import argparse
import shutil
import time
from pathlib import Path
from typing import Iterator, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset, Subset


class SmallTrackerNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        k = 5
        p = 2

        self.conv1a = nn.Conv2d(6, 16, k, padding=p)
        self.conv1b = nn.Conv2d(16, 16, k, padding=p)
        self.pool1 = nn.AvgPool2d(2)

        self.conv2a = nn.Conv2d(16, 32, k, padding=p)
        self.conv2b = nn.Conv2d(32, 32, k, padding=p)
        self.pool2 = nn.AvgPool2d(2)

        self.conv3a = nn.Conv2d(32, 128, k, padding=p)
        self.conv3b = nn.Conv2d(128, 128, k, padding=p)
        self.conv3c = nn.Conv2d(128, 128, k, padding=p)
        self.conv3d = nn.Conv2d(128, 128, k, padding=p)
        self.pool3 = nn.AvgPool2d(2)

        self.conv4a = nn.Conv2d(128, 256, k, padding=p)
        self.conv4b = nn.Conv2d(256, 256, k, padding=p)
        self.head = nn.Conv2d(256, 2, k, padding=p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1a(x))
        x = F.relu(self.conv1b(x))
        x = self.pool1(x)

        x = F.relu(self.conv2a(x))
        x = F.relu(self.conv2b(x))
        x = self.pool2(x)

        x = F.relu(self.conv3a(x))
        x = F.relu(self.conv3b(x))
        x = F.relu(self.conv3c(x))
        x = F.relu(self.conv3d(x))
        x = self.pool3(x)

        x = F.relu(self.conv4a(x))
        x = torch.sigmoid(self.conv4b(x))
        x = self.head(x)
        return x


class OfflineNPZDataset(Dataset[Tuple[torch.Tensor, torch.Tensor]]):
    def __init__(self, root: Path):
        labels_dir = root / "labels"
        search_dir = labels_dir if labels_dir.is_dir() else root
        self.files = sorted(search_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found under {search_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path = self.files[idx]
        with np.load(path) as data:
            x = data["x"].astype(np.float32)  # (W,H,6) in [-0.5,0.5]
            y = data["y"].astype(np.float32)  # (N,M,2)

        x_t = torch.from_numpy(x).permute(2, 0, 1).contiguous()  # (6,W,H)
        y_t = torch.from_numpy(y).permute(2, 0, 1).contiguous()  # (2,N,M)
        return x_t, y_t


def _iter_forever(loader: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def _to_u8_bgr(img_float01: np.ndarray) -> np.ndarray:
    img = np.clip(img_float01, 0.0, 1.0)
    img_u8 = (img * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)


def _draw_pred_arrows(img_float01: np.ndarray, flow_nm2: np.ndarray, scale: int = 5) -> np.ndarray:
    img_bgr = _to_u8_bgr(img_float01)
    img_big = cv2.resize(img_bgr, (img_bgr.shape[1] * scale, img_bgr.shape[0] * scale))

    gh, gw = flow_nm2.shape[0], flow_nm2.shape[1]
    sx = img_bgr.shape[0] / float(gh)
    sy = img_bgr.shape[1] / float(gw)

    color = (0, 255, 255)
    for i in range(gh):
        for j in range(gw):
            x0 = (i + 0.5) * sx
            y0 = (j + 0.5) * sy
            dx = float(flow_nm2[i, j, 0])
            dy = float(flow_nm2[i, j, 1])

            p0 = (int(round(y0 * scale)), int(round(x0 * scale)))
            p1 = (int(round((y0 + dy) * scale)), int(round((x0 + dx) * scale)))
            cv2.arrowedLine(img_big, p0, p1, color, 2, 8, 0, 0.4)

    return img_big


def _load_fixed_val(
    val_loader: DataLoader, *, num: int, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = []
    ys = []
    got = 0
    for xb, yb in val_loader:
        xs.append(xb)
        ys.append(yb)
        got += xb.shape[0]
        if got >= num:
            break
    if not xs:
        raise RuntimeError("Validation loader produced no batches")

    x = torch.cat(xs, dim=0)[:num].to(device=device, dtype=torch.float32)
    y = torch.cat(ys, dim=0)[:num].to(device=device, dtype=torch.float32)
    return x, y


def main() -> int:
    parser = argparse.ArgumentParser(description="Train marker tracking model (PyTorch, TF-style loop).")
    parser.add_argument("--data-dir", type=Path, default=Path("out/dataset_fixed"))
    parser.add_argument("-p", "--prefix", default="test")
    parser.add_argument("-lr", "--lr", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--steps-per-epoch", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--val-size", type=int, default=1000)
    parser.add_argument("--val-save-every", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Resume from best checkpoint if present (resumes from best epoch).",
    )
    args = parser.parse_args()

    prefix = args.prefix
    lr = args.lr
    print(prefix, lr)

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model_dir = Path("models") / prefix
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy("pytorch_train.py", str(model_dir / "pytorch_train.py"))
    if Path("generate_data.py").exists():
        shutil.copy("generate_data.py", str(model_dir / "generate_data.py"))

    best_ckpt_path = model_dir / "checkpoint_best.pt"

    dataset = OfflineNPZDataset(args.data_dir)
    n_total = len(dataset)
    n_val = max(1, int(round(n_total * args.val_ratio)))
    n_train = n_total - n_val
    if n_train <= 0:
        raise SystemExit(f"val split too large: dataset={n_total}, val={n_val}")

    train_ds = Subset(dataset, list(range(0, n_train)))
    val_ds = Subset(dataset, list(range(n_train, n_total)))

    pin = device.type == "cuda"
    workers = max(0, args.workers)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=pin,
        drop_last=False,
        persistent_workers=(workers > 0),
        prefetch_factor=4 if workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin,
        drop_last=False,
        persistent_workers=(workers > 0),
        prefetch_factor=4 if workers > 0 else None,
    )

    model = SmallTrackerNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Fixed validation set, like TF version.
    X_test, Y_test = _load_fixed_val(val_loader, num=args.val_size, device=device)

    start_epoch = 0
    min_loss = float("inf")

    if args.resume and best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location="cpu")
        if isinstance(ckpt, dict) and "model" in ckpt:
            model.load_state_dict(ckpt["model"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = int(ckpt.get("epoch", -1)) + 1
            min_loss = float(ckpt.get("loss", min_loss))
            for pg in optimizer.param_groups:
                pg["lr"] = lr
            print(f"resume from {best_ckpt_path} (epoch={start_epoch}, best_loss={min_loss:.6f})")

    train_iter = _iter_forever(train_loader)

    if start_epoch >= args.epochs:
        print(f"Nothing to do: start_epoch={start_epoch} >= epochs={args.epochs}")
        return 0

    epoch_pbar = tqdm(range(start_epoch, args.epochs), desc="train", unit="epoch", dynamic_ncols=True)
    for epoch in epoch_pbar:
        step_pbar = tqdm(range(args.steps_per_epoch), desc=f"epoch {epoch}", unit="step", leave=False, dynamic_ncols=True)
        model.train()

        train_loss_sum = 0.0
        train_loss_count = 0
        t0 = time.time()

        for step in step_pbar:
            xb, yb = next(train_iter)
            xb = xb.to(device=device, non_blocking=True)
            yb = yb.to(device=device, non_blocking=True)

            pred = model(xb)
            loss = F.mse_loss(pred, yb, reduction="mean")

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            loss_v = float(loss.detach().cpu().item())
            bs = xb.shape[0]
            train_loss_sum += loss_v * bs
            train_loss_count += bs
            if step == 0 or (step + 1) % 10 == 0:
                step_pbar.set_postfix(loss=f"{loss_v:.6f}")

        model.eval()
        with torch.no_grad():
            pred = model(X_test)
            val_loss = ((pred - Y_test) ** 2).mean().item()

        train_loss = train_loss_sum / max(1, train_loss_count)
        epoch_dt = time.time() - t0
        epoch_pbar.set_postfix(train=f"{train_loss:.6f}", val=f"{val_loss:.6f}", best=f"{min_loss:.6f}", s=f"{epoch_dt:.1f}")

        out_dir = Path("out") / "eval" / prefix / f"epoch_{epoch:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        X_test_np = X_test.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)  # (B,W,H,6)
        pred_np = pred.detach().cpu().permute(0, 2, 3, 1).numpy().astype(np.float32)  # (B,N,M,2)

        for k in range(0, X_test_np.shape[0], max(1, args.val_save_every)):
            img_cur = X_test_np[k, :, :, 3:6] + 0.5
            vis = _draw_pred_arrows(img_cur, pred_np[k], scale=5)
            cv2.imwrite(str(out_dir / f"val_{k:04d}.png"), vis)

        if val_loss < min_loss:
            min_loss = float(val_loss)
            ckpt = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss": min_loss,
                "prefix": prefix,
                "lr": lr,
            }
            torch.save(ckpt, best_ckpt_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
