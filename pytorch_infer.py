import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch

from pytorch_train import SmallTrackerNet


def _load_rgb_float01(path: Path) -> np.ndarray:
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _prep_pair(img0: np.ndarray, img1: np.ndarray) -> torch.Tensor:
    if img0.shape != img1.shape:
        raise ValueError(f"shape mismatch: {img0.shape} vs {img1.shape}")
    x = np.dstack([img0 - 0.5, img1 - 0.5]).astype(np.float32)  # (W,H,6)
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).contiguous()  # (1,6,W,H)
    return x


def _save_flow_npy(path: Path, flow: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, flow)


def _flow_to_vis(flow: np.ndarray, scale: float = 3.0) -> np.ndarray:
    dx = flow[..., 0]
    dy = flow[..., 1]
    mag = np.sqrt(dx * dx + dy * dy)
    ang = np.arctan2(dy, dx)
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ((ang + np.pi) * (180.0 / (2 * np.pi))).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(mag * scale * 255.0, 0, 255).astype(np.uint8)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr


def main() -> int:
    parser = argparse.ArgumentParser(description="Inference for marker tracking model (PyTorch).")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to .pt checkpoint (state_dict).")
    parser.add_argument("--img0", type=Path, required=True, help="Reference image (PNG/JPG).")
    parser.add_argument("--img1", type=Path, required=True, help="Current image (PNG/JPG).")
    parser.add_argument("--out", type=Path, default=Path("out/torch_infer"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--save-vis", action="store_true", help="Write flow visualization PNG.")
    args = parser.parse_args()

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    img0 = _load_rgb_float01(args.img0)
    img1 = _load_rgb_float01(args.img1)
    x = _prep_pair(img0, img1).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    model = SmallTrackerNet().to(device)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        pred = model(x)[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)  # (N,M,2)

    args.out.mkdir(parents=True, exist_ok=True)
    _save_flow_npy(args.out / "flow.npy", pred)

    if args.save_vis:
        vis = _flow_to_vis(pred)
        cv2.imwrite(str(args.out / "flow_vis.png"), vis)

    print(f"saved: {args.out / 'flow.npy'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

