import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def _float01_to_u8_bgr(img_rgb_float01: np.ndarray) -> np.ndarray:
    img = np.clip(img_rgb_float01, 0.0, 1.0)
    img_u8 = (img * 255.0 + 0.5).astype(np.uint8)
    return cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)


def _save_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _chunks(total: int, batch_size: int):
    batches = int(math.ceil(total / batch_size))
    for b in range(batches):
        start = b * batch_size
        end = min(total, (b + 1) * batch_size)
        yield start, end, end - start


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic training images.")
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("out/generated"),
        help="Output directory.",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["fixed", "generic"],
        default="fixed",
        help="Data generator mode: fixed uses generate_data.py; generic uses generate_data_generic.py.",
    )
    parser.add_argument("-n", "--num", type=int, default=100, help="Total samples.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument(
        "--save",
        choices=["png", "npz", "both"],
        default="both",
        help="Save images as PNG, arrays as NPZ, or both.",
    )
    parser.add_argument(
        "--setting",
        type=str,
        default="",
        help='Override generator setting as "W,H,N,M" (e.g. "80,112,10,14").',
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed (0 disables).")
    args = parser.parse_args()

    if args.num <= 0:
        raise SystemExit("--num must be > 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")

    setting = None
    if args.setting.strip():
        parts = [int(p.strip()) for p in args.setting.split(",")]
        if len(parts) != 4:
            raise SystemExit('--setting must be "W,H,N,M"')
        setting = tuple(parts)

    if args.seed:
        np.random.seed(args.seed)

    out_dir = args.out
    (out_dir / "meta").mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    meta = {
        "mode": args.mode,
        "num": args.num,
        "batch_size": args.batch_size,
        "save": args.save,
        "setting": setting,
        "seed": args.seed,
    }
    (out_dir / "meta" / "config.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if args.mode == "fixed":
        from generate_data import generate_img

        gen = generate_img(batch_size=args.batch_size, setting=setting)
        for start, end, count in _chunks(args.num, args.batch_size):
            X, Y = next(gen)
            X = X[:count]
            Y = Y[:count]

            for i in range(count):
                idx = start + i
                img0 = X[i, :, :, 0:3] + 0.5
                img1 = X[i, :, :, 3:6] + 0.5

                if args.save in ("png", "both"):
                    cv2.imwrite(
                        str(out_dir / "images" / f"{idx:06d}_0.png"),
                        _float01_to_u8_bgr(img0),
                    )
                    cv2.imwrite(
                        str(out_dir / "images" / f"{idx:06d}_1.png"),
                        _float01_to_u8_bgr(img1),
                    )

                if args.save in ("npz", "both"):
                    _save_npz(
                        out_dir / "labels" / f"{idx:06d}.npz",
                        x=X[i].astype(np.float32),
                        y=Y[i].astype(np.float32),
                    )

    else:
        from generate_data_generic import generate_img

        gen = generate_img(batch_size=args.batch_size, setting=setting)
        for start, end, count in _chunks(args.num, args.batch_size):
            X, Y_list = next(gen)
            X = X[:count]
            Y_list = [y[:count] for y in Y_list]

            for i in range(count):
                idx = start + i
                img0 = X[i, :, :, 0:3] + 0.5
                img1 = X[i, :, :, 3:6] + 0.5

                if args.save in ("png", "both"):
                    cv2.imwrite(
                        str(out_dir / "images" / f"{idx:06d}_0.png"),
                        _float01_to_u8_bgr(img0),
                    )
                    cv2.imwrite(
                        str(out_dir / "images" / f"{idx:06d}_1.png"),
                        _float01_to_u8_bgr(img1),
                    )

                if args.save in ("npz", "both"):
                    arrays = {
                        "x": X[i].astype(np.float32),
                        **{f"y{k}": y[i].astype(np.float32) for k, y in enumerate(Y_list)},
                    }
                    _save_npz(out_dir / "labels" / f"{idx:06d}.npz", **arrays)

    (out_dir / "meta" / "done.txt").write_text("ok\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

