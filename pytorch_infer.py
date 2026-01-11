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
def draw_flow_on_markers(
    img: np.ndarray,
    marker_pos_x: np.ndarray,
    marker_pos_y: np.ndarray,
    dense_flow: np.ndarray,
    img_size: Tuple[int, int],
    scale: float = 1.0,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """
    在marker点上绘制位移场箭头（从稠密流场采样）
    img: (H, W, 3), float[0,1] or uint8
    marker_pos_x: (N, M), marker基准x坐标（在img_size尺度上）
    marker_pos_y: (N, M), marker基准y坐标（在img_size尺度上）
    dense_flow: (H_out, W_out, 2), 稠密流场
    img_size: (W, H), 原始图像尺寸
    """
    if img.dtype != np.uint8:
        vis = (img * 255.0).clip(0, 255).astype(np.uint8)
    else:
        vis = img.copy()

    W, H = img_size
    H_out, W_out = dense_flow.shape[:2]

    N, M = marker_pos_x.shape
    for i in range(N):
        for j in range(M):
            x1 = int(marker_pos_x[i, j])
            y1 = int(marker_pos_y[i, j])

            # 将marker坐标映射到流场网格坐标
            flow_x = int(x1 * W_out / W)
            flow_y = int(y1 * H_out / H)
            flow_x = min(max(flow_x, 0), W_out - 1)
            flow_y = min(max(flow_y, 0), H_out - 1)

            dx = dense_flow[flow_y, flow_x, 0] * scale
            dy = dense_flow[flow_y, flow_x, 1] * scale
            x2 = int(x1 + dx)
            y2 = int(y1 + dy)

            cv2.arrowedLine(
                vis,
                (x1, y1),
                (x2, y2),
                color,
                thickness=1,
                tipLength=0.3,
                line_type=cv2.LINE_AA,
            )
    return vis


def _load_frame_rgb_float01(frame: np.ndarray) -> np.ndarray:
    """将BGR帧转为RGB float[0,1]"""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return (rgb.astype(np.float32) / 255.0).clip(0.0, 1.0)


def _generate_marker_grid(W: int, H: int, N: int = 13, M: int = 18, seed: int = 0):
    """
    生成marker网格位置（与训练数据生成一致）
    返回: (xind, yind, xx_marker_base, yy_marker_base)
    """
    np.random.seed(seed)

    x = np.arange(100, W-100, 1)
    y = np.arange(100, H-100, 1)
    xx0, yy0 = np.meshgrid(y, x)

    interval_x = W-200 / N
    interval_y = H-200 / M

    x = np.arange(interval_x / 2, W, interval_x)[:N]
    y = np.arange(interval_y / 2, H, interval_y)[:M]
    xind, yind = np.meshgrid(y, x)

    xind = (xind.reshape([1, -1])[0]).astype(int)
    yind = (yind.reshape([1, -1])[0]).astype(int)
    xind += (np.random.random(xind.shape) * 2 - 1).astype(int)
    yind += (np.random.random(yind.shape) * 2 - 1).astype(int)
    xind = np.clip(xind, 0, H - 1)
    yind = np.clip(yind, 0, W - 1)

    # Marker基准位置（用于绘制箭头起点）
    xx_marker_base = xx0[yind, xind].reshape([N, M])
    yy_marker_base = yy0[yind, xind].reshape([N, M])

    return xind, yind, xx_marker_base, yy_marker_base


def _load_marker_grid_from_npz(npz_path: Path):
    """
    从npz文件加载真实marker网格
    返回: (xx_marker_base, yy_marker_base, N, M)
    """
    data = np.load(npz_path)
    grid_points = data['grid_points']  # (N, M, 2) 格式
    N, M, _ = grid_points.shape

    # grid_points[..., 0] 是x坐标，[..., 1] 是y坐标
    xx_marker_base = grid_points[..., 0].astype(np.float32)
    yy_marker_base = grid_points[..., 1].astype(np.float32)

    return xx_marker_base, yy_marker_base, N, M


def main() -> int:
    parser = argparse.ArgumentParser(description="Inference for marker tracking model (PyTorch).")
    parser.add_argument("--ckpt", type=Path, required=True, help="Path to .pt checkpoint (state_dict).")
    parser.add_argument("--img0", type=Path, required=True, help="Reference image (PNG/JPG).")
    parser.add_argument("--video", type=Path, required=True, help="Input video file.")
    parser.add_argument("--out", type=Path, default=Path("out/torch_infer"))
    parser.add_argument("--output-video", type=Path, default=None, help="Output video path (default: out/flow_vis.mp4).")
    parser.add_argument("--fps", type=float, default=30.0, help="Output video FPS.")
    parser.add_argument("--marker-npz", type=Path, default=None,
                       help="Path to npz file with real marker grid (e.g., warp_raw_to_px_grid_points_rectified.npz). "
                            "If provided, overrides --N, --M, --marker-seed.")
    parser.add_argument("--N", type=int, default=13, help="Number of markers in X direction (ignored if --marker-npz is set).")
    parser.add_argument("--M", type=int, default=18, help="Number of markers in Y direction (ignored if --marker-npz is set).")
    parser.add_argument("--marker-seed", type=int, default=0, help="Random seed for marker jitter (ignored if --marker-npz is set).")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        print(f"Using device: {device}")

    # 加载参考图
    img0 = _load_rgb_float01(args.img0)
    h_ref, w_ref = img0.shape[:2]
    H, W = h_ref, w_ref

    # 生成或加载marker网格（在原图尺寸上）
    if args.marker_npz is not None:
        print(f"Loading marker grid from: {args.marker_npz}")
        xx_marker_base, yy_marker_base, N, M = _load_marker_grid_from_npz(args.marker_npz)
        print(f"Marker grid: {N}x{M}")
    else:
        print(f"Generating synthetic marker grid: {args.N}x{args.M}")
        N, M = args.N, args.M
        _, _, xx_marker_base, yy_marker_base = _generate_marker_grid(
            W, H, N=N, M=M, seed=args.marker_seed
        )

    # 强制resize到模型输入尺寸 (W=104, H=144)
    target_W, target_H = 144, 104
    print(f"Resizing reference image: {w_ref}x{h_ref} -> {target_W}x{target_H}")
    img0 = cv2.resize(img0, (target_W, target_H), interpolation=cv2.INTER_LINEAR)

    # 调整marker基准位置（如果从npz加载）
    if args.marker_npz is not None:
        scale_x = target_W / w_ref
        scale_y = target_H / h_ref
        xx_marker_base = (xx_marker_base * scale_x).astype(np.float32)
        yy_marker_base = (yy_marker_base * scale_y).astype(np.float32)
        print(f"Resized markers: scale_x={scale_x:.4f}, scale_y={scale_y:.4f}")

    # 打开视频
    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    # 获取视频属性
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {vid_w}x{vid_h}, {total_frames} frames")

    # 输出尺寸固定为目标尺寸
    out_w, out_h = target_W, target_H
    print(f"All frames will be resized to: {out_w}x{out_h}")

    # 设置输出视频路径
    if args.output_video is None:
        args.output_video = args.out / "flow_vis.mp4"
    args.output_video.parent.mkdir(parents=True, exist_ok=True)

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(args.output_video), fourcc, args.fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot create video writer: {args.output_video}")

    # 加载模型
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = SmallTrackerNet().to(device)
    model.load_state_dict(state)
    model.eval()

    # 处理每一帧
    frame_count = 0
    last_flow = None
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = _load_frame_rgb_float01(frame_bgr)

        # 强制resize到目标尺寸
        if frame_rgb.shape[1] != target_W or frame_rgb.shape[0] != target_H:
            frame_rgb = cv2.resize(frame_rgb, (target_W, target_H), interpolation=cv2.INTER_LINEAR)

        # 准备输入
        x = _prep_pair(img0, frame_rgb).to(device)

        # 推理：模型输出 (2, H/8, W/8) -> (H/8, W/8, 2)
        # 经过3次pooling，输出尺寸是输入的1/8
        with torch.no_grad():
            pred_flow = model(x)[0]  # (2, H_out, W_out)
            pred_flow = pred_flow.permute(1, 2, 0).cpu().numpy().astype(np.float32)  # (H_out, W_out, 2)

        # 保存稠密流场
        last_flow = pred_flow

        # 可视化：在marker点上绘制位移场箭头
        vis = draw_flow_on_markers(
            frame_rgb,
            xx_marker_base,
            yy_marker_base,
            pred_flow,
            (target_W, target_H),
            scale=1.0,
        )

        # 写入输出视频
        writer.write(vis)
        frame_count += 1

        if frame_count % 30 == 0:
            print(f"Processed {frame_count}/{total_frames} frames")

    # 释放资源
    cap.release()
    writer.release()

    # 保存最后一帧的flow
    if last_flow is not None:
        args.out.mkdir(parents=True, exist_ok=True)
        _save_flow_npy(args.out / "flow.npy", last_flow)
        print(f"Saved: {args.out / 'flow.npy'}")

    print(f"Output video: {args.output_video}")
    print(f"Processed {frame_count} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

