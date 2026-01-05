import argparse
import math
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import generate_data as gd
from pytorch_train import SmallTrackerNet


def shear(center_x, center_y, sigma, shear_x, shear_y, xx, yy, interval, crazy):
    g = np.exp(-(((xx - center_x) ** 2 + (yy - center_y) ** 2)) / (2.0 * sigma**2))

    dx = shear_x * g
    dy = shear_y * g
    if not crazy:
        thres = 0.7 * interval
        mag = (dx**2 + dy**2) ** 0.5
        mask = mag > thres
        dx[mask] = dx[mask] / mag[mask] * thres
        dy[mask] = dy[mask] / mag[mask] * thres

    return xx + dx, yy + dy


def twist(center_x, center_y, sigma, theta, xx, yy):
    g = np.exp(-(((xx - center_x) ** 2 + (yy - center_y) ** 2)) / (2.0 * sigma**2))
    dx = xx - center_x
    dy = yy - center_y

    rotx = dx * np.cos(theta) - dy * np.sin(theta)
    roty = dx * np.sin(theta) + dy * np.cos(theta)

    return xx + (rotx - dx) * g, yy + (roty - dy) * g


def draw_flow(img, flow, xx0, yy0, K=5, img_raw=None):
    if img_raw is None:
        img_ = cv2.resize(img, (img.shape[1] * K, img.shape[0] * K))
    else:
        img_ = cv2.resize(img_raw, (img.shape[1] * K, img.shape[0] * K))

    scale = 0
    color = (0, 255, 255)

    for i in range(flow.shape[0]):
        for j in range(flow.shape[1]):
            pt1 = (xx0[i, j] * K, yy0[i, j] * K)
            pt2 = (flow[i, j, 0] * K, flow[i, j, 1] * K)
            pt3 = (
                pt2[0] + (pt2[0] - pt1[0]) * scale,
                pt2[1] + (pt2[1] - pt1[1]) * scale,
            )
            pt1 = (int(round(pt1[0])), int(round(pt1[1])))
            pt3 = (int(round(pt3[0])), int(round(pt3[1])))
            cv2.arrowedLine(img_, pt1, pt3, color, 2, 8, 0, 0.4)

    return img_


def cross_product(Ax, Ay, Bx, By, Cx, Cy):
    len1 = ((Bx - Ax) ** 2 + (By - Ay) ** 2) ** 0.5
    len2 = ((Cx - Ax) ** 2 + (Cy - Ay) ** 2) ** 0.5
    return ((Bx - Ax) * (Cy - Ay) - (By - Ay) * (Cx - Ax)) / (len1 * len2 + 1e-6)


def load_torch_model(ckpt_path: Path, device: torch.device) -> SmallTrackerNet:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model = SmallTrackerNet().to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def main() -> int:
    parser = argparse.ArgumentParser(description="Interactive tracking demo (PyTorch).")
    parser.add_argument("--ckpt", type=Path, default="models/smoke/checkpoint_best.pt", help="Path to .pt checkpoint.")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--W", type=int, default=120)
    parser.add_argument("--H", type=int, default=160)
    parser.add_argument("--N", type=int, default=15)
    parser.add_argument("--M", type=int, default=20)
    parser.add_argument("--K", type=int, default=5, help="Display scale.")
    parser.add_argument("--interval", type=int, default=7)
    parser.add_argument("--padding", type=int, default=7)
    parser.add_argument("--mkr-rng", type=float, default=0.0)
    parser.add_argument(
        "--crazy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow larger motions (matches example_tracking_sim default).",
    )
    args = parser.parse_args()

    N = args.N
    M = args.M
    W = args.W
    H = args.H
    K = args.K
    interval = args.interval
    padding = args.padding
    mkr_rng = args.mkr_rng
    crazy = args.crazy

    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = load_torch_model(args.ckpt, device=device)

    x = np.arange(0, W, 1)
    y = np.arange(0, H, 1)
    xx, yy = np.meshgrid(y, x)
    xx0 = xx.copy()
    yy0 = yy.copy()

    img_blur = (np.random.random((W // 3, H // 3, 3)) * 0.9) + 0.1
    img_blur = cv2.resize(img_blur, (H, W))

    traj = []
    moving = False
    rotation = False
    changing_x, changing_y = 0, 0
    svid = 0

    wx = xx0.copy()
    wy = yy0.copy()

    def contrain(xx_, yy_):
        dx = xx_ - xx0
        dy = yy_ - yy0
        if not crazy:
            thres = 1 * interval
            mag = (dx**2 + dy**2) ** 0.5
            mask = mag > thres
            dx[mask] = dx[mask] / mag[mask] * thres
            dy[mask] = dy[mask] / mag[mask] * thres
        return xx0 + dx, yy0 + dy

    def motion_callback(event, x, y, flags, param):
        nonlocal traj, moving, xx, yy, wx, wy, rotation, changing_x, changing_y

        x, y = x / K, y / K

        if event == cv2.EVENT_LBUTTONDOWN:
            traj.append([x, y])
            wx = xx0.copy()
            wy = yy0.copy()
            rotation = False
            moving = True

        elif event == cv2.EVENT_LBUTTONUP:
            traj = []
            moving = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if moving:
                traj.append([x, y])
                sigma = 20
                if not rotation:
                    xx, yy = shear(
                        traj[0][0],
                        traj[0][1],
                        sigma,
                        x - traj[0][0],
                        y - traj[0][1],
                        wx,
                        wy,
                        interval=interval,
                        crazy=crazy,
                    )
                else:
                    theta = math.asin(
                        cross_product(traj[0][0], traj[0][1], changing_x, changing_y, x, y)
                    )
                    theta = max(min(theta, 50 / 180.0 * math.pi), -50 / 180.0 * math.pi)
                    xx, yy = twist(traj[0][0], traj[0][1], sigma, theta, wx, wy)
                if not crazy:
                    xx, yy = contrain(xx, yy)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", motion_callback)

    # Match generate_data.generate_img() marker sampling (fixed grid + small jitter)
    interval_x = W / (N)
    interval_y = H / (M)
    xs = np.arange(interval_x / 2, W, interval_x)[:N]
    ys = np.arange(interval_y / 2, H, interval_y)[:M]
    xind, yind = np.meshgrid(ys, xs)  # xind: col indices, yind: row indices
    xind = (xind.reshape([1, -1])[0]).astype(int)
    yind = (yind.reshape([1, -1])[0]).astype(int)
    xind += (np.random.random(xind.shape) * 2 - 1).astype(int)
    yind += (np.random.random(yind.shape) * 2 - 1).astype(int)
    xind = np.clip(xind, 0, H - 1)
    yind = np.clip(yind, 0, W - 1)

    # Base marker positions (used to draw arrows); consistent with training labels.
    xx_marker_base = xx0[yind, xind].reshape([N, M])
    yy_marker_base = yy0[yind, xind].reshape([N, M])

    def render_reference() -> np.ndarray:
        xx_marker = xx0[yind, xind].reshape([N, M])
        yy_marker = yy0[yind, xind].reshape([N, M])
        return gd.generate(
            xx_marker,
            yy_marker,
            img_blur=img_blur,
            rng=mkr_rng,
            W=W,
            H=H,
            N=N,
            M=M,
            degree=0,
        )

    img0 = render_reference()

    while True:
        xx_marker_, yy_marker_ = xx[yind, xind].reshape([N, M]), yy[yind, xind].reshape([N, M])
        img = gd.generate(
            xx_marker_,
            yy_marker_,
            img_blur=img_blur,
            rng=mkr_rng,
            W=W,
            H=H,
            N=N,
            M=M,
            degree=None,
        )
        img = gd.preprocessing(img, W, H)

        x_in = np.array([np.dstack([img0 - 0.5, img - 0.5])], dtype=np.float32)  # (1,W,H,6)
        x_t = torch.from_numpy(x_in).permute(0, 3, 1, 2).to(device)  # (1,6,W,H)

        st = time.time()
        with torch.no_grad():
            pred = model(x_t)[0].permute(1, 2, 0).cpu().numpy()  # (N,M,2) displacement
        _ = time.time() - st

        pred_xx = xx_marker_base + pred[:, :, 0]
        pred_yy = yy_marker_base + pred[:, :, 1]
        flow = np.stack([pred_xx, pred_yy], axis=-1)
        display_img = draw_flow(img, flow, xx_marker_base, yy_marker_base, K=K, img_raw=img)

        cv2.imshow("image", display_img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            xx = xx0.copy()
            yy = yy0.copy()
            wx, wy = xx.copy(), yy.copy()
            traj = []

        if key == ord("s"):
            rotation = rotation ^ True
            if len(traj) > 0:
                changing_x = traj[-1][0]
                changing_y = traj[-1][1]
            if rotation is False:
                traj = []
            wx, wy = xx.copy(), yy.copy()

        elif key == ord("q") or key == 27:
            break

        elif key == ord("c"):
            img_blur = (np.random.random((W // 3, H // 3, 3)) * 0.9) + 0.1
            img_blur = cv2.resize(img_blur, (H, W))
            img0 = render_reference()

        elif key == ord("p"):
            cv2.imwrite(f"im{svid}.jpg", (display_img * 255).astype(np.uint8))
            svid += 1

        elif key == ord("k"):
            crazy = crazy ^ True

        elif key == ord("z"):
            mkr_rng = mkr_rng - 0.5
            if mkr_rng < 0:
                mkr_rng = 1
            img0 = render_reference()

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
