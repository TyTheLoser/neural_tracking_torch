# Neural Tracking (PyTorch)

该仓库仅保留 PyTorch 运行链路：离线数据生成（`.npz`）→ 训练（保存最佳 checkpoint）→ 交互推理（鼠标拖拽验证）。

## 环境（Conda + uv）

```bash
conda env create -f environment.yml
conda activate neural_tracking
uv sync
```

## 生成离线数据集（来自 `generate_data.py`）

输出目录结构：
- `out/dataset_fixed/labels/*.npz`：训练用数据（必需）
- `out/dataset_fixed/images/*.png`：可选的可视化图片（当 `--save png|both` 时）

```bash
conda run -n neural_tracking uv run python generate_images_main.py \
  -o out/dataset_fixed \
  --setting 120,160,15,20 \
  -n 10000 \
  --batch-size 128 \
  --save npz
```

说明：
- 默认会在已有 `labels/*.npz` 之后继续编号追加（避免覆盖）。需要从 0 重新生成可用 `--start-idx 0` 并手动清空目录。

## 训练（PyTorch）

训练输出：
- 最佳 checkpoint：`models/<prefix>/checkpoint_best.pt`（只保留一个，验证变好时覆盖更新）
- 验证可视化：`out/eval/<prefix>/epoch_XXX/val_*.png`

```bash
conda run -n neural_tracking uv run python pytorch_train.py \
  --data-dir out/dataset_fixed \
  -p test \
  --batch-size 128 \
  --steps-per-epoch 2000 \
  --epochs 100
```

说明：
- 默认开启断点重训（`--resume`）：如果存在 `models/<prefix>/checkpoint_best.pt` 会从最佳 epoch 继续。
- 需要从头训练：加 `--no-resume`（或删除 `models/<prefix>/checkpoint_best.pt`）。

## 交互推理（鼠标拖拽验证）

```bash
conda run -n neural_tracking uv run python pytorch_example_tracking_sim.py \
  --ckpt models/test/checkpoint_best.pt
```

按键说明：
- 鼠标左键拖拽：平移形变
- `s`：切换旋转模式（按住鼠标拖拽）
- `r`：重置形变
- `c`：随机背景并重置参考帧
- `z`：切换 marker 强度/大小效果并重置参考帧
- `k`：切换 `crazy`（限制/放开形变幅度）
- `p`：保存截图 `im*.jpg`
- `q` / `Esc`：退出
