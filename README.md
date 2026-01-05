# Neural Tracking (PyTorch)

该仓库已精简为仅保留 PyTorch 版本的：数据生成（离线 `.npz`）、训练（`.pt`）、交互推理（鼠标拖拽验证）。

## 环境（Conda + uv）

```bash
conda env create -f environment.yml
conda activate neural_tracking
uv sync
```

## 生成离线数据集（来自 `generate_data.py`）

默认输出到 `out/dataset_fixed/labels/*.npz`：

```bash
conda run -n neural_tracking uv run python generate_images_main.py -n 50000 --batch-size 64 --save npz
```

## 训练（PyTorch）

训练并保存到 `models/<prefix>/tracking_XXX_*.pt`，验证可视化保存到 `out/eval/<prefix>/epoch_XXX/`：

```bash
conda run -n neural_tracking uv run python pytorch_train.py --data-dir out/dataset_fixed -p test -lr 1e-5
```

## 交互推理（鼠标拖拽验证）

```bash
conda run -n neural_tracking uv run python pytorch_example_tracking_sim.py --ckpt "$(ls models/test/*.pt | head -n 1)"
```

按键与 `example_tracking_sim.py` 一致：`r/s/c/p/k/z/q`，鼠标左键拖拽平移，`s` 后拖拽旋转。
