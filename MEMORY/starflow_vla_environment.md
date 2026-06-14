# StarFlow-VLA 环境记忆

## 模型路径规则

- 本地模型入口统一放在 `playground/Pretrained_models/`。
- `playground/Pretrained_models/` 下通常使用软链接，不直接保存大模型文件。
- 已确认 `playground/Pretrained_models/Qwen3-VL-4B-Instruct` 指向 `/gemini/pretrain/Qwen3-VL-4B-Instruct/`。
- 后续如需新增模型，模型文件下载保存到 `/gemini/code/models/`，再在 `playground/Pretrained_models/` 下创建软链接指向该目录。
- 未经确认不主动下载新模型。

## Python 环境

- StarFlow-VLA 相关检查优先使用仓库 `.venv`。
- `import torch` 可能耗时约 4 分钟，涉及 framework import 的检查需预留等待时间。

## LIBERO 数据

- 已准备 P0 最小数据集 `libero_goal_no_noops_1.0.0_lerobot`。
- 数据实际目录：`/gemini/code/datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot`。
- 仓库入口软链接：`playground/Datasets/LEROBOT_LIBERO_DATA -> /gemini/code/datasets/LEROBOT_LIBERO_DATA`。
- 已将 `examples/LIBERO/train_files/modality.json` 复制到数据集 `meta/modality.json`。
- 当前 LIBERO registry 使用 7D action 与 8D state：state keys 包含 `x,y,z,roll,pitch,yaw,pad,gripper`。
