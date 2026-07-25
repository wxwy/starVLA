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

## DeepSpeed Universal Checkpoint

- 当前 StarVLA 默认 checkpoint 格式为 DeepSpeed Universal checkpoint。
- StarFlow stage1 默认按单卡常用配置运行：`vla_data.per_device_batch_size=4`、`trainer.gradient_accumulation_steps=8`，有效全局 batch 为 32。
- StarFlow stage1 默认 W&B project 为 `starflow_vla`。
- full-adam Universal checkpoint 保存 ZeRO optimizer-backed fp32 master weights、Adam `exp_avg/exp_avg_sq` 与 `step`，用于跨 GPU 数量恢复优化器动量。
- 该格式仍不包含 scheduler、dataloader、RNG 状态，因此不是完整 bitwise training-state resume。
- 已验证 full-adam Universal checkpoint 可以跨 4 卡与 1 卡恢复加载并继续训练。
- 持久保存路径：
  - full-adam `steps_31500`: `playground/Checkpoints/P0-M5-E-H2a-03_universal_full_adam_250618`
  - 正式实验恢复入口：`playground/Checkpoints/P0-M5-E-H2a-03_starflow_libero-goal_qwen3vl4b_lwfm_ft32_250615/checkpoints/steps_31500`
- StarFlow stage1 默认 `run_id` / W&B run 为 `P0-M5-E-H2a-03_starflow_libero-goal_qwen3vl4b_lwfm_ft32_250615`。
- Universal -> HF safetensors 模型权重导出工具：`tools/convert_universal_checkpoint_to_hf_safetensors.py`。
