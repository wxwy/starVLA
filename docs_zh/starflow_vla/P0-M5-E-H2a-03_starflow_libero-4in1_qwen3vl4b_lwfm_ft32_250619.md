# StarFlow VLA 实验记录

## 实验标识

| 字段 | 值 |
| --- | --- |
| 实验名称 | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 训练脚本 | `starVLA/training/train_starvla.py` |
| 启动脚本 | `examples/LIBERO/train_files/run_starflow_train_ready.sh` |
| 配置文件 | `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml` |
| 框架 | `StarFlowVLA` |
| 基础 VLM | `/disk/rl/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct` |
| 数据集 | `LEROBOT_LIBERO_DATA`，mix `libero_all` |
| 输出目录 | `/disk/rl/starVLA/playground/Checkpoints/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619/` |

## 训练配置

| 字段 | 值 |
| --- | --- |
| 最大训练步数 | 80000 |
| per_device_batch_size | 8 |
| gradient_accumulation_steps | 4 |
| save_interval | 250 |
| logging_frequency | 20 |
| eval_interval | 500 |
| checkpoint_format | lightweight |
| save_format | safetensors |
| 冻结模块 | `qwen_vl_interface` |
| local_checkpoint_root | `/localdisk-tmp` |
| local_checkpoint_keep_count | 1 |

### 原始训练配置（变更前）

| 字段 | 值 |
| --- | --- |
| per_device_batch_size | 4 |
| gradient_accumulation_steps | 8 |
| save_interval | 125 |
| logging_frequency | 10 |
| eval_interval | 250 |
| local_checkpoint_root | `/root/temp` |

## 最新监控状态（2026-06-21 12:45 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-21 12:45 CST |
| 训练状态 | ✅ 已 resume 并正常运行中 |
| 当前步数 | **11250 / 80000** |
| resume 源 checkpoint | `steps_11000`（DeepSpeed ZeRO-2 格式，单进程 world_size=1） |
| 最新完整 checkpoint | `steps_11250` ✅（普通格式，已同步） |
| 设备 | `NVIDIA A100-SXM4-80GB`（80 GB 显存） |
| 计费 | 5.58 元/h |
| GPU 利用率 | 训练中 |
| GPU 显存 | 81920 MiB |
| 训练速度 | 约 7.40 s/it（非 fused AdamW 模式） |
| 备注 | 从 DeepSpeed checkpoint 降级恢复为普通单卡训练；`optimizer_rank_00000.pt` 中的 `base_optimizer_state` 已提取并 reshape 到普通 PyTorch optimizer state；DeepSpeed 特有状态（loss scaler、fp32 分区）已丢弃；`steps_11250` 为普通格式，下次 resume 可恢复 `fused=True` 高速模式 |

## 关键节点

| 时间 | 步数 | checkpoint | 备注 |
| --- | --- | --- | --- |
| 2026-06-19 16:40 | 4000 | `steps_4000` | GPU 98%，本地暂存已清理 |
| 2026-06-19 22:03 | 5250 | `steps_5125` | GPU 短暂降至 0%，root overlay 升至 30% |
| 2026-06-19 22:07 | 5250 | `steps_5250` | GPU 恢复至 96%，root overlay 回落至 3% |
| 2026-06-19 22:12 | 5250 | `steps_5250` | 5 分钟步数未推进，GPU 86% |
| 2026-06-20 07:58 | 7375 | `steps_7375` | overnight 正常推进，GPU 100% |
| 2026-06-21 12:45 | 11250 | `steps_11250` | 从 DeepSpeed checkpoint resume，设备切换为 A100(80G)，训练参数变更，计费 5.58 元/h；新 checkpoint 为普通格式 |

## 注意事项

1. 目标 run_id `250618_r2` 未找到对应进程或目录，实际运行的是 `250619`。
2. 训练 overnight 期间从 step 5250 推进至 step 7375，状态稳定。
3. 输出目录已增长至 1.1T，需关注存储空间。
4. `/root/temp` 清理机制正常，本地暂存未堆积。
5. 2026-06-21 从 `steps_11000` resume 时，原 checkpoint 为 DeepSpeed ZeRO-2 格式，已降级转换为普通单卡 optimizer state 继续训练；转换过程丢弃了 DeepSpeed 特有的 loss scaler 和 fp32 分区信息。
6. resume 后设备切换为 `NVIDIA A100-SXM4-80GB`（80 GB 显存），计费 5.58 元/h，训练参数同步变更（见「训练配置」表）。
7. 由于 DeepSpeed 转换后的 optimizer state 不满足 fused AdamW 的严格 tensor 要求，resume 后临时以 `fused=False` 运行；`steps_11250` 已保存为普通格式，可从该 checkpoint 重新 resume 以恢复 `fused=True` 高速模式。
