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
| 基础 VLM | `/gemini/code/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct` |
| 数据集 | `LEROBOT_LIBERO_DATA`，mix `libero_all` |
| 输出目录 | `/gemini/code/starVLA/playground/Checkpoints/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619/` |

## 训练配置

| 字段 | 值 |
| --- | --- |
| 最大训练步数 | 80000 |
| per_device_batch_size | 4 |
| gradient_accumulation_steps | 8 |
| save_interval | 125 |
| logging_frequency | 10 |
| eval_interval | 250 |
| checkpoint_format | lightweight |
| save_format | safetensors |
| 冻结模块 | `qwen_vl_interface` |
| local_checkpoint_root | `/root/temp` |
| local_checkpoint_keep_count | 1 |

## 最新监控状态（2026-06-20 07:58 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-20 07:58 CST |
| 主训练进程 PID | `84113`（已运行约 31h26m） |
| 数据加载子进程 PID | `124465` |
| 训练状态 | ✅ 正常运行中 |
| 当前步数 | **7375 / 80000** |
| 最新完整 checkpoint | `steps_7375`（约 18G） |
| 主进程 CPU | 163%（ps）/ 125.0%（top 瞬时） |
| 数据 worker CPU | 23.0%（ps） |
| 主进程内存 | RSS 约 2.0 GiB |
| 数据 worker 内存 | RSS 约 5.5 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.7%），功耗 250 W |
| 系统负载 | 22.81 / 22.56 / 21.77 |
| 系统内存 | 503 GiB 总，78 GiB 已用，419 GiB 可用 |
| root overlay | 30G 总，860M 已用，30G 可用（3%） |
| `/root/temp` | 28K（已清理） |
| 输出目录总大小 | 1.1T |
| 备注 | 距上次监控约 9.75 小时，训练从 step 5250 推进至 step 7375，新增 17 个 checkpoint；GPU 利用率恢复并稳定 100% |

## 关键节点

| 时间 | 步数 | checkpoint | 备注 |
| --- | --- | --- | --- |
| 2026-06-19 16:40 | 4000 | `steps_4000` | GPU 98%，本地暂存已清理 |
| 2026-06-19 22:03 | 5250 | `steps_5125` | GPU 短暂降至 0%，root overlay 升至 30% |
| 2026-06-19 22:07 | 5250 | `steps_5250` | GPU 恢复至 96%，root overlay 回落至 3% |
| 2026-06-19 22:12 | 5250 | `steps_5250` | 5 分钟步数未推进，GPU 86% |
| 2026-06-20 07:58 | 7375 | `steps_7375` |  overnight 正常推进，GPU 100% |

## 注意事项

1. 目标 run_id `250618_r2` 未找到对应进程或目录，实际运行的是 `250619`。
2. 训练 overnight 期间从 step 5250 推进至 step 7375，状态稳定。
3. 输出目录已增长至 1.1T，需关注存储空间。
4. `/root/temp` 清理机制正常，本地暂存未堆积。
