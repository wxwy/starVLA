# P0-M5-E-H2a-03: StarFlow LIBERO 4-in-1 Qwen3VL-4B LayerwiseFM ft=32（bs32）

> **实验代号**: E-H2a-03 / P0-M5
> **状态**: 🟢 训练运行中
> **run_id**: `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_260703_0848`
> **启动时间**: 2026-07-03 08:48 CST
> **当前更新**: 2026-07-05 02:27 CST
> **tmux 会话**: `train-0`
> **配置来源**: `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml`

---

## 实验概述

P0-M5 **Stage 1 StarFlowVLA 默认路径**：使用 LayerwiseFM (DiT) action model 在 LIBERO 4-in-1 数据集上从头训练。

本 run 为 **从 scratch 全新训练**（`is_resume=False`），在 4090 上使用 bs32 配置（per_device_batch_size=1 × gradient_accumulation_steps=32）。这是 P0-M5-E-H2a-03 的第三个 run，前两个 run 在 A100 上训练至 step ~28000 后因设备切换废弃。

---

## 训练参数

### 脚本级参数

| 参数 | 值 |
|------|-----|
| `RUN_ID` | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_260703_0848` |
| `CONFIG_YAML` | `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml` |
| `DATA_MIX` | `libero_all` |
| `MAX_TRAIN_STEPS` | 80,000 |
| `SAVE_INTERVAL` | 250 |
| `EVAL_INTERVAL` | 500 |
| `LOGGING_FREQUENCY` | 20 |
| `GRADIENT_ACCUMULATION_STEPS` | **32** |
| `PER_DEVICE_BATCH_SIZE` | **1** |
| `NUM_WORKERS` | 2 |
| `BASE_VLM` | `/disk/rl/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct` |
| `LIBERO_DATA_ROOT` | `/disk/rl/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA` |
| `WANDB_PROJECT` | `starflow_vla` |
| `WANDB_ENTITY` | `silencewx-harbin-institute-of-technology` |
| `is_resume` | **`False`**（从 scratch） |
| `eval_num_batches` | 8 |
| 设备 | NVIDIA GeForce RTX 4090（24564 MiB） |
| 单价 | 本地 4090，暂不记录 |

### 模型 / 优化器参数

| 参数 | 值 |
|------|-----|
| **框架** | StarFlowVLA |
| **Base VLM** | Qwen3-VL-4B-Instruct |
| **Action Model** | LayerwiseFM (DiT, 36 layers, 1024 hidden) |
| **Action Dim** | 7 |
| **State Dim** | 8 |
| **Action Horizon** | 8 |
| **Num Target Vision Tokens** | 32 |
| **DiT Attention Heads** | 16 |
| **Dropout** | 0.2 |
| **Frozen Modules** | `qwen_vl_interface` |
| **Learning Rate (action_model)** | 1.0e-4 |
| **Learning Rate (base)** | 2.5e-5 |
| **Learning Rate (qwen_vl_interface)** | 1.0e-5 |
| **LR Scheduler** | cosine_with_min_lr |
| **Min LR** | 1.0e-6 |
| **Warmup Steps** | 0 |
| **Optimizer** | AdamW (β=(0.9, 0.95), eps=1e-8, wd=1e-8) |
| **Max Grad Norm** | 1.0 |
| **Gradient Checkpointing** | true |
| **Mixed Precision** | no |
| **Effective Global Batch** | **32**（1 × 32） |
| **Checkpoint Format** | lightweight |
| **Save Format** | safetensors |
| **local_checkpoint_root** | `/localdisk-tmp` |
| **local_checkpoint_keep_count** | 1 |
| **Seed** | 42 |

### 数据集

| 数据集 | 样本数 | embodiment |
|--------|--------|------------|
| `libero_object_no_noops_1.0.0_lerobot` | 66,984 | FRANKA |
| `libero_goal_no_noops_1.0.0_lerobot` | 52,042 | FRANKA |
| `libero_spatial_no_noops_1.0.0_lerobot` | 52,970 | FRANKA |
| `libero_10_no_noops_1.0.0_lerobot` | 101,469 | FRANKA |
| **合计** | **273,465** | — |

---

## 训练进度

> 最后更新：2026-07-03 10:16 CST

| 指标 | 值 |
|------|-----|
| **当前 Step** | **21260 / 80000**（26.6%） |
| **完成比例** | 26.5% |
| **单步耗时** | ~7.0 s/it |
| **数据加载耗时** | ~0.000 s |
| **模型前向/反向耗时** | ~0.232 s |
| **已运行时间** | 约 33 小时 28 分钟 |
| **预计剩余时间** | ~126 小时（约 5.3 天） |

---

## 成本估算

| 项目 | 计算 |
|------|------|
| **每 step 耗时** | ~7.34 s/it |
| **每 step 成本** | 本地 4090，暂不记录 |
| **已运行时间** | 约 33 小时 28 分钟 |
| **完整 80000 steps 预估** | 80000 × 7.34s ≈ 163h ≈ 6.8 天 |

---

## 输出目录

```
/disk/rl/starVLA/playground/Checkpoints/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_260703_0848/
├── config.full.yaml          ✅
├── config.yaml               ✅
├── dataset_statistics.json   ✅
├── summary.jsonl             ✅
└── wandb/                    ✅
```

本地暂存：`/localdisk-tmp/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_260703_0848/`

---

## 相关链接

- **WandB Run**: [260703_0848](https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla/runs/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_260703_0848)
- **tmux 会话**: `train-0`

---

## 启动命令

```bash
cd /disk/rl/starVLA
RUN_ID="P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_$(date +%y%m%d_%H%M)" \
CONFIG_YAML=configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml \
DATA_MIX=libero_all \
MAX_TRAIN_STEPS=80000 \
SAVE_INTERVAL=250 \
LOGGING_FREQUENCY=20 \
EVAL_INTERVAL=500 \
GRADIENT_ACCUMULATION_STEPS=32 \
PER_DEVICE_BATCH_SIZE=1 \
NUM_WORKERS=2 \
LOCAL_CHECKPOINT_KEEP_COUNT=1 \
LOCAL_CHECKPOINT_ROOT=/localdisk-tmp \
STARVLA_PYTHON=/opt/conda/envs/starVLA/bin/python \
bash examples/LIBERO/train_files/run_starflow_train_ready.sh
```

---

## 历史 run 参考

| Run | 设备 | 最终 step | 状态 |
|-----|------|----------|------|
| `P0-M5-E-H2a-03_..._250619` | A100 (80G) | ~27407 | 已废弃，设备切换 |
| `P0-M5-E-H2a-03_..._260702_...` | RTX 4090 | ~28381 | 已废弃，从 A100 resume 后 WandB step 回退 |
| **`P0-M5-E-H2a-03_..._260703_0848`** | RTX 4090 | — | 🟢 当前，从 scratch 全新训练 |

---

## 备注

- 本实验为 P0-M5 默认 StarFlowVLA 配置，使用 LayerwiseFM + Qwen3-VL-4B。
- **从 scratch 全新训练**，不从旧 run resume，避免 WandB step 回退问题。
- 显存使用率 96.2%（23.6G/24G），余量仅 442 MiB，接近 OOM 边界。
- Docker 内存用量 27.8 GiB / 56.0 GiB（cgroup 实际用量）。
- gradient accumulation fix（commit `abe1e46`）已应用。

---

*本文档将持续更新。*
