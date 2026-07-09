# MoWA-E-001: StarFlowVLA ft0 robocasa365 baseline（bs32）

> **实验代号**: E-001 / mowa_baseline  
> **状态**: 🟢 训练运行中  
> **run_id**: `MoWA-E-001_starflow_ft0_baseline_4090_long_260706_1730`  
> **启动时间**: 2026-07-06 17:30 CST  
> **当前更新**: 2026-07-09 16:37 CST  
> **进程**: 训练进程于 `pts/4` 直接运行（无 tmux 会话）  
> **配置来源**: `configs/mowa/mowa_e001_starflow_ft0_baseline_long_training_candidate.yaml`

---

## 实验概述

MoWA E-001 **StarFlowVLA ft0 baseline（robocasa365）**：在 StarFlowVLA 框架中使用 FT0 变体（即动作头 `future_action_window_size=7`，`past_action_window_size=0`），在 **robocasa365** 数据集上训练 VLA 模型。

本 run 为 **从 scratch 全新训练**（`is_resume=False`），使用 bs32 配置（per_device_batch_size=2 × gradient_accumulation_steps=16），单卡 RTX 4090 24GB 显存。

**重要配置差异**：`mowa_future_supervision_loss` 未启用（`enable_future_supervision_loss: false`），未来 token 的监督信号不参与训练。

> ⚠️ 前一个 run（`260706_1650`）已在 ~17:20 因未知原因退出后被重启，此为重启后的新 run。

---

## 训练参数

### 脚本级参数

| 参数 | 值 |
|------|-----|
| `run_id` | `MoWA-E-001_starflow_ft0_baseline_4090_long_260706_1730` |
| `CONFIG_YAML` | `configs/mowa/mowa_e001_starflow_ft0_baseline_long_training_candidate.yaml` |
| `DATA_MIX` | `robocasa365_atomic_target_human_all` |
| `MAX_TRAIN_STEPS` | 80,000 |
| `SAVE_INTERVAL` | 2,000 |
| `EVAL_INTERVAL` | 2,000 |
| `LOGGING_FREQUENCY` | 50 |
| `NUM_PROCESSES` | **1**（单卡单进程；额外 3 个进程为 DataLoader worker `num_workers=3`） |
| `GRADIENT_ACCUMULATION_STEPS` | **16** |
| `PER_DEVICE_BATCH_SIZE` | **2** |
| `NUM_WORKERS` | 3 |
| `BASE_VLM` | `./playground/Pretrained_models/Qwen3-VL-4B-Instruct` |
| `DATA_ROOT` | `playground/Datasets/robocasa365` |
| `WANDB_PROJECT` | `MoWA` |
| `WANDB_ENTITY` | `silencewx-harbin-institute-of-technology` |
| `is_resume` | **`False`**（从 scratch） |
| 设备 | NVIDIA GeForce RTX 4090（24564 MiB） |
| 单价 | 本地 RTX 4090，暂不记录 |

### 模型 / 优化器参数

| 参数 | 值 |
|------|-----|
| **框架** | StarFlowVLA（`ft0` variant） |
| **Base VLM** | Qwen3-VL-4B-Instruct |
| **Action Model** | LayerwiseFM (DiT, 36 layers, 1024 hidden, 16 heads) |
| **Action Dim** | 12 |
| **State Dim** | 16 |
| **Action Horizon** | 8 |
| **Future Action Window Size** | 7（FT0） |
| **Past Action Window Size** | 0 |
| **Num Target Vision Tokens** | 0 |
| **Noise Schedule** | BetaAlpha（α=1.5, β=1.0, s=0.999） |
| **Inference Timesteps** | 4 |
| **DiT Dropout** | 0.2 |
| **Frozen Modules** | `qwen_vl_interface` |
| **Learning Rate (action_model)** | 1.0e-4 |
| **Learning Rate (base)** | 2.5e-5 |
| **Learning Rate (qwen_vl_interface)** | 1.0e-5 |
| **LR Scheduler** | cosine_with_min_lr |
| **Min LR** | 1.0e-6 |
| **Warmup Steps** | 500 |
| **Optimizer** | AdamW（β=(0.9, 0.95), eps=1e-8, wd=0） |
| **Max Grad Norm** | 1.0 |
| **Gradient Checkpointing** | true |
| **Gradient Clipping** | 1.0 |
| **Effective Global Batch** | **32**（2 × 16） |
| **Checkpoint Format** | lightweight |
| **Save Format** | safetensors |
| **Seed** | 42 |

### 数据集

| 数据集 | 样本数（transitions）| trajectories | embodiment |
|--------|---------------------|--------------|------------|
| `robocasa365_atomic_target_human_all` | 2,231,347 | 9,126 | new_embodiment |

---

## 训练进度

> 最后更新：2026-07-09 16:37 CST

| 指标 | 值 |
|------|-----|
| **当前 Step** | **~67950 / 80000**（84.9%） |
| **完成比例** | 84.9% ✅ **接近 85%** |
| **单步耗时** | ~3.82 s/it |
| **模型前向/反向耗时** | ~0.20–0.25 s |
| **数据加载耗时** | ~0.0002–0.0005 s（几乎忽略） |
| **已运行时间** | ~71 小时（约 3 天） |
| **预计剩余时间** | ~12.8 小时（约 **0.5 天**，**训练即将结束！** ）|
| **最新 checkpoint** | ✅ `steps_66000`（14:31 保存） |
| **最新 eval mse_score** | 始终未见 eval 输出 |

### Loss 记录

**完整记录共 1359 个数据点**（step 50–67950）。以下列出 milestone 及关键低点：

| Step | action_dit_loss | last_micro_loss | LR (action_model) | LR (base) | 备注 |
|------|----------------|-----------------|-------------------|-----------|------|
| 50 | 0.9581 | 0.7684 | 1.0e-5 | 2.5e-6 | warmup |
| **58550** | — | **0.0031** | 2.06e-5 | 5.16e-6 | 🏆 **全程最佳** |
| 64000 | — | — | — | — | ✅ checkpoint（12:24）|
| 64550 | — | **0.0057** | 1.29e-5 | 3.22e-6 | |
| 64650 | — | **0.0077** | 1.27e-5 | 3.18e-6 | |
| 64950 | — | **0.0078** | 1.22e-5 | 3.06e-6 | |
| 65700 | — | **0.0080** | 1.13e-5 | 2.82e-6 | |
| 65800 | — | **0.0057** | 1.12e-5 | 2.79e-6 | |
| 66000 | — | — | — | — | ✅ checkpoint（14:31）|
| 66250 | — | **0.0067** | 1.08e-5 | 2.70e-6 | |
| 67150 | — | **0.0056** | 1.01e-5 | 2.53e-6 | |
| 67450 | — | **0.0067** | 9.78e-6 | 2.45e-6 | |
| 67850 | — | **0.0067** | 9.43e-6 | 2.36e-6 | |
| 67950 | — | 0.0493 | 9.34e-6 | 2.33e-6 | 🔵 当前 |

### Loss 趋势

- 🏆 **最佳 last_micro_loss 保持 0.0031**（step 58550）。近期仍有多个 < 0.01 低值（step 64550: 0.0057, step 67150: 0.0056, step 67850: 0.0067）。
- ✅ **84.9% 完成**！训练接近终点（剩余 ~12.8h）。
- **Cosine LR decay 超 90%**: action_model LR 1.0e-4 → **9.34e-6**（↓91%），base LR 2.5e-5 → **2.33e-6**（↓91%）。
- **Epoch**: 已完成 **0.62 epoch**（~62% 数据遍历）。
- 近期 loss 在 0.0056–0.28 区间，大部分稳定在 0.005–0.08。

---

## 系统资源占用

> 最后更新：2026-07-09 16:37 CST

### GPU（NVIDIA GeForce RTX 4090）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 5%（测量瞬时）|
| **显存使用** | 20,312 MiB / 24,564 MiB（82.7%） |
| **显存空闲** | ~4,252 MiB |
| **功耗** | 177.58 W / 450.00 W |
| **温度** | 55°C |

### 存储

| 挂载点 | 容量 | 已用 | 可用 | 使用率 |
|--------|------|------|------|--------|
| `/` (overlay) | 30G | 22G | 8.8G | 71% |
| `/localdisk-tmp` | 3.5T | 1.2T | 2.4T | 33% |
| `/disk/rl` | 700T | 555T | 146T | 80% |

---

## 输出目录

```
/disk/rl/starVLA/playground/mowa_ckpt/MoWA-E-001_starflow_ft0_baseline_4090_long_260706_1730/
├── checkpoints/（部分旧 checkpoint 已被清理）
│   ├── steps_6000/       ✅（Jul6 23:55）
│   ├── steps_10000/      ✅（Jul7 04:18）
│   ├── steps_20000/      ✅（Jul7 14:46）
│   ├── steps_24000/      ✅（Jul7 18:57）
│   ├── steps_26000/      ✅（Jul7 21:01）
│   ├── steps_28000/      ✅（Jul7 23:06）
│   ├── steps_30000/      ✅（Jul8 01:11）
│   ├── steps_32000/      ✅（Jul8 03:15）
│   ├── steps_34000/      ✅（Jul8 05:19）
│   ├── steps_36000/      ✅（Jul8 07:23）
│   ├── steps_38000/      ✅（Jul8 09:28）
│   ├── steps_40000/      ✅（Jul8 11:31）
│   ├── steps_42000/      ✅（Jul8 13:35）
│   ├── steps_44000/      ✅（Jul8 15:39）
│   ├── steps_46000/      ✅（Jul8 17:44）
│   ├── steps_48000/      ✅（Jul8 19:48）
│   ├── steps_50000/      ✅（Jul8 21:51）
│   ├── steps_52000/      ✅（Jul8 23:56）
│   ├── steps_54000/      ✅（Jul9 01:59）
│   ├── steps_56000/      ✅（Jul9 04:03）
│   ├── steps_58000/      ✅（Jul9 06:08）
│   ├── steps_60000/      ✅（Jul9 08:12）
│   ├── steps_62000/      ✅（Jul9 10:17）
│   ├── steps_64000/      ✅（Jul9 12:24）
│   └── steps_66000/      ✅（Jul9 14:31）
├── config.full.yaml             ✅
├── config.yaml                  ✅
├── dataset_statistics.json      ✅
└── wandb/                       ✅（运行中）
```

---

## 相关链接

- **WandB Project**: [MoWA](https://wandb.ai/silencewx-harbin-institute-of-technology/MoWA)
- **训练主机**: 本地服务器（RTX 4090）

---

## 启动命令

```bash
cd /disk/rl/starVLA
python starVLA/training/train_starvla.py \
  --config_yaml configs/mowa/mowa_e001_starflow_ft0_baseline_long_training_candidate.yaml
```

---

## 备注

- 本实验为 MoWA 项目 E-001 的正式 long-training baseline，在 robocasa365 数据集上使用 StarFlowVLA（ft0 variant）训练 VLA 模型。
- 与 starflow_vla 的 libero 实验不同，本 run 使用：
  - robocasa365 数据集（2.2M transition，9K trajectory），single embodiment
  - Action dim 12（含 gripper），State dim 16（含 instruction embedding）
  - per_device_batch_size=2，grad_accum=16
- `enable_future_supervision_loss: false`，即不做 future token 的额外监督。
- 🟢 **训练正常运行中** — step ~67950 / 80000（**84.9% ✅**），已稳定运行 71h+（约 3 天）。
- 🏆 **最佳 last_micro_loss: 0.0031**（step 58550）。近期仍有多个 < 0.01 低值（0.0056, 0.0057, 0.0067）。
- ✅ 最新 checkpoint `steps_66000`（14:31）。**训练接近终点！**
- 📉 **Cosine LR decay 超 90%**: action_model LR 1.0e-4 → **9.34e-6**（↓91%），base LR 2.5e-5 → **2.33e-6**（↓91%）。
- 🟢 显存稳定 82.7%。
- ⚠️ 全程未见 eval mse_score 输出。
- 预计完成时间：~12.8h（约 0.5 天），即 **7 月 10 日凌晨/早间** 🏁。
- 每 4 小时由 cron 自动更新本 tracker（定时任务 `dfc091e9`，每 4h :07 触发）。

---

*本文档将持续更新。*
