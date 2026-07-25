# MoWA-E-001: StarFlowVLA ft0 robocasa365 baseline（bs32）

> **实验代号**: E-001 / mowa_baseline  
> **状态**: 🟢 训练运行中  
> **run_id**: `MoWA-E-001_starflow_ft0_baseline_4090_long_260706_1650`  
> **启动时间**: 2026-07-06 16:57 CST  
> **当前更新**: 2026-07-06 17:22 CST
> **进程**: 训练进程于 `pts/4` 直接运行（无 tmux 会话）  
> **配置来源**: `configs/mowa/mowa_e001_starflow_ft0_baseline_long_training_candidate.yaml`

---

## 实验概述

MoWA E-001 **StarFlowVLA ft0 baseline（robocasa365）**：在 StarFlowVLA 框架中使用 FT0 变体（即动作头 `future_action_window_size=7`，`past_action_window_size=0`），在 **robocasa365** 数据集上训练 VLA 模型。

本 run 为 **从 scratch 全新训练**（`is_resume=False`），使用 bs32 配置（per_device_batch_size=2 × gradient_accumulation_steps=16 × num_processes=4），适配 RTX 4090 24GB 显存。

**重要配置差异**：`mowa_future_supervision_loss` 未启用（`enable_future_supervision_loss: false`），未来 token 的监督信号不参与训练。

---

## 训练参数

### 脚本级参数

| 参数 | 值 |
|------|-----|
| `run_id` | `MoWA-E-001_starflow_ft0_baseline_4090_long_260706_1650` |
| `CONFIG_YAML` | `configs/mowa/mowa_e001_starflow_ft0_baseline_long_training_candidate.yaml` |
| `DATA_MIX` | `robocasa365_atomic_target_human_all` |
| `MAX_TRAIN_STEPS` | 80,000 |
| `SAVE_INTERVAL` | 2,000 |
| `EVAL_INTERVAL` | 2,000 |
| `LOGGING_FREQUENCY` | 50 |
| `NUM_PROCESSES` | 4（DeepSpeed） |
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
| **Effective Global Batch** | **32**（2 × 16 × 4 / gradient_accumulation_steps=16） |
| **Checkpoint Format** | lightweight |
| **Save Format** | safetensors |
| **Seed** | 42 |

### 数据集

| 数据集 | 样本数（transitions）| trajectories | embodiment |
|--------|---------------------|--------------|------------|
| `robocasa365_atomic_target_human_all` | 2,231,347 | 9,126 | new_embodiment |

---

## 训练进度

> 最后更新：2026-07-06 17:22 CST

| 指标 | 值 |
|------|-----|
| **当前 Step** | **~350 / 80000**（0.4%） |
| **完成比例** | 0.4% |
| **单步耗时** | ~3.87 s/it |
| **模型前向/反向耗时** | ~0.22–0.42 s |
| **数据加载耗时** | ~0.0002–0.0004 s（几乎忽略） |
| **已运行时间** | ~25 分钟 |
| **预计剩余时间** | ~85.7 小时（约 3.6 天） |
| **最新 checkpoint** | 尚未保存（save_interval=2000） |
| **最新 eval mse_score** | 尚未评估（eval_interval=2000） |

### Loss 记录

| Step | action_dit_loss | last_micro_loss | LR (action_model) | LR (base) | 备注 |
|------|----------------|-----------------|-------------------|-----------|------|
| 50 | 0.9404 | 0.8234 | 1.0e-5 | 2.5e-6 | 初始（warmup 阶段）|
| 100 | 0.7244 | 0.6159 | 2.0e-5 | 5.0e-6 | |
| 150 | 0.5015 | 0.6381 | 3.0e-5 | 7.5e-6 | |
| 200 | 0.4051 | 0.3220 | 4.0e-5 | 1.0e-5 | |
| 250 | 0.4242 | 0.7024 | 5.0e-5 | 1.25e-5 | |
| 300 | 0.5311 | 0.5555 | 6.0e-5 | 1.5e-5 | |
| 350 | 0.3948 | 0.2541 | 7.0e-5 | 1.75e-5 | |

### Loss 趋势

- **Phase 1（0–350 steps）**: 快速下降阶段，action_dit_loss 从 0.94 降至 ~0.39–0.53，呈现暖机（warmup）中的震荡下降。
- 当前处于 **warmup 阶段（0–500 steps）**，LR 从 0 线性增长至目标值。action_model LR 已升至 7e-5（目标 1e-4），base LR 升至 1.75e-5（目标 2.5e-5）。
- `mowa_future_supervision_loss` 恒为 0.0（该 loss 未启用）。
- `last_micro_loss` 在 0.25–0.82 之间震荡，因 grad_accum=16 且每个 micro 样本差异大。
- 单步耗时 ~3.87 s/it，但模型前向/反向仅 ~0.22–0.42 s，数据加载几乎为 0（~0.0003 s），剩余时间为 DeepSpeed 4 卡的同步/通信开销。

---

## 系统资源占用

> 最后更新：2026-07-06 17:22 CST

### GPU（NVIDIA GeForce RTX 4090）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 28% |
| **显存使用** | 20,312 MiB / 24,564 MiB（82.7%） |
| **显存空闲** | ~4,252 MiB |
| **功耗** | 209.77 W / 450.00 W |
| **温度** | 58°C |

### 存储

| 挂载点 | 容量 | 已用 | 可用 | 使用率 |
|--------|------|------|------|--------|
| `/` (overlay) | 30G | 21G | 9.2G | 70% |
| `/localdisk-tmp` | 3.5T | 1.2T | 2.4T | 33% |
| `/disk/rl` | 700T | 554T | 147T | 80% |

---

## 输出目录

```
/disk/rl/starVLA/playground/mowa_ckpt/MoWA-E-001_starflow_ft0_baseline_4090_long_260706_1650/
├── checkpoints/                 （暂无 checkpoint，save_interval=2000）
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
  - per_device_batch_size=2，grad_accum=16，4 卡 DeepSpeed 分布式
- `enable_future_supervision_loss: false`，即不做 future token 的额外监督。
- 🟢 **训练正常运行中** — step ~350（warmup 阶段），loss 从 0.94 降至 ~0.39，仍在 warmup（500 steps）。
- 🔴 **注意**：GPU 利用率仅 28%（偏低），显存占 82.7%~20.3GiB/24.6GiB。data loading 几乎为 0（~0.0003s），但 model time 仅 ~0.22–0.42s，大量时间可能花在 DeepSpeed 4 卡的梯度同步或 grad_accum 的 16 micro-step 之间。
- 预计完成时间：~85.7h（约 3.6 天），即 7 月 10 日凌晨左右。
- 每小时由 cron 自动更新本 tracker（定时任务 `82345eee`，每小时 :07 触发）。

---

*本文档将持续更新。*
