# MoWA-E-001: StarFlowVLA ft0 mowa_main（robocasa365, bs32, 带 future supervision）

> **实验代号**: E-001 / mowa_main
> **状态**: 🟢 训练运行中
> **run_id**: `MoWA-E-001_starflow_ft0_mowa_4090_long_260706_1700`
> **启动时间**: 2026-07-06 17:00 CST
> **当前更新**: 2026-07-06 17:35 CST
> **进程**: PID 284059（4 个子进程），运行于 `pts/2`（建议迁入 tmux）
> **配置来源**: `configs/mowa/mowa_e001_starflow_ft0_long_training_candidate.yaml`

---

## 实验概述

MoWA E-001 **mowa_main**：在 StarFlowVLA 框架中使用 FT0 变体（动作头 `future_action_window_size=7`，`past_action_window_size=0`），在 **robocasa365** 数据集上训练 VLA 模型。

**本 run 为 mowa 主实验**（`experiment_role: mowa_main`），区别于 baseline：
- ✅ **`enable_future_supervision_loss: true`** — 未来 token 的额外监督信号参与训练
- 使用 `mowa_future_feature_heads` 作为 bridge token 特征源
- `layerwise_bridge_active_heads`: task_progress, action_outcome_class
- `enable_layerwise_bridge_token_coupling: true`

Batch size 配置：bs32（per_device_batch_size=2 × gradient_accumulation_steps=16），单 GPU 有效并行，适配 RTX 4090 24GB 显存。

---

## 训练参数

### 脚本级参数

| 参数 | 值 |
|------|-----|
| `run_id` | `MoWA-E-001_starflow_ft0_mowa_4090_long_260706_1700` |
| `CONFIG_YAML` | `configs/mowa/mowa_e001_starflow_ft0_long_training_candidate.yaml` |
| `DATA_MIX` | `robocasa365_atomic_target_human_all` |
| `MAX_TRAIN_STEPS` | 80000 |
| `SAVE_INTERVAL` | 2,000 |
| `EVAL_INTERVAL` | 2,000 |
| `LOGGING_FREQUENCY` | 50 |
| `NUM_PROCESSES` | **1**（单卡训练，无分布式） |
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
| **Action Dim** | 12（含 gripper；注意 dim11 恒为 0.0，可能是 padding 维度） |
| **State Dim** | 16（含 instruction embedding） |
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
| **Effective Global Batch** | **32**（per_device=2 × grad_accum=16 × num_processes=1） |
| **Checkpoint Format** | lightweight |
| **Save Format** | safetensors |
| **Seed** | 42 |

### MoWA 特有参数

| 参数 | 值 |
|------|-----|
| `enable_future_supervision_loss` | **true** ✅ |
| `enable_layerwise_bridge_token_coupling` | true |
| `layerwise_bridge_feature_source` | `mowa_future_feature_heads` |
| `layerwise_bridge_active_heads` | task_progress, action_outcome_class |
| `wam_feature_dim` | 1024 |
| `action_hidden_dim` | 1024 |
| `num_bridge_tokens` | 2 |
| `loss_scale.mowa_future_supervision` | 1.0 |

### 数据集

| 数据集 | 样本数（transitions）| trajectories | embodiment |
|--------|---------------------|--------------|------------|
| `robocasa365_atomic_target_human_all` | 2,231,347 | 9,126 | new_embodiment |

---

## 训练进度

> 最后更新：2026-07-06 17:35 CST
> 训练启动 35分钟，log 更新每 50 steps

| 指标 | 值 |
|------|-----|
| **当前 Step** | ~223 / 80000 |
| **完成比例** | 0.28% |
| **单步耗时** | 4.72s/it |
| **model_time** | 0.402s |
| **data_time** | 0.000s |
| **已运行时间** | 35分钟 |
| **预计剩余时间** | 104h 35m |
| **最新 checkpoint** | 尚未保存（共 0 个） |
| **最新 eval mse_score** | 首次 eval 在 step 2000 |

### Loss 记录

| Step | action_dit_loss | mowa_future_supervision_loss | 备注 |
|------|----------------|------------------------------|------|
| 223 | 0.9083662442862988 | — |  |
| 223 | 0.7216371316462755 | — |  |
| 223 | 0.3781107012182474 | — |  |

---

## 系统资源占用

> 最后更新：2026-07-06 17:35 CST

### GPU（NVIDIA GeForce RTX 4090）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 0% |
| **显存使用** | 20198 MiB / 24564 MiB (82%) |
| **功耗** | 147.77 W |
| **温度** | 44°C |

### 系统内存

| 指标 | 值 |
|------|-----|
| **总量** | 503Gi |
| **已用** | 34Gi |
| **可用** | 465Gi |

### 存储

| 挂载点 | 使用情况 |
|--------|----------|
| `/` (overlay) | 4.3G / 30G (15%) |
| `/localdisk-tmp` | 0 / 100G (0%) |
| `/disk/rl` | 554T / 700T (80%) |

---

## 输出目录

```
playground/mowa_ckpt/MoWA-E-001_starflow_ft0_mowa_4090_long_260706_1700/
├── checkpoints/                 （0 个 checkpoint，最新: 尚未保存）
├── config.full.yaml             ✅
├── config.yaml                  ✅
├── dataset_statistics.json      ✅
└── wandb/                       ✅
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
  --config_yaml configs/mowa/mowa_e001_starflow_ft0_long_training_candidate.yaml
```

---

## 备注

- 本实验为 MoWA 项目 E-001 的 **main 实验**，在 robocasa365 数据集上使用 StarFlowVLA（ft0 variant）训练 VLA 模型。
- **与 baseline 的关键区别**：`enable_future_supervision_loss: true`，即 MoWA future token 额外监督参与训练。
- 使用 mowa_future_feature_heads 作为 bridge token 特征源，含 task_progress 和 action_outcome_class 两个 active head。
- `per_device_batch_size=2`，`grad_accum=16`，`num_processes=1`，`num_workers=3`。RTX 4090 单卡无分布式训练，3 个 `pt_data_worker` 子进程为 PyTorch DataLoader 数据预读取进程。
- Action dim 12（含 gripper），注意 action dim 11 恒为 0.0（可能是 padding 维度）。
- **训练不在 tmux 中**，运行于 `pts/2` 直接启动。建议迁入 tmux `train` 会话以防 pts 断开导致中断。如需迁移：`pkill -f train_starvla` 后重新在 tmux 内启动。
- 每小时由 cron 自动更新本 tracker。

---

*本文档将持续更新。*
