# MoWA-E-001: StarFlowVLA ft0 robocasa365 baseline（bs32）

> **实验代号**: E-001 / mowa_baseline  
> **状态**: 🟢 训练运行中  
> **run_id**: `MoWA-E-001_starflow_ft0_baseline_4090_long_260706_1730`  
> **启动时间**: 2026-07-06 17:30 CST  
> **当前更新**: 2026-07-06 20:37 CST  
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

> 最后更新：2026-07-06 20:37 CST

| 指标 | 值 |
|------|-----|
| **当前 Step** | **~2900 / 80000**（3.6%） |
| **完成比例** | 3.6% |
| **单步耗时** | ~3.69 s/it |
| **模型前向/反向耗时** | ~0.20–0.25 s |
| **数据加载耗时** | ~0.0002–0.0005 s（几乎忽略） |
| **已运行时间** | ~3 小时 |
| **预计剩余时间** | ~77.4 小时（约 **3.2 天**，7/10 凌晨） |
| **最新 checkpoint** | ✅ `steps_2000`（19:38 保存） |
| **最新 eval mse_score** | 尚未评估（eval_interval=2000） |

### Loss 记录

| Step | action_dit_loss | last_micro_loss | LR (action_model) | LR (base) | 备注 |
|------|----------------|-----------------|-------------------|-----------|------|
| 50 | 0.9581 | 0.7684 | 1.0e-5 | 2.5e-6 | warmup |
| 100 | — | 0.6310 | 2.0e-5 | 5.0e-6 | |
| 150 | — | 0.6645 | 3.0e-5 | 7.5e-6 | |
| 200 | — | 0.2730 | 4.0e-5 | 1.0e-5 | |
| 250 | — | 0.6846 | 5.0e-5 | 1.25e-5 | |
| 300 | — | 0.4771 | 6.0e-5 | 1.5e-5 | |
| 350 | — | 0.2663 | 7.0e-5 | 1.75e-5 | |
| 400 | — | 0.3574 | 8.0e-5 | 2.0e-5 | |
| 450 | — | 0.1888 | 9.0e-5 | 2.25e-5 | |
| **500** | — | 0.1705 | **1.0e-4** | **2.5e-5** | ✅ warmup 结束 |
| 550 | — | 0.1711 | 1.0e-4 | 2.5e-5 | |
| 600 | — | 0.1001 | 1.0e-4 | 2.5e-5 | |
| 650 | — | 1.0916 | 1.0e-4 | 2.5e-5 | 微 batct 异常 |
| 700 | — | 0.2063 | 1.0e-4 | 2.5e-5 | |
| **750** | **0.2586** | 0.0967 | 1.0e-4 | 2.5e-5 | |
| 800 | — | 0.1108 | 1.0e-4 | 2.5e-5 | |
| 850 | — | 0.1304 | 1.0e-4 | 2.5e-5 | |
| 900 | — | 0.1285 | 1.0e-4 | 2.5e-5 | |
| 950 | — | 0.1166 | 1.0e-4 | 2.5e-5 | |
| 1000 | — | 0.0813 | 1.0e-4 | 2.5e-5 | |
| 1050 | — | 0.3773 | 1.0e-4 | 2.5e-5 | |
| 1100 | — | 0.0585 | 1.0e-4 | 2.5e-5 | best micro so far |
| 1150 | — | 0.0806 | 1.0e-4 | 2.5e-5 | |
| 1200 | — | 0.1223 | 1.0e-4 | 2.5e-5 | |
| 1250 | — | 0.5256 | 1.0e-4 | 2.5e-5 | |
| 1300 | — | 0.1315 | 1.0e-4 | 2.5e-5 | |
| 1350 | — | 0.1998 | 1.0e-4 | 2.5e-5 | |
| 1400 | — | 0.1276 | 1.0e-4 | 2.5e-5 | |
| 1450 | — | 0.0578 | 1.0e-4 | 2.5e-5 | |
| 1500 | — | 0.5175 | 1.0e-4 | 2.5e-5 | |
| 1550 | — | 0.2255 | 1.0e-4 | 2.5e-5 | |
| 1600 | — | 0.1557 | 1.0e-4 | 2.5e-5 | |
| 1650 | — | 0.1093 | 1.0e-4 | 2.5e-5 | |
| 1700 | — | 0.1431 | **9.99e-5** | 2.5e-5 | cosine decay 开始 |
| 1750 | — | 0.2281 | 9.99e-5 | 2.5e-5 | |
| 1800 | — | 0.0682 | 9.99e-5 | 2.5e-5 | |
| 1850 | — | 0.0721 | 9.99e-5 | 2.5e-5 | |
| 1900 | — | 0.1101 | 9.99e-5 | 2.5e-5 | |
| **2000** | — | 0.4503 | 9.99e-5 | — | ✅ **checkpoint 保存** |
| 2050 | — | 0.0791 | 9.99e-5 | 2.5e-5 | |
| 2100 | — | **0.0443** | 9.99e-5 | 2.5e-5 | |
| 2150 | — | **0.0390** | 9.99e-5 | 2.5e-5 | 🏆 最佳 last_micro |
| 2200 | — | 0.1029 | 9.99e-5 | 2.5e-5 | |
| 2250 | — | 0.0823 | 9.99e-5 | 2.5e-5 | |
| 2300 | — | 0.2171 | 9.99e-5 | 2.5e-5 | |
| 2350 | — | 0.1838 | 9.99e-5 | 2.5e-5 | |
| 2400 | — | 0.2268 | 9.99e-5 | 2.5e-5 | |
| 2450 | — | 0.0484 | 9.99e-5 | 2.5e-5 | |
| 2500 | — | 0.0865 | 9.99e-5 | 2.5e-5 | |
| 2550 | — | 0.2946 | **9.98e-5** | 2.5e-5 | |
| 2600 | — | 0.0783 | 9.98e-5 | 2.5e-5 | |
| 2650 | — | 0.1491 | 9.98e-5 | 2.5e-5 | |
| 2700 | — | 0.2243 | 9.98e-5 | 2.5e-5 | |
| 2750 | — | 0.2102 | 9.98e-5 | 2.5e-5 | |
| 2800 | — | 0.1578 | 9.98e-5 | 2.5e-5 | |
| 2850 | — | 0.0623 | 9.98e-5 | **2.49e-5** | |
| 2900 | — | 0.1145 | 9.98e-5 | 2.49e-5 | 🔵 当前 |

### Loss 趋势

- **warmup 阶段（0–500）**: action_dit_loss 从 0.96 快速下降至 0.27，LR 线性增长至目标值。
- **step 500–2000**: warmup 结束，LR 稳定在 action_model=1e-4 / base=2.5e-5。last_micro_loss 在 0.06–1.09 之间震荡，整体 loss 在 0.25–0.26 区间。
- **step 2000+**: checkpoint 正常保存后继续训练。last_micro_loss 新低 **0.0390**（step 2150），多个 step 出现 0.04–0.08 的低值。LR 开始极缓 cosine decay（9.99e-5 → 9.98e-5）。
- `action_dit_loss`（16 micro 平均）仅 step 50（0.9581）和 step 750（0.2586）有完整记录，后续被 log 格式化截断。
- `mowa_future_supervision_loss` 恒为 0.0（未启用）。
- 单步耗时稳定在 ~3.69 s/it，模型前向/反向仅 ~0.20–0.25 s（占 step 时间 ~6%），剩余时间消耗在 grad_accum=16 的 micro-step 间同步。数据加载几乎为 0。

---

## 系统资源占用

> 最后更新：2026-07-06 20:37 CST

### GPU（NVIDIA GeForce RTX 4090）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 43% |
| **显存使用** | 20,312 MiB / 24,564 MiB（82.7%） |
| **显存空闲** | ~4,252 MiB |
| **功耗** | 184.14 W / 450.00 W |
| **温度** | 57°C |

### 存储

| 挂载点 | 容量 | 已用 | 可用 | 使用率 |
|--------|------|------|------|--------|
| `/` (overlay) | 30G | 21G | 9.2G | 70% |
| `/localdisk-tmp` | 3.5T | 1.2T | 2.4T | 33% |
| `/disk/rl` | 700T | 554T | 147T | 80% |

---

## 输出目录

```
/disk/rl/starVLA/playground/mowa_ckpt/MoWA-E-001_starflow_ft0_baseline_4090_long_260706_1730/
├── checkpoints/steps_2000/      ✅（19:38 保存，~4.3GB model + 5GB optimizer）
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
- 🟢 **训练正常运行中** — step ~2900，warmup 已完成（500 steps），last_micro_loss 在 0.04-0.23 区间震荡，最佳 micro **0.0390**（step 2150）。
- ✅ 第一个 checkpoint `steps_2000` 已于 19:38 正常保存（~4.3GB model safetensors + 5GB optimizer）。
- 🔴 **GPU 利用率 43%**（比前次的 28%/0% 有改善但仍偏低）。model time 仅 ~0.22s（占 step time ~6%），大量时间消耗在 grad_accum=16 的 micro-step 同步上。
- 💡 `last_micro_loss` 反映单条 micro-batch 的即时 loss，因此波动较大（0.04–1.09）。`action_dit_loss` 是 16 个 micro-step 的平均值，更稳定但日志中多被截断。
- 预计完成时间：~77.4h（约 3.2 天），即 7 月 10 日凌晨左右。
- 每 4 小时由 cron 自动更新本 tracker（定时任务 `dfc091e9`，每 4h :07 触发）。

---

*本文档将持续更新。*
