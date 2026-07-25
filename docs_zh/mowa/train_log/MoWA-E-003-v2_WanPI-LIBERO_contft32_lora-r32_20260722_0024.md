# MoWA E-003-v2: WanPI + contFT32 + LoRA rank=32 · 全18原子任务（robocasa365_atomic_target_human_all）

> **实验代号**: E-003-v2 / `WanPI-LIBERO_contft32_lora-r32`
> **状态**: 🟢 训练运行中（B3，从 B2 step 14500 续训）
> **run_id**: `MoWA-E-003-v2_WanPI-LIBERO_contft32_lora-r32_20260722_0024`
> **B3 启动**: 2026-07-22 14:07 CST（`is_resume=True`）
> **当前更新**: 2026-07-22 15:22 CST（step 16300，16.3%，续训正常）
> **tmux 会话**: `train`（accelerate 8 进程）
> **配置来源**: `configs/mowa/mowa_e003_v2_wanpi_contft32_lora.yaml`

---

## 实验概述

MoWA E-003-v2，基于 WanPI 框架 + contFT32 + LoRA rank=32。数据使用全 18 原子任务 `robocasa365_atomic_target_human_all`（559k transitions, 9,126 trajectories），8 × RTX 4090（NCCL, bf16）。

### 重启历史

| 代号 | 启动 | 中断 | 最大 step | 原因分析 |
|------|------|------|-----------|----------|
| **B2** | 07/22 00:48 | 07/22 10:24 | **14500** | 保存 checkpoint 后中断，log 无 error，疑似 tmux 被杀 |
| **B3** | 07/22 14:07 | — | **续训中** | **`is_resume=True`，从 B2 steps_14500 续训** |

B1/B2 均在 checkpoint 保存步骤附近中断，无 traceback。

---

## 训练事件

### B2（00:48–10:24 CST）

| 时间 (CST) | 事件 |
|------------|------|
| 2026-07-22 00:45 | 终端启动，config 已更新为全 18 任务 |
| 2026-07-22 00:49 | Data flow validation 1/2, 2/2 通过（T=9 ✅） |
| 2026-07-22 00:51 | Step 50，`action_dit_loss=0.936` |
| 2026-07-22 00:53 | Step 100，`action_dit_loss=0.490` |
| 2026-07-22 00:59 | Step 250，首次 eval，`mse=0.011`，`action_dit_loss=0.227` |
| 2026-07-22 01:11 | Step 500，`action_dit_loss=0.306`，eval `mse=0.011` |
| 2026-07-22 01:45 | Step 1000，`action_dit_loss=0.239`，eval `mse=0.006`，**ckpt steps_1000** |
| 2026-07-22 02:10 | Step 1500，`action_dit_loss=0.094`，warmup 接近完成 |
| 2026-07-22 02:25 | Step 2000，`action_dit_loss=0.057`，**warmup 完成**，**ckpt steps_2000** |
| 2026-07-22 03:10 | Step 3500，`action_dit_loss=0.087`，eval `mse=0.006`，loss 稳定 |
| 2026-07-22 04:20 | Step 5000，`action_dit_loss=0.106`，**ckpt steps_5000** |
| 2026-07-22 06:00 | Step 7500，`action_dit_loss=0.109`，eval `mse=0.007` |
| 2026-07-22 07:50 | Step 10000，`action_dit_loss=0.102`，**ckpt steps_10000** |
| 2026-07-22 09:24 | Step 13000，**ckpt steps_13000** |
| 2026-07-22 10:04 | Step 14000，**ckpt steps_14000**，action_dit_loss~0.10 |
| 2026-07-22 10:24 | Step 14500，`action_dit_loss=0.114`，eval `mse=0.007`，**ckpt steps_14500** — 保存完成后训练中断 🔴 |

### B3（14:07 CST → 续训中）

| 时间 (CST) | 事件 |
|------------|------|
| 2026-07-22 14:07 | 启动，`is_resume=True`，从 `steps_14500` checkpoint 恢复 |
| 2026-07-22 14:11 | wandb run 141124，data flow 1/2, 2/2 通过 |
| 2026-07-22 14:13 | Step 14550（续训首步），`action_dit_loss=0.128`，与 B2 末尾一致 ✅ |
| 2026-07-22 14:31 | Step 15000，**ckpt steps_15000**（B3 首存） |
| 2026-07-22 14:51 | Step 15500，**ckpt steps_15500** |
| 2026-07-22 15:11 | Step 16000，**ckpt steps_16000** |
| 2026-07-22 15:22 | Step 16300，`action_dit_loss=0.059`，eval `mse=0.0042` 🚀 |

---

## 训练参数

### 脚本级参数

| 参数 | 值 |
|------|-----|
| `run_id` | `MoWA-E-003-v2_WanPI-LIBERO_contft32_lora-r32_20260722_0024` |
| `DATA_MIX` | `robocasa365_atomic_target_human_all` |
| `MAX_TRAIN_STEPS` | 100000 |
| `SAVE_INTERVAL` | 1000 |
| `EVAL_INTERVAL` | 250 |
| `LOGGING_FREQUENCY` | 50 |
| `NUM_PROCESSES` | 8（NCCL，bf16） |
| `GRADIENT_ACCUMULATION_STEPS` | 2 |
| `PER_DEVICE_BATCH_SIZE` | 2 |
| `Effective Global Batch` | **32** |
| `NUM_WORKERS` | 32 |
| `BASE_WM` | Wan2.2-TI2V-5B-Diffusers |
| `is_resume` | false（从 scratch） |

### 模型参数

| 参数 | 值 |
|------|-----|
| 框架 | WanPI（v2） |
| starflow_ft_variant | ft0 |
| Action Model | LayerwiseFM（DiT，30 层，hidden 1024，16 heads） |
| Action Dim | 12（delta_ee） |
| State Dim | 32 |
| State Mode | continuous_head |
| Action Horizon | 32 |
| Future Window | 8 |
| History Window | 0 |
| Noise Schedule | BetaAlpha（α=1.5, β=1.0） |
| Inference Timesteps | 4 |
| Frozen | `backbone` |
| LR (action_model) | 1.0e-4 |
| LR (base) | 2.5e-5 |
| LR (wan_lora) | 1.0e-5 |
| LR Scheduler | cosine_with_min_lr（1.0e-6） |
| Warmup | 2000 steps |
| Optimizer | AdamW（β=(0.9, 0.95)） |

### Wan LoRA

| 参数 | 值 |
|------|-----|
| enabled | true |
| rank | 32 |
| alpha | 64 |
| dropout | 0.0 |
| layers | 0–全部 |
| targets | cross_attention, self_attention |

### MoWA 特有参数

| 参数 | 值 |
|------|-----|
| `enable_future_latent_prior_loss` | true（loss_scale=0.05） |
| `enable_future_supervision_loss` | false |
| `multi_view` | true（main / wrist 双视角，batch flatten） |
| `cross_view` | gate_init=1.0, num_layers=10 |
| `done_head.loss_weight` | 0.1 |

### Pretrained Checkpoint

| 参数 | 值 |
|------|-----|
| checkpoint | `WM4A-Wan2d2-OFT-LIBERO-4in1/steps_60000_pytorch_model.pt` |
| reload_modules | backbone |

### 数据规模

| 指标 | 值 |
|------|-----|
| 数据集 | robocasa365_atomic_target_human_all（18 任务） |
| Transitions | 558,946 |
| Trajectories | 9,126 |
| 动作类型 | delta_ee（12-dim） |

### vs B1 对比

| 项目 | B1（中断） | **B2（本次）** |
|------|-----------|---------------|
| run_id | ...2150 | **...0024** |
| 数据 | OpenDrawer（34k） | **全 18 任务（559k）** |
| 总步数 | 100k | 100k |
| 预计总时长 | ~65h | **~65h（epochs 更少）** |
| 中断前 | step 2350 | — |

---

## 训练指标（B2 完整记录）

| Step | action_dit_loss | loss_action near/far | loss_future main/wrist | eval mse | epoch | 备注 |
|------|----------------|---------------------|----------------------|----------|-------|------|
| 50 | 0.936 | — | — | — | — | warmup 早期 |
| 100 | 0.490 | — | — | — | — | |
| 250 | 0.227 | — | — | 0.011 | — | 首次 eval |
| 500 | 0.306 | — | — | 0.011 | — | |
| 750 | 0.163 | — | — | 0.009 | — | |
| 1000 | 0.239 | 0.104 / 0.106 | 0.426 / 0.566 | 0.006 | — | **ckpt #1** |
| 1500 | 0.094 | — | — | — | — | warmup 接近完成 |
| 2000 | 0.057 | 0.099 / 0.099 | 0.423 / 0.581 | — | — | **warmup 完成, ckpt #2** |
| 2500 | 0.121 | — | — | 0.009 | — | |
| 3500 | 0.087 | — | — | 0.006 | — | |
| 5000 | 0.106 | — | — | 0.008 | — | **ckpt #3** |
| 7500 | 0.109 | — | — | 0.007 | — | |
| 10000 | 0.102 | — | — | 0.007 | ~0.35 | **ckpt #4** |
| 12500 | 0.131 | — | — | 0.007 | — | **ckpt #5** |
| 13000 | — | — | — | 0.008 | — | **ckpt #6** |
| 14000 | 0.128 | — | — | — | ~0.51 | **ckpt #7** |
| 14450 | 0.128 | — | — | — | — | |
| 14500 | 0.114 | 0.099 / 0.101 | 0.425 / 0.591 | 0.007 | 0.53 | B2 最后记录 🔴 |
| **B3 resume** → | | | | | | |
| 14550 | 0.128 | — | 0.352 / 0.712 | — | — | 续训首步，一致性 ✅ |
| 15000 | — | — | — | — | — | **B3 ckpt #1** |
| 15500 | — | — | — | — | — | **B3 ckpt #2** |
| 16000 | — | — | — | 0.004 | — | **B3 ckpt #3**，eval mse 新低 🚀 |
| 16300 | 0.059 | 0.097 / 0.097 | 0.288 / 0.462 | — | 0.59 | future loss 比 B2 降 ~30% |

- **B3 training speed**: ~2.29 s/it（续训后与 B2 一致）
- **B3 LR**（step 16300）: base=2.38e-5, action_model=9.51e-5, wan_lora=9.51e-6（cosine decay ~5%）
- **续训表现**：eval mse 从 0.007 → **0.004**，future loss 从 0.425→**0.288**（main），训练状态健康 🟢
- **epoch**: 0.59（约 55% epoch）
