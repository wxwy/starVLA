# MoWA E-003-v2: WanPI + contFT32 + LoRA rank=32 · OpenDrawer (robocasa365)

> **实验代号**: E-003-v2 / `WanPI-LIBERO_contft32_lora-r32`
> **状态**: 🟢 训练运行中
> **run_id**: `MoWA-E-003-v2_WanPI-LIBERO_contft32_lora-r32_20260721_2150`
> **启动时间**: 2026-07-21 22:41 CST（从 scratch，基于 pretrained checkpoint）
> **当前更新**: 2026-07-21 22:45 CST（step ~100）
> **tmux 会话**: `train`（accelerate 8 进程）
> **配置来源**: `configs/mowa/mowa_e003_v2_wanpi_contft32_lora.yaml`

---

## 实验概述

MoWA E-003-v2，基于 WanPI 框架 + contFT32（Continuous Head + Action Horizon 32）+ LoRA rank=32。数据只使用 `robocasa365_open_drawer_target_human` 单任务（24,424 transitions, 514 trajectories），8 × 未知 GPU（NCCL, bf16）。

从 baseline checkpoint `WM4A-Wan2d2-OFT-LIBERO-4in1/checkpoints/steps_60000_pytorch_model.pt` 加载 backbone 权重。

---

## 训练事件

| 时间 (CST) | 事件 |
|------------|------|
| 2026-07-21 22:09 | 首次启动（run 220911），data flow validation 通过但 DDP 报 `find_unused_parameters` 错误 |
| 2026-07-21 22:41 | 修复后重新启动（run 224115），data flow validation 2/2 通过 |
| 2026-07-21 22:43 | Step 50，`action_dit_loss=1.034`，`future_latent_prior_loss=3.251` |
| 2026-07-21 22:45 | Step 100，`action_dit_loss=0.442`，`future_latent_prior_loss=1.941`，loss 快速下降 |

---

## 训练参数

### 脚本级参数

| 参数 | 值 |
|------|-----|
| `run_id` | `MoWA-E-003-v2_WanPI-LIBERO_contft32_lora-r32_20260721_2150` |
| `DATA_MIX` | `robocasa365_open_drawer_target_human` |
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
| 设备 | 8 × GPU（未知型号） |

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
| 数据集 | robocasa365_open_drawer_target_human |
| Transitions | 34,424 |
| Trajectories | 514 |
| 动作类型 | delta_ee（12-dim） |

---

## 训练指标

| Step | action_dit_loss | future_latent_prior | loss_action near/far | loss_future near/far | 备注 |
|------|----------------|---------------------|---------------------|---------------------|------|
| 50 | 1.034 | 3.251 | 1.068 / — | 1.833 / 1.711 | warmup 早期（LR 极低） |
| 100 | 0.442 | 1.941 | 0.427 / — | 0.948 / 1.011 | loss 快速下降 |

- **training speed**: ~2.33 s/it（8 GPU），data_time ~0.001s，model_time ~1.27–1.31s
- **LR**: warmup 阶段（step 100: base=1.25e-6, wan_lora=5e-7, action_model=5e-6）
- **padding ratio**: front_pad=0, back_pad~14%, both_pad=0（step 100）
- **cross_view**: gate ~1.0，residual_ratio ~0.0008–0.001（早期，gate 尚未学习）
- **future_pred_norm**: main ~469, wrist ~432（step 100，较高因为刚初始化）
- **wan_lora**: param_norm=50.6, delta_norm=0.04, grad_norm=0.017
