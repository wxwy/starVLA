# MoWA E-003-v2: WanPI + contFT32 + LoRA rank=32 · OpenDrawer (robocasa365)

> **实验代号**: E-003-v2 / `WanPI-LIBERO_contft32_lora-r32`
> **状态**: 🔴 训练中断（step 2350），原因未知
> **run_id**: `MoWA-E-003-v2_WanPI-LIBERO_contft32_lora-r32_20260721_2150`
> **启动时间**: 2026-07-21 22:41 CST（从 scratch，基于 pretrained checkpoint）
> **中断时间**: 2026-07-22 00:15 CST（step 2350）
> **当前更新**: 2026-07-22 00:39 CST
> **tmux 会话**: `train`（accelerate 8 进程，已无存活进程）
> **配置来源**: `configs/mowa/mowa_e003_v2_wanpi_contft32_lora.yaml`

---

## 实验概述

MoWA E-003-v2，基于 WanPI 框架 + contFT32（Continuous Head + Action Horizon 32）+ LoRA rank=32。数据使用 `robocasa365_open_drawer_target_human` 单任务（34,424 transitions, 514 trajectories），8 GPU（NCCL, bf16）。

从 baseline checkpoint `WM4A-Wan2d2-OFT-LIBERO-4in1/checkpoints/steps_60000_pytorch_model.pt` 加载 backbone 权重。

训练在 step 2350（~1h34m）意外中断，最后一个 log 无 error trace，进程不再存活。可能原因为 tmux 会话被关闭或 OOM kill。

---

## 训练事件

| 时间 (CST) | 事件 |
|------------|------|
| 2026-07-21 22:09 | 首次启动（run 220911），data flow validation 通过但 DDP 报 `find_unused_parameters` 错误 |
| 2026-07-21 22:41 | 修复后重新启动（run 224115），data flow validation 2/2 通过 |
| 2026-07-21 22:43 | Step 50，`action_dit_loss=1.034`，`future_latent_prior_loss=3.251` |
| 2026-07-21 22:45 | Step 100，`action_dit_loss=0.442`，`future_latent_prior_loss=1.941`，loss 快速下降 |
| 2026-07-21 22:49 | Step 200，`action_dit_loss=0.411`，波动中下降 |
| 2026-07-21 22:53 | Step 250，首次 eval，`mse_score=0.011`，`action_dit_loss=0.229` |
| 2026-07-21 23:01 | Step 400，`action_dit_loss=0.127`，loss 趋于平稳 |
| 2026-07-21 23:05 | Step 500，eval `mse=0.011`，`action_dit_loss=0.114` |
| 2026-07-21 23:09 | Step 600，`action_dit_loss=0.273`（偶尔 spike） |
| 2026-07-21 23:21 | Step 1000，`action_dit_loss=0.248`，eval `mse=0.006`，**保存 checkpoint steps_1000** |
| 2026-07-21 23:43 | Step 1500，`action_dit_loss=0.071`，eval `mse=0.009`，loss 持续下降 |
| 2026-07-21 ~23:55 | Step ~1800，warmup 接近结束，LR 接近峰值 |
| 2026-07-22 00:01 | Step 2000，`action_dit_loss=0.155`，**warmup 完成，保存 checkpoint steps_2000** |
| 2026-07-22 00:11 | Step 2250，`action_dit_loss=0.169`，eval `mse=0.009` |
| 2026-07-22 00:13 | Step 2300，`action_dit_loss=0.139`，`future_latent_prior=1.039` |
| 2026-07-22 00:15 | Step 2350（最后记录），`action_dit_loss=0.084`，`future_latent_prior=0.998`，随后训练中断 |

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

| Step | action_dit_loss | future_latent_prior | loss_action near/far | loss_future main/wrist | eval mse | 备注 |
|------|----------------|---------------------|---------------------|----------------------|----------|------|
| 50 | 1.034 | 3.251 | 1.068 / — | 1.833 / 1.711 | — | warmup 早期 |
| 100 | 0.443 | 1.941 | 0.427 / — | 0.948 / 1.011 | — | loss 快速下降 |
| 200 | 0.411 | — | — | — | — | |
| 250 | 0.229 | — | — | — | 0.011 | 首次 eval |
| 500 | 0.114 | — | — | — | 0.011 | |
| 750 | 0.104 | — | — | — | 0.009 | |
| 1000 | 0.248 | 0.973 | 0.128 / 0.120 | 0.626 / 0.698 | 0.006 | **ckpt #1** |
| 1250 | 0.088 | — | — | — | 0.009 | |
| 1500 | 0.071 | — | — | — | 0.009 | |
| 1750 | 0.060 | — | — | — | — | |
| 2000 | 0.155 | 1.038 | 0.183 / — | 0.641 / 0.701 | 0.012 | **warmup 完成，ckpt #2** |
| 2250 | 0.169 | 0.960 | 0.127 / 0.121 | 0.634 / 0.705 | 0.009 | |
| 2300 | 0.139 | 1.039 | 0.190 / — | 0.340 / 0.472 | — | future loss 异常下降，cross_view residual_ratio=0.28 |
| 2350 | 0.084 | 0.998 | 0.107 / — | 0.450 / 0.479 | — | **训练中断** |

- **training speed**: ~2.33–2.43 s/it（8 GPU），data_time ~0.001s，model_time ~1.27–1.33s
- **LR**: warmup 0→2000 步线性上升；step 2000 时达到峰值 base=2.5e-5, action_model=1e-4, wan_lora=1e-5；step 2350 时 cosine decay 刚开始，LR 仅微降（base=2.4999e-5）
- **padding ratio**: front_pad=0, back_pad~13–14%, both_pad=0（稳定）
- **cross_view**: gate 从 ~1.0000（step 50）逐步上升到 ~1.006（step 2350）；residual_ratio 显著增大：~0.001（step 50）→ ~0.001（step 100）→ ~0.284（step 2350），说明 cross_view 特征融合正在激活
- **future_pred_norm**: main ~353（step 50）→ ~564（step 2350），wrist ~362（step 50）→ ~521（step 2350）
- **wan_lora**: param_norm ~50.6→50.8，delta_norm 从 0.01 增长到 ~4.58，表明 LoRA 权重在正常更新
- **done_head**: done_logit_mean ~-0.56→-3.51（step 2350），done_probability_mean ~0.36→0.08，done_head 在区分 done/not-done 上正在学习
- **eval mse**: 在 0.006–0.012 之间波动，整体呈下降趋势
- **epoch**: step 2350 时 epoch=2.18
