# MoWA E-003-v2: WanPI + contFT32 + LoRA rank=32 · TurnOnElectricKettle（单任务子集）

> **实验代号**: E-003-v2 / `WanPI-TurnOnElectricKettle_contft32_lora-r32`
> **状态**: 🟢 训练运行中
> **run_id**: `MoWA-E-003-v2_WanPI-TurnOnElectricKettle_contft32_lora-r32_20260722_1326`
> **启动时间**: 2026-07-22 13:26 CST
> **当前更新**: 2026-07-22 14:36 CST
> **tmux 会话**: `train`（accelerate 8 进程）
> **配置来源**: `configs/mowa/mowa_e003_v2_wanpi_contft32_lora_TurnOnElectricKettle.yaml`

---

## 实验概述

MoWA E-003-v2，基于 WanPI 框架 + contFT32 + LoRA rank=32。数据使用单原子任务 **TurnOnElectricKettle**（`robocasa365_turn_on_electric_kettle_target_human`），8 × RTX 4090（NCCL, bf16）。

从全 18 任务（B2/B3）中提取单任务子集进行针对性训练，作为 E-003-v2 单任务基线实验。

### 与全任务训练对比

| 项目 | 全任务（E003-v2 B3） | **本次（单任务 Kettle）** |
|------|---------------------|--------------------------|
| run_id | ...0024 | **...1326** |
| 数据 | 全 18 任务（559k） | **TurnOnElectricKettle（~31k）** |
| 总步数 | 100k | 100k |
| 启动时间 | 07/22 14:07 | **07/22 13:26** |
| 预训练 | WM4A-Wan2d2-OFT-LIBERO-4in1 | 同 |

---

## 训练事件

| 时间 (CST) | 事件 |
|------------|------|
| 2026-07-22 13:26 | 终端启动，`accelerate launch` 8 进程 spawn |
| 2026-07-22 13:27 | Data flow validation 通过 |
| 2026-07-22 13:35 | Step 50，`action_dit_loss=0.885` |
| 2026-07-22 13:37 | Step 100，`action_dit_loss=0.409` |
| 2026-07-22 13:42 | Step 200，`action_dit_loss=0.539` |
| 2026-07-22 13:47 | Step 300，`action_dit_loss=0.301` |
| 2026-07-22 13:49 | Step 350，`action_dit_loss=0.285` |
| 2026-07-22 13:51 | Step 400，`action_dit_loss=0.078` |
| 2026-07-22 13:54 | Step 450，首次 eval |
| 2026-07-22 13:57 | Step 500，`action_dit_loss=0.058`，eval mse=0.0085，**ckpt #1 steps_500** |
| 2026-07-22 13:59 | Step 550 |
| 2026-07-22 14:06 | Step 700，eval mse=0.010 |
| 2026-07-22 14:10 | Step 750，eval mse=0.012 |
| 2026-07-22 14:17 | Step 950，eval mse=0.012 |
| 2026-07-22 14:20 | Step 1000，eval mse=0.006，**ckpt #2 steps_1000** |
| 2026-07-22 14:31 | Step 1250，`action_dit_loss=0.089`，eval mse=0.006 |
| 2026-07-22 14:36 | Step 1300，eval mse=0.006 |
| 2026-07-22 14:41 | Step 1500，eval mse=0.006 |

---

## 训练参数

### 脚本级参数

| 参数 | 值 |
|------|-----|
| `run_id` | `MoWA-E-003-v2_WanPI-TurnOnElectricKettle_contft32_lora-r32_20260722_1326` |
| `DATA_MIX` | `robocasa365_turn_on_electric_kettle_target_human` |
| `MAX_TRAIN_STEPS` | 100000 |
| `SAVE_INTERVAL` | 500 |
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
| 数据集 | robocasa365_turn_on_electric_kettle_target_human（单任务） |
| 动作类型 | delta_ee（12-dim） |

---

## 训练指标

| Step | action_dit_loss | loss_action near/far | loss_future main/wrist | eval mse | 备注 |
|------|----------------|---------------------|----------------------|----------|------|
| 50 | 0.885 | — | — | — | warmup 早期 |
| 100 | 0.409 | — | — | — | |
| 200 | 0.539 | — | — | — | |
| 300 | 0.301 | — | — | — | |
| 350 | 0.285 | — | — | — | |
| 400 | 0.078 | — | — | — | |
| 500 | 0.058 | — | — | 0.0085 | **ckpt #1** |
| 600 | 0.168 | — | — | — | |
| 750 | — | — | — | 0.012 | |
| 1000 | — | 0.125 / 0.118 | 0.537 / 0.702 | 0.006 | **ckpt #2** |
| 1250 | 0.089 | 0.124 / 0.118 | 0.537 / 0.702 | 0.006 | |
| 1300 | — | — | — | 0.006 | |

- **training speed**: ~2.0–2.3 s/it（8 × RTX 4090），data_time ~0.001s，model_time ~1.3–2.0s
- **LR**: warmup 2000 步 → peak at step 2000 → cosine decay；step 1250 时 base=1.5e-5, action_model=6e-5, wan_lora=6e-6
- **action_dit_loss**: 0.885（step 50）→ 0.06–0.09（step 500–1250），波动较小
- **eval mse**: 0.006–0.012
- **future latent prior loss**: ~0.87–0.94 range
- **padding ratio**: front_pad=0, back_pad~22%
- **cross_view**: gate 逐步激活，residual_ratio ~0.24
- **epoch**: ~1.88 at step 1250
