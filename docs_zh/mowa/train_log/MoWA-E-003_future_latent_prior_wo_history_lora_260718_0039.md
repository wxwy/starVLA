# MoWA-E-003 (LoRA): Future Latent Prior without History · Wan LoRA 4 卡版（robocasa365_atomic_target_human_all）

> **实验代号**: E-003 LoRA / `mowa_future_latent_prior_wo_history_lora`
> **状态**: 🟢 训练运行中
> **run_id**: `MoWA-E-003_future_latent_prior_wo_history_lora_260718_0039`
> **启动时间**: 2026-07-18 00:39 CST（从 scratch）
> **当前更新**: 2026-07-20 09:00 CST（step ~45,700）
> **tmux 会话**: `train`（accelerate 4 进程）
> **配置来源**: `configs/mowa/mowa_e003_future_latent_prior_lora_long_training_launch_candidate.yaml`

---

## 实验概述

MoWA E-003 future latent prior without history · Wan LoRA 版。WanPI + Wan2.2-TI2V-5B + LoRA（rank=8），`robocasa365_atomic_target_human_all`（559k transitions），4 × RTX 4090。

---

## 训练事件

| 时间 (CST) | 事件 |
|------------|------|
| 2026-07-18 00:39 | 从 scratch 启动 |
| 2026-07-18 00:42 | Step 20，`action_dit_loss=1.218` |
| 2026-07-18 00:43 | Step 30，`action_dit_loss=1.119` |
| 2026-07-18 00:44 | Step 40，`action_dit_loss=1.065` |
| 2026-07-18 ~00:45 | Step ~40，训练正常，~4.36 s/it |
| 2026-07-18 ~04:00 | Step ~3,176，loss 稳步下降，action near 0.147 |
| 2026-07-18 ~08:00 | Step ~6,443，action near 0.122，future near 0.296 |
| 2026-07-18 ~12:00 | Step ~9,693，action near 0.110，future near 0.262 |
| 2026-07-18 ~14:00 | Step ~10,800，action_dit_loss 0.02–0.24 波动（均 ~0.11），future horizon loss 持续下降 |
| 2026-07-18 ~16:40 | Step ~13,000，action near 0.100，future near 0.269，speed 4.36 s/it，无错误 |
| 2026-07-18 ~20:40 | Step ~16,300（20%），action near 0.082，future near 0.257，speed 4.39 s/it |
| 2026-07-19 ~01:00 | Step ~19,600（24%），action near 0.094，future near 0.249，speed 4.33 s/it，epoch 0.71 |
| 2026-07-19 ~05:00 | Step ~22,800（29%），action near 0.075，future near 0.228，speed 4.41 s/it，epoch 0.83 |
| 2026-07-19 ~09:00 | Step ~26,100（33%），action near 0.075 / far 0.074，future near 0.257，speed 4.34 s/it，epoch 0.95 |
| 2026-07-19 ~13:00 | Step ~29,400（37%），action near 0.061 / far 0.064，future near 0.246，speed 4.40 s/it，epoch 1.07 |
| 2026-07-19 ~17:00 | Step ~32,600（41%），action near 0.060 / far 0.066，future near 0.234，speed 4.34 s/it，epoch 1.19 |
| 2026-07-19 ~21:00 | Step ~35,900（45%），action near 0.056 / far 0.060，future near 0.241，speed 4.36 s/it，epoch 1.31 |
| 2026-07-20 ~01:00 | Step ~39,200（49%），action near 0.055 / far 0.061，future near 0.226，speed 4.41 s/it，epoch 1.43 |
| 2026-07-20 ~05:00 | Step ~42,500（53%），action near 0.056 / far 0.065，future near 0.228，speed 4.37 s/it，epoch 1.55 |
| 2026-07-20 ~09:00 | Step ~45,700（57%），action near 0.054 / far 0.050，future near 0.237，speed 4.40 s/it，epoch 1.67 |

---

## 训练参数

### 脚本级参数

| 参数 | 值 |
|------|-----|
| `run_id` | `MoWA-E-003_future_latent_prior_wo_history_lora_260718_0039` |
| `DATA_MIX` | `robocasa365_atomic_target_human_all` |
| `MAX_TRAIN_STEPS` | 80000 |
| `SAVE_INTERVAL` | 500 |
| `LOGGING_FREQUENCY` | 10 |
| `NUM_PROCESSES` | 4（NCCL，bf16） |
| `GRADIENT_ACCUMULATION_STEPS` | 4 |
| `PER_DEVICE_BATCH_SIZE` | 2 |
| `Effective Global Batch` | **32** |
| `NUM_WORKERS` | 16 |
| `BASE_WM` | Wan2.2-TI2V-5B-Diffusers |
| `is_resume` | false（从 scratch） |
| 设备 | 4 × RTX 4090 |

### 模型参数

| 参数 | 值 |
|------|-----|
| 框架 | WanPI |
| Action Model | LayerwiseFM（DiT，30 层，hidden 1024，16 heads） |
| Action Dim | 12 |
| State Dim | 32 |
| Action Horizon | 32 |
| Future Window | 8 |
| History Window | 0 |
| Noise Schedule | BetaAlpha（α=1.5, β=1.0） |
| Inference Timesteps | 4 |
| Frozen | `backbone` |
| LR (action_model) | 1.0e-4 |
| LR (base) | 2.5e-5 |
| LR (wan_lora) | 1.0e-5 |
| LR Scheduler | cosine_with_min_lr |
| Warmup | 1000 steps |
| Optimizer | AdamW（β=(0.9, 0.95)） |
| 总参数 | 5514.707 M |
| 可训练 | 512.794 M |
| 数据规模 | 558,946 transitions / 9,126 trajectories |

### Wan LoRA

| 参数 | 值 |
|------|-----|
| enabled | true |
| rank | 8 |
| alpha | 16 |
| dropout | 0.0 |
| layers | 0–全部 |
| targets | cross_attention, self_attention |

### MoWA 特有参数

| 参数 | 值 |
|------|-----|
| `enable_future_latent_prior_loss` | true（loss_scale=0.05） |
| `enable_future_supervision_loss` | false |
| `multi_view` | true（main / wrist 双视角） |
| `cross_view` | gate_init=1.0, num_layers=10 |

---

## 训练指标

| Step | action_dit_loss | action near/far | future near/far | 备注 |
|------|----------------|-----------------|-----------------|------|
| 20 | 1.218 | — | — | warmup 中 |
| 40 | 1.065 | — | — | |
| ~3,176 | — | 0.147 / 0.155 | 0.299 / 0.326 | ~4h |
| ~6,443 | — | 0.122 / 0.129 | 0.296 / 0.318 | ~8h |
| ~9,693 | — | 0.110 / 0.119 | 0.262 / 0.292 | ~12h |
| ~10,800 | 0.02–0.24 | — | — | ~14h |
| ~13,000 | — | 0.100 / 0.112 | 0.269 / 0.292 | ~16h |
| ~16,300 | — | 0.082 / 0.093 | 0.257 / 0.289 | ~20h，20% |
| ~19,600 | — | 0.094 / 0.103 | 0.249 / 0.281 | ~24h，24% |
| ~22,800 | — | 0.075 / 0.084 | 0.228 / 0.262 | ~28h，29% |
| ~26,100 | — | 0.075 / 0.074 | 0.257 / 0.287 | ~32h，33% |
| ~29,400 | — | 0.061 / 0.064 | 0.246 / 0.271 | ~36h，37% |
| ~32,600 | — | 0.060 / 0.066 | 0.234 / 0.263 | ~40h，41% |
| ~35,900 | — | 0.056 / 0.060 | 0.241 / 0.275 | ~44h，45% |
| ~39,200 | — | 0.055 / 0.061 | 0.226 / 0.258 | ~48h，49% |
| ~42,500 | — | 0.056 / 0.065 | 0.228 / 0.260 | ~52h，53% |
| ~45,700 | — | 0.054 / 0.050 | 0.237 / 0.266 | ~56h，57% |

- **training speed**: ~4.37–4.39 s/it 稳定，data_time ~0.001s，model_time ~0.94–1.5s
- **LR**: warmup 已完成，cosine decay 中（wan_lora 9.72e-6 @ step 9700）
- **padding ratio**: front_pad=0, back_pad~16%, both_pad=0
