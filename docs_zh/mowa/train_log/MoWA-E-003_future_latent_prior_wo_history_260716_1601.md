# MoWA-E-003: Future Latent Prior without History（robocasa365_open_drawer_target_human）

> **实验代号**: E-003 / `mowa_future_latent_prior_wo_history`
> **状态**: 🟢 训练运行中
> **run_id**: `MoWA-E-003_future_latent_prior_wo_history`
> **启动时间**: 2026-07-16 16:01 CST
> **当前更新**: 2026-07-16 20:08 CST
> **tmux 会话**: `e003b1`（Pane PID 140322，训练子进程 PID 1176892）
> **配置来源**: `configs/mowa/mowa_e003_future_latent_prior_long_training_launch_candidate.yaml`

---

## 实验概述

MoWA E-003 **future latent prior without history**：在 **WanPI** 框架（WorldModel→Action）下，以 **Wan2.2-TI2V-5B** 为预训练 world model backbone，在 **robocasa365_open_drawer_target_human** 数据上训练 VLA 模型。

本 run 的核心变体：
- ✅ **`history_window_steps: 0`** — 不使用历史观测帧，仅依赖当前帧与未来潜在先验
- ✅ **`enable_future_latent_prior_loss: true`** — 未来视频 latent 作为额外监督信号
- ✅ **`enable_future_supervision_loss: true`** — future feature heads 监督
- ✅ **`enable_layerwise_bridge_token_coupling: true`** — 向 DiT 每层 cross-attention 注入 2 个 bridge token
- ✅ **双视角输入**（main + wrist），启用 cross-view attention
- backbone 冻结，仅训练 action model / projector / future heads 等模块

Batch size 配置：bs32（per_device_batch_size=4 × gradient_accumulation_steps=8），RTX 4090 24GB 单卡。

---

## 训练事件

| 时间 (CST) | 事件 |
|------------|------|
| 2026-07-16 16:01 | 首次启动，从 scratch 开始训练 |
| 2026-07-16 17:35 | 首次记录，step ~134 |
| 2026-07-16 ~19:14 | 训练继续至 step ~780 后中断（`KeyboardInterrupt`） |
| 2026-07-16 20:01 | **重新启动**，自动从 `steps_500` checkpoint 恢复 |

> 注：重新启动后 WandB 出现警告：`Tried to log to step 520 that is less than the current step 781. Steps must be monotonically increasing`。原因是 WandB 云端/本地已记录到 step 781，而恢复点回到 step 500。后续 WandB step 520–780 的数据可能不会被接受，需关注是否影响可视化。

---

## 训练参数

### 脚本级参数

| 参数 | 值 |
|------|-----|
| `run_id` | `MoWA-E-003_future_latent_prior_wo_history` |
| `CONFIG_YAML` | `configs/mowa/mowa_e003_future_latent_prior_long_training_launch_candidate.yaml` |
| `DATA_MIX` | `robocasa365_open_drawer_target_human` |
| `MAX_TRAIN_STEPS` | 80000 |
| `SAVE_INTERVAL` | 500 |
| `EVAL_INTERVAL` | 500 |
| `LOGGING_FREQUENCY` | 20 |
| `NUM_PROCESSES` | **1**（单卡训练，无分布式） |
| `GRADIENT_ACCUMULATION_STEPS` | **8** |
| `PER_DEVICE_BATCH_SIZE` | **4** |
| `NUM_WORKERS` | 8 |
| `BASE_VLM / BASE_WM` | `./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers` |
| `DATA_ROOT` | `playground/Datasets/robocasa365` |
| `WANDB_PROJECT` | `MoWA` |
| `WANDB_ENTITY` | `silencewx-harbin-institute-of-technology` |
| `is_resume` | **`False`**（配置项，但实际因 `resume_policy=resume_latest_complete_only` 自动从 `steps_500` 恢复） |
| 设备 | NVIDIA GeForce RTX 4090（24564 MiB） |

### 模型 / 优化器参数

| 参数 | 值 |
|------|-----|
| **框架** | WanPI（WorldModel→Action，`starflow_ft_variant: ft0`） |
| **Base World Model** | Wan2.2-TI2V-5B-Diffusers |
| **Action Model** | LayerwiseFM（DiT） |
| **Action Dim** | 12 |
| **State Dim** | 32 |
| **Action Horizon** | 32 |
| **Past Action Window Size** | 0 |
| **Future Window Steps** | 8 |
| **History Window Steps** | 0 |
| **Num Target Vision Tokens** | 0 |
| **Noise Schedule** | BetaAlpha（α=1.5, β=1.0, s=0.999） |
| **Inference Timesteps** | 4 |
| **DiT Hidden Dim** | 1024 |
| **DiT Dropout** | 0.2 |
| **Frozen Modules** | `backbone` |
| **Learning Rate (action_model)** | 1.0e-4 |
| **Learning Rate (base)** | 2.5e-5 |
| **Learning Rate (qwen_vl_interface)** | 1.0e-5 |
| **LR Scheduler** | cosine_with_min_lr |
| **Min LR** | 1.0e-6 |
| **Warmup Steps** | 500 |
| **Optimizer** | AdamW（β=(0.9, 0.95), eps=1e-8, wd=1.0e-8） |
| **Max Grad Norm** | 1.0 |
| **Gradient Checkpointing** | true |
| **Gradient Clipping** | 1.0 |
| **Effective Global Batch** | **32**（per_device=4 × grad_accum=8 × num_processes=1） |
| **Checkpoint Format** | lightweight |
| **Save Format** | safetensors |
| **Seed** | 42 |
| **总参数量** | 5502.911 M |
| **可训练参数量** | 503.123 M |

### MoWA / Future Latent Prior 特有参数

| 参数 | 值 |
|------|-----|
| `enable_future_latent_prior_loss` | **true** ✅ |
| `enable_future_supervision_loss` | **true** ✅ |
| `enable_layerwise_bridge_token_coupling` | **true** ✅ |
| `layerwise_bridge_feature_source` | `mowa_future_feature_heads` |
| `layerwise_bridge_active_heads` | task_progress, action_outcome_class |
| `future_supervision_active_heads` | task_progress, action_outcome_class |
| `wam_feature_dim` | 1024 |
| `action_hidden_dim` | 1024 |
| `num_bridge_tokens` | 2 |
| `loss_scale.mowa_future_supervision` | 1.0 |
| `loss_scale.mowa_future_latent_prior` | 1.0 |
| `multi_view.enabled` | true（main / wrist） |
| `multi_view.shared_wan_backbone` | true |
| `multi_view.view_embedding` | true |
| `cross_view.enabled` | true |
| `cross_view.mode` | per_timestep |
| `cross_view.bidirectional` | true |
| `cross_view.num_layers` | 10 |
| `cross_view.start_layer_ratio` | 0.6667 |
| `cross_view.gate_init` | 0.001 |
| `action_fusion.mode` | learnable_queries（num_queries=8） |

### 数据集

| 数据集 | 样本数（frames/transitions）| embodiment |
|--------|-----------------------------|------------|
| `robocasa365_open_drawer_target_human` | 34,424 | new_embodiment |

---

## 训练进度

> 最后更新：2026-07-16 20:08 CST
> 本次重启后运行约 7 分钟；自首次启动累计约 3 小时 40 分钟

| 指标 | 值 |
|------|-----|
| **当前 Step** | ~532 / 80000 |
| **完成比例** | 0.66% |
| **单步耗时** | ~14.70 s/it |
| **model_time** | ~1.82 s |
| **data_time** | ~0.003 s |
| **本次重启后运行时间** | ~7 分钟 |
| **预计剩余时间** | ~325 小时 |
| **最新 checkpoint** | `steps_500` |
| **最新 eval** | 尚未执行 |

### Loss 记录

| Step | action_dit_loss | mowa_future_latent_prior_loss | loss_multiview_total | loss_total | 备注 |
|------|-----------------|-------------------------------|----------------------|------------|------|
| 120 | 0.4596 | 2.1309 | 2.6866 | 3.0857 | 首个 logging 点 |
| 200 | 0.2867 | 1.9087 | 2.6615 | 2.9654 |  |
| 220 | 0.3234 | 1.8615 | 1.1413 | 1.5724 |  |
| 240 | 0.2515 | 1.6479 | 2.0664 | 2.2919 |  |
| 260 | 0.2787 | 1.7525 | 2.2490 | 2.4527 |  |
| 280 | 0.2045 | 1.5630 | 0.9558 | 1.1457 |  |
| 300 | 0.4300 | 2.0620 | 1.9388 | 2.4378 |  |
| 320 | 0.3105 | 1.7825 | 2.3631 | 2.5987 |  |
| 340 | 0.2847 | 1.8401 | 1.5031 | 1.8958 |  |
| 360 | 0.3071 | 1.3212 | 1.6255 | 2.0003 |  |
| 380 | 0.3015 | 1.2829 | 1.0824 | 1.5590 |  |
| 660 | 0.3030 | 0.9382 | 0.6334 | 0.8296 | 中断前 |
| 680 | 0.3289 | 0.7961 | 0.5178 | 0.6365 | 中断前 |
| 700 | 0.2421 | 0.9130 | 0.9677 | 1.1669 | 中断前 |
| 720 | 0.1541 | 0.9120 | 0.4763 | 0.7237 | 中断前 |
| 740 | 0.1583 | 0.9088 | 0.7745 | 0.9013 | 中断前 |
| 760 | 0.2370 | 0.8044 | 0.9905 | 1.3465 | 中断前 |
| 780 | 0.1572 | 0.9259 | 1.2553 | 1.3872 | 中断前 |
| 520 | 0.2577 | 1.1669 | 1.1935 | 0.2439 | 重启后恢复点 |

> 注：
> - `mowa_future_supervision_loss` 在各 logging step 仍为 0.0。
> - 重启后 step 520 的 `loss_total` 仅为 0.2439，与 `loss_action` 相等，可能 resume 后首个 logging step 的 future latent prior loss 未计入总 loss；后续需继续观察是否恢复一致。
> - 中断前 step 720/780 `action_dit_loss` 已降至 0.15 左右，较早期有明显下降。

---

## 系统资源占用

> 最后更新：2026-07-16 20:08 CST

### GPU（NVIDIA GeForce RTX 4090）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 100% |
| **显存使用** | 24028 MiB / 24564 MiB (98%) |
| **功耗** | 417.68 W |
| **温度** | 69°C |

### 系统内存

| 指标 | 值 |
|------|-----|
| **总量** | 503Gi |
| **已用** | 约 44Gi（参考同机） |
| **可用** | 约 455Gi（参考同机） |

### 存储

| 挂载点 | 使用情况 |
|--------|----------|
| `/` (overlay) | 5.0G / 30G (17%)（参考同机） |
| `/disk/rl` | 533T / 700T (77%)（参考同机） |

---

## 输出目录

```
playground/mowa_ckpt/MoWA-E-003_future_latent_prior_wo_history/
└── MoWA-E-003_future_latent_prior_wo_history/
    ├── checkpoints/
    │   └── steps_500/               ✅ 自动保存的第一个 checkpoint
    ├── config.full.yaml             ✅
    ├── config.yaml                  ✅
    ├── dataset_statistics.json      ✅
    └── wandb/                       ✅
```

---

## 相关链接

- **WandB Project**: [MoWA](https://wandb.ai/silencewx-harbin-institute-of-technology/MoWA)
- **WandB Run**: [MoWA-E-003_future_latent_prior_wo_history](https://wandb.ai/silencewx-harbin-institute-of-technology/MoWA/runs/MoWA-E-003_future_latent_prior_wo_history)
- **训练主机**: 本地服务器（RTX 4090）

---

## 启动命令

```bash
cd /disk/rl/starVLA
CUDA_VISIBLE_DEVICES=0 .venv/bin/python starVLA/training/train_starvla.py \
  --config_yaml configs/mowa/mowa_e003_future_latent_prior_long_training_launch_candidate.yaml
```

---

## 备注

- 本实验为 MoWA 项目 **E-003** 的正式长训启动，探索在 **无历史帧** 条件下，利用 Wan2.2-TI2V 的未来 latent prior 监督 VLA 动作生成。
- 与 E-001 的核心差异：使用 **WanPI 框架**（WorldModel→Action）替代 StarFlowVLA；输入为 Wan VAE latent 而非原始像素；新增 `future_latent_prior_loss` 监督未来视频 latent 重建。
- 双视角输入：`observation.images.robot0_agentview_left`（main）与 `observation.images.robot0_eye_in_hand`（wrist），通过 shared Wan backbone + view embedding 编码。
- `per_device_batch_size=4`，`grad_accum=8`，`num_processes=1`，`num_workers=8`。单卡 RTX 4090 满载，显存占用约 96%。
- 训练已重新启动并从 `steps_500` 恢复。当前 step ~532，仍在 warmup 末尾（`num_warmup_steps=500`，已结束）。
- 中断前训练已推进到 step ~780，`action_dit_loss` 在 step 720/780 降至 ~0.15，较早期 ~0.46 明显下降。
- WandB 因恢复点 step 500 小于云端 step 781 产生单调性警告，后续需关注是否影响曲线。
- 下一次自动 save/eval 预计在 step 1000。
- 训练在 tmux 会话 `e003b1` 中运行，断开会话不会中断训练。

---

*本文档将持续更新。*
