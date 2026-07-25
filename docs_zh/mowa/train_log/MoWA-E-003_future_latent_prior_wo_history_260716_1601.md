# MoWA-E-003: Future Latent Prior without History（robocasa365_open_drawer_target_human）

> **实验代号**: E-003 / `mowa_future_latent_prior_wo_history`
> **状态**: ⏹️ 已停止（2026-07-17 上午，由 4 卡 run `MoWA-E-003_future_latent_prior_wo_history_260717_1120` 取代，见对应日志）
> **run_id**: `MoWA-E-003_future_latent_prior_wo_history`
> **启动时间**: 2026-07-17 00:21 CST
> **当前更新**: 2026-07-17 08:30 CST
> **tmux 会话**: `train`（Pane PID 2231154，训练主进程 PID 2262306）
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
| 2026-07-16 ~19:14 | 训练中断（`KeyboardInterrupt`）|
| 2026-07-16 20:01 | 第一次重新启动，从 `steps_500` checkpoint 恢复 |
| 2026-07-17 00:14 | 第二次重新启动尝试，创建了空目录 `MoWA-E-003_future_latent_prior_wo_history/`，但未成功保存 checkpoint |
| 2026-07-17 00:21 | **本次实际重新启动**，无可用 checkpoint，从 scratch 开始训练；实际运行目录为 `MoWA-E-003_future_latent_prior_wo_history_260717_0020` |
| 2026-07-17 ~11:20 | 本单卡 run 停止；实验切换为 4 卡训练（run_id `MoWA-E-003_future_latent_prior_wo_history_260717_1120`），配置亦有调整（关闭 future supervision / bridge token coupling，latent prior loss_scale 1.0→0.05），详见新日志 |

> 注：
> - 前一次运行最终保存的 checkpoint 位于 `playground/mowa_ckpt/tmp_MoWA-E-003_future_latent_prior_wo_history/MoWA-E-003_future_latent_prior_wo_history/checkpoints/steps_1000.eval`，本次未加载。
> - 本次启动日志显示：`No checkpoints found in playground/mowa_ckpt/MoWA-E-003_future_latent_prior_wo_history/checkpoints`。`is_resume=False`，`resume_from_checkpoint=None`。
> - 由于原目录已存在，训练脚本自动将本次运行的 `run_id` 修正为 `MoWA-E-003_future_latent_prior_wo_history_260717_0020`，输出目录同步更新。
> - WandB 仍显示 `Resuming run MoWA-E-003_future_latent_prior_wo_history`，但本地模型权重为随机初始化。
> - 启动后 WandB 出现上传警告：`Fatal error while uploading data. Some run data will not be synced, but it will still be written to disk. Use wandb sync`。

---

## 训练参数

### 脚本级参数

| 参数 | 值 |
|------|-----|
| `run_id` | `MoWA-E-003_future_latent_prior_wo_history_260717_0020`（实际） |
| `original_run_id` | `MoWA-E-003_future_latent_prior_wo_history`（配置中） |
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
| `is_resume` | `False` |
| `resume_policy` | `resume_latest_complete_only` |
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

> 最后更新：2026-07-17 08:30 CST
> 本次启动后运行约 8 小时 9 分钟

| 指标 | 值 |
|------|-----|
| **当前 Step** | ~2015 / 80000 |
| **完成比例** | 2.52% |
| **单步耗时** | ~14.54 s/it |
| **model_time** | ~1.81 s |
| **data_time** | ~0.002 s |
| **本次启动后运行时间** | ~8 小时 9 分钟 |
| **预计剩余时间** | ~315 小时 |
| **最新 checkpoint** | `steps_2000`（同时触发 eval） |
| **最新 eval** | Step 2000：mse_score=0.0116，eval_num_samples=1536 |

### Loss 记录

| Step | action_dit_loss | mowa_future_latent_prior_loss | loss_multiview_total | loss_total | 备注 |
|------|-----------------|-------------------------------|----------------------|------------|------|
| 2000 | 0.1428 | 2.6095 | 3.7219 | 0.0801 | 同时触发 eval：mse_score=0.0116，eval_num_samples=1536 |

> 注：
> - `mowa_future_supervision_loss` 在 step 2000 仍为 0.0。
> - `loss_total`（0.0801）等于 `loss_action`，与 `action_dit_loss`（0.1428）不同；`action_dit_loss_last_micro` 为 0.0801，说明 total loss 取的是最后一个 micro-step 的 action loss。
> - 当前 tmux pane 缓冲区仅保留到 step 2000 的完整 loss dict，中间 logging step 的历史记录未保留。

---

## 系统资源占用

> 最后更新：2026-07-17 08:30 CST

### GPU（NVIDIA GeForce RTX 4090）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 100% |
| **显存使用** | 24030 MiB / 24564 MiB (98%) |
| **功耗** | 408.54 W |
| **温度** | 63°C |

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
playground/mowa_ckpt/MoWA-E-003_future_latent_prior_wo_history_260717_0020/
└── MoWA-E-003_future_latent_prior_wo_history_260717_0020/
    ├── checkpoints/
    │   ├── steps_1000/              ✅
    │   ├── steps_1500/              ✅
    │   └── steps_2000/              ✅（同时触发 eval）
    ├── config.full.yaml             ✅
    ├── config.yaml                  ✅
    ├── dataset_statistics.json      ✅
    ├── summary.jsonl                ✅
    └── wandb/                       ✅
```

> 历史/未使用输出目录：
> ```
> playground/mowa_ckpt/MoWA-E-003_future_latent_prior_wo_history/     # 00:14 创建，无 checkpoint
> playground/mowa_ckpt/tmp_MoWA-E-003_future_latent_prior_wo_history/ # 早期运行，steps_1000.eval
> ```

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
- `per_device_batch_size=4`，`grad_accum=8`，`num_processes=1`，`num_workers=8`。单卡 RTX 4090 满载，显存占用约 98%。
- 训练已于 2026-07-17 00:21 重新启动，当前 step ~2015，已过 warmup 阶段。
- 本次启动未找到本地 checkpoint，模型从 scratch 开始训练；WandB run 使用同名恢复。
- 由于原目录已存在，实际 `run_id` 自动修正为 `MoWA-E-003_future_latent_prior_wo_history_260717_0020`。
- WandB 启动时出现上传错误警告，后续需关注数据同步情况，必要时可手动执行 `wandb sync`。
- 已保存 checkpoint：steps_1000、steps_1500、steps_2000；step 2000 同时触发 eval（mse_score=0.0116）。
- 下一次自动 save/eval 预计在 step 2500。
- 训练在 tmux 会话 `train` 中运行，断开会话不会中断训练。

---

*本文档将持续更新。*
