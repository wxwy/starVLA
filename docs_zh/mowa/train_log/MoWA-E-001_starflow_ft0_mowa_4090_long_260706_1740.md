# MoWA-E-001: StarFlowVLA ft0 mowa_main（robocasa365, bs32, 带 future supervision）

> **实验代号**: E-001 / mowa_main
> **状态**: 🟢 训练运行中
> **run_id**: `MoWA-E-001_starflow_ft0_mowa_4090_long_260706_1740`
> **启动时间**: 2026-07-06 17:35 CST
> **当前更新**: 2026-07-08 16:37 CST
> **进程**: PID 284059（4 进程），运行于 `pts/2`（建议迁入 tmux）
> **配置来源**: `configs/mowa/mowa_e001_starflow_ft0_long_training_candidate.yaml`

---

## 实验概述

MoWA E-001 **mowa_main**：在 StarFlowVLA 框架中使用 FT0 变体（动作头 `future_action_window_size=7`，`past_action_window_size=0`），在 **robocasa365** 数据集上训练 VLA 模型。

**本 run 为 mowa 主实验**（`experiment_role: mowa_main`），区别于 baseline：
- ✅ **`enable_future_supervision_loss: true`** — 未来 token 的额外监督信号参与训练
- ✅ **`enable_layerwise_bridge_token_coupling: true`** — 向每层 DiT cross-attention 注入 2 个 bridge token
- 使用 `mowa_future_feature_heads` 作为 bridge token 特征源
- `layerwise_bridge_active_heads`: task_progress, action_outcome_class
- `enable_layerwise_bridge_token_coupling: true`

Batch size 配置：bs32（per_device_batch_size=2 × gradient_accumulation_steps=16），RTX 4090 24GB 单卡。

---

## 训练参数

### 脚本级参数

| 参数 | 值 |
|------|-----|
| `run_id` | `MoWA-E-001_starflow_ft0_mowa_4090_long_260706_1740` |
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

### 模型 / 优化器参数

| 参数 | 值 |
|------|-----|
| **框架** | StarFlowVLA（`ft0` variant） |
| **Base VLM** | Qwen3-VL-4B-Instruct |
| **Action Model** | LayerwiseFM (DiT, 36 layers, 1024 hidden, 16 heads) |
| **Action Dim** | 12（含 gripper；dim11 恒为 0.0，可能是 padding 维度） |
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
| `enable_layerwise_bridge_token_coupling` | **true** ✅ — 每层 DiT 拼接 2 个 bridge token |
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

> 最后更新：2026-07-08 16:37 CST
> 训练启动 47h 2m

| 指标 | 值 |
|------|-----|
| **当前 Step** | ~37650 / 80000 |
| **完成比例** | 47.06% |
| **单步耗时** | — |
| **model_time** | 0.24941703502554446s |
| **data_time** | 0.0005370998987928033s |
| **已运行时间** | 47h 2m |
| **预计剩余时间** | 52h 54m |
| **最新 checkpoint** | steps_36000（共 11 个） |
| **最新 eval mse_score** | 首次 eval 在 step 2000 |

### Loss 记录

| Step | action_dit_loss | mowa_future_supervision_loss | 备注 |
|------|----------------|------------------------------|------|
| 34700 | 0.12083146220538765 | 0.05811226804507896 |  |
| 34750 | 0.07764758804114535 | 0.06864365859655663 |  |
| 34800 | 0.1145442333072424 | 0.03343695343937725 |  |
| 34850 | 0.060020782286301255 | 0.059755776659585536 |  |
| 34900 | 0.1039834640105255 | 0.05960521879023872 |  |
| 34950 | 0.08873639290686697 | 0.032401606906205416 |  |
| 35000 | 0.08358662168029696 | 0.059090058028232306 |  |
| 35050 | 0.07240238867234439 | 0.06477491298574023 |  |
| 35100 | 0.07048595906235278 | 0.05139172327471897 |  |
| 35150 | 0.0797275441000238 | 0.08484847613726743 |  |
| 35200 | 0.08314582263119519 | 0.04209171162801795 |  |
| 35250 | 0.08541186747606844 | 0.06480234710033983 |  |
| 35300 | 0.08705629594624043 | 0.15118431230075657 |  |
| 35350 | 0.1094051402178593 | 0.02598662429227261 |  |
| 35400 | 0.13086178721277975 | 0.06150775490095839 |  |
| 35450 | 0.10663013823796064 | 0.06876878015464172 |  |
| 35500 | 0.07690742041449994 | 0.059234441054286435 |  |
| 35550 | 0.07701969402842224 | 0.05833620484918356 |  |
| 35600 | 0.09245738689787686 | 0.031204118698951788 |  |
| 35650 | 0.10546539770439267 | 0.07878118504595477 |  |
| 35700 | 0.11665695393458009 | 0.05074072888237424 |  |
| 35750 | 0.05363693041726947 | 0.03686034577549435 |  |
| 35800 | 0.11031245673075318 | 0.053091967914951965 |  |
| 35850 | 0.06584322213893756 | 0.07661752187414095 |  |
| 35900 | 0.10048696538433433 | 0.07182714394002687 |  |
| 35950 | 0.06420149421319366 | 0.057216102883103304 |  |
| 36000 | 0.08645764790708199 | 0.03985373028262984 | 📊 含 eval/save |
| 36050 | 0.11506053386256099 | 0.08673834166256711 |  |
| 36100 | 0.1088604339165613 | 0.07155966002028435 |  |
| 36150 | 0.07928920106496662 | 0.07292798883281648 |  |
| 36200 | 0.08173154119867831 | 0.05944378549247631 |  |
| 36250 | 0.08122401311993599 | 0.07669429409725126 |  |
| 36300 | 0.1181131883058697 | 0.07093453250126913 |  |
| 36350 | 0.11167820729315281 | 0.10320942221733276 |  |
| 36400 | 0.08394967101048678 | 0.026663925091270357 |  |
| 36450 | 0.06099896185332909 | 0.05841348417743575 |  |
| 36500 | 0.0992679963237606 | 0.046440038189757615 |  |
| 36550 | 0.18554699403466657 | 0.04590004394412972 |  |
| 36600 | 0.06043085257988423 | 0.062160353758372366 |  |
| 36650 | 0.07202041702112183 | 0.09394958542543463 |  |
| 36700 | 0.08814413112122566 | 0.06406842213618802 |  |
| 36750 | 0.0984193931799382 | 0.06919156957883388 |  |
| 36800 | 0.0990709206671454 | 0.055120035030995496 |  |
| 36850 | 0.0997804252547212 | 0.06946191884344444 |  |
| 36900 | 0.120725481887348 | 0.06703990718233399 |  |
| 36950 | 0.06176037067780271 | 0.0772235024778638 |  |
| 37000 | 0.06840874487534165 | 0.05179559046518989 |  |
| 37050 | 0.08425247337436303 | 0.052251581830205396 |  |
| 37100 | 0.09892551880329847 | 0.048952696670312434 |  |
| 37150 | 0.10463419137522578 | 0.05907889414811507 |  |
| 37200 | 0.12095134519040585 | 0.057010691496543586 |  |
| 37250 | 0.07461674965452403 | 0.05026215047109872 |  |
| 37300 | 0.08112311575678177 | 0.03682986611966044 |  |
| 37350 | 0.07221801998093724 | 0.052345233445521444 |  |
| 37400 | 0.10110072698444128 | 0.06336970653501339 |  |
| 37450 | 0.07909555686637759 | 0.056143824476748705 |  |
| 37500 | 0.07931816659402102 | 0.0486115016246913 |  |
| 37550 | 0.07176663621794432 | 0.052679168213217054 |  |
| 37600 | 0.133261221984867 | 0.04426384897669777 |  |
| 37650 | 0.09963933832477778 | 0.06444652847130783 |  |

---

## 系统资源占用

> 最后更新：2026-07-08 16:37 CST

### GPU（NVIDIA GeForce RTX 4090）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 44% |
| **显存使用** | 20406 MiB / 24564 MiB (83%) |
| **功耗** | 223.37 W |
| **温度** | 52°C |

### 系统内存

| 指标 | 值 |
|------|-----|
| **总量** | 503Gi |
| **已用** | 42Gi |
| **可用** | 457Gi |

### 存储

| 挂载点 | 使用情况 |
|--------|----------|
| `/` (overlay) | 4.9G / 30G (17%) |
| `/localdisk-tmp` | 0 / 100G (0%) |
| `/disk/rl` | 531T / 700T (76%) |

---

## 输出目录

```
playground/mowa_ckpt/MoWA-E-001_starflow_ft0_mowa_4090_long_260706_1740/
├── checkpoints/                 （11 个 checkpoint，最新: steps_36000）
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
- **与 baseline 的关键区别**: (1) `enable_future_supervision_loss: true` — future token 额外监督参与训练; (2) `enable_layerwise_bridge_token_coupling: true` — 向每层 DiT cross-attention 注入 2 个 bridge token（源自 pooled VLM hidden + per-layer 偏置）; (3) 额外 ~3.27M 参数（bridge projector + future feature heads + token embedding）。
- `per_device_batch_size=2`，`grad_accum=16`，`num_processes=1`，`num_workers=3`。RTX 4090 单卡，3 个 `pt_data_worker` 为 PyTorch DataLoader 数据进程。
- Action dim 12（含 gripper），注意 action dim 11 恒为 0.0（可能是 padding 维度）。
- **训练不在 tmux 中**，运行于 `pts/2` 直接启动。建议迁入 tmux `train` 会话以防 pts 断开导致中断。
- 每4小时由 Claude cron loop 自动更新本 tracker。

---

*本文档将持续更新。*
