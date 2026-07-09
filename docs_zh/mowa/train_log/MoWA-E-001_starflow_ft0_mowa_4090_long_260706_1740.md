# MoWA-E-001: StarFlowVLA ft0 mowa_main（robocasa365, bs32, 带 future supervision）

> **实验代号**: E-001 / mowa_main
> **状态**: 🟢 训练运行中
> **run_id**: `MoWA-E-001_starflow_ft0_mowa_4090_long_260706_1740`
> **启动时间**: 2026-07-06 17:35 CST
> **当前更新**: 2026-07-09 12:37 CST
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

> 最后更新：2026-07-09 12:37 CST
> 训练启动 67h 2m

| 指标 | 值 |
|------|-----|
| **当前 Step** | ~53550 / 80000 |
| **完成比例** | 66.94% |
| **单步耗时** | — |
| **model_time** | 0.24941703502554446s |
| **data_time** | 0.0004409800749272108s |
| **已运行时间** | 67h 2m |
| **预计剩余时间** | 33h 6m |
| **最新 checkpoint** | steps_52000（共 7 个） |
| **最新 eval mse_score** | 首次 eval 在 step 2000 |

### Loss 记录

| Step | action_dit_loss | mowa_future_supervision_loss | 备注 |
|------|----------------|------------------------------|------|
| 50600 | 0.07957982126390561 | 0.030984395794803277 |  |
| 50650 | 0.1028805898677092 | 0.045584131992654875 |  |
| 50700 | 0.04268430927186273 | 0.04250010952819139 |  |
| 50750 | 0.11450856283772737 | 0.06241761788260192 |  |
| 50800 | 0.09202381374780089 | 0.05590808858687524 |  |
| 50850 | 0.10253222822211683 | 0.06257098625064828 |  |
| 50900 | 0.06460782838985324 | 0.03789134352700785 |  |
| 50950 | 0.09326054900884628 | 0.06375843912246637 |  |
| 51000 | 0.06495242062374018 | 0.06101938683423214 |  |
| 51050 | 0.05926262022694573 | 0.03779118615784682 |  |
| 51100 | 0.07564020354766399 | 0.040393633767962456 |  |
| 51150 | 0.06317674706224352 | 0.05742185469716787 |  |
| 51200 | 0.10291906428756192 | 0.05206229045870714 |  |
| 51250 | 0.09835550969000906 | 0.020035692461533472 |  |
| 51300 | 0.14592507982160896 | 0.05642549411277287 |  |
| 51350 | 0.05926673352951184 | 0.032365543898777105 |  |
| 51400 | 0.06648369668982923 | 0.057509041544108186 |  |
| 51450 | 0.053713720466475934 | 0.08292463200632483 |  |
| 51500 | 0.09241182549158111 | 0.04236516446690075 |  |
| 51550 | 0.07392182346666232 | 0.06593119152239524 |  |
| 51600 | 0.08599653374403715 | 0.06965778327867156 |  |
| 51650 | 0.07756886188872159 | 0.07880415237741545 |  |
| 51700 | 0.06632094251108356 | 0.05932001362089068 |  |
| 51750 | 0.05660495808115229 | 0.04101058181549888 |  |
| 51800 | 0.0916222698870115 | 0.07524491025833413 |  |
| 51850 | 0.10037897247821093 | 0.0773036982427584 |  |
| 51900 | 0.07620715408120304 | 0.047994057851610705 |  |
| 51950 | 0.09081061481265351 | 0.06688793895227718 |  |
| 52000 | 0.09693763090763241 | 0.0694590811181115 | 📊 含 eval/save |
| 52050 | 0.06630766758462414 | 0.02288378734374419 |  |
| 52100 | 0.09408279950730503 | 0.06856901534774806 |  |
| 52150 | 0.07104242616333067 | 0.06604570423951373 |  |
| 52200 | 0.07908562006196007 | 0.04531554123968817 |  |
| 52250 | 0.10594228800619021 | 0.05535739158221986 |  |
| 52300 | 0.07196870446205139 | 0.06702623580349609 |  |
| 52350 | 0.08561873168218881 | 0.04909427236998454 |  |
| 52400 | 0.06920152943348512 | 0.04275698104174808 |  |
| 52450 | 0.09412964805960655 | 0.05321602819458349 |  |
| 52500 | 0.09006943530403078 | 0.034916213757242076 |  |
| 52550 | 0.07317296136170626 | 0.07463222538353875 |  |
| 52600 | 0.06710858689621091 | 0.03674794684047811 |  |
| 52650 | 0.07169088430237025 | 0.042770622865646146 |  |
| 52700 | 0.07619533175602555 | 0.06099140901642386 |  |
| 52750 | 0.08292095374781638 | 0.06789518146251794 |  |
| 52800 | 0.05744840553961694 | 0.059149244902073406 |  |
| 52850 | 0.0628801416605711 | 0.03624705223774072 |  |
| 52900 | 0.08980431302916259 | 0.052293357730377465 |  |
| 52950 | 0.05774242937332019 | 0.04390505803166889 |  |
| 53000 | 0.07886796811362728 | 0.0982901073875837 |  |
| 53050 | 0.05632211943157017 | 0.038452131135272793 |  |
| 53100 | 0.11734723340487108 | 0.04720658526639454 |  |
| 53150 | 0.11631877673789859 | 0.049499563057906926 |  |
| 53200 | 0.1260342289460823 | 0.03751992472825805 |  |
| 53250 | 0.06241697212681174 | 0.02825657273933757 |  |
| 53300 | 0.09944339882349595 | 0.045354423054959625 |  |
| 53350 | 0.11963691760320216 | 0.030267439578892663 |  |
| 53400 | 0.10279136418830603 | 0.07133798918221146 |  |
| 53450 | 0.07760854478692636 | 0.08415719679032918 |  |
| 53500 | 0.05104054056573659 | 0.07215803791768849 |  |
| 53550 | 0.05732289081788622 | 0.03795431801700033 |  |

---

## 系统资源占用

> 最后更新：2026-07-09 12:37 CST

### GPU（NVIDIA GeForce RTX 4090）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 61% |
| **显存使用** | 20406 MiB / 24564 MiB (83%) |
| **功耗** | 196.01 W |
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
| `/` (overlay) | 5.0G / 30G (17%) |
| `/localdisk-tmp` | 0 / 100G (0%) |
| `/disk/rl` | 532T / 700T (76%) |

---

## 输出目录

```
playground/mowa_ckpt/MoWA-E-001_starflow_ft0_mowa_4090_long_260706_1740/
├── checkpoints/                 （7 个 checkpoint，最新: steps_52000）
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
