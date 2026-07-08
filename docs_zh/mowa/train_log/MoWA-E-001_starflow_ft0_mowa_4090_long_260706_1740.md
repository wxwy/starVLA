# MoWA-E-001: StarFlowVLA ft0 mowa_main（robocasa365, bs32, 带 future supervision）

> **实验代号**: E-001 / mowa_main
> **状态**: 🟢 训练运行中
> **run_id**: `MoWA-E-001_starflow_ft0_mowa_4090_long_260706_1740`
> **启动时间**: 2026-07-06 17:35 CST
> **当前更新**: 2026-07-08 12:37 CST
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

> 最后更新：2026-07-08 12:37 CST
> 训练启动 43h 2m

| 指标 | 值 |
|------|-----|
| **当前 Step** | ~34450 / 80000 |
| **完成比例** | 43.06% |
| **单步耗时** | — |
| **model_time** | 0.24941703502554446s |
| **data_time** | 0.00041020801290869713s |
| **已运行时间** | 43h 2m |
| **预计剩余时间** | 56h 55m |
| **最新 checkpoint** | steps_34000（共 10 个） |
| **最新 eval mse_score** | 首次 eval 在 step 2000 |

### Loss 记录

| Step | action_dit_loss | mowa_future_supervision_loss | 备注 |
|------|----------------|------------------------------|------|
| 31500 | 0.08974814915563911 | 0.04977061311365105 |  |
| 31550 | 0.10153096966678277 | 0.0695352963230107 |  |
| 31600 | 0.12668320687953383 | 0.021676899734302424 |  |
| 31650 | 0.09045706171309575 | 0.06874795333715156 |  |
| 31700 | 0.07011795102152973 | 0.024056873691733927 |  |
| 31750 | 0.08734784874832258 | 0.04169401776744053 |  |
| 31800 | 0.11092493636533618 | 0.07301580827333964 |  |
| 31850 | 0.06550024630269036 | 0.0729229269491043 |  |
| 31900 | 0.11491624196060002 | 0.04020433656114619 |  |
| 31950 | 0.11099614464910701 | 0.052229336462914944 |  |
| 32000 | 0.09109600947704166 | 0.03858953376766294 | 📊 含 eval/save |
| 32050 | 0.04807009140495211 | 0.06642126140650362 |  |
| 32100 | 0.09245175955584273 | 0.07546341259148903 |  |
| 32150 | 0.11947661521844566 | 0.0640918182907626 |  |
| 32200 | 0.08667566912481561 | 0.03243987834866857 |  |
| 32250 | 0.06303098856005818 | 0.06845857248117682 |  |
| 32300 | 0.06894720951095223 | 0.06247604399686679 |  |
| 32350 | 0.14424176968168467 | 0.07679711107630283 |  |
| 32400 | 0.08613011328270659 | 0.058693783212220296 |  |
| 32450 | 0.0582740917452611 | 0.0646317763676052 |  |
| 32500 | 0.08733588445466012 | 0.03401464712806046 |  |
| 32550 | 0.08973864489234984 | 0.04389484228522633 |  |
| 32600 | 0.05917675333330408 | 0.026067410544783343 |  |
| 32650 | 0.09330372663680464 | 0.05304656310181599 |  |
| 32700 | 0.13850003737024963 | 0.023860070068622008 |  |
| 32750 | 0.07625885674497113 | 0.05526090046623722 |  |
| 32800 | 0.1252131531946361 | 0.10012506623752415 |  |
| 32850 | 0.11966608659713529 | 0.07508570339996368 |  |
| 32900 | 0.06516326760174707 | 0.0845230768318288 |  |
| 32950 | 0.08284688839921728 | 0.057294401223771274 |  |
| 33000 | 0.09451741341035813 | 0.04966704138496425 |  |
| 33050 | 0.0629242334398441 | 0.047210052609443665 |  |
| 33100 | 0.06605784932617098 | 0.06722318015818018 |  |
| 33150 | 0.08524915587622672 | 0.08471695691696368 |  |
| 33200 | 0.10566052934154868 | 0.06135704166081268 |  |
| 33250 | 0.08392048848327249 | 0.08160863749799319 |  |
| 33300 | 0.09822241507936269 | 0.03916093276347965 |  |
| 33350 | 0.08644086009007879 | 0.021124541584867984 |  |
| 33400 | 0.07544644258450717 | 0.06026368628954515 |  |
| 33450 | 0.06939464528113604 | 0.02902702719438821 |  |
| 33500 | 0.12124001159099862 | 0.07857671563397162 |  |
| 33550 | 0.08587545005138963 | 0.056773178730509244 |  |
| 33600 | 0.05625924532068893 | 0.07343028293689713 |  |
| 33650 | 0.11209281111950986 | 0.049754722422221676 |  |
| 33700 | 0.09721455647377297 | 0.09235916536999866 |  |
| 33750 | 0.07656202506041154 | 0.06324536699685268 |  |
| 33800 | 0.1327510536648333 | 0.10308698314474896 |  |
| 33850 | 0.11130727722775191 | 0.0380717211519368 |  |
| 33900 | 0.0954193738871254 | 0.050939401087816805 |  |
| 33950 | 0.08937894197879359 | 0.06720898333878722 |  |
| 34000 | 0.11359199427533895 | 0.03248549692216329 | 📊 含 eval/save |
| 34050 | 0.08726204436970875 | 0.04343594692181796 |  |
| 34100 | 0.07143704331247136 | 0.07440292921091896 |  |
| 34150 | 0.08178619560203515 | 0.10664120946603362 |  |
| 34200 | 0.1168970933649689 | 0.052064680319745094 |  |
| 34250 | 0.06882674101507291 | 0.024644931210787036 |  |
| 34300 | 0.07596278935670853 | 0.07800563967248308 |  |
| 34350 | 0.11326554871629924 | 0.038815658626845106 |  |
| 34400 | 0.0858007405186072 | 0.07873153784748865 |  |
| 34450 | 0.08562799461651593 | 0.0544431329035433 |  |

---

## 系统资源占用

> 最后更新：2026-07-08 12:37 CST

### GPU（NVIDIA GeForce RTX 4090）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 15% |
| **显存使用** | 20406 MiB / 24564 MiB (83%) |
| **功耗** | 172.08 W |
| **温度** | 47°C |

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
| `/disk/rl` | 539T / 700T (77%) |

---

## 输出目录

```
playground/mowa_ckpt/MoWA-E-001_starflow_ft0_mowa_4090_long_260706_1740/
├── checkpoints/                 （10 个 checkpoint，最新: steps_34000）
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
