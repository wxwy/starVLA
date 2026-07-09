# MoWA-E-001: StarFlowVLA ft0 mowa_main（robocasa365, bs32, 带 future supervision）

> **实验代号**: E-001 / mowa_main
> **状态**: 🟢 训练运行中
> **run_id**: `MoWA-E-001_starflow_ft0_mowa_4090_long_260706_1740`
> **启动时间**: 2026-07-06 17:35 CST
> **当前更新**: 2026-07-09 16:37 CST
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

> 最后更新：2026-07-09 16:37 CST
> 训练启动 71h 2m

| 指标 | 值 |
|------|-----|
| **当前 Step** | ~56750 / 80000 |
| **完成比例** | 70.94% |
| **单步耗时** | — |
| **model_time** | 0.24941703502554446s |
| **data_time** | 0.00042275688610970974s |
| **已运行时间** | 71h 2m |
| **预计剩余时间** | 29h 6m |
| **最新 checkpoint** | steps_56000（共 9 个） |
| **最新 eval mse_score** | 首次 eval 在 step 2000 |

### Loss 记录

| Step | action_dit_loss | mowa_future_supervision_loss | 备注 |
|------|----------------|------------------------------|------|
| 53800 | 0.06397039379226044 | 0.0457096998434281 |  |
| 53850 | 0.11770840338431299 | 0.04329653556487756 |  |
| 53900 | 0.08732674107886851 | 0.04622301331255585 |  |
| 53950 | 0.06758786051068455 | 0.0255267663160339 |  |
| 54000 | 0.0779485184175428 | 0.04224152771348599 | 📊 含 eval/save |
| 54050 | 0.07129917299607769 | 0.03742181434063241 |  |
| 54100 | 0.05100733845029026 | 0.05389127384114545 |  |
| 54150 | 0.06382554658921435 | 0.03276858513709158 |  |
| 54200 | 0.0430442848301027 | 0.04773108629160561 |  |
| 54250 | 0.07054574938956648 | 0.08527845233402331 |  |
| 54300 | 0.08097280171932653 | 0.052717243204824626 |  |
| 54350 | 0.09848586423322558 | 0.022953407307795715 |  |
| 54400 | 0.12290211841173004 | 0.09095692182017956 |  |
| 54450 | 0.0822931150905788 | 0.033280852338066325 |  |
| 54500 | 0.08079265191918239 | 0.022079212503740564 |  |
| 54550 | 0.07861080026486889 | 0.08790657049394213 |  |
| 54600 | 0.138740563968895 | 0.05223680310336931 |  |
| 54650 | 0.0641545529360883 | 0.05297169659752399 |  |
| 54700 | 0.08726247237063944 | 0.030161693452100735 |  |
| 54750 | 0.12990161357447505 | 0.046449078423393075 |  |
| 54800 | 0.07529743190389127 | 0.031156913639279082 |  |
| 54850 | 0.07852619685581885 | 0.04439068678766489 |  |
| 54900 | 0.08013217651750892 | 0.06713442565524019 |  |
| 54950 | 0.09056487653288059 | 0.06921095292273094 |  |
| 55000 | 0.08317176278796978 | 0.044379806917277165 |  |
| 55050 | 0.07614637995720841 | 0.07736174087040126 |  |
| 55100 | 0.06376757135149091 | 0.027961690069787437 |  |
| 55150 | 0.04942093760473654 | 0.024565792438806966 |  |
| 55200 | 0.12727597751654685 | 0.06217526888940483 |  |
| 55250 | 0.08418937656097114 | 0.04480622293340275 |  |
| 55300 | 0.051343422615900636 | 0.0718842800706625 |  |
| 55350 | 0.10605287746875547 | 0.026815814824658446 |  |
| 55400 | 0.08262634190032259 | 0.055357877386995824 |  |
| 55450 | 0.04986476560588926 | 0.04355528030282585 |  |
| 55500 | 0.0874144954723306 | 0.09720663121697726 |  |
| 55550 | 0.10072431183652952 | 0.030323818122269586 |  |
| 55600 | 0.11125032277777791 | 0.07985115534393117 |  |
| 55650 | 0.06966162740718573 | 0.058865523256827146 |  |
| 55700 | 0.06488113838713616 | 0.05003404800663702 |  |
| 55750 | 0.06177494826260954 | 0.06026784701680299 |  |
| 55800 | 0.07926647691056132 | 0.06594986864365637 |  |
| 55850 | 0.07730068953242153 | 0.05217298280331306 |  |
| 55900 | 0.11482535838149488 | 0.08604766993084922 |  |
| 55950 | 0.12124770658556372 | 0.03783936208856176 |  |
| 56000 | 0.09887863291078247 | 0.08075863659541938 | 📊 含 eval/save |
| 56050 | 0.06778478357591666 | 0.06056085845557391 |  |
| 56100 | 0.07809619943145663 | 0.020217504730680957 |  |
| 56150 | 0.04744801117340103 | 0.10199276627099607 |  |
| 56200 | 0.09036742593161762 | 0.09829566894040909 |  |
| 56250 | 0.07808072018087842 | 0.05959516472648829 |  |
| 56300 | 0.05349680318613537 | 0.0569241970370058 |  |
| 56350 | 0.08905915811192244 | 0.05549535030149855 |  |
| 56400 | 0.06511075340677053 | 0.08754041168140247 |  |
| 56450 | 0.07083062711171806 | 0.11736038791786996 |  |
| 56500 | 0.09477404976496473 | 0.01795183123613242 |  |
| 56550 | 0.08879147219704464 | 0.10878758018952794 |  |
| 56600 | 0.04141473618801683 | 0.044388472408172674 |  |
| 56650 | 0.11420477280626073 | 0.03281523166515399 |  |
| 56700 | 0.06583114678505808 | 0.05312949730432592 |  |
| 56750 | 0.0737143037840724 | 0.07429117211722769 |  |

---

## 系统资源占用

> 最后更新：2026-07-09 16:37 CST

### GPU（NVIDIA GeForce RTX 4090）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 48% |
| **显存使用** | 20406 MiB / 24564 MiB (83%) |
| **功耗** | 199.40 W |
| **温度** | 54°C |

### 系统内存

| 指标 | 值 |
|------|-----|
| **总量** | 503Gi |
| **已用** | 44Gi |
| **可用** | 455Gi |

### 存储

| 挂载点 | 使用情况 |
|--------|----------|
| `/` (overlay) | 5.0G / 30G (17%) |
| `/localdisk-tmp` | 0 / 100G (0%) |
| `/disk/rl` | 533T / 700T (77%) |

---

## 输出目录

```
playground/mowa_ckpt/MoWA-E-001_starflow_ft0_mowa_4090_long_260706_1740/
├── checkpoints/                 （9 个 checkpoint，最新: steps_56000）
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
