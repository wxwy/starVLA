# MoWA-E-001: StarFlowVLA ft0 mowa_main（robocasa365, bs32, 带 future supervision）

> **实验代号**: E-001 / mowa_main
> **状态**: 🟢 训练运行中
> **run_id**: `MoWA-E-001_starflow_ft0_mowa_4090_long_260706_1740`
> **启动时间**: 2026-07-06 17:35 CST
> **当前更新**: 2026-07-07 09:28 CST
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

> 最后更新：2026-07-07 09:28 CST
> 训练启动 15h 53m

| 指标 | 值 |
|------|-----|
| **当前 Step** | ~12700 / 80000 |
| **完成比例** | 15.88% |
| **单步耗时** | — |
| **model_time** | 0.24941703502554446s |
| **data_time** | 0.00038191897328943014s |
| **已运行时间** | 15h 53m |
| **预计剩余时间** | 84h 13m |
| **最新 checkpoint** | steps_12000（共 6 个） |
| **最新 eval mse_score** | 首次 eval 在 step 2000 |

### Loss 记录

| Step | action_dit_loss | mowa_future_supervision_loss | 备注 |
|------|----------------|------------------------------|------|
| 9750 | 0.08064575190655887 | 0.06915230525191873 |  |
| 9800 | 0.09477299416903406 | 0.06460452469764277 |  |
| 9850 | 0.14317512838169932 | 0.08105368074029684 |  |
| 9900 | 0.13589666818734258 | 0.05545610631816089 |  |
| 9950 | 0.1727012018673122 | 0.0742253188509494 |  |
| 10000 | 0.1528068482875824 | 0.11422797595150769 | 📊 含 eval/save |
| 10050 | 0.09804783924482763 | 0.05670265975641087 |  |
| 10100 | 0.12848385132383555 | 0.03783566600759514 |  |
| 10150 | 0.12611661583650857 | 0.0813969622249715 |  |
| 10200 | 0.14070474333129823 | 0.04400053851713892 |  |
| 10250 | 0.16901735425926745 | 0.059350454539526254 |  |
| 10300 | 0.11798231094144285 | 0.042705701056547696 |  |
| 10350 | 0.15272843558341265 | 0.05606118278228678 |  |
| 10400 | 0.058826811611652374 | 0.06539816003351007 |  |
| 10450 | 0.09655381680931896 | 0.047687131256680004 |  |
| 10500 | 0.10581232653930783 | 0.07387636788189411 |  |
| 10550 | 0.1241876608110033 | 0.09248752452549525 |  |
| 10600 | 0.10545611009001732 | 0.0758028332493268 |  |
| 10650 | 0.14173744362778962 | 0.17408071414683945 |  |
| 10700 | 0.08542508230311796 | 0.07648662984138355 |  |
| 10750 | 0.09615278651472181 | 0.04865960753522813 |  |
| 10800 | 0.141810669680126 | 0.029024603005382232 |  |
| 10850 | 0.10555841692257673 | 0.032248278715997 |  |
| 10900 | 0.14188953558914363 | 0.045054802612867206 |  |
| 10950 | 0.10627985582686961 | 0.05772917112335563 |  |
| 11000 | 0.11988457338884473 | 0.08839217317290604 |  |
| 11050 | 0.17604645085521042 | 0.06159211427439004 |  |
| 11100 | 0.1354563005734235 | 0.09222532718558796 |  |
| 11150 | 0.08312843611929566 | 0.07749153953045607 |  |
| 11200 | 0.10566686454694718 | 0.0741968784132041 |  |
| 11250 | 0.11659044306725264 | 0.11380100902169943 |  |
| 11300 | 0.08212109713349491 | 0.05780002297979081 |  |
| 11350 | 0.10828169400338084 | 0.07266280788462609 |  |
| 11400 | 0.15465389587916434 | 0.025617483581299894 |  |
| 11450 | 0.11576360079925507 | 0.08344740379834548 |  |
| 11500 | 0.11287481302861124 | 0.07835859712213278 |  |
| 11550 | 0.08743609068915248 | 0.06496297853300348 |  |
| 11600 | 0.11596394143998623 | 0.06634269290952943 |  |
| 11650 | 0.08824287878815085 | 0.07384707912569866 |  |
| 11700 | 0.1390665372600779 | 0.11230632720980793 |  |
| 11750 | 0.09598462225403637 | 0.030683747681905515 |  |
| 11800 | 0.076291250763461 | 0.07974522060249001 |  |
| 11850 | 0.09634145733434707 | 0.07409611728508025 |  |
| 11900 | 0.1639660745859146 | 0.0719178126892075 |  |
| 11950 | 0.11311416863463819 | 0.06824406470695976 |  |
| 12000 | 0.08095732086803764 | 0.06319398595951498 | 📊 含 eval/save |
| 12050 | 0.13368296308908612 | 0.08998410333879292 |  |
| 12100 | 0.081675142981112 | 0.0851819246308878 |  |
| 12150 | 0.10968653787858784 | 0.046750619885642664 |  |
| 12200 | 0.0999940856709145 | 0.061785438669176074 |  |
| 12250 | 0.13196576049085706 | 0.041850916284602135 |  |
| 12300 | 0.14163281529909 | 0.07032581405655947 |  |
| 12350 | 0.1060564024373889 | 0.08305972890229896 |  |
| 12400 | 0.11069537000730634 | 0.05546527309343219 |  |
| 12450 | 0.13045164494542405 | 0.06655825236521196 |  |
| 12500 | 0.1244369424530305 | 0.07836417967337184 |  |
| 12550 | 0.1324742419528775 | 0.06238882905745413 |  |
| 12600 | 0.16508890374097973 | 0.06216305508860387 |  |
| 12650 | 0.1054180838400498 | 0.1275393075775355 |  |
| 12700 | 0.1230369086842984 | 0.03835139451257419 |  |

---

## 系统资源占用

> 最后更新：2026-07-07 09:28 CST

### GPU（NVIDIA GeForce RTX 4090）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 13% |
| **显存使用** | 20406 MiB / 24564 MiB (83%) |
| **功耗** | 184.35 W |
| **温度** | 47°C |

### 系统内存

| 指标 | 值 |
|------|-----|
| **总量** | 503Gi |
| **已用** | 44Gi |
| **可用** | 455Gi |

### 存储

| 挂载点 | 使用情况 |
|--------|----------|
| `/` (overlay) | 4.5G / 30G (15%) |
| `/localdisk-tmp` | 0 / 100G (0%) |
| `/disk/rl` | 555T / 700T (80%) |

---

## 输出目录

```
playground/mowa_ckpt/MoWA-E-001_starflow_ft0_mowa_4090_long_260706_1740/
├── checkpoints/                 （6 个 checkpoint，最新: steps_12000）
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
