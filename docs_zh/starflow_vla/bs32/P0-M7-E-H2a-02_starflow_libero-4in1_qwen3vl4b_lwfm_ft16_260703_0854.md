# P0-M7-E-H2a-02: StarFlow LIBERO 4-in-1 Qwen3VL-4B future_tokens=16（bs32）

> **实验代号**: E-H2a-02 / P0-M7  
> **状态**: 🟢 训练运行中  
> **run_id**: `P0-M7-E-H2a-02_starflow_libero-4in1_qwen3vl4b_lwfm_ft16_260703_0854`  
> **启动时间**: 2026-07-03 09:25:54 CST  
> **当前更新**: 2026-07-03 16:12:20 CST
> **tmux 会话**: `train`（attached）  
> **配置来源**: `configs/starflow_vla/ablations/future_tokens_16.yaml`

---

## 实验概述

P0-M7 **future tokens 对照**：在 StarFlowVLA 框架中将 action head 的 `num_target_vision_tokens` 设为 **16**（对比 H2a-04 的 ft64 和 H2a-03 的 ft32），验证更少目标视觉 token 对收敛速度与显存占用的影响。

本 run 为 **从 scratch 全新训练**（`is_resume=False`），使用 bs32 配置（per_device_batch_size=1 × gradient_accumulation_steps=32）。与 H2a-04（ft64, per_device=8 × grad_accum=4）相比，本 run 将显存压力从 per_device_batch 转移到 grad_accum，以适配 RTX 4090 24GB 显存。

---

## 训练参数

### 脚本级参数

| 参数 | 值 |
|------|-----|
| `RUN_ID` | `P0-M7-E-H2a-02_starflow_libero-4in1_qwen3vl4b_lwfm_ft16_260703_0854` |
| `CONFIG_YAML` | `configs/starflow_vla/ablations/future_tokens_16.yaml` |
| `DATA_MIX` | `libero_all` |
| `MAX_TRAIN_STEPS` | 80,000 |
| `SAVE_INTERVAL` | 250 |
| `EVAL_INTERVAL` | 500 |
| `LOGGING_FREQUENCY` | 20 |
| `NUM_PROCESSES` | 1 |
| `GRADIENT_ACCUMULATION_STEPS` | **32** |
| `PER_DEVICE_BATCH_SIZE` | **1** |
| `NUM_WORKERS` | 2 |
| `BASE_VLM` | `/disk/rl/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct` |
| `LIBERO_DATA_ROOT` | `/disk/rl/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA` |
| `WANDB_PROJECT` | `starflow_vla` |
| `WANDB_ENTITY` | `silencewx-harbin-institute-of-technology` |
| `is_resume` | **`False`**（从 scratch） |
| 设备 | NVIDIA GeForce RTX 4090（24564 MiB） |
| 单价 | 本地 RTX 4090，暂不记录 |

### 模型 / 优化器参数

| 参数 | 值 |
|------|-----|
| **框架** | StarFlowVLA |
| **Base VLM** | Qwen3-VL-4B-Instruct |
| **Action Model** | LayerwiseFM (DiT, 36 layers, 1024 hidden) |
| **Action Dim** | 7 |
| **State Dim** | 8 |
| **Action Horizon** | 8 |
| **Future Action Window Size** | 7 |
| **Num Target Vision Tokens** | **16** |
| **DiT Attention Heads** | 16 |
| **DiT Params** | ~532 M |
| **Dropout** | 0.2 |
| **Total Params** | ~5,071 M |
| **Trainable Params** | ~633 M |
| **Frozen Modules** | `qwen_vl_interface` |
| **Learning Rate (action_model)** | 1.0e-4 |
| **Learning Rate (base)** | 2.5e-5 |
| **Learning Rate (qwen_vl_interface)** | 1.0e-5 |
| **LR Scheduler** | cosine_with_min_lr |
| **Min LR** | 1.0e-6 |
| **Warmup Steps** | 0 |
| **Optimizer** | AdamW (β=(0.9, 0.95), eps=1e-8, wd=1e-8) |
| **Max Grad Norm** | 1.0 |
| **Gradient Checkpointing** | true |
| **Mixed Precision** | no |
| **Effective Global Batch** | **32**（1 × 32） |
| **Checkpoint Format** | lightweight |
| **Save Format** | safetensors |
| **local_checkpoint_root** | `/localdisk-tmp` |
| **local_checkpoint_keep_count** | 2 |
| **Seed** | 42 |

### 数据集

| 数据集 | 样本数 | embodiment |
|--------|--------|------------|
| `libero_object_no_noops_1.0.0_lerobot` | 66,984 | FRANKA |
| `libero_goal_no_noops_1.0.0_lerobot` | 52,042 | FRANKA |
| `libero_spatial_no_noops_1.0.0_lerobot` | 52,970 | FRANKA |
| `libero_10_no_noops_1.0.0_lerobot` | 101,469 | FRANKA |
| **合计** | **273,465** | — |

---

## 训练进度

> 最后更新：2026-07-04 11:00:00 CST

| 指标 | 值 |
|------|-----|
| **当前 Step** | **29035 / 80000**（36.3%） |
| **完成比例** | 36.3% |
| **单步耗时** | ~6.68 s/it |
| **模型前向/反向耗时** | ~0.21 s |
| **数据加载耗时** | ~0.000 s |
| **已运行时间** | 约 53 小时 |
| **预计剩余时间** | ~93 小时（约 3.9 天） |
| **最新 checkpoint** | `steps_29000` |
| **最新 eval mse_score** | **0.0157**（step 28500） |
| **上一个 checkpoint** | `steps_17500` |

### Loss 记录

| Step | action_dit_loss | 备注 |
|------|----------------|------|
| 20 | 1.5602 | 初始 |
| 40 | 0.9320 | |
| 60 | 0.5074 | |
| 80 | 0.6580 | |
| 100 | 0.4781 | |
| 120 | 0.4514 | |
| 140 | 0.3756 | |
| 160 | 0.3818 | |
| 180 | 0.3527 | |
| 200 | 0.3742 | |
| 220 | 0.3160 | |
| 240 | 0.3971 | |
| 260 | 0.3321 | |
| 280 | 0.3236 | |
| 300 | 0.2634 | |
| 320 | 0.2566 | |
| 400 | 0.2715 | |
| 420 | 0.2446 | |
| 440 | 0.2727 | |
| 460 | 0.2532 | |
| 480 | 0.2536 | |
| **500** | **0.2418** | eval mse_score=**0.03544**；✅ checkpoint |
| 520 | 0.2307 | |
| 1000 | 0.1832 | eval mse_score=? |
| 1500 | 0.1582 | |
| 2000 | 0.1811 | |
| 2500 | 0.1367 | eval mse_score=? |
| 3000 | 0.1370 | |
| 3500 | 0.1447 | |
| 4000 | 0.1014 | eval mse_score=? |
| **4200** | **0.1043** | |
| 4400 | 0.1140 | |
| 4600 | 0.1280 | |
| **4740** | **0.1430** | |
| 5000 | 0.1300 | eval mse_score=? |
| **5260** | **0.1395** | |
| 5500 | 0.1280 | eval mse_score=? |
| **5800** | **0.1136** | |
| 6000 | 0.1200 | eval mse_score=? |
| **6340** | **0.1090** | |
| 6500 | 0.1250 | |
| **6880** | **0.1050** | |
| 10000 | 0.0920 | |
| 15000 | 0.0860 | |
| **17080** | **0.0841** | |
| **17640** | **0.0773** | |
| **18180** | **0.0708** | 🔵 当前 |
| **250** | — | ✅ checkpoint |
| **500** | — | ✅ checkpoint |
| **1000** | — | ✅ checkpoint |
| **1500** | — | ✅ checkpoint |
| **2000** | — | ✅ checkpoint |
| **2500** | — | ✅ checkpoint |
| **3000** | — | ✅ checkpoint |
| **3500** | — | ✅ checkpoint |
| **3750** | — | ✅ checkpoint |
| **4000** | — | ✅ checkpoint |
| **4250** | — | ✅ checkpoint |
| **4500** | — | ✅ checkpoint |
| **4750** | — | ✅ checkpoint |
| **5000** | — | ✅ checkpoint |
| **5250** | — | ✅ checkpoint |
| **5500** | — | ✅ checkpoint |
| **5750** | — | ✅ checkpoint |
| **6000** | — | ✅ checkpoint |
| **6250** | — | ✅ checkpoint |
| **6500** | — | ✅ checkpoint |
| **6750** | — | ✅ checkpoint |

### Checkpoint 列表

| Step | 保存时间 |
|------|----------|
| 250 | Jul 3 ~10:02 |
| 500 | Jul 3 10:21 |

### Loss 趋势

- **Phase 1（0–140）**: 快速下降 1.56 → 0.38，网络初步学习动作分布
- **Phase 2（140–520）**: 继续震荡下降，已降至 0.23；下降速度比 H2a-04（ft64）在同等区间更快（H2a-04 step 500 为 0.24，H2a-02 step 500 为 0.24 — 基本持平）
- eval mse_score 在 step 500 为 **0.03544**，比 H2a-04 同期（0.0116）差约 3×；可能因 ft16 目标 token 数减少导致预测精度下降
- grad_accum=32 导致 last_micro_loss 与平均 loss 的方差较大（单条样本 vs 32 条平均）
- 学习率接近初始值：action_model ~9.999e-5，base ~2.5e-5

---

## 系统资源占用

> 最后更新：2026-07-03 16:12:20 CST

### GPU（NVIDIA GeForce RTX 4090）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 90% |
| **显存使用** | 23800 MiB / 24564 MiB（96.9%） |
| **显存空闲** | ~800 MiB |
| **功耗** | 288.39 W / 450.00 W |
| **温度** | 65°C |

### Docker 内存（cgroup v2）

| 指标 | 值 |
|------|-----|
| **Docker 内存上限** | 56 GiB |
| **当前已用** | 22 GiB |
| **使用率** | 39.3% |
| **Swap** | 0 B |

### 存储

| 挂载点 | 容量 | 已用 | 可用 | 使用率 |
|--------|------|------|------|--------|
| `/` (overlay) | 30G | 16G | 15G | 53% |
| `/localdisk-tmp` | 100G | 31G | 70G | 31% |
| `/disk/rl` | 700T | 548T | 153T | 79% |

---

## 成本估算

| 项目 | 计算 |
|------|------|
| **每 step 耗时** | ~6.5 s/it |
| **已运行时间** | ~1.2 小时 |
| **完整 80000 steps 预估** | 80000 × 6.5s ≈ 144h ≈ 6.0 天 |
| **每 step 成本** | 本地 RTX 4090，免费 |

---

## 输出目录

```
/disk/rl/starVLA/playground/Checkpoints/P0-M7-E-H2a-02_starflow_libero-4in1_qwen3vl4b_lwfm_ft16_260703_0854/
├── checkpoints/
│   ├── steps_250/
│   └── steps_500/
├── config.full.yaml          ✅
├── config.yaml               ✅
├── dataset_statistics.json   ✅
├── summary.jsonl             ✅
└── wandb/                    ✅
```

本地暂存：`/localdisk-tmp/P0-M7-E-H2a-02_starflow_libero-4in1_qwen3vl4b_lwfm_ft16_260703_0854/`

---

## 相关链接

- **WandB Run**: [260703_0854](https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla/runs/P0-M7-E-H2a-02_starflow_libero-4in1_qwen3vl4b_lwfm_ft16_260703_0854)
- **tmux 会话**: `train`

---

## 启动命令

```bash
cd /disk/rl/starVLA
RUN_ID="P0-M7-E-H2a-02_starflow_libero-4in1_qwen3vl4b_lwfm_ft16_$(date +%y%m%d_%H%M)" \
CONFIG_YAML=configs/starflow_vla/ablations/future_tokens_16.yaml \
DATA_MIX=libero_all \
MAX_TRAIN_STEPS=80000 \
SAVE_INTERVAL=250 \
LOGGING_FREQUENCY=20 \
EVAL_INTERVAL=500 \
NUM_PROCESSES=1 \
GRADIENT_ACCUMULATION_STEPS=32 \
PER_DEVICE_BATCH_SIZE=1 \
NUM_WORKERS=2 \
FREEZE_MODULES=qwen_vl_interface \
LOCAL_CHECKPOINT_KEEP_COUNT=2 \
LOCAL_CHECKPOINT_ROOT=/localdisk-tmp \
STARVLA_PYTHON=/opt/conda/envs/starVLA/bin/python \
bash examples/LIBERO/train_files/run_starflow_train_ready.sh --trainer.eval_num_batches=8
```

---

## 备注

- 本实验验证 `num_target_vision_tokens=16` 对 StarFlowVLA 动作学习的影响。
- 与 H2a-04（ft64, device_bs=8×acc=4）配置不同，本 run 使用 device_bs=1×acc=32 以适配 RTX 4090 24GB。
- GPU 利用率 54%（远低于 H2a-04 的 100%），可能因 per_device_batch=1 导致 GPU 计算不饱和，大量时间花在 grad_accum 的 CPU-GPU 同步上。
- 显存 96.9% 与 H2a-04 持平；显存瓶颈在 frozen VLM 而非 batch size。
- 单步耗时 ~6.5 s/it，但其中 micro-step 模型前向仅 ~0.22s，grad_accum=32 的同步开销显著。
- 后续可考虑在 4090 上优化 per_device_batch_size 与 grad_accum 的平衡，以提升 GPU 利用率。

---

*本文档将持续更新。*
