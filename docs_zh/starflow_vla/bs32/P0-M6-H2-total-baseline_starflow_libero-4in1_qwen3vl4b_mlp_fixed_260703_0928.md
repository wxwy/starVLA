# P0-M6-H2-total-baseline: StarFlow LIBERO 4-in-1 Qwen3VL-4B MLP baseline（bs32）

> **实验代号**: H2-total-baseline / P0-M6  
> **状态**: 🟢 训练运行中  
> **run_id**: `P0-M6-H2-total-baseline_starflow_libero-4in1_qwen3vl4b_mlp_fixed_260703_0928`  
> **启动时间**: 2026-07-03 09:28 CST  
> **当前更新**: 2026-07-05 14:19:00 CST
> **tmux 会话**: `train`（attached）  
> **配置来源**: `configs/starflow_vla/stage2_mlp_baseline.yaml`

---

## 实验概述

P0-M6 **Stage2 MLP baseline**：使用纯 MLP action head（无 DiT、无 future tokens、无 state conditioning），验证 StarFlowVLA 框架在最简单 baseline 下的收敛行为。

与 P0-M7（ft64 DiT）和 H2a-02（ft16 DiT）形成对照：MLP head 参数量远小于 DiT，显存占用更低，预期收敛速度较慢但单步更快。

本 run 为 **从 scratch 全新训练**（`is_resume=False`），使用 bs32 配置（per_device_batch_size=1 × gradient_accumulation_steps=32）。

---

## 训练参数

### 脚本级参数

| 参数 | 值 |
|------|-----|
| `RUN_ID` | `P0-M6-H2-total-baseline_starflow_libero-4in1_qwen3vl4b_mlp_fixed_260703_0928` |
| `CONFIG_YAML` | `configs/starflow_vla/stage2_mlp_baseline.yaml` |
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
| **框架** | QwenOFT（连续动作并行输出） |
| **Base VLM** | Qwen3-VL-4B-Instruct |
| **Action Model** | **MLP**（纯多层感知机，非 DiT） |
| **Action Hidden Dim** | 2,560 |
| **Action Dim** | 7 |
| **Action Horizon** | 8 |
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

> 最后更新：2026-07-04 09:19:00 CST

| 指标 | 值 |
|------|-----|
| **当前 Step** | **48520 / 80000**（60.7%） |
| **完成比例** | 60.7% |
| **单步耗时** | ~3.27 s/it |
| **数据加载耗时** | ~0.000 s |
| **模型前向/反向耗时** | ~0.098 s |
| **已运行时间** | 约 45 小时 |
| **预计剩余时间** | ~28 小时（约 1.2 天） |
| **最新 checkpoint** | `steps_48500` |

### Loss 记录（部分）

| Step | action_dit_loss | 备注 |
|------|----------------|------|
| 940 | 0.2493 | |
| 1000 | 0.2277 | |
| 5000 | 0.1997 | |
| 10000 | 0.1702 | |
| 20000 | 0.1420 | |
| 30000 | 0.1190 | |
| 47420 | 0.0896 | 🏆 最低 |
| 48480 | 0.1163 | |
| 48500 | 0.1301 | |
| **48520** | **0.1062** | 🔵 当前 |

### Loss 趋势

- 初始 0.25 → 稳步下降至 0.10–0.13 区间
- step 34460 最低 0.105，当前在 0.12–0.13 小幅波动
- MLP baseline 收敛远超预期，已过半程

> 注：虽然 action head 是 MLP 而非 DiT，训练日志中 loss 字段名仍为 `action_dit_loss`（历史遗留命名）。

### Loss 趋势

- 启动阶段（0–1000），loss 在 0.23–0.25 区间波动
- step 1000-5000：缓慢下降至 0.20 附近
- step 5000-9000：在 0.18-0.21 区间震荡
- **step 9000-10600：跌破 0.18，降至 0.15-0.17 区间**，MLP 仍在缓慢下降

---

## 系统资源占用

> 最后更新：2026-07-03 21:19:00 CST

### GPU（NVIDIA GeForce RTX 4090）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 19% |
| **显存使用** | 10330 MiB / 24564 MiB（42.1%）
| **显存空闲** | 14,234 MiB |
| **功耗** | 143.91 W / 450.00 W |
| **温度** | 47°C |

> MLP head 显存仅 ~10.3 GB，loss 已降至 0.105，MLP 收敛远超预期。

### Docker 内存（cgroup v2）

| 指标 | 值 |
|------|-----|
| **Docker 内存总量** | 56.0 GiB
| **Docker 内存已用** | ~20.5 GiB
| **Docker 内存可用** | ~35.5 GiB
| **Swap** | 0 B |

### 存储

| 挂载点 | 容量 | 已用 | 可用 | 使用率 |
|--------|------|------|------|--------|
| `/` (overlay) | 30G | ~1G | ~29G | ~3% |
| `/localdisk-tmp` | 100G | — | — | — |
| `/disk/rl` | 700T | 548T | 153T | 79% |

---

## checkpoint 时间线

| checkpoint | 时间 | 备注 |
|------------|------|------|
| `steps_250` | ~09:56 | ✅ |
| `steps_500` | ~10:10 | ✅ |
| `steps_750` | ~10:24 | ✅ |
| `steps_1000` | ~10:39 | ✅ 最新 |

---

## 成本估算

| 项目 | 计算 |
|------|------|
| **每 step 耗时** | ~3.46 s/it |
| **每 step 成本** | 本地 4090，暂不记录 |
| **已运行时间** | ~58 分钟 |
| **完整 80000 steps 预估** | 80000 × 3.46s ≈ 77h ≈ 3.2 天 |

---

## 输出目录

```
/disk/rl/starVLA/playground/Checkpoints/P0-M6-H2-total-baseline_starflow_libero-4in1_qwen3vl4b_mlp_fixed_260703_0928/
├── checkpoints/
│   ├── steps_250/
│   ├── steps_500/
│   ├── steps_750/
│   └── steps_1000/      ← 最新
├── config.full.yaml          ✅
├── config.yaml               ✅
├── dataset_statistics.json   ✅
├── summary.jsonl             ✅
└── wandb/                    ✅
```

本地暂存：`/localdisk-tmp/P0-M6-H2-total-baseline_starflow_libero-4in1_qwen3vl4b_mlp_fixed_260703_0928/`

---

## 相关链接

- **WandB Run**: [260703_0928](https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla/runs/P0-M6-H2-total-baseline_starflow_libero-4in1_qwen3vl4b_mlp_fixed_260703_0928)
- **对照实验**: P0-M7-E-H2a-04（ft64 DiT）、H2a-02（ft16 DiT）
- **tmux 会话**: `train`

---

## 启动命令

```bash
cd /disk/rl/starVLA
RUN_ID="P0-M6-H2-total-baseline_starflow_libero-4in1_qwen3vl4b_mlp_fixed_$(date +%y%m%d_%H%M)" \
CONFIG_YAML=configs/starflow_vla/stage2_mlp_baseline.yaml \
DATA_MIX=libero_all \
MAX_TRAIN_STEPS=80000 \
SAVE_INTERVAL=250 \
LOGGING_FREQUENCY=20 \
EVAL_INTERVAL=500 \
NUM_PROCESSES=1 \
GRADIENT_ACCUMULATION_STEPS=32 \
PER_DEVICE_BATCH_SIZE=1 \
NUM_WORKERS=2 \
LOCAL_CHECKPOINT_KEEP_COUNT=2 \
LOCAL_CHECKPOINT_ROOT=/localdisk-tmp \
STARVLA_PYTHON=/opt/conda/envs/starVLA/bin/python \
bash examples/LIBERO/train_files/run_starflow_train_ready.sh
```

---

## 备注

- 本实验为 Stage2 MLP baseline，验证纯 MLP action head 在 StarFlowVLA 中的收敛表现。
- MLP head 参数量远小于 DiT，显存仅 ~10.3 GB（42%），单步耗时 ~3.5 s，3.2 天可完成 80000 steps。
- 对照 P0-M7（ft64 DiT）：loss 更低（~0.1 vs ~0.25），但显存更高（96.9% vs 42%），单步更慢（9.2 s vs 3.5 s）。
- GPU 利用率偏低（32%），可能因 per_device_batch_size=1 + grad_accum=32 导致 GPU 在 micro-step 间有空闲。
- 可考虑增大 per_device_batch_size 提高 GPU 利用率（目前显存余量充足，~14 GB 空闲）。
- `action_dit_loss` 命名是历史遗留，实际为 MLP 输出的 MSE loss。

---

*本文档将持续更新。*
