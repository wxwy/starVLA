# P0-M7-E-H2a-01: StarFlow LIBERO 4-in-1 Qwen3VL-4B future_tokens=0（bs32）

> **实验代号**: E-H2a-01 / P0-M7  
> **状态**: 🟢 训练运行中（step 944/80000，约 1.2%）  
> **run_id**: `P0-M7-E-H2a-01_starflow_libero-4in1_qwen3vl4b_lwfm_ft0_260703_0849`  
> **启动时间**: 2026-07-03 08:53 CST  
> **当前更新**: 2026-07-03 15:14:13 CST
> **tmux 会话**: `train`  
> **配置来源**: `configs/starflow_vla/ablations/future_tokens_0.yaml`

---

## 实验概述

P0-M7 **future tokens 对照**：在 StarFlowVLA 框架中将 action head 的 `num_target_vision_tokens` 设为 **0**（纯 state→action），验证零视觉 token 注入对 LIBERO 4-in-1 连续动作学习的收敛下限。

本 run 为 **从 scratch 全新训练**（`is_resume=False`），使用 RTX 4090 bs32 配置（per_device_batch_size=1 × gradient_accumulation_steps=32）。

---

## 训练参数

### 脚本级参数

| 参数 | 值 |
|------|-----|
| `RUN_ID` | `P0-M7-E-H2a-01_starflow_libero-4in1_qwen3vl4b_lwfm_ft0_260703_0849` |
| `CONFIG_YAML` | `configs/starflow_vla/ablations/future_tokens_0.yaml` |
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
| **Num Target Vision Tokens** | **0** |
| **Frozen Modules** | `qwen_vl_interface`（VLM 冻结） |
| **Learning Rate (action_model)** | 1.0e-4 |
| **Learning Rate (base)** | 2.5e-5 |
| **LR Scheduler** | cosine_with_min_lr |
| **Min LR** | 1.0e-6 |
| **Optimizer** | AdamW (β=(0.9, 0.95), eps=1e-8, wd=1e-8) |
| **Max Grad Norm** | 1.0 |
| **Gradient Checkpointing** | true |
| **Effective Global Batch** | 32 (1 × 32) |
| **Mixed Precision** | bf16 |

### 数据集

使用 `libero_all` 混合数据集：

| 数据集 | 样本数 | embodiment |
|--------|--------|------------|
| `libero_object_no_noops_1.0.0_lerobot` | 66,984 | FRANKA |
| `libero_goal_no_noops_1.0.0_lerobot` | 52,042 | FRANKA |
| `libero_spatial_no_noops_1.0.0_lerobot` | 52,970 | FRANKA |
| `libero_10_no_noops_1.0.0_lerobot` | 101,469 | FRANKA |
| **合计** | **273,465** | — |

---

## 训练进度

> 最后更新：2026-07-03 10:48 CST（Docker 容器内实际可用资源）

| 指标 | 值 |
|------|-----|
| **当前 Step** | **3301 / 80000**（4.1%） |
| **完成比例** | 4.1% |
| **单步耗时** | ~7.48 s/it |
| **数据加载耗时** | ~0.000 s |
| **模型前向/反向耗时** | ~0.260 s |
| **已运行时间** | 1 小时 45 分钟 |
| **预计剩余时间** | ~6 天 4 小时 |
| **预计总耗时** | ~6 天 6 小时 |

### Loss 记录

| Step | Loss | 备注 |
|------|------|------|
| 540 | 0.2711 | 初始 warmup |
| 560 | 0.2889 | |
| 580 | 0.1944 | |
| 600 | 0.2138 | |
| 620 | 0.2684 | |
| 640 | 0.2455 | |
| 660 | 0.2224 | |
| 680 | 0.2206 | |
| 700 | 0.2153 | |
| 720 | 0.2062 | |
| 740 | 0.1750 | |
| 760 | 0.2166 | |
| 780 | 0.1606 | |
| 800 | 0.1998 | |
| 820 | 0.1807 | |
| 840 | — | |
| 860 | — | |
| 880 | — | |
| 900 | — | ✅ checkpoint steps_750 |
| 920 | — | |
| 940 | — | 🔵 当前最新（step 944） |

---

## 系统资源占用

> 最后更新：2026-07-03 10:48 CST（Docker 容器内实际可用资源）  
> 注意：以下为 Docker 容器内实际可用资源

### GPU（NVIDIA GeForce RTX 4090）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 32% |
| **显存使用** | 24032 MiB / 24564 MiB（97.8%） |
| **显存空闲** | 50 MiB |
| **功耗** | 215.26 W / 450.00 W |
| **温度** | 57°C |

### CPU / 内存（Docker 容器）

| 指标 | 值 |
|------|-----|
| **内存总量** | 56 GiB（Docker 容器限制） |
| **内存已用** | ~32 GiB |
| **内存可用** | ~24 GiB |

### 存储

| 挂载点 | 总量 | 已用 | 可用 | 使用率 |
|--------|------|------|------|--------|
| `/disk/rl` | 700 T | 549 T | 152 T | 79% |
| `/localdisk-tmp` | 100 G | 20 G | 81 G | 20% |

---

## 输出目录

```
/disk/rl/starVLA/playground/Checkpoints/P0-M7-E-H2a-01_starflow_libero-4in1_qwen3vl4b_lwfm_ft0_260703_0849/
├── checkpoints/              # 模型 checkpoint
│   ├── steps_250/            # 第一个 checkpoint
│   └── steps_500/            # ✅ 最新完整 checkpoint
├── config.full.yaml          # 完整配置
├── config.yaml               # 访问过的配置快照
├── dataset_statistics.json   # 数据集统计信息
├── summary.jsonl             # 训练摘要
└── wandb/                    # WandB 本地日志
```

---

## 相关链接

- **WandB Project**: [starflow_vla](https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla)
- **WandB Run**: [P0-M7-E-H2a-01_starflow_libero-4in1_qwen3vl4b_lwfm_ft0_260703_0849](https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla/runs/P0-M7-E-H2a-01_starflow_libero-4in1_qwen3vl4b_lwfm_ft0_260703_0849)
- **tmux 会话**: `train`

---

## 启动命令（用于 resume）

```bash
cd /disk/rl/starVLA
RUN_ID="P0-M7-E-H2a-01_starflow_libero-4in1_qwen3vl4b_lwfm_ft0_260703_0849" \
CONFIG_YAML=configs/starflow_vla/ablations/future_tokens_0.yaml \
DATA_MIX=libero_all \
MAX_TRAIN_STEPS=80000 \
SAVE_INTERVAL=250 \
EVAL_INTERVAL=500 \
LOGGING_FREQUENCY=20 \
NUM_PROCESSES=1 \
GRADIENT_ACCUMULATION_STEPS=32 \
PER_DEVICE_BATCH_SIZE=1 \
NUM_WORKERS=2 \
bash examples/LIBERO/train_files/run_starflow_train_ready.sh
```

---

## 备注

- 本实验验证 `num_target_vision_tokens=0` 的极端情况：action head 完全基于 state 做决策，无视觉 token 注入。
- 与 E-H2a-02 (ft=16), E-H2a-03 (ft=32), E-H2a-04 (ft=64) 形成 future_tokens 消融扫描。
- 训练从 scratch 开始，无预训练 checkpoint。
- VLM（qwen_vl_interface）冻结，仅训练 action model。
- 由于 RTX 4090 只有 24GB 显存，使用 per_device_batch_size=1 + grad_accum=32 确保有效 batch=32。

---

## 自动监控状态**: 🟢 训练运行中（step 944/80000，约 1.2%）  

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-07-03 10:48 CST |
| 训练状态**: 🟢 训练运行中（step 944/80000，约 1.2%）  
| run_id | `P0-M7-E-H2a-01_starflow_libero-4in1_qwen3vl4b_lwfm_ft0_260703_0849` |
| tmux 会话 | `train` |
| 当前步数 | **944 / 80000** |
| 完成比例 | 1.2% |
| 训练速度 | ~6.76 s/it |
| data_time | 0.000 s |
| model_time | 0.207 s |
| 已运行时间 | 1:44:59 |
| 预计剩余时间 | ~6d 4h |
| 最新完整 checkpoint | `steps_750` |
| GPU | NVIDIA GeForce RTX 4090 |
| GPU 利用率 | 62% |
| 显存使用 | 24032 MiB / 24564 MiB (97.8%) |
| 功耗 | 260.81 W / 450.00 W |
| 温度 | 67°C |
| 内存总量 | 56 GiB（Docker 容器限制） |
| 内存已用 | ~32 GiB |
| 内存可用 | ~24 GiB |
| 存储 `/disk/rl` | 549T / 700T (79% 已用) |
| 存储 `/localdisk-tmp` | 20G / 100G (20% 已用) |

---

*本文档由 Claude 自动监控生成并持续更新。*
