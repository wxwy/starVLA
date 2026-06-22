# P0-M6-H2-total-baseline-fixed: StarFlow LIBERO 4-in-1 Qwen3VL-4B MLP Baseline（修正版）

> **实验代号**: H2-total-baseline-fixed / P0-M6-MLP-4in1  
> **状态**: 🟢 训练运行中（step 22337/80000，约 27.9%）  
> **启动时间**: 2026-06-22 11:48:42 CST  
> **最后训练日志**: 2026-06-22 18:36:44 CST，step 22320  
> **当前更新**: 2026-06-22 18:37:30 CST  
> **tmux 会话**: `train`（本实验正在运行中）  
> **配置来源**: `configs/starflow_vla/stage2_mlp_baseline.yaml`

---

## 实验概述

修正版 MLP baseline：修复了初版 P0-M6（`mlp_260621_1421`）中的问题，重新训练。StarFlowVLA framework + MLP action head 在 LIBERO 4-in-1 上的对照实验，用于与 `future_tokens + cross-DiT` 主路线（E-H2a 系列）进行受控对比。

### 与初版 P0-M6 的关键差异

| 项目 | 初版 (`mlp_260621_1421`) | 修正版 (`mlp_fixed_260622_1148`) |
|------|--------------------------|----------------------------------|
| **Trainable Params** | 633.272 M | 65.623 M |
| **Total Params** | 5,071.088 M | 4,503.439 M |
| **SAVE_INTERVAL** | 250 | 1000 |
| **状态** | 🔴 已停止（step 9750） | 🟢 运行中 |

### 关键设计

- 保持与 H2-a 实验相同的数据、batch、训练预算
- Action head 为 MLP（`MLPResNet`）
- VLM backbone (`qwen_vl_interface`) 冻结
- 训练从 scratch 开始

---

## 训练参数

### 脚本级参数

| 参数 | 值 |
|------|-----|
| `RUN_ID` | `P0-M6-H2-total-baseline_starflow_libero-4in1_qwen3vl4b_mlp_fixed_260622_1148` |
| `CONFIG_YAML` | `configs/starflow_vla/stage2_mlp_baseline.yaml` |
| `DATA_MIX` | `libero_all` |
| `MAX_TRAIN_STEPS` | 80,000 |
| `SAVE_INTERVAL` | 1000 |
| `EVAL_INTERVAL` | 500 |
| `LOGGING_FREQUENCY` | 20 |
| `NUM_PROCESSES` | 1 |
| `GRADIENT_ACCUMULATION_STEPS` | 4 |
| `PER_DEVICE_BATCH_SIZE` | 8 |
| `NUM_WORKERS` | 6 |
| `BASE_VLM` | `/disk/rl/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct` |
| `LIBERO_DATA_ROOT` | `/disk/rl/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA` |
| `WANDB_PROJECT` | `starflow_vla` |
| `WANDB_ENTITY` | `silencewx-harbin-institute-of-technology` |
| `is_resume` | `False` |

### 模型 / 优化器参数

| 参数 | 值 |
|------|-----|
| **框架** | StarFlowVLA（CLI 覆盖 `framework.name=StarFlowVLA`） |
| **Base VLM** | Qwen3-VL-4B-Instruct |
| **Action Model** | MLP (`MLPResNet`) |
| **Action Dim** | 7 |
| **State Dim** | 8 |
| **Action Horizon** | 8 |
| **Future Action Window** | 7 |
| **Past Action Window** | 0 |
| **Total Params** | 4,503.439 M |
| **Trainable Params** | 65.623 M |
| **Frozen Modules** | `qwen_vl_interface` |
| **Learning Rate (base)** | 2.5e-5 |
| **Learning Rate (qwen_vl_interface)** | 1.0e-5 |
| **Learning Rate (action_model)** | 1.0e-4 |
| **LR Scheduler** | cosine_with_min_lr |
| **Min LR** | 1.0e-6 |
| **Optimizer** | AdamW (β=(0.9, 0.95), eps=1e-8, wd=1e-8) |
| **Max Grad Norm** | 1.0 |
| **Gradient Checkpointing** | true |
| **Effective Global Batch** | 32 (8 × 4) |

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

> 最后更新：2026-06-22 14:37:30 CST

| 指标 | 值 |
|------|-----|
| **当前 Step** | 22337 / 80000 |
| **完成比例** | 27.9% |
| **单步耗时** | ~1.21 s/it |
| **数据加载耗时** | ~0.000 s |
| **模型前向/反向耗时** | ~0.31 s |
| **已运行时间** | 6 小时 48 分钟 |
| **预计剩余时间** | ~19 小时 23 分钟 |
| **预计总耗时** | ~26 小时 11 分钟 |

### Loss 记录

| Step | Loss | 备注 |
|------|------|------|
| 200 | 0.2701 | 初始 warmup |
| 400 | 0.3216 | |
| 1000 | — | ✅ checkpoint |
| 2000 | — | ✅ checkpoint |
| 3000 | — | ✅ checkpoint |
| 4000 | — | ✅ checkpoint |
| 5000 | — | ✅ checkpoint |
| 6000 | — | ✅ checkpoint |
| 7000 | — | ✅ checkpoint |
| 8000 | — | ✅ checkpoint |
| 9000 | — | ✅ 最新 checkpoint |
| 9080 | 0.2206 | |
| 9100 | 0.2252 | |
| 9120 | 0.2083 | |
| 9140 | 0.2111 | |
| 9160 | 0.1954 | |
| 9180 | 0.2623 | |
| 9200 | 0.1860 | |
| … | … | 中间步骤省略 |
| 15660 | 0.2063 | |
| 15680 | 0.2339 | |
| 15700 | 0.1706 | |
| … | … | 中间步骤省略 |
| 22300 | 0.1405 | |
| 22320 | 0.1808 | 🔵 当前最新 step |

---

## 系统资源占用

> 最后更新：2026-06-22 12:37:30 CST

### GPU（NVIDIA A100-SXM4-80GB）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 75% |
| **显存使用** | 12,573 MiB / 81,920 MiB (15.3%) |
| **显存空闲** | 68,583 MiB |
| **功耗** | 173.16 W / 400.00 W |
| **温度** | 48°C |

### CPU / 内存

| 指标 | 值 |
|------|-----|
| **CPU 核心数** | 13 |
| **内存总量** | 1007.5 GiB |
| **内存已用** | 205.3 GiB |
| **内存可用** | 802.2 GiB |

---

## 成本估算

> 单价：5.58 元/小时（A100-SXM4-80GB）

| 项目 | 计算 |
|------|------|
| **每 step 耗时** | ~1.21 秒 |
| **每 step 成本** | 5.58 / 3600 × 1.21 ≈ **0.0019 元** |
| **已产生成本** | 6.81 h × 5.58 ≈ **37.99 元** |
| **剩余 57663 steps 预估** | 57663 × 1.21s ≈ 19.4h ≈ **108.17 元** |
| **完整 80000 steps 预估** | 80000 × 1.21s ≈ 26.2h ≈ **146.17 元** |

---

## 输出目录

```
/disk/rl/starVLA/playground/Checkpoints/P0-M6-H2-total-baseline_starflow_libero-4in1_qwen3vl4b_mlp_fixed_260622_1148/
├── checkpoints/              # 模型 checkpoint
├── config.full.yaml          # 完整配置
├── config.yaml               # 访问过的配置快照
├── dataset_statistics.json   # 数据集统计信息
├── summary.jsonl             # 训练摘要
└── wandb/                    # WandB 本地日志
```

---

## 相关链接

- **WandB Project**: [starflow_vla](https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla)
- **WandB Run**: [P0-M6-H2-total-baseline_starflow_libero-4in1_qwen3vl4b_mlp_fixed_260622_1148](https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla/runs/P0-M6-H2-total-baseline_starflow_libero-4in1_qwen3vl4b_mlp_fixed_260622_1148)
- **tmux 会话**: `train`

---

## 启动命令（用于 resume）

```bash
cd /disk/rl/starVLA
RUN_ID="P0-M6-H2-total-baseline_starflow_libero-4in1_qwen3vl4b_mlp_fixed_$(date +%y%m%d_%H%M)" \
CONFIG_YAML=configs/starflow_vla/stage2_mlp_baseline.yaml \
DATA_MIX=libero_all \
MAX_TRAIN_STEPS=80000 \
SAVE_INTERVAL=1000 \
LOGGING_FREQUENCY=20 \
EVAL_INTERVAL=500 \
NUM_PROCESSES=1 \
GRADIENT_ACCUMULATION_STEPS=4 \
PER_DEVICE_BATCH_SIZE=8 \
NUM_WORKERS=6 \
bash examples/LIBERO/train_files/run_starflow_train_ready.sh
```

---

## 备注

- 本实验为 P0-M6 MLP baseline 的**修正版**（`_fixed`），修复了初版中的问题。
- 与初版相比，可训练参数量从 633M 降至 65.6M（约 90% 减少），总参数量从 5071M 降至 4503M。
- SAVE_INTERVAL 从 250 调整为 1000，减少 checkpoint I/O 开销。
- MLP 结构使单步训练速度约 1.12 s/it，显著快于 LayerwiseFM 结构的 ~7 s/it。
- 训练从 scratch 开始，无预训练 checkpoint。
- 初版 P0-M6（`mlp_260621_1421`）训练至 step 9750 后停止，详见 [P0-M6 初版 tracker](P0-M6-H2-total-baseline_starflow_libero-4in1_qwen3vl4b_mlp_260621_1421.md)。

---

*本文档将持续更新。*

## 自动监控状态

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-22 18:37:30 CST |
| 训练状态 | 🟢 运行中（来自 tmux `train`） |
| run_id | `P0-M6-H2-total-baseline_starflow_libero-4in1_qwen3vl4b_mlp_fixed_260622_1148` |
| tmux 会话 | `train` |
| 当前步数 | **22337 / 80000** |
| 完成比例 | 27.9% |
| 训练速度 | ~1.21 s/it |
| data_time | 0.000 s |
| model_time | 0.309 s |
| 已运行时间 | 06:48:29 |
| 预计剩余时间 | 19:23:00 |
| 最新完整 checkpoint | `steps_9000` |
| GPU | NVIDIA A100-SXM4-80GB |
| GPU 利用率 | 75% |
| 显存使用 | 12573 MiB / 81920 MiB (15.3%) |
| 功耗 | 173.16 W / 400.00 W |
| 温度 | 48°C |
| 每 step 成本 | ~0.0019 元 |
| 已产生成本 | ~37.99 元 |
| 完整训练预估成本 | ~146.17 元 |
