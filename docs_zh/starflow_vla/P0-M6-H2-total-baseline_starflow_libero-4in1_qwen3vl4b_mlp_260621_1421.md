# P0-M6-H2-total-baseline: StarFlow LIBERO 4-in-1 Qwen3VL-4B MLP Baseline

> **实验代号**: H2-total-baseline / P0-M6-MLP-4in1  
> **状态**: 🟢 训练运行中（step 9868/80000，约 12.3%）  
> **启动时间**: 2026-06-21 14:21:08 CST  
> **最后 checkpoint**: 2026-06-22 09:51:24 CST (`steps_9750`)  
> **最后训练日志**: 2026-06-22 10:04:31 CST，step 9860  
> **当前更新**: 2026-06-22 10:06:11 CST  
> **tmux 会话**: `train`（本实验正在运行中）  
> **配置来源**: `configs/starflow_vla/stage2_mlp_baseline.yaml`

---

## 实验概述

建立 H2 总 baseline：**StarFlowVLA framework + MLP action head** 在 LIBERO 4-in-1 上的对照。用于与 `future_tokens + cross-DiT` 主路线（E-H2a-01/02/03/04）进行受控对比。

### 关键设计

- 保持与 H2-a 实验相同的数据、batch、训练预算
- 仅将 action head 从 `LayerwiseFM` 替换为 `MLP`
- VLM backbone (`qwen_vl_interface`) 冻结

---

## 训练参数

### 脚本级参数

| 参数 | 值 |
|------|-----|
| `RUN_ID` | `P0-M6-H2-total-baseline_starflow_libero-4in1_qwen3vl4b_mlp_260621_1421` |
| `CONFIG_YAML` | `configs/starflow_vla/stage2_mlp_baseline.yaml` |
| `DATA_MIX` | `libero_all` |
| `MAX_TRAIN_STEPS` | 80,000 |
| `SAVE_INTERVAL` | 250 |
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
| **Action Model** | MLP (`MLPResNet`, hidden_dim = action_hidden_dim × 2 = 4096) |
| **Action Dim** | 7 |
| **State Dim** | 8 |
| **Action Horizon** | 8 |
| **Future Action Window** | 7 |
| **Past Action Window** | 0 |
| **Action Hidden Dim** | 2048 |
| **Total Params** | 5,071.088 M |
| **Trainable Params** | 633.272 M |
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

> 最后更新：2026-06-22 10:06:11 CST

| 指标 | 值 |
|------|-----|
| **当前 Step** | 9868 / 80000 |
| **完成比例** | 12.3% |
| **单步耗时** | ~7.11 s/it |
| **数据加载耗时** | ~0.001 s |
| **模型前向/反向耗时** | ~1.78 s |
| **已运行时间** | 19 小时 44 分钟 |
| **预计剩余时间** | 5 天 18 小时 31 分钟 |
| **预计总耗时** | 6 天 14 小时 15 分钟 |

### Loss / MSE 记录

| Step | Loss | MSE (eval) | 备注 |
|------|------|------------|------|
| 20 | 1.0702 | — | 初始warmup |
| 40 | 1.7984 | — | 初始warmup |
| 60 | 0.5937 | — | 初始warmup |
| 80 | 0.7716 | — | 初始warmup |
| 100 | 0.6184 | — | |
| 120 | 0.5249 | — | |
| 140 | 0.4222 | — | |
| 160 | 0.3884 | — | |
| 180 | 0.5015 | — | |
| 200 | 0.4984 | — | |
| 220 | 0.4152 | — | |
| 240 | 0.4548 | — | |
| 260 | 0.5204 | — | |
| 280 | 0.4433 | — | |
| 300 | 0.4705 | — | |
| 320 | 0.4133 | — | |
| 340 | 0.3672 | — | |
| 360 | 0.3454 | — | |
| 380 | 0.3639 | — | |
| 400 | 0.3618 | — | |
| 420 | 0.3418 | — | |
| 440 | 0.3543 | — | |
| 460 | 0.3307 | — | |
| 480 | 0.3150 | — | |
| 500 | 0.3018 | 0.0160 | ✅ 最后完整 checkpoint / eval |
| … | … | … | 中间步骤省略 |
| 9500 | 0.1410 | 0.0126 | ✅ checkpoint（含 eval） |
| 9750 | 0.2046 | — | ✅ 最新完整 checkpoint |
| 9860 | 0.1407 | — | 🔵 当前最新 step |

---

## 系统资源占用

> 最后更新：2026-06-22 10:06:11 CST

### GPU（NVIDIA A100-SXM4-80GB）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 85% |
| **显存使用** | 60,067 MiB / 81,920 MiB (73.3%) |
| **显存空闲** | 21,089 MiB |
| **功耗** | 366.91 W / 400.00 W |
| **温度** | 56°C |

### CPU / 内存

| 指标 | 值 |
|------|-----|
| **CPU 核心数** | 13 |
| **内存总量** | 1007.5 GiB |
| **内存已用** | 90.3 GiB |
| **内存可用** | 917.2 GiB |

### 存储

| 挂载点 | 总量 | 已用 | 可用 | 使用率 |
|--------|------|------|------|--------|
| `/disk/rl` | 700 T | 561 T | 140 T | 81% |
| `/localdisk-tmp` | 3.5 T | 1.3 T | 2.2 T | 38% |

---

## 成本估算

> 单价：5.58 元/小时（A100-SXM4-80GB）

| 项目 | 计算 |
|------|------|
| **每 step 耗时** | ~7.11 秒 |
| **每 step 成本** | 5.58 / 3600 × 7.11 ≈ **0.0110 元** |
| **已产生成本** | 19.74 h × 5.58 ≈ **110.15 元** |
| **剩余 70132 steps 预估** | 70132 × 7.11s ≈ 138.5h ≈ **772.83 元** |
| **完整 80000 steps 预估** | 80000 × 7.11s ≈ 158.0h ≈ **881.64 元** |

---

## 输出目录

```
/disk/rl/starVLA/playground/Checkpoints/P0-M6-H2-total-baseline_starflow_libero-4in1_qwen3vl4b_mlp_260621_1421/
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
- **WandB Run**: [P0-M6-H2-total-baseline_starflow_libero-4in1_qwen3vl4b_mlp_260621_1421](https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla/runs/P0-M6-H2-total-baseline_starflow_libero-4in1_qwen3vl4b_mlp_260621_1421)
- **tmux 会话**: `train`

---

## 启动命令（用于 resume）

```bash
cd /disk/rl/starVLA
RUN_ID="P0-M6-H2-total-baseline_starflow_libero-4in1_qwen3vl4b_mlp_260621_1421" CONFIG_YAML=configs/starflow_vla/stage2_mlp_baseline.yaml DATA_MIX=libero_all MAX_TRAIN_STEPS=80000 SAVE_INTERVAL=250 LOGGING_FREQUENCY=20 EVAL_INTERVAL=500 NUM_PROCESSES=1 GRADIENT_ACCUMULATION_STEPS=4 PER_DEVICE_BATCH_SIZE=8 NUM_WORKERS=6 bash examples/LIBERO/train_files/run_starflow_train_ready.sh
```

---

## 备注

- 本实验作为 H2 总 baseline，与 `future_tokens + cross-DiT` 主路线对照。
- CLI 启动时覆盖了 `framework.name=StarFlowVLA`，action head 仍为 MLP。
- 训练从 scratch 开始，无预训练 checkpoint。
- **当前状态**: 🟢 本实验正在 tmux 会话 `train` 中运行，当前至 step ~9868/80000（12.3%）。最新完整 checkpoint 为 `steps_9750`。
- 本实验此前曾一度停止在 step 260 / `steps_250`，之后被 resume 并继续训练至 step 500 / `steps_500`，再次停止，于 2026-06-21 重新从 `steps_500` resume 继续训练。
- 系统资源与成本数据反映当前机器状态（由 Claude 监控任务自动采集）。

---

*本文档将持续更新。*

## 自动监控状态

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-22 10:06:11 CST |
| 训练状态 | 🟢 运行中（来自 tmux `train`） |
| run_id | `P0-M6-H2-total-baseline_starflow_libero-4in1_qwen3vl4b_mlp_260621_1421` |
| 当前步数 | **9868 / 80000** |
| 完成比例 | 12.3% |
| 训练速度 | ~7.11 s/it |
| data_time | 0.001 s |
| model_time | 1.782 s |
| 已运行时间 | 19:44:31 |
| 预计剩余时间 | 138:30:42 |
| 最新完整 checkpoint | `steps_9750` |
| GPU | NVIDIA A100-SXM4-80GB |
| GPU 利用率 | 85% |
| 显存使用 | 60067 MiB / 81920 MiB (73.3%) |
| 功耗 | 366.91 W / 400.00 W |
| 温度 | 56°C |
| 内存总量 | 1007.5 GiB |
| 内存已用 | 90.3 GiB |
| 内存可用 | 917.2 GiB |
| 每 step 成本 | ~0.0110 元 |
| 已产生成本 | ~110.15 元 |
| 完整训练预估成本 | ~881.64 元 |