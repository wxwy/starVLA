# P0-M6-H2-total-baseline-fixed: StarFlow LIBERO 4-in-1 Qwen3VL-4B MLP Baseline（修正版）

> **实验代号**: H2-total-baseline-fixed / P0-M6-MLP-4in1  
> **状态**: 🟢 训练运行中（step 39521/80000，约 49.4%）  
> **启动时间**: 2026-06-22 11:48:42 CST  
> **最后训练日志**: 2026-07-02 12:17:19 CST，step 39520  
> **当前更新**: 2026-07-02 12:18:00 CST  
> **tmux 会话**: `train`（本实验正在运行中）  
> **配置来源**: `configs/starflow_vla/stage2_mlp_baseline.yaml`  
> **当前设备**: NVIDIA GeForce RTX 4090（2026-07-02 自 A100 80G 切换，resume 继续训练）

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

> 最后更新：2026-07-02 12:18:00 CST

| 指标 | 值 |
|------|-----|
| **当前 Step** | 39521 / 80000 |
| **完成比例** | 49.4% |
| **单步耗时** | ~3.50 s/it（4090，per_device_batch_size=1，gradient_accumulation_steps=32） |
| **数据加载耗时** | ~0.000 s |
| **模型前向/反向耗时** | ~0.10 s |
| **自 2026-06-22 启动以来累计运行** | 约 10 小时 48 分钟（A100 阶段）+ 本次 4090 resume 持续中 |
| **4090 resume 后已运行** | 约 15 分钟 |
| **4090 resume 后预计剩余时间** | 约 39 小时 |
| **最新完整 checkpoint** | `steps_39500` |

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
| 9000 | — | ✅ checkpoint |
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
| … | … | 中间步骤省略 |
| 28600 | 0.1445 | |
| … | … | 中间步骤省略 |
| 34880 | 0.1760 | |
| 34900 | 0.1336 | |
| 34920 | 0.2008 | A100 阶段最后记录 step |
| … | … | 中间步骤省略 |
| 39260 | 0.1948 | 4090 resume 后 |
| 39300 | 0.2222 | |
| 39400 | 0.1884 | |
| 39500 | 0.2908 | ✅ checkpoint |
| 39520 | 0.0482 | 🔵 当前最新 step |

---

## 系统资源占用

> 最后更新：2026-07-02 12:18:00 CST

### GPU（NVIDIA GeForce RTX 4090）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 26% |
| **显存使用** | 10,330 MiB / 24,564 MiB (42.1%) |
| **显存空闲** | 13,752 MiB |
| **功耗** | 136.73 W / 450.00 W |
| **温度** | 46°C |

### CPU / 内存

| 指标 | 值 |
|------|-----|
| **CPU 核心数** | 待更新 |
| **内存总量** | 503 GiB |
| **内存已用** | 50 GiB |
| **内存可用** | 435 GiB |

### 历史资源占用（A100 80G 阶段，2026-06-22）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 30% |
| **显存使用** | 12,573 MiB / 81,920 MiB (15.3%) |
| **显存空闲** | 68,583 MiB |
| **功耗** | 447.89 W / 400.00 W |
| **温度** | 43°C |
| **内存总量** | 1007.5 GiB |
| **内存已用** | ~224 GiB |
| **内存可用** | ~783 GiB |

---

## 成本估算

### A100 阶段（2026-06-22，已结束）

> 单价：5.58 元/小时（A100-SXM4-80GB）

| 项目 | 计算 |
|------|------|
| **每 step 耗时** | ~1.11 秒 |
| **每 step 成本** | 5.58 / 3600 × 1.11 ≈ **0.0017 元** |
| **A100 阶段已产生成本** | 10.81 h × 5.58 ≈ **60.32 元** |
| **A100 阶段完成 step** | ~34920 |

### RTX 4090 阶段（2026-07-02 resume 起，本地卡，暂不记录费用）

| 项目 | 计算 |
|------|------|
| **每 step 耗时** | ~3.50 秒 |
| **每 step 成本** | 本地 4090，暂不记录 |
| **已产生成本** | 暂不记录 |
| **剩余 ~40479 steps 预估** | 40479 × 3.50s ≈ 39.3h |
| **完整 80000 steps 预估（4090 速度）** | 80000 × 3.50s ≈ 77.8h |

> 注：4090 为本地设备，费用按 0 计算；上表仅按当前速度估算时间。

---

## 切换 RTX 4090 继续训练（2026-07-02）

因计算卡切换为 **NVIDIA GeForce RTX 4090**，在 tmux 会话 `train` 中自 `steps_39250` resume 继续训练。有效全局 batch 保持 **32** 不变，调整为 `per_device_batch_size=1` × `gradient_accumulation_steps=32`。

### 关键信息

| 字段 | 值 |
| --- | --- |
| tmux 会话名 | `train` |
| tmux 创建时间 | 2026-07-02 11:59:50 CST |
| GPU | `NVIDIA GeForce RTX 4090`（24564 MiB） |
| 单价 | 本地 4090，暂不记录 |
| 训练状态 | 运行中（attached） |
| 恢复源 checkpoint | `steps_39250` |
| resume 时间 | 2026-07-02 12:01:49 CST |
| 当前配置 | `configs/starflow_vla/stage2_mlp_baseline.yaml` |
| `is_resume` | `True` |

### 参数变更

| 字段 | A100 80G 阶段 | RTX 4090 resume 阶段 |
| --- | --- | --- |
| `per_device_batch_size` | 8 | 1 |
| `gradient_accumulation_steps` | 4 | 32 |
| **有效全局 batch** | **32** | **32** |
| `save_interval` | 1000 | 250 |
| `eval_interval` | 500 | 500 |
| `logging_frequency` | 20 | 20 |
| `num_workers` | 6 | 2 |
| `local_checkpoint_root` | `/localdisk-tmp` | `/localdisk-tmp` |
| `local_checkpoint_keep_count` | 未设置 | 2 |

> 有效全局 batch 保持 32 不变（8×4 → 1×32）。`save_interval` 从 1000 下调为 250，便于本地训练期间更频繁保存。

### Resume 启动命令

```bash
cd /disk/rl/starVLA
tmux attach -t train
# 在 tmux 会话内执行
RUN_ID="P0-M6-H2-total-baseline_starflow_libero-4in1_qwen3vl4b_mlp_fixed_260622_1148" \
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
bash examples/LIBERO/train_files/run_starflow_train_ready.sh
```

### Resume 关键事件

- 2026-07-02 11:59：创建 tmux 会话 `train`。
- 2026-07-02 12:01：自 `steps_39250` resume，设备切换为 `NVIDIA GeForce RTX 4090`，训练参数同步调整。
- 2026-07-02 12:02：成功写入 `config.yaml` / `config.full.yaml`，训练正常推进。
- 2026-07-02 12:15：成功保存 checkpoint `steps_39500` 并启动后台同步。
- 当前进度：step **39521 / 80000**（约 49.4%），速度约 **3.50 s/it**。

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
- 2026-07-02 设备由 A100-SXM4-80GB 切换为本地 **NVIDIA GeForce RTX 4090**，自 `steps_39250` resume 继续训练；`per_device_batch_size` 从 8 降至 1，`gradient_accumulation_steps` 从 4 升至 32，**有效全局 batch 保持 32 不变**。
- 4090 阶段单步速度约 3.50 s/it，较 A100 阶段（~1.11 s/it）下降，主要因为单卡 batch 减小、gradient accumulation 步数增加。
- 初版 P0-M6（`mlp_260621_1421`）训练至 step 9750 后停止，详见 [P0-M6 初版 tracker](P0-M6-H2-total-baseline_starflow_libero-4in1_qwen3vl4b_mlp_260621_1421.md)。

---

*本文档将持续更新。*

## 自动监控状态

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-07-02 12:18:00 CST |
| 训练状态 | 🟢 运行中（来自 tmux `train`） |
| run_id | `P0-M6-H2-total-baseline_starflow_libero-4in1_qwen3vl4b_mlp_fixed_260622_1148` |
| tmux 会话 | `train` |
| 当前步数 | **39521 / 80000** |
| 完成比例 | 49.4% |
| 训练速度 | ~3.50 s/it |
| data_time | 0.000 s |
| model_time | 0.108 s |
| 4090 resume 后已运行 | ~15 分钟 |
| 预计剩余时间 | ~39 小时 |
| 最新完整 checkpoint | `steps_39500` |
| GPU | NVIDIA GeForce RTX 4090 |
| GPU 利用率 | 26% |
| 显存使用 | 10330 MiB / 24564 MiB (42.1%) |
| 功耗 | 136.73 W / 450.00 W |
| 温度 | 46°C |
| 有效全局 batch | 32（1 × 32） |
| 每 step 成本 | 本地 4090，暂不记录 |
| 已产生成本（A100 阶段） | ~60.32 元 |
| 已产生成本（4090 阶段） | 暂不记录 |
| 完整训练预估成本 | 暂不记录 |
