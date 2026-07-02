# P1-M1-E-H2b-02: StarFlow LIBERO 4-in-1 Qwen3VL-4B continuous_head ft=32

> **实验代号**: E-H2b-02 / P1-M1  
> **状态**: 🟢 训练运行中（step 15392/80000，约 19.2%）  
> **启动时间**: 2026-06-21 19:01:16 CST  
> **最后训练日志**: 2026-07-02 12:17:19 CST，step 15392  
> **当前更新**: 2026-07-02 12:17:19 CST  
> **tmux 会话**: `train-0`（当前恢复后运行）  
> **配置来源**: `configs/starflow_vla/state/continuous_head.yaml`

---

## 实验概述

P1-M1 **state conditioning 对照**：在 StarFlowVLA 框架中启用 `state_mode=continuous_head`，将本体状态（8D state）通过连续向量直接注入 action head，与默认的 `discretized_instruction` 路径进行对比。

### 关键设计

- 保持与 H2-a 实验相同的数据、batch、训练预算
- `state_mode=continuous_head`（由 codex 构建完成）
- Action head 仍为 `LayerwiseFM`
- VLM backbone (`qwen_vl_interface`) 冻结

---

## 训练参数

### 脚本级参数

| 参数 | 值 |
|------|-----|
| `RUN_ID` | `P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260621_1901` |
| `CONFIG_YAML` | `configs/starflow_vla/state/continuous_head.yaml` |
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
| **框架** | StarFlowVLA |
| **State Mode** | `continuous_head` |
| **Base VLM** | Qwen3-VL-4B-Instruct |
| **Action Model** | LayerwiseFM (DiT, 36 layers, 1024 hidden) |
| **Action Dim** | 7 |
| **State Dim** | 8 |
| **Action Horizon** | 8 |
| **Future Action Window** | 7 |
| **Past Action Window** | 0 |
| **Num Target Vision Tokens** | 32 |
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

> 最后更新：2026-07-02 12:17:19 CST

| 指标 | 值 |
|------|-----|
| **当前 Step** | 15392 / 80000 |
| **完成比例** | 19.2% |
| **单步耗时** | ~7.55 s/it |
| **数据加载耗时** | 0.000 s |
| **模型前向/反向耗时** | 0.230 s |
| **已运行时间** | 约 0 小时 18 分钟（本次恢复后） |
| **预计剩余时间** | 约 135 小时 32 分钟 |
| **预计总耗时** | 约 135 小时 50 分钟（本次恢复后预估） |

---

## 系统资源占用

> 最后更新：2026-07-02 12:17:19 CST

### 4.1 GPU（NVIDIA GeForce RTX 4090）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 47% |
| **显存使用** | 22988 MiB / 24564 MiB (93.6%) |
| **显存空闲** | 1094 MiB |
| **功耗** | 288.88 W / 450.00 W |
| **温度** | 68°C |

### 4.2 CPU / 内存

| 指标 | 值 |
|------|-----|
| **CPU 使用率** | 待更新 |
| **内存总量** | 待更新 |
| **内存已用** | 待更新 |
| **内存可用** | 待更新 |

---

## 成本估算

> 最后更新：2026-07-02 12:17:19 CST

| 项目 | 计算 |
|------|------|
| **本次恢复后已产生成本** | 本地 4090，暂不记录 |
| **按当前速度完整训练预估成本** | 本地 4090，暂不记录 |
| **剩余预计成本** | 本地 4090，暂不记录 |

> 注：本次为从 `steps_15250` 恢复训练，上表成本仅统计当前 tmux 会话（恢复后）的耗时。历史会话已产生的累计成本未计入。 |

---

## 输出目录

```
/disk/rl/starVLA/playground/Checkpoints/P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260621_1901/
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
- **WandB Run**: [P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260621_1901](https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla/runs/P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260621_1901)
- **tmux 会话**: `train-2`

---

## 启动命令

```bash
cd /disk/rl/starVLA
RUN_ID="P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260621_1901" \
CONFIG_YAML=configs/starflow_vla/state/continuous_head.yaml \
DATA_MIX=libero_all \
MAX_TRAIN_STEPS=80000 \
SAVE_INTERVAL=250 \
EVAL_INTERVAL=500 \
LOGGING_FREQUENCY=20 \
NUM_PROCESSES=1 \
GRADIENT_ACCUMULATION_STEPS=4 \
PER_DEVICE_BATCH_SIZE=8 \
NUM_WORKERS=6 \
bash examples/LIBERO/train_files/run_starflow_train_ready.sh
```

---

## 恢复训练记录（Resume，2026-07-02）

因计算卡切换为 **NVIDIA GeForce RTX 4090**，在 tmux 会话 `train-0` 中重新 resume 训练。有效全局 batch 仍保持 **32** 不变，调整为 `per_device_batch_size=1` × `gradient_accumulation_steps=32`。

| 字段 | 值 |
| --- | --- |
| tmux 会话名 | `train-0` |
| tmux 创建时间 | 2026-07-02 11:57:44 CST |
| 主机 | `bitahub-a20206348704935936956761` |
| GPU | `NVIDIA GeForce RTX 4090`（24564 MiB） |
| 单价 | 本地 4090，暂不记录 |
| 训练状态 | 运行中（attached） |
| 恢复源 checkpoint | `steps_15250` |
| resume 时间 | 2026-07-02 11:59:16 CST |
| 当前配置 | `configs/starflow_vla/state/continuous_head.yaml` |
| `is_resume` | `True` |

### 变更后的训练参数

| 字段 | 原值（A100 80G） | 本次 resume（4090） |
| --- | --- | --- |
| per_device_batch_size | 8 | 1 |
| gradient_accumulation_steps | 4 | 32 |
| 有效全局 batch | 32 | 32 |
| save_interval | 250 | 250 |
| eval_interval | 500 | 500 |
| logging_frequency | 20 | 20 |
| num_workers | 6 | 2 |
| local_checkpoint_root | 未设置 | `/localdisk-tmp` |
| local_checkpoint_keep_count | 未设置 | 2 |

> 有效全局 batch 保持 32 不变（8×4 → 1×32）。

### Resume 启动命令

```bash
cd /disk/rl/starVLA
tmux attach -t train-0
# 在 tmux 会话内执行
IS_RESUME=True \
RUN_ID="P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260621_1901" \
CONFIG_YAML=configs/starflow_vla/state/continuous_head.yaml \
DATA_MIX=libero_all \
MAX_TRAIN_STEPS=80000 \
SAVE_INTERVAL=250 \
LOGGING_FREQUENCY=20 \
EVAL_INTERVAL=500 \
NUM_PROCESSES=1 \
GRADIENT_ACCUMULATION_STEPS=32 \
PER_DEVICE_BATCH_SIZE=1 \
NUM_WORKERS=2 \
STARVLA_PYTHON=/opt/conda/envs/starVLA/bin/python \
WANDB_MODE=online \
WANDB_PROJECT=starflow_vla \
WANDB_ENTITY=silencewx-harbin-institute-of-technology \
LOCAL_CHECKPOINT_ROOT=/localdisk-tmp \
LOCAL_CHECKPOINT_KEEP_COUNT=2 \
ENABLE_LOCAL_CHECKPOINT_STAGING=True \
bash examples/LIBERO/train_files/run_starflow_train_ready.sh
```

### Resume 关键事件

- 2026-07-02 11:57：创建 tmux 会话 `train-0`。
- 2026-07-02 11:59：从 `steps_15250` resume，设备切换为 `NVIDIA GeForce RTX 4090`，训练参数同步调整。
- 2026-07-02 11:59：成功写入 `config.yaml` / `config.full.yaml`，训练正常推进。
- 当前进度：step **15392 / 80000**（约 19.2%），速度约 **7.55 s/it**，已运行约 18 分钟。

---

## 备注

- 本实验验证 `state_mode=continuous_head` 与默认 `discretized_instruction` 的状态条件注入路径差异。
- `continuous_head` runtime 已由 codex 构建完成。
- 训练从 scratch 开始，无预训练 checkpoint。

---

*本文档将持续更新。*

## 自动监控状态

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-07-02 12:17:19 CST |
| 训练状态 | 🟢 运行中（来自 tmux `train-0`） |
| run_id | `P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260621_1901` |
| tmux 会话 | `train-0` |
| 当前步数 | **15392 / 80000** |
| 最新完整 checkpoint | `steps_15250` |
| GPU | NVIDIA GeForce RTX 4090 |
| GPU 利用率 | 47% |
| 显存使用 | 22988 MiB / 24564 MiB (93.6%) |
| 功耗 | 288.88 W / 450.00 W |
| 温度 | 68°C |
| 有效全局 batch | 32（1 × 32） |
| 每 step 成本 | 本地 4090，暂不记录 |
| 已产生成本（本次 resume） | 暂不记录 |
| 完整训练预估成本 | 暂不记录 |