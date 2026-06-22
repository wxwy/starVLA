# P1-M1-E-H2b-02: StarFlow LIBERO 4-in-1 Qwen3VL-4B continuous_head ft=32

> **实验代号**: E-H2b-02 / P1-M1  
> **状态**: 🟢 训练运行中（step 36/80000，约 0.05%）  
> **启动时间**: 2026-06-21 19:01:16 CST  
> **最后训练日志**: 2026-06-21 19:06:30 CST，step 36  
> **当前更新**: 2026-06-21 19:06:30 CST  
> **tmux 会话**: `train-2`（运行本实验）  
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

> 最后更新：2026-06-22 15:32:00 CST

| 指标 | 值 |
|------|-----|
| **当前 Step** | 10861 / 80000 |
| **完成比例** | 14.0% |
| **单步耗时** | ~6.69 s/it |
| **数据加载耗时** | 0.001 s |
| **模型前向/反向耗时** | 1.652 s |
| **已运行时间** | 20小时 30分钟 |
| **预计剩余时间** | 5天 8小时 28分钟 |
| **预计总耗时** | 6天 4小时 59分钟 |

---

## 系统资源占用

> 最后更新：2026-06-22 15:32:01 CST

### 4.1 GPU（NVIDIA A100-SXM4-80GB）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 100% |
| **显存使用** | 55897 MiB / 81920 MiB (68.2%) |
| **显存空闲** | 25259 MiB |
| **功耗** | 317.33 W / 400.00 W |
| **温度** | 55°C |

### 4.2 CPU / 内存

| 指标 | 值 |
|------|-----|
| **CPU 使用率** | %Cpu(s):  8.1 us,  4.2 sy,  0.0 ni, 87.3 id,  0.0 wa,  0.0 hi,  0.4 si,  0.0 st |
| **内存总量** | 1007.5 GiB |
| **内存已用** | 126.9 GiB |
| **内存可用** | 880.6 GiB |

---

## 成本估算

> 最后更新：2026-06-22 15:32:01 CST

| 项目 | 计算 |
|------|------|
| **已产生成本** | 20小时 30分钟 × 5.58 元/h ≈ **114.40 元** |
| **预计总成本** | 6天 4小时 39分钟 × 5.58 元/h ≈ **829.56 元** |
| **剩余预计成本** | 5天 8小时 9分钟 × 5.58 元/h ≈ **715.16 元** |

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

## 备注

- 本实验验证 `state_mode=continuous_head` 与默认 `discretized_instruction` 的状态条件注入路径差异。
- `continuous_head` runtime 已由 codex 构建完成。
- 训练从 scratch 开始，无预训练 checkpoint。

---

*本文档将持续更新。*

## 自动监控状态

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-22 15:32:01 CST |
| 训练状态 | 🟢 运行中 |
| run_id | `P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260621_1901` |
| tmux 会话 | `train-2` |
| 已运行时间 | 20小时 30分钟 |
| 当前步数 | **10861 / 80000** |
| 完成比例 | 14.0% |
| 训练速度 | ~6.69 s/it |
| data_time | 0.001 s |
| model_time | 1.652 s |
| 最新完整 checkpoint | `steps_10750` |
| GPU | NVIDIA A100-SXM4-80GB |
| GPU 利用率 | 100% |
| 显存使用 | 55897 MiB / 81920 MiB (68.2%) |
| 功耗 | 317.33 W / 400.00 W |
| 温度 | 55°C |