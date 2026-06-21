# 训练跟踪：P0-M7-E-H2a-04_starflow_libero-4in1_qwen3vl4b_lwfm_ft64_260620_1652

> 本文档由 Claude Code 自动生成并维护，记录 `tmux train` 会话中的训练状态、系统资源占用与成本估算。

---

## 1. 基本信息

| 项目 | 内容 |
|------|------|
| **Run ID** | `P0-M7-E-H2a-04_starflow_libero-4in1_qwen3vl4b_lwfm_ft64_260620_1652` |
| **会话** | `tmux train` |
| **配置** | `configs/starflow_vla/ablations/future_tokens_64.yaml` |
| **启动时间** | 2026-06-20 16:52:58 CST |
| **记录时间** | 2026-06-20 16:55:57 CST |
| **已运行** | ~3 分钟 |
| **机器** | A100-SXM4-80GB × 1 |
| **单价** | 5.58 元/小时 |
| **WANDB 项目** | [starflow_vla](https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla) |
| **本次运行** | [Run 链接](https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla/runs/P0-M7-E-H2a-04_starflow_libero-4in1_qwen3vl4b_lwfm_ft64_260620_1652) |

---

## 2. 训练参数

### 2.1 脚本级参数

| 参数 | 值 |
|------|-----|
| `RUN_ID` | `P0-M7-E-H2a-04_starflow_libero-4in1_qwen3vl4b_lwfm_ft64_260620_1652` |
| `CONFIG_YAML` | `configs/starflow_vla/ablations/future_tokens_64.yaml` |
| `DATA_MIX` | `libero_all` |
| `MAX_TRAIN_STEPS` | 80,000 |
| `SAVE_INTERVAL` | 2,500 |
| `EVAL_INTERVAL` | 5,000 |
| `LOGGING_FREQUENCY` | 20 |
| `NUM_PROCESSES` | 1 |
| `GRADIENT_ACCUMULATION_STEPS` | 4 |
| `PER_DEVICE_BATCH_SIZE` | 8 |
| `NUM_WORKERS` | 4 |
| `BASE_VLM` | `/disk/rl/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct` |
| `LIBERO_DATA_ROOT` | `/disk/rl/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA` |
| `WANDB_PROJECT` | `starflow_vla` |
| `WANDB_ENTITY` | `silencewx-harbin-institute-of-technology` |

### 2.2 模型 / 优化器参数

| 参数 | 值 |
|------|-----|
| **框架** | StarFlowVLA |
| **Base VLM** | Qwen3-VL-4B-Instruct |
| **Action Model** | LayerwiseFM |
| **Action Dim** | 7 |
| **State Dim** | 8 |
| **Action Horizon** | 8 |
| **Future Action Window** | 7 |
| **Past Action Window** | 0 |
| **Num Target Vision Tokens** | 64 |
| **DiT Hidden Dim** | 1024 |
| **DiT Layers** | 36 |
| **DiT Attention Heads** | 16 |
| **Attention Head Dim** | 64 |
| **Dropout** | 0.2 |
| **Total Params** | 5,071.121 M |
| **Trainable Params** | 633.305 M |
| **Frozen Modules** | `qwen_vl_interface` |
| **Learning Rate (base)** | 2.5e-5 |
| **Learning Rate (qwen_vl_interface)** | 1.0e-5 |
| **Learning Rate (action_model)** | 1.0e-4 |
| **LR Scheduler** | cosine_with_min_lr |
| **Min LR** | 1.0e-6 |
| **Optimizer** | AdamW (β=(0.9, 0.95), eps=1e-8, wd=1e-8) |
| **Max Grad Norm** | 1.0 |
| **Gradient Checkpointing** | true |
| **Mixed Precision** | no |

### 2.3 数据集

使用 `libero_all` 混合数据集：

| 数据集 | 样本数 |  embodiment |
|--------|--------|-------------|
| `libero_object_no_noops_1.0.0_lerobot` | 66,984 | FRANKA |
| `libero_goal_no_noops_1.0.0_lerobot` | 52,042 | FRANKA |
| `libero_spatial_no_noops_1.0.0_lerobot` | 52,970 | FRANKA |
| `libero_10_no_noops_1.0.0_lerobot` | 101,469 | FRANKA |
| **合计** | **273,465** | — |

---

## 3. 训练进度

> 最后更新：2026-06-21 17:21:12 CST

| 指标 | 值 |
|------|-----|
| **当前 Step** | 13212 / 80000 |
| **完成比例** | 17.0% |
| **单步耗时** | ~7.68 s/it |
| **数据加载耗时** | 0.0 s |
| **模型前向/反向耗时** | 1.789 s |
| **已运行时间** | 4小时 3分钟 |
| **预计剩余时间** | 5天 22小时 28分钟 |
| **预计总耗时** | 6天 2小时 32分钟 |

---

## 4. 系统资源占用

> 最后更新：2026-06-21 17:21:13 CST

### 4.1 GPU（NVIDIA A100-SXM4-80GB）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 100% |
| **显存使用** | 63473 MiB / 81920 MiB (77.5%) |
| **显存空闲** | 17683 MiB |
| **功耗** | 338.45 W / 400.00 W |
| **温度** | 59°C |

### 4.2 CPU / 内存

| 指标 | 值 |
|------|-----|
| **CPU 使用率** | %Cpu(s):  5.9 us,  1.3 sy,  0.0 ni, 92.8 id,  0.0 wa,  0.0 hi,  0.0 si,  0.0 st |
| **内存总量** | 1007.5 GiB |
| **内存已用** | 92.3 GiB |
| **内存可用** | 915.2 GiB |

---

## 5. 成本估算

> 最后更新：2026-06-21 17:21:13 CST

| 项目 | 计算 |
|------|------|
| **已产生成本** | 4小时 3分钟 × 5.58 元/h ≈ **22.65 元** |
| **预计总成本** | 7天 2小时 39分钟 × 5.58 元/h ≈ **952.32 元** |
| **剩余预计成本** | 6天 22小时 36分钟 × 5.58 元/h ≈ **929.67 元** |

---

## 6. 输出目录结构

```
/disk/rl/starVLA/playground/Checkpoints/P0-M7-E-H2a-04_starflow_libero-4in1_qwen3vl4b_lwfm_ft64_260620_1652/
├── checkpoints/              # 模型 checkpoint（按 save_interval 生成）
├── config.full.yaml          # 完整配置
├── config.yaml               # 访问过的配置快照
├── dataset_statistics.json   # 数据集统计信息
└── wandb/                    # WandB 本地日志
```

---

## 7. 备注

- 启动前修复了两个环境问题：
  1. 脚本默认调用不存在的 `.venv/bin/python`，通过 `STARVLA_PYTHON=/opt/conda/envs/starVLA/bin/python` 指定 conda 环境解决。
  2. `pandas` 安装损坏且 `numpy` 2.x 与 `pyarrow` 不兼容，通过重装 `pandas` 并降级 `numpy<2`（最终 1.26.4）解决。
- 训练从 scratch 开始，无预训练 checkpoint。

---

*本文档将持续更新。*

## 自动监控状态

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-21 17:32:55 CST |
| 训练状态 | 🟢 运行中（来自 tmux `train`） |
| run_id | `P0-M7-E-H2a-04_starflow_libero-4in1_qwen3vl4b_lwfm_ft64_260620_1652` |
| 当前步数 | **8627 / 80000** |
| 完成比例 | 11.0% |
| 训练速度 | ~9.51 s/it |
| data_time | 0.001 s |
| model_time | 2.369 s |
| 已运行时间 | 4:19:52 |
| 预计剩余时间 | 188:29:05 |
| 最新完整 checkpoint | `steps_8500` |
| GPU | NVIDIA A100-SXM4-80GB |
| GPU 利用率 | 100% |
| 显存使用 | 65859 MiB / 81920 MiB (80.4%) |
| 功耗 | 313.54 W / 400.00 W |
| 温度 | 55°C |
| 内存总量 | 1.0Ti |
| 内存已用 | 81Gi |
| 内存空闲 | 165Gi |
| 存储 `/disk/rl` | 561T / 700T (81% 已用) |
| 存储 `/localdisk-tmp` | 16G / 100G (16% 已用) |
| 每 step 成本 | ~0.0147 元 |
| 已产生成本 | ~24.17 元 |
| 完整训练预估成本 | ~1179.24 元 |