# P0-M7-E-H2a-01: StarFlow LIBERO 4-in-1 Qwen3VL-4B future_tokens=0（bs32）

> **实验代号**: E-H2a-01 / P0-M7  
> **状态**: 🟢 训练运行中（step 55155/80000，约 68.9%）  
> **run_id**: `P0-M7-E-H2a-01_starflow_libero-4in1_qwen3vl4b_lwfm_ft0_260703_0849`  
> **启动时间**: 2026-07-03 08:53 CST  
> **当前更新**: 2026-07-07 18:26 CST  
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

> 最后更新：2026-07-07 18:26 CST（Docker 容器内实际可用资源）

| 指标 | 值 |
|------|-----|
| **当前 Step** | 55155 / 80,000（68.9%） |
| **完成比例** | 68.9% |
| **单步耗时** | ~6.75 s/it |
| **数据加载耗时** | ~0.000 s |
| **模型前向/反向耗时** | ~0.214 s |
| **已运行时间** | 103 小时 42 分钟 |
| **预计剩余时间** | ~46 小时 30 分钟 |
| **预计总耗时** | ~150 小时 12 分钟 |

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
| 940 | — | |
| 15141 | 0.0867 | |
| 15663 | 0.0734 | |
| 16191 | 0.0881 | |
| 16727 | 0.0752 | |
| 17249 | 0.0739 | |
| 17290 | 0.0840 | |
| 17320 | 0.0843 | |
| 17340 | 0.0946 | |
| 17360 | 0.0721 | |
| 17380 | 0.0792 | |
| 17400 | 0.1076 | |
| 17420 | 0.0707 | |
| 17680 | 0.0775 | |
| 17700 | 0.0814 | |
| 17720 | 0.0714 | |
| 17740 | 0.0929 | |
| 17760 | 0.0746 | |
| 17780 | 0.0740 | |
| 17800 | 0.0666 | |
| 17820 | 0.0609 | |
| 17840 | 0.0822 | |
| 17860 | 0.0722 | |
| 18220 | 0.0839 | |
| 18240 | 0.0724 | |
| 18260 | 0.0685 | |
| 18280 | 0.0755 | |
| 18300 | 0.0620 | |
| 18320 | 0.0751 | |
| 18340 | 0.1020 | |
| 18360 | 0.0666 | |
| 18380 | 0.0899 | |
| 18400 | 0.0691 | |
| 18740 | 0.0710 | |
| 18760 | 0.0715 | |
| 18780 | 0.0760 | |
| 18800 | 0.0674 | |
| 18820 | 0.0658 | |
| 18840 | 0.0811 | |
| 18860 | 0.0789 | |
| 18880 | 0.0523 | |
| 18900 | 0.0560 | |
| 18920 | 0.0851 | |
| 19260 | 0.0617 | |
| 19280 | 0.0653 | |
| 19300 | 0.0647 | |
| 19320 | 0.0790 | |
| 19340 | 0.0654 | |
| 19360 | 0.0721 | |
| 19380 | 0.0845 | |
| 19400 | 0.0689 | |
| 19420 | 0.0754 | |
| 19440 | 0.0792 | |
| 19800 | 0.0840 | |
| 19820 | 0.0877 | |
| 19840 | 0.0609 | |
| 19860 | 0.0698 | |
| 19880 | 0.0730 | |
| 19900 | 0.0629 | |
| 19920 | 0.0796 | |
| 19940 | 0.0763 | |
| 19960 | 0.0838 | |
| 19980 | 0.0817 | |
| 20320 | 0.0742 | |
| 20340 | 0.0673 | |
| 20360 | 0.0774 | |
| 20380 | 0.0719 | |
| 20400 | 0.0788 | |
| 20420 | 0.0774 | |
| 20440 | 0.0820 | |
| 20460 | 0.0896 | |
| 20480 | 0.0838 | |
| 20500 | 0.0803 | |
| 20860 | 0.0714 | |
| 20880 | 0.0742 | |
| 20900 | 0.0732 | |
| 20920 | 0.0690 | |
| 20940 | 0.0686 | |
| 20960 | 0.0761 | |
| 20980 | 0.0658 | |
| 21000 | 0.0714 | |
| 21020 | 0.0646 | |
| 21380 | 0.0778 | |
| 21400 | 0.0566 | |
| 21420 | 0.0728 | |
| 21440 | 0.0724 | |
| 21460 | 0.0714 | |
| 21480 | 0.0592 | |
| 21500 | 0.0922 | |
| 21520 | 0.0695 | |
| 21540 | 0.0726 | |
| 21560 | 0.0794 | |
| 21900 | 0.0654 | |
| 21920 | 0.0685 | |
| 21940 | 0.0595 | |
| 21960 | 0.0512 | |
| 21980 | 0.0782 | |
| 22000 | 0.0536 | |
| 22020 | 0.0909 | |
| 22040 | 0.0731 | |
| 22060 | 0.0821 | |
| 22080 | 0.0812 | |
| 22440 | 0.0668 | |
| 22460 | 0.0675 | |
| 22480 | 0.0724 | |
| 22500 | 0.0665 | |
| 22520 | 0.0714 | |
| 22540 | 0.0632 | |
| 22560 | 0.0688 | |
| 22580 | 0.0702 | |
| 22600 | 0.0784 | |
| 22620 | 0.0652 | 🔵 当前最新（step 22620） |

---

## 系统资源占用

> 最后更新：2026-07-07 18:26 CST（Docker 容器内实际可用资源）  
> 注意：以下为 Docker 容器内实际可用资源

### GPU（NVIDIA GeForce RTX 4090）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 49% |
| **显存使用** | 24032 MiB / 24564 MiB（97.8%） |
| **显存空闲** | 50 MiB |
| **功耗** | 249.10 W / 450.00 W |
| **温度** | 63°C |

### CPU / 内存（Docker 容器）

| 指标 | 值 |
|------|-----|
| **内存总量** | 56 GiB（Docker 容器限制） |
| **内存已用** | ~30.8 GiB |
| **内存可用** | ~25.2 GiB |

### 存储

| 挂载点 | 总量 | 已用 | 可用 | 使用率 |
|--------|------|------|------|--------|
| `/disk/rl` | 700 T | 550 T | 151 T | 79% |
| `/localdisk-tmp` | 100 G | 16 G | 85 G | 16% |

---

## 输出目录

```
/disk/rl/starVLA/playground/Checkpoints/P0-M7-E-H2a-01_starflow_libero-4in1_qwen3vl4b_lwfm_ft0_260703_0849/
├── checkpoints/              # 模型 checkpoint
│   ├── steps_250/            # 第一个 checkpoint
│   ├── steps_22500/          # 历史 checkpoint
│   └── steps_55000/          # ✅ 最新完整 checkpoint
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

## 自动监控状态**: 🟢 训练运行中（step 55155/80000，约 68.9%）  

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-07-07 18:26 CST |
| 训练状态**: 🟢 训练运行中（step 55155/80000，约 68.9%）  
| run_id | `P0-M7-E-H2a-01_starflow_libero-4in1_qwen3vl4b_lwfm_ft0_260703_0849` |
| tmux 会话 | `train` |
| 当前步数 | **55155 / 80000** |
| 完成比例 | 68.9% |
| 训练速度 | ~6.75 s/it |
| data_time | 0.000 s |
| model_time | 0.214 s |
| 已运行时间 | 103:42:56 |
| 预计剩余时间 | ~46h 30m |
| 最新完整 checkpoint | `steps_55000` |
| GPU | NVIDIA GeForce RTX 4090 |
| GPU 利用率 | 49% |
| 显存使用 | 24032 MiB / 24564 MiB (97.8%) |
| 功耗 | 249.10 W / 450.00 W |
| 温度 | 63°C |
| 内存总量 | 56 GiB（Docker 容器限制） |
| 内存已用 | ~30.8 GiB |
| 内存可用 | ~25.2 GiB |
| 存储 `/disk/rl` | 550T / 700T (79% 已用) |
| 存储 `/localdisk-tmp` | 16G / 100G (16% 已用) |

---

*本文档由 Claude 自动监控生成并持续更新。*

---

## LIBERO Goal 评估结果

> 评估时间：2026-07-07 15:46 CST
> Task Suite: **libero_goal** (10 tasks × 50 trials)
> 评估设备: NVIDIA RTX 4090

### steps_50000

| 指标 | 值 |
|------|-----|
| **Success Rate** | **80.94%** |
| 成功/总数 | 395 / 488 |

| 任务 | 成功率 | 成功/总数 |
|------|--------|-----------|
| put the bowl on the plate | 100.00% | 50/50 |
| turn on the stove | 100.00% | 50/50 |
| put the bowl on the stove | 94.00% | 47/50 |
| put the wine bottle on top of the cabinet | 92.00% | 46/50 |
| open the middle drawer of the cabinet | 90.00% | 45/50 |
| push the plate to the front of the stove | 90.00% | 45/50 |
| put the bowl on top of the cabinet | 74.00% | 37/50 |
| put the cream cheese in the bowl | 72.00% | 36/50 |
| open the top drawer and put the bowl inside | 68.00% | 34/50 |
| put the wine bottle on the rack | 13.16% | 5/38 |

---

## LIBERO Goal 评估结果

> 评估时间：2026-07-07 15:47:31 CST
> Task Suite: **libero_goal**
> Trials per Task: **50**
> 评估设备: NVIDIA RTX 4090

| Step | Success Rate | 成功数/总数 |
|------|-------------|-------------|
| 50000 | **0.7900** (0.7900%) | 395/500 |

### 各任务成功率

```
open the middle drawer of the cabinet: 45/50=90.00%
open the top drawer and put the bowl inside: 34/50=68.00%
push the plate to the front of the stove: 45/50=90.00%
put the bowl on the plate: 50/50=100.00%
put the bowl on the stove: 47/50=94.00%
put the bowl on top of the cabinet: 37/50=74.00%
put the cream cheese in the bowl: 36/50=72.00%
put the wine bottle on the rack: 5/50=10.00%
put the wine bottle on top of the cabinet: 46/50=92.00%
turn on the stove: 50/50=100.00%
```

*评估报告路径：/disk/rl/starVLA/playground/eval_results/libero_goal/P0-M7-E-H2a-01_starflow_libero-4in1_qwen3vl4b_lwfm_ft0_260703_0849/steps_50000/eval_report.json*

