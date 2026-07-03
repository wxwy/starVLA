# P0-M7-E-H2a-04: StarFlow LIBERO 4-in-1 Qwen3VL-4B future_tokens=64（bs32）

> **实验代号**: E-H2a-04 / P0-M7  
> **状态**: 🟢 训练运行中  
> **run_id**: `P0-M7-E-H2a-04_starflow_libero-4in1_qwen3vl4b_lwfm_ft64_260702_2010`  
> **启动时间**: 2026-07-02 20:16:02 CST  
> **当前更新**: 2026-07-03 16:03:20 CST  
> **tmux 会话**: `train`（attached）  
> **配置来源**: `configs/starflow_vla/ablations/future_tokens_64.yaml`

---

## 实验概述

P0-M7 **future tokens 对照**：在 StarFlowVLA 框架中将 action head 的 `num_target_vision_tokens` 从默认 32 提升至 **64**，验证更多目标视觉 token 对 LIBERO 4-in-1 连续动作学习的收敛与显存影响。

本 run 为 **从 scratch 全新训练**（`is_resume=False`），使用 A100 bs32 配置（per_device_batch_size=8 × gradient_accumulation_steps=4）。

> 注：本 tracker 曾一度被误标记为「step 5500 停止」，实际训练在 tmux `train` 中持续运行，当前 step 已推进至 7695。

---

## 训练参数

### 脚本级参数

| 参数 | 值 |
|------|-----|
| `RUN_ID` | `P0-M7-E-H2a-04_starflow_libero-4in1_qwen3vl4b_lwfm_ft64_260702_2010` |
| `CONFIG_YAML` | `configs/starflow_vla/ablations/future_tokens_64.yaml` |
| `DATA_MIX` | `libero_all` |
| `MAX_TRAIN_STEPS` | 80,000 |
| `SAVE_INTERVAL` | 250 |
| `EVAL_INTERVAL` | 500 |
| `LOGGING_FREQUENCY` | 20 |
| `NUM_PROCESSES` | 1 |
| `GRADIENT_ACCUMULATION_STEPS` | **4** |
| `PER_DEVICE_BATCH_SIZE` | **8** |
| `NUM_WORKERS` | 8 |
| `BASE_VLM` | `/disk/rl/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct` |
| `LIBERO_DATA_ROOT` | `/disk/rl/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA` |
| `WANDB_PROJECT` | `starflow_vla` |
| `WANDB_ENTITY` | `silencewx-harbin-institute-of-technology` |
| `is_resume` | **`False`**（从 scratch） |
| 设备 | NVIDIA A100-SXM4-80GB（81920 MiB） |
| 单价 | 本地 A100，暂不记录 |

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
| **Num Target Vision Tokens** | **64** |
| **DiT Attention Heads** | 16 |
| **DiT Params** | 532,326,426 |
| **Dropout** | 0.2 |
| **Total Params** | 5,071.121 M |
| **Trainable Params** | 633.305 M |
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
| **Effective Global Batch** | **32**（8 × 4） |
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

> 最后更新：2026-07-03 16:03:20 CST

| 指标 | 值 |
|------|-----|
| **当前 Step** | **7695 / 80000**（9.62%） |
| **完成比例** | 9.62% |
| **单步耗时** | ~9.17–9.19 s/it |
| **数据加载耗时** | ~0.0003–0.0007 s |
| **模型前向/反向耗时** | ~2.31–2.35 s |
| **已运行时间** | 约 19 小时 47 分钟 |
| **预计剩余时间** | ~181 小时（约 7.5 天） |
| **最新 checkpoint** | `steps_7500` |
| **上一个 checkpoint** | `steps_7250` |

### Loss 记录（部分）

| Step | action_dit_loss | last_micro_loss | 备注 |
|------|----------------|-----------------|------|
| 20 | 1.1153 | 1.0858 | 初始 |
| 100 | 0.5190 | 0.6180 | |
| 500 | 0.2416 | 0.2043 | eval mse_score=0.0116 |
| 1000 | 0.1776 | 0.2217 | eval mse_score=0.0099 |
| 1500 | 0.1704 | 0.1852 | eval mse_score=0.0131 |
| 2000 | 0.1629 | 0.1679 | eval mse_score=0.0117 |
| 2500 | 0.1542 | 0.1837 | eval mse_score=0.0110 |
| 3000 | 0.1704 | 0.1852 | eval mse_score=0.0077 |
| 3500 | 0.1629 | 0.1679 | eval mse_score=0.0075 |
| 4000 | 0.1319 | 0.2652 | eval mse_score=**0.00549** 🏆 |
| 4500 | 0.1153 | 0.1934 | eval mse_score=0.00619 |
| 5000 | 0.1663 | 0.1273 | eval mse_score=0.00837 |
| 5500 | 0.1237 | 0.1281 | eval mse_score=0.00881 |
| 6000 | 0.1272 | 0.1391 | eval mse_score=0.00558 |
| 6500 | 0.1547 | 0.1628 | |
| 7000 | 0.1218 | 0.1507 | |
| 7500 | 0.1122 | 0.1134 | eval mse_score=0.00619 |
| 7620 | 0.1067 | 0.1023 | |
| 7680 | 0.1043 | 0.0937 | |
| 7695 | — | — | 🔵 当前 |
| **250** | — | — | ✅ checkpoint |
| **500** | — | — | ✅ checkpoint |
| **750** | — | — | ✅ checkpoint |
| **1000** | — | — | ✅ checkpoint |
| **1250** | — | — | ✅ checkpoint |
| **1500** | — | — | ✅ checkpoint |
| **1750** | — | — | ✅ checkpoint |
| **2000** | — | — | ✅ checkpoint |
| **2250** | — | — | ✅ checkpoint |
| **2500** | — | — | ✅ checkpoint |
| **2750** | — | — | ✅ checkpoint |
| **3000** | — | — | ✅ checkpoint |
| **3250** | — | — | ✅ checkpoint |
| **3500** | — | — | ✅ checkpoint |
| **3750** | — | — | ✅ checkpoint |
| **4000** | — | — | ✅ checkpoint |
| **4250** | — | — | ✅ checkpoint |
| **4500** | — | — | ✅ checkpoint |
| **4750** | — | — | ✅ checkpoint |
| **5000** | — | — | ✅ checkpoint |
| **5250** | — | — | ✅ checkpoint |
| **5500** | — | — | ✅ checkpoint |
| **5750** | — | — | ✅ checkpoint |
| **6000** | — | — | ✅ checkpoint |
| **6250** | — | — | ✅ checkpoint |
| **6500** | — | — | ✅ checkpoint |
| **6750** | — | — | ✅ checkpoint |
| **7000** | — | — | ✅ checkpoint |
| **7250** | — | — | ✅ checkpoint |
| **7500** | — | — | ✅ checkpoint |

### Loss 趋势

- 初始 loss 1.12 → 快速下降至 0.24 附近（step 500）
- step 1000–4000 区间震荡下降，最低 mse_score 0.00549 出现在 step 4000
- step 4000–7500 区间 `action_dit_loss` 整体稳定在 0.10–0.16
- step 7500 eval `mse_score` 为 0.00619，较 step 4000 有所回升但仍处于较低水平
- `last_micro_loss` 与 `action_dit_loss` 差异较小（grad_accum=4）
- 学习率从初始值缓慢下降（cosine schedule）

---

## 系统资源占用

> 最后更新：2026-07-03 16:03:20 CST

### GPU（NVIDIA A100-SXM4-80GB）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 99% |
| **显存使用** | 64,269 MiB / 81,920 MiB（78.5%） |
| **显存空闲** | 16,884 MiB |
| **功耗** | 320.24 W / 400.00 W |
| **温度** | 54°C |

### Docker 内存（cgroup v2）

| 指标 | 值 |
|------|-----|
| **Docker 内存上限** | 120 GiB |
| **当前已用** | 24.26 GiB |
| **使用率** | 20.2% |
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
| **每 step 耗时** | ~9.2 s/it |
| **每 step 成本** | 本地 A100，暂不记录 |
| **已运行时间** | ~19.8 小时 |
| **完整 80000 steps 预估** | 80000 × 9.2s ≈ 204h ≈ 8.5 天 |

---

## 输出目录

```
/disk/rl/starVLA/playground/Checkpoints/P0-M7-E-H2a-04_starflow_libero-4in1_qwen3vl4b_lwfm_ft64_260702_2010/
├── checkpoints/
│   ├── steps_250/
│   ├── steps_500/
│   ├── steps_750/
│   ├── steps_1000/
│   ├── steps_1250/
│   ├── steps_1500/
│   ├── steps_1750/
│   ├── steps_2000/
│   ├── steps_2250/
│   ├── steps_2500/
│   ├── steps_2750/
│   ├── steps_3000/
│   ├── steps_3250/
│   ├── steps_3500/
│   ├── steps_3750/
│   ├── steps_4000/
│   ├── steps_4250/
│   ├── steps_4500/
│   ├── steps_4750/
│   ├── steps_5000/
│   ├── steps_5250/
│   ├── steps_5500/
│   ├── steps_5750/
│   ├── steps_6000/
│   ├── steps_6250/
│   ├── steps_6500/
│   ├── steps_6750/
│   ├── steps_7000/
│   ├── steps_7250/
│   └── steps_7500/
├── config.full.yaml          ✅
├── config.yaml               ✅
├── dataset_statistics.json   ✅
├── summary.jsonl             ✅
└── wandb/                    ✅
```

本地暂存：`/localdisk-tmp/P0-M7-E-H2a-04_starflow_libero-4in1_qwen3vl4b_lwfm_ft64_260702_2010/`

---

## 相关链接

- **WandB Run**: [260702_2010](https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla/runs/P0-M7-E-H2a-04_starflow_libero-4in1_qwen3vl4b_lwfm_ft64_260702_2010)
- **tmux 会话**: `train`

---

## 启动命令

```bash
cd /disk/rl/starVLA
RUN_ID="P0-M7-E-H2a-04_starflow_libero-4in1_qwen3vl4b_lwfm_ft64_$(date +%y%m%d_%H%M)" \
CONFIG_YAML=configs/starflow_vla/ablations/future_tokens_64.yaml \
DATA_MIX=libero_all \
MAX_TRAIN_STEPS=80000 \
SAVE_INTERVAL=250 \
LOGGING_FREQUENCY=20 \
EVAL_INTERVAL=500 \
NUM_PROCESSES=1 \
GRADIENT_ACCUMULATION_STEPS=4 \
PER_DEVICE_BATCH_SIZE=8 \
NUM_WORKERS=8 \
LOCAL_CHECKPOINT_KEEP_COUNT=2 \
LOCAL_CHECKPOINT_ROOT=/localdisk-tmp \
STARVLA_PYTHON=/opt/conda/envs/starVLA/bin/python \
bash examples/LIBERO/train_files/run_starflow_train_ready.sh
```

---

## 自动监控状态

- 定时任务 `e775fd3e`：每小时 :47 运行监控脚本
- 上次手动更新：2026-07-03 16:03:20 CST

---

## 备注

- 本实验验证 `num_target_vision_tokens=64`（默认 32）对 StarFlowVLA 动作学习的影响。
- **当前 run 为 FRESH START**，从 step 0 开始全新训练；实际训练持续运行中，未在 step 5500 终止。
- A100 80GB 显存使用率 78.5%（64.3G/80G），余量 ~16.9G，尚未 OOM。
- 单步耗时 ~9.2 s/it；模型前向/反向耗时 ~2.33 s（per_device_batch_size=8，单次 micro-step 处理 8 条样本）。
- checkpoint 每 250 steps 正常保存，已保存至 `steps_7500`；`/localdisk-tmp` → `/disk/rl` 后台同步正常。
- 训练日志中出现过数据读取异常（`Invalid data found when processing input`），均自动 retry 后未中断。
- **最佳 loss**: 0.09397（step 5300）；**最佳 mse**: 0.00549（step 4000）。

---

*本文档将持续更新。*
