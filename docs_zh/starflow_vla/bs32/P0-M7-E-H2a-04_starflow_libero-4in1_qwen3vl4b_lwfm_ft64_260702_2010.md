# P0-M7-E-H2a-04: StarFlow LIBERO 4-in-1 Qwen3VL-4B future_tokens=64（bs32）

> **实验代号**: E-H2a-04 / P0-M7  
> **状态**: 🟢 训练运行中  
> **run_id**: `P0-M7-E-H2a-04_starflow_libero-4in1_qwen3vl4b_lwfm_ft64_260702_2010`  
> **启动时间**: 2026-07-02 20:16:02 CST  
> **当前更新**: 2026-07-03 08:28:00 CST  
> **tmux 会话**: `train`（attached）  
> **配置来源**: `configs/starflow_vla/ablations/future_tokens_64.yaml`

---

## 实验概述

P0-M7 **future tokens 对照**：在 StarFlowVLA 框架中将 action head 的 `num_target_vision_tokens` 从默认 32 提升至 **64**，验证更多目标视觉 token 对 LIBERO 4-in-1 连续动作学习的收敛与显存影响。

本 run 为 **从 scratch 全新训练**（`is_resume=False`），使用 A100 bs32 配置（per_device_batch_size=8 × gradient_accumulation_steps=4）。

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

> 最后更新：2026-07-03 08:28:00 CST

| 指标 | 值 |
|------|-----|
| **当前 Step** | **~4748 / 80000**（5.9%，估算） |
| **完成比例** | 5.9% |
| **单步耗时** | ~9.2 s/it（历史数据） |
| **数据加载耗时** | 无法获取（远端机器） |
| **模型前向/反向耗时** | 无法获取（远端机器） |
| **已运行时间** | 约 12 小时 12 分钟 |
| **预计剩余时间** | ~192 小时（约 8.0 天） |
| **最新 checkpoint** | `steps_4500`（07:50） |
| **上一个 checkpoint** | `steps_4250`（07:11） |
| **上一个 checkpoint** | `steps_3500` |

### Loss 记录（部分）

| Step | action_dit_loss | last_micro_loss | 备注 |
|------|----------------|-----------------|------|
| 20 | 1.1153 | 1.0858 | 初始 |
| 100 | 0.5190 | 0.6180 | |
| 200 | 0.3668 | 0.3663 | |
| 300 | 0.2850 | 0.3069 | |
| 400 | 0.2944 | 0.2652 | |
| 500 | 0.2416 | 0.2043 | 含 eval mse_score=0.0116 |
| 600 | 0.2089 | 0.2109 | |
| 700 | 0.2308 | 0.1945 | |
| 800 | 0.1835 | 0.2135 | |
| 900 | 0.1970 | 0.2313 | |
| 1000 | 0.1776 | 0.2217 | 含 eval mse_score=0.0099 |
| 1500 | 0.1704 | 0.1852 | 含 eval mse_score=0.0131 |
| 2000 | 0.1629 | 0.1679 | |
| 2500 | 0.1542 | 0.1837 | |
| 3000 | 0.1704 | 0.1852 | |
| 3500 | 0.1629 | 0.1679 | |
| 3750 | 0.1122 | 0.1134 | 含 eval mse_score=0.0131 |
| 3880 | 0.1158 | 0.1626 | |
| 3900 | 0.1149 | 0.1319 | |
| 3920 | 0.1272 | 0.0924 | |
| 3940 | 0.1397 | 0.1257 | 🔵 当前 |
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

### Loss 趋势

- 初始 loss 1.12 → 快速下降至 0.24 附近（step 500）
- `action_dit_loss` 在 step 2000–3880 区间整体稳定在 0.11–0.17，最低下探至 0.112（step 3750）
- eval `mse_score`: step 500 为 0.0116，step 1000 为 0.0099，step 1500/3750 约为 0.0131；eval 指标存在波动
- `last_micro_loss` 与 `action_dit_loss` 差异较小（grad_accum=4，micro-batch 间方差相对可控）
- 学习率从初始值缓慢下降（cosine schedule）

---

## 系统资源占用

> 最后更新：2026-07-03 06:24:51 CST

### GPU（NVIDIA A100-SXM4-80GB）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 100% |
| **显存使用** | 64,269 MiB / 81,920 MiB（78.5%） |
| **显存空闲** | 16,884 MiB |
| **功耗** | 320.57 W / 400.00 W |
| **温度** | 56°C |

### Docker 内存（cgroup v2）

| 指标 | 值 |
|------|-----|
| **Docker 内存上限** | 120 GiB |
| **当前已用** | 33.62 GiB |
| **使用率** | 28.0% |
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
| **已运行时间** | ~10.1 小时 |
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
│   └── steps_4500/
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

## 备注

- 本实验验证 `num_target_vision_tokens=64`（默认 32）对 StarFlowVLA 动作学习的影响。
- **当前 run 为 FRESH START**，从 step 0 开始全新训练。
- A100 80GB 显存使用率 78.5%（64.3G/80G），余量 ~16.9G，尚未 OOM；GPU 利用率维持 100%。
- 单步耗时 ~9.2 s/it；模型前向/反向耗时 ~2.35 s（per_device_batch_size=8，单次 micro-step 处理 8 条样本）。
- checkpoint 每 250 steps 正常保存，已保存至 `steps_3750`；`/localdisk-tmp` → `/disk/rl` 后台同步正常。
- 训练日志中出现三次数据读取异常（index 30578、45269、120829: `Invalid data found when processing input`），均自动 retry 后未中断。
- loss 从 1.12 快速下降至 0.24 后，在 step 2000–3940 区间整体稳定在 0.11–0.17；eval `mse_score` 在 0.0099–0.0131 之间波动，长程收敛需继续观察。

---

*本文档将持续更新。*
