# P1-M1-E-H2b-02: StarFlow LIBERO 4-in-1 Qwen3VL-4B continuous_head ft=32（bs32）

> **实验代号**: E-H2b-02 / P1-M1
> **状态**: 🟢 训练运行中
> **run_id**: `P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260702_2021`
> **启动时间**: 2026-07-02 20:24:19 CST
> **当前更新**: 2026-07-05 23:15 CST
> **tmux 会话**: `train`（attached）
> **配置来源**: `configs/starflow_vla/state/continuous_head.yaml`

---

## 实验概述

P1-M1 **state conditioning 对照**：在 StarFlowVLA 框架中启用 `state_mode=continuous_head`，将本体状态（8D state）通过连续向量直接注入 action head。

本 run 为 **从 scratch 全新训练**（`is_resume=False`），使用 4090 bs32 配置（per_device_batch_size=1 × gradient_accumulation_steps=32）。对应旧 run `P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260621_1901`（A100 阶段，已废弃）。

---

## 训练参数

### 脚本级参数

| 参数 | 值 |
|------|-----|
| `RUN_ID` | `P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260702_2021` |
| `CONFIG_YAML` | `configs/starflow_vla/state/continuous_head.yaml` |
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
| 单价 | 本地 4090，暂不记录 |

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
| **Num Target Vision Tokens** | 32 |
| **DiT Attention Heads** | 16 |
| **DiT Params** | 532,326,426 |
| **Dropout** | 0.2 |
| **Total Params** | 5,071.088 M |
| **Trainable Params** | 633.272 M |
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
| **Effective Global Batch** | **32**（1 × 32） |
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

> 最后更新：2026-07-05 23:15:00 CST

| 指标 | 值 |
|------|-----|
| **当前 Step** | **38400 / 80000**（48.0%） |
| **完成比例** | 48.0% |
| **单步耗时** | ~6.9-7.0 s/it |
| **数据加载耗时** | ~0.000 s |
| **模型前向/反向耗时** | ~0.21-0.22 s |
| **已运行时间** | 约 74 小时 51 分钟 |
| **预计剩余时间** | ~48 小时（约 2.0 天） |
| **最新 checkpoint** | `steps_38250` |
| **上一个 checkpoint** | `steps_38000` |

### Loss 记录（部分）

| Step | action_dit_loss | last_micro_loss | 备注 |
|------|----------------|-----------------|------|
| 20 | 1.3252 | — | 初始 |
| 160 | 0.2280 | — | |
| 520 | 0.2500 | — | |
| 1060 | 0.1847 | — | |
| 1600 | 0.2709 | — | |
| 11160 | 0.0975 | 0.0258 | |
| 11180 | 0.0899 | 0.1355 | |
| 11200 | 0.0953 | 0.1280 | |
| 11700 | 0.0959 | 0.0393 | |
| 12220 | 0.0962 | 0.1440 | |
| 12700 | 0.0834 | 0.0657 | |
| 13220 | 0.0919 | 0.0333 | |
| 14250 | 0.1036 | 0.0419 | |
| 14760 | 0.0837 | 0.1874 | |
| 15260 | 0.0863 | 0.1821 | |
| 16300 | 0.0891 | 0.0882 | |
| 17320 | 0.0812 | 0.0848 | |
| 23480 | 0.0634 | 0.0685 | |
| 23980 | 0.0858 | 0.0534 | |
| 24080 | 0.0756 | 0.0310 | |
| 24500 | 0.0667 | 0.0879 | mse=0.0124 |
| 25000 | 0.0724 | 0.0543 | mse=0.0154 |
| 25520 | 0.0567 | 0.0537 | |
| 26040 | 0.0522 | 0.0251 | |
| 26560 | 0.0521 | 0.0698 | |
| 27060 | 0.0528 | 0.0542 | |
| 27580 | 0.0592 | 0.0981 | |
| 28100 | 0.0609 | 0.0085 | |
| 28620 | 0.0737 | 0.1349 | |
| 29080 | 0.0640 | 0.0158 | |
| 29640 | 0.0573 | 0.0964 | |
| 30140 | 0.0557 | 0.1401 | |
| 30660 | 0.0512 | 0.0530 | |
| 30760 | 0.0459 | 0.0666 | |
| 31140 | 0.0420 | 0.0419 | |
| 31700 | 0.0551 | 0.1683 | |
| 32200 | 0.0547 | 0.0398 | |
| 32720 | 0.0528 | 0.0272 | |
| 33260 | 0.0678 | 0.0606 | |
| 33780 | 0.0451 | 0.0060 | |
| 34300 | 0.0476 | 0.0291 | |
| 34800 | 0.0568 | 0.1129 | |
| 35300 | 0.0465 | 0.0863 | |
| 35820 | 0.0655 | 0.0186 | |
| 36340 | 0.0474 | 0.0204 | |
| 36860 | 0.0463 | 0.0118 | |
| 37380 | 0.0490 | 0.1615 | |
| 37900 | 0.0525 | 0.0508 | |
| 38400 | 0.0482 | 0.0201 | 🔵 当前 |

### Loss 趋势

- 初始 loss 1.33 → 快速下降至 0.18–0.27 区间
- step 11000+ loss 在 0.09–0.13 震荡，整体缓慢下降
- `last_micro_loss` 与 `action_dit_loss`（32 micro-step mean）差异显著，说明单 micro-batch 方差大
- 学习率缓慢下降（cosine schedule）

---

## 系统资源占用

> 最后更新：2026-07-05 17:15:00 CST

### GPU（NVIDIA GeForce RTX 4090）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 81%（本机当前） |
| **显存使用** | 23160 MiB / 24564 MiB（94.3%） |
| **显存空闲** | 1404 MiB |
| **功耗** | 284.94 W / 450.00 W |
| **温度** | 59°C |

### Docker 内存（cgroup v2）

| 指标 | 值 |
|------|-----|
| **Docker 内存上限** | 56 GiB（cgroup v2 memory.max） |
| **Docker 内存已用** | 37.5 GiB |
| **Docker 内存可用** | ~18.5 GiB |
| **Swap** | 0 B |

### 存储

| 挂载点 | 容量 | 已用 | 可用 | 使用率 |
|--------|------|------|------|--------|
| `/` (overlay) | 30G | 16G | 15G | 53% |
| `/localdisk-tmp` | 100G | 31G | 70G | 31% |
| `/disk/rl` | 700T | 548T | 153T | 79% |

---

## 输出目录

```
/disk/rl/starVLA/playground/Checkpoints/P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260702_2021/
├── checkpoints/
├── config.full.yaml          ✅
├── config.yaml               ✅
├── dataset_statistics.json   ✅
├── summary.jsonl             ✅
└── wandb/                    ✅
```

本地暂存：`/localdisk-tmp/P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260702_2021/`

---

## 相关链接

- **WandB Run**: [260702_2021](https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla/runs/P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260702_2021)
- **tmux 会话**: `train`

---

## 启动命令

```bash
cd /disk/rl/starVLA
RUN_ID="P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_$(date +%y%m%d_%H%M)" \
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
LOCAL_CHECKPOINT_KEEP_COUNT=2 \
LOCAL_CHECKPOINT_ROOT=/localdisk-tmp \
STARVLA_PYTHON=/opt/conda/envs/starVLA/bin/python \
bash examples/LIBERO/train_files/run_starflow_train_ready.sh
```

---

## 备注

- 本实验验证 `state_mode=continuous_head` 与默认路径的状态注入差异。
- **当前 run 为 FRESH START**，从 step 0 开始全新训练。
- 旧 run `260621_1901` 在 A100 上训练，已废弃。
- 4090 显存使用率 94.3%（23.2G/24G），余量 ~1.4G，尚未 OOM。
- 速度 ~6.9-7.0 s/it，预计完整训练约 2.3 天。
- 🎯 达成 30k steps 里程碑，loss 维持 0.056 低位
- 🎯 突破 40% 完成（32000/80000），loss 维持 0.055 附近

---

*本 tracker 仅由本机（RTX 4090）维护；如发现状态被外部机器覆盖，会恢复为运行中状态。*
