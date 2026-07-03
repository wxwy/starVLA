# P1-M1-E-H2b-02: StarFlow LIBERO 4-in-1 Qwen3VL-4B continuous_head ft=32（bs32）

> **实验代号**: E-H2b-02 / P1-M1  
> **状态**: 🔴 已停止（最后 checkpoint steps_10750）  
> **run_id**: `P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260702_2021`  
> **启动时间**: 2026-07-02 20:24:19 CST  
> **当前更新**: 2026-07-03 17:36 CST
> **tmux 会话**: 无（已停止）  
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

> 最后更新：2026-07-03 17:16:03 CST

| 指标 | 值 |
|------|-----|
| **当前 Step** | **10750 / 80000**（13.4%，已停止） |
| **完成比例** | 13.4% |
| **单步耗时** | ~7.45 s/it
| **数据加载耗时** | ~0.000 s
| **模型前向/反向耗时** | ~0.214 s
| **已运行时间** | 约 20 小时 43 分钟 |
| **预计剩余时间** | ~137 小时（约 5.7 天） |
| **最新 checkpoint** | `steps_10750` |
| **最近 checkpoint** | `steps_10000` |

### Loss 记录（部分）

| Step | action_dit_loss | 备注 |
|------|----------------|------|
| 20 | 1.3252 | 初始 |
| 1000 | 0.1776 | |
| 2500 | 0.1542 | |
| 5000 | 0.1663 | eval mse_score=0.0084 |
| 7500 | ~0.12 | |
| 10000 | ~0.11 | |
| **10750** | — | 🔴 最终 checkpoint |

| Step | action_dit_loss | 备注 |
|------|----------------|------|

### Loss 趋势
- Step 0→10750：loss 从 1.33 逐步降至 ~0.11，收敛正常
- eval mse_score 最低 0.00837（step 5000）
- lr 平稳 cosine 衰减


---

## 系统资源占用

> 最后更新：2026-07-03 17:16:03 CST

### GPU（NVIDIA GeForce RTX 4090）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | N/A（实验已停止） |
| **显存使用** | N/A（实验已停止） |
| **显存空闲** | N/A |
| **功耗** | N/A |
| **温度** | N/A |

### Docker 内存

| 指标 | 值 |
|------|-----|
| **Docker 内存总量** | 56.0 GiB |
| **Docker 内存已用** | N/A（实验已停止） |
| **Docker 内存可用** | N/A |
| **Swap** | 0 B |

### 存储

| 挂载点 | 容量 | 已用 | 可用 | 使用率 |
|--------|------|------|------|--------|
| `/` (overlay) | 30G | 831M | 30G | 3% |
| `/localdisk-tmp` | 100G | 16G | 85G | 16% |
| `/disk/rl` | 700T | 548T | 153T | 79% |

---

## checkpoint 时间线

| checkpoint | 时间 | 备注 |
|------------|------|------|
| steps_250  | Jul 2 21:23 | ✅ |
| steps_500  | Jul 2 21:53 | ✅ |
| steps_750  | Jul 2 22:22 | ✅ |
| steps_1000 | Jul 2 22:51 | ✅ |
| steps_2000 | Jul 3 ~01:24 | ✅ |
| steps_3000 | Jul 3 ~03:58 | ✅ |
| steps_4000 | Jul 3 ~06:33 | ✅ |
| steps_5000 | Jul 3 ~09:07 | ✅ |
| steps_6000 | Jul 3 ~11:40 | ✅ |
| steps_7000 | Jul 3 ~14:13 | ✅ |
| steps_8000 | Jul 3 ~16:46 | ✅ |
| steps_9000 | Jul 3 ~19:19 | ✅ |
| steps_10000 | Jul 3 ~21:52 | ✅ |
| steps_10250 | Jul 3 ~23:00 | ✅ |
| steps_10500 | Jul 4 ~00:08 | ✅ |
| **steps_10750** | **Jul 4 ~00:30** | 🔴 最终 |

| checkpoint | 时间 | 备注 |
|------------|------|------|

---

## 成本估算

| 项目 | 计算 |
|------|------|
| **每 step 耗时** | ~7.45 s/it |
| **每 step 成本** | 本地 4090，暂不记录 |
| **已运行时间** | ~22 小时（Jul 2 20:24 → Jul 3 ~00:30） |
| **完整 80000 steps 预估** | 80000 × 7.15s ≈ 159h ≈ 6.6 天 |

---

## 输出目录

```
/disk/rl/starVLA/playground/Checkpoints/P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260702_2021/
├── checkpoints/
│   ├── steps_250/
│   ├── ...
│   ├── steps_10000/
│   ├── steps_10250/
│   ├── steps_10500/
│   └── steps_10750/    ← 最终 checkpoint
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

## probe 实验（gradient accumulation 修复验证）

| Run | 配置 | 最后 checkpoint |
|-----|------|----------------|
| `accum_fix_bs8_probe_260702` | bs=8, grad_accum=4 | steps_15250 |
| `accum_fix_bs32_probe_260702` | bs=1, grad_accum=32 | steps_15250 |
| `accum_fix_bs8_meanloss_probe_260702` | bs=8, mean loss | steps_15250 |
| `accum_fix_bs32_meanloss_probe_260702` | bs=1, mean loss | steps_15250 |

> 以上 probe 实验从旧 run 的 `steps_15250` resume，用于验证 gradient accumulation 修复在不同 bs 配置下的 `action_dit_loss`（micro-step 均值 vs 汇总值）行为。

---

## 备注

- 本实验验证 `state_mode=continuous_head` 与默认路径的状态注入差异。
- **当前 run 为 FRESH START**，从 step 0 开始全新训练。
- 旧 run `260621_1901` 在 A100 上训练至 step ~17000，已废弃。

---

