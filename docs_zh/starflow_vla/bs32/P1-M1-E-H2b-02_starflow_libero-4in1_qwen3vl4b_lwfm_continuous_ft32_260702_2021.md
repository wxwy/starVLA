# P1-M1-E-H2b-02: StarFlow LIBERO 4-in-1 Qwen3VL-4B continuous_head ft=32（bs32）

> **实验代号**: E-H2b-02 / P1-M1  
> **状态**: 🟢 训练运行中  
> **run_id**: `P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260702_2021`  
> **启动时间**: 2026-07-02 20:24:19 CST  
> **当前更新**: 2026-07-03 10:15:38 CST
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

> 最后更新：2026-07-03 10:15:38 CST

| 指标 | 值 |
|------|-----|
| **当前 Step** | **5441 / 80000**（6.8%） |
| **完成比例** | 6.8% |
| **单步耗时** | ~9.23 s/it |
| **数据加载耗时** | ~0.001 s |
| **模型前向/反向耗时** | ~2.375 s |
| **已运行时间** | 约 3 小时 12 分钟 |
| **预计剩余时间** | ~151 小时（约 6.3 天） |
| **最新 checkpoint** | `steps_7000` |
| **最近 checkpoint** | `steps_1250` |

### Loss 记录（部分）

| Step | action_dit_loss | 备注 |
|------|----------------|------|
| 20 | 1.3252 | 初始 |
| 160 | 0.2280 | |
| 520 | 0.2500 | |
| 600 | 0.2117 | |
| 1060 | 0.1847 | |
| 1600 | 0.2709 | |
| **1620** | **0.2298** | 🔵 当前 |
| **1628** | 检查点中 | `steps_250`–`steps_1500` ✅ |

### Loss 趋势

- 初始 loss 1.33 → 快速下降至 0.18–0.27 区间
- 2% 完成，loss 正常波动，无明显发散
- 学习率缓慢下降（cosine schedule）：action_model 9.99e-5 → 9.99e-5，base 2.50e-5 → 2.50e-5

---

## 系统资源占用

> 最后更新：2026-07-03 10:15:38 CST

### GPU（NVIDIA GeForce RTX 4090）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 100% |
| **显存使用** | 64269 MiB / 81920 MiB（78.5%）
| **显存空闲** | 1,404 MiB |
| **功耗** | 339.40 W / 400.00 W |
| **温度** | 59°C |

### Docker 内存

| 指标 | 值 |
|------|-----|
| **Docker 内存总量** | 1.0Ti
| **Docker 内存已用** | ~65 GiB
| **Docker 内存可用** | ~935 GiB
| **Docker 缓存** | ~427 GiB |
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
| `steps_250` | 21:23 | ✅ |
| `steps_500` | 21:53 | ✅ |
| `steps_750` | 22:22 | ✅ |
| `steps_1000` | 22:51 | ✅ |
| `steps_1250` | 23:21 | ✅ |
| **`steps_1500`** | **23:51** | ✅ 最新 |

---

## 成本估算

| 项目 | 计算 |
|------|------|
| **每 step 耗时** | ~7.15 s/it |
| **每 step 成本** | 本地 4090，暂不记录 |
| **已运行时间** | ~2.1 小时 |
| **完整 80000 steps 预估** | 80000 × 7.15s ≈ 159h ≈ 6.6 天 |

---

## 输出目录

```
/disk/rl/starVLA/playground/Checkpoints/P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260702_2021/
├── checkpoints/
│   ├── steps_250/
│   ├── steps_500/
│   ├── steps_750/
│   └── steps_1000/
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
- 4090 显存使用率 94.3%（23.2G/24G），余量 ~1.4G，尚未 OOM。
- checkpoint 每 250 steps 正常保存，`/localdisk-tmp` → `/disk/rl` 后台同步正常。
- `last_micro_loss` 与 `action_dit_loss`（32 micro-step mean）差异显著，说明单 micro-batch 方差大，gradient accumulation 有效平滑。
- 速度 ~7.15 s/it，预计完整训练约 6.6 天。

---

*本文档将持续更新。*
