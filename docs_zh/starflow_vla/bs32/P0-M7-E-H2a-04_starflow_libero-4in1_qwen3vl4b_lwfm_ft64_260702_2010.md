# P0-M7-E-H2a-04: StarFlow LIBERO 4-in-1 Qwen3VL-4B future_tokens=64（bs32）

> **实验代号**: E-H2a-04 / P0-M7  
> **状态**: 🔴 已停止（被 H2a-02 ft16 替代）  
> **run_id**: `P0-M7-E-H2a-04_starflow_libero-4in1_qwen3vl4b_lwfm_ft64_260702_2010`  
> **启动时间**: 2026-07-02 20:16:02 CST  
> **停止时间**: 2026-07-03 ~10:24 CST（steps_5500 checkpoint 保存后 kill）  
> **当前更新**: 2026-07-03 10:43:00 CST
> **配置来源**: `configs/starflow_vla/ablations/future_tokens_64.yaml`

---

## 实验概述

P0-M7 **future tokens 对照**：在 StarFlowVLA 框架中将 action head 的 `num_target_vision_tokens` 从默认 32 提升至 **64**，验证更多目标视觉 token 对 LIBERO 4-in-1 连续动作学习的收敛与显存影响。

本 run 为 **从 scratch 全新训练**（`is_resume=False`），使用 bs32 配置（per_device_batch_size=8 × gradient_accumulation_steps=4）。

> ⚠️ 本实验于 2026-07-03 ~10:20 被手动停止，为 H2a-02（future_tokens=16, per_device_batch_size=1, grad_accum=32）腾出 GPU 资源。最终 step = **5480**（6.9%），未跑完目标 80000 steps。

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

## 训练进度（最终状态）

> 最后更新：2026-07-03 10:30:00 CST

| 指标 | 值 |
|------|-----|
| **最终 Step** | **5480 / 80000**（6.9%） |
| **完成比例** | 6.9% |
| **单步耗时** | ~9.2 s/it（全程稳定） |
| **模型前向/反向耗时** | ~2.35 s（per micro-step） |
| **数据加载耗时** | ~0.0003 s（可忽略） |
| **已运行时间** | 约 14 小时 4 分钟（Jul 2 20:16 → Jul 3 ~10:20） |
| **若跑完 80000 steps 需时** | ~204 小时（约 8.5 天） |
| **最后 checkpoint** | `steps_5500`（10:24，程序收到 kill 信号后仍完成了此 checkpoint 保存） |
| **最终 WandB logged step** | `5480`（~10:20 CST） |

### Loss 记录

| Step | action_dit_loss | 备注 |
|------|----------------|------|
| 20 | 1.1153 | 初始 |
| 100 | 0.5190 | |
| 200 | 0.3668 | |
| 300 | 0.2850 | |
| 400 | 0.2944 | |
| 500 | 0.2416 | eval mse_score=0.0116 |
| 600 | 0.2089 | |
| 700 | 0.2308 | |
| 800 | 0.1835 | |
| 900 | 0.1970 | |
| 1000 | 0.1776 | eval mse_score=0.0099 |
| 1500 | 0.1704 | eval mse_score=0.0131 |
| 2000 | 0.1629 | |
| 2500 | 0.1542 | |
| 3000 | 0.1704 | |
| 3500 | 0.1629 | |
| 3750 | 0.1122 | eval mse_score=0.0131 |
| 4000 | 0.1319 | checkpoint |
| 4250 | 0.1089 | checkpoint |
| 4500 | 0.1153 | checkpoint |
| 4750 | 0.1222 | checkpoint |
| 4880 | 0.1267 | |
| 4900 | 0.1235 | |
| 4920 | 0.1112 | |
| 4940 | 0.1036 | |
| 4960 | 0.1206 | |
| 4980 | 0.1182 | |
| **5000** | **0.1663** | ⚠️ spike；eval mse_score=**0.00837**（最佳） |
| 5020 | 0.1352 | |
| 5040 | 0.1093 | |
| 5060 | 0.1261 | |
| 5080 | 0.1248 | |
| 5100 | 0.1233 | |
| 5120 | 0.1229 | |
| 5140 | 0.1135 | |
| 5160 | 0.1339 | |
| 5180 | 0.1191 | |
| 5200 | 0.1284 | |
| 5220 | 0.1195 | |
| 5240 | 0.0996 | |
| **5250** | — | ✅ checkpoint |
| 5260 | 0.1322 | |
| 5280 | 0.1050 | |
| **5300** | **0.09397** | 🏆 最低 loss |
| 5320 | 0.1313 | |
| 5340 | 0.09686 | |
| 5380 | 0.1137 | |
| 5400 | 0.1143 | |
| 5420 | 0.1100 | |
| 5440 | 0.1241 | |
| **5460** | **0.1211** | |
| **5480** | **0.1242** | 🔴 最终记录（WandB 最后日志） |

### 完整 Checkpoint 列表（22 个）

| Step | 保存时间 |
|------|----------|
| 250 | Jul 2 20:54 |
| 500 | Jul 2 21:33 |
| 750 | Jul 2 22:11 |
| 1000 | Jul 2 22:50 |
| 1250 | Jul 2 23:28 |
| 1500 | Jul 3 00:07 |
| 1750 | Jul 3 00:46 |
| 2000 | Jul 3 01:24 |
| 2250 | Jul 3 02:03 |
| 2500 | Jul 3 02:41 |
| 2750 | Jul 3 03:20 |
| 3000 | Jul 3 03:58 |
| 3250 | Jul 3 04:37 |
| 3500 | Jul 3 05:16 |
| 3750 | Jul 3 05:54 |
| 4000 | Jul 3 06:33 |
| 4250 | Jul 3 07:11 |
| 4500 | Jul 3 07:50 |
| 4750 | Jul 3 08:29 |
| 5000 | Jul 3 09:07 |
| **5250** | Jul 3 09:43 |
| **5500** | Jul 3 ~10:20 |

> 每 250 steps 正常保存，无遗漏。steps_5500 为最终 checkpoint。

### Loss 趋势分析

- **Phase 1（0–500）**: 快速下降 1.12 → 0.24，网络迅速学习动作分布
- **Phase 2（500–3750）**: 震荡下降至 0.11–0.17 区间；eval mse_score 在 0.0099–0.0131 之间
- **Phase 3（3750–5460）**: loss 进一步下探，最低达 **0.09397（step 5300）**，整体在 0.10–0.13 震荡
- **eval mse_score**: best = **0.00837（step 5000）**；整体在 0.008–0.013 之间波动，呈下降趋势
- **学习率**: action_model 从 1e-4 缓慢 cosine 衰减至 ~9.89e-5（step ~5400）；base 从 2.5e-5 衰减至 ~2.47e-5
- **last_micro_loss** 方差中等，grad_accum=4 时 micro-batch 间差异可控

---

## 系统资源占用（运行中采样）

> 以下均为 H2a-04 运行期间的资源占用快照。

### GPU（NVIDIA GeForce RTX 4090）

| 指标 | 值（H2a-04 运行中） |
|------|---------------------|
| **GPU 利用率** | 100% |
| **显存使用** | ~23,800 MiB / 24,564 MiB（96.9%） |
| **显存空闲** | ~800 MiB |
| **功耗** | ~280–291 W / 300 W |
| **温度** | ~70°C |

> ⚠️ **更正**：原日志错误标注 GPU 为 "A100-SXM4-80GB"，实为 **NVIDIA GeForce RTX 4090（24564 MiB）**。24GB VRAM 下 bs32（per_device=8 × grad_accum=4）已接近显存上限（96.9%），但全程未 OOM。原日志"64,269 MiB / 81,920 MiB"为错误数据。

### Docker 内存（cgroup v2）

| 指标 | 值 |
|------|-----|
| **Docker 内存上限** | **56 GiB**（60129542144 bytes） |
| **H2a-04 运行中已用** | ~33.6 GiB |
| **使用率** | ~60% |
| **Swap** | 0 B |

> ⚠️ **更正**：原日志错误标注 Docker 上限为 120 GiB，实际 cgroup v2 `memory.max` = **56 GiB**。

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
| **已运行时间** | ~14 小时 |
| **完成 step 数** | 5,480 |
| **完整 80000 steps 预估** | 80000 × 9.2s ≈ 204h ≈ 8.5 天 |
| **每 step 成本** | 本地 RTX 4090，免费 |

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
│   └── steps_5500/    ← 最后 checkpoint
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
- **后续实验**: H2a-02（future_tokens=16, bs32 via per_device=1 × grad_accum=32），当前在 tmux `train` 中运行

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
- **于 2026-07-03 ~10:20 被手动停止**（最后 WandB 日志 step 5480），为 H2a-02（ft16）腾出 GPU 资源。最终完成 6.9%。
- RTX 4090 24GB 显存使用率 96.9%，全程未 OOM；per_device_batch_size=8 已接近显存上限。
- 单步耗时 ~9.2 s/it，模型前向/反向耗时 ~2.35 s（per micro-step，8 条样本）。
- checkpoint 每 250 steps 正常保存，共 22 个；`/localdisk-tmp` → `/disk/rl` 后台同步正常。
- 训练日志中出现三次数据读取异常（index 30578、45269、120829: `Invalid data found when processing input`），均自动 retry 后未中断。
- loss 从 1.12 快速下降，最低 **0.09397（step 5300）**；eval mse_score 最低 **0.00837（step 5000）**。
- 本 run 可提供 ft64 在 LIBERO 4-in-1 上的 loss 曲线参考，后续如有需要可从 steps_5500 checkpoint 恢复训练。

---

*本文档已停止更新（实验已终止，被 H2a-02 替代）。*
