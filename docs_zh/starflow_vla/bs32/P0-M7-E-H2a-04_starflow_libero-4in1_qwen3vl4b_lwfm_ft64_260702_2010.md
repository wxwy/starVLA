# P0-M7-E-H2a-04: StarFlow LIBERO 4-in-1 Qwen3VL-4B future_tokens=64（bs32）

> **实验代号**: E-H2a-04 / P0-M7  
> **状态**: 🟢 训练运行中  
> **run_id**: `P0-M7-E-H2a-04_starflow_libero-4in1_qwen3vl4b_lwfm_ft64_260702_2010`  
> **启动时间**: 2026-07-02 20:16:02 CST  
> **当前更新**: 2026-07-04 18:00:00 CST  
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
| `is_resume` | **`False`** |
| 设备 | NVIDIA A100-SXM4-80GB（81920 MiB） |

### 模型 / 优化器参数

| 参数 | 值 |
|------|-----|
| **框架** | StarFlowVLA |
| **Base VLM** | Qwen3-VL-4B-Instruct |
| **Action Model** | LayerwiseFM (DiT, 36 layers, 1024 hidden) |
| **Action Dim** | 7 |
| **State Dim** | 8 |
| **Action Horizon** | 8 |
| **Num Target Vision Tokens** | **64** |
| **Total Params** | 5,071.121 M |
| **Trainable Params** | 633.305 M |
| **Frozen Modules** | `qwen_vl_interface` |
| **LR Scheduler** | cosine_with_min_lr |
| **Min LR** | 1.0e-6 |
| **Optimizer** | AdamW (β=(0.9, 0.95), eps=1e-8, wd=1e-8) |
| **Effective Global Batch** | **32**（8 × 4） |

---

## 训练进度

> 最后更新：2026-07-04 18:00:00 CST

| 指标 | 值 |
|------|-----|
| **当前 Step** | **17883 / 80000**（22.35%） |
| **单步耗时** | ~9.18 s/it |
| **已运行时间** | 约 45 小时 59 分钟 |
| **预计剩余时间** | ~158 小时（约 6.6 天） |
| **最新 checkpoint** | `steps_17750` |
| **上一个 checkpoint** | `steps_17500` |

### Loss 记录（部分）

| Step | action_dit_loss | last_micro_loss | 备注 |
|------|----------------|-----------------|------|
| 20 | 1.1153 | 1.0858 | 初始 |
| 500 | 0.2416 | 0.2043 | eval mse=0.0116 |
| 1000 | 0.1776 | 0.2217 | eval mse=0.0099 |
| 2000 | 0.1629 | 0.1679 | eval mse=0.0117 |
| 3000 | 0.1704 | 0.1852 | eval mse=0.0077 |
| 4000 | 0.1319 | 0.2652 | eval mse=**0.00549** |
| 5000 | 0.1663 | 0.1273 | eval mse=0.00837 |
| 6000 | 0.1272 | 0.1391 | eval mse=0.00558 |
| 7000 | 0.1218 | 0.1507 | |
| 8000 | 0.1178 | 0.1067 | |
| 8500 | 0.1147 | 0.1023 | eval mse=0.00619 |
| 9000 | 0.1147 | 0.1023 | |
| 10000 | 0.1178 | 0.1067 | eval mse=0.00619 |
| 11000 | 0.0907 | 0.0860 | eval mse=**0.00502** 🏆 |
| 12000 | 0.1051 | 0.1056 | |
| 13000 | 0.0842 | — | eval mse=0.00700 |
| 13300 | 0.0674 | 0.0760 | |
| 13500 | 0.1173 | — | eval mse=0.00657 ✅ ckpt |
| 14000 | 0.0853 | — | eval mse=0.00732 ✅ ckpt |
| 14500 | 0.0894 | — | eval mse=0.00786 ✅ ckpt |
| 15000 | 0.0800 | — | eval mse=**0.00518** ✅ ckpt |
| 15120 | 0.0572 | 0.0573 | best last_micro |
| 15500 | 0.0858 | — | eval mse=0.00732 ✅ ckpt |
| 16000 | 0.0827 | — | eval mse=**0.00510** ✅ ckpt |
| 16500 | 0.0822 | — | eval mse=— ✅ ckpt |
| 16750 | 0.0980 | 0.1024 | ✅ ckpt |
| 17000 | 0.0977 | 0.1116 | eval mse=**0.00515** ✅ ckpt |
| 17200 | 0.0666 | 0.0411 | best last_micro |
| 17250 | 0.0805 | 0.0781 | ✅ ckpt |
| 17500 | 0.0631 | 0.0745 | eval mse=0.00554 ✅ ckpt |
| 17660 | 0.0965 | 0.0801 | |
| 17700 | 0.0845 | 0.0849 | |
| 17750 | 0.0795 | 0.0746 | ✅ ckpt (step 17740) |
| 17780 | 0.0790 | 0.1059 | |
| 17800 | 0.0687 | 0.0696 | |
| 17820 | 0.0691 | 0.0650 | |
| 17840 | 0.0857 | 0.0707 | |
| 17860 | 0.0820 | 0.0874 | |
| 17880 | 0.0864 | 0.0937 | |
| 17883 | — | — | 🔵 当前 |
| **250–17750** | — | — | ✅ 每 250 steps checkpoint |

### Loss 趋势

- 初始快速下降至 0.24（step 500），随后逐渐收敛
- 最佳 eval mse **0.00502** 在 step 11000，后续 step 15000/16000/17000 也接近此水平（0.00518/0.00510/0.00515）
- step 13000–17880 损失在 0.057–0.12 之间震荡，无明显上升
- 学习率按 cosine schedule 缓慢衰减（当前 base LR ~2.22e-5）
- 最新 checkpoint **steps_17750** 已于 17:54 保存并同步至持久存储

---

## 系统资源占用

> 最后更新：2026-07-04 18:00:00 CST

### GPU（NVIDIA A100-SXM4-80GB）

| 指标 | 值 |
|------|-----|
| **GPU 利用率** | 100% |
| **显存使用** | 64,269 MiB / 81,920 MiB（78.5%） |
| **功耗** | 334.81 W / 400.00 W |
| **温度** | 54°C |

### Docker 内存（cgroup v2）

| 指标 | 值 |
|------|-----|
| **上限** | 120 GiB |
| **当前已用** | 27.81 GiB |
| **使用率** | 23.2% |

### 存储

| 挂载点 | 容量 | 已用 | 可用 | 使用率 |
|--------|------|------|------|--------|
| `/` | 30G | 16G | 15G | 53% |
| `/localdisk-tmp` | 100G | 31G | 70G | 31% |
| `/disk/rl` | 700T | 557T | 144T | 80% |

---

## 输出目录

```
/disk/rl/starVLA/playground/Checkpoints/P0-M7-E-H2a-04_starflow_libero-4in1_qwen3vl4b_lwfm_ft64_260702_2010/
├── checkpoints/steps_250/ ... steps_17750/  ✅
├── config.full.yaml          ✅
├── config.yaml               ✅
├── dataset_statistics.json   ✅
├── summary.jsonl             ✅
└── wandb/                    ✅
```

---

## 自动监控状态

- 定时任务 `4288195d`：每小时 :07 直接监控 tmux `train`，自动更新 H2a-04 tracker
- 上次更新：2026-07-04 18:00:00 CST

---

## 备注

- 本实验验证 `num_target_vision_tokens=64` 的效果。
- A100 显存使用率 78.5%，未 OOM。
- 单步耗时 ~9.18 s/it，模型前向/反向 ~2.35 s。
- checkpoint 每 250 steps 正常保存至 `steps_17750`。
- step 17000 eval mse=0.00515，step 17500 eval mse=0.00554，接近 best（0.00502 at step 11000）。等待 step 18000 下一轮 eval。
- **重要声明**：本 tracker 文件只由本 cron/手动任务维护；之前的 P1-M1 tracker 被 `.monitor_bs32.py` 误写，已单独还原。

---

*本文档持续更新。*
