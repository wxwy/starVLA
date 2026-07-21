# MoWA-E-003 (b1): Future Latent Prior without History · 4 卡版（robocasa365_open_drawer_target_human）

> **实验代号**: E-003 b1 / `mowa_future_latent_prior_wo_history`
> **状态**: ⏹️ 已停止（2026-07-17 12:15 前后，由 LoRA 版 run `MoWA-E-003_future_latent_prior_wo_history_lora_260717_1200` 取代，见对应日志）
> **run_id**: `MoWA-E-003_future_latent_prior_wo_history_260717_1120`
> **启动时间**: 2026-07-17 11:27 CST
> **当前更新**: 2026-07-17 11:45 CST
> **tmux 会话**: `train`（accelerate 4 进程，主训练进程 PID 624776–624779）
> **配置来源**: `configs/mowa/mowa_e003_future_latent_prior_long_training_launch_candidate.yaml`

---

## 实验概述

MoWA E-003 **future latent prior without history** 的 **4 卡重启动版本**（用户称 b1）：在 **WanPI** 框架（WorldModel→Action）下，以 **Wan2.2-TI2V-5B** 为预训练 world model backbone，在 **robocasa365_open_drawer_target_human** 数据上训练 VLA 模型。

与前一版单卡 run（`..._260717_0020`，见同名日志）相比，本次的核心变化：
- ✅ **4 × RTX 4090 分布式训练**（accelerate，num_processes=4，bf16，NCCL）
- ✅ `per_device_batch_size: 2` × `grad_accum: 4` × 4 卡 = **global batch 32**（与单卡版等效）
- ⚠️ **`enable_future_supervision_loss: false`**（单卡版为 true）— future feature heads 监督关闭
- ⚠️ **`enable_layerwise_bridge_token_coupling: false`**（单卡版为 true）— 不再向 DiT 注入 bridge token
- ⚠️ **`loss_scale.mowa_future_latent_prior: 0.05`**（单卡版为 1.0）— latent prior 损失权重大幅调低
- ✅ 保留：`history_window_steps: 0`、`enable_future_latent_prior_loss: true`、双视角（main + wrist）+ cross-view attention
- `num_workers: 48`，`num_warmup_steps: 1000`（单卡版为 500）
- 启动命令附带 `--validate-data-flow --validation-steps 2`（先做 2 步数据流验证再进入正式训练）
- backbone 冻结，仅训练 action model / projector / future heads 等模块（可训练 500.997 M / 5502.911 M）

---

## 训练事件

| 时间 (CST) | 事件 |
|------------|------|
| 2026-07-17 10:02 | tmux 会话 `train` 创建 |
| 2026-07-17 11:20 | 运行目录 `MoWA-E-003_future_latent_prior_wo_history_260717_1120/` 创建 |
| 2026-07-17 11:27 | **正式训练启动**：4 卡 accelerate，从 scratch 开始（`is_resume=False`，目录内无 checkpoint） |
| 2026-07-17 ~11:13–11:27 | 数据流验证（validation_steps=2）通过后进入正式训练循环 |
| 2026-07-17 12:00–12:01 | 保存首个 checkpoint `steps_500`（17G，含 4 卡 optimizer 分片） |
| 2026-07-17 ~12:15 | **本 run 停止**；实验切换为 **Wan LoRA 版**（run_id `..._lora_260717_1200`）：backbone 改挂 LoRA adapter（rank=8, alpha=16），cross_view gate_init 0.001→1.0，新增 wan_lora 学习率组，详见新日志 |

> 注：
> - 前一单卡 run（`..._260717_0020`）已于当日上午停止，本次为其 4 卡替代版本，**未加载任何旧 checkpoint**，模型从随机初始化开始。
> - WandB run：`run-20260717_112751-MoWA-E-003_future_latent_prior_wo_history_260717_1120`。

---

## 训练参数

### 脚本级参数

| 参数 | 值 |
|------|-----|
| `run_id` | `MoWA-E-003_future_latent_prior_wo_history_260717_1120` |
| `CONFIG_YAML` | `configs/mowa/mowa_e003_future_latent_prior_long_training_launch_candidate.yaml` |
| `DATA_MIX` | `robocasa365_open_drawer_target_human` |
| `MAX_TRAIN_STEPS` | 80000 |
| `SAVE_INTERVAL` | 500 |
| `EVAL_INTERVAL` | 500（config.full.yaml 另有一处为 200） |
| `LOGGING_FREQUENCY` | 10 |
| `NUM_PROCESSES` | **4**（accelerate 多卡，NCCL，bf16） |
| `GRADIENT_ACCUMULATION_STEPS` | **4** |
| `PER_DEVICE_BATCH_SIZE` | **2** |
| `NUM_WORKERS` | 48 |
| `BASE_VLM / BASE_WM` | `./playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers` |
| `DATA_ROOT` | `playground/Datasets/robocasa365` |
| `is_resume` | `False`（从 scratch） |
| 设备 | 4 × NVIDIA GeForce RTX 4090（每张 24564 MiB），`CUDA_VISIBLE_DEVICES=0,1,2,3` |
| 其他 | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`，`--main_process_port 29501` |

### 模型 / 优化器参数

| 参数 | 值 |
|------|-----|
| **框架** | WanPI（WorldModel→Action） |
| **Base World Model** | Wan2.2-TI2V-5B-Diffusers |
| **Action Model** | LayerwiseFM（DiT，30 层，hidden 1024，16 heads） |
| **Action Dim** | 12 |
| **State Dim** | 32 |
| **Action Horizon** | 32 |
| **Future Window Steps** | 8 |
| **History Window Steps** | 0 |
| **Noise Schedule** | BetaAlpha（α=1.5, β=1.0） |
| **Inference Timesteps** | 4 |
| **Frozen Modules** | `backbone` |
| **Learning Rate (action_model)** | 1.0e-4 |
| **Learning Rate (base)** | 2.5e-5 |
| **Learning Rate (qwen_vl_interface)** | 1.0e-5 |
| **LR Scheduler** | cosine_with_min_lr（min_lr=1.0e-6） |
| **Warmup Steps** | 1000 |
| **Optimizer** | AdamW（β=(0.9, 0.95), eps=1e-8, wd=1.0e-8） |
| **Effective Global Batch** | **32**（per_device=2 × grad_accum=4 × 4 卡） |
| **Checkpoint Format** | lightweight（keep_latest_count=3，permanent steps: 5000/10000/20000/.../80000） |
| **Seed** | 42 |
| **总参数量** | 5502.911 M |
| **可训练参数量** | 500.997 M |

### MoWA / Future Latent Prior 特有参数

| 参数 | 值 |
|------|-----|
| `enable_future_latent_prior_loss` | **true** ✅ |
| `enable_future_supervision_loss` | **false** ⚠️（单卡版为 true） |
| `enable_layerwise_bridge_token_coupling` | **false** ⚠️（单卡版为 true） |
| `loss_scale.mowa_future_latent_prior` | **0.05**（单卡版为 1.0） |
| `loss_scale.mowa_future_supervision` | 0.0 |
| `future_loss` | main_weight=1.0, wrist_weight=1.0 |
| `done_head` | hidden_dim=512, loss_weight=0.25 |
| `multi_view.enabled` | true（main / wrist 双视角） |
| `cross_view.enabled` | true（num_layers=10, start_layer_ratio=0.6667, gate_init=0.001, bottleneck=512, zero_init_output） |
| `action_fusion` | num_queries=8, num_heads=8 |
| `validate_data_flow` | true（validation_steps=2） |

### 数据集

| 数据集 | 样本数（frames/transitions）| embodiment |
|--------|-----------------------------|------------|
| `robocasa365_open_drawer_target_human`（`v1.0/target/atomic/OpenDrawer/20250816/lerobot`） | 34,424 | new_embodiment（panda_omron_robocasa365） |

> 使用 Wan2.2 VAE latent 缓存：`playground/Datasets/robocasa365_wan2.2_latent/v1.0/target/atomic`（window manifest: `wan2.2_h10_f8_train.parquet`）。

---

## 训练进度

> 最后更新：2026-07-17 11:45 CST
> 本次启动后运行约 18 分钟

| 指标 | 值 |
|------|-----|
| **当前 Step** | ~212 / 80000 |
| **完成比例** | 0.27% |
| **单步耗时** | ~4.0 s/it |
| **model_time** | ~0.9–1.2 s |
| **data_time** | ~0.002 s |
| **本次启动后运行时间** | ~18 分钟 |
| **预计剩余时间** | ~88–90 小时（约 3.7 天，预计 2026-07-21 凌晨完成） |
| **最新 checkpoint** | 尚无（首次保存预计在 step 500，约 13:00 CST） |
| **最新 eval** | 尚无 |

### Loss 记录

| Step | action_dit_loss | mowa_future_latent_prior_loss | loss_multiview_total | loss_action | loss_total | 备注 |
|------|-----------------|-------------------------------|----------------------|-------------|------------|------|
| 150 | 0.4670 | 1.3854 | 0.6219 | 0.3381 | 0.3692 | lr 仍在 warmup（action_model 1.4e-5） |
| 160 | 0.2278 | 2.1088 | 0.8326 | 0.3010 | 0.3426 | |
| 200 | 0.3396 | 1.0662 | 0.3920 | 0.1780 | ≈0.1976 | loss_total 为推算值（= loss_action + weighted_future_total） |

> 注：
> - `mowa_future_supervision_loss` 恒为 0.0（该监督已关闭，loss_scale=0.0）。
> - `loss_total` = `loss_action` + `weighted_future_total`（done 损失未计入 total）。
> - latent prior loss 以 loss_scale=0.05 加权（step 200 时 weighted_future_total≈0.0196）。
> - logging_frequency=10，以上取自 tmux pane 缓冲区可见的完整记录。

---

## 系统资源占用

> 最后更新：2026-07-17 11:45 CST

### GPU（4 × NVIDIA GeForce RTX 4090）

| GPU | 利用率 | 显存 | 功耗 | 温度 |
|-----|--------|------|------|------|
| 0 | 100% | 23036 / 24564 MiB (94%) | 331.8 W | 72°C |
| 1 | 100% | 23036 / 24564 MiB (94%) | 344.9 W | 74°C |
| 2 | 100% | 23036 / 24564 MiB (94%) | 338.6 W | 69°C |
| 3 | 100% | 23536 / 24564 MiB (96%) | 366.6 W | 68°C |

### 容器（Docker）资源 — 通过 cgroup v2 实测

> 本环境运行在 Docker 容器内且无 `docker` CLI，以下为容器 cgroup 实测值（非宿主机全局值）。

| 指标 | 值 |
|------|-----|
| **内存使用** | 93.4 GiB / 224 GiB 限额（41.7%） |
| **CPU 使用** | ~6.7 核 / 52 核限额（12.8%，10 秒采样 `cpu.stat usage_usec`） |
| **CPU throttle** | nr_throttled=11 / 69566 periods（可忽略） |
| **主要进程** | 4 × 训练主进程各 ~98% CPU、RSS ~9.9 GB；若干 pt_data_worker 各 RSS ~8.5 GB；4 × inductor compile_worker 各 ~10% CPU |

### 存储

| 挂载点 | 使用情况 |
|--------|----------|
| `/` (overlay) | 928M / 30G (4%) |
| `/disk/rl` | 528T / 700T (76%) |

---

## 输出目录

```
playground/mowa_ckpt/MoWA-E-003_future_latent_prior_wo_history_260717_1120/
├── checkpoints/                     （空，首次保存预计 step 500）
├── config.full.yaml                 ✅
├── config.yaml                      ✅（实际访问配置快照）
├── dataset_statistics.json          ✅
└── wandb/wandb/
    └── run-20260717_112751-MoWA-E-003_future_latent_prior_wo_history_260717_1120  ✅
```

> 历史/未使用输出目录：
> ```
> playground/mowa_ckpt/MoWA-E-003_future_latent_prior_wo_history_260717_0020/  # 单卡版 run，已停止
> playground/mowa_ckpt/MoWA-E-003_future_latent_prior_wo_history/              # 07-17 00:14 创建，无 checkpoint
> playground/mowa_ckpt/tmp_MoWA-E-003_future_latent_prior_wo_history/          # 早期运行，steps_1000.eval
> ```

---

## 相关链接

- **WandB Project**: [MoWA](https://wandb.ai/silencewx-harbin-institute-of-technology/MoWA)
- **WandB Run**: `run-20260717_112751-MoWA-E-003_future_latent_prior_wo_history_260717_1120`
- **前版（单卡）日志**: `docs_zh/mowa/train_log/MoWA-E-003_future_latent_prior_wo_history_260716_1601.md`
- **训练主机**: 本地服务器（4 × RTX 4090，Docker 容器）

---

## 启动命令

```bash
cd /disk/rl/starVLA
CUDA_VISIBLE_DEVICES=0,1,2,3 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
.venv/bin/accelerate launch \
  --num_processes 4 \
  --main_process_port 29501 \
  starVLA/training/train_starvla.py \
  --config_yaml configs/mowa/mowa_e003_future_latent_prior_long_training_launch_candidate.yaml \
  --run_root_dir playground/mowa_ckpt \
  --validate-data-flow \
  --validation-steps 2
```

---

## 备注

- 本 run 为 E-003 的 4 卡版本（用户称 **b1**），取代 2026-07-17 00:21 启动的单卡 run（`..._260717_0020`）。
- 与单卡版相比，除分布式外还有实质配置变化：`enable_future_supervision_loss=false`、`enable_layerwise_bridge_token_coupling=false`、`mowa_future_latent_prior` loss_scale 1.0→0.05、warmup 500→1000、num_workers 8→48。
- 从 scratch 开始训练，未加载旧 checkpoint；`is_resume=False`。
- 单步 ~4.0 s/it（4 卡，global batch 32），对比单卡版 ~14.5 s/it，吞吐约 3.6 倍。
- 4 卡均满载（100% util，~94–96% 显存），容器 CPU 余量充足（~13%）。
- 首次 checkpoint 预计 step 500；permanent checkpoint 在 5000/10000/20000/.../80000。
- **2026-07-17 12:15 更新**：本 run 已于 step ~500 后停止，最终仅保留 `steps_500` checkpoint（17G）；目录下另有 `mowa-e003-b1-loss-2.png` loss 曲线图与 `summary.jsonl`。后续由 LoRA 版 run（`..._lora_260717_1200`）接替，监控日志见新文档。
- 训练在 tmux 会话 `train` 中运行，断开会话不会中断训练。
- 本日志由定时监控任务每 4 小时自动更新（训练进度、文件生成、GPU 与容器资源占用）。

---

*本文档将持续更新。*
