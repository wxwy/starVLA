# MoWA-E-003 (LoRA): Future Latent Prior without History · Wan LoRA 4 卡版（robocasa365_open_drawer_target_human）

> **实验代号**: E-003 LoRA / `mowa_future_latent_prior_wo_history_lora`
> **状态**: 🟢 训练运行中（15:34 首次进程崩溃，15:38 已重启，当前为第二次进程）
> **run_id**: `MoWA-E-003_future_latent_prior_wo_history_lora_260717_1200`
> **启动时间**: 2026-07-17 12:59 CST（首次）；2026-07-17 15:38 CST（当前进程，从 step 0 重训）
> **当前更新**: 2026-07-17 17:15 CST
> **tmux 会话**: `train`（accelerate 4 进程）
> **配置来源**: `configs/mowa/mowa_e003_future_latent_prior_lora_long_training_launch_candidate.yaml`

---

## 实验概述

MoWA E-003 **future latent prior without history** 的 **Wan LoRA 版本**：在 **WanPI** 框架（WorldModel→Action）下，以 **Wan2.2-TI2V-5B** 为预训练 world model backbone，backbone 基座权重冻结、**挂载 LoRA adapter 参与训练**，在 **robocasa365_open_drawer_target_human** 数据上训练 VLA 模型。4 × RTX 4090 分布式。

与上一版 4 卡全量 run（`..._260717_1120`，已停止）相比，本次的核心变化：
- ✅ **Wan LoRA 启用**（adapter=`mowa_wan`，rank=8，alpha=16，dropout=0.0，层 0–末尾，target=cross_attention + self_attention）— world model backbone 不再完全冻结，LoRA 适配层可训练
- ✅ **新增 `wan_lora` 学习率组**：lr=1.0e-5（480 个参数组）
- ⚠️ **`cross_view.gate_init: 1.0`**（上版为 0.001）— cross-view 门控从 1.0 开始，日志中 `cross_view_gate_mean` 确认为 1.0
- 其余配置与上版一致：`history_window_steps: 0`、`enable_future_latent_prior_loss: true`（loss_scale=0.05）、`enable_future_supervision_loss: false`、`enable_layerwise_bridge_token_coupling: false`、双视角 + cross-view attention
- 可训练参数 512.794 M / 5514.707 M（较上版 500.997 M / 5502.911 M 增加约 11.8 M LoRA 参数）
- 从 scratch 开始，未加载上一版 `steps_500` checkpoint

---

## 训练事件

| 时间 (CST) | 事件 |
|------------|------|
| 2026-07-17 ~12:15 | 上一版 4 卡全量 run（`..._260717_1120`）停止（仅保留 steps_500 checkpoint） |
| 2026-07-17 12:59 | 运行目录创建；`Wan LoRA enabled: adapter=mowa_wan, groups=['cross_attention','self_attention']`；从 scratch 启动（`is_resume=False`） |
| 2026-07-17 13:00 | 数据流验证 1/2、2/2 通过，进入正式训练循环 |
| 2026-07-17 13:35–15:23 | 依次保存 `steps_500`（后被轮转删除）、`steps_1000`（14:11）、`steps_1500`（14:47）、`steps_2000`（15:23）；step 2000 触发 robocasa 异步 eval |
| 2026-07-17 ~15:34 | ⚠️ **首次进程崩溃**（step ~2120，epoch 1.97）：wandb `output.log` 在打印 traceback 途中中断（栈位于 `Wan2.py:539` forward），异常类型未被完整记录；崩溃前一步 `timing/model` 尖峰 29.7s |
| 2026-07-17 15:36–15:38 | **进程重启**（同一启动命令，pt_elastic 15:36:41 拉起 4 个 worker）；**未恢复 checkpoint，从 step 0 重新训练** |
| 2026-07-17 16:09–16:47 | 首次 run 遗留的 eval 任务写出 `steps_2000.eval`：仅完成 1 条 episode（`success_rate=0.0`，样本不足，**不具代表性**） |
| 2026-07-17 16:15 / 16:52 | 第二次进程保存 `steps_500`（16:52 轮转删除）与 `steps_1000`（覆盖写入同名目录） |
| 2026-07-17 17:15 | 当前 step ~1290，训练正常，~4.43 s/it |

> WandB run：`run-20260717_125949-MoWA-E-003_future_latent_prior_wo_history_lora_260717_1200`。
> ⚠️ 重启后未创建新的 wandb run 目录，`latest-run` 仍指向 12:59 的 run，其 `output.log` 自 15:34 起停止写入——第二次进程的 wandb 记录可能不完整，必要时可手动 `wandb sync`。

---

## 训练参数

### 脚本级参数

| 参数 | 值 |
|------|-----|
| `run_id` | `MoWA-E-003_future_latent_prior_wo_history_lora_260717_1200` |
| `CONFIG_YAML` | `configs/mowa/mowa_e003_future_latent_prior_lora_long_training_launch_candidate.yaml` |
| `DATA_MIX` | `robocasa365_open_drawer_target_human` |
| `MAX_TRAIN_STEPS` | 80000 |
| `SAVE_INTERVAL` | 500 |
| `EVAL_INTERVAL` | 500 |
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
| **Base World Model** | Wan2.2-TI2V-5B-Diffusers + **LoRA adapter** |
| **Action Model** | LayerwiseFM（DiT，30 层，hidden 1024，16 heads） |
| **Action Dim** | 12 |
| **State Dim** | 32 |
| **Action Horizon** | 32 |
| **Future Window Steps** | 8 |
| **History Window Steps** | 0 |
| **Noise Schedule** | BetaAlpha（α=1.5, β=1.0） |
| **Inference Timesteps** | 4 |
| **Frozen Modules** | `backbone`（基座权重冻结，LoRA 适配层除外） |
| **Learning Rate (action_model)** | 1.0e-4 |
| **Learning Rate (base)** | 2.5e-5 |
| **Learning Rate (qwen_vl_interface)** | 1.0e-5 |
| **Learning Rate (wan_lora)** | **1.0e-5**（新增 LR 组） |
| **LR Scheduler** | cosine_with_min_lr（min_lr=1.0e-6） |
| **Warmup Steps** | 1000 |
| **Optimizer** | AdamW（β=(0.9, 0.95), eps=1e-8, wd=1.0e-8） |
| **Effective Global Batch** | **32**（per_device=2 × grad_accum=4 × 4 卡） |
| **Checkpoint Format** | lightweight（keep_latest_count=3，permanent steps: 5000/10000/20000/.../80000） |
| **Seed** | 42 |
| **总参数量** | 5514.707 M |
| **可训练参数量** | 512.794 M |

### Wan LoRA 参数

| 参数 | 值 |
|------|-----|
| `lora.enabled` | **true** ✅ |
| adapter 名称 | `mowa_wan` |
| `rank` | 8 |
| `alpha` | 16 |
| `dropout` | 0.0 |
| `start_layer` / `end_layer` | 0 / null（全部层） |
| `target_groups` | cross_attention, self_attention |

### MoWA / Future Latent Prior 特有参数

| 参数 | 值 |
|------|-----|
| `enable_future_latent_prior_loss` | **true** ✅ |
| `enable_future_supervision_loss` | false |
| `enable_layerwise_bridge_token_coupling` | false |
| `loss_scale.mowa_future_latent_prior` | 0.05 |
| `loss_scale.mowa_future_supervision` | 0.0 |
| `future_loss` | main_weight=1.0, wrist_weight=1.0 |
| `done_head` | hidden_dim=512, loss_weight=0.25 |
| `multi_view.enabled` | true（main / wrist 双视角） |
| `cross_view.enabled` | true（num_layers=10, start_layer_ratio=0.6667, **gate_init=1.0**, bottleneck=512, zero_init_output） |
| `action_fusion` | num_queries=8, num_heads=8 |
| `validate_data_flow` | true（validation_steps=2） |

### 数据集

| 数据集 | 样本数（frames/transitions）| embodiment |
|--------|-----------------------------|------------|
| `robocasa365_open_drawer_target_human`（`v1.0/target/atomic/OpenDrawer/20250816/lerobot`） | 34,424 | new_embodiment（panda_omron_robocasa365） |

> 使用 Wan2.2 VAE latent 缓存：`playground/Datasets/robocasa365_wan2.2_latent/v1.0/target/atomic`（window manifest: `wan2.2_h10_f8_train.parquet`）。

---

## 训练进度

> 最后更新：2026-07-17 17:15 CST
> 当前为第二次进程（15:38 从 step 0 重启），已运行约 1 小时 37 分钟

| 指标 | 值 |
|------|-----|
| **当前 Step** | ~1290 / 80000 |
| **完成比例** | 1.6% |
| **单步耗时** | ~4.43 s/it |
| **model_time** | ~0.9–1.5 s |
| **data_time** | ~0.002 s |
| **本次进程运行时间** | ~1 小时 37 分钟（首次进程 12:59–15:34 曾跑到 step ~2120） |
| **预计剩余时间** | ~96 小时（约 4 天，预计 2026-07-21 傍晚完成） |
| **最新 checkpoint** | `steps_1000`（第二次进程，16:52 覆盖写入）；`steps_1500`/`steps_2000`（15:23 前首次进程遗留） |
| **最新 eval** | `steps_2000.eval`（首次进程遗留，16:47 完成写入）：仅 1 episode，success_rate=0.0，不具代表性 |

### Loss 记录

| Step | action_dit_loss | mowa_future_latent_prior_loss | loss_multiview_total | loss_action | loss_total | 备注 |
|------|-----------------|-------------------------------|----------------------|-------------|------------|------|
| 10 | 1.4523 | 1.7939 | 1.6701 | 1.3920 | 1.4755 | 首次进程；lr warmup 中；cross_view_gate_mean=1.0 |
| 30 | 1.0787 | 2.1291 | 1.5898 | 1.0593 | ≈1.1387 | 首次进程；loss_total 为推算值 |
| ~2120 | — | — | — | 0.0562 | 0.0767 | 首次进程崩溃前最后记录（epoch 1.97，取自 wandb output.log）；aux_to_action_ratio=0.365 |
| 1280 | 0.1372 | 0.6040 | 0.7596 | 0.0666 | ≈0.1045 | 第二次进程；cross_view_output_norm=585.8，loss_total 为推算值 |

> 注：
> - `mowa_future_supervision_loss` 恒为 0.0（该监督关闭，loss_scale=0.0）。
> - `loss_total` = `loss_action` + `weighted_future_total`（done 损失未计入 total）。
> - `cross_view_gate_mean` 从 1.0 起步，step 1280 时约 1.0016（缓慢上移）；`cross_view_output_norm` 已从初始 0.17 升至 ~586，新增日志键 `cross_view_residual_ratio`（step 1280 为 0.243）与 `done_positive_ratio`（0.0）。
> - logging_frequency=10，以上取自 tmux pane 缓冲区 / wandb output.log。

---

## 系统资源占用

> 最后更新：2026-07-17 17:08 CST

### GPU（4 × NVIDIA GeForce RTX 4090）

| GPU | 利用率 | 显存 | 功耗 | 温度 |
|-----|--------|------|------|------|
| 0 | 100% | 22570 / 24564 MiB (92%) | 407.3 W | 76°C |
| 1 | 100% | 22590 / 24564 MiB (92%) | 391.4 W | 74°C |
| 2 | 100% | 22570 / 24564 MiB (92%) | 375.9 W | 75°C |
| 3 | 100% | 22570 / 24564 MiB (92%) | 407.6 W | 78°C |

> 功耗较启动初期（~300 W）明显上升至 375–408 W，温度 74–78°C，仍在正常范围。

### 容器（Docker）资源 — 通过 cgroup v2 实测

> 本环境运行在 Docker 容器内且无 `docker` CLI，以下为容器 cgroup 实测值（非宿主机全局值）。

| 指标 | 值 |
|------|-----|
| **内存使用** | **185.2 GiB / 224 GiB 限额（82.7%）** ⚠️ 较启动初期（81.3 GiB）大幅上升 |
| **CPU 使用** | ~6.6 核 / 52 核限额（12.6%，10 秒采样 `cpu.stat usage_usec`） |
| **主要进程** | 4 × 训练主进程各 ~99.6% CPU、RSS ~9.9 GB；多个 pt_data_worker 各 RSS ~8.6 GB（内存占用主体） |

### 存储

| 挂载点 | 使用情况 |
|--------|----------|
| `/` (overlay) | 930M / 30G (4%) |
| `/disk/rl` | 528T / 700T (76%) |

---

## 输出目录

```
playground/mowa_ckpt/MoWA-E-003_future_latent_prior_wo_history_lora_260717_1200/   （共 52G）
├── checkpoints/
│   ├── steps_1000/                ✅（第二次进程 16:52 覆盖写入）
│   ├── steps_1500/                ⚠️ 首次进程遗留（14:47，崩溃前权重）
│   ├── steps_2000/                ⚠️ 首次进程遗留（15:23，崩溃前权重）
│   └── steps_2000.eval/           ⚠️ 首次进程 eval，仅 1 episode（success_rate=0.0，不具代表性）
├── config.full.yaml               ✅
├── config.yaml                    ✅（实际访问配置快照）
├── dataset_statistics.json        ✅
├── summary.jsonl                  ✅（记录 steps 500/1000/1500/2000 保存标记）
└── wandb/wandb/
    └── run-20260717_125949-MoWA-E-003_future_latent_prior_wo_history_lora_260717_1200  ✅（output.log 止于 15:34）
```

> 历史/前版输出目录：
> ```
> playground/mowa_ckpt/MoWA-E-003_future_latent_prior_wo_history_260717_1120/  # 4 卡全量版，已停止（保留 steps_500）
> playground/mowa_ckpt/MoWA-E-003_future_latent_prior_wo_history_260717_0020/  # 单卡版，已停止
> playground/mowa_ckpt/MoWA-E-003_future_latent_prior_wo_history/              # 07-17 00:14 创建，无 checkpoint
> playground/mowa_ckpt/tmp_MoWA-E-003_future_latent_prior_wo_history/          # 早期运行，steps_1000.eval
> ```

---

## 相关链接

- **WandB Project**: [MoWA](https://wandb.ai/silencewx-harbin-institute-of-technology/MoWA)
- **WandB Run**: `run-20260717_125949-MoWA-E-003_future_latent_prior_wo_history_lora_260717_1200`
- **前版（4 卡全量）日志**: `docs_zh/mowa/train_log/MoWA-E-003_future_latent_prior_wo_history_260717_1120.md`
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
  --config_yaml configs/mowa/mowa_e003_future_latent_prior_lora_long_training_launch_candidate.yaml \
  --run_root_dir playground/mowa_ckpt \
  --validate-data-flow \
  --validation-steps 2
```

---

## 备注

- 本 run 为 E-003 的 **Wan LoRA 版**，取代同日 11:27 启动的 4 卡全量 run（`..._260717_1120`，step ~500 停止）。
- 核心变化：backbone 挂 LoRA（rank=8, alpha=16, 全部层, cross+self attention），新增 `wan_lora` LR 组（1e-5），`cross_view.gate_init` 0.001→1.0。
- 从 scratch 开始训练，未加载旧 checkpoint；`is_resume=False`。
- 单步 ~4.4 s/it（4 卡，global batch 32），与上版基本持平。
- **2026-07-17 17:15 监控更新**：
  - 首次进程于 ~15:34 在 step ~2120 崩溃（栈在 `Wan2.py:539` forward，异常信息未完整落盘；崩溃前出现 29.7s 的 model_time 尖峰），15:38 以同一命令重启，**从 step 0 重训**，未加载 steps_2000。
  - 当前 checkpoint 目录为混合状态：`steps_1000` 已被第二次进程覆盖，`steps_1500`/`steps_2000` 仍是首次进程（崩溃前）权重，引用时需注意区分。
  - `steps_2000.eval` 仅含 1 条 episode（success_rate=0.0），为首次进程遗留的不完整 eval，不代表模型真实水平。
  - 容器内存占用从启动初期 81 GiB 升至 185 GiB（限额 224 GiB，82.7%），主体为 dataloader workers；后续需持续关注，接近限额时有 OOM 风险。
  - 重启后 wandb 未建新 run，`output.log` 止于 15:34，训练曲线以 tmux 日志为准。
- 训练在 tmux 会话 `train` 中运行，断开会话不会中断训练。
- 本日志由定时监控任务每 4 小时自动更新（训练进度、文件生成、GPU 与容器资源占用）。

---

*本文档将持续更新。*
