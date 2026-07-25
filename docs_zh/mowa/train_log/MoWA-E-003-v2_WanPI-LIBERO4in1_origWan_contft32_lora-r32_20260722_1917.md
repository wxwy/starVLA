# MoWA E-003-v2 LIBERO4in1 训练记录（正式运行）

> run id：`MoWA-E-003-v2_WanPI-LIBERO4in1_origWan_contft32_lora-r32_20260722_1917`
> 启动时间：2026-07-22 19:18 CST
> 配置：`configs/mowa/mowa_e003_v2_wanpi_libero4in1_contft32_lora.yaml`
> Git SHA：`ecfc5f2`
> 状态：✅ 已完成（KeyboardInterrupt 手动停止，最终 step 6650，~23:39 CST）

## 运行配置

- **硬件**：8×RTX 4090 (24 GiB)，bf16 mixed precision
- **Batch**：per-device=2，gradient accumulation=2，global batch=32
- **数据集**：`libero_4in1_wanpi`（4 子集：object/goal/spatial/10）
- **框架**：WanPI (LayerwiseFM)，original Wan2.2 初始化（无 pretrained checkpoint overlay）
- **LoRA**：r=32，alpha=64，cross_attention + self_attention，0-29 层
- **Action**：delta_ee / abs mode，7 dim，horizon=32
- **State**：8 dim → sin/cos → 16 dim
- **Multi-view**：main + wrist，flatten_into_batch
- **Future latent prior**：enabled，loss weight=0.05
- **Done head**：loss weight=0.1
- **LR**：base=2.5e-5，wan_lora=1e-5，action_model=1e-4；cosine_with_min_lr，min_lr=1e-6，warmup=2000
- **优化器**：AdamW (0.9, 0.95)，eps=1e-8，weight_decay=1e-8
- **Max steps**：100000

## 进度与关键指标

| Step | action_dit_loss | mse_score (eval) | 耗时 |
|------|----------------|-------------------|------|
| 50 | 0.998 | — | 1m |
| 100 | 0.519 | — | 3m |
| 150 | 0.334 | — | 5m |
| 200 | 0.528 | — | 7m |
| 250 | 0.399 | **0.0194** | 10m |
| 300 | 0.709 | — | 12m |
| 350 | 0.591 | — | 14m |
| 400 | 0.212 | — | 16m |
| 450 | 0.521 | — | 19m |
| 500 | 0.196 | **0.0216** | 21m |
| 550 | 0.273 | — | 23m |
| 600 | 0.424 | — | 25m |
| 650 | 0.221 | — | 27m |
| 700 | 0.456 | — | 29m |
| 750 | 0.323 | **0.0228** | 31m |
| 800 | 0.668 | — | 34m |
| 850 | 0.343 | — | 36m |
| 900 | 0.274 | — | 38m |
| 950 | 0.644 | — | 40m |
| 1000 | 0.526 | **0.0226** | 42m |
| 1050 | 0.734 | — | 44m |
| 1100 | 0.226 | — | 46m |
| 1150 | **0.123** | — | 47m |

## ETA

- 速度：~24.5 steps/min
- 当前进度：4000 / 100000 (4%)
- 预计总耗时：~68 小时
- 预计完成：2026-07-25 ~15:00 CST

## Checkpoints

| Step | 大小 | 保存时间 |
|------|------|---------|
| 500 | 34 GiB | ~19:39 |
| 1000 | 34 GiB | ~19:58 |
| 2000 | — | ~21:xx |
| 3000 | — | ~22:xx |
| 4000 | — | ~23:xx |

## 已知问题

- 无。shm 问题已修复（`instruction_text_latent=None`，text_embeds 不通过 DataLoader queue 传输，由每个 rank 独立查表补齐）
- GPU 显存稳定在 ~22.8 GiB / 24 GiB
