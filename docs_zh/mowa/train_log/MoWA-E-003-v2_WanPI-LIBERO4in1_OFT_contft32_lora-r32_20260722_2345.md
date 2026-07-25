# MoWA E-003-v2 LIBERO4in1 训练记录（OFT backbone）

> run id：`MoWA-E-003-v2_WanPI-LIBERO4in1_OFT_contft32_lora-r32_20260722_2345`
> 启动时间：2026-07-22 23:46 CST
> 配置：`configs/mowa/mowa_e003_v2_wanpi_libero4in1_oft_contft32_lora.yaml`
> Git SHA：`ecfc5f2`
> 状态：🟢 运行中

## 与 origWan 的关键差异

| 维度 | origWan (1917) | **OFT (2345)** |
|------|---------------|-----------------|
| backbone 初始化 | 原始 Wan2.2 | WM4A-Wan2d2-OFT-LIBERO-4in1 (`steps_60000`) |
| `pretrained_checkpoint` | null | `/disk/rl/models/mowa_init_candidates/.../steps_60000_pytorch_model.pt` |
| `reload_modules` | null | backbone |
| 预期 | 从头训练 | OFT backbone + 新 action head |

## 运行配置

- **硬件**：8×RTX 4090 (24 GiB)，bf16 mixed precision
- **Batch**：per-device=2，gradient accumulation=2，global batch=32
- **数据集**：`libero_4in1_wanpi`（4 子集：object/goal/spatial/10）
- **框架**：WanPI (LayerwiseFM)，**OFT backbone 初始化**
- **LoRA**：r=32，alpha=64，cross_attention + self_attention，0-29 层
- **Action**：delta_ee / abs mode，7 dim，horizon=32
- **State**：8 dim → sin/cos → 16 dim
- **Multi-view**：main + wrist，flatten_into_batch
- **Future latent prior**：enabled，loss weight=0.05
- **LR**：base=2.5e-5，wan_lora=1e-5，action_model=1e-4；cosine_with_min_lr
- **Max steps**：100000

## 进度

| Step | action_dit_loss | mse_score | wrist_far | 耗时 |
|------|----------------|-----------|-----------|------|
| 50 | 1.001 | — | — | 初始 |
| 250 | 0.371 | — | 1.362 | ~10m |
| 500 | 0.208 | — | 1.211 | ~20m |
| 750 | 0.254 | — | 1.100 | ~30m |
| 1000 | 0.522 | — | 1.008 | ~40m |
| 1250 | 0.168 | — | 0.959 | ~50m |
| 1500 | 0.155 | — | 0.921 | ~1h |
| 1750 | 0.156 | **0.0187** | ~0.88 | ~1.1h |
| 2000 | 0.341 | — | **0.847** | ~1.3h |

## 与 origWan 对照

| 指标 | origWan @2000 | **OFT @2000** | 对比 |
|------|--------------|---------------|------|
| action_dit_loss | 0.196-0.525 | **0.167-0.521** | ≈ |
| mse_score @1750 | 0.0164 | **0.0187** | +14% |
| wrist_far | 0.594 | **0.847** | +43% |

**结论**：OFT backbone 在 action loss 上无明显优势（action head 从零开始），wrist future 预测收敛较慢（OFT backbone 可能只优化了 main camera 特征）。wrist_far 差距正在持续缩小（从 +0.56 缩小到 +0.25）。

## 对照实验设计

两个训练并行运行，形成直接对照：
- **origWan (1917)**：原始 Wan2.2 → 68h，已到 step 6650+，稳定收敛
- **OFT (2345)**：OFT backbone + 新 action head → 预期更快收敛
