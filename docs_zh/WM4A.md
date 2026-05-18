# WM4A：World Model for Action（以世界模型驱动动作）

<a href="https://huggingface.co/collections/StarVLA/world-model-to-vla"><img src="https://img.shields.io/badge/HuggingFace-WM4A%20检查点-orange?style=for-the-badge&logo=huggingface" alt="WM4A 检查点 on HuggingFace"></a>

WM4A 将**预训练的视频生成世界模型**重新用作机器人动作预测的视觉编码器。WM4A 不使用视觉-语言模型（VLM）骨干，而是将图像馈入**Diffusion Transformer（DiT）**——与视频预测相同的架构——并在其中间表示之上附加一个轻量级动作头。

## 动机

视频预测模型学习丰富的时空表示，涵盖物理、物体持久性和动力学。WM4A 直接利用这些表示：

```
VLM 方案：  图像 → VLM 编码器 → 语言对齐特征 → 动作头
WM4A 方案： 图像 → 视频 DiT 编码器 → 物理对齐特征 → 动作头
```

## 支持的骨干网络

| 骨干 | 参数量 | 层数 | 隐藏维度 | 来源 |
|----------|--------|--------|------------|--------|
| **Cosmos-Predict2-2B** | 2B | 28 | 2048 | NVIDIA Cosmos |
| **Wan2.2-T2V** | 5B | 30 | 3072 | Alibaba Wan |

## 架构

```
输入: {images: [PIL], instruction: str, actions: [T, action_dim]}
  │
  ├─ 文本编码器 (T5 / UMT5)
  │   instruction → [B, L_text, text_hidden]
  │
  ├─ VAE 编码器
  │   images → latents [B, 16, T_latent, H/8, W/8]
  │
  └─ DiT Transformer（冻结或微调）
      latents + text_embeds → hidden_states [B, N_tokens, hidden_dim]
      │
      └─ 动作头（以下三种变体之一）
          → predicted actions [B, chunk_len, action_dim]
```

## 动作头变体

每种骨干可以搭配三种动作头类型：

| 变体 | 动作头 | 使用内容 | 速度 | 框架 |
|---------|------------|--------------|-------|------------|
| **OFT** | MLP 回归 | 仅最后一层 | 最快 | `CosmoPredict2OFT`、`WanOFT` |
| **GR00T** | 流匹配扩散（单层） | 仅最后一层 | 中等 | `CosmoPredict2GR00T`、`WanGR00T` |
| **PI** | 逐层交叉注意力 DiT | 所有 Transformer 层 | 最慢 | `CosmoPredict2PI`、`WanPI` |

这给出了 **7 种框架组合**（包括一个通用的 `WM4A_OFT`）。

## 快速入门：检查数据流

运行内置 demo 查看完整的前向传播（训练 + 推理）：

```bash
python starVLA/model/framework/WM4A/CosmoPredict2GR00T.py
```

此脚本：
1. 加载配置并实例化 `CosmoPredict2_GR00T` 模型
2. 创建一个包含多视图图像和随机动作的合成批次
3. 运行**训练前向传播**并打印动作损失
4. 运行**推理前向传播**并打印预测动作

> **注意：** 你需要本地下载 Cosmos-Predict2-2B 权重。
> 在脚本中设置路径，或通过以下方式下载：
> ```bash
> huggingface-cli download nvidia/Cosmos-Predict2-2B-Video2World \
>     --local-dir ./playground/Pretrained_models/nvidia/Cosmos-Predict2-2B-Video2World
> ```

## 训练

### 推荐：LIBERO 配合 CosmoPredict2OFT

训练 WM4A 模型最简单的方式是在 LIBERO 基准上使用 OFT（MLP）动作头：

```bash
# 从 starVLA 项目根目录
bash examples/LIBERO/train_files/run_libero_train.sh
```

运行前编辑脚本设置本地路径：

```bash
Framework_name=CosmoPredict2OFT          # 或 CosmoPredict2GR00T、WanOFT 等
base_wm=nvidia/Cosmos-Predict2-2B-Video2World  # tokenizer/processor 仍然需要
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=<LIBERO 数据集路径>
data_mix=libero_all                       # 或 libero_goal
```

关键训练参数：

| 参数 | 描述 | 推荐值 |
|----------|-------------|-------------|
| `--framework.name` | 框架类名 | `CosmoPredict2OFT` |
| `--framework.action_model.future_action_window_size` | 动作预测视野 | 7 |
| `--datasets.vla_data.per_device_batch_size` | 每 GPU batch size | 16 (OFT) / 8 (GR00T) |
| `--trainer.max_train_steps` | 总训练步数 | 80000 |
| `--trainer.save_interval` | 检查点保存频率 | 10000 |

### 切换骨干

要使用 Wan 替代 Cosmos，只需更改框架名称：

```bash
Framework_name=WanOFT       # 或 WanGR00T、WanPI
```

配置 YAML 保持不变——框架类根据其注册名称自动处理骨干加载。

## 代码结构

```
starVLA/model/
├── framework/
│   └── WM4A/
│       ├── CosmoPredict2GR00T.py   # Cosmos + 流匹配头
│       ├── CosmoPredict2OFT.py     # Cosmos + MLP 头
│       ├── CosmoPredict2PI.py      # Cosmos + 逐层交叉 DiT 头
│       ├── WanGR00T.py             # Wan + 流匹配头
│       ├── WanOFT.py               # Wan + MLP 头
│       ├── WanPI.py                # Wan + 逐层交叉 DiT 头
│       └── WM4A_OFT.py            # 通用 WM 后端 + MLP 头
└── modules/
    ├── world_model/
    │   └── CosmoPredict2.py        # Cosmos 骨干封装（VAE + T5 + DiT）
    └── action_model/
        ├── MLP_ActionHeader.py             # OFT 动作头
        ├── flow_matching_head/             # GR00T 动作头（单层 FM）
        └── LayerwiseFM_ActionHeader.py     # PI 动作头（逐层 FM）
```

## 关键实现细节

- **精度**：DiT 前向传播使用 `bfloat16` 以提高速度；动作头使用 `float32` 以保证数值稳定性。
- **特征提取**：在选定的 DiT 块上使用 PyTorch hooks 捕获中间隐藏状态，无需修改骨干代码。
- **时间步技巧**：特征提取期间设置 `timestep=0`——这在接近干净噪声级别提取表示，而非去噪。
- **形状重塑**：DiT 输出 `[B, C, T, H, W]` 视频张量，重塑为 `[B, T*H*W, C]` token 序列供动作头使用。
- **框架注册**：所有 WM4A 类通过 `@FRAMEWORK_REGISTRY.register("FrameworkName")` 注册自身，并在导入时自动发现——无需手动导入。
