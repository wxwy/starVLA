# 快速入门：用 LIBERO 训练和评估你的第一个 VLA

本指南将带你走通完整的 StarVLA 工作流——从安装到训练再到评估——以 **LIBERO** 基准为具体的端到端示例。完成后你将拥有一个训练好的 VLA 策略，并知道如何在仿真中评估它。

> **只想评估已发布的检查点？** 跳到[评估预训练检查点](#7-评估预训练检查点) — 无需训练。
>
> **使用你自己的数据集 / 机器人？** 阅读 [`integrate_your_dataset.md`](integrate_your_dataset.md)
> 获取端到端的"使用我自己的数据"指南，或激活捆绑的
> [agent skill](agent_skills/integrate-starvla-dataset/README.md) 让
> 编程助手（Claude Code、VS Code Copilot 等）驱动集成过程。

---

## 目录

- [0. 安装](#0-安装)
- [1. 验证安装](#1-验证安装)
- [2. 准备训练数据](#2-准备训练数据)
- [3. 准备预训练模型](#3-准备预训练模型)
- [4. 理解训练配置](#4-理解训练配置)
- [5. 训练数据流与原理介绍](../examples/LIBERO/train_files/training_log_1229_libero4in1_qwen3oft.md)
- [5. 理解训练脚本](#5-理解训练脚本)
- [6. 启动训练](#6-启动训练)
- [7. 评估预训练检查点](#7-评估预训练检查点)
- [后续步骤](#后续步骤)

---

## 0. 安装

```bash
git clone https://github.com/starVLA/starVLA
cd starVLA

conda create -n starVLA python=3.10 -y
conda activate starVLA

pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
pip install -e .
```

> **wxwy 备注：flash-attn 版本固定为 2.7.4.post1。**
>
> 选择原因：当前环境为 Python 3.10 + torch 2.6.0+cu124 + CUDA 12.4。相比源码编译 flash-attn 2.8.x，2.7.4.post1 可以直接安装官方预编译 whl，避免长时间 CUDA 编译、OOM、GitHub 下载中断等问题，整体更适合云服务器稳定部署 starVLA。
>
> 也可直接下载预编译 whl 安装：
> ```bash
> wget -c "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"
> pip install ./flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
> ```

<details>
<summary><b>⚠️ 常见问题</b></summary>

flash-attn 可能难以安装，因为它必须与系统的 CUDA 工具链（nvcc）和 PyTorch 版本匹配。`--no-build-isolation` 标志可以解决大多数问题，但在较新的系统上可能需要手动选择兼容的 flash-attn 版本。确保 CUDA 驱动/工具链和 torch 版本对齐。检查你的环境：

```bash
nvcc -V
pip list | grep -E 'torch|transformers|flash-attn'
```

如果问题仍存在，选择一个与你版本（CUDA 和 torch）匹配的 flash-attn 发行版，或使用 ChatGPT 的搜索功能根据上述输出来帮你诊断。

我们已验证 `flash-attn==2.7.4.post1` 在 nvcc 版本 `12.0` 和 `12.4` 上运行良好。

</details>

<details>
<summary><b>⚠️ flash-attn 2.7.4.post1 相比 2.8.x 的局限（wxwy 备注）</b></summary>

1. 不包含 2.8.x 中新增或强化的部分 Hopper / Blackwell / FP8 / cute kernel 支持。
2. 对 H100、B100/B200、sm90/sm100 等新架构的性能优化不如 2.8.x。
3. 如果后续项目明确依赖 flash-attn 2.8.x 新 API 或新 kernel，需要重新评估升级。
4. 对当前 sm86/Ampere GPU 和 starVLA 训练/推理场景影响较小，稳定性优先级高于新特性。

</details>

---

## 1. 验证安装

运行一个快速的 smoke test 确保框架能正确加载：

```bash
python starVLA/model/framework/VLM4A/QwenGR00T.py
```

这需要将 [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) 放在 `./playground/Pretrained_models/Qwen3-VL-4B-Instruct`（参见[步骤 3](#3-准备预训练模型)）。它应该打印模型架构并在假数据上运行前向传播，不出错。

### 验证 Flash Attention

独立的 `flash-attn` 包对训练吞吐至关重要。运行附带测试脚本检查导入、CUDA kernel 及与 PyTorch 内置 SDPA 的性能对比：

```bash
python docs_zh/test_flash_attn.py
```

此脚本检查三项：
1. `flash_attn` 是否可导入
2. `flash_attn_func` CUDA kernel 是否正确运行
3. 性能基准对比：`flash_attn_func` vs `F.scaled_dot_product_attention`

**样例输出**（GPU: A100 / sm80，torch 2.6.0+cu124，flash_attn 2.7.4.post1）：

```
============================================================
PyTorch 信息
============================================================
torch version: 2.6.0+cu124
cuda available: True
cuda device: P1.gpu.medium
compute capability: (8, 0)

============================================================
FlashAttention 导入检测
============================================================
flash_attn import: SUCCESS
flash_attn version: 2.7.4.post1

============================================================
FlashAttention CUDA Kernel 检测
============================================================
flash_attn_func: SUCCESS
output shape: torch.Size([2, 128, 8, 64])
output dtype: torch.float16

============================================================
FlashAttention 性能对比
============================================================
SDPA backends: {'flash_sdp': True, 'mem_efficient': True, 'math': True}
PyTorch SDPA  : 0.0155 sec (avg 0.309 ms/iter)
flash_attn    : 0.0096 sec (avg 0.192 ms/iter)
加速比        : 1.61x (flash_attn 更快)

============================================================
FlashAttention 安装正常
============================================================
```

关键看**加速比**（> 1.0x 表示 flash_attn 比 PyTorch 默认 SDPA 路径更快）。如果加速比约 1.0x，说明 PyTorch 内部可能已经将 SDPA 调用分发到自身的 flash-sdp 后端。

---

## 2. 准备训练数据

StarVLA 使用 **LeRobot 格式**的数据集。我们提供了一键脚本下载四个 LIBERO 套件（Spatial、Object、Goal、Long-Horizon）以及联合训练 VLM 数据：

```bash
# 设置 DEST 为你想要存放原始数据的路径（可以是共享磁盘）
export DEST=/path/to/your/data/directory
bash examples/LIBERO/data_preparation.sh
```

此脚本将：
1. 从 HuggingFace 下载 4 个 LIBERO 子集（`libero_spatial`、`libero_object`、`libero_goal`、`libero_10`）
2. 下载 VLM 联合训练数据（[LLaVA-OneVision-COCO](https://huggingface.co/datasets/StarVLA/LLaVA-OneVision-COCO)）
3. 在 `playground/Datasets/` 下创建符号链接
4. 将 `modality.json` 复制到每个数据集的 `meta/` 文件夹

<details>
<summary><b>手动下载（替代方案）</b></summary>

```bash
# 逐个下载数据集
huggingface-cli download IPEC-COMMUNITY/libero_spatial_no_noops_1.0.0_lerobot --repo-type dataset --local-dir playground/Datasets/LEROBOT_LIBERO_DATA/libero_spatial_no_noops_1.0.0_lerobot
huggingface-cli download IPEC-COMMUNITY/libero_object_no_noops_1.0.0_lerobot  --repo-type dataset --local-dir playground/Datasets/LEROBOT_LIBERO_DATA/libero_object_no_noops_1.0.0_lerobot
huggingface-cli download IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot    --repo-type dataset --local-dir playground/Datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot
huggingface-cli download IPEC-COMMUNITY/libero_10_no_noops_1.0.0_lerobot      --repo-type dataset --local-dir playground/Datasets/LEROBOT_LIBERO_DATA/libero_10_no_noops_1.0.0_lerobot

# 将 modality.json 复制到每个子集
for d in playground/Datasets/LEROBOT_LIBERO_DATA/*/; do
  cp examples/LIBERO/train_files/modality.json "$d/meta/"
done
```
</details>

此步骤完成后，你的 `playground/Datasets/` 应如下所示：

```
playground/Datasets/
├── LEROBOT_LIBERO_DATA/
│   ├── libero_spatial_no_noops_1.0.0_lerobot/
│   │   ├── meta/
│   │   │   ├── modality.json        ← 必需
│   │   │   └── ...
│   │   └── ...
│   ├── libero_object_no_noops_1.0.0_lerobot/
│   ├── libero_goal_no_noops_1.0.0_lerobot/
│   └── libero_10_no_noops_1.0.0_lerobot/
└── LLaVA-OneVision-COCO/             ← VLM 联合训练数据
```

### 验证你的 Dataloader

确保数据能正确加载：

```bash
python starVLA/dataloader/lerobot_datasets.py \
  --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml
```

---

## 3. 准备预训练模型

将基础 VLM 下载到 `playground/Pretrained_models/`：

```bash
# Qwen3-VL（推荐）
huggingface-cli download Qwen/Qwen3-VL-4B-Instruct --local-dir playground/Pretrained_models/Qwen3-VL-4B-Instruct

# 对于 FAST 框架，使用动作扩展版本：
# huggingface-cli download StarVLA/Qwen3-VL-4B-Instruct-Action --local-dir playground/Pretrained_models/Qwen3-VL-4B-Instruct-Action
```

所有可用的基础模型和微调检查点参见[模型库](model_zoo.md)。

---

## 4. 理解训练配置

配置文件完整路径为 [`examples/LIBERO/train_files/starvla_cotrain_libero.yaml`](../examples/LIBERO/train_files/starvla_cotrain_libero.yaml)。以下按实际 YAML 结构逐段解释每个参数的含义。

---

### 顶层参数

```yaml
run_id: starvla                   # 本次运行的唯一标识，用于命名输出目录
run_root_dir: playground/Checkpoints  # 所有运行的输出根目录
seed: 42                          # 随机种子，保证可复现性
wandb_entity: your_wandb_entity   # Weights & Biases 实体名（团队/用户名）
wandb_project: llavavla           # W&B 项目名
is_debug: false                   # 调试模式：true 时仅加载少量数据快速验证流程
version_id: "0.21"                # 配置 schema 版本号，用于 apply_config_compat 兼容旧格式
```

---

### Framework

```yaml
framework:
  name: QwenGR00T                # 使用哪个 VLA 架构（见下表）
  qwenvl:
    base_vlm: ./playground/Pretrained_models/Qwen3-VL-4B-Instruct  # 预训练 VLM 路径
    attn_implementation: flash_attention_2  # 注意力实现：flash_attention_2（推荐）| sdpa | eager
    vl_hidden_dim: 2048          # VLM 隐藏层维度，作为交叉注意力维度传入动作头
  action_model:
    action_dim: 7                # 动作向量维度。LIBERO 为 7（x, y, z, roll, pitch, yaw, gripper）
    state_dim: 7                 # 本体感知状态维度。通常与 action_dim 一致
    action_horizon: 8            # 动作块总长度（含当前步 + 未来步）
```

> **关于 `action_horizon` 与 `future_action_window_size` 的关系**：VLA 的动作分块（action chunking）预测一个长度为 `action_horizon` 的动作序列，其中第 1 步为当前动作（立即执行），第 2~N 步为未来动作（用于时序平滑）。因此 `future_action_window_size = action_horizon - 1 = 7`。
>
> YAML 中只写了 `action_horizon: 8`，`future_action_window_size` 由 `apply_config_compat()` 自动推导（`share_tools.py:465-477`），无需手动指定。两者同时存在且不一致时以 `action_horizon` 为准。

StarVLA 支持四种框架变体——只需修改 `framework.name`：

| 框架 | 名称 | 描述 |
|-----------|------|-------------|
| StarVLA-OFT | `QwenOFT` | MLP 动作头，简单且快 |
| StarVLA-FAST | `QwenFAST` | 离散动作 token，自回归 |
| StarVLA-π | `QwenPI` | 流匹配扩散动作头 |
| StarVLA-GR00T | `QwenGR00T` | 双系统：VLM (系统 2) + 流匹配 (系统 1) |

---

### Datasets——VLM 联合训练数据

```yaml
datasets:
  vlm_data:
    dataset_py: vlm_datasets     # 使用哪个 dataloader 模块（starVLA/dataloader/vlm_datasets.py）
    dataformat: llava_json       # 数据格式：llava_json 为 LLaVA 格式的 JSON 文件
    dataset_use: sharegpt4v_coco # 使用的 VLM 数据集名称（注册在 dataloader 中）
    eval_dataset: sharegpt4v_coco # 评估用数据集名称
    data_flatten: false          # 是否将图像 token 展平拼接到文本序列中
    base_interval: 2             # VLM 数据采样间隔：每 2 步取一个 VLM batch（与 VLA 交替）
    max_pixels: 307200           # 图像最大像素数（超过会缩放到此值）
    min_pixels: 784              # 图像最小像素数（不足会放大到此值），即 28×28
    model_max_length: 2048       # tokenizer 最大序列长度
    model_type: qwen2.5vl        # VLM 类型，影响 processor 初始化方式
    per_device_batch_size: 4     # 每 GPU 的 VLM 数据 batch size
```

### Datasets——VLA 机器人动作数据

```yaml
  vla_data:
    dataset_py: lerobot_datasets # 使用 LeRobot 格式的 dataloader
    data_root_dir: playground/Datasets/LEROBOT_LIBERO_DATA  # 数据集根目录
    data_mix: libero_all         # 数据集混合名称：libero_all=4 套件全用 | libero_goal=单套件
    action_type: delta_qpos      # 动作类型：delta_qpos=关节增量 | absolute_qpos=绝对位置
    sequential_step_sampling: False  # 是否按时间顺序连续采样帧（False=随机采样）
    CoT_prompt: "Your task is {instruction}. To identify the key objects for your task. Locate their bounding boxes in [x1,y1,x2,y2] format."
                                 # 思维链提示模板，{instruction} 会被替换为任务描述
    per_device_batch_size: 16    # 每 GPU 的 VLA 数据 batch size
    load_all_data_for_training: true  # 是否在训练开始时将全部数据加载到内存（加速训练）
    video_backend: torchvision_av # 视频解码后端：torchvision_av | decord（av1 编码用前者）
```

`data_mix` 的注册定义在 [`examples/LIBERO/train_files/data_registry/data_config.py`](../examples/LIBERO/train_files/data_registry/data_config.py)：

```python
DATASET_NAMED_MIXTURES = {
    "libero_all": [                              # 全部 4 个套件
        ("libero_object_no_noops_1.0.0_lerobot",  1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot",    1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot",  1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot",      1.0, "libero_franka"),
    ],
    "libero_goal": [                             # 单个套件
        ("libero_goal_no_noops_1.0.0_lerobot",    1.0, "libero_franka"),
    ],
}
```

每个元组为 `(数据集目录名, 采样权重, 机器人类型)`。

---

### Trainer

```yaml
trainer:
  max_train_steps: 100000         # 最大训练步数
  num_warmup_steps: 5000          # 学习率 warmup 步数
  save_interval: 5000             # 每 N 步保存一次检查点
  eval_interval: 100              # 每 N 步记录一次评估指标到 W&B
  learning_rate:
    base: 2.5e-05                 # 默认学习率（未单独指定的模块使用此值）
    qwen_vl_interface: 1.0e-05    # VLM 模块学习率（通常较低以保护预训练知识）
    action_model: 1.0e-04         # 动作头学习率（较高以加速新模块收敛）
  lr_scheduler_type: cosine_with_min_lr  # 学习率调度器类型（transformers get_scheduler）
  scheduler_specific_kwargs:
    min_lr: 1.0e-06               # 余弦退火的最小学习率
  freeze_modules: 'qwen_vl_interface'  # 冻结的模块名（逗号分隔），此处冻结 VLM 骨干
  loss_scale:
    vla: 1.0                      # 动作损失权重
    vlm: 0.1                      # VLM 联合训练损失权重（VLM 为辅助任务，权重较低）
  max_grad_norm: 1.0              # 梯度裁剪的最大范数
  weight_decay: 0.0               # 权重衰减（L2 正则化），此处不使用
  logging_frequency: 10           # 每 N 步打印一次训练日志到控制台
  gradient_clipping: 1.0          # 梯度裁剪阈值（与 max_grad_norm 等效）
  gradient_accumulation_steps: 4  # 梯度累积步数（有效 batch = per_device_batch × GPU数 × 此值）
  gradient_checkpointing: true    # 是否开启梯度检查点（省显存，稍慢）

  optimizer:
    name: AdamW                   # 优化器名称
    betas: [0.9, 0.95]            # AdamW 的 β 参数
    eps: 1.0e-08                  # AdamW 的 ε 参数（数值稳定性）
    weight_decay: 1.0e-08         # 优化器内 weight_decay（与上层 weight_decay 独立）
```

---

## 5. 理解训练脚本

训练脚本 [`examples/LIBERO/train_files/run_libero_train.sh`](../examples/LIBERO/train_files/run_libero_train.sh) 封装了 `accelerate launch`。需要自定义的关键变量：

```bash
###########################################################################################
# === 根据你的环境修改以下内容 ===
Framework_name=QwenOFT              # QwenOFT | QwenFAST | QwenPI | QwenGR00T
freeze_module_list=''               # 如 'qwen_vl' 冻结 VLM 骨干
base_vlm=playground/Pretrained_models/Qwen3-VL-4B-Instruct
config_yaml=./examples/LIBERO/train_files/starvla_cotrain_libero.yaml
libero_data_root=playground/Datasets/LEROBOT_LIBERO_DATA
data_mix=libero_all                 # 或 libero_goal 单套件
run_root_dir=./results/Checkpoints
run_id=my_first_libero_run          # 唯一实验名称
###########################################################################################
```

脚本使用 DeepSpeed ZeRO-2 启动分布式训练：

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 8 \                    # GPU 数量
  starVLA/training/train_starvla.py \
  --config_yaml ${config_yaml} \
  --framework.name ${Framework_name} \
  ...
```

> **注意：** 命令行参数会覆盖 YAML 配置值。这让你可以保留一份基础配置，按实验变化参数。

---

## 6. 启动训练

```bash
# 从仓库根目录
bash examples/LIBERO/train_files/run_libero_train.sh
```

### 预期效果

- 检查点保存至 `results/Checkpoints/{run_id}/checkpoints/`
- 训练日志发送至 W&B（设置 `WANDB_MODE=disabled` 跳过）
- 脚本将自身复制到输出目录以保证可复现性
- 使用 8× A100/H800 GPU，在 `libero_all` 上训练大约需要 30K 步（约 10 个 epoch）

### 调整 GPU 数量

在脚本中编辑 `--num_processes` 以及 `starVLA/config/deepseeds/deepspeed_zero2.yaml` 中的 `num_processes`，使之匹配你可用的 GPU。

---

## 7. 评估预训练检查点

评估采用**客户端-服务器架构**：策略服务器（在 `starVLA` 环境中）提供模型推理，仿真客户端（在独立的 `LIBERO` 环境中）发送观测并接收动作。

### 步骤 0：设置 LIBERO 环境

为 LIBERO 仿真器创建独立的 conda 环境：

```bash
conda create -n libero python=3.10 -y
conda activate libero
pip install mujoco==3.2.3

# Clone 并安装 LIBERO
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO && pip install -e . && cd ..

# 安装评估依赖
pip install tyro matplotlib mediapy websockets msgpack numpy==1.24.4
```

或使用提供的脚本：

```bash
bash examples/LIBERO/eval_files/install_libero.sh
```

### 步骤 1：下载检查点

从 [🤗 StarVLA/bench-libero](https://huggingface.co/collections/StarVLA/bench-libero) 下载预训练检查点：

```bash
huggingface-cli download StarVLA/Qwen3-VL-OFT-LIBERO-4in1 \
  --local-dir playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1
```

### 步骤 2：启动策略服务器（终端 1 — starVLA 环境）

编辑 [`examples/LIBERO/eval_files/run_policy_server.sh`](../examples/LIBERO/eval_files/run_policy_server.sh) 中的检查点路径：

```bash
CKPT=playground/Pretrained_models/StarVLA/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
```

然后启动：

```bash
conda activate starVLA
bash examples/LIBERO/eval_files/run_policy_server.sh
```

等待直到看到 `server listening on 0.0.0.0:6694`。

### 步骤 3：运行评估（终端 2 — LIBERO 环境）

编辑 [`examples/LIBERO/eval_files/eval_libero.sh`](../examples/LIBERO/eval_files/eval_libero.sh) 中的路径，然后：

```bash
conda activate libero
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
bash examples/LIBERO/eval_files/eval_libero.sh
```

### 预期效果

- 每个任务默认运行 50 轮
- 视频保存在 `results/{task_suite}/{checkpoint_name}/` 下
- 成功率在最后打印出来

> **提示：** 要评估你自己训练的检查点，将 `CKPT` 指向 `results/Checkpoints/{run_id}/checkpoints/steps_XXXXX_pytorch_model.pt` 即可。

---

## 后续步骤

- **尝试不同框架：** 将训练脚本中的 `Framework_name` 改为 `QwenGR00T`、`QwenPI` 或 `QwenFAST`
- **探索其他基准：** 查看 [SimplerEnv](../examples/SimplerEnv/)、[RoboTwin](../examples/Robotwin/)、[Calvin](../examples/calvin/) 或 [Behavior-1K](../examples/Behavior/)
- **World-Model-for-Action：** 使用视频生成模型（Cosmos、Wan）作为动作骨干 — 参见 [WM4A](WM4A.md)
- **VLM 联合训练：** 了解多目标训练的工作原理 — 参见 [CoTrainVLM](../examples/CoTrainVLM/)
- **部署到真实机器人：** 参见 [Franka 示例](../examples/Franka/) 的真实世界部署
- **RL 后训练：** 查看 [StarVLA × RLinf](https://rlinf.readthedocs.io/en/latest/rst_source/examples/embodied/starvla.html)
