# StarVLA LIBERO 训练代码走读

> 通用代码架构解析文档。以 `run_libero_train.sh` + `QwenGR00T` 框架为例，逐层拆解从启动命令到训练循环的完整调用链。
> 配套 `training_log_1229_libero4in1_qwen3oft.md`（实验记录），本文档聚焦代码逻辑与架构。

---

## 一、整体调用链（TL;DR）

```
run_libero_train.sh
  └─ accelerate launch
       └─ starVLA/training/train_starvla.py  (main入口)
            ├─ 1. OmegaConf 加载 YAML + CLI 覆盖 → cfg
            ├─ 2. build_framework(cfg) → Qwen_GR00T 模型
            ├─ 3. build_dataloader(cfg) → LeRobotMixtureDataset → DataLoader
            ├─ 4. setup_optimizer_and_scheduler(cfg) → AdamW + cosine scheduler
            ├─ 5. VLATrainer.__init__()
            ├─ 6. VLATrainer.prepare_training()
            └─ 7. VLATrainer.train()
                 └─ while step < max_steps:
                      ├─ _get_next_batch()    → 原始 dict 列表
                      ├─ _train_step(batch)   → model.forward() → loss.backward()
                      ├─ _log_metrics()        → wandb + logger
                      └─ _save_checkpoint()    → 按间隔保存
```

---

## 二、启动命令与配置加载

### 2.1 shell 脚本 → accelerate launch

`run_libero_train.sh` 做三件事：

1. **环境变量设置**（第1-26行）：CUDA 路径、NCCL 网络接口、超时时间等
2. **用户配置变量**（第27-47行）：框架名、模型路径、batch size、数据路径等
3. **accelerate launch 命令**（第66-97行）

关键命令结构：

```bash
accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \  # DeepSpeed ZeRO-2 配置
  --num_processes 2 \                                            # GPU 数量
  --gradient_accumulation_steps 2 \                              # 梯度累积
  starVLA/training/train_starvla.py \                            # 训练入口脚本
  --config_yaml examples/LIBERO/train_files/starvla_cotrain_libero.yaml \  # YAML 配置文件
  --framework.name QwenGR00T \                                   # 框架选择（CLI 覆盖 YAML）
  --framework.qwenvl.base_vlm playground/Pretrained_models/Qwen3-VL-4B-Instruct \
  ...                                                            # 更多 CLI 覆盖
```

**CLI 优先级 > YAML**：`--xxx.yyy=value` 格式的参数会通过 `OmegaConf.merge()` 覆盖 YAML 中的对应值。

### 2.2 YAML 配置结构

`starvla_cotrain_libero.yaml` 的顶层结构：

```yaml
framework:        # 模型架构配置
  name: QwenGR00T
  qwenvl: {...}   # VLM 骨干参数
  action_model: {...}  # 动作头参数

datasets:         # 数据配置
  vla_data: {...} # 机器人动作数据（LIBERO）
  vlm_data: {...} # 通用图文数据（COCO，联合训练用）

trainer:          # 训练超参数
  max_train_steps, learning_rate, freeze_modules, ...
```

### 2.3 main() 函数配置加载流程

`train_starvla.py:1583-1613`：

```python
# 1. 解析命令行参数
args, clipargs = parser.parse_known_args()

# 2. 加载 YAML 配置
cfg = OmegaConf.load(args.config_yaml)

# 3. CLI 参数覆盖（--xxx.yyy=zzz → OmegaConf.from_dotlist → merge）
dotlist = normalize_dotlist_args(clipargs)
cli_cfg = OmegaConf.from_dotlist(dotlist)
cfg = OmegaConf.merge(cfg, cli_cfg)

# 4. 配置兼容性规范化（旧字段 → 新字段映射）
cfg = apply_config_compat(cfg)
```

---

## 三、数据集构建与数据流

### 3.1 调用链

```
prepare_data() [train_starvla.py:648]
  └─ build_dataloader(cfg, dataset_py="lerobot_datasets") [dataloader/__init__.py:36]
       └─ get_vla_dataset(data_cfg) [dataloader/lerobot_datasets.py:60]
            ├─ DATASET_NAMED_MIXTURES["libero_all"] → 数据集名/权重/机器人类型列表
            └─ for each dataset:
                 └─ make_LeRobotSingleDataset()
                      └─ LeRobotSingleDataset(parquet文件, 模态配置, 视频后端)
            └─ LeRobotMixtureDataset(dataset_mixture, mode="train")
       └─ DataLoader(dataset, batch_size=8, collate_fn=collate_fn)
```

### 3.2 数据格式

Dataloader 不做任何模型相关预处理，返回**原始 Python 对象列表**：

```python
# collate_fn 直接返回 list，不做 stack/tensor 化
def collate_fn(batch):
    return batch

# 每个 batch 是 List[dict]，每个 dict 为：
{
    "image":  [PIL.Image, PIL.Image, ...],  # 多视角图像（LIBERO 为单视角）
    "lang":   "put the bowl on the plate",  # 任务指令字符串
    "action": np.ndarray [T, 7],            # 动作序列 (xyz + rpy + gripper)
    "state":  np.ndarray [8],               # 本体感知状态（关节角度等）
}
```

**设计意图**：dataloader 与模型完全解耦。换框架只需换 `forward()` 实现，dataloader 不用动。

### 3.3 LIBERO 数据统计

| 指标 | 值 |
|------|-----|
| 数据集名 | `libero_all`（4合1：Goal + Spatial + Object + Long） |
| 总 episode | 1,693 |
| 总 transition | 273,465 |
| action 维度 | 7 (`xyz + rpy + gripper`) |
| action 类型 | `delta_qpos`（相对位移） |
| 单样本 action 帧数 | 8（`action_horizon`，包含当前帧+未来7帧） |

### 3.4 底层数据读取

`LeRobotSingleDataset` 基于 LeRobot v2.0 格式：
- 元数据：`meta/modality.json`, `meta/episodes.jsonl`, `meta/tasks.jsonl`
- 数据：`data/*/*.parquet`（按 chunk 存储的 Parquet 文件）
- 视频帧通过 `video_backend`（默认 `torchvision_av`）从视频文件中按时间戳读取

---

## 四、模型架构初始化

### 4.1 build_framework() → Qwen_GR00T

`base_framework.py:55-76`：

```python
def build_framework(cfg):
    # 1. 自动扫描 starVLA/model/framework/ 下所有框架并注册
    _auto_import_framework_modules()

    # 2. 从注册表查找框架类
    framework_id = cfg.framework.name  # "QwenGR00T"
    model_class = FRAMEWORK_REGISTRY[framework_id]

    # 3. 实例化
    return model_class(cfg)
```

### 4.2 Qwen_GR00T.__init__()

`VLM4A/QwenGR00T.py:138-165`：

```python
class Qwen_GR00T(baseframework):
    def __init__(self, config):
        # 1. 合并默认配置与 YAML 配置
        self.config = merge_framework_config(QwenGR00TDefaultConfig, config)

        # 2. 构建 VLM 骨干
        self.qwen_vl_interface = get_vlm_model(config=self.config)
        #    └─ 根据 base_vlm 路径中的 "Qwen3-VL" 关键字
        #       选择 _QWen3_VL_Interface (QWen3.py)
        #       └─ 内部加载 Qwen3VLForConditionalGeneration + AutoProcessor

        # 3. 对齐 cross_attention_dim
        self.config.framework.action_model.diffusion_model_cfg.cross_attention_dim = \
            self.qwen_vl_interface.model.config.hidden_size  # Qwen3-VL-4B: 2560

        # 4. 构建动作头
        self.action_model = get_action_model(config=self.config)
        #    └─ FlowmatchingActionHead
        #       ├─ DiT-B (16层交叉注意力 Transformer)
        #       ├─ ActionEncoder (7→768, 含 timestep 调制)
        #       ├─ ActionDecoder (MLP 1024→1024→7)
        #       ├─ StateEncoder (MLP 7→1024→768)
        #       ├─ future_tokens (32个 learnable query)
        #       └─ position_embedding
```

### 4.3 VLM 接口层：_QWen3_VL_Interface

`modules/vlm/QWen3.py`：

```python
class _QWen3_VL_Interface(nn.Module):
    def __init__(self, config):
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            base_vlm, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        self.processor = AutoProcessor.from_pretrained(base_vlm)

    def build_qwenvl_inputs(self, images, instructions):
        # 构建 Qwen3-VL 标准消息格式
        messages = []
        for imgs, instruction in zip(images, instructions):
            content = [{"type": "image", "image": img} for img in imgs]
            content.append({"type": "text", "text": instruction})
            messages.append([{"role": "user", "content": content}])

        # 通过 processor 做 tokenize + image processing + padding
        batch_inputs = self.processor.apply_chat_template(
            messages, tokenize=True, padding=True, return_dict=True, return_tensors="pt"
        )
        return batch_inputs.to(self.model.device)
```

### 4.4 动作头：FlowmatchingActionHead

`modules/action_model/GR00T_ActionHeader.py:196-363`：

**训练 forward 流程**：

```python
def forward(self, vl_embs, actions, state=None):
    # vl_embs: [B, seq_len, 2560] -- VLM 输出的最后一层 hidden states
    # actions: [B*R, 8, 7]       -- repeated_diffusion_steps 次重复

    # 1. 采样噪声和时间步
    noise = torch.randn(actions.shape)
    t = sample_time(B)  # Beta(1.5, 1.0) 分布，上界 s=0.999

    # 2. 流匹配插值：noisy_traj = (1-t)*noise + t*action
    noisy_trajectory = (1 - t) * noise + t * actions
    velocity = actions - noise  # 目标：预测速度场

    # 3. 编码含噪动作
    action_features = action_encoder(noisy_trajectory, t_discretized)

    # 4. 拼装 DiT 输入序列：[state | future_tokens | action_features]
    sa_embs = cat(state_features, future_tokens, action_features)

    # 5. DiT 交叉注意力去噪（vl_embs 作为 cross-attn 的 KV）
    model_output = self.model(sa_embs, encoder_hidden_states=vl_embs, timestep=t)

    # 6. 解码 → 预测速度场 → MSE loss
    pred_actions = action_decoder(model_output[:, -8:, :])
    loss = MSE(pred_actions, velocity)
    return loss
```

**推理 predict_action 流程**：

```python
def predict_action(self, vl_embs, state=None):
    actions = torch.randn(B, 8, 7)  # 从纯噪声开始

    for step in range(num_inference_steps):  # 默认 4 步 Euler ODE
        t = step / num_steps
        velocity = self.model(sa_embs, vl_embs, t)  # 预测速度场
        actions = actions + velocity * dt           # Euler 积分

    return actions
```

**DiT-B 结构**（16层交叉注意力 Transformer）：

```
DiT-B
├── TimestepEncoder: Timesteps(256) + Embedding(256→768)
├── TransformerBlocks × 16（interleaved 模式）
│   ├── [0]  CrossAttn: Q(768), KV(2560) + FF(768→3072→768) + AdaLN
│   ├── [1]  SelfAttn:  QKV(768)          + FF(768→3072→768) + AdaLN
│   ├── [2]  CrossAttn: ...
│   ├── ...
│   └── [15] SelfAttn:  ...
├── norm_out: LayerNorm(768)
└── proj_out: 768→1024（送入 action_decoder）
```

**精度分离**：
- VLM forward 使用 `bfloat16`（省显存、速度快）
- 动作头 forward 使用 `float32`（流匹配数值敏感，需高精度）

---

## 五、优化器与学习率

### 5.1 分模块学习率

`trainer_utils/trainer_tools.py:94-150`：

```python
def build_param_lr_groups(model, cfg):
    # cfg.trainer.learning_rate = {
    #     "base": 2.5e-05,
    #     "qwen_vl_interface": 1.0e-05,
    #     "action_model": 1.0e-04,
    # }

    for module_name, lr in lr_cfg.items():
        if module_name == "base": continue
        module = getattr(model, module_name)  # 通过点号路径查找子模块
        params = [p for p in module.parameters() if not frozen]
        param_groups.append({"params": params, "lr": lr, "name": module_name})

    # 剩余未分配的参数用 base_lr
    other_params = [p for p in model.parameters() if not assigned and not frozen]
    param_groups.append({"params": other_params, "lr": base_lr, "name": "base"})
```

**实际效果**（本实验 `freeze_modules=qwen_vl_interface`）：
- `qwen_vl_interface`：被冻结，LR 不生效
- `action_model`：LR = 1.0e-04（动作头从头训练，用更高 LR）
- 其余（base）：LR = 2.5e-05

### 5.2 学习率调度

```python
lr_scheduler = get_scheduler(
    name="cosine_with_min_lr",
    optimizer=optimizer,
    num_warmup_steps=5000,       # 0→5000 线性 warmup
    num_training_steps=80000,    # 5000→80000 余弦衰减
    scheduler_specific_kwargs={"min_lr": 1.0e-06},
)
```

---

## 六、训练循环详解

### 6.1 VLATrainer 初始化

`train_starvla.py:687-729`：

```python
class VLATrainer(TrainerUtils):
    def __init__(self, cfg, model, dataloader, optimizer, scheduler, accelerator):
        self.completed_steps = 0
        self.total_batch_size = per_device_bsz × num_processes × grad_accum_steps
        # 示例：8 × 2 × 2 = 32（有效全局 batch）
```

### 6.2 prepare_training()

`train_starvla.py:730-767`：

```python
def prepare_training(self):
    # 1. 等待启动阶段 checkpoint 后台同步完成
    _wait_for_startup_checkpoint_stage()

    # 2. 设置本地/网络盘双写 checkpoint 存储
    self._setup_checkpoint_storage()

    # 3. 保存 config.full.yaml 和 config.yaml
    self._save_initial_configs()

    # 4. 初始化 checkpoint 目录、处理 resume
    self._init_checkpointing()
    #    ├─ is_resume=True → 查找最新 checkpoint → 加载权重
    #    ├─ pretrained_checkpoint → 加载预训练权重
    #    └─ 否则 → 从头训练

    # 5. 追平 LR scheduler（resume 场景）
    self._adjust_lr_scheduler_for_resume()

    # 6. 冻结指定模块
    self.model = self.freeze_backbones(self.model, freeze_modules="qwen_vl_interface")

    # 7. Accelerate 分布式包装（DeepSpeed + DDP）
    self.model, self.optimizer, self.dataloader = accelerator.prepare(
        self.model, self.optimizer, self.dataloader
    )

    # 8. 加载 optimizer/scheduler 状态（lightweight / DeepSpeed 模式）
    if resume:
        self._load_checkpoint() or self._load_lightweight_training_state()
```

### 6.3 train() 主循环

`train_starvla.py:1115-1164`：

```python
def train(self):
    self._create_data_iterators()
    progress_bar = tqdm(total=max_train_steps, initial=completed_steps)

    while self.completed_steps < max_train_steps:
        # ─── 1. 取数据 ───
        batch_vla = self._get_next_batch()
        #    └─ next(vla_iter)，遇到 StopIteration 自动重置 dataloader

        # ─── 2. 单步训练 ───
        step_metrics = self._train_step(batch_vla)

        # ─── 3. 更新进度条 ───
        if sync_gradients:
            progress_bar.update(1)
            self.completed_steps += 1

        # ─── 4. 定期评估 ───
        if step % eval_interval == 0:
            step_metrics = self.eval_action_model(step_metrics)

        # ─── 5. 日志记录 ───
        step_metrics["timing/data"] = data_time
        step_metrics["timing/model"] = model_time
        if sync_gradients:
            self._log_metrics(step_metrics)  # → wandb.log()

        # ─── 6. 定期保存 checkpoint ───
        if step % save_interval == 0 and step > 0:
            self._save_checkpoint()

    self._finalize_training()  # 保存最终模型
```

### 6.4 _train_step() 单步训练

`train_starvla.py:1195-1221`：

```python
def _train_step(self, batch_vla):
    with self.accelerator.accumulate(self.model):  # 梯度累积上下文
        self.optimizer.zero_grad()

        # ─── VLM 编码（bfloat16）───
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output_dict = self.model.forward(batch_vla)
            #   内部流程：
            #   1. qwen_vl_interface.build_qwenvl_inputs(images, instructions)
            #   2. qwen_vl_interface(**inputs) → hidden_states[-1] [B, L, 2560]
            #   3. action_model(last_hidden, actions) → action_loss

        action_loss = output_dict["action_loss"]
        total_loss = action_loss  # VLA-only 训练

        # ─── 反向传播 ───
        self.accelerator.backward(total_loss)

        # ─── 梯度裁剪 ───
        if gradient_clipping is not None:
            self.accelerator.clip_grad_norm_(model.parameters(), gradient_clipping)

        # ─── 参数更新 ───
        self.optimizer.step()
        if self.accelerator.sync_gradients:  # 只在真正同步梯度时更新 LR
            self.lr_scheduler.step()

    return {"action_dit_loss": action_loss.item()}
```

### 6.5 model.forward() 内部数据流

`QwenGR00T.py:167-215`（完整注释版）：

```python
def forward(self, examples):
    # ── 步骤1：从原始 dict 提取数据 ──
    batch_images = [example["image"] for example in examples]    # List[List[PIL.Image]]
    instructions = [example["lang"] for example in examples]     # List[str]
    actions = [example["action"] for example in examples]        # List[np.ndarray]
    state = [example["state"] for example in examples]           # List[np.ndarray]

    # ── 步骤2：VLM 编码（bfloat16 精度）──
    with torch.autocast("cuda", dtype=torch.bfloat16):
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images, instructions=instructions
        )
        # qwen_inputs = {"input_ids": [B, L], "pixel_values": [...], ...}

        qwenvl_outputs = self.qwen_vl_interface(
            **qwen_inputs,
            output_hidden_states=True,    # 需要拿到 hidden states
        )
        last_hidden = qwenvl_outputs.hidden_states[-1]  # [B, L, 2560]

    # ── 步骤3：动作头计算（float32 精度）──
    with torch.autocast("cuda", dtype=torch.float32):
        actions = torch.tensor(np.array(actions))                # [B, T, 7]
        actions_target = actions[:, -action_horizon:, :]         # [B, 8, 7]

        # 重复扩散步：每个样本采样 multiple 次噪声 → 增大有效 batch
        actions_target_repeated = actions_target.repeat(R, 1, 1) # [B*R, 8, 7]
        last_hidden_repeated = last_hidden.repeat(R, 1, 1)       # [B*R, L, 2560]

        action_loss = self.action_model(
            last_hidden_repeated, actions_target_repeated, state_repeated
        )
        # action_loss 内部：流匹配 → 预测速度场 → MSE loss

    return {"action_loss": action_loss}
```

### 6.6 repeated_diffusion_steps 的作用

一个重要的训练技巧：每个 batch 的动作重复 `R=8` 次，每次对同一 action 序列采样不同的噪声和时间步。这使得：
- 有效 DiT batch size = `B × R`
- 更多样的 (noise, t) 对 → 更稳定的流匹配训练
- 不增加 dataloader 负担（VLM hidden states 只需算一次，内存复用）

### 6.7 DiT 内部数据流（cross_attention_dit.py）

`FlowmatchingActionHead` 调用 `self.model(...)` 后，数据进入 DiT（`cross_attention_dit.py`）。这里拆解从输入到输出的完整 tensor 变化。

#### 输入拼装（FlowmatchingActionHead.forward 中）

```python
# sa_embs = [state_features, future_tokens, action_features] 沿 seq 维 cat
sa_embs = [B, 1, 768] + [B, 32, 768] + [B, 8, 768]  # = [B, 41, 768]
vl_embs = [B, S_vl, 2560]   # VLM 最后一层 hidden states
t_discretized = [B,]        # 已离散化到 0~999
```

#### DiT.forward() 逐层数据流

```python
def forward(self, hidden_states, encoder_hidden_states, timestep, ...):
    # hidden_states: [B, 41, 768]   (sa_embs)
    # encoder_hidden_states: [B, S_vl, 2560]  (vl_embs)
```

**步骤 1：时间步编码**

```python
temb = self.timestep_encoder(timestep)   # [B, 768]
# 内部：Timesteps(256) + TimestepEmbedding(256 → 768)
```

**步骤 2：交错 Transformer 块（16 层）**

`interleave_self_attention=True` 时，偶数层做 Cross-Attention，奇数层做 Self-Attention：

```
Block [0]  CrossAttn: Q(768) + KV(2560)   ← VLM 语义注入
Block [1]  SelfAttn:  QKV(768)            ← 动作序列内部交互
Block [2]  CrossAttn: Q(768) + KV(2560)
Block [3]  SelfAttn:  QKV(768)
    ...
Block [14] CrossAttn: Q(768) + KV(2560)
Block [15] SelfAttn:  QKV(768)
```

每层 `BasicTransformerBlock.forward` 内部：

```python
# 1. AdaLayerNorm（时间步条件归一化）
norm_hidden = AdaLayerNorm(hidden_states, temb)
#    temb → Linear → (scale, shift)
#    hidden = LayerNorm(hidden) * (1 + scale) + shift
#    shape 不变: [B, 41, 768]

# 2. Attention
attn_output = attn1(norm_hidden, encoder_hidden_states=...)
#    偶数层: encoder_hidden_states=[B,S_vl,2560] → Cross-Attention
#            Q: [B,41,768], K/V: [B,S_vl,2560]（通过 cross_attention_dim 映射）
#            输出: [B, 41, 768]
#    奇数层: encoder_hidden_states=None → Self-Attention
#            Q/K/V 都来自 norm_hidden [B, 41, 768]
#            输出: [B, 41, 768]

# 残差连接
hidden_states = attn_output + hidden_states   # [B, 41, 768]

# 3. FeedForward (GeGLU, 768→3072→768)
ff_output = ff(norm3(hidden_states))          # [B, 41, 768]
hidden_states = ff_output + hidden_states     # [B, 41, 768]
```

**步骤 3：输出调制**

```python
conditioning = temb                                    # [B, 768]
shift, scale = proj_out_1(SiLU(conditioning)).chunk(2) # [B, 768], [B, 768]

# AdaLN 输出调制
hidden_states = norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
# 结果: [B, 41, 768]

# 投影到输出维度
output = proj_out_2(hidden_states)  # Linear(768 → 1024)
# 结果: [B, 41, 1024]
```

#### 回到 FlowmatchingActionHead

```python
model_output = self.model(sa_embs, vl_embs, t)   # [B, 41, 1024]
pred = self.action_decoder(model_output)         # MLP: [B, 41, 1024] → [B, 41, 7]

# 只取最后 action_horizon=8 个位置（对应 action_features 的位置）
pred_actions = pred[:, -8:, :]                   # [B, 8, 7]

# loss = MSE(pred_actions, velocity)  其中 velocity = actions - noise
```

#### 为什么 Cross-Attn / Self-Attn 是交错而非连续？

| 层类型 | 作用 |
|--------|------|
| **Cross-Attention** | 把 VLM 语义特征（看懂场景+指令）注入到动作序列中 |
| **Self-Attention** | 让动作序列内部各位置（state / future / action）互相交流 |

交错模式的好处：语义注入和内部交互交替进行，每注入一次语义就允许动作序列内部重新分配注意力，避免连续多层 cross-attn 导致动作序列同质化。

#### future_tokens 的作用

32 个可学习的 `future_tokens` 插在 state 和 action 之前，在 Self-Attention 层中它们会和 action 位置互相 attend。这相当于给模型提供了**专用的规划槽位**，让模型先在 future_tokens 上做"预规划"，再把规划结果传递到具体的 action 位置。这是连续动作预测中的一个重要归纳偏置。

#### 完整形状变化总结

```
输入:
  sa_embs:          [B, 41, 768]   (state=1 + future=32 + action=8)
  vl_embs:          [B, S_vl, 2560]
  timestep:         [B,]

DiT 内部:
  temb:             [B, 768]
  Block 0 (Cross):  [B, 41, 768]  ← 引入 [B, S_vl, 2560]
  Block 1 (Self):   [B, 41, 768]
  ...
  Block 15 (Self):  [B, 41, 768]
  output:           [B, 41, 1024]

FlowmatchingActionHead 后续:
  pred[:, -8:]:     [B, 8, 7]     ← 只取 action 位置
  loss: MSE(pred_velocity, action - noise)
```

---

## 七、Checkpoint 保存与恢复

### 7.1 三类 checkpoint 格式

| 格式 | 条件 | 文件形态 | 恢复方式 |
|------|------|---------|---------|
| **单文件** | `save_checkpoint_as_directory=False` | `steps_N_model.safetensors` | `load_state_dict()` |
| **轻量目录式** | `save_checkpoint_as_directory=True`（默认） | 目录含模型分片 + `scheduler.pt` + `optimizer_rank_*.pt` + `trainer_state.json` | `_load_lightweight_training_state()` |
| **DeepSpeed 完整态** | `save_with_training_state=True` | 目录含 DeepSpeed 引擎状态 | `_load_checkpoint()` (DeepSpeed 加载) |

### 7.2 轻量目录式 checkpoint 结构（推荐/默认）

```
checkpoints/steps_1000/
├── model-00001.safetensors      # 模型权重分片 1
├── model-00002.safetensors      # 模型权重分片 2
├── model.safetensors.index.json # 分片索引（weight_map）
├── scheduler.pt                 # LR scheduler 状态
├── optimizer_rank_00000.pt      # rank 0 optimizer 状态
├── optimizer_rank_00001.pt      # rank 1 optimizer 状态
└── trainer_state.json           # 训练元信息
    {
      "completed_steps": 1000,
      "save_format": "safetensors",
      "checkpoint_type": "lightweight_training",
      "optimizer_format": "rank_sharded",
      "optimizer_world_size": 2
    }
```

### 7.3 本地/网络盘双写机制

当 `enable_local_checkpoint_staging=True` 时：

```
训练时写入：本地 NVMe 高速盘 (/tmp/nvme/.../checkpoints/)
后台同步：  独立 Python 进程异步复制到网络盘 (output_dir/checkpoints/)
保留策略：  local_checkpoint_keep_count=2（本地仅保留最近 2 个完整 checkpoint）
```

启动时（`_setup_checkpoint_storage`）：
1. 比较本地和网络盘的最新 checkpoint
2. 本地更新 → 后台复制到网络盘
3. 本地落后 → 从网络盘引导
4. 网络盘无 checkpoint → 本地开始全新训练

### 7.4 resume 流程

`_init_checkpointing()` 中：

```python
if is_resume:
    # 1. 扫描 checkpoint_dir，找到最大 step 的完整 checkpoint
    resume_path, completed_steps = self._get_latest_checkpoint(checkpoint_dir)

    # 2. 根据 checkpoint 类型选择恢复策略
    if is_deepspeed_dir:
        self.resume_requires_training_state = True    # 后续 load_state
    elif is_lightweight_dir:
        self.resume_requires_lightweight_state = True # 后续 _load_lightweight
        self.model = self.load_pretrained_backbones(model, resume_path)
    else:
        self.model = self.load_pretrained_backbones(model, resume_path)
```

**重要**：`--trainer.is_resume True` 必须在 accelerate 命令的续行符链中，否则不会被传递给 Python 脚本。

---

## 八、关键代码路径速查表

| 功能 | 文件 | 关键函数/类 |
|------|------|-----------|
| 训练入口 | `starVLA/training/train_starvla.py` | `main()`, `VLATrainer` |
| 框架注册 | `starVLA/model/framework/base_framework.py` | `build_framework()`, `FRAMEWORK_REGISTRY` |
| QwenGR00T 框架 | `starVLA/model/framework/VLM4A/QwenGR00T.py` | `Qwen_GR00T.forward()`, `predict_action()` |
| VLM 接口 (Qwen3) | `starVLA/model/modules/vlm/QWen3.py` | `_QWen3_VL_Interface.build_qwenvl_inputs()` |
| 流匹配动作头 | `starVLA/model/modules/action_model/GR00T_ActionHeader.py` | `FlowmatchingActionHead.forward()`, `predict_action()` |
| DiT Transformer | `starVLA/model/modules/action_model/flow_matching_head/cross_attention_dit.py` | `DiT` |
| 数据加载 | `starVLA/dataloader/lerobot_datasets.py` | `get_vla_dataset()`, `collate_fn` |
| 底层数据集 | `starVLA/dataloader/gr00t_lerobot/datasets.py` | `LeRobotSingleDataset`, `LeRobotMixtureDataset` |
| 优化器/调度器 | `starVLA/training/trainer_utils/trainer_tools.py` | `build_param_lr_groups()`, `setup_optimizer_and_scheduler()` |
| Checkpoint 保存 | `train_starvla.py:1050-1083` | `VLATrainer._save_checkpoint()` |
| Checkpoint 恢复 | `train_starvla.py:811-866` | `VLATrainer._init_checkpointing()` |
| 本地/网络双写 | `train_starvla.py:502-619` | `_stage_local_checkpoint_storage()`, `_launch_startup_checkpoint_stage()` |
| DeepSpeed 配置 | `starVLA/config/deepseeds/deepspeed_zero2.yaml` | ZeRO-2 配置 |
| 训练 shell 脚本 | `examples/LIBERO/train_files/run_libero_train.sh` | — |
| 训练 YAML 配置 | `examples/LIBERO/train_files/starvla_cotrain_libero.yaml` | — |

---

## 九、从零到一的完整数据流示意

```
┌─────────────────────────────────────────────────────────────────────────┐
│  1. CLI / YAML 配置                                                      │
│     run_libero_train.sh → accelerate launch → train_starvla.py main()    │
│     OmegaConf.merge(YAML, CLI) → cfg                                     │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  2. 模型构建                                                              │
│     build_framework(cfg)                                                  │
│     ├─ _QWen3_VL_Interface: Qwen3VLForConditionalGeneration (冻结, bf16) │
│     └─ FlowmatchingActionHead:                                           │
│         ├─ DiT-B (16层, cross_attn_dim=2560, 可训练, fp32)              │
│         ├─ ActionEncoder / ActionDecoder / StateEncoder                   │
│         └─ future_tokens (32 learnable queries)                          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  3. 数据加载                                                              │
│     LeRobotMixtureDataset (libero_all: 1693 episodes, 273K transitions)  │
│     → DataLoader(batch_size=8, collate_fn=lambda x: x)                   │
│     → 每个 batch: List[dict], 原始 PIL.Image + str + np.ndarray          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  4. 训练循环 (while step < 80000)                                        │
│                                                                          │
│     _train_step(batch):                                                  │
│     ┌──────────────────────────────────────────────────────────────────┐ │
│     │ a. VLM 编码 (bf16)                                                │ │
│     │    images + instructions                                          │ │
│     │    → build_qwenvl_inputs() → tokenize + image processing          │ │
│     │    → Qwen3VLForConditionalGeneration(**inputs)                    │ │
│     │    → hidden_states[-1]  [B, L, 2560]                             │ │
│     ├──────────────────────────────────────────────────────────────────┤ │
│     │ b. 动作头 (fp32)                                                  │ │
│     │    hidden × R, action × R  (R=8 repeated_diffusion_steps)        │ │
│     │    → sample_time(Beta(1.5,1.0)) → t, noise                       │ │
│     │    → noisy_traj = (1-t)×noise + t×action                         │ │
│     │    → ActionEncoder(noisy_traj, t) → action_features               │ │
│     │    → cat(state, future_tokens, action_features)                   │ │
│     │    → DiT(sa_embs, cross_attn_kv=hidden, timestep=t)              │ │
│     │    → ActionDecoder → pred_velocity                                │ │
│     │    → loss = MSE(pred_velocity, action - noise)                   │ │
│     ├──────────────────────────────────────────────────────────────────┤ │
│     │ c. 反向传播                                                       │ │
│     │    accelerator.backward(loss)                                     │ │
│     │    → clip_grad_norm_(max_norm=1.0)                                │ │
│     │    → optimizer.step()                                             │ │
│     │    → lr_scheduler.step()                                          │ │
│     └──────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│     每 100 步 → eval_action_model() → MSE score                          │
│     每 100 步 → wandb.log(loss, lr, epoch, timing)                       │
│     每 500 步 → _save_checkpoint() → 本地 NVMe → 后台同步网络盘           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 十、常见配置调整指南

| 需求 | 修改位置 | 参数 |
|------|---------|------|
| 换框架 | shell 脚本 | `Framework_name=QwenFast`（或 QwenPI/QwenOFT） |
| 换 VLM | shell 脚本或 YAML | `--framework.qwenvl.base_vlm` |
| 调整有效 batch | shell 脚本 | `per_device_batch_size` × `num_processes` × `gradient_accumulation_steps` |
| 冻结/解冻模块 | shell 脚本 | `freeze_module_list`（逗号分隔） |
| 调整动作头学习率 | YAML `trainer.learning_rate` | `action_model: 1.0e-04` |
| 调整 warmup | YAML | `trainer.num_warmup_steps` |
| 调整保存频率 | shell 脚本 | `--trainer.save_interval` |
| resume 训练 | shell 脚本 | `--trainer.is_resume True`（必须在续行符链中） |
| 切换 action 类型 | YAML `datasets.vla_data` | `action_type: delta_qpos` / `absolute_qpos` |

---

## 十一、LIBERO 评测流程

> 训练完成后，通过 `examples/LIBERO/eval_files/` 下的脚本在 LIBERO 仿真环境中评测模型。评测采用**客户端-服务端架构**：服务端加载 checkpoint 并提供推理 API，客户端通过 websocket 调用模型并在 LIBERO 环境中执行动作。

### 11.1 评测整体调用链

```
eval_libero.sh
  ├─ 1. 解析 checkpoint 路径（支持单文件 .pt / 轻量目录 / DeepSpeed 目录）
  ├─ 2. 设置 LIBERO_HOME / PYTHONPATH / MUJOCO_GL=egl
  ├─ 3. 启动推理服务（policy_server，监听 websocket）
  └─ 4. eval_libero.py
       ├─ benchmark.get_benchmark_dict()["libero_goal"]() → 加载任务集
       ├─ OffScreenRenderEnv → 创建无头仿真环境
       ├─ ModelClient → 连接推理服务 (websocket)
       └─ 循环：task → episode → step
            ├─ env.set_init_state() → 设置固定初始状态
            ├─ 前 10 步 dummy action（等待物体稳定）
            ├─ 观测 → ModelClient.step() → 模型推理 → 动作
            ├─ env.step(action) → obs, reward, done, info
            └─ done=True → 统计成功/失败，保存视频
```

### 11.2 eval_libero.sh 启动脚本

`examples/LIBERO/eval_files/eval_libero.sh` 做三件事：

1. **解析 checkpoint**：自动识别三种 checkpoint 形态
   - 单文件：`steps_N_pytorch_model.pt`
   - 轻量目录：`steps_N/`（含 `model.safetensors.index.json`）
   - DeepSpeed 目录：`steps_N/pytorch_model/...`

2. **环境配置**
   - `LIBERO_HOME`：LIBERO 库安装路径
   - `PYTHONPATH`：让 eval 脚本找到 LIBERO 工具，也让 LIBERO 找到 websocket 客户端
   - `MUJOCO_GL=egl` / `PYOPENGL_PLATFORM=egl`：无头渲染（无需显示器）

3. **调用 eval_libero.py**

```bash
python ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${CKPT} \
    --args.host 127.0.0.1 \
    --args.port 6694 \
    --args.task-suite-name libero_goal \
    --args.num-trials-per-task 50 \
    --args.video-out-path ${output_dir}
```

### 11.3 LIBERO 任务集与环境接口

`eval_libero.py:84-104`：

```python
# 1. 加载任务集
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict["libero_goal"]()
num_tasks = task_suite.n_tasks   # libero_goal = 10 个任务

# 2. 获取任务描述与初始状态
task = task_suite.get_task(task_id)
initial_states = task_suite.get_task_init_states(task_id)   # [N, state_dim]

# 3. 创建环境
env, task_description = _get_libero_env(task, resolution=256, seed=7)

# _get_libero_env 内部：
#   OffScreenRenderEnv(
#       bddl_file_name=task_bddl_file,   # BDDL 任务定义文件
#       camera_heights=256,
#       camera_widths=256,
#   )
```

**关键设计**：LIBERO 评测使用**固定初始状态**（`set_init_state`），而非随机初始化。这意味着同一 checkpoint 多次评测应该得到相同结果（只要 seed 固定）。

### 11.4 单 episode 执行循环

`eval_libero.py:130-226`：

```python
for episode_idx in range(args.num_trials_per_task):   # 每任务 50 个 episode
    env.reset()
    obs = env.set_init_state(initial_states[episode_idx])

    while t < max_steps + num_steps_wait:
        # 前 10 步 dummy action，等待物体落稳
        if t < num_steps_wait:
            obs, reward, done, info = env.step(DUMMY_ACTION)
            continue

        # 1. 观测预处理
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])      # 旋转 180°
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        state = np.concatenate([
            obs["robot0_eef_pos"],           # [3] 末端执行器位置
            _quat2axisangle(obs["robot0_eef_quat"]),  # [3] 轴角
            obs["robot0_gripper_qpos"],      # [1] 夹爪位置
        ])  # = [7]

        # 2. 构造模型输入
        example_dict = {
            "image": [img, wrist_img],       # List[np.ndarray], 双视角
            "lang": task_description,        # str, 如 "put the bowl on the plate"
        }

        # 3. 调用模型推理
        response = client_model.step(example=example_dict, step=step)

        # 4. 解析动作并执行
        raw_action = response["raw_action"]   # dict: world_vector, rotation_delta, open_gripper
        delta_action = np.concatenate([
            raw_action["world_vector"],      # [3] xyz 增量
            raw_action["rotation_delta"],    # [3] rpy 增量
            _binarize_gripper_open(raw_action["open_gripper"]),  # [1] 夹爪开关
        ])  # = [7]

        obs, reward, done, info = env.step(delta_action.tolist())
        if done:
            task_successes += 1
            break
```

**图像预处理**：`[::-1, ::-1]` 表示水平和垂直翻转（旋转 180°）。这是因为 LIBERO 训练数据在预处理时也做了同样的旋转，评测必须保持一致的坐标系。

### 11.5 ModelClient：推理服务客户端

`model2libero_interface.py` 封装了与推理服务的通信，核心职责：

```python
class ModelClient:
    def __init__(self, host, port, ...):
        # 1. websocket 连接推理服务
        self.client = WebsocketClientPolicy(host, port)
        # 2. 握手获取模型元信息
        meta = self.client.get_server_metadata()
        self.action_chunk_size = int(meta["action_chunk_size"])   # 通常 = 8

    def step(self, example: dict, step: int) -> dict:
        # Action Chunking：每隔 action_chunk_size 步才调用一次推理
        if step % self.action_chunk_size == 0 or self.raw_actions is None:
            response = self.client.predict_action({
                "examples": [example],
                "use_ddim": True,
                "num_ddim_steps": 10,
            })
            self.raw_actions = response["data"]["actions"][0]   # [8, 7]

        # 从 chunk 中取出当前步对应动作
        raw_action = self.raw_actions[step % self.action_chunk_size]
        return {
            "raw_action": {
                "world_vector": raw_action[:3],
                "rotation_delta": raw_action[3:6],
                "open_gripper": raw_action[6:7],
            }
        }
```

**Action Chunking（动作块缓存）**：
- 模型每次推理输出 `action_horizon=8` 步动作序列
- `ModelClient` 每 8 步才发送一次 websocket 请求，中间 7 步直接从缓存中取
- 这大幅减少了推理开销（50 episode × 300 step 只需要约 50 × 300/8 ≈ 1875 次推理调用）

**Action Ensemble（动作平滑）**：
- 当 `action_ensemble=True` 时，使用 `AdaptiveEnsembler` 对重叠窗口的动作做加权平均
- 这消除了 chunk 边界处的动作跳变，使机械臂运动更平滑

### 11.6 推理服务侧简要流程

服务端（`deployment/model_server/` 下）加载 checkpoint 后：

```python
# 1. 加载模型
model = Qwen_GR00T.from_pretrained(checkpoint_path)

# 2. 接收客户端请求
examples = [{"image": [img], "lang": instruction}]

# 3. 模型推理
output = model.predict_action(examples=examples, use_ddim=True, num_ddim_steps=10)
#    ├─ build_qwenvl_inputs() → tokenize
#    ├─ qwen_vl_interface() → hidden_states [B, L, 2560]
#    └─ action_model.predict_action() → 4 步 Euler ODE 去噪 → [B, 8, 7]

# 4. 反归一化（unnormalization）
#    模型输出是 normalized action，服务端根据 dataset_statistics.json 做反归一化
#    转换为物理意义上的 delta_qpos

# 5. 返回客户端
{"data": {"actions": [B, 8, 7]}}
```

### 11.7 评测结果统计

评测完成后，结果通过两个口径统计：

1. **日志输出**：`eval_libero.py` 实时打印每任务成功率
2. **视频文件名**：`rollout_{task}_episode{N}_{success|failure}.mp4`
   - 直接数 `success` 文件名数量即可得到成功率

```python
# 保存视频
imageio.mimwrite(
    f"rollout_{task_segment}_episode{episode_idx}_{suffix}.mp4",
    replay_images,   # 列表，每帧为 np.ndarray [H, W, 3]
    fps=10,
)
```

### 11.8 评测速查表

| 项目 | 值 |
|------|-----|
| 任务集 | `libero_goal`（10 任务）、`libero_spatial`（10）、`libero_object`（10）、`libero_10`（10）、`libero_90`（90） |
| 每任务 episode | 50 |
| 图像分辨率 | 256×256 |
| 视角 | agentview（第三人称）+ wrist（手腕） |
| 最大步数 | `libero_goal=300`、`libero_spatial=220`、`libero_object=280` |
| 等待步数 | 10（dummy action，等物体落稳） |
| 动作格式 | delta_qpos（xyz + rpy + gripper） |
| 动作块长度 | 8（`action_chunk_size=8`） |
| 推理去噪步数 | 10（`num_ddim_steps=10`，评测时比训练更精细） |
| 夹爪处理 | 二值化：`1=open, -1=close` |
| 初始状态 | 固定（`set_init_state`），非随机 |
| 渲染后端 | EGL（无头，无需显示器） |
| 视频保存 | `playground/eval_results/{task_suite}/{ckpt_name}/` |
