# Codex Review: Backbone Embedding Infrastructure

## 当前阶段

M6：新 backbone 接入基础设施评审与修复。2026-07-21 评审的 12 个文件改动集，涉及 LoRA 兼容加载、Qwen VLM LoRA 注入、eval payload 三路重构、视频录制升级和 server 批量推理。

## 背景

前序（Kimi 会话 2026-07-21）结论：
- E-001 v2：PI_v3-Bridge 的 Qwen3-VL-4B backbone + cont.ft32 头（~600M）从头训
- E-003 v2：WM4A-Wan2d2-OFT 的 Wan2.2 backbone + cont.ft32 头（input dim 改 1024）
- backbone 冻结优先，LoRA 做第二臂

当前改动集是上述方案所需的前置基础设施，**尚未实现 backbone swap 本身**。

## 评审改动文件

| 文件 | 改动量 | 主题 |
|---|---|---|
| `starVLA/model/modules/lora_utils.py` | **新建 47 行** | LoRA 兼容加载工具 |
| `starVLA/model/modules/vlm/QWen3.py` | +4 行 | Qwen backbone 接入 `load_pretrained_state_dict` |
| `starVLA/model/modules/world_model/Wan2.py` | +10 行 | Wan backbone 接入（`optional_prefixes=("text_encoder.","vae.")`） |
| `starVLA/model/framework/VLM4A/QwenPI_v3.py` | +55 行 | `_configure_qwen_lora()`：PEFT LoRA 注入 Qwen |
| `starVLA/model/framework/WM4A/WanPI.py` | +40 行 | CPU UMT5 fallback 编码缺失指令 |
| `starVLA/training/trainer_utils/trainer_tools.py` | +5 行 | `load_chkp_not_strict` 优先调用 `load_pretrained_state_dict` |
| `starVLA/dataloader/lerobot_datasets.py` | +5 行 | action_chunk_steps 覆盖逻辑修复 |
| `deployment/model_server/policy_wrapper.py` | +25 行 | 诊断日志（device/backbone_device/latency） |
| `deployment/model_server/server_policy.py` | +12 行 | `max_batch_size` / `batch_timeout_ms` 参数 |
| `examples/Robocasa_365/eval_files/model2robocasa365_interface.py` | **~340 行净增** | payload_style 三路（wanpi/starflow/groot）+ 数据契约校验 + server meta 校验 |
| `examples/Robocasa_365/eval_files/simulation_env.py` | **~200 行净增** | split/seed/EpisodeProgressBar/视频参数/no-video |
| `examples/Robocasa_tabletop/eval_files/wrappers/video_recording_wrapper.py` | **~200 行净增** | 多视角横拼/任务名文件名/视图别名/obs 帧复用 |

## 需要修复的问题

按严重程度排序。

### P0 —— argparse 漏注册 `groot` payload_style

**文件**：`examples/Robocasa_365/eval_files/simulation_env.py:348`

```python
parser.add_argument(
    "--args.payload-style",
    dest="payload_style",
    choices=["wanpi", "starflow"],   # ⚠️ 漏了 "groot"
```

tyro 路径支持三种，但 `_parse_args_without_tyro` 路径拒绝 `--args.payload-style groot`。EXT-B1 eval script 如果依赖 tyro（可能未安装）就会走到 argparse 路径，导致无法启动。

**修复**：`choices` 加 `"groot"`。

### P1——`starflow` 分支缺 3 视角校验

**文件**：`model2robocasa365_interface.py:_validate_observations()`

`groot` 分支显式校验了 3 个视角键；`wanpi` 分支校验了 left/wrist 的 shape。但 `starflow` 分支（第 236-280 行内部）依赖后续的 `STATE_KEY_ORDER` 检查——如果 `agentview_right` 缺失，错误信息不够明确。应该和 `groot` 一样在 `starflow` 分支开头显式校验三个 `video.robot0_agentview_*` 键都存在。

### P1——`lora_utils.py` candidates 可能有歧义匹配

**文件**：`starVLA/model/modules/lora_utils.py:21-36`

```python
candidates = [key]
if key.startswith("model."):
    candidates.append("model.base_model.model." + key[len("model."):])
for candidate in list(candidates):
    for suffix in (".weight", ".bias"):
        if candidate.endswith(suffix):
            candidates.append(candidate[:-len(suffix)] + ".base_layer" + suffix)
target = next((candidate for candidate in candidates if candidate in current_keys), None)
```

如果 PEFT 的某些配置同时存在 `model.layers.0.xxx.weight`（合入权重）和 `model.base_model.model.layers.0.xxx.base_layer.weight`（分离权重），`next()` 拿到的可能不是目标路径。建议加 `ambiguous` 检查：

```python
matched = [c for c in candidates if c in current_keys]
if len(matched) > 1:
    logger.warning("Ambiguous LoRA key mapping for %s: %s", key, matched)
target = matched[0] if matched else None
```

### P2——`WanPI.py` CPU UMT5 fallback 的设备安全

**文件**：`starVLA/model/framework/WM4A/WanPI.py:_encode_instruction_on_cpu()`

`_ensure_mowa_text_fallback_encoder()` 将 encoder 移到 CPU，但 `encode_instruction()` 内部如果不做设备防护，可能把 tensor 移回 GPU。建议在调用时显式断言或在编码时用 `with torch.device("cpu")` 包裹：

```python
device_before = next(encoder._text_encoder.parameters()).device
result = encoder.encode_instruction(instruction)
assert device_before == next(encoder._text_encoder.parameters()).device, "UMT5 encoder moved from CPU"
```

### P2——`simulation_env.py` tyro 路径未校验 split

tyro 传 `--args.split foo` 不会报错，直到 `gym.make(..., split="foo")` 才崩。main() 里加一行：

```python
assert config.split in ("pretrain", "target"), f"Invalid split: {config.split}"
```

### P3——`video_recording_wrapper.py` fallback 渲染无日志

`_render_view` 的 fallback 路径（`base.env.sim.render`）如果频繁触发，会影响 eval 帧率但没有任何日志提示。建议 fallback 时打印 debug 日志。

### P3——`QwenPI_v3.py` LoRA 注入后缺 trainable 日志

`_configure_qwen_lora()` 注入 LoRA 后，建议打印 backbone 中可训练参数的信息，方便调试冻结策略：

```python
trainable = [(n, p.numel()) for n, p in self.qwen_vl_interface.named_parameters() if p.requires_grad]
logger.info("Qwen LoRA trainable params: %d (%.2fM)", len(trainable), sum(s for _, s in trainable) / 1e6)
```

## 未实现（当前改动集不包含）

以下为 E-001 v2 / E-003 v2 方案的核心部分，**尚未在代码中实现**：

### 1. PI_v3-Bridge backbone 权重抽取

从 `/disk/rl/models/mowa_init_candidates/Qwen3VL-PI_v3-Bridge-RT_1/checkpoints/steps_50000_pytorch_model.pt` 中提取 `qwen_vl_interface.*` 权重（4.8B），丢掉 `action_model.*` / `project_layers.*`（634M）。用于初始化新框架的 backbone。

### 2. cont.ft32 头接入 YAML

定义新框架配置（可继承 QwenPI_v3 但替换 action_model 为 StarFlow continuous_head + ft32 variant）：
- `action_model_type: continuous`（而非 DiT-B）
- `starflow_ft_variant: ft32`
- `state_mode: continuous_head`
- `input_embedding_dim: 1024`（PI_v3-Bridge 的 qwen_vl_interface 输出是 1024，与 cont.ft32 对齐，无需 adapter）

### 3. Wan 线：input_embedding_dim 改 1024

Wan2.2 backbone 的输出 hidden 是 1024 维，而现有 cont.ft32（基于 Qwen 搭建）的 cross-attention input_embedding_dim 需要从 2560 改成 1024。这是一个 YAML 配置级改动。

### 4. Backbone 冻结 + LoRA 对照

- 主干：freeze qwen_vl_interface（backbone 完全冻结）
- 对照臂：freeze qwen_vl_interface + 启用 Qwen LoRA（已实现的 `_configure_qwen_lora()` 为此服务）

## 测试命令

```bash
# 回归测试
pytest tests/mowa -q
# 专项测试
python -m unittest tests.test_starflow_vla_reuse -v
# LoRA 注入测试
python -c "
from starVLA.model.framework.VLM4A.QwenPI_v3 import Qwen_PI_v3
# 需配置 YAML 启用 qwenvl.lora.enabled=true
"
# payload_style smoke
python examples/Robocasa_365/eval_files/simulation_env.py --help
# argparse 校验
python examples/Robocasa_365/eval_files/simulation_env.py --args.payload-style groot  # 应不报错
```

## 验收标准

1. argparse `groot` 被接受 → server 可启动
2. `starflow` 分支缺视角报明确错误
3. `lora_utils.py` 歧义匹配不静默
4. WanPI CPU 编码不意外移回 GPU
5. 回归测试全部通过

## 阶段边界

本 prompt 只处理上述评审问题的修复和未实现项的前置设计。不包含：
- 实际 backbone swap 的训练实验（属于 E-001 v2 / E-003 v2 正式实验）
- 数据混采策略改动
- 评测门禁阈值改动
