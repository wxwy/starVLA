# StarFlow-VLA：DeepSpeed Universal Checkpoint 迁移尝试记录

> 记录时间：2026-06-17  
> 背景：用户希望将 StarFlow-VLA 项目从自定义 lightweight checkpoint 迁移到 DeepSpeed Universal Checkpoint，实现跨 world-size 恢复。

## 目标

- 将现有最新的 lightweight checkpoint（`steps_31500`）转换为 DeepSpeed Universal Checkpoint。
- 验证 4 卡 DeepSpeed ZeRO-2 进程可以从该 Universal Checkpoint 恢复。
- 保持现有 lightweight checkpoint 逻辑不被破坏。

## 已完成的工作

### 1. 代码修改

#### 1.1 `starVLA/training/train_starvla.py`

新增 `VLATrainer.save_deepspeed_zero_checkpoint()` 方法：

```python
def save_deepspeed_zero_checkpoint(self, output_dir, tag=None):
    """Save the current DeepSpeed engine state as a native ZeRO checkpoint."""
    if self.accelerator.distributed_type != DistributedType.DEEPSPEED:
        raise RuntimeError("save_deepspeed_zero_checkpoint requires a DeepSpeed engine")
    output_dir = Path(output_dir)
    if self.accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
    self.accelerator.wait_for_everyone()
    if tag is None:
        tag = f"steps_{self.completed_steps}"
    logger.info(f"Saving DeepSpeed ZeRO checkpoint to {output_dir} with tag {tag}")
    self.model.save_checkpoint(str(output_dir), tag=tag)
    self.accelerator.wait_for_everyone()
    logger.info(f"DeepSpeed ZeRO checkpoint saved to {output_dir / tag}")
```

#### 1.2 `tools/convert_lightweight_checkpoint_to_universal.py`（新建）

实现了一个转换脚本，流程如下：

1. 验证 source lightweight checkpoint（检查 `trainer_state.json`、world_size、必需文件）。
2. 创建临时 run 目录，将 source checkpoint 以 symlink 形式放到 `checkpoints/steps_31500`。
3. 加载 source 的 `config.full.yaml`，覆盖 `trainer.is_resume=True`、`max_train_steps=0` 等字段。
4. 复用 `train_starvla.py` 的 `build_framework`、`prepare_data`、`setup_optimizer_and_scheduler` 构建模型与优化器。
5. 调用 `VLATrainer.prepare_training()` 将 lightweight checkpoint 加载进 DeepSpeed engine。
6. 调用 `save_deepspeed_zero_checkpoint()` 保存中间 ZeRO checkpoint。
7. rank 0 调用 `python -m deepspeed.checkpoint.ds_to_universal` 转换为 Universal Checkpoint。
8. 原子重命名临时输出目录到最终目录。

已修复的 bug：
- 并发 symlink 创建竞态 → 改为仅 rank 0 创建，其他 rank 通过 marker 文件等待。
- `ds_to_universal` 的 `--input_folder` 错误指向父目录 → 改为指向 tag 子目录（如 `tmp_zero/steps_31500`）。

### 2. 执行的命令与结果

#### 2.1 4 卡 lightweight checkpoint 恢复验证（前置验证）

已确认现有 lightweight checkpoint 可以在 4 卡 ZeRO-2 上完整恢复，优化器状态匹配。

#### 2.2 运行转换脚本

```bash
export WANDB_MODE=disabled
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/compat:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
/gemini/code/starVLA/.venv/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  tools/convert_lightweight_checkpoint_to_universal.py \
  --source_checkpoint playground/Checkpoints/P0-M5-E-H2a-03_starflow_libero-goal_qwen3vl4b_lwfm_ft32_250615/checkpoints/steps_31500 \
  --output_universal_dir playground/Checkpoints/P0-M5-E-H2a-03_universal/steps_31500
```

结果：脚本成功将 lightweight checkpoint 加载并保存为 DeepSpeed ZeRO checkpoint，但在调用 `ds_to_universal` 时失败。

### 3. `ds_to_universal` 转换尝试

#### 第一次尝试（网络存储）

输入：`playground/Checkpoints/P0-M5-E-H2a-03_universal/.tmp_zero_steps_31500/steps_31500`  
错误：

```text
AssertionError: Required universal_checkpoint_info state is missing in checkpoint.
Verify that client creates this state.
```

说明：DeepSpeed 保存的 ZeRO checkpoint 缺少 `universal_checkpoint_info` 元数据。该元数据通常由训练框架（如 Megatron-DeepSpeed）在 `client_state` 中提供，Accelerate + 自定义训练脚本不会自动写入。

#### 第二次尝试（`--inject_missing_state`）

输入：同上，附加 `--inject_missing_state`  
结果：越过了 `universal_checkpoint_info` 检查，但在 **Merging slices** 阶段失败：

```text
RuntimeError: torch.cat(): expected a non-empty list of Tensors
```

失败位置：`deepspeed/checkpoint/ds_to_universal.py:320`，`merge_tp_slices` 函数。

原因分析：`--inject_missing_state` 注入的是一个空的 `universal_checkpoint_info`（仅含 `universal_checkpoint_version=0.2`）。合并切片逻辑依赖 `vocabulary_parameter_patterns`、`tp_replicated_parameter_patterns` 等字段来决定如何拼接参数；空对象导致部分参数找不到切片，最终 `slices` 为空。

#### 第三次尝试（本地 /tmp 存储）

为排除网络存储 I/O 瓶颈，将 21 GB ZeRO checkpoint 复制到本地 `/tmp`：

```bash
cp -r .../.tmp_zero_steps_31500/steps_31500 /tmp/starflow_universal_conv/zero_steps_31500
```

复制耗时约 20 分钟。随后运行：

```bash
python -m deepspeed.checkpoint.ds_to_universal \
  --input_folder /tmp/starflow_universal_conv/zero_steps_31500 \
  --output_folder /tmp/starflow_universal_conv/universal_steps_31500.tmp \
  --inject_missing_state
```

结果：提取 ZeRO fragments 成功（4/4），但在合并 672 个参数的第一个时就报同样的 `torch.cat` 空列表错误。

## 4. 关键发现

### 4.1 ZeRO checkpoint 结构

保存的中间 ZeRO checkpoint 包含：

```text
/tmp/starflow_universal_conv/zero_steps_31500/
├── latest
├── mp_rank_00_model_states.pt          # 18 GB，完整模型参数
├── bf16_zero_pp_rank_0_mp_rank_00_optim_states.pt  # 604 MB
├── bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt  # 604 MB
├── bf16_zero_pp_rank_2_mp_rank_00_optim_states.pt  # 604 MB
├── bf16_zero_pp_rank_3_mp_rank_00_optim_states.pt  # 604 MB
└── zero_to_fp32.py
```

模型状态文件大小 18 GB，远大于原始 safetensors 的 ~10 GB，说明 DeepSpeed 的 ZeRO-2 模型状态保存了额外信息（如 fp32 master weights 引用、buffer 等）。

### 4.2 模型参数命名

从 `mp_rank_00_model_states.pt` 中解析出的关键参数：

- 词表相关：
  - `qwen_vl_interface.model.model.language_model.embed_tokens.weight`: `(151936, 2560)`
  - `qwen_vl_interface.model.lm_head.weight`: `(151936, 2560)`
- 视觉部分：`qwen_vl_interface.model.model.visual.*`
- 动作头：`action_model.model.*`
- 总参数键数量：1386

词表大小为 **151936**，无显式 padding 信息。

### 4.3 缺少 `universal_checkpoint_info` 是上游已知问题

GitHub 上存在相同问题：

- [deepspeedai/DeepSpeed#5430](https://github.com/deepspeedai/DeepSpeed/issues/5430): *No `universal_checkpoint_info` in the Accelerate+Deepspeed Checkpoint*
- [huggingface/transformers#33157](https://github.com/huggingface/transformers/issues/33157): *Failed to load universal_checkpoint with deepspeed integration*

目前 Accelerate / Transformers 不会自动注入该元数据，需要训练脚本在调用 `engine.save_checkpoint()` 时通过 `client_state` 手动提供。

## 5. 当前阻塞点

要成功运行 `ds_to_universal`，必须在保存 ZeRO checkpoint 时或转换前向 `mp_rank_00_model_states.pt` 写入正确的 `universal_checkpoint_info`，至少包含：

```python
{
    "universal_checkpoint_version": 0.2,
    "vocabulary_parameter_patterns": [
        r".*embed_tokens\.weight$",
        r".*lm_head\.weight$",
    ],
    "original_vocab_size": 151936,
    "padded_vocab_size": 151936,
    "tp_replicated_parameter_patterns": [],
    "pipeline_replicated_parameter_patterns": [],
    "parameter_to_average_patterns": [],
    "parameter_with_row_parallelism_patterns": [],
    "parameter_with_2_sub_params_cat_dim_0": [],
    "parameter_with_sub_params": [],
}
```

但即使写入上述信息，仍不确定是否能解决 `torch.cat` 空列表错误，因为该错误可能还与其他参数分片映射不一致有关（例如冻结参数、共享参数、action_model 参数等）。

## 6. 临时文件位置

如要继续，可利用以下已保存的 ZeRO checkpoint，无需重新加载模型：

- 网络存储：`playground/Checkpoints/P0-M5-E-H2a-03_universal/.tmp_zero_steps_31500/steps_31500`
- 本地 `/tmp`：`/tmp/starflow_universal_conv/zero_steps_31500`

最终目标输出目录（尚未生成）：

- `playground/Checkpoints/P0-M5-E-H2a-03_universal/steps_31500`

## 7. 后续可选方向

### 方向 A：继续完成 Universal Checkpoint 转换

1. 在 `save_deepspeed_zero_checkpoint()` 中通过 `client_state` 注入 `universal_checkpoint_info`。
2. 重新运行完整转换流程（需要再次加载模型，约 5 分钟）。
3. 运行 `ds_to_universal`（本地存储）完成转换。
4. 执行 4 卡 / 1 卡 / 4 卡弹性恢复验证。

### 方向 B：放弃 `ds_to_universal`，直接验证 ZeRO checkpoint 跨卡恢复

如果项目当前最迫切的诉求是“能跨卡恢复训练”，可以：

1. 直接用 4 卡保存的 ZeRO checkpoint 做 4 卡恢复验证。
2. 研究 DeepSpeed ZeRO checkpoint 是否支持同 world-size 恢复即可，不追求 Universal Checkpoint 的弹性能力。

### 方向 C：换用其他模型或更小 checkpoint 先做 POC

如用户所言“换模型处理”，可先用更小的模型（如更小的 VLM 或 toy action head）跑通整个 lightweight → universal → 弹性恢复流程，再迁移到完整 4B checkpoint。

## 8. 已修改文件清单

- `starVLA/training/train_starvla.py`：新增 `save_deepspeed_zero_checkpoint()`。
- `tools/convert_lightweight_checkpoint_to_universal.py`：新建转换脚本。
- 本文件：`docs_zh/starflow_vla/UNIVERSAL_CHECKPOINT_CONVERSION_ATTEMPT.md`。

## 9. 参考命令速查

```bash
# 4 卡转换
export WANDB_MODE=disabled
export LD_LIBRARY_PATH=/usr/local/cuda-12.3/compat:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH}
/gemini/code/starVLA/.venv/bin/accelerate launch \
  --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml \
  --num_processes 4 \
  tools/convert_lightweight_checkpoint_to_universal.py \
  --source_checkpoint playground/Checkpoints/P0-M5-E-H2a-03_starflow_libero-goal_qwen3vl4b_lwfm_ft32_250615/checkpoints/steps_31500 \
  --output_universal_dir playground/Checkpoints/P0-M5-E-H2a-03_universal/steps_31500

# 本地 ds_to_universal
python -m deepspeed.checkpoint.ds_to_universal \
  --input_folder /tmp/starflow_universal_conv/zero_steps_31500 \
  --output_folder /tmp/starflow_universal_conv/universal_steps_31500.tmp \
  --inject_missing_state
```
