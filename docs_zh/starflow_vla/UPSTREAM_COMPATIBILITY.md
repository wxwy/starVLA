# StarFlow-VLA 上游兼容策略

## Task ID
P0-M1

## Scope
本文件定义 StarFlow-VLA 进入代码实现前的上游兼容检查策略，不修改 StarVLA 源码。

## Compatibility Rule
- P0 实现必须绑定一个明确的 StarVLA baseline commit。
- 若 StarVLA 上游 commit、`pyproject.toml` 版本、`version_id` 或 framework/action head 接口发生变化，必须先做 compatibility audit。
- 未完成 compatibility audit 前，不进入 P0-M2 framework 实现。

## Audit Items
| 检查项 | 文件或入口 | 影响 |
| --- | --- | --- |
| Framework registry | `starVLA/model/framework/base_framework.py` | `StarFlowVLA` 注册和构建 |
| QwenPI_v3 framework | `starVLA/model/framework/VLM4A/QwenPI_v3.py` | 继承/委托边界 |
| LayerwiseFM action head | `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py` | Flow Matching 主路径 |
| GR00T action head | `starVLA/model/modules/action_model/GR00T_ActionHeader.py` | 可选 action head 对照 |
| Baseline action headers | `MLP_ActionHeader.py`、`VLA_AdapterHeader.py`、`QwenOFT.py` | H2 baseline |
| Config schema | `examples/LIBERO/train_files/starvla_cotrain_libero.yaml` | `version_id` 与配置兼容 |
| Checkpoint save/load | 训练入口与共享加载工具 | `starflow_mapping` 后续写入 |

## Current Status
当前工作区位于 `merge-official-starvla-dev`，已合并官方 `starVLA_dev` 最新 `cdf5434438f4449cff85e3588956f7706a5c9cc3`，同时保留 fork 中已有 LIBERO 训练、checkpoint、eval 与文档资产，详见 `BASELINE_VERSION.md`。

进入 P0-M2 前必须先完成 compatibility audit，重点检查：

- `FRAMEWORK_REGISTRY` 与 `build_framework(cfg)` 是否仍保持 P0 所需入口能力。
- `QwenPI_v3.py` 在官方合并后是否仍适合被 `StarFlowVLA` 继承或委托。
- `LayerwiseFM_ActionHeader.py` / `GR00T_ActionHeader.py` 的 `future_tokens`、state handling 与 Euler solver 接口是否与设计一致。
- fork 保留的 checkpoint loader、lightweight checkpoint、safetensors 分片和 resume 逻辑是否与官方新增训练逻辑兼容。
- `starVLA/dataloader/__init__.py` 中官方新增的 balance 参数与 fork 保留的 `num_workers` / `prefetch_factor` 配置是否可共同使用。

## Compatibility Audit DOC-M9

### Scope
本次审计只做静态阅读与语法检查，不构建真实模型，不加载 checkpoint，不运行训练、评测或部署。

### Result
P0-M2 可以进入实现准备阶段。当前未发现阻断 `StarFlowVLA` facade 入口实现的接口变更。

### Findings
| 检查项 | 结论 | 说明 |
| --- | --- | --- |
| Framework registry | 通过 | `base_framework.py` 仍通过 `_auto_import_framework_modules()` 自动导入 framework module，并由 `FRAMEWORK_REGISTRY` + `build_framework(cfg)` 按 `cfg.framework.name` 构建。 |
| Existing `StarFlowVLA` registry | 未存在，符合预期 | 当前未注册 `StarFlowVLA`，这是 P0-M2 的新增任务，不是合并回归。 |
| QwenPI_v3 reuse points | 通过 | `QwenPI_v3.py` 仍注册 `QwenPI_v3`，并保留 `qwen_vl_interface`、`project_layers`、`action_model`、`forward()`、`predict_action()`、state-to-instruction 路径。 |
| LayerwiseFM action head | 通过 | `LayerwiseFM_ActionHeader.py` 仍保留 `future_tokens`、`num_target_vision_tokens`、`state_encoder`、layer-wise cross-attention、Euler `predict_action()`。 |
| GR00T action head | 通过 | `GR00T_ActionHeader.py` 仍保留 `future_tokens`、`state_encoder`、Euler `predict_action()`，可作为 StarVLA-native 对照或后续备选。 |
| Checkpoint loader | 通过，需 P0-M2 后复验 | `share_tools.py` / `trainer_tools.py` 仍保留单文件、目录、DeepSpeed、safetensors 分片和 lightweight training checkpoint 解析逻辑。 |
| Dataloader merge | 通过，需数据 smoke 复验 | `starVLA/dataloader/__init__.py` 同时保留官方 `balance_dataset_weights` / `balance_trajectory_weights` 和 fork 的 `num_workers` / `prefetch_factor` 配置。 |
| Config schema | 通过 | `pyproject.toml` 仍为 `1.0.1`，LIBERO 示例配置仍为 `version_id: "0.21"`。 |

### P0-M2 Guardrails
- 新增 `StarFlowVLA` 时只新增独立 framework 入口，不复制 QwenPI_v3 主体逻辑。
- `StarFlowVLA` 应继承或委托 `Qwen_PI_v3`，复用 `qwen_vl_interface`、`project_layers`、`action_model`、`forward()` 与 `predict_action()` 主路径。
- P0 默认继续使用 state-to-instruction，不强制启用显式 `FlowCondition` runtime。
- P0 默认继续使用 `action_dim=7`，不引入 `14D action_mask`。
- P0-M2 只做 import / registry / config parse / dry-run 级验证；真实 forward/backward、loss finite 和 single batch overfit 留到 P0-M4 / P0-M5。

### Checks
```bash
sed -n '1,260p' starVLA/model/framework/base_framework.py
rg -n "FRAMEWORK_REGISTRY|def build_framework|register\(" starVLA/model/framework -S
sed -n '1,260p' starVLA/dataloader/__init__.py
sed -n '1,620p' starVLA/model/framework/VLM4A/QwenPI_v3.py
rg -n "num_target_vision_tokens|future_tokens|state_encoder|predict_action|forward\(|action_model|action_dim|num_inference_timesteps|Euler|euler|cross" starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py starVLA/model/modules/action_model/GR00T_ActionHeader.py
python -m py_compile starVLA/model/framework/base_framework.py starVLA/model/framework/VLM4A/QwenPI_v3.py starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py starVLA/model/modules/action_model/GR00T_ActionHeader.py starVLA/model/framework/share_tools.py starVLA/training/train_starvla.py starVLA/training/trainer_utils/trainer_tools.py starVLA/dataloader/__init__.py
```

## Not Run
未运行 StarFlowVLA 代码测试、真实模型加载、训练、评测或部署。
