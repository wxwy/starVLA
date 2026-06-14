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
当前工作区已切换到设计文档锚点 `42170b2a4df3877ccf6581948e2198d37c363c7f`，详见 `BASELINE_VERSION.md`。进入 P0-M2 前建议先从该 commit 创建专用实现分支，并继续保持上游兼容审计要求。

## Not Run
未运行 StarFlowVLA 代码测试、真实模型加载、训练、评测或部署。
