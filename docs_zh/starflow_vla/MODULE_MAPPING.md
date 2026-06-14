# StarFlow-VLA 模块映射

## Scope
本文件记录设计抽象到 StarVLA-native 实现路径的 P0 映射。当前已新增 `StarFlowVLA` framework facade 与 `starflow_mapping` schema / 旁路保存工具；checkpoint 自动写入留到 P0-M9。

## P0 Mapping Draft
| 设计抽象 | StarVLA-native 映射 | P0 状态 |
| --- | --- | --- |
| StarFlowVLA framework | `starVLA/model/framework/VLM4A/StarFlowVLA.py` | P0-M2 已新增 facade 入口 |
| Base framework | `QwenPI_v3` | DOC-M9 静态审计确认可继承或委托复用 |
| VLM backbone | Qwen3-VL via `get_vlm_model(config=...)` | DOC-M9 静态审计确认保留现有路径 |
| Projection layers | `QwenPI_v3.project_layers` | DOC-M9 静态审计确认保留现有路径 |
| Flow Matching action head | `LayerwiseFM_ActionHeader.py` | DOC-M9 静态审计确认 P0 默认路径仍可用 |
| H2 baseline | `MLP_ActionHeader.py`、`VLA_AdapterHeader.py`、`QwenOFT.py` | 待 P0-M6 |
| Action token / future token route | `future_tokens + cross-DiT` | DOC-M9 静态审计确认 LayerwiseFM / GR00T 均保留 `future_tokens` |
| State default path | QwenPI_v3 state-to-instruction | DOC-M9 静态审计确认 P0 默认路径保留 |
| Explicit FlowCondition runtime | 可选 dataclass / wrapper | P2，不阻断 P0 |
| PerceiverAdapter | 可选 token compressor | P2，不阻断 P0 |
| 7/14DoF action mask | `max_action_dim=14 + action_mask + masked loss` | P1，不阻断 P0 |
| starflow_mapping manifest | `starVLA/model/modules/starflow_vla/mapping.py` | P0-M3 已新增 schema 构造与 `starflow_mapping.json` 旁路保存工具；checkpoint 自动写入待 P0-M9 |

## Boundary
P0 验收不以新增同名抽象类为标准，而以 StarVLA-native 路线是否可注册、可配置、可 dry-run、可追踪为标准。

## Not Run
未运行 StarFlowVLA 代码测试、真实模型加载、训练、评测或部署。
