# StarFlow-VLA Patch Manifest

## Scope
本文件记录 StarFlow-VLA 相关 patch 和文档变更。当前只记录 `docs_zh/starflow_vla` 文档整理，不记录任何源码 patch。

## Manifest
| ID | 类型 | 文件 | 状态 | 说明 |
| --- | --- | --- | --- | --- |
| DOC-M5 | docs | `README.md` | added | 补全文档索引 |
| DOC-M5 | docs | `IMPLEMENTATION_LOG.md` | modified | 追加文档整理记录 |
| DOC-M6 | docs | `DESIGN_FREEZE_CHECK.md` | added | P0-M0 冻结检查记录 |
| DOC-M6 | docs | `BASELINE_VERSION.md` | added | P0-M1 基线版本记录 |
| DOC-M6 | docs | `UPSTREAM_COMPATIBILITY.md` | added | P0-M1 上游兼容策略 |
| DOC-M6 | docs | `PATCH_MANIFEST.md` | added | patch manifest 初始骨架 |
| DOC-M6 | docs | `MODULE_MAPPING.md` | added | StarVLA-native 映射骨架 |
| DOC-M6 | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 P0-M0/P0-M1 文档记录 |
| DOC-M8 | docs | `README.md` | modified | 将已补全文档改为索引链接 |
| DOC-M8 | docs | `BASELINE_VERSION.md` | modified | 记录官方合并后的当前基线状态 |
| DOC-M8 | docs | `UPSTREAM_COMPATIBILITY.md` | modified | 更新 P0-M2 前 compatibility audit 范围 |
| DOC-M8 | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| DOC-M8 | docs | `IMPLEMENTATION_LOG.md` | modified | 追加官方合并后的文档同步记录 |
| DOC-M9 | docs | `UPSTREAM_COMPATIBILITY.md` | modified | 追加 P0-M2 前 compatibility audit 结论 |
| DOC-M9 | docs | `MODULE_MAPPING.md` | modified | 同步 DOC-M9 静态审计后的映射状态 |
| DOC-M9 | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| DOC-M9 | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 compatibility audit 实施记录 |
| P0-M2 | code | `starVLA/model/framework/VLM4A/StarFlowVLA.py` | added | 新增 StarFlowVLA framework facade 入口 |
| P0-M2 | docs | `MODULE_MAPPING.md` | modified | 标记 StarFlowVLA facade 与 mapping 方法状态 |
| P0-M2 | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| P0-M2 | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 P0-M2 实施记录 |

## Source Code Patches
`P0-M2` 新增 StarFlowVLA facade 文件，但不修改 `QwenPI_v3.py`、`LayerwiseFM_ActionHeader.py` 或 `GR00T_ActionHeader.py` 主体逻辑。

## Not Run
未运行 StarFlowVLA 代码测试、真实模型加载、训练、评测或部署。
