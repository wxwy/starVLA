# StarFlow-VLA Patch Manifest

## Scope
本文件记录 StarFlow-VLA 相关 patch 和文档变更。

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
| P0-M3 | code | `starVLA/model/modules/starflow_vla/__init__.py` | added | 新增 StarFlow-VLA 模块工具入口 |
| P0-M3 | code | `starVLA/model/modules/starflow_vla/mapping.py` | added | 新增 starflow_mapping schema 构造与 JSON 旁路保存工具 |
| P0-M3 | code | `starVLA/model/framework/VLM4A/StarFlowVLA.py` | modified | `describe_starflow_mapping()` 委托 mapping 模块 |
| P0-M3 | docs | `MODULE_MAPPING.md` | modified | 标记 P0-M3 mapping schema 状态 |
| P0-M3 | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| P0-M3 | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 P0-M3 实施记录 |
| P0-M4 | test | `tests/test_starflow_vla_reuse.py` | added | 新增 QwenPI_v3 复用 smoke 测试 |
| P0-M4 | docs | `ACCEPTANCE_CHECKLIST.md` | modified | 标记已完成的 Stage A / P0-M3 / P0-M4 验收项 |
| P0-M4 | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| P0-M4 | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 P0-M4 实施记录 |
| P0-M5 | config | `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml` | added | 新增 StarFlowVLA + QwenPI_v3 native + LayerwiseFM 7DoF smoke 配置 |
| P0-M5 | docs | `ACCEPTANCE_CHECKLIST.md` | modified | 标记 P0 默认 7DoF / 不启用 14D mask 配置项 |
| P0-M5 | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| P0-M5 | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 P0-M5 配置级实施记录 |
| P0-M8 | test | `tests/test_starflow_libero_batch.py` | added | 新增 LIBERO batch schema smoke 测试，数据缺失时显式 skip |
| P0-M8 | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| P0-M8 | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 P0-M8 数据可用性实施记录 |
| P0-M6 | config | `configs/starflow_vla/stage2_mlp_baseline.yaml` | added | 新增 QwenOFT + MLP 7DoF baseline smoke 配置 |
| P0-M6 | docs | `ACCEPTANCE_CHECKLIST.md` | modified | 标记 baseline 配置级 dry-run 通过 |
| P0-M6 | docs | `MODULE_MAPPING.md` | modified | 同步 H2 baseline P0-M6 状态 |
| P0-M6 | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| P0-M6 | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 P0-M6 配置级实施记录 |
| P0-M7 | config | `configs/starflow_vla/stage3_future_token_ablation.yaml` | added | 新增 future_tokens `0/8/16/32/64` 消融配置 |
| P0-M7 | docs | `MODULE_MAPPING.md` | modified | 同步 future token ablation 配置状态 |
| P0-M7 | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| P0-M7 | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 P0-M7 配置级实施记录 |
| P0-M9 | code | `starVLA/model/modules/starflow_vla/mapping.py` | modified | 新增 checkpoint sidecar mapping 保存函数 |
| P0-M9 | code | `starVLA/model/modules/starflow_vla/__init__.py` | modified | 导出 checkpoint sidecar mapping 保存函数 |
| P0-M9 | test | `tests/test_starflow_checkpoint_mapping.py` | added | 验证目录和单文件 checkpoint 旁路 mapping JSON |
| P0-M9 | docs | `ACCEPTANCE_CHECKLIST.md` | modified | 标记旁路 mapping 保存与 patch manifest hash 可记录 |
| P0-M9 | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| P0-M9 | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 P0-M9 工具级实施记录 |
| P0-M10 | docs | `EVAL_SMOKE.md` | added | 新增 LIBERO eval smoke 前置条件与阻塞状态记录 |
| P0-M10 | test | `tests/test_starflow_eval_preflight.py` | added | 新增 eval shell 语法和 P0 checkpoint mapping 前置检查 |
| P0-M10 | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| P0-M10 | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 P0-M10 preflight 实施记录 |
| P0-M11 | docs | `EXPERIMENT_MATRIX.md` | added | 新增 P0 配置、测试、阻塞状态矩阵 |
| P0-M11 | test | `tests/test_starflow_docs_governance.py` | added | 新增文档治理测试 |
| P0-M11 | docs | `README.md` | modified | 将实验矩阵加入文档索引 |
| P0-M11 | docs | `ACCEPTANCE_CHECKLIST.md` | modified | 标记 `EXPERIMENT_MATRIX.md` 已存在 |
| P0-M11 | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| P0-M11 | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 P0-M11 文档治理记录 |
| P0-M7a | test | `tests/test_starflow_future_token_variants.py` | added | 新增 future token 5 组 mapping 变体轻量测试 |
| P0-M7a | test | `tests/test_starflow_docs_governance.py` | modified | 将 future token 变体测试纳入 manifest 覆盖检查 |
| P0-M7a | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| P0-M8-DATA | config | `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml` | modified | 对齐 LIBERO registry，P0 使用 7D action / 8D state |
| P0-M8-DATA | config | `configs/starflow_vla/stage2_mlp_baseline.yaml` | modified | 对齐 LIBERO registry，P0 使用 7D action / 8D state |
| P0-M8-DATA | config | `configs/starflow_vla/stage3_future_token_ablation.yaml` | modified | 对齐 LIBERO registry，P0 使用 7D action / 8D state |
| P0-M8-DATA | test | `tests/test_starflow_libero_batch.py` | modified | 使用配置声明校验 action/state 维度，真实 LIBERO batch 通过 |
| P0-M8-DATA | docs | `ACCEPTANCE_CHECKLIST.md` | modified | 标记 LIBERO minimal batch schema 通过 |
| P0-M8-DATA | docs | `EXPERIMENT_MATRIX.md` | modified | 更新 LIBERO 数据已准备和 batch smoke 状态 |
| P0-M8-DATA | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |

## Source Code Patches
`P0-M2` 新增 StarFlowVLA facade 文件；`P0-M3` 新增独立 mapping schema / 旁路 JSON 保存工具，并让 facade 委托该工具返回映射；`P0-M4` 新增不加载真实模型的 QwenPI_v3 复用 smoke 测试；`P0-M5` 新增 Stage1 7DoF action / 8D state smoke 配置；`P0-M8` 新增 LIBERO batch schema smoke 测试；`P0-M6` 新增 QwenOFT + MLP baseline smoke 配置；`P0-M7` 新增 future_tokens 消融配置；`P0-M9` 新增 checkpoint sidecar mapping 保存工具；`P0-M10` 新增 eval smoke preflight；`P0-M11` 新增实验矩阵和文档治理测试；`P0-M8-DATA` 对齐真实 LIBERO registry 的 8D state schema。

本阶段不修改 `QwenPI_v3.py`、`LayerwiseFM_ActionHeader.py` 或 `GR00T_ActionHeader.py` 主体逻辑。

当前未对 StarVLA 原始主体文件做需要 `STARFLOW_PATCH_BEGIN / END` 标记的 inline patch。

## Not Run
未运行真实模型加载、训练、评测或部署。
