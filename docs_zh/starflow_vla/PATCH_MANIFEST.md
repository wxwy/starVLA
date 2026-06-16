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
| P0-M7 | config | `configs/starflow_vla/stage3_future_token_ablation.yaml` | added | 新增 future_tokens `0/8/16/32/64` 消融配置（历史记录；当前 P0/P1 可执行矩阵已收敛为 `0/16/32/64`） |
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
| P0-M5-STAGEB | docs | `ACCEPTANCE_CHECKLIST.md` | modified | 标记 Stage1 forward/backward、loss finite、single batch overfit 前置验证通过 |
| P0-M5-STAGEB | docs | `EXPERIMENT_MATRIX.md` | modified | 更新 Stage1 Stage B smoke 状态 |
| P0-M5-STAGEB | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 Stage B smoke 实施记录 |
| P0-M5-STAGEB | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| P0-M9-STAGEB | docs | `ACCEPTANCE_CHECKLIST.md` | modified | 标记 eval preflight 通过并保留 rollout 待验证 |
| P0-M9-STAGEB | docs | `EXPERIMENT_MATRIX.md` | modified | 记录 smoke checkpoint save/load 与 preflight 状态 |
| P0-M9-STAGEB | docs | `EVAL_SMOKE.md` | modified | 更新 checkpoint 与 mapping 当前状态 |
| P0-M9-STAGEB | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 checkpoint/eval preflight 实施记录 |
| P0-M9-STAGEB | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| P0-M6-STAGEB | docs | `ACCEPTANCE_CHECKLIST.md` | modified | 标记 MLP baseline single batch overfit 前置验证通过 |
| P0-M6-STAGEB | docs | `EXPERIMENT_MATRIX.md` | modified | 更新 MLP baseline Stage B smoke 状态 |
| P0-M6-STAGEB | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 MLP baseline Stage B smoke 实施记录 |
| P0-M6-STAGEB | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| P0-M7-STAGEB | docs | `ACCEPTANCE_CHECKLIST.md` | modified | 标记 future token Stage B overfit 前置验证通过 |
| P0-M7-STAGEB | docs | `EXPERIMENT_MATRIX.md` | modified | 更新 future token 5 组 Stage B smoke 状态 |
| P0-M7-STAGEB | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 future token Stage B smoke 实施记录 |
| P0-M7-STAGEB | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| P0-M10-STAGEB | docs | `ACCEPTANCE_CHECKLIST.md` | modified | 标记 LIBERO rollout smoke 和 success_rate 输出通过 |
| P0-M10-STAGEB | docs | `EXPERIMENT_MATRIX.md` | modified | 更新最小 LIBERO rollout smoke 状态 |
| P0-M10-STAGEB | docs | `EVAL_SMOKE.md` | modified | 记录 1 task × 1 trial rollout smoke 结果 |
| P0-M10-STAGEB | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 LIBERO rollout smoke 实施记录 |
| P0-M10-STAGEB | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| P0-M4-STAGEB | docs | `ACCEPTANCE_CHECKLIST.md` | modified | 标记 QwenPI_v3 baseline 真实 forward/backward 通过 |
| P0-M4-STAGEB | docs | `EXPERIMENT_MATRIX.md` | modified | 记录 QwenPI_v3 baseline compatibility smoke |
| P0-M4-STAGEB | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 QwenPI_v3 baseline Stage B 实施记录 |
| P0-M4-STAGEB | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| P0-M10-REPORT | code | `examples/LIBERO/eval_files/starflow_eval_report.py` | added | 新增 eval report 元数据/hash/聚合写入工具 |
| P0-M10-REPORT | code | `examples/LIBERO/eval_files/eval_libero.py` | modified | 在 episode 结束后、视频编码前写出 `eval_report.json` |
| P0-M10-REPORT | test | `tests/test_starflow_eval_report.py` | added | 验证 eval report 元数据提取与 JSON 输出 |
| P0-M10-REPORT | docs | `ACCEPTANCE_CHECKLIST.md` | modified | 标记 eval report failure category 和 metadata 字段通过 |
| P0-M10-REPORT | docs | `EXPERIMENT_MATRIX.md` | modified | 记录 eval report 产物状态 |
| P0-M10-REPORT | docs | `EVAL_SMOKE.md` | modified | 记录 eval report 路径与退出阶段写盘策略 |
| P0-M10-REPORT | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 eval report 实施记录 |
| P0-M10-REPORT | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| P0-M9-RESUME | code | `starVLA/training/train_starvla.py` | modified | lightweight checkpoint 新增 RNG state 保存/恢复 |
| P0-M9-RESUME | test | `tests/test_starflow_resume_100_steps.py` | added | bootstrap checkpoint + 50/50 resume 100-step consistency smoke |
| P0-M9-RESUME | docs | `ACCEPTANCE_CHECKLIST.md` | modified | 标记 resume 100 step 偏差 <1% 通过 |
| P0-M9-RESUME | docs | `EXPERIMENT_MATRIX.md` | modified | 记录 resume 100 step smoke 通过 |
| P0-M9-RESUME | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 resume consistency 实施记录 |
| P0-M9-RESUME | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| P0-M9-RESUME | docs | `examples/LIBERO/SESSION.md` | modified | 记录 bootstrap checkpoint 与 resume 100 step 结果 |
| P0-M9-SCALER | code | `starVLA/training/train_starvla.py` | modified | lightweight checkpoint 新增 scaler sidecar 与 checkpoint metadata 自动落盘 |
| P0-M9-SCALER | code | `starVLA/training/trainer_utils/trainer_tools.py` | modified | 抽出 lightweight scaler state 与 checkpoint metadata 保存工具 |
| P0-M9-SCALER | test | `tests/test_starflow_checkpoint_mapping.py` | modified | 增补 scaler placeholder / metadata 落盘与 scaler roundtrip 测试 |
| P0-M9-SCALER | docs | `ACCEPTANCE_CHECKLIST.md` | modified | 标记 checkpoint 含 model / optimizer / scaler / config / starflow_mapping 通过 |
| P0-M9-SCALER | docs | `EXPERIMENT_MATRIX.md` | modified | 记录 lightweight checkpoint 自动写入 scaler 与 metadata |
| P0-M9-SCALER | docs | `EVAL_SMOKE.md` | modified | 记录 `steps_1/scaler.pt` 占位文件状态 |
| P0-M9-SCALER | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 checkpoint scaler / metadata 实施记录 |
| P0-M9-SCALER | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| P0-M9-SCALER | docs | `examples/LIBERO/SESSION.md` | modified | 记录 `steps_1` scaler sidecar 与短测结果 |
| P0-GUARDRAIL | test | `tests/test_starflow_docs_governance.py` | modified | 新增 P0 guardrail 检查，防止 Perceiver / 显式 FlowCondition / 14D mask 回流为 P0 阻断项 |
| P0-GUARDRAIL | docs | `ACCEPTANCE_CHECKLIST.md` | modified | 标记 no Perceiver / FlowCondition runtime / 14D mask blocking P0 通过 |
| P0-GUARDRAIL | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 P0 guardrail 治理测试记录 |
| P0-GUARDRAIL | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| P0-GUARDRAIL | docs | `examples/LIBERO/SESSION.md` | modified | 记录 P0 guardrail 文档治理测试结果 |
| P0-M5-TRAINLOOP | code | `starVLA/training/train_starvla.py` | modified | 修复单卡未初始化分布式时 `prepare_data()` 无条件 `dist.barrier()` 的训练入口阻塞 |
| P0-M5-TRAINLOOP | docs | `EXPERIMENT_MATRIX.md` | modified | 记录 Stage1 真实 10 step 训练闭环结果与环境前提 |
| P0-M5-TRAINLOOP | docs | `IMPLEMENTATION_LOG.md` | modified | 追加真实训练主链路闭环记录 |
| P0-M5-TRAINLOOP | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| P0-M5-TRAINLOOP | docs | `examples/LIBERO/SESSION.md` | modified | 记录 `steps_10` / `final_model` 训练产物和环境变量要求 |
| P0-M10-FULLTASK | docs | `EXPERIMENT_MATRIX.md` | modified | 记录 `steps_10` 的 `libero_goal` 全 10 task × 1 trial task sweep 结果 |
| P0-M10-FULLTASK | docs | `EVAL_SMOKE.md` | modified | 记录真实训练产物的全 task sweep 目录、成功率和 failure category |
| P0-M10-FULLTASK | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 `steps_10` 全 task sweep 评测记录 |
| P0-M10-FULLTASK | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| P0-M10-FULLTASK | docs | `examples/LIBERO/SESSION.md` | modified | 记录 `steps_10` 的 `libero_goal` 全 task sweep 结果 |
| P0-M10-REGRESSION | code | `examples/LIBERO/eval_files/run_starflow_eval_regression.sh` | added | 新增训练产物到 policy server 到 LIBERO eval 的一键快速回归入口 |
| P0-M10-REGRESSION | code | `examples/LIBERO/eval_files/eval_libero.sh` | modified | 补充 `MAX_TASKS` 透传，支持 quick regression |
| P0-M10-REGRESSION | test | `tests/test_starflow_eval_preflight.py` | modified | 扩展回归脚本 shell 语法和 quick regression 接线检查 |
| P0-M10-REGRESSION | docs | `EVAL_SMOKE.md` | modified | 记录一键回归入口与 server 冷启动等待窗口 |
| P0-M10-REGRESSION | docs | `IMPLEMENTATION_LOG.md` | modified | 追加回归入口固化记录 |
| P0-M10-REGRESSION | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |
| P0-M10-REGRESSION | docs | `examples/LIBERO/SESSION.md` | modified | 记录一键回归入口和实际运行观察 |
| P0-M10-COLDSTART | code | `deployment/model_server/server_policy.py` | modified | 将重依赖改为 `main()` 内懒导入，并补充冷启动阶段耗时日志 |
| P0-M10-COLDSTART | code | `deployment/model_server/policy_wrapper.py` | modified | 补充 `from_pretrained`、dtype/device 迁移和 metadata 阶段耗时日志 |
| P0-M10-COLDSTART | code | `starVLA/model/framework/base_framework.py` | modified | 补充 `read_mode_config`、`build_framework`、`load_model_weights` 耗时日志 |
| P0-M10-COLDSTART | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 policy server 冷启动定位记录 |
| P0-M10-COLDSTART | docs | `examples/LIBERO/SESSION.md` | modified | 记录冷启动阶段拆解结果 |
| P0-M5-TRAINREADY | code | `examples/LIBERO/train_files/run_starflow_train_ready.sh` | added | 新增训练就绪启动器，固化已验证的 Stage1 训练前提 |
| P0-M5-TRAINREADY | test | `tests/test_starflow_train_ready.py` | added | 新增训练就绪脚本的语法和关键默认值回归检查 |
| P0-M5-TRAINREADY | docs | `IMPLEMENTATION_LOG.md` | modified | 追加训练就绪启动器记录 |
| P0-M5-TRAINREADY | docs | `examples/LIBERO/SESSION.md` | modified | 记录训练就绪启动器默认值 |
| DOC-Matrix-Cleanup | docs | `DESIGN.md` | modified | 统一 H2-a/H2-b 实验编号、future_tokens 取值、MLP baseline 定位和当前 P0/P1 覆盖边界 |
| DOC-Matrix-Cleanup | docs | `ALGORITHM_OPTIMIZATION_PLAN.md` | modified | 统一 H2-a/H2-b 实验编号、future_tokens 取值和当前 P0/P1 覆盖边界 |
| DOC-Matrix-Cleanup | docs | `EXPERIMENT_MATRIX.md` | modified | 重构为 Engineering / Algorithm / Appendix / Coverage 四张表，统一口径并修正可执行命令 |
| DOC-Matrix-Cleanup | docs | `ACCEPTANCE_CHECKLIST.md` | modified | 修正 future_tokens 验收口径为 0/16/32/64 |
| DOC-Matrix-Cleanup | docs | `P0_IMPLEMENTATION_PLAN.md` | modified | 修正 future_tokens 执行口径为 0/16/32/64 |
| DOC-Matrix-Cleanup | docs | `CODEX_ISSUES.md` | modified | 修正 future_tokens 执行口径为 0/16/32/64 |
| DOC-Matrix-Cleanup | docs | `CODEX_EXECUTION_GUIDE.md` | modified | 修正 future_tokens 执行口径为 0/16/32/64 |
| DOC-Matrix-Cleanup | docs | `MODULE_MAPPING.md` | modified | 修正 future_tokens 映射路径和 MLP baseline 定位 |
| DOC-Matrix-Cleanup | docs | `IMPLEMENTATION_LOG.md` | modified | 追加 DOC-Matrix-Cleanup 记录 |
| DOC-Matrix-Cleanup | docs | `PATCH_MANIFEST.md` | modified | 追加本记录 |


| DOC-H1-ACT-Downgrade | docs | `DESIGN.md`, `EXPERIMENT_MATRIX.md`, `ALGORITHM_OPTIMIZATION_PLAN.md`, `ACCEPTANCE_CHECKLIST.md`, `IMPLEMENTATION_LOG.md` | modified | 将 H1 Flow Matching vs ACT 降级为后续完整论文扩展 / optional baseline，ACT 不进入当前 P0/P1 必跑矩阵，当前主线收敛为 H2/H2-a |

## Source Code Patches
`P0-M2` 新增 StarFlowVLA facade 文件；`P0-M3` 新增独立 mapping schema / 旁路 JSON 保存工具，并让 facade 委托该工具返回映射；`P0-M4` 新增不加载真实模型的 QwenPI_v3 复用 smoke 测试；`P0-M5` 新增 Stage1 7DoF action / 8D state smoke 配置；`P0-M8` 新增 LIBERO batch schema smoke 测试；`P0-M6` 新增 QwenOFT + MLP baseline smoke 配置；`P0-M7` 新增 future_tokens 消融配置；`P0-M9` 新增 checkpoint sidecar mapping 保存工具；`P0-M10` 新增 eval smoke preflight；`P0-M11` 新增实验矩阵和文档治理测试；`P0-M8-DATA` 对齐真实 LIBERO registry 的 8D state schema；`P0-M5-STAGEB` 记录真实 Stage1 smoke 验证结果；`P0-M9-STAGEB` 记录 smoke checkpoint save/load 与 eval preflight；`P0-M6-STAGEB` 记录 MLP baseline Stage B smoke 验证结果；`P0-M7-STAGEB` 记录 future token 5 组 Stage B smoke 验证结果；`P0-M10-STAGEB` 记录最小 LIBERO rollout smoke；`P0-M4-STAGEB` 记录 QwenPI_v3 baseline compatibility smoke；`P0-M10-REPORT` 新增 eval report helper，并将报告写入前移到 episode 结束后、视频编码前；`P0-M9-RESUME` 为 lightweight checkpoint 增加 RNG state，并跑通 bootstrap checkpoint + 100-step resume consistency smoke；`P0-M9-SCALER` 为 lightweight checkpoint 增加 scaler sidecar、config/mapping 自动落盘，并补齐 `steps_1/scaler.pt`；`P0-GUARDRAIL` 将 Perceiver / 显式 FlowCondition / 14D mask 的 P0 边界固化为可执行文档治理测试；`P0-M5-TRAINLOOP` 修复单卡 barrier 阻塞，并跑通 `train_starvla.py` 的真实 10 step 训练闭环；`P0-M10-FULLTASK` 追加 `steps_10` 在 `libero_goal` 上的全 10 task × 1 trial task sweep；`P0-M10-REGRESSION` 固化训练产物到 policy server 到单 episode eval 的一键快速回归入口；`P0-M10-COLDSTART` 为 policy server 冷启动增加阶段耗时日志并确认 import/build 才是首要瓶颈；`P0-M5-TRAINREADY` 固化训练就绪默认值，减少手工拼接环境变量的时间。

本阶段不修改 `QwenPI_v3.py`、`LayerwiseFM_ActionHeader.py` 或 `GR00T_ActionHeader.py` 主体逻辑。

当前未对 StarVLA 原始主体文件做需要 `STARFLOW_PATCH_BEGIN / END` 标记的 inline patch。

## Not Run
未运行真实模型加载、训练、评测或部署。
