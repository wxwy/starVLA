# StarFlow-VLA P0 实施计划

版本：V0.1  
日期：2026-06-14  
范围：仅 P0 最小闭环实施计划，不包含 P1/P2 advanced 功能。

## Milestone 0: 文档冻结与基线锁定

**目标**

确认 V4.6.2 文档、StarVLA baseline、P0/P1/P2 边界已经冻结，后续实现不再讨论路线方向。

**包含任务**

- P0-M0：V4.6.2 文档冻结检查
- P0-M1：StarVLA 基线锁定与分支初始化
- P0-M11：文档与 patch 管理的初始骨架

**预计产物**

- `docs/starflow_vla/DESIGN_FREEZE_CHECK.md`
- `docs/starflow_vla/BASELINE_VERSION.md`
- `docs/starflow_vla/UPSTREAM_COMPATIBILITY.md`
- `docs/starflow_vla/PATCH_MANIFEST.md`
- `docs/starflow_vla/MODULE_MAPPING.md`

**建议执行顺序**

1. 检查最终设计文档是否为 `V4.6.2 Implementation Trace Patch`。
2. 记录 StarVLA branch、commit、package version、config schema。
3. 创建 `docs/starflow_vla/` 文档骨架。
4. 明确上游更新必须走 compatibility audit。

**验收标准**

- H2 统一为 `future_tokens + cross-DiT vs MLP/OFT/VLA_AdapterHeader baseline`。
- P0 不包含 PerceiverAdapter、显式 FlowCondition runtime、14D action_mask。
- baseline 记录包含 `starVLA_dev`、`42170b2a4df3877ccf6581948e2198d37c363c7f`、`starVLA 1.0.1`、`version_id=0.21`。

**失败回滚方式**

若文档或 baseline 无法确认，停止代码实现，回退到 V4.6.2 文档重新审查。

## Milestone 1: StarFlowVLA framework smoke

**目标**

新增 StarFlowVLA framework 入口，并验证它是 StarVLA-native 路线：对外可注册，对内继承/委托 QwenPI_v3。

**包含任务**

- P0-M2：新增 StarFlowVLA framework 入口任务
- P0-M3：新增 starflow_mapping manifest
- P0-M4：QwenPI_v3 reuse smoke

**预计产物**

- `starVLA/model/framework/VLM4A/StarFlowVLA.py`
- `starVLA/model/modules/starflow_vla/mapping.py`
- `docs/starflow_vla/MODULE_MAPPING.md`
- `starflow_mapping.json` schema 草案

**建议执行顺序**

1. 新增 StarFlowVLA framework 入口，不复制 QwenPI_v3 主体 forward。
2. 接入 `FRAMEWORK_REGISTRY.register("StarFlowVLA")`。
3. 验证 `build_framework(cfg)` 可构建 StarFlowVLA。
4. 增加 `describe_starflow_mapping()` 或等价 manifest 构造方法。
5. 记录 QwenPI_v3 reuse smoke 的 Stage B 复验归属，实际 forward/backward 在 P0-M4/P0-M5 执行。

**验收标准**

- Stage A：`FRAMEWORK_REGISTRY` 可找到 `StarFlowVLA`。
- Stage A：`framework.name=StarFlowVLA` 可被 config 调用。
- Stage A：import / config parse / build_framework dry-run pass。
- Stage A：StarFlowVLA 继承或委托 QwenPI_v3。
- Stage A：StarFlowVLA 不复制 QwenPI_v3 大段主体逻辑。
- Stage B：single batch forward/backward、loss finite、predict_action shape、single batch overfit 由 P0-M4/P0-M5 复验。

**失败回滚方式**

删除新增 StarFlowVLA framework 和配置，保留原 QwenPI_v3 / QwenGR00T / QwenOFT baseline 不变。

## Milestone 2: LayerwiseFM 7DoF train smoke

**目标**

用 LIBERO small split 跑通 StarFlowVLA + QwenPI_v3 reuse + LayerwiseFM 的单臂 7DoF 最小训练路径。

**包含任务**

- P0-M5：LayerwiseFM 单臂 7DoF 训练闭环
- P0-M8：LIBERO 最小数据闭环

**预计产物**

- `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml`
- LIBERO batch schema smoke report
- finite loss log
- single batch overfit log

**建议执行顺序**

1. 以 `examples/LIBERO/train_files/starvla_cotrain_libero.yaml` 为参考创建 Stage1 config。
2. 固定 `action_dim=7`，不启用 `max_action_dim=14 + action_mask`。
3. 先加载一个 LIBERO batch，检查 image / instruction / state / action。
4. 运行 single batch forward/backward。
5. 运行 single batch overfit。

**验收标准**

- batch 含 image / instruction / state / action。
- `action_dim=7`。
- loss finite，无 NaN/Inf。
- single batch overfit pass。
- predict_action shape 正确。

**失败回滚方式**

回退到原始 LIBERO 示例配置；若 StarFlowVLA 入口失败，则临时用 QwenPI_v3 baseline 对照定位问题。

## Milestone 3: MLP baseline

**目标**

建立 H2 的 MLP/OFT/VLA_AdapterHeader baseline，确保 future_tokens + cross-DiT 有公平对照。

**包含任务**

- P0-M6：MLP/OFT/VLA_AdapterHeader baseline

**预计产物**

- `configs/starflow_vla/stage2_mlp_baseline.yaml`
- baseline dry-run log
- baseline single batch overfit log
- baseline checkpoint

**建议执行顺序**

1. 确认 `MLP_ActionHeader.py`、`VLA_AdapterHeader.py`、`QwenOFT.py` 可作为 baseline 来源。
2. 创建 Stage2 MLP baseline config。
3. 保持数据、batch、训练预算与 Stage1 可对照。
4. 运行 dry-run 和 single batch overfit。

**验收标准**

- MLP/OFT/VLA_AdapterHeader baseline 可 dry-run。
- single batch overfit pass。
- 日志记录 baseline 类型。
- 输出可与 future_tokens + cross-DiT 路线公平对照。

**失败回滚方式**

保留 Stage1 StarFlowVLA native 路线，单独修复 baseline config；不得因 baseline 问题改动 P0-M5 主闭环。

## Milestone 4: future_tokens 消融

**目标**

完成 `num_target_vision_tokens=0/8/16/32/64` 消融配置，为 H2 提供主线实验。

**包含任务**

- P0-M7：future_tokens + cross-DiT 消融

**预计产物**

- `configs/starflow_vla/stage3_future_token_ablation.yaml`
- 5 组 token 数配置
- ablation dry-run log
- checkpoint manifest 中的 `adapter_mode=future_token_cross_dit`

**建议执行顺序**

1. 从 Stage1 config 派生 future token ablation config。
2. 先跑默认 `num_target_vision_tokens=32`。
3. 再跑 `8/16/64`。
4. 最后检查 `0` token 边界，如失败则记录为需小 patch 的边界问题。

**验收标准**

- 所有配置可解析。
- forward pass 通过。
- single batch overfit 通过。
- 日志记录 `num_target_vision_tokens`。
- manifest 记录 `adapter_mode=future_token_cross_dit`。

**失败回滚方式**

回退默认 `num_target_vision_tokens=32`；若 `0` token 不兼容，先不阻断 H2 主线，单独登记边界 patch。

## Milestone 5: checkpoint + eval smoke

**目标**

确保训练产物可恢复、可追踪、可评测。

**包含任务**

- P0-M9：Checkpoint 与恢复
- P0-M10：最小评测闭环
- P0-M11：文档与 patch 管理完善

**预计产物**

- checkpoint
- config snapshot
- `starflow_mapping.json`
- `PATCH_MANIFEST.md`
- LIBERO eval smoke report

**建议执行顺序**

1. 在 checkpoint 旁保存 `starflow_mapping.json`。
2. 记录 `patch_manifest_hash`。
3. 测试 checkpoint save/load。
4. resume 后跑 100 step 对比 loss。
5. 启动 LIBERO eval smoke，输出 success_rate 和 failure category。

**验收标准**

- checkpoint 包含 model / optimizer / scaler / config / starflow_mapping。
- resume 后 100 step 内 loss 偏差 <1%。
- LIBERO eval smoke pass。
- 报告包含 checkpoint hash、config hash、data version、starflow_mapping。

**失败回滚方式**

若 checkpoint manifest 嵌入困难，先使用 checkpoint 旁路 JSON；若 eval smoke 失败，先回退到原 StarVLA LIBERO eval 流程定位接口差异。

## 算法优化补充：P0-M7a / H2-a / H2-b

本补充不改变 P0 主闭环顺序，只把两个算法优化方向纳入后续执行基线。当前任务只定义文档、配置和验收边界，不实现 StarFlowVLA 代码，不修改 QwenPI_v3 / LayerwiseFM / GR00T 主体逻辑。

### VGGT / Geometry Fusion 不进入 P0

P0 不接入 VGGT，不实现 RGB-3D fusion，不新增 geometry encoder 依赖。P0 仅保留 RGB / language / state 到 action chunk 的 Flow Matching VLA 最小闭环。VGGT 相关内容仅作为 P2 optional extension 和《基于世界模型的移动操作规划与决策框架研究》衔接接口。

### P0-M7a：Future Tokens Planning Slot Dry-run

**目标**

将 `future_tokens` / `num_target_vision_tokens` 作为动作条件 token 预算和规划槽位容量变量，形成 `0/8/16/32/64` 五组配置和 manifest 记录，为 H2-a 提供 P0 级最小实验入口。

**验收标准**

- Stage A：`num_target_vision_tokens=0/8/16/32/64` 配置存在或在 issue 中明确列出。
- Stage A：config parse / build_framework dry-run pass。
- Stage A：`starflow_mapping` 记录 `num_target_vision_tokens`、`adapter_mode=future_token_cross_dit`、`state_mode`、`action_dim`、`action_horizon`。
- Stage A：不修改 QwenPI_v3 / LayerwiseFM / GR00T 主体逻辑。
- Stage B：至少一组 token 数完成 single batch overfit，记录 loss finite/NaN、peak memory、latency、action smoothness。
- Stage B：LIBERO full success、RoboCasa/RoboTwin cross benchmark 不作为 P0-M7a 完成条件，归入 P1/P2。

### Milestone 4 补充验收口径

Milestone 4 原有 P0-M7 继续作为 future_tokens + cross-DiT 主线配置任务；P0-M7a 进一步要求把 `future_tokens` 明确写成算法变量，而不是只写成工程参数。

```text
Stage A：future_tokens 配置 / import / config parse / starflow_mapping dry-run pass。
Stage B：single batch overfit 只要求至少一组完成，完整 LIBERO/RoboCasa/RoboTwin 指标后移。
```

### H2-b 状态条件注入路径执行边界

- P0：默认 `state_mode=discretized_instruction`，复用 QwenPI_v3 state-to-instruction。
- P1：新增 `state_mode=continuous_head` 对照，复用或包装 action head `state_encoder`。
- P2：新增 `state_mode=hybrid_gated`，仅在 P1 结果显示单一路径存在短板时启用。
- 所有状态路径指标在未完成 Stage B 前写为 `[待 Stage B A100 复验]`，不得用本地 dry-run 冒充训练或评测结果。
