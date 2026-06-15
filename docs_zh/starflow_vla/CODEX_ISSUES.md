# StarFlow-VLA Codex Issue 清单

版本：V0.1  
日期：2026-06-14  
说明：本文件用于后续拆成 GitHub Issues。本次只生成计划，不实现代码。

## Global Issue Execution Requirements

### Environment Stage

- P0-M0 ~ P0-M3 可在 Stage A 1×A100 40G lightweight validation 完成。
- P0-M4 ~ P0-M10 需要 Stage B 1×A100 40G target smoke validation 复验。
- P0-M11 可在 Stage A 完成，但涉及测试结果时必须引用 Stage B 记录。
- Stage A 1×A100 40G lightweight validation：文档、代码构建、仓库阅读、registry/config/import/mock 测试和轻量静态检查。
- Stage B 1×A100 40G target smoke validation：真实模型 P0 smoke、QwenPI_v3 reuse smoke、LayerwiseFM 7DoF forward/backward、single batch overfit、checkpoint save/load、predict_action shape、LIBERO eval smoke、future_tokens 消融 dry-run。
- Target verification required：任何训练、评测、部署、性能、显存、延迟和真实机器人结论。

### Implementation Record

完成任一 issue 后必须更新：

- `docs/starflow_vla/IMPLEMENTATION_LOG.md`
- 对应 `ACCEPTANCE_CHECKLIST.md` 项
- 必要时更新 `PATCH_MANIFEST.md`
- 必要时更新 `MODULE_MAPPING.md`

记录必须包含：

- 设计章节引用；
- 当前环境阶段：Stage A 1×A100 40G lightweight validation 或 Stage B 1×A100 40G target smoke validation；
- 实际修改文件；
- 执行步骤；
- 测试命令；
- 测试结果；
- 未运行测试的原因；
- 需要目标环境复验的内容；
- 与设计不一致的地方；
- 回滚方式。

Stage A 无法验证的训练、评测、部署或性能结论必须写明：`未在 Stage B 目标模型 smoke 或正式训练/评测环境验证，需在对应目标环境验证。`

所有 P0 issue 的完成标准不仅是文件修改完成，还必须形成可读的 Implementation Record。

## [P0-M0] Verify V4.6.2 design freeze

Priority: P0  
Type: docs  
Scope: design

### Goal

确认当前设计文档已冻结为 StarVLA-native framework 路线，并且 P0/P1/P2 边界一致。

### Files

- Add: `docs/starflow_vla/DESIGN_FREEZE_CHECK.md`
- Modify: none
- Do not modify: StarVLA model source

### Acceptance Criteria

- [ ] 文档不再把 StarFlowVLA 描述成外部项目。
- [ ] PerceiverAdapter 不作为 P0 必选。
- [ ] 显式 FlowCondition runtime 不作为 P0 必选。
- [ ] 14D action_mask 不作为 P0 阻断项。
- [ ] H2 表述为 `future_tokens + cross-DiT vs MLP/OFT/VLA_AdapterHeader baseline`。
- [ ] 允许修改 StarVLA 原文件：否。

### Tests

```bash
rg -n "models/qwen3_starvla_flow|H2 Action Token vs MLP Adapter|Perceiver.*P0|FlowCondition.*P0|14D.*P0" docs/ *.md
```

### Notes

这是所有实现 issue 的前置门。本 issue 的完成标准不仅是文件修改完成，还必须形成可读的 Implementation Record。

## [P0-M1] Lock StarVLA baseline version

Priority: P0  
Type: docs  
Scope: baseline

### Goal

记录 StarVLA branch、commit、package version、config schema 和 compatibility audit 规则。

### Files

- Add: `docs/starflow_vla/BASELINE_VERSION.md`
- Add: `docs/starflow_vla/UPSTREAM_COMPATIBILITY.md`
- Modify: none
- Do not modify: `.git/config`

### Acceptance Criteria

- [ ] 记录 branch `starVLA_dev`。
- [ ] 记录 commit `42170b2a4df3877ccf6581948e2198d37c363c7f`。
- [ ] 记录 package version `starVLA 1.0.1`。
- [ ] 记录 config schema `version_id=0.21`。
- [ ] 说明后续上游更新必须先 compatibility audit。
- [ ] 允许修改 StarVLA 原文件：否。

### Tests

```bash
Get-Content .git/HEAD
Get-Content pyproject.toml
rg -n "version_id" examples/LIBERO/train_files/starvla_cotrain_libero.yaml
```

### Notes

不要为了解决 safe.directory 提示修改用户全局 git 配置；可通过只读 `.git/HEAD` 和 refs 记录版本。本 issue 的完成标准不仅是文件修改完成，还必须形成可读的 Implementation Record。

## [P0-M2] Add StarFlowVLA framework entry

Priority: P0  
Type: feature  
Scope: framework

### Goal

新增 StarFlowVLA framework 入口，使其通过 `framework.name=StarFlowVLA` 构建。

### Files

- Add: `starVLA/model/framework/VLM4A/StarFlowVLA.py`
- Modify: `starVLA/model/framework/VLM4A/__init__.py` only if auto-import does not discover the new module
- Do not modify: `QwenPI_v3.py` main forward logic

### Acceptance Criteria

Stage A 1×A100 40G lightweight validation:

- [ ] `FRAMEWORK_REGISTRY` 可找到 `StarFlowVLA`。
- [ ] `build_framework(cfg)` 可构建 `StarFlowVLA`。
- [ ] import / config parse 通过。
- [ ] StarFlowVLA 继承或委托 QwenPI_v3。
- [ ] 不复制 QwenPI_v3 主体 forward / predict_action。
- [ ] 不重复构建 Qwen3-VL、project_layers、LayerwiseFM。
- [ ] 允许修改 StarVLA 原文件：仅必要 import/registry 辅助，不改主逻辑。

Stage B 1×A100 40G target smoke validation 复验归属：

- [ ] single batch forward/backward pass 由 P0-M4 / P0-M5 验收。
- [ ] loss finite 由 P0-M5 验收。
- [ ] predict_action shape 由 P0-M5 / P0-M10 验收。

### Tests

```bash
python -c "from starVLA.model.framework.base_framework import build_framework; print('TODO: build StarFlowVLA config')"
pytest tests/test_starflow_vla_registry.py -q
```

### Notes

保持 facade + hook，不做 hard fork。本 issue 的完成标准不仅是文件修改完成，还必须形成可读的 Implementation Record。

## [P0-M3] Add starflow_mapping manifest

Priority: P0  
Type: feature / docs / test  
Scope: manifest

### Goal

记录抽象设计到 StarVLA-native runtime 的映射，并写入 checkpoint 或 checkpoint 旁路 JSON。

### Files

- Add: `starVLA/model/modules/starflow_vla/mapping.py`
- Add: `docs/starflow_vla/MODULE_MAPPING.md`
- Modify: checkpoint saving only if needed
- Do not modify: action head math

### Acceptance Criteria

- [ ] 可生成 `starflow_mapping.json`。
- [ ] 字段包含 framework_name、implementation_mode、base_framework、action_head、state_mode、adapter_mode、flow_condition_runtime、perceiver_enabled、num_target_vision_tokens、solver、num_inference_timesteps、patch_manifest_hash、starvla_commit、config_schema。
- [ ] JSON 可序列化。
- [ ] checkpoint 或旁路目录保存该 JSON。
- [ ] 允许修改 StarVLA 原文件：可选，优先旁路 JSON。

### Tests

```bash
python -m json.tool outputs/.../starflow_mapping.json
pytest tests/test_starflow_mapping.py -q
```

### Notes

manifest 是后续论文复现和部署候选筛选的硬门。本 issue 的完成标准不仅是文件修改完成，还必须形成可读的 Implementation Record。

## [P0-M4] Verify QwenPI_v3 reuse

Priority: P0  
Type: test  
Scope: framework

### Goal

确认 StarFlowVLA 真实复用 QwenPI_v3 的 qwen_vl_interface、project_layers、action_model、forward / predict_action 主路径。

### Files

- Add: `tests/test_starflow_vla_reuse.py`
- Modify: none
- Do not modify: `QwenPI_v3.py`

### Acceptance Criteria

- [ ] StarFlowVLA 不重复创建第二套 Qwen3-VL。
- [ ] StarFlowVLA 不复制 QwenPI_v3.py 大段代码。
- [ ] QwenPI_v3 baseline 仍可运行。
- [ ] state-to-instruction 默认路径可复用。
- [ ] 允许修改 StarVLA 原文件：否。

### Tests

```bash
pytest tests/test_starflow_vla_reuse.py -q
```

### Notes

这个 issue 防止架构退化成复制粘贴。本 issue 的完成标准不仅是文件修改完成，还必须形成可读的 Implementation Record。

## [P0-M5] Add Stage1 LayerwiseFM 7DoF train smoke config

Priority: P0  
Type: config / test  
Scope: training

### Goal

跑通 StarFlowVLA + QwenPI_v3 reuse + LayerwiseFM 的单臂 7DoF 最小训练闭环。

### Files

- Add: `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml`
- Modify: none unless config schema requires registration
- Do not modify: LayerwiseFM loss logic for 14D mask

### Acceptance Criteria

- [ ] `action_dim=7`。
- [ ] loss finite。
- [ ] single batch forward/backward pass。
- [ ] single batch overfit pass。
- [ ] checkpoint save/load pass。
- [ ] predict_action shape 正确。
- [ ] 允许修改 StarVLA 原文件：否，优先配置化。

### Tests

```bash
accelerate launch starVLA/training/train_starvla_cotrain.py --config_yaml configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml
```

### Notes

不要启用 `max_action_dim=14 + action_mask`。本 issue 的完成标准不仅是文件修改完成，还必须形成可读的 Implementation Record。

## [P0-M6] Add MLP/OFT/VLA_AdapterHeader baseline config

Priority: P0  
Type: config / test  
Scope: baseline

### Goal

为 H2 提供 MLP/OFT/VLA_AdapterHeader baseline。

### Files

- Add: `configs/starflow_vla/stage2_mlp_baseline.yaml`
- Modify: none
- Do not modify: create new MLPAdapter runtime class

### Acceptance Criteria

- [ ] baseline 可 dry-run。
- [ ] single batch overfit pass。
- [ ] 日志记录 baseline 类型。
- [ ] 可与 future_tokens + cross-DiT 公平对照。
- [ ] 允许修改 StarVLA 原文件：否。

### Tests

```bash
accelerate launch starVLA/training/train_starvla_cotrain.py --config_yaml configs/starflow_vla/stage2_mlp_baseline.yaml
```

### Notes

`MLPAdapter` 是文档抽象，P0 映射为已有 StarVLA baseline。本 issue 的完成标准不仅是文件修改完成，还必须形成可读的 Implementation Record。

## [P0-M7] Add future_tokens ablation configs

Priority: P0  
Type: config / experiment  
Scope: action_head

### Goal

支持 `num_target_vision_tokens=0/8/16/32/64` 消融。

### Files

- Add: `configs/starflow_vla/stage3_future_token_ablation.yaml`
- Modify: none unless zero-token edge case requires small guard
- Do not modify: P2 PerceiverAdapter

### Acceptance Criteria

- [ ] 5 个 token 数配置均可解析。
- [ ] forward pass 通过。
- [ ] single batch overfit 通过。
- [ ] 日志记录 `num_target_vision_tokens`。
- [ ] manifest 记录 `adapter_mode=future_token_cross_dit`。
- [ ] 允许修改 StarVLA 原文件：仅允许 zero-token edge guard。

### Tests

```bash
# TODO: replace with project sweep runner
for n in 0 8 16 32 64; do echo "run num_target_vision_tokens=$n"; done
```

### Notes

这是 H2 的核心 issue。本 issue 的完成标准不仅是文件修改完成，还必须形成可读的 Implementation Record。

## [P0-M8] Verify LIBERO minimal data loop

Priority: P0  
Type: test  
Scope: data

### Goal

跑通 LIBERO 数据到训练 batch 的最小链路。

### Files

- Add: optional `tests/test_starflow_libero_batch.py`
- Modify: optional StarFlowVLA config
- Do not modify: RoboCasa / RoboTwin adapters for this issue

### Acceptance Criteria

- [ ] 数据转换或读取成功。
- [ ] batch 含 image / instruction / state / action。
- [ ] `action_dim=7`。
- [ ] 无 NaN/Inf。
- [ ] schema test 通过。
- [ ] 允许修改 StarVLA 原文件：否。

### Tests

```bash
python -c "TODO: load one LIBERO batch through StarVLA dataloader"
```

### Notes

RoboCasa / RoboTwin 完整接入放 P1。本 issue 的完成标准不仅是文件修改完成，还必须形成可读的 Implementation Record。

## [P0-M9] Add checkpoint manifest and resume smoke

Priority: P0  
Type: feature / test  
Scope: checkpoint

### Goal

保证 checkpoint 可恢复、可追踪，并包含 starflow_mapping。

### Files

- Add: optional `docs/starflow_vla/CHECKPOINT_MANIFEST.md`
- Modify: checkpoint save path only if needed
- Do not modify: model architecture

### Acceptance Criteria

- [x] checkpoint 包含 model / optimizer / scaler / config / starflow_mapping。
- [ ] resume 后 100 step 内 loss 偏差 <1%。
- [ ] `patch_manifest_hash` 可记录。
- [ ] 允许修改 StarVLA 原文件：可选，优先旁路 JSON。

### Tests

```bash
pytest tests/test_starflow_checkpoint_resume.py -q
```

### Notes

如果嵌入 checkpoint 风险高，先保存 checkpoint 旁路 JSON。本 issue 的完成标准不仅是文件修改完成，还必须形成可读的 Implementation Record。

## [P0-M10] Run LIBERO eval smoke from checkpoint

Priority: P0  
Type: test  
Scope: evaluation

### Goal

从 P0 checkpoint 跑通 LIBERO eval smoke。

### Files

- Add: optional `docs/starflow_vla/EVAL_SMOKE.md`
- Modify: optional eval config
- Do not modify: RoboCasa / RoboTwin eval for this issue

### Acceptance Criteria

- [ ] LIBERO eval smoke pass。
- [ ] 输出 success_rate。
- [ ] 输出 failure category。
- [ ] 报告包含 checkpoint hash、config hash、data version、starflow_mapping。
- [ ] 允许修改 StarVLA 原文件：否。

### Tests

```bash
bash examples/LIBERO/eval_files/run_policy_server.sh
bash examples/LIBERO/eval_files/eval_libero.sh
```

### Notes

先跑最小任务，不做完整 benchmark matrix。本 issue 的完成标准不仅是文件修改完成，还必须形成可读的 Implementation Record。

## [P0-M11] Maintain module mapping and patch manifest docs

Priority: P0  
Type: docs  
Scope: governance

### Goal

保证所有新增文件和 StarVLA 原文件 patch 可审计。

### Files

- Add: `docs/starflow_vla/MODULE_MAPPING.md`
- Add: `docs/starflow_vla/PATCH_MANIFEST.md`
- Add: `docs/starflow_vla/EXPERIMENT_MATRIX.md`
- Add: `docs/starflow_vla/UPSTREAM_COMPATIBILITY.md`
- Modify: none

### Acceptance Criteria

- [ ] 每个新增文件都有记录。
- [ ] 每个修改 StarVLA 原文件的 patch 都有记录。
- [ ] 后续代码实现中原文件 patch 使用 `STARFLOW_PATCH_BEGIN / END`。
- [ ] checkpoint manifest 中的 patch hash 可追溯。
- [ ] 允许修改 StarVLA 原文件：否。

### Tests

```bash
Test-Path docs/starflow_vla/MODULE_MAPPING.md
Test-Path docs/starflow_vla/PATCH_MANIFEST.md
```

### Notes

这是 P0 的工程治理 issue。本 issue 的完成标准不仅是文件修改完成，还必须形成可读的 Implementation Record。

## P1 issue backlog

- [P1-M1] Add continuous_head state path
- [P1-M2] Add max_action_dim=14 + action_mask support
- [P1-M3] Add masked flow loss
- [P1-M4] Complete solver manifest
- [P1-M5] Complete RoboCasa / RoboTwin integration
- [P1-M6] Add latency profiling
- [P1-M7] Automate upstream compatibility audit

这些 issue 不阻断 P0，必须在 issue 描述中写明前置条件、是否影响 checkpoint、是否修改 StarVLA 原文件。

## P2 issue backlog

- [P2-M1] Add PerceiverAdapter
- [P2-M2] Add Observation Geometry Adapter Interface for world-model mobile manipulation planning bridge
- [P2-M3] Add explicit FlowCondition runtime dataclass
- [P2-M4] Add advanced condition injection ablation
- [P2-M5] Add multi-view long-token compression
- [P2-M6] Add bimanual coordination loss / advanced embodiment token
- [P2-M7] Add advanced real-robot deployment extensions

这些 issue 是 advanced，不允许混入 P0 issue。

## [P2-M2] Observation Geometry Adapter Interface for World-Model Mobile Manipulation Planning Bridge

Priority: P2  
Type: optional / research extension  
Scope: observation geometry / world-model mobile manipulation planning bridge

### Goal

为后续《基于世界模型的移动操作规划与决策框架研究》预留 VGGT / RGB-3D geometry token 接口，使该研究产生的 world tokens 或 VGGT geometry tokens 能作为额外 condition 输入 StarFlow-VLA policy。

### Files

- Optional Add in P2: `starVLA/model/modules/starflow_vla/geometry_adapter.py`
- Optional Add in P2: `starVLA/model/modules/starflow_vla/geometry_fusion.py`
- Optional Modify in P2: StarFlowVLA config only
- Do not modify in P0/P1: `QwenPI_v3.py`
- Do not modify in P0/P1: `LayerwiseFM_ActionHeader.py`

### Acceptance Criteria

- [ ] P0/P1 不依赖 VGGT。
- [ ] `observation_geometry.enabled=false` 时 StarFlow-VLA 行为与 RGB-only pathway 一致。
- [ ] VGGT / geometry token 只作为 P2 optional extension。
- [ ] 《基于世界模型的移动操作规划与决策框架研究》可将 world tokens / geometry tokens 注入 StarFlow-VLA policy condition。
- [ ] 不声称已完成 VGGT 训练或评测，除非有真实 Implementation Record。

### Tests

```bash
rg -n "VGGT|geometry_fusion|observation_geometry|GeometryAdapter" docs/starflow_vla docs/design
```

### Notes

该 issue 不阻断 P0/P1。StarFlow-VLA 保留接口，《基于世界模型的移动操作规划与决策框架研究》完成主线研究。

## Algorithm Optimization Issues

以下 issue 用于补充 H2-a / H2-b 算法优化方向。它们不代表代码已经实现，也不代表训练、评测或部署已经完成。所有未完成指标必须显式标注 `[待 Stage B A100 复验]`、`[待 LIBERO eval]` 或 `[待 RoboCasa / RoboTwin eval]`。

## [P0-M7a] Future Tokens Planning Slot Dry-run

Priority: P0  
Type: config/docs/test  
Scope: algorithm optimization

### Goal

把 `future_tokens` / `num_target_vision_tokens` 作为动作条件 token 预算和规划槽位容量变量，建立 `0/8/16/32/64` 五组配置与 manifest 字段，为 H2-a 提供最小可执行入口。

### Files

- Add/modify: optional `configs/starflow_vla/ablations/future_tokens_{0,8,16,32,64}.yaml`
- Modify: checkpoint / report manifest schema only if needed
- Modify: docs/starflow_vla/ALGORITHM_OPTIMIZATION_PLAN.md
- Do not modify: QwenPI_v3 / LayerwiseFM / GR00T 主体逻辑

### Acceptance Criteria

- [ ] `num_target_vision_tokens=0/8/16/32/64` 配置存在或在 issue 中明确列出。
- [ ] Stage A config parse / build_framework dry-run pass。
- [ ] `starflow_mapping` 记录 `num_target_vision_tokens`、`adapter_mode=future_token_cross_dit`、`state_mode`、`action_dim`、`action_horizon`。
- [ ] 至少一条 Implementation Record 记录本 issue 的 Stage A / Stage B 边界。
- [ ] P0 不要求完整 LIBERO full success rate。
- [ ] Stage B 至少一组 token 数完成 single batch overfit 后，记录 loss finite/NaN、peak memory、latency、action smoothness。

### Tests

```bash
rg -n "num_target_vision_tokens|future_tokens|starflow_mapping" configs docs/starflow_vla
```

### Notes

`num_target_vision_tokens=0` 可能暴露空 token 边界问题。若必须修改 StarVLA 原文件，需另开 patch issue 并使用 `STARFLOW_PATCH_BEGIN / END` 标记。

## [P1-M1] Future Tokens Ablation on LIBERO

Priority: P1  
Type: experiment  
Scope: algorithm optimization

### Goal

在 LIBERO full split 上比较 `0/8/16/32/64` planning slots 对 success、loss、显存、延迟和动作平滑性的影响。

### Acceptance Criteria

- [ ] 五组配置均能进入 LIBERO 训练/评测脚本。
- [ ] 报告包含 success_rate、loss curve、single batch overfit speed、action chunk smoothness、peak memory、inference latency、按任务长度分组 success。
- [ ] 未完成指标写 `[待 LIBERO eval]`。
- [ ] 不使用 Stage A dry-run 结果冒充 LIBERO full 结果。

### Tests

```bash
rg -n "E-H2a|future_tokens_0|future_tokens_8|future_tokens_16|future_tokens_32|future_tokens_64" docs/starflow_vla
```

## [P1-M2] State Conditioning Path Comparison

Priority: P1  
Type: config/experiment  
Scope: algorithm optimization

### Goal

比较 `state-to-instruction` 与 `continuous state encoder` 两条状态条件注入路径，回答本体状态通过语言 token 还是连续 action head 条件更适合 Flow Matching VLA。

### Files

- Add/modify: optional `configs/starflow_vla/state/discretized_instruction.yaml`
- Add/modify: optional `configs/starflow_vla/state/continuous_head.yaml`
- Do not modify: QwenPI_v3 / LayerwiseFM / GR00T 主体逻辑；优先 wrapper/hook/config

### Acceptance Criteria

- [ ] `state_mode=discretized_instruction` baseline 记录完整。
- [ ] `state_mode=continuous_head` 配置存在或列入 P1 待实现清单。
- [ ] 报告包含 LIBERO success、state-sensitive task success、long-horizon success、loss curve、overfit speed、smoothness、latency、state noise robustness。
- [ ] 未完成指标写 `[待 Stage B A100 复验]`。

### Tests

```bash
rg -n "state_mode|discretized_instruction|continuous_head|state_encoder" docs/starflow_vla
```

## [P2-M1] Hybrid Gated State Conditioning

Priority: P2  
Type: advanced experiment  
Scope: algorithm optimization

### Goal

新增 `hybrid_gated` 状态条件路径，同时使用 instruction state 与 continuous state，并通过 gate 控制语义对齐与控制精度之间的权衡。

### Acceptance Criteria

- [ ] `state_mode=hybrid_gated` 可选启用，默认关闭。
- [ ] 不影响 P0/P1 checkpoint load。
- [ ] 报告包含 cross benchmark drop、missing state robustness、state noise robustness、long-horizon success、latency。
- [ ] 未完成指标写 `[待 RoboCasa / RoboTwin eval]`。

### Tests

```bash
rg -n "hybrid_gated|missing_state|state_noise|E-H2b" docs/starflow_vla
```
