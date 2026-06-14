# StarFlow-VLA 验收清单

版本：V0.1  
日期：2026-06-14  
说明：本清单用于 StarFlow-VLA P0/P1/P2 实施验收。

## P0 验收清单：最小闭环门禁

- [x] V4.6.2 设计冻结检查通过。
- [x] StarVLA baseline version 已记录：branch、commit、package version、config schema、审计日期。
- [x] `docs_zh/starflow_vla/BASELINE_VERSION.md` exists。
- [x] `docs_zh/starflow_vla/UPSTREAM_COMPATIBILITY.md` exists。
- [x] StarFlowVLA framework registry build pass。
- [x] `framework.name=StarFlowVLA` 可被 config 调用。
- [x] QwenPI_v3 reuse documented。
- [x] StarFlowVLA 不复制 QwenPI_v3 主体 forward / predict_action。
- [ ] QwenPI_v3 baseline 仍可运行。
- [x] LayerwiseFM 7DoF single batch forward/backward pass。
- [x] loss finite。
- [x] single batch overfit pass。
- [x] `action_dim=7` 为 P0 默认配置。
- [x] 不强制启用 `max_action_dim=14 + action_mask`。
- [x] LIBERO minimal batch schema pass。
- [x] batch 含 image / instruction / state / action。
- [x] MLP/OFT/VLA_AdapterHeader baseline dry-run pass。
- [x] MLP/OFT/VLA_AdapterHeader baseline single batch overfit pass。
- [x] future_tokens ablation configs `0/8/16/32/64` pass。
- [ ] `num_target_vision_tokens` 写入日志。
- [x] `adapter_mode=future_token_cross_dit` 写入 manifest。
- [x] `starflow_mapping.json` saved in checkpoint 或 checkpoint 旁路目录。
- [x] `starflow_mapping.json` 可 JSON 序列化。
- [ ] checkpoint 包含 model / optimizer / scaler / config / starflow_mapping。
- [ ] resume 后 100 step 内 loss 偏差 <1%。
- [x] `patch_manifest_hash` 可记录。
- [x] LIBERO eval preflight pass。
- [ ] LIBERO rollout eval smoke pass。
- [ ] eval report 输出 success_rate。
- [ ] eval report 输出 failure category。
- [ ] eval report 包含 checkpoint hash、config hash、data version、starflow_mapping。
- [x] `MODULE_MAPPING.md` exists。
- [x] `PATCH_MANIFEST.md` exists。
- [x] `EXPERIMENT_MATRIX.md` exists。
- [ ] no Perceiver / FlowCondition runtime / 14D mask blocking P0。

## P1 验收清单：增强项，不阻断 P0

- [ ] `state_mode=continuous_head` 可 dry-run。
- [ ] continuous_head state path 与 `discretized_instruction` 有对照报告。
- [ ] `max_action_dim=14 + action_mask` mixed batch shape test pass。
- [ ] masked flow loss finite。
- [ ] 单臂样本后 7 维不参与 loss。
- [ ] mask denominator clamp pass。
- [ ] solver manifest 记录 solver type、num_inference_timesteps、latency profile。
- [ ] RoboCasa train / eval smoke pass。
- [ ] RoboTwin train / eval smoke pass。
- [ ] latency profiling 输出 p50 / p95 / p99。
- [ ] upstream compatibility audit automation 可运行。
- [ ] P1 配置不会改变 P0 默认配置。
- [ ] P1 checkpoint 与 P0 checkpoint 的兼容性说明已记录。

## P2 验收清单：Advanced，不阻断 P0/P1

- [ ] PerceiverAdapter dry-run pass。
- [ ] PerceiverAdapter 可通过 `perceiver_enabled` 开关启用/关闭。
- [ ] 显式 FlowCondition runtime dataclass 可选启用。
- [ ] P0 不依赖显式 FlowCondition runtime。
- [ ] 高级 Condition Injection 消融至少一组配置可 dry-run。
- [ ] 多视角长 token compression 输出 token 数、显存、吞吐对照。
- [ ] 双臂 coordination loss 可开关。
- [ ] advanced embodiment token 不影响 7DoF P0 checkpoint load。
- [ ] 真实机器人 shadow mode pass。
- [ ] 真实机器人 action replay pass。
- [ ] 低速闭环 safety gate pass。

## P2 / 与《基于世界模型的移动操作规划与决策框架研究》衔接验收项：Observation Geometry Adapter

- [ ] VGGT 不作为 P0 验收项。
- [ ] VGGT 不作为 P1 验收项。
- [ ] `observation_geometry.enabled=false` 为默认值。
- [ ] 文档明确 VGGT 是本项目 P2 optional extension / 与《基于世界模型的移动操作规划与决策框架研究》的接口预留。
- [ ] 文档明确《基于世界模型的移动操作规划与决策框架研究》将 VGGT 作为 geometric world state encoder。
- [ ] 未实现 VGGT 时不得声称完成 RGB-3D fusion 训练或评测。

## 禁止项检查

- [ ] 不把 StarFlowVLA 写成外部项目。
- [ ] 不复制 QwenPI_v3.py 形成大体重复文件。
- [ ] 不在 P0 中要求 PerceiverAdapter。
- [ ] 不在 P0 中要求显式 FlowCondition runtime dataclass。
- [ ] 不在 P0 中要求 14D action_mask。
- [ ] 不把 H2 写回 “ActionTokenAdapter vs MLPAdapter” 的抽象口径。
- [x] 不运行长训练作为 P0 smoke 的前置条件。
- [ ] 不删除或重命名 StarVLA 现有文件。
- [ ] 不把 VGGT 写入 P0/P1 必选验收项。
- [ ] 不声称已完成 VGGT 接入、训练、评测或指标，除非有真实 Implementation Record。

## 算法优化验收项

### Future Tokens Planning Slot Optimization

- [x] `num_target_vision_tokens=0/8/16/32/64` 五组配置存在，或在 P0-M7a issue 中明确列出待创建路径。
- [x] 五组配置均可完成 config parse。
- [x] 每组配置都能写入或计划写入 `starflow_mapping`。
- [x] `starflow_mapping` 至少包含 `num_target_vision_tokens`、`adapter_mode`、`state_mode`、`action_dim`、`action_horizon`、StarVLA upstream commit。
- [ ] 至少一组 token 数完成 Stage B single batch overfit 后，记录 loss finite/NaN、peak memory、latency、action smoothness。
- [ ] 报告包含 success_rate、loss curve、memory、latency、task length success 或对应占位符。
- [ ] 未完成指标均标注 `[待 Stage B A100 复验]`、`[待 LIBERO eval]` 或 `[待 RoboCasa / RoboTwin eval]`。
- [ ] 不使用本地 dry-run 或 mock 结果冒充 LIBERO/RoboCasa/RoboTwin 指标。

### State Conditioning Path Optimization

- [ ] `state_mode=discretized_instruction` baseline 已记录。
- [ ] `state_mode=continuous_head` 配置存在，或作为 P1 待实现项记录。
- [ ] `state_mode=hybrid_gated` 仅作为 P2 advanced，默认不阻断 P0/P1。
- [ ] 报告包含 state-sensitive success、long-horizon success、loss curve、overfit speed、smoothness、cross benchmark drop、latency、state noise robustness 或对应占位符。
- [ ] 未完成指标均标注 `[待 Stage B A100 复验]`。
- [ ] 后续实现优先使用配置、wrapper、hook 和模块化替代，不改写 QwenPI_v3 / LayerwiseFM / GR00T 主体逻辑。

## 实施记录验收项

- [ ] `CODEX_EXECUTION_GUIDE.md` exists。
- [ ] `IMPLEMENTATION_LOG.md` exists。
- [ ] 每个 P0 issue 都有 Implementation Record。
- [ ] 每个 Implementation Record 都包含 Design Reference。
- [ ] 每个 Implementation Record 都包含 Environment 边界说明。
- [ ] 每个 Implementation Record 都标注 Stage A / Stage B。
- [ ] 本地未验证内容均标注“未在当前本地环境验证，需在目标训练/部署环境验证”。
- [ ] 每个代码改动都能追溯到 `TASK_BREAKDOWN.md` / `CODEX_ISSUES.md`。
- [ ] 每个测试命令都有结果记录：Pass / Fail / Not run / Need target verification。
- [ ] 未运行测试均说明原因。
- [ ] 需要 A100 复验的任务已标注。
- [ ] 修改 StarVLA 原文件时同步更新 `PATCH_MANIFEST.md`。
- [ ] 新增 StarFlow-VLA 模块时同步更新 `MODULE_MAPPING.md`。

## 人工确认项

- [ ] 当前文档最终应落到 `E:\projects\starVLA\docs\starflow_vla\` 还是保持在当前工作区 `C:\Users\fast_\Desktop\xx\docs\starflow_vla\`。
- [ ] StarFlowVLA 实现分支名称。
- [ ] LIBERO small split 的本地数据路径。
- [ ] 是否允许为了 git 命令添加 safe.directory。
- [ ] Stage A 环境已确认：默认 1×A100 40G lightweight validation。
- [ ] 当前 Codex 本地文档编辑路径与 StarVLA 目标仓库路径已确认。
- [ ] 当前环境是否具备 GPU / 数据 / 训练依赖已确认。
- [ ] Stage B 环境已确认：默认 1×A100 40G target smoke validation。
- [ ] 哪些测试只允许在目标训练环境运行已确认。
- [ ] Stage A 与 Stage B 的区别已确认：二者默认均为 1×A100 40G，区别是验证强度，不是硬件差异。
