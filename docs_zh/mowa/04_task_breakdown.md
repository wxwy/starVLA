# MoWA 任务拆解

任务卡继承 `02_detailed_design.md` 的 Implementation Task、Ablation Task 与 Risk Item。本文件只拆解执行任务，不启动模型、数据或训练代码实现。

## 当前落实顺序（2026-07-05）

本节是当前 checkout 的执行准则。后续推进按本顺序落实，不因单次对话里的“可以训练”“先看命名”等临时话题改变优先级；临时话题只记录为待办或兼容性约束，除非用户明确要求修改本落实顺序。

| 顺序 | 工作包 | 当前状态 | 下一步可执行项 | 不做事项 / Gate |
|---|---|---|---|---|
| 1 | E-001 paired baseline/MoWA 训练前一致性 | StarFlow ft0 baseline candidate 与 MoWA launch candidate 已存在，launch guard 物理阻断已验证。 | 补齐 baseline/MoWA runtime symmetry checker：只允许 MoWA bridge、future labels、feature source 等字段不同，其余数据、action head、trainer、checkpoint、wandb/offline 策略必须一致。 | 不启动正式训练；不修改 StarFlow 原项目配置。 |
| 2 | E-001 launch readiness 收敛 | save/resume smoke、eval-load、bounded bs4 runtime smoke 已完成；`launch_ready=false`、`policy_confirmed=false` 保持。 | 将 symmetry checker 接入 readiness，并确认 launch candidate 的最终参数、checkpoint root、resume、offline logging 都可追溯。 | 不能因为用户说“可以训练”就直接训练；必须先完成 readiness 更新和人工确认字段。 |
| 3 | E-006 action-coupling 证据 | synthetic intervention、checkpoint-backed forward、rollout preflight 已完成；真实 rollout 被 RoboCasa assets 缺失阻断。 | 先把 assets blocker 输出为结构化环境阻塞，或在 assets 补齐后重跑 baseline/zero/batch_shuffle/head_mask rollout。 | 没有真实 success_rate 前，不声明 action-gain。 |
| 4 | 命名语义化收敛 | 新运行时代码已开始使用 `future_feature` / `future_supervision` / `latent_cache` 语义别名，旧 `P0/P1` 字段保留兼容。 | 只在碰到相关运行时代码或新报告时继续收敛；历史配置、checkpoint metadata、旧 JSON 不批量改名。 | 不能把命名问题插队到训练 gate 之前做大规模重命名。 |
| 5 | 正式 E-001 训练 | 尚未放行。 | 只有当 1-3 完成且用户明确确认 `policy_confirmed=true`、`human_confirmed=true`、`launch_ready=true` 后，才执行正式训练命令。 | launch_guard 未放行时必须物理拒绝；wandb 默认保持 offline/disabled 策略。 |
| 6 | P1 / latent-cache 后续 | latent cache contract 仅 plan-only 通过。 | E-001/E-006 证据闭环后，再进入真实 latent cache builder 或 P1-b0 接口实现。 | 不在 E-001 训练前并行展开 P1 主实现。 |

### 执行纪律

1. 每轮开始先对照“当前落实顺序”选择最高优先级未完成项。
2. 用户临时提到训练、命名、review 或外部建议时，先判断是否属于当前工作包；不属于则记录，不插队实施。
3. 任何正式训练前必须满足 readiness、launch guard、人工确认、checkpoint/resume/logging 可追溯四个条件。
4. 每个工程步必须最小修改、可验证、更新 `SESSION.md` 和 `07_implementation_log.md`，再提交。

## M0-001：MoWA 文档入口与 Agent 规则

| 项 | 内容 |
|---|---|
| 任务编号 | M0-001 |
| 任务名称 | MoWA 文档入口与 Agent 规则 |
| 所属阶段 | M0 |
| 目标 | 生成 README、AGENTS、阶段计划、任务拆解、实验 Registry、Data Gate 模板、实现日志和 prompts。 |
| 输入依据 | 三份核心文档；用户任务约束。 |
| 允许修改文件 | `docs_zh/mowa/README.md`、`AGENTS.md`、`03` 至 `07`、`prompts/*.md`。 |
| 禁止修改文件 | 三份核心文档核心结论；任何 Python 业务代码。 |
| 预期输出 | MoWA 工程启动包。 |
| 单元测试或 smoke test | `find docs_zh/mowa -maxdepth 2 -type f | sort`。 |
| 验收标准 | 文件齐全、命名一致、无 E-021、无业务代码改动。 |
| 失败后动作 | 补文档或记录待确认项。 |
| 是否计入实验预算 | 否。 |
| 依赖任务 | 无。 |

## M1-001：G0 DataGate skeleton

| 项 | 内容 |
|---|---|
| 任务编号 | M1-001 |
| 任务名称 | G0 DataGate skeleton |
| 所属阶段 | M1 |
| 目标 | 建立 G0 数据核验入口和 DataGateReport 输出结构。 |
| 输入依据 | 详细设计第 4、9、11、12、14 章；`06_data_gate_report.md`。 |
| 允许修改文件 | `tools/mowa/`、`tests/mowa/`、`docs_zh/mowa/06_data_gate_report.md`、`07_implementation_log.md`。 |
| 禁止修改文件 | P0/P1/P2 训练代码；LayerwiseFM 内部；SOT。 |
| 预期输出 | G0 skeleton、报告 schema、smoke test。 |
| 单元测试或 smoke test | `pytest tests/mowa -q` 或 DataGate CLI smoke。 |
| 验收标准 | 未测字段为 `Data Gate` / `TBD`；leakage 失败能阻断训练。 |
| 失败后动作 | 记录缺字段和降级数据域。 |
| 是否计入实验预算 | 否。 |
| 依赖任务 | M0-001。 |

## M1-002：UnifiedEpisode schema draft

| 项 | 内容 |
|---|---|
| 任务编号 | M1-002 |
| 任务名称 | UnifiedEpisode schema draft |
| 所属阶段 | M1 |
| 目标 | 起草 episode 统一 schema，覆盖 dataset、task、observations、actions、wam_targets、metadata。 |
| 输入依据 | 详细设计第 4.5 节和附录 B。 |
| 允许修改文件 | `starVLA/dataloader/mowa/` 或现有 dataloader 下 `mowa_*`；`tests/mowa/`；日志。 |
| 禁止修改文件 | 原有数据加载主干的大规模重构；SOT。 |
| 预期输出 | `MoWAUnifiedEpisode` schema draft。 |
| 单元测试或 smoke test | schema required fields test。 |
| 验收标准 | episode 内时间戳单调；字段状态清楚标注。 |
| 失败后动作 | 记录缺字段，降级为部分标签数据域。 |
| 是否计入实验预算 | 否。 |
| 依赖任务 | M1-001。 |

## M1-003：WindowSample schema draft

| 项 | 内容 |
|---|---|
| 任务编号 | M1-003 |
| 任务名称 | WindowSample schema draft |
| 所属阶段 | M1 |
| 目标 | 起草模型 forward 级样本 schema，保证 history/current/future/action target 边界清晰。 |
| 输入依据 | 详细设计第 4.6 节、6.11 节和附录 B。 |
| 允许修改文件 | dataloader `mowa_*`、`tests/mowa/`、日志。 |
| 禁止修改文件 | 模型训练代码；future action 输入 WAM。 |
| 预期输出 | `MoWAWindowSample` schema draft。 |
| 单元测试或 smoke test | boundary mask test、future action leakage test。 |
| 验收标准 | `history <= anchor < future`；future action 只在 target。 |
| 失败后动作 | 阻断 P0/P1，修 sampler。 |
| 是否计入实验预算 | 否。 |
| 依赖任务 | M1-002。 |

## M1-004：Episode-to-window sampler 设计与 leakage test 计划

| 项 | 内容 |
|---|---|
| 任务编号 | M1-004 |
| 任务名称 | Episode-to-window sampler 设计与 leakage test 计划 |
| 所属阶段 | M1 |
| 目标 | 设计 random anchor 训练采样、fixed-step 推理窗口和 leakage test。 |
| 输入依据 | 详细设计第 3.5、3.6、4.6、6.8、6.11 节。 |
| 允许修改文件 | `tests/mowa/`、dataloader `mowa_*`、报告和日志。 |
| 禁止修改文件 | P0/P1 模型实现；未通过 G0 时启动训练。 |
| 预期输出 | sampler design、leakage test plan。 |
| 单元测试或 smoke test | `test_future_action_leakage`、`test_window_boundary`。 |
| 验收标准 | 训练/推理 WAM Hz、history、future、action chunk 对齐。 |
| 失败后动作 | 阻断训练并修正切片。 |
| 是否计入实验预算 | 否。 |
| 依赖任务 | M1-003。 |

## M2-001：P0FullHeads 接口设计

| 项 | 内容 |
|---|---|
| 任务编号 | M2-001 |
| 任务名称 | P0FullHeads 接口设计 |
| 所属阶段 | M2 |
| 目标 | 设计七类冻结 P0 heads、mask loss 和 P0FutureFeatures。 |
| 输入依据 | 详细设计第 5 章。 |
| 允许修改文件 | `starVLA/model/modules/mowa/`、`configs/mowa/mowa_full_heads_interface.yaml`、`tests/mowa/`。 |
| 禁止修改文件 | 新增未冻结 head；LayerwiseFM 内部。 |
| 预期输出 | `MoWAP0FullHeads` interface draft。 |
| 单元测试或 smoke test | head list/mask loss shape test。 |
| 验收标准 | head 清单为 task_progress、manipulation_readiness、failure_risk、next_best_view_score、subgoal_feasibility、object_visibility_future、action_outcome_class。 |
| 失败后动作 | 回到标签 coverage，mask 缺失 head。 |
| 是否计入实验预算 | E-001 训练阶段计入；接口设计不计入。 |
| 依赖任务 | M1-004。 |

## M2-002：P0 labels coverage report

| 项 | 内容 |
|---|---|
| 任务编号 | M2-002 |
| 任务名称 | P0 labels coverage report |
| 所属阶段 | M2 |
| 目标 | 统计各数据域七类 P0 标签可构造性。 |
| 输入依据 | G0 输出；详细设计第 4.3、5.3 节。 |
| 允许修改文件 | DataGateReport、P0 label report、日志。 |
| 禁止修改文件 | 新增标签含义或新增 head。 |
| 预期输出 | `p0_label_coverage`。 |
| 单元测试或 smoke test | label builder smoke。 |
| 验收标准 | 缺失标签以 mask 处理。 |
| 失败后动作 | 降级为可构造 heads。 |
| 是否计入实验预算 | 否。 |
| 依赖任务 | M2-001。 |

## M2-003：P0-GatedHeads optional task

| 项 | 内容 |
|---|---|
| 任务编号 | M2-003 |
| 任务名称 | P0-GatedHeads optional task |
| 所属阶段 | M2 |
| 目标 | 设计 P0-GatedHeads 可选接口和 gate 日志。 |
| 输入依据 | 详细设计第 5.5、8.4 节。 |
| 允许修改文件 | P0 模块、配置、测试、日志。 |
| 禁止修改文件 | per-head/leave-one-out/selected-head sweep。 |
| 预期输出 | `MoWAP0GatedHeads` optional interface。 |
| 单元测试或 smoke test | gate range/logging test。 |
| 验收标准 | 仅与 P0FullHeads 一次对照；gate 可解释。 |
| 失败后动作 | 保留 FullHeads。 |
| 是否计入实验预算 | E-002 若训练则计入。 |
| 依赖任务 | M2-001、M2-002。 |

## M3-001：Wan latent cache builder 设计

| 项 | 内容 |
|---|---|
| 任务编号 | M3-001 |
| 任务名称 | Wan latent cache builder 设计 |
| 所属阶段 | M3 |
| 目标 | 设计 current/future/history RGB 到 Wan latent cache 的 smoke 流程。 |
| 输入依据 | 详细设计第 4.3、6、9 章。 |
| 允许修改文件 | `tools/mowa/`、`tests/mowa/`、`configs/mowa/mowa_latent_cache_builder_design.yaml`、`configs/mowa/mowa_future_latent_prior_interface.yaml`、日志。 |
| 禁止修改文件 | 训练新 decoder；把 future frames 输入 WAM input。 |
| 预期输出 | `MoWALatentCacheBuilder` design。 |
| 单元测试或 smoke test | latent cache smoke、cache hash test。 |
| 验收标准 | latent shape、cache hash、latency 状态可追踪；未测为 Data Gate。 |
| 失败后动作 | P1 阻断，保留 P0。 |
| 是否计入实验预算 | 否。 |
| 依赖任务 | M1-004。 |

## M3-002：P1-b0 latent prior 接口设计

| 项 | 内容 |
|---|---|
| 任务编号 | M3-002 |
| 任务名称 | P1-b0 latent prior 接口设计 |
| 所属阶段 | M3 |
| 目标 | 设计视觉/语言 future latent prior，不注入 robot history latent。 |
| 输入依据 | 详细设计第 6.2、8.4 节。 |
| 允许修改文件 | `starVLA/model/modules/mowa/`、配置、测试、日志。 |
| 禁止修改文件 | P1-b1 HLC-GCI 默认逻辑；future action leakage。 |
| 预期输出 | `MoWAP1B0FutureLatentPrior` interface。 |
| 单元测试或 smoke test | latent prior shape test、action bridge input test。 |
| 验收标准 | future latent 是 target；当前 latent + text 是输入。 |
| 失败后动作 | 检查 VAE/cache/bridge，回退 P0。 |
| 是否计入实验预算 | E-003 若训练则计入。 |
| 依赖任务 | M3-001。 |

## M4-001：HLCGCI 模块接口设计

| 项 | 内容 |
|---|---|
| 任务编号 | M4-001 |
| 任务名称 | HLCGCI 模块接口设计 |
| 所属阶段 | M4 |
| 目标 | 设计 `C_hist/h_hist/g_hist` 输出接口和 condition-path injection 输入。 |
| 输入依据 | 详细设计第 6.3 至 6.7 节。 |
| 允许修改文件 | `starVLA/model/modules/mowa/`、测试、配置、日志。 |
| 禁止修改文件 | Wan DiT full fine-tune 默认路线；LayerwiseFM 内部重写。 |
| 预期输出 | `MoWAHLCGCI` interface。 |
| 单元测试或 smoke test | shape/gate test。 |
| 验收标准 | gate 初始化与输出范围可检查；shape 均为 TBD/Profile 前不写 measured。 |
| 失败后动作 | 缩小 history、降低注入强度。 |
| 是否计入实验预算 | E-004 若训练则计入。 |
| 依赖任务 | M3-002。 |

## M4-002：history sampling train/inference consistency test

| 项 | 内容 |
|---|---|
| 任务编号 | M4-002 |
| 任务名称 | history sampling train/inference consistency test |
| 所属阶段 | M4 |
| 目标 | 验证训练 random anchor 与推理 fixed-step sliding window 使用同一 WAM Hz、history、future、chunk。 |
| 输入依据 | 详细设计第 3.5、3.6、6.8 节。 |
| 允许修改文件 | `tests/mowa/`、sampler、日志。 |
| 禁止修改文件 | 训练/推理窗口漂移。 |
| 预期输出 | consistency test。 |
| 单元测试或 smoke test | `test_mowa_history_sampling_consistency`。 |
| 验收标准 | 参数来源统一，未测数值为 Data Gate。 |
| 失败后动作 | 阻断 P1-b1。 |
| 是否计入实验预算 | 否。 |
| 依赖任务 | M1-004、M4-001。 |

## M4-003：shuffled-robot sanity plan

| 项 | 内容 |
|---|---|
| 任务编号 | M4-003 |
| 任务名称 | shuffled-robot sanity plan |
| 所属阶段 | M4 |
| 目标 | 设计 E-005，验证 robot history latent 的真实贡献并排查泄漏。 |
| 输入依据 | 详细设计第 6.11、8.4、10.5 节。 |
| 允许修改文件 | sanity eval 脚本/报告、测试、日志。 |
| 禁止修改文件 | 将防泄漏包装成算法创新；future action 输入 WAM。 |
| 预期输出 | shuffled sanity report plan。 |
| 单元测试或 smoke test | shuffle pair construction smoke。 |
| 验收标准 | shuffled 后指标下降才支持 robot history 有效。 |
| 失败后动作 | 检查泄漏或 robot projector 无效。 |
| 是否计入实验预算 | E-005 若训练/评测 run 计入。 |
| 依赖任务 | M4-001、M4-002。 |

## M5-001：WAMActionBridge 接口设计

| 项 | 内容 |
|---|---|
| 任务编号 | M5-001 |
| 任务名称 | WAMActionBridge 接口设计 |
| 所属阶段 | M5 |
| 目标 | 将 P0/P1 WAM features 映射为 LayerwiseFM 可消费的 layerwise condition features。 |
| 输入依据 | 详细设计第 3.3、5.6、6.7、11 章。 |
| 允许修改文件 | bridge/adapter 模块、测试、配置、日志。 |
| 禁止修改文件 | `LayerwiseFM_ActionHeader.py` 内部主逻辑。 |
| 预期输出 | `MoWAActionBridge` interface。 |
| 单元测试或 smoke test | LayerwiseFM compatibility test。 |
| 验收标准 | bridge 输出维度和层数可对齐；默认不改 action head 内部层。 |
| 失败后动作 | 保持诊断信号或重新设计 bridge。 |
| 是否计入实验预算 | 否；E-006 训练/评测阶段计入。 |
| 依赖任务 | M2-001 或 M3-002。 |

## M5-002：E-006 coupling / feature removal eval plan

| 项 | 内容 |
|---|---|
| 任务编号 | M5-002 |
| 任务名称 | E-006 coupling / feature removal eval plan |
| 所属阶段 | M5 |
| 目标 | 设计 feature removal、shuffle、correlation、latency-cost coupling 分析。 |
| 输入依据 | 详细设计第 8.4、10.5 节。 |
| 允许修改文件 | eval report、tests、日志。 |
| 禁止修改文件 | planner/FSM/RL/Scene Graph 对照；action head 主干重写。 |
| 预期输出 | coupling eval plan。 |
| 单元测试或 smoke test | feature removal smoke。 |
| 验收标准 | 去掉 WAM feature 后 action 指标有可解释变化。 |
| 失败后动作 | WAM 降级为诊断。 |
| 是否计入实验预算 | E-006。 |
| 依赖任务 | M5-001。 |

## M6-001：P2 frozen decoder diagnostic plan

| 项 | 内容 |
|---|---|
| 任务编号 | M6-001 |
| 任务名称 | P2 frozen decoder diagnostic plan |
| 所属阶段 | M6 |
| 目标 | 设计只对 selected cases 触发的 frozen decoder keyframe/short clip 诊断。 |
| 输入依据 | 详细设计第 7 章。 |
| 允许修改文件 | `tools/mowa/`、P2 report、日志。 |
| 禁止修改文件 | 训练新 decoder；微调 Wan2.2-VAE decoder；continuous video generation。 |
| 预期输出 | P2DiagnosticReport plan。 |
| 单元测试或 smoke test | decode smoke 或 latent fallback smoke。 |
| 验收标准 | P2 只服务 failure analysis/debug/demo。 |
| 失败后动作 | 降级 latent retrieval/PCA/attention map。 |
| 是否计入实验预算 | E-008 不计入。 |
| 依赖任务 | M3-002 或 M4-001。 |

## M6-002：final report / known limitations template

| 项 | 内容 |
|---|---|
| 任务编号 | M6-002 |
| 任务名称 | final report / known limitations template |
| 所属阶段 | M6 |
| 目标 | 汇总实验、失败路径、SOT 影响、Known Limitations 和下一步。 |
| 输入依据 | 详细设计第 12、13、14、15 章；全部阶段报告。 |
| 允许修改文件 | `docs_zh/mowa/` 报告与日志。 |
| 禁止修改文件 | 改写实验编号含义；包装失败结果为成功。 |
| 预期输出 | final report template。 |
| 单元测试或 smoke test | 文档一致性检查。 |
| 验收标准 | 所有 TBD/Risk/Implementation Task 有状态。 |
| 失败后动作 | 保留 Known Limitations。 |
| 是否计入实验预算 | 否。 |
| 依赖任务 | M1 至 M6 已执行项。 |
