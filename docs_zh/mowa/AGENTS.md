# MoWA Agent 执行规则

本文件适用于 Codex / Claude 在 MoWA（Mobile World Action Model，面向移动操作的世界动作模型）项目中的所有执行任务。执行前必须先阅读 `README.md`、三份核心 Source-of-Truth 文档、当前任务卡和 `07_implementation_log.md`。

若未来 `00_project_proposal.md`、`01_technical_survey.md`、`02_detailed_design.md` 再次缺失，Agent 必须把该缺口视为 readiness 风险：不得临时补写或局部改写核心 SOT，只能读取 `03` 至 `10` 的派生执行文档、记录未解决项，并保持正式训练 launch gate 关闭。

## 语言规则

1. 所有回复、解释、注释、日志和提交摘要默认使用简体中文。
2. 代码中的既有英文 API、类名、字段名、实验名保持原样。

## 修改规则

1. 只做最小必要修改，不重构无关代码。
2. 保留原有风格、命名、注释和结构。
3. 不新增无关依赖，不改公共接口，不破坏原有逻辑。
4. 优先复用现有代码、工具函数和 StarVLA-native 接入方式。
5. 非必要不创建新文件；确需创建时必须使用 MoWA / `mowa` 命名口径。
6. 修改前先阅读相关上下文代码和 MoWA Source-of-Truth。

## 命名规则

1. 项目短名统一写 `MoWA`。
2. 工程 namespace、目录、配置、测试、日志、输出名统一使用 `mowa` 前缀。
3. 配置文件命名使用 `mowa_*.yaml`。
4. 实验显示名前缀使用 `MoWA-E-001` 至 `MoWA-E-020`。
5. 禁止使用 `StarWAM`、`WAM-Project2`、`项目二`、`MobileWAM` 等临时命名。

## 禁止事项

1. 不得新增 E-021 或更高实验编号。
2. 不得改写 `00_project_proposal.md`、`01_technical_survey.md`、`02_detailed_design.md` 的核心结论或 Source-of-Truth。
3. 不得引入 planner / FSM / RL / Scene Graph 作为默认路线。
4. 不得把 frozen decoder diagnostic 扩展为 continuous video generation。
5. 不得将 future action label 输入 WAM；future action 只能作为 action target 或 invariant 检查对象。
6. 不得重写 `LayerwiseFM_ActionHeader.py` 内部主逻辑；默认只通过 bridge / adapter 接入 WAM features。
7. 不得新增未冻结 future supervision head，不得做 per-head / leave-one-out / selected-head sweep。
8. 未实测的 fps、Hz、history window、future window、batch、显存、训练时长、阈值必须标注为 `target` / `TBD` / `Data Gate`，不得写成 `measured`。

## 文档同步规则

1. 每次 MoWA 修改必须更新 `docs_zh/mowa/07_implementation_log.md`。
2. 若发现核心文档之间不一致，不得局部改写 Source-of-Truth，只能记录到“未解决问题 / 待确认项”。
3. 新增 TODO 必须简洁、可执行，并指向具体阶段或任务编号。
4. `09_p0_*`、`10_p1_*`、`mowa_p0_*` 等旧命名产物仅作兼容留档；新的 readiness、launch、实现入口优先使用 `future` / `full_heads` 新命名。

## 测试规则

1. 每个实现任务必须给出测试命令、运行结果和失败说明。
2. G0、`full_heads`、`future_latent_prior`、`frozen_decoder_diagnostic` 相关实现至少包含 smoke test 或单元测试计划。
3. 数据与采样任务必须包含 leakage test 或 boundary assert 计划。
4. 未运行测试时必须明确写明“未运行”及原因，不得省略。

## 安全规则

以下操作必须先请求人工确认：

1. 删除文件。
2. 批量重构。
3. 修改训练主干。
4. 修改 checkpoint / resume / save 逻辑。
5. 修改密钥、配置凭据或系统环境。
6. 外网访问、下载数据或安装新依赖。

## 阶段推进规则

1. 必须先执行 M1 / G0 Data Verification Gate，再进入 `full_heads` / `future_latent_prior` 主训练实现。
2. G0 不通过时不得强行启动这些主训练链路；只能缩小数据域、补 schema 或输出降级报告。
3. 各阶段实验按 `03_agent_implementation_plan.md` 与 `05_experiment_registry.md` 推进。
4. 新增训练实验只能占用 E-011 至 E-020 的 UNALLOCATED 槽位，并必须先记录理由。
