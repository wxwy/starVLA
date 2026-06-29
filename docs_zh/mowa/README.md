# MoWA 文档入口

MoWA（Mobile World Action Model，面向移动操作的世界动作模型）是面向 long-horizon household mobile manipulation 的 World Action Model 工程启动包。MoWA 不替代 VLA 策略项目，不训练通用视频生成模型，而是围绕 action-relevant future representation 建立数据门禁、P0/P1/P2 路线、实验 Registry、Agent 执行规则与日志追踪。

## 核心阅读顺序

1. `00_project_proposal.md`：理解项目目标、研究边界、P0/P1/P2 的立项定位。
2. `01_technical_survey.md`：理解技术路线、数据候选、P0/P1/P2 的调研依据和风险。
3. `02_detailed_design.md`：理解已冻结的 Source-of-Truth、工程命名、G0、实验编号、schema、模块边界和执行计划。

以上三份文件是 MoWA 的核心 Source-of-Truth，后续执行文件只能继承、拆解和追踪，不得改写其核心结论。

## Agent 执行文件作用

| 文件 | 作用 |
|---|---|
| `AGENTS.md` | Codex / Claude 的 MoWA 项目级执行规则。 |
| `03_agent_implementation_plan.md` | M0 至 M6 的阶段计划、允许修改范围、验收标准与失败动作。 |
| `04_task_breakdown.md` | 从详细设计抽取的 MoWA 任务卡。 |
| `05_experiment_registry.md` | G0 与 E-001 至 E-020 的唯一执行侧实验表。 |
| `06_data_gate_report.md` | G0 Data Verification Gate 报告模板。 |
| `07_implementation_log.md` | 所有 MoWA 修改、测试、待确认项与 SOT 影响记录。 |
| `08_starvla_data_benchmark_support_matrix.md` | 基于 StarVLA 当前仓库生成的 MoWA 数据 / benchmark 支持矩阵。 |
| `prompts/*.md` | 可独立复制给 Codex / Claude 的阶段 prompt。 |

## Codex / Claude 必读文件

每次执行 MoWA 任务前，Codex / Claude 必须先读：

1. `docs_zh/mowa/AGENTS.md`
2. `docs_zh/mowa/README.md`
3. `docs_zh/mowa/02_detailed_design.md` 的第 0、1、4、8、11、14 章和附录 A/B/D/F
4. 当前任务对应的 prompt 或任务卡
5. `docs_zh/mowa/07_implementation_log.md`

## Source-of-Truth

MoWA 的 Source-of-Truth 包括：

- 项目短名：MoWA
- 英文全称：Mobile World Action Model
- 中文全称：面向移动操作的世界动作模型
- 工程 namespace：`mowa`
- 文档目录：`docs_zh/mowa/`
- 配置前缀：`mowa_*.yaml`
- 实验显示前缀：`MoWA-E-001` 至 `MoWA-E-020`
- G0 是 Data Verification Gate，属于工程门禁，不计入算法实验数。
- P0/P1/P2 阶段定义、实验编号、负面清单、future action 防泄漏、LayerwiseFM 默认 bridge/adapter 接入边界均以 `02_detailed_design.md` 为准。

## 执行门禁

不得直接跳过 G0 进入 P0/P1。G0 未完成前，只允许生成数据门禁 skeleton、schema 草案、window sampler 设计、leakage test 计划和报告模板；不得启动 P0/P1 主训练，不得把任何 fps、Hz、window、batch、显存、训练时长或阈值写成 `measured`。
