# MoWA M0 Agent Pack Prompt

## 当前阶段

M0：项目启动包生成。

## 目标

基于 `docs_zh/mowa/00_project_proposal.md`、`01_technical_survey.md`、`02_detailed_design.md` 生成或更新 MoWA 文档入口、Agent 规则、阶段计划、任务拆解、实验 Registry、Data Gate 报告模板、实现日志和 prompts。

## 必读文档

1. `docs_zh/mowa/00_project_proposal.md`
2. `docs_zh/mowa/01_technical_survey.md`
3. `docs_zh/mowa/02_detailed_design.md`
4. `docs_zh/mowa/AGENTS.md`

## 允许修改文件

- `docs_zh/mowa/README.md`
- `docs_zh/mowa/AGENTS.md`
- `docs_zh/mowa/03_agent_implementation_plan.md`
- `docs_zh/mowa/04_task_breakdown.md`
- `docs_zh/mowa/05_experiment_registry.md`
- `docs_zh/mowa/06_data_gate_report.md`
- `docs_zh/mowa/07_implementation_log.md`
- `docs_zh/mowa/prompts/*.md`

## 禁止修改文件

- 三份核心文档的核心结论。
- 任何模型、数据、训练、评测 Python 业务代码。
- `LayerwiseFM_ActionHeader.py`。

## 具体任务

1. 抽取 Source-of-Truth、项目边界、实验编号、P0/P1/P2 路线、G0、命名规范和 Agent 执行约束。
2. 生成启动包文档。
3. 保证所有新增文件使用 MoWA / `mowa` 口径。
4. 不得新增 E-021。
5. 更新 `07_implementation_log.md`。

## 测试命令

```bash
find docs_zh/mowa -maxdepth 2 -type f | sort
```

## 验收标准

1. 文件齐全。
2. 无业务代码改动。
3. 无 E-021。
4. G0 被写成工程门禁，不计入算法实验。
5. 未实测数值标注为 `target` / `TBD` / `Data Gate`。

## 输出格式

只输出：新增/修改文件列表、每个文件作用、SOT 摘要、下一步推荐任务、未执行测试说明。

## 阶段边界

不得越界实现 M1 或后续代码。
