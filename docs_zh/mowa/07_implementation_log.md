# MoWA 实现日志

每次 MoWA 修改必须更新本文件。若未运行测试，必须写明原因。若影响 Source-of-Truth 或新增实验编号，必须显式标注。

## 日志模板

| 日期 | 任务编号 | 修改文件 | 修改摘要 | 测试命令 | 测试结果 | 未解决问题 | 下一步建议 | 是否影响 Source-of-Truth | 是否新增实验编号 |
|---|---|---|---|---|---|---|---|---|---|
| YYYY-MM-DD | Mx-xxx | `path` | TBD | TBD | TBD | TBD | TBD | 否 / 是，说明 | 否 / 是，编号 |

## 当前记录

| 日期 | 任务编号 | 修改文件 | 修改摘要 | 测试命令 | 测试结果 | 未解决问题 | 下一步建议 | 是否影响 Source-of-Truth | 是否新增实验编号 |
|---|---|---|---|---|---|---|---|---|---|
| 2026-06-29 | M0-001 | `docs_zh/mowa/README.md`、`AGENTS.md`、`03_agent_implementation_plan.md`、`04_task_breakdown.md`、`05_experiment_registry.md`、`06_data_gate_report.md`、`07_implementation_log.md`、`prompts/*.md` | 生成 MoWA 工程启动包、Agent 执行规则、阶段计划、任务卡、实验 Registry、Data Gate 模板和阶段 prompts。 | `find docs_zh/mowa -maxdepth 2 -type f \| sort` | 通过，文件列表齐全；`docs_zh/mowa/` 当前被 git ignore，`git status --short --ignored docs_zh/mowa` 显示 `!! docs_zh/mowa/`。 | 未运行 Python 业务代码测试；未执行数据处理。 | 下一步执行 M1-001：G0 DataGate skeleton。 | 否，仅继承三份核心文档。 | 否。 |
| 2026-06-29 | M0-002 | `docs_zh/mowa/08_starvla_data_benchmark_support_matrix.md`、`docs_zh/mowa/README.md`、`docs_zh/mowa/07_implementation_log.md` | 检查 StarVLA 已支持的数据集与 benchmark，生成面向 MoWA 的支持矩阵，并给出 `robocasa365` 提前准备建议。 | `find docs_zh/mowa -maxdepth 2 -type f \| sort` | 通过，矩阵文档已生成。 | 未运行 Python 业务代码测试；未实际下载任何外部数据。 | 若要执行 G0，优先用 `robocasa365` 的最小闭环任务。 | 否，仅整理仓库现状与 MoWA 优先级。 | 否。 |
| 2026-06-29 | M1-001/M1-002/M1-003/M1-004 | `starVLA/dataloader/mowa/__init__.py`、`schema.py`、`sampler.py`、`data_gate.py`、`tests/mowa/test_mowa_data_gate.py`、`starVLA/dataloader/__init__.py`、`docs_zh/mowa/07_implementation_log.md` | 新增 MoWA G0 DataGate skeleton、UnifiedEpisode / WindowSample schema、Episode-to-window sampler 和 future action leakage 单测；将 `starVLA.dataloader` 的训练重依赖延迟到 `build_dataloader` 调用，避免纯 schema 测试依赖 `accelerate`。 | `.venv/bin/python -m pytest tests/mowa -q`; `.venv/bin/python -m unittest tests.mowa.test_mowa_data_gate -v` | `.venv` 中未安装 pytest；`.venv/bin/python -m unittest ...` 通过 4 项测试。 | 未运行真实数据处理；未下载 `robocasa365`；G0 report 仍为 skeleton，所有 profile 字段保持 `Data Gate` / `TBD`。 | 下一步执行 M1 数据核验 smoke：准备 `robocasa365` 最小任务数据后读取 metadata，并生成实际 DataGateReport。 | 否，不改写 SOT。 | 否。 |
