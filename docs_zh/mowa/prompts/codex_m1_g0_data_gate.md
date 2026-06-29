# MoWA M1 G0 Data Gate Prompt

## 当前阶段

M1：G0 Data Verification Gate。

## 目标

建立 MoWA G0 DataGate skeleton、UnifiedEpisode schema draft、WindowSample schema draft、Episode-to-window sampler 设计与 leakage test 计划。不要启动 P0/P1 主训练。

## 必读文档

1. `docs_zh/mowa/AGENTS.md`
2. `docs_zh/mowa/README.md`
3. `docs_zh/mowa/02_detailed_design.md` 第 4、9、11、12、14 章和附录 B/F
4. `docs_zh/mowa/06_data_gate_report.md`
5. `docs_zh/mowa/04_task_breakdown.md` 的 M1 任务卡
6. `docs_zh/mowa/07_implementation_log.md`

## 允许修改文件

- `docs_zh/mowa/06_data_gate_report.md`
- `docs_zh/mowa/07_implementation_log.md`
- 后续实现阶段可新增 `tools/mowa/`、`tests/mowa/`、`starVLA/dataloader/mowa/` 或现有 dataloader 下 `mowa_*` 文件。

## 禁止修改文件

- P0/P1/P2 模型训练代码。
- `LayerwiseFM_ActionHeader.py`。
- 三份核心 Source-of-Truth 文档。
- 任何 checkpoint 逻辑。

## 具体任务

1. 设计 G0 report 输出字段。
2. 设计 UnifiedEpisode / WindowSample schema。
3. 设计 episode-to-window sampling 与 boundary mask。
4. 设计 future action leakage test。
5. 所有未实测 fps、Hz、window、batch、显存、阈值写为 `Data Gate` / `TBD`。
6. 更新实现日志。

## 测试命令

```bash
pytest tests/mowa -q
```

若尚未创建测试，只运行可用的最小 smoke test，并在日志中写明未运行原因。

## 验收标准

1. G0 不计入实验预算。
2. leakage test 失败必须阻断 P0/P1。
3. future action label 不进入 WAM input。
4. G0 未通过前不得进入 P0/P1 主训练。

## 输出格式

输出修改文件、schema 字段、测试命令与结果、Go/No-Go 状态、下一步。

## 阶段边界

不得实现 P0FullHeads、P1 latent prior、HLC-GCI 或 action bridge。
