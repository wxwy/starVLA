# MoWA Agent 实施计划

本计划继承 `00_project_proposal.md`、`01_technical_survey.md` 与 `02_detailed_design.md`，用于约束 Codex / Claude 的阶段执行。G0 是工程门禁，不计入算法实验预算；所有训练实验必须使用 E-001 至 E-020，禁止新增 E-021。

## M0：项目启动包生成

| 项 | 内容 |
|---|---|
| 阶段目标 | 生成 MoWA 文档入口、Agent 规则、阶段计划、任务拆解、实验 Registry、G0 模板、实现日志和 prompts。 |
| 输入文档 | 三份核心文档、用户任务要求。 |
| 允许修改范围 | `docs_zh/mowa/README.md`、`AGENTS.md`、`03` 至 `07`、`prompts/*.md`。 |
| 禁止修改范围 | 模型、数据、训练、评测业务代码；三份核心文档核心结论。 |
| 输出文件 | 本启动包全部 Markdown 文件。 |
| 测试命令 | `find docs_zh/mowa -maxdepth 2 -type f | sort`；文档生成阶段不运行 Python 业务测试。 |
| 验收标准 | 文件齐全；命名统一为 MoWA / `mowa`；无 E-021；无业务代码改动。 |
| 失败后动作 | 补齐缺失文件或将不一致写入 `07_implementation_log.md`。 |
| 实验预算 | 否。 |

## M1：G0 Data Verification Gate

| 项 | 内容 |
|---|---|
| 阶段目标 | 建立 G0 skeleton、UnifiedEpisode / WindowSample schema 草案、DataGateReport 输出、leakage test 计划。 |
| 输入文档 | `02_detailed_design.md` 第 4、9、11、12、14 章，`06_data_gate_report.md`，`prompts/codex_m1_g0_data_gate.md`。 |
| 允许修改范围 | `docs_zh/mowa/06_data_gate_report.md`；后续实现阶段可新增 `tools/mowa/`、`tests/mowa/`、`starVLA/dataloader/mowa/` 或现有 dataloader 下 `mowa_*` 文件。 |
| 禁止修改范围 | P0/P1/P2 模型训练代码；LayerwiseFM 内部；三份核心 SOT。 |
| 输出文件 | DataGateReport、schema skeleton、leakage test skeleton、实现日志。 |
| 测试命令 | `pytest tests/mowa -q` 或最小 schema/leakage smoke test；未实现代码时写 `TBD`。 |
| 验收标准 | 至少一个候选数据域可进入 schema/profile/leakage 判定；未测数值均为 `Data Gate` / `TBD`。 |
| 失败后动作 | 降级数据域、记录缺字段、阻断 P0/P1 主训练。 |
| 实验预算 | 否。G0 不计入 E-001 至 E-020。 |

## M2：P0 Video-Generation-Free WAM

| 项 | 内容 |
|---|---|
| 阶段目标 | 设计并实现 P0FullHeads 与可选 P0GatedHeads 接口，建立 P0 label coverage report 和 action bridge 输入。 |
| 输入文档 | `02_detailed_design.md` 第 5、8、10、11 章，M1 G0 输出。 |
| 允许修改范围 | `starVLA/model/modules/mowa/`、`configs/mowa/mowa_full_heads_interface.yaml`、`tests/mowa/`、P0 文档与日志。 |
| 禁止修改范围 | P1 latent 模型；P2 decoder；LayerwiseFM 内部主逻辑；新增 P0 head。 |
| 输出文件 | P0FullHeads skeleton、P0 label report、可选 P0GatedHeads 计划。 |
| 测试命令 | `pytest tests/mowa -q`；P0 smoke train/eval 命令由实现阶段补充。 |
| 验收标准 | 七类冻结 heads 仅在标签可构造时参与 loss；mask 行为清晰；WAM features 可进入 bridge。 |
| 失败后动作 | P0 降级为诊断模型，阻断 P1 主训练。 |
| 实验预算 | E-001；可选 E-002。 |

## M3：P1-b0 Latent-Only future prior

| 项 | 内容 |
|---|---|
| 阶段目标 | 设计 Wan latent cache builder 与 P1-b0 latent future prior 接口，不默认解码像素。 |
| 输入文档 | `02_detailed_design.md` 第 6、8、9、10、11 章，G0 latent cache smoke。 |
| 允许修改范围 | `tools/mowa/`、`starVLA/model/modules/mowa/`、`configs/mowa/mowa_latent_cache_builder_design.yaml`、`configs/mowa/mowa_future_latent_prior_interface.yaml`、`tests/mowa/`。 |
| 禁止修改范围 | P1-b1 HLC-GCI 训练主逻辑；P2 continuous video；Wan DiT full fine-tune 默认路线。 |
| 输出文件 | latent cache design、P1-b0 interface、cache smoke report。 |
| 测试命令 | latent cache smoke、shape test、bridge compatibility test。 |
| 验收标准 | future latent 只作为 target/cache；latent loss 与 action bridge 分离记录。 |
| 失败后动作 | 检查 VAE/cache/bridge；回退 P0。 |
| 实验预算 | E-003。 |

## M4：P1-b1 HLC-GCI

| 项 | 内容 |
|---|---|
| 阶段目标 | 设计 HLC-GCI 模块接口、history sampling 一致性测试和 shuffled-robot sanity 计划。 |
| 输入文档 | `02_detailed_design.md` 第 6、8、10、11、14 章，M3 输出。 |
| 允许修改范围 | `starVLA/model/modules/mowa/`、`tests/mowa/`、`configs/mowa/mowa_hlc_gci_interface.yaml`、日志和报告。 |
| 禁止修改范围 | future action leakage；LayerwiseFM 内部重写；Rec-HLC 默认主线；P2 主训练闭环。 |
| 输出文件 | HLC-GCI skeleton、shape/gate tests、history consistency test、shuffled sanity report。 |
| 测试命令 | `pytest tests/mowa -q`，重点覆盖 shape、gate、leakage、history sampling consistency。 |
| 验收标准 | 输出 `C_hist/h_hist/g_hist`；训练和推理 window 参数一致；E-005 能检测 robot history 贡献。 |
| 失败后动作 | 降低 history 注入强度、缩短窗口、回退 P1-b0。 |
| 实验预算 | E-004、E-005；Rec-HLC 仅条件触发 E-009。 |

## M5：WAM-to-action bridge 与 coupling eval

| 项 | 内容 |
|---|---|
| 阶段目标 | 设计 WAMActionBridge，将 P0/P1 WAM features 映射为 LayerwiseFM 可消费的 layerwise condition features，并设计 coupling eval。 |
| 输入文档 | `02_detailed_design.md` 第 3、5、6、8、10、11 章。 |
| 允许修改范围 | bridge / adapter 文件、配置、测试、评测报告。 |
| 禁止修改范围 | `LayerwiseFM_ActionHeader.py` 内部主逻辑；action head 主干重写；planner/FSM/RL/Scene Graph 对照。 |
| 输出文件 | WAMActionBridge 接口、feature removal eval、coupling report。 |
| 测试命令 | bridge shape compatibility test、feature removal smoke eval。 |
| 验收标准 | bridge 输出可被 action path 消费；feature removal 或 shuffle 能解释 WAM 是否被使用。 |
| 失败后动作 | WAM 降级为诊断信号，记录 action coupling 风险。 |
| 实验预算 | E-006。 |

## M6：P2 diagnostic 与 final report

| 项 | 内容 |
|---|---|
| 阶段目标 | 设计 P2 frozen decoder diagnostic 与 final report / known limitations 模板。 |
| 输入文档 | `02_detailed_design.md` 第 7、8、10、12、13、15 章。 |
| 允许修改范围 | `tools/mowa/` 诊断工具、`docs_zh/mowa/` 报告、`outputs/mowa/`。 |
| 禁止修改范围 | 训练新 decoder；微调 Wan2.2-VAE decoder；continuous video generation。 |
| 输出文件 | P2DiagnosticReport、fallback latent diagnostic、final report / known limitations。 |
| 测试命令 | P2 decode smoke 或 latent fallback smoke；未执行写明原因。 |
| 验收标准 | 只对 selected cases 触发诊断；结果服务失败分析，不进入主训练闭环。 |
| 失败后动作 | 降级为 latent retrieval/PCA/attention map 报告。 |
| 实验预算 | E-008 不训练、不计入；E-010 eval-only 不计入。 |
