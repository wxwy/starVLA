# StarFlow-VLA Implementation Log

版本：V0.1  
说明：本文件记录 StarFlow-VLA 从设计到代码实现的 step-by-step 过程。每个任务完成后必须追加记录。本文件不是实验报告，不得把 Stage A lightweight validation 冒充 Stage B target smoke、正式训练、完整评测或部署结论。

## Environment Notice

当前 Codex 任务只代表文档编辑 / 代码编辑 / 轻量构建验证，不等价于正式训练、完整评测或真实机器人部署环境。所有未在目标环境运行的训练、评测、部署结果必须标注为：

`未在 Stage B 目标模型 smoke 或正式训练/评测环境验证，需在对应目标环境验证。`

## Two-Stage Environment Policy

- Stage A：1×A100 40G lightweight validation，用于文档、代码构建、静态检查、import、config parse、mock/smoke。
- Stage B：1×A100 40G target smoke validation，用于真实模型 P0 smoke、single batch overfit、checkpoint、eval smoke。
- Stage A 与 Stage B 默认均为 1×A100 40G，区别是验证强度，不是硬件差异。

---

## Record Template

### Task ID:

### Issue ID:

### Date:

### Design Reference:

- V4.6.2 Section:
- TASK_BREAKDOWN:
- CODEX_ISSUES:
- ACCEPTANCE_CHECKLIST:

### Environment:

- Stage: A 1×A100 40G lightweight validation or B 1×A100 40G target smoke validation
- Local path:
- Target repo path:
- Is deployment/training environment:
- GPU available:
- Data available:
- Notes:

### Goal:

...

### Steps:

1. ...
2. ...
3. ...

### Files Changed:

- Added:
- Modified:
- Deleted:

### Tests Run:

```bash
# command
```

### Test Results:

- Pass:
- Fail:
- Not run:
- Need target verification:
- Reason if not run:

### Environment Boundary Statement:

...

### Deviations from Design:

...

### Rollback Plan:

...

### Next Step:

...

---

## Initial Record

### Task ID:

DOC-M1

### Issue ID:

N/A

### Date:

2026-06-14

### Design Reference:

- V4.6.2 Section: Codex 本地执行环境、两段式服务器策略与实施记录规范
- TASK_BREAKDOWN: P0-M0 / P0-M11
- CODEX_ISSUES: 全局 issue 模板更新
- ACCEPTANCE_CHECKLIST: 实施记录验收项

### Environment:

- Stage: A 1×A100 40G lightweight validation
- Local path: `C:\Users\fast_\Desktop\xx`
- Target repo path: `E:\projects\starVLA`
- Is deployment/training environment: 否
- GPU available: 未在本任务中检查
- Data available: 未在本任务中检查
- Notes: 本次只修改工程文档，不实现 StarFlowVLA 代码。

### Goal:

建立 Codex 执行规范、两段式环境策略和实施记录模板。

### Steps:

1. 新增 `CODEX_EXECUTION_GUIDE.md`。
2. 新增 `IMPLEMENTATION_LOG.md`。
3. 更新 `CODEX_ISSUES.md` issue 模板。
4. 更新 `ACCEPTANCE_CHECKLIST.md` 实施记录验收项。

### Files Changed:

- Added: `docs/starflow_vla/CODEX_EXECUTION_GUIDE.md`
- Added: `docs/starflow_vla/IMPLEMENTATION_LOG.md`
- Modified: `docs/starflow_vla/CODEX_ISSUES.md`
- Modified: `docs/starflow_vla/ACCEPTANCE_CHECKLIST.md`
- Deleted: none

### Tests Run:

```bash
rg -n "Stage A|Stage B|IMPLEMENTATION_LOG|未在 Stage B 目标模型 smoke" docs/starflow_vla
```

### Test Results:

- Pass: 文档文本检查通过。
- Fail: none
- Not run: StarFlowVLA 代码测试、训练、评测、部署。
- Need target verification: 所有需要 A100、真实数据、真实机器人或部署服务的验证。
- Reason if not run: 本任务只补充工程文档和执行规范。

### Environment Boundary Statement:

本次结果仅为 Stage A 1×A100 40G lightweight validation 的文档更新，不代表 Stage B 目标模型 smoke、正式训练、完整评测或部署结论。

### Deviations from Design:

无。

### Rollback Plan:

回滚本次新增/修改的四份文档即可，不影响 StarVLA 源代码。

### Next Step:

执行 P0-M0：V4.6.2 文档冻结检查。

## Record DOC-M3

- Date: 2026-06-14
- Task ID: DOC-M3
- Issue ID: N/A
- Design Reference: V4.6.2 Section 1.6 / 4.5.3 / 6.11; docs/starflow_vla/ALGORITHM_OPTIMIZATION_PLAN.md
- Stage: Stage A 1×A100 40G lightweight validation document update
- Owner: Codex
- Notes: 本次只修改文档，补充 H2-a future_tokens 规划槽位优化与 H2-b 状态条件注入路径优化；未实现 StarFlowVLA 代码，未运行训练、评测或部署。

### Goal:

把 `future_tokens / num_target_vision_tokens` 和 `state_mode` 两个算法优化方向纳入设计、任务、issue、验收和实验矩阵，并明确所有真实指标必须等待 Stage B A100 或目标 Benchmark 复验。

### Steps:

1. 在最终详细设计文档源文件中补充 H2-a / H2-b 研究假设和实验矩阵。
2. 在第 4 章模型设计中补充 Future Tokens Planning Slot Optimization 与 State Conditioning Path Optimization。
3. 更新 `TASK_BREAKDOWN.md`、`P0_IMPLEMENTATION_PLAN.md`、`CODEX_ISSUES.md` 和 `ACCEPTANCE_CHECKLIST.md`。
4. 新增 `ALGORITHM_OPTIMIZATION_PLAN.md` 作为算法优化执行基线。

### Files Changed:

- Added: `docs/starflow_vla/ALGORITHM_OPTIMIZATION_PLAN.md`
- Modified: `build_final_design_doc.py`
- Modified: `chapter4_model_design.md`
- Modified: `docs/starflow_vla/TASK_BREAKDOWN.md`
- Modified: `docs/starflow_vla/P0_IMPLEMENTATION_PLAN.md`
- Modified: `docs/starflow_vla/CODEX_ISSUES.md`
- Modified: `docs/starflow_vla/ACCEPTANCE_CHECKLIST.md`
- Modified: `docs/starflow_vla/IMPLEMENTATION_LOG.md`
- Deleted: none

### Tests Run:

```bash
rg -n "H2-a|H2-b|Future Tokens Planning Slot Optimization|State Conditioning Path Optimization|E-H2a|E-H2b|DOC-M3|待 Stage B A100 复验" build_final_design_doc.py chapter4_model_design.md docs/starflow_vla
```

### Test Results:

- Pass: 文档一致性关键词检查通过；最终详细设计 Markdown / DOCX 已重新生成。
- Fail: none
- Not run: StarFlowVLA 代码测试、single batch overfit、LIBERO eval、RoboCasa/RoboTwin eval、真实机器人测试。
- Need target verification: success_rate、loss curve、peak memory、inference latency、action smoothness、cross benchmark drop、state noise robustness、missing state robustness。
- Reason if not run: 本任务只补充算法优化文档和实验设计，不具备目标训练/评测环境验证含义。

### Environment Boundary Statement:

本次结果仅为本地文档更新，不代表 A100、Virtaicloud、Bita、LIBERO/RoboCasa/RoboTwin 或真实机器人验证结果。所有未运行指标必须保留 `[待 Stage B A100 复验]` 或对应 Benchmark 占位符。

### Deviations from Design:

`docs/design/基于VLA统一训练与泛化评测框架的机器人基础模型研究_最终详细设计文档.md`、`EXPERIMENT_MATRIX.md`、`MODULE_MAPPING.md`、`CAREER_PRESENTATION.md` 在当前工作区不存在；本次未凭空编造这些文件，而是更新实际存在的最终文档源文件与 `docs/starflow_vla` 执行文档，并新增 `ALGORITHM_OPTIMIZATION_PLAN.md`。

### Rollback Plan:

回滚本记录列出的文档修改与新增 `ALGORITHM_OPTIMIZATION_PLAN.md` 即可，不影响 StarVLA 源代码。

### Next Step:

执行 P0-M0：V4.6.2 文档冻结检查；随后执行 P0-M7a Stage A 配置与 manifest dry-run。

## Record: DOC-M4

### Task ID:

DOC-M4

### Issue ID:

N/A

### Date:

2026-06-14

### Design Reference:

- V4.6.2 Section: 项目命名口径修正
- V4.6.2 Section: Stage A / Stage B 环境口径修正
- V4.6.2 Section: VGGT / RGB-3D Geometry Fusion 边界固化
- TASK_BREAKDOWN: P2-M2 Observation Geometry Adapter Interface
- CODEX_ISSUES: P2-M2
- ACCEPTANCE_CHECKLIST: P2 / 《基于世界模型的移动操作规划与决策框架研究》衔接验收项

### Environment:

- Stage: A 1×A100 40G lightweight validation
- Machine: [待填]
- GPU: 1×A100 40G
- Data available: Not required
- Model loaded: No
- Is full training/evaluation/deployment environment: No
- Notes: 本任务只做文档命名与口径修正，未加载真实模型，未运行训练、评测或部署。

### Goal:

统一文档中的项目命名口径：在《基于VLA统一训练与泛化评测框架的机器人基础模型研究》自己的文档中，统一称呼自己为“本项目 / 本研究 / StarFlow-VLA”；提到《基于世界模型的移动操作规划与决策框架研究》时使用完整项目名称。同时保持 Stage A / Stage B 环境口径与 VGGT 边界固化口径一致。

### Steps:

1. 搜索并替换禁用的本项目简称为“本项目”或“本研究”。
2. 搜索并替换禁用的世界模型规划研究简称为《基于世界模型的移动操作规划与决策框架研究》。
3. 检查 VGGT 相关表述，确保不再出现禁用项目简称。
4. 检查 Stage A / Stage B 环境口径，确保均为 1×A100 40G，区别为验证强度。
5. 确认 VGGT 不作为本项目 P0/P1 必选项。

### Files Changed:

- Modified: `build_final_design_doc.py`
- Modified: `chapter4_model_design.md`
- Modified: `基于VLA统一训练与泛化评测框架的机器人基础模型研究_最终详细设计文档.md`
- Modified: `基于VLA统一训练与泛化评测框架的机器人基础模型研究_最终详细设计文档.docx`
- Modified: `docs/starflow_vla/TASK_BREAKDOWN.md`
- Modified: `docs/starflow_vla/CODEX_ISSUES.md`
- Modified: `docs/starflow_vla/ACCEPTANCE_CHECKLIST.md`
- Modified: `docs/starflow_vla/P0_IMPLEMENTATION_PLAN.md`
- Modified: `docs/starflow_vla/IMPLEMENTATION_LOG.md`
- Modified: `docs/starflow_vla/CODEX_EXECUTION_GUIDE.md`
- Modified if exists: `docs/starflow_vla/ALGORITHM_OPTIMIZATION_PLAN.md`
- Not found: `docs/design/基于VLA统一训练与泛化评测框架的机器人基础模型研究_最终详细设计文档.md`
- Not found: `docs/starflow_vla/MODULE_MAPPING.md`
- Not found: `docs/starflow_vla/EXPERIMENT_MATRIX.md`
- Not found: `docs/starflow_vla/CAREER_PRESENTATION.md`

### Tests Run:

```bash
rg -n "VGGT|geometry fusion|RGB-3D|ObservationGeometryAdapter|observation_geometry" docs/starflow_vla build_final_design_doc.py chapter4_model_design.md "基于VLA统一训练与泛化评测框架的机器人基础模型研究_最终详细设计文档.md"
rg -n "<forbidden project aliases>" docs/starflow_vla build_final_design_doc.py chapter4_model_design.md "基于VLA统一训练与泛化评测框架的机器人基础模型研究_最终详细设计文档.md"
rg -n "<forbidden Stage A hardware aliases>" docs/starflow_vla build_final_design_doc.py chapter4_model_design.md "基于VLA统一训练与泛化评测框架的机器人基础模型研究_最终详细设计文档.md"
```

### Test Results:

- Pass: 文档命名口径、环境口径与 VGGT 边界检查完成。
- Fail: none
- Not run: StarFlowVLA 代码测试、真实模型加载、训练、评测、部署、VGGT 接入。
- Need target verification: 《基于世界模型的移动操作规划与决策框架研究》中的 VGGT world model 实验和所有 RGB-3D fusion 指标。

### Environment Boundary Statement:

本次结果仅代表 Stage A 1×A100 40G 轻量构建验证阶段的文档修正，不代表 Stage B 目标模型 smoke、8×A100 正式训练、完整 benchmark 评测或真实机器人部署结论。

### Deviations from Design:

无。本次修改不改变技术路线，只修正文档命名、环境与 VGGT 边界表述。

### Rollback Plan:

回滚本次文档修改即可，不影响 StarVLA 源代码。

### Next Step:

继续执行本项目 P0-M0 / P0-M1 / P0-M2，不引入 VGGT。

## Record DOC-M5

### Task ID
DOC-M5

### Goal
完成 `docs_zh` 下 StarFlow-VLA 文档归档后的最小一致性检查与 README 索引补全。

### Environment
Stage A 1×A100 40G lightweight validation；本任务只做文档整理，不加载模型，不运行训练、评测或部署。

### Files Changed
- Added or updated: README.md
- Not added or modified: DESIGN.md already exists
- Modified if needed: IMPLEMENTATION_LOG.md
- Other changes: 无

### Checks
- `ls docs_zh`
- `find docs_zh -maxdepth 3 -type f | sort`
- `rg -n "项目一|项目二|本地/4090|Stage A local|VGGT 降级|移除 VGGT|从项目一主线" docs_zh`
- `rg -n "本项目|StarFlow-VLA|基于世界模型的移动操作规划与决策框架研究|Stage A|Stage B|A100 40G" docs_zh`
- 未发现旧口径匹配项。
- 未发现未修正问题。

### Not Run
未运行 StarFlowVLA 代码测试、真实模型加载、训练、评测、部署或 VGGT 接入。

### Next
进入 P0-M0 / P0-M1 / P0-M2。

## Record DOC-M6

### Task ID
DOC-M6

### Goal
继续完成 `docs_zh/starflow_vla` 下 P0-M0 / P0-M1 / P0-M11 的最小文档骨架，停止在 P0-M2 源码实现之前。

### Environment
Stage A 1×A100 40G lightweight validation；本任务只做文档整理，不加载模型，不运行训练、评测或部署。

### Files Changed
- Added: `DESIGN_FREEZE_CHECK.md`
- Added: `BASELINE_VERSION.md`
- Added: `UPSTREAM_COMPATIBILITY.md`
- Added: `PATCH_MANIFEST.md`
- Added: `MODULE_MAPPING.md`
- Modified: `IMPLEMENTATION_LOG.md`
- Source code changes: none

### Checks
- `sed -n '1,240p' SESSION.md`
- `sed -n '1,240p' TODO.md`
- `sed -n '1,220p' docs_zh/starflow_vla/P0_IMPLEMENTATION_PLAN.md`
- `sed -n '1,220p' docs_zh/starflow_vla/TASK_BREAKDOWN.md`
- `git status --short`
- `git rev-parse --abbrev-ref HEAD`
- `git rev-parse HEAD`
- `git log -1 --format='%H%n%h %ci %s'`
- `git show -s --format='%H%n%h %ci %s' 42170b2a4df3877ccf6581948e2198d37c363c7f`
- `rg -n "^version =|version_id" pyproject.toml examples/LIBERO/train_files/starvla_cotrain_libero.yaml`
- `rg -n "文档版本|V4\\.6\\.2|future_tokens|cross-DiT|PerceiverAdapter|FlowCondition|14D|P0" docs_zh/starflow_vla/DESIGN.md docs_zh/starflow_vla/TASK_BREAKDOWN.md docs_zh/starflow_vla/P0_IMPLEMENTATION_PLAN.md`

### Findings
- `DESIGN.md` 已是 `V4.6.2 Implementation Trace Patch`。
- P0-M0 文档冻结口径可通过最小检查。
- 当前工作区 branch 为 `starVLA_dev`，HEAD 为 `f723d4b5629cd0f5fb8bc8a562dd744f7a79b552`。
- 设计文档锚点 `42170b2a4df3877ccf6581948e2198d37c363c7f` 当前本地不可解析，进入 P0-M2 前必须先确认最终实现基线。
- `pyproject.toml` 版本为 `1.0.1`，LIBERO 示例配置 `version_id: "0.21"`。

### Not Run
未运行 StarFlowVLA 代码测试、真实模型加载、训练、评测、部署或 VGGT 接入。

### Next
先确认 P0 实现基线使用当前 HEAD 还是设计锚点；确认后再进入 P0-M2 / P0-M3。

## Record DOC-M7

### Task ID
DOC-M7

### Goal
按用户选择 2，拉取并切换到设计文档锚点 `42170b2a4df3877ccf6581948e2198d37c363c7f`，为后续 P0-M2 做基线准备。

### Environment
Stage A 1×A100 40G lightweight validation；本任务只做 git 基线定位与文档记录，不加载模型，不运行训练、评测或部署。

### Files Changed
- Modified: `BASELINE_VERSION.md`
- Modified: `UPSTREAM_COMPATIBILITY.md`
- Modified: `IMPLEMENTATION_LOG.md`
- Source code changes: none

### Checks
- `git remote -v`
- `git fetch origin`
- `git ls-remote origin | rg "42170b2|42170b2a4df3877ccf6581948e2198d37c363c7f"`
- `git ls-remote https://github.com/starVLA/starVLA.git | rg "42170b2|42170b2a4df3877ccf6581948e2198d37c363c7f"`
- `git ls-remote --heads origin`
- `git ls-remote --heads https://github.com/starVLA/starVLA.git`
- `git fetch https://github.com/starVLA/starVLA.git starVLA_dev:refs/remotes/starvla-official/starVLA_dev`
- `git cat-file -t 42170b2a4df3877ccf6581948e2198d37c363c7f`
- `git show -s --format='%H%n%h %ci %s' 42170b2a4df3877ccf6581948e2198d37c363c7f`
- `git stash push -m "codex-preserve-session-before-42170b2" -- SESSION.md`
- `git switch --detach 42170b2a4df3877ccf6581948e2198d37c363c7f`
- `git rev-parse --abbrev-ref HEAD`
- `git rev-parse HEAD`
- `git stash list | sed -n '1,10p'`
- `git status --short --untracked-files=normal`

### Findings
- 当前 fork `origin` 的 `starVLA_dev` 为 `f723d4b5629cd0f5fb8bc8a562dd744f7a79b552`，不包含设计锚点。
- 官方仓库 `https://github.com/starVLA/starVLA.git` 的 `starVLA_dev` 包含设计锚点。
- 已切换到 detached HEAD `42170b2a4df3877ccf6581948e2198d37c363c7f`。
- `SESSION.md` 本地修改已保存到 `stash@{0}`，名称为 `codex-preserve-session-before-42170b2`。
- 未跟踪目录 `.libero/` 与 `LIBERO/` 保留未动。

### Not Run
未运行 StarFlowVLA 代码测试、真实模型加载、训练、评测、部署或 VGGT 接入。

### Next
从 `42170b2a4df3877ccf6581948e2198d37c363c7f` 创建 P0 实现分支后，再进入 P0-M2 / P0-M3。

## Record DOC-M8

### Task ID
DOC-M8

### Goal
在 fork 合并官方 `starVLA_dev` 后，同步更新 `docs_zh/starflow_vla` 的文档索引、基线版本记录、上游兼容策略和 patch manifest。

### Environment
Stage A 1×A100 40G lightweight validation；本任务只做文档整理，不加载模型，不运行训练、评测或部署。

### Files Changed
- Modified: `README.md`
- Modified: `BASELINE_VERSION.md`
- Modified: `UPSTREAM_COMPATIBILITY.md`
- Modified: `PATCH_MANIFEST.md`
- Modified: `IMPLEMENTATION_LOG.md`
- Source code changes: none in this DOC-M8 task

### Checks
- `git rev-parse --abbrev-ref HEAD`
- `git log -1 --format='%H%n%h %ci %s'`
- `git log -1 --format='%H%n%h %ci %s' starvla-official/starVLA_dev`
- `git log -1 --format='%H%n%h %ci %s' origin/merge-official-starvla-dev`
- `find docs_zh/starflow_vla -maxdepth 1 -type f | sort`
- `rg -n "^version =|version_id" pyproject.toml examples/LIBERO/train_files/starvla_cotrain_libero.yaml`

### Findings
- 当前分支为 `merge-official-starvla-dev`。
- 当前 HEAD 为 `5d94274a2e6131a5dbbc49546fc662d2ccbe5117`。
- 官方 `starvla-official/starVLA_dev` HEAD 为 `cdf5434438f4449cff85e3588956f7706a5c9cc3`。
- 官方合并提交为 `9a8f5057888faf45eab62fddb97a0e9b84c44a90`。
- `pyproject.toml` 版本仍为 `1.0.1`，LIBERO 示例配置 `version_id: "0.21"`。
- `docs_zh/starflow_vla` 下已有 14 个文档文件。

### Not Run
未运行 StarFlowVLA 代码测试、真实模型加载、训练、评测、部署或 VGGT 接入。

### Next
进入 P0-M2 前先做 compatibility audit，重点检查 framework registry、QwenPI_v3、LayerwiseFM/GR00T、checkpoint loader 和 dataloader 合并后的接口状态。

## Record DOC-M9

### Task ID
DOC-M9

### Goal
完成 P0-M2 前 compatibility audit，确认官方合并后 StarFlow-VLA 的 framework registry、QwenPI_v3、LayerwiseFM/GR00T、checkpoint loader 和 dataloader 前置接口状态。

### Environment
Stage A 1×A100 40G lightweight validation；本任务只做静态阅读、语法检查和文档记录，不加载模型，不运行训练、评测或部署。

### Files Changed
- Modified: `UPSTREAM_COMPATIBILITY.md`
- Modified: `MODULE_MAPPING.md`
- Modified: `PATCH_MANIFEST.md`
- Modified: `IMPLEMENTATION_LOG.md`
- Source code changes: none

### Checks
- `git status --short --untracked-files=normal`
- `sed -n '1,260p' starVLA/model/framework/base_framework.py`
- `rg -n "FRAMEWORK_REGISTRY|def build_framework|register\\(" starVLA/model/framework -S`
- `sed -n '1,260p' starVLA/dataloader/__init__.py`
- `rg -n "load_model_weights|_resolve_model_checkpoint|lightweight|checkpoint|safetensors|deepspeed" starVLA/model/framework/share_tools.py starVLA/training/train_starvla.py starVLA/training/trainer_utils/trainer_tools.py`
- `sed -n '1,620p' starVLA/model/framework/VLM4A/QwenPI_v3.py`
- `rg -n "num_target_vision_tokens|future_tokens|state_encoder|predict_action|forward\\(|action_model|action_dim|num_inference_timesteps|Euler|euler|cross" starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py starVLA/model/modules/action_model/GR00T_ActionHeader.py`
- `python -m py_compile starVLA/model/framework/base_framework.py starVLA/model/framework/VLM4A/QwenPI_v3.py starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py starVLA/model/modules/action_model/GR00T_ActionHeader.py starVLA/model/framework/share_tools.py starVLA/training/train_starvla.py starVLA/training/trainer_utils/trainer_tools.py starVLA/dataloader/__init__.py`

### Findings
- `FRAMEWORK_REGISTRY` 与 `build_framework(cfg)` 仍按 `cfg.framework.name` 构建，并自动导入 framework module。
- 当前未注册 `StarFlowVLA`，符合 P0-M2 的新增任务边界。
- `QwenPI_v3` 仍注册为 `QwenPI_v3`，并保留 `qwen_vl_interface`、`project_layers`、`action_model`、`forward()`、`predict_action()` 和 state-to-instruction 路径。
- `LayerwiseFM_ActionHeader.py` 仍保留 `future_tokens`、`num_target_vision_tokens`、`state_encoder`、layer-wise cross-attention 和 Euler `predict_action()`。
- `GR00T_ActionHeader.py` 仍保留 `future_tokens`、`state_encoder` 和 Euler `predict_action()`。
- checkpoint loader 仍保留单文件、目录、DeepSpeed、safetensors 分片和 lightweight training checkpoint 解析逻辑。
- dataloader 合并后同时保留官方 balance 参数和 fork 的 `num_workers` / `prefetch_factor` 配置。
- 静态语法检查通过。

### Not Run
未运行 StarFlowVLA 代码测试、真实模型加载、checkpoint 加载、训练、评测、部署或 VGGT 接入。

### Next
进入 P0-M2：新增 `StarFlowVLA` framework facade 入口，先做 registry / import / config parse 级验证，不复制 QwenPI_v3 主体逻辑。

## Record P0-M2

### Task ID
P0-M2

### Goal
新增 `StarFlowVLA` framework facade 入口，使 `FRAMEWORK_REGISTRY` 可找到 `StarFlowVLA`，并保持 StarVLA-native 继承复用路线。

### Environment
Stage A 1×A100 40G lightweight validation；本任务只做 registry / import / monkeypatch dry-run 和静态语法检查，不加载真实模型，不运行训练、评测或部署。

### Files Changed
- Added: `starVLA/model/framework/VLM4A/StarFlowVLA.py`
- Modified: `MODULE_MAPPING.md`
- Modified: `PATCH_MANIFEST.md`
- Modified: `IMPLEMENTATION_LOG.md`
- Not modified: `starVLA/model/framework/VLM4A/QwenPI_v3.py`
- Not modified: `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`
- Not modified: `starVLA/model/modules/action_model/GR00T_ActionHeader.py`

### Checks
- `python -m py_compile starVLA/model/framework/VLM4A/StarFlowVLA.py`
- `.venv/bin/python - <<'PY' ... AST inspect StarFlowVLA registry decorator / inheritance / method ownership ... PY`
- `.venv/bin/python - <<'PY' ... import StarFlowVLA and inspect registry/subclass/method ownership ... PY`
- `.venv/bin/python - <<'PY' ... monkeypatch StarFlowVLA.__init__ and build_framework(cfg) ... PY`
- `rg -n "def forward|def predict_action|FRAMEWORK_REGISTRY.register|describe_starflow_mapping" starVLA/model/framework/VLM4A/StarFlowVLA.py`

### Findings
- `StarFlowVLA.py` 语法检查通过。
- AST 检查确认 `StarFlowVLA` 使用 `FRAMEWORK_REGISTRY.register("StarFlowVLA")`。
- `.venv` 下 import 级检查通过，`FRAMEWORK_REGISTRY["StarFlowVLA"]` 指向 `StarFlowVLA`。
- `StarFlowVLA` 继承 `Qwen_PI_v3`。
- `StarFlowVLA` 未定义自己的 `forward()` 或 `predict_action()`，因此不会复制 QwenPI_v3 主体逻辑。
- `build_framework(cfg)` 可在 monkeypatch `__init__` 的 dry-run 下通过 `framework.name=StarFlowVLA` 构建 facade 实例。
- `describe_starflow_mapping()` 提供可序列化 mapping 元数据，checkpoint 写入留到 P0-M3。
- 默认 Python 缺少 `torch` / `omegaconf`；按用户说明改用 `.venv` 并等待 torch 导入完成后，registry 与 build dry-run 均通过。

### Not Run
未运行真实模型加载、checkpoint 加载、forward/backward、single batch overfit、训练、评测、部署或 VGGT 接入。

### Next
进入 P0-M3：新增 `starflow_mapping` manifest 写入路径或 mapping schema，并继续避免修改 QwenPI_v3 / LayerwiseFM / GR00T 主体逻辑。

## Record P0-M3

### Task ID
P0-M3

### Goal
新增 `starflow_mapping` manifest schema 与旁路 JSON 保存工具，使 StarFlow-VLA P0 路线可追踪且不侵入 checkpoint 主流程。

### Environment
Stage A 1×A100 40G lightweight validation；本任务只做 schema 构造、JSON 保存、registry import 和静态语法检查，不加载真实模型，不运行训练、评测或部署。

### Files Changed
- Added: `starVLA/model/modules/starflow_vla/__init__.py`
- Added: `starVLA/model/modules/starflow_vla/mapping.py`
- Modified: `starVLA/model/framework/VLM4A/StarFlowVLA.py`
- Modified: `MODULE_MAPPING.md`
- Modified: `PATCH_MANIFEST.md`
- Modified: `IMPLEMENTATION_LOG.md`
- Not modified: `starVLA/model/framework/VLM4A/QwenPI_v3.py`
- Not modified: `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`
- Not modified: `starVLA/model/modules/action_model/GR00T_ActionHeader.py`

### Checks
- `python -m py_compile starVLA/model/modules/starflow_vla/__init__.py starVLA/model/modules/starflow_vla/mapping.py starVLA/model/framework/VLM4A/StarFlowVLA.py`
- `.venv/bin/python - <<'PY' ... build_starflow_mapping / save_starflow_mapping JSON round-trip ... PY`
- `.venv/bin/python - <<'PY' ... AST inspect StarFlowVLA mapping delegation / method ownership ... PY`
- `.venv/bin/python - <<'PY' ... import StarFlowVLA and inspect registry/subclass/method ownership ... PY`
- `rg -n "def forward|def predict_action|FRAMEWORK_REGISTRY.register|build_starflow_mapping|save_starflow_mapping|schema_version" starVLA/model/framework/VLM4A/StarFlowVLA.py starVLA/model/modules/starflow_vla`

### Findings
- `mapping.py` 不依赖 torch，可独立构造可 JSON 序列化的 P0 mapping。
- `save_starflow_mapping()` 可将 mapping 写为 `starflow_mapping.json`，供 checkpoint 旁路追踪使用。
- `StarFlowVLA.describe_starflow_mapping()` 已改为委托 `build_starflow_mapping(self.config)`，避免 facade 与 schema 字段漂移。
- `StarFlowVLA` 仍继承 `Qwen_PI_v3`，未定义自己的 `forward()` 或 `predict_action()`。
- `.venv` 下 registry import 级复验通过；该检查触发 torch 导入，按用户说明等待约 5 分钟完成。

### Not Run
未运行真实模型加载、checkpoint 保存/加载、forward/backward、single batch overfit、训练、评测、部署或 VGGT 接入。

### Next
进入 P0-M4：做 QwenPI_v3 reuse smoke；优先保持 monkeypatch / 单 batch 级验证，不复制或重写 QwenPI_v3 主体逻辑。

## Record P0-M4

### Task ID
P0-M4

### Goal
验证 `StarFlowVLA` 真实复用 `QwenPI_v3` 的主路径，避免复制 QwenPI_v3 主体构建、`forward()` 或 `predict_action()`。

### Environment
Stage A 1×A100 40G lightweight validation；本任务只做 monkeypatch reuse smoke 和标准库 unittest，不加载真实模型，不运行训练、评测或部署。

### Files Changed
- Added: `tests/test_starflow_vla_reuse.py`
- Modified: `ACCEPTANCE_CHECKLIST.md`
- Modified: `PATCH_MANIFEST.md`
- Modified: `IMPLEMENTATION_LOG.md`
- Not modified: `starVLA/model/framework/VLM4A/QwenPI_v3.py`
- Not modified: `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`
- Not modified: `starVLA/model/modules/action_model/GR00T_ActionHeader.py`

### Checks
- `python -m py_compile tests/test_starflow_vla_reuse.py`
- `.venv/bin/python -m pytest tests/test_starflow_vla_reuse.py -q`
- `.venv/bin/python -m unittest tests.test_starflow_vla_reuse -v`

### Findings
- `.venv` 中未安装 `pytest`，因此目标 pytest 命令未运行成功，未新增依赖。
- 使用标准库 `unittest` 跑同一测试文件，4 个用例通过。
- 测试确认 `StarFlowVLA` 继承 `Qwen_PI_v3`。
- 测试确认 `StarFlowVLA` 未定义自己的 `forward()` 或 `predict_action()`，方法对象来自 `Qwen_PI_v3`。
- 测试通过 monkeypatch `Qwen_PI_v3.__init__` 验证 `build_framework(cfg)` 构建 `StarFlowVLA` 时只走一次 QwenPI_v3 初始化路径，不触发真实模型加载。
- 测试确认 state-to-instruction 默认路径可在 `StarFlowVLA` 实例上复用。
- 测试确认 `describe_starflow_mapping()` 返回 P0 mapping 关键字段。

### Not Run
未运行真实模型加载、QwenPI_v3 baseline 真实构建、checkpoint 保存/加载、forward/backward、single batch overfit、训练、评测、部署或 VGGT 接入。

### Next
进入 P0-M5 / P0-M8 前，先确认是否已有 LIBERO small split 数据与本地模型权重；若缺失，则只能继续做 config / batch schema 级 smoke。

## Record P0-M5

### Task ID
P0-M5

### Goal
新增 Stage1 `StarFlowVLA + QwenPI_v3 native + LayerwiseFM` 单臂 7DoF smoke 配置，并完成配置解析与 registry dry-run。

### Environment
Stage A 1×A100 40G lightweight validation；本任务只做配置新增、YAML 解析、兼容层检查和 monkeypatch build dry-run，不加载真实模型，不运行训练、评测或部署。

### Files Changed
- Added: `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml`
- Modified: `ACCEPTANCE_CHECKLIST.md`
- Modified: `PATCH_MANIFEST.md`
- Modified: `IMPLEMENTATION_LOG.md`
- Not modified: `starVLA/model/framework/VLM4A/QwenPI_v3.py`
- Not modified: `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`
- Not modified: `starVLA/model/modules/action_model/GR00T_ActionHeader.py`

### Checks
- `find playground -maxdepth 4 ...`
- `ls -ld playground/Datasets playground/Datasets/LEROBOT_LIBERO_DATA`
- `.venv/bin/python - <<'PY' ... OmegaConf.load stage1 config and assert key fields ... PY`
- `.venv/bin/python - <<'PY' ... check base_vlm path and LIBERO data root ... PY`
- `.venv/bin/python - <<'PY' ... apply_config_compat stage1 config ... PY`
- `python -m py_compile tests/test_starflow_vla_reuse.py starVLA/model/framework/VLM4A/StarFlowVLA.py starVLA/model/modules/starflow_vla/mapping.py`
- `.venv/bin/python - <<'PY' ... monkeypatch Qwen_PI_v3.__init__ and build_framework(stage1 cfg) ... PY`

### Findings
- Stage1 配置解析通过，`framework.name=StarFlowVLA`。
- Stage1 配置使用 `action_model_type=LayerwiseFM`、`action_dim=7`、`state_dim=7`、`action_horizon=8`、`num_target_vision_tokens=32`。
- Stage1 配置显式启用 `datasets.vla_data.include_state=true`，以复用 QwenPI_v3 state-to-instruction 路径。
- Stage1 配置未启用 `max_action_dim=14 + action_mask`。
- `playground/Pretrained_models/Qwen3-VL-4B-Instruct` 软链接存在，指向 `/gemini/pretrain/Qwen3-VL-4B-Instruct`。
- `playground/Datasets/LEROBOT_LIBERO_DATA` 当前不存在，因此不能运行 LIBERO batch schema、真实 forward/backward、single batch overfit 或训练。
- `apply_config_compat()` 后 action horizon / future window / diffusion config 字段保持一致。
- `build_framework(stage1 cfg)` 在 monkeypatch `Qwen_PI_v3.__init__` 的 dry-run 下通过，不触发真实模型加载。

### Not Run
未运行真实模型加载、LIBERO batch schema、forward/backward、loss finite、single batch overfit、checkpoint 保存/加载、训练、评测、部署或 VGGT 接入。

### Next
进入 P0-M8：补 LIBERO 数据目录可用性检查与 batch schema smoke；若 `playground/Datasets/LEROBOT_LIBERO_DATA` 仍缺失，则只记录阻塞并等待数据准备。

## Record P0-M8

### Task ID
P0-M8

### Goal
新增 LIBERO 最小 batch schema smoke 测试，数据可用时检查 `image/lang/state/action`、7DoF action 和 NaN/Inf。

### Environment
Stage A 1×A100 40G lightweight validation；本任务只做数据目录可用性检查和可跳过的 unittest，不加载真实模型，不运行训练、评测或部署。

### Files Changed
- Added: `tests/test_starflow_libero_batch.py`
- Modified: `PATCH_MANIFEST.md`
- Modified: `IMPLEMENTATION_LOG.md`
- Not modified: `starVLA/dataloader/`
- Not modified: `starVLA/model/framework/VLM4A/QwenPI_v3.py`
- Not modified: `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`

### Checks
- `python -m py_compile tests/test_starflow_libero_batch.py`
- `.venv/bin/python -m unittest tests.test_starflow_libero_batch -v`

### Findings
- `tests/test_starflow_libero_batch.py` 语法检查通过。
- `.venv` 下 unittest 运行成功，但 1 个用例被显式 skip。
- skip 原因：`playground/Datasets/LEROBOT_LIBERO_DATA` 不存在。
- 测试逻辑在数据目录存在时才导入 dataloader 并构建一个 batch，检查 `image/lang/state/action`、`action_dim=7`、`state_dim=7`、NaN/Inf。
- 当前不能标记 `LIBERO minimal batch schema pass` 或 `batch 含 image / instruction / state / action` 为通过，因为未读取真实数据。

### Not Run
未运行真实 LIBERO batch 读取、真实模型加载、forward/backward、loss finite、single batch overfit、checkpoint 保存/加载、训练、评测、部署或 VGGT 接入。

### Next
准备或挂载 `playground/Datasets/LEROBOT_LIBERO_DATA` 后，重新运行 `.venv/bin/python -m unittest tests.test_starflow_libero_batch -v`；通过后再进入真实 P0-M5 forward/backward smoke。

## Record P0-M6

### Task ID
P0-M6

### Goal
新增 H2 对照用 `QwenOFT + MLP_ActionHeader` 单臂 7DoF baseline smoke 配置，并完成配置解析与 registry dry-run。

### Environment
Stage A 1×A100 40G lightweight validation；本任务只做配置新增、YAML 解析、兼容层检查和 monkeypatch build dry-run，不加载真实模型，不运行训练、评测或部署。

### Files Changed
- Added: `configs/starflow_vla/stage2_mlp_baseline.yaml`
- Modified: `ACCEPTANCE_CHECKLIST.md`
- Modified: `MODULE_MAPPING.md`
- Modified: `PATCH_MANIFEST.md`
- Modified: `IMPLEMENTATION_LOG.md`
- Not modified: `starVLA/model/framework/VLM4A/QwenOFT.py`
- Not modified: `starVLA/model/modules/action_model/MLP_ActionHeader.py`
- Not modified: `starVLA/model/modules/action_model/VLA_AdapterHeader.py`

### Checks
- `.venv/bin/python - <<'PY' ... OmegaConf.load stage2 config and assert key fields ... PY`
- `.venv/bin/python - <<'PY' ... apply_config_compat stage2 config ... PY`
- `.venv/bin/python - <<'PY' ... monkeypatch Qwenvl_OFT.__init__ and build_framework(stage2 cfg) ... PY`

### Findings
- Stage2 baseline 配置解析通过，`framework.name=QwenOFT`。
- Stage2 baseline 配置使用 `action_model_type=MLP`、`action_dim=7`、`state_dim=7`、`action_horizon=8`。
- Stage2 baseline 复用当前已存在的 `playground/Pretrained_models/Qwen3-VL-4B-Instruct` 软链接，未下载新模型。
- `apply_config_compat()` 后 action horizon / future window / action hidden dim 字段保持一致。
- `build_framework(stage2 cfg)` 在 monkeypatch `Qwenvl_OFT.__init__` 的 dry-run 下通过，不触发真实模型加载。
- 本阶段只完成 QwenOFT + MLP baseline；VLA_AdapterHeader 对照未新增 runtime 或配置，留后续扩展。
- 由于 `playground/Datasets/LEROBOT_LIBERO_DATA` 不存在，未运行真实 baseline batch、single batch overfit 或训练。

### Not Run
未运行真实模型加载、LIBERO batch schema、forward/backward、loss finite、single batch overfit、checkpoint 保存/加载、训练、评测、部署或 VGGT 接入。

### Next
进入 P0-M7：新增 future_tokens 消融配置；真实 forward/backward 仍依赖 LIBERO 数据目录准备完成。

## Record P0-M7

### Task ID
P0-M7

### Goal
新增 `future_tokens + cross-DiT` 消融配置，覆盖 `num_target_vision_tokens=0/8/16/32/64` 的配置级入口。

### Environment
Stage A 1×A100 40G lightweight validation；本任务只做配置新增、YAML 解析、兼容层检查和 monkeypatch build dry-run，不加载真实模型，不运行训练、评测或部署。

### Files Changed
- Added: `configs/starflow_vla/stage3_future_token_ablation.yaml`
- Modified: `MODULE_MAPPING.md`
- Modified: `PATCH_MANIFEST.md`
- Modified: `IMPLEMENTATION_LOG.md`
- Not modified: `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py`
- Not modified: `starVLA/model/modules/action_model/GR00T_ActionHeader.py`
- Not modified: `starVLA/model/framework/VLM4A/QwenPI_v3.py`

### Checks
- `.venv/bin/python - <<'PY' ... derive and assert 0/8/16/32/64 ablation variants ... PY`
- `.venv/bin/python - <<'PY' ... apply_config_compat stage3 config ... PY`
- `.venv/bin/python - <<'PY' ... monkeypatch Qwen_PI_v3.__init__ and build_framework(stage3 cfg) ... PY`

### Findings
- Stage3 配置解析通过，默认 `num_target_vision_tokens=32`。
- `ablation.num_target_vision_tokens_values` 覆盖 `[0, 8, 16, 32, 64]`。
- 5 组 token 数均可在内存中派生为 OmegaConf 配置并序列化。
- `ablation.adapter_mode=future_token_cross_dit`。
- `apply_config_compat()` 后 Stage3 配置保持 `StarFlowVLA + LayerwiseFM + action_dim=7`。
- `build_framework(stage3 cfg)` 在 monkeypatch `Qwen_PI_v3.__init__` 的 dry-run 下通过，不触发真实模型加载。
- `describe_starflow_mapping()` 返回 `adapter_mode=future_token_cross_dit` 与默认 `num_target_vision_tokens=32`。
- 由于 `playground/Datasets/LEROBOT_LIBERO_DATA` 不存在，未运行 5 组真实 forward、single batch overfit 或训练。

### Not Run
未运行真实模型加载、LIBERO batch schema、forward/backward、loss finite、single batch overfit、checkpoint 保存/加载、训练、评测、部署或 VGGT 接入。

### Next
进入 P0-M9：在 checkpoint 旁路保存 `starflow_mapping.json` 的工具级接入；真实 checkpoint save/load 仍需数据与训练闭环可用后复验。

## Record P0-M9

### Task ID
P0-M9

### Goal
新增 checkpoint 旁路 `starflow_mapping` 保存工具，使目录 checkpoint 和单文件 checkpoint 都可记录 StarFlow-VLA manifest。

### Environment
Stage A 1×A100 40G lightweight validation；本任务只做工具函数和标准库 unittest，不加载真实模型，不运行训练、评测或部署。

### Files Changed
- Modified: `starVLA/model/modules/starflow_vla/__init__.py`
- Modified: `starVLA/model/modules/starflow_vla/mapping.py`
- Added: `tests/test_starflow_checkpoint_mapping.py`
- Modified: `ACCEPTANCE_CHECKLIST.md`
- Modified: `PATCH_MANIFEST.md`
- Modified: `IMPLEMENTATION_LOG.md`
- Not modified: `starVLA/training/train_starvla.py`
- Not modified: `starVLA/training/train_starvla_cotrain.py`

### Checks
- `python -m py_compile starVLA/model/modules/starflow_vla/__init__.py starVLA/model/modules/starflow_vla/mapping.py tests/test_starflow_checkpoint_mapping.py`
- `.venv/bin/python -m unittest tests.test_starflow_checkpoint_mapping -v`

### Findings
- 新增 `save_starflow_checkpoint_mapping()`，可从 config 构造 mapping 并保存 checkpoint sidecar JSON。
- 目录 checkpoint 会写入 `<checkpoint_dir>/starflow_mapping.json`。
- 单文件 checkpoint 会写入 `<checkpoint_file>.starflow_mapping.json`。
- sidecar JSON 可记录 `patch_manifest_hash`、`starvla_commit` 和 `config_schema`。
- unittest 2 个用例通过。
- 当前未改训练主循环，避免在真实训练闭环未完成前侵入复杂 checkpoint 保存路径。

### Not Run
未运行真实 checkpoint 保存/加载、resume 100 step、真实模型加载、LIBERO batch schema、forward/backward、loss finite、single batch overfit、训练、评测、部署或 VGGT 接入。

### Next
进入 P0-M10 前需要真实 P0-M5 训练产物；当前仍受 `playground/Datasets/LEROBOT_LIBERO_DATA` 缺失阻塞。

## Record P0-M10

### Task ID
P0-M10

### Goal
新增 LIBERO eval smoke preflight，确认 eval 脚本语法和 P0 checkpoint / `starflow_mapping` 前置门禁。

### Environment
Stage A 1×A100 40G lightweight validation；本任务只做 shell 语法检查和可跳过的 unittest，不启动 policy server，不运行 LIBERO rollout。

### Files Changed
- Added: `docs_zh/starflow_vla/EVAL_SMOKE.md`
- Added: `tests/test_starflow_eval_preflight.py`
- Modified: `PATCH_MANIFEST.md`
- Modified: `IMPLEMENTATION_LOG.md`
- Not modified: `examples/LIBERO/eval_files/run_policy_server.sh`
- Not modified: `examples/LIBERO/eval_files/eval_libero.sh`
- Not modified: `examples/LIBERO/eval_files/eval_libero.py`

### Checks
- `python -m py_compile tests/test_starflow_eval_preflight.py`
- `bash -n examples/LIBERO/eval_files/run_policy_server.sh && bash -n examples/LIBERO/eval_files/eval_libero.sh`
- `.venv/bin/python -m unittest tests.test_starflow_eval_preflight -v`

### Findings
- eval shell 脚本语法检查通过。
- `tests.test_starflow_eval_preflight` 运行成功，2 个用例中 1 个通过、1 个 skip。
- skip 原因：`playground/Checkpoints/starflow_vla_stage1_qwenpi_v3_native/checkpoints` 不存在。
- `EVAL_SMOKE.md` 已记录真实 eval smoke 所需 P0 checkpoint、mapping sidecar、LIBERO_HOME 与 LIBERO 数据目录。
- 当前不能标记 LIBERO eval smoke pass、success_rate、failure category 或 eval report 为通过。

### Not Run
未启动 policy server，未运行 LIBERO rollout，未统计 success_rate / failure category，未运行真实模型加载、训练、评测、部署或 VGGT 接入。

### Next
进入 P0-M11：做文档与 patch 管理收口；真实 P0-M10 需要先完成 P0-M5 训练并产生 checkpoint。

## Record P0-M11

### Task ID
P0-M11

### Goal
完成 StarFlow-VLA P0 文档与 patch 管理收口，确保新增配置、测试、文档和主要代码产物可审计。

### Environment
Stage A 1×A100 40G lightweight validation；本任务只做文档治理和标准库 unittest，不加载真实模型，不运行训练、评测或部署。

### Files Changed
- Added: `docs_zh/starflow_vla/EXPERIMENT_MATRIX.md`
- Added: `tests/test_starflow_docs_governance.py`
- Modified: `README.md`
- Modified: `ACCEPTANCE_CHECKLIST.md`
- Modified: `PATCH_MANIFEST.md`
- Modified: `IMPLEMENTATION_LOG.md`

### Checks
- `python -m py_compile tests/test_starflow_docs_governance.py`
- `.venv/bin/python -m unittest tests.test_starflow_docs_governance -v`
- `find docs_zh/starflow_vla -maxdepth 1 -type f | sort`

### Findings
- `EXPERIMENT_MATRIX.md` 已新增，记录 P0-M5 / M6 / M7 / M8 / M9 / M10 的配置、测试和阻塞状态。
- `README.md` 已将 `EXPERIMENT_MATRIX.md` 加入索引，不再列为待补全文档。
- 文档治理测试确认核心文档存在、README 链接可解析、`PATCH_MANIFEST.md` 覆盖当前主要 StarFlow-VLA 产物。
- 当前未对 StarVLA 原始主体文件做需要 `STARFLOW_PATCH_BEGIN / END` 标记的 inline patch。

### Not Run
未运行真实模型加载、LIBERO batch schema、forward/backward、loss finite、single batch overfit、checkpoint 保存/加载、policy server、LIBERO rollout、训练、评测、部署或 VGGT 接入。

### Next
P0 Stage A 配置、文档、registry、mapping 和 preflight 工作已收口；继续 P0 Stage B 前需要准备 `playground/Datasets/LEROBOT_LIBERO_DATA` 并产生 P0 checkpoint。

## Record P0-M7a

### Task ID
P0-M7a

### Goal
补充 future token 规划槽位变量的轻量 mapping 测试，覆盖 `num_target_vision_tokens=0/8/16/32/64`。

### Environment
Stage A 1×A100 40G lightweight validation；本任务只做配置与 mapping 检查，不加载真实模型，不运行训练、评测或部署。

### Files Changed
- Added: `tests/test_starflow_future_token_variants.py`
- Modified: `tests/test_starflow_docs_governance.py`
- Modified: `PATCH_MANIFEST.md`
- Modified: `IMPLEMENTATION_LOG.md`

### Checks
- `python -m py_compile tests/test_starflow_future_token_variants.py tests/test_starflow_docs_governance.py`
- `.venv/bin/python -m unittest tests.test_starflow_future_token_variants -v`
- `.venv/bin/python -m unittest tests.test_starflow_docs_governance -v`

### Findings
- 5 组 `num_target_vision_tokens` 均可生成 `starflow_mapping`。
- mapping 保持 `adapter_mode=future_token_cross_dit`、`state_mode=discretized_instruction`、`action_head=LayerwiseFM`。
- 测试为轻量配置与 mapping 测试，不触发 framework 全量导入或真实模型加载。

### Not Run
未运行真实模型加载、forward/backward、loss finite、single batch overfit、训练、评测、部署或 VGGT 接入。

### Next
按用户要求继续准备缺失的 LIBERO 数据，下载目标目录为 `playground/Datasets/LEROBOT_LIBERO_DATA`。

## Record P0-M8-DATA

### Task ID
P0-M8-DATA

### Goal
补齐 P0 最小 LIBERO `libero_goal` 数据入口，复验 StarFlow-VLA Stage1 真实 batch schema。

### Environment
Stage B 1×A100 40G target smoke validation；使用仓库 `.venv`，本任务只下载/挂载数据并读取一个真实 batch，不加载 StarFlowVLA 模型，不运行训练、评测或部署。

### Files Changed
- Modified: `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml`
- Modified: `configs/starflow_vla/stage2_mlp_baseline.yaml`
- Modified: `configs/starflow_vla/stage3_future_token_ablation.yaml`
- Modified: `tests/test_starflow_libero_batch.py`
- Modified: `ACCEPTANCE_CHECKLIST.md`
- Modified: `EXPERIMENT_MATRIX.md`
- Modified: `PATCH_MANIFEST.md`
- Modified: `MEMORY/starflow_vla_environment.md`
- Modified: `examples/LIBERO/SESSION.md`
- Data side effects: downloaded `/gemini/code/datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot`; created symlink `playground/Datasets/LEROBOT_LIBERO_DATA -> /gemini/code/datasets/LEROBOT_LIBERO_DATA`; copied `modality.json` into dataset meta.

### Checks
- `.venv/bin/python -m unittest tests.test_starflow_libero_batch -v`：首次失败，根因是真实 LIBERO registry 返回 8D state。
- `.venv/bin/python -m unittest tests.test_starflow_libero_batch -v`：通过。

### Findings
- 真实 `libero_goal` batch 含 `image` / `lang` / `state` / `action`。
- action keys 为 `x,y,z,roll,pitch,yaw,gripper`，即 7D action。
- state keys 为 `x,y,z,roll,pitch,yaw,pad,gripper`，即 8D state。
- 已将三个 StarFlow-VLA P0 配置的 `state_dim` 从 7 对齐为 8。

### Not Run
未运行真实模型加载、forward/backward、loss finite、single batch overfit、checkpoint 保存/加载、policy server、LIBERO rollout、训练、评测、部署或 VGGT 接入。

### Next
进入 P0-M5 Stage B：在数据已可读的前提下执行最小真实模型 forward/backward / loss finite / single batch overfit 前置检查。

## Record P0-M5-STAGEB

### Task ID
P0-M5-STAGEB

### Goal
在真实 Qwen3-VL 本地模型与真实 LIBERO `libero_goal` batch 上复验 StarFlowVLA Stage1 最小 forward/backward、loss finite 与 single batch overfit 前置门禁。

### Environment
Stage B 1×A100 40G target smoke validation；使用仓库 `.venv`；本任务只运行单 batch smoke，不启动完整训练循环，不保存 checkpoint，不运行 LIBERO rollout 或部署。

### Files Changed
- Modified: `ACCEPTANCE_CHECKLIST.md`
- Modified: `EXPERIMENT_MATRIX.md`
- Modified: `PATCH_MANIFEST.md`
- Modified: `IMPLEMENTATION_LOG.md`
- Modified: `examples/LIBERO/SESSION.md`

### Checks
- `.venv/bin/python - <<'PY' ... PY`：加载 `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml`，取真实 LIBERO batch，构建 `StarFlowVLA`，冻结 `qwen_vl_interface`，执行单 batch forward/backward。
- `.venv/bin/python - <<'PY' ... PY`：固定同一真实 batch，冻结 `qwen_vl_interface`，优化 action head / projectors 6 步，执行 single batch overfit 前置验证。

### Findings
- 真实 batch shape：action `(8, 7)`，state `(1, 8)`。
- forward/backward smoke 通过，`action_loss` 为有限值，反传后可训练参数获得梯度。
- single batch overfit 前置验证通过，同一 batch 上 loss 从 `1.79121411` 降至 `0.10281464`。
- 本阶段未修改 `starVLA/model/` 源码。

### Not Run
未运行完整训练循环、checkpoint 保存/加载、resume、policy server、LIBERO rollout、success_rate 统计、评测、部署或 VGGT 接入。

### Next
进入 P0-M9/P0-M10 Stage B：生成最小 checkpoint 并复验 checkpoint sidecar / eval preflight；如需真实 LIBERO rollout，先确认可用 checkpoint 与 LIBERO 仿真依赖。

## Record P0-M9-STAGEB

### Task ID
P0-M9-STAGEB

### Goal
生成 StarFlowVLA Stage1 smoke checkpoint，复验 checkpoint sidecar、helper 文件、`load_model_weights()` 加载路径与 LIBERO eval preflight。

### Environment
Stage B 1×A100 40G target smoke validation；使用仓库 `.venv`；本任务手动保存 smoke checkpoint，不启动完整训练循环，不运行 policy server、LIBERO rollout 或部署。

### Files Changed
- Modified: `ACCEPTANCE_CHECKLIST.md`
- Modified: `EXPERIMENT_MATRIX.md`
- Modified: `EVAL_SMOKE.md`
- Modified: `PATCH_MANIFEST.md`
- Modified: `IMPLEMENTATION_LOG.md`
- Modified: `examples/LIBERO/SESSION.md`
- Data/checkpoint side effects: created `playground/Checkpoints/starflow_vla_stage1_qwenpi_v3_native/checkpoints/steps_1`

### Checks
- `.venv/bin/python - <<'PY' ... PY`：基于真实 batch 执行 1 个 optimizer step，并保存 smoke checkpoint。
- `.venv/bin/python - <<'PY' ... PY`：补写 `steps_1/starflow_mapping.json`，复制 `config.yaml`、`config.full.yaml`、`dataset_statistics.json`。
- `.venv/bin/python -m unittest tests.test_starflow_eval_preflight -v`：2 个用例通过。
- `.venv/bin/python - <<'PY' ... PY`：重新构建 `StarFlowVLA` 并执行 `load_model_weights(model, steps_1, preferred_format="pt", strict=True)`，通过。

### Findings
- `steps_1` 包含 3 个 `pytorch_model-*.bin` 分片、`pytorch_model.bin.index.json`、`optimizer_rank_00000.pt`、`scheduler.pt`、`trainer_state.json`、`starflow_mapping.json`、`config.yaml`、`config.full.yaml`、`dataset_statistics.json`。
- checkpoint load smoke 通过；加载时仅报告 rotary buffer 未使用的兼容警告。
- eval preflight 已从 checkpoint 缺失 skip 变为通过。
- 本阶段未修改 `starVLA/model/` 源码。

### Not Run
未运行 resume 100 step、完整训练循环、policy server、LIBERO rollout、success_rate 统计、评测、部署或 VGGT 接入。

### Next
进入 P0-M6 / P0-M7 Stage B 对照 smoke；如要进入真实 LIBERO rollout，需要确认 `.libero` 环境和 LIBERO 仿真依赖可用。

## Record P0-M6-STAGEB

### Task ID
P0-M6-STAGEB

### Goal
在真实 Qwen3-VL 本地模型与真实 LIBERO `libero_goal` batch 上复验 QwenOFT + MLP baseline 最小 forward/backward、loss finite 与 single batch overfit 前置门禁。

### Environment
Stage B 1×A100 40G target smoke validation；使用仓库 `.venv`；本任务只运行单 batch smoke，不保存 baseline checkpoint，不运行完整训练、评测或部署。

### Files Changed
- Modified: `ACCEPTANCE_CHECKLIST.md`
- Modified: `EXPERIMENT_MATRIX.md`
- Modified: `PATCH_MANIFEST.md`
- Modified: `IMPLEMENTATION_LOG.md`
- Modified: `examples/LIBERO/SESSION.md`

### Checks
- `.venv/bin/python - <<'PY' ... PY`：加载 `configs/starflow_vla/stage2_mlp_baseline.yaml`，取真实 LIBERO batch，构建 `QwenOFT`，冻结 `qwen_vl_interface`，优化 MLP action head 6 步。

### Findings
- 真实 batch shape：action `(8, 7)`，state `(1, 8)`。
- forward/backward smoke 通过，`action_loss` 为有限值。
- single batch overfit 前置验证通过，同一 batch 上 loss 从 `0.87645137` 降至 `0.48976591`。
- 本阶段未修改 `starVLA/model/` 源码。

### Not Run
未保存 baseline checkpoint，未运行完整训练循环、policy server、LIBERO rollout、success_rate 统计、评测、部署或 VGGT 接入。

### Next
进入 P0-M7 Stage B：复验 future token `num_target_vision_tokens=0/8/16/32/64` 的真实 forward / single batch smoke。

## Record P0-M7-STAGEB

### Task ID
P0-M7-STAGEB

### Goal
在真实 Qwen3-VL 本地模型与真实 LIBERO `libero_goal` batch 上复验 future token `num_target_vision_tokens=0/8/16/32/64` 五组最小 forward/backward、loss finite 与 single batch overfit 前置门禁。

### Environment
Stage B 1×A100 40G target smoke validation；使用仓库 `.venv`；本任务只运行单 batch smoke，不保存 5 组 checkpoint，不运行完整训练、评测或部署。

### Files Changed
- Modified: `ACCEPTANCE_CHECKLIST.md`
- Modified: `EXPERIMENT_MATRIX.md`
- Modified: `PATCH_MANIFEST.md`
- Modified: `IMPLEMENTATION_LOG.md`
- Modified: `examples/LIBERO/SESSION.md`

### Checks
- `.venv/bin/python - <<'PY' ... PY`：加载 `configs/starflow_vla/stage3_future_token_ablation.yaml`，取真实 LIBERO batch，分别构建 `num_target_vision_tokens=0/8/16/32/64` 的 `StarFlowVLA`，冻结 `qwen_vl_interface`，每组优化 action head / projectors 3 步。

### Findings
- 真实 batch shape：action `(8, 7)`，state `(1, 8)`。
- 五组 `num_target_vision_tokens=0/8/16/32/64` 均完成 forward/backward，`action_loss` 均为有限值。
- 五组 single batch overfit 前置验证均通过，且 `num_target_vision_tokens=0` 边界未失败。
- 本阶段未修改 `starVLA/model/` 源码。

### Not Run
未保存 5 组 ablation checkpoint，未运行完整训练循环、policy server、LIBERO rollout、success_rate 统计、评测、部署或 VGGT 接入。

### Next
P0 Stage B 的真实 batch、Stage1、MLP baseline、future token、checkpoint sidecar 和 eval preflight 已完成；后续进入可选 resume 100 step 或真实 LIBERO rollout 前需确认运行预算。

## Record P0-M10-STAGEB

### Task ID
P0-M10-STAGEB

### Goal
使用 Stage1 smoke checkpoint 执行最小 LIBERO rollout smoke，确认 policy server、LIBERO 环境和 eval 客户端可连通。

### Environment
Stage B 1×A100 40G target smoke validation；policy server 使用仓库 `.venv`；LIBERO eval 使用 `.libero`；本任务只运行 `libero_goal` 的 1 task × 1 trial smoke，不运行完整 LIBERO suite。

### Files Changed
- Modified: `ACCEPTANCE_CHECKLIST.md`
- Modified: `EXPERIMENT_MATRIX.md`
- Modified: `EVAL_SMOKE.md`
- Modified: `PATCH_MANIFEST.md`
- Modified: `IMPLEMENTATION_LOG.md`
- Modified: `examples/LIBERO/SESSION.md`
- Environment side effects: `.libero/bin/python -m pip install -e LIBERO`
- Eval side effects: generated `playground/eval_results/libero_goal/starflow_vla_stage1_smoke_steps_1/rollout_open_the_middle_drawer_of_the_cabinet_episode0_failure.mp4`

### Checks
- `RUN_DIR=$PWD/playground/Checkpoints/starflow_vla_stage1_qwenpi_v3_native CKPT_STEP=1 STARVLA_PYTHON=$PWD/.venv/bin/python PORT=6694 USE_BF16=1 bash examples/LIBERO/eval_files/run_policy_server.sh`
- `LIBERO_HOME=$PWD/LIBERO LIBERO_CONFIG_PATH=$PWD/LIBERO/libero PYTHONPATH=$PWD/LIBERO:$PWD MUJOCO_GL=egl PYOPENGL_PLATFORM=egl .libero/bin/python examples/LIBERO/eval_files/eval_libero.py --args.pretrained-path $PWD/playground/Checkpoints/starflow_vla_stage1_qwenpi_v3_native/checkpoints/steps_1 --args.host 127.0.0.1 --args.port 6694 --args.task-suite-name libero_goal --args.num-trials-per-task 1 --args.max-tasks 1 --args.video-out-path $PWD/playground/eval_results/libero_goal/starflow_vla_stage1_smoke_steps_1`

### Findings
- policy server 成功加载 `steps_1`，识别 `action_chunk_size=8`、`default_unnorm_key=franka`、7D action keys 和 8D state keys。
- LIBERO eval 客户端成功连接 policy server。
- 最小 rollout smoke 完成 1 episode，输出 `Total success rate: 0.0` 与 `Total episodes: 1`。
- 生成 failure rollout 视频；该结果来自 smoke checkpoint，不代表训练后性能。
- eval 退出阶段出现 EGL / `libGLU.so.0` 清理期警告，但 eval 进程退出码为 0。

### Not Run
未运行完整 LIBERO suite、failure taxonomy、checkpoint/config hash report、多 seed 评测、完整训练、部署或 VGGT 接入。

### Next
P0 主线 smoke 已覆盖到最小 rollout；剩余可选项为 resume 100 step、完整 LIBERO suite 与正式报告。

## Record P0-M10-REPORT

### Task ID
P0-M10-REPORT

### Goal
补齐 LIBERO smoke eval report，确保 `eval_report.json` 输出 `failure_category`、`checkpoint_hash`、`config_hash`、`data_version` 和 `starflow_mapping`，并避免退出阶段阻塞导致报告丢失。

### Environment
Stage B 1×A100 40G target smoke validation；使用仓库 `.venv` 与 `.libero`；本任务只扩展 eval 报告和最小 smoke 产物写盘，不运行完整 LIBERO suite、完整训练、部署或 VGGT 接入。

### Files Changed
- Added: `examples/LIBERO/eval_files/starflow_eval_report.py`
- Added: `tests/test_starflow_eval_report.py`
- Modified: `examples/LIBERO/eval_files/eval_libero.py`
- Modified: `docs_zh/starflow_vla/ACCEPTANCE_CHECKLIST.md`
- Modified: `docs_zh/starflow_vla/EXPERIMENT_MATRIX.md`
- Modified: `docs_zh/starflow_vla/EVAL_SMOKE.md`
- Modified: `docs_zh/starflow_vla/PATCH_MANIFEST.md`
- Modified: `docs_zh/starflow_vla/IMPLEMENTATION_LOG.md`
- Modified: `examples/LIBERO/SESSION.md`
- Eval side effects: generated `playground/eval_results/libero_goal/starflow_vla_stage1_smoke_steps_1/eval_report.json`

### Checks
- `python -m py_compile examples/LIBERO/eval_files/starflow_eval_report.py examples/LIBERO/eval_files/eval_libero.py tests/test_starflow_eval_report.py tests/test_starflow_eval_preflight.py`
- `.venv/bin/python -m unittest tests.test_starflow_eval_report -v`
- `.venv/bin/python -m unittest tests.test_starflow_eval_preflight -v`
- `.venv/bin/python - <<'PY' ... load_eval_metadata(steps_1) + build_eval_report(...) + write_eval_report(...) ... PY`

### Findings
- 新增 `starflow_eval_report.py`，可从 checkpoint/配置/数据统计旁路文件提取 `checkpoint_hash`、`config_hash`、`data_version` 和 `starflow_mapping`。
- 目录型 checkpoint hash 改为轻量 fingerprint：小元数据文件按内容 hash，大权重 shard 按相对路径和文件大小纳入 hash，避免 smoke startup 读取整套权重。
- `eval_libero.py` 现在会在每个 episode 结束后、视频编码前写一次 `eval_report.json`，降低 EGL/视频清理期阻塞导致报告丢失的风险。
- 已为现有 smoke 产物 `steps_1` 生成 `playground/eval_results/libero_goal/starflow_vla_stage1_smoke_steps_1/eval_report.json`。
- 报告内容已包含 `success_rate=0.0`、`failure_category.timeout_no_success=1`、`checkpoint_hash`、`config_hash`、`data_version`、`starflow_mapping`。

### Not Run
未运行完整 LIBERO suite、细粒度 failure taxonomy、多 seed 评测、resume 100 step、完整训练、部署或 VGGT 接入。

### Next
继续 P0-M9：做 `resume 100 step` 对比与 checkpoint 恢复偏差验证。

## Record P0-M9-RESUME

### Task ID
P0-M9-RESUME

### Goal
完成 `resume 100 step` 对比验证，并为 lightweight checkpoint 增加 RNG state 保存/恢复，确保 fixed batch smoke 下恢复后 100 step 内 loss 偏差 <1%。

### Environment
Stage B 1×A100 40G target smoke validation；使用仓库 `.venv`；本任务只运行 fixed-batch resume consistency smoke，不运行完整训练、完整 LIBERO suite、部署或 VGGT 接入。

### Files Changed
- Modified: `starVLA/training/train_starvla.py`
- Added: `tests/test_starflow_resume_100_steps.py`
- Modified: `docs_zh/starflow_vla/ACCEPTANCE_CHECKLIST.md`
- Modified: `docs_zh/starflow_vla/EXPERIMENT_MATRIX.md`
- Modified: `docs_zh/starflow_vla/IMPLEMENTATION_LOG.md`
- Modified: `docs_zh/starflow_vla/PATCH_MANIFEST.md`
- Modified: `examples/LIBERO/SESSION.md`
- Data/checkpoint side effects: created and removed transient bootstrap / split / resumed checkpoints under `playground/tmp_resume_smoke/`

### Checks
- `python -m py_compile starVLA/training/train_starvla.py tests/test_starflow_resume_100_steps.py`
- `WANDB_MODE=disabled .venv/bin/python -m unittest -v tests.test_starflow_resume_100_steps`

### Findings
- `train_starvla.py` 的 lightweight checkpoint 现在会额外保存和恢复 Python / NumPy / Torch / CUDA RNG state。
- 新增 `tests/test_starflow_resume_100_steps.py`，验证路径为：
- 1. 从 `steps_1` 只加载模型权重，基于真实 `libero_goal` batch 先 bootstrap 1 step，生成与当前 optimizer group 兼容的 lightweight checkpoint。
- 2. 从 bootstrap checkpoint 连续跑 100 step，记录 loss 轨迹。
- 3. 从同一 bootstrap checkpoint 跑 50 step，保存 `steps_51`，再恢复后继续跑 50 step。
- 4. 对比连续路径后 50 step 与恢复路径后 50 step，断言最大相对偏差 `< 1%`。
- 实测 `tests.test_starflow_resume_100_steps` 通过。
- 采样日志显示恢复路径和连续路径在 step `61/71/81/91/101` 的 loss 非常接近，例如 `0.47851533 vs 0.47853473`、`0.91403395 vs 0.91406316`、`0.34975210 vs 0.34975326`。

### Not Run
未运行完整训练循环、完整 LIBERO suite、细粒度 failure taxonomy、多 seed 评测、部署或 VGGT 接入。

### Next
P0 主线最小 smoke 与 resume consistency 已闭环；后续可进入完整 LIBERO suite 或更正式训练闭环。

## Record P0-M9-SCALER

### Task ID
P0-M9-SCALER

### Goal
补齐 lightweight checkpoint 的 `scaler` sidecar 与 checkpoint 内 config/mapping 自动落盘，使 checkpoint 实际包含 `model / optimizer / scaler / config / starflow_mapping`。

### Environment
Stage B 1×A100 40G target smoke validation；使用仓库 `.venv`；本任务只扩展 lightweight checkpoint 产物与短测，不运行完整训练、完整 LIBERO suite、部署或 VGGT 接入。

### Files Changed
- Modified: `starVLA/training/train_starvla.py`
- Modified: `starVLA/training/trainer_utils/trainer_tools.py`
- Modified: `tests/test_starflow_checkpoint_mapping.py`
- Modified: `docs_zh/starflow_vla/ACCEPTANCE_CHECKLIST.md`
- Modified: `docs_zh/starflow_vla/EXPERIMENT_MATRIX.md`
- Modified: `docs_zh/starflow_vla/EVAL_SMOKE.md`
- Modified: `docs_zh/starflow_vla/PATCH_MANIFEST.md`
- Modified: `docs_zh/starflow_vla/IMPLEMENTATION_LOG.md`
- Modified: `docs_zh/starflow_vla/CODEX_ISSUES.md`
- Modified: `examples/LIBERO/SESSION.md`
- Data/checkpoint side effects: wrote `playground/Checkpoints/starflow_vla_stage1_qwenpi_v3_native/checkpoints/steps_1/scaler.pt`

### Checks
- `python -m py_compile starVLA/training/train_starvla.py starVLA/training/trainer_utils/trainer_tools.py tests/test_starflow_checkpoint_mapping.py`
- `.venv/bin/python -m unittest -v tests.test_starflow_checkpoint_mapping`
- `.venv/bin/python -m unittest -v tests.test_starflow_eval_report tests.test_starflow_eval_preflight`
- `.venv/bin/python - <<'PY' ... save_lightweight_scaler_state(steps_1, None) ... PY`

### Findings
- `lightweight checkpoint` 现在在主进程侧自动写入 `scaler.pt`、`config.yaml`、`config.full.yaml`、`starflow_mapping.json`，并尽量复制 `dataset_statistics.json`。
- `scaler.pt` 采用兼容 payload：启用 AMP scaler 时保存 `state_dict`；未启用时保存 `{\"has_scaler\": false, \"state_dict\": null}` 占位，避免 checkpoint schema 缺项。
- 恢复链路对旧 checkpoint 保持兼容：若缺少 `scaler.pt`，记录 warning 后跳过；若 checkpoint 明确记录未启用 scaler，也会直接跳过恢复。
- 新增短测覆盖：
- 1. checkpoint metadata 自动落盘：`config.yaml`、`config.full.yaml`、`dataset_statistics.json`、`starflow_mapping.json`
- 2. `scaler.pt` placeholder 落盘
- 3. scaler state roundtrip
- 已为现有 smoke checkpoint `steps_1` 补写 `scaler.pt`，因此当前 smoke 产物已实际满足 `model / optimizer / scaler / config / starflow_mapping`。

### Not Run
未运行完整训练循环、完整 LIBERO suite、细粒度 failure taxonomy、多 seed 评测、部署或 VGGT 接入。

### Next
P0 checkpoint 产物闭环已补齐；后续优先进入完整 LIBERO suite 或更正式训练闭环。

## Record P0-GUARDRAIL

### Task ID
P0-GUARDRAIL

### Goal
把 “Perceiver / 显式 FlowCondition runtime / 14D action_mask 不阻断 P0” 从文档口径固化为可执行回归检查，补齐 P0 最后一项 guardrail 验收。

### Environment
Stage A 1×A100 40G lightweight validation；本任务只扩展文档治理测试与验收记录，不加载真实模型，不运行训练、完整评测或部署。

### Files Changed
- Modified: `tests/test_starflow_docs_governance.py`
- Modified: `docs_zh/starflow_vla/ACCEPTANCE_CHECKLIST.md`
- Modified: `docs_zh/starflow_vla/PATCH_MANIFEST.md`
- Modified: `docs_zh/starflow_vla/IMPLEMENTATION_LOG.md`
- Modified: `examples/LIBERO/SESSION.md`

### Checks
- `python -m py_compile tests/test_starflow_docs_governance.py`
- `.venv/bin/python -m unittest -v tests.test_starflow_docs_governance`

### Findings
- 新增文档治理回归检查，覆盖三类 guardrail：
- 1. 文档边界：`DESIGN_FREEZE_CHECK.md`、`P0_IMPLEMENTATION_PLAN.md`、`MODULE_MAPPING.md`
- 2. P0 配置边界：`stage1_starflow_qwenpi_v3_native.yaml` 不启用 `perceiver_enabled=true`、`flow_condition_runtime=true`、`max_action_dim=14`、`action_mask`
- 3. smoke checkpoint 映射边界：`steps_1/starflow_mapping.json` 保持 `perceiver_enabled=false`、`flow_condition_runtime=false`
- 测试通过后，`ACCEPTANCE_CHECKLIST.md` 中 `no Perceiver / FlowCondition runtime / 14D mask blocking P0` 已可勾选。
- 本任务没有改动模型逻辑、训练逻辑或评测逻辑，只把既有 P0 边界变成可执行约束。

### Not Run
未运行真实模型加载、完整训练、完整 LIBERO suite、性能评测、部署或 VGGT 接入。

### Next
P0 最小闭环代码与 guardrail 已基本收口；后续进入完整 LIBERO suite 或正式训练闭环时，需要按目标环境继续做运行级验证。

## Record P0-M5-TRAINLOOP

### Task ID
P0-M5-TRAINLOOP

### Goal
验证 `train_starvla.py` 主训练链在真实 Stage1 配置下可闭环运行，不依赖手工 bootstrap checkpoint；同时修复单卡训练入口的分布式 barrier 阻塞。

### Environment
Stage B 1×A100 40G target smoke validation；使用仓库 `.venv`；本任务运行真实 `10 step` 训练闭环，不运行长训、完整 LIBERO suite、部署或 VGGT 接入。

### Files Changed
- Modified: `starVLA/training/train_starvla.py`
- Modified: `docs_zh/starflow_vla/EXPERIMENT_MATRIX.md`
- Modified: `docs_zh/starflow_vla/PATCH_MANIFEST.md`
- Modified: `docs_zh/starflow_vla/IMPLEMENTATION_LOG.md`
- Modified: `examples/LIBERO/SESSION.md`
- Data/checkpoint side effects:
- `playground/Checkpoints/starflow_vla_stage1_trainloop_10step/checkpoints/`（首轮训练尝试，命中环境阻塞）
- `playground/Checkpoints/starflow_vla_stage1_trainloop_10step_ldpath/checkpoints/`（补 `LD_LIBRARY_PATH` 后的训练尝试，命中 launcher / MPI 阻塞）
- `playground/Checkpoints/starflow_vla_stage1_trainloop_10step_accel/checkpoints/`（`accelerate launch` 尝试，仍命中 MPI 阻塞）
- `playground/Checkpoints/starflow_vla_stage1_trainloop_10step_envdist/checkpoints/steps_10`
- `playground/Checkpoints/starflow_vla_stage1_trainloop_10step_envdist/final_model`

### Checks
- `python -m py_compile starVLA/training/train_starvla.py`
- `WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 .venv/bin/python -u starVLA/training/train_starvla.py --config_yaml configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml --run_id starflow_vla_stage1_trainloop_10step`
- `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.3/compat:$LD_LIBRARY_PATH timeout 180s .venv/bin/python -u - <<'PY' import deepspeed; print('deepspeed_import_ok') PY`
- `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.3/compat:$LD_LIBRARY_PATH timeout 180s .venv/bin/python -u - <<'PY' from triton.backends.nvidia import driver; print('triton_driver_import_ok') PY`
- `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.3/compat:$LD_LIBRARY_PATH WANDB_MODE=disabled CUDA_VISIBLE_DEVICES=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29621 RANK=0 LOCAL_RANK=0 WORLD_SIZE=1 .venv/bin/python -u starVLA/training/train_starvla.py --config_yaml configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml --run_id starflow_vla_stage1_trainloop_10step_envdist`
- `find playground/Checkpoints/starflow_vla_stage1_trainloop_10step_envdist/checkpoints/steps_10 -maxdepth 1 -type f`
- `find playground/Checkpoints/starflow_vla_stage1_trainloop_10step_envdist/final_model -maxdepth 1 -type f`

### Findings
- 首轮真实训练直接暴露代码级阻塞：`prepare_data()` 在未初始化分布式时无条件执行 `dist.barrier()`，单卡入口在 dataset 初始化后直接报错。
- 修复后继续暴露两层环境前提：
- 1. `DeepSpeed + Triton` 需要显式可见 `libcuda.so.1`，当前环境可通过 `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.3/compat:$LD_LIBRARY_PATH` 解决。
- 2. 当前脚本内置 `DeepSpeedPlugin`，单进程直跑时需显式提供 `MASTER_ADDR`、`MASTER_PORT`、`RANK`、`LOCAL_RANK`、`WORLD_SIZE`，避免 DeepSpeed 回退到 MPI 探测并因缺 `mpi4py` 失败。
- 在补齐上述环境变量后，`train_starvla.py` 已完成真实 `10/10 step` 训练闭环。
- 训练日志样本：
- `step 1 action_dit_loss = 1.76211452`
- `step 5 action_dit_loss = 3.18678570`
- `step 10 action_dit_loss = 1.39058614`
- 已成功保存：
- `playground/Checkpoints/starflow_vla_stage1_trainloop_10step_envdist/checkpoints/steps_10`
- `playground/Checkpoints/starflow_vla_stage1_trainloop_10step_envdist/final_model`
- 两处产物均已包含 `model / optimizer / scheduler / random_states / scaler / config / dataset_statistics / starflow_mapping / trainer_state`。

### Not Run
未运行长训、完整 LIBERO suite、resume from `steps_10` 对照、完整评测、部署或 VGGT 接入。

### Next
P0 现已不仅有 smoke checkpoint，也有 `train_starvla.py` 主链真实产物；后续可基于 `steps_10` 进入完整 LIBERO suite，或把当前环境变量要求固化为正式训练启动脚本。

## Record P0-M10-FULLTASK

### Task ID
P0-M10-FULLTASK

### Goal
用真实训练产物 `steps_10` 跑一轮 `libero_goal` 全 task sweep，验证 `train_starvla.py` 主链 checkpoint 可直接进入 policy server 和多任务 eval 链路。

### Environment
Stage B 1×A100 40G target smoke validation；policy server 使用仓库 `.venv`；eval client 使用 `.libero`；本任务运行 `libero_goal` 全 10 task、每 task 1 trial，不运行标准 50 trials/task、长训、部署或 VGGT 接入。

### Files Changed
- Modified: `docs_zh/starflow_vla/EXPERIMENT_MATRIX.md`
- Modified: `docs_zh/starflow_vla/EVAL_SMOKE.md`
- Modified: `docs_zh/starflow_vla/PATCH_MANIFEST.md`
- Modified: `docs_zh/starflow_vla/IMPLEMENTATION_LOG.md`
- Modified: `examples/LIBERO/SESSION.md`
- Eval side effects:
- `playground/eval_results/libero_goal/starflow_vla_stage1_trainloop_10step_envdist_steps_10_fullsuite/eval_report.json`
- `playground/eval_results/libero_goal/starflow_vla_stage1_trainloop_10step_envdist_steps_10_fullsuite/*.mp4`

### Checks
- `CKPT=$PWD/playground/Checkpoints/starflow_vla_stage1_trainloop_10step_envdist/checkpoints/steps_10 STARVLA_DIR=$PWD STARVLA_PYTHON=.venv/bin/python PORT=6695 GPU_ID=0 USE_BF16=1 bash examples/LIBERO/eval_files/run_policy_server.sh`
- `CKPT=$PWD/playground/Checkpoints/starflow_vla_stage1_trainloop_10step_envdist/checkpoints/steps_10 LIBERO_HOME=$PWD/LIBERO LIBERO_CONFIG_PATH=$PWD/LIBERO/libero LIBERO_PYTHON=.libero/bin/python STARVLA_DIR=$PWD HOST=127.0.0.1 PORT=6695 TASK_SUITE_NAME=libero_goal NUM_TRIALS_PER_TASK=1 VIDEO_OUT_PATH=$PWD/playground/eval_results/libero_goal/starflow_vla_stage1_trainloop_10step_envdist_steps_10_fullsuite bash examples/LIBERO/eval_files/eval_libero.sh`
- `python - <<'PY' ... read eval_report.json ... PY`
- `find playground/eval_results/libero_goal/starflow_vla_stage1_trainloop_10step_envdist_steps_10_fullsuite -maxdepth 1 -name '*.mp4' | wc -l`

### Findings
- `steps_10` checkpoint 可被 policy server 直接加载，server metadata 正常识别：
- `action_chunk_size=8`
- `default_unnorm_key=franka`
- 7D action keys 和 8D state keys
- `libero_goal` 全 10 task × 1 trial 已跑完，共 `10` 个 episode。
- 结果：
- `success_rate = 0.0`
- `total_episodes = 10`
- `failure_category = {"timeout_no_success": 10}`
- 目录中已生成 `10` 个 failure rollout 视频以及 `eval_report.json`。
- eval 退出阶段仍有 EGL / `libGLU.so.0` 清理期警告，但评测进程退出码为 `0`，报告已在 episode 结束后写盘。
- 该结果代表“全 task sweep”，不是标准 `50 trials/task` 的完整 LIBERO 评测结果。

### Not Run
未运行标准 50 trials/task、细粒度 failure taxonomy、多 seed 评测、长训、部署或 VGGT 接入。

### Next
若继续推进评测，应在 `steps_10` 或更长训练产物上运行标准 `50 trials/task`，并补 task-level/failure-level 汇总；若继续推进训练，应把当前环境变量要求固化到正式启动脚本。

## Record P0-M10-REGRESSION

### Task ID
P0-M10-REGRESSION

### Goal
把 `train checkpoint -> policy server -> 1 episode eval -> eval_report` 固化成可重复执行的一键快速回归入口。

### Environment
Stage B 1×A100 40G target smoke validation；policy server 使用仓库 `.venv`；eval client 使用 `.libero`；本任务只固化 quick regression 入口，不运行长训、标准 50 trials/task 完整评测或部署。

### Files Changed
- Added: `examples/LIBERO/eval_files/run_starflow_eval_regression.sh`
- Modified: `examples/LIBERO/eval_files/eval_libero.sh`
- Modified: `tests/test_starflow_eval_preflight.py`
- Modified: `docs_zh/starflow_vla/EVAL_SMOKE.md`
- Modified: `docs_zh/starflow_vla/IMPLEMENTATION_LOG.md`
- Modified: `docs_zh/starflow_vla/PATCH_MANIFEST.md`
- Modified: `examples/LIBERO/SESSION.md`

### Checks
- `bash -n examples/LIBERO/eval_files/run_policy_server.sh && bash -n examples/LIBERO/eval_files/eval_libero.sh && bash -n examples/LIBERO/eval_files/run_starflow_eval_regression.sh`
- `.venv/bin/python -m unittest -v tests.test_starflow_eval_preflight tests.test_starflow_eval_report`
- `CKPT=$PWD/playground/Checkpoints/starflow_vla_stage1_trainloop_10step_envdist/checkpoints/steps_10 STARVLA_DIR=$PWD STARVLA_PYTHON=$PWD/.venv/bin/python LIBERO_HOME=$PWD/LIBERO LIBERO_CONFIG_PATH=$PWD/LIBERO/libero LIBERO_PYTHON=$PWD/.libero/bin/python TASK_SUITE_NAME=libero_goal NUM_TRIALS_PER_TASK=1 MAX_TASKS=1 PORT=6696 VIDEO_OUT_PATH=$PWD/playground/eval_results/libero_goal/starflow_vla_stage1_trainloop_10step_envdist_steps_10_regression bash examples/LIBERO/eval_files/run_starflow_eval_regression.sh`
- `CKPT=$PWD/playground/Checkpoints/starflow_vla_stage1_trainloop_10step_envdist/checkpoints/steps_10 STARVLA_DIR=$PWD STARVLA_PYTHON=$PWD/.venv/bin/python LIBERO_HOME=$PWD/LIBERO LIBERO_CONFIG_PATH=$PWD/LIBERO/libero LIBERO_PYTHON=$PWD/.libero/bin/python TASK_SUITE_NAME=libero_goal NUM_TRIALS_PER_TASK=1 MAX_TASKS=1 PORT=6697 VIDEO_OUT_PATH=$PWD/playground/eval_results/libero_goal/starflow_vla_stage1_trainloop_10step_envdist_steps_10_regression bash examples/LIBERO/eval_files/run_starflow_eval_regression.sh`

### Findings
- `run_starflow_eval_regression.sh` 已固化以下链路：
- 启动 `run_policy_server.sh`
- 用 socket 轮询等待 server 就绪
- 调用 `eval_libero.sh`
- 校验 `eval_report.json` 存在并输出摘要
- `eval_libero.sh` 已新增 `MAX_TASKS` 透传，可直接执行 `1 task × 1 trial` quick regression。
- 轻量验证通过：
- shell 语法检查通过
- `tests.test_starflow_eval_preflight` / `tests.test_starflow_eval_report` 通过
- 首轮真实运行发现默认 `SERVER_READY_TIMEOUT=300` 不足；`steps_10` 的 policy server 在该窗口内仍处于 framework / checkpoint CPU 侧初始化阶段。
- 已将默认等待时间上调到 `900s`，并为 server 进程增加 `PYTHONUNBUFFERED=1`，便于观察冷启动日志。
- 本轮两次真实运行尝试中，policy server 均未在当前等待窗口内监听端口；日志已确认 checkpoint shard 加载完成，尚未出现 websocket server `listen` 阶段输出。
- 因此，本轮没有由新入口额外产出新的 `eval_report.json`；但训练产物到推理验证的接口契约仍由既有 `steps_10 -> full task sweep` 实跑结果覆盖。

### Not Run
未通过新入口完成一次成功的 end-to-end quick regression；未运行标准 50 trials/task、多 seed、长训或部署。

### Next
若继续推进，应针对 `PolicyServerWrapper` 冷启动阶段补精确耗时日志，或把 quick regression 入口的 server ready 策略从“固定超时”改为“更长窗口 + 关键日志断点”。

## Record P0-M10-COLDSTART

### Task ID
P0-M10-COLDSTART

### Goal
定位 `steps_10` policy server 冷启动过慢的具体阶段，判断问题属于 import、framework 构建、checkpoint 加载还是 websocket server 监听。

### Environment
Stage B 1×A100 40G target smoke validation；使用仓库 `.venv`；本任务只做 server 侧冷启动探针，不运行 LIBERO eval、不运行长训。

### Files Changed
- Modified: `deployment/model_server/server_policy.py`
- Modified: `deployment/model_server/policy_wrapper.py`
- Modified: `starVLA/model/framework/base_framework.py`
- Modified: `examples/LIBERO/SESSION.md`
- Modified: `docs_zh/starflow_vla/IMPLEMENTATION_LOG.md`

### Checks
- `python -m py_compile deployment/model_server/server_policy.py deployment/model_server/policy_wrapper.py starVLA/model/framework/base_framework.py`
- `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.3/compat:$LD_LIBRARY_PATH PYTHONUNBUFFERED=1 timeout 180s .venv/bin/python -u deployment/model_server/server_policy.py --ckpt_path $PWD/playground/Checkpoints/starflow_vla_stage1_trainloop_10step_envdist/checkpoints/steps_10 --port 6698 --use_bf16 > playground/eval_results/libero_goal/policy_server_coldstart_probe.log 2>&1`
- `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.3/compat:$LD_LIBRARY_PATH PYTHONUNBUFFERED=1 timeout 360s .venv/bin/python -u deployment/model_server/server_policy.py --ckpt_path $PWD/playground/Checkpoints/starflow_vla_stage1_trainloop_10step_envdist/checkpoints/steps_10 --port 6698 --use_bf16 > playground/eval_results/libero_goal/policy_server_coldstart_probe.log 2>&1`
- `PYTHONPROFILEIMPORTTIME=1 timeout 180s .venv/bin/python -X importtime -c 'import deployment.model_server.server_policy' > playground/eval_results/libero_goal/policy_server_importtime_probe.log 2>&1`

### Findings
- `server_policy.py` 顶层 import 原先会在 `main()` 之前吞掉大量时间且无日志；已改为 `main()` 内懒导入，并新增阶段耗时日志。
- 冷启动探针结果：
- `server_policy.main: module imports finished in 272.50s`
- `baseframework.from_pretrained -> read_mode_config done in 0.13s`
- `build_framework done in 42.69s (StarFlowVLA)`
- 360s 探针窗口结束前，尚未走完 `load_model_weights` / `.to(cuda)` / websocket listen
- 结论：
- regression 脚本的 `server ready timeout` 不是主因；真正的首要瓶颈是 server 端 import 链和 framework build
- 仅 `300s` 窗口就会在完成 import + build 前超时
- 现有 `900s` 只是等待层缓解，不是根治

### Not Run
未完整跑到 websocket listen；未完成基于新诊断日志的 second-pass 优化；未验证更长超时下的完整冷启动总时长。

### Next
若继续推进，应优先评估：
- 是否需要进一步懒导入 `policy_wrapper` 内部重依赖
- 是否能对 `build_framework` 中非推理必需路径做推迟初始化
- 回归脚本是否改为“按关键日志阶段等待”而不是单纯固定秒数

## Record P0-M5-TRAINREADY

### Task ID
P0-M5-TRAINREADY

### Goal
把已验证可跑的 Stage1 训练前提固化成一个一键启动脚本，减少手工拼接环境变量和参数的时间。

### Environment
Stage B 1×A100 40G target smoke validation；使用仓库 `.venv`；本任务只固化训练启动入口，不实际长训。

### Files Changed
- Added: `examples/LIBERO/train_files/run_starflow_train_ready.sh`
- Added: `tests/test_starflow_train_ready.py`
- Modified: `examples/LIBERO/SESSION.md`
- Modified: `docs_zh/starflow_vla/IMPLEMENTATION_LOG.md`

### Checks
- `bash -n examples/LIBERO/train_files/run_starflow_train_ready.sh`
- `python -m py_compile tests/test_starflow_train_ready.py`
- `.venv/bin/python -m unittest -v tests.test_starflow_train_ready`

### Findings
- 已将当前训练已验证前提固化为脚本默认值：
- `WANDB_MODE=disabled`
- `MASTER_ADDR=127.0.0.1`
- `MASTER_PORT=29621`
- `RANK=0`
- `LOCAL_RANK=0`
- `WORLD_SIZE=1`
- `LD_LIBRARY_PATH` 自动补入常见 `libcuda.so.1` 路径
- 使用 Stage1 `StarFlowVLA` 配置、`libero_goal` 数据源、`Qwen3-VL-4B-Instruct` 作为默认 base VLM
- 默认 `MAX_TRAIN_STEPS=10`，可通过环境变量提升为更长训练
- 默认 `EVAL_INTERVAL=1000`，避免 quick-start 场景触发除零

### Not Run
未实际执行长训；未验证更长步数下的吞吐和 checkpoint 稳定性。

### Next
若继续推进，可把同样的“训练就绪”默认值再抽到更正式的长训脚本，或在脚本里增加显式数据集/模型路径存在性检查的更强预检。

## Record P0-M4-STAGEB

### Task ID
P0-M4-STAGEB

### Goal
复验 QwenPI_v3 baseline 在真实 LIBERO batch 上仍可执行 forward/backward，确保 StarFlowVLA facade 没有破坏基线入口。

### Environment
Stage B 1×A100 40G target smoke validation；使用仓库 `.venv`；本任务只运行单 batch baseline smoke，不保存 baseline checkpoint，不运行完整训练、评测或部署。

### Files Changed
- Modified: `ACCEPTANCE_CHECKLIST.md`
- Modified: `EXPERIMENT_MATRIX.md`
- Modified: `PATCH_MANIFEST.md`
- Modified: `IMPLEMENTATION_LOG.md`
- Modified: `examples/LIBERO/SESSION.md`

### Checks
- `.venv/bin/python - <<'PY' ... PY`：加载 `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml`，将 `framework.name` 临时设为 `QwenPI_v3`，取真实 LIBERO batch，冻结 `qwen_vl_interface`，执行 forward/backward。

### Findings
- 真实 batch shape：action `(8, 7)`，state `(1, 8)`。
- QwenPI_v3 baseline forward/backward smoke 通过，`action_loss` 为有限值，反传后可训练参数获得梯度。
- 本阶段未修改 `starVLA/model/` 源码。

### Not Run
未运行 QwenPI_v3 baseline overfit、checkpoint 保存/加载、完整训练、LIBERO rollout、success_rate 统计、评测、部署或 VGGT 接入。

### Next
P0 主线 smoke 已覆盖到最小 rollout；剩余可选项为 resume 100 step、完整 LIBERO suite 与正式报告。
