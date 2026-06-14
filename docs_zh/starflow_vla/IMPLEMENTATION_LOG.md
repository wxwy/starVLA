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
