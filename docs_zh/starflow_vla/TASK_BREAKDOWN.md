# StarFlow-VLA 任务拆解文档

版本：V0.1  
日期：2026-06-14  
范围：仅任务拆解与工程计划，不实现 StarFlowVLA 代码，不运行长训练，不修改模型主逻辑。

## 0. 只读仓库检查结论

审计对象：`E:\projects\starVLA`  
当前可写文档输出目录：`C:\Users\fast_\Desktop\xx\docs\starflow_vla`

| 检查项 | 结论 |
| --- | --- |
| QwenPI_v3 framework 文件路径 | `starVLA/model/framework/VLM4A/QwenPI_v3.py` |
| QwenPI_v3 registry 注册 | `@FRAMEWORK_REGISTRY.register("QwenPI_v3")`，位于 `QwenPI_v3.py:138` |
| framework build 入口 | `starVLA/model/framework/base_framework.py`，`build_framework(cfg)` 读取 `cfg.framework.name` |
| qwen_vl_interface 构建 | `QwenPI_v3.py` 中 `self.qwen_vl_interface = get_vlm_model(config=self.config)` |
| project_layers 构建 | `QwenPI_v3.py` 中 `self.project_layers = nn.ModuleList(...)` |
| action_model 构建 | `QwenPI_v3.py` 中 `self.action_model = get_action_model(config=self.config)` |
| LayerwiseFM action head | `starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py` |
| GR00T action head | `starVLA/model/modules/action_model/GR00T_ActionHeader.py` |
| future_tokens / num_target_vision_tokens | `LayerwiseFM_ActionHeader.py` 与 `GR00T_ActionHeader.py`，默认 `num_target_vision_tokens=32` |
| state-to-instruction | `QwenPI_v3.py`，`add_discretized_state_to_instruction()`；QwenOFT 也复用 share_tools 版本 |
| action_head.state_encoder | `LayerwiseFM_ActionHeader.py` / `GR00T_ActionHeader.py` 中 `self.state_encoder` |
| VLA_AdapterHeader / MLP baseline | `starVLA/model/modules/action_model/VLA_AdapterHeader.py`；`starVLA/model/modules/action_model/MLP_ActionHeader.py`；`QwenOFT.py` |
| config schema | 示例 `examples/LIBERO/train_files/starvla_cotrain_libero.yaml`，`version_id: "0.21"`，`framework.name`，`framework.action_model` |
| 训练入口 | `starVLA/training/train_starvla.py`、`train_starvla_cotrain.py`、`train_starvlm.py` |
| eval 入口 | `examples/LIBERO/eval_files/eval_libero.py`；`examples/Robocasa_tabletop/eval_files/`；`examples/Robotwin/eval_files/` |
| checkpoint 保存 | `train_starvla_cotrain.py` 中 `_save_checkpoint()` 保存 `steps_*_pytorch_model.pt` 或 safetensors，并保存 config |
| `docs/starflow_vla/` | Not found in current repository inspection |
| `configs/starflow_vla/` | Not found in current repository inspection |
| git branch / commit | `.git/HEAD` 显示 `starVLA_dev`；当前本地 ref 为 `42170b2a4df3877ccf6581948e2198d37c363c7f` |
| package version | `pyproject.toml` 中 `version = "1.0.1"` |

说明：直接调用 `git -C E:\projects\starVLA ...` 会触发 safe.directory 所有权保护。本次未修改 git 全局配置，commit 通过只读 `.git/HEAD` 与 refs 文件确认。

## 1. P0 任务：最小闭环

### P0-M0：V4.6.2 文档冻结检查

| 字段 | 内容 |
| --- | --- |
| Task ID | P0-M0 |
| 任务名称 | V4.6.2 文档冻结检查 |
| 优先级 | P0 |
| 目标 | 确认当前详细设计文档已统一为 StarVLA-native framework 路线，并且 P0/P1/P2 边界一致 |
| 动机 | 防止后续实现重新滑回外部项目、复制 QwenPI_v3 或把 P2 advanced 功能塞进 P0 |
| 涉及文件 | `基于VLA统一训练与泛化评测框架的机器人基础模型研究_最终详细设计文档.md` |
| 新增文件 | `docs/starflow_vla/DESIGN_FREEZE_CHECK.md`，或在本文档维护冻结检查章节 |
| 修改文件 | 无，除非发现文档冲突 |
| 输入 | V4.6.2 Implementation Trace Patch 文档 |
| 输出 | 冻结检查结论 |
| 依赖任务 | 无 |
| 验收标准 | 不再把 StarFlowVLA 描述成外部项目；不再把 PerceiverAdapter / FlowCondition runtime / 14D mask 作为 P0 必选；H2 统一为 future_tokens + cross-DiT vs MLP/OFT/VLA_AdapterHeader baseline |
| 建议测试命令 | `rg -n "models/qwen3_starvla_flow|H2 Action Token vs MLP Adapter|Perceiver.*P0|FlowCondition.*P0|14D.*P0" docs/ *.md` |
| 风险 | 文档源和导出稿不一致 |
| 回滚方式 | 回退到 V4.6.2 生成脚本和输出稿 |
| 是否修改 StarVLA 原文件 | 否 |
| 是否阻断 P0 闭环 | 是，文档未冻结前不进入实现 |

### P0-M1：StarVLA 基线锁定与分支初始化

| 字段 | 内容 |
| --- | --- |
| Task ID | P0-M1 |
| 任务名称 | StarVLA 基线锁定与分支初始化 |
| 优先级 | P0 |
| 目标 | 固定 StarVLA upstream commit、config schema、package version，作为 P0 实验基线 |
| 动机 | 防止 StarVLA 上游滚动更新导致实验结果不可追踪 |
| 涉及文件 | `.git/HEAD`、`pyproject.toml`、`examples/LIBERO/train_files/starvla_cotrain_libero.yaml` |
| 新增文件 | `docs/starflow_vla/UPSTREAM_COMPATIBILITY.md`、`docs/starflow_vla/BASELINE_VERSION.md` |
| 修改文件 | 无 |
| 输入 | 当前 StarVLA branch、commit、version、config schema |
| 输出 | 基线版本记录和上游兼容策略 |
| 依赖任务 | P0-M0 |
| 验收标准 | 记录 upstream commit `42170b2a4df3877ccf6581948e2198d37c363c7f`；记录 branch `starVLA_dev`；记录 package version `1.0.1`；记录 config schema `0.21`；记录审计日期；说明后续上游更新必须走 compatibility audit |
| 建议测试命令 | `Get-Content .git/HEAD; Get-Content pyproject.toml; rg -n "version_id" examples/LIBERO/train_files/starvla_cotrain_libero.yaml` |
| 风险 | git safe.directory 导致直接 git 命令失败 |
| 回滚方式 | 保留旧 baseline 记录，不切换正式实验基线 |
| 是否修改 StarVLA 原文件 | 否 |
| 是否阻断 P0 闭环 | 是 |

### P0-M2：新增 StarFlowVLA framework 入口任务

| 字段 | 内容 |
| --- | --- |
| Task ID | P0-M2 |
| 任务名称 | 新增 StarFlowVLA framework 入口 |
| 优先级 | P0 |
| 目标 | 后续实现新增 `StarFlowVLA` framework，对外通过 `framework.name=StarFlowVLA` 构建 |
| 动机 | 让 StarFlow-VLA 成为 StarVLA-native framework 路线，而不是外部训练脚本 |
| 涉及文件 | `starVLA/model/framework/base_framework.py`、`starVLA/model/framework/VLM4A/QwenPI_v3.py` |
| 新增文件 | `starVLA/model/framework/VLM4A/StarFlowVLA.py` |
| 修改文件 | 可能需要 `starVLA/model/framework/VLM4A/__init__.py`，若当前自动 import 已覆盖则不改 |
| 输入 | `cfg.framework.name=StarFlowVLA` |
| 输出 | 可注册、可 build 的 framework |
| 依赖任务 | P0-M1 |
| 验收标准 | Stage A：`FRAMEWORK_REGISTRY` 可找到 `StarFlowVLA`；`framework.name=StarFlowVLA` 可被 config 调用；`build_framework(cfg)` 可构建或至少通过不加载完整大模型的 dry-run；import / config parse 通过；StarFlowVLA 继承或委托 QwenPI_v3；不复制 QwenPI_v3 主体 forward / predict_action；不重复构建 Qwen3-VL、project_layers、LayerwiseFM；形成 Implementation Record。Stage B 复验归属：single batch forward/backward pass 由 P0-M4 / P0-M5 验收；loss finite 由 P0-M5 验收；predict_action shape 由 P0-M5 / P0-M10 验收；single batch overfit 由 P0-M5 验收 |
| 建议测试命令 | `python -c "from starVLA.model.framework.base_framework import build_framework; print('TODO: build StarFlowVLA config')"` |
| 风险 | 复制 QwenPI_v3 导致后续上游兼容困难 |
| 回滚方式 | 删除新增 `StarFlowVLA.py` 和配置，保留 QwenPI_v3 baseline |
| 是否修改 StarVLA 原文件 | 尽量否；新增文件为主 |
| 是否阻断 P0 闭环 | 是 |

### P0-M3：新增 starflow_mapping manifest

| 字段 | 内容 |
| --- | --- |
| Task ID | P0-M3 |
| 任务名称 | 新增 starflow_mapping manifest |
| 优先级 | P0 |
| 目标 | 记录文档抽象模块到 StarVLA-native 实现路径的映射 |
| 动机 | 让 checkpoint、实验报告和设计文档保持一致 |
| 涉及文件 | `train_starvla_cotrain.py` 的 checkpoint 保存路径；`StarFlowVLA.py` 的映射描述接口 |
| 新增文件 | `starVLA/model/modules/starflow_vla/mapping.py`、`docs/starflow_vla/MODULE_MAPPING.md` |
| 修改文件 | checkpoint 保存逻辑可在后续实现中最小扩展 |
| 输入 | config、runtime component、StarVLA baseline version |
| 输出 | `starflow_mapping.json` |
| 依赖任务 | P0-M1、P0-M2 |
| 验收标准 | 训练 checkpoint 中可保存 `starflow_mapping`；字段可序列化；`MODULE_MAPPING.md` 能解释每个抽象模块对应的 StarVLA-native 实现 |
| 建议测试命令 | `python -m json.tool outputs/.../starflow_mapping.json` |
| 风险 | manifest 字段与真实 runtime 不一致 |
| 回滚方式 | 不改变模型权重，仅回退 manifest 写入逻辑 |
| 是否修改 StarVLA 原文件 | 可能，若 checkpoint 保存入口不支持附加 manifest |
| 是否阻断 P0 闭环 | 是 |

`starflow_mapping.json` P0 字段：

```yaml
framework_name: StarFlowVLA
implementation_mode: starvla_native
base_framework: QwenPI_v3
action_head: LayerwiseFM
state_mode: discretized_instruction
adapter_mode: future_token_cross_dit
flow_condition_runtime: false
perceiver_enabled: false
num_target_vision_tokens: 32
solver: euler
num_inference_timesteps: 4
patch_manifest_hash: xxx
starvla_commit: xxx
config_schema: xxx
```

### P0-M4：QwenPI_v3 reuse smoke

| 字段 | 内容 |
| --- | --- |
| Task ID | P0-M4 |
| 任务名称 | QwenPI_v3 reuse smoke |
| 优先级 | P0 |
| 目标 | 确认 StarFlowVLA 后续实现真实复用 QwenPI_v3 |
| 动机 | 防止新增 framework 变成复制版 QwenPI_v3 |
| 涉及文件 | `QwenPI_v3.py`，`base_framework.py` |
| 新增文件 | smoke test，可命名为 `tests/test_starflow_vla_reuse.py` |
| 修改文件 | 无，或仅增加测试 |
| 输入 | StarFlowVLA config |
| 输出 | reuse smoke report |
| 依赖任务 | P0-M2 |
| 验收标准 | StarFlowVLA 构建时不重复创建第二套 Qwen3-VL；不复制 QwenPI_v3.py 大段代码；`qwen_vl_interface`、`project_layers`、`action_model`、`forward/predict_action` 主路径、state-to-instruction 默认路径可复用；QwenPI_v3 原始 baseline 仍可运行 |
| 建议测试命令 | `pytest tests/test_starflow_vla_reuse.py -q` |
| 风险 | hook 过深导致 QwenPI_v3 baseline 受污染 |
| 回滚方式 | 关闭 StarFlowVLA，直接运行原 QwenPI_v3 baseline |
| 是否修改 StarVLA 原文件 | 否 |
| 是否阻断 P0 闭环 | 是 |

### P0-M5：LayerwiseFM 单臂 7DoF 训练闭环

| 字段 | 内容 |
| --- | --- |
| Task ID | P0-M5 |
| 任务名称 | LayerwiseFM 单臂 7DoF 训练闭环 |
| 优先级 | P0 |
| 目标 | 跑通最小 Flow Matching 训练路径 |
| 动机 | 验证 StarFlowVLA 的最小可训练闭环 |
| 涉及文件 | `LayerwiseFM_ActionHeader.py`、`QwenPI_v3.py`、训练入口 |
| 新增文件 | `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml` |
| 修改文件 | 配置新增为主；必要时最小 checkpoint manifest 扩展 |
| 输入 | LIBERO small split batch，action_dim=7 |
| 输出 | finite loss、checkpoint、predict_action 输出 |
| 依赖任务 | P0-M2、P0-M4、P0-M8 |
| 验收标准 | loss finite；single batch overfit；checkpoint save/load；predict_action shape 正确；LIBERO small split eval smoke 可运行 |
| 建议测试命令 | `accelerate launch starVLA/training/train_starvla_cotrain.py --config_yaml configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml` |
| 风险 | 长训练资源不足；配置与现有 schema 不兼容 |
| 回滚方式 | 回退到 `examples/LIBERO/train_files/starvla_cotrain_libero.yaml` |
| 是否修改 StarVLA 原文件 | 否，优先配置化 |
| 是否阻断 P0 闭环 | 是 |

注意：P0 默认 `action_dim=7`；不要强制启用 `max_action_dim=14 + action_mask`，14D mask 放 P1。

### P0-M6：MLP/OFT/VLA_AdapterHeader baseline

| 字段 | 内容 |
| --- | --- |
| Task ID | P0-M6 |
| 任务名称 | MLP/OFT/VLA_AdapterHeader baseline |
| 优先级 | P0 |
| 目标 | 为 H2 提供 MLP baseline |
| 动机 | H2 需要与 future_tokens + cross-DiT 做公平对照 |
| 涉及文件 | `MLP_ActionHeader.py`、`VLA_AdapterHeader.py`、`QwenOFT.py` |
| 新增文件 | `configs/starflow_vla/stage2_mlp_baseline.yaml` |
| 修改文件 | 无，优先复用已有 baseline |
| 输入 | 与 P0-M5 同一数据、同一训练预算 |
| 输出 | baseline logs / checkpoint / eval report |
| 依赖任务 | P0-M5 |
| 验收标准 | MLP/OFT/VLA_AdapterHeader baseline 可 dry-run；single batch overfit；输出可与 future_tokens + cross-DiT 路线公平对照 |
| 建议测试命令 | `accelerate launch starVLA/training/train_starvla_cotrain.py --config_yaml configs/starflow_vla/stage2_mlp_baseline.yaml` |
| 风险 | baseline 与 StarFlowVLA 配置字段不一致 |
| 回滚方式 | 单独保留 baseline config，不影响 P0-M5 |
| 是否修改 StarVLA 原文件 | 否 |
| 是否阻断 P0 闭环 | 是 |

注意：不要强制新增独立 `MLPAdapter` runtime 类；`MLPAdapter` 是文档抽象，P0 映射为 StarVLA 已有 baseline。

### P0-M7：future_tokens + cross-DiT 消融

| 字段 | 内容 |
| --- | --- |
| Task ID | P0-M7 |
| 任务名称 | future_tokens + cross-DiT 消融 |
| 优先级 | P0 |
| 目标 | 支撑 H2：StarVLA-native future_tokens + cross-DiT vs MLP/OFT/VLA_AdapterHeader baseline |
| 动机 | 验证 planning slots 数量对跨 Benchmark 泛化的影响 |
| 涉及文件 | `LayerwiseFM_ActionHeader.py`、`GR00T_ActionHeader.py` |
| 新增文件 | `configs/starflow_vla/stage3_future_token_ablation.yaml` |
| 修改文件 | 无，优先配置化 |
| 输入 | `num_target_vision_tokens = 0 / 8 / 16 / 32 / 64` |
| 输出 | ablation configs、logs、manifest |
| 依赖任务 | P0-M5、P0-M6 |
| 验收标准 | 所有配置可解析；forward pass 通过；single batch overfit 通过；日志中记录 `num_target_vision_tokens`；checkpoint manifest 中记录 `adapter_mode=future_token_cross_dit` |
| 建议测试命令 | `for n in 0 8 16 32 64; do echo $n; done # TODO: replace with config sweep runner` |
| 风险 | `num_target_vision_tokens=0` 可能触发空 embedding 边界问题 |
| 回滚方式 | 回退默认 `num_target_vision_tokens=32` |
| 是否修改 StarVLA 原文件 | 否，除非 0-token 边界需小 patch |
| 是否阻断 P0 闭环 | 是 |

### P0-M8：LIBERO 最小数据闭环

| 字段 | 内容 |
| --- | --- |
| Task ID | P0-M8 |
| 任务名称 | LIBERO 最小数据闭环 |
| 优先级 | P0 |
| 目标 | 先跑通 LIBERO 数据转换、DataLoader、Batch schema、训练输入 |
| 动机 | 避免三数据集完整接入阻断 framework smoke |
| 涉及文件 | `examples/LIBERO/train_files/data_registry/data_config.py`、`examples/LIBERO/train_files/starvla_cotrain_libero.yaml` |
| 新增文件 | 可选 `docs/starflow_vla/LIBERO_SCHEMA_SMOKE.md` |
| 修改文件 | 可能新增 StarFlowVLA LIBERO 配置 |
| 输入 | LIBERO small split |
| 输出 | 可训练 batch |
| 依赖任务 | P0-M1 |
| 验收标准 | 数据转换成功；batch 含 image / instruction / state / action；`action_dim=7`；无 NaN/Inf；schema test 通过 |
| 建议测试命令 | `python -c "TODO: load one LIBERO batch through StarVLA dataloader"` |
| 风险 | 本地数据路径未配置 |
| 回滚方式 | 使用 StarVLA 原始 LIBERO 示例配置 |
| 是否修改 StarVLA 原文件 | 否 |
| 是否阻断 P0 闭环 | 是 |

### P0-M9：Checkpoint 与恢复

| 字段 | 内容 |
| --- | --- |
| Task ID | P0-M9 |
| 任务名称 | Checkpoint 与恢复 |
| 优先级 | P0 |
| 目标 | 保证训练可恢复、实验可追踪 |
| 动机 | 长训练必须支持断点恢复和 manifest 审计 |
| 涉及文件 | `train_starvla_cotrain.py`、`train_starvla.py` |
| 新增文件 | `docs/starflow_vla/CHECKPOINT_MANIFEST.md` 可选 |
| 修改文件 | checkpoint 保存逻辑可能需要最小扩展 |
| 输入 | P0-M5 / P0-M6 / P0-M7 训练 run |
| 输出 | checkpoint + config + starflow_mapping |
| 依赖任务 | P0-M3、P0-M5 |
| 验收标准 | checkpoint 包含 model / optimizer / scaler / config / starflow_mapping；resume 后 100 step 内 loss 偏差 <1%；`patch_manifest_hash` 可记录 |
| 建议测试命令 | `pytest tests/test_starflow_checkpoint_resume.py -q` |
| 风险 | 现有 checkpoint 格式未内置 manifest |
| 回滚方式 | 将 manifest 作为 checkpoint 旁路 JSON 文件保存 |
| 是否修改 StarVLA 原文件 | 可能，优先旁路 JSON 降低侵入 |
| 是否阻断 P0 闭环 | 是 |

### P0-M10：最小评测闭环

| 字段 | 内容 |
| --- | --- |
| Task ID | P0-M10 |
| 任务名称 | 最小评测闭环 |
| 优先级 | P0 |
| 目标 | 从 checkpoint 跑通 eval smoke |
| 动机 | 验证训练产物可进入评测流程 |
| 涉及文件 | `examples/LIBERO/eval_files/eval_libero.py`、`run_policy_server.sh`、`model2libero_interface.py` |
| 新增文件 | `outputs/eval/*` 报告；可选 `docs/starflow_vla/EVAL_SMOKE.md` |
| 修改文件 | 尽量不改；必要时新增 StarFlowVLA eval config |
| 输入 | P0 checkpoint |
| 输出 | success_rate、failure category、评测报告 |
| 依赖任务 | P0-M5、P0-M9 |
| 验收标准 | LIBERO eval smoke pass；输出 success_rate；输出 failure category；报告包含 checkpoint hash、config hash、data version、starflow_mapping |
| 建议测试命令 | `bash examples/LIBERO/eval_files/run_policy_server.sh && bash examples/LIBERO/eval_files/eval_libero.sh` |
| 风险 | eval 环境与训练环境不一致 |
| 回滚方式 | 回退到原 StarVLA LIBERO eval 流程 |
| 是否修改 StarVLA 原文件 | 否 |
| 是否阻断 P0 闭环 | 是 |

### P0-M11：文档与 patch 管理

| 字段 | 内容 |
| --- | --- |
| Task ID | P0-M11 |
| 任务名称 | 文档与 patch 管理 |
| 优先级 | P0 |
| 目标 | 让所有改动可审计 |
| 动机 | 支撑长期训练、论文复现和上游兼容 |
| 涉及文件 | `docs/starflow_vla/` |
| 新增文件 | `MODULE_MAPPING.md`、`PATCH_MANIFEST.md`、`EXPERIMENT_MATRIX.md`、`UPSTREAM_COMPATIBILITY.md` |
| 修改文件 | 无 |
| 输入 | P0-M1 至 P0-M10 的实现记录 |
| 输出 | 可审计文档集 |
| 依赖任务 | P0-M1、P0-M3 |
| 验收标准 | 每个新增文件、每个修改 StarVLA 原文件的 patch 都有记录；必须修改原文件时，要求后续代码实现使用 `STARFLOW_PATCH_BEGIN / END` 标记 |
| 建议测试命令 | `Test-Path docs/starflow_vla/MODULE_MAPPING.md; Test-Path docs/starflow_vla/PATCH_MANIFEST.md` |
| 风险 | 实现改动和文档脱节 |
| 回滚方式 | 以 patch manifest 为准回滚对应改动 |
| 是否修改 StarVLA 原文件 | 否 |
| 是否阻断 P0 闭环 | 是 |

## 2. P1 任务：增强能力，不阻断 P0

### P1-M1：continuous_head state path

| 字段 | 内容 |
| --- | --- |
| 任务名称 | continuous_head state path |
| 为什么不是 P0 | P0 默认走 QwenPI_v3 state-to-instruction，continuous state path 是对照增强 |
| 前置条件 | P0-M5 通过 |
| 涉及文件 | `LayerwiseFM_ActionHeader.py` state_encoder；新增 `state_bridge.py` |
| 是否需要修改 StarVLA 原文件 | 尽量否，优先 hook / wrapper |
| 对应测试 | state shape test、single batch overfit、与 discretized_instruction 对照 |
| 是否影响已有 checkpoint | 可能影响 config 与 state_dict key；需标记不兼容字段 |
| 验收标准 | `state_mode=continuous_head` 可 dry-run；不影响 P0 默认配置 |
| 是否阻断 P0 闭环 | 否 |

### P1-M2：max_action_dim=14 + action_mask

| 字段 | 内容 |
| --- | --- |
| 任务名称 | 7DoF/14DoF mixed action 接口 |
| 为什么不是 P0 | P0 单臂 7DoF 已足够跑通主闭环，14D mask 是 bimanual-ready 扩展 |
| 前置条件 | P0-M5、P0-M8 |
| 涉及文件 | collator、action head loss、config |
| 是否需要修改 StarVLA 原文件 | 可能，需要最小 masked loss patch |
| 对应测试 | mixed batch shape test、mask denominator test |
| 是否影响已有 checkpoint | 可能影响 action_dim 和 action_decoder shape |
| 验收标准 | 单臂后 7 维不参与 loss；14D 样本可参与训练 |
| 是否阻断 P0 闭环 | 否 |

### P1-M3：masked flow loss

| 字段 | 内容 |
| --- | --- |
| 任务名称 | masked flow loss |
| 为什么不是 P0 | P0 不需要 14D mixed batch |
| 前置条件 | P1-M2 |
| 涉及文件 | `LayerwiseFM_ActionHeader.py`、`GR00T_ActionHeader.py` |
| 是否需要修改 StarVLA 原文件 | 可能，需要 `STARFLOW_PATCH_BEGIN / END` |
| 对应测试 | finite loss、zero-mask clamp、single-arm regression |
| 是否影响已有 checkpoint | 不影响权重结构，但影响 loss 逻辑 |
| 验收标准 | mask 分母正确；无 NaN/Inf；P0 7DoF 路径结果不回退 |
| 是否阻断 P0 闭环 | 否 |

### P1-M4：solver manifest 完善

| 字段 | 内容 |
| --- | --- |
| 任务名称 | solver manifest 完善 |
| 为什么不是 P0 | P0 只需记录 Euler 默认步数；完整 solver sweep 属于增强 |
| 前置条件 | P0-M3、P0-M9 |
| 涉及文件 | `mapping.py`、checkpoint manifest、eval report |
| 是否需要修改 StarVLA 原文件 | 可能，优先旁路 JSON |
| 对应测试 | manifest schema test |
| 是否影响已有 checkpoint | 不影响权重，影响 metadata |
| 验收标准 | 记录 solver、num_inference_timesteps、latency profile |
| 是否阻断 P0 闭环 | 否 |

### P1-M5：RoboCasa / RoboTwin 完整接入

| 字段 | 内容 |
| --- | --- |
| 任务名称 | RoboCasa / RoboTwin 完整接入 |
| 为什么不是 P0 | P0 先跑通 LIBERO 最小闭环，三数据集完整接入不阻断 framework smoke |
| 前置条件 | P0-M8、P0-M10 |
| 涉及文件 | `examples/Robocasa_tabletop/`、`examples/Robotwin/` |
| 是否需要修改 StarVLA 原文件 | 可能新增 dataset adapter/config |
| 对应测试 | data schema test、eval smoke |
| 是否影响已有 checkpoint | 数据 mixture 改变会影响训练结果，需新 run_id |
| 验收标准 | 两个 benchmark 的 train/eval smoke 均可运行 |
| 是否阻断 P0 闭环 | 否 |

### P1-M6：latency profiling

| 字段 | 内容 |
| --- | --- |
| 任务名称 | latency profiling |
| 为什么不是 P0 | P0 关注可训练可评测，性能优化可后置 |
| 前置条件 | P0-M10 |
| 涉及文件 | `deployment/model_server/`、eval report |
| 是否需要修改 StarVLA 原文件 | 否，优先外部 profiling |
| 对应测试 | p50/p95/p99 latency report |
| 是否影响已有 checkpoint | 否 |
| 验收标准 | 输出 latency profile 并写入 eval report |
| 是否阻断 P0 闭环 | 否 |

### P1-M7：upstream compatibility audit automation

| 字段 | 内容 |
| --- | --- |
| 任务名称 | 上游兼容审计自动化 |
| 为什么不是 P0 | P0 需要手工冻结基线即可，自动化属于长期维护增强 |
| 前置条件 | P0-M1、P0-M11 |
| 涉及文件 | `docs/starflow_vla/UPSTREAM_COMPATIBILITY.md`、可选脚本 |
| 是否需要修改 StarVLA 原文件 | 否 |
| 对应测试 | registry smoke、checkpoint load smoke、eval smoke |
| 是否影响已有 checkpoint | 否 |
| 验收标准 | 新 upstream commit 进入正式训练前必须有 audit record |
| 是否阻断 P0 闭环 | 否 |

## 3. P2 任务：Advanced，不阻断 P0/P1

### P2-M1：PerceiverAdapter

触发条件：多视角长 token 或高分辨率输入导致 token 过长、吞吐下降。  
为什么当前不做：P0 已有 future_tokens + cross-DiT 主线，Perceiver 会增加结构变量。  
与论文主实验关系：可作为 advanced ablation，不作为 H2 主线必要条件。  
最小验收方式：`perceiver_enabled=true` dry-run；token shape 正确；不影响 P0 config。

### P2-M2：Observation Geometry Adapter Interface for 《基于世界模型的移动操作规划与决策框架研究》 Bridge

| 字段 | 内容 |
| --- | --- |
| Task ID | P2-M2 |
| 任务名称 | Observation Geometry Adapter Interface |
| 优先级 | P2 |
| 目标 | 为《基于世界模型的移动操作规划与决策框架研究》衔接预留 VGGT / geometry token 输入接口 |
| 动机 | 保持本项目 P0/P1 聚焦 Flow Matching VLA，同时为《基于世界模型的移动操作规划与决策框架研究》的 3D world representation 输出提供 policy condition 接口 |
| 涉及文件 | `starVLA/model/modules/starflow_vla/geometry_adapter.py`、`starVLA/model/modules/starflow_vla/geometry_fusion.py`、配置文件、MODULE_MAPPING.md |
| 新增文件 | P2 阶段再新增，不阻断 P0/P1 |
| 输入 | depth / point_map / camera_pose / point_tracks / world_tokens |
| 输出 | geometry_tokens 或 enhanced_vl_embs_list |
| 依赖任务 | P0-M0 ~ P0-M10 完成后再考虑 |
| 验收标准 | P2 dry-run pass；`observation_geometry.enabled=false` 时不影响 P0/P1；不修改 QwenPI_v3 / LayerwiseFM 主逻辑 |
| 是否阻断 P0 闭环 | 否 |
| 是否阻断 P1 闭环 | 否 |

### P2-M3：显式 FlowCondition runtime dataclass

触发条件：多个 adapter/head 需要统一日志、诊断或跨模块接口。  
为什么当前不做：StarVLA-native P0 可用隐式张量 `vl_embs_list + state_features + future_tokens + action_features` 表达条件。  
与论文主实验关系：接口工程增强，不作为主贡献。  
最小验收方式：显式 dataclass 可选启用；P0 不依赖它；manifest 能记录 `flow_condition_runtime=true/false`。

### P2-M4：高级 Condition Injection 消融

触发条件：future_tokens + cross-DiT 与 MLP baseline 差异不明确，需要进一步定位条件注入机制。  
为什么当前不做：P0 只验证主路线，不展开 Additive / AdaLN / Cross Attention 多轴消融。  
与论文主实验关系：补充分析，不作为 P0 主线。  
最小验收方式：至少一组对照配置可 dry-run，并记录 condition ablation 结果。

### P2-M5：多视角长 token compression

触发条件：多摄像头或高分辨率输入导致 VLM hidden states 过长。  
为什么当前不做：P0 LIBERO 7DoF smoke 不需要长 token 压缩。  
与论文主实验关系：可服务多视角泛化附录实验。  
最小验收方式：压缩前后 token 数、显存、success_rate 对照完整。

### P2-M6：双臂 coordination loss / advanced embodiment token

触发条件：RoboTwin 或真实双臂任务进入主线，并出现左右臂协调失败。  
为什么当前不做：P0 是 single-arm first，14D mask 也仅为 P1。  
与论文主实验关系：未来扩展或附录，不作为当前 H1/H2/H3 主线。  
最小验收方式：coordination loss 可开关；不影响 7DoF checkpoint load。

### P2-M7：真实机器人高级部署扩展

触发条件：仿真 P0/P1 闭环稳定，并准备进入 Piper / MOZ1 等真实机器人。  
为什么当前不做：真实机器人部署依赖安全、标定、延迟和硬件接口，超出 P0 训练闭环。  
与论文主实验关系：工程展示或未来工作，不作为仿真主实验必要条件。  
最小验收方式：shadow mode、action replay、低速闭环安全门全通过。

## 4. 算法优化任务补充

本节补充两个算法优化方向：`future_tokens` 规划槽位优化和状态条件注入路径优化。它们基于 StarVLA 当前 `future_tokens / num_target_vision_tokens`、QwenPI_v3 `state-to-instruction` 与 action head `state_encoder` 能力设计；后续实现应优先采用配置、wrapper、hook 和模块化替代，不修改 QwenPI_v3 / LayerwiseFM / GR00T 主体逻辑。

### P0-M7a：Future Tokens Planning Slot Dry-run

| 字段 | 内容 |
| --- | --- |
| Task ID | P0-M7a |
| 任务名称 | future_tokens 规划槽位优化 / 动作条件 token 预算优化 |
| 优先级 | P0 |
| 目标 | 将 `num_target_vision_tokens=0/16/32/64` 固化为算法变量，支撑 H2-a；`ft=8` 可作为可选 dense-sweep，不进入当前 P0/P1 执行矩阵 |
| 涉及文件 | StarVLA config、`starflow_mapping`、checkpoint manifest、实验报告 |
| 新增文件 | `configs/starflow_vla/ablations/future_tokens_{0,16,64}.yaml`；`ft=32` 由 P0-M5-Stage1 baseline 覆盖，不重复创建 |
| 修改 StarVLA 原文件 | 否；如 0-token 边界必须 patch，需另开 issue 并标注 `STARFLOW_PATCH_BEGIN / END` |
| Stage A 验收 | 四组配置存在或在 issue 中列出；config parse 通过；dry-run 不加载完整大模型也能验证字段；`starflow_mapping` 记录 `num_target_vision_tokens`、`state_mode`、`action_dim`、`action_horizon` |
| Stage B 复验 | 至少一组 token 数完成 single batch overfit；记录 loss finite/NaN、peak memory、latency、action smoothness |
| 不作为 P0 完成条件 | LIBERO full success rate、RoboCasa/RoboTwin cross benchmark、完整 latency Pareto |
| 输出 | Implementation Record + H2-a 实验占位表，未完成指标写 `[待 Stage B A100 复验]` |

### P1-M1：Future Tokens Ablation on LIBERO（算法优化优先项）

| 字段 | 内容 |
| --- | --- |
| Task ID | P1-M1 |
| 任务名称 | Future Tokens Ablation on LIBERO |
| 目标 | 在 LIBERO full split 上比较 `0/16/32/64` planning slots；`ft=8` 不作为当前 P0/P1 编号 |
| 前置条件 | P0-M7a Stage B single batch overfit 至少一组通过 |
| 指标 | success_rate、loss curve、single batch overfit speed、action chunk smoothness、peak memory、inference latency、按任务长度分组 success |
| 输出 | LIBERO ablation report；所有未运行项写 `[待 LIBERO eval]` |
| 是否阻断 P0 | 否 |

### P1-M2：State Conditioning Path Comparison（算法优化优先项）

| 字段 | 内容 |
| --- | --- |
| Task ID | P1-M2 |
| 任务名称 | 状态条件注入路径优化 / 本体状态条件建模优化 |
| 目标 | 比较 `discretized_instruction` 与 `continuous_head` |
| 默认路径 | P0 保持 `state-to-instruction`，复用 QwenPI_v3 |
| P1 路径 | `continuous_head`，复用或包装 action head `state_encoder` |
| 指标 | LIBERO success、state-sensitive task success、long-horizon success、loss curve、overfit speed、smoothness、latency、state noise robustness |
| 输出 | state path comparison report；未完成指标写 `[待 Stage B A100 复验]` |
| 是否阻断 P0 | 否 |

### P2-M1：Hybrid Gated State Conditioning（算法优化优先项）

| 字段 | 内容 |
| --- | --- |
| Task ID | P2-M1 |
| 任务名称 | hybrid gated state conditioning |
| 目标 | 同时保留 instruction state 与 continuous state，通过 gate 融合语义对齐和控制精度 |
| 前置条件 | P1-M2 完成并证明单一路径仍有明显短板 |
| 指标 | cross benchmark drop、missing state robustness、state noise robustness、long-horizon success、latency |
| 输出 | hybrid gated ablation report；未完成指标写 `[待 RoboCasa / RoboTwin eval]` |
| 是否阻断 P0/P1 | 否 |
