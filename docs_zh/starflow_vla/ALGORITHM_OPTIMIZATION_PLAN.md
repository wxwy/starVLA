# StarFlow-VLA Algorithm Optimization Plan

Version: V0.1  
Design Reference: V4.6.2 Implementation Trace Patch  
Scope: algorithm optimization only  
Status: 已完成实验设计，待 Stage B A100 复验

本文件补充本项目主线算法优化与《基于世界模型的移动操作规划与决策框架研究》衔接方向。StarFlow-VLA 主线算法优化包括 Future Tokens Planning Slot Optimization 与 State Conditioning Path Optimization；RGB-Geometry Observation Fusion with VGGT 不属于本项目 P0/P1 主线，仅作为本项目 P2 optional extension / 与《基于世界模型的移动操作规划与决策框架研究》的接口预留。

本文件只细化 H2-a future_tokens planning slot optimization 与 H2-b state conditioning path optimization，不承担 H1 Flow Matching vs ACT 的完整证明。ACT baseline 已降级为后续完整论文扩展 / optional baseline，不进入当前 P0/P1 执行矩阵。本文件不替代 DESIGN.md 中 H1-H8 的完整研究假设矩阵。当前 P0/P1 执行矩阵只覆盖 StarFlow-VLA 最小闭环与 H2 局部优化，不宣称已经完成 H1/H3/Cross Benchmark/Sim2Real 等完整验证。当前文件只定义研究假设、配置变量、实验矩阵、验收指标和 P0/P1/P2 边界；不代表 StarFlowVLA 代码已经实现，不代表 A100、LIBERO、RoboCasa、RoboTwin、VGGT 或真实机器人结果已经完成。

## 0. Optimization Scope

本项目主线算法优化：

1. Future Tokens Planning Slot Optimization
2. State Conditioning Path Optimization

本项目 P2 / 与《基于世界模型的移动操作规划与决策框架研究》的衔接：

3. RGB-Geometry Observation Fusion with VGGT

第三项不阻断本项目 P0/P1，完整验证进入《基于世界模型的移动操作规划与决策框架研究》。本项目只保留 `observation_geometry` 配置、`GeometryAdapter` 抽象说明和 `geometry_fusion` hook，不实现完整 VGGT 训练闭环。

## 1. Future Tokens Planning Slot Optimization

中文名：`future_tokens 规划槽位优化 / 动作条件 token 预算优化`

### 1.1 Research Hypothesis H2-a

`future_tokens` / `num_target_vision_tokens` 不是单纯的工程 token 数，而是 Flow Matching VLA 中的 latent planning slot。不同 token 预算会影响动作规划容量、长时序任务稳定性、动作 chunk 平滑性、推理延迟和跨 Benchmark 泛化。

核心问题：

```text
future_tokens 是冗余视觉 token，
还是承载 planning / goal / temporal abstraction 的动作条件槽位？
```

### 1.2 StarVLA Basis

StarVLA 当前 LayerwiseFM / GR00T action head 已具备 `future_tokens` / `num_target_vision_tokens` 与 cross-DiT 条件机制。本项目优先通过配置、manifest 和 wrapper 记录实现该消融，不修改 QwenPI_v3 / LayerwiseFM / GR00T 主体逻辑。

### 1.3 Variables

默认变量：

```text
num_target_vision_tokens = 0 / 16 / 32 / 64
```

解释：

| Value | Meaning | Risk |
| --- | --- | --- |
| 0 | 移除 future planning slots | 可能触发空 token 边界 |
| 16 | 低预算 planning slots | 可能是低延迟折中 |
| 32 | StarVLA 常用默认 | 作为默认基线 |
| 64 | 高预算 planning slots | 显存和延迟增加 |

### 1.4 Metrics

- single batch overfit convergence speed
- loss finite / NaN stability
- LIBERO eval smoke success
- action chunk smoothness
- peak memory
- inference latency
- cross benchmark drop
- success by task length

### 1.5 P0 / P1 / P2 Boundary

| Stage | Requirement | Output |
| --- | --- | --- |
| P0 | configs dry-run + 至少一组 single batch overfit comparison entry | `[待 Stage B A100 复验]` 记录 |
| P1 | LIBERO full split success/loss/memory/latency | `[待 LIBERO eval]` 报告 |
| P2 | RoboCasa/RoboTwin + cross benchmark | `[待 RoboCasa / RoboTwin eval]` 报告 |

## 2. State Conditioning Path Optimization

中文名：`状态条件注入路径优化 / 本体状态条件建模优化`

### 2.1 Research Hypothesis H2-b

VLA 的本体状态条件既可以通过语言 token 与 VLM 对齐，也可以通过连续 state encoder 直接进入 action head。不同注入路径会影响 state-sensitive task、长时序控制、动作平滑性、跨 Benchmark 泛化和状态噪声鲁棒性。

核心问题：

```text
proprio state 应该通过 language token 进入 VLM，
还是通过 continuous state encoder 进入 action head？
hybrid gated path 是否能同时保留语义对齐和控制精度？
```

### 2.2 StarVLA Basis

StarVLA 当前 QwenPI_v3 已具备 state-to-instruction 路径，LayerwiseFM / GR00T action head 已具备 `state_encoder` 路径。本项目默认 P0 使用 state-to-instruction；P1 才做 continuous state encoder 对照；P2 才做 hybrid gated state conditioning。

### 2.3 Paths

| Path | state_mode | StarVLA Mapping | Priority |
| --- | --- | --- | --- |
| A. state-to-instruction | `discretized_instruction` | QwenPI_v3 `add_discretized_state_to_instruction` | P0 default |
| B. continuous state encoder | `continuous_head` | action head `state_encoder` | P1 comparison |
| C. hybrid gated state conditioning | `hybrid_gated` | instruction state + continuous state + gate | P2 advanced |

### 2.4 Metrics

- LIBERO success
- state-sensitive task success
- long-horizon success
- loss curve
- single batch overfit speed
- action smoothness
- cross benchmark drop
- latency
- robustness to state noise
- robustness to missing state

## 3. Experiment Matrix

所有指标在未完成目标环境验证前必须保持占位符，不得填写推测值。

| exp_id | hypothesis_id | config path | benchmark | state_mode | num_target_vision_tokens | action_dim | action_horizon | metrics | stage | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E-H2a-01 | H2-a | `configs/starflow_vla/ablations/future_tokens_0.yaml` | LIBERO small/full | `discretized_instruction` | 0 | 7 | 8 | overfit_steps/loss_finite/NaN/peak_memory/latency/action_smoothness | P0 Stage B | `[待 Stage B A100 复验]` |
| E-H2a-02 | H2-a | `configs/starflow_vla/ablations/future_tokens_16.yaml` | LIBERO full | `discretized_instruction` | 16 | 7 | 8 | success_rate/loss_curve/peak_memory/latency/task_length_success | P0/P1 | `[待 Stage B A100 复验]` |
| E-H2a-03 | H2-a | `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml` | LIBERO full | `discretized_instruction` | 32 | 7 | 8 | success_rate/loss_curve/peak_memory/latency/task_length_success | P0/P1 | `P0-M5-Stage1 baseline 默认配置，长训进行中；run_id 命名标签误写为 E-H2a-04，实际对应 E-H2a-03；当前约 steps_15500/100000，不代表最终成功率` |
| E-H2a-04 | H2-a | `configs/starflow_vla/ablations/future_tokens_64.yaml` | LIBERO full（P1）/ RoboCasa/RoboTwin（P2） | `discretized_instruction` | 64 | 7/14 | 8/16 | cross_drop/worst_family/peak_memory/latency/task_length_success | P1/P2 | `[待 LIBERO eval] / [待 RoboCasa / RoboTwin eval]` |
| E-H2b-01 | H2-b | `configs/starflow_vla/state/discretized_instruction.yaml` | LIBERO small/full | `discretized_instruction` | 32 | 7 | 8 | success_rate/state_sensitive_success/overfit_steps/smoothness/noise_robustness | P0/P1 | `P0-M5-Stage1 默认路径，与 E-H2a-03 同一场训练` |
| E-H2b-02 | H2-b | `configs/starflow_vla/state/continuous_head.yaml` | LIBERO full | `continuous_head` | 32 | 7 | 8 | success_rate/state_sensitive_success/loss_curve/smoothness/latency | P1 | `[待 LIBERO eval]` |
| E-H2b-03 | H2-b | `configs/starflow_vla/state/hybrid_gated.yaml` | LIBERO full | `hybrid_gated` | 32 | 7 | 8 | success_rate/state_sensitive_success/noise_robustness/missing_state_robustness/latency | P2 | `[待 Stage B A100 复验]` |
| E-H2b-04 | H2-b | `configs/starflow_vla/state/hybrid_gated_cross.yaml` | RoboCasa/RoboTwin/cross | `hybrid_gated` | 32 | 7/14 | 8/16 | cross_drop/long_horizon_success/noise_robustness/worst_family | P2 | `[待 RoboCasa / RoboTwin eval]` |

### 矩阵说明

- `E-H2a-03` 与 `E-H2b-01` 的变量组合（`num_target_vision_tokens=32` + `state_mode=discretized_instruction`）正是 `P0-M5-Stage1` 的默认配置，因此由同一场训练覆盖。
- 当前 `tmux starflow_train` 中的长训 run_id 写为 `P0-M5-E-H2a-04_..._ft32_250615`，属于命名标签笔误，实际配置为 `ft32`，应在实验记录中更正为 `E-H2a-03`。
- `E-H2a-04`（`future_tokens=64`）尚未开始训练，仍按 P1/P2 占位。

## 3.5 RGB-Geometry Observation Fusion with VGGT：P2 / 与《基于世界模型的移动操作规划与决策框架研究》的接口预留

VGGT 本来就不属于本项目 P0/P1 主线。本项目 P0/P1 继续聚焦 Qwen3-VL + StarVLA + Flow Matching，完成 RGB / language / state 到 action chunk 的策略学习闭环。

本项目允许预留如下配置，但默认关闭：

```yaml
observation_geometry:
  enabled: false
  encoder: vggt
  fusion_mode: cross_attention
  geo_token_num: 128
  coordinate_frame: camera
```

当 `enabled=false` 时，StarFlow-VLA 行为应与 RGB-only policy pathway 一致。P2 或《基于世界模型的移动操作规划与决策框架研究》启用时，VGGT 输出的 depth、point map、camera pose、point tracks 或 world tokens 可通过 `GeometryAdapter` 转换为 geometry tokens，再作为额外 condition 注入 policy。

完整算法验证属于《基于世界模型的移动操作规划与决策框架研究》：该研究将使用 VGGT 作为 geometric world state encoder，预测未来几何状态、可交互区域、遮挡关系和策略条件表征。

## 4. Implementation Rules

- P0 不新增独立 ActionTokenAdapter runtime 作为硬要求。
- P0 不改写 QwenPI_v3 / LayerwiseFM / GR00T 主体逻辑。
- P0/P1 不接入 VGGT，不新增 VGGT 依赖，不实现 RGB-3D fusion 训练闭环。
- 后续实现优先使用 config、wrapper、hook 和 manifest 记录。
- 如必须 patch StarVLA 原文件，需要另开 issue，并使用 `STARFLOW_PATCH_BEGIN / END` 标记。
- 所有实验产物必须写入 Implementation Record。
- Stage A 1×A100 40G lightweight validation 只说明配置、导入链路和轻量构建可检查，不代表训练或评测通过。

## 5. Presentation Notes

面向论文、企业预研或求职表达时，这两个方向可概括为：

- H2-a：把 StarVLA 的 future tokens 从工程参数提升为可解释的动作规划容量变量。
- H2-b：系统比较 VLA 本体状态通过语言通道、连续控制通道和混合门控通道的差异。
- H2-c / P2 bridge：预留 VGGT geometry adapter 作为《基于世界模型的移动操作规划与决策框架研究》接口，不声称本项目已完成 RGB-3D fusion 训练闭环。
- 当前状态：已完成实验设计，待 Stage B A100 复验。
