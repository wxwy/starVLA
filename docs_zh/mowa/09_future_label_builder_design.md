# MoWA Future Label Builder 设计索引

本文件是当前仓库内 future label coverage 与 label/mask builder 的派生设计索引，不是核心 Source-of-Truth。若与 `00_project_proposal.md`、`01_technical_survey.md`、`02_detailed_design.md` 冲突，以核心 SOT 为准；若未来核心 SOT 再次缺失，也不得在本文件中补写或改写其结论。

## 当前冻结范围

- future supervision heads 仍为七类冻结 head。
- 生产默认可构造 heads 保持 `task_progress` 和 `action_outcome_class`。
- `subgoal_feasibility` 与 `manipulation_readiness` 已通过 18 个 atomic task 的 sidecar 预计算和分布审查，进入 **ablation-ready** 状态；它们将在 E-001 4-head ablation 中显式解 mask，但尚未升级为全局默认 constructible heads。
- `failure_risk` 继续保持 masked：在纯 human demo 数据上为正样本空类。
- `next_best_view_score` 与 `object_visibility_future` 已通过 MuJoCo forward kinematics + 相机投影 + ray-cast 遮挡检查实现视觉 proxy。`object_visibility_future` 默认主摄像头改为 `robot0_eye_in_hand` 以获得更有区分度的标签分布。sidecar 已写入并审查：
  - `next_best_view_score` 在 18/18 任务上为 `candidate`，可进入 **5-head ablation**。
  - `object_visibility_future` 仅 5/18 任务为 `candidate`，10/18 为 `review_high_majority_rate`，3/18 为 `blocked`；6-head ablation **暂缓**，OVF 保持 masked 直至 proxy 进一步改进或支持 per-task mask。
- future action 不得作为 WAM 输入，只能作为 action target 或 leakage invariant 检查对象。

## 当前实现入口

- label coverage: `starVLA/dataloader/mowa/full_head_label_coverage.py`
- label builder: `starVLA/dataloader/mowa/full_head_label_builder.py`
- 生产 dataloader label/mask attachment: `starVLA/dataloader/gr00t_lerobot/datasets.py`
- 共享 head 常量: `starVLA/mowa_constants.py`
- future heads module: `starVLA/model/modules/mowa/full_heads.py`

## 数据格式约定

- smoke label report 可以保留 rich dict，用于展示 `class_mapping_status` 等元信息。
- production dataloader 传给模型的是 tensor-ready target 值。
- `action_outcome_class` 当前冻结为 `[next_reward, next_done_flag]` 二维 tensor-ready target，映射版本为 `reward_done_vector_v1`，`class_mapping_status=confirmed_for_e001_initial_target`。
- future heads 默认 `action_outcome_loss_type=mse`，保持当前 `[reward, done]` tensor target 行为；这一步冻结的是标签语义，不是 loss 迁移。
- `action_outcome_loss_type=cross_entropy_done` 仅作为 opt-in 兼容接口，使用 target 最后一维 `next_done` 作为二分类标签；不得在类别映射未冻结前把它设为默认。

## 未解决项

- `action_outcome_class` 是否切换为 CE/Focal，需要先冻结类别映射；当前仍为 MSE 回归 `[next_reward, next_done_flag]`。
- `failure_risk` 在纯 human demo 数据上无正样本，保持 masked；若后续引入失败/截断数据，需重新评估。
- `next_best_view_score` 视觉 proxy 已审查通过，进入 5-head ablation；`object_visibility_future` 因全量分布中仅 5/18 任务为 `candidate`，3/18 任务 `blocked`，暂缓解 mask。后续可改进方向：增加主摄像头候选、调整 occlusion 容差或 max_distance、或改为 per-task mask 后再评估 6-head。
- 若将 `subgoal_feasibility` / `manipulation_readiness` 从 ablation-ready 升级为全局默认 constructible heads，需要：
  - 4-head vs 2-head ablation 结果证明收益或至少无回归；
  - 决定是否把 `OpenCabinet` 的 `subgoal_feasibility` 纳入训练（其正样本率 4.18%，低于 5% data gate 门槛）；
  - 同步 `MOWA_FUTURE_CONSTRUCTIBLE_HEADS` 与 `full_head_label_coverage.py`。
- 若新增可构造 head，必须同步 dataloader label builder、mask、QwenOFT/QwenPI_v3 supervision probe、单测和实现日志。

## Proxy 解 mask 验收标准

任何 masked head 从草案进入 production loss 前，至少必须满足以下条件：

- 输入字段在当前 parquet schema 或已冻结的 task schema 中真实存在，并能在固定 episode 上读取。
- 标签规则能写成无未定义中间变量的伪代码；窗口长度、阈值、predicate 版本必须冻结。
- 已生成 smoke label report，记录 label coverage、正负样本比例、mask 比例和边界样本。
- 标签分布不能接近全 0 / 全 1；若有效样本比例低于 5%，必须在 data gate 中显式记录训练价值风险。
- 已新增单测覆盖 episode 末尾、提前 `done`、字段缺失、schema 缺失和 mask 分支。
- 已同步 `full_head_label_coverage.py`、`full_head_label_builder.py`、production dataloader mask attachment、QwenOFT/QwenPI_v3 supervision probe 和实现日志。

当前 `task_progress` 使用 `frame_index / episode_length` 作为时间进度 proxy，不是真实语义进度。若后续任何 head 依赖 `task_progress`，必须明确这是时间进度；一旦 `task_progress` 定义升级，相关 proxy 规则需要重新校准。

## Proxy 标签构造方案草案

以下方案只冻结标签语义和数据前置条件，不代表已经接入 production dataloader / loss。三项仍是 proxy，不是真实人工标签；未满足输入前置条件的样本必须保持 mask。

### 1) `failure_risk`

- 目标语义：预测 anchor step 之后短窗口内是否出现可观测失败终止，而不是预测“多久后成功”。
- 固定窗口：`H_risk = 10` 个后续 step；窗口为 `(t, t + H_risk]`，截断到 episode 末尾。
- 输入：窗口内 `next.reward`、`next.done`；可选使用 `frame_index` 只做窗口边界和 episode 末尾定位。
- 成功 proxy：窗口内某一步满足 `next.done == true AND next.reward > 0`。
- 失败 proxy：窗口内某一步满足 `next.done == true AND next.reward <= 0`，且该步之前没有成功 proxy。
- 输出：二分类 scalar，`1.0 = high risk`，`0.0 = low risk`。
- **关键数据发现**：在 `robocasa365 target/human` pure human demo 数据上，所有 `done=true` 都伴随 `reward=1.0`。`failure_risk_data_gate_smoke.json` 显示 10/10 任务的 `positive_count=0`，即当前数据域没有失败样本。该 head 只有在混入含失败样本的数据源（machine-generated、noisy demo、truncated episode 等）后才可能有训练价值。在当前数据上建议保持 mask，或将语义切换为 continuous progress-stall predictor。
- 标签规则：
  - 若窗口内先出现失败 proxy，`failure_risk = 1.0`。
  - 若窗口内先出现成功 proxy，`failure_risk = 0.0`。
  - 若窗口内没有任何 `next.done == true`，该 anchor 为 censored sample，`failure_risk` 必须 mask，不用“未在 H 内成功”直接当高风险。
- 伪代码：

```python
future_rewards = rewards[t + 1 : t + 1 + H_risk]
future_dones = dones[t + 1 : t + 1 + H_risk]

for reward, done in zip(future_rewards, future_dones):
    if not done:
        continue
    if reward > 0:
        return 0.0, True
    return 1.0, True
return 0.0, False
```

- reward 假设：该 proxy 依赖 `next.reward > 0` 能代表成功终止。data gate 必须验证 RoboCasa365 reward 是否为稀疏 0/1；若 reward 是稠密递增信号，则 success proxy 需要改为 `next.done == true AND next.reward >= success_threshold`，并单独校准 `success_threshold`。
- failure 噪声：当前把所有 `done == true AND reward <= 0` 统一视作非成功终止，可能混合 timeout、truncation、真实失败和数据采集边界。若 data gate 发现非成功终止中 timeout/truncation 比例较高，必须保持 mask 或改用软标签。
- 覆盖率预期：对长度 `L=200-600` 的 episode，`H_risk=10` 可能只覆盖 episode 末尾约 `H_risk/L ~= 2%-5%` 的 anchor。低覆盖率是预期风险，必须在 smoke report 中量化。
- 独立性检查：在 sampled anchors 上计算 `action_outcome_class` 的 `next_done` 分量与 `failure_risk` 标签的相关系数或 mutual information。若相关性过高，说明该 head 的增量信息有限。
- mask 条件：缺少 `next.reward` / `next.done`、窗口内无任何 `done`、reward 分布假设未通过 data gate、非成功终止噪声不可控。
- 当前状态：**保持 masked**。`mowa_all_atomic_task_future_label_distribution.json` 显示 18/18 任务的 `failure_risk` 均为 `blocked_single_class`（positive_count=0）。在纯 human demo 数据上无训练信号，因此不参与 4-head ablation。

### 2) `subgoal_feasibility`

- 目标语义：给定一个候选 subgoal，预测当前 anchor 是否能在短窗口内推进到该 subgoal 的完成条件；不能退化为 `failure_risk` 的取反，也不能退化为单纯 `frame_index` 进度。
- 固定窗口：`H_subgoal = 20` 个后续 step；窗口为 `(t, t + H_subgoal]`，截断到 episode 末尾。
- 必需输入：`task_name`、候选 `subgoal_id`、该 subgoal 的 completion predicate、窗口内 predicate 状态；`next.reward` / `next.done` 只作为成功或失败终止的辅助信号。
- subgoal schema：每个 atomic task 需要一份有序 subgoal schema，至少包含 `subgoal_id`、`description`、`completion_predicate`、`failure_predicate(optional)`。没有 schema 的 task 不能构造该 head。
- 输出：二分类 scalar，`1.0 = feasible`，`0.0 = infeasible`。
- 标签规则：
  - 若窗口内满足候选 subgoal 的 completion predicate，且此前没有失败 proxy，`subgoal_feasibility = 1.0`。
  - 若窗口内先出现失败 proxy，或到窗口结束仍未满足 completion predicate，`subgoal_feasibility = 0.0`。
  - 若缺少候选 subgoal、completion predicate 或 predicate 所需字段，该 anchor 必须 mask。
- 不接受的降级：不能把 `t + K` 的未来帧、未来 reward 峰值或更长窗口的 success 直接当作候选 subgoal；这些会把该 head 退化成 progress / outcome predictor。
- 示例 schema，仅用于暴露依赖，不表示当前字段已存在：
- OpenDrawer 试点 schema：基于 `mowa_opendrawer_state_mapping_audit.json`，当前至少可以冻结一个真实 completion predicate；但还不能把整套 subgoal 链都当作已解锁。

```yaml
task: OpenDrawer
subgoals:
  - subgoal_id: approach_drawer
    description: "base reaches drawer operation range"
    completion_predicate:
      type: state_threshold
      field: base_distance_to_drawer
      operator: "<"
      value: 0.3
  - subgoal_id: grasp_handle
    description: "gripper closes on drawer handle"
    completion_predicate:
      type: state_threshold
      field: gripper_handle_contact
      operator: "=="
      value: true
  - subgoal_id: open_drawer
    description: "drawer joint moves beyond open threshold"
    completion_predicate:
      type: state_threshold
      field: drawer_joint_position
      operator: ">"
      value: 0.15
```

- OpenDrawer 当前可落地的最小真实版本：

```yaml
task: OpenDrawer
task_source:
  fixture_ref_field: ep_meta.fixture_refs.drawer
  fixture_ref_value_example: stack_1_left_group_4
  joint_name_rule: "{fixture_ref}_slidejoint"
  raw_state_source: extras/states.npz
  state_layout: [time, qpos, qvel]
subgoals:
  - subgoal_id: open_drawer
    description: "target drawer reaches open state"
    completion_predicate:
      type: simulator_joint_threshold
      field: raw_drawer_qpos
      source_joint_name: "{fixture_ref}_slidejoint"
      state_index_rule: "derive qpos index from XML joint order"
      qpos_index_example: 25
      normalization:
        raw_to_progress: "(-raw_qpos) / (0.55 * abs(joint_range_min) / 2)"
        joint_range_example: [-0.6, 0.0]
      operator: ">="
      value: 0.95
```

- 注意：`qpos_index_example=25` 只对应示例 episode 0；真实 `OpenDrawer` 数据里 `drawer_ref` 和 `slidejoint` 的 qpos index 会随 episode 变化，因此任何可用实现都必须按 episode 的 `ep_meta.json` / `model.xml.gz` 动态解析 joint index，不能把单个 episode 的 index 固定推广到全量数据。

- OpenDrawer 当前不能冻结的部分：
  - `approach_drawer`：parquet `observation.state` 虽含 `base_position`、`end_effector_position_relative`、`gripper_qpos`，但没有 drawer handle 或 fixture 的相对位姿；目前不能稳定定义 “到达操作位”。
  - `grasp_handle`：当前没有 handle contact、handle relative pose 或 gripper-handle 距离，不能定义稳定 predicate。
- OpenDrawer 当前推荐口径：
  - `subgoal_feasibility` 可以先试点为单 subgoal schema，只评估 `open_drawer` 是否在 `H_subgoal` 内达成。
  - 不要强行补 `approach_drawer` / `grasp_handle` 的假阈值；在 handle site / contact 映射未完成前，它们必须保持 blocked。
- OpenDrawer 额外观察：
  - 早先那版只拿 episode 0 的 joint index 去扫全量数据，导致看起来只有极少数 episode 成功；按 episode 动态解析 joint index 后，`open_drawer` 的 state predicate 与 terminal success 在 514/514 个 episode 上都对齐，说明原始失配来自固定 index 口径，而不是任务本身无法构造。

- 阻塞链：当前缺少 subgoal schema；schema predicate 需要的 object distance、contact、joint position 等字段不在当前 parquet 标量列中；即使这些字段可从 `observation.state` 提取，也必须先冻结 state 维度语义和 per-task 阈值。
- **Production dataloader 断裂与预计算方案**：`subgoal_feasibility` 依赖的 `drawer_progress` 来自 `extras/<ep>/states.npz`，而当前 production dataloader（`datasets.py:_attach_mowa_future_labels`）只读取 parquet 标量列。已新增 `starVLA/dataloader/mowa/opendrawer_label_cache.py` 和 `tools/mowa/precompute_opendrawer_future_labels.py`，可离线解析每个 episode 的 `ep_meta.json` / `model.xml.gz` / `states.npz`，把 `drawer_progress` 和三 head 标签写成 parquet sidecar（`<dataset>/mowa_future_labels/opendrawer/episode_XXXXXX.parquet`）。dataloader 在运行时检测 sidecar 存在并直接合并，避免 online XML 解析开销。当前只有 OpenDrawer 完成这条链路；其它 task 需要各自 state mapping 后才能复用该模式。
- 更新后的阻塞链：
  - 通用情况仍 blocked：大多数 atomic task 还没有 task-specific schema，也没有 simulator-state 到 predicate 字段的映射。
  - OpenDrawer 已部分解锁：`open_drawer` completion predicate 的 source chain 已确认，可进入单 task schema 草案。
  - OpenDrawer 其余 manipulation-phase subgoal 仍 blocked：缺 handle pose / contact / EEF-to-handle 距离。
- mask 条件：未冻结 subgoal schema、缺少候选 `subgoal_id`、completion predicate 不可计算、predicate 依赖字段缺失、阈值版本缺失。
- 当前状态：**ablation-ready**。所有 18 个 atomic task 已通过 `SingleDofTaskBuilder` / 自定义 builder 构造单 subgoal `complete` 的 completion predicate；sidecar 已预计算并合并到 production dataloader。`mowa_all_atomic_task_future_label_distribution.json` 显示 17/18 任务为 `candidate`，仅 `OpenCabinet` 为 `review_low_minority_rate`（正样本 4.18%）。4-head ablation 将包含该 head，但 OpenCabinet 样本作为已知低少数类风险项记录。

### 3) `manipulation_readiness`

- 目标语义：预测当前 anchor 是否已经处于“可以开始有效物体操作”的状态；它描述导航到操作的切换条件，不描述任务最终是否成功。
- 固定窗口：`H_ready = 5` 个后续 step，用于校验 ready 状态是否紧邻有效 manipulation onset。
- 必需输入：`task_name`、`observation.state` 的维度语义映射、task-specific readiness predicate；`action` 只能作为校准或一致性检查信号，不能单独定义 ready。
- readiness schema：每个 atomic task 需要一份 readiness schema，至少包含 `state_fields`、`ready_predicate`、`manipulation_onset_predicate`、`threshold_version`。没有 state 维度语义或阈值版本的 task 不能构造该 head。
- 输出：二分类 scalar，`1.0 = ready`，`0.0 = not_ready`。
- 标签规则：
  - 若当前 `observation.state` 满足 task-specific `ready_predicate`，且 `(t, t + H_ready]` 内出现 `manipulation_onset_predicate`，`manipulation_readiness = 1.0`。
  - 若当前 state 明确不满足 `ready_predicate`，`manipulation_readiness = 0.0`。
  - 若 state 维度语义、predicate 或阈值缺失，该 anchor 必须 mask。
- 前置验证：编写任何 readiness predicate 前，必须先确认 RoboCasa `observation.state` 是否包含 base pose、gripper state、object-relative pose、contact / joint state 等 readiness 判定所需信号。若 state 不含这些信号，该 head 本质上依赖视觉或额外 simulator state，当前标量字段不能支撑。
- mask 条件：state 维度语义未冻结、task-specific readiness predicate 缺失、manipulation onset predicate 缺失、阈值版本缺失、所需物理量不在当前字段中。
- 当前状态：**ablation-ready**。当前实现采用两级 fallback：
  - 若 builder schema 提供 `handle_site_template` 且 MuJoCo 可用，则使用 `eef_to_handle_distance <= 0.05m` 的 proximity predicate（当前仅 OpenDrawer 走此分支）；
  - 否则使用 `progress-imminence` proxy：未来 `H_ready` 步内 progress 增量 ≥ 0.05。
  18 个任务 sidecar 已生成，`manipulation_readiness` 在全部任务上均为 `candidate`（正样本率 7.7%–46.1%）。该 proxy 尚未升级为全任务 proximity/contact 语义，但已满足 ablation 训练信号要求。
- **语义升级（OpenDrawer 已部分解决）**：最初 Codex 的实现把 “未来 `H_ready` 步内 drawer progress 增量 ≥ `readiness_progress_delta`” 当作 readiness，这实质是 `progress_imminence`。现已通过 MuJoCo forward kinematics 从 `model.xml.gz` + `states.npz` 提取 `eef_to_handle_distance` 和 `gripper_handle_contact`，并把 OpenDrawer 的 `manipulation_readiness` 升级为 proximity-based 定义：`eef_to_handle_distance <= readiness_distance_threshold`（默认 0.05m）且尚未完成 subgoal。这仍然不是完整的 manipulation readiness（缺少 force/contact 闭合语义），但已经从“drawer 即将动”升级到“末端执行器已接近把手”。
- **MuJoCo FK 不是交互仿真**：只需加载 episode 的 `model.xml.gz`，把 `states.npz` 的 qpos/qvel 写入 `mjData`，调用 `mj_forward()` 即可获得世界坐标系下的 handle/EEF 位置和 contact。不需要启动 RoboCasa env，也不需要渲染。
- OpenDrawer 审计型 sidecar 例外：当前已在 `future_label_audit.py` 中补出一个 audit-only `open` proxy，不要求 handle pose/contact，而是把 “未来 `H_ready` 步内 drawer progress 增量是否超过阈值” 当作 manipulation onset 的弱事件。这个版本的价值是让 `predicate/event -> sidecar schema -> mask/debug` 工程闭环跑通；它仍不能替代正式 readiness predicate，也不能直接作为 production loss 标签解 mask。

### Head 间逻辑一致性约束

- 若 `failure_risk = 1.0` 且 `subgoal_feasibility = 1.0` 出现在同一窗口语义下，必须作为异常样本检查；除非 subgoal 窗口显式长于风险窗口且文档解释该差异。
- 若 `manipulation_readiness = 0.0`，针对 manipulation-only subgoal 的 `subgoal_feasibility = 1.0` 需要额外审查。
- 这些约束只用于 data gate / label validation，不在未验证前作为训练 loss 或硬规则。

### 全 atomic task 预计算 sidecar 使用方式

为了让 `subgoal_feasibility` / `manipulation_readiness` / `failure_risk` 进入 production training，已改为按 task 注册 builder、离线预计算 sidecar 的模式：

```bash
# 预计算全部 18 个 atomic task 的 sidecar
python tools/mowa/precompute_all_atomic_task_future_labels.py \
  --output docs_zh/mowa/mowa_all_atomic_task_future_label_cache_manifest.json

# 审查每任务分布
python tools/mowa/report_future_label_distribution.py \
  --output docs_zh/mowa/mowa_all_atomic_task_future_label_distribution.json
```

输出目录为 `<dataset_path>/mowa_future_labels/<task_name>/`。每个 episode 对应一个 `episode_XXXXXX.parquet`，至少包含：

- `frame_index`
- `task_progress`（或任务相关的 progress 列）
- `failure_risk` / `failure_risk_mask`
- `subgoal_feasibility` / `subgoal_feasibility_mask`
- `manipulation_readiness` / `manipulation_readiness_mask`
- `object_visibility_future` / `object_visibility_future_mask`
- `next_best_view_score` / `next_best_view_score_mask`

当 builder schema 提供 `handle_site_template` 且 MuJoCo 可用时，`manipulation_readiness` 使用 proximity predicate：
`eef_to_handle_distance <= readiness_distance_threshold`（默认 0.05m）且尚未完成 subgoal。否则自动回退到 `progress_imminence` proxy。

`datasets.py:_attach_mowa_future_labels` 在 `enable_mowa_future_labels=true` 时会检测 sidecar：

- 若存在，通过 `_merge_task_label_cache` 合并 `subgoal_feasibility` / `manipulation_readiness`，并更新 mask；
- `failure_risk` 仅在缓存中存在正类时才解 mask，否则保持 masked（单类 override）。

`full_head_label_builder.py` 的 smoke 也会优先读取 sidecar，方便在不上训练的情况下验证标签分布。

### E-001 ablation 解 mask 决策

基于 `mowa_all_atomic_task_future_label_distribution.json`（2026-07-08，eye-in-hand 主摄 + ray-cast 遮挡版本）的分布结果，当前解 mask 决策如下：

- **2-head baseline**：`task_progress` + `action_outcome_class`，对应 `configs/mowa/mowa_e001_starflow_ft0_2head_baseline_ablation.yaml`。
- **4-head**：`task_progress` + `subgoal_feasibility` + `manipulation_readiness` + `action_outcome_class`，对应 `configs/mowa/mowa_e001_starflow_ft0_4head_ablation.yaml`。
- **5-head (+NBV)**：4-head + `next_best_view_score`，对应 `configs/mowa/mowa_e001_starflow_ft0_5head_nbv_ablation.yaml`。**可启用**：NBV 在 18/18 任务上为 `candidate`，std 在 0.06–0.40 之间，有足够区分度。
- **6-head (+NBV +OVF)**：5-head + `object_visibility_future`，对应 `configs/mowa/mowa_e001_starflow_ft0_6head_ablation.yaml`。**暂缓**：OVF 仅 5/18 任务为 `candidate`（CloseBlenderLid、CloseFridge、CloseToasterOvenDoor、OpenCabinet、PickPlaceDrawerToCounter），10/18 为 `review_high_majority_rate`，3/18 为 `blocked`（NavigateKitchen `blocked_all_masked`、PickPlaceSinkToCounter / TurnOnElectricKettle `blocked_single_class`）。全局启用 OVF 会让 blocked/review 任务贡献无效或极偏信号。

所有配置都显式通过 `layerwise_bridge_active_heads` / `future_supervision_active_heads` 控制激活 head，不依赖全局常量。`failure_risk` 不参与任何消融（18/18 `blocked_single_class`）。

**已知风险与处理**：

- `OpenCabinet` 的 `subgoal_feasibility` 正样本率为 4.18%，低于 5% 门槛，标记为 `review_low_minority_rate`。ablation 先保留该任务，观察其对 loss 和 eval 的影响；若出现明显不稳定，再在后续迭代中把 OpenCabinet 从 4-head 的 subgoal_feasibility loss 中排除。
- `manipulation_readiness` 在多数任务上仍是 `progress-imminence` proxy，不是完整的 proximity/contact readiness。ablation 结果将决定是否需要投入 MuJoCo FK 把其余任务也升级到 proximity predicate。
- `failure_risk` 全任务单类，维持屏蔽；若未来引入非成功终止数据，再按本文档 proxy 规则重启评估。
- `object_visibility_future` 分布不够健康（仅 5/18 `candidate`），6-head 配置已创建但暂不推荐启动。若后续要启用，必须先满足以下任一条件：
  1. OVF proxy 改进后重新审查，使 candidate 任务比例显著提高；
  2. 或 dataloader/trainer 支持 per-task head mask，让 blocked/review 任务不贡献 OVF loss。

启动命令（需人工确认 `launch_guard`）：

```bash
VARIANT=2head ./tools/mowa/launch_e001_ablation.sh
VARIANT=4head ./tools/mowa/launch_e001_ablation.sh
VARIANT=5head_nbv ./tools/mowa/launch_e001_ablation.sh
VARIANT=6head ./tools/mowa/launch_e001_ablation.sh
```

### Coverage 口径说明

`full_head_label_coverage.py` 当前对部分 head 做 field-level 检查，例如 `failure_risk` 可能仍以 `failure_annotation` 作为原始理想字段。本文档中的 `next.reward` / `next.done` 是替代 proxy 方案。后续若要实现该 proxy，必须同步更新 coverage 工具的字段口径，并明确区分 `ideal_annotation` 与 `proxy_source_fields`。

### 视觉 proxy 已构造（带遮挡与视角选择）

`next_best_view_score` 与 `object_visibility_future` 已通过 MuJoCo forward kinematics + 相机投影 + ray-cast 遮挡检查实现，不解码视频。标签随 sidecar 一起预计算，最终是否解 mask 取决于全量分布报告。

#### `object_visibility_future`

- 目标语义：未来短窗口内，任务目标 3D 点是否至少被一个主摄像头看到。
- 主摄像头：默认改为 `robot0_eye_in_hand`。固定视角摄像头（`agentview_*`）下目标几乎总在画面内，导致正样本率过高；eye-in-hand 相机随机械臂运动，能产生更有区分度的可见/不可见变化。
- 固定窗口：`H_visibility = 10` 个后续 step；窗口为 `(t, t + H_visibility]`，截断到 episode 末尾。
- 输入：episode `model.xml.gz`、`states.npz`、`ep_meta.json` 中的相机配置；任务相关的 3D target point（handle site、object default site 或 manipulated joint 的 parent body 中心）。
- 可见 proxy：目标点投影到主摄像头画面内、相机到目标距离 < `max_distance = 2.0m`，且从相机到目标点的 ray-cast 不被其它物体（目标 body 自身除外）遮挡。
- 输出：二分类 scalar，`1.0 = visible`，`0.0 = not visible`。
- mask 条件：任务已完成（`progress >= completion_threshold`）、目标点无法解析、缺少可用相机、模型加载失败。
- 伪代码：

```python
visible = False
for name in main_camera_names:  # default ("robot0_eye_in_hand",)
    future_scores = per_camera_visibility[name][t + 1 : t + 1 + H_visibility]
    if future_scores.size > 0 and np.any(future_scores > 0.0):
        visible = True
        break
object_visibility_future[t] = 1.0 if visible else 0.0
object_visibility_future_mask[t] = (progress[t] >= completion_threshold)
```

其中 `per_camera_scores` 在计算主摄像头时已通过 `mujoco.mj_ray` 剔除被其它 body 遮挡的样本（目标 body 自身命中且距离在 5cm 容差内仍视为可见）。

#### `next_best_view_score`

- 目标语义：未来短窗口内，所有候选摄像头中能达到的最佳可见度分数，用于指导 viewpoint 选择。
- 固定窗口：与 `object_visibility_future` 相同，`H_visibility = 10`。
- 输入：与 `object_visibility_future` 相同，但评估更多摄像头（默认加上 `robot0_frontview`、`robot0_robotview`、`robot0_eye_in_hand`）。
- score proxy：`score = max(0, 1 - distance / max_distance)`，当目标投影在画面内且距离 < `max_distance` 时非零；否则为 0。该 score **不做遮挡检查**，保留连续信号；遮挡检查只用于 `object_visibility_future`。
- 输出：连续 scalar，`[0.0, 1.0]`，用 MSE loss。
- mask 条件：与 `object_visibility_future` 相同。
- 伪代码：

```python
best_score = 0.0
for name in alternative_camera_names:
    future_scores = per_camera_visibility[name][t + 1 : t + 1 + H_visibility]
    if future_scores.size > 0:
        best_score = max(best_score, float(np.max(future_scores)))
next_best_view_score[t] = best_score
next_best_view_score_mask[t] = (progress[t] >= completion_threshold)
```

#### 已知限制

- **遮挡近似**：ray-cast 忽略透明/微小几何，且把目标 body 自身 5cm 内的命中视为可见；对 body-center fallback 目标可能过宽。
- **正样本率仍偏高**：全量审查显示 `object_visibility_future` 在 10/18 任务上 `positive_rate > 0.95`（review_high_majority_rate），3/18 任务 blocked。eye-in-hand 主摄虽比固定视角有区分度，但操作过程中目标仍长时间位于手爪视野内。这是 6-head 暂缓的主要原因。
- **摄像头可变性**：部分 episode 的 fixture / object 命名不一致，builder 已加入候选 site fallback（如微波炉多候选 site），若报告里某任务 ovf/nbv 异常低，优先检查 target site/body 解析。

#### 暂不构造

- 无。`next_best_view_score` 与 `object_visibility_future` 已实现为 proxy，`failure_risk` 因数据域无正样本继续保持 masked。

### 不推荐事项

- 不要把未来 `reward` / `done` 作为当前 step 输入特征；它们只能用于 label 构造。
- 不要在没有 data gate 的情况下把 proxy head 加入 `MOWA_FUTURE_CONSTRUCTIBLE_HEADS`。
- 不要为这些 proxy head 单独调整 loss scale，除非已有成对消融和标签分布证据。
- 不要用未来 action 定义 subgoal 或 readiness；future action 只能作为 action target 或 leakage invariant 检查对象。
