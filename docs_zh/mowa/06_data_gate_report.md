# MoWA G0 Data Verification Gate 报告模板

本文件是模板，不运行数据处理，不填入未实测结果。所有未测字段必须写 `TBD` 或 `Data Gate`。G0 是工程门禁，不计入 E-001 至 E-020；G0 未通过前不得启动 P0/P1 主训练。

## 0. 报告元信息

| 字段 | 值 |
|---|---|
| report_id | `mowa_g0_datagate_TBD` |
| date | TBD |
| git_commit | TBD |
| script_version | TBD |
| data_root | TBD |
| operator | TBD |
| conclusion | TBD |

## 1. 候选数据集清单

| 数据集 | 状态 | 默认用途 | 下载 / license | 备注 |
|---|---|---|---|---|
| RoboCasa / RoboCasa365 | Data Gate | 第一闭环、P0/P1 主训练候选 | TBD | 需核验 mobile base 强度。 |
| AIRoA MoMa | Data Gate | 真实移动操作关键候选 | TBD | 需核验 Base/EEF/Success/Lang、Hz、任务长度。 |
| EBench | Data Gate | readiness / handoff / failure 诊断 | TBD | 需核验任务字段和 eval adapter。 |
| MoMa-Kitchen | Data Gate | P0 final-pose readiness 诊断 | TBD | 视觉真实感弱，不作主视觉 benchmark。 |
| LIBERO / RoboTwin | Data Gate | action schema sanity、操作辅助 | TBD | 不能作为移动操作主证据。 |
| MobileManiBench / Kitchen-R | Data Gate | 开放后优先核验 | TBD | coming soon / 未开放时不进近期主矩阵。 |

## 2. 字段完整性

| 数据集 | RGB | language | robot_state | base_pose | action | success/stage | timestamps | license | 字段覆盖结论 |
|---|---|---|---|---|---|---|---|---|---|
| RoboCasa / RoboCasa365 | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | TBD |
| AIRoA MoMa | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | TBD |
| EBench | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | TBD |
| MoMa-Kitchen | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | TBD |
| LIBERO / RoboTwin | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | TBD |

## 3. Episode 统计

| 数据集 | episode_count | split | episode_frames | episode_seconds | task_count | success_rate | 状态 |
|---|---|---|---|---|---|---|---|
| RoboCasa / RoboCasa365 | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | TBD |
| AIRoA MoMa | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | TBD |
| EBench | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | TBD |
| MoMa-Kitchen | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | TBD |
| LIBERO / RoboTwin | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | TBD |

## 4. Obs FPS / Action Hz

| 数据集 | obs fps | action Hz | WAM Hz | history window | future horizon | action chunk | 结论 |
|---|---|---|---|---|---|---|---|
| RoboCasa / RoboCasa365 | Data Gate | Data Gate | target / Data Gate | target: 16 frames 起步 | target: 28 frames 起步 | Data Gate | TBD |
| AIRoA MoMa | Data Gate | Data Gate | target / Data Gate | Data Gate | Data Gate | Data Gate | TBD |
| EBench | Data Gate | Data Gate | target / Data Gate | Data Gate | Data Gate | Data Gate | TBD |
| MoMa-Kitchen | Data Gate | Data Gate | target / Data Gate | Data Gate | Data Gate | Data Gate | TBD |
| LIBERO / RoboTwin | Data Gate | Data Gate | target / Data Gate | Data Gate | Data Gate | Data Gate | TBD |

说明：SparseVideoNav profile baseline 中的 4Hz、16 history、28 future、chunk=4、约70G 仅可作为 profile baseline，不是 MoWA measured 数字。

## 5. P0 Label 可构造性

| 数据集 | task_progress | manipulation_readiness | failure_risk | next_best_view_score | subgoal_feasibility | object_visibility_future | action_outcome_class | 结论 |
|---|---|---|---|---|---|---|---|---|
| RoboCasa / RoboCasa365 | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | TBD |
| AIRoA MoMA | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | TBD |
| EBench | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | TBD |
| MoMa-Kitchen | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | TBD |
| LIBERO / RoboTwin | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | Data Gate | TBD |

缺失标签必须 mask，不得新增未冻结 P0 head。

## 6. P1 Latent Cache 可行性

| 数据集 | current RGB 可编码 | future RGB target 可编码 | history RGB 可编码 | latent_shape | cache_hash | latency | 结论 |
|---|---|---|---|---|---|---|---|
| RoboCasa / RoboCasa365 | Data Gate | Data Gate | Data Gate | Data Gate | TBD | Data Gate | TBD |
| AIRoA MoMa | Data Gate | Data Gate | Data Gate | Data Gate | TBD | Data Gate | TBD |
| EBench | Data Gate | Data Gate | Data Gate | Data Gate | TBD | Data Gate | TBD |
| MoMa-Kitchen | Data Gate | Data Gate | Data Gate | Data Gate | TBD | Data Gate | TBD |
| LIBERO / RoboTwin | Data Gate | Data Gate | Data Gate | Data Gate | TBD | Data Gate | TBD |

## 7. Leakage 检查

| 检查项 | 规则 | 结果 | 失败动作 |
|---|---|---|---|
| history action boundary | `max(history_action_t) <= anchor_t` | TBD | 阻断训练并修 sampler。 |
| future frames input whitelist | future frames 只进入 target/cache，不进入 WAM input | TBD | 阻断 P0/P1。 |
| future action label | future action 只作为 action target 或 invariant 检查对象 | TBD | 阻断训练。 |
| boundary mask | window 不跨 episode 边界 | TBD | 修 sampler。 |
| shuffled robot pair | shuffled robot history pair 可构造 | TBD | 暂缓 E-005。 |

## 8. Go / No-Go 结论

| 阶段 | Go 条件 | 当前结论 | 动作 |
|---|---|---|---|
| G0 → P0 | 至少一个训练域 schema/profile/P0 labels/leakage 通过 | TBD | TBD |
| G0 → P1-b0 | latent cache smoke 通过，且 P0 有可对照数据域 | TBD | TBD |
| G0 → P1-b1 | history/state/action 字段可构造，leakage test 通过 | TBD | TBD |
| G0 → P2 | P1 能输出 predicted clean future latent | TBD | TBD |

## 9. 降级路径

| 失败情况 | 降级路径 | 输出 |
|---|---|---|
| 数据不可下载或 license 不明 | 不进入主闭环，仅保留文献参考 | DataGateReport 风险项 |
| mobile base 字段不足 | 降级为 household manipulation / action sanity | 数据域用途调整 |
| P0 标签覆盖不足 | 只训练可构造 heads，缺失 head mask | P0 label coverage report |
| latent cache 不通过 | 阻断 P1，保留 P0 | latent cache failure report |
| leakage test 不通过 | 阻断所有训练 | sampler fix TODO |
| eval adapter 不通过 | 降级为离线指标或等待适配 | eval adapter risk |
