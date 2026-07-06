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

## 10. 当前只读 Smoke 记录

本节记录本机 `robocasa365_open_drawer_target_human` 只读 metadata smoke 结果。该检查不读取 parquet / video 内容，不 profile 训练吞吐，不启动 P0/P1，不计入实验预算。

| 字段 | 当前值 |
|---|---|
| report_json | `docs_zh/mowa/mowa_g0_robocasa365_open_drawer_smoke.json` |
| data_root | `playground/Datasets/robocasa365` |
| relative_path | `v1.0/target/atomic/OpenDrawer/20250816/lerobot` |
| path_exists | true |
| schema_smoke_available | true |
| declared_total_episodes | 514 |
| declared_total_frames | 137431 |
| declared_fps | 20（dataset metadata declared，不作为 MoWA measured profile） |
| declared_robot_type | `PandaOmron` |
| feature_keys_present | `observation.state`、`action`、`timestamp`、`frame_index`、`episode_index`、`task_index` |
| video_feature_keys | `observation.images.robot0_eye_in_hand`、`observation.images.robot0_agentview_left`、`observation.images.robot0_agentview_right` |
| parquet_file_count | 514 |
| video_file_count | 1542 |
| episode_extra_dir_count | 514 |
| obs_fps_status | Data Gate |
| action_hz_status | Data Gate |
| history_window_status | Data Gate |
| future_window_status | Data Gate |
| leakage_check_status | TBD |
| go_no_go | `No-Go: schema smoke available; profile/leakage pending` |

当前结论：OpenDrawer target/human 最小数据闭环已经具备只读 schema smoke 条件；G0 仍未放行 P0/P1，因为 profile、P0 label coverage、latent cache 可行性和 leakage 检查尚未完成。

## 11. 当前 UnifiedEpisode Schema Smoke

本节记录 `M1-002/M1-003` 的只读 parquet schema smoke。该检查只读取一个 episode 的 parquet schema 和少量标量预览，不读取视频内容，不启动训练。

| 字段 | 当前值 |
|---|---|
| adapter | `starVLA/dataloader/mowa/robocasa365_adapter.py` |
| episode | `episode_000000.parquet` |
| row_count | 334 |
| instruction | `Open the right drawer.` |
| observation.state shape | `(16,)` |
| action shape | `(12,)` |
| required columns | `observation.state`、`action`、`timestamp`、`frame_index`、`episode_index`、`task_index` |
| missing columns | none |
| sample timestamps | `0.0, 0.05, 0.10, 0.15, 0.20`（parquet preview，不作为 MoWA official profile result） |
| window smoke | history `(1, 2, 3)`、future `(4, 5)`、action target `(3, 4)` with anchor `3` |
| future action leakage | action target 不进入 `inputs` |
| obs_fps | Data Gate |
| action_hz | Data Gate |
| history_window | Data Gate |
| future_window | Data Gate |

当前结论：RoboCasa365 Lerobot parquet 可映射到 `MoWAUnifiedEpisode` 草案，并可通过现有 `MoWAEpisodeToWindowSampler` 生成不泄漏 future action 的 `MoWAWindowSample`。G0 仍未放行 P0/P1，因为尚未完成 profile、P0 label coverage、latent cache 可行性和跨 episode leakage 检查。

## 12. 当前 Dataset Boundary / Leakage Smoke 与 P0 Coverage 初判

本节记录 `M1-004` 的只读 dataset-level smoke。该检查读取 `episodes.jsonl` 和 3 个 sampled episode 的 parquet schema，不读取视频内容，不启动训练。

| 字段 | 当前值 |
|---|---|
| report_json | `docs_zh/mowa/mowa_g0_robocasa365_open_drawer_dataset_smoke.json` |
| cli | `tools/mowa/g0_dataset_smoke.py` |
| sampled_episode_indices | `0, 1, 4` |
| sampled_row_counts | `334, 235, 208` |
| metadata_episode_count | 514 |
| metadata_min_episode_length | 130 |
| metadata_max_episode_length | 603 |
| smoke window config | history_steps=`3`、future_steps=`2`、action_chunk_steps=`2`，仅 smoke target |
| sampled history/action boundary | `history_indices <= anchor_index`，smoke passed |
| sampled future boundary | `future_indices > anchor_index`，smoke passed |
| future action leakage | `action_chunk_target` 不进入 `inputs`，smoke passed |
| cross_episode_leakage_status | `smoke_passed` |
| obs_fps / action_hz | Data Gate |
| production window | Data Gate |

P0 label coverage 初判：

| P0 head | 当前结论 |
|---|---|
| task_progress | candidate_from_frame_index_and_episode_length; Data Gate |
| manipulation_readiness | candidate_from_state_action_reward; Data Gate |
| failure_risk | insufficient_without_failure_annotation; Data Gate |
| next_best_view_score | requires_view_label_or_proxy_definition; Data Gate |
| subgoal_feasibility | candidate_from_reward_done_and_task_progress; Data Gate |
| object_visibility_future | requires_video_decode_or_visibility_proxy; Data Gate |
| action_outcome_class | candidate_from_next.reward_next.done; Data Gate |

当前结论：M1-004 sampled boundary / leakage smoke 已通过，但它不是完整门禁。P0 label 只是可构造性初判，尚未实现或验证 label builder；G0 仍未放行 P0/P1。

## 13. 固定主对比 Recipe 可用性

本节记录 MoWA P0/P1/P2 固定主对比数据配方的下载可用性。该检查只验证任务目录和基础 Lerobot 结构是否存在，不读取 parquet / video 内容。

| 字段 | 当前值 |
|---|---|
| recipe_name | `mowa_robocasa365_target_human_atomic_core_v1` |
| report_json | `docs_zh/mowa/mowa_g0_robocasa365_atomic_core_recipe_smoke.json` |
| cli | `tools/mowa/g0_recipe_smoke.py` |
| split / source | `target` / `human` |
| task_type | atomic |
| task_count | 10 |
| available_task_count | 10 |
| missing_task_count | 0 |
| go_no_go | `TBD: recipe available; profile/leakage/labels still Data Gate` |

当前固定任务清单：

| task | 当前可用性 |
|---|---|
| OpenDrawer | available |
| OpenCabinet | available |
| CloseFridge | available |
| CloseToasterOvenDoor | available |
| CoffeeSetupMug | available |
| NavigateKitchen | available |
| PickPlaceCounterToCabinet | available |
| PickPlaceToasterToCounter | available |
| PickPlaceSinkToCounter | available |
| TurnOnSinkFaucet | available |

当前结论：固定主对比 recipe 的 10 个 target/human/atomic 任务已具备基础 Lerobot 结构。该结论只代表目录、meta、data、videos 可用；P0/P1/P2 主对比仍必须继续通过 profile、label coverage、leakage 和 latent cache 相关 G0 检查后才能启动。

## 14. 当前只读 Profile Smoke

本节记录 `OpenDrawer target/human` 的只读 profile smoke。该检查读取 sampled episode 的少量 parquet 行，只用于检查 timestamp / frame_index / shape 一致性，不读取视频内容，不给出正式 fps / Hz / window 结论。

| 字段 | 当前值 |
|---|---|
| report_json | `docs_zh/mowa/mowa_g0_robocasa365_open_drawer_profile_smoke.json` |
| cli | `tools/mowa/g0_profile_smoke.py` |
| sampled_episode_indices | `0, 1, 4` |
| sampled_row_counts | `334, 235, 208` |
| timestamp_monotonic | true |
| frame_index_monotonic | true |
| timestamp_delta_preview | `0.05`（parquet preview，不作为 MoWA official fps / Hz profile result） |
| action_shape_consistent | true，shape=`(12,)` |
| state_shape_consistent | true，shape=`(16,)` |
| next_done_seen | false（preview rows only） |
| reward_signal_seen | false（preview rows only） |
| obs_fps_status | Data Gate |
| action_hz_status | Data Gate |
| history_window_status | Data Gate |
| future_window_status | Data Gate |

当前结论：sampled parquet 的 timestamp、frame_index、state/action shape 一致性通过 smoke；生产用 WAM Hz、history window、future window、action chunk 仍保持 Data Gate。

## 15. 当前 P0 Label Coverage Smoke

本节记录 `OpenDrawer target/human` 的 P0 label coverage smoke。该检查只做字段级可构造性判断，不生成训练标签，不定义阈值，不启动 P0 模型。

| 字段 | 当前值 |
|---|---|
| report_json | `docs_zh/mowa/mowa_g0_p0_label_coverage_open_drawer_smoke.json` |
| cli | `tools/mowa/g0_p0_label_coverage_smoke.py` |
| sampled_episode_indices | `0, 1, 4` |
| available_columns | `action`、`annotation.human.task_description`、`annotation.human.task_name`、`episode_index`、`frame_index`、`index`、`next.done`、`next.reward`、`observation.state`、`task_index`、`timestamp` |
| constructible_heads | `task_progress`、`action_outcome_class` |
| masked_or_data_gate_heads | `manipulation_readiness`、`failure_risk`、`next_best_view_score`、`subgoal_feasibility`、`object_visibility_future` |

| P0 head | status | source_fields | mask_rule |
|---|---|---|---|
| task_progress | candidate_constructible | `frame_index`、`timestamp`、`episode_index` | mask if frame_index/timestamp/episode length unavailable |
| manipulation_readiness | Data Gate | `observation.state`、`action` | mask until readiness proxy is validated |
| failure_risk | masked | `failure_annotation` | mask by default |
| next_best_view_score | masked | `view_score`、`visibility_label` | mask by default |
| subgoal_feasibility | Data Gate | `next.reward`、`next.done`、`frame_index` | mask until feasibility proxy is validated |
| object_visibility_future | masked | `future_video`、`object_visibility_proxy` | mask by default |
| action_outcome_class | candidate_constructible | `next.reward`、`next.done` | mask if reward/done unavailable |

当前结论：P0 FullHeads 首轮只能以 mask 方式处理不可构造 heads；若要先做训练 smoke，候选可构造 head 仅为 `task_progress` 和 `action_outcome_class`。所有阈值、class mapping 和 proxy 定义仍为 Data Gate。

## 16. 当前 Latent Cache Manifest Smoke

本节记录 `OpenDrawer target/human` 的 latent cache manifest smoke。该检查只验证视频路径和确定性 cache key，不执行 Wan encoder / VAE，不生成 latent tensor，不写 cache 文件。

| 字段 | 当前值 |
|---|---|
| report_json | `docs_zh/mowa/mowa_g0_latent_cache_manifest_open_drawer_smoke.json` |
| cli | `tools/mowa/g0_latent_cache_manifest_smoke.py` |
| sampled_episode_indices | `0, 1, 4` |
| video_keys | `observation.images.robot0_eye_in_hand`、`observation.images.robot0_agentview_left`、`observation.images.robot0_agentview_right` |
| manifest_entries | 9 |
| missing_video_count | 0 |
| latent_shape_status | Data Gate |
| cache_hash_status | Data Gate |
| encoder_status | Data Gate |

当前结论：OpenDrawer sampled episode 的三路视频路径齐全，latent cache manifest 的路径和 cache key 规则可以稳定生成。真实 latent 编码、latent shape、cache hash 和吞吐仍为 Data Gate。

## 17. 当前 10 项 Atomic Core 批量 Smoke

本节记录 `mowa_robocasa365_target_human_atomic_core_v1` 的 10 项批量只读 smoke。该检查逐项运行 profile、P0 label coverage 和 latent cache manifest smoke；不启动训练，不解码视频，不执行 Wan encoder / VAE。

| 字段 | 当前值 |
|---|---|
| summary_json | `docs_zh/mowa/g0_atomic_core_smoke/mowa_g0_atomic_core_batch_smoke_summary.json` |
| output_dir | `docs_zh/mowa/g0_atomic_core_smoke/` |
| task_count | 10 |
| passed_task_count | 10 |
| failed_task_count | 0 |
| checks | `profile`、`p0_label_coverage`、`latent_cache_manifest` |
| sampled_episode_indices | `0, 1, 4` |
| profile_smoke | 10/10 timestamp monotonic、frame_index monotonic、state/action shape consistent |
| p0_label_coverage | 10/10 仅 `task_progress`、`action_outcome_class` 为 candidate constructible；其余五类 head 继续 mask / Data Gate |
| latent_manifest | 10/10 missing_video_count=0 |
| go_no_go | `TBD: batch smoke passed; production profile/labels/leakage remain Data Gate` |

当前结论：10 项核心 recipe 已通过训练前只读 smoke，足以支持下一步进入 P0 label builder / ConstructibleHeads smoke 设计。它仍不是完整 G0 放行：生产 WAM Hz、history/future/action chunk、真实 label builder 阈值、完整 leakage gate、真实 latent cache 编码和吞吐仍保持 Data Gate。

## 18. 当前 P0 ConstructibleHeads Label Builder Dry-run

本节记录 P0 ConstructibleHeads label builder 的 dry-run 结果。该检查只生成当前可构造 head 的 smoke targets 和七类 head mask，不启动训练，不定义 production 阈值，不冻结 `action_outcome_class` class mapping。

| 字段 | 当前值 |
|---|---|
| open_drawer_report_json | `docs_zh/mowa/mowa_g0_p0_label_builder_open_drawer_smoke.json` |
| batch_summary_json | `docs_zh/mowa/g0_atomic_core_smoke/mowa_g0_atomic_core_p0_label_builder_summary.json` |
| cli | `tools/mowa/g0_p0_label_builder_smoke.py` |
| task_count | 10 |
| passed_task_count | 10 |
| failed_task_count | 0 |
| constructible_heads | `task_progress`、`action_outcome_class` |
| masked_heads | `manipulation_readiness`、`failure_risk`、`next_best_view_score`、`subgoal_feasibility`、`object_visibility_future` |
| action_outcome_class_mapping | Data Gate；dry-run 只保留 `next.reward` / `next.done` |
| go_no_go | `TBD: constructible label smoke passed; production labels remain Data Gate` |

当前结论：P0 的最小 ConstructibleHeads 训练前目标可以从 10 项 recipe 中 dry-run 生成，但仍不能等价为正式 label builder 放行。正式训练前还需要确认 batch 级 dataloader 接入、完整 leakage gate、production profile，以及是否接受只用两个 head 的 P0 smoke 配置。

## 19. 当前 Batch 级 Dataloader Smoke

本节记录 10 项 recipe 的 batch 级 dataloader smoke。该检查组合 `WindowSample` 边界、action target、ConstructibleHeads labels/masks 和 leakage invariant；不实例化生产 dataloader，不读取视频内容，不启动训练。

| 字段 | 当前值 |
|---|---|
| report_json | `docs_zh/mowa/g0_atomic_core_smoke/mowa_g0_atomic_core_batch_dataloader_smoke.json` |
| cli | `tools/mowa/g0_batch_dataloader_smoke.py` |
| task_count | 10 |
| sampled_episode_indices | `0, 1, 4` |
| sample_count | 30 |
| window_config | history_steps=`3`、future_steps=`2`、action_chunk_steps=`2`，smoke_only_target |
| future_action_leakage_status | `smoke_passed` |
| constructible_label_status | `smoke_passed` |
| input_keys | `current_index`、`history_actions`、`history_indices`、`instruction`、`observations` |
| target_keys | `action_chunk_target`、`future_indices`、`wam_targets` |
| p0_label_keys | `action_outcome_class`、`task_progress` |
| go_no_go | `TBD: batch dataloader smoke passed; production dataloader remains Data Gate` |

当前结论：10 项 recipe 的 sampled windows 可以与 ConstructibleHeads dry-run targets/masks 对齐，且 future action target 不进入 WAM inputs。该检查为最小 P0 ConstructibleHeads 训练入口提供前置依据，但生产 dataloader、完整 leakage gate、正式 window 参数和训练配置仍需单独放行。

## 20. 当前 Metadata 级 Leakage Gate

本节记录 10 项 recipe 的 metadata 级 leakage gate。该检查读取每个任务的 `meta/episodes.jsonl`，对每个 episode 选择 start/mid/end 三类 anchor，验证 sampler 边界和 future action target-only invariant；不读取视频内容，不实例化生产 dataloader workers，不启动训练。

| 字段 | 当前值 |
|---|---|
| report_json | `docs_zh/mowa/g0_atomic_core_smoke/mowa_g0_atomic_core_leakage_gate_smoke.json` |
| cli | `tools/mowa/g0_leakage_gate_smoke.py` |
| task_count | 10 |
| episode_count | 5055 |
| checked_window_count | 15165 |
| failed_window_count | 0 |
| window_config | history_steps=`3`、future_steps=`2`、action_chunk_steps=`2`，metadata_level_smoke |
| future_action_leakage_status | `smoke_passed` |
| cross_episode_leakage_status | `smoke_passed` |
| go_no_go | `TBD: metadata leakage smoke passed; production dataloader remains Data Gate` |

当前结论：10 项 recipe 的 metadata 级 sampler 边界和 target-only action invariant 已通过 smoke。该结果关闭了 G0 的主要 metadata leakage 风险，但仍不等价于生产 dataloader workers / distributed sampler / train-val split 的正式放行。

## 21. 当前 10 项 Temporal Profile

本节记录 10 项 recipe 的 full-recipe temporal profile。该检查读取所有 sampled recipe parquet 的 scalar columns，用于确认 timestamp / frame_index / shape / reward-done 轮廓；不读取视频内容，不测试训练吞吐，不自动冻结 WAM Hz、history window、future horizon 或 action chunk。

| 字段 | 当前值 |
|---|---|
| report_json | `docs_zh/mowa/g0_atomic_core_smoke/mowa_g0_atomic_core_temporal_profile.json` |
| cli | `tools/mowa/g0_temporal_profile.py` |
| task_count | 10 |
| episode_count | 5055 |
| parquet_count | 5055 |
| metadata_total_frames | 1342150 |
| parquet_total_rows | 1342150 |
| timestamp_monotonic | true |
| frame_index_monotonic | true |
| timestamp_delta_values | `0.049999`、`0.05`、`0.050001`、`0.050003`（parquet scalar profile；浮点舍入差异） |
| state_shapes | `(16,)` |
| action_shapes | `(12,)` |
| reward_signal_seen | true |
| next_done_seen | true |
| obs_fps_status | Data Gate |
| action_hz_status | Data Gate |
| history_window_status | Data Gate |
| future_window_status | Data Gate |
| go_no_go | `TBD: temporal profile passed; production WAM Hz/window remain Data Gate` |

任务级行数：

| task | episode_count | parquet_total_rows | min_episode_rows | max_episode_rows |
|---|---:|---:|---:|---:|
| OpenDrawer | 514 | 137431 | 130 | 603 |
| OpenCabinet | 500 | 184024 | 133 | 664 |
| CloseFridge | 513 | 155443 | 133 | 615 |
| CloseToasterOvenDoor | 505 | 86401 | 95 | 305 |
| CoffeeSetupMug | 502 | 117168 | 142 | 434 |
| NavigateKitchen | 500 | 72786 | 48 | 282 |
| PickPlaceCounterToCabinet | 502 | 131904 | 148 | 638 |
| PickPlaceToasterToCounter | 512 | 148353 | 183 | 497 |
| PickPlaceSinkToCounter | 501 | 194952 | 215 | 712 |
| TurnOnSinkFaucet | 506 | 113688 | 139 | 523 |

当前结论：10 项 recipe 的 scalar temporal profile 已通过，metadata frame 数与 parquet row 数一致，timestamp / frame_index 单调，state/action shape 一致。该结果可以作为 P0 ConstructibleHeads 最小训练入口的数据轮廓依据；正式训练配置仍必须显式声明 WAM Hz、history/future window 和 action chunk 的来源，不能把本节自动解释为最终窗口冻结。

## 22. 当前 P0 ConstructibleHeads 最小训练入口准备

本节记录 P0 ConstructibleHeads one-step train smoke 的入口准备状态。该入口只用于验证当前 G0 已可构造的两个 head 能完成 forward / loss / backward / one-step update，不计入 E-001 至 E-020，不放开其他五类 P0 head。

| 字段 | 当前值 |
|---|---|
| config | `configs/mowa/mowa_p0_constructible_heads_smoke.yaml` |
| module | `starVLA/model/modules/mowa/p0_heads.py` |
| cli | `tools/mowa/p0_constructible_heads_train_smoke.py` |
| test | `tests/mowa/test_mowa_p0_heads.py` |
| constructible_heads | `task_progress`、`action_outcome_class` |
| masked_heads | `manipulation_readiness`、`failure_risk`、`next_best_view_score`、`subgoal_feasibility`、`object_visibility_future` |
| labels_source | G0 smoke labels |
| update_step | 手写 one-step SGD；不使用 `torch.optim` |
| expected_vram | 0；默认 CPU 运行，不调用 `.cuda()` |
| smoke_report_json | `docs_zh/mowa/mowa_p0_constructible_heads_train_smoke.json` |
| sample_count | 30 |
| input_shape | `[30, 4]` |
| loss_before | 0.024293631315231323 |
| loss_after | 0.024045661091804504 |
| class_mapping_status | Data Gate |
| go_no_go | `TBD: train smoke passed; production training remains Data Gate` |

当前结论：P0 ConstructibleHeads 最小训练入口已完成 one-step smoke：当前 G0 已可构造的 `task_progress` 与 `action_outcome_class` 两个 head 可以完成 forward / loss / backward / one-step update，loss 在一次手写 SGD step 后下降。该结果不计入 E-001，不放开其他五类 P0 head，不等价于 production dataloader / train-val split / distributed sampler / P0 主训练放行；`class_mapping_status`、WAM Hz 和 window 仍保持 Data Gate。

## 23. 当前 Production Preflight Smoke

本节记录 P0 主训练前的 production-entry preflight。该检查只验证 deterministic train/val split、rank 分片和多 worker 少量 schema/window 读取，不启动训练主干，不计入 E-001，不冻结 production window。

| 字段 | 当前值 |
|---|---|
| report_json | `docs_zh/mowa/g0_atomic_core_smoke/mowa_g0_atomic_core_production_preflight_smoke.json` |
| cli | `tools/mowa/g0_production_preflight_smoke.py` |
| module | `starVLA/dataloader/mowa/production_preflight.py` |
| split_strategy | `episode_index_modulo` |
| val_every | 10 |
| train_episode_count | 4544 |
| val_episode_count | 511 |
| split_overlap_count | 0 |
| worker_count | 2 |
| worker_sample_count | 16 |
| failed_sample_count | 0 |
| rank_count | 2 |
| distributed_overlap_count | 0 |
| split_status | `smoke_passed` |
| worker_status | `smoke_passed` |
| distributed_sampler_status | `smoke_passed` |
| future_action_leakage_status | `smoke_passed` |
| go_no_go | `TBD: production preflight smoke passed; main training still requires explicit E-001 launch` |

当前结论：固定 10 项 recipe 的 production-entry smoke 已通过，train/val split、rank 分片和多 worker 读取的最小不变量成立，future action 仍未进入 WAM inputs。该结果只说明 E-001 前置数据入口风险下降，不代表已启动或放行 P0 主训练；正式训练前仍需显式确认 E-001、训练配置、资源预算和保存/恢复策略。

## 24. 当前 E-001 Readiness Smoke

本节记录 E-001 启动前 readiness 聚合检查。该检查只读取已有 G0/P0/preflight 报告，不启动训练，不修改训练主干，不计入 E-001。

| 字段 | 当前值 |
|---|---|
| report_json | `docs_zh/mowa/mowa_e001_readiness_smoke.json` |
| cli | `tools/mowa/e001_readiness_smoke.py` |
| recipe_available | true |
| production_preflight_passed | true |
| p0_one_step_smoke_passed | true |
| p0_fullheads_interface_created | true |
| e001_launch_draft_created | true |
| e001_launch_ready_false | true |
| runtime_policy_draft_created | true |
| runtime_policy_confirmed_false | true |
| task_count | 10 |
| available_task_count | 10 |
| train_episode_count | 4544 |
| val_episode_count | 511 |
| split_overlap_count | 0 |
| distributed_overlap_count | 0 |
| worker_sample_count | 16 |
| failed_sample_count | 0 |
| p0_smoke_sample_count | 30 |
| p0_fullheads_interface_config | `configs/mowa/mowa_full_heads_interface.yaml` |
| e001_launch_draft_config | `configs/mowa/mowa_e001_launch_draft.yaml` |
| e001_runtime_policy_draft_config | `configs/mowa/mowa_e001_runtime_policy_draft.yaml` |
| class_mapping_status | Data Gate |
| training_started | false |
| go_no_go | `TBD: E-001 prerequisites mostly passed; launch draft exists but resource/save-resume remain Data Gate` |

未解决项：

- `docs_zh/mowa/00_project_proposal.md`、`01_technical_survey.md`、`02_detailed_design.md` 当前不可读。
- P0 FullHeads interface draft、E-001 launch draft 和 runtime policy draft 已创建，但 launch draft 明确 `launch_ready=false`，runtime policy 明确 `policy_confirmed=false`，不可执行。
- production WAM Hz/window 仍为 Data Gate。
- `class_mapping_status` 仍为 Data Gate。
- runtime policy draft 尚未确认。
- resource budget 仍为 TBD。

当前结论：E-001 的数据、最小 P0 smoke、P0 FullHeads interface、launch draft 与 runtime policy draft 前置条件基本成立，但 launch draft 明确不可执行，runtime policy 仍未确认，不能自动进入主训练。下一步必须先明确资源预算和保存/恢复策略，并将 `policy_confirmed` 与 `launch_ready` 从 false 显式改为 true；任何修改训练主干、checkpoint/resume/save 逻辑或正式启动训练都需要单独确认。
