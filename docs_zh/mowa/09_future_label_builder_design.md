# MoWA Future Label Builder 设计索引

本文件是当前仓库内 future label coverage 与 label/mask builder 的派生设计索引，不是核心 Source-of-Truth。若与 `00_project_proposal.md`、`01_technical_survey.md`、`02_detailed_design.md` 冲突，以核心 SOT 为准；若未来核心 SOT 再次缺失，也不得在本文件中补写或改写其结论。

## 当前冻结范围

- future supervision heads 仍为七类冻结 head。
- 当前生产可构造 heads 仅为 `task_progress` 和 `action_outcome_class`。
- 其余 heads 必须保持 masked，直到数据定义和标签构造规则被单独确认。
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

- `action_outcome_class` 是否切换为 CE/Focal，需要先冻结类别映射。
- `failure_risk`、`subgoal_feasibility`、`object_visibility_future` 等 head 的 proxy/threshold 未确认前不得参与 loss。
- 若新增可构造 head，必须同步 dataloader label builder、mask、QwenOFT supervision probe、单测和实现日志。
