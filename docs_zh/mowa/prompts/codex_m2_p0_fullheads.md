# MoWA M2 P0 FullHeads Prompt

## 当前阶段

M2：P0 Video-Generation-Free WAM。

## 目标

在 G0 通过后，设计或实现 `MoWAP0FullHeads` 接口、P0 labels coverage report 和可选 `MoWAP0GatedHeads` 计划。P0 不生成未来视频。

## 必读文档

1. `docs_zh/mowa/AGENTS.md`
2. `docs_zh/mowa/02_detailed_design.md` 第 5、8、10、11 章
3. `docs_zh/mowa/04_task_breakdown.md` 的 M2 任务卡
4. `docs_zh/mowa/05_experiment_registry.md`
5. `docs_zh/mowa/07_implementation_log.md`
6. M1 G0 输出报告

## 允许修改文件

- `starVLA/model/modules/mowa/`
- `configs/mowa/mowa_full_heads_interface.yaml`
- `configs/mowa/mowa_e002_future_gated_heads_candidate.yaml`，仅 optional
- `tests/mowa/`
- `docs_zh/mowa/07_implementation_log.md`

## 禁止修改文件

- `LayerwiseFM_ActionHeader.py` 内部主逻辑。
- P1 latent 训练代码。
- P2 decoder 代码。
- 三份核心 SOT。
- 新增未冻结 P0 head。

## 具体任务

1. 固定七类 P0 heads：task_progress、manipulation_readiness、failure_risk、next_best_view_score、subgoal_feasibility、object_visibility_future、action_outcome_class。
2. 缺失标签用 mask，不新增替代 head。
3. 输出 `P0FutureFeatures` 给后续 bridge。
4. P0-GatedHeads 只能作为 E-002 optional，禁止 per-head/leave-one-out/selected-head sweep。
5. 更新实现日志。

## 测试命令

```bash
pytest tests/mowa -q
```

## 验收标准

1. P0 是独立 WAM branch，不是 VLA action head 改名。
2. P0 head loss 与 action loss 分开记录。
3. P0 features 可进入 WAMActionBridge。
4. E-001/E-002 编号含义不变。

## 输出格式

输出修改文件、P0 接口摘要、测试命令与结果、实验编号影响、下一步。

## 阶段边界

不得实现 P1-b0、HLC-GCI、P2 decode 或 action head 主干重写。
