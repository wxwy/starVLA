# MoWA M5 Action Bridge Eval Prompt

## 当前阶段

M5：WAM-to-action bridge 与 coupling eval。

## 目标

设计或实现 `MoWAActionBridge`，将 P0/P1 WAM features 映射为 StarFlow / LayerwiseFM 可消费的 layerwise condition features，并设计 E-006 coupling / feature removal eval。

## 必读文档

1. `docs_zh/mowa/AGENTS.md`
2. `docs_zh/mowa/02_detailed_design.md` 第 3、5、6、8、10、11 章
3. `docs_zh/mowa/04_task_breakdown.md` 的 M5 任务卡
4. `docs_zh/mowa/05_experiment_registry.md`
5. `docs_zh/mowa/07_implementation_log.md`

## 允许修改文件

- `starVLA/model/modules/mowa/`
- bridge / adapter 文件
- `configs/mowa/mowa_*bridge*.yaml`
- `tests/mowa/`
- coupling eval 报告
- `docs_zh/mowa/07_implementation_log.md`

## 禁止修改文件

- `LayerwiseFM_ActionHeader.py` 内部主逻辑。
- action head 主干重写。
- planner / FSM / RL / Scene Graph 对照。
- 新增 E-021。
- 三份核心 SOT。

## 具体任务

1. 设计 `MoWAActionBridge` 输入：P0FutureFeatures、predicted future latent、`h_hist/g_hist`。
2. 输出 LayerwiseFM 可消费的 layerwise condition features 或可选 global token。
3. 默认通过 bridge / adapter 接入，不改 LayerwiseFM 内部层。
4. 设计 feature removal、shuffle、correlation、latency-cost coupling 分析。
5. 更新实现日志。

## 测试命令

```bash
pytest tests/mowa -q
```

## 验收标准

1. bridge 输出 shape 与 action path 对齐。
2. feature removal 或 shuffle 能解释 WAM 是否真正进入 action path。
3. E-006 编号含义不变。
4. coupling 失败时 WAM 降级为诊断信号。

## 输出格式

输出修改文件、bridge 接口摘要、coupling eval 计划、测试命令与结果、下一步。

## 阶段边界

不得实现 P2 diagnostic；不得修改 action head 主干；不得新增未分配训练实验。
