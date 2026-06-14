# StarFlow-VLA Codex 执行规范

版本：V0.1  
日期：2026-06-14  
范围：本规范约束 Codex 在 StarFlow-VLA 项目中的文档编辑、代码编辑、轻量检查和实施记录方式。

## 1. 环境边界

当前 Codex 任务只代表文档编辑 / 代码编辑 / 轻量构建验证，不等价于项目训练、评测或部署结论。

Stage A lightweight validation 可以做：

- 阅读仓库；
- 生成和修改文档；
- 生成代码 patch；
- 运行轻量静态检查；
- 运行不依赖目标 GPU / 真实数据 / 真实机器人环境的 smoke test；
- mock batch / config parse / import / registry 级验证。

Stage A lightweight validation 不应宣称完成：

- 8 卡正式训练验证；
- Virtaicloud / Bita 云训练验证；
- A100 性能结论；
- 真实机器人闭环验证；
- 部署性能、推理延迟、吞吐、显存峰值的最终结论；
- 任何需要目标训练/部署环境才能确认的结论。

无法在当前环境验证的内容必须标注：

`未在 Stage B 目标模型 smoke 或正式训练/评测环境验证，需在对应目标环境验证。`

## 2. 两段式环境策略

StarFlow-VLA 开发采用两段式环境策略。

### Stage A：1×A100 40G lightweight validation

Stage A 默认使用 `1×A100 40G`，用于：

- 文档编辑；
- 代码构建；
- 仓库阅读；
- framework registry 静态检查；
- config parse；
- import test；
- mock batch test；
- json manifest serialize test；
- 不加载完整大模型的 dry-run。

Stage A 不用于：

- 正式训练；
- 真实模型性能结论；
- 完整显存 / 延迟 / 成功率结论；
- 真实机器人部署结论；
- 多阶段长训结论。

Stage A 的所有结论必须标注为：

`Stage A 1×A100 40G lightweight validation，仅代表轻量构建验证，不代表 Stage B 目标模型 smoke、正式训练、完整评测或部署结论。`

### Stage B：1×A100 40G target smoke validation

Stage B 默认也使用 `1×A100 40G`，区别不是硬件类型，而是验证强度。Stage B 用于：

- QwenPI_v3 reuse smoke；
- LayerwiseFM 7DoF forward/backward；
- single batch overfit；
- checkpoint save/load；
- predict_action shape check；
- LIBERO eval smoke；
- future_tokens 8/16/32/64 dry-run；
- 必要时进行 `num_target_vision_tokens=0` 边界测试。

Stage A 与 Stage B 默认均为 `1×A100 40G`。Stage A 偏文档、配置、import、registry 和 dry-run；Stage B 偏真实模型 smoke、single batch overfit、checkpoint 和 eval smoke。除非明确进入 P1/P2 advanced 或正式训练，不以切换硬件作为默认验证策略。

## 3. Codex 工作方式

每个任务必须按以下顺序执行：

1. 读取任务 ID；
2. 找到对应详细设计章节；
3. 检查当前仓库真实路径；
4. 写出实施计划；
5. 最小修改代码或文档；
6. 运行允许范围内的轻量测试；
7. 更新实施记录；
8. 更新验收清单；
9. 总结结果、偏离和下一步。

## 4. 设计到实现追踪

每个代码任务必须关联：

- 详细设计文档章节；
- `TASK_BREAKDOWN.md` 任务 ID；
- `CODEX_ISSUES.md` issue ID；
- `ACCEPTANCE_CHECKLIST.md` 验收项；
- 相关文件路径；
- 测试命令和测试结果；
- 当前运行环境；
- 是否需要阶段 B A100 复验。

## 5. 实施记录要求

每个 P0/P1/P2 任务完成后，必须更新 `IMPLEMENTATION_LOG.md`。

每条实施记录必须包含：

- Task ID；
- Issue ID；
- Design Reference；
- Environment；
- Stage：A 1×A100 40G lightweight validation 或 B 1×A100 40G target smoke validation；
- Steps；
- Files Changed；
- Tests Run；
- Results；
- Not Run / Need Target Verification；
- Deviations；
- Rollback；
- Next。

## 6. 禁止事项

禁止：

- 把本地文档编辑环境当作部署环境；
- 把 Stage A lightweight validation 写成 Stage B target smoke validation；
- 把未运行的测试写成已通过；
- 把无法验证的性能指标写成结论；
- 复制 QwenPI_v3 形成大体重复文件；
- 把 PerceiverAdapter、显式 FlowCondition runtime、14D action_mask 放进 P0；
- 删除或重命名 StarVLA 原始文件；
- 无记录地修改 StarVLA 原文件；
- 不更新 `IMPLEMENTATION_LOG.md` 就结束任务。
