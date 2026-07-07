# MoWA M4 P1-b1 HLC-GCI Prompt

## 当前阶段

M4：P1-b1 HLC-GCI robot-history-conditioned latent WAM。

## 目标

设计或实现 `MoWAHLCGCI` 接口，输出 `C_hist/h_hist/g_hist`，并建立 history sampling train/inference consistency test 与 shuffled-robot sanity 计划。

当前前置条件：

1. E-003 的 fake-encoder cache 闭环已经存在。
2. P1-b0 已至少完成 config preview / train dry-run。
3. 本阶段只推进 HLC-GCI 接口、history contract 和 sanity 计划，不直接启动 E-004/E-005。

## 必读文档

1. `docs_zh/mowa/AGENTS.md`
2. `docs_zh/mowa/02_detailed_design.md` 第 6、8、10、11、14 章
3. `docs_zh/mowa/04_task_breakdown.md` 的 M4 任务卡
4. `docs_zh/mowa/05_experiment_registry.md`
5. M3 P1-b0 输出
6. `docs_zh/mowa/07_implementation_log.md`

## 允许修改文件

- `starVLA/model/modules/mowa/`
- `configs/mowa/mowa_hlc_gci_interface.yaml`
- `tests/mowa/`
- `docs_zh/mowa/07_implementation_log.md`

## 禁止修改文件

- future action label 输入 WAM。
- `LayerwiseFM_ActionHeader.py` 内部主逻辑。
- Rec-HLC 默认主线。
- Wan DiT full fine-tune 默认路线。
- P2 continuous video generation。

## 具体任务

1. 实现或设计 HLC-GCI 输入：visual history latent、text condition tokens、robot history latent。
2. 输出 `C_hist/h_hist/g_hist`。
3. 默认采用 condition-path injection；mid/late modulation 只作备选。
4. 把 predicted future latent 到 action bridge 的输出 contract 写清楚，但不强绑到正式训练主干。
5. 编写 history sampling 训练/推理一致性测试。
6. 编写 shuffled-robot sanity 计划或 smoke test，明确它依赖有效 E-004 checkpoint 后才进入真实指标验证。
7. 更新实现日志。

## 测试命令

```bash
pytest tests/mowa -q
```

## 验收标准

1. shape、token 数、LoRA rank、batch、显存未实测时标注 `TBD` / `Data Gate`。
2. future action leakage test 必须通过。
3. E-004/E-005 编号含义不变。
4. shuffled 后指标下降才支持 robot history 贡献，但这条只能在有有效 checkpoint 后判断，不能拿 plan/smoke 冒充。
5. gate 初始化必须接近 0，不应默认盖过 current/text condition。

## 输出格式

输出修改文件、HLC-GCI 接口、测试命令与结果、风险、下一步。

## 阶段边界

不得启动 Rec-HLC E-009，除非已有明确触发条件和人工确认；不得实现 P2。
不得绕过 E-003 前置，直接把 HLC-GCI 接入正式训练。
