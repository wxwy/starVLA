# MoWA M3 P1-b0 Latent Prior Prompt

## 当前阶段

M3：P1-b0 Latent-Only future prior。

## 目标

设计或实现 Wan latent cache builder 与 P1-b0 latent future prior 接口。P1-b0 只使用视觉/语言 future latent prior，不注入 robot history latent，不解码像素作为默认路径。

## 必读文档

1. `docs_zh/mowa/AGENTS.md`
2. `docs_zh/mowa/02_detailed_design.md` 第 6、8、9、10、11 章
3. `docs_zh/mowa/04_task_breakdown.md` 的 M3 任务卡
4. `docs_zh/mowa/05_experiment_registry.md`
5. M1 G0 latent cache smoke 输出
6. `docs_zh/mowa/07_implementation_log.md`

## 允许修改文件

- `tools/mowa/`
- `starVLA/model/modules/mowa/`
- `configs/mowa/mowa_latent_cache_builder_design.yaml`
- `configs/mowa/mowa_future_latent_prior_interface.yaml`
- `tests/mowa/`
- `docs_zh/mowa/07_implementation_log.md`

## 禁止修改文件

- P1-b1 HLC-GCI 默认主逻辑。
- P2 continuous video generation。
- Wan DiT full fine-tune 默认路线。
- future action label 输入 WAM。
- 三份核心 SOT。

## 具体任务

1. 设计 current/future/history RGB 的 latent cache smoke。
2. 设计 `MoWAP1B0FutureLatentPrior` 输入输出。
3. future latent 只能作为 target/cache，不得进入 WAM input 泄漏未来。
4. 输出可供 action bridge 使用的 latent future features。
5. 更新实现日志。

## 测试命令

```bash
pytest tests/mowa -q
```

## 验收标准

1. latent shape、cache hash、latency 未实测时为 `Data Gate` / `TBD`。
2. VAE/text encoder 默认冻结。
3. E-003 编号含义不变。
4. P1-b0 失败时能回退 P0。

## 输出格式

输出修改文件、latent cache 状态、接口摘要、测试命令与结果、下一步。

## 阶段边界

不得实现 HLC-GCI、Rec-HLC、P2 decoder 或 LayerwiseFM 内部改写。
