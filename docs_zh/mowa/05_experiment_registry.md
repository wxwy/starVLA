# MoWA 实验 Registry

本文件从 `02_detailed_design.md` 第 8 章和附录 D 抽取并固化执行侧实验表。编号含义不得改变；G0 是 Data Verification Gate，不训练，不计入算法实验数；E-008 与 E-010 不计入训练实验数；新增训练实验只能使用 E-011 至 E-020，禁止新增 E-021。

| 编号 | 类型 | 阶段 | 实验 / 门禁 | 是否训练 | 是否计入≤20 | 状态 | 决策作用 |
|---|---|---|---|---|---|---|---|
| G0 | Gate | 前置 | Data Verification Gate | 否 | 否 | Required | 决定哪些数据域可进入 P0/P1/P2；失败则降级数据域或阻断训练。 |
| E-001 | R1 | P0 | P0-FullHeads vs VLA baseline | 是 | 是 | Required | 验证 Video-Generation-Free WAM 是否具备 action-relevant future supervision 收益。 |
| E-002 | O4 | P0 | P0-GatedHeads vs P0-FullHeads | 是 | 是 | Optional | 验证 learnable gates 是否能替代 per-head sweep，作为 P0 可选增强。 |
| E-003 | R2 | P1-b0 | P1-b0 latent future prior vs P0 | 是 | 是 | Required | 判断 latent-only future prior 是否值得进入 P1-b1。 |
| E-004 | R3/O1 | P1-b1 | P1-b1-HLC-GCI vs P1-b0 | 是 | 是 | Required | 验证 history latent 压缩与 gated injection 是否带来额外收益。 |
| E-005 | R3 sanity | P1-b1 | P1-b1 shuffled-robot sanity | 是 | 是 | Required-light | 证明 robot history latent 的真实贡献，排查泄漏或无效 projector。 |
| E-006 | O2 | P0/P1 | WAM-to-action coupling | 是 | 是 | Optional | 验证 WAM features 是否真正进入 action path；失败则降级为诊断信号。 |
| E-007 | O3 | Data mix | proxy learned α vs uniform α | 是 | 是 | Optional | 小规模 P0 proxy 学数据域权重 α，正式 P0/P1 固定使用。 |
| E-008 | D1 | P2 | P2 frozen decoder diagnostic | 否 | 否 | Diagnostic | 解释 P1 predicted latent 与失败模式；不作为性能收益来源。 |
| E-009 | Conditional | P1-b2 | Rec-HLC conditional enhancement | 是 | 是，仅触发时 | Conditional | 仅当 short-window HLC 覆盖不足时验证固定长度 recurrent memory。 |
| E-010 | Eval | P0/P1 | multi-benchmark eval tracking | 否 | 否 | Eval-only | 同一 checkpoint 的跨域评测追踪，不新增训练实验。 |
| E-011 | UNALLOCATED | — | UNALLOCATED | — | — | UNALLOCATED | 预留槽位。 |
| E-012 | UNALLOCATED | — | UNALLOCATED | — | — | UNALLOCATED | 预留槽位。 |
| E-013 | UNALLOCATED | — | UNALLOCATED | — | — | UNALLOCATED | 预留槽位。 |
| E-014 | UNALLOCATED | — | UNALLOCATED | — | — | UNALLOCATED | 预留槽位。 |
| E-015 | UNALLOCATED | — | UNALLOCATED | — | — | UNALLOCATED | 预留槽位。 |
| E-016 | UNALLOCATED | — | UNALLOCATED | — | — | UNALLOCATED | 预留槽位。 |
| E-017 | UNALLOCATED | — | UNALLOCATED | — | — | UNALLOCATED | 预留槽位。 |
| E-018 | UNALLOCATED | — | UNALLOCATED | — | — | UNALLOCATED | 预留槽位。 |
| E-019 | UNALLOCATED | — | UNALLOCATED | — | — | UNALLOCATED | 预留槽位。 |
| E-020 | UNALLOCATED | — | UNALLOCATED | — | — | UNALLOCATED | 预留槽位。 |

## 预算状态

当前预分配训练实验为 E-001 至 E-007，E-009 为条件触发，最多 8 个训练实验；E-008、E-010 和 G0 不计入训练实验数。预算满足“计入预算的算法训练实验 ≤20”。

## 新增实验规则

1. 新增训练实验必须先占用 E-011 至 E-020 中的一个 UNALLOCATED 槽位。
2. 新增实验必须写明阶段、变量、对照组、数据域、是否训练、是否计入预算、成功标准和失败动作。
3. 不得新增 E-021 或更高编号。
4. 不得通过 benchmark eval、checkpoint eval 或诊断 decode 变相新增训练实验编号。
