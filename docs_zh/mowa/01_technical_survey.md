# 面向移动操作的 World Action Model 构建与评测研究
# 技术调研报告 v2.11（Rec-HLC 与 Episode 采样增强版）

> 文档类型：技术调研报告  
> 版本：v2.11-Rec-HLC 与 Episode 采样增强版  
> 适用对象：技术负责人、算法专家、机器人系统工程师、后续详细设计文档撰写者。  
> 本项目名称：**《面向移动操作的 World Action Model 构建与评测研究》**  
> 关联 VLA 策略项目：**《基于 Vision-Language Model 与 Flow Matching 的语言条件机器人操作策略研究》**。正文如需引用该工作，统一称为“VLA 策略项目”。

## 修订记录

| 版本 | 主要修改 |
|---|---|
| v1.1 | 在初始技术路线基础上细化 P0/P1/P2 分层，明确本项目不是普通 VLA 策略训练，而是面向移动操作的 WAM 构建与评测。 |
| v1.2 | 对齐数据集 / Benchmark v0.8 核对结果，重构第四章数据集分层、候选优先级与可用性判断。 |
| v1.3 | 新增多数据域混合训练与采样权重学习策略，提出按数据域建模、proxy WAM 学习采样权重、正式 P0/P1 固定权重的思路。 |
| v1.4 | 明确 P1 采用抽象阶段名 Latent-Only WAM，并将 Wan2.2 TI2V-5B[14] 定位为候选视频基础模型；视觉编码和语言编码优先沿用 Wan2.2-VAE[14] 与 Wan2.2 T5/UMT5[14] text encoder。 |
| v1.5 | 明确 P1-b 的 robot-history-conditioned 路线：P1-b0 为视觉/语言 future latent prior 基线，P1-b1 为历史 state/action 条件化 DiT-LoRA[14][25] 候选增强路线；强调 Wan2.2 是推荐候选实现而非阶段命名。 |
| v1.6 | 重写第五至第八章的调研叙事结构，补充每章目的、承接关系、研究发展、代表工作、优劣分析、路线决策和注意点。 |
| v1.7 | 补全文献引用和来源链，将第五至第八章中的重点参考工作与文末参考资料编号重新对应。 |
| v1.8 | 增加研究发展谱系与典型工作清单，区分“研究发展中的典型工作”和“本章重点参考工作”。 |
| v1.9 | 回到 v1.6 作为正文底稿，在不删减原有展开的前提下补充研究发展谱系，避免因结构调整压缩已有分析。 |
| v2.0 | 清理结构与术语：将原第四章 4.11 多数据域混合内容完整迁移并强化到第五章 5.8；第四章只保留数据集 / benchmark 筛选结论；统一正式称谓，本研究对象称为“本项目”，关联 VLA 工作称为“VLA 策略项目”。 |
| v2.1 | 补全文献引用覆盖：恢复并扩展参考资料清单至 70+ 条；正文中对研究发展、代表工作、重点参考工作、数据集 / benchmark、模型与方法名称补充编号标注，避免“提到工作但无来源”的问题。 |
| v2.2 | 术语解绑定与引用核验：恢复 P0/P1/P2 抽象阶段名，将 Qwen3VL[29]、Wan2.2 TI2V-5B[14] 等改为推荐候选实现；补充立项书遗留问题对照表、详细设计输入总表、章节客观结论/决策建议、P0 任务收益风险降级路径；将未核验或仅作线索的参考资料显式标注为“待核验/扩展资料”。 |
| v2.3 | 在不改动第五至第八章主体内容的前提下，补强第二、三、四、九、十、十一章的“发展背景—分类/阶段谱系—典型工作—重点参考—本项目决策”链条；明确第二章任务形态分析主要依据 WAM 综述[84]和移动操作 benchmark/真实系统来源；全篇扫描术语、章节衔接、引用标注和数据状态表述。 |
| v2.4 | 针对文献索引和链接进行专项核验：直接打开 arXiv / 官方仓库 / Hugging Face 等来源核验 WAM 综述[84]、Fast-WAM[1]、LaWAM[2]、Light-WAM[15]、AdaWAM[9]、OSCAR[18]、Efficient-WAM[26]、MotionWAM[27]、Cosmos 3[6]、UWM[8]、1X World Model Challenge[16]、Wan2.2[14] 等关键条目；补充 DreamZero[87]、Metis[85]、AHA-WAM[86] 的编号引用；将“核心/已核验”和“待核验/扩展资料”的标注收紧为可追溯来源状态。 |
| v2.5 | 根据外部校验意见补强精确化与执行核验内容：新增快速参考索引和术语对照表；补充 Wan2.2[14] 技术边界、LoRA[25] 注入原则、P0/P1/P2 阶段转换逻辑、Data Verification Gate 通过/不通过判定、P0 收益阈值、P1-b1 工程 invariant 检查、第一周核验交付物和 12 周里程碑；修正 [72]/[73] 同源引用说明、VLA-JEPA[3] GitHub 链接、Diffusion Policy[79] arXiv 链接、ManiSkill-HAB[47] 来源，并强化 MobileManiBench[33] coming soon 与 Kitchen-R[34] 未开源口径。 |
| v2.6 | 基于详细设计前讨论，新增 P1-b 历史 latent 压缩与注入调研：将历史 RGB/video、语言、机器人 state/action 统一表述为与 Wan2.2[14] DiT 条件空间对齐的 history latent / condition tokens；补充 learned-query compression、Perceiver Resampler[88]、adapter / gated injection、T2I-Adapter[89]、ControlNet[90]、memory-augmented history retrieval[91] 等典型工作；将 P1-b1 默认机制从“历史 state/action encoder”升级为本项目建议方案 History Latent Compressor + Gated Condition Injection，并明确 future action 防泄漏是工程 invariant，不进入实验矩阵。 |
| v2.7 | 明确 History Latent Compressor + Gated Condition Injection（HLC-GCI）不是已有论文的标准方法名，也不是对单一工作的直接复现，而是本项目基于 learned-query compression、adapter-based condition alignment、gated / zero-initialized condition injection、latent WAM / JEPA-style future prediction 等已有技术脉络组合形成的 P1-b 默认候选实现方案；要求正文区分“已有工作支撑的组件思想”与“本项目自定义组合方案”，避免将 HLC-GCI 误写为已有工作。 |
| v2.8 | 在 7.6 节新增“history latent 注入位置”调研，严格按技术调研报告口径组织：先分析条件注入位置的发展脉络与工程约束，再结合 StarVLA / WM4A / LayerwiseFM 的现有接口确认本项目路线。结论收敛为：HLC-GCI 输出 compact history condition tokens 与 pooled history embedding；Wan DiT 侧默认采用 condition-path injection，mid/late modulation 作为备选增强；action head 侧默认与 VLA 策略项目 StarFlow / LayerwiseFM 对齐，通过 layerwise WAM features 进入 action DiT，暂不改变 action head 内部层；若 downstream gain 不足，再增加一个 WAM global token，layerwise action modulation 暂缓。 |
| v2.9 | 对 v2.8 中过度详细的接口变量、公式、action head 内部拼接和具体实验编号进行上浮 / 下沉整理：技术调研报告保留技术脉络、代表工作、路线对比、工程约束和收敛判断；将 `C_hist/h_hist/g_hist`、`C_wan_cond` 公式、`sa_embs` 拼接、action encoder/state encoder/future tokens 保持不变等实现级细节下沉到详细设计文档 Prompt。7.6.6 改写为调研报告风格的“history latent 注入位置路线分析”，7.6.7 改写为调研结论收敛，7.6.8 改写为面向详细设计的输入边界。 |
| v2.10 | 新增“历史压缩时间尺度与记忆更新策略”调研，补充 sliding window、event-aware window、recurrent / compressive memory、retrieval memory、state-space / Mamba-style streaming memory 等路线对比；结合 Transformer-XL、Compressive Transformer、RMT、TokenLearner、Mamba、AEM、Mem-World 等参考工作，收敛为本项目默认采用“短中期滑动窗口 + learned-query compression + 可选事件关键帧 / 低频摘要”的保守路线。不默认采用“旧压缩记忆 + 新历史递归压缩以保留全部历史信息”，该路线作为长期 memory 扩展或条件触发增强下沉到详细设计。 |
| v2.11 | 补充 episode-to-window 训练采样、拟采用数据集 episode length / Hz profiling、SparseVideoNav 显存 profile baseline 与 Rec-HLC 条件增强路线：明确完整 episode 是采样容器而非默认模型输入；P1-b 默认从 episode 中采样 anchor timestep，切出 recent history window、future latent target 与 action chunk target；SparseVideoNav 作为工程上界参考（Wan2.1 T2V-1.3B full fine-tune、256×256、4Hz、16 history、28 future、chunk=4、direct concat、T5 offload、per-device batch size=2、约70G）；Rec-HLC 可用固定长度 memory 覆盖 episode 前缀，但不是无损全历史输入，默认作为 P1-b2 / Conditional 而非 P1-b1 主线。 |

---

## 引用标注与核验说明

本文采用正文编号引用方式。文献与链接核验状态截至 **2026-06-26**。研究发展谱系中的典型工作、从中筛选出的重点参考工作、数据集 / benchmark、模型、方法和评测框架，均应在首次或关键出现处标注文献编号；同一工作在同一段或同一表格中重复出现时，不机械重复标注。

本版对参考资料采用分层使用口径：

1. **核心引用 / 已核验来源**：优先使用 arXiv 原文、官方 GitHub、官方项目页、Hugging Face 页面或 benchmark 官方文档。这类来源可直接支撑正文中的路线判断。
2. **扩展资料 / 待核验候选**：对尚未完成逐条复核、仅有二手材料、仅论文未开放代码/数据、或开放状态随时间变化较快的条目，文末显式标注“待核验”或在正文中标注“开放后优先 / 仅作线索”。这类条目不作为当前详细设计的硬依赖。
3. **内部核对材料**：用户核对版 v0.8 候选表作为本项目内部数据集状态依据，用于记录可用性、人工备注、S/V/M 标记和下一步核验队列；其结论仍需在详细设计前通过 Data Verification Gate 复核。

本版修订重点不是增加引用数量，而是降低“错误编号 / 幽灵引用 / 候选资源被误写成已可用”的风险。后续若进入正式对外版本，应继续对文末待核验条目逐条打开论文原文或官方仓库，确认标题、编号、开放状态、许可证与数据字段。

---

## 快速参考索引

| 阶段 / 模块 | 核心问题 | 推荐路线 | 关键数据资源 | 三个月可行性 | 首要风险 |
|---|---|---|---|---|---|
| P0：Video-Generation-Free WAM | 不生成像素，如何验证 future supervision 对策略有帮助 | 轻量 WAM-specific heads；候选实现可复用 VLA 策略项目中的视觉语言 backbone，但 WAM 分支保持独立 | RoboCasa[32] / RoboCasa365[19]、AIRoA MoMa[36]、EBench[35]、MoMa-Kitchen[39] | 高 | WAM-specific 指标提升但 downstream task gain 不明显 |
| P1-b0：future latent prior 基线 | 只用视觉/语言预测 future latent 是否有下游价值 | Wan2.2[14] 体系内视觉 latent 与文本条件，默认不注入机器人历史条件 | RoboCasa[32] / RoboCasa365[19]、DROID[22]、BridgeData[60] | 中 | WAM 味道偏弱，可能只是 latent visual subgoal |
| P1-b1：robot-history-conditioned Latent-Only WAM | 多模态历史 latent 是否能让 future latent 更 action-relevant | Wan2.2[14] DiT-LoRA[25] + 本项目建议方案 HLC-GCI；历史 RGB/video、语言、机器人 state/action 均转为可注入 DiT 的 history condition 表征；Wan DiT 侧优先采用 condition-path 注入，action head 侧优先对齐 VLA 策略项目 StarFlow / LayerwiseFM；future action 防泄漏通过工程 invariant 保证；HLC-GCI 不是已有方法名，而是本项目组合方案；具体接口变量与注入公式下沉到详细设计文档 | AIRoA MoMa[36]、RoboCasa[32] / RoboCasa365[19] | 中 | history latent token 压缩、条件空间对齐、注入稳定性与推理延迟 |
| P2：Render-and-Decode WAM | 何时解码像素用于诊断和展示 | 仅触发式 decode P1 predicted latent，生成 keyframe / short clip | RoboCasa[32] / RoboCasa365[19] 局部样本、Isaac 系候选 | 低-中 | 滑向 continuous video generation 主线 |

## 术语对照表

| 术语 | 正式定义 | 禁止的误用 |
|---|---|---|
| WAM | 学习 action-relevant future representation，且该表征能够被下游策略、诊断或评测消费 | 不等同于普通 VLA action head，也不等同于通用视频生成模型 |
| P0：Video-Generation-Free WAM | 推理时不生成像素，只通过轻量 future representation / auxiliary heads 引入未来监督 | 不写成 “Qwen3VL-based” 的阶段名，不写成 VLA action head 改装 |
| P1：Latent-Only WAM | 在 latent space 建模未来，不默认解码像素 | 不写成 “Wan2.2 TI2V-5B-derived” 的阶段名；Wan2.2[14] 只是当前推荐候选底座 |
| P1-b0 | 视觉/语言 future latent prior 基线 | 不等同于完整 robot-history-conditioned WAM |
| P1-b1 | 将历史 RGB/video latent、语言 condition、历史 robot state/action condition tokens 压缩并注入 future latent prediction | 不写成只注入低维 state/action；禁止把数据集中的 future action label 输入 WAM；future action 防泄漏属于工程 invariant |
| P2：Render-and-Decode WAM | 解码 P1 predicted latent，用于可解释性、误差诊断和局部 demo | 不作为 continuous video generation 主线 |
| VLA 策略项目 | 关联项目《基于 Vision-Language Model 与 Flow Matching 的语言条件机器人操作策略研究》 | 可以复用数据管线、schema 和部分 backbone 经验，但不能把 WAM 研究目标写成 VLA 本体优化 |

---

## 执行摘要

本报告围绕“面向移动操作的 World Action Model（WAM）构建与评测研究”展开技术调研。与立项书不同，本报告不再重复“为什么做 WAM”，而是回答“当前有哪些可参考路线、如何实现、效果如何、优劣是什么、如何进入详细设计”。本项目研究对象不是 VLA 本体，而是面向 long-horizon household mobile manipulation 的 action-relevant future 建模、接入与评测系统。VLA、导航器、规划器、操作 primitive 或执行器只作为 WAM 输出的下游消费方。

术语上，本文中的“本项目”均指《面向移动操作的 World Action Model 构建与评测研究》；“VLA 策略项目”指《基于 Vision-Language Model 与 Flow Matching 的语言条件机器人操作策略研究》。正文中不再使用临时编号称谓，以避免正式报告中的指代不清。

本版在 v1.1 的技术路线基础上，完整吸收《本项目 WAM 数据集 / Benchmark 全量候选表 v0.8》的核对结果。v0.8 的关键价值在于新增“可用性 / 核对状态”和“人工核对备注”，将 MobileManiBench[33] coming soon、Kitchen-R[34] 未开源、HomeRobot/OVMM[7][20][21] 抓取吸附、AIRoA MoMa[36] 偏短程任务、SAGE-3D[11] 非标准 benchmark、MoMa-Kitchen[39] 视觉真实感弱等边界显式结构化，从而避免把论文强、视觉强或名称前沿的候选错误写成三个月内可落地主闭环。

本项目路线仍保持三阶段：

- **P0：Video-Generation-Free WAM**  
  阶段目标是在不生成未来视频的前提下，引入 future auxiliary learning、task progress、manipulation-readiness、failure-risk、next-best-view 或 subgoal feasibility 等 action-relevant future 监督。Qwen3VL[29] / Qwen-VL[30] 系视觉语言主干可作为当前优先候选实现，主要来自 VLA 策略项目的工程复用价值，但 P0 的正式定义不绑定具体 backbone。

- **P1：Latent-Only WAM**  
  阶段目标是在不解码像素的前提下学习 future latent representation。Wan2.2 TI2V-5B[14] 是当前优先候选视频基础模型：视觉侧可用 Wan2.2-VAE encoder 将当前/未来片段编码为 video latent，语言侧可用 Wan2.2[14] 自带 T5/UMT5 text encoder 形成文本条件。P1-b0 作为低风险基线，不修改候选视频模型生成流程，只用视觉/语言预测 future latent；P1-b1 作为推荐增强路线，继承候选视频 DiT 的 latent dynamics prior，通过 LoRA[25]/adapter 注入多模态 history latent：历史 RGB/video 经 Wan2.2-VAE[14] 编码为 visual history latent，语言经 Wan2.2[14] text encoder 形成 text condition，机器人 state/action 经 trainable adapter 映射到 DiT condition space，再经本项目建议方案 History Latent Compressor + Gated Condition Injection（HLC-GCI）形成 robot-history-conditioned latent WAM。HLC-GCI 不是已有论文的标准方法名，而是基于已有组件思想形成的项目内候选实现方案。P1 不默认使用 Qwen-VL[30]，不输入未来 action label，推理阶段不解码像素。

- **P2：Render-and-Decode WAM**  
  阶段目标是对 P1 predicted latent 做局部 keyframe / short-clip 解码，用于可解释性、误差诊断和局部 demo。P2 不作为三个月完整视频生成主线，也不默认运行 continuous video rollout。

本版更新后的数据集与 benchmark 结论如下：

1. **近期主闭环优先**：RoboCasa[32] / RoboCasa365[19] 是最现实的第一闭环入口，工程生态成熟，厨房任务资产强；但必须继续核实 mobile base / navigation-manipulation 的真实性，不能无条件写成完整移动操作主 benchmark。
2. **真实移动操作必须保留**：AIRoA MoMa[36] 是真实移动操作、多模态、层级标注的 P0/P1 核心候选，需重点核验数据开放、schema、Base/EEF/Success/Lang 与任务长度。
3. **Isaac Sim 系上调但不能高估**：MobileManiBench[33]、Kitchen-R[34]、EBench[35] 贴近移动操作和高真实感仿真。MobileManiBench[33] 目前代码/数据 coming soon，Kitchen-R[34] 未找到代码/数据，不能作为近期已可用主闭环；EBench[35] 可作为 VLA/mobile manipulation 诊断候选。
4. **MoMa-Kitchen[39] 明确降级**：MoMa-Kitchen[39] 基于 BestMan/PyBullet，视觉真实感较弱，适合作 P0 final-pose affordance / manipulation-readiness 诊断集，不作主视觉 benchmark。
5. **高难验证后置**：HomeRobot/OVMM[7][20][21]、LAMBDA[38]、BEHAVIOR-1K[24]/OmniGibson[45]、Habitat[42][43][44] 2.0/HAB 价值高，但复现难度、在线评测属性、抓取机制或工程复杂度使其更适合作后期外推和长期验证。
6. **非移动操作辅助集不能越位**：LIBERO[48]、RoboTwin[49]、DROID[22]、Open X-Embodiment[23]、BridgeData[60]、RLBench[50]、CALVIN[51] 等适合快速消融、操作泛化、表征学习和数据工程参考，但不能作为移动操作主证据。

7. **多数据域混合策略**：本项目不建议按数据量比例采样，也不建议粗暴划分为 mobile-only / manipulation-only 两类数据。更合理的做法是将 RoboCasa[32]/RoboCasa365[19]、AIRoA MoMa[36]、EBench[35] mobile、MoMa-Kitchen[39] 以及一个高质量纯操作数据集分别视为独立数据域，借鉴 Re-Mix[12] / DRO 类思想，用小规模 P0 proxy WAM 估计域贡献并学习采样权重 α，正式 P0/P1 训练阶段固定使用同一组 α。该机制定位为数据混合策略，不作为新的核心算法路线。

最终，本报告建议采用“**近期可落地闭环 + 真实移动操作核验 + P0 诊断集 + 高难后期验证 + 操作辅助集 + world model evaluation 方法参考**”六层数据策略。三个月内最优先验证的 8–12 个候选为：RoboCasa365[19]、RoboCasa[32]、AIRoA MoMa[36]、EBench[35]、ManiSkill3[40]、MoMa-Kitchen[39]、LIBERO[48]/LIBERO-Long[48]、RoboTwin 2.0[49]、MobileManiBench[33]（开放后）、Kitchen-R[34]（开放后）、HomeRobot/OVMM[7][20][21]、BEHAVIOR-1K[24]/OmniGibson[45]。


阶段命名上，本报告恢复抽象术语：**P0 = Video-Generation-Free WAM，P1 = Latent-Only WAM，P2 = Render-and-Decode WAM**。具体模型只出现在“候选实现 / 推荐路线”层面：P0 可优先复用 VLA 策略项目中的视觉语言编码、数据处理和状态 / 动作 schema，但 P0 的研究对象是独立 WAM 分支而非 VLA action head；P1 可优先评估 Wan2.2 TI2V-5B[14] 作为视频 latent foundation candidate，但 Wan2.2[14] 不是阶段名称，也不是不可替换底座。

同时，本版强化一个关键风险判断：P0/P1 的 success 不能只看 WAM-specific loss 或 latent prediction 变好，必须证明其能被下游消费并带来任务级收益。如果 future representation 无法提升 sub-stage success、navigation-to-manipulation handoff、recovery success 或 overall success rate，则应将 WAM 输出降级为诊断信号，推迟 P1/P2 扩展。

# 第一章 调研目标、范围与方法

## 1.1 调研目标

本报告回答以下问题：

1. 移动操作场景下，什么样的 future representation 对任务有用？
2. P0 能否基于 VLA 策略项目的 Qwen3VL-VLA 经验改造，而不是从零做一个 WAM？
3. P1 是否应该使用 Wan 系列[5] / Wan2.2[14] / video diffusion 等本来能生成像素的模型，但只取 latent 或 hidden feature？
4. Cosmos[6]应该放在 P1 还是 P2？
5. 当前 benchmark 和数据集哪些真正支持移动操作，而不是固定桌面操作？
6. 哪些方法进入详细设计，哪些只作为备选或长期路线？

## 1.2 立项书遗留问题对照表

本报告承接立项书中的开放问题，不重新讨论“为什么做本项目”，而是把立项阶段尚未冻结的技术路线、数据资源、评测方法和工程边界转化为可进入详细设计的输入。下表用于明确“立项书问题—调研章节—当前状态”的闭环关系。

| 序号 | 立项书遗留问题 | 对应立项书主题 | 本报告负责章节 | 当前状态 | 进入详细设计方式 |
|---|---|---|---|---|---|
| 1 | P0 是否必须生成未来像素？是否存在低成本 WAM 路线？ | P0 Video-Generation-Free WAM | 第六章、第十章 | 已有明确结论：P0 不生成像素，采用 future supervision / lightweight heads / WAM-specific metrics | 进入 P0 网络结构、标签构造、loss 与下游接入设计 |
| 2 | P1 是否应使用 video latent？是否从零训练 WAM？ | P1 Latent-Only WAM | 第七章 | 已有明确结论：P1 使用 latent-only；Wan2.2[14] 作为候选视频基础模型；P1-b0/b1 分级推进 | 进入 latent extraction、history condition、LoRA/adapters、缓存策略设计 |
| 3 | P2 是否作为三个月主线？ | P2 Render-and-Decode WAM | 第八章 | 已有明确结论：P2 仅作 decode P1 predicted latent 的局部诊断 / demo，不进入主训练闭环 | 进入 triggered keyframe / short-clip demo 设计 |
| 4 | 哪些 benchmark 可作为移动操作主证据？ | 数据集与 benchmark 筛选 | 第四章、第九章、附录 A | 已有初步结论：RoboCasa/RoboCasa365[19][32] 先作 household manipulation 强闭环底座，真实移动操作需 AIRoA MoMa[36] 等复核；MobileManiBench[33]/Kitchen-R[34] 开放后优先 | 进入 Data Verification Gate，不通过 gate 不写入主实验矩阵 |
| 5 | 多数据域如何混合，是否需要复杂 DRO？ | 数据工程与 scaling | 第五章 5.8 | 已有明确结论：proxy WAM 学 α，正式 P0/P1 固定 α；不把完整 Re-Mix/DRO 复现作为主贡献 | 进入数据采样器、domain schema 与 proxy 指标设计 |
| 6 | 如何证明 WAM 对任务有用？ | 评测体系与下游接入 | 第十章、第十一章、第十二章 | 已有框架，需在详细设计中细化指标、消融和相关性分析 | 进入 WAM-specific、downstream、coupling、cost 四类指标设计 |
| 7 | 如何控制三个月单人执行风险？ | 风险与执行计划 | 第十三章、第十四章 | 已有 Data Verification Gate 和降级路径；本版新增 P0 收益不明显风险 | 进入里程碑、验收标准和停/转向条件设计 |

## 1.3 调研范围

调研范围包括：

- WAM 与 VLA / world model / video generation 的边界；
- Fast-WAM[1]、LaWAM[2]、VLA-JEPA[3]、UWM[8]、AdaWAM[9]、Qwen-RobotManip[4]、Wan 系列[5]、Cosmos[6]等典型路线，并参考 Awesome-WAM 阅读清单[10]补充候选谱系；
- HomeRobot / OVMM[7][20][21]、MobileManiBench[33]、LIBERO[48]、RoboTwin[49]、RoboCasa[32]、BEHAVIOR[24] / OmniGibson[45]、ManiSkill[40][41] 等 benchmark 与数据集；
- P0/P1/P2 的工程可行性、训练目标、下游接入和评测指标；
- 单人三个月、最高 8 卡 A100 条件下的实施约束。

## 1.4 证据等级

| 证据等级 | 含义 | 使用方式 |
|---|---|---|
| A | 论文原文 + 官方项目 / 代码 / benchmark 支持 | 可作为推荐主依据 |
| B | 论文原文或官方技术报告支持，但复现尚需验证 | 可作为重要候选依据 |
| C | 项目主页 / README / 二级综述支持 | 作为调研线索 |
| D | 社交平台 / 媒体描述 / 未核实数字 | 不能作为定论 |
| E | 无来源 | 不使用 |

报告中涉及 2025–2026 年论文、benchmark、开源状态和性能结论，均应以论文或官方资料为准。

## 1.5 推荐等级

| 等级 | 含义 |
|---|---|
| 推荐 | 进入详细设计主线 |
| 备选 | 可在主线失败或资源充足时推进 |
| 暂缓 | 长期方向，不进入三个月主线 |
| 不推荐 | 与项目边界或资源约束不匹配 |

---

# 第二章 移动操作任务需求与系统挑战调研

## 2.1 本章目的、作用与来源说明

本章回答“本项目为什么必须面向移动操作，而不能只沿用固定桌面操作 benchmark”的问题。它在全文中承担需求约束作用：第二章定义移动操作任务链和失败模式，第三章据此界定 WAM 与 VLA 的差异，第四章到第五章再把这些需求转化为数据集筛选和数据工程要求。

本章的研究背景与任务链划分主要参考 WAM 综述《World Action Models: A Survey: Dream Less, Act More》[84] 对 WAM、VLA、video-generation-free / latent-only / render-and-decode 路线边界的总结，并结合 HomeRobot/OVMM[7][20][21]、Mobile ALOHA[37]、AIRoA MoMa[36]、LAMBDA[38]、BEHAVIOR-1K[24]、MoMa-Kitchen[39] 等移动操作或 household benchmark 的任务设置进行归纳。这里的典型工作用于说明移动操作任务形态如何演化；真正进入本项目路线决策的数据资源和 benchmark，会在第四章、第九章和第十二章中进一步筛选。

## 2.2 移动操作任务形态的发展背景与典型工作

从已有研究和 benchmark 看，机器人操作任务大致经历了从“固定桌面操作”到“家庭环境重排”，再到“真实移动操作系统”和“WAM 驱动的未来状态建模”的扩展。不同阶段的典型工作如下：

| 发展阶段 | 典型工作 / 资源 | 主要任务形态 | 对本项目的启发 |
|---|---|---|---|
| 固定桌面语言条件操作 | LIBERO[48]、RLBench[50]、CALVIN[51]、RoboTwin[49] | 固定机械臂或局部桌面任务，语言条件 pick/place/open/close 等 | 可用于操作端 sanity check、长时序操作和语言泛化，但不能证明移动操作 WAM 能力 |
| 大规模操作数据与 VLA 训练 | DROID[22]、Open X-Embodiment[23]、BridgeData V2[60]、Qwen-RobotManip[4] | 多机器人、多任务、多场景操作数据或 VLA 对齐流程 | 提供数据清洗、action-visual 对齐和 scaling recipe 参考，但移动底盘与导航—操作交接通常不足 |
| household rearrangement / 高复杂仿真 | Habitat 2.0 / Rearrangement[42][43]、HomeRobot/OVMM[7][20][21]、BEHAVIOR-1K[24]、OmniGibson[45] | 家庭环境中的搜索、抓取、重排、放置和物体状态变化 | 体现移动操作的长链条、高真实感和任务复杂度，但工程成本较高，适合后期验证 |
| 真实移动操作系统与数据 | Mobile ALOHA[37]、AIRoA MoMa[36]、LAMBDA[38]、SHOPPER 系真实系统 | 移动底盘 + 机械臂 + 多相机 / 真实传感器 / 长时序任务 | 提供真实失败模式、真实传感器噪声和 whole-body mobile manipulation 形态，是本项目必须保留的核验方向 |
| last-mile / readiness / 诊断型 benchmark | MoMa-Kitchen[39]、EBench[35]、MobileManiBench[33]、Kitchen-R[34] | 最后接近、可操作位置、视角调整、移动操作五轴诊断 | 更直接对应 P0 的 readiness、failure-risk、next-best-view 和 navigation-to-manipulation handoff 指标 |
| WAM / future-aware 方法 | Fast-WAM[1]、LaWAM[2]、VLA-JEPA[3]、Light-WAM[15]、ChronoDreamer[17]、OSCAR[18]、WAM 综述[84] | 通过未来预测、latent future、history/action conditioning 或 video-free reasoning 辅助动作决策 | 说明本项目不应只追求动作输出，而要研究可被动作策略消费的 future representation |

这一发展脉络说明：移动操作的核心难点不是“把桌面操作数据放大”，而是任务链从局部操作扩展到目标搜索、接近、可操作视角选择、操作、搬运、放置和失败恢复。WAM 的价值也应围绕这些阶段是否能被更早预测、更稳定接入和更有效评测来判断。

## 2.3 移动操作与固定桌面操作的差异

固定桌面操作通常假设机器人基座固定、目标区域有限、视角变化小、任务长度短。移动操作则要求机器人在家庭环境中完成“找物—接近—调整视角—处理遮挡—拿取—移动—放置”的长链条。它不仅需要操作策略，还需要导航、场景理解、目标搜索、视角选择、失败恢复和状态预测。

因此，本项目的主证据不能只来自 LIBERO[48]、RoboTwin[49] 这类偏桌面或操作泛化的 benchmark。它们适合验证 P0/P1 的操作端收益，但不足以证明移动操作 WAM 的完整价值。主 benchmark 必须包含导航—操作衔接、目标不可见或初始远距离、目标接近、可操作视角选择、拿取和放置等阶段。

## 2.4 移动操作失败模式

| 阶段 | 失败模式 | WAM 需求 |
|---|---|---|
| 找物 | 目标不可见、类别识别不稳、重复搜索 | 预测目标可见性、next-best-view、搜索进度 |
| 接近 | 走到不可操作位置、目标被遮挡 | manipulation-readiness、可操作视角预测 |
| 视角调整 | 相机姿态变化导致 VLA 不稳定 | future observation / latent dynamics |
| 拿取 | 遮挡、抓取姿态错误、夹爪接触失败 | failure-risk、action-conditioned future state |
| 移动 | 持物移动导致目标丢失或碰撞 | object-state consistency、trajectory risk |
| 放置 | receptacle 识别错误、放置位置不可行 | subgoal feasibility、place-readiness |

## 2.5 重点参考工作与本章决策依据

本章的重点参考不是所有典型工作，而是对本项目需求定义最有直接约束的几类来源：WAM 综述[84]用于界定 WAM 不等于视频生成或 VLA action head；HomeRobot/OVMM[7][20][21]、BEHAVIOR-1K[24]和 Habitat Rearrangement[42][43]说明 household mobile manipulation 的任务链复杂度；Mobile ALOHA[37]、AIRoA MoMa[36]和 LAMBDA[38]提供真实移动操作形态和真实失败模式；MoMa-Kitchen[39]、EBench[35]、MobileManiBench[33]、Kitchen-R[34]则更直接支撑 readiness、mobility、horizon 和 navigation-to-manipulation handoff 等诊断指标。

这些来源共同支撑本章决策：本项目不能以固定桌面成功率作为主证据，必须把 WAM 输出绑定到移动操作任务链中的 future-related 判断。

## 2.6 WAM 对移动操作的价值

WAM 的价值不在于“生成漂亮未来视频”，而在于为任务推进提供 action-relevant future。例如：

- 当前视角是否足以操作目标；
- 执行动作后物体是否会进入可抓取状态；
- 是否需要先调整视角；
- 是否存在高失败风险；
- 候选子目标是否更接近任务完成；
- 当前阶段是否应该从导航切换到操作。

这些信息可以以 P0 的轻量标签形式出现，也可以以 P1 的 latent future substrate 形式出现，还可以在 P2 中以可见未来帧形式出现。

---

## 2.7 本章客观调研结论

移动操作与固定桌面操作的核心差异不只是机器人是否具备移动底盘，而是任务链从局部操作扩展为“找物—接近—视角调整—可操作状态判断—操作—搬运/放置—失败恢复”的长时序闭环。移动过程会引入视角变化、遮挡、导航—操作交接、接触前姿态不确定和失败恢复等问题，这些问题难以由普通 VLA action head 直接覆盖。

## 2.8 对本项目的决策建议

本项目的 WAM 输出必须服务于移动操作特有的 future-related 判断：目标可见性、manipulation-readiness、failure-risk、task progress、next-best-view、subgoal feasibility 和 action outcome。后续 P0/P1/P2 的指标与数据筛选都应围绕这些问题展开，而不是仅以固定桌面成功率作为主证据。

# 第三章 WAM 概念、技术分类与评价维度

## 3.1 本章目的、作用与承接关系

第二章给出了移动操作任务链和失败模式，本章进一步回答“什么样的模型才算本项目意义上的 WAM”。本章不是直接选择具体模型，而是建立概念边界、技术分类和评价维度，为第六章 P0、第七章 P1、第八章 P2 的路线调研提供统一语言。

本章的分类框架主要参考 WAM 综述[84]关于 rendered future、latent future、video-generation-free action reasoning 以及 predictive substrate、action coupling、deployment regime 等维度的划分，并结合 Fast-WAM[1]、LaWAM[2]、VLA-JEPA[3]、Light-WAM[15]、UWM[8]、ChronoDreamer[17]、OSCAR[18]等典型工作进行本项目化整理。

## 3.2 WAM 研究发展谱系与典型工作

从研究发展看，WAM 并不是单一架构，而是从 VLA、world model、video generation model 和 action-conditioned prediction 多条路线汇合而来。典型谱系如下：

| 谱系 | 典型工作 | 主要特点 | 对本项目分类的影响 |
|---|---|---|---|
| VLA / action policy | RT-2[78]、OpenVLA[77]、Qwen-RobotManip[4]、StarVLA[31] | 从视觉、语言、状态直接预测动作，重点是动作策略泛化 | 提供下游消费方和工程复用基础，但不能直接等同 WAM |
| Video-generation-free future supervision | Fast-WAM[1]、Light-WAM[15]、VLA-JEPA[3] | 训练时利用 future supervision，推理时不生成视频 | 对应本项目 P0：Video-Generation-Free WAM |
| Latent-only / feature substrate | LaWAM[2]、VLA-JEPA[3]、V-JEPA 类方法、Wan latent substrate[5][14] | 不解码像素，用 latent / feature 形式承载未来信息 | 对应本项目 P1：Latent-Only WAM |
| Video foundation / render-and-decode | Wan 系列[5]、Wan2.2[14]、Cosmos[6]、UWM[8] | 利用视频生成模型、DiT 或 diffusion prior 表征未来 | 为 P1 提供候选 latent foundation，为 P2 提供局部解码能力 |
| History/action-conditioned world model | ChronoDreamer[17]、1X World Model Challenge[16]、OSCAR[18] | 将历史 state/action 或动作条件注入未来预测 | 支撑 P1-b1 robot-history-conditioned latent WAM 的合理性 |
| Adaptive / omnimodal WAM | AdaWAM[9]、Cosmos[6]、Metis[85]、AHA-WAM[86] | 按需触发、联合多模态或专家模型 | 长期参考，不作为三个月第一闭环 |

因此，本项目采用 P0/P1/P2 三阶段不是任意设计，而是对当前 WAM 发展路线的工程化归纳：先用 video-generation-free 信号验证收益，再引入 latent substrate，最后仅在需要诊断时解码可见未来。

## 3.3 WAM 与 VLA 的区别

VLA 通常学习从当前观测和语言指令到动作的映射。WAM 则要求显式或隐式建模“动作会如何改变未来世界”，并让动作生成与未来预测对齐。若一个模型只输出动作，而没有任何 future representation、world representation supervision 或 future-conditioned 接入，它不能被称为 WAM。

因此，VLA 策略项目的 Qwen3VL-VLA 不是 WAM。但在它上面加入 future auxiliary learning、task-progress prediction、failure-risk prediction 或 latent future prediction 后，可以构成本项目的 P0/P1 候选。这里的“构成候选”指 WAM 分支学习 future representation 并被下游消费，而不是把 VLA action head 本身改名为 WAM。

## 3.4 三类路线定义

| 阶段 | 核心问题 | 是否生成像素 | 代表思路 |
|---|---|---|---|
| P0 | 不生成未来，如何让模型具备 action-relevant future awareness | 否 | future auxiliary learning；Fast-WAM-style training[1]；Light-WAM-style video-free / low-cost future supervision[15] |
| P1 | 不生成像素，如何用 latent future substrate 提升策略 | 否 | Wan 系列[5] / Wan2.2[14] latent、LaWAM[2]、VLA-JEPA[3]、history-conditioned latent WAM[16][17] |
| P2 | 什么时候值得生成可见未来 | 是 | keyframe、route video、局部 Render-and-Decode、video foundation world model[5][6][8][14] |

## 3.5 值得参考的工作与概念边界

本章从研究发展中的典型工作里进一步筛选出三类对本项目最关键的参考来源：第一类是 WAM 综述[84]，用于冻结概念边界，避免将普通 VLA、通用视频生成模型或宽泛 world model 直接写成 WAM；第二类是 Fast-WAM[1]、VLA-JEPA[3]、Light-WAM[15]和 LaWAM[2]，用于支撑 P0/P1 的 low-cost future supervision 和 latent future 路线；第三类是 ChronoDreamer[17]、1X World Model Challenge[16]和 OSCAR[18]，用于说明 history/action-conditioned future prediction 的价值和工程风险。

据此，本项目的 WAM 定义应强调两个条件：一是模型必须学习某种 action-relevant future representation；二是该 representation 必须能被下游策略、诊断器、风险预测器或评测体系消费。

## 3.6 统一评价维度

| 维度 | 说明 |
|---|---|
| future representation | 任务进度、风险、latent、keyframe、video 等 |
| 是否 test-time video-free | 推理时是否生成未来视觉 |
| 数据需求 | 是否需要视频、动作、状态、语言、camera 参数 |
| 下游接入 | 作为条件、评分器、风险预测器、子目标 |
| 延迟与算力 | 是否能实时或近实时 |
| 三个月可行性 | 单人 + 最高 8 卡 A100 |
| 与VLA 策略项目复用 | 是否复用 Qwen3VL-VLA 代码 |
| 移动操作适配 | 是否支持导航—操作衔接 |
| 可诊断性 | 是否能定位失败原因 |
| 推荐等级 | 推荐 / 备选 / 暂缓 / 不推荐 |

---

## 3.7 本章客观调研结论

WAM 与普通 VLA 的区别在于是否显式学习 action-relevant future representation。仅从视觉、语言和状态直接输出动作，即使使用大型 VLM 或先进 action head，也仍属于 VLA；只有当模型学习了可被下游消费的 future state、future latent、risk、progress 或 action outcome，才构成本项目意义上的 WAM 分支。

## 3.8 对本项目的决策建议

正式术语应冻结为：P0 = Video-Generation-Free WAM，P1 = Latent-Only WAM，P2 = Render-and-Decode WAM。Qwen3VL[29]、Wan2.2 TI2V-5B[14]、Cosmos[6]等只能作为候选实现或参考路线，不能写入阶段正式命名。详细设计阶段必须继续保持 WAM 与 VLA 策略项目的边界：工程资产可复用，研究对象必须独立。

# 第四章 移动操作数据集、Benchmark 与 Leaderboard 调研

## 4.1 本章目标与 v0.8 核对结论

本章根据《本项目 WAM 数据集 / Benchmark 全量候选表 v0.8》重新梳理数据集与 benchmark 选择。v0.8 的关键贡献不是新增若干名字，而是将候选资源统一到“可用性 / 核对状态、人工核对备注、空间等级、视觉等级、生态成熟度、模态、任务规模、本项目定位”的筛选框架中。由此，本项目的数据集选择不再以“论文是否前沿”作为唯一标准，而是以“能否支撑 P0/P1/P2 可复现实验闭环”为主线。

本章采用以下原则：

1. **主 benchmark 必须服务移动操作**：优先考虑 S2–S4 空间，即单房间移动操作、多区域单房间和跨房间同楼层任务；S0 固定桌面任务不能作为移动操作主证据。
2. **近期闭环优先可用性**：论文强但代码/数据 coming soon 的资源，只能进入高价值候选或待核验队列，不能写成三个月内主闭环。
3. **诊断集与主 benchmark 分离**：MoMa-Kitchen[39]、EBench[35]、Habitat[42][43][44] Rearrangement、ManiSkill-HAB[47]等可用于 readiness、failure-risk、task-progress、affordance 等诊断，不等于主 benchmark。
4. **真实数据单独保留**：AIRoA MoMa[36]、Mobile ALOHA[37] 等真实移动操作数据要保留为 P0/P1 核心参考，但需注意任务长度、模态完整性与开放状态。
5. **操作辅助集不能越位**：LIBERO[48]、RoboTwin[49]、RoboCasa[32]、DROID[22]、Open X-Embodiment[23] 等对操作泛化、表征学习和快速消融非常重要，但不能简单当作移动操作主证据。

## 4.2 Benchmark 研究发展与典型资源谱系

移动操作相关 benchmark 的发展可以分为六类。这里列出的典型资源用于说明数据生态的覆盖面；真正支撑本章最终决策的资源，会在 4.3 进一步筛选。

| 类别 | 典型资源 | 主要贡献 | 对本项目的限制 |
|---|---|---|---|
| 固定桌面 / 局部操作 benchmark | LIBERO[48]、RLBench[50]、CALVIN[51]、RoboTwin[49] | 语言条件操作、多任务泛化、操作技能评测成熟 | S0–S1 为主，不能证明导航—操作衔接能力 |
| 大规模真实操作数据 | DROID[22]、Open X-Embodiment[23]、BridgeData V2[60]、AgiBot World[61] | 真实视觉、多机器人、多任务数据，有利于表征学习和数据工程 | 移动操作链条、base/action schema、任务阶段标注不一定完整 |
| household manipulation / kitchen benchmark | RoboCasa[32]、RoboCasa365[19]、ManiSkill3[40] | 工程生态成熟，任务资产丰富，可快速形成训练/评测闭环 | mobile base 与 navigation-manipulation 强度必须单独核验 |
| mobile manipulation benchmark | HomeRobot/OVMM[7][20][21]、MobileManiBench[33]、Kitchen-R[34]、LAMBDA[38] | 更接近移动操作主问题，覆盖搜索、接近、抓取、放置和长 horizon | 部分资源在线评测、未开源或工程复杂，不适合直接作为第一闭环 |
| 高真实感 household / rearrangement | BEHAVIOR-1K[24]、OmniGibson[45]、Habitat 2.0 / 3.0[42][43][44] | 高真实感、高复杂任务、物体状态与场景交互丰富 | 工程成本较高，更适合后期外推验证 |
| WAM / world model evaluation | WorldEval[70]、dWorldEval[71]、WPE[72]、WorldGym[73]、OSCAR[18] | 关注 world model 对 policy evaluation、未来预测和诊断的价值 | 多数是方法论参考，不一定直接提供移动操作训练数据 |

这个谱系说明，benchmark 选择不能只按“真实感”“论文新”“数据量大”排序。本项目需要同时评估空间移动性、视觉质量、生态成熟度、模态字段和 WAM 监督信号可用性。

## 4.3 值得重点参考的 Benchmark 与数据资源

从上述典型资源中，本章真正用于支撑本项目路线决策的重点资源包括：RoboCasa/RoboCasa365[19][32] 作为近期可落地 household manipulation 强闭环入口；AIRoA MoMa[36] 作为真实移动操作数据核验核心；EBench[35]、MoMa-Kitchen[39]和 ManiSkill3[40] 作为 P0 readiness / failure-risk / 操作子任务诊断候选；HomeRobot/OVMM[7][20][21]、BEHAVIOR-1K[24]、Habitat[42][43][44]、LAMBDA[38] 作为后期高难验证；MobileManiBench[33]和 Kitchen-R[34] 作为开放后优先上调的高价值候选。

这一区分很重要：典型资源用于说明领域发展，重点参考资源才进入本章的路线决策。未开放、仅论文、在线评测或字段未核验的资源，即使方向高度相关，也不能被写入三个月第一闭环。

## 4.4 筛选标记体系：S / V / M

本报告沿用 v0.8 的三组标记：

| 维度 | 等级 | 对本项目的意义 |
|---|---|---|
| 空间难度 S0–S5 | S0 固定桌面，S1 局部工作台，S2 单房间移动操作，S3 多区域单房间，S4 跨房间同楼层，S5 跨楼层/开放长距离 | S2–S4 是本项目最核心范围；S0 只能做操作辅助；S5 多为长期外推 |
| 视觉等级 V0–V4 | V4 高真实感/Omniverse/real scan，V3 高质量合成资产，V2 可用但仿真感明显，V1 功能/几何仿真，V0 toy sim | P1/P2 更依赖 V3–V4；P0 诊断集可接受 V1–V2，但不能作为视觉主证据 |
| 生态成熟度 M0–M4 | M4 社区成熟，M3 可用，M2 新仓需试跑，M1 开源弱，M0 无可靠代码 | 三个月闭环优先 M3–M4；M1–M2 需要先做小规模核验 |

这个体系解决了一个常见误区：**视觉真实感高不等于可落地，生态成熟不等于移动操作相关，论文强不等于三个月能用。**

同时，benchmark 选择不能只看成功率表或任务数量，也要关注评测到底在衡量什么。相关 benchmark 反思工作[28]提示，如果评测维度没有区分场景、技能、horizon、precision、mobility 和失败类型，就容易把“桌面操作成功率”误解为“移动操作世界模型能力”。

## 4.5 第一层：近期可落地主闭环候选

近期主闭环应优先选择“可用性相对明确、工程生态较好、能够支撑 P0/P1 训练或评测”的数据源。根据 v0.8，第一层候选为：

| 候选 | 类型 | 平台/仿真器 | 核对状态 | 关键价值 | 主要风险 | 当前结论 |
|---|---|---|---|---|---|---|
| RoboCasa365[19] | 仿真 benchmark / 数据集 | RoboCasa[32] / robosuite / MuJoCo | 可用 | 365 tasks、2500 kitchen scenes、600h+ human demos、1600h+ synthetic demos，工程生态成熟 | 需核实 mobile base / navigation-manipulation 强度 | 近期闭环高优先，但在移动强度核验前只作为 household manipulation 强闭环底座 |
| RoboCasa[32] | 仿真 benchmark | robosuite / MuJoCo | 可用 | 成熟、100 tasks、厨房资产质量较好，适合 P0/P1 快速闭环 | 不是纯移动操作主证据 | 近期闭环高优先 |
| AIRoA MoMa[36] | 真实移动操作数据集 | Toyota HSR / Real | 数据/代码待核实 | 真实移动操作、多模态、层级标注，适合 progress/readiness/failure 标签 | 当前备注偏短程任务，需核实开放与模态完整性 | 真实 P0/P1 核心候选 |
| EBench[35] | VLA / mobile manipulation 诊断 benchmark | Isaac Sim | 可参考 / 需核验 | 含 mobile manipulation，五轴诊断：Scene、Atomic Skill、Horizon、Precision、Mobility | 新仓需试跑，模态细节待核 | P0/P1 诊断候选 |
| ManiSkill3[40] | 仿真器 + benchmark | SAPIEN / ManiSkill[40][41] | 可用 / 需版本核验 | GPU 并行、模态丰富、适合数据生成与操作子任务 | 移动操作强度需按任务核验 | 工程底座与辅助闭环 |
| MobileManiBench[33] | 仿真移动操作 benchmark | Isaac Sim | coming soon / 尚未开源 | 论文强，multi-view RGB-D-Seg、2 mobile platforms、100 scenes、300K trajectories | 代码/数据未开放，不能作为近期主闭环 | 高价值候选，待开放 |
| Kitchen-R[34] | 仿真移动操作 benchmark | Isaac Sim | 未开源 / 仅论文 | 500+ complex language instructions，厨房规划 + 低层控制 | 未找到代码/数据入口 | 仅论文候选 |

这一层的核心判断是：**RoboCasa[32] / RoboCasa365[19] 可作为近期工程闭环入口；AIRoA MoMa[36] 必须进入真实数据核验；EBench[35] 和 ManiSkill3[40] 可作为诊断/生成候选；MobileManiBench[33] 与 Kitchen-R[34] 只能按“高价值待开放”处理，不能写成已可用主 benchmark。**

## 4.6 第二层：真实移动操作关键候选

真实数据对 P0/P1 的价值在于真实传感器噪声、真实接触、真实任务阶段和真实失败模式。v0.8 中真实移动操作候选如下：

| 候选 | 平台 | 空间 | 模态 | 任务/规模 | 当前定位 |
|---|---|---|---|---|---|
| AIRoA MoMa[36] | Toyota HSR | S2–S4，需按 episode 判断 | RGB、Joints、FT、internal State、primitive Action、Stage/Subgoal；Depth/Lang/Base/EEF/Success 需核验 | 25,469 episodes，约 94h | 真实 P0/P1 核心候选，必须验证 |
| Mobile ALOHA[37] | mobile base + 双臂 ALOHA | S2–S5 | multi-camera RGB、dual-arm Action、Joints、gripper、Base；Depth/Lang/Success 需核验 | 50 demos/task，厨房、开柜、进电梯等 | 真实 whole-body mobile manipulation 形态参考 |
| LAMBDA[38] / λ Benchmark[38] | sim + real mobile manipulator | S4–S5 | Lang、demo trajectories；RGB/RGB-D、State、Action、Base、EEF、Joints 需下载核验 | 571 human-collected demos | 高难空间外推 / 数据效率 benchmark，不作第一步 |
| HomeRobot / OVMM[7][20][21] real | Hello Robot Stretch | S4 | RGB-D、Lang goal、nav/manip Action、object/receptacle target | OVMM[7][20][21] real component | 真实高难评测参考 |
| SHOPPER 系 | grocery-store mobile manipulator | S5 | logs、metrics、failure modes；模态待核 | grocery store 多周 field tests | failure taxonomy / 真实系统指标参考 |

本报告建议：**AIRoA MoMa[36] 和 Mobile ALOHA[37] 进入真实数据重点核验；LAMBDA[38]、HomeRobot[7][20][21] real、SHOPPER 更适合作高难外推、系统指标或失败模式参考，不作为第一阶段闭环。**

## 4.7 第三层：高真实感但高复杂度验证集

这一层更接近完整 household activity 或跨房间移动操作，但工程复杂度高，适合后期验证或长期目标。

| 候选 | 平台 | 空间 | 优势 | 风险 | 定位 |
|---|---|---|---|---|---|
| BEHAVIOR-1K[24] | OmniGibson[45] / Omniverse | S3–S5 | 1000 everyday activities、50 scenes、9000+ objects、物理复杂度高 | 工程重，不宜第一步全量接入 | 长期高复杂 WAM benchmark |
| OmniGibson[45] | Omniverse / PhysX | S3–S5 | 可生成复杂物理状态和多模态观测 | 更像仿真底座，不是单一数据集 | P1/P2 长期路线 |
| HomeRobot / OVMM[7][20][21] sim | Habitat[42][43][44] / HomeRobot[7][20][21] | S4，可构造单房间 | open-vocabulary object search + grasp + place，接近移动操作主证据 | 抓取为 attach/吸附机制，偏在线评测，无静态 human traces | 主移动操作评测候选，非第一步训练 |
| Habitat[42][43][44] 2.0 / HAB[42] | Habitat[42][43][44] / ReplicaCAD | S3–S4 | 支持 rearrangement、导航—操作—物体状态 | 高难跨区域/跨房间验证 | 后期验证 |
| Habitat[42][43][44] 3.0[44] | Habitat[42][43][44] | S3–S4 | 动态人类、social rearrangement | 超出当前主线 | 长期协作 WAM 参考 |
| LAMBDA[38] | sim + real | S4–S5 | multi-room / multi-floor / data efficiency | 难度高，仓库生态小 | 高难空间外推 |

这层的处理原则是：**写入正文，但不要作为第一阶段必须跑通对象。**

## 4.8 第四层：P0 readiness / affordance 诊断集

P0 的关键不是“最终 benchmark 一步到位”，而是需要可靠诊断任务来验证 readiness、failure-risk、task-progress、next-best-view 和 subgoal feasibility。v0.8 指出的诊断候选包括：

| 候选 | 平台 | 适合诊断什么 | 限制 | 当前定位 |
|---|---|---|---|---|
| MoMa-Kitchen[39] | BestMan / PyBullet | final-pose affordance、manipulation-readiness、last-mile navigation | V1，视觉真实感弱，不是 Isaac，不作主 benchmark | P0 readiness / affordance 诊断集 |
| EBench[35] | Isaac Sim | Scene、Atomic Skill、Horizon、Precision、Mobility 五轴诊断 | 新仓需试跑 | WAM 诊断评测候选 |
| Habitat[42][43][44] Rearrangement Challenge[43] | Habitat[42][43][44] | rearrangement、pick/place/open/close、object state | 偏 leaderboard / online eval | 后期诊断 |
| ManiSkill3[40] / ManiSkill-HAB[47]| SAPIEN / ManiSkill[40][41] | 高速生成操作与家庭 rearrangement 子任务 | 移动性需按任务核验 | 操作子任务与中间路线 |
| SAGE-3D[11] / Isaac 自建路线 | Isaac / 3DGS / 自定义 | 空间视觉、轨迹、RGB、PCD、collision labels | 不是标准移动操作 benchmark | demo / ablation / 自有补充数据 |

本报告明确下调 MoMa-Kitchen[39]：它适合 P0 final-pose affordance / manipulation-readiness 诊断集，不适合做主视觉 benchmark。

## 4.9 第五层：非移动操作辅助集

这一层很重要，但必须避免“越位”：

| 候选 | 类型 | 为什么有用 | 为什么不能作主证据 |
|---|---|---|---|
| LIBERO[48] / LIBERO-Long[48] | 语言条件操作 | 生态成熟，适合 P0/P1 sanity check 和长时序操作消融 | S0 固定桌面，无移动 |
| LIBERO-Plus[48] / LIBERO-PRO[48] | robustness / OOD | 适合 perturbation 与 anti-memorization 诊断 | 仍是 LIBERO[48] setting |
| RoboTwin 2.0[49] | 双臂操作 benchmark | 双臂泛化、domain randomization、合成数据强 | S0–S1，移动弱 |
| RLBench[50] | 多任务操作 | 经典视觉操作，多模态可得 | CoppeliaSim 固定机械臂为主 |
| CALVIN[51] | long-horizon language manipulation | 语言长时序、任务进度参考 | PyBullet 桌面环境 |
| FurnitureBench[52] | 真实+仿真装配 | 真实接触丰富、长时序 | 家具装配，非 household mobile manipulation |
| MimicGen[53] / SkillMimicGen[54] | 数据生成系统 | demo 扩增与 scaling 方法参考 | 主要服务操作技能，不是移动主 benchmark |

结论：**这一层进入辅助验证与数据工程章节，不进入移动操作主证据。**

## 4.10 第六层：World Model Evaluation / Leaderboard / 方法论参考

这一层不是训练数据主来源，但对 WAM 评测有启发。

| 候选 | 类型 | 价值 | 定位 |
|---|---|---|---|
| HomeRobot[7][20][21] Challenge / OVMM[7][20][21] Challenge | mobile manipulation leaderboard | 在线评测、真实/仿真移动操作指标 | 主移动操作评测参考 |
| VLA-Arena[69]| VLA benchmark / framework | safety、distractor、extrapolation、long horizon 等 robustness 维度 | robustness 指标参考 |
| WorldEval[70] | world model policy evaluator | 用 world model 评估真实机器人 policy | P1/P2 评测方法核心参考 |
| dWorldEval[71]| discrete diffusion WM eval | progress token 对 P0 有启发 | P0/P1 指标参考 |
| WPE[72]/ WorldGym[73]| action-conditioned video WM eval | video world model evaluator | P2/WAM-eval 长期参考 |
| OSCAR[18] | action-conditioned video WAM | 架构参考，非主数据集 | 长期 WAM 参考 |
| SimplerEnv[74] | real-policy sim eval | 代理真实策略评估方法论 | 方法参考 |

## 4.11 待核实与暂不作为正式 benchmark 的名称

v0.8 明确将以下名称列为待核实，不得在正文中作为确定 benchmark 使用：

| 名称 | 当前处理 |
|---|---|
| LIBERO-Extended | 未确认官方正式 benchmark，暂缓 |
| RoboCasa-Lite | 未确认官方正式 benchmark，暂缓 |
| RoboTwin-Diagnosis | 未确认官方正式 benchmark，可能只是诊断方向泛称，暂缓 |

## 4.12 本章结论

本项目数据与 benchmark 应采取“近期闭环 + 真实数据核验 + 高难后期验证 + 诊断集 + 辅助集”的组合路线：

1. **近期主闭环**：RoboCasa[32] / RoboCasa365[19] + AIRoA MoMa[36] 核验 + EBench[35]/ManiSkill3[40] 诊断试跑。
2. **强移动操作候选**：MobileManiBench[33]、Kitchen-R[34]、HomeRobot/OVMM[7][20][21]、LAMBDA[38]、BEHAVIOR-1K[24]，但需按开放性和工程难度分阶段处理。
3. **P0 诊断集**：MoMa-Kitchen[39]、EBench[35]、Habitat[42][43][44] Rearrangement、ManiSkill-HAB[47]、自建 SAGE-3D[11]/Isaac。
4. **操作辅助集**：LIBERO[48]、RoboTwin[49]、RoboCasa[32]、DROID[22]、Open X-Embodiment[23] 只能作为快速消融、泛化和表征学习参考。
5. **附录保留**：v0.8 全量候选表进入附录，正文只保留 8–12 个真正影响路线选择的核心候选。

> 多数据域混合训练与采样权重学习属于训练数据工程问题，不再放在本章展开；相关策略统一迁移至第五章 5.8。第四章只保留数据集、benchmark 与 leaderboard 的筛选和定位结论。

# 第五章 数据工程与可持续 Scaling 调研

## 5.1 本章目的、作用与承接关系

第四章已经完成数据集与 benchmark 的分层核对，回答了“哪些资源可用、哪些资源暂缓、哪些资源只能作为辅助参考”。本章在此基础上进一步回答一个更工程化的问题：**这些数据如何转化为 P0/P1/P2 可用的监督信号、训练样本和评测闭环**。

本章在全文中的作用是承上启下。向前看，它承接第四章的数据可用性判断，避免把 coming soon、未开源、在线评测型资源误写成近期训练数据。向后看，它直接决定第六章 P0、第七章 P1 和第八章 P2 的路线是否能落地：P0 需要 progress、readiness、failure-risk 等轻量 future 标签；P1 需要 current/future Wan2.2[14] latent、历史 state/action 与语言条件；P2 需要少量可解释性解码样本，而不是全量视频生成数据。

因此，本章不是单纯列数据源，而是围绕三个问题组织调研：第一，现有机器人数据工程的发展趋势是什么；第二，哪些代表性工作的数据处理方式值得参考；第三，结合本项目三个月周期和算力约束，应采用怎样的数据工程路线。

## 5.2 数据工程研究的发展与现状

机器人学习的数据工程大致经历了三个阶段。

第一阶段是 **benchmark-centric imitation learning**。这一阶段以 LIBERO[48]、RLBench[50]、CALVIN[51]、RoboTwin[49] 等固定或半固定操作 benchmark 为代表，重点是把语言、视觉、状态和动作整理成统一模仿学习样本。它的优点是工程成熟、任务可复现、适合快速训练和消融；缺点是空间移动、导航—操作衔接、失败恢复和真实家庭场景复杂性不足。对于本项目，这类数据不能作为移动操作主证据，但可以作为 action/state/lang schema 的快速 sanity check。

第二阶段是 **household / mobile manipulation 数据与仿真环境扩展**。RoboCasa[32]/RoboCasa365[19]、HomeRobot/OVMM[7][20][21]、Habitat[42][43][44] Rearrangement、BEHAVIOR-1K[24]、ManiSkill3[40]、AIRoA MoMa[36]、Mobile ALOHA[37] 等资源开始覆盖更复杂的家庭场景、移动平台、长时序任务和真实机器人演示。它们更接近本项目目标，但工程代价和数据 schema 差异也明显上升：有的偏仿真、有的偏真实但任务短；有的支持在线评测但缺少离线轨迹；有的视觉真实感强但物理交互或抓取机制有简化。

第三阶段是 **world-model-oriented data engineering**。这一阶段的数据不再只服务“当前观测到动作”的监督，而是需要构造 future supervision：未来 latent、未来可见性、任务进度、失败风险、接触/可操作状态、历史动作导致的未来变化等。对 WAM 来说，数据工程的核心不再是“样本越多越好”，而是“哪些字段能支持 action-relevant future 建模”。这也是本项目与普通 VLA 数据工程的区别。

### 5.2.1 研究发展谱系与典型工作补充

为避免“研究发展”部分直接跳到本项目结论，本节在 v1.6 三阶段叙述基础上，补充按研究重心划分的典型工作谱系。这里列出的工作用于说明方向演化，并不意味着它们都会成为本项目的重点参考或近期实现对象。

| 阶段/类别 | 核心问题 | 典型工作/资源 | 对本项目的直接含义 |
|---|---|---|---|
| **Benchmark-centric imitation learning** | 如何把语言、视觉、状态和动作整理成可复现模仿学习样本 | LIBERO[48]、RLBench[50]、CALVIN[51]、RoboTwin[49]、ManiSkill3[40] 等，详见 v0.8 候选表 [11] | 适合作 schema sanity check、快速消融和操作能力辅助验证；不能作为移动操作主证据 |
| **Household / mobile manipulation 数据与仿真扩展** | 如何覆盖家庭场景、移动平台、开放词汇目标和长时序任务 | RoboCasa[32]/RoboCasa365[19] [19]、HomeRobot/OVMM[7][20][21] [7][20][21]、BEHAVIOR-1K[24]/OmniGibson[45] [24]、AIRoA MoMa[36]、Mobile ALOHA[37]、EBench[35]、MoMa-Kitchen [11] | 决定 P0/P1/P2 的主数据和评测候选；需要逐项核验空间尺度、抓取机制、离线轨迹和 state/action 字段 |
| **Robot data alignment 与跨 embodiment 数据工程** | 如何把不同机器人、相机、动作空间和任务语义对齐 | Qwen-RobotManip [4]、DROID [22]、Open X-Embodiment[23] / RT-X [23]、RoboCasa365 [19] | 为多来源数据清洗、动作标准化、相机/状态对齐和数据域划分提供方法参考 |
| **World-model-oriented future supervision** | 如何让数据不只服务 action cloning，而是服务未来状态、未来 latent、风险和可操作性建模 | Fast-WAM [1]、LaWAM [2]、VLA-JEPA [3]、Light-WAM [15]、Efficient-WAM [26]、Wan2.2 [14]、ChronoDreamer [17]、1X World Model Challenge [16] | 决定 P0 的轻量 future 标签、P1 的 Wan2.2[14] latent target 与历史机器人条件、P2 的局部解码诊断样本 |
| **Data mixture / sampling weight learning** | 多域数据如何混合，避免大数据域淹没小但关键的数据域 | Re-Mix [12]、DoReMi[13] / DRO 类数据混合思想 [13] | 支持“proxy WAM 学 α，正式 P0/P1 固定 α”的轻量数据混合策略 |

因此，第 5.3 节中的“值得参考的数据工程做法”不是对上述典型工作的重复罗列，而是从这些类别中筛出真正支撑本项目数据工程决策的来源：Qwen-RobotManip[4]、RoboCasa[32]/RoboCasa365[19]、DROID[22]/Open X[23]、HomeRobot/OVMM[7][20][21]、BEHAVIOR-1K[24]、Wan2.2[14] 与 Re-Mix[12]/DoReMi[13]。

## 5.3 值得参考的数据工程做法及优劣分析

本节与上一节的关系需要明确区分：5.2 负责罗列数据工程发展的类别和典型工作，5.3 只从这些典型工作中筛选真正影响本项目数据工程决策的重点来源。筛选标准包括：是否能指导多源数据对齐、是否能支撑 P0/P1/P2 的监督信号构造、是否能帮助确定近期主闭环与后期验证集、是否能降低三个月内工程风险。


**Qwen-RobotManip[4] / Qwen 系机器人工作** 的价值在于展示了大规模机器人数据清洗、筛选、动作—视觉对齐、相机参数注入和训练 recipe 的重要性。它提醒本项目：即使模型架构先进，如果数据中的相机、动作、状态、语言和任务阶段没有对齐，future representation 很容易学成视觉相关性，而不是机器人动作相关性。其局限是它主要服务 VLA / manipulation policy，而不是直接服务移动操作 WAM。

**DROID[22]、Open X-Embodiment[23]、BridgeData[60]、AgiBot World[61] 等真实操作数据** 的价值在于真实分布、多 embodiment、多场景和多动作形态。它们适合帮助 Wan2.2[14] latent 或 P0/P1 表征接触真实视觉分布，避免只在仿真厨房中有效。局限在于移动属性不足、动作空间不统一、语言和 success/failure 标注不一定完整，因此更适合作为辅助数据域或表征预训练资源，而不是移动操作主 benchmark。

**RoboCasa[32] / RoboCasa365[19]** 的价值在于 household 场景、任务数量、工程生态和可控仿真流程。它适合构造 P0 的 progress/readiness 标签，也适合为 P1 缓存 current/future Wan2.2[14] latent。局限是其移动操作强度、导航—操作耦合程度仍需核验，不能因为任务丰富就直接把它写成完整移动操作证据。

**AIRoA MoMa[36]、Mobile ALOHA[37]、HomeRobot/OVMM[7][20][21]、LAMBDA[38] 等真实或移动操作资源** 的价值在于提供更接近移动操作的历史 state/action、base/arm/EEF 运动、真实失败和长时序阶段。局限是下载、字段、任务长度、评测接口和复现成本都需要逐项核验。它们更适合用于 P0/P1 关键验证，而不是一开始就全量接入。

**BEHAVIOR-1K[24]、Habitat[42][43][44]、OmniGibson[45]、ManiSkill3[40]、EBench[35] 等仿真环境** 的价值在于可以构造诊断任务、控制场景变量和生成失败样本。局限是工程复杂度高，不同仿真器的物理真实性和动作接口差异大。本项目应优先把它们作为诊断和后期验证资源，而不是把全部环境都接入主训练。

## 5.4 本项目的数据需求与监督信号映射

结合上述调研，本项目的数据工程可以按“监督信号”而不是“数据集名称”组织。不同数据源只要能提供某类有效监督，就可以作为独立数据域参与训练；不要求所有数据集都同时具备 RGB、语言、base、EEF、joints、gripper、success、failure 和完整任务阶段。

| 数据源 | 可构造 P0 监督 | 可构造 P1 监督 | 可用于 P2 | 主要风险 | 推荐用途 |
|---|---|---|---|---|---|
| AIRoA MoMa[36] | progress、primitive action、subgoal、failure/readiness 线索、force/contact risk | 真实 RGB latent、状态转移 latent、接触相关 latent | 可做真实关键帧分析，不做生成主线 | 偏短程任务；Depth/Lang/Base/EEF/Success 需核验 | 真实 P0/P1 核心候选 |
| RoboCasa365[19] | task progress、success/failure、object state、操作阶段 | synthetic future latent、视觉/状态对齐 latent | 可做 keyframe demo | mobile base / navigation-manipulation 强度需核验 | 近期闭环与 P1 辅助 |
| RoboCasa[32] | progress、readiness、object-state change | future visual/state latent | 可做局部 keyframe | 移动性弱于真正 mobile benchmark | 第一闭环 |
| MobileManiBench[33] | readiness、navigation-to-manipulation、Base/arm action、multi-view future | multi-view RGB-D-Seg latent、state/action latent | 可做 Isaac future demo | coming soon，不能按已可用写 | 若开放则为仿真主候选 |
| Kitchen-R[34] | task plan、low-level control、kitchen stage | task-plan-conditioned latent | 可做厨房 keyframe | 未开源，仅论文候选 | 暂缓，待开放 |
| EBench[35] | Scene、Atomic Skill、Horizon、Precision、Mobility 诊断标签 | 诊断维度 latent | 局部可视化 | 新仓需试跑 | P0/P1 诊断 |
| MoMa-Kitchen[39] | final-pose affordance、manipulation-readiness | affordance latent | 不建议 P2 主用 | V1 / PyBullet，视觉弱 | P0 readiness 诊断 |
| HomeRobot / OVMM[7][20][21] | object/receptacle target、nav/manip success、open-vocabulary goal | navigation-manipulation latent | 评测用 keyframe | attach grasp，在线评测，离线数据不足 | 后期主评测 |
| BEHAVIOR-1K[24] | object state、task state、复杂 household transition | rich household latent | P2 长期 | 工程重 | 高难长期验证 |
| LIBERO[48] / RoboTwin[49] | action/state/lang、success、扰动/OOD | 操作 latent、动作泛化 latent | 低优先 | 非移动主证据 | 快速 sanity / 泛化辅助 |
| DROID[22] / Open X[23] / BridgeData[60] | 真实操作状态/action/lang | 真实操作 latent pretraining | 低优先 | 移动属性不足、schema 复杂 | 表征学习和数据工程参考 |

## 5.5 P0 数据工程：从任务轨迹到轻量 future 标签

P0 的数据工程目标不是生成未来图像，而是从现有轨迹中抽取可以反映未来状态的轻量标签。这类标签需要具备两个特点：一是能从已有数据自动或半自动构造，二是能被下游策略直接消费。

| P0 标签 | 主要数据来源 | 自动构造方式 | 对 WAM 的作用 |
|---|---|---|---|
| task progress | RoboCasa[32]/RoboCasa365[19]、LIBERO[48]、AIRoA MoMa[36]、ALFRED[65] | 根据 subtask、stage、success 条件或 primitive action 序列切分 | 判断任务是否推进 |
| manipulation-readiness | MoMa-Kitchen[39]、HomeRobot[7][20][21]、MobileManiBench[33]、RoboCasa[32] | 根据目标可见性、距离、朝向、可达性、final pose affordance 构造 | 判断导航—操作切换 |
| failure-risk | AIRoA MoMa[36]、HomeRobot[7][20][21]、EBench[35]、RoboTwin[49]、DuoBench | 根据失败、force/contact 异常、动作后状态未变、timeout 标注 | 失败恢复和安全约束 |
| next-best-view | MobileManiBench[33]、Habitat[42][43][44]、SAGE-3D[11]、自建 Isaac | 根据未来可见性、遮挡减少、目标区域占比变化生成 | 视角调整 |
| subgoal feasibility | RoboCasa365[19]、HomeRobot[7][20][21]、BEHAVIOR[24]、ALFRED[65] | 根据候选子目标是否导致成功或阶段完成 | 子目标评分 |
| action outcome class | RoboCasa[32]、AIRoA MoMa[36]、LIBERO[48]、RoboTwin[49] | 根据动作后物体状态、夹爪状态、目标可见性变化分类 | action-relevant future |

P0 的优势是数据要求较低，适合三个月内建立第一闭环。其局限是标签粒度较粗，不能完整表达未来视觉细节和历史动作导致的细微状态变化。因此，P0 更适合作为 video-generation-free baseline 和 WAM-specific 诊断层，而不是最终 latent world model。

## 5.6 P1 数据工程：Wan2.2[14] latent target、语言条件与历史机器人条件

P1 的数据工程应收敛到一个明确口径：**Latent-Only WAM**。Wan2.2[14] 的价值不是直接生成视频，而是提供统一的视觉 latent space、文本条件空间和可继承的视频 latent dynamics prior。

| Wan2.2[14] 组件 | P1 推荐用法 | 是否作为默认主线 | 说明 |
|---|---|---:|---|
| Wan2.2-VAE encoder | 编码当前/未来机器人视频片段，构造 current/future latent pair | 是 | 作为 P1 latent target 与 P2 解码衔接点 |
| Wan2.2 T5/UMT5[14] text encoder | 编码任务语言，提供 text condition | 是 | 保持与 Wan2.2[14] video latent 体系一致 |
| Wan2.2 TI2V-5B[14] DiT | 继承视频 latent dynamics prior，并通过 LoRA[25]/adapter 注入历史机器人条件 | 是，P1-b1 | 避免从零训练 WAM，但不输入未来 action label |
| Wan2.2-VAE decoder | P1 不使用；P2 局部 demo 时使用 | 否 | 一旦解码像素即进入 P2 |

P1 数据构造至少包括四步：第一，从机器人轨迹中抽取历史片段、当前片段和未来片段；第二，用 frozen Wan2.2-VAE encoder 缓存历史/current/future RGB 或 short-video latent，其中 future latent 只作为预测目标；第三，用 Wan2.2[14] text encoder 缓存或在线构造语言 condition；第四，将历史 robot state/action 通过 trainable projector / adapter 映射为与 Wan2.2 DiT condition space 对齐的 robot history latent。这里的 action 必须是历史 action，而不是未来动作标签。历史 RGB/video latent、语言 condition 和历史 state/action condition tokens 共同构成 P1-b1 的 history latent 输入；future action 防泄漏应通过 dataset slicing invariant、batch assert 与 unit test 保证，不应作为实验矩阵中的独立消融。

P1 的评估重点也不是视频生成质量，而是 predicted future latent 是否提升 readiness、failure-risk、progress、subgoal feasibility 和 downstream task metrics。P2 才对少量 predicted latent 做解码，用于可解释性和误差诊断。

## 5.7 P2 数据工程：只服务局部解码与诊断

P2 的数据工程不能反向主导整个项目。对于三个月周期而言，P2 不应要求全量连续视频生成数据，而应从 P1 预测的 Wan2.2[14] latent 中选择少量关键样本进行解码。

适合 P2 的数据不是“所有轨迹视频”，而是以下几类：第一，关键阶段样本，如接近目标、抓取前、放置前、失败恢复前；第二，P1 预测错误或不确定性高的样本；第三，P0/P1 指标出现矛盾的样本，例如 readiness 高但 action 失败，或 future latent 看似合理但 progress 不提升。这样 P2 才能服务 WAM 诊断，而不是变成独立视频生成项目。

## 5.8 多数据域混合与可持续 scaling 闭环

本节承接第四章的数据集 / benchmark 筛选结论，专门讨论“这些异构数据源如何共同进入训练闭环”。它原本容易被放在第四章之后作为附加结论，但从性质上看，它不是 benchmark 选择问题，而是数据工程与可持续 scaling 问题，因此统一放在第五章展开。

完成 v0.8 数据集与 Benchmark 候选表核对后，一个重要结论是：本项目的数据来源天然异构，不能简单按数据量或粗粒度类别混合。RoboCasa[32] / RoboCasa365[19]、AIRoA MoMa[36]、EBench[35] mobile、MoMa-Kitchen[39]、LIBERO[48] / RoboTwin[49] / DROID[22] / Open X-Embodiment[23] 等候选资源，在仿真器、机器人本体、视觉真实度、空间范围、动作空间和标注字段上均存在显著差异。它们对 P0 / P1 的贡献方式也不同：有的数据适合构造 task-progress、manipulation-readiness、failure-risk 等 P0 标签；有的数据适合提供未来帧、动作轨迹或 latent target；还有的数据只适合做操作泛化、表征学习或诊断验证。

因此，本报告不建议采用两类常见但过于粗糙的数据混合方式。第一种是按数据量比例采样，这会导致大规模但不一定最相关的数据集淹没小规模但关键的数据域，例如真实移动操作数据或 P0 readiness 诊断集。第二种是简单划分为 mobile-only / manipulation-only 两类数据，这会掩盖不同数据源之间更细粒度的差异，例如 RoboCasa[32] 与 MoMa-Kitchen[39] 同属厨房场景但视觉真实度、仿真器、动作字段和任务定位完全不同；AIRoA MoMa[36] 与 Mobile ALOHA[37] 同属真实移动操作数据，但机器人平台、动作空间、任务长度和标注层级也并不一致。

更合理的调研判断是：**将每个可用数据集视为一个独立数据域**。例如，RoboCasa[32] / RoboCasa365[19] 可作为一个或两个 household kitchen manipulation 数据域；AIRoA MoMa[36] 可作为真实移动操作数据域；EBench[35] mobile 可作为 Isaac Sim 诊断评测数据域；MoMa-Kitchen[39] 可作为 P0 final-pose affordance / manipulation-readiness 诊断域；再加入一个高质量纯操作数据域，例如 LIBERO[48]、RoboTwin[49]、DROID[22] 或 Open X-Embodiment[23] 的相关子集，作为操作泛化和表征补充。每个数据域根据自身可用字段参与 WAM 训练，不强制所有数据集提供完全一致的监督信号。例如，具备 Stage/Subgoal 的真实数据可以更多服务 progress/readiness，具备未来视觉序列的数据可以服务 latent target，只有高质量操作轨迹的数据则作为动作与视觉表征的补充。

从研究现状看，Re-Mix[12] / DRO 类工作提供了一个重要启发：机器人数据混合不应完全依赖人工经验比例，而可以通过小模型或 proxy training 学习不同数据域的采样权重。Re-Mix[12] 针对大规模 imitation learning 数据混合，使用 distributionally robust optimization 思想学习不同机器人数据域的权重，以提升下游域上的稳健性；类似的 domain reweighting 思路在语言模型预训练中也被用于先训练 proxy model，再用得到的 domain weights 指导大模型训练。本文借鉴这一类思想，但不把“数据混合权重学习”本身升级为新的核心算法路线。

结合三个月周期和实验预算，本报告建议采用轻量折中策略：先训练一个小规模 P0 proxy WAM，用于估计不同数据域对当前 WAM 训练目标的有效贡献；随后得到一组 learned sampling weights α，并在正式 P0 / P1 训练中固定使用这组 α。这样做的意义不是追求复杂的动态采样算法，而是以低成本缓解两个问题：一是减少人工设定数据比例的不确定性；二是避免大规模辅助操作数据主导训练，使 AIRoA MoMa[36]、EBench[35] mobile、MoMa-Kitchen[39] 等小规模但与移动操作 WAM 关键能力相关的数据域被稀释。

该策略还可以增强 P0 与 P1 的公平比较。P0 与 P1 的主要差异应来自 future representation 形态：P0 使用 video-free lightweight future supervision，P1 使用 latent-only future substrate。如果二者使用不同数据配比，则路线差异会与数据差异混淆。因此，正式 P0 / P1 训练阶段应尽量共享同一组 learned sampling weights α，使比较更聚焦于 P0/P1 方法本身，而不是数据分布变化。

本项目的数据 scaling 闭环应为：

```text
候选数据核验
→ schema 统一
→ P0 标签构造 / P1 latent target 缓存
→ 诊断集生成
→ WAM 训练
→ WAM-specific metrics
→ downstream task gain
→ 失败样本回流
→ 新数据/新诊断任务补充
```

其中，v0.8 的“可用性 / 核对状态”和“人工核对备注”是第一道 gate：coming soon、未开源、仅论文候选不能进入近期训练闭环；在线评测或仿真器不能被误写成离线数据集。纯操作数据仍然有价值，但它的定位应是补充性数据域，而不是移动操作主 benchmark。LIBERO[48]、RoboTwin[49]、DROID[22]、Open X-Embodiment[23] 等可以提供高质量动作、视觉和语言监督，帮助 WAM 学习更稳定的视觉—动作表征；但它们不能替代 RoboCasa[32] / RoboCasa365[19]、AIRoA MoMa[36]、EBench[35] mobile、HomeRobot / OVMM[7][20][21]、LAMBDA[38] 等移动操作相关候选在任务链、空间移动、导航—操作衔接和真实失败模式上的价值。

综上，本报告将该机制定位为**数据混合策略 / 训练数据配比策略**，而非新的核心算法路线。三个月内不建议完整复现 Re-Mix[12] 或开展复杂动态采样研究；更合适的做法是采用“proxy WAM 学 α，正式 P0/P1 固定 α”的轻量策略，作为多来源数据集共同服务 WAM 训练的工程折中。

## 5.9 本章路线判断与注意点

本章最终建议是：数据工程先服务 P0/P1 闭环，而不是追求全量数据接入。近期优先级应为：RoboCasa[32]/RoboCasa365[19] 跑通第一闭环，AIRoA MoMa[36] 作为真实移动操作关键候选核验，EBench[35]/ManiSkill3/MoMa-Kitchen 作为诊断资源，LIBERO[48]/RoboTwin[49]/DROID[22]/Open X[23] 作为辅助数据域。

需要特别注意五点：第一，纯操作数据可以用，但不能作为移动操作主证据；第二，P1 latent target 要优先缓存，避免训练时反复跑 Wan-VAE[5]；第三，历史 action/state 可进入 P1-b1，但未来 action label 不进入 WAM；第四，P2 只服务局部解码和误差分析；第五，P0/P1 公平比较时应尽量共享同一组数据域权重 α。

# 第六章 P0：Video-Generation-Free WAM 路线调研

## 6.1 本章目的、作用与承接关系

第五章解决了数据如何转化为监督信号的问题。本章进一步讨论 P0：在不生成未来视频、不引入重型 latent video model 的情况下，如何构建一个轻量、可被下游策略消费的 Video-Generation-Free WAM 分支。Qwen3VL[29] / Qwen-VL[30] 系视觉语言主干是当前优先候选实现之一，主要来自 VLA 策略项目的工程复用价值，但不构成 P0 的正式阶段命名。

P0 在全文中的作用是建立最低风险、最快可验证的 WAM 闭环。它承接第五章中的 progress、readiness、failure-risk、next-best-view、subgoal feasibility 等轻量 future 标签；同时为第七章 P1 提供比较基线。如果 P0 已经能通过 lightweight future supervision 带来稳定收益，那么 P1 必须证明 Wan2.2[14] latent 的额外复杂度确实值得。

因此，本章不直接给“P0 必须怎么做”的结论，而是先分析 video-generation-free WAM 的研究动因、相关工作的启发和局限，再结合本项目给出推荐路线。

## 6.2 研究发展：从 VLA action head 到 video-free future supervision

传统 VLA 主要学习从当前视觉、语言和机器人状态到动作的映射。它的问题是：模型可能在短 horizon 内拟合动作，但没有显式学习“任务是否推进、当前是否可操作、下一步是否更容易失败、目标是否会被遮挡”等 future-related 信息。对固定桌面操作来说，这个问题有时不明显；对移动操作来说，长时序、视角变化、导航—操作切换和失败恢复会放大这个缺陷。

video-generation WAM 试图通过生成未来视频解决这个问题，但完整未来视频生成代价高、延迟高，而且生成质量不一定等价于任务成功。由此出现了 video-generation-free 或 latent/lightweight future supervision 的路线：训练阶段让模型学习未来状态、未来 latent 或 future-aware representation，推理阶段不显式生成像素，只把这些信息压缩进策略特征或轻量 head。

P0 正是这一类路线在本项目中的落地版本。它不追求可视化未来，而是追求让策略具备对任务进度、风险和可操作性的前瞻判断。

### 6.2.1 研究发展谱系与典型工作补充

P0 所在方向不是单一模型路线，而是从传统 policy-only VLA 向 video-generation-free future supervision 演化出的低成本 WAM 分支。典型工作可以按以下类别理解：

| 发展类别 | 核心思想 | 典型工作 | 与 P0 的关系 |
|---|---|---|---|
| **Policy-only / VLA action head** | 直接学习 observation-language-state 到 action chunk 的映射 | Open X-Embodiment[23] / RT-X [23]、Qwen-RobotManip [4] 以及 LIBERO[48]/RoboTwin 等 VLA benchmark [11] | 提供 P0 的基线和数据 schema，但缺少显式 future supervision |
| **训练时 future modeling、测试时不显式想象** | 训练阶段通过未来建模塑造 representation，推理阶段不生成未来视频 | Fast-WAM [1]、Efficient-WAM [26]| 支持 P0 “video-generation-free”的核心设定 |
| **Latent / JEPA-style future supervision** | 用未来帧或未来 latent 作为 target，避免 pixel-level generation 进入推理路径 | VLA-JEPA [3]、LaWAM [2]、Light-WAM [15] | 支持 P0 的 future auxiliary heads、progress/readiness/failure-risk 标签设计 |
| **按需 future reasoning / adaptive WAM** | 不是每一步都运行重型 WAM，而是在不确定或高风险阶段触发 | AdaWAM [9]| 支持 P0/P1/P2 后续的 adaptive triggering 和风险驱动评估 |

第 6.3 节只展开其中对本项目 P0 决策最关键的工作：Fast-WAM[1]、VLA-JEPA[3]、Light-WAM[15]、Qwen-RobotManip[4] 与 AdaWAM[9]。其他工作主要承担背景、基线或长期参考作用。

除上述与本项目 P0 直接相关的代表工作外，policy-only/VLA action head 的背景还包括 OpenVLA[77]、RT-2[78] 和 Diffusion Policy[79] 等路线：它们说明“直接从视觉/语言到动作”的策略学习已经形成成熟范式，但也进一步凸显了本项目引入 future supervision 的必要性。

## 6.3 值得参考的工作与启发

上一节给出的是 P0 方向的研究谱系；本节只展开其中最能支撑本项目 P0 决策的工作。也就是说，Fast-WAM[1]、VLA-JEPA[3]、Light-WAM[15]、Qwen-RobotManip[4] 与 AdaWAM[9]不是“所有相关工作”，而是从典型工作中筛出的决策依据：它们分别支撑 test-time 解耦、防泄漏 future supervision、轻量 action decoding、机器人数据对齐和按需触发。


**Fast-WAM[1] 类思路** 的核心启发是：未来建模对训练有价值，但推理阶段未必需要显式生成未来。对本项目而言，这支持 P0 的基本假设：可以在训练中加入 future auxiliary learning，而在测试时只运行轻量 WAM heads 或将 future representation 融入 action policy。

**VLA-JEPA[3] / JEPA-style latent prediction** 的启发是：future information 可以作为监督目标，而不应在输入端泄漏。它强调 target encoder 编码未来，student pathway 只看当前或历史观测，从而学习更稳健的 future-aware representation。P0 可借鉴这种思想，把未来 progress、future visibility 或 future state change 作为监督，而不是把未来标签作为推理输入。

**Light-WAM[15] / lightweight future supervision 类工作** 的启发是：不一定要生成高分辨率未来视频，也可以在 downsampled latent 或 compact representation 上进行 future supervision，并用轻量 action expert 消费这些状态。它证明了“轻量 WAM + action head”可以成为重型视频生成 WAM 的低成本替代方案。

**Qwen-RobotManip[4] / Qwen-VL[30] 机器人化工作** 的启发在于数据清洗、动作—视觉对齐、相机参数和机器人数据 recipe，而不是直接提供 P0 架构。它说明 Qwen 系 VLM 可以作为机器人策略 backbone，但本项目必须在其上增加 WAM-specific future supervision，否则仍只是普通 VLA。

**AdaWAM-style adaptive reasoning** 的启发是不同任务阶段需要不同强度的 reasoning。P0 可以借鉴其 gating 思想，用 failure risk、readiness 或 stage change 判断是否需要触发更重的 P1/P2 模块；但三个月内不建议复现完整动态多模态路由系统。

## 6.4 优劣分析：为什么 P0 适合作为第一闭环

P0 的优势是工程风险低。它可以复用 VLA 策略项目中的视觉语言编码、StarVLA[31] 数据管线和状态 / 动作 schema，也可以直接消费第五章构造的轻量标签，不需要运行 Wan2.2 DiT[14]、缓存大规模 video latent 或设计复杂 action/state adapter。这里的“复用”是工程复用，不意味着 P0 被定义为 VLA action head 改装；P0 的研究对象仍是独立 WAM 分支。P0 的指标也更容易解释：progress 是否更准、readiness 是否能判断导航—操作切换、failure-risk 是否提前发现失败。

P0 的局限同样明确。它不能提供细粒度 future visual latent，也不能自然衔接 P2 解码；如果任务需要复杂未来视觉变化、遮挡恢复或长期场景预测，P0 的标量/类别标签可能表达能力不足。因此，P0 应被定位为 video-generation-free WAM baseline 和 lightweight future supervision 路线，而不是本项目的终点。

## 6.5 本项目采用的 P0 路线

P0 的推荐定义是：

```text
Video-Generation-Free WAM =
Qwen3VL[29] visual-language backbone
+ robot state / action-history condition
+ action-relevant future auxiliary learning
+ lightweight WAM heads
+ downstream task gain evaluation
```

它不是：

```text
Qwen3VL + action head = WAM
```

因为后者只是普通 VLA。P0 必须有独立的 future supervision 或 WAM-specific heads，才能构成本项目中的 WAM 分支。

P0 可预测的 lightweight future representation 包括：

| 输出 | 形式 | 下游用途 |
|---|---|---|
| task progress | scalar / class | 判断是否完成子任务 |
| manipulation-readiness | scalar | 导航—操作切换 |
| failure risk | scalar | 失败恢复 |
| next-best-view score | candidate score | 视角选择 |
| subgoal feasibility | score | 子目标筛选 |
| object visibility future | class / score | 找物与接近 |
| action outcome class | class | 动作结果诊断 |

P0 训练目标可以采用多任务形式：

```text
L = L_action + λ1 L_progress + λ2 L_readiness + λ3 L_failure + λ4 L_subgoal + λ5 L_visibility
```

其中 action loss 可复用VLA 策略项目的 VLA / Flow Matching 训练逻辑；其他 loss 来自第五章的数据工程自动或半自动构造。

## 6.6 P0 候选路线横向对比

| P0 候选路线 | 核心机制 | 候选底座 / 工程来源 | 优点 | 风险 | 推荐等级 |
|---|---|---:|---|---|---|
| **lightweight WAM heads（可选复用 Qwen3VL[29]）** | 在视觉语言 / 视觉状态 backbone 上加 progress/readiness/risk 等 future heads | Qwen3VL[29] / Qwen-VL[30] / StarVLA[31] 是候选实现 | 复用 VLA 策略项目资产，闭环最快 | 容易被误写成 VLA action head 改装，需保持 WAM 分支独立 | **推荐主线** |
| JEPA-style future latent auxiliary | 当前观测预测 future feature / future state | 可选 | 表征学习更强，避免像素生成 | target 构造和防泄漏要谨慎 | 推荐增强 |
| Fast-WAM-style train-time future supervision | 训练时引入 future loss，推理时不生成 | 可选 | 延迟低，符合 P0 | 需证明 downstream gain | 推荐参考 |
| AdaWAM-style gating | 根据阶段/风险触发不同模块 | 可选 | 可连接 P1/P2 | 完整复现复杂 | 简化采用 |
| 直接 VLA action head | 只预测动作 | VLA 策略项目 | 简单 | 不是 WAM | 不作为本项目贡献 |

## 6.7 本章路线判断与注意点

P0 的最终定位是：**用最小工程代价验证 future supervision 是否能提升移动操作任务中的阶段判断、可操作性判断和失败预警**。它是 P1/P2 的对照基线，也是本项目三个月内最稳的第一闭环。

后续设计中需要注意三点。第一，P0 的贡献不能写成“Qwen3VL[29] 做动作预测”，而要写成 WAM-specific future supervision。第二，P0 标签必须可追溯到数据工程流程，不能手写经验规则后又无法复现。第三，P0 与 P1 比较时应共享相同数据域权重和下游 action head，否则无法判断收益来自路线差异还是数据差异。

# 第七章 P1：Latent-Only WAM 路线调研

## 7.1 本章目的、作用与承接关系

第六章讨论了不生成视频的轻量 P0 路线。本章讨论更进一步的 P1：利用视频生成基础模型的 latent space 和视频 dynamics prior，在不解码像素的前提下构建 latent-only WAM。

P1 在全文中的作用是连接 P0 和 P2。相对 P0，P1 希望获得更丰富的 future representation；相对 P2，P1 又保持 latent-only，不把任务推进到昂贵的像素生成。P1 的成败将决定本项目是否能够从 lightweight future heads 走向更具扩展性的 world-action latent modeling。

本章重点不是泛泛比较所有视频生成模型，而是讨论 Latent-Only WAM 的可行路线：latent target 来自哪里、future latent 由什么条件预测、历史多模态信息如何被压缩为 compact history condition tokens、以及是否继承 pretrained video DiT 的 dynamics prior。Wan2.2 TI2V-5B[14] 是当前优先评估的候选视频基础模型，但不是 P1 的正式阶段名称。

## 7.2 研究发展：从 latent future target 到 pretrained video DiT adaptation

P1 类路线的发展可以分为三个层次。

第一层是 **future latent target / latent visual subgoal**。这类方法不直接生成像素，而是用预训练视觉模型或视频模型把未来帧编码为 latent target，再训练策略或世界模型利用这个 future latent。它的优点是训练和推理成本低于视频生成，缺点是如果只用视觉和语言预测未来，容易变成 task-conditioned future prior，而不是机器人历史条件化 WAM。

第二层是 **history-conditioned future prediction**。这类方法不仅看当前图像和语言，还引入历史 state/action、接触信息、关节状态或相机运动历史。它更符合机器人任务，因为机器人未来视觉并不只由图像和语言决定，还与底盘、手臂、夹爪和最近动作趋势有关。移动操作尤其需要这类条件，因为相机变化往往由机器人运动本身造成。

第三层是 **pretrained video DiT adaptation**。随着 Wan 系列[5] / Wan2.2[14]、Cosmos[6]等视频基础模型出现，从零训练 world model 的必要性下降。更合理的做法是继承视频模型的 VAE latent space、text condition 和 DiT video dynamics prior，再通过 LoRA[25]/adapter 注入机器人条件。这样既保留大模型先验，又降低机器人数据不足导致的从头训练风险。

### 7.2.1 研究发展谱系与典型工作补充

P1 的核心不是“是否使用 Wan2.2[14] latent”这么简单，而是 latent 来自哪里、future latent 由什么条件预测、是否继承 pretrained video DiT 的 dynamics prior。相关研究可以按五类理解：

| 发展类别 | 核心思想 | 典型工作 | 对 P1 的启发 |
|---|---|---|---|
| **Latent visual subgoal / future prior** | 当前视觉和语言预测 future latent，作为策略或 action head 的条件 | LaWAM [2]、VLA-JEPA [3]、Light-WAM [15] | 对应 P1-b0：低风险验证 future latent 是否有下游价值 |
| **Pretrained video foundation latent substrate** | 使用视频生成模型的 VAE latent、text condition 和 DiT prior，而不是从零定义 latent space | Wan / Wan2.1[5]、Wan2.2 TI2V-5B[14] | 决定 P1 使用 Wan2.2-VAE、Wan2.2[14] text encoder 和 TI2V-5B DiT prior |
| **History latent compression / injection** | 将历史 RGB/video latent、语言 condition、历史 robot state/action condition tokens 压缩为 compact history tokens，再注入 future latent prediction | Flamingo Perceiver Resampler[88]、T2I-Adapter[89]、ControlNet[90]、ChronoDreamer [17]、1X World Model Challenge[16] 中的 Wan2.2[14] state-conditioned adaptation [16]、Mem-World[91] | 对应 P1-b1：核心不是简单 concat 历史，而是 history latent 的压缩、对齐与稳定注入；future action 防泄漏是工程 invariant |
| **Video-action coupled / candidate-action evaluator** | 把视频生成和动作生成/动作评估放在同一世界模型中，或用候选动作预测后果 | UWM [8]、OSCAR [18] | 后期可用于 action scoring / counterfactual evaluation，但不作为当前默认闭环 |
| **Adaptive / omnimodal world model** | 统一多模态或按需触发更重的 world reasoning | Cosmos 3[6]、AdaWAM [9]| 提供长期架构参考，不作为三个月内实现目标 |

第 7.4 节中的三类“值得参考工作”来自上表的筛选：P1-b0 重点参考 latent visual subgoal / future prior；P1-b1 重点参考 history latent compression / injection、history-conditioned future prediction 与 Wan2.2[14] state-conditioned adaptation；candidate-action evaluator 只作为后期增强参考。

## 7.3 Wan2.2 TI2V-5B[14] 作为候选视频基础模型的调研判断

Wan2.2 TI2V-5B[14] 是 P1 当前最适合作为优先候选的视频基础模型之一。它同时支持 text-to-video 与 image-to-video，具备 high-compression Wan2.2-VAE，并以 TI2V-5B DiT 承载视频 latent dynamics prior。对本项目而言，Wan2.2[14] 的价值不是“能生成漂亮视频”，而是提供三个可继承基础：

1. **视觉 latent space**：用 Wan2.2-VAE encoder 将当前/未来机器人视频片段编码为 latent；
2. **语言条件空间**：用 Wan2.2[14] 体系内 T5/UMT5 text encoder 编码任务语言；
3. **视频 dynamics prior**：用 TI2V-5B DiT 继承从大规模视频中学到的时序变化先验。

但 Wan2.2[14] 原生不是机器人 WAM。它缺少 robot state、base pose、EEF、joints、gripper、force/tactile、success/failure、manipulation-readiness 和历史 action 等条件。因此，本项目不能直接把 Wan2.2[14] 当作完整 WAM，而应把它作为 latent world foundation model，在其上补充机器人历史条件和 WAM-specific 评测。

**技术边界与参数口径。** 根据官方仓库说明，Wan2.2[14] 的 TI2V-5B 是 5B 级别的 text-image-to-video 模型，使用 high-compression Wan2.2-VAE，压缩设计写作 16×16×4，并支持 720P、24fps 的 T2V/I2V。这里的 16×16×4 应在本报告中理解为时空压缩比 / 压缩设计，而不应直接误写成“单帧 latent 等于 1024 维向量”。P1 详细设计必须以实际代码中的 VAE latent tensor shape、patchification 方式、history window 和 future horizon 实测为准。

对本项目而言，Wan2.2[14] 的关键不是完整视频生成速度，而是：1）VAE encoder 能否稳定抽取 current/future latent target；2）TI2V-5B DiT 前向是否能在不解码像素的情况下承担 latent future prediction；3）LoRA[25]/adapter 注入历史条件后，是否能在可接受延迟内带来 downstream gain。因此，P1-b1 的显存、吞吐和延迟必须在详细设计初期通过小 batch profile 实测，不应直接引用视频生成端到端耗时来判断 latent-only 路线。

这也解释了为什么 P1 不默认使用 Qwen-VL[30]。既然 P1 的视觉 latent 和语言条件都来自 Wan2.2[14] 体系，那么再引入 Qwen-VL[30] 会增加表征对齐负担，并把 P1 拉回 VLM/VLA 路线。Qwen3VL[29] 更适合 P0 和VLA 策略项目，P1 则优先保持候选视频基础模型体系内部一致；如果后续替换为其他 latent video foundation model，也应遵循同样原则。

## 7.4 值得参考的工作类型及优劣分析

上一节给出 P1 的五类研究谱系；本节只展开其中对本项目路线选择最关键的三类。P1-b0 重点参考视觉/语言 future prior，P1-b1 重点参考 history-conditioned future prediction 与 Wan2.2[14] state-conditioned adaptation，candidate-action evaluator 只保留为后期增强边界。


现有相关工作可以分为三类。

**第一类：视觉/语言 future prior。** 这类方法只用当前视觉和语言预测 future latent 或 future video，再让 action head 消费 future representation。它的优点是工程风险低，不需要改视频生成模型流程；缺点是 world model 没有看到机器人历史动作和状态，因此更像 latent visual subgoal provider，而不是完整机器人 WAM。本项目可将其作为 P1-b0 低风险基线。

**第二类：历史状态/动作条件化 future prediction。** 这类方法使用历史视觉、历史 state/action、语言共同预测未来视频、future latent 或 future state。它更符合机器人时序建模，因为历史动作说明了相机为什么变化、末端是否正在接近目标、夹爪是否刚闭合、底盘是否正在调整视角。本项目的 P1-b1 应归入这一类。

**第三类：候选动作后果评估。** 这类方法输入当前观测和 candidate future action，预测该候选动作的后果，用于 action scoring 或反事实评估。它最接近严格的 action-conditioned rollout，但工程复杂、推理成本高，且需要 action proposal / selector 机制。本项目当前不建议把 P1-b1 做成这一类，后期可以作为增强方向。

三类路线的关系如下：

| 路线类别 | 典型做法 | 对本项目的含义 |
|---|---|---|
| 视觉/语言 future prior | 当前观测和语言预测 future latent，action head 再消费 future latent | 对应 P1-b0，适合作低风险基线 |
| 历史状态/动作条件化未来预测 | 历史视觉、历史 state/action、语言共同预测未来 latent/state | 对应 P1-b1，建议作为推荐主线 |
| 候选动作后果评估 | 当前观测 + candidate future action → predicted outcome | 后期增强，不作为当前 P1 默认闭环 |

## 7.5 P1-b0 与 P1-b1 的路线判断

**P1-b0** 不修改 Wan2.2[14] 生成流程，只使用视觉和语言条件预测 future latent：

```text
current / history RGB
→ Wan2.2-VAE encoder
→ current Wan latent

language
→ Wan2.2 T5/UMT5 text encoder
→ text condition

current Wan latent + text condition
→ Wan2.2 future latent prior
→ predicted future Wan latent

predicted future Wan latent + robot state/action history
→ action head / WAM-specific heads
```

这条路线适合建立低风险基线。它的主要价值是验证 Wan2.2[14] future latent 是否能作为 action head 的额外条件。其限制是 WAM 本身没有接收机器人历史 state/action，因此对移动操作中的相机运动、底盘趋势和夹爪状态不敏感。

**P1-b1** 是本报告建议的推荐主线。其核心不再表述为“历史 state/action encoder”本身，而是 **History Latent Compressor + Gated Condition Injection**：

```text
history RGB / short video
→ Wan2.2-VAE encoder
→ visual history latent

language
→ Wan2.2 T5/UMT5 text encoder
→ text condition tokens

history robot state + executed history action
→ state/action projector 或 adapter
→ robot history latent / robot condition tokens

visual history latent + text condition tokens + robot history latent
→ History Latent Compressor
→ compact history condition tokens

compact history condition tokens
→ Gated Condition Injection / cross-attention / adapter
→ Wan2.2 TI2V-5B DiT-LoRA
→ predicted clean future Wan latent

predicted future Wan latent + current/history robot state/action
→ action head / WAM-specific heads
```

这里输入给 WAM 的 action 是历史 action，不是数据集中的未来 action label。更准确地说，future action 防泄漏不是一个独立实验，而是 dataset slicing、batch construction 和 unit test 必须保证的工程 invariant。历史 RGB/video、语言、机器人 state/action 都应被转化为与 Wan2.2[14] DiT condition / hidden space 对齐的 history latent 或 condition tokens；真正需要研究的是这些 history latent 如何压缩、如何稳定注入、以及是否带来 downstream gain。


## 7.6 P1-b 历史 latent 压缩与注入方式调研

本节补充 P1-b 的关键机制调研。前文已经说明 P1-b1 不能只是“在 Wan2.2[14] 上加一个历史 state/action encoder”。更准确的问题是：**如何将历史 RGB/video、语言和机器人 state/action 转化为与 Wan2.2 DiT 条件空间对齐的 history latent / condition tokens，并在不破坏预训练视频 prior 的前提下稳定注入 future latent prediction 过程。**


### 7.6.0 来源边界：HLC-GCI 是本项目组合方案，不是已有工作名

需要先明确本文对 **History Latent Compressor + Gated Condition Injection（HLC-GCI）** 的使用边界。HLC-GCI 不是当前文献中已经固定使用的标准方法名，也不是对某一篇论文或某一个开源项目的直接复现。它是本项目为了 P1-b1 路线提出的候选实现方案，用来概括两个紧密相连的设计问题：

```text
1. History Latent Compressor：
   如何把历史 RGB/video latent、语言 condition tokens、robot state/action latent
   压缩为固定长度 compact history condition tokens。

2. Gated Condition Injection：
   如何把 compact history condition tokens 稳定注入 Wan2.2 DiT-LoRA 的 condition path，
   同时尽量不破坏预训练视频 latent dynamics prior。
```

该组合方案的组成思想分别有已有工作支撑，但完整组合与命名属于本项目设计。已有工作提供的支撑包括：

| 组成思想 | 代表来源 | 对本项目的启发 |
|---|---|---|
| learned-query / Perceiver-style compression | Flamingo / Perceiver Resampler[88] | 用少量 learnable query 压缩可变长度视觉 / 视频 token，避免直接拼接长历史 token |
| adapter-based condition alignment | T2I-Adapter[89] | 冻结大模型主干，通过轻量 adapter 对齐外部控制信号与预训练模型内部表征 |
| gated / zero-initialized condition injection | ControlNet[90] 及相关条件注入路线 | 外部条件应以渐进、稳定的方式进入预训练扩散 / DiT 模型，避免训练初期破坏已有 prior |
| latent future prediction / latent WAM | LaWAM[2]、VLA-JEPA[3] | future representation 可以在 latent space 预测并被策略消费，而不必在推理时解码完整像素 |
| history / memory selection | Mem-World[91] 等 memory-augmented world model | 长历史选择与压缩对未来预测有价值，但复杂 retrieval 不进入当前三个月主线 |

因此，本文后续如使用 “HLC-GCI”，均表示“本项目建议的 P1-b1 历史 latent 压缩与注入方案”，而不是“已有工作已经提出 HLC-GCI”。正式详细设计和后续论文 / 简历表述中也应遵守这一来源边界：可以说“基于已有多模态压缩、adapter 条件注入和 latent WAM 技术脉络，设计 HLC-GCI 作为本项目 P1-b1 的默认实现”，不能说“复现已有 HLC-GCI 方法”。

### 7.6.1 问题定义：P1-b 不是简单 state/action 注入

P1-b0 的输入主要是当前或短历史视觉 latent 与语言 condition，因此更像视觉/语言 future latent prior。P1-b1 的目标则是让 future latent prediction 受机器人历史影响：相机为何移动、底盘是否接近目标、末端是否进入操作姿态、夹爪是否刚闭合、历史动作是否导致遮挡或接触变化。这些信息不能只由当前 RGB 和语言恢复。

因此，本项目将 P1-b1 的输入统一表述为多模态 history latent：

```text
history RGB / video → Wan2.2-VAE encoder → visual history latent
language instruction → Wan2.2 text encoder → text condition tokens
history robot state/action → trainable projector / adapter → robot history latent
```

其中，RGB/video 对应 Wan2.2-VAE latent；语言和机器人状态/动作不应误写为 Wan-VAE latent，而应称为 text condition tokens 与 robot history latent / robot condition tokens。三者最终都需要对齐到 Wan2.2[14] DiT 可消费的 condition / hidden token space。

### 7.6.2 发展脉络一：从直接拼接到 learned-query compression

历史信息天然是长序列。若直接将多个历史帧的 latent patches、文本 token 和机器人状态 token 全部 concat 到 DiT condition 中，token 数会随历史窗口线性增长，推理延迟、显存和 cross-attention 成本都会快速上升。更合理的方向是 learned-query compression：用少量可学习 query 从长历史 token 中读取任务相关信息，得到固定长度 compact history condition tokens。

Flamingo[88] 的 Perceiver Resampler 提供了重要范式：它用固定数量的 latent queries 对可变数量的视觉输入进行重采样，使模型能够处理 interleaved image/video 与文本，同时控制视觉 token 数。该思想对本项目的启发是：P1-b1 不应直接注入全量 history visual latent，而应通过 History Latent Compressor 将历史视觉、文本和机器人条件压缩为固定长度 history condition tokens。

### 7.6.3 发展脉络二：从单模态历史到多模态 history latent 对齐

机器人移动操作的未来变化由多种因素共同决定。视觉历史说明“看见过什么”和“场景如何变化”；语言 condition 说明任务目标；机器人 state/action 历史说明“机器人做了什么”以及“相机和末端为何发生变化”。LaWAM[2] 等 latent WAM 工作说明 latent future subgoal 可以作为低延迟策略条件；VLA-JEPA[3] 则强调 future latent 作为监督目标时要避免学生路径看到未来信息。

本项目的 P1-b1 在此基础上进一步强调 history latent 对齐：机器人 state/action 不经过 Wan-VAE，而是通过 trainable adapter 映射到与 Wan2.2 DiT condition space 相容的 robot history latent。这样既保留视频 foundation model 的视觉 latent prior，又让未来预测受已执行动作和机器人状态变化影响。

### 7.6.4 发展脉络三：从普通条件注入到 adapter / gated injection

在大规模扩散或视频生成模型上加入外部条件，直接修改主干参数通常风险较高。T2I-Adapter[89] 的核心思想是冻结原始大模型，只训练轻量 adapter 将外部控制信号对齐到预训练模型内部知识；ControlNet[90] 则通过锁定原始扩散模型并使用零初始化连接，使外部条件在训练初期不会破坏已有生成能力。这些思路共同指向一个原则：**外部机器人 history latent 应通过 adapter、cross-attention 或 gate 渐进注入，而不是直接扰动 Wan2.2 主干。**

因此，本项目建议 P1-b1 采用本项目设计中的 Gated Condition Injection：

```text
C_hist = HistoryLatentCompressor(visual_history_latent, text_condition, robot_history_latent)
g = sigmoid(MLP(pool(C_hist)))
C_cond = concat(C_text, C_current, g · Project(C_hist))
```

这里的 gate 建议初始化接近 0，使训练初期模型退化为接近 P1-b0 / 原始 Wan condition prior；随着训练推进，模型再逐步学习使用 history latent。默认优先注入 DiT-LoRA 的 condition path / cross-attention KV，而不是一开始改动 latent self-attention 主路径。

### 7.6.5 发展脉络四：memory-augmented history retrieval

固定窗口压缩并不是唯一方案。Mem-World[91] 指出，在复杂操纵中，末端遮挡和 wrist-camera 快速运动会使当前观察不足以预测未来，因此它引入 memory-augmented、multi-view、action-conditioned world model，通过几何化记忆检索相关历史。该方向说明“历史选择”本身可能成为世界模型的关键能力。

但对本项目三个月周期而言，复杂几何 memory、surfel retrieval 或 long-term memory 不应进入近期主线。它们更适合作为后续扩展。当前主线应先采用固定窗口 + learned-query compression 的中间路线，在工程可控的前提下验证 history latent 注入是否带来收益。


### 7.6.6 发展脉络五：历史压缩时间尺度与记忆更新策略

前述小节讨论了 history latent 的模态对齐和压缩方式，但还需要回答一个更基础的问题：**到底压缩多长时间的历史，以及历史记忆如何随时间更新**。这直接影响 P1-b1 的工程复杂度、显存成本、训练稳定性和长期信息保留能力。

本节按照技术调研报告口径讨论路线，不固定具体窗口长度、stride、token 数量或缓存实现；这些实现级参数应在详细设计文档中结合数据帧率、episode 长度、任务阶段和显存 profile 决定。

#### 7.6.6.1 问题定义：历史不是越长越好

移动操作任务中，历史信息确实重要：目标物可能曾经出现但当前被遮挡，机器人视角变化会造成短时观测缺失，导航到操作的交接也依赖过去的接近过程。但历史不是越长越好，原因包括：

1. **token 与显存成本增长**：历史 RGB/video latent、机器人状态和动作都会增加 condition token 数量；
2. **信息稀释**：过长历史中大量无关移动、等待或重复帧会稀释关键事件；
3. **训练 / 推理不一致风险**：如果训练时能访问完整 episode 历史，而推理时只能在线维护有限记忆，会造成分布偏移；
4. **递归压缩误差累积**：如果持续用“旧压缩记忆 + 新历史”再压缩，长期记忆可能逐步漂移或遗忘关键细节；
5. **任务相关性不均匀**：对抓取前精细操作，最近几秒通常更重要；对遮挡、回访和跨房间移动，更早历史可能有价值。

因此，本项目需要区分“短中期历史窗口”和“长期 episode memory”两个问题，不能简单假设保留全部历史一定更优。

#### 7.6.6.2 路线一：固定长度滑动窗口

最直接的路线是固定长度 sliding window：每次只取最近一段历史 RGB/video、state 和 action，再进行 learned-query compression。Transformer / video / robot policy 中大量方法采用固定窗口或固定上下文，因为它实现简单、训练推理一致、显存可控，并且容易与 batch 训练对齐。

该路线的优点是稳定、便宜、可解释，适合本项目主线。缺点是无法保留窗口之外的早期信息，遇到长时间遮挡、回访目标物或跨阶段任务时可能不足。

对本项目而言，sliding window 适合作为 P1-b1 的默认路线，但窗口长度不应在调研报告中写死，而应在详细设计中由 Data Gate / Profiling 决定。更合理的描述是：以“秒级覆盖范围 + 采样 stride + 关键事件覆盖率”定义历史长度，而不是只用原始帧数定义。

#### 7.6.6.3 路线二：事件感知窗口与关键帧补充

第二类路线是在 sliding window 基础上增加 event-aware selection，例如保留目标物首次出现、接近目标物、base 停止、arm 开始动作、gripper 接触、失败恢复等关键帧。这与 TokenLearner 类工作中“自适应选择少量重要视觉 token”的思想相近：不是处理所有密集 patch / frame，而是学习或规则化地提取少量高价值 token / frame[95]。

该路线比完整长期 memory 更轻量，也更符合移动操作中“关键事件比均匀历史更重要”的特点。缺点是需要额外定义事件或重要性评分，可能依赖数据字段和任务阶段标注。

对本项目而言，event-aware window 更适合作为 sliding window 的轻量增强：默认仍使用最近历史窗口，必要时补充少量低频关键帧或 episode summary tokens。

#### 7.6.6.4 路线三：递归 / 压缩式长期记忆

第三类路线是维护一个可递归更新的长期 memory：将旧压缩记忆与新历史片段再次压缩，以期保留整个 episode 的信息。Transformer-XL 使用 segment-level recurrence 来突破固定上下文限制[96]；Compressive Transformer 进一步将过去 memory 压缩后保留为更长期的 compressed memory[97]；Recurrent Memory Transformer 通过 memory tokens 在片段之间传递信息[98]。这些工作证明了“分段处理 + 记忆传递 / 压缩”是长序列建模中的重要方向。

但直接将这类路线用于本项目 P1-b1 存在风险：移动操作中的历史视觉与机器人动作是多模态、高维、非平稳的，递归压缩可能积累错误；如果没有专门的 memory supervision，模型很难保证旧记忆中保留的正是未来动作需要的信息；同时训练和在线推理的 memory 更新逻辑也更复杂。

因此，本项目不应默认采用“旧压缩记忆 + 新历史递归压缩以保留所有历史信息”。它可以作为长期扩展或条件触发增强，但不适合作为第一版主线。

#### 7.6.6.5 路线四：检索式 / 几何式长期记忆

第四类路线是将长期历史存入 memory bank，推理时按相关性检索。Mem-World 针对机器人操纵中的遮挡和 wrist-camera 快速运动问题，提出 memory-augmented multi-view action-conditioned world model，并使用几何化 memory 检索相关历史帧以改善持续性建模[91]。这类路线对于长时遮挡、跨阶段回访、场景持久性建模非常有吸引力。

但它需要额外的 memory indexing、检索评分、几何或多视角对齐机制，工程量较大，也可能要求更完整的相机位姿或场景结构信息。结合本项目当前约束，retrieval memory 不建议进入三个月主线，但应作为未来扩展方向保留接口。

#### 7.6.6.6 路线五：状态空间 / Mamba-style streaming history

第五类路线是使用状态空间模型或 Mamba-style sequence model 对长历史做流式压缩。Mamba 通过 selective state spaces 实现线性时间序列建模，并强调根据输入选择性传播或遗忘信息[99]。机器人方向也出现了类似思路，例如 AEM 将视觉与动作历史交错建模，通过 Mamba 编码获得 compact history representation，用于下游控制[100]。

该路线对“长期、高频、在线历史压缩”很有吸引力，尤其适合机器人连续控制。但在本项目中，它会引入新的时序 backbone，增加与 Wan DiT / HLC-GCI 的集成复杂度。更稳妥的做法是将 Mamba-style history encoder 作为未来替代 History Latent Compressor 的候选，而不是当前默认路线。

#### 7.6.6.7 路线比较与本项目收敛

| 路线 | 核心思想 | 优点 | 风险 | 本项目定位 |
|---|---|---|---|---|
| 固定滑动窗口 | 只压缩最近一段历史 | 简单、稳定、训练推理一致、显存可控 | 无法保留窗口外信息 | 默认主线 |
| 事件感知窗口 / 关键帧补充 | 最近窗口 + 关键事件帧 / summary | 轻量保留重要历史，适合遮挡和阶段切换 | 需要事件定义或重要性评分 | 轻量增强 |
| 递归 / 压缩式长期记忆 | 旧压缩记忆 + 新片段再压缩 | 理论上可覆盖更长 episode | 误差累积、训练复杂、难保证保留有效信息 | 暂缓 / 条件触发 |
| 检索式 / 几何式 memory | memory bank + relevance retrieval | 适合长期遮挡、回访和场景持久性 | 工程量大，依赖位姿 / 几何或检索机制 | 长期扩展 |
| Mamba / SSM streaming memory | 线性时间流式历史编码 | 高效长序列建模，适合在线控制 | 新增 backbone，集成复杂 | 长期扩展 / 替代 HLC 候选 |

本项目的合理收敛路线是：

> P1-b1 默认采用短中期 sliding window history compression，以 learned-query / Perceiver-style 压缩为核心；在数据支持时，可补充少量事件关键帧或低频 episode summary。暂不默认采用递归压缩所有历史的长期 memory，也不默认引入 retrieval memory 或 Mamba-style streaming encoder。具体历史时间长度、采样 stride、窗口覆盖秒数、关键帧数量和是否启用轻量 summary，应在详细设计中通过 Data Gate / Profiling 决定。

这一收敛与本项目整体策略一致：先建立稳定、可解释、可训练的 P1-b1 链路，再在必要时引入长期 memory 扩展。



### 7.6.7 发展脉络六：从 episode 到训练样本——窗口采样、future target 与训练/推理一致性

历史压缩时间尺度确定之后，还需要明确训练样本如何从完整 episode 中构造。对于本项目的 P1-b 路线，完整 episode 更适合作为数据容器、采样单位和离线缓存单位，而不是直接作为单次模型 forward 的输入。

#### 7.6.7.1 全 episode 输入的风险

将完整 episode 直接输入模型，看似可以保留全部历史，但在移动操作中存在明显风险：episode 长度差异大，batch padding 和显存浪费严重；视频 latent token 数随 episode 时长线性增长，Wan DiT activation 成本难以控制；大量移动、等待或重复帧与当前动作弱相关；训练阶段如果可看到完整 episode，而推理阶段只能在线看到过去，会造成训练/推理不一致；同时 future observation / future action 泄漏风险更高。

因此，完整 episode 不应默认作为单次模型输入。更合理的方式是：episode 作为轨迹容器，训练时采样 anchor timestep，并围绕该时间点切出 history input 与 future/action target。

#### 7.6.7.2 Anchor-based temporal window sampling

本项目推荐的训练样本构造方式是：

```text
episode e
→ sample anchor timestep t
→ recent history window before / at t
→ optional sparse summary or memory from earlier prefix
→ future latent target after t
→ action chunk target from t
```

这种采样方式与 action chunk / receding-horizon policy 训练更一致。训练时可从同一 episode 产生多个 anchor 样本，提高数据利用率；推理时则用 rolling history buffer 维护同样形式的 recent history，保证训练/推理一致。完整 episode 仍用于 dataset / domain / task-balanced sampling、anchor sampling、stage / event / progress label 构造、Wan VAE latent cache、language embedding cache、offline statistics 和 evaluation replay。

#### 7.6.7.3 Anchor sampling：uniform 不是唯一选择

移动操作 episode 中不同时间点的信息密度不同。纯 uniform anchor sampling 可能过采样长时间导航、等待或低变化片段，而低估目标出现、接近、接触、失败恢复等关键阶段。因此，建议采用：

```text
uniform anchor sampling
+ optional stage-balanced / event-aware oversampling
```

可优先关注目标物首次出现、导航到操作交接、base 停止与 arm 开始动作、gripper 接触 / 开合、失败恢复和任务阶段边界。具体 event 定义、采样权重和标签来源应下沉到详细设计文档，由 Data Gate 根据数据字段决定。

#### 7.6.7.4 SparseVideoNav 工程 profiling 参考

用户当前执行的 SparseVideoNav 提供了重要工程参考：

```text
backbone: Wan2.1 T2V-1.3B
training: full fine-tune
input resize: 256×256
video fps: 4Hz
history: 16 frames ≈ 4s
future prediction: 28 frames ≈ 7s
chunk size: 4 frames ≈ 1s
history handling: direct concat
language: T5 编码后移出 GPU / offload
per-device batch size: 2
peak GPU memory: ≈70G
```

该结果不能简单归因于 history direct concat。更合理的解释是：约 70G 峰值显存由 full fine-tune 的 persistent training states、batch=2 的 44-frame video DiT activation、history direct concat 带来的 activation 增量、future prediction activation、attention workspace、runtime buffer 和 fragmentation 共同构成。

这条工程经验对本项目的启发是：full fine-tune + long video span + direct history concat 的组合扩展性有限；即使输入分辨率为 256×256，44-frame temporal span 的 video DiT 训练 activation 仍然很大；本项目若使用更大 Wan backbone 或引入 robot history condition，默认不应采用 full fine-tune + long dense history concat；SparseVideoNav 的 4Hz / 16-frame history / 28-frame future / 4-frame chunk 可作为 P1-b Data Gate / Profiling 的初始参考点。

#### 7.6.7.5 与拟采用数据集 episode 长度 / Hz 的关系

是否需要 episode-level summary 或 recurrent memory，不能只由模型设计决定，还必须由数据统计决定。详细设计阶段应统计每个拟采用数据集的原始 observation fps、action Hz、episode 帧数分布、episode 秒数分布、navigation / approach / manipulation / recovery 阶段时长、目标遮挡 / 回访 / 失败恢复等长程依赖比例，以及不同 history window 对 episode 和任务阶段的覆盖比例。

核心判断应基于：

```text
history_seconds = history_frames / model_video_fps
coverage_ratio = history_seconds / episode_seconds
```

如果 4 秒窗口能覆盖大多数操作阶段，则 sliding-window HLC 足以作为默认主线。如果 episode 很长、4 秒窗口覆盖率低，且存在长遮挡 / 回访 / 导航-操作长间隔，则应优先加入 sparse episode summary、event-aware keyframes 或 Rec-HLC，而不是直接输入 dense full episode。

#### 7.6.7.6 调研结论

本项目的默认训练数据构造应收敛为：

```text
episode-level container
+ anchor timestep sampling
+ recent sliding history window
+ optional sparse past summary / memory
+ future latent target
+ action chunk target
```

完整 episode 不默认进入模型 forward；它用于采样、缓存、标签构造、统计和评估。只有当 Data Gate 显示 episode 短、下采样后 token 数可控，并且训练/推理一致性可以保证时，才可考虑 whole-episode compressed input。对于长程移动操作，更推荐 recent window + sparse summary / recurrent memory，而不是 dense full-episode input。


### 7.6.8 发展脉络七：Recurrent / Compressive History Latent Memory

用户提出的“在前一步或前几步压缩得到的 latent 基础上，补充新周期信息，再压缩成同样长度 latent”的方法是存在的，可归入 recurrent memory、compressive memory 或 memory token 路线。本项目可将其命名为 Recurrent History Latent Compressor，简称 Rec-HLC。其抽象形式是：

```text
M_{k-1}: 上一周期压缩后的历史 memory
x_k: 当前周期新增的 visual / state / action latent
M_k = Update(M_{k-1}, x_k)
```

其中 `M_k` 长度固定，可作为当前时刻的长期历史条件输入。

#### 7.6.8.1 相关工作脉络

Transformer-XL 通过 segment-level recurrence 在片段之间传递记忆，用于突破固定上下文限制[96]。Compressive Transformer 进一步将更早的 memory 压缩为 compressed memory，以支持更长程依赖[97]。Recurrent Memory Transformer 使用 memory tokens 在片段之间传递信息，与本项目中固定长度 history latent tokens 的设想更接近[98]。机器人和 WAM 方向也在出现相关探索，例如 Mem-World 针对遮挡和视角变化引入 memory-augmented world model 与历史检索机制[91]，HiMem-WAM 面向长程 manipulation 引入 boundary-triggered memory updates 和 compact task states[101]。

#### 7.6.8.2 Rec-HLC 能解决什么

Rec-HLC 的吸引力在于，它可以让模型以固定长度 memory 接收从 episode 起点到当前时刻的历史流，而不需要把所有历史 dense tokens 同时输入 Wan DiT。它适合处理目标物早期出现但后续被遮挡、导航阶段看到目标但操作阶段目标不在视野中心、长程导航-操作交接、失败恢复依赖更早动作、episode 较长而 recent window 覆盖率低等情形。

#### 7.6.8.3 Rec-HLC 的风险

Rec-HLC 不能被表述为“无损输入整个 episode”。更准确地说，它是将整个 episode 的历史前缀递归压缩为固定容量 memory。因此它存在信息瓶颈、误差累积、训练复杂度、训练/推理一致性和诊断困难等风险。训练时若从 episode 起点滚动更新 memory，会增加计算成本；若离线预缓存 memory，又需要处理 stale memory、teacher-forcing、detach 或 truncated BPTT 等问题。推理时 memory 必须在线更新，不能依赖 oracle full-episode summary。

#### 7.6.8.4 本项目中的合理定位

本项目更稳妥的分级是：

| 路线 | 压缩范围 | 更新方式 | 本项目定位 |
|---|---|---|---|
| Sliding-window HLC | 最近 L 秒 | 每个 anchor 独立压缩 | P1-b1 默认主线 |
| Sliding-window + sparse summary | 最近 L 秒 + 低频关键帧 / summary | 非递归或低频更新 | P1-b1 可选增强 |
| Rec-HLC | 从 episode 起点到当前前缀 | `M_k = Update(M_{k-1}, x_k)` | P1-b2 / Conditional |
| Retrieval / geometry memory | 全 episode memory bank | 检索相关历史 | Deferred / 长期扩展 |

推荐路线是先完成 P1-b1 的 recent sliding-window HLC，验证 history compression + Wan DiT condition injection 是否有效；只有当 Data Gate 显示 recent window 覆盖率不足，且任务确实存在长程记忆需求时，再进入 P1-b2 的 Rec-HLC。

#### 7.6.8.5 Rec-HLC 的最小候选设计边界

如果进入 P1-b2，Rec-HLC 应采用最小实现：以 1 秒或若干帧为 memory update segment；memory tokens 长度固定；使用 `stopgrad(M_{k-1})` 或 truncated BPTT 控制显存；memory 在 episode boundary reset；memory update 只使用当前及过去信息；recent dense window 仍然保留，用于避免近端细节被长期递归压缩损失；Rec-HLC 只作为 Conditional 实验进入 Experiment Registry。

调研报告层面的收敛结论是：Rec-HLC 可以让模型以固定长度 memory 覆盖整个 episode 前缀，但不能保证无损保留全部历史信息。它是 long-horizon mobile manipulation 的有价值增强方向，但不应替代 P1-b1 默认的 sliding-window HLC。

### 7.6.9 发展脉络八：history latent 注入位置——从条件控制到动作收益

前述小节回答了“历史如何被压缩为可消费的 latent / condition 表征”，但 P1-b1 的实际价值还取决于这些历史表征如何进入模型链路。对本项目而言，history latent 注入位置需要同时考虑两个层面：一是它如何影响 Wan2.2 DiT 的 future latent prediction，二是它如何被动作生成模块消费并转化为 downstream action gain。

本节只做技术调研报告层面的路线分析，不固定具体接口变量、张量公式或 action head 内部拼接实现。这些实现级内容应在详细设计文档中展开。

#### 7.6.9.1 问题定义：注入位置是 WAM 是否产生动作收益的关键中介

Latent WAM 的目标不是单纯降低 future latent prediction loss，而是使 future-aware / history-aware 表征最终服务于机器人动作生成。如果 history latent 只改善了内部预测指标，却无法被 action head 有效消费，项目收益会停留在 representation 层，无法体现到任务成功率、导航-操作交接、恢复能力或长时序稳定性上。

因此，history latent 注入位置本质上是一个“表征—预测—动作”的中介问题：

```text
history latent 是否改善 future latent prediction？
↓
history-aware future representation 是否被 action head 消费？
↓
是否带来 downstream action gain？
```

技术调研报告需要回答的是“有哪些路线、各自风险是什么、为什么本项目应收敛到哪一类路线”；具体 token 名称、公式、层号和模块实现应留给详细设计。

#### 7.6.9.2 相关路线一：Wan DiT 侧 condition-path 注入

第一类路线是将 compressed history representation 作为额外条件，接入 Wan DiT 原有的 text / image condition 路径。该路线与 T2I-Adapter[89]、ControlNet[90] 等可控生成工作中的思想一致：尽量冻结或少改预训练生成主干，通过轻量 adapter、gated connection 或 zero-initialized connection 将外部控制信号对齐到预训练模型内部表征空间。

该路线的优势是稳定、低风险、与预训练视频模型的条件机制相容；劣势是历史条件的影响强度可能不如直接调制主干层。对于本项目三个月周期和资源约束而言，它最适合作为默认路线。

#### 7.6.9.3 相关路线二：Wan DiT 中后层 modulation / adapter 注入

第二类路线是在 condition-path 注入之外，进一步让 history representation 调制 Wan DiT 的中后层 hidden states，例如通过 AdaLN、residual adapter、gated adapter 或 LoRA 扩展来增强历史条件对 future latent prediction 的影响。

这类路线表达力更强，可能在复杂移动操作中更好地利用历史视觉、机器人状态和已执行动作。但它也带来更高风险：更容易扰动视频 foundation model 的已有 prior，训练稳定性更难控制，也会增加消融实验数量。因此，它适合作为默认方案之后的备选增强，而不适合作为首轮主线。

#### 7.6.9.4 相关路线三：激进主干注入与全层搜索

第三类路线是将 history latent 注入 early blocks、all-layer blocks、self-attention 主路径或进行更大范围的 Wan DiT fine-tuning。这类方法理论表达力最高，但变量最多、成本最大、风险最高，也最容易把“history latent 是否有用”与“主干被大幅改造后是否更强”混在一起。

结合本项目单人、两到三个月、实验总量受限的约束，这类路线应作为长期扩展而非当前主线。

#### 7.6.9.5 Action head 侧路线：从 layerwise feature 对齐到额外 WAM coupling

Action head 侧的调研重点不是重做动作头，而是如何保证项目二与 VLA 策略项目具有可比性。StarVLA 的总体设计强调 backbone、action head、data 和 trainer 等组件解耦[92]；WM4A 方向把视频生成 DiT 用作动作预测 backbone，并支持 OFT、GR00T、PI 等不同 action head，其中 PI 使用 layer-wise cross-attention 读取所有 transformer layers[93]。项目一 StarFlow 当前对应 LayerwiseFM / PI 风格的 flow-matching action head，核心是 action-side tokens 逐层 cross-attend 到 backbone 提供的 layerwise condition features[94]。

因此，本项目 action head 侧有三类路线：

| 路线 | 思路 | 优点 | 风险 | 本项目定位 |
|---|---|---|---|---|
| 对齐 StarFlow / LayerwiseFM 的 layerwise feature 接口 | 不重构 action head，只替换或增强 backbone 提供的 layerwise condition features | 与项目一可比，变量干净，工程风险低 | 若 WAM 表征未被充分消费，动作收益可能偏弱 | 默认主线 |
| 增加轻量 WAM global token / global condition | 在 action head 侧提供一个额外全局 WAM 条件 | 改动小，可增强 WAM-to-action coupling | 需要额外验证是否带来 downstream gain | 备选增强 |
| 重写 action head 内部层级调制或额外 cross-attention | 在 action head 多层中显式加入 WAM adapter / modulation | 表达力强 | 破坏项目一对照，变量混杂，调参复杂 | 暂缓 |

这一判断的核心是：项目二要证明的是 WAM / history latent / future latent 的价值，而不是证明一个新的 action head 架构更强。因此 action head 侧应优先对齐项目一 StarFlow / LayerwiseFM，把主要研究变量留在 history latent 压缩与 Wan DiT 侧注入。

#### 7.6.9.6 调研结论

从技术路线和工程约束综合判断，本项目应采用“保守默认 + 轻量备选”的策略：

| 问题 | 默认路线 | 备选增强 | 暂缓路线 |
|---|---|---|---|
| Wan DiT 如何接收 history latent | condition-path 注入 | 中后层 modulation / adapter | early/all-layer 主干注入、full fine-tune、全层搜索 |
| Action head 如何消费 WAM 表征 | 对齐 VLA 策略项目 StarFlow / LayerwiseFM 的 layerwise condition interface | 轻量 WAM global token / global condition | 重构 action head、layerwise action modulation、额外 cross-attention 重写 |

因此，调研报告层面的收敛结论是：

> 本项目不选择高风险的 Wan DiT 全层主干注入，也不将 action head 重构作为主变量；更合理的路线是：Wan DiT 侧优先采用 condition-path history injection，action head 侧优先复用 / 对齐 VLA 策略项目 StarFlow / LayerwiseFM 的 layerwise condition interface。具体 HLC-GCI 输出接口、history token 命名、注入公式、action-side token 组织和消融实验编号下沉到详细设计文档。


### 7.6.10 本项目推荐路线：History Latent Compressor + Gated Condition Injection

在明确来源边界、历史 latent 压缩路线、历史时间尺度路线和注入位置路线后，本报告建议将 HLC-GCI 作为 P1-b1 的默认候选实现方案。这里的“默认”表示详细设计阶段优先实现与验证，并不表示该方案已经被外部工作完整验证。其有效性仍需通过 P1-b0、P1-b1-HLC-GCI 与 robot-history shuffle 等最小消融确认。

从调研报告层面看，本项目推荐路线可以概括为：

```text
history RGB/video latent + text condition + robot history latent
→ history latent compression
→ compact history condition representation
→ conservative gated condition injection into Wan DiT
→ history-aware future latent / layerwise WAM features
→ StarFlow / LayerwiseFM-aligned action head
→ action chunk
```

该路线的关键判断包括：

1. P1-b1 的研究重点不是简单加入历史 state/action，而是多模态 history latent 的压缩与稳定注入；
2. HLC-GCI 是本项目组合方案，不是已有论文标准方法名；
3. 历史时间尺度默认采用短中期 sliding window，并可在数据支持时补充事件关键帧 / 低频 summary；不默认递归压缩全部历史；
4. Wan DiT 侧优先采用保守的 condition-path 注入，避免过早扰动视频 foundation model 主干；
5. action head 侧优先与 VLA 策略项目 StarFlow / LayerwiseFM 对齐，避免把 action head 重构变成新的混杂变量；
6. 更激进的 Wan DiT 中后层调制和 action-side WAM global condition 可作为备选增强；
7. all-layer 主干注入、full fine-tune、action head layerwise modulation、长期 retrieval memory、Mamba-style streaming memory 等高复杂度方案暂缓。

这一收敛路线使本项目的核心变量保持清晰：验证 history-aware latent WAM 是否能在不重构 action head 的情况下，为移动操作任务提供额外动作收益。


### 7.6.11 对详细设计的输入

详细设计文档应继承以下调研结论，并在设计文档中展开具体接口、公式、模块边界和实验编号：

| 设计项 | 调研结论 | 详细设计动作 |
|---|---|---|
| HLC-GCI 来源边界 | HLC-GCI 不是已有标准方法名，而是本项目基于多模态压缩、adapter 条件注入和 latent WAM 技术脉络提出的组合方案 | 明确“已有工作支撑的组件思想”和“本项目自定义组合方案”，避免误写为已有论文方法 |
| history latent 定义 | RGB/video、language、state/action 均需转为 DiT 可消费的 history latent / condition 表征；但只有 RGB/video 是 Wan-VAE latent | 区分 visual history latent、text condition tokens、robot history latent，并定义各自编码接口 |
| history latent 压缩 | 不直接 concat 全历史 token，优先采用 learned-query / Perceiver-style compression | 设计 History Latent Compressor 的输入、输出、token 数量和 profile 计划 |
| 历史时间尺度 | 默认采用短中期 sliding window；事件关键帧 / 低频 summary 可作为轻量增强；递归压缩全部历史、retrieval memory、Mamba-style streaming memory 暂缓 | 在 Data Gate / Profiling 中确定窗口覆盖秒数、采样 stride、历史帧数、关键帧数量；在 P1 章节定义默认 sliding-window 实现和可选增强 |
| Episode-to-window 采样 | 完整 episode 是采样容器，不是默认模型输入；训练从 episode 采样 anchor timestep，切出 history window、future latent target 与 action chunk target | 在数据章节定义 episode sampling、anchor sampling、window slicing、future target slicing、action chunk slicing、边界 mask 和 leakage invariant |
| SparseVideoNav profile baseline | Wan2.1 T2V-1.3B full fine-tune、256×256、4Hz、16 history、28 future、chunk=4、direct concat、T5 offload、batch=2、约 70G；该显存不能单独归因于 direct concat | 在资源章节拆分 persistent states、video activation、history concat 增量、future activation 和 workspace，并将其作为 P1-b 显存评估上界参考 |
| Rec-HLC 长期记忆 | `M_k = Update(M_{k-1}, x_k)` 可用固定长度 memory 覆盖 episode 前缀，但不是无损全历史输入；默认不进入 P1-b1 主线 | 在详细设计中作为 P1-b2 / Conditional 方案，定义 memory update segment、memory reset、detach / truncated BPTT、训练/推理一致性与触发条件 |
| Wan DiT 注入位置 | 调研结论收敛为 condition-path 注入优先，中后层 modulation 为备选，激进主干注入暂缓 | 在详细设计中定义具体注入公式、gate 初始化、LoRA / adapter 训练范围和可选 modulation 消融 |
| Action head 对齐 | 默认对齐 VLA 策略项目 StarFlow / LayerwiseFM 的 layerwise condition interface，不重构 action head 内部层 | 在详细设计中定义 Wan/HLC-GCI layerwise features 如何输入 LayerwiseFM；WAM global token 作为备选增强 |
| 最小验证 | 不做大量 history encoder / injection placement sweep | 在实验 Registry 中约束 P1-b0、P1-b1-HLC-GCI、robot-history shuffle、可选 WAM global token 等最小实验 |
| 防泄漏 | future action label 不进入 WAM input | 作为 dataset slicing invariant 和 unit test，不进入实验矩阵 |


## 7.7 LoRA[25] 与全面微调的判断

P1-b1 推荐 LoRA[25]/adapter 适配 Wan2.2 DiT[14]，不建议默认全面微调 TI2V-5B。原因有三点。

第一，机器人数据规模和多样性不足以稳定支撑 5B DiT 全量微调。第二，全量微调容易破坏 Wan2.2[14] 原有的视频 latent dynamics prior，使模型过拟合少数仿真或真实数据域。第三，三个月周期内还需要完成数据核验、latent 缓存、P0/P1 对比和下游评测，全量微调会显著增加训练、显存、checkpoint 和稳定性风险。

推荐训练边界是：

```text
冻结或基本冻结：Wan2.2-VAE、Wan2.2 text encoder、DiT 原始主干大部分权重
训练：state/action projector、History Latent Compressor、Gated Condition Injection、condition adapter、下游 action head / WAM-specific heads
LoRA：Wan2.2 TI2V-5B DiT 的关键 condition / attention / MLP 层
不默认做：全参数微调 Wan2.2 DiT
```

推荐的 LoRA[25] 注入优先级如下：

1. **优先注入 cross-attention / 条件交互相关层**：让 compact history condition tokens 能影响文本条件、图像条件和 latent dynamics 的交互；
2. **次优注入 self-attention / 时序建模层**：用于适配机器人相机运动、底盘移动和手臂/夹爪状态变化；
3. **谨慎注入 MLP 层**：仅在 attention-only LoRA[25] 不足时增加，避免训练参数和显存快速上升；
4. **默认冻结 VAE encoder / decoder 与 text encoder**：保持 Wan2.2[14] latent space 与 P2 decoder 的一致性；
5. **默认不做 DiT 全量微调**：除非 LoRA[25] 已证明成为瓶颈，且数据规模、schema 和评测闭环已经稳定。

LoRA rank、注入层数和 history window 属于详细设计阶段的可调参数，本报告只冻结“LoRA 优先、全量微调暂缓”的路线判断。

全面微调只作为后期升级：当 P1-b1 LoRA[25] 已经跑通、数据规模和 schema 足够稳定、且明确 LoRA[25] 成为性能瓶颈时，再考虑小范围全量微调或分层解冻。

## 7.8 与 P2 的连续性

P1 与 P2 的连接点是 Wan2.2-VAE latent space，而不是某个额外自定义表征。P1-b0 和 P1-b1 都输出 future Wan2.2[14] latent；P2 在局部 demo 或误差诊断中，用 Wan2.2-VAE decoder 解码 P1 预测的 latent，形成 future keyframe 或 short clip。

因此，P2 的第一步应是 decode P1-predicted latent，而不是直接运行完整 TI2V-5B diffusion rollout。完整 diffusion rollout 可以作为长期增强或展示性 demo，但不应反向挤占 P1-b 的主线。

## 7.9 P1 候选路线横向对比

| P1 候选路线 | WAM 输入 | 是否改 Wan2.2 DiT[14] | latent 来源 | 是否解码像素 | 下游接入方式 | 优点 | 风险 | 推荐等级 |
|---|---|---:|---|---:|---|---|---|---|
| **P1-b0：future latent prior** | 视觉 + 语言 | 否 / 极少 | Wan2.2-VAE latent | 否 | future latent + state/action history → action head | 工程稳、继承原生视频先验、P2 连续 | 不显式建模机器人历史动作影响，WAM 味道偏弱 | 低风险基线 |
| **P1-b1：history-latent-conditioned DiT-LoRA WAM** | visual history latent + text condition + robot history latent | 是，History Latent Compressor + Gated Condition Injection + LoRA[25]/adapter | Wan2.2-VAE latent + Wan2.2 DiT[14] prior | 否 | future latent + state/action history → action head / WAM heads | 继承预训练视频动力学，同时注入压缩后的多模态历史上下文 | history latent 压缩、condition space 对齐和注入稳定性要求更高 | **推荐主线** |
| 从零训练小型 latent predictor | 视觉 + 语言 + 历史 state/action | 否 | Wan2.2-VAE latent | 否 | policy condition / WAM heads | 简单、可控 | 不继承 Wan2.2 DiT[14] dynamics prior，数据和步数要求更高 | 备选/降级 |
| DiT hidden / denoising feature probing | 视觉 + 语言 | 不一定 | DiT 中间特征 | 否 | feature adapter / probe | 可分析 DiT 表征 | timestep/layer 选择复杂，非主闭环 | 暂缓 |
| candidate-action evaluator | 视觉 + 语言 + 当前 state + candidate future action | 是 | video/action coupled latent | 可选 | action scoring / refiner | 可做反事实动作后果评估 | 推理成本高、工程复杂 | 后期增强 |

## 7.10 本章路线判断与注意点

P1 推荐从“泛化的 Wan-derived latent WAM”收敛为 **Latent-Only WAM**，并将 P1-b 分为低风险基线和推荐主线：P1-b0 只用视觉/语言预测 future Wan2.2[14] latent；P1-b1 则将 history RGB/video latent、text condition 与 robot history latent 经 History Latent Compressor 压缩后，通过 Gated Condition Injection 与 LoRA[25]/adapter 注入 Wan2.2 DiT[14]，适配为 robot-history-conditioned latent WAM。

后续需要注意五点。第一，P1-b1 输入的是 history latent / condition tokens，其中 robot action 只能是已执行历史 action，不是未来 action label。第二，future action 防泄漏是 dataset slicing invariant 和 unit test，不应占用实验矩阵。第三，Wan2.2 DiT-LoRA[14][25] 不是为了生成视频，而是为了预测 future latent；P1 阶段不解码像素。第四，P1 与 P0 对比时应共享数据配比、下游 action head 和评测任务。第五，P2 只解码 P1 预测 latent 的少量关键样本，用于可解释性和误差诊断。

# 第八章 P2：Render-and-Decode WAM 路线调研

## 8.1 本章目的、作用与承接关系

第七章把 P1 定义为 latent-only WAM：模型预测 future Wan2.2[14] latent，但不解码像素。本章讨论何时需要把 future latent 解码出来，以及这种 Render-and-Decode 路线在本项目中应该承担什么角色。

P2 在全文中的作用不是替代 P0/P1，而是提供可解释性、误差诊断和局部展示能力。它向前承接 P1 的 future Wan2.2[14] latent，向后影响第九章和第十章中的 RGB 真实感评估、world-model-specific metrics 和 downstream task metrics。若 P2 被写成 full video generation 主线，项目将偏离三个月可执行范围；若完全不讨论 P2，则 P1 与 video foundation model 的关系又不完整。因此，本章需要明确 P2 的边界。

## 8.2 研究发展：从 goal image 到 continuous video world model

可视化未来的路线大致从轻到重分为四类。

第一类是 **goal image / keyframe-only**。模型只生成一个未来关键帧、目标视角或局部 visual subgoal，用于解释“模型希望场景变成什么样”。这类方法成本较低，适合作为 P2 的短期形式。

第二类是 **route video / trajectory-conditioned video**。模型沿导航或相机轨迹生成短视频，适合移动导航或接近目标阶段，但需要相机位姿、地图/几何和轨迹条件，对数据和系统要求更高。

第三类是 **continuous video WAM**。模型持续生成多步未来视频，用作想象、规划或策略评估。这是最完整的 video world model，但推理延迟、物理一致性、长期漂移和任务收益验证都很难。

第四类是 **omnimodal world model**。Cosmos-style 架构试图统一语言、图像、视频、动作和其他模态，为长期 Physical AI 提供通用世界模拟能力。它代表长期趋势，但不是本项目三个月内的可复现目标。

### 8.2.1 研究发展谱系与典型工作补充

P2 的研究谱系可以从轻到重拆成五类。这里的“典型工作/来源”用于说明可视化未来路线的发展，不代表本项目需要在三个月内实现所有方向。

| 发展类别 | 核心思想 | 典型工作/来源 | 与 P2 的关系 |
|---|---|---|---|
| **Goal image / keyframe-only / visual subgoal** | 只生成或解码一个未来关键帧、目标视角或局部 visual subgoal | LaWAM[2] 的 latent visual subgoal 思路 [2]、VLA-JEPA[3] / Light-WAM[15] 的 future latent supervision [3][15] | 适合作 P2 的最小可视化形式 |
| **Decode predicted latent** | 不重新运行完整视频生成流程，只把 P1 预测的 latent 解码成关键帧或短片段 | Wan2.2-VAE decoder 与 TI2V-5B latent space [14] | 是本项目 P2 的推荐短期路线 |
| **Route video / trajectory-conditioned video** | 沿导航轨迹或相机运动生成局部未来视频 | HomeRobot/OVMM[7][20][21] 暴露的导航—操作链路需求 [7][20][21]，以及 SAGE-3D[11] 等内部自建路线 [11] | 对移动操作有吸引力，但需要几何、轨迹和相机条件，当前暂缓 |
| **Continuous video WAM / video-action coupled WAM** | 连续生成多步未来视频，或将 video/action dynamics 统一建模 | Fast-WAM [1]、DreamZero[87]、UWM [8]、OSCAR [18]、Efficient-WAM [26]| 长期有价值，但推理成本和系统复杂度高 |
| **Omnimodal / adaptive world model** | 统一多模态世界模拟，或按需触发重型未来生成 | Cosmos 3[6]、AdaWAM [9]| 提供长期架构想象和触发策略，不作为当前主线 |

因此，第 8.3 节的重点参考工作不是完整视频生成方向的全量展开，而是围绕“decode P1 predicted latent”筛选：keyframe-only / goal image、route video 作为后期方向、continuous video WAM 作为长期边界、Cosmos[6]/AdaWAM 作为架构与触发策略参考。

从更早的 video prediction / model-based control 传统看，DreamerV3[80]、Visual Foresight[81]、UniPi[82]和 Genie[83]等工作分别代表了 latent dynamics、视觉前瞻、文本引导视频规划和交互式生成环境等方向。它们不是本项目短期要复现的对象，但有助于说明 P2 为什么应先收敛到 keyframe / latent decode，而不是直接进入完整连续视频世界模型。

## 8.3 值得参考的 P2 思路及优劣分析

上一节列出的 P2 发展谱系包括 keyframe、latent decode、route video、continuous video WAM 和 omnimodal/adaptive world model。本节只展开其中与本项目短期决策相关的参考思路：哪些支撑“decode P1 predicted latent”，哪些只适合后期方向，哪些只提供长期架构启发。


**keyframe-only / goal image** 的优点是可解释、成本相对低、容易与 P1 predicted latent 衔接。它适合在接近目标、抓取前、放置前和失败恢复前使用。缺点是单张关键帧不能表达完整动态，也可能出现视觉合理但物理不可执行的问题。

**route video / trajectory-conditioned video** 对移动操作有吸引力，因为移动平台需要理解从当前视角到目标区域的未来视觉变化。但它要求相机轨迹、环境几何、目标条件和可控渲染，工程复杂度明显高于 keyframe-only。本项目可把它作为后续研究方向，而不是当前主线。

**continuous video generation** 最接近完整世界模型，但也是风险最高的路线。它可能带来更强的规划想象能力，也可能消耗大量算力却不能提升机器人成功率。对本项目而言，continuous video generation 只能作为长期路线或展示性 demo，不能作为 P0/P1 的前置依赖。

**Cosmos-style omnimodal world model** 的意义在于提供长期架构想象：未来 WAM 可能不只是 video model，而是统一 language、image、video、action sequence、audio、state 的 world-action model。但当前项目不应从零复现 Cosmos[6]，也不应把 Cosmos[6]写成三个月内可验证的技术路线。

**AdaWAM-style adaptive triggering** 的启发是 P2 不应每一步都运行。只有当 P0/P1 判断 readiness 低、failure risk 高、阶段切换或 future latent 不确定时，才触发 keyframe 或短视频解码。这种按需触发比持续生成更符合移动操作实时性要求。

## 8.4 本项目采用的 P2 路线

本报告建议把 P2 定义为：**Decode P1-predicted Wan2.2[14] latent for local diagnosis and demonstration**。也就是说，P2 的第一阶段不是重新跑完整 TI2V diffusion rollout，而是把 P1-b0 或 P1-b1 预测出的 future Wan2.2[14] latent 用 Wan2.2-VAE decoder 解码成 keyframe 或 short clip。

推荐 P2 数据流为：

```text
P0/P1 默认运行
→ 选择关键样本或高不确定样本
→ 取 P1 predicted future Wan latent
→ Wan2.2-VAE decoder
→ future keyframe / short clip
→ 可解释性、误差诊断、局部 demo
```

触发条件可以包括：

```text
failure risk 高
readiness 低
任务阶段切换
P1 future latent 与 P0 progress/risk 判断冲突
action head 输出不稳定
需要展示或人工分析
```

这样 P2 才是 P1 的自然延伸，而不是一条独立的视频生成路线。

## 8.5 P2 候选路线横向对比

| P2 候选路线 | 输出形式 | 是否生成像素 | 代表模型/工作 | 适用阶段 | 成本 | 延迟 | 可解释性 | 任务收益可能性 | 三个月可行性 | 推荐等级 |
|---|---|---:|---|---|---|---|---|---|---|---|
| **decode P1 predicted latent** | future keyframe / short clip | 是 | Wan2.2-VAE decoder | 误差诊断、关键阶段展示 | 中 | 中 | 高 | 中 | 中 | **推荐局部验证** |
| keyframe-only / goal image | future keyframe / goal image | 是 | Wan 系列[5] / diffusion model | 关键操作前 | 中 | 中 | 高 | 中-高 | 中 | 备选 |
| route video | 短视频 | 是 | video generation / navigation video | 导航接近 | 高 | 高 | 中 | 中 | 低 | 暂缓 |
| continuous video WAM | 多步未来视频 | 是 | DreamZero[87] / UWM[8] / Cosmos-style | 全流程想象 | 很高 | 高 | 高 | 不确定 | 低 | 暂缓 |
| Cosmos[6]omnimodal model | 多模态 world simulation | 是/可多模态 | Cosmos-style | 长期架构 | 很高 | 高 | 高 | 长期高 | 低 | 长期参考 |
| adaptive triggering | 按需触发 keyframe/video | 是/可选 | AdaWAM-style | 任务切换/风险高 | 中 | 可控 | 高 | 高 | 中 | 推荐策略 |

## 8.6 本章路线判断与注意点

P2 的短期合理形态不是 full continuous video WAM，而是 keyframe-only 或 decode P1 predicted latent 的局部可视化。Wan2.2[14] 在 P2 中优先作为 VAE decoder 使用，服务可解释性、误差诊断和局部 demo。完整 TI2V diffusion rollout 只作为长期增强或展示性 demo，不应反向挤占 P0/P1 的主线。

后续需要注意三点。第一，P2 指标不能只看图像好不好看，而要看解码结果是否解释了 P1/P0 的判断和下游动作成败。第二，P2 应采用 adaptive triggering，避免每一步生成视频。第三，P2 的结论必须回流到数据工程和 WAM 评测，例如哪些场景的 future latent 漂移、哪些历史动作导致未来预测不稳定、哪些 failure-risk 样本值得补充训练数据。


# 第九章 RGB 真实感、任务链覆盖与数据可用性评估

## 9.1 本章目的、作用与承接关系

第四章已经完成候选数据和 benchmark 的分层筛选，但还需要解释为什么某些视觉真实感很高的资源不进入第一闭环，某些生态成熟的数据也不能作为移动操作主证据。本章从评价标准角度补充这一问题：数据资源的价值不只取决于画面质量、数据量或是否开源，而取决于它是否同时覆盖任务链、模态字段、监督信号和工程可用性。

本章向前承接第四章的候选资源，向后影响第十章评测指标和第十二章详细设计输入。它的目标是把“RGB 真实感、任务链覆盖、数据可用性”三类容易混淆的判断拆开，防止详细设计阶段把 V4 资源误当作主 benchmark，或把 M4 资源误当作移动操作证据。

## 9.2 数据评价标准的发展背景与典型做法

机器人数据评价从早期的“是否能跑、成功率多少”，逐步扩展到任务链覆盖、模态字段完整性和 world-model supervision 可用性。本项目采用的 S/V/M 与可用性矩阵，是在这一发展基础上的工程化筛选。

| 评价阶段 | 典型资源 / 工作 | 关注重点 | 对本项目的启发 |
|---|---|---|---|
| 操作 benchmark 成熟度评价 | LIBERO[48]、RLBench[50]、CALVIN[51]、RoboTwin[49] | 任务数量、语言条件、多任务泛化、操作成功率 | 适合操作端 sanity check，但不足以证明移动操作 WAM |
| 真实数据规模与多样性评价 | DROID[22]、Open X-Embodiment[23]、BridgeData[60]、AgiBot World[61] | 数据量、机器人多样性、真实传感器、action schema | 适合数据工程与表征学习，但仍需判断移动操作任务链覆盖 |
| 高真实感仿真评价 | BEHAVIOR-1K[24]、OmniGibson[45]、Habitat[42][43][44]、HomeRobot/OVMM[7][20][21] | 场景真实感、物理交互、household activity、跨房间任务 | V3–V4 对 P1/P2 有价值，但工程成本可能超过第一闭环 |
| 移动操作诊断评价 | MoMa-Kitchen[39]、EBench[35]、MobileManiBench[33]、Kitchen-R[34] | mobility、horizon、precision、readiness、last-mile | 更贴近 P0/P1 诊断需求，但开放状态和视觉质量需要核验 |
| WAM-specific 数据评价 | WAM 综述[84]、Fast-WAM[1]、LaWAM[2]、VLA-JEPA[3]、Light-WAM[15] | future supervision、latent target、action/history condition、可被下游消费的未来表征 | 数据集必须能产生 future-related 标签或 latent target，而不是只有动作标签 |

因此，本项目不采用单一“真实/仿真”二分，也不采用“V4 优先”或“数据量优先”的简单规则，而采用 S/V/M、模态完整性、可用性状态和项目定位的组合判断。

## 9.3 评价框架更新

v0.8 表明，仅用“真实/仿真”或“是否开源”判断数据价值是不够的。本项目需要同时评估：

1. **空间等级 S**：是否真正涉及移动、跨区域、跨房间或 last-mile 操作；
2. **视觉等级 V**：是否具备足够真实感支撑 P1/P2 视觉 latent 或可见未来；
3. **生态等级 M**：是否能在三个月内被单人试跑、读取、转换和评测；
4. **可用性状态**：可用、数据可下、未开源、coming soon、在线评测、仅论文、仿真器而非数据集；
5. **模态完整性**：是否有 action、state、base pose、EEF、joints、success/failure、stage/subgoal、force-torque；
6. **本项目定位**：主 benchmark、训练数据、诊断集、辅助验证、长期参考。

## 9.4 关键判断规则

| 规则 | 含义 |
|---|---|
| V4 不等于主 benchmark | BEHAVIOR-1K[24]、OmniGibson[45]、HomeRobot[7][20][21] 视觉强，但工程复杂或在线评测，不适合第一步全量接入 |
| M4 不等于移动相关 | LIBERO[48]、RoboTwin[49]、Open X[23] 生态成熟，但移动操作属性不足 |
| S0 不能作移动主证据 | 固定桌面只能做快速消融和操作泛化 |
| S2–S4 是核心 | 单房间移动、厨房多区域、跨房间同楼层最适合本项目 |
| S5 是长期外推 | 开放商店、多楼层、长距离真实系统价值高，但不适合作第一闭环 |
| coming soon 不能进主闭环 | MobileManiBench[33]、Kitchen-R[34] 等需等待代码/数据开放 |
| 在线评测不是离线训练集 | HomeRobot[7][20][21] Challenge、BEHAVIOR[24] Challenge[24] 等要和数据集区分 |

## 9.5 候选数据可用性矩阵

| 候选 | S等级 | V等级 | M等级 | 移动操作强度 | 工程可行性 | 是否适合第一闭环 | 是否适合主证据 | 是否适合附录保留 |
|---|---|---|---|---|---|---|---|---|
| RoboCasa365[19] | S2–S3 | V3 | M4 | 中-高，需核实 mobile base | 高 | 是 | 候选，但需边界说明 | 是 |
| RoboCasa[32] | S1–S3 | V3 | M4 | 中 | 高 | 是 | 辅助/闭环，不作完整移动主证据 | 是 |
| AIRoA MoMa[36] | S2–S4 | Real | M2–M3 | 高，真实移动操作 | 中，需核验数据/代码 | 是，作为真实数据核验 | 是，真实数据核心候选 | 是 |
| MobileManiBench[33] | S2–S3 | V4 | M1–M2 | 高 | 低，coming soon | 否 | 未来主候选 | 是 |
| Kitchen-R[34] | S2–S3 | V4 | M1–M2 | 高 | 低，未开源 | 否 | 未来候选 | 是 |
| EBench[35] | S0–S3 | V4 | M2 | 中-高 | 中，需试跑 | 诊断闭环 | 诊断证据 | 是 |
| BEHAVIOR-1K[24] | S3–S5 | V4 | M4 | 高 | 低-中，工程重 | 否 | 后期高难证据 | 是 |
| HomeRobot / OVMM[7][20][21] | S4 | V3–V4 | M3 | 高 | 中-低，在线/attach grasp | 否 | 后期主评测候选 | 是 |
| Habitat[42][43][44] 2.0 / HAB[42] | S3–S4 | V3–V4 | M4 | 高 | 中 | 后期 | 后期证据 | 是 |
| LAMBDA[38] | S4–S5 | Real+Sim | M2 | 高 | 中-低 | 否 | 高难外推 | 是 |
| MoMa-Kitchen[39] | S2–S3 | V1 | M2 | 中，last-mile | 中 | 诊断 | 否，诊断集 | 是 |
| ManiSkill3[40] | S0–S3 | V3 | M4 | 中 | 高 | 诊断/数据生成 | 辅助 | 是 |
| LIBERO[48] | S0 | V2 | M4 | 低 | 高 | 快速 sanity | 否 | 是 |
| RoboTwin 2.0[49] | S0–S1 | V2–V3 | M4 | 低-中 | 高 | 操作泛化 | 否 | 是 |
| DROID[22] | S0–S2 | Real | M4 | 中 | 中 | P1 表征 | 否，真实操作辅助 | 是 |
| Open X-Embodiment[23] | S0–S2 | Real | M4 | 中 | 中 | 数据工程 | 否，背景数据 | 是 |
| WorldEval[70] | 任务依赖 | 方法型 | M2 | 取决于输入 | 中 | 否 | 评测方法参考 | 是 |

## 9.6 对 P0/P1/P2 的含义

### P0

P0 对视觉真实感要求低于 P1/P2，但对 stage、success/failure、action outcome、final-pose affordance、readiness 标注要求高。因此 MoMa-Kitchen[39]、EBench[35]、AIRoA MoMa[36]、RoboCasa[32]/RoboCasa365[19]、ManiSkill3[40] 都有价值。

### P1

P1 需要未来帧、连续轨迹、状态、动作和稳定视觉模态来构造 latent target。AIRoA MoMa[36]、RoboCasa[32]/RoboCasa365[19]、DROID[22]、BridgeData[60]、AgiBot World[61]、RoboTwin[49]、LIBERO[48] 可以作为不同强度的数据源，但只有 AIRoA MoMa[36] 更接近真实移动操作。

### P2

P2 对 V3–V4 视觉更敏感，应优先考虑 RoboCasa[32]/RoboCasa365[19]、Isaac 系候选、BEHAVIOR[24]/OmniGibson[45]、Habitat[42][43][44]/HomeRobot 等。但 P2 只做 keyframe-only 或局部 demo，不做 full continuous video WAM 主线。

## 9.7 重点参考工作与本章决策依据

本章真正支撑决策的不是所有候选，而是三组对比：第一，RoboCasa/RoboCasa365[19][32] 与 LIBERO/RoboTwin[48][49] 的对比说明“生态成熟”不等于“移动操作主证据”；第二，AIRoA MoMa[36]、Mobile ALOHA[37]和 LAMBDA[38]说明真实移动操作数据必须单独核验；第三，MoMa-Kitchen[39]、EBench[35]、MobileManiBench[33]、Kitchen-R[34]说明移动操作诊断集对 P0/P1 更直接，但开放状态和视觉等级必须写清楚。

据此，本项目采用“近期可落地 + 真实核验 + 诊断补充 + 高难后置”的组合，而不是单纯追求 V4 视觉或最大数据规模。

## 9.8 本章客观调研结论

本项目的数据可用性判断应优先看“能否服务 P0/P1/P2 可复现实验闭环”。因此：

- RoboCasa[32]/RoboCasa365[19] 虽非完美移动操作主证据，但适合第一闭环；
- AIRoA MoMa[36] 是真实移动操作核心数据，必须核验；
- MobileManiBench[33]/Kitchen-R[34] 因开放状态限制，不能进入近期主闭环；
- HomeRobot[7][20][21]/BEHAVIOR[24]/LAMBDA[38] 是高难验证集，不适合第一步全量接入；
- LIBERO[48]/RoboTwin[49]/DROID[22]/Open X[23] 等只能做辅助层，不能越位成移动操作主证据。

## 9.9 对本项目的决策建议

详细设计阶段应将 RGB 真实感、任务链覆盖和数据可用性拆开处理：高真实感数据不自动进入主线，高可用数据也不自动证明移动操作能力。RoboCasa/RoboCasa365[19][32] 在完成 mobile base / navigation-manipulation 强度核验前，应写作近期可落地 household manipulation 强闭环底座；AIRoA MoMa[36]、EBench[35]、MobileManiBench[33]、Kitchen-R[34] 等才是移动操作主证据需要重点核验的对象。

# 第十章 WAM 评测指标体系调研

## 10.1 本章目的、作用与承接关系

第六至第八章给出了 P0/P1/P2 的路线选择，但路线是否成立不能只看训练 loss 或生成效果。本章的作用是定义“怎样证明 WAM 对移动操作真的有用”。它承接第四、五章的数据与监督信号，面向第十一章的下游接入和第十二章的详细设计输入，建立 WAM-specific、downstream、coupling 和 cost 四类指标。

## 10.2 WAM 评测方法的发展谱系与典型工作

WAM 评测方法的演化可以概括为从 policy success、prediction quality、world-model evaluator 到 WAM-policy coupling 的转变。WAM 综述[84]也指出，WAM 不能只报告未来生成质量，还要报告 future representation 与动作决策之间的关系。

| 评测阶段 | 典型工作 / 框架 | 常见指标 | 对本项目的启发 |
|---|---|---|---|
| Policy-only task success | LIBERO[48]、RLBench[50]、CALVIN[51]、RoboTwin[49]、OpenVLA[77] | TSR、成功率、任务完成率、操作技能成功率 | 必须保留 downstream 指标，但它无法解释 WAM 表征是否有效 |
| Mobile manipulation task metrics | HomeRobot/OVMM[7][20][21]、Mobile ALOHA[37]、LAMBDA[38]、MoMa-Kitchen[39] | navigation success、grasp/place success、last-mile affordance、stage success | 本项目应拆分找物、接近、视角调整、操作、放置和恢复等阶段 |
| Future representation / latent quality | LaWAM[2]、VLA-JEPA[3]、Light-WAM[15]、Fast-WAM[1] | future latent distance、progress/readiness/risk prediction、future supervision loss | P0/P1 需要 WAM-specific 指标，但不能只报告 loss |
| Video / render quality | Wan 系列[5]、Wan2.2[14]、Cosmos[6]、UWM[8] | FVD、LPIPS、frame consistency、physical plausibility | P2 可参考，但视频好看不等于任务收益 |
| World model as evaluator | WorldEval[70]、dWorldEval[71]、WPE[72]、WorldGym[73]、OSCAR[18] | policy evaluation correlation、rollout outcome、action-conditioned future consistency | 对 P1/P2 的评测方法有启发，但不一定作为第一阶段实现 |
| Coupling / ablation analysis | Fast-WAM[1]、Light-WAM[15]、1X World Model Challenge[16]、WAM 综述[84] | 去掉 future representation 后性能下降、WAM 指标与 downstream gain 相关性 | 本项目必须证明 WAM 输出被下游消费，而不是只优化辅助任务 |

## 10.3 指标分类

| 指标类型 | 具体指标 | 适用阶段 | 用途 |
|---|---|---|---|
| **WAM-specific** | Future Visibility Accuracy / F1 | P0/P1 | 衡量目标未来是否可见、遮挡是否被提前预测 |
| **WAM-specific** | Manipulation-Readiness Prediction accuracy / AUROC | P0/P1 | 衡量导航—操作切换前的可操作状态判断 |
| **WAM-specific** | Failure-Risk AUROC / early-warning lead time | P0/P1 | 衡量失败预警能力和提前量 |
| **WAM-specific** | Task Progress MAE / stage classification accuracy | P0/P1 | 衡量子任务阶段推进预测 |
| **WAM-specific** | Future Latent Distance / latent consistency | P1 | 衡量 predicted future latent 与 target latent 的一致性 |
| **WAM-specific** | Latent-to-State / Latent-to-Outcome correlation | P1 | 判断 latent 是否包含可被动作策略消费的信息 |
| **Downstream task** | Task Success Rate（TSR） | P0/P1/P2 | 衡量整体任务收益 |
| **Downstream task** | Sub-stage Success Rate（SSR） | P0/P1/P2 | 分析找物、接近、视角调整、抓取、放置等阶段收益 |
| **Downstream task** | Navigation-to-Manipulation Success（NMS） | P0/P1 | 衡量移动到可操作状态的交接质量 |
| **Downstream task** | Recovery Success Rate（RSR） | P0/P1 | 衡量失败恢复能力 |
| **Downstream task** | Repetition / oscillation count | P0/P1 | 衡量重复尝试和局部振荡是否减少 |
| **Downstream task** | Collision / unsafe action rate | P0/P1/P2 | 衡量安全性与物理合理性 |
| **Coupling metric** | WAM metric gain vs downstream gain correlation | P0/P1 | 判断 WAM 指标提升是否真正转化为任务收益 |
| **Coupling metric** | ablation drop after removing future representation | P0/P1 | 衡量 future representation 的必要性 |
| **Cost metric** | WAM forward latency、显存、latent extraction 成本、训练时长 | P0/P1/P2 | 衡量三个月和 8 卡 A100 约束下的可行性 |

## 10.4 P0/P1/P2 指标差异

| 阶段 | 核心指标 | 成功判断 | 失败信号 |
|---|---|---|---|
| P0 | readiness accuracy、failure-risk AUROC、task-progress accuracy、SSR / NMS / RSR gain、latency | 轻量 future heads 在不生成像素的前提下提升阶段判断和下游成功率 | WAM-specific 指标提升但 TSR/SSR/NMS 无提升 |
| P1 | future latent distance、latent-outcome correlation、SR gain over P0、latency、显存 | Latent-Only WAM 提供比 P0 更有用的 future representation | latent loss 下降但 action head 不受益；推理延迟过高 |
| P2 | decoded keyframe consistency、physical plausibility、diagnostic usefulness、triggered SR gain | P2 能解释 P1 错误、支持局部可视化诊断，而不是吞噬主线 | 视频好看但与任务收益无关；P2 占用 P0/P1 资源 |

## 10.5 WAM-specific 与 downstream 的关联分析

详细设计阶段至少需要做三类关联分析：

1. **相关性分析**：统计 readiness/risk/progress/latent distance 与 TSR、SSR、NMS、RSR 的相关性，避免只报告 isolated WAM metrics。
2. **去除式消融**：比较 full model、no-WAM、no-future-latent、no-history-condition、manual sampling α 等变体，验证收益来自 WAM 而不是数据或 action head。
3. **阶段性诊断**：将任务链拆成找物、接近、视角调整、可操作状态判断、操作、放置、恢复等阶段，分析 WAM 对哪个阶段真正有帮助。

## 10.6 重点参考工作与本章决策依据

本章的关键参考来源分为三类：第一，WorldEval[70]、dWorldEval[71]、WPE[72]、WorldGym[73]说明 world model 可以作为 policy evaluator 或 rollout evaluator，但其工程成本和数据要求较高；第二，Fast-WAM[1]、VLA-JEPA[3]、LaWAM[2]、Light-WAM[15]说明 WAM-specific 指标必须与下游策略收益绑定；第三，HomeRobot/OVMM[7][20][21]、MoMa-Kitchen[39]、LAMBDA[38]等移动操作 benchmark 提醒本项目必须报告阶段性成功率，而不是只报告整体 TSR。

因此，第十章的决策不是采用某一个外部 leaderboard，而是在详细设计中建立“WAM-specific + downstream + coupling + cost”的组合评测。

## 10.7 本章客观调研结论

WAM 的评测不能停留在 prediction loss 或 decoded video quality。对本项目而言，最重要的是证明 future representation 是否可被下游消费，并在移动操作任务链中带来 measurable gain。P0/P1 的核心指标应优先围绕 readiness、risk、progress、visibility、latent-outcome correlation 和 downstream success 设计。

## 10.8 对本项目的决策建议

详细设计中应把 WAM-specific、downstream、coupling、cost 四类指标写入验收标准。若 P0/P1 的 WAM-specific 指标提升但 downstream gain 不显著，则不得继续扩大 P1/P2 规模，应先执行第十三章定义的降级路径。

# 第十一章 下游接入方式与系统集成形态调研

## 11.1 本章目的、作用与承接关系

前面章节回答了 WAM 该学什么、用什么数据学、如何评测；本章回答 WAM 输出如何被下游系统消费。它承接第六至第八章的 P0/P1/P2 表征形式，也承接第十章的指标体系，并为第十二章详细设计输入提供系统接入边界。

本章不展开 ROS、部署接口或 batch size 等详细工程实现，而是比较 WAM 输出在策略系统中的几类接入形态，避免把 WAM 分支、VLA action head、规划器、评测器混写。

## 11.2 WAM 下游接入方式的发展谱系与典型工作

不同 WAM 路线对应不同接入方式。结合 WAM 综述[84]以及 Fast-WAM[1]、LaWAM[2]、VLA-JEPA[3]、Light-WAM[15]、ChronoDreamer[17]、OSCAR[18]、WorldEval[70]等工作，可以归纳出以下谱系：

| 接入方式 | 典型工作 / 思路 | WAM 输出 | 下游消费方式 | 工程风险 |
|---|---|---|---|---|
| Future auxiliary heads | Fast-WAM[1]、Light-WAM[15]、P0 类 video-free WAM | progress、readiness、risk、visibility | 作为 action head 条件、诊断信号或辅助 loss | 风险低，最适合第一闭环 |
| Latent future conditioning | LaWAM[2]、VLA-JEPA[3]、P1 类 latent-only WAM | future latent / latent subgoal | 输入 action head、subgoal head 或 WAM-specific heads | 需要验证 latent 是否被下游有效消费 |
| History-conditioned future model | ChronoDreamer[17]、1X Wan2.2 adaptation[16] | 历史 state/action 条件下的 future latent / future frame | 提供更贴近机器人运动趋势的未来表征 | 需要保证只输入历史 action/state，禁止 future action label 泄漏 |
| Candidate-action evaluator | OSCAR[18]、UWM[8]、WorldEval[70] | 候选动作导致的 future / outcome score | action scoring、selector、refiner 或 policy evaluator | 推理成本高，不适合当前第一闭环 |
| Render-and-decode diagnostic | Wan 系列[5]、Wan2.2[14]、Cosmos[6]、P2 类局部解码 | keyframe / short clip / 可见未来 | 用于解释错误、诊断 latent drift 和展示 demo | 容易滑向视频生成主线，需要触发式使用 |
| Adaptive trigger / router | AdaWAM[9]、omnimodal / adaptive WAM 候选 | 是否触发 reasoning、latent 或 decode | 决定何时使用 WAM 输出 | 长期参考，当前不作为主线 |

这个谱系说明：WAM 的接入不只有“把 future latent 拼给 action head”一种形式。不同路线在延迟、动作闭环、可解释性和评测可控性上差异很大，必须与 P0/P1/P2 阶段目标对应。

## 11.3 接入形态

| 接入方式 | P0 | P1 | P2 |
|---|---|---|---|
| 条件输入 | readiness / progress embedding | latent future embedding | keyframe embedding |
| 评分器 | subgoal score | latent subgoal score | visual subgoal score |
| 风险预测器 | failure risk | latent anomaly / mismatch | predicted failure video |
| router | 是否操作 / 是否恢复 | 是否使用 latent future | 是否触发 video decode |
| 诊断器 | 阶段失败定位 | latent drift | visual inconsistency |

## 11.4 值得参考的接入方式与优劣分析

对本项目当前三个月目标而言，最值得参考的是三种接入方式。第一，P0 采用 future auxiliary heads / diagnostics：readiness、progress、risk、visibility 等输出可以直接接入 action head 或作为诊断指标，延迟低、可解释性强。第二，P1 采用 latent future conditioning：future latent 与历史 state/action 一起输入 action head 或 WAM-specific heads，但必须通过第十章的 coupling metrics 验证其带来下游收益。第三，P2 仅采用 triggered decode diagnostic：只在分析 P1 latent drift、失败案例或可解释性展示时解码 keyframe / short clip，不进入常规控制闭环。

不建议当前阶段采用 candidate-action evaluator 作为主线。它需要对多个候选动作做 future rollout 或 outcome scoring，适合长期 action evaluation，但会显著增加推理成本和实验复杂度。

## 11.5 推荐接入方案

第一阶段推荐采用：

```text
policy / action head tokens
+ robot state
+ P0 WAM heads
→ action / subgoal score / risk signal
```

这里的 policy / action head 可以复用 VLA 策略项目的工程资产，但 P0 WAM heads 是独立的 future-aware 分支。P0 输出应优先用于 readiness、progress、failure-risk、next-best-view 或 subgoal feasibility，而不是直接把 VLA action head 改名为 WAM。

第二阶段引入：

```text
future latent / latent subgoal
+ history state/action
→ latent adapter / action head / WAM-specific heads
```

P1-b1 若使用 Wan2.2[14] 作为候选视频基础模型，则历史 robot state/action 只作为历史条件注入 WAM，不输入未来 action label。该 future latent 是否进入主闭环，取决于第十章定义的 downstream gain 和 coupling analysis。

P2 接入仅限：

```text
predicted future latent
→ triggered decoder
→ keyframe / short clip diagnostic
```

P2 不进入高频控制闭环，不作为 P0/P1 的替代路线。

---

## 11.6 本章客观调研结论

WAM 的下游接入方式主要包括直接给 action head 提供 future representation、作为风险 / readiness / progress 诊断头、作为 action selector / refiner 的辅助信号，以及 P2 中的可视化诊断。不同接入方式对延迟、数据字段和评测指标的要求不同，不能在详细设计中混写。

## 11.7 对本项目的决策建议

P0 优先采用“future heads + action head conditioning / diagnostics”的轻量接入；P1 优先采用“future latent + history state/action → action head / WAM-specific heads”的 latent-only 接入；candidate-action evaluator 和 P2 decode 只作为后期增强，不进入当前第一闭环。

# 第十二章 路线对比、推荐结论与详细设计输入

## 12.1 数据与 benchmark 对路线的约束

v0.8 核对结果改变了本项目的实验路径：不能直接将“最前沿或最像移动操作”的 benchmark 作为第一阶段主线，因为很多强候选存在未开源、coming soon、仅论文、在线评测或工程过重问题。因此，详细设计应采用“先闭环、再外推、再高难”的策略。

推荐顺序：

```text
近期可落地闭环
→ 真实移动操作数据核验
→ P0/P1 诊断集
→ 强移动操作仿真候选
→ 高难 household / 跨房间验证
→ P2 局部 demo / world model evaluation
```


补充数据混合结论：正式 P0 / P1 不应分别手工设置不同数据比例，而应共享一组由小规模 P0 proxy WAM 学得的域采样权重 α。该策略作为数据混合折中，不新增复杂实验矩阵，也不把 Re-Mix[12] / DRO 复现作为主研究目标。

## 12.2 数据与 benchmark 详细设计输入表

| 用途 | 首选 | 备选 | 暂缓 | 不推荐 | 进入详细设计动作 |
|---|---|---|---|---|---|
| 第一闭环 | RoboCasa[32] / RoboCasa365[19] | LIBERO[48] / RoboTwin[49] 做 sanity check；ManiSkill3[40] 做生成辅助 | MobileManiBench[33]/Kitchen-R[34] 若未开放 | 直接全量 BEHAVIOR[24] 或 HomeRobot[7][20][21] | 先跑通数据读取、replay、P0 标签构造、P1 latent 抽取 |
| 真实移动操作关键候选 | AIRoA MoMa[36] | Mobile ALOHA[37]、HomeRobot[7][20][21] real、LAMBDA[38] | SHOPPER / LaNMP 若数据不可得 | 仅系统论文无数据 | 下载/核验 schema，检查 RGB/Joints/FT/Stage/Action/Success/Base |
| 强移动操作仿真候选 | EBench[35]、MobileManiBench[33]（开放后）、Kitchen-R[34]（开放后） | Habitat[42][43][44] 2.0 / HAB[42]、HomeRobot/OVMM[7][20][21] | BEHAVIOR-1K[24] 全量 | coming soon 资源直接进主闭环 | 建立可用性 gate：代码、数据、样例任务、模态导出 |
| P0 readiness / affordance 诊断 | MoMa-Kitchen[39]、EBench[35]、Habitat[42][43][44] Rearrangement、ManiSkill-HAB[47]| SAGE-3D[11] / Isaac 自建补充 | full high-fidelity household | 只用 LIBERO[48] 判断 readiness | 构造 readiness/risk/progress/NBV/subgoal feasibility 诊断集 |
| P1 latent 数据 | AIRoA MoMa[36]、RoboCasa[32]/RoboCasa365[19]、DROID[22]、BridgeData V2[60] | Open X[23]、AgiBot World[61]、RoboTwin[49]、LIBERO[48] | BEHAVIOR[24] 全量、Cosmos[6]/OSCAR 类大模型数据 | 无 action/state 的视频数据直接训练 WAM | 抽取 Wan2.2[14] latent / VAE latent / DiT hidden states，记录成本 |
| P2 局部 demo | RoboCasa[32]/RoboCasa365[19]、Isaac 系候选、HomeRobot[7][20][21] 局部 | BEHAVIOR[24] / OmniGibson[45] / Habitat[42][43][44] | continuous video WAM | 三个月 full video WAM 主线 | 只做 keyframe-only / triggered visual subgoal |
| 评测方法 | WorldEval[70]、dWorldEval[71]、VLA-Arena[69]、HomeRobot[7][20][21] Challenge | SimplerEnv[74]、WPE[72]、BEHAVIOR[24] Challenge[24] | WorldGym[73]/ OSCAR[18] 全量 | 只看 offline loss | 设计 WAM-specific + downstream + correlation metrics |

## 12.3 推荐验证队列：8–12 个核心候选

### 第一优先级：三个月内最应该验证

1. **RoboCasa365[19]**  
   用作近期可落地闭环。重点验证：数据读取、replay、action/state/lang、是否能构造 P0 progress/readiness/risk、是否能抽取 P1 latent。风险：mobile base / navigation-manipulation 强度需核实；在完成该核验前，RoboCasa365[19] 只能写作 kitchen household manipulation 强闭环底座，不能单独作为完整移动操作主 benchmark。

2. **RoboCasa[32]**  
   用作稳定工程底座和第一版闭环。重点验证：与 RoboCasa365[19] 的兼容性、数据模态、任务成功/失败信息、对象状态导出。

3. **AIRoA MoMa[36]**  
   用作真实移动操作 P0/P1 核心数据。重点验证：HF 数据是否完整、schema、Stage/Subgoal、primitive Action、Joints、FT、internal State、Base/EEF/Success/Lang 是否可用。注意：当前核对备注为短程任务。

4. **EBench[35]**  
   用作 Isaac Sim 诊断评测候选。重点验证：mobile manipulation 任务是否可跑通、五轴诊断是否可映射到 WAM-specific metrics。

5. **ManiSkill3[40]**  
   用作工程成熟的操作/数据生成底座。重点验证：RGB-D/Seg/PCD/State/Action 导出、是否支持 S2–S3 mobile-base 子任务或可构造类似诊断。

6. **MoMa-Kitchen[39]**  
   只作为 P0 readiness / final-pose affordance 诊断集。重点验证：affordance floor labels、robot-specific parameters、target object、是否可构造 manipulation-readiness。不要作为主视觉 benchmark。

7. **LIBERO[48] / LIBERO-Long[48]**  
   用作 sanity check 和VLA 策略项目经验复用。重点验证：P0/P1 模块是否能在成熟 VLA benchmark 上稳定训练。必须标注非移动主证据。

8. **RoboTwin 2.0[49]**  
   用作操作泛化和 OOD 辅助。重点验证：dual-arm State/Action、domain randomization、语言 variation、P1 latent 泛化。

### 第二优先级：开放后或后期验证

9. **MobileManiBench[33]**  
   Isaac Sim 强候选，但当前 coming soon，不能作为近期主闭环。开放后优先验证。

10. **Kitchen-R[34]**  
   强 kitchen mobile manipulation 候选，但当前仅论文、未找到代码/数据。暂不进入主闭环。

11. **HomeRobot / OVMM[7][20][21]**  
   高价值移动操作评测候选，但抓取机制偏 attach/吸附、在线评测、离线训练数据不足，不适合作第一步训练。

12. **BEHAVIOR-1K[24] / OmniGibson[45]**  
   高真实感、高复杂 household 长期验证集。工程重，适合后期高难验证。

### 附录保留 / 暂不推进

- LAMBDA[38]：高难 multi-room / multi-floor / data efficiency，暂作为后期外推；
- DROID[22] / Open X[23] / BridgeData[60] / AgiBot World[61]：P1 表征学习与数据工程参考，不作移动主证据；
- WorldEval[70] / dWorldEval[71]/ WPE[72]：评测方法参考；
- LIBERO-Extended / RoboCasa-Lite / RoboTwin-Diagnosis：未确认正式 benchmark，暂缓。

## 12.4 P0/P1/P2 与数据资源的最终绑定

| 阶段 | 推荐数据资源 | 验证目标 |
|---|---|---|
| P0 | RoboCasa[32]/RoboCasa365[19]、AIRoA MoMa[36]、EBench[35]、MoMa-Kitchen[39]、ManiSkill3[40] | readiness、risk、progress、NBV、subgoal feasibility |
| P1 | AIRoA MoMa[36]、RoboCasa[32]/RoboCasa365[19]、DROID[22]、BridgeData[60]、RoboTwin[49]/LIBERO[48] 辅助 | P1-b0 视觉/语言 future latent prior；P1-b1 robot-history-conditioned Wan2.2 DiT-LoRA[14][25]，使用历史 state/action 条件预测 future Wan2.2[14] latent |
| P2 | RoboCasa[32]/RoboCasa365[19]、Isaac 系候选、HomeRobot[7][20][21]/BEHAVIOR 局部 | decode P1-predicted Wan2.2[14] latent、keyframe-only、triggered visual subgoal |
| 高难验证 | MobileManiBench[33]、Kitchen-R[34]、HomeRobot/OVMM[7][20][21]、LAMBDA[38]、BEHAVIOR-1K[24] | 移动操作主证据与长期外推 |
| 辅助验证 | LIBERO[48]、RoboTwin[49]、CALVIN[51]、RLBench[50]、FurnitureBench[52] | 快速消融、操作泛化、长时序 sanity check |

## 12.5 进入详细设计文档的统一输入清单

新增 P1-b 设计输入：详细设计文档必须将 P1-b1 的默认机制写成 History Latent Compressor + Gated Condition Injection，而不是笼统的 history state/action encoder；必须区分 visual history latent、text condition tokens、robot history latent，并将 future action 防泄漏放入工程 invariant，而不是实验 Registry。

下表将前文分散的路线、数据、评测和风险结论压缩为详细设计文档的直接输入。详细设计阶段应逐项展开网络结构、数据 schema、训练目标、评测脚本和验收阈值。

| 输入项 | 推荐选项 | 备选选项 | 暂缓 / 排除 | 证据等级 | 主要风险 | 详细设计待确认 |
|---|---|---|---|---|---|---|
| P0 正式路线 | Video-Generation-Free WAM | JEPA-style latent future auxiliary | 像素级视频生成 | B | 下游收益不明显 | WAM heads、标签来源、loss 权重、action head 接入 |
| P0 候选实现 | 可复用 Qwen3VL[29] / Qwen-VL[30] / StarVLA[31] 资产 | 其他 VLM / visual-state encoder | 直接 VLA action head 当作 WAM | B/C | 被误写成 VLA 改装 | backbone 冻结/微调、WAM 分支独立性 |
| P1 正式路线 | Latent-Only WAM | P1-b0 future latent prior | P1 阶段默认解码像素 | B | latent 不可消费 | latent target、history condition、policy interface |
| P1 候选实现 | Wan2.2 TI2V-5B[14] + VAE latent + DiT-LoRA[25] | Wan2.1[5] / 其他 video latent foundation model | 从零训练大型 WAM / 全量微调 5B DiT | B/C | 显存、数据规模、许可证与引用状态 | LoRA 注入位置、history encoder、latent 缓存 |
| P1-b0 | 视觉/语言 future latent prior | 加轻量 state summary | 当作完整 action-conditioned WAM | B | WAM 味道偏弱 | 是否能带来 downstream gain |
| P1-b1 | 历史 state/action 条件化 latent WAM | 小型 latent predictor 降级 | 输入 future action label | B | 数据切片泄漏、schema 不统一 | history window、future horizon、防泄漏检查 |
| P2 路线 | Decode P1 predicted latent 的 keyframe / short clip | triggered visual subgoal demo | continuous video rollout 主线 | B/C | 偏离主线、成本过高 | 触发条件、解码频率、诊断指标 |
| 第一闭环数据 | RoboCasa[32] / RoboCasa365[19] | ManiSkill3[40] / LIBERO[48] sanity check | 直接全量 BEHAVIOR[24] | B | mobile 强度需核验 | replay、schema、success/failure、P0 标签 |
| 真实数据核验 | AIRoA MoMa[36] | Mobile ALOHA[37] / HomeRobot real[7][20][21] | 仅系统论文无下载数据 | B/C | 字段不完整、任务短 | RGB、state、action、stage、success、base/EEF |
| 移动操作强候选 | EBench[35]、MobileManiBench[33]、Kitchen-R[34] | Habitat[42][43][44] / HomeRobot[7][20][21] | 未开放即进主闭环 | B/C | 开放状态变化 | 代码/数据、mobile manipulation task、模态导出 |
| P0 诊断集 | MoMa-Kitchen[39] / EBench[35] / ManiSkill-HAB[47]| SAGE-3D 自建补充[11] | 只用 LIBERO 判断 readiness | B/C | 视觉保真度或任务覆盖不足 | readiness/risk/progress/NBV 标注 |
| 数据混合 | proxy WAM 学 α，正式 P0/P1 固定 α | 手工域采样比例 | 完整动态 DRO 作为主贡献 | B | proxy 与正式训练不一致 | proxy 指标、域划分、α 固定策略 |
| 评测指标 | WAM-specific + downstream + coupling + cost | 只做 offline loss 作为辅助 | 只看生成质量 | B | 指标提升不转化任务收益 | TSR/SSR/NMS/RSR、相关性、消融 |
| 风险 gate | Data Verification Gate | 人工小样本核验 | 未核验直接进入实验矩阵 | A/B | 候选过多拖慢主线 | 下载、schema、license、sample replay、字段完整性 |

## 12.6 P0→P1→P2 阶段转换决策流

```text
P0 运行中
├── WAM-specific 指标是否提升？
│   ├── 否：降级为诊断模型或缩小 heads，停止推进 P1
│   └── 是
│       └── downstream TSR / SSR / NMS / RSR 是否有可测收益？
│           ├── 否：保留 readiness / risk / NBV 等最可消费信号，重新验证 action-head 接入
│           └── 是：进入 P1-b0

P1-b0 运行中
├── future latent 预测质量是否稳定改善？
│   ├── 否：检查 latent target、VAE 缓存和数据切片，必要时退回 P0
│   └── 是
│       └── 相比 P0 是否带来额外 downstream gain？
│           ├── 否：P1-b0 保留为分析基线，暂缓 P1-b1
│           └── 是：进入 P1-b1

P1-b1 运行中
├── history state/action condition 是否带来额外 gain？
│   ├── 否：回退到 P1-b0，不扩大 LoRA 注入范围
│   └── 是
│       └── 显存与延迟是否可接受？
│           ├── 否：降低 LoRA rank、缩短 history window 或退回 P1-b0
│           └── 是：进入 P2 局部 Render-and-Decode 诊断
```

该决策流的作用不是增加实验矩阵，而是防止路线在没有下游收益证据时继续膨胀。

## 12.7 本章结论

详细设计阶段应首先完成 8 个动作：

1. 跑通 RoboCasa[32]/RoboCasa365[19]；
2. 核验 AIRoA MoMa[36] 下载与 schema；
3. 试跑 EBench[35] 或至少确认其 mobile manipulation 任务可用；
4. 试跑 ManiSkill3[40] 数据导出；
5. 将 MoMa-Kitchen[39] 固定为 P0 readiness 诊断候选；
6. 维持 LIBERO[48]/RoboTwin 的辅助消融地位；
7. 将 MobileManiBench[33]/Kitchen-R[34] 标记为“开放后优先”；
8. 将 HomeRobot[7][20][21]/BEHAVIOR[24]/LAMBDA[38] 推迟到后期高难验证。

# 第十三章 风险、未决问题与后续调研计划

## 13.1 风险矩阵：v0.8 更新版

| 风险 | 概率 | 影响 | 典型候选 | 缓解策略 |
|---|---|---|---|---|
| P0 任务收益不明显 | 中 | 高 | P0 WAM-specific 指标提升但 TSR/SSR/NMS/RSR 无显著提升 | 降级路径：缩小到 readiness/risk/NBV 三类最可消费信号；从 overall TSR 改为阶段性 SSR/NMS/RSR；推迟 P1，先验证 P0-action head 最小闭环；若仍无收益，将 WAM 定位降级为诊断模型 |
| 长时序失败传播 | 中 | 高 | WAM 早期错误导致后续 action head 持续偏离 | 引入阶段性评测、short-horizon ablation、error reset 机制；先在 S2–S3 局部链路验证，再外推到 S4/S5 |
| 数据使用口径漂移 | 中 | 中-高 | 待核验 benchmark、二手资料或内部候选被误写成可用主证据 | 正文区分“已核验来源 / 待核验候选 / 内部核对材料”；所有主实验数据必须通过 Data Verification Gate |
| 引用编号或开放状态错误 | 中 | 中-高 | 2026 新论文、coming soon 项目、二手列表 | 文末显式标注待核验条目；详细设计前逐条打开 arXiv/官方仓库/HF/许可证页面；错误编号不得进入正式引用 |
| 未开源 / coming soon | 高 | 高 | MobileManiBench[33]、Kitchen-R[34]、AutoMoMa、部分新 benchmark | 不进入近期主闭环，放入“开放后优先验证” |
| 仅论文候选被误写成可用数据 | 中 | 高 | Kitchen-R[34]、AutoMoMa、部分 2026 新工作 | 正文标注“仅论文”，附录保留，不作实验承诺 |
| 视觉真实度不足 | 中 | 中 | MoMa-Kitchen[39]、CALVIN[51]、PyBullet/BestMan | 降级为 P0 readiness / affordance 诊断，不作主视觉 benchmark |
| 仿真器 / benchmark / 数据集混淆 | 中 | 高 | OmniGibson[45]、Habitat[42][43][44] 3.0[44]、AI2-THOR[68]、WorldGym[73]| 明确“仿真器而非数据集”或“在线评测而非离线数据” |
| 抓取机制不真实 | 中 | 中-高 | HomeRobot / OVMM[7][20][21] attach grasp | 用作高难评测参考，不作为真实接触抓取证据 |
| 数据模态不完整 | 高 | 高 | AIRoA MoMa[36]、Mobile ALOHA[37]、DROID[22]、BridgeData[60]、RoboCasa[32] 版本差异 | 核验 RGB、D、State、Action、Base、Joints、EEF、Success、Stage |
| 非移动操作被误当主证据 | 高 | 高 | LIBERO[48]、RoboTwin[49]、DROID[22]、Open X[23]、RLBench[50]、CALVIN[51] | 统一标注“辅助验证 / 数据工程参考”，不写主 benchmark |
| 高难环境拖慢主线 | 中 | 高 | BEHAVIOR-1K[24]、HomeRobot[7][20][21]、LAMBDA[38] | 推迟到后期外推验证 |
| P2 抢占 P0/P1 | 中 | 高 | BEHAVIOR[24]、Cosmos[6]/OSCAR/WPE 类 | P2 只做 keyframe-only /局部 demo |
| 真实数据任务太短 | 中 | 中 | AIRoA MoMa[36] | 按 episode 统计任务长度，短程任务用于标签/表征，不直接证明长时序 |
| schema 统一成本高 | 高 | 中 | Open X[23]、DROID[22]、BridgeData[60]、AgiBot World[61] | 只选 1–2 个先做 latent extraction，不全量集成 |
| P1-b1 条件注入过重 | 中 | 中-高 | Wan2.2 DiT-LoRA[14][25] | 先做 P1-b0 低风险基线，再做历史 state/action 条件注入；LoRA[25] 优先，全量微调暂缓 |


## 13.2 P0 收益不明显风险的量化触发条件

P0 是否进入 P1，不应只看 auxiliary loss 是否下降，而应看 WAM-specific 指标是否能转化为可测 downstream gain。详细设计阶段建议采用以下触发阈值作为初始判定口径：

| 触发条件 | 判定 | 降级路径 |
|---|---|---|
| TSR 绝对提升 < 5% | P0 对整体任务收益不明显 | 将 WAM heads 降级为诊断信号，不进入主闭环 |
| SSR 在“接近 / 视角调整 / 可操作姿态”阶段无提升 | 对 navigation-to-manipulation handoff 价值不足 | 只保留 readiness / NBV 相关 heads，移除弱相关 heads |
| NMS 无提升 | future representation 未被 action head 有效消费 | 调整 action-head 接入方式，先做最小闭环 |
| RSR 无提升 | failure-risk head 对恢复帮助不足 | 降低 failure-risk 权重，仅保留为分析指标 |
| WAM-specific 指标提升但 downstream 不提升 | 表征质量与任务收益脱节 | 推迟 P1，先做 coupling analysis 和 ablation |

这些阈值不是最终验收标准，而是详细设计阶段的初始 gate；具体数值需结合 baseline 方差和任务数量调整。

## 13.3 未决问题

1. RoboCasa365[19] 的 mobile base / navigation-manipulation 强度是否足以支撑移动操作主张？
2. AIRoA MoMa[36] 的数据是否可完整下载？是否包含 Base、EEF、Success、Lang？短程任务比例多高？
3. EBench[35] 的 mobile manipulation 任务是否能在本地快速跑通？
4. ManiSkill3[40] 是否有可直接用于 S2–S3 mobile-base / household 子任务的数据生成脚本？
5. MobileManiBench[33] 和 Kitchen-R[34] 何时开放代码/数据？
6. HomeRobot/OVMM[7][20][21] 的 attach grasp 对 WAM readiness / failure-risk 评估影响多大？
7. MoMa-Kitchen[39] 的 V1 视觉是否足够支撑 P0 readiness 诊断？
8. DROID[22]/Open X/BridgeData 的 schema 统一成本是否超过 P1 的三个月预算？
9. 如果 P0 的 WAM-specific 指标提升但下游任务收益不明显，是否降级为诊断模型，还是继续推进 P1？
10. 文末待核验参考资料在详细设计冻结前是否已经逐条确认标题、编号、开放状态和许可证？

## 13.4 后续核验计划

### 第一周：可用性核验

- RoboCasa365[19] / RoboCasa[32]：代码安装、示例任务、数据下载、replay；
- AIRoA MoMa[36]：HF 数据下载、schema 检查、模态完整性；
- EBench[35]：仓库安装、示例任务、mobile manipulation 是否可运行；
- ManiSkill3[40]：安装、数据导出、RGB-D/State/Action schema。

第一周建议交付物不是口头结论，而是四份标准核验记录：

| 数据资源 | 最小交付物 | 不通过时处理 |
|---|---|---|
| RoboCasa[32] / RoboCasa365[19] | 安装记录、10 个 episode replay、RGB/State/Action/Lang schema 表、P0 标签可构造性说明 | 降级为 sanity check，只保留可读数据子集 |
| AIRoA MoMa[36] | 下载记录、episode schema、RGB/State/Action/Stage/Success/Base/EEF 字段完整性表、短程/长程统计 | 字段不完整则降级为真实数据参考，不进入主实验 |
| EBench[35] | 任务列表、可运行样例、mobile manipulation 五轴诊断映射表 | 无法运行则仅保留为评测方法参考 |
| ManiSkill3[40] | 安装脚本、RGB-D/Seg/State/Action 导出样例、可生成任务清单 | 降级为操作/数据生成辅助底座 |


### 第二周：监督信号核验

- 对 RoboCasa[32]/RoboCasa365[19] 构造 task progress、success/failure、object-state-change；
- 对 AIRoA MoMa[36] 构造 primitive action、Stage/Subgoal、failure/contact；
- 对 MoMa-Kitchen[39] 构造 readiness / final-pose affordance；
- 对 LIBERO[48]/RoboTwin[49] 构造 P0/P1 sanity check。

### 第三周：latent extraction 核验

- 从 RoboCasa[32]/AIRoA MoMa[36]/DROID[22]/BridgeData[60] 中选 1–2 个抽取 Wan2.2[14] VAE latent；
- 测量 latent extraction 成本；
- 检查 latent 与 task progress / action outcome 的相关性。

### 第四周：详细设计输入冻结

- 决定第一闭环 benchmark；
- 决定真实数据核验集；
- 决定 P0 诊断集；
- 决定 P1 latent source；
- 决定后期高难验证集。


## 13.5 Data Verification Gate 判定标准与 P1-b1 工程 invariant 检查

以 AIRoA MoMa[36] 为例，Data Verification Gate 应至少覆盖以下判定项：

| 核验项 | 通过标准 | 不通过时降级路径 |
|---|---|---|
| 数据可下载 | 能完成数据下载并读取样例 episode | 降级为仅论文/真实系统参考 |
| schema 可读 | 能解析 RGB、State、Action、时间戳和 episode 边界 | 不进入主实验，只做背景引用 |
| Base / EEF / joints 字段 | 至少存在一种可稳定映射到机器人运动状态的字段 | 标注为操作数据为主，mobile 强度下调 |
| 任务长度分布 | 能区分短程/长程 episode，并统计长度 | 短程用于标签构造，不证明长时序 |
| success / failure 或 stage 标注 | 存在可用显式标签，或可由状态变化构造弱标签 | 增加人工核验成本，降低优先级 |
| P0/P1 监督可构造 | 能构造 progress/readiness/risk 或 current/future latent target | 不进入 P0/P1 主线 |

P1-b1 的防泄漏检查必须单独执行：

```text
□ WAM 输入窗口只包含 t <= T_history 的视觉、state、action。
□ future latent target 只由 t > T_history 的未来帧经 Wan2.2-VAE[14] 编码得到。
□ 禁止把 t > T_history 的未来 action label 输入 WAM。
□ 允许 action head 使用历史 state/action 与 predicted future latent 预测未来 action chunk。
□ 训练脚本需要断言 WAM input tensor 中不存在 future action token / future action array。
□ 数据切片需要保存 history window、future horizon、action label 三者的索引边界，便于复查。
```

## 13.6 12 周里程碑建议

| 周次 | 里程碑 | 验收标准 | 失败时降级路径 |
|---|---|---|---|
| 1–2 | Data Verification Gate | 至少 3–4 个候选通过可用性核验 | 缩小到 RoboCasa[32]/RoboCasa365[19] + LIBERO[48] sanity check |
| 3–4 | P0 原型 | P0 heads 可计算 loss，标签构造自动化 | 减少 head 数量，优先 readiness / progress / risk |
| 5–6 | P0 downstream 评估 | WAM-specific 与 TSR/SSR/NMS/RSR 至少一类下游指标相关 | 降级为诊断模型，暂缓 P1 |
| 7–8 | P1-b0 原型 | Wan2.2[14] latent 缓存完成，future latent prior 可训练 | 检查 VAE target 和数据切片，必要时回退 P0 |
| 9–10 | P1-b1 原型 | history condition 注入成功，无 future action 泄漏，latency 可 profile | 回退 P1-b0，降低 LoRA rank 或缩短 history window |
| 11–12 | P2 局部验证 | 能 decode 少量 P1 predicted latent 形成诊断 keyframe / short clip | P2 降级为报告与误差分析，不进入主闭环 |

## 13.7 本章结论

数据集与 benchmark 的最大风险不是“候选不够”，而是“候选过多但可用性、移动性、模态和工程成本没有核验”。因此，详细设计必须设置 Data Verification Gate：任何数据集进入主线前，必须先确认可下载/可运行、schema 可读、动作/状态/语言/成功失败/阶段标签是否足够、是否能构造 P0/P1 监督信号。

# 第十四章 结论

本报告将项目路线收敛为可实施的三阶段结构，并在 v1.2 中根据数据集 / benchmark v0.8 核对结果重构了实验验证闭环：

```text
P0：Video-Generation-Free WAM，训练时引入 action-relevant future supervision，推理时不生成未来像素。
P1：Latent-Only WAM，P1-b0 作为视觉/语言 future latent prior 基线，P1-b1 作为 robot-history-conditioned DiT-LoRA 候选增强路线。
P2：Render-and-Decode WAM，优先解码 P1 predicted latent 做 keyframe / short-clip 诊断，continuous video generation 仅作长期参考。
```

最重要的技术路线结论仍然是：

1. P0 采用 Video-Generation-Free WAM 合理，而且是三个月内最推荐路线；Qwen3VL[29] 可作为优先候选 backbone，但不能写入阶段正式命名；
2. P1 采用 Latent-Only WAM 合理，Wan2.2[14] 可作为优先候选 video latent foundation model；P1-b 应区分 b0/b1：b0 是不改候选视频模型的 future latent prior，b1 是注入历史 state/action 的 DiT-LoRA WAM，推荐主线为 b1；
3. P0 与 P1 必须分章，因为它们解决的问题不同；
4. Cosmos[6]不应作为 P1 默认底座，主要放入 P2 / 长期架构；
5. 技术调研报告必须以“典型路线—实现机制—效果—优劣—适配性—推荐结论”为主体；
6. 详细设计阶段应优先实现 P0 快速闭环，再推进 P1-b0/P1-b1 latent 对比，P2 只做局部验证。

根据 v0.8 数据集 / benchmark 核对结果，本报告进一步明确：

1. **第一闭环不应从最高难 benchmark 开始**。RoboCasa[32] / RoboCasa365[19] 更适合近期落地，但要标注其 mobile base / navigation-manipulation 强度仍需核实。
2. **AIRoA MoMa[36] 必须保留**。它是少见的真实移动操作、多模态、层级标注数据，是 P0/P1 真实数据核心候选，但要注意当前核对备注中的短程任务边界。
3. **MobileManiBench[33] / Kitchen-R[34] / EBench[35] 上调但分级处理**。MobileManiBench[33] 和 Kitchen-R[34] 因未开放不能作为近期闭环；EBench[35] 可作为 Isaac Sim 诊断候选。
4. **MoMa-Kitchen[39] 定位下调**。它适合作 P0 readiness / final-pose affordance 诊断集，不作主视觉 benchmark。
5. **HomeRobot / OVMM[7][20][21]、LAMBDA[38]、BEHAVIOR-1K[24] 是后期高难验证集**。它们更能体现完整移动操作，但不宜第一步全量接入。
6. **LIBERO[48]、RoboTwin[49]、DROID[22]、Open X[23] 等不能作移动主证据**。它们进入辅助验证、数据工程和表征学习层。

因此，详细设计阶段的首要任务不是继续扩大候选清单，而是建立 Data Verification Gate，对 RoboCasa365[19]、RoboCasa[32]、AIRoA MoMa[36]、EBench[35]、ManiSkill3[40]、MoMa-Kitchen[39]、LIBERO[48]、RoboTwin 2.0[49] 进行可用性、schema、模态、replay/eval、P0 监督构造和 P1 latent extraction 的核验。只有通过该 gate 的数据，才进入 P0/P1 主实验矩阵。


# 附录 A：本项目 WAM 数据集 / Benchmark 全量候选表 v0.8

> 本附录保留 v0.8 中的全量候选信息，用于避免正文过度堆表，同时确保数据集/benchmark 专项调研的关键信息不丢失。正式执行前，仍需对可用性、社区信号、数据开放状态、schema 和 license 做实时核验。

# 本项目 WAM 数据集 / Benchmark 全量候选表 v0.8

> 研究题目：**《面向移动操作的 World Action Model 构建与评测研究》**  
> 本表用于本项目数据集与 Benchmark 专项调研，目标是服务 **P0 = Video-Generation-Free WAM**、**P1 = Latent-Only WAM**、**P2 = Render-and-Decode WAM** 的路线选择与可复现实验闭环。  
> 当前版本重点覆盖：移动操作相关性、空间难度、仿真器/平台、视觉真实感、开源生态、可用性/核对状态、人工核对备注、数据模态、任务规模、本项目定位和相关链接。  
> 社区信号、stars、forks、issues、数据开放状态会变化；正式写入报告前建议逐项实时核验。

---

## 0. 字段说明

| 字段 | 说明 |
|---|---|
| 顺序 | 当前建议验证优先级，不等于最终推荐等级 |
| 名称 | 数据集 / benchmark / simulator / evaluation framework |
| 可用性 / 核对状态 | 可用 / 未开源 / coming soon / 数据可下 / 仅论文 / 代码可用但数据待核 / 仿真器而非数据集 / 非移动操作主证据等 |
| 人工核对备注 | 人工核对时发现的关键边界，例如“短程任务”“抓取吸附”“导航非移动操作”“渲染逼真”等 |
| 相关链接 | 项目主页 / 论文 / 代码仓 / 数据 / Challenge |
| 当前社区信号 | 社区成熟度、仓库活跃度、是否需要实时核验 stars/forks/issues |
| 类型 | 真实 / 仿真 / 混合 / 评测框架 / 数据生成系统 / 方法参考 |
| 仿真器 / 平台 | Isaac Sim / Habitat[42][43][44] / OmniGibson[45] / MuJoCo / SAPIEN / PyBullet / Real 等 |
| 空间 | S0–S5，是否跨房间或跨楼层 |
| 数据类型 / 模态 | RGB、D、RGB-D、Seg、PCD、State、Joints、EEF、Base、Action、Lang、FT、Tactile、Audio、Stage/Subgoal、Success/Fail 等 |
| 数据/任务 | 规模、任务数、场景数、demo/trajectory 数、时长 |
| 视觉/生态 | V0–V4 视觉真实度；M0–M4 开源生态成熟度 |
| 数据情况与任务 | 任务链覆盖、标注质量、split、是否可 replay/eval、特殊注意事项 |
| 本项目定位 | 主 benchmark / 训练数据 / 诊断集 / 辅助验证 / 长期参考 |

---

## 1. 标记说明

### 1.1 空间难度 S0–S5

| 等级 | 含义 | 典型形式 |
|---|---|---|
| S0 | 固定桌面 / 无移动 | LIBERO[48]、RLBench[50]、CALVIN[51]、RoboTwin[49] 多数任务 |
| S1 | 单工作台 / 局部操作空间 | 局部厨房台面、装配桌、局部抓取 |
| S2 | 单房间移动操作 / last-mile | 厨房内移动、接近、视角调整、操作位姿选择 |
| S3 | 多区域单房间 | 同一厨房内 fridge / cabinet / sink / table / island 跨区域 |
| S4 | 跨房间同楼层 | bedroom → kitchen，living room → table |
| S5 | 跨楼层 / 开放长距离 | multi-floor、真实 grocery store、开放商店 |

### 1.2 视觉等级 V0–V4

| 等级 | 含义 | 代表 |
|---|---|---|
| V4 | 高真实感 / Omniverse / real scan / photorealistic | Isaac Sim、OmniGibson[45]、Habitat-HM3D/HSSD |
| V3 | 高质量合成资产 / 纹理丰富 | RoboCasa[32]、RoboCasa365[19]、部分 ManiSkill3[40] |
| V2 | 可用但仿真感明显 | AI2-THOR[68]、iGibson[46]、基础 SAPIEN、RLBench[50] |
| V1 | 功能/几何仿真为主 | PyBullet/BestMan、基础 MuJoCo、CoppeliaSim |
| V0 | 低保真 / toy sim | 简化占位视觉 |

### 1.3 生态成熟度 M0–M4

| 等级 | 含义 |
|---|---|
| M4 | 社区成熟，文档/代码/issue/release/数据生态较完整 |
| M3 | 可用且有一定社区，适合优先验证 |
| M2 | 新仓或小社区，论文附带代码为主，需小规模试跑 |
| M1 | 开源弱 / 数据可用性待确认 |
| M0 | 无可靠代码或名称未确认 |

---

# A. 高真实感 / 工程生态较强 / 移动操作相关

| 顺序 | 名称 | 可用性 / 核对状态 | 人工核对备注 | 相关链接 | 当前社区信号 | 类型 | 仿真器 / 平台 | 空间 | 数据类型 / 模态 | 数据/任务 | 视觉/生态 | 数据情况与任务 | 本项目定位 |
|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **RoboCasa365[19]** | 可用 | 可用 | [主页](https://robocasa.ai/)<br>[论文](https://arxiv.org/abs/2603.04356)<br>[代码](https://github.com/robocasa/robocasa)<br>[Release](https://github.com/robocasa/robocasa/releases) | 高；RoboCasa[32] 生态成熟，有 release；stars/forks/issues 需实时复核 | 仿真 benchmark / 数据集 | RoboCasa[32] / robosuite / MuJoCo | S2–S3，厨房 | RGB、State、Action、Lang；可能含 Joints、EEF、gripper、object state；Depth/Seg 需核验 | 365 tasks，2500 kitchen scenes，600h+ human demos，1600h+ synthetic demos | V3 / M4 | 大规模厨房 household manipulation；官方称 RoboCasa365[19] 是在 RoboCasa[32] 上扩展的 household mobile manipulation benchmark | 高优先；稳定工程底座；需核实 mobile base / navigation-manipulation 强度 |
| 2 | **RoboCasa[32]** | 可用 | 可用 | [主页](https://robocasa.ai/)<br>[论文](https://arxiv.org/abs/2406.02523)<br>[代码](https://github.com/robocasa/robocasa) | 高；成熟仓库、文档和数据工具较完整 | 仿真 benchmark | robosuite / MuJoCo | S1–S3，厨房 | RGB、State、Action、Lang、sim state；Depth/Seg/EEF/Joints 需看数据版本 | 100 tasks，厨房环境，human + synthetic demos | V3 / M4 | 视觉和厨房资产比 LIBERO[48]/MoMa-Kitchen 更好，工程复现性强 | P0/P1 快速闭环；不是纯移动操作主证据 |
| 3 | **MobileManiBench[33]** | 尚未开源 / coming soon | 尚未开源；代码/数据未开放，暂不能作为近期主闭环 | [主页](https://dexhand.github.io/MobileManiBench_Website/)<br>[论文](https://arxiv.org/abs/2602.05233)<br>[HTML](https://arxiv.org/html/2602.05233v1)<br>代码/数据：coming soon | 待核；论文强，开源成熟度需查 | 仿真移动操作 benchmark | Isaac Sim | S2–S3，是否跨房间待核 | Lang、multi-view RGB-D-Seg、object/robot states、Action；应含 Base/arm/gripper，需核验 | 2 mobile platforms，2 cameras，630 objects，20 categories，5 skills，100 scenes，300K trajectories | V4 / M1–M2 | 论文明确基于 Isaac Sim，自动生成多模态移动操作轨迹，含语言、RGB-depth-seg、同步状态和动作 | 若代码/数据开放，应作为仿真移动操作主候选 |
| 4 | **Kitchen-R[34]** | 未开源 / 仅论文候选 | 未开源；未找到代码/数据入口 | [论文](https://arxiv.org/abs/2508.15663)<br>[HTML](https://arxiv.org/html/2508.15663v1)<br>代码/数据：没找到 | 待核；论文强，仓库/数据需确认 | 仿真移动操作 benchmark | Isaac Sim | S2–S3，厨房 | Lang、task-plan、low-level control、RGB/RGB-D、State、Action、Base/Joints/EEF 需核验 | 500+ complex language instructions | V4 / M1–M2 | 厨房任务规划 + 低层控制联合评测；是否有离线 trajectory 是关键 | 单厨房移动操作强候选；需先查数据开放性 |
| 5 | **EBench[35]** | 可参考 / 需可用性核验 | 有vla和mobile manipulation | [主页](https://internrobotics.github.io/EBench-doc/)<br>[代码](https://github.com/InternRobotics/EBench)<br>[论文](https://arxiv.org/abs/2606.18239)<br>[HTML](https://arxiv.org/html/2606.18239v2) | 中低；新仓，需试跑 | VLA / mobile manipulation 诊断 benchmark | Isaac Sim | S0–S3，含 mobile manipulation | RGB、Lang、State、Action；Depth/Seg/EEF/Joints/Stage 需核验 | long-horizon、precision、dexterous、mobile manipulation 诊断 | V4 / M2 | 价值在五轴诊断：Scene、Atomic Skill、Horizon、Precision、Mobility | WAM 诊断评测候选 |
| 6 | **BEHAVIOR-1K[24]** | 可用 / 需版本核验 | 渲染逼真 | [主页](https://behavior.stanford.edu/)<br>[代码](https://github.com/StanfordVL/BEHAVIOR-1K)<br>[论文](https://arxiv.org/abs/2403.09227)<br>[HF 镜像](https://huggingface.co/datasets/StarVLA/BEHAVIOR-1K) | 高；社区/论文影响力高，但工程重 | household activity benchmark | OmniGibson[45] / Omniverse | S3–S5 | RGB-D、Seg、proprioception、Action、object state、task annotation；Joints/EEF/Base 需按任务核验 | 1000 everyday activities，50 scenes，9000+ objects | V4 / M4 | 覆盖刚体、软体、液体和复杂 household activity，任务最全但工程复杂 | 长期高复杂 household WAM benchmark，不宜第一步全量接入 |
| 7 | **OmniGibson[45]** | 仿真器可用 / 非单一数据集 | 仿真器，非数据集/benchmark；非单一数据集 | [主页](https://behavior.stanford.edu/)<br>[代码](https://github.com/StanfordVL/OmniGibson)<br>[论文/BEHAVIOR-1K](https://arxiv.org/abs/2403.09227) | 高；BEHAVIOR[24] 底座 | 仿真器 / 平台 | Omniverse / PhysX | S3–S5 | 可生成 RGB-D、Seg、State、Action、object state、复杂物理状态 | BEHAVIOR-1K[24] 底层仿真器 | V4 / M3–M4 | 更像仿真底座，不是单一数据集 | P1/P2 长期路线 |
| 8 | **HomeRobot / OVMM[7][20][21] sim** | 评测框架 / 在线评测 | 渲染较真实，但是抓取物品是直接吸附到夹爪上；偏在线评测/leaderboard | [主页](https://ovmm.github.io/)<br>[代码](https://github.com/facebookresearch/home-robot)<br>[论文](https://arxiv.org/abs/2306.11565)<br>[Challenge](https://aihabitat.org/challenge/2023_homerobot_ovmm/) | 中高；Habitat[42][43][44] 生态强 | 混合 benchmark | Habitat[42][43][44] / HomeRobot[7][20][21] | S4，可构造单房间 | RGB-D、Lang goal、object/receptacle target、continuous nav/manip Action、sim state | open-vocabulary object search + grasp + place | V3–V4 / M3 | Challenge 主要是在线评测，无静态 human traces；OVMM[7][20][21] 难度高 | 主移动操作评测候选；不适合第一步全量训练 |
| 9 | **Habitat[42][43][44] 2.0 / HAB[42]** | 可用 / 需版本核验 | — | [论文](https://arxiv.org/abs/2106.14405)<br>[Habitat-Lab](https://github.com/facebookresearch/habitat-lab)<br>[Habitat-Sim](https://github.com/facebookresearch/habitat-sim) | 高；Habitat[42][43][44] 生态成熟 | 移动操作 / rearrangement benchmark | Habitat[42][43][44] / ReplicaCAD | S3–S4 | RGB-D、robot state、Action、object state、proprioception | Fetch 执行 tidy house、prepare groceries、set table 等 | V3–V4 / M4 | 适合导航—操作—rearrangement；可在线生成数据 | 高难跨区域/跨房间验证 |
| 10 | **Habitat[42][43][44] Rearrangement Challenge[43]** | 评测框架 / 在线评测 | 偏在线评测/leaderboard | [Challenge](https://aihabitat.org/challenge/2022_rearrange/)<br>[Habitat-Lab](https://github.com/facebookresearch/habitat-lab) | 中高 | challenge | Habitat[42][43][44] | S3–S4 | RGB-D、State、Action、object pose/state、success metrics | rearrangement、pick/place/open/close | V3–V4 / M3 | 更像 leaderboard / online eval | 后期诊断集 |
| 11 | **Habitat[42][43][44] 3.0[44]** | 需核验 | — | [主页](https://aihabitat.org/habitat3/)<br>[Habitat-Lab](https://github.com/facebookresearch/habitat-lab)<br>[Habitat-Sim](https://github.com/facebookresearch/habitat-sim) | 高 | 仿真器 / HRI benchmark | Habitat[42][43][44] | S3–S4 | RGB-D、agent state、human/humanoid state、Action | social navigation / social rearrangement | V3–V4 / M4 | 动态人类和协作任务强 | 长期协作 WAM 参考 |
| 12 | **ManiSkill3[40]** | 可用 / 需版本核验 | — | [文档](https://maniskill.readthedocs.io/)<br>[代码](https://github.com/haosulab/ManiSkill)<br>[论文](https://arxiv.org/abs/2410.00425) | 高；社区成熟、活跃 | 仿真器 + benchmark | SAPIEN / ManiSkill[40][41] | S0–S3 | RGB、D、Seg、PCD、State、Action；Lang 通常需模板化 | GPU 并行仿真/渲染，多任务操作 | V3 / M4 | 适合快速生成视觉/状态/动作数据，支持多种 observation mode | 数据生成和操作子任务工程底座 |
| 13 | **ManiSkill2[41] / ManiSkill[40][41]** | 需核验 | — | [文档](https://maniskill.readthedocs.io/)<br>[代码](https://github.com/haosulab/ManiSkill)<br>[ManiSkill2[41] 论文](https://arxiv.org/abs/2302.04659) | 高 | 操作 benchmark | SAPIEN | S0–S2 | RGB-D、PCD、State、Action、controllers；Joints/EEF 需按 task 核验 | 20 task families，2000+ objects，4M+ demo frames | V2–V3 / M4 | 支持 stationary/mobile-base、single/dual-arm、rigid/soft-body | 操作技能泛化 / P0-P1 辅助 |

---

# B. 真实移动操作 / 真实数据

| 顺序 | 名称 | 可用性 / 核对状态 | 人工核对备注 | 相关链接 | 当前社区信号 | 类型 | 仿真器 / 平台 | 空间 | 数据类型 / 模态 | 数据/任务 | 视觉/生态 | 数据情况与任务 | 本项目定位 |
|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 14 | **AIRoA MoMa[36]** | 数据/代码待核实 | 短程任务；需继续核实链接/数据开放 | [HF 数据](https://huggingface.co/datasets/airoa-org/airoa-moma)<br>[HF 文件树](https://huggingface.co/datasets/airoa-org/airoa-moma/tree/main)<br>[论文](https://arxiv.org/abs/2509.25032)<br>代码：待核实 | 中；HF 数据，真实移动操作稀缺，社区热度需核验 | 真实移动操作数据集 | Toyota HSR | S2–S4，需按 episode 判断 | RGB、Joints、FT、internal State、primitive Action、Stage/Subgoal；Depth/Lang/Base/EEF/Success 需核验 | 25,469 episodes，约 94h | Real / M2–M3 | 真实、长时序、多模态、层级标注；非常适合 progress/readiness/failure 标签 | 真实 P0/P1 核心候选，必须验证 |
| 15 | **Mobile ALOHA[37]** | 可用 / 需版本核验 | 不错 | [主页](https://mobile-aloha.github.io/)<br>[论文](https://arxiv.org/abs/2401.02117)<br>[代码](https://github.com/MarkFzp/mobile-aloha) | 高；影响力大 | 真实移动双臂系统 / 数据 | mobile base + 双臂 ALOHA | S2–S5 | multi-camera RGB、dual-arm Action、Joints、gripper、Base；Depth/Lang/Success 需核验 | 50 demos/task，厨房、开柜、进电梯等 | Real / M4 | 真实 whole-body mobile manipulation 形态很关键 | 真实移动操作任务形态参考 |
| 16 | **LAMBDA[38] / λ Benchmark[38]** | 可参考 / 需可用性核验 | — | [主页](https://lambdabenchmark.github.io/)<br>[代码](https://github.com/h2r/LAMBDA)<br>[论文](https://arxiv.org/abs/2412.05313) | 低-中；仓库小，但数据/文档明确 | 真实+仿真 benchmark | sim + real mobile manipulator | S4–S5 | Lang、demo trajectories；RGB/RGB-D、State、Action、Base、EEF、Joints 需下载核验 | 571 human-collected demos | Real+Sim / M2 | 明确是 multi-room、multi-floor、long-horizon pick-and-place；难度高 | 高难空间外推 / 数据效率 benchmark，不作第一步 |
| 17 | **HomeRobot / OVMM[7][20][21] real** | 可用 / 需版本核验 | — | [主页](https://ovmm.github.io/)<br>[代码](https://github.com/facebookresearch/home-robot)<br>[论文](https://arxiv.org/abs/2306.11565) | 中高；复现成本高 | 真实+仿真 benchmark | Hello Robot Stretch | S4 | RGB-D、Lang goal、nav/manip Action、object/receptacle target | OVMM[7][20][21] real-world component | Real / M3 | 真实 Stretch 复现价值高，但离线数据不一定完整 | 真实高难评测参考 |
| 18 | **LaNMP** | 待核实 | 需继续核实链接/数据开放 | [论文/HTML](https://arxiv.org/html/2412.05313v1)<br>项目主页/代码：待核实 | 低-中；需核验开源 | 真实/混合移动操作 | Spot / mobile platform | S4 | Lang、RGB-D、Seg、pose、nav/manip data；State/Action/EEF/Base 需核验 | sim + real long-horizon room-to-room tasks | Real+Sim / M1–M2 | 可能与 LAMBDA[38] 类似，适合高难真实/混合验证 | 补充候选 |
| 19 | **Demonstrating Mobile Manipulation in the Wild / SHOPPER** | 数据/代码待核实 | 需继续核实链接/数据开放 | [论文](https://arxiv.org/abs/2401.01474)<br>[RSS 页面](https://roboticsconference.org/2023/program/papers/055/)<br>代码/数据：待核实 | 低；系统论文为主 | 真实系统评测 | grocery-store mobile manipulator | S5 | robot logs、metrics、failure modes；RGB/RGB-D、State、Action 需核验 | grocery store 多周 field tests | Real / M1 | 真实开放环境，数据未必开放 | 失败 taxonomy / 系统指标参考 |
| 20 | **SHOPPER grasping follow-up** | 数据/代码待核实 | 需继续核实链接/数据开放 | [论文](https://arxiv.org/abs/2504.12512)<br>[HTML](https://arxiv.org/html/2504.12512)<br>代码/数据：待核实 | 低；需核验 | 真实抓取评测 | SHOPPER | S5 | grasp attempts、Success/Fail；RGB-D/State/Action 需核验 | grocery grasping attempts | Real / M1 | 适合真实抓取失败模式 | failure-risk 参考 |

---

# C. 单房间 / last-mile 诊断集

| 顺序 | 名称 | 可用性 / 核对状态 | 人工核对备注 | 相关链接 | 当前社区信号 | 类型 | 仿真器 / 平台 | 空间 | 数据类型 / 模态 | 数据/任务 | 视觉/生态 | 数据情况与任务 | 本项目定位 |
|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 21 | **MoMa-Kitchen[39]** | 可参考 / 需可用性核验 | — | [主页](https://momakitchen.github.io/)<br>[代码](https://github.com/MoMaKitchen/MoMaKitchen)<br>[论文](https://arxiv.org/abs/2503.11081)<br>[CVF PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_MoMa-Kitchen_A_100K_Benchmark_for_Affordance-Grounded_Last-Mile_Navigation_in_Mobile_ICCV_2025_paper.pdf) | 低-中；小仓，论文附带代码为主 | last-mile affordance benchmark | BestMan / PyBullet | S2–S3，厨房 | RGB-D、Affordance floor labels、robot-specific parameters、target object；Action/Joints/EEF/Base/success reason 需核验 | 100K+ samples | V1 / M2 | 不是 Isaac；偏 final navigation position / manipulation-readiness | P0 readiness / final-pose affordance 诊断集，不作主 benchmark |
| 22 | **SAGE-3D[11] / Isaac 自建路线** | 自建路线 / 非公共 benchmark | 导航，非移动操作 | [代码](https://github.com/Galery23/SAGE-3D_Official)<br>[解析站](https://zread.ai/Galery23/SAGE-3D_Official) | 自有工程 | 自建数据生成路线 | Isaac / 3DGS / 自定义 | 可控 S2–S4 | 可生成 RGB、D、Pose/Calib、trajectory、PCD、Seg、occupancy、collision labels | 自建轨迹和渲染数据 | 取决于资产 / 自控 | 公共 benchmark 属性弱，但空间难度和字段可控 | demo / ablation / 自有补充数据 |
| 23 | **AutoMoMa** | 数据/代码待核实 | 需继续核实链接/数据开放 | [论文](https://arxiv.org/abs/2604.12565)<br>代码/数据：待核实 | 待核 | trajectory generation | 待核 | S2–S3 | trajectory、Base/Arm Action、State 可能有 | whole-body trajectory generation | 待核 | 方法相关性强，但数据开放需查 | 方法参考 |

---

# D. 高复杂 household / 跨房间仿真

| 顺序 | 名称 | 可用性 / 核对状态 | 人工核对备注 | 相关链接 | 当前社区信号 | 类型 | 仿真器 / 平台 | 空间 | 数据类型 / 模态 | 数据/任务 | 视觉/生态 | 数据情况与任务 | 本项目定位 |
|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 24 | **iGibson[46] 2.0[46]** | 可用 / 需版本核验 | — | [主页](https://svl.stanford.edu/igibson/)<br>[代码](https://github.com/StanfordVL/iGibson)<br>[论文](https://arxiv.org/abs/2108.03272) | 中高 | household embodied sim | iGibson[46] | S3–S4 | RGB-D、Seg、State、Action、object states、predicate states | household tasks / object states | V2 / M3 | 适合可操作状态、predicate、任务逻辑 | 状态变化 / 可操作性参考 |
| 25 | **BEHAVIOR[24] Challenge[24]** | 评测框架 / 在线评测 | 偏在线评测/leaderboard | [Challenge](https://behavior.stanford.edu/challenge/call_for_participation.html)<br>[BEHAVIOR[24] 主页](https://behavior.stanford.edu/)<br>[代码](https://github.com/StanfordVL/BEHAVIOR-1K) | 中高 | challenge | OmniGibson[45] | S3–S5 | RGB-D、proprioception、object GT state、Action、fine-grained annotation | BEHAVIOR[24] subset challenge | V4 / M3 | 工程重，适合作长期评测 | 长期 leaderboard |
| 26 | **Galactic** | 数据/代码待核实 | 需继续核实链接/数据开放 | [论文](https://arxiv.org/abs/2306.07552)<br>代码/数据：待核实 | 中低 | mobile rearrangement RL framework | Habitat[42][43][44] | S3–S4 | RGB-D、State、Action、object state，可生成 | mobile rearrangement RL data stream | V3 / M2 | 偏 RL 吞吐，不是离线 benchmark 优先 | 数据生成参考 |
| 27 | **ManiSkill-HAB[47]/ MS-HAB[47]** | 来源已核验 / 需试跑 | — | [主页](https://arth-shukla.github.io/mshab/)<br>[论文](https://arxiv.org/abs/2412.13211)<br>[OpenReview](https://openreview.net/forum?id=6bKEWevgSd)<br>[匿名仓库](https://github.com/anonsubmit0/maniskill-hab) | 中；新近，需试跑 | HAB + ManiSkill[40][41] | ManiSkill[40][41] / HAB | S2–S4 | RGB-D、PCD、State、Action、object state、Base/Joints/EEF | in-home rearrangement / low-level manipulation | V3 / M2–M3 | 如果跑通，可能是高速 household 中间路线 | 候选中间路线 |

---

# E. 非移动但成熟的操作辅助 benchmark

| 顺序 | 名称 | 可用性 / 核对状态 | 人工核对备注 | 相关链接 | 当前社区信号 | 类型 | 仿真器 / 平台 | 空间 | 数据类型 / 模态 | 数据/任务 | 视觉/生态 | 数据情况与任务 | 本项目定位 |
|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 28 | **LIBERO[48]** | 可用 / 需版本核验 | — | [主页](https://libero-project.github.io/main.html)<br>[代码](https://github.com/Lifelong-Robot-Learning/LIBERO)<br>[论文](https://papers.neurips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html) | 高；VLA 常用，生态成熟 | 语言条件操作 benchmark | robosuite / MuJoCo | S0 | RGB/multi-view、State、Action、Lang；Joints/EEF/gripper/object state 可从 robosuite 获得；Depth/Seg 看版本 | 130 language-conditioned tasks | V2 / M4 | 官方论文说明 130 个语言条件任务，分 object/goal/spatial/long-horizon 等 suite | 快速 sanity check；不能作移动主证据 |
| 29 | **LIBERO-Long[48]** | 需核验 | — | [主页](https://libero-project.github.io/main.html)<br>[代码](https://github.com/Lifelong-Robot-Learning/LIBERO)<br>[论文](https://papers.neurips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html) | 高 | LIBERO[48] 子集 | robosuite / MuJoCo | S0 | 同 LIBERO[48] | long-horizon subset | V2 / M4 | 长时序桌面任务 | 长时序操作辅助 |
| 30 | **LIBERO-Plus[48]** | 数据/代码待核实 | 需继续核实链接/数据开放 | [论文](https://arxiv.org/abs/2510.13626)<br>代码/数据：待核实 | 中；新近 | robustness benchmark | LIBERO[48] setting | S0 | RGB、State、Action、Lang、perturbation metadata | 多扰动鲁棒性任务 | V2 / M2–M3 | viewpoint、lighting、language、layout、noise 等扰动 | P0/P1 robustness diagnostic |
| 31 | **LIBERO-PRO[48]** | 需核验 | — | [论文](https://arxiv.org/abs/2510.03827)<br>[代码](https://github.com/Zxy-MLlab/LIBERO-PRO) | 中；新近 | OOD / anti-memorization benchmark | LIBERO[48] setting | S0 | 同 LIBERO[48] + OOD metadata | OOD / anti-memorization | V2 / M2 | 防止只背 benchmark | OOD 辅助 |
| 32 | **RoboTwin 2.0[49]** | 可用 / 需版本核验 | — | [主页](https://robotwin-platform.github.io/)<br>[代码](https://github.com/robotwin-Platform/robotwin)<br>[论文](https://arxiv.org/abs/2506.18088) | 高；社区热度较高 | 双臂操作 benchmark | RoboTwin[49] / 仿真 | S0–S1 | RGB/RGB-D 需核验、dual-arm State/Action、Joints/EEF、Lang variation、domain randomization metadata | 50 dual-arm tasks，731 objects，147 categories，5 embodiments | V2–V3 / M4 | 双臂泛化、合成数据、domain randomization 强 | 操作泛化辅助 |
| 33 | **RoboTwin[49] 1.0** | 需核验 | — | [主页](https://robotwin-platform.github.io/)<br>[代码](https://github.com/robotwin-Platform/robotwin)<br>[论文](https://arxiv.org/abs/2504.13059) | 高 | 双臂 benchmark | RoboTwin[49] | S0–S1 | RGB/RGB-D、State、Action、task code 需核验 | 双臂 digital twins | V2–V3 / M4 | 2.0 已是主线版本 | 历史对照 |
| 34 | **RLBench[50]** | 可用 / 需版本核验 | — | [主页](https://sites.google.com/view/rlbench)<br>[代码](https://github.com/stepjam/RLBench)<br>[论文](https://arxiv.org/abs/1909.12271) | 中高 | 多任务操作 benchmark | CoppeliaSim / PyRep | S0 | RGB、D、Seg、proprioception、Action；EEF/Joints 通常可得 | 100 hand-designed tasks | V2 / M3–M4 | 经典视觉操作 benchmark | 操作辅助 |
| 35 | **CALVIN[51]** | 可用 / 需版本核验 | — | [主页](https://calvin.cs.uni-freiburg.de/)<br>[代码](https://github.com/mees/calvin)<br>[论文](https://arxiv.org/abs/2112.03227) | 中高 | long-horizon language manipulation | PyBullet | S0–S1 | Lang、RGB、robot_obs、scene_obs、Action；Depth/Seg 看版本 | 长时序语言条件桌面操作 | V1–V2 / M3 | 强在 language + long-horizon，不是移动操作 | 任务进度 / world model 辅助 |
| 36 | **FurnitureBench[52]** | 可用 / 需版本核验 | — | [主页](https://clvrai.github.io/furniture-bench/)<br>[代码](https://github.com/clvrai/furniture-bench)<br>[论文](https://arxiv.org/abs/2305.12821) | 中高 | 真实+仿真长时序操作 | Real + FurnitureSim | S1 | RGB/multi-view、State、Action、task metadata；Depth/Seg/FT 需核验 | 219.6h，5100 demos | Real+Sim / M3 | 长时序接触丰富装配 | 真实接触任务参考 |
| 37 | **FurnitureSim** | 需核验 | — | [FurnitureBench[52] 主页](https://clvrai.github.io/furniture-bench/)<br>[代码](https://github.com/clvrai/furniture-bench)<br>[论文 PDF](https://www.roboticsproceedings.org/rss19/p041.pdf) | 中 | 仿真组件 | FurnitureSim | S1 | RGB、D、State、Action、object/furniture state 可生成 | FurnitureBench[52] sim | V2–V3 / M3 | 配套 FurnitureBench[52] | 仿真-真实对照 |
| 38 | **MimicGen[53]** | 可用 / 需版本核验 | — | [主页](https://mimicgen.github.io/)<br>[代码](https://github.com/NVlabs/mimicgen)<br>[论文](https://arxiv.org/abs/2310.17596) | 中高 | 数据生成系统 | robosuite / 多仿真 / real | S0–S1 | State、Action、robot/object state；RGB/D/EEF/Joints 取决于采集配置 | 50K+ demos，18 tasks | V2 / M3–M4 | 少量 human demos 扩增为大量 demos | 数据 scaling 参考 |
| 39 | **SkillMimicGen[54]** | 待核实 | 需继续核实链接/数据开放 | [论文](https://arxiv.org/abs/2410.18907)<br>项目主页/代码：待核实 | 中 | 数据生成系统 | sim + real | S0–S1 | RGB、State、Action、stage metadata 需核验 | 长时序技能数据扩增 | V2 / M2 | MimicGen[53] 扩展 | scaling 参考 |
| 40 | **SoftMimicGen[55]** | 待核实 | 需继续核实链接/数据开放 | [论文](https://arxiv.org/abs/2603.25725)<br>项目主页/代码：待核实 | 待核 | deformable data generation | 待核 | S0–S1 | RGB/RGB-D、State、Action、deformable state 需核验 | rope/towel/tissue 等 deformable | 待核 / M1–M2 | 新近候选 | 长期 deformable 参考 |
| 41 | **MuBlE / SHOP-VRB2** | 待核实 | 需继续核实链接/数据开放 | [论文](https://arxiv.org/abs/2503.02834)<br>项目主页/代码：待核实 | 待核 | 物理/视觉推理 benchmark | MuJoCo + Blender / robosuite | S0–S1 | RGB/render、State、Action、physical logs、reasoning labels 需核验 | long-horizon visual reasoning + physical interaction | V2–V3 / M1–M2 | 对 action-outcome reasoning 有参考 | P2 / 推理参考 |
| 42 | **Meta-World[56]** | 可用 / 需版本核验 | — | [主页](https://meta-world.github.io/)<br>[代码](https://github.com/Farama-Foundation/Metaworld)<br>[论文](https://arxiv.org/abs/1910.10897) | 高；经典 | multi-task/meta-RL benchmark | MuJoCo | S0 | low-dim State、Action；RGB 可渲染，Lang 无 | 50 manipulation tasks | V1 / M4 | 与 WAM 移动操作距离远 | 背景 |
| 43 | **BulletArm[57]** | 待核实 | 需继续核实链接/数据开放 | [论文](https://arxiv.org/abs/2205.14292)<br>代码：待核实 | 中低 | PyBullet 操作 benchmark | PyBullet | S0 | State、heightmap/RGB-D-like obs、Action；Lang/Seg 待核 | 31 manipulation tasks | V1 / M2 | 旧式操作 benchmark | 低优先 |
| 44 | **DexJoCo[58]** | 待核实 | 需继续核实链接/数据开放 | [论文](https://arxiv.org/abs/2605.16257)<br>项目主页/代码：待核实 | 待核 | dexterous benchmark | MuJoCo | S0–S1 | State、Action、RGB、dexterous hand joints、object pose 需核验 | dexterous / bimanual / tool-use | V2 / M1–M2 | 灵巧手方向 | 长期参考 |
| 45 | **VLABench[59]** | 需核验 | — | [主页](https://vlabench.github.io/)<br>[论文](https://arxiv.org/abs/2412.18194)<br>[CVF PDF](https://openaccess.thecvf.com/content/ICCV2025/papers/Zhang_VLABench_A_Large-Scale_Benchmark_for_Language-Conditioned_Robotics_Manipulation_with_Long-Horizon_ICCV_2025_paper.pdf) | 中 | VLA / LCM benchmark | 仿真操作 | S0–S1 | Lang、RGB-D、proprioception、Action、object metadata 需核验 | 100 task categories，2000+ objects | V2–V3 / M2 | 强调 world knowledge/common sense | VLA/WAM 诊断 |

---

# F. 真实大规模操作数据 / 数据工程参考

| 顺序 | 名称 | 可用性 / 核对状态 | 人工核对备注 | 相关链接 | 当前社区信号 | 类型 | 仿真器 / 平台 | 空间 | 数据类型 / 模态 | 数据/任务 | 视觉/生态 | 数据情况与任务 | 本项目定位 |
|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 46 | **Open X-Embodiment[23]** | 可用 / 需版本核验 | — | [主页](https://robotics-transformer-x.github.io/)<br>[代码](https://github.com/google-deepmind/open_x_embodiment)<br>[论文](https://arxiv.org/abs/2310.08864) | 高；VLA 基础数据生态 | 多机器人真实数据集合 | 22 robots / 21 institutions | S0–S2 混合 | RGB/video、Action、State、Lang/task metadata；Depth/Joints/EEF/gripper 因子数据集不同 | 527 skills，160,266 tasks | Real / M4 | 跨 22 robots、21 institutions 的大规模集合 | 数据工程 / VLA 背景，不作移动主证据 |
| 47 | **DROID[22]** | 可用 / 需版本核验 | — | [主页](https://droid-dataset.github.io/)<br>[代码](https://github.com/droid-dataset/droid)<br>[论文](https://arxiv.org/abs/2403.12945) | 高；真实数据影响力大 | 真实 in-the-wild manipulation | 多真实 setup | S0–S2 | 多相机 RGB、Depth、calibration、Lang、Action、State；EEF/Joints/gripper 需看 schema | 76k demos，350h，564 scenes，86 tasks | Real / M4 | 官方主页说明 76k demonstrations / 350h / 564 scenes / 86 tasks | P1 表征 / 真实操作泛化参考 |
| 48 | **BridgeData V2[60]** | 数据可下 / 需 schema 核验 | — | [主页](https://rail-berkeley.github.io/bridgedata/)<br>[代码](https://github.com/rail-berkeley/bridge_data_v2)<br>[论文](https://arxiv.org/abs/2308.12952)<br>[LeRobot v3 数据](https://huggingface.co/datasets/nvidia/BridgeData2_LeRobot_v3) | 中高 | 真实操作数据 | WidowX / low-cost robot | S0–S2 | RGB/video、Action、State、goal image / Lang；Depth/EEF/Joints 需核验 | 60k+ trajectories，24 environments | Real / M3–M4 | goal-conditioned / language-conditioned 操作参考 | 真实操作辅助 |
| 49 | **AgiBot World[61] / Colosseo** | 数据可下 / 需 schema 核验 | — | [主页](https://opendrivelab.com/AgiBot-World/)<br>[代码/数据](https://github.com/OpenDriveLab/AgiBot-World)<br>[论文](https://arxiv.org/abs/2503.06669) | 高但需核验开放性 | 大规模真实操作平台 | AgiBot[61] 多平台 | S0–S3 | RGB/RGB-D、Action、State、Joints/EEF、可能 tactile/force；需按数据包核验 | 1M+ trajectories，217 tasks，5 scenarios | Real / M3–M4 | 前沿大规模真实数据 | scaling 参考 |
| 50 | **ALOHA / ACT[62]** | 需核验 | — | [ALOHA 主页](https://tonyzhaozh.github.io/aloha/)<br>[ACT 代码](https://github.com/tonyzhaozh/act)<br>[Mobile ALOHA[37] 主页](https://mobile-aloha.github.io/) | 高 | 真实双臂操作数据 / 方法 | ALOHA 双臂 | S0 | multi-camera RGB、dual-arm Joints/Action、gripper；Lang/Depth/Success 取决于任务 | 50 demos/task，ACT chunking | Real / M4 | action chunking 和 teleop 方法影响大 | 方法与动作建模参考 |
| 51 | **RH20T[63]** | 需核验 | — | [主页](https://rh20t.github.io/)<br>[论文](https://arxiv.org/abs/2307.00595) | 中 | contact-rich 真实数据 | 多机器人 | S0–S1 | visual、FT、Audio、Action、human demo video；Joints/EEF/Lang/Stage 需核验 | 110K+ contact-rich sequences | Real / M2–M3 | 多模态接触数据强 | failure/contact 标签参考 |
| 52 | **ARMBench[64]** | 需核验 | — | [主页](https://www.armbench.com/)<br>[论文](https://arxiv.org/abs/2303.16382) | 中 | 仓储真实操作数据 | Amazon warehouse | S0–S1 | images、videos、metadata、stage labels；low-level Action/Joints/EEF 需核验 | 235K+ pick-place activities | Real / M2 | 阶段化视觉数据 | readiness/failure 参考 |
| 53 | **LongBench** | 数据/代码待核实 | 需继续核实链接/数据开放 | [论文](https://arxiv.org/abs/2604.16788)<br>项目主页/代码/数据：待核实 | 待核 | 真实长时序操作 benchmark | 真实机器人 | S0–S2 | RGB、State、Action、success/failure 需核验 | 1000+ real-world episodes，待核 | Real / M1–M2 | 新近长时序真实评测 | failure analysis |
| 54 | **RoboCOIN** | 数据/代码待核实 | 需继续核实链接/数据开放 | [论文](https://arxiv.org/abs/2511.17441)<br>项目主页/代码/数据：待核实 | 待核 | 真实双臂多本体数据 | 多平台 | S0–S2 | RGB、Joints/EEF、Action、task labels；Depth/Seg/Lang/FT 需核验 | 180K+ demos，421 tasks，16 scenarios | Real / M1–M2 | 层级标注很有价值 | 层级标注参考 |
| 55 | **DuoBench** | 数据/代码待核实 | 需继续核实链接/数据开放 | [论文](https://arxiv.org/abs/2606.11901)<br>项目主页/代码/数据：待核实 | 待核 | 双臂 sim + real | 双臂平台 | S0–S1 | RGB、State、Action、Joints/EEF、stage/failure labels 需核验 | 11 bimanual tasks | Sim+Real / M1–M2 | 语义失败分析 | 双臂 failure taxonomy |
| 56 | **RoboEval** | 数据/代码待核实 | 需继续核实链接/数据开放 | [主页](https://robo-eval.github.io/)<br>论文/代码/数据：待核实 | 待核 | 双臂诊断框架 | 双臂操作 | S0–S1 | rollout logs、State、Action、success/failure/stage 需核验 | structured metrics | M1–M2 | 非二元成功率评价 | 指标参考 |

---

# G. 语言、导航、任务进度类 household 环境

| 顺序 | 名称 | 可用性 / 核对状态 | 人工核对备注 | 相关链接 | 当前社区信号 | 类型 | 仿真器 / 平台 | 空间 | 数据类型 / 模态 | 数据/任务 | 视觉/生态 | 数据情况与任务 | 本项目定位 |
|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 57 | **ALFRED[65]** | 可用 / 需版本核验 | — | [主页](https://askforalfred.com/)<br>[代码](https://github.com/askforalfred/alfred)<br>[论文](https://arxiv.org/abs/1912.01734) | 高；经典 | household instruction benchmark | AI2-THOR[68] | S3–S4 | RGB、Lang high-level/low-level instructions、discrete Action、interaction masks、object state；Depth/Seg 可 replay 生成 | 25k+ language directives / expert demos | V2 / M4 | 强语言长时序，但低层操作抽象 | progress/subgoal 参考 |
| 58 | **TEACh[66]** | 可用 / 需版本核验 | — | [代码](https://github.com/alexa/teach)<br>[论文](https://arxiv.org/abs/2110.00534) | 中高 | embodied dialogue benchmark | AI2-THOR[68] | S3–S4 | dialogue/Lang、RGB、Action trajectory、object interaction info；Depth/Seg 需核验 | 3000+ human-human dialogues | V2 / M3 | 对话式 household task | 纠错 / 语言恢复参考 |
| 59 | **ProcTHOR[67]** | 需核验 | — | [主页](https://procthor.allenai.org/)<br>[代码](https://github.com/allenai/procthor)<br>[论文](https://arxiv.org/abs/2206.06994) | 高 | 程序生成家庭环境 | AI2-THOR[68] | S3–S4 | 可生成 RGB、D、Seg、object metadata、layout、navigation trajectories | PROCTHOR-10K 等 generated houses | V2–V3 / M4 | 强在大规模导航/场景泛化 | 视觉/导航泛化参考 |
| 60 | **AI2-THOR[68] Rearrangement[68] / RoomR** | 可用 / 需版本核验 | — | [主页](https://ai2thor.allenai.org/rearrangement/)<br>[AI2-THOR[68] 代码](https://github.com/allenai/ai2thor) | 中高 | rearrangement benchmark | AI2-THOR[68] | S3–S4 | RGB、D、Seg、object pose/state、discrete interaction Action | room rearrangement | V2 / M3 | 状态恢复 / room-level changes | 状态变化参考 |
| 61 | **AI2-THOR[68]** | 需核验 | — | [主页](https://ai2thor.allenai.org/)<br>[代码](https://github.com/allenai/ai2thor)<br>[论文](https://arxiv.org/abs/1712.05474) | 高 | 仿真器 | Unity3D | S3–S4 | RGB、Depth、Seg、normals、object metadata、interaction actions | 120 rooms，2000+ objects | V2–V3 / M4 | 不是 joint-level robot manipulation | 任务进度/状态参考 |

---

# H. 评测框架 / leaderboard / world model evaluation

| 顺序 | 名称 | 可用性 / 核对状态 | 人工核对备注 | 相关链接 | 当前社区信号 | 类型 | 仿真器 / 平台 | 空间 | 数据类型 / 模态 | 数据/任务 | 视觉/生态 | 数据情况与任务 | 本项目定位 |
|---:|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 62 | **HomeRobot[7][20][21] Challenge / OVMM[7][20][21] Challenge** | 评测框架 / 在线评测 | 偏在线评测/leaderboard | [Challenge](https://aihabitat.org/challenge/2023_homerobot_ovmm/)<br>[HomeRobot[7][20][21] 主页](https://ovmm.github.io/)<br>[代码](https://github.com/facebookresearch/home-robot) | 中高 | mobile manipulation leaderboard | Habitat[42][43][44] / Stretch | S4 | RGB-D、Lang goal、continuous Action、target object/receptacle、metrics | online challenge | V3–V4 / M3 | 无静态 human traces，在线评测 | 主移动操作评测参考 |
| 63 | **VLA-Arena[69]** | 需核验 | — | [代码](https://github.com/PKU-Alignment/VLA-Arena)<br>[论文](https://arxiv.org/abs/2512.22539) | 中；新近 | VLA benchmark / framework | 多仿真/任务 | 多为 S0–S2 | Lang、visual obs、Action、perturbation metadata；RGB/D/State/EEF 需核验 | 170 tasks，11 suites | M2–M3 | safety、distractor、extrapolation、long horizon | robustness 指标参考 |
| 64 | **WorldEval[70]** | 需核验 | — | [论文](https://arxiv.org/abs/2505.19017)<br>[代码](https://github.com/liyaxuanliyaxuan/Worldeval) | 中；方法新 | world model policy evaluator | world model / real robot eval | 任务依赖 | policy videos、latent Action/Policy2Vec、generated video、ranking/safety labels | world model as policy evaluator | M2 | 与 WAM 评测思想高度相关 | P1/P2 评测方法核心参考 |
| 65 | **dWorldEval[71]** | 待核实 | 需继续核实链接/数据开放 | [论文](https://arxiv.org/abs/2604.22152)<br>项目主页/代码：待核实 | 待核 | discrete diffusion WM eval | LIBERO[48] / RoboTwin[49] / real tasks | 任务依赖 | vision、Lang、Action tokens、future obs、progress token | WM evaluation | M1–M2 | progress token 对 P0 很有启发 | P0/P1 指标参考 |
| 66 | **WPE[72]/ Evaluating Robot Policies in a World Model** | 来源已核验 / 需试跑 | — | [主页](https://world-model-eval.github.io/abstract)<br>[论文](https://arxiv.org/abs/2506.00613) | 来源已核验，方法需试跑 | action-conditioned video WM eval | world model | 任务依赖 | past video、Action conditioning、future video、VLM reward、policy ranking | video world model evaluator | M1–M2 | P2/WAM-eval 方法 | 长期评测参考 |
| 67 | **WorldGym[73]** | 与 WPE[72] 同源 / 需试跑 | — | [主页](https://world-model-eval.github.io/abstract)<br>[论文](https://arxiv.org/abs/2506.00613) | 与 WPE[72] 同源 | world model environment | world model | 任务依赖 | video obs、Action、future rollout、policy metrics 需核验 | WM environment | M1 | 正式入口需查 | 长期参考 |
| 68 | **OSCAR[18]** | 数据可下 / 需 schema 核验 | — | [主页](https://wuzy2115.github.io/oscar-project-page/)<br>[论文](https://arxiv.org/abs/2606.04463)<br>[HF 数据](https://huggingface.co/datasets/zywu2115/OSCAR_robot) | 待核 | action-conditioned video WAM | Cosmos-like / video WM | 任务依赖 | video、Action、skeleton/latent condition、future video 需核验 | WAM architecture / data | M1–M2 | 架构参考，不是主数据集 | 长期 WAM 参考 |
| 69 | **ManiSkill[40][41] Challenge** | 评测框架 / 在线评测 | 偏在线评测/leaderboard | [Challenge](https://sapien.ucsd.edu/challenges/maniskill/)<br>[ManiSkill[40][41] 文档](https://maniskill.readthedocs.io/)<br>[代码](https://github.com/haosulab/ManiSkill) | 高 | 操作 challenge | ManiSkill[40][41] / SAPIEN | S0–S2 | RGB-D/PCD/State/Action，任务依赖 | manipulation challenge | M4 | 操作技能 leaderboard | 操作辅助评测 |
| 70 | **BEHAVIOR[24] Challenge[24]** | 评测框架 / 在线评测 | 偏在线评测/leaderboard | [Challenge](https://behavior.stanford.edu/challenge/call_for_participation.html)<br>[主页](https://behavior.stanford.edu/)<br>[代码](https://github.com/StanfordVL/BEHAVIOR-1K) | 中高 | household challenge | OmniGibson[45] | S3–S5 | RGB-D、proprioception、object GT state、Action、fine-grained annotation | BEHAVIOR[24] subset | V4 / M3 | 高复杂 household challenge | 长期评测 |
| 71 | **SimplerEnv[74] / SIMPLER[74]** | 可用 / 需版本核验 | — | [主页](https://simpler-env.github.io/)<br>[代码](https://github.com/simpler-env/SimplerEnv)<br>[论文](https://arxiv.org/abs/2405.05941) | 高；方法影响力强 | real-policy sim eval | Google Robot / WidowX sim | S0–S2 | RGB、Action、State、rollout metrics | sim-to-real policy eval | M3–M4 | 代理真实策略评估方法论 | 方法参考 |
| 72 | **RoboBenchMart[75]** | 待核实 | 需继续核实链接/数据开放 | [论文](https://arxiv.org/abs/2511.10276)<br>[HTML](https://arxiv.org/html/2511.10276v2)<br>项目主页/代码：待核实 | 待核 | retail/grocery mobile manipulation | 待核 | S4–S5 | RGB-D、State、Action、object/receptacle metadata、Success 需核验 | retail manipulation | M1 | SHOPPER 类场景 | 长期参考 |
| 73 | **MolmoSpaces[76]** | 待核实 | — | [代码](https://github.com/allenai/molmospaces)<br>[HF paper](https://huggingface.co/papers/2602.11337) | 待核 | large-scale indoor policy benchmark | 待核 | S3–S4 | RGB/RGB-D、scene metadata、navigation/interaction tasks 需核验 | 230k+ indoor envs，待核 | M1–M2 | 新近候选 | 泛化评测参考 |

---

# I. 暂不作为正式 benchmark 的待核实名称

| 名称 | 可用性 / 核对状态 | 人工核对备注 | 相关链接 | 当前社区信号 | 类型 | 仿真器 / 平台 | 空间 | 数据类型 / 模态 | 数据/任务 | 视觉/生态 | 数据情况与任务 | 本项目定位 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **LIBERO-Extended** | 需核验 | — | [LIBERO[48] 主页](https://libero-project.github.io/main.html)<br>[LIBERO[48] 代码](https://github.com/Lifelong-Robot-Learning/LIBERO) | 未确认官方正式 benchmark | 待核 | 可能同 LIBERO[48] | S0 | 大概率 RGB、State、Action、Lang | 待核 | 待核 | 不能当正式 benchmark 写 | 暂缓 |
| **RoboCasa-Lite** | 需核验 | — | [RoboCasa[32] 主页](https://robocasa.ai/)<br>[RoboCasa[32] 代码](https://github.com/robocasa/robocasa) | 未确认官方正式 benchmark | 待核 | 可能同 RoboCasa[32] | S1–S3 | 大概率 RGB、State、Action、Lang | 待核 | 待核 | 不能当正式 benchmark 写 | 暂缓 |
| **RoboTwin-Diagnosis** | 需核验 | — | [RoboTwin[49] 主页](https://robotwin-platform.github.io/)<br>[RoboTwin[49] 代码](https://github.com/robotwin-Platform/robotwin) | 未确认官方正式 benchmark | 待核 | 可能同 RoboTwin[49] | S0–S1 | 大概率 RGB/RGB-D、State、Action、Stage/Failure | 待核 | 待核 | 可能只是诊断方向泛称 | 暂缓 |

---

# J. 建议的下一步核验队列

## J.1 先核验前 10 个主候选

1. RoboCasa365[19]
2. RoboCasa[32]
3. MobileManiBench[33]
4. Kitchen-R[34]
5. EBench[35]
6. BEHAVIOR-1K[24]
7. HomeRobot / OVMM[7][20][21]
8. Habitat 2.0 / HAB[42]
9. ManiSkill3[40]
10. AIRoA MoMa[36]

## J.2 每个候选统一核验字段

```text
1. 可用性 / 核对状态：可用 / 未开源 / coming soon / 数据可下 / 仅论文 / 代码可用但数据待核 / 暂缓
2. 人工核对备注：人工核对时发现的关键边界或风险点
3. 当前社区信号：stars / forks / issues / release / last commit / docs / HF likes-downloads
4. 类型：真实 / 仿真 / 混合 / 评测框架
5. 仿真器 / 平台：Isaac / Habitat / OmniGibson / MuJoCo / SAPIEN / PyBullet / Real
6. 空间：S0-S5，是否跨房间
7. 数据类型 / 模态：RGB、D、Seg、PCD、State、Joints、EEF、Base、Action、Lang、FT、Stage、Success
8. 数据/任务：规模、任务数、场景数、demo/trajectory 数、时长
9. 视觉/生态：V0-V4，M0-M4
10. 数据情况与任务：任务链覆盖、标签质量、是否有 split、是否可 replay/eval
11. 本项目定位：主 benchmark / 训练数据 / 诊断集 / 辅助验证 / 长期参考
```

---

# K. v0.8 相对 v0.7 的结构化更新

1. 新增 **可用性 / 核对状态** 列，将“可用、未开源、coming soon、数据可下、仿真器而非数据集、非移动操作主证据”等信息从名称括号中迁移为结构化字段。
2. 新增 **人工核对备注** 列，保留人工核对得到的关键边界，例如“短程任务”“抓取吸附”“导航非移动操作”“渲染逼真”等。
3. 对 MobileManiBench[33]、Kitchen-R[34]、EBench[35]、HomeRobot/OVMM[7][20][21]、AIRoA MoMa[36]、SAGE-3D[11] 等条目做了状态化标记，避免仅按论文描述高估近期可落地性。
4. 后续进入正式调研报告时，应将主表放入附录，并在正文只保留经过筛选后的 8–12 个核心候选。

# L. 当前阶段结论

1. **MoMa-Kitchen[39] 保留，但定位下调**：BestMan/PyBullet，视觉真实感较弱，适合作 P0 final-pose affordance / manipulation-readiness 诊断集，不作主 benchmark。  
2. **RoboCasa[32] / RoboCasa365[19] 上调**：工程生态成熟、厨房任务资产强，适合单房间 household 操作闭环，但需要核实 mobile base 与 navigation-manipulation 真实性。  
3. **MobileManiBench[33] / Kitchen-R[34] / EBench[35] 上调**：Isaac Sim 系视觉和物理基础更好，如果代码/数据开放，应优先验证。  
4. **AIRoA MoMa[36] 必须保留**：真实移动操作、多模态、层级标注，是 P0/P1 真实数据最关键候选之一。  
5. **HomeRobot[7][20][21] / LAMBDA[38] / BEHAVIOR-1K[24] 是高难验证集**：空间难度偏高，适合作后期外推和长期验证，不适合作第一步闭环。  
6. **LIBERO[48] / RoboTwin[49] / DROID[22] / Open X[23] 等不能作移动主证据**：但非常适合快速消融、操作泛化、数据工程和表征学习参考。

# 参考资料

> **参考资料核验说明**：本节保留 70+ 条线索，但使用口径分层。标注为“核心 / 已核验”的条目可直接支撑正文判断；标注为“待核验 / 扩展资料”的条目仅作为后续检索线索，不作为详细设计的硬依赖。对于 2026 年新论文、coming soon benchmark、项目主页或二手列表，详细设计冻结前必须重新打开 arXiv / 官方仓库 / Hugging Face / 许可证页面核对。
## A. WAM / Latent World Model / Future Supervision
[84] 【核心/已核验】World Action Models: A Survey. PDF 副标题：Dream Less, Act More. arXiv:2606.20781v1. https://arxiv.org/abs/2606.20781 ; Survey homepage: https://world-action-models.github.io/  
[1] 【核心/已核验】Fast-WAM: Do World Action Models Need Test-time Future Imagination? arXiv:2603.16666. https://arxiv.org/abs/2603.16666  
[2] 【核心/已核验】LaWAM: Latent World Action Models for Efficient Dynamics-Aware Robot Policies. arXiv:2606.15768. https://arxiv.org/abs/2606.15768  
[3] 【核心/已核验】VLA-JEPA: Enhancing Vision-Language-Action Model with Latent World Model. arXiv:2602.10098. https://arxiv.org/abs/2602.10098 ; GitHub: https://github.com/ginwind/VLA-JEPA/  
[15] 【核心/已核验】Light-WAM: Efficient World Action Models with State-Fusion Action Decoding. arXiv:2606.08242. https://arxiv.org/abs/2606.08242  
[26] 【核心/已核验】Efficient-WAM: A 1B-Parameter World-Action Model with Low-Cost Future Imagination. arXiv:2606.10040. https://arxiv.org/abs/2606.10040  
[27] 【核心/已核验】MotionWAM: Towards Foundation World Action Models for Real-Time Humanoid Loco-Manipulation. arXiv:2606.09215. https://arxiv.org/abs/2606.09215  
[85] 【核心/已核验】Metis: A Generalizable and Efficient World-Action Model for Autonomous Driving and Urban Navigation. arXiv:2606.15869. https://arxiv.org/abs/2606.15869  
[86] 【核心/已核验】AHA-WAM: Asynchronous Horizon-Adaptive World-Action Modeling with Observation-Guided Context Routing. arXiv:2606.09811. https://arxiv.org/abs/2606.09811  
[87] 【核心/已核验】World Action Models are Zero-shot Policies（DreamZero）. arXiv:2602.15922. https://arxiv.org/abs/2602.15922  
[8] Unified World Models: Coupling Video and Action Diffusion for Pretraining on Large Robotic Datasets. arXiv:2504.02792. https://arxiv.org/abs/2504.02792  
[9] 【核心/已核验】AdaWAM: Dreaming when Necessary: Advancing World Action Models with Adaptive Multi-Modal Reasoning. arXiv:2606.07089. https://arxiv.org/abs/2606.07089  
[10] Awesome-WAM / WAM reading list. https://openmoss.ai/Awesome-WAM/  
[17] 【核心/已核验】ChronoDreamer: Action-Conditioned World Model as an Online Simulator for Robotic Planning. arXiv:2512.18619. https://arxiv.org/abs/2512.18619  
[18] 【核心/已核验】OSCAR: Omni-Embodiment Skeleton-Conditioned World Action Model for Robotics. arXiv:2606.04463. https://arxiv.org/abs/2606.04463 ; https://wuzy2115.github.io/oscar-project-page/  

## B. Robot VLA / Backbone / Action Policy 相关
[4] Qwen-RobotManip Technical Report: Alignment Unlocks Scale for Robotic Manipulation Foundation Models. arXiv:2606.17846. https://arxiv.org/abs/2606.17846  
[29] 【核心/已核验】Qwen3-VL Technical Report. arXiv:2511.21631. https://arxiv.org/abs/2511.21631  
[30] Qwen2.5-VL Technical Report. arXiv:2502.13923. https://arxiv.org/abs/2502.13923  
[31] 【核心/已核验】StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing. arXiv:2604.05014. https://arxiv.org/abs/2604.05014 ; https://github.com/starVLA/starVLA  
[77] OpenVLA: An Open-Source Vision-Language-Action Model. arXiv:2406.09246. https://arxiv.org/abs/2406.09246  
[78] RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control. arXiv:2307.15818. https://arxiv.org/abs/2307.15818  
[79] Diffusion Policy: Visuomotor Policy Learning via Action Diffusion. arXiv:2303.04137. https://arxiv.org/abs/2303.04137 ; https://diffusion-policy.cs.columbia.edu/  
[62] ALOHA / ACT: Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware. https://tonyzhaozh.github.io/aloha/ ; https://github.com/tonyzhaozh/act  

## C. Wan2.2 / Video Foundation Model / LoRA
[5] 【核心/已核验】Wan: Open and Advanced Large-Scale Video Generative Models. arXiv:2503.20314. https://arxiv.org/abs/2503.20314  
[14] 【核心/已核验】Wan2.2 官方仓库与 TI2V-5B 模型说明. 主链接：https://github.com/Wan-Video/Wan2.2 ; HF 模型页：https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B（部分环境可能需要登录或访问权限）  
[16] 【核心/已核验】Generative World Modelling for Humanoids: 1X World Model Challenge Technical Report. arXiv:2510.07092. https://arxiv.org/abs/2510.07092  
[25] 【核心/已核验】LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685. https://arxiv.org/abs/2106.09685  
[6] 【核心/已核验】Cosmos 3: Omnimodal World Models for Physical AI. arXiv:2606.02800. https://arxiv.org/abs/2606.02800  
[80] 【待核验/扩展资料】DreamerV3. https://github.com/danijar/dreamerv3  
[81] 【待核验/扩展资料】Visual Foresight / Video Prediction for Robotic Control. https://visual-foresight.cs.berkeley.edu/  
[82] 【待核验/扩展资料】UniPi: Learning Universal Policies via Text-Guided Video Generation. https://unipi.cs.berkeley.edu/  
[83] 【待核验/扩展资料】Genie: Generative Interactive Environments. https://deepmind.google/discover/blog/genie/  

## D. 数据混合与 Scaling 方法
[12] Re-Mix: Optimizing Data Mixtures for Large Scale Imitation Learning. arXiv:2408.14037. https://arxiv.org/abs/2408.14037  
[13] DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining. arXiv:2305.10429. https://arxiv.org/abs/2305.10429  

## E. Mobile Manipulation / Household Benchmark 与仿真平台
[7] HomeRobot: Open-Vocabulary Mobile Manipulation. arXiv:2306.11565. https://arxiv.org/abs/2306.11565 ; https://github.com/facebookresearch/home-robot ; https://ovmm.github.io/  
[19] 【核心/已核验】RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots. arXiv:2603.04356. https://arxiv.org/abs/2603.04356  
[20] UniTeam: Open Vocabulary Mobile Manipulation Challenge. arXiv:2312.08611. https://arxiv.org/abs/2312.08611  
[21] Towards Open-World Mobile Manipulation in Homes: Lessons from the NeurIPS 2023 HomeRobot Open Vocabulary Mobile Manipulation Challenge. arXiv:2407.06939. https://arxiv.org/abs/2407.06939  
[24] BEHAVIOR-1K: A Human-Centered, Embodied AI Benchmark with 1,000 Everyday Activities and Realistic Simulation. arXiv:2403.09227. https://arxiv.org/abs/2403.09227  
[28] 【待核验/扩展资料】What Are We Actually Benchmarking in Robot Manipulation? arXiv:2606.04233. https://arxiv.org/abs/2606.04233  
[32] 【核心/已核验】RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots. arXiv:2406.02523. https://arxiv.org/abs/2406.02523 ; https://robocasa.ai/  
[33] 【核心/已核验；代码/数据 coming soon】MobileManiBench: Benchmarking Mobile Manipulation in Isaac Sim. arXiv:2602.05233. https://arxiv.org/abs/2602.05233 ; https://dexhand.github.io/MobileManiBench_Website/  
[34] 【核心/已核验；未找到代码/数据入口】Kitchen-R: Kitchen Robotic Benchmark for Complex Language Instructions and Low-Level Control. arXiv:2508.15663. https://arxiv.org/abs/2508.15663  
[35] 【核心/已核验】EBench: Embodied / VLA Benchmark with Mobile Manipulation Diagnostics. arXiv:2606.18239. https://arxiv.org/abs/2606.18239 ; https://internrobotics.github.io/EBench-doc/  
[36] 【核心/已核验】AIRoA MoMa dataset. HF 数据页：https://huggingface.co/datasets/airoa-org/airoa-moma（部分环境可能需要登录或访问权限）；arXiv:2509.25032. https://arxiv.org/abs/2509.25032  
[37] 【核心/已核验】Mobile ALOHA: Learning Bimanual Mobile Manipulation with Low-Cost Whole-Body Teleoperation. arXiv:2401.02117. https://arxiv.org/abs/2401.02117 ; https://mobile-aloha.github.io/  
[38] 【核心/已核验】LAMBDA / λ Benchmark for long-horizon mobile manipulation. arXiv:2412.05313. https://arxiv.org/abs/2412.05313 ; https://lambdabenchmark.github.io/  
[39] 【核心/已核验】MoMa-Kitchen: A 100K Benchmark for Affordance-Grounded Last-Mile Navigation in Mobile Manipulation. arXiv:2503.11081. https://arxiv.org/abs/2503.11081 ; https://momakitchen.github.io/  
[40] ManiSkill3: GPU-Parallelized Robotics Simulation and Rendering for Generalizable Embodied AI. arXiv:2410.00425. https://arxiv.org/abs/2410.00425 ; https://maniskill.readthedocs.io/  
[41] ManiSkill2 / ManiSkill: A Unified Benchmark for Generalizable Manipulation Skills. arXiv:2302.04659. https://arxiv.org/abs/2302.04659 ; https://github.com/haosulab/ManiSkill  
[42] Habitat 2.0: Training Home Assistants to Rearrange their Habitat. arXiv:2106.14405. https://arxiv.org/abs/2106.14405 ; https://github.com/facebookresearch/habitat-lab  
[43] Habitat Rearrangement Challenge. https://aihabitat.org/challenge/2022_rearrange/ ; Habitat-Lab: https://github.com/facebookresearch/habitat-lab  
[44] Habitat 3.0. https://aihabitat.org/habitat3/ ; https://github.com/facebookresearch/habitat-lab  
[45] OmniGibson simulator. https://github.com/StanfordVL/OmniGibson ; BEHAVIOR project: https://behavior.stanford.edu/  
[46] iGibson 2.0: Object-Centric Simulation for Robot Learning of Everyday Household Tasks. https://github.com/StanfordVL/iGibson  
[47] 【核心/已核验】ManiSkill-HAB: A Benchmark for Low-Level Manipulation in Home Rearrangement Tasks. arXiv:2412.13211. https://arxiv.org/abs/2412.13211  

## F. 操作数据集 / 真实数据 / 辅助 Benchmark
[22] DROID: A Large-Scale In-The-Wild Robot Manipulation Dataset. arXiv:2403.12945. https://arxiv.org/abs/2403.12945 ; https://droid-dataset.github.io/  
[23] Open X-Embodiment: Robotic Learning Datasets and RT-X Models. arXiv:2310.08864. https://arxiv.org/abs/2310.08864 ; https://robotics-transformer-x.github.io/  
[48] LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning. https://github.com/Lifelong-Robot-Learning/LIBERO  
[49] RoboTwin 2.0 benchmark and dataset. https://robotwin-benchmark.github.io/ ; https://github.com/RoboTwin-Platform/RoboTwin  
[50] RLBench: The Robot Learning Benchmark and Learning Environment. https://github.com/stepjam/RLBench  
[51] CALVIN: A Benchmark for Language-Conditioned Policy Learning for Long-Horizon Robot Manipulation Tasks. https://github.com/mees/calvin  
[52] FurnitureBench: Reproducible Real-World Benchmark for Long-Horizon Complex Manipulation. https://clvrai.github.io/furniture-bench/  
[53] MimicGen: A Data Generation System for Scalable Robot Learning using Human Demonstrations. https://mimicgen.github.io/  
[54] SkillMimicGen: Automated Demonstration Generation for Efficient Skill Learning. https://skillmimicgen.github.io/  
[55] 【待核验/扩展资料】SoftMimicGen: data generation for soft-body / deformable manipulation. https://softmimicgen.github.io/  
[56] Meta-World: A Benchmark and Evaluation for Multi-Task and Meta Reinforcement Learning. https://meta-world.github.io/  
[57] BulletArm: An Open-Source Robotic Manipulation Benchmark and Learning Framework. https://github.com/ColinKohler/BulletArm  
[58] 【待核验/扩展资料】DexJoCo: A Benchmark for Dexterous Robotic Manipulation. https://github.com/kevinzakka/dexjoco  
[59] 【待核验/扩展资料】VLABench: A Large-Scale Benchmark for Language-Conditioned Robotics. https://github.com/OpenMOSS/VLABench  
[60] BridgeData V2: A Dataset for Robot Learning at Scale. arXiv:2308.12952. https://arxiv.org/abs/2308.12952 ; https://rail-berkeley.github.io/bridgedata/  
[61] AgiBot World / Colosseo. arXiv:2503.06669. https://arxiv.org/abs/2503.06669 ; https://github.com/OpenDriveLab/AgiBot-World  
[63] RH20T: A Comprehensive Robotic Dataset for Learning Diverse Skills in One-Shot. arXiv:2307.00595. https://arxiv.org/abs/2307.00595 ; https://rh20t.github.io/  
[64] ARMBench: A Large-Scale Robot Manipulation Benchmark. arXiv:2303.16382. https://arxiv.org/abs/2303.16382 ; https://www.armbench.com/  
[65] ALFRED: A Benchmark for Interpreting Grounded Instructions for Everyday Tasks. arXiv:1912.01734. https://arxiv.org/abs/1912.01734 ; https://askforalfred.com/  
[66] TEACh: Task-driven Embodied Agents that Chat. arXiv:2110.00534. https://arxiv.org/abs/2110.00534  
[67] ProcTHOR: Large-Scale Embodied AI Using Procedural Generation. arXiv:2206.06994. https://arxiv.org/abs/2206.06994 ; https://procthor.allenai.org/  
[68] AI2-THOR: An Interactive 3D Environment for Visual AI. arXiv:1712.05474. https://arxiv.org/abs/1712.05474 ; https://ai2thor.allenai.org/  

## G. World Model Evaluation / Leaderboard / 方法论参考
[69] 【待核验/扩展资料】VLA-Arena: VLA robustness / safety benchmark. arXiv:2512.22539. https://arxiv.org/abs/2512.22539 ; https://github.com/PKU-Alignment/VLA-Arena  
[70] WorldEval: World Model as a Policy Evaluator. arXiv:2505.19017. https://arxiv.org/abs/2505.19017 ; https://github.com/liyaxuanliyaxuan/Worldeval  
[71] 【待核验/扩展资料】dWorldEval: Discrete Diffusion World Model Evaluation. arXiv:2604.22152. https://arxiv.org/abs/2604.22152  
[72] 【核心/已核验】WorldGym / Evaluating Robot Policies in a World Model（WPE）. arXiv:2506.00613. https://arxiv.org/abs/2506.00613 ; https://world-model-eval.github.io/abstract  
[73] 【同源别称/索引保留】WorldGym: World Model as An Environment for Policy Evaluation；与 [72] 指向同一 arXiv:2506.00613，本编号仅为兼容正文既有引用，不作为独立论文。 https://arxiv.org/abs/2506.00613  
[74] SimplerEnv / SIMPLER: Evaluating Real-World Robot Policies in Simulation. arXiv:2405.05941. https://arxiv.org/abs/2405.05941 ; https://simpler-env.github.io/  
[75] 【待核验/扩展资料】RoboBenchMart: retail / grocery mobile manipulation benchmark candidate. arXiv:2511.10276. https://arxiv.org/abs/2511.10276  
[76] 【待核验/扩展资料】MolmoSpaces indoor policy benchmark candidate. https://github.com/allenai/molmospaces ; https://huggingface.co/papers/2602.11337  

## H. 用户内部核对材料
[11] 本项目 WAM 数据集 / Benchmark 全量候选表 v0.8，用户核对版，2026。  



## I. History Latent Compression / Condition Injection 参考
[88] 【核心/已核验】Flamingo: a Visual Language Model for Few-Shot Learning. arXiv:2204.14198. https://arxiv.org/abs/2204.14198  
[89] 【核心/已核验】T2I-Adapter: Learning Adapters to Dig out More Controllable Ability for Text-to-Image Diffusion Models. arXiv:2302.08453. https://arxiv.org/abs/2302.08453  
[90] 【核心/已核验】Adding Conditional Control to Text-to-Image Diffusion Models（ControlNet）. arXiv:2302.05543. https://arxiv.org/abs/2302.05543  
[91] 【核心/已核验】Mem-World: Memory-Augmented Action-Conditioned World Models for Persistent Robot Manipulation. arXiv:2606.18960. https://arxiv.org/abs/2606.18960  

[92] 【工程参考/已核对】StarVLA README：StarVLA: A Lego-like Codebase for Vision-Language-Action Model Developing；模块化 backbone / action head / trainer 设计。https://github.com/wxwy/starVLA/tree/merge-official-starvla-dev  
[93] 【工程参考/已核对】StarVLA WM4A 文档：World Model for Action；说明视频生成 DiT 作为 action prediction backbone，OFT / GR00T / PI 三类 action head，以及 WanOFT / WanGR00T / WanPI 组合。https://github.com/wxwy/starVLA/blob/merge-official-starvla-dev/docs/WM4A.md  
[94] 【工程参考/已核对】StarVLA LayerwiseFM_ActionHeader.py：LayerwiseFlowmatchingActionHead；当前 action DiT 使用 layerwise cross-attention 到 `vl_embs_list[layer_idx]`，并将 state_features、future_tokens、action_features 拼接为 action-side sequence。https://github.com/wxwy/starVLA/blob/merge-official-starvla-dev/starVLA/model/modules/action_model/LayerwiseFM_ActionHeader.py  
[95] 【核心/已核验】TokenLearner: What Can 8 Learned Tokens Do for Images and Videos? arXiv:2106.11297. https://arxiv.org/abs/2106.11297  
[96] 【核心/已核验】Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context. arXiv:1901.02860. https://arxiv.org/abs/1901.02860  
[97] 【核心/已核验】Compressive Transformers for Long-Range Sequence Modelling. arXiv:1911.05507. https://arxiv.org/abs/1911.05507  
[98] 【核心/已核验】Recurrent Memory Transformer. arXiv:2207.06881. https://arxiv.org/abs/2207.06881  
[99] 【核心/已核验】Mamba: Linear-Time Sequence Modeling with Selective State Spaces. arXiv:2312.00752. https://arxiv.org/abs/2312.00752  
[100] 【核心/已核验】Action-Effect Memory Pretraining for Robot Manipulation. arXiv:2606.12499. https://arxiv.org/abs/2606.12499  
[101] 【核心/已核验】HiMem-WAM: Hierarchical Memory-Gated World Action Models for Robotic Manipulation. arXiv:2606.10363. https://arxiv.org/abs/2606.10363  

