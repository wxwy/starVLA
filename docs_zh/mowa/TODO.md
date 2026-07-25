# MoWA TODO

## 2026-07-23 E003-B1 v2 优化与逐项验收清单

状态定义：`[x]` 已实现且有直接证据；`[~]` 已实现但尚缺目标 checkpoint 的闭环证据；`[ ]` 尚未实施。后续每完成一项，必须补充验证命令、结果路径和通过/不通过结论，不能只按代码存在判定完成。

### P0：先确认现有训练—推理链路

- [x] **原始 Wan2.2 fail-fast 加载**：原始 transformer 加载要求 `missing=0 / unexpected=0 / mismatched=0 / errors=0`；`steps_6500`、`steps_18000` 实际服务日志均已通过。轻量 checkpoint 只能在原始 backbone 严格加载成功后允许省略冻结的非 LoRA backbone 权重。
- [x] **LoRA 与 checkpoint 恢复**：LoRA 已进入独立 optimizer group，lightweight checkpoint 同时保存 `wan_lora.safetensors` / config；恢复时不得静默忽略 LoRA key。后续每个正式 checkpoint 仍需记录 LoRA tensor 数、missing/unexpected key 和相邻 checkpoint delta。
- [x] **训练 cache 与在线 VAE 编码逐元素对齐**：在线改为按 episode/view 独立持久化 Wan causal `feature_cache`；首帧初始化，此后每 4 个真实新帧生成 1 个 regular latent；main/wrist 独立 stream id。真实 LIBERO ep6 在 endpoint 4,8,...,40 的 main/wrist 均为 MSE=0、max_abs=0，报告：`mowa_wan_vae_streaming_alignment_{main,wrist}.json`。
- [~] **RoboCasa 在线帧推进**：首帧初始化，首次执行 8 个 action 收集 8 个真实帧，之后每轮新增 8 帧并生成 2 个 latent；已修复遗留的“必须等于旧 `wan_history_frames=5`”校验，24/24 回归通过。仍需用正式多 episode 评测确认 episode reset 后 stream/cache 不串样本。
- [~] **LIBERO 在线帧推进**：warmup 从 10 对齐为 12 个真实环境步，首轮覆盖 frame 0..12；`steps_6500` 已连续完成 1 episode、无 cache/shape 异常，但该次 SR=0/1，尚不能证明策略能力。
- [ ] **训练/推理视觉语义全量审计**：逐项固定 main/wrist 顺序、同步帧、180°旋转、crop/resize、uint8/range、VAE mode、时间采样；shape 相同不能代替视角语义检查。需要保存一组训练源帧、在线帧和 latent 可视/数值对照。
- [~] **动作/state 契约**：LIBERO 使用 action 7维、state raw8→sin/cos16；RoboCasa 使用 action 12维、state raw16→sin/cos32；反归一化 key 和 key 顺序已有 fail-fast 检查。仍需对每个目标 checkpoint 做 training forward / `predict_action` / server 输出三路同输入一致性检查。
- [~] **夹爪二值化与方向**：LIBERO client 已二值化 gripper；RoboCasa action key/反归一化已对齐。需要在离线 expert probes 和闭环视频中分别记录 gripper accuracy、开合方向、饱和/clip 比例。
- [ ] **单 batch 过拟合**：固定 32–128 样本，关闭 future/cross-view，只训练动作路径；通过标准为 normalized action L1 接近0、gripper准确率接近100%，且 training forward、`predict_action`、server 输出一致。

### P1：恢复“已有成功策略 + 功能保持增量”

- [x] **原生 WanOFT 正对照**：官方 `steps_60000` 在修复 legacy VAE/state 后为 14/15；作为 LIBERO 评测金标准。新增残差框架零训练闭环进一步取得 9/9，报告在 `playground/eval_results/libero_goal/E003_WanOFT_multiview_residual_zero_init_3tasks3eps/eval_report.json`。
- [x] **完整继承 WanOFT 策略能力**：新路径严格加载 `backbone + action_query_proj + action_model`，实际 key 数为 1264+2+16；旧 checkpoint 只允许缺失新增 `multi_view_fusion.{weight,bias}`，其他 missing/unexpected 继续 fail-fast。
- [x] **简化双视角 WanOFT**：为严格保持官方策略，主路径保留原生 `[main,wrist]` 两帧联合编码和 mean pooling；新增 wrist 独立编码残差，二者拼接后经 `fusion_proj 6144→3072` 接原生 `action_query_proj + MLP action head`。共享同一个 Wan backbone，不复制参数。
- [x] **功能保持初始化**：`fusion.weight[:, :3072]=I`、`fusion.weight[:,3072:]=0`、bias=0。真实 LIBERO 专家观测端到端对比 `max_abs_diff=0`，报告为 `wanoft_multiview_equivalence.json`；零训练 `replan_interval=1` 闭环 9/9。
- [x] **第一轮降复杂度**：新 WanOFT 路径不包含 future latent、future token fusion、LayerwiseFM、done、cross-view attention、memory 或 state adapter；Stage1 配置只保留原生策略与新增 fusion。
- [x] **Stage1 单步训练 smoke**：使用官方 `steps_60000` 完整加载 `backbone + action_query_proj + action_model`，冻结后三者，仅 `multi_view_fusion` 18,877,440 参数可训练；持久化 UMT5 表 40/40 加载成功。bs=2、GA=2 的第一个 optimizer step 完成，`action_dit_loss=0.01682`，无 checkpoint 写出；首微步 4.97s，第二微步 0.66s。
- [x] **Stage1 3k 与首轮闭环**：fusion-only 训练已完成 3000 step，`final_model` 已落盘。LIBERO goal 前三任务 ×3、`replan_interval=1` 为 7/9（77.8%）：drawer 1/3、bowl-on-stove 3/3、wine-bottle 3/3；结果目录 `playground/eval_results/libero_goal/E003_WanOFT_stage1_3k_3tasks3eps`。该小样本低于零初始化正对照 9/9，不能宣称 Stage1 提升，下一步应先扩同协议评测再决定是否进入 LoRA Stage2。
- [x] **UMT5 指令缓存接入**：复用 `playground/Datasets/libero_wan2.2_latent/instruction_text_latents.pt` 的 40 条、4096 维 UMT5 fp16 embedding；WanOFT 训练/推理按 `lang` 查表并传入 Wan2，cache miss 直接报错，避免在 24GB 卡常驻约 12.54 GiB UMT5。已通过 shape 回归和真实 server probe：轻量 `steps_1000` 在 4090 占 11.67 GiB，返回 `[8,7]` 动作块；视觉 VAE 仍是在线路径，不能误标为 visual cache。
- [x] **无文件缓存回退**：设置 `world_model.use_text_cache=true`、`preload_text_cache=true` 且不提供有效持久化表时，训练入口扫描数据根目录的去重指令，临时加载 UMT5 生成内存表；完成后显式释放 UMT5、tokenizer 与 CUDA allocator cache。Wan2 后续只接受内存表或 batch embedding，miss fail-fast；该路径不为在线未知指令提供隐式编码。真实 `MoWA-E-003-WanOFT-preload-text-smoke` full-path dry-run 完成 40/40 编码、首 batch `[2, 8, 7]`，日志确认 `UMT5/tokenizer released`，无训练启动。
- [ ] **分阶段解冻**：Stage1 只训练 fusion（1k–3k）；Stage2 fusion+Wan LoRA；Stage3 小学习率开放 action query/head。建议起始 LR：fusion `1e-4`，query/head `1e-5~3e-5`，LoRA `5e-6~1e-5`。每阶段先做离线指标与短闭环门禁。
- [ ] **LIBERO 控制口径**：恢复 SR 阶段固定 action_dim=7、action_horizon=8、replan_interval=1；稳定后再测试 replan 4/8，避免 32 步开环误差与 future horizon 混淆。

### P2：在稳定基线上逐步增加能力

- [ ] **状态注入**：E001/E003 使用兼容相同 action head 的状态注入；新增 state adapter 必须零初始化/残差门控，先验证关闭时严格等价，再验证开启后的离线和闭环收益。状态位置量 min-max 归一化属于需重训项。
- [ ] **多视角融合升级**：顺序固定为 mean-pool concat → gated view residual → cross-view attention → camera-pose-aware fusion；每一步单独验收，不把多个模块同时打开。
- [ ] **pooling 升级**：mean pooling 是功能保持基线；稳定后再测试 residual attention pooling、state-conditioned pooling 或 action-query cross-attention，新增分支 gate 初始化为0。
- [ ] **future action path 重构**：必须保留 current feature 直连，使用 `current + gate × future_adapter(future)`，gate 初始化为0/很小，禁止 future-hidden-only 接管动作。
- [ ] **降低 future 训练—推理 gap**：依次比较 current-only、current+GT-future、current+predicted-future；根据结果选择 scheduled sampling、GT/pred action consistency，或先把 future 仅作为 auxiliary loss。
- [ ] **future latent/done 监督验收**：确认 `enable_future_latent_prior_loss`、trainer scale、done loss 开关一致；W&B 必须持续记录 main/wrist future、done、action、aux/action ratio。未证明闭环收益前不得用 loss 下降代替成功率结论。
- [ ] **记忆模块**：基础 SR 稳定后，按短时视觉、动作历史、state历史、failure-aware memory 顺序加入；统一 gated residual、零初始化，并验证 episode reset。
- [ ] **空间理解**：后续再加入 camera intrinsics/extrinsics、base-camera-wrist 坐标、target-centric spatial token 或轻量3D adapter；必须直接验证对 action path 的增益。

### 训练工程与验收口径

- [x] **LoRA 量级**：当前 r32 self+cross attention 为 47,185,920 参数；已修复冻结 backbone 时 LoRA 被错误排除 optimizer 的问题。后续调整 rank 前先报告 LoRA/依附 backbone 和 LoRA/总 trainable 的比例，不按固定百分比拍脑袋。
- [~] **显存与有效 batch**：目标有效 batch=32；现有 8卡口径 per-device BS=2、GA=2。action head gradient checkpointing、吞吐和峰值显存已有 smoke 记录，但正式机器仍需记录首10个 optimizer step 的均值/峰值。
- [~] **W&B 监控**：既有 action/future/done/cross-view/LoRA 梯度与参数变化指标保留；每次正式启动后必须在 W&B 页面确认实际出现，而不是只检查 YAML。
- [ ] **统一 Go/No-Go**：每个新阶段至少通过严格加载、数据流2/2、单batch过拟合、离线 action 指标、三路推理一致性和闭环评测；恢复基线阶段 20k 后 `pos_corr<0.5` 或闭环持续0，应停止加步数并回到链路/架构诊断。
- [x] **当前旧 E003 checkpoint 现场结果**：RoboCasa multitask `steps_18000` 的 `OpenDrawer/TurnOnSinkFaucet/PickPlaceCounterToCabinet` 均为 0/3，单任务 TurnOnElectricKettle `steps_4000` 为 0/3；LIBERO origWan `steps_6500` 中断前为 0/4，OFT `steps_15500` 中断前为 0/5。流式编码已对齐但未恢复 SR，因此不再给旧复杂架构追加训练步数。

- 默认 cache 设计已从 window-level artifact（方案 A）重构为 episode-level latent store + window manifest references（方案 C）；新增 `starVLA/dataloader/mowa/episode_latent_store.py`、`window_manifest.py`、`window_latent_sample.py` 和对应 CLI / 单测；遗留 `tools/mowa/build_future_latent_cache.py` 作为方案 A 短期兼容。
- E-001 至 E-007 的活跃训练实验已在 `docs_zh/mowa/mowa_experiment_launch_readiness_matrix.json` 收口为 `launchable_now`；E-009 是 conditional，未触发时不阻断；E-010 eval-only tracking plan/smoke 已补齐。
- E-001 的 MoWA formal long-training candidate 已在 A100 上启动，当前 checkpoint 写入 `playground/mowa_ckpt`，先观察首个 checkpoint 与 step 统计，再决定是否补 baseline 对照或继续扩量。
- E-003 / E-004 已迁移到 WanPI 路径，`train_starvla.py` 数据路径可消费 episode-level latent/history batch；当前仍需在全量 `vae_spatial` cache 完成后补 E-003/E-004 WanPI full-path dry-run evidence。
- E-007 proxy-alpha candidate/readiness smoke 已补齐；实际 proxy optimization 与 frozen alpha 报告仍需单独执行后再接入下游 P0/P1 配置。
- 真实 Wan2.2 已接到 `playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers`；`tools/mowa/e003_wan2_2_latent_cache_smoke.py --execute --episode-index 0 --episode-index 1 --episode-index 4` 已完成 3 episode 真实写盘和 `validate_mowa_latent_cache()` 复验，且 execute 模式现在会自动覆盖专用 cache_root，重复运行仍保持可复跑。E-003 还新增了 launch smoke，和 fake-cache dry-run 一起把 launch readiness 推到 `launchable_now`。
- E-004 也已补上 synthetic launch smoke、checkpoint-backed preflight 与 policy rollout smoke；E-005 现已补上 shuffled-robot rollout smoke，checkpoint preflight 也可通过 `latest_complete` 解析真实长训 checkpoint，matrix 中为 `launchable_now`。
- `allow_partial_windows`、dataset payload cache、`video_backend` 的 decord/opencv 双后端和 `_single_dof.py` / reward-based / object-pose 的 schema 阈值回退都已接上；后续如果继续收 review，优先看视频解码在更大视频集上的稳健性。
- 明确 `class_mapping_status` 从 `Data Gate` 到确认态的收口条件，并同步到 readiness / matrix。
- 使用更强 checkpoint 重新执行 `E-006` checkpoint-forward 与真实 rollout，对 bridge 干预拿到有信息量的对比证据；当前 eval-load、synthetic intervention、rollout preflight 已通过。
- E-003 launch smoke 已把临时 overwrite 配置改成 `TemporaryDirectory` 清理；latent cache dataset 的 payload cache 现在是有上限的 LRU，后续如继续扩大 cache 规模再看更系统的内存 profile。
- `MoWAWindowSample` 的 forbidden input keys 现在补了 `mowa_future_latent` / `mowa_future_latent_target` / `mowa_predicted_future_latent`，后续再扫 batch contract 时继续关注是否还有 `mowa_*` 前缀遗漏。
- E-003 / E-004 的 formal long-training candidate 已补成独立 YAML + preview，并在矩阵中按工程启动口径放行；收益结论仍依赖正式训练与后续 rollout/eval。
- E-004 checkpoint-backed policy rollout smoke 已补齐入口并生成结构化报告；在当前机器上真实 client 被 `robocasa_render_backend_unavailable` 阻断，后续若要继续拿 rollout success_rate，需要先修复或切换 RoboCasa 渲染后端环境，而不是继续堆 smoke。
- `experiment_launch_readiness_matrix.py` 已接入 E-004 rollout smoke；当前 matrix 会同时显示 `hlcgci_policy_rollout_passed=true` 与 `hlcgci_policy_rollout_zero_success=true`，但这不改变 E-004 的 launchable 状态，因为 success 仍是 0.0。
- 仿真全 0% 根因定位已完成并落盘 `13_sim_zero_offline_diagnosis.md`：数据↔仿真动作/状态语义经 expert 回放验证一致（~1mm），E-001/E-003 的 0% 归因为模型闭环质量不足；新增离线动作检查工具 `tools/mowa/e003_offline_action_check.py`、批跑器 `e003_offline_action_curve.sh`、expert 回放 `e003_expert_replay_sim.py`、语言探针 `mowa_lang_sensitivity_probe.py`；eval client 增加 `payload_style`（wanpi/starflow）开关。E-001 在 2026-07-16 前的仿真结果因 client 视角 bug 作废（baseline 已用修正 client 重测 0/10；E-001-mowa 尚无干净测量）。
- 待办：E-003 继续训练并按 `e003_offline_action_curve.sh` 跟踪 eef_pos/z 维 corr（当前 30k 步 pos_corr≈0.51，曲线仍在上升）；E-004 出 ckpt 后用同一离线脚本做 h>0 同族对照；state 位置量 min_max 归一化消融（需重训）；action_horizon 32→8/16 与 LoRA rank 消融作为候选；仿真评测协议固定为每任务 ≥10 episode。
- 2026-07-19/20 进展：离线曲线已跟踪至 46k（40k 为顶点 pos_corr≈0.60，45k/46k 未再超过，recipe 已进入平台）；steps_40000 按 v1.0.1 对齐 horizon=750 跑 OpenDrawer 10 eps 仍 0/10，坐实"pos_corr≈0.6 闭环不足"与"继续加步数无边际收益"；外部基线代码调研（GR00T N1.5 / π0 / WorldDreamer，仓库在 `playground/Code/`）完成，确认动作契约与我们逐维一致、差异在初始化/数据多样性/参数化细节。
- 待办（新增）：EXT-B1 外部正对照按 `14_ext_b1_groot_n15_external_baseline_design.md` v2 执行（合并 codex/claude 评审修订；更名自 E000-B1；checkpoint 用户下载中，先 Phase -1/0A preflight）；E-003-B2（history=10）smoke 待用户拍板后按门禁启动。
