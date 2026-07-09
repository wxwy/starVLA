# MoWA TODO

- 默认 cache 设计已从 window-level artifact（方案 A）重构为 episode-level latent store + window manifest references（方案 C）；新增 `starVLA/dataloader/mowa/episode_latent_store.py`、`window_manifest.py`、`window_latent_sample.py` 和对应 CLI / 单测；遗留 `tools/mowa/build_future_latent_cache.py` 作为方案 A 短期兼容。
- E-001 至 E-007 的活跃训练实验已在 `docs_zh/mowa/mowa_experiment_launch_readiness_matrix.json` 收口为 `launchable_now`；E-009 是 conditional，未触发时不阻断；E-010 eval-only tracking plan/smoke 已补齐。
- E-003 / E-004 已迁移到 WanPI 路径，`train_starvla.py` 数据路径可消费 episode-level latent/history batch；当前仍需在全量 `vae_spatial` cache 完成后补 E-003/E-004 WanPI full-path dry-run evidence。
- E-007 proxy-alpha candidate/readiness smoke 已补齐；实际 proxy optimization 与 frozen alpha 报告仍需单独执行后再接入下游 P0/P1 配置。
- 真实 Wan2.2 已接到 `playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers`；`tools/mowa/e003_wan2_2_latent_cache_smoke.py --execute --episode-index 0 --episode-index 1 --episode-index 4` 已完成 3 episode 真实写盘和 `validate_mowa_latent_cache()` 复验，且 execute 模式现在会自动覆盖专用 cache_root，重复运行仍保持可复跑。E-003 还新增了 launch smoke，和 fake-cache dry-run 一起把 launch readiness 推到 `launchable_now`。
- E-004 也已补上 synthetic launch smoke、checkpoint-backed preflight 与 policy rollout smoke；E-005 checkpoint preflight 现在可通过 `latest_complete` 解析真实长训 checkpoint，matrix 中为 `launchable_now`。
- `allow_partial_windows`、dataset payload cache、`video_backend` 的 decord/opencv 双后端和 `_single_dof.py` / reward-based / object-pose 的 schema 阈值回退都已接上；后续如果继续收 review，优先看视频解码在更大视频集上的稳健性。
- 明确 `class_mapping_status` 从 `Data Gate` 到确认态的收口条件，并同步到 readiness / matrix。
- 使用更强 checkpoint 重新执行 `E-006` checkpoint-forward 与真实 rollout，对 bridge 干预拿到有信息量的对比证据；当前 eval-load、synthetic intervention、rollout preflight 已通过。
- E-003 launch smoke 已把临时 overwrite 配置改成 `TemporaryDirectory` 清理；latent cache dataset 的 payload cache 现在是有上限的 LRU，后续如继续扩大 cache 规模再看更系统的内存 profile。
- `MoWAWindowSample` 的 forbidden input keys 现在补了 `mowa_future_latent` / `mowa_future_latent_target` / `mowa_predicted_future_latent`，后续再扫 batch contract 时继续关注是否还有 `mowa_*` 前缀遗漏。
- E-003 / E-004 的 formal long-training candidate 已补成独立 YAML + preview，并在矩阵中按工程启动口径放行；收益结论仍依赖正式训练与后续 rollout/eval。
- E-004 checkpoint-backed policy rollout smoke 已补齐入口并生成结构化报告；在当前机器上真实 client 被 `robocasa_render_backend_unavailable` 阻断，后续若要继续拿 rollout success_rate，需要先修复或切换 RoboCasa 渲染后端环境，而不是继续堆 smoke。
- `experiment_launch_readiness_matrix.py` 已接入 E-004 rollout smoke；当前 matrix 会同时显示 `hlcgci_policy_rollout_passed=true` 与 `hlcgci_policy_rollout_zero_success=true`，但这不改变 E-004 的 launchable 状态，因为 success 仍是 0.0。
