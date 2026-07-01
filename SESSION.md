# MoWA 会话状态

## 当前阶段
- M1 G0 Data Verification Gate 数据复验已通过；P0 ConstructibleHeads one-step train smoke 已通过，仍不计入 E-001 主训练。

## 数据状态
- `.robocase` venv 已就绪：`/gemini/code/starVLA/.robocase`。
- 已删除临时的 `robocasa365` conda 环境；改为使用项目根目录 `.robocase` venv 运行 robocasa 下载/评测脚本。
- `examples/Robocasa_365/train_files/download_target_human.sh` 已改为激活 `.robocase` venv。
- `DATASET_BASE_PATH` 已配置为 `/gemini/code/datasets/robocasa365`，并通过软链映射到 `playground/Datasets/robocasa365`。
- **固定主对比 recipe 数据已补齐**：`mowa_robocasa365_target_human_atomic_core_v1` 10 个 target/human/atomic 任务全部下载并解压完成。
- 已用 `tools/mowa/g0_recipe_smoke.py` 重新核验固定配方 `mowa_robocasa365_target_human_atomic_core_v1`：
  - task_count=10，available_task_count=10，missing_task_count=0。
  - go_no_go=`TBD: recipe available; profile/leakage/labels still Data Gate`。
- 已完成本轮 G0 复验：
  - 10 项 profile / P0 label coverage / latent cache manifest smoke 已逐任务重跑，30 个命令全部通过；latent manifest 的 OpenDrawer 代表性检查 missing_video_count=0。
  - batch dataloader smoke：10 个任务、30 个 sampled windows，通过 future action target-only 检查。
  - metadata leakage gate：5055 个 episode、15165 个 anchor windows、failed_window_count=0。
  - temporal profile：5055 个 parquet、1342150 行，metadata_total_frames 与 parquet_total_rows 对齐；WAM Hz/window 仍不冻结。
- 真实 label builder 阈值 / class mapping、生产 dataloader workers、真实 latent cache 仍为 Data Gate。
- P0 ConstructibleHeads 最小训练入口已完成代码准备：
  - 仅启用 `task_progress`、`action_outcome_class` 两个当前可构造 head。
  - 其他五类 P0 head 继续 mask，不进入训练 smoke。
  - one-step 更新使用手写 SGD step，避免 `torch.optim` 触发额外 `torch._dynamo` / `triton` 导入。
  - 默认 CPU 运行，预计显存占用为 0。
  - `.venv/bin/python -m unittest tests.mowa.test_mowa_p0_heads -v` 已通过 1 项测试，用时 166.764s；本机 `torch` 导入需要分钟级等待窗口。
  - `p0_constructible_heads_train_smoke.py` 已通过并生成 `docs_zh/mowa/mowa_p0_constructible_heads_train_smoke.json`：sample_count=30、input_shape=[30, 4]、loss_before=0.024293631315231323、loss_after=0.024045661091804504、class_mapping_status=Data Gate。

## 进行中的任务
- 2026-07-01 已完成 10 个 target/human/atomic core 任务下载、G0 复验和 P0 one-step smoke。
- 当前没有启动 P0/P1 主训练；本轮仅完成 G0 复验与 P0 one-step smoke。

## 下一步
- 若继续推进 P0，先明确 E-001 是否从 smoke 进入主训练；主训练前仍需复验 production dataloader workers、train/val split、distributed sampler。
- 训练入口必须显式引用 G0 temporal profile，但不得把 WAM Hz/window 当作已冻结结论。
- 真实 Wan latent cache builder 仍需单独放行。
