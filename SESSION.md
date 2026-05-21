# Session Log

## 2026-05-17 — 项目初始化

- 创建 CLAUDE.md、AGENTS.md、SESSION.md、TODO.md、MEMORY/
- 当前分支：未知（非 git 仓库或有 detached HEAD）
- 环境就绪，可开始开发

## 2026-05-20 — LIBERO 训练中断排查

- 定位到 `train_starvla.py` 在 `Step 20000` 后进入 checkpoint 保存流程时被 cgroup OOM 杀死
- 证据：`/sys/fs/cgroup/memory/memory.limit_in_bytes=34359738368`，`memory.oom_control` 显示 `oom_kill=2`
- 根因：DeepSpeed 训练仍用 `accelerator.get_state_dict()` + `torch.save()` 保存 9.3G 单文件 checkpoint，保存期内存峰值触发 32GB 限制
- 修复：DeepSpeed 周期性/最终 checkpoint 改为 `accelerator.save_state()` 目录式保存，并补充目录式 checkpoint 的自动恢复
- 顺手修复：`starVLA/dataloader/__init__.py` 在未初始化分布式时直接 `dist.get_rank()` 的异常

## 2026-05-20 — 本地临时 checkpoint 工作区

- `train_starvla.py` 支持可选 `trainer.local_checkpoint_root`
- 启用后 checkpoint 先保存到本地临时目录 `<local_checkpoint_root>/<run_id>/checkpoints`
- 若本地临时目录为空，则从网络盘 run 目录复制“最新完整 checkpoint + config/dataset statistics/summary”
- 每次本地保存完成后，启动独立后台进程同步该 checkpoint 到网络盘 run 目录
- 若网络盘存在同名旧目录，后台同步会先改名为 `.stale_<ts>`，再放入新目录，避免半成品 checkpoint 阻塞恢复

## 2026-05-20 — checkpoint 保存模式与本地空间预检

- `train_starvla.py` 新增 `trainer.save_with_training_state`，默认 `False`
- `train_starvla.py` 新增 `trainer.save_checkpoint_as_directory`，默认 `True`
- 默认改为“轻量目录式保存”：模型使用 Accelerate 标准分片目录保存，optimizer/scheduler/trainer state 分文件保存
- 轻量目录式恢复时，模型通过分片逐步加载，减少 CPU 侧峰值内存
- 仅当 `save_with_training_state=True` 且使用 DeepSpeed 时，才走 `accelerator.save_state()` 完整训练态目录式保存
- checkpoint 读取同时兼容旧单文件、轻量目录式分片目录和 DeepSpeed 完整训练态目录式格式
- 启用本地临时 checkpoint 中转时，保存前会根据最近 checkpoint 大小估算所需空间，空间不足则提前报错，避免写到一半失败
- `run_libero_train.sh` 增加 `save_checkpoint_as_directory=True`、`save_with_training_state=False` 和 `checkpoint_max_shard_size=5GB` 启动参数
- 本地临时 checkpoint 的“最新性检查/必要时复制”已前移到程序启动阶段后台执行，并在 `prepare_training()` 前等待完成，以便和模型构建、数据初始化并行
- 每次本地 checkpoint 后台同步回网络盘完成后，会自动删除本地源 checkpoint，避免临时路径持续堆积
- 启动阶段若发现本地临时路径存在多个 checkpoint，只保留最新一个
- 启动阶段若发现本地最新 checkpoint 新于网络盘且完整，则直接复用本地版本，并后台继续向网络盘补同步，不再用网络旧版本覆盖本地
- 后台 checkpoint 同步改为单 worker 串行消费 `.checkpoint_sync_queue`，避免多个同步进程并发运行
- 新增 `trainer.local_checkpoint_keep_count`，用于控制本地临时路径保留的 checkpoint 版本数；脚本默认设为 `2`
- 启动阶段从网络盘整理到临时路径时，仍然只保证本地有一份最新可恢复版本；`local_checkpoint_keep_count` 只作用于训练过程中新 checkpoint 的本地裁剪
- 正常训练同步完成后，不再直接删除当前本地 checkpoint，而是按 `local_checkpoint_keep_count` 裁剪较旧版本

## 2026-05-21 — LIBERO 训练降峰值参数调整

- `run_libero_train.sh` 中 `checkpoint_max_shard_size` 从 `5GB` 调整为 `4GB`
- `run_libero_train.sh` 中 `per_device_batch_size` 从 `32` 调整为 `16`
- `starVLA/config/deepseeds/ds_config.yaml` 中 `gradient_accumulation_steps` 从 `1` 调整为 `2`
- 目标是在保持全局 batch 基本不变的前提下，降低单步激活/保存时的内存峰值
- 进一步将 `local_checkpoint_keep_count` 调整为 `1`，避免本地临时盘同时保留两个 checkpoint 导致保存前空间预检失败
- 确认 `Accelerator()` 的梯度累积步数不直接读取训练 yaml 或 DeepSpeed 配置文件中的该字段；实际生效值来自 `accelerate launch --gradient_accumulation_steps`
- `run_libero_train.sh` 已显式传入 `--gradient_accumulation_steps 2`，确保 `accelerator.gradient_accumulation_steps` 与预期一致
- 对齐 `local_checkpoint_keep_count` 的真实语义：该值表示“本地临时路径最多可占用的 checkpoint 配额数”
- 后台同步完成后的本地保留数改为 `max(local_checkpoint_keep_count - 1, 0)`，与保存前空间预检逻辑一致
- 因此当 `local_checkpoint_keep_count=1` 时，同步完成后本地应清空；当 `=2` 时，同步完成后本地保留 1 份最新 checkpoint
