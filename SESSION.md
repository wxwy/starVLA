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
- 定位到一次 checkpoint 保存报错 `DataLoader worker ... killed by signal: Killed` 的直接根因是 cgroup OOM：`accelerator.save_model()` 内部会在 CPU 侧克隆模型 state_dict，峰值期间 dataloader worker 被系统先杀掉
- `starVLA/dataloader/__init__.py` 中 VLA dataloader 的 `num_workers` 改为可配置，默认回落到 `0`
- `run_libero_train.sh` 显式传入 `--datasets.vla_data.num_workers 0`，优先降低 checkpoint 保存期的 CPU 内存压力
- `run_libero_train.sh` 中 `save_interval` 从 `250` 调整为 `500`，降低 checkpoint 保存频率
- `train_starvla.py` 中补充了轻量训练态加载/保存路径下的 `del + gc.collect()`：
  - 加载后回收 `trainer_state`
  - 单文件保存后回收 `state_dict`
  - 轻量目录式保存后回收 `optimizer_state`、`scheduler_state`、`trainer_state`
- 轻量目录式模型保存不再调用 `accelerator.save_model()`；改为按参数/缓冲区逐个搬到 CPU、按 shard 流式写盘并生成 index 文件
- 目标是绕开 `accelerator.get_state_dict()` / `clone_tensors_for_torch_save()` 带来的整份模型 `state_dict` CPU 克隆峰值
- `train_starvla.py` 中未显式配置时的默认 `save_format` 改为 `safetensors`
- `run_libero_train.sh` 显式传入 `--trainer.save_format safetensors`，使轻量目录式默认产出 `model-xxxxx.safetensors` 与 `model.safetensors.index.json`
- 目录 checkpoint 读取逻辑改为“优先按 `preferred_format` 探测，缺失时再回退到其他格式”
- 对分片 index 会校验其 `weight_map` 中引用的 shard 文件是否真实存在，避免因为目录中残留了另一种格式的 index 文件而误读
- 训练侧 `TrainerUtils.load_pretrained_backbones()` 与推理侧 `baseframework.from_pretrained()` 现在统一调用 `share_tools.py` 中的共享模型权重加载入口
- 共享入口负责：
  - 单文件 / 目录式 / 分片目录 checkpoint 解析
  - 指定格式优先、缺失回退
  - 分片目录加载与单文件加载
- `examples/LIBERO/eval_files/eval_libero.sh` 已改为当前仓库相对路径启动，不再硬编码作者机器路径
- `eval_libero.sh` 默认 `CKPT` 对齐到 `playground/trained_model/.../steps_23000`
- `eval_libero.sh` 对结果目录的推导同时兼容旧的 `/checkpoints/steps_xxx` 路径和新的 `trained_model/.../steps_xxx` 路径
- `eval_libero.sh` 本身不直接加载模型权重；真正的 checkpoint 解析和权重加载仍由 `run_policy_server.sh` 启动的 policy server 负责
- `safetensors` 分片写入前会再次确保目标目录存在，避免保存过程中因目录缺失触发 `SavetensorError: I/O error: No such file or directory`
- 针对 2 卡 24G / 64G 内存机器，`run_libero_train.sh` 已调整为：
  - `per_device_batch_size=8`
  - `gradient_accumulation_steps=2`
  - `num_processes=2`
  - `num_workers=4`
- 临时 checkpoint 存储扩容到约 100G 后，`local_checkpoint_keep_count` 调整为 `2`，表示本地临时路径在同步完成后可保留 1 份旧版本作为缓冲
- 对应全局 batch 为 `8 x 2 x 2 = 32`
- 启动阶段本地 checkpoint 最新性检查/复制逻辑已前移到 `main()` 刚完成配置归一化之后，早于 `setup_directories()`、模型构建和数据集初始化
- 记录遗留问题：LIBERO 数据集构建/初始化链路仍偏慢，主要热点在 `LeRobotSingleDataset._get_metadata()`、`_load_or_compute_statistics()`、`_get_all_steps()` 和 `LeRobotMixtureDataset.update_metadata()`；后续应加阶段计时日志量化
- 2 卡 ZeRO2 恢复轻量 checkpoint 时，旧版单文件 `optimizer.pt` 会导致 rank1 在 DeepSpeed `state_dict_list[dp_rank]` 处越界
- 轻量训练态 optimizer 保存已改为 per-rank 分片：`optimizer_rank_00000.pt`、`optimizer_rank_00001.pt` 等，并在 `trainer_state.json` 记录 `optimizer_format=rank_sharded` 与 `optimizer_world_size`
- 轻量训练态恢复只在 `optimizer_format=rank_sharded` 且保存时 `optimizer_world_size` 与当前一致时恢复 optimizer
- 旧版单文件 `optimizer.pt` 只在当前 `world_size=1` 时恢复；多卡场景下会跳过 optimizer 状态，继续恢复模型、scheduler 与 step，避免伪造 rank 分片造成错误恢复
- 现有 `steps_26000/optimizer.pt` 曾在本地临时路径和网络 checkpoint 路径下复制出 2 份 rank 文件，但由于 `trainer_state.json` 未声明 `rank_sharded`，当前加载逻辑不会把这些文件当作可靠 ZeRO2 optimizer 分片
- 2 卡 ZeRO2 + `gradient_accumulation_steps=2` 下，`accelerator.accumulate()` 默认会在非同步步进入 DeepSpeed `no_sync()`，触发 `no_sync context manager is incompatible with gradient partitioning logic of ZeRO stage 2`
- `Accelerator` 初始化改为显式使用 `GradientAccumulationPlugin(sync_each_batch=True)`，并从 `ACCELERATE_GRADIENT_ACCUMULATION_STEPS` 读取累积步数；保留累积步数语义，同时避免 ZeRO2 下进入 `no_sync`
- 发现网络 checkpoint 目录几乎每个 step 都留下 `steps_xxx.stale_*`，原因是后台同步在目标目录已存在时永久保留旧目录
- 后台同步逻辑改为：源目录与目标目录文件名/大小一致时直接跳过；确需替换时只在替换期间临时保留旧目录，替换成功后删除该临时旧目录，避免后续继续堆积 `.stale_*`
- 继续定位 `.stale_*` 大量产生的直接触发点：gradient accumulation 下非同步 micro-batch 不会递增 `completed_steps`，但原训练循环仍会执行 eval/log/save，导致同一个 `steps_xxx` 在下一次 micro-batch 被再次保存和同步
- 训练循环已改为仅在 `accelerator.sync_gradients=True` 的真实 optimizer update 步执行 eval、日志和 checkpoint 保存，避免同一个 step 重复保存
- `examples/LIBERO/train_files/starvla_cotrain_libero.yaml` 中显式添加 `datasets.vla_data.num_workers: 8`
- `run_libero_train.sh` 默认 `num_workers` 同步调整为 `8`，CLI override 仍会传入 `--datasets.vla_data.num_workers`
- W&B 初始化改为使用稳定 run id：默认由 `run_id` 归一化得到 `wandb.init(id=..., resume="allow")`
- 后续同一 `run_id` 的训练重启会续写同一个 W&B run，避免每次重启在网站上生成新的碎片 run
- 已确认当前本地历史 W&B run 目录有 30 个；历史碎片不会因代码修改自动合并，若需要网站全局视图，需单独解析历史日志并上传为一个新 W&B run
- 已将历史碎片 W&B 日志解析去重后上传为 clean 合并 run：`1229_libero4in1_qwen3oft_merged_history_clean`
- 合并 run 包含原始记录 466 条，去重后 264 个 step，范围 `100..43200`
- 本地导出文件：`playground/Checkpoints/1229_libero4in1_qwen3oft/wandb_merged_history_clean.csv` 与 `.jsonl`
- 定位到 `Attempt ... Cannot allocate memory` 的直接失败点在 DataLoader worker 内部 PyAV / `torchvision.io.VideoReader` 打开视频 codec context
- 当前 2 卡配置下 `num_workers=8` 等价于 16 个 worker，PyTorch 默认 `prefetch_factor=2` 会放大到最多 32 个预取 batch，视频解码并发过高
- `starVLA/dataloader/__init__.py` 增加 `datasets.vla_data.prefetch_factor` 配置透传
- LIBERO 启动脚本与 yaml 默认调整为 `num_workers=5`、`prefetch_factor=2`，2 卡并发预取峰值为 10 个 worker / 20 个 batch
