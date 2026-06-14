# Session Log

## 2026-06-14 — StarFlow-VLA P0-M3 / P0-M4

- 当前分支：`merge-official-starvla-dev`
- 已完成 P0-M3：新增 `starVLA/model/modules/starflow_vla/mapping.py` 与 `__init__.py`，提供 `starflow_mapping` schema 构造和旁路 JSON 保存工具
- 已完成 P0-M4：新增 `tests/test_starflow_vla_reuse.py`，验证 `StarFlowVLA` 继承复用 `Qwen_PI_v3`，不复制 `forward()` / `predict_action()`
- P0-M3 提交：`8efdb81 Add StarFlow-VLA mapping manifest`
- P0-M4 提交：`e9c8af4 Add StarFlow-VLA reuse smoke test`
- `.venv` 中未安装 `pytest`，P0-M4 改用标准库 `unittest` 跑通 4 个用例
- 当前未跟踪目录：`.libero/`、`LIBERO/`，不应误提交

## 2026-06-14 — StarFlow-VLA P0-M5 至 P0-M9

- 已完成 P0-M5 配置级产物：`configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml`
- 已完成 P0-M8 可跳过数据 smoke：`tests/test_starflow_libero_batch.py`
- 已完成 P0-M6 配置级产物：`configs/starflow_vla/stage2_mlp_baseline.yaml`
- 已完成 P0-M7 配置级产物：`configs/starflow_vla/stage3_future_token_ablation.yaml`
- 已完成 P0-M9 工具级产物：`save_starflow_checkpoint_mapping()` 与 `tests/test_starflow_checkpoint_mapping.py`
- 提交：`362dc38 Add StarFlow-VLA stage1 smoke config`
- 提交：`e253189 Add StarFlow-VLA LIBERO batch smoke test`
- 提交：`74a7d2a Add StarFlow-VLA MLP baseline config`
- 提交：`f641a3e Add StarFlow-VLA future token ablation config`
- 提交：`10331d7 Add StarFlow-VLA checkpoint mapping sidecar`
- 当前阻塞：`playground/Datasets/LEROBOT_LIBERO_DATA` 不存在，无法运行真实 LIBERO batch schema、forward/backward、single batch overfit、训练、checkpoint save/load 或 eval smoke

## 2026-06-14 — StarFlow-VLA P0-M10 / P0-M11

- 已完成 P0-M10 preflight：新增 `docs_zh/starflow_vla/EVAL_SMOKE.md` 与 `tests/test_starflow_eval_preflight.py`
- 已完成 P0-M11 文档治理：新增 `docs_zh/starflow_vla/EXPERIMENT_MATRIX.md` 与 `tests/test_starflow_docs_governance.py`
- P0-M10 只完成 shell 语法和 checkpoint/mapping 前置检查；未启动 policy server 或 LIBERO rollout
- P0-M11 文档治理测试已通过：README 链接、核心文档存在性、patch manifest 主要产物覆盖
- 当前仍需准备 `playground/Datasets/LEROBOT_LIBERO_DATA` 和 P0 checkpoint，才能进入真实 Stage B 验证

## 2026-06-14 — StarFlow-VLA 模型路径规则

- 本地模型入口统一放在 `playground/Pretrained_models/`
- 该目录通常保存软链接；已确认 `Qwen3-VL-4B-Instruct` 指向 `/gemini/pretrain/Qwen3-VL-4B-Instruct/`
- 后续新增模型文件下载到 `/gemini/code/models/`，再在 `playground/Pretrained_models/` 下创建软链接
- 相关长期记忆已写入 `MEMORY/starflow_vla_environment.md`
- StarFlow-VLA 检查优先使用 `.venv`；涉及 `import torch` 的命令需预留约 4 分钟

## 2026-06-14 — skill 评估记录

- 评估目标：`humanizer`、`deep-research`、`skill-creator`、`ideation`
- 本地缓存中未找到 `humanizer`、`deep-research`、`ideation` 对应的 `SKILL.md`
- 已确认 `skill-creator` 为元 skill，核心用途是创建/更新 skill，而不是直接处理业务任务
- 后续若需要精确评估前三者，需要先定位其实际 `SKILL.md` 或安装来源

## 2026-06-14 — 官方来源复核

- 官方 Codex 文档确认 `skill-creator` 是内置 skill，且用于创建/更新 skill
- 官方 Deep research 文档确认 `deep research` 是 ChatGPT 功能/工作流，不是公开技能目录中的通用 skill 名称
- 官方 OpenAI 文档把 `ideation` 作为业务/产品使用场景描述，而不是独立 skill 条目
- 在官方 OpenAI/Developers 页面未找到 `humanizer` 作为公开 skill 名称的证据

## 2026-06-14 — 安装结果

- 已安装 `gh-fix-ci`
- 已安装 `gh-address-comments`
- 已安装 `yeet`
- 安装位置：`/root/.codex/skills`

## 2026-05-24 — LIBERO eval 适配 checkpoints 目录下单文件 pt

- 确认早期 checkpoint 不是目录，而是直接位于 `playground/trained_model/starVLA_QwenGR00T_libero4in1_qwen3_dit/checkpoints/` 下的单文件：
  - `steps_1000_pytorch_model.pt`
  - `steps_2000_pytorch_model.pt`
  - `steps_5000_pytorch_model.pt`
  - `steps_10000_pytorch_model.pt`
  - `steps_15000_pytorch_model.pt`
- 已对 `examples/LIBERO/eval_files/run_policy_server.sh` 和 `examples/LIBERO/eval_files/eval_libero.sh` 做最小适配：
  - 当未显式传入 `CKPT` 时，优先选择 `checkpoints/steps_<step>_pytorch_model.pt`
  - 其次回退到旧式 DeepSpeed `checkpoints/steps_<step>/pytorch_model/mp_rank_00_model_states.pt`
  - 最后再回退到 `checkpoints/steps_<step>` 目录
- `eval_libero.sh` 同时补充了这类单文件 checkpoint 的结果目录命名逻辑，输出目录会稳定落到 `playground/eval_results/<task_suite>/steps_<step>`
- 两个脚本都已通过 `bash -n` 语法检查

## 2026-05-24 — LIBERO `libero_goal` 早期阶段评测结果补齐

- 已基于 `playground/eval_results/libero_goal/steps_1000`、`steps_2000`、`steps_5000`、`steps_10000`、`steps_15000`、`steps_20000_pytorch_model_mp_rank_00_model_states.pt`、`starVLA_QwenGR00T_libero4in1_qwen3_dit_steps_40000` 目录中的 rollout 视频文件名，按 `success/failure` 统计各阶段整体和分任务成功率
- 已将 `steps_2000`、`steps_5000`、`steps_10000`、`steps_15000` 的整体结果、分任务表格和阶段分析补入 `examples/LIBERO/train_files/training_log_1229_libero4in1_qwen3oft.md`
- 当前 `libero_goal` overall success rate 时间线：
  - `steps_1000`: `0.0%`
  - `steps_2000`: `1.0%`
  - `steps_5000`: `4.4%`
  - `steps_10000`: `33.4%`
  - `steps_15000`: `36.4%`
  - `steps_20000`: `46.0%`
  - `steps_40000`: `36.8%`
- 已在训练日志中补充阶段趋势总结：第一次明显跃迁出现在 `steps_10000`，当前已统计阶段 overall 最优 checkpoint 是 `steps_20000`

## 2026-05-25 — LIBERO `libero_goal` 补测 `steps_70000/80000`

- 已基于以下结果目录统计 `steps_70000/80000` 的整体和分任务成功率：
  - `playground/eval_results/libero_goal/starVLA_QwenGR00T_libero4in1_qwen3_dit_checkpoints_steps_70000`
  - `playground/eval_results/libero_goal/starVLA_QwenGR00T_libero4in1_qwen3_dit_checkpoints_steps_80000`
- 统计结果：
  - `steps_70000`: `370/500 = 74.0%`
  - `steps_80000`: `357/500 = 71.4%`
- 关键结论：
  - `steps_70000` 是当前 `libero_goal` overall 最优 checkpoint
  - `steps_80000` 虽然 overall 略低，但在 `open_the_middle_drawer_of_the_cabinet`、`put_the_cream_cheese_in_the_bowl`、`put_the_wine_bottle_on_the_rack` 等长尾任务上优于 `steps_70000`
- 已将 `steps_70000`、`steps_80000` 的表格、分析，以及 `70000 vs 80000` 对比和更新后的阶段趋势总结补入 `examples/LIBERO/train_files/training_log_1229_libero4in1_qwen3oft.md`

## 2026-05-25 — 仿真评测记录按训练联动框架重构

- 已重写 `examples/LIBERO/train_files/training_log_1229_libero4in1_qwen3oft.md` 中的“仿真评测记录”部分，不再按 checkpoint 逐段叙述
- 新结构改为：
  - `评测设置`
  - `训练阶段与恢复连续性`
  - `Checkpoint 总览`
  - `关键结论`
  - `任务演化总表`
  - `训练-测评联动分析`
  - `任务类型分析`
  - `Checkpoint 选择建议`
  - `附录：各 checkpoint 详细结果`
- 新框架已显式纳入：
  - 单卡到 2 卡训练切换
  - batch size / effective batch 变化
  - warmup / cosine scheduler 区间
  - 2 卡切换时 optimizer 丢失导致的恢复不连续风险
  - `steps_40000/70000/80000` 的 `rank_sharded` optimizer 恢复稳定阶段
- 当前仿真分析的核心判断已调整为：
  - `steps_20000 -> steps_40000` 的变化不能只按 step 增长解释
  - `steps_70000` 是 overall 最优 checkpoint
  - `steps_80000` 是长尾任务对照 checkpoint

## 2026-05-25 — 实验概况与训练过程更新到 `80000 step`

- 已同步更新 `examples/LIBERO/train_files/training_log_1229_libero4in1_qwen3oft.md` 中以下部分，使其与当前最终训练状态一致：
  - `实验概况`
  - `训练过程`
  - `训练配置`
  - `性能优化记录`
  - `W&B Step-Epoch 曲线分析`
  - `已知问题`
- 主要修正：
  - 将“当前步数 `19000 / 80000`”改为“训练完成步数 `80000 / 80000`”
  - 将训练过程重写为单卡早期、单卡连续训练、2 卡切换、2 卡稳定训练四/五个阶段
  - 显式写入“切到 2 卡时发生 optimizer 丢失 / 恢复不连续”
  - 将训练配置中的 `save_interval` 更新为 `500`
  - 将 `datasets.vla_data.per_device_batch_size` 更新为最终稳定阶段使用的 `8`
  - 将单卡历史吞吐表标注为历史测算，避免与后期 2 卡阶段混淆

## 2026-05-25 — 增补 LIBERO SOTA 对齐与不足分析

- 已在 `examples/LIBERO/train_files/training_log_1229_libero4in1_qwen3oft.md` 末尾新增 `对齐当前 LIBERO SOTA 与不足分析`
- 内容包括：
  - `LIBERO-Goal` 单套件与公开方法的可比性边界
  - 与 `TraceVLA / OpenVLA / PixelVLA` 的目标成功率对齐
  - 当前结果的主要不足：评测范围、长尾任务、训练协议可比性、鲁棒性评测缺失
  - 下一步建议：补齐 `Spatial/Object/Long`、定向补长尾、固定训练协议、增加鲁棒性评测
- 当前在日志中的结论：
  - `steps_70000=74.0%` 已接近已发表强基线 `TraceVLA=75.1%`
  - 但距离更前沿公开结果 `PixelVLA=85.8%` 仍有明显差距
  - 目前还不能声称“对齐完整 LIBERO SOTA”

## 2026-05-23 — LIBERO eval 环境缺包定位

- `tmux` 会话 `sim` 当前稳定复现报错：`ModuleNotFoundError: No module named 'robosuite'`
- 直接原因不是 `eval_libero.py` 路径错误，而是 `.libero` 虚拟环境只安装了 `libero` editable 包本体，没有安装 `LIBERO/requirements.txt` 中声明的运行依赖
- 证据：
  - `.libero` 中 `pip show libero` 显示 editable project 指向 `/gemini/code/starVLA/LIBERO`
  - `LIBERO/setup.py` 中 `install_requires=[]`，因此 `pip install -e LIBERO` 不会自动带上依赖
  - 当前 `.libero` 缺失的关键包包括：`robosuite`、`bddl`、`robomimic`、`hydra-core`、`easydict`、`transformers`、`opencv-python`、`einops`、`thop`、`future`、`gym`、`cloudpickle`
- 已修正 `examples/LIBERO/eval_files/install_libero.sh`：
  - 改为使用仓库内 `.libero` 虚拟环境
  - 改为基于仓库相对路径定位 `LIBERO`
  - 安装顺序改为先 `python -m pip install -r requirements.txt`，再 `python -m pip install -e .`
  - 验证步骤增加 `robosuite`、`bddl` 导入检查
- 当前 `sim` 会话尚未恢复；仍需在允许联网安装依赖的前提下重新执行安装脚本或等价安装命令

## 2026-05-23 — LIBERO eval 输出路径只读

- `tmux` 会话 `sim` 在依赖补齐后继续运行到评测入口，但 `eval_libero.py` 创建视频输出目录时失败
- 直接报错：`OSError: [Errno 30] Read-only file system: '/gemini/code/starVLA/playground/trained_model/.../results'`
- 根因：`examples/LIBERO/eval_files/eval_libero.sh` 默认把 `video_out_path` 写到 checkpoint 所在的 `playground/trained_model/.../results`，该路径在当前环境只读
- 已做最小修复：将 `video_out_path` 改为仓库内可写路径 `playground/eval_results/${task_suite_name}/${folder_name}`

## 2026-05-23 — LIBERO init_states 与 PyTorch 2.6 兼容

- `tmux` 会话 `sim` 在修复输出路径后继续报错：`_pickle.UnpicklingError: Weights only load failed`
- 根因：PyTorch `2.6.0` 将 `torch.load` 的默认 `weights_only` 从 `False` 改为 `True`，而 LIBERO 的 `init_states` 文件是可信任的普通 pickle 数据，不是纯模型权重
- 已做最小修复：在 `examples/LIBERO/eval_files/eval_libero.py` 导入 `libero` 前为 `torch.load` 补兼容包装；当调用方未显式传入 `weights_only` 时，默认按 `False` 处理，兼容 LIBERO 的 `init_states` 旧格式文件
- 已重新拉起 `tmux sim` 验证：当前评测已进入真实 rollout 阶段，日志显示 `Task: open the middle drawer of the cabinet`，并已完成多个 episode

## 2026-05-22 — policy 推理加载分片 checkpoint 误报 missing keys

- `tmux` 会话 `policy` 的报错不是 `tmux` 故障，而是 `deployment/model_server/server_policy.py` 在加载 `steps_40000` 时失败
- 根因一：`starVLA/model/framework/share_tools.py` 对分片目录调用 `accelerate.load_checkpoint_in_model(..., strict=True)`，而当前 `accelerate` 会对每个 shard 单独执行 `model.load_state_dict(..., strict=True)`，把“尚未加载到当前 shard 的参数”误判成 `Missing key(s)`
- 修复：分片目录在共享加载入口里先根据 `*.index.json` 做完整 key 校验，再用 `strict=False` 逐 shard 实际加载，避免分片级误报
- 根因二：误报消掉后，暴露出 HF/Qwen safetensors 的兼容差异：`lm_head.weight` 作为 tied weight 未单独落盘，`rotary*_inv_freq` 作为非持久/缓存 buffer 出现在 index 中
- 修复：严格 key 校验里过滤上述已知无害差异，保留其它真实 missing/unexpected keys 的报错能力

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
- 推理侧 checkpoint 辅助文件解析增强：`_resolve_inference_run_files()` 现在按 checkpoint 目录、直接父目录、旧训练 run 目录，以及 symlink resolve 后的对应目录查找 `config.yaml` 和 `dataset_statistics.json`
- 该逻辑兼容 `.../Checkpoints/run/checkpoints/steps_xxx` 与 `.../trained_model/name/steps_xxx` 两种布局

## 2026-06-14 — StarFlow-VLA P0 LIBERO goal 数据准备与 batch smoke

- 已下载 P0 最小数据集 `IPEC-COMMUNITY/libero_goal_no_noops_1.0.0_lerobot` 到 `/gemini/code/datasets/LEROBOT_LIBERO_DATA/libero_goal_no_noops_1.0.0_lerobot`
- 已创建仓库入口软链接 `playground/Datasets/LEROBOT_LIBERO_DATA -> /gemini/code/datasets/LEROBOT_LIBERO_DATA`
- 已复制 `examples/LIBERO/train_files/modality.json` 到数据集 `meta/modality.json`
- 已确认当前 LIBERO registry 使用 7D action 与 8D state，state keys 包含 `x,y,z,roll,pitch,yaw,pad,gripper`
- `.venv/bin/python -m unittest tests.test_starflow_libero_batch -v` 已通过，真实 batch 含 image / lang / state / action

## 2026-06-14 — StarFlow-VLA P0 Stage B forward/backward 与单 batch overfit smoke

- 已使用 `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml`、本地 `Qwen3-VL-4B-Instruct` symlink 和真实 `libero_goal` batch 构建 `StarFlowVLA`
- 已冻结 `qwen_vl_interface`，执行单 batch forward/backward，`action_loss` 为有限值且可训练参数获得梯度
- 已在同一真实 batch 上优化 action head / projectors 6 步，loss 从 `1.79121411` 降至 `0.10281464`
- 未运行完整训练循环、checkpoint 保存/加载、LIBERO rollout、评测或部署

## 2026-06-15 — StarFlow-VLA P0 smoke checkpoint 与 eval preflight

- 已创建 `playground/Checkpoints/starflow_vla_stage1_qwenpi_v3_native/checkpoints/steps_1`
- `steps_1` 包含模型分片、optimizer、scheduler、trainer_state、config、dataset_statistics 和 `starflow_mapping.json`
- `.venv/bin/python -m unittest tests.test_starflow_eval_preflight -v` 已通过，checkpoint mapping 检查不再 skip
- 已用 `load_model_weights(..., strict=True)` 复验 `steps_1` 可加载；仅出现 rotary buffer 未使用的兼容警告
- 未运行 resume 100 step、policy server、LIBERO rollout、success_rate 统计或部署

## 2026-06-15 — StarFlow-VLA P0 MLP baseline Stage B smoke

- 已使用 `configs/starflow_vla/stage2_mlp_baseline.yaml`、本地 `Qwen3-VL-4B-Instruct` symlink 和真实 `libero_goal` batch 构建 `QwenOFT`
- 已冻结 `qwen_vl_interface`，执行 MLP baseline 单 batch forward/backward 与 6 步 overfit smoke
- 同一真实 batch 上 loss 从 `0.87645137` 降至 `0.48976591`
- 未保存 baseline checkpoint，未运行完整训练循环、LIBERO rollout、评测或部署

## 2026-06-15 — StarFlow-VLA P0 future token Stage B smoke

- 已使用 `configs/starflow_vla/stage3_future_token_ablation.yaml` 和真实 `libero_goal` batch 复验 `num_target_vision_tokens=0/8/16/32/64`
- 五组均完成真实 forward/backward、loss finite 与 3 步 single batch overfit smoke
- `num_target_vision_tokens=0` 边界通过，未触发构建或 forward blocker
- 未保存 5 组 ablation checkpoint，未运行完整训练循环、LIBERO rollout、评测或部署

## 2026-06-15 — StarFlow-VLA P0 最小 LIBERO rollout smoke

- 已用 `.venv` 启动 policy server，checkpoint 为 `playground/Checkpoints/starflow_vla_stage1_qwenpi_v3_native/checkpoints/steps_1`
- 已用 `.libero` 运行 `libero_goal` 的 `max_tasks=1`、`num_trials_per_task=1` eval smoke
- eval 客户端成功连接 server，完成 1 episode，输出 `Total success rate: 0.0`
- 生成 `playground/eval_results/libero_goal/starflow_vla_stage1_smoke_steps_1/rollout_open_the_middle_drawer_of_the_cabinet_episode0_failure.mp4`
- 退出阶段有 EGL / `libGLU.so.0` 清理期警告，但 eval 进程退出码为 0
- 未运行完整 LIBERO suite、failure taxonomy、多 seed 评测或正式性能报告

## 2026-06-15 — StarFlow-VLA P0 QwenPI_v3 baseline compatibility smoke

- 已将 `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml` 临时设为 `framework.name=QwenPI_v3`
- 已使用真实 `libero_goal` batch 执行 QwenPI_v3 baseline forward/backward smoke
- `action_loss` 为有限值，反传后可训练参数获得梯度
- 未运行 baseline overfit、checkpoint 保存/加载、完整训练、LIBERO rollout 或评测
