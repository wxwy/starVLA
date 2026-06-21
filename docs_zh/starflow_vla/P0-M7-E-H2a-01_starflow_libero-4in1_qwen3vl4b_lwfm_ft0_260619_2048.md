# E-H2a-01-4in1 训练记录

## 实验元信息

| 字段 | 值 |
| --- | --- |
| 实验 ID | E-H2a-01-4in1 |
| run_id | `P0-M7-E-H2a-01_starflow_libero-4in1_qwen3vl4b_lwfm_ft0_260619_2048` |
| 框架 | StarFlowVLA |
| VLM | Qwen3-VL-4B-Instruct |
| 数据 | LIBERO / `libero_all` |
| 服务器 | 单卡 A100 40G |
| GPU 单价 | 3.39 元/小时 |
| 启动时间 | 2026-06-19 20:48:09 CST |
| 记录时间 | 2026-06-20 08:07:55 CST |
| 已运行 | 11.33 小时 |
| 当前步数 | 3250 / 80000 |
| 累计费用 | 38.41 元 |
| 平均每 step 费用 | 0.01182 元（约 1.18 分） |

## 训练配置

| 字段 | 值 |
| --- | --- |
| per_device_batch_size | 4 |
| gradient_accumulation_steps | 8 |
| 有效全局 batch | 32 |
| max_train_steps | 80000 |
| save_interval | 125 |
| eval_interval | 250 |
| logging_frequency | 10 |
| freeze_modules | qwen_vl_interface |

## 每 step / 累计成本（按 save_interval 汇总）

| step | 累计费用（元） | 本区间费用（元） |
| ---: | ---: | ---: |
| 125 | 1.48 | 1.48 |
| 250 | 2.95 | 1.48 |
| 375 | 4.43 | 1.48 |
| 500 | 5.91 | 1.48 |
| 625 | 7.39 | 1.48 |
| 750 | 8.86 | 1.48 |
| 875 | 10.34 | 1.48 |
| 1000 | 11.82 | 1.48 |
| 1125 | 13.29 | 1.48 |
| 1250 | 14.77 | 1.48 |
| 1375 | 16.25 | 1.48 |
| 1500 | 17.73 | 1.48 |
| 1625 | 19.20 | 1.48 |
| 1750 | 20.68 | 1.48 |
| 1875 | 22.16 | 1.48 |
| 2000 | 23.64 | 1.48 |
| 2125 | 25.11 | 1.48 |
| 2250 | 26.59 | 1.48 |
| 2375 | 28.07 | 1.48 |
| 2500 | 29.54 | 1.48 |
| 2625 | 31.02 | 1.48 |
| 2750 | 32.50 | 1.48 |
| 2875 | 33.98 | 1.48 |
| 3000 | 35.45 | 1.48 |
| 3125 | 36.93 | 1.48 |
| 3250 | 38.41 | 1.48 |

> 注：费用按“当前已运行时间 ÷ 当前步数”线性估算。由于训练仍在继续，后续 step 的实际单价会随总耗时变化而下降。

## 关键事件

- 原 run（`..._1644` / `..._1949`）因 `_log_metrics` 中 `dist.get_rank()` 在单进程下未初始化进程组而报错终止。
- 当前 run 应用修复：将 `dist.get_rank() == 0` 改为 `self.accelerator.is_main_process`。
- 当前 run 以 `is_resume=False` 从头启动。

## 恢复训练记录（Resume，2026-06-21）

该 run 于 2026-06-21 在 tmux 会话 `train` 中恢复训练，设备由 A100 40G 切换为 A100 80G。

| 字段 | 值 |
| --- | --- |
| tmux 会话名 | `train` |
| tmux 创建时间 | 2026-06-21 12:09:01 CST |
| tmux 窗口 | `bash`（pane PID 1675） |
| 主机 | `bitahub-a20205015879249920489293` |
| GPU | `NVIDIA A100-SXM4-80GB`（81920 MiB） |
| 单价 | 5.58 元/小时 |
| 训练状态 | 运行中（attached） |
| 恢复源 checkpoint | `steps_7875` |
| resume 时间 | 2026-06-21 12:44 CST |
| 当前配置 | `configs/starflow_vla/ablations/future_tokens_0.yaml` |
| `is_resume` | `True` |

### 变更后的训练参数

| 字段 | 原值 | resume 后 |
| --- | --- | --- |
| per_device_batch_size | 4 | 8 |
| gradient_accumulation_steps | 8 | 4 |
| 有效全局 batch | 32 | 32 |
| save_interval | 125 | 250 |
| eval_interval | 250 | 500 |
| logging_frequency | 10 | 20 |
| num_workers | 未设置 | 3 |
| local_checkpoint_root | 未设置 | `/localdisk-tmp` |
| local_checkpoint_keep_count | 未设置 | 2 |

> 有效全局 batch 保持 32 不变（4×8 → 8×4）。

### Resume 启动命令

```bash
cd /disk/rl/starVLA
tmux attach -t train
# 在 tmux 会话内执行
/opt/conda/envs/starVLA/bin/python -u starVLA/training/train_starvla.py \
  --config_yaml configs/starflow_vla/ablations/future_tokens_0.yaml \
  --framework.name StarFlowVLA \
  --framework.qwenvl.base_vlm /disk/rl/starVLA/playground/Pretrained_models/Qwen3-VL-4B-Instruct \
  --datasets.vla_data.data_root_dir /disk/rl/starVLA/playground/Datasets/LEROBOT_LIBERO_DATA \
  --datasets.vla_data.data_mix libero_all \
  --datasets.vla_data.per_device_batch_size 8 \
  --datasets.vla_data.num_workers 3 \
  --trainer.freeze_modules qwen_vl_interface \
  --trainer.max_train_steps 80000 \
  --trainer.save_interval 250 \
  --trainer.logging_frequency 20 \
  --trainer.eval_interval 500 \
  --trainer.gradient_accumulation_steps 4 \
  --trainer.checkpoint_format lightweight \
  --trainer.enable_local_checkpoint_staging True \
  --trainer.local_checkpoint_root /localdisk-tmp \
  --trainer.local_checkpoint_keep_count 2 \
  --trainer.save_checkpoint_as_directory True \
  --trainer.save_with_training_state False \
  --trainer.save_format safetensors \
  --run_root_dir /disk/rl/starVLA/playground/Checkpoints \
  --run_id P0-M7-E-H2a-01_starflow_libero-4in1_qwen3vl4b_lwfm_ft0_260619_2048 \
  --wandb_run_id P0-M7-E-H2a-01_starflow_libero-4in1_qwen3vl4b_lwfm_ft0_260619_2048 \
  --wandb_name P0-M7-E-H2a-01_starflow_libero-4in1_qwen3vl4b_lwfm_ft0_260619_2048 \
  --wandb_project starflow_vla \
  --wandb_entity silencewx-harbin-institute-of-technology \
  --trainer.is_resume True
```

### Resume 关键事件

- 2026-06-21 12:09：创建 tmux 会话 `train`。
- 2026-06-21 12:44：从 `steps_7875` resume，设备切换为 A100(80G)，训练参数同步调整。
- 2026-06-21 12:45：成功写入 `config.yaml` / `config.full.yaml`，训练正常推进。

## 风险提醒

当前有两个训练进程同时以相同 `run_id` 运行并写入同一 checkpoint 目录：

- PID `130534`，启动于 2026-06-19 20:48:09
- PID `154248`，启动于 2026-06-19 20:54:50

这会导致 `summary.jsonl` 与 `checkpoints/` 被双方同时修改。如果不是预期行为，建议停止其中一个进程。

> 注：2026-06-21 resume 后，上述双进程问题已不复存在；当前仅在 tmux 会话 `train` 内运行单一训练进程。

## 相关文件

- 会话记录：`/gemini/code/starVLA/SESSION.md`
- 训练输出：`/gemini/code/starVLA/playground/Checkpoints/P0-M7-E-H2a-01_starflow_libero-4in1_qwen3vl4b_lwfm_ft0_260619_2048/`
- 启动脚本：`examples/LIBERO/train_files/run_starflow_train_ready.sh`

## 自动监控状态

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-21 19:16:30 CST |
| 训练状态 | 🟢 运行中（来自 tmux `train`） |
| run_id | `P0-M7-E-H2a-01_starflow_libero-4in1_qwen3vl4b_lwfm_ft0_260619_2048` |
| 当前步数 | **12526 / 80000** |
| 完成比例 | 16.0% |
| 训练速度 | ~4.91 s/it |
| data_time | 0.0 s |
| model_time | 1.273 s |
| 已运行时间 | 6:31:17 |
| 预计剩余时间 | 92:07:08 |
| 最新完整 checkpoint | `steps_12500` |
| GPU | NVIDIA A100-SXM4-80GB |
| GPU 利用率 | 45% |
| 显存使用 | 48939 MiB / 81920 MiB (59.7%) |
| 功耗 | 441.67 W / 400.00 W |
| 温度 | 60°C |
| 内存总量 | 1.0Ti |
| 内存已用 | 68Gi |
| 内存空闲 | 21Gi |
| 存储 `/disk/rl` | 562T / 700T (81% 已用) |
| 存储 `/localdisk-tmp` | 1.3T / 3.5T (38% 已用) |
| 每 step 成本 | ~0.0076 元 |
| 已产生成本 | ~36.39 元 |
| 完整训练预估成本 | ~608.84 元 |