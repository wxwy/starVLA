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

## 风险提醒

当前有两个训练进程同时以相同 `run_id` 运行并写入同一 checkpoint 目录：

- PID `130534`，启动于 2026-06-19 20:48:09
- PID `154248`，启动于 2026-06-19 20:54:50

这会导致 `summary.jsonl` 与 `checkpoints/` 被双方同时修改。如果不是预期行为，建议停止其中一个进程。

## 相关文件

- 会话记录：`/gemini/code/starVLA/SESSION.md`
- 训练输出：`/gemini/code/starVLA/playground/Checkpoints/P0-M7-E-H2a-01_starflow_libero-4in1_qwen3vl4b_lwfm_ft0_260619_2048/`
- 启动脚本：`examples/LIBERO/train_files/run_starflow_train_ready.sh`
