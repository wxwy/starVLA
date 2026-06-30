# Session Log

## 2026-06-30 — StarFlow 标准化评测连通性 smoke

| 字段 | 值 |
| --- | --- |
| 评测目标 | 验证 StarFlow 目录式 checkpoint 可通过标准 LIBERO 评测链路启动 |
| checkpoint | `playground/Checkpoints/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619/checkpoints/steps_10000` |
| 任务集 | `libero_goal` |
| smoke 设置 | `MAX_TASKS=1`，`NUM_TRIALS_PER_TASK=1`，总 1 episode |
| 输出目录 | `playground/eval_results/libero_goal/std_smoke_P0-M5-ft32-4in1_steps_10000` |
| 结果 | `eval_report.json` 已生成；`total_episodes=1`，`total_successes=0`，`success_rate=0.0` |
| 环境观察 | `.venv` 可用 CUDA 1 卡；policy server 首次 import 极慢，`module imports finished` 耗时约 531.75s；模型加载后显存约 9.9GiB |
| 异常说明 | rollout 完成后 EGL 清理阶段出现 `libGLU.so.0` / `EGL_NOT_INITIALIZED`，符合既有记录中“episode 完成后清理报错不计入评测失败”的口径 |
| 下一步 | 正式标准化评测使用 `libero_goal`、10 任务、每任务 50 trials；当前单卡环境预计单 checkpoint 耗时很长，建议按 checkpoint 顺序串行跑并保留 `eval_report.json` |

## 2026-06-30 — StarFlow 标准化评测正式启动

| 字段 | 值 |
| --- | --- |
| checkpoint | `playground/Checkpoints/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619/checkpoints/steps_10000` |
| 输出目录 | `playground/eval_results/libero_goal/std_P0-M5-ft32-4in1_steps_10000` |
| 任务集 | `libero_goal` |
| 正式设置 | `NUM_TRIALS_PER_TASK=50`，`MAX_TASKS=-1`，预期 10 任务共 500 episodes |
| 后台 PID | `211097` |
| 日志 | `run.log`、`policy_server.log` |
| 状态 | 已启动；policy server 正在加载 checkpoint |

## 2026-06-30 — StarFlow 标准化评测运行快照（16:56 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | `2026-06-30 16:56 CST` |
| 主进程 | `server_policy.py` PID `211134`，CPU `22.8%`，RSS `3528544 KB` |
| client 进程 | `eval_libero.py` PID `276422`，CPU `3.9%`，RSS `331372 KB` |
| 任务集进度 | `libero_goal`，已进入正式评测阶段，`MAX_TASKS=-1`，仍在执行 |
| GPU | `nvidia-smi` 当前无可见 GPU 进程，显存 `0MiB / 24258MiB`，GPU-Util `0%` |
| 系统内存 | `503Gi` 总，`121Gi` 已用，`30Gi` 空闲，`377Gi` 可用 |
| 目录空间 | `/gemini/code` 使用率 `51%`，评测输出与 checkpoint 所在盘可用空间充足 |
| 备注 | 这轮记录按实验规范补入了资源占用信息，后续每个 checkpoint 的正式结果也应同步记录对应资源快照 |

## 2026-06-30 — StarFlow 标准化评测结果（steps_10000）

| 字段 | 值 |
| --- | --- |
| checkpoint | `playground/Checkpoints/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619/checkpoints/steps_10000` |
| 任务集 | `libero_goal` |
| 评测规模 | `10` tasks, `50` trials/task, `500` episodes |
| 最终结果 | `180 / 500` |
| 最终成功率 | `36.0%` |
| 任务明细 | `turn on the stove 50/50`，`put the bowl on the plate 45/50`，`put the bowl on top of the cabinet 30/50`，`open the middle drawer of the cabinet 24/50`，`put the bowl on the stove 19/50`，`open the top drawer and put the bowl inside 4/50`，`push the plate to the front of the stove 3/50`，`put the cream cheese in the bowl 3/50`，`put the wine bottle on top of the cabinet 2/50`，`put the wine bottle on the rack 0/50` |
| 服务端资源 | `server_policy.py` PID `211134`，CPU 约 `193%`，GPU 显存约 `10.16GiB` |
| 测试端资源 | `eval_libero.py` 在评测过程中持续运行，最终写出 `eval_report.json`；结尾阶段已退出 |
| 系统资源 | `503Gi` 总内存，`121Gi` 已用，`377Gi` 可用；GPU `0%` 利用率，显存 `10158MiB / 24258MiB` |
| 产物 | `eval_report.json`、各 task rollout 视频、`run.log`、`policy_server.log` |
| 结论 | 这组 checkpoint 已完成标准化评测，可作为后续其他 step 的对照基线 |

## 2026-06-21 — P1 continuous_head build / forward-backward / 10-step smoke

| 字段 | 值 |
| --- | --- |
| 配置 | `configs/starflow_vla/state/continuous_head.yaml` |
| run_id | `P1-M1-E-H2b-02_continuous_head_10step_smoke_260621_1849` |
| 数据 | `libero_goal`，真实 LIBERO batch，batch size 1 |
| build | 通过；`StarFlowVLA` + `LayerwiseFM`，`num_target_vision_tokens=32`，action head `state_encoder` 存在 |
| single-batch forward/backward | 通过；action `(8, 7)`，state `(1, 8)`；loss `1.6213243` finite；捕获到 action head 收到 state shape `(1, 1, 8)`；668 个可训练参数张量获得 finite grad |
| 10-step smoke | 通过；最终 step 10 `action_dit_loss=1.1300001`，`mse_score=0.1388952` |
| checkpoint | `playground/Checkpoints/P1-M1-E-H2b-02_continuous_head_10step_smoke_260621_1849/checkpoints/steps_10` |
| final model | `playground/Checkpoints/P1-M1-E-H2b-02_continuous_head_10step_smoke_260621_1849/final_model` |
| manifest | `state_mode=continuous_head`，`state_enters_instruction=false`，`state_enters_action_head=true`，`num_target_vision_tokens=32` |
| W&B | offline run written under the run directory |
| 下一步 | 可进入 `E-H2b-02-4in1` 正式训练；正式训练需覆盖 `DATA_MIX=libero_all` 并沿用 P0 主线 batch/step/保存策略 |

## 2026-06-21 — P1 continuous_head runtime 最小实现

| 字段 | 值 |
| --- | --- |
| 任务 | 实现 `state_mode=continuous_head` 的 StarFlowVLA runtime 分流 |
| 状态 | 已完成最小代码改动与单元测试 |
| 关键改动 | `QwenPI_v3` 抽出 `_prepare_state_condition()` 默认 hook；`StarFlowVLA` 按 `framework.state_mode` 选择 `discretized_instruction` / `continuous_head` / `none`；`hybrid_gated` 当前显式报错，避免误跑 P2 |
| manifest | `starflow_mapping` 现在读取真实 `state_mode`，并记录 `state_enters_instruction` 与 `state_enters_action_head` |
| 已通过测试 | `/opt/conda/envs/starVLA/bin/python -m unittest tests.test_starflow_vla_reuse -v`；`/opt/conda/envs/starVLA/bin/python -m unittest tests.test_starflow_checkpoint_mapping -v` |
| 未通过/非本次引入 | `tests.test_starflow_future_token_variants` 仍期待 `[0, 8, 16, 32, 64]`，但当前配置为 `[0, 16, 32, 64]`；`tests.test_starflow_docs_governance` 依赖的 `playground/Checkpoints/starflow_vla_stage1_qwenpi_v3_native/checkpoints/steps_1/starflow_mapping.json` 当前不存在 |
| 下一步 | 跑 `continuous_head.yaml` 的 build / single-batch forward-backward / 10-step smoke，再启动 `E-H2b-02-4in1` 正式训练 |

## 2026-06-20 — StarFlow Train 会话状态（08:07 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-20 08:07 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行约 31h35m） |
| 数据加载子进程 PID | `124465` |
| 训练状态 | ✅ **正常运行中**（主进程 `RLl+`，数据 worker `Sl+`） |
| 当前步数 | **7375 / 80000**（与 08:02 相同） |
| 最新完整 checkpoint | `steps_7375`（约 18G） |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 163%（ps）/ **512.5%**（top 瞬时） |
| 数据 worker CPU | 23.0%（ps） |
| 主进程内存 | RSS 约 2.0 GiB |
| 数据 worker 内存 | RSS 约 5.5 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.7%），功耗 250 W |
| 系统负载 | 22.04 / 22.40 / 22.02 |
| 系统内存 | 503 GiB 总，77 GiB 已用，421 GiB 可用 |
| root overlay | 30G 总，860M 已用，30G 可用（**3%**） |
| `/root/temp` | 28K（已清理） |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_7375/` |
| 输出目录总大小 | 1.1T |
| 备注 | 距 08:02 约 5 分钟步数仍未推进；主进程 CPU 瞬时飙升至 **512.5%**，GPU 保持满载，可能处于 checkpoint 前准备或 CPU 密集型操作；`steps_7375` 保持完整；建议继续观察下一个监控点 |


## 2026-06-20 — StarFlow Train 会话状态（08:02 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-20 08:02 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行约 31h30m） |
| 数据加载子进程 PID | `124465` |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`/`R`） |
| 当前步数 | **7375 / 80000**（与 07:58 相同） |
| 最新完整 checkpoint | `steps_7375`（约 18G） |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 163%（ps）/ 133.3%（top 瞬时） |
| 数据 worker CPU | 23.0%（ps）/ 73.3%（top 瞬时） |
| 主进程内存 | RSS 约 2.0 GiB |
| 数据 worker 内存 | RSS 约 5.5 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.7%），功耗 250 W |
| 系统负载 | 24.48 / 23.19 / 22.17 |
| 系统内存 | 503 GiB 总，78 GiB 已用，420 GiB 可用 |
| root overlay | 30G 总，860M 已用，30G 可用（**3%**） |
| `/root/temp` | 28K（已清理） |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_7375/` |
| 输出目录总大小 | 1.1T |
| 备注 | 距 07:58 约 4 分钟步数未推进，GPU 仍满载，数据 worker 瞬时 CPU 升高，可能处于训练迭代或 checkpoint 前准备；`steps_7375` 保持完整；预计下一记录 step 7500 |


## 2026-06-20 — StarFlow Train 会话状态（07:58 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-20 07:58 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行约 31h26m） |
| 数据加载子进程 PID | `124465` |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | **7375 / 80000**（`summary.jsonl` 最新记录） |
| 最新完整 checkpoint | `steps_7375`（约 18G） |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 163%（ps）/ 125.0%（top 瞬时） |
| 数据 worker CPU | 23.0%（ps） |
| 主进程内存 | RSS 约 2.0 GiB |
| 数据 worker 内存 | RSS 约 5.5 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.7%），功耗 250 W |
| 系统负载 | 22.81 / 22.56 / 21.77 |
| 系统内存 | 503 GiB 总，78 GiB 已用，419 GiB 可用 |
| root overlay | 30G 总，860M 已用，30G 可用（**3%**） |
| `/root/temp` | 28K（已清理） |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_7375/` |
| 输出目录总大小 | 1.1T |
| 备注 | 距上次监控约 9.75 小时，训练从 step 5250 推进至 step 7375，新增 17 个 checkpoint；GPU 利用率恢复并稳定 100%；输出目录从 744G 增至 1.1T；预计向 step 7500 推进 |


## 2026-06-19 — StarFlow Train 会话状态（22:12 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 22:12 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行约 21:40） |
| 数据加载子进程 PID | `124465` |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | **5250 / 80000**（`summary.jsonl` 最新记录与 22:07 相同） |
| 最新完整 checkpoint | `steps_5250`（约 18G） |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 163%（ps）/ 126.7%（top 瞬时） |
| 数据 worker CPU | 24.7%（ps） |
| 主进程内存 | RSS 约 2.0 GiB |
| 数据 worker 内存 | RSS 约 5.5 GiB |
| GPU | `P1.gpu.medium`，利用率 **86%**，显存 39540/40488 MiB（97.7%），功耗 250 W |
| 系统负载 | 30.76 / 26.74 / 26.71 |
| 系统内存 | 503 GiB 总，80 GiB 已用，417 GiB 可用 |
| root overlay | 30G 总，860M 已用，30G 可用（**3%**） |
| `/root/temp` | 28K（已清理） |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_5250/` |
| 输出目录总大小 | 744G |
| 备注 | 从 22:07 到 22:12 的 5 分钟内 `summary.jsonl` 未新增步数记录，可能处于 checkpoint 同步或数据加载等待；GPU 利用率从 96% 略降至 86%；`steps_5250` 保持完整；预计下一记录 step 5375 |


## 2026-06-19 — StarFlow Train 会话状态（22:07 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 22:07 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行约 21:35） |
| 数据加载子进程 PID | `124465` |
| 训练状态 | ✅ **正常运行中**（主进程 `RLl+`，数据 worker `Dl+`） |
| 当前步数 | **5250 / 80000**（`summary.jsonl` 最新记录） |
| 最新完整 checkpoint | `steps_5250`（约 18G，已同步至网络） |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 163%（ps）/ 126.7%（top 瞬时） |
| 数据 worker CPU | 24.7%（ps）/ 80.0%（top 瞬时） |
| 主进程内存 | RSS 约 2.0 GiB |
| 数据 worker 内存 | RSS 约 5.5 GiB |
| GPU | `P1.gpu.medium`，利用率 **96%**，显存 39540/40488 MiB（97.7%），功耗 250 W |
| 系统负载 | 21.75 / 26.85 / 26.90 |
| 系统内存 | 503 GiB 总，80 GiB 已用，418 GiB 可用 |
| root overlay | 30G 总，860M 已用，30G 可用（**3%**） |
| `/root/temp` | 28K（已清理） |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_5250/` |
| 输出目录总大小 | 744G |
| 备注 | GPU 利用率从 22:03 的 0% 恢复至 96%，训练恢复正常；`steps_5250` 已完整保存并同步至网络；`/root/temp` 已清理，root overlay 从 30% 回落至 3%；输出目录从 728G 增至 744G；预计向 step 5375 推进 |


## 2026-06-19 — StarFlow Train 会话状态（22:03 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 22:03 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行约 21:31） |
| 数据加载子进程 PID | `124465` |
| 训练状态 | ✅ **正常运行中**（主进程 `DLl+`，数据 worker `Sl+`） |
| 当前步数 | **5250 / 80000**（`summary.jsonl` 最新记录） |
| 最新完整 checkpoint | `steps_5125`（约 18G，已同步至网络） |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 163%（ps）/ 33.3%（top 瞬时） |
| 数据 worker CPU | 24.7%（ps） |
| 主进程内存 | RSS 约 6.6 GiB |
| 数据 worker 内存 | RSS 约 5.5 GiB |
| GPU | `P1.gpu.medium`，利用率 **0%**，显存 39540/40488 MiB（97.7%），功耗 250 W |
| 系统负载 | 28.34 / 27.42 / 26.71 |
| 系统内存 | 503 GiB 总，83 GiB 已用，414 GiB 可用 |
| root overlay | 30G 总，8.9G 已用，22G 可用（**30%**） |
| `/root/temp` | 18G（`steps_5125` 本地暂存） |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_5125/` |
| 输出目录总大小 | 728G |
| 备注 | GPU 利用率从 16:40 的 98% 降至 0%，显存仍占 97.7%，可能处于 checkpoint 保存或数据加载等待阶段；`steps_5125` 已完整保存；`/root/temp` 仍有 18G 暂存，说明轻量 checkpoint 同步尚未完成或正在排队；root overlay 从 3% 升至 30%，需关注磁盘空间 |


## 2026-06-19 — StarFlow Train 会话状态（16:40 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 16:40 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行 16:08:31） |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `69612`（已变为 defunct，同步已完成） |
| 训练状态 | ✅ **正常运行中**（主进程 `R`/`SLl+`，数据 worker `D`/`Dl+`） |
| 当前步数 | **4000 / 80000**（`steps_4000` 已同步至网络） |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 163%（ps）/ 112.5%（top 瞬时） |
| 数据 worker CPU | 25.1%（ps）/ 未进入 top 前列 |
| 主进程内存 | RSS 约 4.6 GiB（checkpoint 后仍处高位） |
| 数据 worker 内存 | RSS 约 5.4 GiB |
| GPU | `P1.gpu.medium`，利用率 **98%**，SM **98%**，显存 39540/40488 MiB（97.6%），功耗 250 W |
| `/root/temp` | `checkpoints/` 子目录为空（已清理） |
| root overlay | **3%（860M / 30G）**（本地暂存已清理） |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_4000/` |
| 负载 | 11.25 / 13.06 / 15.10（1 分钟负载略有回升） |
| 数据加载 | 数据 worker `D`/`Dl+`，线程数 1645（较 16:35 的 1741 回落） |
| 目标 r2 目录 | 不存在 |
| 备注 | `steps_4000` 已完成本地保存并同步至网络；sync worker PID 69612 已变为 defunct；`/root/temp` 已清理，root overlay 从 62% 回落至 3%；GPU 持续高负载 98%，训练向 step 4125 推进；数据 worker 线程数降至 1645；主进程内存 4.6 GiB 仍处高位；预计 steps_4125 约 17:10 |

## 2026-06-19 — StarFlow Train 会话状态（16:35 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 16:35 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行 16:03:42） |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `69612`（sync worker，运行中） |
| 训练状态 | ✅ **正常运行中**（主进程 `R`/`RLl+`，数据 worker `D`/`Dl+`，sync worker `Rs`） |
| 当前步数 | **4000 / 80000**（`steps_4000` 本地保存并同步中） |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 163%（ps）/ 126.7%（top 瞬时） |
| 数据 worker CPU | 25.1%（ps）/ 126.7%（top 瞬时） |
| sync worker CPU | 35.9%（ps）/ 46.7%（top 瞬时） |
| 主进程内存 | RSS 约 4.6 GiB（checkpoint 期间上升） |
| 数据 worker 内存 | RSS 约 5.4 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39540/40488 MiB（97.6%），功耗 250 W |
| `/root/temp` | `checkpoints/steps_4000/` 已创建（16:34），占用约 19G |
| root overlay | **62%（19G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_3875/`（`steps_4000` 同步中） |
| 负载 | **10.99 / 13.34 / 15.84**（1 分钟负载显著下降） |
| 数据加载 | 数据 worker `D`/`Dl+`，线程数 1741（较 16:30 的 2221 回落） |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练成功推进至 step 4000；`steps_4000` 本地已保存于 16:34，sync worker PID 69612 正在同步至网络；root overlay 因本地暂存升至 62%；主进程内存因 checkpoint 升至 4.6 GiB；数据 worker 线程数降至 1741；负载显著降至 10.99；GPU 持续满载；预计 steps_4125 约 17:05 |

## 2026-06-19 — StarFlow Train 会话状态（16:30 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 16:30 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行 15:58:09） |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（同步已完成） |
| 训练状态 | ✅ **正常运行中**（主进程 `R`/`RLl+`，数据 worker `D`/`Dl+`） |
| 当前步数 | **3875 / 80000**（向 step 4000 推进中） |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 163%（ps）/ 120.0%（top 瞬时） |
| 数据 worker CPU | 25.1%（ps）/ 未进入 top 前列 |
| 主进程内存 | RSS 约 2.0 GiB |
| 数据 worker 内存 | RSS 约 5.4 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39540/40488 MiB（97.6%），功耗 250 W |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **3%（860M / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_3875/` |
| 负载 | 16.13 / 16.64 / 17.52（1 分钟负载下降） |
| 数据加载 | 数据 worker `D`/`Dl+`，线程数 2221（较 16:25 的 1165 再次上升） |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练向 step 4000 推进；GPU 持续满载；数据 worker 线程数升至 2221；负载降至 16.13；主进程内存稳定 2.0 GiB；本地暂存干净；预计 steps_4000 约 16:40-16:50 |

## 2026-06-19 — StarFlow Train 会话状态（16:25 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 16:25 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行 15:53:22） |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（同步已完成） |
| 训练状态 | ✅ **正常运行中**（主进程 `S`/`RLl+`，数据 worker `S`/`Dl+`） |
| 当前步数 | **3875 / 80000**（向 step 4000 推进中） |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 163%（ps）/ 126.7%（top 瞬时） |
| 数据 worker CPU | 25.1%（ps）/ 80.0%（top 瞬时） |
| 主进程内存 | RSS 约 2.0 GiB |
| 数据 worker 内存 | RSS 约 5.4 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39540/40488 MiB（97.6%），功耗 250 W |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **3%（860M / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_3875/` |
| 负载 | 17.60 / 17.83 / 18.20（1 分钟负载回升） |
| 数据加载 | 数据 worker `S`/`Dl+`，CPU 瞬时 80.0%，线程数 1165（较 16:20 的 2125 下降） |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练向 step 4000 推进；GPU 持续满载；数据 worker 线程数降至 1165，但 CPU 瞬时 80.0%；负载回升至 17.60；主进程内存稳定 2.0 GiB；本地暂存干净；建议继续观察 steps_4000 是否按时出现 |

## 2026-06-19 — StarFlow Train 会话状态（16:20 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 16:20 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行 15:48:39） |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（同步已完成） |
| 训练状态 | ✅ **正常运行中**（主进程 `S`/`RLl+`，数据 worker `S`/`Dl+`） |
| 当前步数 | **3875 / 80000**（向 step 4000 推进中） |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 163%（ps）/ 120.0%（top 瞬时） |
| 数据 worker CPU | 25.1%（ps）/ 未进入 top 前列 |
| 主进程内存 | RSS 约 2.0 GiB（已从 checkpoint 后的 4.6 GiB 回落） |
| 数据 worker 内存 | RSS 约 5.4 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39540/40488 MiB（97.6%），功耗 250 W |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **3%（860M / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_3875/` |
| 负载 | 14.79 / 17.47 / 18.36（1 分钟负载下降） |
| 数据加载 | 数据 worker `S`/`Dl+`，线程数 2125（较 16:15 的 2605 回落） |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练向 step 4000 推进；GPU 利用率从 74% 恢复至 100%，数据加载瓶颈缓解；数据 worker 线程数回落至 2125；主进程内存从 4.6 GiB 回落至 2.0 GiB；负载降至 14.79；本地暂存干净；预计 steps_4000 约 16:40 |

## 2026-06-19 — StarFlow Train 会话状态（16:15 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 16:15 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行 15:43:55） |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（同步已完成） |
| 训练状态 | ✅ **正常运行中**（主进程 `S`/`SLl+`，数据 worker `D`/`Dl+`） |
| 当前步数 | **3875 / 80000**（向 step 4000 推进中） |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 163%（ps）/ 133.3%（top 瞬时） |
| 数据 worker CPU | 25.1%（ps）/ 未进入 top 前列 |
| 主进程内存 | RSS 约 4.6 GiB |
| 数据 worker 内存 | RSS 约 5.4 GiB |
| GPU | `P1.gpu.medium`，利用率 **74%**，SM **74%**，显存 39540/40488 MiB（97.6%），功耗 250 W |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **3%（860M / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_3875/` |
| 负载 | **20.55 / 19.46 / 18.95**（1 分钟负载上升） |
| 数据加载 | 数据 worker `D`/`Dl+`，线程数 **2605**（较 16:10 的 877 大幅上升，历史最高） |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练向 step 4000 推进；数据 worker 线程数骤升至 2605（历史最高），GPU 利用率降至 74%，数据加载可能再次成为瓶颈；负载升至 20.55；主进程内存 4.6 GiB；本地暂存干净；建议关注线程数是否继续增长，若超过 3000 或 GPU 持续低于 70% 需进一步诊断；预计 steps_4000 约 16:40 |

## 2026-06-19 — StarFlow Train 会话状态（16:10 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 16:10 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行 15:39:10） |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（同步已完成） |
| 训练状态 | ✅ **正常运行中**（主进程 `S`/`SLl+`，数据 worker `S`/`Dl+`） |
| 当前步数 | **3875 / 80000**（`steps_3875` 已同步至网络） |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 163%（ps）/ 120.0%（top 瞬时） |
| 数据 worker CPU | 25.1%（ps）/ 未进入 top 前列 |
| 主进程内存 | RSS 约 4.6 GiB（checkpoint 后仍处高位） |
| 数据 worker 内存 | RSS 约 5.4 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39540/40488 MiB（97.6%），功耗 250 W |
| `/root/temp` | `checkpoints/` 子目录为空（已清理） |
| root overlay | **3%（860M / 30G）**（本地暂存已清理） |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_3875/` |
| 负载 | 16.49 / 19.44 / 18.67（1 分钟负载下降） |
| 数据加载 | 数据 worker `S`/`Dl+`，线程数 877（较 16:06 的 1837 显著下降） |
| 目标 r2 目录 | 不存在 |
| 备注 | `steps_3875` 已完成本地保存并同步至网络；`/root/temp` 已清理，root overlay 从 62% 回落至 3%；GPU 恢复满载，训练向 step 4000 推进；数据 worker 线程数降至 877；主进程内存 4.6 GiB 仍处高位；负载降至 16.49；预计 steps_4000 约 16:40 |

## 2026-06-19 — StarFlow Train 会话状态（16:06 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 16:06 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行 15:34:14） |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 存在同步队列文件与 `.worker.lock` |
| 训练状态 | ✅ **正常运行中**（主进程 `S`/`SLl+`，数据 worker `S`/`Dl+`） |
| 当前步数 | **3875 / 80000**（`steps_3875` 本地保存并同步中） |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 163%（ps）/ 112.5%（top 瞬时） |
| 数据 worker CPU | 25.1%（ps）/ 未进入 top 前列 |
| 主进程内存 | RSS 约 4.6 GiB（checkpoint 期间上升） |
| 数据 worker 内存 | RSS 约 5.4 GiB |
| GPU | `P1.gpu.medium`，利用率 **72%**，SM **72%**，显存 39540/40488 MiB（97.6%），功耗 250 W |
| `/root/temp` | `checkpoints/steps_3875/` 已创建（16:04），占用约 19G |
| root overlay | **62%（19G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_3750/`（`steps_3875` 同步中） |
| 负载 | **21.12 / 20.35 / 18.45**（1 分钟负载继续上升） |
| 数据加载 | 数据 worker `S`/`Dl+`，线程数 1837（较 16:01 的 1645 上升） |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练成功推进至 step 3875；`steps_3875` 本地已保存于 16:04，正在同步至网络；root overlay 因本地暂存升至 62%；主进程内存因 checkpoint 升至 4.6 GiB；数据 worker 线程数达 1837；GPU 因同步降至 72%；负载升至 21.12；建议等待同步完成并观察 root overlay 回落 |

## 2026-06-19 — StarFlow Train 会话状态（16:01 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 16:01 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行 15:29:28） |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（同步已完成） |
| 训练状态 | ✅ **正常运行中**（主进程 `S`/`SLl+`，数据 worker `R`/`Sl+`） |
| 当前步数 | **3750 / 80000**（向 step 3875 推进中） |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 163%（ps）/ 125.0%（top 瞬时） |
| 数据 worker CPU | 25.1%（ps）/ 未进入 top 前列 |
| 主进程内存 | RSS 约 2.0 GiB |
| 数据 worker 内存 | RSS 约 5.4 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39540/40488 MiB（97.6%），功耗 250 W |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **3%（860M / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_3750/` |
| 负载 | **19.81 / 18.23 / 17.25**（1 分钟负载上升） |
| 数据加载 | 数据 worker `R`/`Sl+`，线程数 1645（较 15:56 的 1261 再次上升） |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练向 step 3875 推进；GPU 持续满载；数据 worker 线程数再次升至 1645；负载上升至 19.81；主进程内存稳定 2.0 GiB；本地暂存干净；建议继续观察 steps_3875 是否按时出现 |

## 2026-06-19 — StarFlow Train 会话状态（15:56 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 15:56 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行 15:24:42） |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（同步已完成） |
| 训练状态 | ✅ **正常运行中**（主进程 `S`/`SLl+`，数据 worker `D`/`Dl+`） |
| 当前步数 | **3750 / 80000**（向 step 3875 推进中） |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 163%（ps）/ 580.0%（top 瞬时峰值） |
| 数据 worker CPU | 25.1%（ps）/ 26.7%（top 瞬时） |
| 主进程内存 | RSS 约 2.0 GiB（已从 checkpoint 后的 4.6 GiB 回落） |
| 数据 worker 内存 | RSS 约 5.4 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39540/40488 MiB（97.6%），功耗 250 W |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **3%（860M / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_3750/` |
| 负载 | 17.06 / 16.87 / 16.53（相对稳定） |
| 数据加载 | 数据 worker `D`/`Dl+`，线程数 1261（较 15:51 的 1741 回落） |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练向 step 3875 推进；GPU 利用率从 74% 恢复至 100%，数据加载瓶颈缓解；主进程内存从 4.6 GiB 回落至 2.0 GiB；数据 worker 线程数回落至 1261；负载稳定；本地暂存干净；预计 steps_3875 约 16:04-16:15 |

## 2026-06-19 — StarFlow Train 会话状态（15:51 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 15:51 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行 15:20:02） |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（同步已完成） |
| 训练状态 | ✅ **正常运行中**（主进程 `S`/`SLl+`，数据 worker `R`/`Sl+`） |
| 当前步数 | **3750 / 80000**（向 step 3875 推进中） |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 163%（ps）/ 126.7%（top 瞬时） |
| 数据 worker CPU | 25.1%（ps）/ 86.7%（top 瞬时） |
| 主进程内存 | RSS 约 4.6 GiB |
| 数据 worker 内存 | RSS 约 5.4 GiB |
| GPU | `P1.gpu.medium`，利用率 **74%**，SM **74%**，显存 39540/40488 MiB（97.6%），功耗 250 W |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **3%（860M / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_3750/` |
| 负载 | 17.01 / 17.85 / 16.74（1 分钟负载下降） |
| 数据加载 | 数据 worker `R`/`Sl+`，CPU 瞬时 86.7%，线程数 1741（较 15:47 的 1165 上升） |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练向 step 3875 推进；GPU 利用率从 100% 降至 74%，数据 worker CPU 升至 86.7%、线程数升至 1741，数据加载压力增大，可能成为瓶颈；负载降至 17.01；本地暂存干净；预计 steps_3875 可能略有延迟 |

## 2026-06-19 — StarFlow Train 会话状态（15:47 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 15:47 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行 15:15:11） |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（同步已完成） |
| 训练状态 | ✅ **正常运行中**（主进程 `S`/`SLl+`，数据 worker `D`/`Dl+`） |
| 当前步数 | **3750 / 80000**（向 step 3875 推进中） |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 163%（ps）/ **580.0%**（top 瞬时峰值） |
| 数据 worker CPU | 25.1%（ps）/ 33.3%（top 瞬时） |
| 主进程内存 | RSS 约 4.6 GiB |
| 数据 worker 内存 | RSS 约 5.4 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39540/40488 MiB（97.6%），功耗 250 W |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **3%（860M / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_3750/` |
| 负载 | **23.15 / 17.76 / 16.33**（1 分钟负载显著上升） |
| 数据加载 | 数据 worker `D`/`Dl+`，线程数 1165 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练向 step 3875 推进；主进程 CPU 出现 580% 瞬时峰值；系统负载显著上升至 23.15；GPU 仍满载；数据 worker 线程数稳定 1165；本地暂存干净；预计 steps_3875 约 16:12 |

## 2026-06-19 — StarFlow Train 会话状态（15:42 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 15:42 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行 15:10:17） |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（同步已完成） |
| 训练状态 | ✅ **正常运行中**（主进程 `R`/`SLl+`，数据 worker `D`/`Dl+`） |
| 当前步数 | **3750 / 80000**（`steps_3750` 已同步至网络） |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 163%（ps）/ 126.7%（top 瞬时） |
| 数据 worker CPU | 25.1%（ps）/ 160.0%（top 瞬时） |
| 主进程内存 | RSS 约 4.6 GiB（checkpoint 后仍处高位） |
| 数据 worker 内存 | RSS 约 5.4 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39540/40488 MiB（97.6%），功耗 250 W |
| `/root/temp` | `checkpoints/` 子目录为空（已清理） |
| root overlay | **3%（860M / 30G）**（本地暂存已清理） |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_3750/` |
| 负载 | 12.92 / 15.18 / 15.62（1 分钟负载下降） |
| 数据加载 | 数据 worker `D`/`Dl+`，线程数 1165（较 15:37 的 1933 显著下降） |
| 目标 r2 目录 | 不存在 |
| 备注 | `steps_3750` 已完成本地保存并同步至网络；`/root/temp` 已清理，root overlay 从 62% 回落至 3%；GPU 持续满载，训练向 step 3875 推进；数据 worker 线程数降至 1165；主进程内存 4.6 GiB 仍处高位；预计约 16:12 到达 steps_3875 |

## 2026-06-19 — StarFlow Train 会话状态（15:37 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 15:37 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行 15:05:34） |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 待确认（存在同步队列文件） |
| 训练状态 | ✅ **正常运行中**（主进程 `S`/`SLl+`，数据 worker `D`/`Dl+`） |
| 当前步数 | **3750 / 80000**（`summary.jsonl` 已更新） |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 163%（ps）/ 106.7%（top 瞬时） |
| 数据 worker CPU | 25.1%（ps）/ 26.7%（top 瞬时） |
| 主进程内存 | RSS 约 4.6 GiB（**checkpoint 期间上升**） |
| 数据 worker 内存 | RSS 约 5.4 GiB |
| GPU | `P1.gpu.medium`，利用率 **94%**，SM **94%**，显存 39540/40488 MiB（97.6%），功耗 250 W |
| `/root/temp` | `checkpoints/steps_3750/` 已创建（15:34），占用约 19G |
| root overlay | **62%（19G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_3625/`（`steps_3750` 同步中） |
| 负载 | 17.73 / 15.56 / 15.73（1 分钟负载回升） |
| 数据加载 | 数据 worker `D`/`Dl+`，线程数 1933（较 15:32 的 1453 大幅上升） |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练成功推进至 step 3750；`steps_3750` 本地已保存于 15:34，正在同步至网络；root overlay 因本地暂存升至 62%；主进程内存因 checkpoint 升至 4.6 GiB；数据 worker 线程数达 1933；GPU 仍接近满载；建议等待同步完成并观察 root overlay 回落 |

## 2026-06-19 — StarFlow Train 会话状态（15:32 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 15:32 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行 15:00:44） |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（已完成） |
| 训练状态 | ✅ **正常运行中**（主进程 `S`/`RLl+`，数据 worker `D`/`Rl+`） |
| 当前步数 | **3625 / 80000**（`steps_3750` 尚未出现） |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 163%（ps）/ 133.3%（top 瞬时） |
| 数据 worker CPU | 25.1%（ps）/ 66.7%（top 瞬时） |
| 主进程内存 | RSS 约 2.0 GiB |
| 数据 worker 内存 | RSS 约 5.4 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39540/40488 MiB（97.6%），功耗 250 W |
| `/root/temp` | run 目录存在，`checkpoints/` 子目录为空 |
| root overlay | **3%（860M / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_3625/`（`steps_3750` 待出现） |
| 负载 | 12.39 / 13.61 / 15.41（1 分钟负载下降） |
| 数据加载 | 数据 worker `D`/`Rl+`，线程数 1453（较 15:28 的 1357 上升） |
| 目标 r2 目录 | 不存在 |
| 备注 | `steps_3625` 已保存 28 分钟，按此前节奏 `steps_3750` 预计即将到达；GPU 持续满载说明计算正常；数据 worker 线程数持续增长至 1453；负载降至 12.39；建议再等 5-10 分钟确认 3750 checkpoint 是否出现 |

## 2026-06-19 — StarFlow Train 会话状态（15:28 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 15:28 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行 14:55:52） |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（已完成） |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `S`/`Dl+`） |
| 当前步数 | **3625 / 80000**（`summary.jsonl` 仅在 checkpoint 时更新） |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 163%（ps）/ 126.7%（top 瞬时） |
| 数据 worker CPU | 25.1%（ps）/ 66.7%（top 瞬时） |
| 主进程内存 | RSS 约 2.0 GiB |
| 数据 worker 内存 | RSS 约 5.4 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39540/40488 MiB（97.6%），功耗 250 W |
| `/root/temp` | run 目录存在，`checkpoints/` 子目录为空 |
| root overlay | **3%（860M / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_3625/` |
| 负载 | 17.45 / 16.25 / 16.66（1 分钟负载回升） |
| 数据加载 | 数据 worker `S`/`Dl+`，线程数 1357（较 15:23 的 1261 上升） |
| 目标 r2 目录 | 不存在 |
| 备注 | `summary.jsonl` 仅在每 125 步 checkpoint 时更新，`steps_3625` 保存于 15:04，预计 `steps_3750` 约 15:34；GPU 持续满载说明训练计算正常；数据 worker 线程数 1357，CPU 瞬时 66.7%；负载回升至 17.45；建议等待至 15:40 确认 3750 checkpoint 是否出现 |

## 2026-06-19 — StarFlow Train 会话状态（15:23 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 15:23 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行 14:50:52） |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（已完成） |
| 训练状态 | ✅ **正常运行中**（主进程 `S`/`RLl+`，数据 worker `D`/`Rl+`） |
| 当前步数 | **3625 / 80000**（连续三次采样未变） |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 163%（ps）/ 133.3%（top 瞬时） |
| 数据 worker CPU | 25.1%（ps）/ 6.7%（top 瞬时） |
| 主进程内存 | RSS 约 2.0 GiB |
| 数据 worker 内存 | RSS 约 5.4 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39540/40488 MiB（97.6%），功耗 250 W |
| `/root/temp` | run 目录存在，`checkpoints/` 子目录为空 |
| root overlay | **3%（860M / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_3625/` |
| 负载 | 9.62 / 14.21 / 16.49（1 分钟负载持续下降） |
| 数据加载 | 数据 worker `D`/`Rl+`，线程数 1261（较 15:18 的 1069 回升） |
| 目标 r2 目录 | 不存在 |
| 备注 | step 仍在 3625，但 GPU 利用率 100%、SM 100%，训练计算未停；`summary.jsonl` 连续 10 分钟未更新，可能处于长 micro-batch 序列、eval 或元数据整理阶段；数据 worker 线程数波动；负载降至 9.62；建议继续观察是否推进至 3750，必要时抓取主进程栈 |

## 2026-06-19 — StarFlow Train 会话状态（15:18 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 15:18 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行 14:46:09） |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（已完成） |
| 训练状态 | ✅ **正常运行中**（主进程 `R`/`SLl+`，数据 worker `Dl+`） |
| 当前步数 | **3625 / 80000**（与 15:13 相同） |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 163%（ps）/ 26.7%（top 瞬时） |
| 数据 worker CPU | 25.1% |
| 主进程内存 | RSS 约 2.0 GiB |
| 数据 worker 内存 | RSS 约 5.7 GiB |
| GPU | `P1.gpu.medium`，SM 利用率 **100%**，显存 39540/40488 MiB（97.6%），功耗 250 W |
| `/root/temp` | run 目录存在，`checkpoints/` 子目录为空 |
| root overlay | **3%（860M / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_3625/` |
| 负载 | 13.74 / 16.46 / 17.58（1 分钟负载较 15:13 回升） |
| 数据加载 | 数据 worker `Dl+`，线程数 1069（较 15:13 的 2317 下降）；`sdb` IO 活跃 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 仍在 3625，但 `nvidia-smi pmon` 显示 GPU SM 100%、显存 97%，说明仍在计算；`summary.jsonl` 未更新可能因采样间隙或数据加载间歇；数据 worker 线程数显著下降；建议继续观察是否推进至 3750 |

## 2026-06-19 — StarFlow Train 会话状态（15:13 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 15:13 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113`（已运行 14:41:14） |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（已完成） |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | **3625 / 80000** |
| 当前 loss | `output.log` 无最新可读值；历史 step 50 的 `action_dit_loss=0.4859` |
| 主进程 CPU | 162% |
| 数据 worker CPU | 25.1% |
| 主进程内存 | RSS 约 2.0 GiB |
| 数据 worker 内存 | RSS 约 5.7 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **3%（860M / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_3625/` |
| 负载 | 11.27 / 14.15 / 17.14（较 12:40 明显下降） |
| 数据加载 | 数据 worker `Dl+`，线程数 2317 |
| 目标 r2 目录 | 不存在 |
| 备注 | 自 12:40 的 step 3000 已推进 625 步至 3625；GPU 利用率从 54% 恢复至 100%，数据加载瓶颈缓解；负载回落至 11.27；主进程内存回归 2.0 GiB 基线；`output.log` 末尾为 06/18 历史 KeyboardInterrupt，当前 06/19 00:32 重启进程已正常运行 |

## 2026-06-19 — StarFlow Train 会话状态（12:40 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 12:40 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（已完成） |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 3000 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 161% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 4.6 GiB（checkpoint 后仍处高位） |
| GPU | `P1.gpu.medium`，利用率 **54%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空（已清理） |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_3000/` |
| 负载 | 20.26 / 18.57 / 17.08（数据加载高峰期） |
| 数据加载 | 数据 worker `Sl+` |
| 目标 r2 目录 | 不存在 |
| 备注 | step 3000 checkpoint 同步完成；sync worker 已退出；`/root/temp` 已清理，root overlay 回落至 4%；但 GPU 骤降至 54%，负载升高至 20.26，显示数据加载成为当前瓶颈；主进程内存仍 4.6 GiB；关注 GPU 恢复及负载回落 |

## 2026-06-19 — StarFlow Train 会话状态（12:36 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 12:36 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `58527`（sync worker，运行中） |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`，sync worker `Rs`） |
| 当前步数 | **3000 / 80000** |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 161% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 4.6 GiB（**checkpoint 保存期间上升**） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/steps_3000/` 占用约 **18G**，正在同步 |
| root overlay | **63%（19G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_3000/` |
| 负载 | 15.07 / 15.27 / 15.89 |
| 数据加载 | 数据 worker `Sl+` |
| 目标 r2 目录 | 不存在 |
| 备注 | step 3000 checkpoint 已触发，sync worker PID 58527 正在将 18G 本地暂存同步至网络；主进程内存升至 4.6 GiB 与 checkpoint 周期一致；GPU 仍维持 100%；负载处于正常区间；关注同步完成及本地清理 |

## 2026-06-19 — StarFlow Train 会话状态（12:31 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 12:31 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2875 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 161% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持在基线） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2875/` |
| 负载 | 21.18 / 18.48 / 16.97（15 分钟均值已开始回落） |
| 数据加载 | 数据 worker `Sl+`，已从 `Dl+` 恢复 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；数据 worker 恢复 `Sl+`，GPU 回升至 100%；主进程内存维持 2.0 GiB 基线；1 分钟负载仍高（21.18），但 15 分钟均值 16.97 显示整体压力在缓解；本地暂存空；summary.jsonl 最新 2875；关注负载完全回落及 step 3000 checkpoint |

## 2026-06-19 — StarFlow Train 会话状态（12:26 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 12:26 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | 2875 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 160% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（**已回落至基线**） |
| GPU | `P1.gpu.medium`，利用率 **90%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2875/` |
| 负载 | 20.32 / 19.85 / 17.05（**数据加载高峰期升高**） |
| 数据加载 | 数据 worker `Dl+`，再次出现不可中断睡眠 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；主进程内存已回落至 2.0 GiB 基线；数据 worker 再次进入 `Dl+`，GPU 降至 90%，负载升高至 20.32，呈现周期性数据加载瓶颈；本地暂存空；summary.jsonl 最新 2875；关注数据 worker 恢复及 GPU 回升 |

## 2026-06-19 — StarFlow Train 会话状态（12:21 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 12:21 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `RLl+`，数据 worker `Sl+`） |
| 当前步数 | 2875 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 160% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 4.6 GiB（checkpoint 后仍处高位） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2875/` |
| 负载 | 15.79 / 15.28 / 14.86 |
| 数据加载 | 数据 worker `Sl+`，已从 `Dl+` 恢复 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；主进程转为 `RLl+` 运行态，数据 worker 恢复 `Sl+`；GPU 维持 100%；主进程内存仍 4.6 GiB，尚未回落至 2.0 GiB 基线；本地暂存空；summary.jsonl 最新 2875；关注内存回落及 step 3000 checkpoint 触发 |

## 2026-06-19 — StarFlow Train 会话状态（12:17 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 12:17 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | 2875 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 160% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 4.6 GiB（checkpoint 后仍处高位） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2875/` |
| 负载 | 14.11 / 13.46 / 14.28 |
| 数据加载 | 数据 worker `Dl+`，再次出现不可中断睡眠 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；数据 worker 再次进入 `Dl+`，但 GPU 仍维持 100%，说明数据加载瓶颈未显著影响计算；主进程内存仍 4.6 GiB，尚未回落至基线；本地暂存空；summary.jsonl 最新 2875；关注内存回落及数据 worker 恢复 |

## 2026-06-19 — StarFlow Train 会话状态（12:12 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 12:12 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（已完成） |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2875 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 160% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 4.6 GiB（checkpoint 后仍处高位） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空（已清理） |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2875/` |
| 负载 | 11.06 / 14.02 / 14.85（**已从峰值回落**） |
| 数据加载 | 数据 worker `Sl+` |
| 目标 r2 目录 | 不存在 |
| 备注 | step 2875 checkpoint 同步完成；sync worker 已退出；`/root/temp` 已清理，root overlay 回落至 4%；负载回落至 11.06；主进程内存仍 4.6 GiB，与之前 checkpoint 后滞留高位模式一致；GPU 100%；关注内存是否回落及 step 3000 准备 |

## 2026-06-19 — StarFlow Train 会话状态（12:07 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 12:07 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `187968`（sync worker，运行中） |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`，sync worker `Ss`） |
| 当前步数 | **2875 / 80000** |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 160% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 4.6 GiB（**checkpoint 保存期间上升**） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/steps_2875/` 占用约 **18G**，正在同步 |
| root overlay | **63%（19G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2875/` |
| 负载 | 20.66 / 16.27 / 15.58（**sync worker 导致冲高**） |
| 数据加载 | 数据 worker `Sl+` |
| 目标 r2 目录 | 不存在 |
| 备注 | step 2875 checkpoint 已触发，sync worker PID 187968 正在将 18G 本地暂存同步至网络；主进程内存升至 4.6 GiB 与 checkpoint 周期一致；GPU 仍维持 100%；root overlay 上升至 63%；关注同步完成及本地清理 |

## 2026-06-19 — StarFlow Train 会话状态（12:02 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 12:02 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2750 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 160% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持在基线） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2750/` |
| 负载 | 12.35 / 14.53 / 15.14 |
| 数据加载 | 数据 worker `Sl+` |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；GPU 维持 100%；负载稳定在 12–15 区间；主进程内存 2.0 GiB 基线；本地暂存空；summary.jsonl 最新 2750；关注 step 2875 checkpoint 触发 |

## 2026-06-19 — StarFlow Train 会话状态（11:58 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 11:58 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Rl+`） |
| 当前步数 | 2750 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 160% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持在基线） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2750/` |
| 负载 | 10.85 / 13.77 / 15.03（**已回落至正常区间**） |
| 数据加载 | 数据 worker `Rl+`，处于运行态 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；数据 worker 转为 `Rl+`，GPU 恢复 100%，负载显著回落至 10.85；主进程内存 2.0 GiB 基线；本地暂存空；summary.jsonl 最新 2750；数据加载瓶颈已解除，关注 step 2875 checkpoint 触发 |

## 2026-06-19 — StarFlow Train 会话状态（11:53 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 11:53 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2750 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 160% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持在基线） |
| GPU | `P1.gpu.medium`，利用率 **92%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2750/` |
| 负载 | 19.04 / 16.76 / 16.17（1 分钟负载再次升高） |
| 数据加载 | 数据 worker `Sl+`，已从 `Dl+` 恢复 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；数据 worker 已恢复 `Sl+`，但 GPU 仍未回到 100%，1 分钟负载升至 19.04；可能处于新一轮数据加载/预处理高峰或接近 checkpoint；主进程内存 2.0 GiB 基线；本地暂存空；summary.jsonl 最新 2750；关注 GPU 恢复及 step 2875 |

## 2026-06-19 — StarFlow Train 会话状态（11:48 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 11:48 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | 2750 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 160% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持在基线） |
| GPU | `P1.gpu.medium`，利用率 **88%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2750/` |
| 负载 | 15.62 / 16.81 / 15.91 |
| 数据加载 | 数据 worker `Dl+`，再次出现不可中断睡眠 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；数据 worker 再次进入 `Dl+`，GPU 利用率降至 88%，提示数据加载阶段性瓶颈；主进程内存 2.0 GiB 基线；本地暂存空；summary.jsonl 最新 2750；关注数据 worker 是否恢复及 GPU 是否回到 100% |

## 2026-06-19 — StarFlow Train 会话状态（11:43 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 11:43 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2750 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 160% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持在基线） |
| GPU | `P1.gpu.medium`，利用率 **94%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2750/` |
| 负载 | 16.04 / 16.46 / 15.50（已从 checkpoint 峰值回落） |
| 数据加载 | 数据 worker `Sl+` |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；负载从 22.28 回落至 16.04；GPU 94% 略降，仍在高位；主进程内存 2.0 GiB 基线；本地暂存空；summary.jsonl 最新 2750；关注 step 2875 checkpoint 触发 |

## 2026-06-19 — StarFlow Train 会话状态（11:39 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 11:39 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | **2750 / 80000** |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 160% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持在基线） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空（11:38 创建，已清理） |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2750/` |
| 负载 | 22.28 / 17.52 / 15.42（**因 checkpoint 保存显著升高**） |
| 数据加载 | 数据 worker `Sl+`，已从 `Dl+` 恢复 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 2750 checkpoint 已保存并同步完成；负载短期冲高至 22.28，属于 checkpoint 周期正常现象；GPU 维持 100%；主进程内存 2.0 GiB 基线；本地暂存已清理；关注负载回落及 step 2875 准备 |

## 2026-06-19 — StarFlow Train 会话状态（11:34 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 11:34 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | 2625 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 160% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持在基线） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2625/` |
| 负载 | 15.17 / 14.97 / 14.24 |
| 数据加载 | 数据 worker `Dl+`，处于不可中断睡眠 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；GPU 维持 100%；1 分钟负载上升至 15.17；数据 worker 进入 `Dl+` 状态，通常是数据读取/同步 IO；主进程内存维持 2.0 GiB 基线；本地暂存空；summary.jsonl 最新 2625；关注 step 2750 checkpoint 触发 |

## 2026-06-19 — StarFlow Train 会话状态（11:31 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 11:31 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2625 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 160% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持在基线） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2625/` |
| 负载 | 14.17 / 13.77 / 13.72 |
| 数据加载 | 数据 worker `Sl+`，线程数待采样 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；GPU 恢复至 100%；负载略有上升至 14.17；主进程内存维持 2.0 GiB 基线；本地暂存空；summary.jsonl 最新 2625；关注 step 2750 checkpoint 触发 |

## 2026-06-19 — StarFlow Train 会话状态（11:24 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 11:24 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `RLl+`，数据 worker `Dl+`） |
| 当前步数 | 2625 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 160% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持在基线） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2625/` |
| 负载 | 11.33 / 12.18 / 13.55 |
| 数据加载 | 数据 worker `Dl+`，线程数 493 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；GPU 维持 100%；数据 worker 线程数从 589 降至 493；负载稳定；主进程内存维持 2.0 GiB 基线；本地暂存空；summary.jsonl 最新 2625；关注 step 2750 触发 |

## 2026-06-19 — StarFlow Train 会话状态（11:19 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 11:19 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2625 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 160% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持在基线） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2625/` |
| 负载 | 9.28 / 12.40 / 14.05 |
| 数据加载 | 数据 worker `Sl+`，线程数 589 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；GPU 利用率从 20% 恢复至 100%；1 分钟负载回落至 9.28；数据 worker 线程数从 1645 降至 589；主进程内存维持 2.0 GiB 基线；本地暂存空；summary.jsonl 最新 2625；关注 step 2750 触发 |

## 2026-06-19 — StarFlow Train 会话状态（11:15 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 11:15 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | 2625 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 160% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（**已回落至基线**） |
| GPU | `P1.gpu.medium`，利用率 **20%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2625/` |
| 负载 | 18.52 / 17.77 / 15.80 |
| 数据加载 | 数据 worker `Dl+`，线程数 1645 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；主进程内存回落至 2.0 GiB 基线；GPU 利用率骤降至 20%，数据加载成为明显瓶颈；1 分钟负载显著上升至 18.52；数据 worker 线程数升至 1645；本地暂存空；summary.jsonl 最新 2625；关注 GPU 恢复及负载回落 |

## 2026-06-19 — StarFlow Train 会话状态（11:10 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 11:10 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `RLl+`，数据 worker `Dl+`） |
| 当前步数 | 2625 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 160% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 4.6 GiB（checkpoint 保存期间上升） |
| GPU | `P1.gpu.medium`，利用率 **98%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空（时间戳 11:09，已清理） |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2625/` |
| 负载 | 12.72 / 15.61 / 14.62 |
| 数据加载 | 数据 worker `Dl+`，线程数 877 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 2625 checkpoint 已保存并同步完成；主进程内存升至 4.6 GiB 与 checkpoint 周期一致；GPU 98%；负载稳定；本地暂存已清理；summary.jsonl 最新 2625；关注内存回落及 step 2750 准备 |

## 2026-06-19 — StarFlow Train 会话状态（11:04:42 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 11:04:42 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2500 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 160% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持在基线） |
| GPU | `P1.gpu.medium`，利用率 **86%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2500/` |
| 负载 | 12.36 / 13.52 / 13.29 |
| 数据加载 | 数据 worker `Sl+`，线程数 685 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；GPU 86%；数据 worker 线程数从 781 降至 685；1 分钟负载从 16.79 回落至 12.36；主进程内存维持 2.0 GiB 基线；本地暂存空；summary.jsonl 最新 2500；关注 step 2625 触发 |

## 2026-06-19 — StarFlow Train 会话状态（11:04 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 11:04 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2500 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 160% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持在基线） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2500/` |
| 负载 | 16.79 / 14.34 / 13.54 |
| 数据加载 | 数据 worker `Sl+`，线程数 781 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；GPU 利用率从 56% 恢复至 100%；数据 worker 线程数从 1741 降至 781；1 分钟负载仍处 16.79 高位；主进程内存维持 2.0 GiB 基线；本地暂存空；summary.jsonl 最新 2500；关注负载回落及 step 2625 触发 |

## 2026-06-19 — StarFlow Train 会话状态（10:59 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 10:59 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2500 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 160% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持在基线） |
| GPU | `P1.gpu.medium`，利用率 **56%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2500/` |
| 负载 | 15.45 / 12.89 / 13.08 |
| 数据加载 | 数据 worker `Sl+`，线程数 1741 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；GPU 利用率降至 56%，可能处于数据等待期或 checkpoint 前准备；1 分钟负载上升至 15.45；数据 worker 线程数升至 1741；主进程内存维持 2.0 GiB；本地暂存空；summary.jsonl 最新 2500；关注 GPU 恢复及 step 2625 触发 |

## 2026-06-19 — StarFlow Train 会话状态（10:54:43 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 10:54:43 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | 2500 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 160% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持在基线） |
| GPU | `P1.gpu.medium`，利用率 **96%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2500/` |
| 负载 | 9.54 / 12.58 / 13.30 |
| 数据加载 | 数据 worker `Dl+`，线程数 1165 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；主进程内存维持 2.0 GiB 基线；GPU 96%；数据 worker 线程数从 2893 降至 1165；负载继续回落至 9.54；本地暂存空；summary.jsonl 最新 2500；关注 step 2625 触发 |

## 2026-06-19 — StarFlow Train 会话状态（10:54 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 10:54 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | 2500 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 160% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（**已回落至基线**） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2500/` |
| 负载 | 12.59 / 13.50 / 13.62 |
| 数据加载 | 数据 worker `Dl+`，线程数 2893 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；主进程内存回落至 2.0 GiB 基线；GPU 维持 100%；数据 worker 线程数大幅升至 2893；负载稳定；本地暂存空；summary.jsonl 最新 2500；关注 step 2625 触发 |

## 2026-06-19 — StarFlow Train 会话状态（10:49:42 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 10:49:42 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2500 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 159% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 4.6 GiB |
| GPU | `P1.gpu.medium`，利用率 **94%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2500/` |
| 负载 | 11.64 / 12.16 / 13.26 |
| 数据加载 | 数据 worker `Sl+`，线程数 1165 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；GPU 94%；数据 worker 线程数从 877 升至 1165；负载略升至 11.64；主进程内存维持 4.6 GiB 已约 10 分钟，视为新常态；本地暂存空；summary.jsonl 最新 2500；关注 step 2625 触发 |

## 2026-06-19 — StarFlow Train 会话状态（10:49 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 10:49 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2500 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 159% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 4.6 GiB（step 2500 后持续维持该水平） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2500/` |
| 负载 | 8.65 / 11.88 / 13.22 |
| 数据加载 | 数据 worker `Sl+`，线程数 877 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；负载已回落至 8.65；GPU 维持 100%；数据 worker 线程数从 1645 降至 877；主进程内存稳定在 4.6 GiB，可能成为新的常态；本地暂存空；summary.jsonl 最新 2500；关注内存长期趋势及 step 2625 触发 |

## 2026-06-19 — StarFlow Train 会话状态（10:44 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 10:44 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `RLl+`，数据 worker `Rl+`） |
| 当前步数 | 2500 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 160% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 4.6 GiB（仍处 checkpoint 保存后高位） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2500/` |
| 负载 | 17.64 / 16.01 / 14.52 |
| 数据加载 | 数据 worker `Rl+`，线程数 1645 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；负载从 21.38 回落至 17.64；GPU 恢复至 100%；数据 worker 线程数从 589 升至 1645；主进程内存仍 4.6 GiB 未回落；本地暂存空；summary.jsonl 最新 2500；关注内存回落及 step 2625 准备 |

## 2026-06-19 — StarFlow Train 会话状态（10:39:47 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 10:39:47 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2500 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 159% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 4.6 GiB（仍处 checkpoint 保存后高位） |
| GPU | `P1.gpu.medium`，利用率 **84%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2500/` |
| 负载 | 21.38 / 15.91 / 14.11 |
| 数据加载 | 数据 worker `Sl+`，线程数 589 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；1 分钟负载显著上升至 21.38（可能为 checkpoint 后 I/O 或数据加载高峰）；GPU 降至 84%；数据 worker 线程数从 1261 降至 589；主进程内存仍 4.6 GiB；本地暂存空；summary.jsonl 最新 2500；关注负载回落、GPU 恢复及内存回落 |

## 2026-06-19 — StarFlow Train 会话状态（10:39 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 10:39 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `RLl+`，数据 worker `Dl+`） |
| 当前步数 | 2500 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 159% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 4.6 GiB（checkpoint 保存期间上升） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空（时间戳 10:38，已清理） |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2500/` |
| 负载 | 12.45 / 13.96 / 13.44 |
| 数据加载 | 数据 worker `Dl+`，线程数 1261 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 2500 checkpoint 已保存并同步完成；主进程内存升至 4.6 GiB 与 checkpoint 周期一致；GPU 维持 100%；负载因 checkpoint 回升至 12.45；本地暂存已清理；summary.jsonl 最新 2500；关注内存回落及 step 2625 准备 |

## 2026-06-19 — StarFlow Train 会话状态（10:34 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 10:34 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2375 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 159% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持在基线） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2375/` |
| 负载 | 9.49 / 12.20 / 12.63 |
| 数据加载 | 数据 worker `Sl+`，线程数 1261 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；GPU 利用率维持 100%；数据 worker 线程数从 973 升至 1261；1 分钟负载继续回落至 9.49；主进程内存稳定在 2.0 GiB 基线；本地暂存空，网络 checkpoint 稳定；summary.jsonl 最新 2375；关注 step 2500 checkpoint 触发 |

## 2026-06-19 — StarFlow Train 会话状态（10:32 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 10:32 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2375 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 159% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持在基线） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2375/` |
| 负载 | 11.64 / 13.45 / 13.03 |
| 数据加载 | 数据 worker `Sl+`，线程数 973 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；GPU 利用率从 80% 回升至 100%；数据 worker 线程数从 1741 降至 973；1 分钟负载回落至 11.64；主进程内存稳定在 2.0 GiB 基线；本地暂存空，网络 checkpoint 稳定；summary.jsonl 最新 2375；关注 step 2500 checkpoint 触发 |

## 2026-06-19 — StarFlow Train 会话状态（10:29 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 10:29 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `RLl+`，数据 worker `Rl+`） |
| 当前步数 | 2375 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 159% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（**已回落至基线**） |
| GPU | `P1.gpu.medium`，利用率 **80%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2375/` |
| 负载 | 17.74 / 14.73 / 13.29 |
| 数据加载 | 数据 worker `Rl+`（disk sleep），线程数 1741 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；主进程内存回落至 2.0 GiB 基线；1 分钟负载上升至 17.74；GPU 80%；数据 worker 线程数从 1261 升至 1741；本地暂存空，网络 checkpoint 稳定；summary.jsonl 最新 2375；关注 GPU 回升及 step 2500 触发 |

## 2026-06-19 — StarFlow Train 会话状态（10:24 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 10:24 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | 2375 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 159% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 4.6 GiB（仍未回落至基线） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2375/` |
| 负载 | 12.64 / 12.61 / 12.41 |
| 数据加载 | 数据 worker `Dl+`，线程数 1261 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；GPU 回升至 100%；数据 worker 线程数从 1549 降至 1261；负载稳定 12.64 / 12.61 / 12.41；主进程内存仍 4.6 GiB 未回落，可能成为新的常态；本地暂存空，网络 checkpoint 稳定；summary.jsonl 最新 2375；关注内存趋势及 step 2500 触发 |

## 2026-06-19 — StarFlow Train 会话状态（10:20 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 10:20 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | 2375 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 159% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 4.6 GiB（仍未回落至基线） |
| GPU | `P1.gpu.medium`，利用率 **72%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2375/` |
| 负载 | 11.06 / 11.80 / 12.18 |
| 数据加载 | 数据 worker `Dl+`，线程数 1549 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；GPU 利用率从 46% 回升至 72%；数据 worker 线程数从 2413 降至 1549；负载回落至 11.06 / 11.80 / 12.18；主进程内存仍 4.6 GiB 未回落；本地暂存空，网络 checkpoint 稳定；summary.jsonl 最新 2375；关注 GPU 进一步回升及内存回落 |

## 2026-06-19 — StarFlow Train 会话状态（10:15 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 10:15 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Rl+`） |
| 当前步数 | 2375 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 159% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 4.6 GiB（仍未回落至基线） |
| GPU | `P1.gpu.medium`，利用率 **46%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2375/` |
| 负载 | 16.33 / 12.60 / 12.40 |
| 数据加载 | 数据 worker `Rl+`，线程数 2413 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；GPU 利用率大幅下降至 46%，数据 worker 线程数升至 2413，数据加载成为明显瓶颈；1 分钟负载上升至 16.33；主进程内存仍 4.6 GiB 未回落；本地暂存空，网络 checkpoint 稳定；summary.jsonl 最新 2375；关注 GPU 回升、线程数回落及内存回落 |

## 2026-06-19 — StarFlow Train 会话状态（10:10 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 10:10 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中，step 2375 同步完成**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | 2375 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 159% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 4.6 GiB（同步完成后尚未回落至基线） |
| GPU | `P1.gpu.medium`，利用率 **88%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录已清空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2375/` |
| 负载 | 9.24 / 12.15 / 12.50 |
| 数据加载 | 数据 worker `Dl+`，线程数 1165 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 2375 同步完成；网络目录已包含 steps_2375；本地暂存清空，root overlay 恢复 4%；主进程内存仍 4.6 GiB 未回落；1 分钟负载降至 9.24；GPU 88%；数据 worker 线程数 1165；summary.jsonl 最新 2375；关注内存回落及 step 2500 触发 |

## 2026-06-19 — StarFlow Train 会话状态（10:05 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 10:05 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中，step 2375 checkpoint 刚触发，本地保存完成**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | 2375 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 159% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 4.6 GiB（因 checkpoint 上升） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `steps_2375/` 约 18G，root overlay 升至 25% |
| root overlay | **25%（7.4G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2250/`（steps_2375 尚未同步） |
| 负载 | 12.27 / 11.91 / 12.46 |
| 数据加载 | 数据 worker `Dl+`（disk sleep），线程数 1165 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 2375 checkpoint 已触发；summary.jsonl 更新至 2375；本地暂存 steps_2375 约 18G；网络同步尚未启动；主进程内存升至 4.6 GiB；GPU 维持 100%；数据 worker 处于 disk sleep；负载正常；关注同步完成与本地清理 |

## 2026-06-19 — StarFlow Train 会话状态（10:01 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 10:01 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2250 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 159% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持基线） |
| GPU | `P1.gpu.medium`，利用率 **86%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2250/` |
| 负载 | 14.05 / 12.58 / 12.92 |
| 数据加载 | 数据 worker `Sl+`（disk sleep），线程数 2509 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；GPU 降至 86%，数据 worker 线程数升至 2509 且存在 disk sleep，数据加载 IO 成为瓶颈；负载稳定；主进程内存维持 2.0 GiB；本地暂存空，网络 checkpoint 稳定；summary.jsonl 最新 2250；关注 GPU 是否回升及 step 2375 触发 |

## 2026-06-19 — StarFlow Train 会话状态（09:56 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 09:56 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2250 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 159% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持基线） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2250/` |
| 负载 | 12.59 / 13.14 / 13.42 |
| 数据加载 | 数据 worker `Sl+`，线程数 973 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；负载回落至 12.59 / 13.14 / 13.42；GPU 回升至 100%；数据 worker 线程数从 2413 降至 973；主进程内存维持 2.0 GiB；本地暂存空，网络 checkpoint 稳定；summary.jsonl 最新 2250；继续向 step 2375 推进 |

## 2026-06-19 — StarFlow Train 会话状态（09:51 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 09:51 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Rl+`） |
| 当前步数 | 2250 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 159% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持基线） |
| GPU | `P1.gpu.medium`，利用率 **74%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2250/` |
| 负载 | 15.02 / 13.98 / 13.70 |
| 数据加载 | 数据 worker `Rl+`，线程数 2413 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；GPU 利用率降至 74%，数据 worker 线程数升至 2413，可能存在数据加载瓶颈；负载稳定 15.02 / 13.98 / 13.70；主进程内存维持 2.0 GiB；本地暂存空，网络 checkpoint 稳定；summary.jsonl 最新 2250；关注 GPU 是否回升及 step 2375 触发 |

## 2026-06-19 — StarFlow Train 会话状态（09:46 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 09:46 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2250 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 159% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持基线） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2250/` |
| 负载 | 15.64 / 14.29 / 13.72 |
| 数据加载 | 数据 worker `Sl+`，线程数 1069 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；负载从 23.32 回落至 15.64；GPU 维持 100%；数据 worker 线程数从 1933 降至 1069；主进程内存维持 2.0 GiB；本地暂存空，网络 checkpoint 稳定；summary.jsonl 最新 2250；继续向 step 2375 推进 |

## 2026-06-19 — StarFlow Train 会话状态（09:42 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 09:42 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中，step 2250 同步完成**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | 2250 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 159% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（已回落至基线） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录已清空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2250/` |
| 负载 | 23.32 / 15.83 / 13.91 |
| 数据加载 | 数据 worker `Dl+`（IO 等待），线程数 1933 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 2250 同步完成；网络目录已包含 steps_2250；本地暂存清空，root overlay 恢复 4%；主进程内存回落至 2.0 GiB；1 分钟负载 spike 至 23.32；GPU 维持 100%；数据 worker 线程数升至 1933；summary.jsonl 最新 2250；继续向 step 2375 推进 |

## 2026-06-19 — StarFlow Train 会话状态（09:37 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 09:37 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（前 defunct worker 已消失） |
| 训练状态 | ✅ **正常运行中，step 2250 checkpoint 刚触发，本地保存完成**（主进程 `RLl+`，数据 worker `Sl+`） |
| 当前步数 | 2250 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 159% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 4.6 GiB（因 checkpoint 上升） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `steps_2250/` 约 18G，root overlay 升至 63% |
| root overlay | **63%（19G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2125/`（steps_2250 尚未同步完成） |
| 负载 | 11.50 / 11.94 / 12.68 |
| 数据加载 | 数据 worker `Sl+`，线程数 2125 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 2250 checkpoint 已触发；summary.jsonl 更新至 2250；本地暂存 steps_2250 约 18G；网络同步尚未完成；主进程内存升至 4.6 GiB；前 defunct 同步 worker 已消失；GPU 维持 100%；负载正常；关注同步完成与本地清理 |

## 2026-06-19 — StarFlow Train 会话状态（09:32 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 09:32 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `59408`（defunct，已存在 27m17s，待回收） |
| 训练状态 | ✅ **正常运行中**（主进程 `RLl+`，数据 worker `Dl+`） |
| 当前步数 | 2125 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 159% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持基线） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2125/` |
| 负载 | 12.29 / 13.05 / 13.27 |
| 数据加载 | 数据 worker `Dl+`（IO 等待），线程数 781 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；负载从 19.52 回落至 12.29；GPU 回升至 100%；数据 worker 线程数从 2413 大幅降至 781；主进程内存维持 2.0 GiB；defunct 同步 worker 仍未回收；summary.jsonl 无新增；继续向 step 2250 推进 |

## 2026-06-19 — StarFlow Train 会话状态（09:27 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 09:27 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `59408`（defunct，已存在 22m32s，待回收） |
| 训练状态 | ✅ **正常运行中**（主进程 `RLl+`，数据 worker `Dl+`） |
| 当前步数 | 2125 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 158% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持基线） |
| GPU | `P1.gpu.medium`，利用率 **96%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2125/` |
| 负载 | 19.52 / 13.72 / 13.55 |
| 数据加载 | 数据 worker `Dl+`（IO 等待），线程数 2413 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；1 分钟负载持续上升至 19.52；数据 worker 线程数从 1645 大幅升至 2413 并进入 `Dl+` IO 等待；GPU 96%；主进程内存维持 2.0 GiB；defunct 同步 worker 仍未回收；summary.jsonl 无新增；继续向 step 2250 推进，关注负载趋势 |

## 2026-06-19 — StarFlow Train 会话状态（09:22 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 09:22 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `59408`（defunct，已存在 17m46s，待回收） |
| 训练状态 | ✅ **正常运行中**（主进程 `RLl+`，数据 worker `Rl+`） |
| 当前步数 | 2125 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 158% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持基线） |
| GPU | `P1.gpu.medium`，利用率 **92%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2125/` |
| 负载 | 17.04 / 14.94 / 14.06 |
| 数据加载 | 数据 worker `Rl+`，线程数 1645 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练正常；主进程内存维持 2.0 GiB 基线；1 分钟负载反弹至 17.04；GPU 92%；数据 worker 进入 `Rl+` 运行态，线程数升至 1645；defunct 同步 worker 仍未回收；summary.jsonl 无新增；继续向 step 2250 推进 |

## 2026-06-19 — StarFlow Train 会话状态（09:18 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 09:18 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `59408`（defunct，待回收） |
| 训练状态 | ✅ **正常运行中，step 2125 同步完成后恢复**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2125 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 158% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（**已回落至基线**） |
| GPU | `P1.gpu.medium`，利用率 **90%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2125/` |
| 负载 | 10.72 / 12.52 / 13.33 |
| 数据加载 | 数据 worker `Sl+`，线程数 1453 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 2125 同步完成后系统完全恢复；主进程内存回落至 2.0 GiB 基线；负载继续降至 10.72；GPU 略降至 90%；数据 worker 线程数从 781 升至 1453；defunct 同步 worker 待回收；继续向 step 2250 推进 |

## 2026-06-19 — StarFlow Train 会话状态（09:13 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 09:13 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `59408`（已 defunct，同步完成） |
| 训练状态 | ✅ **正常运行中，step 2125 同步完成**（主进程 `RLl+`，数据 worker `Dl+`） |
| 当前步数 | 2125 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 158% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 4.6 GiB（尚未回落至基线） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录已清空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2125/` |
| 负载 | 12.68 / 12.99 / 13.65 |
| 数据加载 | 数据 worker `Dl+`（IO 等待），线程数 781 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 2125 同步完成；同步 worker 已 defunct 待回收；`/root/temp` 清空；root overlay 恢复至 4%；负载回落至正常区间；主进程内存仍 4.6 GiB 未回落；GPU 维持 100%；数据 worker 线程数从 1933 降至 781；继续向 step 2250 推进 |

## 2026-06-19 — StarFlow Train 会话状态（09:08 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 09:08 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `59408` |
| 训练状态 | ✅ **正常运行中，step 2125 checkpoint 同步中**（主进程 `SLl+`，数据 worker `Dl+`，同步 worker `R`） |
| 当前步数 | 2125 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 158% |
| 数据worker CPU | 25.1% |
| 同步 worker CPU | 35.4% |
| 主进程内存 | 约 4.6 GiB（因 checkpoint 上升） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `steps_2125/` 约 18G，root overlay 升至 63% |
| root overlay | **63%（19G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2125/` |
| 负载 | 14.50 / 14.16 / 14.15 |
| 数据加载 | 数据 worker `Dl+`（IO 等待），线程数 1933 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 2125 checkpoint 已触发；同步 worker 正在将 18G 本地暂存复制到网络；主进程内存升至 4.6 GiB；数据 worker 因 IO 等待进入 Dl+；summary.jsonl 已更新至 2125；关注同步完成后本地清理与内存回落 |

## 2026-06-19 — StarFlow Train 会话状态（09:03 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 09:03 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2000 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 158% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持基线） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2000/` |
| 负载 | 15.48 / 13.35 / 14.04 |
| 数据加载 | 数据 worker `Sl+`，线程数 1165 |
| 目标 r2 目录 | 不存在 |
| 备注 | 1 分钟负载从 7.50 略反弹至 15.48；GPU 维持 100%；数据 worker 线程数略升至 1165；主进程内存维持 2.0 GiB 基线；summary.jsonl 无新增；向 step 2125 推进 |

## 2026-06-19 — StarFlow Train 会话状态（09:01 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 09:01 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `R`，数据 worker `S`） |
| 当前步数 | 2000 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 158% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（维持基线） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2000/` |
| 负载 | 7.50 / 11.94 / 13.76 |
| 数据加载 | 数据 worker `S`，线程数 1069 |
| 目标 r2 目录 | 不存在 |
| 备注 | 负载进一步回落至 7.50；GPU 利用率回升至 100%；数据 worker 线程数从 2030 降至 1069；主进程内存维持 2.0 GiB 基线；summary.jsonl 无新增；向 step 2125 推进 |

## 2026-06-19 — StarFlow Train 会话状态（08:54 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 08:54 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2000 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 158% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 4.6 GiB（持续未回落） |
| GPU | `P1.gpu.medium`，利用率 **92%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2000/` |
| 负载 | 23.28 / 19.33 / 16.05 |
| 数据加载 | 数据 worker `Sl+`，线程数 494 |
| 目标 r2 目录 | 不存在 |
| 备注 | 1 分钟负载再次升高至 23.28，5/15 分钟负载也上升；数据 worker 线程数从 1550 降至 494；GPU 升至 92%；主进程内存仍维持 4.6 GiB；summary.jsonl 无新增；需关注负载趋势 |

## 2026-06-19 — StarFlow Train 会话状态（08:49 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 08:49 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2000 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 158% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 4.6 GiB（自 step 2000 后未回落） |
| GPU | `P1.gpu.medium`，利用率 **80%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2000/` |
| 负载 | 16.48 / 14.11 / 13.91 |
| 数据加载 | 数据 worker `Sl+`，线程数 1550 |
| 目标 r2 目录 | 不存在 |
| 备注 | 主进程内存维持 4.6 GiB 未回落；GPU 80%；1 分钟负载略升至 16.48；数据 worker 线程数维持 1550；summary.jsonl 无新增；step 2125 预计即将触发 |

## 2026-06-19 — StarFlow Train 会话状态（08:44 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 08:44 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2000 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 158% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 4.6 GiB |
| GPU | `P1.gpu.medium`，利用率 **78%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2000/` |
| 负载 | 12.91 / 15.17 / 14.23 |
| 数据加载 | 数据 worker `Sl+`，线程数 1550 |
| 目标 r2 目录 | 不存在 |
| 备注 | 1 分钟负载从 27.60 回落至正常区间 12.91；GPU 降至 78%；主进程内存仍维持 4.6 GiB；数据 worker 线程数维持 1550；summary.jsonl 无新增；向 step 2125 推进 |

## 2026-06-19 — StarFlow Train 会话状态（08:40 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 08:40 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中，step 2000 已同步完成**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2000 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 158% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 4.6 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录已清空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_2000/`（step 2000 已同步完成） |
| 负载 | **27.60 / 17.51 / 14.29** |
| 数据加载 | 数据 worker `Sl+`，线程数 1550 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 2000 本地保存+网络同步+本地清理完成；网络盘已出现 `steps_2000/`；1 分钟负载显著飙升至 27.60，需关注是否持续；主进程内存仍维持 4.6 GiB；数据 worker 线程数升至 1550；GPU 100%；向 step 2125 推进 |

## 2026-06-19 — StarFlow Train 会话状态（08:35 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 08:35 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（保存中） |
| 训练状态 | ✅ **正常运行中，step 2000 checkpoint 保存刚开始**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 2000 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 157% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 4.6 GiB（从 2.0 GiB 回升） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/steps_2000/` 刚创建（08:35） |
| root overlay | **4%（1.1G / 30G）**（暂存刚开始，尚未显著增长） |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_1875/`（`steps_2000/` 尚未同步） |
| 负载 | 16.13 / 12.83 / 12.58 |
| 数据加载 | 数据 worker `Sl+`，线程数 878 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 2000 保存刚开始；`summary.jsonl` 已更新至 2000；主进程内存回升至 4.6 GiB；`/root/temp` 刚出现 `steps_2000/`；root overlay 仍为 4%；GPU 100%；需观察暂存增长、同步完成与本地清理 |

## 2026-06-19 — StarFlow Train 会话状态（08:30 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 08:30 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 1875 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 158% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_1875/` |
| 负载 | 11.56 / 12.59 / 12.72 |
| 数据加载 | 数据 worker `Sl+`，线程数 1550 |
| 目标 r2 目录 | 不存在 |
| 备注 | 主进程回到 `SLl+`，worker 回到 `Sl+`；数据 worker 线程数升至 1550；GPU 100%；1 分钟负载从 16.32 回落至 11.56；主进程内存维持 2.0 GiB；summary.jsonl 无新增；step 2000 预计即将触发 |

## 2026-06-19 — StarFlow Train 会话状态（08:25 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 08:25 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `RLl+`，数据 worker `Dl+`） |
| 当前步数 | 1875 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 158% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_1875/` |
| 负载 | 16.32 / 13.16 / 12.90 |
| 数据加载 | 数据 worker `Dl+`，线程数 1358 |
| 目标 r2 目录 | 不存在 |
| 备注 | 主进程 CPU 微升至 158%；1 分钟负载从 9.99 回升至 16.32；GPU 100%；主进程内存维持 2.0 GiB；数据 worker 线程数维持 1358；summary.jsonl 无新增；step 2000 预计即将触发 |

## 2026-06-19 — StarFlow Train 会话状态（08:21 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 08:21 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `RLl+`，数据 worker `Dl+`） |
| 当前步数 | 1875 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 157% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_1875/` |
| 负载 | 9.99 / 12.65 / 12.98 |
| 数据加载 | 数据 worker `Dl+`，线程数 1358 |
| 目标 r2 目录 | 不存在 |
| 备注 | 主进程 `RLl+` 但内存维持 2.0 GiB；数据 worker 线程数升至 1358；GPU 100%；1 分钟负载降至 9.99；summary.jsonl 无新增；step 2000 预计不久后触发 |

## 2026-06-19 — StarFlow Train 会话状态（08:16 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 08:16 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | 1875 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 157% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB |
| GPU | `P1.gpu.medium`，利用率 **90%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_1875/` |
| 负载 | 14.36 / 12.75 / 12.98 |
| 数据加载 | 数据 worker `Dl+`，线程数 206 |
| 目标 r2 目录 | 不存在 |
| 备注 | 主进程回到 `SLl+`，内存维持 2.0 GiB；数据 worker 线程数骤降至 206；GPU 90%；负载稳定；summary.jsonl 无新增；向 step 2000 推进 |

## 2026-06-19 — StarFlow Train 会话状态（08:11 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 08:11 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中，step 1875 已同步完成**（主进程 `RLl+`，数据 worker `Dl+`） |
| 当前步数 | 1875 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 157% |
| 数据worker CPU | 25.1% |
| 主进程内存 | 约 2.0 GiB（从 4.6 GiB 回落至基线） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录已清空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_1875/`（step 1875 已同步完成） |
| 负载 | 9.53 / 12.60 / 13.18 |
| 数据加载 | 数据 worker `Dl+`，线程数 590 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 1875 本地保存+网络同步+本地清理完成；网络盘已出现 `steps_1875/`；主进程内存回落至 2.0 GiB 基线；数据 worker 线程数降至 590；GPU 100%；主进程状态 `RLl+`；向 step 2000 推进 |

## 2026-06-19 — StarFlow Train 会话状态（08:06 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 08:06 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（保存中） |
| 训练状态 | ✅ **正常运行中，step 1875 checkpoint 保存中**（主进程 `SLl+`，数据 worker `Rl+`） |
| 当前步数 | 1875 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 157% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 4.6 GiB（从 2.0 GiB 回升） |
| GPU | `P1.gpu.medium`，利用率 **64%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/steps_1875/` 本地暂存中 |
| root overlay | **63%（19G / 30G）**（checkpoint 暂存导致急剧上升） |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_1750/`（`steps_1875/` 尚未同步） |
| 负载 | 13.86 / 14.17 / 13.67 |
| 数据加载 | 数据 worker `Rl+`，线程数 1166 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 1875 保存中；`summary.jsonl` 已更新至 1875；主进程内存回升至 4.6 GiB；root overlay 从 4% 升至 63%；GPU 降至 64%；数据 worker `Rl+`；需观察同步完成与本地清理 |

## 2026-06-19 — StarFlow Train 会话状态（08:02 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 08:02 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 1750 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 157% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.0 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_1750/` |
| 负载 | 17.38 / 14.75 / 13.71 |
| 数据加载 | 数据 worker `Sl+`，线程数 1358 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 回到 100%；数据 worker 线程数降至 1358；1 分钟负载升至 17.38；主进程内存维持 2.0 GiB；summary.jsonl 无新增；step 1875 尚未触发 |

## 2026-06-19 — StarFlow Train 会话状态（07:57 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 07:57 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 1750 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 157% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.0 GiB |
| GPU | `P1.gpu.medium`，利用率 **72%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_1750/` |
| 负载 | 13.50 / 13.33 / 13.14 |
| 数据加载 | 数据 worker `Sl+`，线程数 2126 |
| 目标 r2 目录 | 不存在 |
| 备注 | 数据 worker 从 `Dl+` 回到 `Sl+`，线程数维持 2126；GPU 降至 72%；负载稳定；主进程内存维持 2.0 GiB；summary.jsonl 无新增；step 1875 尚未触发 |

## 2026-06-19 — StarFlow Train 会话状态（07:52 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 07:52 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | 1750 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 157% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.0 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_1750/` |
| 负载 | 14.00 / 12.71 / 12.74 |
| 数据加载 | 数据 worker `Dl+`，线程数 2222 |
| 目标 r2 目录 | 不存在 |
| 备注 | 数据 worker 线程数从 878 升至 2222；GPU 维持 100%；负载稳定；主进程内存维持 2.0 GiB；summary.jsonl 无新增；step 1875 checkpoint 预计即将触发 |

## 2026-06-19 — StarFlow Train 会话状态（07:47 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 07:47 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | 1750 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 157% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.0 GiB（从 4.6 GiB 回落至基线） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_1750/` |
| 负载 | 15.38 / 13.08 / 13.04 |
| 数据加载 | 数据 worker `Dl+`，线程数 878 |
| 目标 r2 目录 | 不存在 |
| 备注 | 主进程内存回落至 2.0 GiB 基线；GPU 回到 100%；数据 worker 线程数降至 878；负载稳定；summary.jsonl 无新增；向 step 1875 推进 |

## 2026-06-19 — StarFlow Train 会话状态（07:42 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 07:42 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中，step 1750 已同步完成**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | 1750 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 157% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 4.6 GiB |
| GPU | `P1.gpu.medium`，利用率 **92%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录已清空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_1750/`（step 1750 已同步完成） |
| 负载 | 15.18 / 13.45 / 13.22 |
| 数据加载 | 数据 worker `Dl+`，线程数 1166 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 1750 本地保存+网络同步+本地清理完成；网络盘已出现 `steps_1750/`；主进程状态回到 `SLl+`；数据 worker 线程数降至 1166；主进程内存仍维持 4.6 GiB；向 step 1875 推进 |

## 2026-06-19 — StarFlow Train 会话状态（07:38 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 07:38 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（保存中） |
| 训练状态 | ✅ **正常运行中，step 1750 checkpoint 保存中**（主进程 `RLl+`，数据 worker `Rl+`） |
| 当前步数 | 1750 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 157% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 4.6 GiB（从 2.0 GiB 回升） |
| GPU | `P1.gpu.medium`，利用率 **76%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/steps_1750/` 本地暂存中 |
| root overlay | **63%（19G / 30G）**（checkpoint 暂存导致急剧上升） |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_1625/`（`steps_1750/` 尚未同步） |
| 负载 | 10.35 / 11.77 / 12.72 |
| 数据加载 | 数据 worker `Rl+`，线程数 1646 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 1750 保存中；`summary.jsonl` 已更新至 1750；主进程状态 `RLl+`、worker `Rl+`；root overlay 从 4% 升至 63%；主进程内存回升至 4.6 GiB；GPU 降至 76%；需观察同步完成与本地清理 |

## 2026-06-19 — StarFlow Train 会话状态（07:33 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 07:33 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | 1625 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 157% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.0 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_1625/` |
| 负载 | 11.98 / 12.32 / 13.17 |
| 数据加载 | 数据 worker `Dl+`，线程数 2030 |
| 目标 r2 目录 | 不存在 |
| 备注 | 数据 worker 线程数升至 2030；GPU 回到 100%；负载继续回落至 12 左右；主进程内存维持 2.0 GiB；summary.jsonl 无新增；step 1750 checkpoint 预计即将触发 |

## 2026-06-19 — StarFlow Train 会话状态（07:28 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 07:28 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | 1625 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 157% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.0 GiB |
| GPU | `P1.gpu.medium`，利用率 **90%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_1625/` |
| 负载 | 13.84 / 13.89 / 13.87 |
| 数据加载 | 数据 worker `Dl+`，线程数 1262 |
| 目标 r2 目录 | 不存在 |
| 备注 | 负载趋于平稳；GPU 90%；主进程内存维持 2.0 GiB 基线；数据 worker 线程数 1262；summary.jsonl 无新增；预计 step 1750 checkpoint 将在未来几次监控内触发 |

## 2026-06-19 — StarFlow Train 会话状态（07:23 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 07:23 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | 1625 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 157% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.0 GiB（从 4.6 GiB 回落至基线） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_1625/` |
| 负载 | 20.15 / 15.82 / 14.31 |
| 数据加载 | 数据 worker `Dl+`，线程数 1454 |
| 目标 r2 目录 | 不存在 |
| 备注 | 主进程内存回落至 2.0 GiB 基线；数据 worker 回到 `Dl+`，线程数 1454；1 分钟负载显著升至 20.15；GPU 回到 100%；summary.jsonl 无新增；接近 step 1750 |

## 2026-06-19 — StarFlow Train 会话状态（07:19 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 07:19 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 1625 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 157% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 4.6 GiB |
| GPU | `P1.gpu.medium`，利用率 **82%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_1625/` |
| 负载 | 8.12 / 12.85 / 13.23 |
| 数据加载 | 数据 worker `Sl+`，线程数 1358 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 利用率从 100% 降至 82%；1 分钟负载从 15.77 降至 8.12；数据 worker 线程数降至 1358；主进程内存维持 4.6 GiB；summary.jsonl 无新增；向 step 1750 推进 |

## 2026-06-19 — StarFlow Train 会话状态（07:14 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 07:14 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 1625 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 157% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 4.6 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录为空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_1625/` |
| 负载 | 15.77 / 15.50 / 13.77 |
| 数据加载 | 数据 worker `Sl+`，线程数 1646 |
| 目标 r2 目录 | 不存在 |
| 备注 | 训练处于 checkpoint 间隔正常阶段；数据 worker 线程数从 302 回升至 1646；负载略有上升；GPU 100%；主进程内存维持 4.6 GiB；向 step 1750 推进 |

## 2026-06-19 — StarFlow Train 会话状态（07:09 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 07:09 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中，step 1625 已同步完成**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 1625 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 157% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 4.6 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录已清空 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_1625/`（step 1625 已同步完成） |
| 负载 | 15.61 / 12.64 / 12.48 |
| 数据加载 | 数据 worker `Sl+`，线程数 302 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 1625 本地保存+网络同步+本地清理完成；网络盘已出现 `steps_1625/`；数据 worker 线程数降至 302；GPU 100%；主进程内存维持 4.6 GiB；向 step 1750 推进 |

## 2026-06-19 — StarFlow Train 会话状态（07:04 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 07:04 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（保存中） |
| 训练状态 | ✅ **正常运行中，step 1625 checkpoint 保存中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | 1625 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 157% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 4.6 GiB（从 2.0 GiB 回升） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/steps_1625/` 本地暂存中 |
| root overlay | **53%（16G / 30G）**（checkpoint 暂存导致急剧上升） |
| 网络 checkpoint | `checkpoints/steps_125/` 至 `steps_1500/`（同步完成），`steps_1625/` 尚未同步 |
| 负载 | 15.96 / 12.57 / 12.50 |
| 数据加载 | 数据 worker `Sl+`，线程数 1262 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 1625 保存中；`summary.jsonl` 已更新至 1625；root overlay 从 4% 升至 53%；主进程内存回升至 4.6 GiB；需观察同步完成与本地清理 |

## 2026-06-19 — StarFlow Train 会话状态（07:00 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 07:00 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | >1500 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 157% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.0 GiB（较 4.6 GiB 明显下降） |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | `checkpoints/` 子目录已清空，本地副本释放 |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` + `checkpoints/steps_250/` + `checkpoints/steps_375/` + `checkpoints/steps_500/` + `checkpoints/steps_625/` + `checkpoints/steps_750/` + `checkpoints/steps_875/` + `checkpoints/steps_1000/` + `checkpoints/steps_1125/` + `checkpoints/steps_1250/` + `checkpoints/steps_1375/` + `checkpoints/steps_1500/`（同步完成） |
| 负载 | 6.53 / 10.91 / 12.32 |
| 数据加载 | 数据 worker `Sl+`，线程数 1646 |
| 目标 r2 目录 | 不存在 |
| 备注 | 主进程内存从 4.6 GiB 回落至 2.0 GiB；`/root/temp/checkpoints/` 已清空；GPU 回到 100%；1 分钟负载大幅降至 6.53；summary.jsonl 仍无新增；接近 step 1625 |

## 2026-06-19 — StarFlow Train 会话状态（06:55 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 06:55 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | >1500 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 156% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 4.6 GiB |
| GPU | `P1.gpu.medium`，利用率 **86%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | 约 16K 目录，保留 `checkpoints/steps_1500/` |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` + `checkpoints/steps_250/` + `checkpoints/steps_375/` + `checkpoints/steps_500/` + `checkpoints/steps_625/` + `checkpoints/steps_750/` + `checkpoints/steps_875/` + `checkpoints/steps_1000/` + `checkpoints/steps_1125/` + `checkpoints/steps_1250/` + `checkpoints/steps_1375/` + `checkpoints/steps_1500/`（同步完成） |
| 负载 | 11.99 / 14.18 / 13.35 |
| 数据加载 | 数据 worker `Sl+`，线程数 1646 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 86%；数据 worker 状态从 `Dl+` 回到 `Sl+`，线程数降至 1646；1 分钟负载从 18.80 回落至 11.99；内存稳定；summary.jsonl 无新增；接近 step 1625 |

## 2026-06-19 — StarFlow Train 会话状态（06:53 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 06:53 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | >1500 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 156% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 4.6 GiB |
| GPU | `P1.gpu.medium`，利用率 **94%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | 约 20K（保留 run 目录 config/summary 及 `checkpoints/steps_1500/`） |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `checkpoints/steps_125/` + `checkpoints/steps_250/` + `checkpoints/steps_375/` + `checkpoints/steps_500/` + `checkpoints/steps_625/` + `checkpoints/steps_750/` + `checkpoints/steps_875/` + `checkpoints/steps_1000/` + `checkpoints/steps_1125/` + `checkpoints/steps_1250/` + `checkpoints/steps_1375/` + `checkpoints/steps_1500/`（同步完成） |
| 负载 | 18.80 / 14.55 / 13.28 |
| 数据加载 | 数据 worker `Dl+`，线程数 2318 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 利用率回升至 94%；数据 worker 线程数升至 2318；1 分钟负载显著升至 18.80；主进程内存维持 4.6 GiB；接近 step 1625 |

## 2026-06-19 — StarFlow Train 会话状态（06:50 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 06:50 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | >1500 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 156% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 4.6 GiB |
| GPU | `P1.gpu.medium`，利用率 **58%**，SM **54%–100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/` + `steps_1125/` + `steps_1250/` + `steps_1375/` + `steps_1500/`（同步完成） |
| 负载 | 15.37 / 12.93 / 12.67 |
| 数据加载 | 数据 worker `Dl+`，线程数 1166 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 采样 58% 但 SM 54%–100%；数据 worker 线程数升至 1166；1 分钟负载升至 15.37；接近 step 1625 |

## 2026-06-19 — StarFlow Train 会话状态（06:46 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 06:46 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | >1500 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 156% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 4.6 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **60%–100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/` + `steps_1125/` + `steps_1250/` + `steps_1375/` + `steps_1500/`（同步完成） |
| 负载 | 12.87 / 12.56 / 12.60 |
| 数据加载 | 数据 worker `Sl+`，线程数 494 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 100%；数据 worker 线程数降至 494；主进程内存维持 4.6 GiB；接近 step 1625 |

## 2026-06-19 — StarFlow Train 会话状态（06:41 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 06:41 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（同步完成） |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | >1500 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 156% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 4.6 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **34%–88%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/` + `steps_1125/` + `steps_1250/` + `steps_1375/` + `steps_1500/`（同步完成） |
| 负载 | 12.09 / 12.53 / 12.58 |
| 数据加载 | 数据 worker `Dl+`，线程数 1262 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 1500 已同步完成；本地暂存清理；主进程内存维持 4.6 GiB；GPU 100% 但 SM 34%–88%；负载回落 |

## 2026-06-19 — StarFlow Train 会话状态（06:36 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 06:36 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `137899`（活跃同步 steps_1500） |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`，同步 worker `Ss`） |
| 当前步数 | **1500 / 80000** |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 156% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 4.6 GiB（较上次回升） |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **62%–94%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | **18G**（含 `steps_1500` 暂存） |
| root overlay | **63%（19G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/` + `steps_1125/` + `steps_1250/` + `steps_1375/`（`steps_1500` 同步中） |
| 负载 | 13.07 / 12.87 / 12.66 |
| 数据加载 | 数据 worker `Dl+`，线程数 878 |
| 目标 r2 目录 | 不存在 |
| 备注 | 到达 step 1500，本地 checkpoint 18G 正在同步；GPU 100%；主进程内存回升至 4.6 GiB |

## 2026-06-19 — StarFlow Train 会话状态（06:32 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 06:32 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | >1375 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 156% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.0 GiB |
| GPU | `P1.gpu.medium`，利用率 **92%**，SM **88%–100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/` + `steps_1125/` + `steps_1250/` + `steps_1375/`（同步完成） |
| 负载 | 6.87 / 10.30 / 11.93 |
| 数据加载 | 数据 worker `Dl+`，线程数 1166 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 92%、SM 88%–100%；1 分钟负载显著下降至 6.87；数据 worker 线程数 1166；接近 step 1500 |

## 2026-06-19 — StarFlow Train 会话状态（06:27 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 06:27 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `RLl+`，数据 worker `Rl+`） |
| 当前步数 | >1375 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 156% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.0 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **40%–100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/` + `steps_1125/` + `steps_1250/` + `steps_1375/`（同步完成） |
| 负载 | 14.37 / 13.01 / 13.00 |
| 数据加载 | 数据 worker `Rl+`，线程数 974 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 100% 但 SM 40%–100%；数据 worker 状态变为 `Rl+`，线程数降至 974；主进程内存稳定 2.0 GiB；接近 step 1500 |

## 2026-06-19 — StarFlow Train 会话状态（06:22 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 06:22 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | >1375 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 155% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.0 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **74%–100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/` + `steps_1125/` + `steps_1250/` + `steps_1375/`（同步完成） |
| 负载 | 13.04 / 12.87 / 13.15 |
| 数据加载 | 数据 worker `Sl+`，线程数 2126 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 恢复 100%；数据 worker 线程数升至 2126；主进程内存稳定 2.0 GiB；接近 step 1500 |

## 2026-06-19 — StarFlow Train 会话状态（06:17 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 06:17 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | >1375 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 155% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.0 GiB |
| GPU | `P1.gpu.medium`，利用率 **36%**，SM **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/` + `steps_1125/` + `steps_1250/` + `steps_1375/`（同步完成） |
| 负载 | 12.68 / 12.70 / 13.19 |
| 数据加载 | 数据 worker `Dl+`，线程数 878 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 采样 36% 但 SM 100%；数据 worker 线程数降至 878；负载回落至 12.68；接近 step 1500 |

## 2026-06-19 — StarFlow Train 会话状态（06:12 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 06:12 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（同步完成） |
| 训练状态 | ✅ **正常运行中**（主进程 `RLl+`，数据 worker `Sl+`） |
| 当前步数 | >1375 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 155% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.0 GiB（较上次下降） |
| GPU | `P1.gpu.medium`，利用率 **70%**，SM **100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/` + `steps_1125/` + `steps_1250/` + `steps_1375/`（同步完成） |
| 负载 | 18.31 / 14.43 / 13.73 |
| 数据加载 | 数据 worker `Sl+`，线程数 1550 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 1375 已同步完成；本地暂存清理；主进程内存回落至 2.0 GiB；GPU 采样 70% 但 SM 100%；负载 18.31 |

## 2026-06-19 — StarFlow Train 会话状态（06:07 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 06:07 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `21970`（活跃同步 steps_1375） |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`，同步 worker `Rs`） |
| 当前步数 | **1375 / 80000** |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 155% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 4.6 GiB（较上次回升） |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **92%–100%**，显存 39540/40488 MiB（97.6%） |
| `/root/temp` | **18G**（含 `steps_1375` 暂存） |
| root overlay | **63%（19G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/` + `steps_1125/` + `steps_1250/`（`steps_1375` 同步中） |
| 负载 | 17.61 / 14.45 / 13.72 |
| 数据加载 | 数据 worker `Sl+`，线程数 1934 |
| 目标 r2 目录 | 不存在 |
| 备注 | 到达 step 1375，本地 checkpoint 18G 正在同步；GPU 满载；主进程内存回升至 4.6 GiB |

## 2026-06-19 — StarFlow Train 会话状态（06:03 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 06:03 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `RLl+`，数据 worker `Dl+`） |
| 当前步数 | >1250 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 155% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.0 GiB |
| GPU | `P1.gpu.medium`，利用率 **94%**，SM **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/` + `steps_1125/` + `steps_1250/`（同步完成） |
| 负载 | 17.17 / 13.93 / 13.58 |
| 数据加载 | 数据 worker `Dl+`，线程数 1454 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 94%、SM 100%；数据 worker 状态变为 `Dl+`，线程数 1454；1 分钟负载升至 17.17；接近 step 1375 |

## 2026-06-19 — StarFlow Train 会话状态（05:58 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 05:58 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | >1250 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 154% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.0 GiB |
| GPU | `P1.gpu.medium`，利用率 **90%**，SM **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/` + `steps_1125/` + `steps_1250/`（同步完成） |
| 负载 | 13.76 / 14.07 / 13.66 |
| 数据加载 | 数据 worker `Sl+`，线程数 2030 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 采样 90% 但 SM 100%；数据 worker 线程数降至 2030；主进程内存稳定 2.0 GiB；接近 step 1375 |

## 2026-06-19 — StarFlow Train 会话状态（05:53 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 05:53 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | >1250 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 154% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.0 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **80%–94%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/` + `steps_1125/` + `steps_1250/`（同步完成） |
| 负载 | 14.18 / 14.32 / 13.56 |
| 数据加载 | 数据 worker `Sl+`，线程数 2318 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 100%；数据 worker 线程数再创新高 2318；主进程内存稳定 2.0 GiB；接近 step 1375 |

## 2026-06-19 — StarFlow Train 会话状态（05:49 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 05:49 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `RLl+`，数据 worker `Sl+`） |
| 当前步数 | >1250 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 154% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.0 GiB（较上次下降） |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/` + `steps_1125/` + `steps_1250/`（同步完成） |
| 负载 | 14.62 / 12.99 / 12.97 |
| 数据加载 | 数据 worker `Sl+`，线程数 1454 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 满载；主进程内存回落至 2.0 GiB；数据 worker 线程数降至 1454；接近 step 1375 |

## 2026-06-19 — StarFlow Train 会话状态（05:44 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 05:44 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `RLl+`，数据 worker `Sl+`） |
| 当前步数 | >1250 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 154% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 4.7 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **86%–100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/` + `steps_1125/` + `steps_1250/`（同步完成） |
| 负载 | 10.28 / 11.48 / 12.48 |
| 数据加载 | 数据 worker `Sl+`，线程数 2222 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 100%；数据 worker 线程数升至 2222；主进程内存稳定 4.7 GiB；负载回落；接近 step 1375 |

## 2026-06-19 — StarFlow Train 会话状态（05:39 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 05:39 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（step 1250 已同步完成） |
| 训练状态 | ✅ **正常运行中**（主进程 `RLl+`，数据 worker `Sl+`） |
| 当前步数 | **1250 / 80000** |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 154% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 4.7 GiB（较上次回升） |
| GPU | `P1.gpu.medium`，利用率 **82%**，SM **88%–100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/` + `steps_1125/` + `steps_1250/`（同步完成） |
| 负载 | 13.36 / 14.31 / 13.47 |
| 数据加载 | 数据 worker `Sl+`，线程数 1550 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 1250 已保存并同步完成；主进程内存回升至 4.7 GiB；数据 worker 线程数 1550；GPU 采样 82% 但 SM 88%–100% |

## 2026-06-19 — StarFlow Train 会话状态（05:34 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 05:34 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | >1125 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 154% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.0 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **52%–98%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/` + `steps_1125/`（同步完成） |
| 负载 | 10.56 / 12.42 / 12.75 |
| 数据加载 | 数据 worker `Sl+`，线程数 686 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 100%；数据 worker 状态变为 `Sl+`，线程数降至 686；主进程内存稳定 2.0 GiB；接近 step 1250 |

## 2026-06-19 — StarFlow Train 会话状态（05:30 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 05:30 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `RLl+`，数据 worker `Dl+`） |
| 当前步数 | >1125 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 153% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.0 GiB |
| GPU | `P1.gpu.medium`，利用率 **94%**，SM **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/` + `steps_1125/`（同步完成） |
| 负载 | 10.90 / 11.93 / 12.60 |
| 数据加载 | 数据 worker `Dl+`，线程数 2030 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 恢复满载；数据 worker 线程数回升至 2030；主进程内存稳定在 2.0 GiB；接近 step 1250 |

## 2026-06-19 — StarFlow Train 会话状态（05:25 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 05:25 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | >1125 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 153% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.0 GiB（较上次明显下降） |
| GPU | `P1.gpu.medium`，利用率 **62%**，SM **72%–100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/` + `steps_1125/`（同步完成） |
| 负载 | 8.23 / 11.46 / 12.66 |
| 数据加载 | 数据 worker `Dl+`，线程数 974 |
| 目标 r2 目录 | 不存在 |
| 备注 | 主进程内存降至 2.0 GiB；数据 worker 线程数降至 974；负载显著下降至 8.23；GPU 采样 62% 但 SM 72%–100%；接近 step 1250 |

## 2026-06-19 — StarFlow Train 会话状态（05:20 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 05:20 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | >1125 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 153% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 4.7 GiB |
| GPU | `P1.gpu.medium`，利用率 **92%**，SM **72%–100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/` + `steps_1125/`（同步完成） |
| 负载 | 12.33 / 13.60 / 13.51 |
| 数据加载 | 数据 worker `Dl+`，线程数 2030 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 92%、SM 72%–100%；负载回落至 12.33；数据 worker 线程数 2030；接近 step 1250 |

## 2026-06-19 — StarFlow Train 会话状态（05:15 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 05:15 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | >1125 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 153% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 4.7 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **52%–100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/` + `steps_1125/`（同步完成） |
| 负载 | 16.26 / 14.68 / 13.74 |
| 数据加载 | 数据 worker `Dl+`，线程数 2126 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 恢复 100%；数据 worker 线程数升至 2126；负载回落；接近 step 1250 |

## 2026-06-19 — StarFlow Train 会话状态（05:10 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 05:10 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（同步完成） |
| 训练状态 | ✅ **正常运行中**（主进程 `RLl+`，数据 worker `Dl+`） |
| 当前步数 | >1125 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 153% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 4.7 GiB |
| GPU | `P1.gpu.medium`，利用率 **74%**，SM **90%–100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/` + `steps_1125/`（同步完成） |
| 负载 | 18.79 / 13.93 / 13.25 |
| 数据加载 | 数据 worker `Dl+`，线程数 1262 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 1125 已同步完成；本地暂存清理；GPU 采样 74% 但 SM 90%–100%；1 分钟负载升至 18.79 |

## 2026-06-19 — StarFlow Train 会话状态（05:05 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 05:05 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `118753`（活跃同步 steps_1125） |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`，同步 worker `Rs`） |
| 当前步数 | **1125 / 80000** |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 152% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 4.8 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **94%–100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **18G**（含 `steps_1125` 暂存） |
| root overlay | **62%（19G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/`（`steps_1125` 同步中） |
| 负载 | 12.93 / 12.31 / 12.88 |
| 数据加载 | 数据 worker `Sl+`，线程数 1358 |
| 目标 r2 目录 | 不存在 |
| 备注 | 到达 step 1125，本地 checkpoint 18G 正在同步；GPU 恢复满载；数据 worker 线程数升至 1358 |

## 2026-06-19 — StarFlow Train 会话状态（05:01 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 05:01 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | >1000 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 152% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.1 GiB |
| GPU | `P1.gpu.medium`，利用率 **36%**，SM **36%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/`（同步完成） |
| 负载 | 13.86 / 13.68 / 13.61 |
| 数据加载 | 数据 worker `Dl+`，线程数 685 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 降至 36%；本地无 `output.log`；接近 step 1125 |

## 2026-06-19 — StarFlow Train 会话状态（04:56 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 04:56 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`/`Dl+`） |
| 当前步数 | >1000 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 152% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.1 GiB |
| GPU | `P1.gpu.medium`，利用率 **92%**，SM **92%**（已从 72% 回升），显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/`（同步完成） |
| 负载 | 11.23 / 13.81 / 13.70（负载回落） |
| 数据加载 | 数据 worker `Sl+`/`Dl+`，线程数 685 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 回升至 92%；本地无 `output.log`；接近 step 1125 |

## 2026-06-19 — StarFlow Train 会话状态（04:51 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 04:51 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`/`RLl+`，数据 worker `Sl+`/`Dl+`） |
| 当前步数 | >1000 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 152% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.1 GiB |
| GPU | `P1.gpu.medium`，利用率 **72%**，SM **72%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/`（同步完成） |
| 负载 | 15.49 / 13.12 / 13.23 |
| 数据加载 | 数据 worker `Sl+`/`Dl+`，线程数 1453 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 降至 72%；本地无 `output.log`；接近 step 1125 |

## 2026-06-19 — StarFlow Train 会话状态（04:46 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 04:46 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`/`RLl+`，数据 worker `Dl+`） |
| 当前步数 | >1000 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 152% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.1 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/`（同步完成） |
| 负载 | 9.61 / 12.54 / 13.19（负载继续回落） |
| 数据加载 | 数据 worker `Dl+`，线程数 877 |
| 目标 r2 目录 | 不存在 |
| 备注 | 本地无 `output.log`；下个保存点 step 1125 |

## 2026-06-19 — StarFlow Train 会话状态（04:42 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 04:42 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **step 1000 同步完成，训练恢复中**（`RLl+`/`SLl+`，`Dl+`/`Sl+`） |
| 当前步数 | 1000 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 151% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.1 GiB（已回落） |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K**（已清理） |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/` + `steps_1000/`（同步完成） |
| 负载 | 13.09 / 14.09 / 13.65 |
| 数据加载 | 数据 worker `Dl+`/`Sl+`，线程数 685 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 1000 评估/checkpoint 已完成；本地无 `output.log`；下个保存点 step 1125 |

## 2026-06-19 — StarFlow Train 会话状态（04:37 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 04:37 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `3608` |
| 训练状态 | ⚠️ **step 1000 checkpoint 保存并后台同步中**（主进程 `RLl+`，GPU 100%） |
| 当前步数 | 1000 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 151% |
| 数据worker CPU | 25.2% |
| 同步 worker CPU | 37.4% |
| 主进程内存 | 约 4.7 GiB（checkpoint 期间升高） |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **18G**（`steps_1000/`） |
| root overlay | **63%（19G / 30G）** |
| 网络 checkpoint | `steps_125/` 到 `steps_875/`，`steps_1000/` 同步中 |
| 负载 | 14.54 / 13.18 / 13.19 |
| 数据加载 | 数据 worker `Dl+`/`Sl+`，线程数 2413 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 1000 同时是 `eval_interval=250` 评估点；同步完成后 `/root/temp` 会清理 |

## 2026-06-19 — StarFlow Train 会话状态（04:32 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 04:32 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **恢复正常运行**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | >875 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 151% |
| 数据worker CPU | 25.3% |
| 主进程内存 | 约 2.1 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**（已从 20% 恢复），显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/`（同步完成） |
| 负载 | 11.13 / 14.31 / 13.69 |
| 数据加载 | 数据 worker `Dl+`，线程数 1165 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 已从 20% 恢复至 100%；本地无 `output.log`；接近 step 1000 |

## 2026-06-19 — StarFlow Train 会话状态（04:27 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 04:27 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ⚠️ **运行中，GPU 利用率显著降至 20%**（主进程 `SLl+`，数据 worker `Sl+`/`Dl+`） |
| 当前步数 | >875 / 80000（接近 step 1000） |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 151% |
| 数据worker CPU | 25.3% |
| 主进程内存 | 约 2.1 GiB |
| GPU | `P1.gpu.medium`，利用率 **20%**，SM **20%**（显著下降），显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/`（同步完成） |
| 负载 | 9.00 / 10.80 / 12.31（负载回落） |
| 数据加载 | 数据 worker `Sl+`/`Dl+`，线程数 1837 |
| 目标 r2 目录 | 不存在 |
| 备注 | 无 checkpoint 活动；可能接近 step 1000 评估点；本地无 `output.log` |

## 2026-06-19 — StarFlow Train 会话状态（04:23 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 04:23 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`/`RLl+`，数据 worker `Sl+`/`Dl+`） |
| 当前步数 | >875 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 151% |
| 数据worker CPU | 25.3% |
| 主进程内存 | 约 2.1 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/`（同步完成） |
| 负载 | 11.61 / 11.79 / 12.95（保持平稳） |
| 数据加载 | 数据 worker `Sl+`/`Dl+`，线程数 2221 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 已恢复至 100%；本地无 `output.log`；接近 step 1000 保存点 |

## 2026-06-19 — StarFlow Train 会话状态（04:18 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 04:18 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`/`Sl+`） |
| 当前步数 | >875 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 150% |
| 数据worker CPU | 25.3% |
| 主进程内存 | 约 2.1 GiB（已完全回落） |
| GPU | `P1.gpu.medium`，利用率 **92%**，SM **92%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/`（同步完成） |
| 负载 | 11.08 / 13.18 / 13.66（负载显著回落） |
| 数据加载 | 数据 worker `Dl+`/`Sl+`，线程数 1453 |
| 目标 r2 目录 | 不存在 |
| 备注 | 主进程内存已回落至 2.1 GiB；本地无 `output.log`；下个保存点 step 1000 |

## 2026-06-19 — StarFlow Train 会话状态（04:13 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 04:13 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **step 875 同步完成，训练恢复中**（`SLl+`/`Sl+`） |
| 当前步数 | 875 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 150% |
| 数据worker CPU | 25.3% |
| 主进程内存 | 约 4.7 GiB（checkpoint 后未完全回落） |
| GPU | `P1.gpu.medium`，利用率 **86%**，SM **86%**（已从 58% 恢复），显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K**（已清理） |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/` + `steps_875/`（同步完成） |
| 负载 | 16.88 / 14.87 / 14.18 |
| 数据加载 | 数据 worker `Sl+`，线程数 1261 |
| 目标 r2 目录 | 不存在 |
| 备注 | `summary.jsonl` 已更新至 875；本地无 `output.log`；下个保存点 step 1000 |

## 2026-06-19 — StarFlow Train 会话状态（04:08 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 04:08 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `215338`（新同步 worker） |
| 训练状态 | ⚠️ **step 875 checkpoint 保存并后台同步中**（主进程 `SLl+`/`RLl+`） |
| 当前步数 | 875 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 150% |
| 数据worker CPU | 25.3% |
| 同步 worker CPU | 34.8% |
| 主进程内存 | 约 4.7 GiB（checkpoint 期间升高） |
| GPU | `P1.gpu.medium`，利用率 **58%**，SM **58%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **18G**（`steps_875/`） |
| root overlay | **63%（19G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/`，`steps_875/` 同步中 |
| 负载 | 18.08 / 15.93 / 14.38 |
| 数据加载 | 数据 worker `Dl+`，线程数 2221 |
| 目标 r2 目录 | 不存在 |
| 备注 | `summary.jsonl` 已更新至 875；同步完成后 `/root/temp` 会清理 |

## 2026-06-19 — StarFlow Train 会话状态（04:03 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 04:03 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`/`Sl+`） |
| 当前步数 | >750 / 80000（接近 step 875） |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 150% |
| 数据worker CPU | 25.3% |
| 主进程内存 | 约 2.1 GiB |
| GPU | `P1.gpu.medium`，利用率 **74%**，SM **74%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/`（同步完成） |
| 负载 | 13.23 / 12.61 / 13.11（保持平稳） |
| 数据加载 | 数据 worker `Dl+`/`Sl+`，线程数 2125 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 利用率降至 74%，可能与数据加载有关；本地无 `output.log`；接近 step 875 |

## 2026-06-19 — StarFlow Train 会话状态（03:59 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 03:59 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`/`RLl+`，数据 worker `Dl+`/`Rl+`） |
| 当前步数 | >750 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 149% |
| 数据worker CPU | 25.3% |
| 主进程内存 | 约 2.1 GiB |
| GPU | `P1.gpu.medium`，利用率 **94%**，SM **94%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/`（同步完成） |
| 负载 | 13.22 / 13.13 / 13.44（负载平稳） |
| 数据加载 | 数据 worker `Dl+`/`Rl+`，线程数 1837 |
| 目标 r2 目录 | 不存在 |
| 备注 | 本地无 `output.log`；接近 step 875 保存点 |

## 2026-06-19 — StarFlow Train 会话状态（03:54 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 03:54 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`/`RLl+`，数据 worker `Sl+`/`Dl+`） |
| 当前步数 | >750 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 149% |
| 数据worker CPU | 25.3% |
| 主进程内存 | 约 2.1 GiB（已回落至 checkpoint 前水平） |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/`（同步完成） |
| 负载 | **16.84** / 13.99 / 13.70（1 分钟负载再次升高） |
| 数据加载 | 数据 worker `Sl+`/`Dl+`，线程数 1356 |
| 目标 r2 目录 | 不存在 |
| 备注 | 主进程内存已回落；本地无 `output.log`；下个保存点 step 875 |

## 2026-06-19 — StarFlow Train 会话状态（03:49 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 03:49 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `RLl+`/`SLl+`，数据 worker `Sl+`/`Dl+`） |
| 当前步数 | >750 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 149% |
| 数据worker CPU | 25.3% |
| 主进程内存 | 约 4.7 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**（已恢复），显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/`（同步完成） |
| 负载 | 12.96 / 14.69 / 14.00（负载回落） |
| 数据加载 | 数据 worker `Sl+`/`Dl+`，线程数 972 |
| 目标 r2 目录 | 不存在 |
| 备注 | 本地无 `output.log`；下个保存点 step 875 |

## 2026-06-19 — StarFlow Train 会话状态（03:44 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 03:44 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | >750 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 148% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 4.7 GiB |
| GPU | `P1.gpu.medium`，利用率 **90%**，SM **90%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/`（同步完成） |
| 负载 | 14.45 / 15.58 / 13.98 |
| 数据加载 | 数据 worker `Sl+`，线程数 1644 |
| 目标 r2 目录 | 不存在 |
| 备注 | 本地无 `output.log`；下个保存点 step 875 |

## 2026-06-19 — StarFlow Train 会话状态（03:40 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 03:40 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **step 750 同步完成，训练恢复中**（`SLl+`/`Sl+`） |
| 当前步数 | 750 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 148% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 4.7 GiB（checkpoint 后未完全回落） |
| GPU | `P1.gpu.medium`，利用率 **94%**，SM **94%**（从 0% 恢复），显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K**（已清理） |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/` + `steps_750/`（同步完成） |
| 负载 | 17.95 / 14.98 / 13.32 |
| 数据加载 | 数据 worker `Sl+`，线程数 1260 |
| 目标 r2 目录 | 不存在 |
| 备注 | `summary.jsonl` 已更新至 750；本地无 `output.log`；下个保存点 step 875 |

## 2026-06-19 — StarFlow Train 会话状态（03:35 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 03:35 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ⚠️ **step 750 checkpoint 保存中**（主进程 `RLl+`，GPU idle） |
| 当前步数 | 750 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 148% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 6.6 GiB（checkpoint 期间升高） |
| GPU | `P1.gpu.medium`，利用率 **0%**，SM **0%**（checkpoint 期间空闲），显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **15G**（`steps_750/` 保存中） |
| root overlay | **5%（1.3G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/`（`steps_750/` 待同步） |
| 负载 | 14.43 / 11.49 / 12.09 |
| 数据加载 | 数据 worker `Sl+`/`Dl+`，线程数 1164 |
| 目标 r2 目录 | 不存在 |
| 备注 | step 750 已本地保存，同步尚未开始；GPU 将在保存完成后恢复 |

## 2026-06-19 — StarFlow Train 会话状态（03:30 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 03:30 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Sl+`） |
| 当前步数 | >625 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 147% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.1 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/`（同步完成） |
| 负载 | 9.17 / 11.66 / 12.57（负载回落） |
| 数据加载 | 数据 worker `Sl+`，线程数 1260，正常等待 |
| 目标 r2 目录 | 不存在 |
| 备注 | 本地无 `output.log`；接近 step 750 保存点 |

## 2026-06-19 — StarFlow Train 会话状态（03:25 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 03:25 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `SLl+`，数据 worker `Dl+`） |
| 当前步数 | >625 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 148% |
| 数据worker CPU | 25.2% |
| 主进程内存 | 约 2.1 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**（已恢复），显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/`（同步完成） |
| 负载 | **18.25** / 13.95 / 13.24（1 分钟负载再次升高） |
| 数据加载 | 数据 worker `Dl+`，线程数 780 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 已恢复；本地无 `output.log`；下个保存点 step 750 |

## 2026-06-19 — StarFlow Train 会话状态（03:21 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 03:21 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（主进程 `RLl+`，数据 worker `Dl+`/`Rl+`） |
| 当前步数 | >625 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 147% |
| 数据worker CPU | 25.3% |
| 主进程内存 | 约 2.1 GiB |
| GPU | `P1.gpu.medium`，利用率 **76%**，SM **76%**（较前次下降），显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/`（同步完成） |
| 负载 | 9.91 / 10.59 / 12.43（持续回落） |
| 数据加载 | 数据 worker 线程数 1472，存在间歇性等待 |
| 目标 r2 目录 | 不存在 |
| 备注 | GPU 利用率短时降至 76%；本地无 `output.log`；下个保存点 step 750 |

## 2026-06-19 — StarFlow Train 会话状态（03:16 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 03:16 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中**（`SLl+`/`Dl+`） |
| 当前步数 | >625 / 80000 |
| 当前 loss | 网络目录无 `output.log`，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 147% |
| 数据worker CPU | 25.3% |
| 主进程内存 | 约 2.1 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/`（同步完成） |
| 负载 | 10.57 / 12.57 / 13.59 |
| 数据加载 | 数据 worker `Dl+`，间歇性等待 |
| 目标 r2 目录 | 不存在 |
| 备注 | `logging_frequency` CLI 覆盖为 10；本地无 `output.log`；下个保存点 step 750 |

## 2026-06-19 — StarFlow Train 会话状态（03:11 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 03:11 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无（`steps_625/` 已同步完成） |
| 训练状态 | ✅ **step 625 同步完成，训练继续**（`RLl+`/`Dl+`） |
| 当前步数 | >625 / 80000 |
| 当前 loss | 日志被 tqdm 覆盖，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 147% |
| 数据worker CPU | 25.3% |
| 主进程内存 | 约 2.1 GiB |
| GPU | `P1.gpu.medium`，利用率 **90%**，SM **98%**（已恢复），显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **24K**（已清理） |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/` + `steps_625/`（同步完成） |
| 负载 | 8.73 / 12.44 / 13.76（已回落） |
| 数据加载 | 数据 worker `Dl+`，间歇性等待 |
| 目标 r2 目录 | 不存在 |
| 备注 | 下一个保存点预计 step 750 |

## 2026-06-19 — StarFlow Train 会话状态（03:07 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 03:07 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `311873`（新同步 worker） |
| 训练状态 | ⚠️ **step 625/80000 checkpoint 保存并后台同步中** |
| 当前 loss | 日志被 tqdm 覆盖，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 147% |
| 数据worker CPU | 25.3% |
| 同步 worker CPU | 34.7% |
| 主进程内存 | 约 2.1 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **88%**（从 58% 恢复），显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **18G**（`steps_625/`） |
| root overlay | **63%（19G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/`，`steps_625/` 同步中 |
| 负载 | 15.52 / 15.39 / 14.78（已回落） |
| 数据加载 | 数据 worker `Sl+`，正常等待 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（03:02 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 03:02 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ⚠️ **运行中，GPU SM 降至 58%，负载升高至 20.67** |
| 当前步数 | >500 / 80000 |
| 当前 loss | 日志被 tqdm 覆盖，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 146% |
| 数据worker CPU | 25.3% |
| 主进程内存 | 约 2.1 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **58%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **24K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/`（同步完成） |
| 负载 | 1 分钟 **20.67**，5/15 分钟约 14-15 |
| 数据加载 | 数据 worker `Dl+`，可能存在数据加载瓶颈 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（02:57 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 02:57 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中，主进程 `SLl+`，step >500/80000** |
| 当前 loss | 日志被 tqdm 覆盖，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 146% |
| 数据worker CPU | 25.3% |
| 主进程内存 | 约 2.1 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **24K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/`（同步完成） |
| 负载 | 1 分钟负载 **18.59**（瞬时升高），5/15 分钟约 15 |
| 数据加载 | 数据 worker `Dl+`，间歇性等待 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（02:52 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 02:52 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无 |
| 训练状态 | ✅ **正常运行中，主进程 `SLl+`，数据 worker `Rl+`** |
| 当前步数 | >500 / 80000 |
| 当前 loss | 日志被 tqdm 覆盖，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 146% |
| 数据worker CPU | 25.3% |
| 主进程内存 | 约 2.1 GiB |
| GPU | `P1.gpu.medium`，利用率 **96%**，SM **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **24K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/`（同步完成） |
| 数据加载 | 数据 worker `Rl+`，活跃运行 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（02:48 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 02:48 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | 无活跃进程 |
| 训练状态 | ✅ **正常运行中，主进程 `RLl+`，step >500/80000** |
| 当前 loss | 日志被 tqdm 覆盖，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 145% |
| 数据worker CPU | 25.4% |
| 主进程内存 | 约 2.1 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **92%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **24K** |
| root overlay | **3%（860M / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/`（同步完成） |
| 数据加载 | 数据 worker `Dl+`，间歇性等待 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（02:43 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 02:43 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `196987`（已结束，temp 已清理） |
| 训练状态 | ✅ **正常运行中，step 500 checkpoint 已同步完成** |
| 当前步数 | 500 / 80000 |
| 当前 loss | 日志被 tqdm 覆盖，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 145% |
| 数据worker CPU | 25.4% |
| 主进程内存 | 约 2.1 GiB（checkpoint 后回落） |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **24K**（已清理） |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/` + `steps_500/`（同步完成） |
| 数据加载 | 数据 worker `Sl+`，正常等待 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（02:38 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 02:38 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `196987`（新同步 worker） |
| 训练状态 | ⚠️ **step 500/80000 checkpoint 保存并后台同步中** |
| 当前 loss | 日志被 tqdm 覆盖，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 144% |
| 数据worker CPU | 25.4% |
| 同步 worker CPU | 28.9% |
| 主进程内存 | 约 4.9 GiB（checkpoint 期间回升） |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **18G**（`steps_500/`） |
| root overlay | **63%（19G / 30G）** |
| 网络 checkpoint | `steps_125/` + `steps_250/` + `steps_375/`，`steps_500/` 同步中 |
| 注意 | 早期 `steps_50/` 已从网络盘移除；`output.log` 仍无清晰 Step 行 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（02:33 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 02:33 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `81059`（defunct） |
| 训练状态 | ✅ **正常运行中，主进程 `RLl+`，已运行约 2 小时** |
| 当前步数 | >375 / 80000 |
| 当前 loss | 日志被 tqdm 覆盖，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 144% |
| 数据worker CPU | 25.5% |
| 主进程内存 | 约 2.1 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **90%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **24K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_50/` + `steps_125/` + `steps_250/` + `steps_375/`（同步完成） |
| 数据加载 | 数据 worker `Dl+`，间歇性等待 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（02:29 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 02:29 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `81059`（defunct） |
| 训练状态 | ✅ **正常运行中，GPU SM 100%，状态稳定** |
| 当前步数 | >375 / 80000 |
| 当前 loss | 日志被 tqdm 覆盖，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 143% |
| 数据worker CPU | 25.4% |
| 主进程内存 | 约 2.1 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **24K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_50/` + `steps_125/` + `steps_250/` + `steps_375/`（同步完成） |
| 数据加载 | 数据 worker `Dl+`，间歇性等待 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（02:24 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 02:24 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `81059`（defunct） |
| 训练状态 | ✅ **正常运行中，GPU SM 利用率 100%** |
| 当前步数 | >375 / 80000 |
| 当前 loss | 日志被 tqdm 覆盖，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 143% |
| 数据worker CPU | 25.3% |
| 主进程内存 | 约 2.1 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **24K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_50/` + `steps_125/` + `steps_250/` + `steps_375/`（同步完成） |
| 数据加载 | 数据 worker `Dl+`，存在间歇性等待 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（02:19 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 02:19 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `81059`（defunct） |
| 训练状态 | ✅ **正常运行中，step >375/80000** |
| 当前 loss | 日志被 tqdm 覆盖，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 143% |
| 数据worker CPU | 25.3% |
| 主进程内存 | 约 2.1 GiB（回落至低位） |
| GPU | `P1.gpu.medium`，利用率 **92%**，SM **90%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **24K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_50/` + `steps_125/` + `steps_250/` + `steps_375/`（同步完成） |
| 数据加载 | 数据 worker `Dl+`，存在间歇性等待 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（02:14 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 02:14 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `81059`（defunct） |
| 训练状态 | ✅ **正常运行中，主进程 `RLl+`，数据 worker `Rl+`** |
| 当前步数 | >375 / 80000 |
| 当前 loss | 日志被 tqdm 覆盖，上次可读值 step 360 的 `0.1705` |
| 主进程 CPU | 142% |
| 数据worker CPU | 25.3% |
| 主进程内存 | 约 4.9 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM **96%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **24K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_50/` + `steps_125/` + `steps_250/` + `steps_375/`（同步完成） |
| 数据加载 | 数据 worker `Rl+`，恢复正常 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（02:10 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 02:10 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `81059`（已结束，defunct） |
| 训练状态 | ✅ **正常运行中，step 375 checkpoint 已同步完成** |
| 当前步数 | ≥375 / 80000 |
| 当前 loss | 日志被进度条覆盖，上次清晰值 step 360 的 `0.1705` |
| 主进程 CPU | 141% |
| 数据worker CPU | 25.3% |
| 主进程内存 | 约 4.9 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM 76%，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **24K**（已清理） |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_50/` + `steps_125/` + `steps_250/` + `steps_375/`（同步完成） |
| 注意 | `output.log` 被 tqdm 进度条覆盖，清晰 Step 行暂不可读 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（02:05 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 02:05 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| checkpoint 同步进程 PID | `81059` |
| 训练状态 | ⚠️ **step 375/80000 checkpoint 保存并后台同步中** |
| 当前 loss | 上次清晰日志 `action_dit_loss: 0.1705`（step 360） |
| 主进程 CPU | 140% |
| 数据worker CPU | 25.3% |
| 同步 worker CPU | 34.7% |
| 主进程内存 | 约 4.9 GiB（checkpoint 期间回升） |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM 80%，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **18G**（`steps_375/`） |
| root overlay | **7–8%（2.1–2.2G / 30G）** |
| 网络 checkpoint | `steps_50/` + `steps_125/` + `steps_250/`，`steps_375/` 同步中 |
| 数据加载 | 数据 worker `Dl+`，与 checkpoint 写盘相关 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（02:02 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 02:02 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| 训练状态 | ✅ **正常运行中，step 360/80000** |
| 当前 loss | `action_dit_loss: 0.1705` |
| 主进程 CPU | 139% |
| 数据worker CPU | 25.4% |
| 主进程内存 | 约 2.1 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，SM 78%，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **24K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_50/` + `steps_125/` + `steps_250/` |
| 数据加载 | `timing/data=0.0005`，恢复正常 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（01:55 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 01:55 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| 训练状态 | ✅ **正常运行中，step 330/80000** |
| 当前 loss | `action_dit_loss: 0.246` |
| 主进程 CPU | 138% |
| 数据worker CPU | 25.4% |
| 主进程内存 | 约 2.1 GiB |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_50/` + `steps_125/` + `steps_250/` |
| 数据加载 | `timing/data=0.0006`，恢复正常 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（01:50 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 01:50 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| 训练状态 | ✅ **正常运行中，step 310/80000** |
| 当前 loss | `action_dit_loss: 0.297` |
| 主进程 CPU | 137% |
| 数据worker CPU | 25.5% |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_50/` + `steps_125/` + `steps_250/` |
| 注意 | `timing/data=1.449`，数据加载出现延迟 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（01:46 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 01:46 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| 训练状态 | ✅ **正常运行中，step 290/80000** |
| 当前 loss | `action_dit_loss: 0.490` |
| 主进程 CPU | 136% |
| 数据worker CPU | 25.4% |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_50/` + `steps_125/` + `steps_250/` |
| loss 趋势 | 波动较大（0.239 → 0.391 → 0.304 → 0.490） |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（01:41 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 01:41 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| 训练状态 | ✅ **正常运行中，step 270/80000** |
| 当前 loss | `action_dit_loss: 0.304` |
| 主进程 CPU | 134% |
| 数据worker CPU | 25.4% |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** ✅ |
| 网络 checkpoint | `steps_50/` + `steps_125/` + `steps_250/` |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（01:36 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 01:36 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| 训练状态 | ✅ **正常运行中，step 250 checkpoint 保存并同步中** |
| 当前步数 | ≥250 / 80000 |
| 主进程 CPU | 132% |
| 数据worker CPU | 25.3% |
| 主进程内存 | 约 4.9 GiB |
| GPU | `P1.gpu.medium`，利用率 **72%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **18G**（step 250 保存中） |
| root overlay | **63%（19G / 30G）** ⚠️ |
| 网络 checkpoint | `steps_50/` + `steps_125/` + `.steps_250.sync_tmp_*` 同步中 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（01:31 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 01:31 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| 训练状态 | ✅ **正常运行中，step 230/80000** |
| 当前 loss | `action_dit_loss: 0.329` |
| 主进程 CPU | 130% |
| 数据worker CPU | 25.4% |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_50/` + `steps_125/` |
| 数据加载 | `timing/data=0.0004`，恢复正常 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（01:26 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 01:26 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| 训练状态 | ✅ **正常运行中，step 210/80000** |
| 当前 loss | `action_dit_loss: 0.391` |
| 主进程 CPU | 127% |
| 数据worker CPU | 25.2% |
| GPU | `P1.gpu.medium`，利用率 **64%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_50/` + `steps_125/` |
| 注意 | `timing/data=0.384`，某步数据加载出现延迟 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（01:22 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 01:22 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| 训练状态 | ✅ **正常运行中，step 190/80000** |
| 当前 loss | `action_dit_loss: 0.239` |
| 主进程 CPU | 124% |
| 数据worker CPU | 25.2% |
| GPU | `P1.gpu.medium`，利用率 **54%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_50/` + `steps_125/` |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（01:17 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 01:17 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| 训练状态 | ✅ **正常运行中，step 170/80000** |
| 当前 loss | `action_dit_loss: 0.348` |
| 主进程 CPU | 121% |
| 数据worker CPU | 25.3% |
| GPU | `P1.gpu.medium`，利用率 **76%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_50/` + `steps_125/` |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（01:12 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 01:12 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| 训练状态 | ✅ **正常运行中，step 150/80000** |
| 当前 loss | `action_dit_loss: 0.401` |
| 主进程 CPU | 116% |
| 数据worker CPU | 24.7% |
| 主进程内存 | 约 2.2 GiB（checkpoint 保存后释放） |
| GPU | `P1.gpu.medium`，利用率 **66%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K**（step 125 已清理） |
| root overlay | **4%（1.1G / 30G）** ✅ |
| 网络 checkpoint | `steps_50/` + `steps_125/`（同步完成） |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（01:07 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 01:07 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| 训练状态 | ✅ **正常运行中，step 130/80000，正在保存 step 125 checkpoint** |
| 当前 loss | `action_dit_loss: 0.340` |
| 主进程 CPU | 111% |
| 数据worker CPU | 24.6% |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **18G**（step 125 本地保存中） |
| root overlay | **63%（19G / 30G）** ⚠️ |
| 网络 checkpoint | `steps_50/` + `.steps_125.sync_tmp_*` 同步中 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（01:03 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 01:03 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| 训练状态 | ✅ **正常运行中，step 120/80000** |
| 当前 loss | `action_dit_loss: 0.542` |
| 主进程 CPU | 107% |
| 数据worker CPU | 24.7% |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_50/` 18G 保留 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（00:58 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 00:58 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| 训练状态 | ✅ **正常运行中，step 100/80000** |
| 当前 loss | `action_dit_loss: 0.506` |
| 主进程 CPU | 99.1% |
| 数据worker CPU | 23.8% |
| GPU | `P1.gpu.medium`，利用率 **76%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_50/` 18G 保留 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（00:53 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 00:53 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465` |
| 训练状态 | ✅ **正常运行中，step 80/80000** |
| 当前 loss | `action_dit_loss: 0.572` |
| 主进程 CPU | 92.5% |
| 数据worker CPU | 24.0% |
| GPU | `P1.gpu.medium`，利用率 **100%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K** |
| root overlay | **4%（1.1G / 30G）** |
| 网络 checkpoint | `steps_50/` 18G 保留 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（00:48 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 00:48 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 主训练进程 PID | `84113` |
| 数据加载子进程 PID | `124465`（`pt_data_worker`） |
| 训练状态 | ✅ **正常运行中，step 60/80000** |
| 当前 loss | `action_dit_loss: 0.643` |
| 主进程 CPU | 70.0% |
| 数据worker CPU | 20.0% |
| 主进程内存 | 约 5.1 GiB |
| 数据worker内存 | 约 4.8 GiB |
| GPU | `P1.gpu.medium`，利用率 **92%**，显存 39538/40488 MiB（97.6%） |
| `/root/temp` | **28K**（本地 checkpoint 已清理） |
| root overlay | **4%（1.1G / 30G）** ✅ |
| 网络 checkpoint | `steps_50/` 18G 保留 |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（00:44 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 00:44 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 训练进程 PID | `84113` |
| 训练状态 | ✅ **resume 成功，已进入训练循环** |
| 当前步数 | **50 / 80000** |
| CPU 使用率 | 约 49.1% |
| 内存 RSS | 约 4.5 GiB（0.8%） |
| 线程数 | 194 |
| GPU | `P1.gpu.medium`，显存 19118/40488 MiB（47.2%），PID 84113 占用 |
| `/root/temp` | 18G |
| root overlay | 63%（19G / 30G） |
| W&B | resume 成功，新 run dir `run-20260619_004339-...` |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（00:39 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 00:39 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 训练进程 PID | `84113` |
| 训练状态 | 运行中，仍在初始化/加载（`DLl+`） |
| resume 来源 | step 50 |
| CPU 使用率 | 约 4.0% |
| 内存 %MEM | 0.1% |
| GPU | `P1.gpu.medium`，显存 0/40488 MiB，利用率 0%，无活跃进程 |
| `/root/temp` | 18G |
| root overlay | 63%（19G / 30G） |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（00:37 CST）

| 字段 | 值 |
| --- | --- |
| 监控时间 | 2026-06-19 00:37 CST |
| 目标 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2` |
| 实际运行 run_id | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619` |
| 训练进程 PID | `84113` |
| 训练状态 | 运行中，仍在初始化/加载 |
| resume 来源 | step 50 |
| CPU 使用率 | 约 4.0% |
| 内存 RSS | 约 312 MiB |
| GPU | 可见但无活跃进程 |
| `/root/temp` | 18G |
| root overlay | 63%（19G / 30G） |
| 目标 r2 目录 | 不存在 |

## 2026-06-19 — StarFlow Train 会话状态（00:35 CST）

- 监控时间：2026-06-19 00:35 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619`
- 实际进程 PID：`84113`，父进程 `84107`，tmux 会话：`starflow_train`
- 训练状态：🔄 **训练已重新启动，正在 resume / 初始化**
- resume 来源：`/root/temp/...250619/checkpoints/steps_50/`（`summary.jsonl` 显示 `"steps": 50`）
- 当前步数：加载中，尚未进入训练循环
- 启动命令：用户在 tmux 中执行，使用 `local_checkpoint_root=/root/temp`，`RUN_ID=...250619`，`IS_RESUME=True`
- 当前进程状态：主进程 `D`（disk sleep，多线程 `l`，NLWP 64）
- 主进程资源：RSS 约 **312 MiB**，**64 线程**，已运行约 2 分钟
- GPU：可见但显示 `SMI N/A`，当前无活跃 GPU 进程（仍在加载）
- checkpoint：
  - 本地：`/root/temp/...250619/checkpoints/steps_50/`，大小 **18G**
  - 网络：`/gemini/code/...Checkpoints/...250619/checkpoints/steps_50/`，大小 **18G**
- 本地 staging 磁盘：⚠️ **root overlay 已用 63%（19G / 30G）**
- 同步状态：同步队列为空（训练刚启动）
- 系统资源：总内存 503 GiB，已用 53 GiB，可用 445 GiB，负载 11.67/13.18/14.50
- 目标 r2 状态：未运行，输出目录不存在
- 后续观察：
  - 进程何时完成加载并进入训练循环
  - W&B 在线同步是否正常
  - 训练速度是否恢复约 23 s/it
  - root overlay 空间在下次保存时是否够用

## 2026-06-19 — StarFlow Train 会话状态（00:29 CST）

- 监控时间：2026-06-19 00:29 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：无训练进程运行
- 训练状态：⏹️ **训练仍处于停止状态**
- 当前步数：—
- 当前进程状态：无 train_starvla 进程
- GPU：当前无活跃 GPU 进程（`nvidia-smi` 无输出）
- checkpoint：
  - 本地：`/root/temp/...250619/checkpoints/steps_50/`，大小 **18G**
  - 网络：`/gemini/code/...Checkpoints/...250619/checkpoints/steps_50/`，大小 **18G**
  - 新 staging 路径：`/gemini/code/...Checkpoints/_temp_staging/` 总大小 **0**
- 本地 staging 磁盘：⚠️ **root overlay 已用 63%（19G / 30G）**
- 同步状态：无同步队列/日志（训练已停止）
- 系统资源：总内存 503 GiB，已用 52 GiB，可用 446 GiB，负载 15.58/13.82/15.05
- 目标 r2 状态：未运行，输出目录不存在
- 与上次对比：系统资源、checkpoint 位置均无变化
- **配置更新（00:30 CST）**：按用户要求，本地临时路径改回 `/root/temp`，`local_checkpoint_keep_count=1`。新训练启动后路径为 `/root/temp/<RUN_ID>/checkpoints/steps_XX/`，只保留 1 个历史版本。

## 2026-06-19 — StarFlow Train 会话状态（00:25 CST）

- 监控时间：2026-06-19 00:25 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：无训练进程运行
- 训练状态：⏹️ **训练仍处于停止状态**
- 当前步数：—
- 当前进程状态：无 train_starvla 进程
- GPU：当前无活跃 GPU 进程（`nvidia-smi` 无输出）
- checkpoint：
  - 本地：`/root/temp/...250619/checkpoints/steps_50/`，大小 **18G**
  - 网络：`/gemini/code/...Checkpoints/...250619/checkpoints/steps_50/`，大小 **18G**
  - 新 staging 路径：`/gemini/code/...Checkpoints/_temp_staging/` 总大小 **0**，仍未放入 checkpoint
- 本地 staging 磁盘：⚠️ **root overlay 已用 63%（19G / 30G）**
- 同步状态：无同步队列/日志（训练已停止）
- 系统资源：总内存 503 GiB，已用 52 GiB，可用 446 GiB，负载 14.42/14.05/15.72
- 目标 r2 状态：未运行，输出目录不存在
- 与上次对比：系统资源、checkpoint 位置均无变化，训练仍未启动

## 2026-06-19 — StarFlow Train 会话状态（00:20 CST）

- 监控时间：2026-06-19 00:20 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：无训练进程运行
- 训练状态：⏹️ **训练仍处于停止状态**
- 当前步数：—
- 当前进程状态：无 train_starvla 进程
- GPU：当前无活跃 GPU 进程（`nvidia-smi` 无输出）
- checkpoint：
  - 本地：`/root/temp/...250619/checkpoints/steps_50/`，大小 **18G**
  - 网络：`/gemini/code/...Checkpoints/...250619/checkpoints/steps_50/`，大小 **18G**
  - 新 staging 路径：`/gemini/code/...Checkpoints/_temp_staging/...250619/` **尚未创建**
- 本地 staging 磁盘：⚠️ **root overlay 已用 63%（19G / 30G）**
- 同步状态：无同步队列/日志（训练已停止）
- 系统资源：总内存 503 GiB，已用 53 GiB，可用 445 GiB，负载 10.02/14.01/16.26
- 目标 r2 状态：未运行，输出目录不存在
- 与 2 分钟前对比：系统资源、checkpoint 位置均无变化

## 2026-06-19 — StarFlow Train 会话状态（00:18 CST）

- 监控时间：2026-06-19 00:18 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：无训练进程运行
- 训练状态：⏹️ **训练仍处于停止状态**
- 当前步数：—
- 当前进程状态：无 train_starvla 进程
- GPU：当前无活跃 GPU 进程（`nvidia-smi` 无输出）
- checkpoint：
  - 本地：`/root/temp/...250619/checkpoints/steps_50/`，大小 **18G**
  - 网络：`/gemini/code/...Checkpoints/...250619/checkpoints/steps_50/`，大小 **18G**
  - 新 staging 路径：`/gemini/code/...Checkpoints/_temp_staging/...250619/` **尚未创建**
- 本地 staging 磁盘：⚠️ **root overlay 已用 63%（19G / 30G）**
- 同步状态：无同步队列/日志（训练已停止）
- 系统资源：总内存 503 GiB，已用 52 GiB，可用 446 GiB，负载 13.48/15.73/17.04
- 目标 r2 状态：未运行，输出目录不存在
- 配置修复：
  - `examples/LIBERO/train_files/run_starflow_train_ready.sh` 已修复默认 resume（`IS_RESUME=True` 并正确传入 `--trainer.is_resume`）
  - `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml` 已增加 `is_resume: true`
- 后续观察：
  - 用户是否移动 checkpoint 并使用新配置重启训练
  - 重启后 staging 路径是否正确使用网络盘

## 2026-06-19 — StarFlow Train 会话状态（00:14 CST）

- 监控时间：2026-06-19 00:14 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：无训练进程运行
- 训练状态：⏹️ **训练仍处于停止状态**
- 当前步数：—
- 退出原因：上一次为 `KeyboardInterrupt`（00:09 CST 之前）
- 当前进程状态：无 train_starvla 进程
- GPU：当前无活跃 GPU 进程
- checkpoint：
  - 本地：`/root/temp/...250619/checkpoints/steps_50/`，大小 **18G**
  - 网络：`/gemini/code/...Checkpoints/...250619/checkpoints/steps_50/`，大小 **18G**
  - 新 staging 路径：`/gemini/code/...Checkpoints/_temp_staging/...250619/` **尚未创建**
- 本地 staging 磁盘：⚠️ **root overlay 已用 63%（19G / 30G）**
- 系统资源：总内存 503 GiB，已用 52 GiB，可用 446 GiB，负载 18.02/16.84/17.61
- 目标 r2 状态：未运行，输出目录不存在
- 后续观察：
  - 用户是否移动 checkpoint 并使用 `--trainer.is_resume True` 重启

## 2026-06-19 — StarFlow Train 会话状态（00:09 CST）

- 监控时间：2026-06-19 00:09 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619`
- 实际进程 PID：`83923`（**已退出**）
- 实际进程启动时间：23:28 CST 左右，最后运行约 **35 分 25 秒**
- 训练状态：⏹️ **训练进程已退出（KeyboardInterrupt）**
- 最后已知步数：**64 / 80000**
- 退出原因：`KeyboardInterrupt`，退出位置在数据预处理 `state_action.py:499`
- 当前进程状态：无 train_starvla 进程运行
- 主进程资源：—
- 启动配置（旧）：
  - `NUM_WORKERS=0`
  - `WANDB_MODE=online`
  - `trainer.checkpoint_format=lightweight`
  - `trainer.enable_local_checkpoint_staging=True`
  - `trainer.local_checkpoint_root=/root/temp`
  - `trainer.local_checkpoint_keep_count=2`
- W&B：run 已结束，日志已保存
- GPU：当前无活跃 GPU 进程
- checkpoint：
  - 本地：`/root/temp/...250619/checkpoints/steps_50/`，大小 **18G**
  - 网络：`/gemini/code/...Checkpoints/...250619/checkpoints/steps_50/`，大小 **18G**，同步已完成
- 本地 staging 磁盘：⚠️ **root overlay 已用 63%（19G / 30G）**
- 同步状态：step 50 同步已完成
- 系统资源：总内存 503 GiB，已用 52 GiB，可用 446 GiB，负载 11.35/14.70/17.44
- 目标 r2 状态：未运行，输出目录不存在
- ⚠️ **注意**：
  - tmux 中待执行的重新启动命令仍包含 `SAVE_INTERVAL=50` 和 `local_checkpoint_root=/root/temp`，会覆盖新默认值
  - 新训练建议清理 `/root/temp/...250619/` 释放空间，并使用新配置
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - 用户是否使用新配置重启训练
  - 是否从 step 50 resume

## 2026-06-19 — StarFlow Train 会话状态（00:04 CST）

- 监控时间：2026-06-19 00:04 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619`
- 实际进程 PID：`83923`，父进程 `83878`，tmux 会话：`starflow_train`
- 实际进程启动时间：23:28 CST 左右，已运行约 **35 分 25 秒**
- 训练状态：✅ **训练正常运行中，当前 step 64/80000**
- 当前步数：**64 / 80000**
- 训练速度：约 **22.99 s/it**
- 当前进程状态：主进程 `D`（disk sleep，多线程 `l`，NLWP 881）
- 主进程资源：RSS **4.33 GiB**，**881 线程**
- 启动配置：
  - `NUM_WORKERS=0`
  - `WANDB_MODE=online`
  - `trainer.checkpoint_format=lightweight`
  - `trainer.enable_local_checkpoint_staging=True`
  - `trainer.local_checkpoint_root=/root/temp`（旧配置）
  - `trainer.local_checkpoint_keep_count=2`（旧配置）
- W&B：在线同步正常
- GPU：`P1.gpu.medium`，显存 **39518/40488 MiB（97.6%）**，利用率 **76%**，温度 31°C
- checkpoint：
  - 本地：`/root/temp/...250619/checkpoints/steps_50/`，大小 **18G**
  - 网络：`/gemini/code/...Checkpoints/...250619/checkpoints/steps_50/`，大小 **18G**，**同步已完成**
- 本地 staging 磁盘：⚠️ **root overlay 已用 63%（19G / 30G）**
- 同步状态：step 50 同步已完成，`.checkpoint_sync_queue` 当前为空
- 系统资源：总内存 503 GiB，已用 59 GiB，可用 439 GiB，负载 12.66/18.60/19.35
- 目标 r2 状态：未运行，输出目录不存在
- ⚠️ **紧急风险**：
  - 当前进程仍使用旧配置，step 100 保存第二个 checkpoint 时，root overlay 必将超过 30G
  - 建议尽快终止当前进程并使用新配置重启
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - step 100 是否会因磁盘空间失败
  - 新配置重启后 checkpoint 保存行为

## 2026-06-19 — 配置修改规避磁盘空间风险（00:03 CST）

- 修改时间：2026-06-19 00:03 CST
- 修改原因：step 50 首次 checkpoint 大小 18G，`local_checkpoint_keep_count=2` 需保留 2 个 = 36G，超过 `/root/temp` 所在 root overlay 的 30G 总量
- 修改文件：
  1. `examples/LIBERO/train_files/run_starflow_train_ready.sh`
  2. `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml`
- 具体修改：
  - `local_checkpoint_root` 从 `/root/temp` 改为 `/gemini/code/starVLA/playground/Checkpoints/_temp_staging`
  - `local_checkpoint_keep_count` 从 `2` 改为 `1`
- 效果：
  - 临时 checkpoint 改存到网络盘（1.3P 可用空间），不再受 root overlay 30G 限制
  - 只保留 1 个本地 checkpoint，进一步降低空间占用
- 已创建目录：`/gemini/code/starVLA/playground/Checkpoints/_temp_staging`
- 注意：
  - 当前运行的 PID 83923 仍使用旧配置（`/root/temp`），**需要重启训练后新配置才生效**
  - 如果手动启动时显式传入 `LOCAL_CHECKPOINT_ROOT=/root/temp`，会覆盖脚本默认值，需避免
- 建议启动命令（无需再传 local_checkpoint_root，保存间隔 125）：
  ```bash
  WANDB_MODE=online \
  NUM_WORKERS=0 \
  DATA_MIX=libero_all \
  MAX_TRAIN_STEPS=80000 \
  NUM_PROCESSES=1 \
  PER_DEVICE_BATCH_SIZE=4 \
  GRADIENT_ACCUMULATION_STEPS=8 \
  LOGGING_FREQUENCY=50 \
  SAVE_INTERVAL=125 \
  EVAL_INTERVAL=250 \
  RUN_ID=P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619 \
  bash examples/LIBERO/train_files/run_starflow_train_ready.sh
  ```
- 说明：脚本和配置文件的默认 `save_interval` 已经是 125；如果命令行显式传入 `SAVE_INTERVAL=50` 会覆盖默认值

## 2026-06-18 — StarFlow Train 会话状态（23:58 CST）

- 监控时间：2026-06-18 23:58 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619`
- 实际进程 PID：`83923`，父进程 `83878`，tmux 会话：`starflow_train`
- 实际进程启动时间：23:28 CST 左右，已运行约 **29 分 35 秒**
- 训练状态：✅ **训练正常运行中，已完成首次 checkpoint 保存，当前 step 51/80000**
- 当前步数：**51 / 80000**
- 首次 checkpoint：step 50，大小 **18G**，已保存至 `/root/temp`
- 后台同步：已启动，从 `/root/temp/.../steps_50` 同步到网络盘
- 训练速度：进度条显示 **30.21 s/it**（受 checkpoint 保存/同步影响）
- 当前进程状态：主进程 `S`（sleeping/interruptible，多线程 `l`，NLWP 305）
- 主进程资源：RSS **4.30 GiB**，**305 线程**
- 启动配置：
  - `NUM_WORKERS=0`
  - `WANDB_MODE=online`
  - `trainer.checkpoint_format=lightweight`
  - `trainer.enable_local_checkpoint_staging=True`
  - `trainer.local_checkpoint_root=/root/temp`
  - `trainer.local_checkpoint_keep_count=2`
- W&B：在线同步正常
- GPU：`P1.gpu.medium`，显存 **39518/40488 MiB（97.6%）**，利用率 **20%**（checkpoint 期间下降），温度 31°C
- 本地暂存目录：`/root/temp/...250619/checkpoints/steps_50/`，大小 **18G**
- 网络输出目录：`/gemini/code/...Checkpoints/...250619/`，大小 **659M**，checkpoints 子目录含 `.steps_50.sync_tmp_*` 临时目录
- 同步状态：**进行中**，`.checkpoint_sync_queue` 和 `checkpoint_sync.log` 已创建
- 系统资源：总内存 503 GiB，已用 72 GiB，可用 426 GiB，负载 16.11/18.26/19.14
- root overlay 使用率：**32%（9.6G / 30G）**
- 目标 r2 状态：未运行，输出目录不存在
- ⚠️ **紧急风险**：
  - 单个 checkpoint 18G，`local_checkpoint_keep_count=2` 需保留 2 个 = 36G
  - root overlay 仅 30G，step 100 保存时极有可能因空间不足失败
- 建议：
  - 将 `local_checkpoint_root` 改到 `/gemini/code` 下（网络盘 1.3P 可用）
  - 或将 `local_checkpoint_keep_count` 改为 1
  - 或增大保存间隔（如 250/500 步）
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - step 50 同步何时完成
  - step 100 保存是否会因磁盘空间失败
  - 同步完成后训练速度是否恢复

## 2026-06-18 — StarFlow Train 瓶颈诊断（23:52 CST）

- 监控时间：2026-06-18 23:52 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619`
- 实际进程 PID：`83923`，tmux 会话：`starflow_train`
- 训练状态：✅ **训练正常运行中，当前 step 43/80000，速度约 19.78 s/it**
- **核心结论：GPU 利用率其实已经很高，瓶颈在模型计算量本身**

### 诊断结果

| 检查项 | 结果 |
| --- | --- |
| GPU SM 利用率 | **98% ~ 100%**（`nvidia-smi dmon` 连续 5 次采样） |
| GPU 显存占用 | **39518/40488 MiB（97.6%）** |
| CPU 利用率 | 大部分核心空闲 80-90%，不是瓶颈 |
| I/O wait | 约 1-2%，不是瓶颈 |
| 磁盘利用率 | sdb 约 3-8%，不是瓶颈 |
| 数据加载 | `NUM_WORKERS=0`，数据集在 seaweedfs 网络盘上，有优化空间但不是当前瓶颈 |
| 精度 | bf16 已启用，fp16 禁用 |
| 梯度检查点 | 已启用 |
| Flow Matching | `repeated_diffusion_steps=2`，`num_inference_timesteps=4`，每步动作预测需 8 次采样 |
| 可用 GPU | 仅 1 张 `P1.gpu.medium` |

### 瓶颈定位

1. **GPU 计算单元已满载**：SM 利用率 100%，说明 GPU 没有空闲，只是每步需要大量计算。
2. **主要计算来源**：
   - Qwen3-VL-4B 视觉-语言编码
   - 5B 参数模型前向/反向传播
   - Flow Matching 动作头：每步 2×4=8 次扩散采样
3. **显存接近满载**：batch size 提升空间有限。
4. **数据加载可优化但非当前瓶颈**：GPU 未因等待数据而空闲。

### 优化建议

| 方向 | 说明 |
| --- | --- |
| 减小 `num_inference_timesteps`（4→2） | 最直接提速，但可能降低动作质量 |
| 减小 `repeated_diffusion_steps`（2→1） | 类似效果 |
| 增加 `NUM_WORKERS` | 减少数据加载开销，但当前 GPU 已满载，提升有限 |
| 降低图像分辨率/帧数 | 减少 VLM 计算，需改配置 |
| 多卡训练 | 当前环境只有 1 张 GPU，不可行 |

- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - step 50 首次 checkpoint 保存
  - 训练速度是否继续稳定在 19-22 s/it

## 2026-06-18 — StarFlow Train 会话状态（23:50 CST）

- 监控时间：2026-06-18 23:50 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619`（日期后缀 250619，无 r2/r3 后缀）
- 实际进程 PID：`83923`，父进程 `83878`，tmux 会话：`starflow_train`
- 实际进程启动时间：23:28 CST 左右，已运行约 **22 分 12 秒**
- 训练状态：✅ **训练正常运行中，当前 step 30/80000**
- 当前步数：**30 / 80000**
- 训练速度：约 **22.03 s/it**，速度稳定
- 当前进程状态：主进程 `D`（disk sleep，多线程 `l`，NLWP 497）
- 主进程资源：RSS **4.18 GiB**，**497 线程**
- 启动配置：
  - `NUM_WORKERS=0`
  - `WANDB_MODE=online`
  - `trainer.checkpoint_format=lightweight`
  - `trainer.enable_local_checkpoint_staging=True`
  - `trainer.local_checkpoint_root=/root/temp`
  - `trainer.local_checkpoint_keep_count=2`
- DeepSpeed：ZeRO-2，`train_batch_size=32`，`train_micro_batch_size_per_gpu=4`，`gradient_accumulation_steps=8`
- 模型参数：5071.088M 总量，633.272M 可训练
- W&B：在线同步正常，`output.log` 已生成
- GPU：`P1.gpu.medium`，显存 **39518/40488 MiB（97.6%）**，利用率 **90%**，温度 31°C
- checkpoint：无，预计 step 50 首次保存
- 输出目录大小：132K，内容：`config.yaml`、`config.full.yaml`、`dataset_statistics.json`、`wandb/`、空 `checkpoints/`
- 本地暂存目录大小：20K，内容：`config.yaml`、`config.full.yaml`、`dataset_statistics.json`、空 `checkpoints/`
- 本地 staging 磁盘：⚠️ **root overlay**，总量 30 GiB，可用 29 GiB
- 同步状态：尚未开始，无 `.checkpoint_sync_queue`，无 `checkpoint_sync.log`
- 系统资源：总内存 503 GiB，已用 71 GiB，可用 426 GiB，负载 13.72/19.59/19.72
- 目标 r2 状态：未运行，输出目录不存在
- 风险与建议：
  - 训练速度稳定在 22 s/it，估算总时间约 20.4 天
  - GPU 显存已占 97.6%，接近满载
  - `/root/temp` 位于 root overlay（30G），保留 2 个 checkpoint 可能空间不足
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - step 50 首次 lightweight checkpoint 保存时间及空间占用
  - 网络盘同步行为
  - 训练速度稳定性

## 2026-06-18 — StarFlow Train 会话状态（23:46 CST）

- 监控时间：2026-06-18 23:46 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619`（日期后缀 250619，无 r2/r3 后缀）
- 实际进程 PID：`83923`，父进程 `83878`，tmux 会话：`starflow_train`
- 实际进程启动时间：23:28 CST 左右，已运行约 **17 分 26 秒**
- 训练状态：✅ **训练正常运行中，当前 step 17/80000**
- 当前步数：**17 / 80000**
- 训练速度：约 **22.42 s/it**（较 23:41 提升）
- 当前进程状态：主进程 `D`（disk sleep，多线程 `l`，NLWP 305）
- 主进程资源：RSS **4.12 GiB**，**305 线程**
- 启动配置：
  - `NUM_WORKERS=0`
  - `WANDB_MODE=online`
  - `trainer.checkpoint_format=lightweight`
  - `trainer.enable_local_checkpoint_staging=True`
  - `trainer.local_checkpoint_root=/root/temp`
  - `trainer.local_checkpoint_keep_count=2`
- DeepSpeed：ZeRO-2，`train_batch_size=32`，`train_micro_batch_size_per_gpu=4`，`gradient_accumulation_steps=8`
- 模型参数：5071.088M 总量，633.272M 可训练
- W&B：在线同步正常，`output.log` 已生成
- GPU：`P1.gpu.medium`，显存 **39518/40488 MiB（97.6%）**，利用率 **82%**，温度 31°C
- checkpoint：无，预计 step 50 首次保存
- 输出目录大小：94K，内容：`config.yaml`、`config.full.yaml`、`dataset_statistics.json`、`wandb/`、空 `checkpoints/`
- 本地暂存目录大小：20K，内容：`config.yaml`、`config.full.yaml`、`dataset_statistics.json`、空 `checkpoints/`
- 本地 staging 磁盘：⚠️ **root overlay**，总量 30 GiB，可用 29 GiB
- 同步状态：尚未开始，无 `.checkpoint_sync_queue`，无 `checkpoint_sync.log`
- 系统资源：总内存 503 GiB，已用 72 GiB，可用 425 GiB，负载 16.18/19.37/19.26
- 目标 r2 状态：未运行，输出目录不存在
- 风险与建议：
  - 训练速度从 33.17 s/it 提升至 22.42 s/it，估算总时间约 20.7 天
  - GPU 显存已占 97.6%，接近满载
  - `/root/temp` 位于 root overlay（30G），保留 2 个 checkpoint 可能空间不足
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - step 50 首次 lightweight checkpoint 保存时间及空间占用
  - 网络盘同步行为
  - 训练速度稳定性

## 2026-06-18 — StarFlow Train 会话状态（23:41 CST）

- 监控时间：2026-06-18 23:41 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619`（日期后缀 250619，无 r2/r3 后缀）
- 实际进程 PID：`83923`，父进程 `83878`，tmux 会话：`starflow_train`
- 实际进程启动时间：23:28 CST 左右，已运行约 **12 分 41 秒**
- 训练状态：✅ **训练正式开始！当前 step 5/80000**
- 当前步数：**5 / 80000**
- 训练速度：约 **33.17 s/it**（前 5 步估计）
- 当前进程状态：主进程 `D`（disk sleep，多线程 `l`，NLWP 497）
- 主进程资源：RSS **4.06 GiB**，**497 线程**
- 启动配置：
  - `NUM_WORKERS=0`
  - `WANDB_MODE=online`
  - `trainer.checkpoint_format=lightweight`
  - `trainer.enable_local_checkpoint_staging=True`
  - `trainer.local_checkpoint_root=/root/temp`
  - `trainer.local_checkpoint_keep_count=2`
- DeepSpeed：ZeRO-2，`train_batch_size=32`，`train_micro_batch_size_per_gpu=4`，`gradient_accumulation_steps=8`
- 模型参数：5071.088M 总量，633.272M 可训练
- W&B：23:37:29 初始化成功，run URL 已生成
- GPU：`P1.gpu.medium`，显存 **39518/40488 MiB（97.6%）**，利用率 **96%**，温度 31°C
- checkpoint：无
- 输出目录大小：25K，内容：`config.yaml`、`config.full.yaml`、`dataset_statistics.json`、`wandb/`、空 `checkpoints/`
- 本地暂存目录大小：20K，内容：`config.yaml`、`config.full.yaml`、`dataset_statistics.json`、空 `checkpoints/`
- 本地 staging 磁盘：⚠️ **root overlay**，总量 30 GiB，可用 29 GiB
- 同步状态：尚未开始，无 `.checkpoint_sync_queue`，无 `checkpoint_sync.log`
- 系统资源：总内存 503 GiB，已用 69 GiB，可用 428 GiB，负载 19.91/18.48/18.94
- 目标 r2 状态：未运行，输出目录不存在
- 风险与建议：
  - 训练速度较慢（33.17 s/it），完成 80000 步约需 30 天，建议观察速度是否稳定
  - GPU 显存已占 97.6%，接近满载
  - `/root/temp` 位于 root overlay（30G），保留 2 个 checkpoint 可能空间不足
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - step 50 首次 lightweight checkpoint 保存时间及空间占用
  - 网络盘同步行为
  - 训练速度稳定性

## 2026-06-18 — StarFlow Train 会话状态（23:36 CST）

- 监控时间：2026-06-18 23:36 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619`（日期后缀 250619，无 r2/r3 后缀）
- 实际进程 PID：`83923`，父进程 `83878`，tmux 会话：`starflow_train`
- 实际进程启动时间：23:28 CST 左右，已运行约 **7 分 55 秒**
- 训练状态：🔄 **新训练进程初始化持续推进，已进入模型加载/扩展编译阶段**
- 当前步数：初始化中，尚未输出训练步骤
- 当前进程状态：主进程 `Dl+`（D = disk sleep，l = 多线程，NLWP 141）
- 主进程资源：RSS **3.53 GiB**（较 23:32 的 0.38 GiB 显著增长），**141 线程**
- 启动配置：
  - `NUM_WORKERS=0`
  - `WANDB_MODE=online`
  - `trainer.checkpoint_format=lightweight`
  - `trainer.enable_local_checkpoint_staging=True`
  - `trainer.local_checkpoint_root=/root/temp`
  - `trainer.local_checkpoint_keep_count=2`
- DeepSpeed：ZeRO-2，`train_batch_size=32`，`train_micro_batch_size_per_gpu=4`，`gradient_accumulation_steps=8`
- 模型参数：5071.088M 总量，633.272M 可训练
- W&B：尚未初始化（无 `wandb/` 目录），需继续关注
- GPU：`P1.gpu.medium` 已恢复可见，显存 **758/40488 MiB（1.9%）**，利用率 0%，温度 31°C
- checkpoint：无
- 输出目录大小：8.5K，内容：`config.yaml`、`config.full.yaml`、`dataset_statistics.json`、空 `checkpoints/`
- 本地暂存目录大小：20K，内容同上
- 本地 staging 磁盘：⚠️ **root overlay**，总量 30 GiB，可用 29 GiB
- 同步状态：尚未开始，无 `.checkpoint_sync_queue`，无 `checkpoint_sync.log`
- 系统资源：总内存 503 GiB，已用 68 GiB，可用 430 GiB，负载 17.38/19.62/19.51
- 目标 r2 状态：未运行，输出目录不存在
- 风险与建议：
  - 初始化进展顺利，但 W&B 仍是在线模式且尚未初始化，需警惕再次触发 90 秒 init 超时
  - `/root/temp` 位于 root overlay（30G），保留 2 个 checkpoint 可能空间不足
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - W&B 初始化是否成功
  - GPU 显存是否进一步增长
  - 首次 `steps_50` lightweight checkpoint 保存及同步速度

## 2026-06-18 — StarFlow Train 会话状态（23:32 CST）

- 监控时间：2026-06-18 23:32 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250619`（日期后缀 250619，无 r2/r3 后缀）
- 实际进程 PID：`83923`，父进程 `83878`，tmux 会话：`starflow_train`
- 上一个进程 PID：`4712`（已消失）
- 实际进程启动时间：23:28 CST 左右，已运行约 **3 分 25 秒**
- 训练状态：🔄 **新训练进程已启动，正在初始化中（D 状态）**
- 当前步数：初始化中
- 当前进程状态：主进程 `Dl+`（D = disk sleep，l = 多线程，NLWP 64）
- 主进程资源：RSS **0.38 GiB**，**64 线程**
- 启动配置变更：
  - `NUM_WORKERS=0`（从 1 改为 0）
  - `WANDB_MODE=online`
  - `trainer.checkpoint_format=lightweight`
  - `trainer.enable_local_checkpoint_staging=True`
  - `trainer.local_checkpoint_root=/root/temp`
  - `trainer.local_checkpoint_keep_count=2`
- DeepSpeed：ZeRO-2，`train_batch_size=32`，`train_micro_batch_size_per_gpu=4`，`gradient_accumulation_steps=8`
- W&B：新 run_id 初始化中
- GPU：`nvidia-smi` 当前不可见 GPU 信息（SMI/Driver/CUDA 均为 N/A）
- checkpoint：无
- 新输出目录：`/gemini/code/starVLA/playground/Checkpoints/P0-M5-E-H2a-03...250619/`（已创建，当前为空）
- 新本地暂存目录：尚未创建
- 旧输出目录：`/gemini/code/starVLA/playground/Checkpoints/P0-M5-E-H2a-03...250618/`（残留 18K）
- 旧本地暂存目录：`/root/temp/P0-M5-E-H2a-03...250618/`（残留 20K）
- 本地 staging 磁盘：⚠️ **root overlay**，总量 30 GiB，可用 29 GiB
- 同步状态：尚未开始，无 `.checkpoint_sync_queue`，无 `checkpoint_sync.log`
- 系统资源：总内存 503 GiB，已用 64 GiB，可用 434 GiB，负载 20.98/19.64/19.44
- 目标 r2 状态：未运行，输出目录不存在
- 风险与建议：
  - 新进程仍使用 `WANDB_MODE=online`，需警惕再次触发 90 秒 init 超时
  - `/root/temp` 位于 root overlay（30G），保留 2 个 checkpoint 可能空间不足
  - GPU 当前不可见，需继续观察是否恢复
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - W&B 初始化是否成功
  - GPU 是否恢复可见
  - 首次 `steps_50` lightweight checkpoint 保存及同步速度

## 2026-06-18 — StarFlow Train 会话状态（23:27 CST）

- 监控时间：2026-06-18 23:27 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`（无 r2/r3 后缀）
- 实际进程 PID：`4712`，父进程 `4681`，tmux 会话：`starflow_train`
- 实际进程启动时间：23:06 CST，已运行约 **20 分 45 秒**
- 训练状态：❌ **训练初始化失败并挂起：W&B init 90 秒超时，已挂起约 10 分钟**
- 当前步数：初始化失败，未开始训练
- 当前进程状态：主进程 `SLl+`（S = sleeping，L = 多线程），**挂起未退出**
- 主进程资源：RSS **3.52 GiB**，**195 线程**
- 子进程：`wandb-core` PID 39625、`torch/inductor/compile_worker` PID 37569 仍在运行
- 启动配置：
  - `NUM_WORKERS=1`
  - `trainer.checkpoint_format=lightweight`
  - `trainer.enable_local_checkpoint_staging=True`
  - `trainer.local_checkpoint_root=/root/temp`
  - `trainer.local_checkpoint_keep_count=2`
- DeepSpeed：ZeRO-2，`train_batch_size=32`，`train_micro_batch_size_per_gpu=4`，`gradient_accumulation_steps=8`
- W&B：run 创建于 23:16:09，`debug.log` 为空，`files/` 为空，init 超时失败
- GPU：`P1.gpu.medium`，显存 **16718/40488 MiB（41.3%）**，利用率 0%，温度 31°C
- checkpoint：无
- 输出目录大小：18K，内容：`config.yaml`、`config.full.yaml`、`dataset_statistics.json`、`wandb/`、空 `checkpoints/`
- 本地暂存目录大小：20K，内容：`config.yaml`、`config.full.yaml`、`dataset_statistics.json`、空 `checkpoints/`
- 本地 staging 磁盘：⚠️ **root overlay**，总量 30 GiB，可用 29 GiB
- 同步状态：尚未开始，无 `.checkpoint_sync_queue`，无 `checkpoint_sync.log`
- 系统资源：总内存 503 GiB，已用 72 GiB，可用 426 GiB，负载 19.16/20.30/19.63
- 目标 r2 状态：未运行，输出目录不存在
- 风险与建议：
  - ⚠️ **当前进程已因 W&B init 超时挂起约 10 分钟，需尽快人工干预**
  - 建议方案：终止 PID 4712，使用 `WANDB_MODE=offline` 重新启动
  - `/root/temp` 位于 root overlay（30G），保留 2 个 checkpoint 可能空间不足
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - 用户是否终止/重启训练
  - 若重试，W&B 是否仍超时

## 2026-06-18 — StarFlow Train 会话状态（23:21 CST）

- 监控时间：2026-06-18 23:21 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`（无 r2/r3 后缀）
- 实际进程 PID：`4712`，父进程 `4681`，tmux 会话：`starflow_train`
- 实际进程启动时间：23:06 CST，已运行约 **15 分 26 秒**
- 训练状态：❌ **训练初始化失败并挂起：W&B init 90 秒超时**
- 当前步数：初始化失败，未开始训练
- 当前进程状态：主进程 `SLl+`（S = sleeping，L = 多线程），**挂起未退出**
- 主进程资源：RSS **3.52 GiB**，**195 线程**
- 启动配置：
  - `NUM_WORKERS=1`
  - `trainer.checkpoint_format=lightweight`
  - `trainer.enable_local_checkpoint_staging=True`
  - `trainer.local_checkpoint_root=/root/temp`
  - `trainer.local_checkpoint_keep_count=2`
- DeepSpeed：ZeRO-2，`train_batch_size=32`，`train_micro_batch_size_per_gpu=4`，`gradient_accumulation_steps=8`
- W&B：run 创建于 23:16:09，但 `debug.log` 为空，`files/` 为空，init 超时失败；`wandb-core` 子进程 PID 39625 仍在运行
- GPU：`P1.gpu.medium`，显存 **16718/40488 MiB（41.3%）**，利用率 0%，温度 31°C
- checkpoint：无
- 输出目录大小：16K，内容：`config.yaml`、`config.full.yaml`、`dataset_statistics.json`、`wandb/`、空 `checkpoints/`
- 本地暂存目录大小：20K，内容：`config.yaml`、`config.full.yaml`、`dataset_statistics.json`、空 `checkpoints/`
- 本地 staging 磁盘：⚠️ **root overlay**，总量 30 GiB，可用 29 GiB
- 同步状态：尚未开始，无 `.checkpoint_sync_queue`，无 `checkpoint_sync.log`
- 系统资源：总内存 503 GiB，已用 68 GiB，可用 429 GiB，负载 13.93/17.66/18.58
- 目标 r2 状态：未运行，输出目录不存在
- 风险与建议：
  - ⚠️ **当前进程已因 W&B init 超时挂起，需人工干预**
  - 建议方案：终止当前进程，使用 `WANDB_MODE=offline` 重新启动，或检查网络/W&B 服务端状态
  - `/root/temp` 位于 root overlay（30G），保留 2 个 checkpoint 可能空间不足
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - 用户是否终止/重启训练
  - 若重试，W&B 是否仍超时

## 2026-06-18 — StarFlow Train 会话状态（23:19 CST）

- 监控时间：2026-06-18 23:19 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`（无 r2/r3 后缀）
- 实际进程 PID：`4712`，父进程 `4681`，tmux 会话：`starflow_train`
- 实际进程启动时间：23:06 CST，已运行约 **13 分 19 秒**
- 训练状态：🔄 **训练进程仍在初始化中，尚未输出训练步骤**
- 当前步数：初始化中
- 当前进程状态：主进程 `S`（sleeping/interruptible，多线程 `l`，NLWP 195）
- 主进程资源：RSS **3.52 GiB**，**195 线程**
- 启动配置：
  - `NUM_WORKERS=1`
  - `trainer.checkpoint_format=lightweight`
  - `trainer.enable_local_checkpoint_staging=True`
  - `trainer.local_checkpoint_root=/root/temp`
  - `trainer.local_checkpoint_keep_count=2`
- DeepSpeed：ZeRO-2，`train_batch_size=32`，`train_micro_batch_size_per_gpu=4`，`gradient_accumulation_steps=8`
- W&B：run 已创建（23:16:09），output.log 尚未生成
- GPU：`P1.gpu.medium`，显存 **16718/40488 MiB（41.3%）**，利用率 0%，温度 31°C
- checkpoint：无
- 输出目录：`.../Checkpoints/P0-M5-E-H2a-03...250618/` 已有 `config.yaml`、`config.full.yaml`、`dataset_statistics.json`、`wandb/`
- 本地暂存目录：`/root/temp/.../checkpoints/`（已创建，当前为空）
- 本地 staging 磁盘：⚠️ **root overlay**，总量 30 GiB，可用 29 GiB
- 同步状态：尚未开始，无 `.checkpoint_sync_queue`，无 `checkpoint_sync.log`
- 系统资源：总内存 503 GiB，已用 68 GiB，可用 429 GiB，负载 13.40/17.76/18.64
- 目标 r2 状态：未运行，输出目录不存在
- 风险与建议：
  - `/root/temp` 位于 root overlay（30G），保留 2 个 checkpoint 可能空间不足
  - 初始化已持续 13 分钟以上，需继续观察是否成功进入训练循环
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - 训练初始化是否成功
  - 首次 `steps_50` lightweight checkpoint 保存及同步速度
  - `/root/temp` 空间消耗

## 2026-06-18 — StarFlow Train 会话状态（23:12 CST）

- 监控时间：2026-06-18 23:12 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`（无 r2/r3 后缀）
- 实际进程 PID：`4712`，父进程 `4681`，tmux 会话：`starflow_train`
- 实际进程启动时间：23:06 CST，已运行约 **5 分 52 秒**
- 训练状态：🔄 **训练进程初始化中，进程仍在 D 状态**
- 当前步数：初始化中
- 当前进程状态：主进程 `Dl+`（D = disk sleep，l = 多线程）
- 主进程资源：RSS **0.44 GiB**，**64 线程**
- 启动配置：
  - `NUM_WORKERS=1`
  - `trainer.checkpoint_format=lightweight`
  - `trainer.enable_local_checkpoint_staging=True`
  - `trainer.local_checkpoint_root=/root/temp`
  - `trainer.local_checkpoint_keep_count=2`
- DeepSpeed：ZeRO-2，`train_batch_size=32`，`train_micro_batch_size_per_gpu=4`，`gradient_accumulation_steps=8`
- W&B：待初始化，无 output.log
- GPU：`nvidia-smi` 当前仍不可读 GPU 信息
- checkpoint：无
- 输出目录：`.../Checkpoints/P0-M5-E-H2a-03...250618/`（23:04 创建，仍为空）
- 本地暂存目录：不存在
- 本地 staging 磁盘：⚠️ **root overlay**，总量 30 GiB，可用 29 GiB
- 同步状态：尚未开始
- 系统资源：总内存 503 GiB，已用 63 GiB，可用 434 GiB，负载 23.52/20.47/19.51
- 目标 r2 状态：未运行，输出目录不存在
- 风险与建议：
  - `/root/temp` 位于 root overlay（30G），保留 2 个 checkpoint 可能空间不足
  - 初始化时间较长（近 6 分钟），需继续观察是否成功
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - 训练初始化是否成功
  - GPU 是否恢复可见
  - W&B 是否连接

## 2026-06-18 — StarFlow Train 会话状态（23:07 CST）

- 监控时间：2026-06-18 23:07 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`（无 r2/r3 后缀）
- 实际进程 PID：`4712`，父进程 `4681`，tmux 会话：`starflow_train`
- 上一个进程 PID：`275915`（已退出）
- 实际进程启动时间：23:06 CST，已运行约 **1 分 07 秒**
- 训练状态：🔄 **训练进程已重新启动，初始化中**
- 当前步数：初始化中
- 当前进程状态：主进程 `Dl+`（D = disk sleep，l = 多线程）
- 主进程资源：RSS **0.97 GiB**，**64 线程**
- 启动配置变更：
  - `NUM_WORKERS` 从 0 改为 **1**
  - `trainer.checkpoint_format=lightweight`
  - `trainer.enable_local_checkpoint_staging=True`
  - `trainer.local_checkpoint_root=/root/temp`
  - `trainer.local_checkpoint_keep_count=2`
- DeepSpeed：ZeRO-2，`train_batch_size=32`，`train_micro_batch_size_per_gpu=4`，`gradient_accumulation_steps=8`
- W&B：待初始化
- GPU：`nvidia-smi` 当前不可读 GPU 信息
- checkpoint：无
- 输出目录：`.../Checkpoints/P0-M5-E-H2a-03...250618/`（23:04 重新创建，当前为空）
- 本地暂存目录：不存在
- 本地 staging 磁盘：⚠️ **root overlay**，总量 30 GiB，可用 29 GiB
- 同步状态：尚未开始
- 系统资源：总内存 503 GiB，已用 64 GiB，可用 434 GiB，负载 15.75/16.94/18.30
- 目标 r2 状态：未运行，输出目录不存在
- 风险与建议：
  - `/root/temp` 位于 root overlay（30G），保留 2 个 checkpoint 可能空间不足
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - 训练初始化是否成功
  - GPU 是否恢复可见
  - W&B 是否连接
  - 本地 `/root/temp` staging 目录何时创建

## 2026-06-18 — StarFlow Train 会话状态（23:02 CST）

- 监控时间：2026-06-18 23:02 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`（无 r2/r3 后缀）
- 实际进程 PID：`275915`，父进程 `275903`，tmux 会话：`starflow_train`
- 实际进程启动时间：22:50 CST，已运行约 **12 分 38 秒**
- 训练状态：🔄 **新训练进程初始化中**
- 当前步数：初始化中，尚未输出训练步骤
- 当前进程状态：主进程 `SLl+`（S = sleeping，L = 多线程）
- 主进程资源：RSS **3.54 GiB**（较 22:58 增长），**195 线程**（较 22:58 增长）
- 启动配置：
  - `trainer.checkpoint_format=lightweight`
  - `trainer.enable_local_checkpoint_staging=True`
  - `trainer.local_checkpoint_root=/root/temp`
  - `trainer.local_checkpoint_keep_count=2`
- DeepSpeed：ZeRO-2，`train_batch_size=32`，`train_micro_batch_size_per_gpu=4`，`gradient_accumulation_steps=8`
- W&B：run 已创建（22:59:39），output.log 尚未生成
- GPU：`P1.gpu.medium`，显存 **16718/40488 MiB（41.3%）**，利用率 0%，温度 31°C
- checkpoint：无
- 输出目录：`.../Checkpoints/P0-M5-E-H2a-03...250618/` 已生成 `config.yaml`、`config.full.yaml`、`dataset_statistics.json`、`wandb/`
- 本地暂存目录：`/root/temp/.../checkpoints/`（已创建，当前为空）
- 本地 staging 磁盘：⚠️ **root overlay**，总量 30 GiB，可用 29 GiB
- 同步状态：尚未开始，无 `.checkpoint_sync_queue`，无 `checkpoint_sync.log`
- 系统资源：总内存 503 GiB，已用 70 GiB，可用 427 GiB，负载 17.61/19.57/19.34
- 目标 r2 状态：未运行，输出目录不存在
- 风险与建议：
  - `/root/temp` 位于 root overlay（30G），保留 2 个 checkpoint 可能空间不足
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - 训练初始化是否成功
  - W&B output.log 何时生成
  - 首次 `steps_50` lightweight checkpoint 保存及同步速度

## 2026-06-18 — StarFlow Train 会话状态（22:58 CST）

- 监控时间：2026-06-18 22:58 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`（无 r2/r3 后缀）
- 实际进程 PID：`275915`，父进程 `275903`，tmux 会话：`starflow_train`
- 实际进程启动时间：22:50 CST，已运行约 **7 分 55 秒**
- 训练状态：🔄 **新训练进程初始化中**
- 当前步数：初始化中，尚未输出训练步骤
- 当前进程状态：主进程 `DLl+`（D = disk sleep，L = 多线程）
- 主进程资源：RSS **3.15 GiB**（较 22:55 增长），**136 线程**（较 22:55 增长）
- 启动配置：
  - `trainer.checkpoint_format=lightweight`
  - `trainer.enable_local_checkpoint_staging=True`
  - `trainer.local_checkpoint_root=/root/temp`
  - `trainer.local_checkpoint_keep_count=2`
- DeepSpeed：ZeRO-2，`train_batch_size=32`，`train_micro_batch_size_per_gpu=4`，`gradient_accumulation_steps=8`
- W&B：初始化中
- GPU：`P1.gpu.medium`，显存 40488 MiB，当前 0 MiB 已用，温度 31°C（GPU 已恢复可见）
- checkpoint：无
- 输出目录：`/gemini/code/starVLA/playground/Checkpoints/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618/`（已创建，`checkpoints/` 子目录已创建，均为空）
- 本地暂存目录：`/root/temp/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618/checkpoints/`（已创建，当前为空）
- 本地 staging 磁盘：⚠️ **root overlay**，总量 30 GiB，可用 29 GiB
- 同步状态：尚未开始，无 `.checkpoint_sync_queue`，无 `checkpoint_sync.log`
- 系统资源：总内存 503 GiB，已用 67 GiB，可用 430 GiB，负载 19.92/20.29/19.37
- 目标 r2 状态：未运行，输出目录不存在
- 风险与建议：
  - `/root/temp` 位于 root overlay（30G），保留 2 个 checkpoint 可能空间不足
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - 训练初始化是否成功
  - W&B 是否连接
  - 首次 `steps_50` lightweight checkpoint 保存及同步速度

## 2026-06-18 — StarFlow Train 会话状态（22:55 CST）

- 监控时间：2026-06-18 22:55 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`（无 r2/r3 后缀）
- 实际进程 PID：`275915`，父进程 `275903`，tmux 会话：`starflow_train`
- 实际进程启动时间：22:50 CST，已运行约 **5 分 46 秒**
- 训练状态：🔄 **新训练进程初始化中**
- 当前步数：初始化中，尚未输出训练步骤
- 当前进程状态：主进程 `Dl+`（D = disk sleep，l = 多线程）
- 主进程资源：RSS **0.47 GiB**，**64 线程**
- 启动配置：
  - `trainer.checkpoint_format=lightweight`
  - `trainer.enable_local_checkpoint_staging=True`
  - `trainer.local_checkpoint_root=/root/temp`
  - `trainer.local_checkpoint_keep_count=2`
- DeepSpeed：ZeRO-2，`train_batch_size=32`，`train_micro_batch_size_per_gpu=4`，`gradient_accumulation_steps=8`
- W&B：初始化中
- GPU：`nvidia-smi` 当前仍不可读 GPU 信息
- checkpoint：无
- 输出目录：`/gemini/code/starVLA/playground/Checkpoints/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618/`（已重新创建，当前为空）
- 本地暂存目录：`/root/temp/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618/`（尚未创建）
- 本地 staging 磁盘：⚠️ **root overlay**，总量 30 GiB，可用 29 GiB
- 同步状态：尚未开始，无 `.checkpoint_sync_queue`，无 `checkpoint_sync.log`
- 系统资源：总内存 503 GiB，已用 64 GiB，可用 434 GiB，负载 16.44/18.87/18.80
- 目标 r2 状态：未运行，输出目录不存在
- 风险与建议：
  - `/root/temp` 位于 root overlay（30G），保留 2 个 checkpoint（预计 36G）可能空间不足，建议更换为独立大容量磁盘路径
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - 训练初始化是否成功
  - W&B 是否连接
  - 本地 `/root/temp` staging 目录何时创建
  - 首次 `steps_50` lightweight checkpoint 保存及同步速度

## 2026-06-18 — StarFlow Train 会话状态（22:52 CST）

- 监控时间：2026-06-18 22:52 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`（无 r2/r3 后缀）
- 实际进程 PID：`275915`，父进程 `275903`，tmux 会话：`starflow_train`
- 实际进程启动时间：22:50 CST，已运行约 **2 分 48 秒**
- 训练状态：🔄 **新训练进程已启动，初始化中**
- 当前步数：初始化中，尚未输出训练步骤
- 当前进程状态：主进程 `Dl+`（D = disk sleep，l = 多线程）
- 主进程资源：RSS **0.35 GiB**，**64 线程**
- 启动配置变更：
  - `trainer.checkpoint_format=lightweight`
  - `trainer.enable_local_checkpoint_staging=True`
  - `trainer.local_checkpoint_root=/root/temp`
  - `trainer.local_checkpoint_keep_count=2`
- DeepSpeed：ZeRO-2，`train_batch_size=32`，`train_micro_batch_size_per_gpu=4`，`gradient_accumulation_steps=8`
- W&B：待初始化完成
- GPU：`nvidia-smi` 当前仍不可读 GPU 信息
- checkpoint：无
- 输出目录：`/gemini/code/starVLA/playground/Checkpoints/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618/`（已重新创建，当前为空）
- 本地暂存目录：`/root/temp/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618/`（尚未创建）
- 系统资源：总内存 503 GiB，已用 65 GiB，可用 432 GiB，负载 24.15/19.88/18.97
- 目标 r2 状态：未运行，输出目录不存在
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - 训练初始化是否成功
  - W&B 是否连接
  - 首次 `steps_50` lightweight checkpoint 保存速度
  - 本地 `/root/temp` staging 是否生效

## 2026-06-18 — StarFlow Train 会话状态（22:48 CST）

- 监控时间：2026-06-18 22:48 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：无
- 实际进程 PID：无
- 训练状态：❌ **训练已停止，未重新启动**
- 当前步数：无
- 当前进程状态：无 train_starvla 进程
- 输出目录：不存在
- tmux 会话：`starflow_train` 仍存在但为空
- GPU：`nvidia-smi` 当前仍不可读 GPU 信息
- checkpoint：无
- 系统资源：总内存 503 GiB，已用 63 GiB，可用 434 GiB，负载 13.60/20.00/18.85
- 目标 r2 状态：未运行，输出目录不存在
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续建议：
  - 确认 GPU 可用后再启动训练
  - 改用 lightweight/deepspeed_state checkpoint 格式或配置 local fast disk staging

## 2026-06-18 — StarFlow Train 会话状态（22:43 CST）

- 监控时间：2026-06-18 22:43 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：无
- 实际进程 PID：无
- 训练状态：❌ **训练已停止，未重新启动**
- 当前步数：无
- 当前进程状态：无 train_starvla 进程
- 输出目录：不存在
- tmux 会话：`starflow_train` 仍存在但为空
- GPU：`nvidia-smi` 当前仍不可读 GPU 信息
- checkpoint：无
- 系统资源：总内存 503 GiB，已用 62 GiB，可用 436 GiB，负载 20.57/19.46/17.84
- 目标 r2 状态：未运行，输出目录不存在
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续建议：
  - 确认 GPU 可用后再启动训练
  - 改用 lightweight/deepspeed_state checkpoint 格式或配置 local fast disk staging

## 2026-06-18 — StarFlow Train 会话状态（22:38 CST）

- 监控时间：2026-06-18 22:38 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：无
- 实际进程 PID：无
- 训练状态：❌ **训练已停止，未重新启动**
- 当前步数：无
- 当前进程状态：无 train_starvla 进程
- 输出目录：不存在
- tmux 会话：`starflow_train` 仍存在但为空
- GPU：`nvidia-smi` 当前仍不可读 GPU 信息
- checkpoint：无
- 系统资源：总内存 503 GiB，已用 62 GiB，可用 436 GiB，负载 20.89/19.70/17.46
- 目标 r2 状态：未运行，输出目录不存在
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续建议：
  - 确认 GPU 可用后再启动训练
  - 改用 lightweight/deepspeed_state checkpoint 格式或配置 local fast disk staging

## 2026-06-18 — StarFlow Train 会话状态（22:33 CST）

- 监控时间：2026-06-18 22:33 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：无
- 实际进程 PID：无
- 训练状态：❌ **训练已停止，输出目录已删除**
- 当前步数：无
- 当前进程状态：无 train_starvla 进程
- 输出目录：已删除（`rm -rf .../P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618/`）
- tmux 会话：`starflow_train` 仍存在但为空
- W&B：历史记录 `https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla/runs/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`
- GPU：`nvidia-smi` 当前不可读 GPU 信息
- checkpoint：无（目录已删除）
- 系统资源：总内存 503 GiB，已用 61 GiB，可用 436 GiB，负载 21.63/18.53/16.38
- 目标 r2 状态：未运行，输出目录不存在
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续建议：
  - 确认 GPU 可用后再启动训练
  - 改用 lightweight/deepspeed_state checkpoint 格式或配置 local fast disk staging
  - 如需 universal 格式，离线转换

## 2026-06-18 — StarFlow Train 会话状态（22:29 CST）

- 监控时间：2026-06-18 22:29 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`（无 r2/r3 后缀）
- 实际进程 PID：`51127` **已退出**
- 实际进程启动时间：21:32 CST，已运行约 **52 分钟后中断**
- 训练状态：❌ **已中断**：Universal 转换在 600/672 参数处收到 `KeyboardInterrupt`
- 当前步数：`50 / 80000`（保存/转换中中断）
- step 50 loss：`action_dit_loss=0.6000`
- 中断位置：`universal.py:244` `_write_universal` → `torch.save()`
- 当前进程状态：**不存在**
- DeepSpeed：ZeRO-2
- W&B：训练进程已退出，同步停止
- GPU：`P1.gpu.medium`，**无运行进程**，显存已释放
- checkpoint：**不完整**：`steps_50/` 6.9 GiB（部分写入），临时目录未清理
  - `.steps_50_lightweight_tmp`：18 GiB
  - `.steps_50_zero_tmp`：26 GiB
  - 临时目录合计：44 GiB
- 输出目录文件：`config.yaml`、`config.full.yaml`、`dataset_statistics.json`、wandb/
- 系统资源：总内存 503 GiB，已用 **60 GiB**（较 22:24 下降 15 GiB），可用 437 GiB，负载 17.36/16.07/15.11
- 目标 r2 状态：未运行，输出目录不存在
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续建议：
  - 清理 `checkpoints/.steps_50*` 和 `checkpoints/steps_50/`
  - 如需继续训练，建议改用 `trainer.checkpoint_format=lightweight` 或 `deepspeed_state`，或配置本地 fast disk staging
  - 如需 universal 格式，建议离线转换

## 2026-06-18 — StarFlow Train 会话状态（22:24 CST）

- 监控时间：2026-06-18 22:24 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`（无 r2/r3 后缀）
- 实际进程 PID：`51127`，父进程 `51122`，tmux 会话：`starflow_train`
- 实际进程启动时间：21:32 CST，已运行约 **52 分 09 秒**
- 训练状态：⏳ **step 50 首次 checkpoint 保存中，Universal 转换已推进到 400/672 参数**
- 当前步数：`50 / 80000`（未推进）
- step 50 loss：`action_dit_loss=0.6000`
- 当前进程状态：主进程 `SLl+`（S = sleeping，L = 多线程）
- 主进程资源：RSS **25.05 GiB**（基本稳定），**210 线程**
- DeepSpeed：ZeRO-2，`train_batch_size=32`，`train_micro_batch_size_per_gpu=4`，`gradient_accumulation_steps=8`
- W&B：✅ 已连接并同步
- W&B Run：`https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla/runs/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`
- GPU：`P1.gpu.medium`，显存 **39518/40488 MiB（97.6%）**，利用率 0%，温度 31°C
- checkpoint：**Universal 转换进行中，已写入 400/672 参数**；最终 `steps_50/` 已有 **5.1 GiB** 内容
  - `.steps_50_lightweight_tmp`：**18 GiB**
  - `.steps_50_zero_tmp`：**26 GiB**
  - 临时目录合计：**44 GiB**
  - 保存已耗时：约 **25 分钟**
- 输出目录文件：`config.yaml`、`config.full.yaml`、`dataset_statistics.json`、wandb/
- 系统资源：总内存 503 GiB，已用 75 GiB，可用 424 GiB，负载 10.41/13.32/14.28
- 目标 r2 状态：未运行，输出目录不存在
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - Universal 转换何时完成（672/672）
  - 临时目录是否清理
  - 保存完成后内存是否回落
  - 训练是否恢复并推进到 step 51+

## 2026-06-18 — StarFlow Train 会话状态（22:19 CST）

- 监控时间：2026-06-18 22:19 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`（无 r2/r3 后缀）
- 实际进程 PID：`51127`，父进程 `51122`，tmux 会话：`starflow_train`
- 实际进程启动时间：21:32 CST，已运行约 **47 分 25 秒**
- 训练状态：⏳ **step 50 首次 checkpoint 保存中，Universal 转换已推进到 200/672 参数**
- 当前步数：`50 / 80000`（未推进）
- step 50 loss：`action_dit_loss=0.6000`
- 当前进程状态：主进程 `SLl+`（S = sleeping，L = 多线程）
- 主进程资源：RSS **24.94 GiB**（增速放缓），**210 线程**
- DeepSpeed：ZeRO-2，`train_batch_size=32`，`train_micro_batch_size_per_gpu=4`，`gradient_accumulation_steps=8`
- W&B：✅ 已连接并同步
- W&B Run：`https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla/runs/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`
- GPU：`P1.gpu.medium`，显存 **39518/40488 MiB（97.6%）**，利用率 0%，温度 31°C
- checkpoint：**Universal 转换进行中，已写入 200/672 参数**；最终 `steps_50/` 已有 **2.4 GiB** 内容
  - `.steps_50_lightweight_tmp`：**18 GiB**
  - `.steps_50_zero_tmp`：**26 GiB**
  - 临时目录合计：**44 GiB**
  - 保存已耗时：约 **20 分钟**
- 输出目录文件：`config.yaml`、`config.full.yaml`、`dataset_statistics.json`、wandb/
- 系统资源：总内存 503 GiB，已用 75 GiB，可用 424 GiB，负载 13.51/16.30/15.20
- 目标 r2 状态：未运行，输出目录不存在
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - Universal 转换何时完成（672/672）
  - 临时目录是否清理
  - 保存完成后内存是否回落
  - 训练是否恢复并推进到 step 51+

## 2026-06-18 — StarFlow Train 会话状态（22:14 CST）

- 监控时间：2026-06-18 22:14 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`（无 r2/r3 后缀）
- 实际进程 PID：`51127`，父进程 `51122`，tmux 会话：`starflow_train`
- 实际进程启动时间：21:32 CST，已运行约 **42 分 37 秒**
- 训练状态：⚠️ **step 50 首次 checkpoint 保存中，Universal 转换已持续约 15 分钟，训练长时间暂停**
- 当前步数：`50 / 80000`（未推进）
- step 50 loss：`action_dit_loss=0.6000`
- 当前进程状态：主进程 `SLl+`（S = sleeping，L = 多线程）
- 主进程资源：RSS **23.61 GiB**（较 22:10 的 16.75 GiB 继续大幅上升），**210 线程**
- DeepSpeed：ZeRO-2，`train_batch_size=32`，`train_micro_batch_size_per_gpu=4`，`gradient_accumulation_steps=8`
- W&B：✅ 已连接并同步
- W&B Run：`https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla/runs/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`
- GPU：`P1.gpu.medium`，显存 **39518/40488 MiB（97.6%）**，利用率 0%，温度 31°C
- checkpoint：**Universal 转换进度缓慢**；最终 `steps_50/` 仍为空
  - `.steps_50_lightweight_tmp`：**18 GiB**
  - `.steps_50_zero_tmp`：**26 GiB**
  - 临时目录合计：**44 GiB**
  - 最新日志：仍停留在 "Loading optimizer shard 0"
  - 保存已耗时：约 **15 分钟**
- 输出目录文件：`config.yaml`、`config.full.yaml`、`dataset_statistics.json`、wandb/
- 系统资源：总内存 503 GiB，已用 **74 GiB**（较 22:10 增加 3 GiB），可用 425 GiB，负载 **23.53/19.72/15.72**
- 目标 r2 状态：未运行，输出目录不存在
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 风险与建议：
  - Universal 转换已耗时 15 分钟，进度停滞在加载 optimizer shard，可能因网络盘 I/O 或转换算法效率低
  - 内存快速上涨（23.61 GiB 且仍在升），需警惕 OOM
  - 若后续 5–10 分钟仍未完成，建议中断并改用 `trainer.checkpoint_format=lightweight` 或 `deepspeed_state`，或配置本地 fast disk staging

## 2026-06-18 — StarFlow Train 会话状态（22:10 CST）

- 监控时间：2026-06-18 22:10 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`（无 r2/r3 后缀）
- 实际进程 PID：`51127`，父进程 `51122`，tmux 会话：`starflow_train`
- 实际进程启动时间：21:32 CST，已运行约 **37 分 51 秒**
- 训练状态：⏳ **step 50 首次 checkpoint 保存中，Universal 转换进行中，训练暂停**
- 当前步数：`50 / 80000`（未推进）
- step 50 loss：`action_dit_loss=0.6000`
- 当前进程状态：主进程 `SLl+`（S = sleeping，L = 多线程）
- 主进程资源：RSS **16.75 GiB**（较 22:05 的 10.75 GiB 继续上升），**210 线程**
- DeepSpeed：ZeRO-2，`train_batch_size=32`，`train_micro_batch_size_per_gpu=4`，`gradient_accumulation_steps=8`
- W&B：✅ 已连接并同步
- W&B Run：`https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla/runs/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`
- GPU：`P1.gpu.medium`，显存 **39518/40488 MiB（97.6%）**，利用率 0%，温度 31°C
- checkpoint：**Universal 转换进行中**；最终 `steps_50/` 仍为空
  - `.steps_50_lightweight_tmp`：**18 GiB**
  - `.steps_50_zero_tmp`：**26 GiB**
  - 临时目录合计：**44 GiB**
  - 转换最新日志：正在加载 `optimizer_rank_00000.pt`
  - 保存已耗时：约 11 分钟
- 输出目录文件：`config.yaml`、`config.full.yaml`、`dataset_statistics.json`、wandb/
- 系统资源：总内存 503 GiB，已用 **71 GiB**（较 22:05 增加 2 GiB），可用 428 GiB，负载 14.37/15.93/13.59
- 目标 r2 状态：未运行，输出目录不存在
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - `steps_50/` 最终内容何时写入完成
  - 转换完成后内存是否回落
  - 临时目录是否清理
  - 训练是否恢复并推进到 step 51+

## 2026-06-18 — StarFlow Train 会话状态（22:05 CST）

- 监控时间：2026-06-18 22:05 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`（无 r2/r3 后缀）
- 实际进程 PID：`51127`，父进程 `51122`，tmux 会话：`starflow_train`
- 实际进程启动时间：21:32 CST，已运行约 **33 分钟**
- 训练状态：⏳ **step 50 首次 checkpoint 保存进行中，DeepSpeed Universal 转换阶段**
- 当前步数：`50 / 80000`（保存中，未继续训练）
- step 50 loss：`action_dit_loss=0.6000`
- 当前进程状态：主进程 `DLl+`（D = disk sleep，L = 多线程）
- 主进程资源：RSS **10.75 GiB**（较 22:00 的 7.05 GiB 继续上升），**210 线程**
- DeepSpeed：ZeRO-2，`train_batch_size=32`，`train_micro_batch_size_per_gpu=4`，`gradient_accumulation_steps=8`
- W&B：✅ 已连接并同步
- W&B Run：`https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla/runs/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`
- GPU：`P1.gpu.medium`，显存 **39518/40488 MiB（97.6%）**，利用率 0%（保存阶段），温度 31°C
- checkpoint：**保存进行中**；最终目录 `steps_50/` 已创建但为空
  - `.steps_50_lightweight_tmp`：**18 GiB**
  - `.steps_50_zero_tmp`：**26 GiB**
  - 临时目录合计：**44 GiB**
  - 关键时间线：22:01 DeepSpeed model state → 22:04 完成 → 22:05 zero optim → 22:05 universal 转换开始
- 输出目录文件：`config.yaml`、`config.full.yaml`、`dataset_statistics.json`、wandb/
- 系统资源：总内存 503 GiB，已用 69 GiB，可用 430 GiB，负载 8.71/11.12/11.50
- 目标 r2 状态：未运行，输出目录不存在
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - `steps_50/` 最终内容何时写入完成
  - 临时目录是否清理
  - 保存完成后内存是否回落
  - 训练是否恢复并推进到 step 51+

## 2026-06-18 — StarFlow Train 会话状态（22:00 CST）

- 监控时间：2026-06-18 22:00 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`（无 r2/r3 后缀）
- 实际进程 PID：`51127`，父进程 `51122`，tmux 会话：`starflow_train`
- 实际进程启动时间：21:32 CST，已运行约 **28 分 22 秒**
- 训练状态：✅ **训练中，step 50 到达，首次 checkpoint 保存进行中**
- 当前步数：`50 / 80000`
- step 50 loss：`action_dit_loss=0.6000`，`learning_rate/action_model=1.00e-4`，`learning_rate/base=2.50e-5`
- 训练速度：`19.58 s/it`，`data_times=0.659`，`model_times=1.471`
- 当前进程状态：主进程 `SLl+`（S = sleeping，L = 多线程）
- 主进程资源：RSS **7.05 GiB**（较 21:55 的 4.21 GiB 上升），**210 线程**
- DeepSpeed：ZeRO-2，`train_batch_size=32`，`train_micro_batch_size_per_gpu=4`，`gradient_accumulation_steps=8`
- W&B：✅ 已连接并同步
- W&B Run：`https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla/runs/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`
- GPU：`P1.gpu.medium`，显存 **39518/40488 MiB（97.6%）**，利用率 0%（保存阶段），温度 31°C
- checkpoint：**首次保存进行中**：`checkpoints/.steps_50_lightweight_tmp/` 已创建，大小 **11 GiB**；最终 `steps_50/` 尚未生成
- 临时目录内容：3 个 model shard（共 ~11.4 GiB）+ optimizer/scheduler/scaler/config 等
- 输出目录文件：`config.yaml`、`config.full.yaml`、`dataset_statistics.json`、wandb/
- 系统资源：总内存 503 GiB，已用 **69 GiB**（较 21:55 增加 4 GiB），可用 430 GiB，负载 12.98/11.69/11.51
- 目标 r2 状态：未运行，输出目录不存在
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - `steps_50` universal 转换是否成功完成
  - 临时目录是否清理
  - 保存完成后内存是否回落
  - 训练是否继续推进到 step 51+

## 2026-06-18 — StarFlow Train 会话状态（21:55 CST）

- 监控时间：2026-06-18 21:55 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`（无 r2/r3 后缀）
- 实际进程 PID：`51127`，父进程 `51122`，tmux 会话：`starflow_train`
- 实际进程启动时间：21:32 CST，已运行约 **23 分 35 秒**
- 训练状态：✅ **训练中，步数 38/80000**
- 当前步数：`38 / 80000`
- 训练速度：`21.83 s/it`，`data_times=2.186`，`model_times=1.164`
- 当前进程状态：主进程 `DLl+`（D = disk sleep，L = 多线程）
- 主进程资源：RSS **4.21 GiB**，**401 线程**（较 21:51 的 689 回落）
- DeepSpeed：ZeRO-2，`train_batch_size=32`，`train_micro_batch_size_per_gpu=4`，`gradient_accumulation_steps=8`
- W&B：✅ 已连接并同步
- W&B Run：`https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla/runs/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`
- GPU：`P1.gpu.medium`，显存 **39518/40488 MiB（97.6%）**，利用率 20%，温度 31°C
- checkpoint：`checkpoints/` 目录已创建但为空；预计首次保存 `steps_50`（约 4–5 分钟后）
- 输出目录文件：`config.yaml`、`config.full.yaml`、`dataset_statistics.json`、wandb/
- 系统资源：总内存 503 GiB，已用 65 GiB，可用 433 GiB，负载 9.58/11.01/11.38
- 目标 r2 状态：未运行，输出目录不存在
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - 首个 `step 50` loss 日志
  - `steps_50` checkpoint 保存时临时目录、内存与显存峰值
  - 线程数是否再次反弹

## 2026-06-18 — StarFlow Train 会话状态（21:51 CST）

- 监控时间：2026-06-18 21:51 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`（无 r2/r3 后缀）
- 实际进程 PID：`51127`，父进程 `51122`，tmux 会话：`starflow_train`
- 实际进程启动时间：21:32 CST，已运行约 **18 分 50 秒**
- 训练状态：✅ **训练中，步数 24/80000**
- 当前步数：`24 / 80000`
- 训练速度：`21.17 s/it`，`data_times=0.409`，`model_times=1.200`
- 当前进程状态：主进程 `DLl+`（D = disk sleep，L = 多线程）
- 主进程资源：RSS **4.19 GiB**，**689 线程**（较 21:48 的 497 继续增长）
- DeepSpeed：ZeRO-2，`train_batch_size=32`，`train_micro_batch_size_per_gpu=4`，`gradient_accumulation_steps=8`
- W&B：✅ 已连接并同步
- W&B Run：`https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla/runs/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`
- GPU：`P1.gpu.medium`，显存 **39518/40488 MiB（97.6%）**，利用率 40%，温度 31°C
- checkpoint：`checkpoints/` 目录已创建但为空；预计首次保存 `steps_50`
- 输出目录文件：`config.yaml`、`config.full.yaml`、`dataset_statistics.json`、wandb/
- 系统资源：总内存 503 GiB，已用 65 GiB，可用 433 GiB，负载 14.09/12.40/11.88
- 目标 r2 状态：未运行，输出目录不存在
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - 首个 `step 50` loss 日志
  - `steps_50` checkpoint 保存时内存是否溢出
  - 线程数持续快速增长（497 → 689），需重点关注

## 2026-06-18 — StarFlow Train 会话状态（21:48 CST）

- 监控时间：2026-06-18 21:48 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`（无 r2/r3 后缀）
- 实际进程 PID：`51127`，父进程 `51122`，tmux 会话：`starflow_train`
- 实际进程启动时间：21:32 CST，已运行约 **16 分钟**
- 训练状态：✅ **训练中，步数 18/80000**
- 当前步数：`18 / 80000`
- 训练速度：`20.35 s/it`，`data_times=0.751`，`model_times=1.334`
- 当前进程状态：主进程 `DLl+`（D = disk sleep，L = 多线程）
- 主进程资源：RSS **4.14 GiB**，**497 线程**（较 21:42 的 208 显著增长）
- DeepSpeed：ZeRO-2，`train_batch_size=32`，`train_micro_batch_size_per_gpu=4`，`gradient_accumulation_steps=8`
- W&B：✅ 已连接并同步
- W&B Run：`https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla/runs/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`
- GPU：`P1.gpu.medium`，显存 **39518/40488 MiB（97.6%）**，利用率 88%，温度 31°C
- checkpoint：`checkpoints/` 目录已创建但为空；预计首次保存 `steps_50`
- 输出目录文件：`config.yaml`、`config.full.yaml`、`dataset_statistics.json`、wandb/
- 系统资源：总内存 503 GiB，已用 65 GiB，可用 433 GiB，负载 8.93/10.75/11.35
- 目标 r2 状态：未运行，输出目录不存在
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - 首个 `step 50` loss 日志
  - `steps_50` checkpoint 保存时内存是否溢出
  - 线程数是否继续增长

## 2026-06-18 — StarFlow Train 会话状态（21:42 CST）

- 监控时间：2026-06-18 21:42 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`（无 r2/r3 后缀）
- 实际进程 PID：`51127`，父进程 `51122`，tmux 会话：`starflow_train`
- 实际进程启动时间：21:32 CST，已运行约 10 分钟
- 训练状态：✅ **初始化成功，训练循环已启动**
- 当前步数：`0 / 80000`
- 当前进程状态：主进程 `DLl+`（D = disk sleep，L = 多线程）
- 主进程资源：VSZ 459.5 GiB，RSS 3.78 GiB，**208 线程**，CPU 13.0%
- 已读取数据：约 14.1 GiB；已写入数据：约 436 MiB
- DeepSpeed：ZeRO-2，`train_batch_size=32`，`train_micro_batch_size_per_gpu=4`，`gradient_accumulation_steps=8`
- W&B：✅ 已连接并同步
- W&B Run：`https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla/runs/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`
- GPU：`P1.gpu.medium`，显存 16724/40488 MiB（41.3%），利用率 0%，温度 31°C
- checkpoint：`checkpoints/` 目录已创建但为空；预计首次保存 `steps_50`
- 输出目录文件：`config.yaml`、`config.full.yaml`、`dataset_statistics.json`、wandb/
- 系统资源：总内存 503 GiB，已用 64 GiB，可用 435 GiB，负载 11.77/11.42/11.65
- 目标 r2 状态：未运行，输出目录不存在
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - 首个 `step 50` loss 日志
  - `steps_50` checkpoint 保存时内存是否溢出
  - 线程数是否继续增长

## 2026-06-18 — StarFlow Train 会话状态（21:37 CST）

- 监控时间：2026-06-18 21:37 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`（无 r2/r3 后缀）
- 实际进程 PID：`51127`，父进程 `51122`，tmux 会话：`starflow_train`
- 实际进程启动时间：21:32 CST，已运行约 4 分 30 秒
- 当前进程状态：主进程 `D`（disk sleep）
- 当前步数：初始化中，尚未输出训练步骤
- 主进程资源：VSZ 5.69 GiB，RSS 467 MiB，64 线程，CPU 4.5%
- 已读取数据：约 4.1 GiB
- 启动配置：`NUM_PROCESSES=1`、`PER_DEVICE_BATCH_SIZE=4`、`GRADIENT_ACCUMULATION_STEPS=8`、`NUM_WORKERS=0`、`SAVE_INTERVAL=50`、`LOGGING_FREQUENCY=50`、`EVAL_INTERVAL=250`、`MAX_TRAIN_STEPS=80000`
- W&B：`WANDB_MODE=online`，但输出目录暂无 wandb 文件
- checkpoint：无（`checkpoints/` 目录尚未创建）
- 输出目录：`/gemini/code/starVLA/playground/Checkpoints/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618/`（空目录）
- 系统资源：总内存 503 GiB，已用 59 GiB，可用 440 GiB，负载 9.15/11.69/11.86
- GPU：`nvidia-smi` 当前环境不可读
- 目标 r2 状态：未运行，输出目录不存在
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - W&B 初始化是否成功
  - 首次 `step 50` loss 日志
  - `steps_50` checkpoint 保存时内存是否溢出

## 2026-06-18 — StarFlow Train 会话状态（21:32 CST）

- 监控时间：2026-06-18 21:32 CST
- 目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- 当前状态：**无训练进程运行**
- P0-M5-E-H2a-04 进程：已退出，无 checkpoint
- 用户最新一次尝试：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`（无 r2/r3 后缀），在 `import sympy` 阶段被中断
- 训练进程：未找到任何 `train_starvla.py` 进程
- checkpoint 进展：无
- 输出目录：
  - `..._250618_r2/`：不存在
  - `P0-M5-E-H2a-04..._250618/`：空目录
- 系统资源：
  - 总内存 503 GiB，已用 57 GiB，可用 442 GiB
  - 负载 10.29 / 10.97 / 11.60
  - 磁盘使用率 2%
- GPU：`nvidia-smi` 当前环境不可读
- tmux 会话 `starflow_train`：存在
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`

## 2026-06-18 — StarFlow Train 会话状态（21:28 CST）

- 监控时间：2026-06-18 21:28 CST
- 用户目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- **实际运行 run_id**：`P0-M5-E-H2a-04_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618`
- 实际进程 PID：`34581`，父进程 `34574`，tmux 会话：`starflow_train`
- 实际进程启动时间：21:26 CST，已运行约 2 分钟
- 当前进程状态：主进程 `D`（disk sleep）
- 当前步数：初始化中，尚未输出训练步骤
- 主进程资源：VSZ 5.51 GiB，RSS 386 MiB，64 线程，CPU 9.0%
- 已读取数据：约 2.4 GiB
- 启动配置：`NUM_PROCESSES=1`、`PER_DEVICE_BATCH_SIZE=4`、`GRADIENT_ACCUMULATION_STEPS=8`、`NUM_WORKERS=0`、`SAVE_INTERVAL=50`、`LOGGING_FREQUENCY=50`、`EVAL_INTERVAL=250`、`MAX_TRAIN_STEPS=80000`
- W&B：`WANDB_MODE=online`，但输出目录暂无 wandb 文件
- checkpoint：无（`checkpoints/` 目录尚未创建）
- 输出目录：`/gemini/code/starVLA/playground/Checkpoints/P0-M5-E-H2a-04_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618/`（空目录）
- 系统资源：总内存 503 GiB，已用 59 GiB，可用 440 GiB，负载 7.03/9.28/11.37
- GPU：`nvidia-smi` 当前环境不可读
- 目标 r2 状态：未运行，输出目录不存在
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 后续观察：
  - W&B 初始化是否成功
  - 首次 `step 50` loss 日志
  - `steps_50` checkpoint 保存时内存是否溢出

## 2026-06-18 — StarFlow Train 会话状态（21:22 CST）

- 监控时间：2026-06-18 21:22 CST
- 目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- 当前状态：**无训练进程运行**，目标 r2 输出目录仍不存在
- 训练进程：未找到任何 `train_starvla.py` 进程
- checkpoint 进展：无
- 系统资源：
  - 总内存 503 GiB，已用 55 GiB，可用 444 GiB
  - 负载 13.30 / 10.98 / 12.46
  - 磁盘使用率 2%
- GPU：`nvidia-smi` 当前环境不可读
- tmux 会话 `starflow_train`：存在，在命令行状态
- 最近一次操作：用户尝试启动 run，在 `import torch` 阶段手动中断
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`

## 2026-06-18 — StarFlow Train 会话状态（21:18 CST）

- 监控时间：2026-06-18 21:18 CST
- 目标 run_id：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- 当前状态：**无训练进程运行**，目标 r2 输出目录已不存在
- 训练进程：未找到任何 `train_starvla.py` 进程
- checkpoint 进展：无
- 系统资源：
  - 总内存 503 GiB，已用 56 GiB，可用 443 GiB
  - 负载 7.56 / 9.95 / 12.75
  - 磁盘使用率 2%
- GPU：`nvidia-smi` 当前环境不可读
- tmux 会话 `starflow_train`：存在，在命令行状态
- 最近一次操作：用户尝试启动 run_id `_250618`（无 r2/r3 后缀），在 `import torch` 阶段手动中断
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`

## 2026-06-18 — StarFlow Train 会话状态（21:05 CST）

- 监控时间：2026-06-18 21:05 CST
- 当前状态：**无训练进程运行**，等待用户手动启动
- 已停止进程：`286225`（r3，由 Claude Code 自动启动后按用户要求停止）
- 已删除：`/gemini/code/starVLA/playground/Checkpoints/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2/wandb`
- 已修改：`examples/LIBERO/train_files/run_starflow_train_ready.sh` 默认 `RUN_ID` 改为 `_r3`
- 已知问题：
  - r2 的 W&B run id 已在服务端被删除，不能复用 → **已通过换 _r3 解决**
  - W&B `init()` 存在 **90 秒超时** 问题，可能为网络/服务端响应慢，不是认证问题
- 建议启动命令：
  ```bash
  # online，增加 init_timeout
  WANDB_MODE=online WANDB_INIT_TIMEOUT=180 bash examples/LIBERO/train_files/run_starflow_train_ready.sh
  # 或 offline
  WANDB_MODE=offline bash examples/LIBERO/train_files/run_starflow_train_ready.sh
  ```
- 详细监控记录：`MEMORY/starflow_train_monitoring_2026-06-18_r2.md`

## 2026-06-18 — StarFlow Train 会话监控记录（r2，最新 20:54）

- 监控时间：2026-06-18 20:54 CST
- 当前活跃训练会话：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- 进程 PID：`243908`（最新主进程）+ 父进程 `243903`（bash 启动脚本），tmux 会话：`starflow_train`
- 旧进程 PID：`211355`（已终止）、`167071`（已终止）、`156282`（已终止）
- 新进程启动时间：2026-06-18 20:49 CST，已运行约 5 分 20 秒
- 数据集：`libero_all`（4in1），框架：`StarFlowVLA`，VLM：`Qwen3-VL-4B-Instruct`
- 训练配置：`per_device_batch_size=4`、`gradient_accumulation_steps=8`、`total_batch_size=32`、`logging_frequency=50`、`save_interval=125`、`eval_interval=250`、`max_train_steps=80000`
- 训练进展：
  - ✅ DeepSpeed 初始化完成，`Device: cuda:0`，backend `nccl`
  - ✅ 模型 checkpoint shards 加载完成
  - ✅ DiT 参数总量：`532,326,426`
  - 🔄 当前阶段：`Creating VLA Dataset with Mixture libero_all`
  - ⏳ 尚未输出训练步骤日志
- 当前进程状态：主进程 `D`（disk sleep）
- 主进程资源：VSZ 45.9 GiB，RSS 3.15 GiB，**136 线程**，CPU 8.4%
- 已读取数据：约 5.5 GiB
- GPU 状态：`Device: cuda:0` 已识别，`nvidia-smi` 当前环境不可读
- 系统资源：总内存 503 GiB，已用 92 GiB，可用 405 GiB，负载 11.60/11.73/12.12
- 尚未保存 checkpoint（`checkpoints/` 为空），预计首次保存 `steps_125`
- W&B 状态：
  - API 可达（`api.wandb.ai` 返回 200）
  - 认证已加载（`/root/.netrc`）
  - ❌ **HTTP 409**：`run P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2 was previously created and deleted; try a new run id`
  - 根因：该 run id 在 W&B 服务端已被创建并删除，不能复用
  - 训练可继续，但 metrics 不会同步到 W&B 网站
- 详细监控记录已写入 `MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 风险与后续观察：
  - 模型已加载完成，数据集创建中，预计不久将进入真实训练循环
  - 如需 W&B 在线记录，必须更换新的 `wandb_run_id`（例如后缀 `_r3`）
  - 线程数 136，需继续观察是否异常增长
  - 等待首个 `step 50` loss 日志输出

## 2026-06-18 — StarFlow Train 会话监控记录（r2，20:40）

- 监控时间：2026-06-18 20:40 CST
- 当前活跃训练会话：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- 进程 PID：`211355`（已终止），tmux 会话：`starflow_train`
- 启动时间：2026-06-18 20:39 CST
- 用户前置操作：执行了 `wandb login` 后重启训练
- 当前进程状态：主进程 `D`（disk sleep）
- 当前步数：初始化中，尚无训练步骤日志
- 主进程资源：VSZ 5.42 GiB，RSS 313 MiB，**64 线程**，CPU 9.4%
- 已读取数据：约 3.8 GiB
- GPU 状态：PyTorch CUDA 可用（1 device），`nvidia-smi` 当前环境不可读
- 系统资源：总内存 503 GiB，已用 88 GiB，可用 409 GiB，负载 10.97/12.51/13.06
- 尚未保存 checkpoint（`checkpoints/` 为空）
- 详细监控记录已写入 `MEMORY/starflow_train_monitoring_2026-06-18_r2.md`

## 2026-06-18 — StarFlow Train 会话监控记录（r2，20:26）

- 监控时间：2026-06-18 20:26 CST（上一轮）
- 当前活跃训练会话：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- 进程 PID：`167071`（已终止），tmux 会话：`starflow_train`
- 启动时间：2026-06-18 20:24:58 CST
- 当前进程状态：主进程 `D`（disk sleep）
- 当前步数：初始化中，尚无训练步骤日志
- 主进程资源：VSZ 5.67 GiB，RSS 526 MiB，**131 线程**，CPU 4.4%
- 已读取数据：约 5.7 GiB
- GPU 状态：PyTorch CUDA 可用（1 device），`nvidia-smi` 当前环境不可读
- 系统资源：总内存 503 GiB，已用 88 GiB，可用 408 GiB，负载 13.46/12.61/13.34
- 尚未保存 checkpoint（`checkpoints/` 为空）
- W&B Run：`https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla/runs/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- 详细监控记录已写入 `MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 风险与后续观察：
  - 进程仍在数据集初始化/元数据构建阶段
  - 主进程线程数 131 并持续增长，需继续观察
  - 已按用户要求将监控频率调整为 **每 5 分钟**

## 2026-06-18 — StarFlow Train 会话监控记录（r2，旧）

- 监控时间：2026-06-18 20:01 / 20:03 / 20:04 / 20:05 / 20:06 / 20:07 / 20:08 / 20:09 / 20:10 / 20:11 / 20:12 / 20:13 / 20:14 / 20:15 / 20:16 / 20:17 / 20:19 CST
- 当前活跃训练会话：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- 进程 PID：`156282`（主进程）+ `187513`（`pt_data_worker` 子进程），tmux 会话：`starflow_train`
- 启动时间：2026-06-18 19:47（主进程），worker 于 19:57 启动
- 数据集：`libero_all`（4in1），框架：`StarFlowVLA`，VLM：`Qwen3-VL-4B-Instruct`
- 训练配置：`per_device_batch_size=4`、`gradient_accumulation_steps=8`、`total_batch_size=32`、`logging_frequency=50`、`save_interval=500`、`max_train_steps=80000`
- 当前进程状态：主进程 `S`，worker `D`
- 当前步数：约 `73 / 80000`，速度约 `16.49 s/it`
- 首个 loss 日志：Step 50 `action_dit_loss=0.6034`
- 主进程资源：VSZ 471 GiB，RSS 4.10 GiB，72 线程，CPU 114%
- DataLoader worker 资源：VSZ 502 GiB，RSS 4.24 GiB，**1069 线程**（1645 → 1069 下降），CPU 22.3%
- GPU 状态：`P1.gpu.medium`，显存 39518/40488 MiB（97.6%），GPU 利用率 100%
- 系统资源：总内存 503 GiB，已用 96 GiB，可用 401 GiB，负载 12.53/13.30/14.27
- 尚未保存 checkpoint（`checkpoints/` 为空），预计首次保存 `steps_500`
- W&B Run：`https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla/runs/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- 详细监控记录已写入 `MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 风险与后续观察：
  - DataLoader worker 线程数高位波动：1645 → **1069**，仍处于高位
  - DataLoader worker RSS 4.24 GiB
  - 系统 1 min 负载回落至 12.53
  - GPU 利用率 100%
  - 首个 loss 0.6034，后续需观察下降趋势
  - 旧进程已于 20:24 被中断，本段记录为历史快照

## 2026-06-18 — StarFlow-VLA DeepSpeed Universal Checkpoint 验证

- 监控时间：2026-06-18 20:01 / 20:03 / 20:04 / 20:05 / 20:06 / 20:07 / 20:08 / 20:09 / 20:10 / 20:11 / 20:12 / 20:13 / 20:14 / 20:15 / 20:16 / 20:17 / 20:19 CST
- 当前活跃训练会话：`P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- 进程 PID：`156282`（主进程）+ `187513`（`pt_data_worker` 子进程），tmux 会话：`starflow_train`
- 启动时间：2026-06-18 19:47（主进程），worker 于 19:57 启动
- 数据集：`libero_all`（4in1），框架：`StarFlowVLA`，VLM：`Qwen3-VL-4B-Instruct`
- 训练配置：`per_device_batch_size=4`、`gradient_accumulation_steps=8`、`total_batch_size=32`、`logging_frequency=50`、`save_interval=500`、`max_train_steps=80000`
- 当前进程状态：主进程 `S`，worker `D`
- 当前步数：约 `73 / 80000`，速度约 `16.49 s/it`
- 首个 loss 日志：Step 50 `action_dit_loss=0.6034`
- 主进程资源：VSZ 471 GiB，RSS 4.10 GiB，72 线程，CPU 114%
- DataLoader worker 资源：VSZ 502 GiB，RSS 4.24 GiB，**1069 线程**（1645 → 1069 下降），CPU 22.3%
- GPU 状态：`P1.gpu.medium`，显存 39518/40488 MiB（97.6%），GPU 利用率 100%
- 系统资源：总内存 503 GiB，已用 96 GiB，可用 401 GiB，负载 12.53/13.30/14.27
- 尚未保存 checkpoint（`checkpoints/` 为空），预计首次保存 `steps_500`
- W&B Run：`https://wandb.ai/silencewx-harbin-institute-of-technology/starflow_vla/runs/P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r2`
- 详细监控记录已写入 `MEMORY/starflow_train_monitoring_2026-06-18_r2.md`
- 风险与后续观察：
  - DataLoader worker 线程数高位波动：1645 → **1069**，仍处于高位
  - DataLoader worker RSS 4.24 GiB
  - 系统 1 min 负载回落至 12.53
  - GPU 利用率 100%
  - 首个 loss 0.6034，后续需观察下降趋势
  - 旧进程已于 20:24 被中断，本段记录为历史快照

## 2026-06-18 — StarFlow-VLA DeepSpeed Universal Checkpoint 验证

- 已生成并验证 full-adam DeepSpeed Universal checkpoint：包含 `fp32.pt`、Adam `exp_avg.pt`、`exp_avg_sq.pt` 与 `step.pt`
- full-adam Universal 持久路径：`playground/Checkpoints/P0-M5-E-H2a-03_universal_full_adam_250618`
- 已将 `steps_31500` 复制到正式实验目录：`playground/Checkpoints/P0-M5-E-H2a-03_starflow_libero-goal_qwen3vl4b_lwfm_ft32_250615/checkpoints/steps_31500`
- 原实验目录已写入 `checkpoints/latest_universal=steps_31500`
- 目标目录校验结果：`total_files=2690`、`fp32=672`、`exp_avg=672`、`exp_avg_sq=672`、`step=672`、`part_files=0`
- `starVLA/training/train_starvla.py` 当前默认 `trainer.checkpoint_format=universal`；如需旧 lightweight 保存，必须显式配置 `trainer.checkpoint_format=lightweight`
- StarFlow stage1 与 `run_starflow_train_ready.sh` 当前默认单卡训练配置：`datasets.vla_data.per_device_batch_size=4`、`trainer.gradient_accumulation_steps=8`，有效全局 batch 为 32
- StarFlow stage1 与 `run_starflow_train_ready.sh` 当前默认 `run_id` / W&B run 为 `P0-M5-E-H2a-03_starflow_libero-goal_qwen3vl4b_lwfm_ft32_250615`，W&B project 为 `starflow_vla`
- 新增 Universal -> HF safetensors 导出工具：`tools/convert_universal_checkpoint_to_hf_safetensors.py`，默认只导出模型权重，输出 `model-*.safetensors` 与 `model.safetensors.index.json`
- 已确认 DeepSpeed 0.16.9 原生 `ds_to_universal` 不适配当前 StarVLA checkpoint：optimizer state 中无 Adam `exp_avg/exp_avg_sq`，只有 `single_partition_of_fp32_groups` 与 `param_slice_mappings`
- 新增 fp32-only Universal 转换路线：从 ZeRO optimizer shard 提取并合并 672 个 optimizer-backed fp32 master weights，输出 DeepSpeed Universal `zero/<param>/fp32.pt`
- 已验证 4 卡 ZeRO source `steps_31500` 转 fp32-only Universal 后，可用 1 卡加载并训练 1 step 到 `steps_31501`
- 已验证 1 卡保存后的 fp32-only Universal `steps_31501` 可被 4 卡 DeepSpeed ZeRO-2 加载，验证脚本退出码为 0
- 持久保存路径：
  - `steps_31500`: `playground/Checkpoints/P0-M5-E-H2a-03_universal_fp32_only_250618`
  - `steps_31501`: `playground/Checkpoints/P0-M5-E-H2a-03_universal_fp32_only_250618_step31501`
- 两个持久目录均约 21G；`steps_31501` 校验结果为 `latest_universal=steps_31501`、`zero/**/fp32.pt=672`
- 限制：当前 Universal 是 fp32-only，不含 Adam 动量、scheduler、dataloader、RNG 状态；可跨 GPU 数量恢复 fp32 master weights 继续训练，但不是完整 bitwise training-state resume

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

## 2026-06-15 — StarFlow-VLA eval report write-path

- 已新增 `examples/LIBERO/eval_files/starflow_eval_report.py`，用于提取 `checkpoint_hash`、`config_hash`、`data_version`、`starflow_mapping`
- 已新增 `tests/test_starflow_eval_report.py`，验证 eval report helper 和 JSON 落盘
- `eval_libero.py` 现改为在每个 episode 结束后、视频编码前先写 `eval_report.json`
- 已生成 `playground/eval_results/libero_goal/starflow_vla_stage1_smoke_steps_1/eval_report.json`
- 当前 smoke report 记录：
- `success_rate = 0.0`
- `failure_category = {"timeout_no_success": 1}`
- 已包含 `checkpoint_hash`、`config_hash`、`data_version`、`starflow_mapping`

## 2026-06-15 — StarFlow-VLA resume 100 step consistency smoke

- `starVLA/training/train_starvla.py` 的 lightweight checkpoint 已新增 Python / NumPy / Torch / CUDA RNG state 保存与恢复
- 已新增 `tests/test_starflow_resume_100_steps.py`
- 验证路径：
- 从 `steps_1` 只加载模型权重，基于真实 `libero_goal` batch bootstrap 1 step，生成当前格式兼容的 lightweight checkpoint
- 从 bootstrap checkpoint 连续跑 100 step
- 从 bootstrap checkpoint 跑 50 step 保存 `steps_51`，恢复后再跑 50 step
- `WANDB_MODE=disabled .venv/bin/python -m unittest -v tests.test_starflow_resume_100_steps` 已通过
- 日志采样显示恢复路径与连续路径后半段 loss 基本重合，例如：
- `step 61: 0.47851533 vs 0.47853473`
- `step 71: 0.91403395 vs 0.91406316`
- `step 81: 0.29172632 vs 0.29170248`
- `step 91: 0.58911145 vs 0.58910495`
- `step 101: 0.34975210 vs 0.34975326`

## 2026-06-15 — StarFlow-VLA checkpoint scaler / metadata sidecar

- `lightweight checkpoint` 现会自动写入 `scaler.pt`、`config.yaml`、`config.full.yaml` 和 `starflow_mapping.json`
- `scaler.pt` 即使在未启用 AMP scaler 时也会写入占位 payload，避免 checkpoint schema 漏项
- 已为现有 smoke checkpoint `playground/Checkpoints/starflow_vla_stage1_qwenpi_v3_native/checkpoints/steps_1` 补写 `scaler.pt`
- `.venv/bin/python -m unittest -v tests.test_starflow_checkpoint_mapping` 已通过，覆盖 scaler placeholder / metadata 落盘和 scaler roundtrip

## 2026-06-15 — StarFlow-VLA P0 guardrail governance

- 已在 `tests/test_starflow_docs_governance.py` 增加 P0 guardrail 回归检查
- 检查内容包括：
- `DESIGN_FREEZE_CHECK.md`、`P0_IMPLEMENTATION_PLAN.md`、`MODULE_MAPPING.md` 明确 Perceiver / 显式 FlowCondition runtime / 14D mask 不阻断 P0
- `stage1_starflow_qwenpi_v3_native.yaml` 不启用 `perceiver_enabled=true`、`flow_condition_runtime=true`、`max_action_dim=14` 或 `action_mask`
- `steps_1/starflow_mapping.json` 中 `perceiver_enabled=false`、`flow_condition_runtime=false`
- `.venv/bin/python -m unittest -v tests.test_starflow_docs_governance` 已通过

## 2026-06-15 — StarFlow-VLA Stage1 真实训练主链路闭环

- 已修复 `starVLA/training/train_starvla.py` 中 `prepare_data()` 在未初始化分布式时无条件调用 `dist.barrier()` 的单卡阻塞
- 已确认当前环境中 `DeepSpeed + Triton` 还需要显式设置：
- `LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.3/compat:$LD_LIBRARY_PATH`
- `MASTER_ADDR=127.0.0.1`
- `MASTER_PORT=<free_port>`
- `RANK=0`
- `LOCAL_RANK=0`
- `WORLD_SIZE=1`
- 使用上述环境变量运行 `.venv/bin/python -u starVLA/training/train_starvla.py --config_yaml configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml --run_id starflow_vla_stage1_trainloop_10step_envdist`
- 真实训练已完成 `10/10 step`
- 已生成：
- `playground/Checkpoints/starflow_vla_stage1_trainloop_10step_envdist/checkpoints/steps_10`
- `playground/Checkpoints/starflow_vla_stage1_trainloop_10step_envdist/final_model`
- `steps_10` 与 `final_model` 均包含 model shards、`optimizer_rank_00000.pt`、`scheduler.pt`、`random_states_0.pkl`、`scaler.pt`、`config.yaml`、`config.full.yaml`、`dataset_statistics.json`、`starflow_mapping.json`、`trainer_state.json`

## 2026-06-15 — StarFlow-VLA `steps_10` 全 task sweep eval

- 已使用 `playground/Checkpoints/starflow_vla_stage1_trainloop_10step_envdist/checkpoints/steps_10` 启动 policy server
- 已运行 `libero_goal` 全 task sweep：`10 task × 1 trial = 10 episodes`
- 结果目录：`playground/eval_results/libero_goal/starflow_vla_stage1_trainloop_10step_envdist_steps_10_fullsuite`
- 结果：
- `success_rate = 0.0`
- `total_episodes = 10`
- `failure_category = {"timeout_no_success": 10}`
- 已生成 `10` 个 failure rollout 视频和 `eval_report.json`
- 该结果是全 task sweep，不等价于标准 `50 trials/task` 完整评测

## 2026-06-15 — StarFlow-VLA 训练产物到推理验证回归入口

- 已新增 `examples/LIBERO/eval_files/run_starflow_eval_regression.sh`
- 入口职责：
- 后台启动 `run_policy_server.sh`
- 轮询 `HOST:PORT` 等待 policy server 就绪
- 调用 `eval_libero.sh` 执行可配置的 quick regression
- 检查 `eval_report.json` 是否产出，并打印结果摘要
- `examples/LIBERO/eval_files/eval_libero.sh` 已补充 `MAX_TASKS` 透传，可直接跑 `1 task × 1 trial`
- `.venv/bin/python -m unittest -v tests.test_starflow_eval_preflight tests.test_starflow_eval_report` 已通过
- 首轮实际运行发现 `SERVER_READY_TIMEOUT=300` 不足；server 在 5 分钟窗口内仍处于 framework / checkpoint CPU 侧初始化
- 已将默认等待时间上调到 `900s`，并为 server 进程启用 `PYTHONUNBUFFERED=1`
- 本轮未等待到 server 真正监听端口，未形成新的回归 `eval_report.json`
- 结论：一键回归入口代码已固化，剩余风险点在 policy server 冷启动时长，不在训练/推理接口契约

## 2026-06-15 — Policy server 冷启动分阶段定位

- 已在以下文件补充最小耗时日志：
- `deployment/model_server/server_policy.py`
- `deployment/model_server/policy_wrapper.py`
- `starVLA/model/framework/base_framework.py`
- `server_policy.py` 已改为在 `main()` 内懒导入 `PolicyServerWrapper` / `WebsocketPolicyServer`
- 实测冷启动探针（`steps_10`）结果：
- `server_policy.main: module imports finished in 272.50s`
- `baseframework.from_pretrained -> read_mode_config done in 0.13s`
- `build_framework done in 42.69s (StarFlowVLA)`
- 在 360s 探针窗口结束前，尚未走完 `load_model_weights` / `.to(cuda)` / websocket listen
- 结论：
- 根因不是 regression 脚本等待逻辑，也不是训练/推理接口不匹配
- 首个大头是顶层依赖导入链；第二个大头是 framework build
- `300s` 冷启动窗口必然不够；`900s` 只是保守等待，不是根治

## 2026-06-15 — StarFlow-VLA 训练就绪启动器

- 已新增 `examples/LIBERO/train_files/run_starflow_train_ready.sh`
- 默认绑定当前已验证的训练前提：
- `STARVLA_PYTHON=.venv/bin/python`
- `WANDB_MODE=disabled`
- `MASTER_ADDR=127.0.0.1`
- `MASTER_PORT=29621`
- `RANK=0`
- `LOCAL_RANK=0`
- `WORLD_SIZE=1`
- `LIBERO_DATA_ROOT=playground/Datasets/LEROBOT_LIBERO_DATA`
- `BASE_VLM=playground/Pretrained_models/Qwen3-VL-4B-Instruct`
- 默认使用 `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml`
- 默认 `MAX_TRAIN_STEPS=10`，可通过环境变量覆盖为更长训练
- 脚本内已显式避免 `eval_interval=0` 的除零风险，默认设为 `1000`
- 新增 `tests/test_starflow_train_ready.py` 做 shell 语法和关键默认值回归

## 2026-06-18 — StarFlow-VLA 4in1 长训默认入口修正

- 已将 `examples/LIBERO/train_files/run_starflow_train_ready.sh` 默认口径切换为 LIBERO 4in1 baseline：
- `RUN_ID=P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250615`
- `LIBERO_DATA_ROOT=/gemini/code/datasets/LEROBOT_LIBERO_DATA`
- `DATA_MIX=libero_all`
- `MAX_TRAIN_STEPS=80000`
- `SAVE_INTERVAL=125`
- `LOGGING_FREQUENCY=50`
- `EVAL_INTERVAL=250`
- `PER_DEVICE_BATCH_SIZE=4`
- `GRADIENT_ACCUMULATION_STEPS=8`
- `NUM_WORKERS=1`
- `trainer.checkpoint_format=lightweight`
- `trainer.enable_local_checkpoint_staging=true`
- `trainer.local_checkpoint_root=/root/temp`
- `trainer.local_checkpoint_keep_count=2`
- 已同步导出 `ACCELERATE_GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}`，修正单卡直接运行时 Accelerator 实际梯度累积退回 `1` 的问题。
- 已同步更新 `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml` 的默认数据集、run_id、步数和日志/保存间隔。
- 已验证 `bash -n examples/LIBERO/train_files/run_starflow_train_ready.sh` 通过。
