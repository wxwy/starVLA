# StarFlow-VLA Eval Smoke

## Scope

本文件记录 P0-M10 的 LIBERO eval smoke 前置条件、命令和当前状态。当前只记录最小 1 task × 1 trial smoke，不记录完整 LIBERO suite 结果。

## Required Inputs

- P0 checkpoint：`playground/Checkpoints/starflow_vla_stage1_qwenpi_v3_native/checkpoints/steps_*`
- Mapping sidecar：目录 checkpoint 下的 `starflow_mapping.json`，或单文件 checkpoint 旁的 `*.starflow_mapping.json`
- LIBERO 环境：`LIBERO_HOME` 指向可用 LIBERO 源码目录
- LIBERO 数据：`playground/Datasets/LEROBOT_LIBERO_DATA`

## Commands

```bash
bash -n examples/LIBERO/eval_files/run_policy_server.sh
bash -n examples/LIBERO/eval_files/eval_libero.sh
.venv/bin/python -m unittest tests.test_starflow_eval_preflight -v
```

真实 eval smoke 在 P0 checkpoint 和 LIBERO 数据可用后运行：

```bash
RUN_DIR=playground/Checkpoints/starflow_vla_stage1_qwenpi_v3_native \
CKPT_STEP=<step> \
STARVLA_PYTHON=.venv/bin/python \
bash examples/LIBERO/eval_files/run_policy_server.sh

RUN_DIR=playground/Checkpoints/starflow_vla_stage1_qwenpi_v3_native \
CKPT_STEP=<step> \
LIBERO_PYTHON=.libero/bin/python \
TASK_SUITE_NAME=libero_goal \
NUM_TRIALS_PER_TASK=1 \
bash examples/LIBERO/eval_files/eval_libero.sh
```

## Current Status

- Eval shell scripts pass `bash -n`.
- `tests/test_starflow_eval_preflight.py` 已通过，checkpoint mapping 检查不再 skip。
- P0 smoke checkpoint 当前为 `playground/Checkpoints/starflow_vla_stage1_qwenpi_v3_native/checkpoints/steps_1`。
- `steps_1/starflow_mapping.json` 已存在。
- `steps_1/scaler.pt` 已存在，当前记录为未启用 AMP scaler 的占位状态。
- policy server 可加载 `steps_1` 并监听 `0.0.0.0:6694`。
- 最小 LIBERO rollout smoke 已完成：`libero_goal`，`max_tasks=1`，`num_trials_per_task=1`。
- smoke 输出 `Total success rate: 0.0`，`Total episodes: 1`。
- 生成视频：`playground/eval_results/libero_goal/starflow_vla_stage1_smoke_steps_1/rollout_open_the_middle_drawer_of_the_cabinet_episode0_failure.mp4`。
- 评测报告：`playground/eval_results/libero_goal/starflow_vla_stage1_smoke_steps_1/eval_report.json`。
- `eval_report.json` 已包含 `success_rate=0.0`、`failure_category.timeout_no_success=1`、`checkpoint_hash`、`config_hash`、`data_version`、`starflow_mapping`。
- 退出阶段出现 EGL / `libGLU.so.0` 清理期警告，但 eval 进程退出码为 0。
- 为避免退出阶段卡住导致报告丢失，`eval_libero.py` 已改为在每个 episode 结束后、视频编码前先写一次 `eval_report.json`。
- 基于真实训练产物 `playground/Checkpoints/starflow_vla_stage1_trainloop_10step_envdist/checkpoints/steps_10` 已完成一轮 `libero_goal` 全 task sweep：
- `10 task × 1 trial = 10 episodes`
- `success_rate = 0.0`
- `failure_category.timeout_no_success = 10`
- 结果目录：`playground/eval_results/libero_goal/starflow_vla_stage1_trainloop_10step_envdist_steps_10_fullsuite`
- 已生成 `10` 个 failure rollout 视频和 `eval_report.json`
- 该结果代表“全 task sweep”，不等价于标准 `50 trials/task` 的完整 LIBERO 评测。
- 未运行标准 50 trials/task、细粒度 failure taxonomy 或多 seed 评测。

## 一键回归入口

- 新增 `examples/LIBERO/eval_files/run_starflow_eval_regression.sh`
- 默认链路：
- 后台启动 `run_policy_server.sh`
- 轮询等待 policy server 监听端口
- 调用 `eval_libero.sh`
- 校验 `eval_report.json`
- `eval_libero.sh` 已支持 `MAX_TASKS`，可直接执行 `1 task × 1 trial` quick regression
- 默认 `SERVER_READY_TIMEOUT=900`；该值来自本地 `steps_10` 冷启动观察，`300s` 不足以覆盖 policy server 的 CPU 侧 framework / checkpoint 初始化
