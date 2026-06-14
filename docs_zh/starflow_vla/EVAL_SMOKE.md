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
- policy server 可加载 `steps_1` 并监听 `0.0.0.0:6694`。
- 最小 LIBERO rollout smoke 已完成：`libero_goal`，`max_tasks=1`，`num_trials_per_task=1`。
- smoke 输出 `Total success rate: 0.0`，`Total episodes: 1`。
- 生成视频：`playground/eval_results/libero_goal/starflow_vla_stage1_smoke_steps_1/rollout_open_the_middle_drawer_of_the_cabinet_episode0_failure.mp4`。
- 退出阶段出现 EGL / `libGLU.so.0` 清理期警告，但 eval 进程退出码为 0。
- 未运行完整 LIBERO suite、failure taxonomy、checkpoint/config hash report 或多 seed 评测。
