# P0-M5-E-H2a-03: StarFlow LIBERO 4-in-1 Qwen3VL-4B future_tokens=32（bs32）state-fix 评估追踪

> **实验代号**: E-H2a-03 / P0-M5
> **状态**: ✅ 评估全部完成
> **run_id**: `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_260703_0848`
> **启动时间**: 2026-07-16 11:48 CST
> **当前更新**: 2026-07-17 12:08 CST
> **完成时间**: 2026-07-17 03:30 CST
> **tmux 会话**: `test`
> **配置来源**: `configs/starflow_vla/ablations/future_tokens_32.yaml`
> **评测输出**: `playground/starflow_eval_result`

---

## 评估概述

本 run 为 StarFlowVLA `num_target_vision_tokens=32`（ft32）变体。2026-07-16 修复 `eval_libero.py` 未传入 robot state 的问题后，使用 `playground/starflow_eval_plan.sh` 对全部关键 ckpt 进行重新评测，结果固定输出到 `playground/starflow_eval_result/`。

### 评测参数

| 参数 | 值 |
|------|-----|
| `RUN_ID` | `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_260703_0848` |
| `workers` | 10 |
| `num_trials` | 50 |
| `suites` | libero_goal, libero_10, libero_object, libero_spatial |
| `ckpt 列表` | steps_5000, steps_10000, steps_20000, steps_30000, steps_40000, steps_50000, steps_60000, steps_70000, steps_80000 |
| `policy server` | 单 server，固定端口 6720-6728 |
| `server 优化` | server-side batching（max_batch=8, timeout=30ms） |

---

## 评估进度

> 最后更新：2026-07-17 12:08 CST

| ckpt | 状态 | 总 eps | 总 successes | 总 SR |
|------|------|--------|--------------|-------|
| steps_5000 | ✅ 完成 | 2000 / 2000 | 933 | 46.7% |
| steps_10000 | ✅ 完成 | 2000 / 2000 | 1362 | 68.1% |
| steps_20000 | ✅ 完成 | 2000 / 2000 | 1678 | 83.9% |
| steps_30000 | ✅ 完成 | 2000 / 2000 | 1625 | 81.2% |
| steps_40000 | ✅ 完成 | 2000 / 2000 | 1750 | 87.5% |
| steps_50000 | ✅ 完成 | 2000 / 2000 | 1834 | 91.7% |
| steps_60000 | ✅ 完成 | 2000 / 2000 | 1860 | 93.0% |
| steps_70000 | ✅ 完成 | 2000 / 2000 | 1851 | 92.5% |
| steps_80000 | ✅ 完成 | 2000 / 2000 | 1863 | 93.2% |

### 已完成的 ckpt 分 suite 结果

#### steps_5000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 333 | 66.6% |
| libero_10 | 500 | 62 | 12.4% |
| libero_object | 500 | 293 | 58.6% |
| libero_spatial | 500 | 245 | 49.0% |
| **合计** | **2000** | **933** | **46.7%** |

#### steps_10000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 356 | 71.2% |
| libero_10 | 500 | 124 | 24.8% |
| libero_object | 500 | 456 | 91.2% |
| libero_spatial | 500 | 426 | 85.2% |
| **合计** | **2000** | **1362** | **68.1%** |

#### steps_20000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 446 | 89.2% |
| libero_10 | 500 | 282 | 56.4% |
| libero_object | 500 | 490 | 98.0% |
| libero_spatial | 500 | 460 | 92.0% |
| **合计** | **2000** | **1678** | **83.9%** |

#### steps_30000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 438 | 87.6% |
| libero_10 | 500 | 248 | 49.6% |
| libero_object | 500 | 470 | 94.0% |
| libero_spatial | 500 | 469 | 93.8% |
| **合计** | **2000** | **1625** | **81.2%** |

#### steps_40000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 435 | 87.0% |
| libero_10 | 500 | 358 | 71.6% |
| libero_object | 500 | 480 | 96.0% |
| libero_spatial | 500 | 477 | 95.4% |
| **合计** | **2000** | **1750** | **87.5%** |

#### steps_50000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 472 | 94.4% |
| libero_10 | 500 | 410 | 82.0% |
| libero_object | 500 | 467 | 93.4% |
| libero_spatial | 500 | 485 | 97.0% |
| **合计** | **2000** | **1834** | **91.7%** |

#### steps_60000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 468 | 93.6% |
| libero_10 | 500 | 419 | 83.8% |
| libero_object | 500 | 488 | 97.6% |
| libero_spatial | 500 | 485 | 97.0% |
| **合计** | **2000** | **1860** | **93.0%** |

#### steps_70000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 465 | 93.0% |
| libero_10 | 500 | 421 | 84.2% |
| libero_object | 500 | 479 | 95.8% |
| libero_spatial | 500 | 486 | 97.2% |
| **合计** | **2000** | **1851** | **92.5%** |

#### steps_80000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 469 | 93.8% |
| libero_10 | 500 | 422 | 84.4% |
| libero_object | 500 | 495 | 99.0% |
| libero_spatial | 500 | 477 | 95.4% |
| **合计** | **2000** | **1863** | **93.2%** |

---

## state 修复前后对比

本次重新评测前，`playground/eval_results/` 中 ft32 的 steps_5000/steps_10000 为未传 state 的结果。对比如下：

| ckpt | 老结果（eval_results） | 新结果（starflow_eval_result） | 差异 |
|------|------------------------|-------------------------------|------|
| steps_5000 | 7.8% | 46.7% | +38.9% |
| steps_10000 | 40.9% | 68.1% | +27.2% |
| steps_20000 | 66.1% | 83.9% | +17.8% |
| steps_30000 | 64.3% | 81.2% | +16.9% |
| steps_40000 | 75.9% | 87.5% | +11.6% |
| steps_50000 | — | 91.7% | — |
| steps_60000 | 82.0% | 93.0% | +11.0% |
| steps_70000 | 81.3% | 92.5% | +11.2% |
| steps_80000 | 80.4% | 93.2% | +12.8% |

### 老结果详细分 suite

#### steps_5000（老）

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 58 | 11.6% |
| libero_10 | 500 | 3 | 0.6% |
| libero_object | 500 | 2 | 0.4% |
| libero_spatial | 500 | 93 | 18.6% |
| **合计** | **2000** | **156** | **7.8%** |

#### steps_10000（老）

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 263 | 52.6% |
| libero_10 | 500 | 14 | 2.8% |
| libero_object | 500 | 261 | 52.2% |
| libero_spatial | 500 | 281 | 56.2% |
| **合计** | **2000** | **819** | **40.9%** |

#### steps_20000（老）

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 376 | 75.2% |
| libero_10 | 500 | 166 | 33.2% |
| libero_object | 500 | 408 | 81.6% |
| libero_spatial | 500 | 373 | 74.6% |
| **合计** | **2000** | **1323** | **66.1%** |

#### steps_30000（老）

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 366 | 73.2% |
| libero_10 | 500 | 105 | 21.0% |
| libero_object | 500 | 384 | 76.8% |
| libero_spatial | 500 | 431 | 86.2% |
| **合计** | **2000** | **1286** | **64.3%** |

#### steps_60000（老）

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 392 | 78.4% |
| libero_10 | 500 | 236 | 47.2% |
| libero_object | 500 | 441 | 88.2% |
| libero_spatial | 500 | 451 | 90.2% |
| **合计** | **2000** | **1520** | **82.0%** |

---

## 备注

- 修复 commit：
  - `b7af6758 fix(LIBERO): pass robot state to model during evaluation`
  - 系统修复：安装 `libegl1` 以解决 PyOpenGL / mujoco EGL 初始化失败
- 评测过程中使用 workers=10，speed 约 18-25 eps/min。
- 每个 ckpt 固定端口：steps_5000→6720，steps_10000→6721，...，steps_80000→6728。
- 日志文件已调整为按秒命名格式：`starflow_eval_plan_YYYYMMDD_HHMMSS.log`，避免同一分钟内多次启动冲突。

---

## 中断说明

2026-07-16 15:15 左右，ft32 在启动 `steps_20000` 的 policy server 时遇到 **Server 启动超时**（日志：`[Thu Jul 16 03:15:08 PM CST 2026] ❌ Server 启动超时`）。后续尝试重启后再次在 `steps_20000` 启动阶段超时（`[Thu Jul 16 03:18:33 PM CST 2026] ❌ Server 启动超时`）。

当前状态：
- `test` tmux 会话仍在，但 ft32 评估已全部完成，无 `672x` 端口监听。
- 无 ft32 相关 `server_policy` / `eval_pool_manager` / `eval_libero` 进程在运行。
- 全部 9 个 ckpt 结果已落盘。

**17:34 更新**：发现一个新的 `server_policy.py` 进程（PID 431170）正在启动 `steps_20000`（端口 6722），CPU 占用约 188%，但端口尚未监听，可能仍在加载模型。状态已改为「评估恢复中」。

**18:33 更新**：`steps_20000` 评估继续进行中，当前已完成 1506/2000 trials，整体 SR 84.3%。

**20:17 更新**：`steps_30000` 评估继续进行中（1993/2000，SR 81.4%）。

**21:31 更新**：`steps_30000` 已完成（2000/2000，SR 81.2%），`steps_40000` 运行中（1906/2000，SR 87.4%）。当前 `test` tmux 会话实际在跑 `ft16`，ft32 由其他 worker/会话推进。

**22:52 更新**：`steps_40000` 已完成（2000/2000，SR 87.5%），`steps_50000` 运行中（1983/2000，SR 91.6%），即将完成。

**23:55 更新**：`steps_50000` 已完成（2000/2000，SR 91.7%），`steps_60000` 运行中（1664/2000，SR 92.9%）。

**00:30 更新**：`steps_60000` 已完成（2000/2000，SR 93.0%），`steps_70000` 运行中（572/2000，SR 92.5%）。

**03:30 更新**：**ft32 全部 9 个 ckpt 评估已完成**。最终 `steps_80000` SR 达到 **93.2%**，为本次 ft32 评估中 SR 最高的 ckpt。state-fix 后的新结果全面高于老结果，提升幅度 11-39 个百分点。

---

---

## 每个 ckpt 的平均完成时长

> 单位：秒。基于 `video_duration_sec` 统计。

| ckpt | libero_goal | libero_10 | libero_object | libero_spatial | 总体平均 |
|------|-------------|-----------|---------------|----------------|----------|
| steps_5000 | 17.1 | 48.8 | 20.8 | 16.0 | 25.7 |
| steps_10000 | 17.3 | 46.5 | 15.7 | 12.9 | 23.1 |
| steps_20000 | 13.1 | 37.4 | 13.8 | 11.5 | 18.9 |
| steps_30000 | 13.2 | 39.0 | 15.2 | 11.1 | 19.6 |
| steps_40000 | 13.7 | 33.2 | 14.9 | 11.3 | 18.3 |
| steps_50000 | 12.4 | 30.9 | 15.9 | 11.0 | 17.5 |
| steps_60000 | 12.6 | 30.3 | 14.4 | 11.0 | 17.1 |
| steps_70000 | 12.7 | 30.6 | 14.7 | 10.9 | 17.2 |
| steps_80000 | 12.3 | 30.0 | 14.0 | 11.0 | 16.8 |

## 每个 suite 的 per-task 成功率矩阵

### libero_goal

| task_id | 任务描述 |
|---------|----------|
| 0 | open the middle drawer of the cabinet |
| 1 | open the top drawer and put the bowl inside |
| 2 | push the plate to the front of the stove |
| 3 | put the bowl on the plate |
| 4 | put the bowl on the stove |
| 5 | put the bowl on top of the cabinet |
| 6 | put the cream cheese in the bowl |
| 7 | put the wine bottle on the rack |
| 8 | put the wine bottle on top of the cabinet |
| 9 | turn on the stove |

| ckpt | task_0 | task_1 | task_2 | task_3 | task_4 | task_5 | task_6 | task_7 | task_8 | task_9 |
|------|------|------|------|------|------|------|------|------|------|------|
| steps_5000 | 10% | 8% | 90% | 92% | 94% | 94% | 86% | 4% | 88% | 100% |
| steps_10000 | 10% | 74% | 24% | 96% | 96% | 88% | 56% | 84% | 84% | 100% |
| steps_20000 | 100% | 86% | 52% | 100% | 98% | 98% | 98% | 64% | 96% | 100% |
| steps_30000 | 88% | 32% | 90% | 98% | 100% | 96% | 100% | 74% | 98% | 100% |
| steps_40000 | 94% | 76% | 74% | 100% | 100% | 98% | 94% | 36% | 98% | 100% |
| steps_50000 | 94% | 84% | 96% | 100% | 100% | 98% | 100% | 74% | 98% | 100% |
| steps_60000 | 84% | 84% | 94% | 100% | 100% | 100% | 94% | 80% | 100% | 100% |
| steps_70000 | 92% | 92% | 90% | 100% | 96% | 94% | 96% | 70% | 100% | 100% |
| steps_80000 | 90% | 90% | 98% | 100% | 100% | 98% | 94% | 68% | 100% | 100% |

### libero_10

| task_id | 任务描述 |
|---------|----------|
| 0 | pick up the book and place it in the back compartment of the caddy |
| 1 | put both moka pots on the stove |
| 2 | put both the alphabet soup and the cream cheese box in the basket |
| 3 | put both the alphabet soup and the tomato sauce in the basket |
| 4 | put both the cream cheese box and the butter in the basket |
| 5 | put the black bowl in the bottom drawer of the cabinet and close it |
| 6 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 7 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 8 | put the yellow and white mug in the microwave and close it |
| 9 | turn on the stove and put the moka pot on it |

| ckpt | task_0 | task_1 | task_2 | task_3 | task_4 | task_5 | task_6 | task_7 | task_8 | task_9 |
|------|------|------|------|------|------|------|------|------|------|------|
| steps_5000 | 8% | 0% | 18% | 0% | 22% | 0% | 0% | 2% | 4% | 70% |
| steps_10000 | 22% | 0% | 22% | 20% | 38% | 28% | 44% | 32% | 6% | 36% |
| steps_20000 | 84% | 14% | 76% | 70% | 96% | 90% | 12% | 52% | 6% | 64% |
| steps_30000 | 88% | 6% | 20% | 36% | 90% | 76% | 44% | 56% | 6% | 74% |
| steps_40000 | 86% | 22% | 82% | 70% | 98% | 64% | 66% | 76% | 72% | 80% |
| steps_50000 | 90% | 60% | 90% | 78% | 94% | 88% | 78% | 80% | 82% | 80% |
| steps_60000 | 94% | 52% | 94% | 70% | 94% | 100% | 92% | 76% | 82% | 84% |
| steps_70000 | 96% | 60% | 80% | 82% | 100% | 98% | 88% | 72% | 80% | 86% |
| steps_80000 | 94% | 56% | 92% | 82% | 96% | 90% | 74% | 82% | 82% | 96% |

### libero_object

| task_id | 任务描述 |
|---------|----------|
| 0 | pick up the alphabet soup and place it in the basket |
| 1 | pick up the bbq sauce and place it in the basket |
| 2 | pick up the butter and place it in the basket |
| 3 | pick up the chocolate pudding and place it in the basket |
| 4 | pick up the cream cheese and place it in the basket |
| 5 | pick up the ketchup and place it in the basket |
| 6 | pick up the milk and place it in the basket |
| 7 | pick up the orange juice and place it in the basket |
| 8 | pick up the salad dressing and place it in the basket |
| 9 | pick up the tomato sauce and place it in the basket |

| ckpt | task_0 | task_1 | task_2 | task_3 | task_4 | task_5 | task_6 | task_7 | task_8 | task_9 |
|------|------|------|------|------|------|------|------|------|------|------|
| steps_5000 | 34% | 38% | 84% | 70% | 100% | 38% | 16% | 66% | 78% | 62% |
| steps_10000 | 76% | 80% | 100% | 100% | 88% | 100% | 98% | 98% | 98% | 74% |
| steps_20000 | 94% | 100% | 98% | 100% | 98% | 96% | 100% | 98% | 98% | 98% |
| steps_30000 | 100% | 98% | 100% | 96% | 98% | 94% | 84% | 84% | 100% | 86% |
| steps_40000 | 94% | 96% | 96% | 90% | 94% | 100% | 100% | 96% | 100% | 94% |
| steps_50000 | 96% | 90% | 94% | 90% | 98% | 100% | 100% | 80% | 100% | 86% |
| steps_60000 | 100% | 100% | 98% | 94% | 100% | 98% | 98% | 90% | 100% | 98% |
| steps_70000 | 100% | 98% | 92% | 92% | 96% | 96% | 100% | 96% | 98% | 90% |
| steps_80000 | 100% | 98% | 98% | 96% | 100% | 100% | 98% | 100% | 100% | 100% |

### libero_spatial

| task_id | 任务描述 |
|---------|----------|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate |
| 1 | pick up the black bowl from table center and place it on the plate |
| 2 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate |
| 3 | pick up the black bowl next to the cookie box and place it on the plate |
| 4 | pick up the black bowl next to the plate and place it on the plate |
| 5 | pick up the black bowl next to the ramekin and place it on the plate |
| 6 | pick up the black bowl on the cookie box and place it on the plate |
| 7 | pick up the black bowl on the ramekin and place it on the plate |
| 8 | pick up the black bowl on the stove and place it on the plate |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate |

| ckpt | task_0 | task_1 | task_2 | task_3 | task_4 | task_5 | task_6 | task_7 | task_8 | task_9 |
|------|------|------|------|------|------|------|------|------|------|------|
| steps_5000 | 96% | 90% | 2% | 62% | 34% | 18% | 84% | 40% | 64% | 0% |
| steps_10000 | 92% | 90% | 76% | 94% | 88% | 92% | 98% | 52% | 90% | 80% |
| steps_20000 | 98% | 82% | 86% | 98% | 98% | 100% | 96% | 74% | 94% | 94% |
| steps_30000 | 98% | 98% | 84% | 100% | 90% | 98% | 96% | 90% | 96% | 88% |
| steps_40000 | 100% | 98% | 90% | 96% | 92% | 98% | 98% | 94% | 90% | 98% |
| steps_50000 | 98% | 98% | 100% | 96% | 90% | 98% | 96% | 98% | 96% | 100% |
| steps_60000 | 98% | 100% | 96% | 100% | 90% | 96% | 100% | 96% | 98% | 96% |
| steps_70000 | 100% | 98% | 100% | 98% | 94% | 96% | 94% | 94% | 100% | 98% |
| steps_80000 | 100% | 100% | 94% | 100% | 90% | 96% | 96% | 88% | 96% | 94% |

---

## 每个 ckpt 的成功 episode 平均完成时长

> 单位：秒。仅统计 `success=true` 的 episode，基于 `video_duration_sec` 计算。

| ckpt | libero_goal | libero_10 | libero_object | libero_spatial | 总体平均 |
|------|-------------|-----------|---------------|----------------|----------|
| steps_5000 | 10.6 | 26.8 | 15.7 | 9.8 | 13.1 |
| steps_10000 | 12.2 | 30.0 | 14.5 | 11.3 | 14.3 |
| steps_20000 | 11.0 | 26.0 | 13.5 | 10.6 | 14.2 |
| steps_30000 | 10.8 | 25.8 | 14.3 | 10.4 | 14.0 |
| steps_40000 | 11.2 | 25.8 | 14.3 | 10.8 | 14.9 |
| steps_50000 | 11.3 | 26.3 | 15.0 | 10.6 | 15.4 |
| steps_60000 | 11.5 | 26.1 | 14.0 | 10.7 | 15.2 |
| steps_70000 | 11.3 | 26.6 | 14.1 | 10.6 | 15.3 |
| steps_80000 | 11.1 | 25.9 | 13.8 | 10.4 | 15.0 |

## 每个 suite 的 per-task 成功率矩阵

### libero_goal

| task_id | 任务描述 |
|---------|----------|
| 0 | open the middle drawer of the cabinet |
| 1 | open the top drawer and put the bowl inside |
| 2 | push the plate to the front of the stove |
| 3 | put the bowl on the plate |
| 4 | put the bowl on the stove |
| 5 | put the bowl on top of the cabinet |
| 6 | put the cream cheese in the bowl |
| 7 | put the wine bottle on the rack |
| 8 | put the wine bottle on top of the cabinet |
| 9 | turn on the stove |

| ckpt | task_0 | task_1 | task_2 | task_3 | task_4 | task_5 | task_6 | task_7 | task_8 | task_9 |
|------|------|------|------|------|------|------|------|------|------|------|
| steps_5000 | 10% | 8% | 90% | 92% | 94% | 94% | 86% | 4% | 88% | 100% |
| steps_10000 | 10% | 74% | 24% | 96% | 96% | 88% | 56% | 84% | 84% | 100% |
| steps_20000 | 100% | 86% | 52% | 100% | 98% | 98% | 98% | 64% | 96% | 100% |
| steps_30000 | 88% | 32% | 90% | 98% | 100% | 96% | 100% | 74% | 98% | 100% |
| steps_40000 | 94% | 76% | 74% | 100% | 100% | 98% | 94% | 36% | 98% | 100% |
| steps_50000 | 94% | 84% | 96% | 100% | 100% | 98% | 100% | 74% | 98% | 100% |
| steps_60000 | 84% | 84% | 94% | 100% | 100% | 100% | 94% | 80% | 100% | 100% |
| steps_70000 | 92% | 92% | 90% | 100% | 96% | 94% | 96% | 70% | 100% | 100% |
| steps_80000 | 90% | 90% | 98% | 100% | 100% | 98% | 94% | 68% | 100% | 100% |

### libero_10

| task_id | 任务描述 |
|---------|----------|
| 0 | pick up the book and place it in the back compartment of the caddy |
| 1 | put both moka pots on the stove |
| 2 | put both the alphabet soup and the cream cheese box in the basket |
| 3 | put both the alphabet soup and the tomato sauce in the basket |
| 4 | put both the cream cheese box and the butter in the basket |
| 5 | put the black bowl in the bottom drawer of the cabinet and close it |
| 6 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 7 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 8 | put the yellow and white mug in the microwave and close it |
| 9 | turn on the stove and put the moka pot on it |

| ckpt | task_0 | task_1 | task_2 | task_3 | task_4 | task_5 | task_6 | task_7 | task_8 | task_9 |
|------|------|------|------|------|------|------|------|------|------|------|
| steps_5000 | 8% | 0% | 18% | 0% | 22% | 0% | 0% | 2% | 4% | 70% |
| steps_10000 | 22% | 0% | 22% | 20% | 38% | 28% | 44% | 32% | 6% | 36% |
| steps_20000 | 84% | 14% | 76% | 70% | 96% | 90% | 12% | 52% | 6% | 64% |
| steps_30000 | 88% | 6% | 20% | 36% | 90% | 76% | 44% | 56% | 6% | 74% |
| steps_40000 | 86% | 22% | 82% | 70% | 98% | 64% | 66% | 76% | 72% | 80% |
| steps_50000 | 90% | 60% | 90% | 78% | 94% | 88% | 78% | 80% | 82% | 80% |
| steps_60000 | 94% | 52% | 94% | 70% | 94% | 100% | 92% | 76% | 82% | 84% |
| steps_70000 | 96% | 60% | 80% | 82% | 100% | 98% | 88% | 72% | 80% | 86% |
| steps_80000 | 94% | 56% | 92% | 82% | 96% | 90% | 74% | 82% | 82% | 96% |

### libero_object

| task_id | 任务描述 |
|---------|----------|
| 0 | pick up the alphabet soup and place it in the basket |
| 1 | pick up the bbq sauce and place it in the basket |
| 2 | pick up the butter and place it in the basket |
| 3 | pick up the chocolate pudding and place it in the basket |
| 4 | pick up the cream cheese and place it in the basket |
| 5 | pick up the ketchup and place it in the basket |
| 6 | pick up the milk and place it in the basket |
| 7 | pick up the orange juice and place it in the basket |
| 8 | pick up the salad dressing and place it in the basket |
| 9 | pick up the tomato sauce and place it in the basket |

| ckpt | task_0 | task_1 | task_2 | task_3 | task_4 | task_5 | task_6 | task_7 | task_8 | task_9 |
|------|------|------|------|------|------|------|------|------|------|------|
| steps_5000 | 34% | 38% | 84% | 70% | 100% | 38% | 16% | 66% | 78% | 62% |
| steps_10000 | 76% | 80% | 100% | 100% | 88% | 100% | 98% | 98% | 98% | 74% |
| steps_20000 | 94% | 100% | 98% | 100% | 98% | 96% | 100% | 98% | 98% | 98% |
| steps_30000 | 100% | 98% | 100% | 96% | 98% | 94% | 84% | 84% | 100% | 86% |
| steps_40000 | 94% | 96% | 96% | 90% | 94% | 100% | 100% | 96% | 100% | 94% |
| steps_50000 | 96% | 90% | 94% | 90% | 98% | 100% | 100% | 80% | 100% | 86% |
| steps_60000 | 100% | 100% | 98% | 94% | 100% | 98% | 98% | 90% | 100% | 98% |
| steps_70000 | 100% | 98% | 92% | 92% | 96% | 96% | 100% | 96% | 98% | 90% |
| steps_80000 | 100% | 98% | 98% | 96% | 100% | 100% | 98% | 100% | 100% | 100% |

### libero_spatial

| task_id | 任务描述 |
|---------|----------|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate |
| 1 | pick up the black bowl from table center and place it on the plate |
| 2 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate |
| 3 | pick up the black bowl next to the cookie box and place it on the plate |
| 4 | pick up the black bowl next to the plate and place it on the plate |
| 5 | pick up the black bowl next to the ramekin and place it on the plate |
| 6 | pick up the black bowl on the cookie box and place it on the plate |
| 7 | pick up the black bowl on the ramekin and place it on the plate |
| 8 | pick up the black bowl on the stove and place it on the plate |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate |

| ckpt | task_0 | task_1 | task_2 | task_3 | task_4 | task_5 | task_6 | task_7 | task_8 | task_9 |
|------|------|------|------|------|------|------|------|------|------|------|
| steps_5000 | 96% | 90% | 2% | 62% | 34% | 18% | 84% | 40% | 64% | 0% |
| steps_10000 | 92% | 90% | 76% | 94% | 88% | 92% | 98% | 52% | 90% | 80% |
| steps_20000 | 98% | 82% | 86% | 98% | 98% | 100% | 96% | 74% | 94% | 94% |
| steps_30000 | 98% | 98% | 84% | 100% | 90% | 98% | 96% | 90% | 96% | 88% |
| steps_40000 | 100% | 98% | 90% | 96% | 92% | 98% | 98% | 94% | 90% | 98% |
| steps_50000 | 98% | 98% | 100% | 96% | 90% | 98% | 96% | 98% | 96% | 100% |
| steps_60000 | 98% | 100% | 96% | 100% | 90% | 96% | 100% | 96% | 98% | 96% |
| steps_70000 | 100% | 98% | 100% | 98% | 94% | 96% | 94% | 94% | 100% | 98% |
| steps_80000 | 100% | 100% | 94% | 100% | 90% | 96% | 96% | 88% | 96% | 94% |

---

## 每个 ckpt 成功 episode 的平均完成部署步数 / 时长

> `steps_executed` 为成功 episode 中实际执行的 env step 数；按 LIBERO 10Hz 控制频率换算为秒。仅统计 `success=true` 的 episode。

| ckpt | goal_steps | goal_sec | 10_steps | 10_sec | object_steps | object_sec | spatial_steps | spatial_sec | overall_steps | overall_sec |
|------|------------|----------|----------|--------|--------------|------------|---------------|-------------|---------------|-------------|
| steps_5000 | 105.2 | 10.5 | 267.2 | 26.7 | 155.6 | 15.6 | 97.1 | 9.7 | 129.7 | 13.0 |
| steps_10000 | 120.6 | 12.1 | 298.6 | 29.9 | 144.3 | 14.4 | 112.5 | 11.2 | 142.2 | 14.2 |
| steps_20000 | 109.3 | 10.9 | 259.4 | 25.9 | 133.8 | 13.4 | 105.3 | 10.5 | 140.6 | 14.1 |
| steps_30000 | 107.3 | 10.7 | 256.9 | 25.7 | 142.4 | 14.2 | 103.2 | 10.3 | 139.1 | 13.9 |
| steps_40000 | 111.1 | 11.1 | 257.1 | 25.7 | 142.3 | 14.2 | 107.3 | 10.7 | 148.5 | 14.8 |
| steps_50000 | 112.2 | 11.2 | 261.6 | 26.2 | 149.1 | 14.9 | 105.1 | 10.5 | 153.1 | 15.3 |
| steps_60000 | 113.5 | 11.4 | 259.8 | 26.0 | 139.2 | 13.9 | 105.9 | 10.6 | 151.2 | 15.1 |
| steps_70000 | 112.5 | 11.2 | 264.9 | 26.5 | 140.0 | 14.0 | 104.7 | 10.5 | 152.2 | 15.2 |
| steps_80000 | 109.9 | 11.0 | 258.3 | 25.8 | 137.2 | 13.7 | 103.5 | 10.3 | 149.1 | 14.9 |

## 每个 suite 的 per-task 成功率矩阵

### libero_goal

| task_id | 任务描述 |
|---------|----------|
| 0 | open the middle drawer of the cabinet |
| 1 | open the top drawer and put the bowl inside |
| 2 | push the plate to the front of the stove |
| 3 | put the bowl on the plate |
| 4 | put the bowl on the stove |
| 5 | put the bowl on top of the cabinet |
| 6 | put the cream cheese in the bowl |
| 7 | put the wine bottle on the rack |
| 8 | put the wine bottle on top of the cabinet |
| 9 | turn on the stove |

| ckpt | task_0 | task_1 | task_2 | task_3 | task_4 | task_5 | task_6 | task_7 | task_8 | task_9 |
|------|------|------|------|------|------|------|------|------|------|------|
| steps_5000 | 10% | 8% | 90% | 92% | 94% | 94% | 86% | 4% | 88% | 100% |
| steps_10000 | 10% | 74% | 24% | 96% | 96% | 88% | 56% | 84% | 84% | 100% |
| steps_20000 | 100% | 86% | 52% | 100% | 98% | 98% | 98% | 64% | 96% | 100% |
| steps_30000 | 88% | 32% | 90% | 98% | 100% | 96% | 100% | 74% | 98% | 100% |
| steps_40000 | 94% | 76% | 74% | 100% | 100% | 98% | 94% | 36% | 98% | 100% |
| steps_50000 | 94% | 84% | 96% | 100% | 100% | 98% | 100% | 74% | 98% | 100% |
| steps_60000 | 84% | 84% | 94% | 100% | 100% | 100% | 94% | 80% | 100% | 100% |
| steps_70000 | 92% | 92% | 90% | 100% | 96% | 94% | 96% | 70% | 100% | 100% |
| steps_80000 | 90% | 90% | 98% | 100% | 100% | 98% | 94% | 68% | 100% | 100% |

### libero_10

| task_id | 任务描述 |
|---------|----------|
| 0 | pick up the book and place it in the back compartment of the caddy |
| 1 | put both moka pots on the stove |
| 2 | put both the alphabet soup and the cream cheese box in the basket |
| 3 | put both the alphabet soup and the tomato sauce in the basket |
| 4 | put both the cream cheese box and the butter in the basket |
| 5 | put the black bowl in the bottom drawer of the cabinet and close it |
| 6 | put the white mug on the left plate and put the yellow and white mug on the right plate |
| 7 | put the white mug on the plate and put the chocolate pudding to the right of the plate |
| 8 | put the yellow and white mug in the microwave and close it |
| 9 | turn on the stove and put the moka pot on it |

| ckpt | task_0 | task_1 | task_2 | task_3 | task_4 | task_5 | task_6 | task_7 | task_8 | task_9 |
|------|------|------|------|------|------|------|------|------|------|------|
| steps_5000 | 8% | 0% | 18% | 0% | 22% | 0% | 0% | 2% | 4% | 70% |
| steps_10000 | 22% | 0% | 22% | 20% | 38% | 28% | 44% | 32% | 6% | 36% |
| steps_20000 | 84% | 14% | 76% | 70% | 96% | 90% | 12% | 52% | 6% | 64% |
| steps_30000 | 88% | 6% | 20% | 36% | 90% | 76% | 44% | 56% | 6% | 74% |
| steps_40000 | 86% | 22% | 82% | 70% | 98% | 64% | 66% | 76% | 72% | 80% |
| steps_50000 | 90% | 60% | 90% | 78% | 94% | 88% | 78% | 80% | 82% | 80% |
| steps_60000 | 94% | 52% | 94% | 70% | 94% | 100% | 92% | 76% | 82% | 84% |
| steps_70000 | 96% | 60% | 80% | 82% | 100% | 98% | 88% | 72% | 80% | 86% |
| steps_80000 | 94% | 56% | 92% | 82% | 96% | 90% | 74% | 82% | 82% | 96% |

### libero_object

| task_id | 任务描述 |
|---------|----------|
| 0 | pick up the alphabet soup and place it in the basket |
| 1 | pick up the bbq sauce and place it in the basket |
| 2 | pick up the butter and place it in the basket |
| 3 | pick up the chocolate pudding and place it in the basket |
| 4 | pick up the cream cheese and place it in the basket |
| 5 | pick up the ketchup and place it in the basket |
| 6 | pick up the milk and place it in the basket |
| 7 | pick up the orange juice and place it in the basket |
| 8 | pick up the salad dressing and place it in the basket |
| 9 | pick up the tomato sauce and place it in the basket |

| ckpt | task_0 | task_1 | task_2 | task_3 | task_4 | task_5 | task_6 | task_7 | task_8 | task_9 |
|------|------|------|------|------|------|------|------|------|------|------|
| steps_5000 | 34% | 38% | 84% | 70% | 100% | 38% | 16% | 66% | 78% | 62% |
| steps_10000 | 76% | 80% | 100% | 100% | 88% | 100% | 98% | 98% | 98% | 74% |
| steps_20000 | 94% | 100% | 98% | 100% | 98% | 96% | 100% | 98% | 98% | 98% |
| steps_30000 | 100% | 98% | 100% | 96% | 98% | 94% | 84% | 84% | 100% | 86% |
| steps_40000 | 94% | 96% | 96% | 90% | 94% | 100% | 100% | 96% | 100% | 94% |
| steps_50000 | 96% | 90% | 94% | 90% | 98% | 100% | 100% | 80% | 100% | 86% |
| steps_60000 | 100% | 100% | 98% | 94% | 100% | 98% | 98% | 90% | 100% | 98% |
| steps_70000 | 100% | 98% | 92% | 92% | 96% | 96% | 100% | 96% | 98% | 90% |
| steps_80000 | 100% | 98% | 98% | 96% | 100% | 100% | 98% | 100% | 100% | 100% |

### libero_spatial

| task_id | 任务描述 |
|---------|----------|
| 0 | pick up the black bowl between the plate and the ramekin and place it on the plate |
| 1 | pick up the black bowl from table center and place it on the plate |
| 2 | pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate |
| 3 | pick up the black bowl next to the cookie box and place it on the plate |
| 4 | pick up the black bowl next to the plate and place it on the plate |
| 5 | pick up the black bowl next to the ramekin and place it on the plate |
| 6 | pick up the black bowl on the cookie box and place it on the plate |
| 7 | pick up the black bowl on the ramekin and place it on the plate |
| 8 | pick up the black bowl on the stove and place it on the plate |
| 9 | pick up the black bowl on the wooden cabinet and place it on the plate |

| ckpt | task_0 | task_1 | task_2 | task_3 | task_4 | task_5 | task_6 | task_7 | task_8 | task_9 |
|------|------|------|------|------|------|------|------|------|------|------|
| steps_5000 | 96% | 90% | 2% | 62% | 34% | 18% | 84% | 40% | 64% | 0% |
| steps_10000 | 92% | 90% | 76% | 94% | 88% | 92% | 98% | 52% | 90% | 80% |
| steps_20000 | 98% | 82% | 86% | 98% | 98% | 100% | 96% | 74% | 94% | 94% |
| steps_30000 | 98% | 98% | 84% | 100% | 90% | 98% | 96% | 90% | 96% | 88% |
| steps_40000 | 100% | 98% | 90% | 96% | 92% | 98% | 98% | 94% | 90% | 98% |
| steps_50000 | 98% | 98% | 100% | 96% | 90% | 98% | 96% | 98% | 96% | 100% |
| steps_60000 | 98% | 100% | 96% | 100% | 90% | 96% | 100% | 96% | 98% | 96% |
| steps_70000 | 100% | 98% | 100% | 98% | 94% | 96% | 94% | 94% | 100% | 98% |
| steps_80000 | 100% | 100% | 94% | 100% | 90% | 96% | 96% | 88% | 96% | 94% |
