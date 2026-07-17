# P0-M7-E-H2a-02: StarFlow LIBERO 4-in-1 Qwen3VL-4B future_tokens=16（bs32）state-fix 评估追踪

> **实验代号**: E-H2a-02 / P0-M7  
> **状态**: ✅ 评估完成  
> **run_id**: `P0-M7-E-H2a-02_starflow_libero-4in1_qwen3vl4b_lwfm_ft16_260703_0854`  
> **启动时间**: 2026-07-16 12:51 CST  
> **当前更新**: 2026-07-17 12:08 CST  
> **tmux 会话**: `test`  
> **配置来源**: `configs/starflow_vla/ablations/future_tokens_16.yaml`  
> **评测输出**: `playground/starflow_eval_result`

---

## 评估概述

本 run 为 StarFlowVLA `num_target_vision_tokens=16`（ft16）variant。2026-07-16 修复 `eval_libero.py` 未传入 robot state 及系统缺少 EGL loader 的问题后，使用 `playground/starflow_eval_plan.sh` 对全部关键 ckpt 进行重新评测，结果固定输出到 `playground/starflow_eval_result/`。

### 评测参数

| 参数 | 值 |
|------|-----|
| `RUN_ID` | `P0-M7-E-H2a-02_starflow_libero-4in1_qwen3vl4b_lwfm_ft16_260703_0854` |
| `workers` | 10 |
| `num_trials` | 50 |
| `suites` | libero_goal, libero_10, libero_object, libero_spatial |
| `ckpt 列表` | steps_5000, steps_10000, steps_20000, steps_30000, steps_40000, steps_50000, steps_60000, steps_70000, steps_80000 |
| `policy server` | 单 server，固定端口 6710-6718 |
| `server 优化` | server-side batching（max_batch=8, timeout=30ms） |

---

## 评估进度

> 最后更新：2026-07-17 12:08 CST

| ckpt | 状态 | 总 eps | 总 successes | 总 SR |
|------|------|--------|--------------|-------|
| steps_5000 | ✅ 完成 | 2000 / 2000 | 791 | 39.6% |
| steps_10000 | ✅ 完成 | 2000 / 2000 | 1389 | 69.5% |
| steps_20000 | ✅ 完成 | 2000 / 2000 | 1638 | 81.9% |
| steps_30000 | ✅ 完成 | 2000 / 2000 | 1633 | 81.7% |
| steps_40000 | ✅ 完成 | 2000 / 2000 | 1750 | 87.5% |
| steps_50000 | ✅ 完成 | 2000 / 2000 | 1855 | 92.8% |
| steps_60000 | ✅ 完成 | 2000 / 2000 | 1843 | 92.2% |
| steps_70000 | ✅ 完成 | 2000 / 2000 | 1902 | 95.1% |
| steps_80000 | ✅ 完成 | 2000 / 2000 | 1891 | 94.5% |

### 已完成的 ckpt 分 suite 结果

#### steps_5000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 247 | 49.4% |
| libero_10 | 500 | 51 | 10.2% |
| libero_object | 500 | 283 | 56.6% |
| libero_spatial | 500 | 210 | 42.0% |
| **合计** | **2000** | **791** | **39.6%** |

#### steps_10000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 352 | 70.4% |
| libero_10 | 500 | 158 | 31.6% |
| libero_object | 500 | 426 | 85.2% |
| libero_spatial | 500 | 453 | 90.6% |
| **合计** | **2000** | **1389** | **69.5%** |

#### steps_20000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 405 | 81.0% |
| libero_10 | 500 | 284 | 56.8% |
| libero_object | 500 | 478 | 95.6% |
| libero_spatial | 500 | 471 | 94.2% |
| **合计** | **2000** | **1638** | **81.9%** |

#### steps_30000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 433 | 86.6% |
| libero_10 | 500 | 302 | 60.4% |
| libero_object | 500 | 458 | 91.6% |
| libero_spatial | 500 | 440 | 88.0% |
| **合计** | **2000** | **1633** | **81.7%** |

#### steps_40000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 445 | 89.0% |
| libero_10 | 500 | 332 | 66.4% |
| libero_object | 500 | 491 | 98.2% |
| libero_spatial | 500 | 482 | 96.4% |
| **合计** | **2000** | **1750** | **87.5%** |

#### steps_50000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 463 | 92.6% |
| libero_10 | 500 | 422 | 84.4% |
| libero_object | 500 | 479 | 95.8% |
| libero_spatial | 500 | 491 | 98.2% |
| **合计** | **2000** | **1855** | **92.8%** |

#### steps_60000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 474 | 94.8% |
| libero_10 | 500 | 412 | 82.4% |
| libero_object | 500 | 482 | 96.4% |
| libero_spatial | 500 | 475 | 95.0% |
| **合计** | **2000** | **1843** | **92.2%** |

#### steps_70000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 481 | 96.2% |
| libero_10 | 500 | 441 | 88.2% |
| libero_object | 500 | 495 | 99.0% |
| libero_spatial | 500 | 485 | 97.0% |
| **合计** | **2000** | **1902** | **95.1%** |

#### steps_80000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 483 | 96.6% |
| libero_10 | 500 | 432 | 86.4% |
| libero_object | 500 | 492 | 98.4% |
| libero_spatial | 500 | 484 | 96.8% |
| **合计** | **2000** | **1891** | **94.5%** |

---

## state 修复 / 流程修复前后对比

本次重新评测前，`playground/eval_results/` 中 ft16 的老结果存在 state 未传入、EGL 库缺失导致子进程频繁崩溃等问题。对比如下：

| ckpt | 老结果（eval_results） | 新结果（starflow_eval_result） | 差异 |
|------|------------------------|-------------------------------|------|
| steps_5000 | 21.2% | 39.6% | +18.4% |
| steps_10000 | 39.9% | 69.5% | +29.6% |
| steps_20000 | 68.4% | 81.9% | +13.5% |
| steps_30000 | 58.4% | 81.7% | +23.3% |
| steps_40000 | 78.0% | 87.5% | +9.5% |
| steps_50000 | — | 92.8% | — |
| steps_60000 | 73.7% | 92.2% | +18.5% |
| steps_70000 | 83.4% | 95.1% | +11.7% |
| steps_80000 | 82.3% | 94.5% | +12.2% |

---

## 备注

- 修复 commit：
  - `b7af6758 fix(LIBERO): pass robot state to model during evaluation`
  - 系统修复：安装 `libegl1` 以解决 PyOpenGL / mujoco EGL 初始化失败
- 评测过程中使用 workers=10，当前 speed 约 18-20 eps/min。
- 每个 ckpt 固定端口：steps_5000→6710，steps_10000→6711，...，steps_80000→6718。
- 新增 `eval_pool_manager.py` 子进程诊断日志，便于定位崩溃原因。
- **2026-07-17 09:30 更新**：ft16 全部 9 个 ckpt 评估已完成。最终 `steps_70000` SR 达到 **95.1%**，为本次 ft16 评估中 SR 最高的 ckpt。

---

---

## 每个 ckpt 的平均完成时长

> 单位：秒。基于 `video_duration_sec` 统计。

| ckpt | libero_goal | libero_10 | libero_object | libero_spatial | 总体平均 |
|------|-------------|-----------|---------------|----------------|----------|
| steps_5000 | 20.3 | 49.9 | 21.0 | 16.9 | 27.0 |
| steps_10000 | 17.0 | 43.6 | 16.2 | 11.8 | 22.1 |
| steps_20000 | 14.5 | 36.6 | 13.7 | 11.0 | 19.0 |
| steps_30000 | 13.9 | 37.5 | 15.9 | 12.0 | 19.8 |
| steps_40000 | 13.0 | 34.5 | 14.0 | 11.1 | 18.1 |
| steps_50000 | 12.5 | 30.0 | 14.4 | 10.7 | 16.9 |
| steps_60000 | 12.1 | 31.1 | 14.4 | 11.2 | 17.2 |
| steps_70000 | 11.8 | 28.9 | 13.9 | 10.8 | 16.3 |
| steps_80000 | 11.6 | 29.5 | 13.9 | 10.7 | 16.5 |

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
| steps_5000 | 24% | 0% | 98% | 50% | 36% | 96% | 18% | 2% | 70% | 100% |
| steps_10000 | 68% | 44% | 32% | 100% | 94% | 88% | 66% | 60% | 80% | 72% |
| steps_20000 | 44% | 92% | 52% | 98% | 94% | 98% | 100% | 36% | 96% | 100% |
| steps_30000 | 98% | 64% | 82% | 98% | 100% | 86% | 78% | 66% | 94% | 100% |
| steps_40000 | 96% | 66% | 94% | 100% | 98% | 98% | 96% | 42% | 100% | 100% |
| steps_50000 | 92% | 80% | 88% | 100% | 94% | 94% | 98% | 82% | 98% | 100% |
| steps_60000 | 96% | 84% | 96% | 100% | 98% | 96% | 96% | 82% | 100% | 100% |
| steps_70000 | 100% | 84% | 98% | 100% | 98% | 94% | 98% | 90% | 100% | 100% |
| steps_80000 | 98% | 90% | 96% | 100% | 100% | 98% | 94% | 90% | 100% | 100% |

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
| steps_5000 | 0% | 0% | 26% | 0% | 16% | 14% | 0% | 0% | 2% | 44% |
| steps_10000 | 64% | 2% | 40% | 16% | 34% | 76% | 38% | 16% | 4% | 26% |
| steps_20000 | 82% | 14% | 74% | 80% | 82% | 88% | 18% | 74% | 14% | 42% |
| steps_30000 | 60% | 6% | 68% | 56% | 92% | 90% | 54% | 8% | 86% | 84% |
| steps_40000 | 92% | 12% | 90% | 86% | 98% | 30% | 64% | 38% | 68% | 86% |
| steps_50000 | 92% | 70% | 96% | 86% | 96% | 78% | 88% | 72% | 82% | 84% |
| steps_60000 | 84% | 60% | 86% | 86% | 96% | 98% | 74% | 70% | 84% | 86% |
| steps_70000 | 92% | 60% | 98% | 92% | 96% | 96% | 88% | 84% | 78% | 98% |
| steps_80000 | 90% | 76% | 98% | 84% | 96% | 88% | 82% | 76% | 90% | 84% |

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
| steps_5000 | 16% | 36% | 66% | 78% | 92% | 42% | 14% | 62% | 86% | 74% |
| steps_10000 | 48% | 70% | 98% | 90% | 94% | 78% | 96% | 96% | 96% | 86% |
| steps_20000 | 88% | 96% | 100% | 100% | 96% | 100% | 100% | 82% | 100% | 94% |
| steps_30000 | 100% | 96% | 96% | 74% | 86% | 100% | 98% | 92% | 100% | 74% |
| steps_40000 | 98% | 96% | 100% | 98% | 98% | 100% | 100% | 92% | 100% | 100% |
| steps_50000 | 100% | 94% | 98% | 100% | 96% | 100% | 100% | 80% | 100% | 90% |
| steps_60000 | 100% | 90% | 100% | 96% | 100% | 100% | 100% | 84% | 98% | 96% |
| steps_70000 | 100% | 96% | 100% | 98% | 100% | 100% | 100% | 98% | 100% | 98% |
| steps_80000 | 100% | 98% | 100% | 96% | 100% | 100% | 100% | 92% | 100% | 98% |

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
| steps_5000 | 72% | 86% | 2% | 58% | 16% | 0% | 60% | 42% | 64% | 20% |
| steps_10000 | 80% | 98% | 92% | 92% | 98% | 100% | 96% | 88% | 90% | 72% |
| steps_20000 | 100% | 84% | 96% | 94% | 100% | 98% | 96% | 82% | 94% | 98% |
| steps_30000 | 98% | 74% | 84% | 100% | 84% | 86% | 82% | 76% | 100% | 96% |
| steps_40000 | 100% | 100% | 94% | 96% | 100% | 98% | 100% | 86% | 96% | 94% |
| steps_50000 | 98% | 100% | 90% | 100% | 100% | 100% | 100% | 98% | 96% | 100% |
| steps_60000 | 100% | 98% | 98% | 100% | 98% | 92% | 90% | 88% | 90% | 96% |
| steps_70000 | 100% | 100% | 96% | 100% | 98% | 100% | 96% | 88% | 98% | 94% |
| steps_80000 | 98% | 100% | 98% | 96% | 88% | 100% | 100% | 96% | 92% | 100% |

---

## 每个 ckpt 的成功 episode 平均完成时长

> 单位：秒。仅统计 `success=true` 的 episode，基于 `video_duration_sec` 计算。

| ckpt | libero_goal | libero_10 | libero_object | libero_spatial | 总体平均 |
|------|-------------|-----------|---------------|----------------|----------|
| steps_5000 | 10.4 | 30.9 | 15.7 | 9.9 | 13.5 |
| steps_10000 | 11.5 | 26.5 | 14.2 | 10.9 | 13.8 |
| steps_20000 | 10.9 | 24.8 | 13.0 | 10.3 | 13.8 |
| steps_30000 | 11.4 | 28.1 | 14.8 | 10.6 | 15.2 |
| steps_40000 | 10.9 | 25.6 | 13.8 | 10.7 | 14.4 |
| steps_50000 | 11.1 | 25.9 | 13.8 | 10.5 | 15.0 |
| steps_60000 | 11.2 | 26.6 | 13.9 | 10.6 | 15.2 |
| steps_70000 | 11.1 | 25.8 | 13.8 | 10.4 | 15.0 |
| steps_80000 | 11.0 | 26.0 | 13.7 | 10.4 | 15.0 |

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
| steps_5000 | 24% | 0% | 98% | 50% | 36% | 96% | 18% | 2% | 70% | 100% |
| steps_10000 | 68% | 44% | 32% | 100% | 94% | 88% | 66% | 60% | 80% | 72% |
| steps_20000 | 44% | 92% | 52% | 98% | 94% | 98% | 100% | 36% | 96% | 100% |
| steps_30000 | 98% | 64% | 82% | 98% | 100% | 86% | 78% | 66% | 94% | 100% |
| steps_40000 | 96% | 66% | 94% | 100% | 98% | 98% | 96% | 42% | 100% | 100% |
| steps_50000 | 92% | 80% | 88% | 100% | 94% | 94% | 98% | 82% | 98% | 100% |
| steps_60000 | 96% | 84% | 96% | 100% | 98% | 96% | 96% | 82% | 100% | 100% |
| steps_70000 | 100% | 84% | 98% | 100% | 98% | 94% | 98% | 90% | 100% | 100% |
| steps_80000 | 98% | 90% | 96% | 100% | 100% | 98% | 94% | 90% | 100% | 100% |

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
| steps_5000 | 0% | 0% | 26% | 0% | 16% | 14% | 0% | 0% | 2% | 44% |
| steps_10000 | 64% | 2% | 40% | 16% | 34% | 76% | 38% | 16% | 4% | 26% |
| steps_20000 | 82% | 14% | 74% | 80% | 82% | 88% | 18% | 74% | 14% | 42% |
| steps_30000 | 60% | 6% | 68% | 56% | 92% | 90% | 54% | 8% | 86% | 84% |
| steps_40000 | 92% | 12% | 90% | 86% | 98% | 30% | 64% | 38% | 68% | 86% |
| steps_50000 | 92% | 70% | 96% | 86% | 96% | 78% | 88% | 72% | 82% | 84% |
| steps_60000 | 84% | 60% | 86% | 86% | 96% | 98% | 74% | 70% | 84% | 86% |
| steps_70000 | 92% | 60% | 98% | 92% | 96% | 96% | 88% | 84% | 78% | 98% |
| steps_80000 | 90% | 76% | 98% | 84% | 96% | 88% | 82% | 76% | 90% | 84% |

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
| steps_5000 | 16% | 36% | 66% | 78% | 92% | 42% | 14% | 62% | 86% | 74% |
| steps_10000 | 48% | 70% | 98% | 90% | 94% | 78% | 96% | 96% | 96% | 86% |
| steps_20000 | 88% | 96% | 100% | 100% | 96% | 100% | 100% | 82% | 100% | 94% |
| steps_30000 | 100% | 96% | 96% | 74% | 86% | 100% | 98% | 92% | 100% | 74% |
| steps_40000 | 98% | 96% | 100% | 98% | 98% | 100% | 100% | 92% | 100% | 100% |
| steps_50000 | 100% | 94% | 98% | 100% | 96% | 100% | 100% | 80% | 100% | 90% |
| steps_60000 | 100% | 90% | 100% | 96% | 100% | 100% | 100% | 84% | 98% | 96% |
| steps_70000 | 100% | 96% | 100% | 98% | 100% | 100% | 100% | 98% | 100% | 98% |
| steps_80000 | 100% | 98% | 100% | 96% | 100% | 100% | 100% | 92% | 100% | 98% |

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
| steps_5000 | 72% | 86% | 2% | 58% | 16% | 0% | 60% | 42% | 64% | 20% |
| steps_10000 | 80% | 98% | 92% | 92% | 98% | 100% | 96% | 88% | 90% | 72% |
| steps_20000 | 100% | 84% | 96% | 94% | 100% | 98% | 96% | 82% | 94% | 98% |
| steps_30000 | 98% | 74% | 84% | 100% | 84% | 86% | 82% | 76% | 100% | 96% |
| steps_40000 | 100% | 100% | 94% | 96% | 100% | 98% | 100% | 86% | 96% | 94% |
| steps_50000 | 98% | 100% | 90% | 100% | 100% | 100% | 100% | 98% | 96% | 100% |
| steps_60000 | 100% | 98% | 98% | 100% | 98% | 92% | 90% | 88% | 90% | 96% |
| steps_70000 | 100% | 100% | 96% | 100% | 98% | 100% | 96% | 88% | 98% | 94% |
| steps_80000 | 98% | 100% | 98% | 96% | 88% | 100% | 100% | 96% | 92% | 100% |

---

## 每个 ckpt 成功 episode 的平均完成部署步数 / 时长

> `steps_executed` 为成功 episode 中实际执行的 env step 数；按 LIBERO 10Hz 控制频率换算为秒。仅统计 `success=true` 的 episode。

| ckpt | goal_steps | goal_sec | 10_steps | 10_sec | object_steps | object_sec | spatial_steps | spatial_sec | overall_steps | overall_sec |
|------|------------|----------|----------|--------|--------------|------------|---------------|-------------|---------------|-------------|
| steps_5000 | 102.9 | 10.3 | 308.3 | 30.8 | 155.7 | 15.6 | 97.9 | 9.8 | 133.7 | 13.4 |
| steps_10000 | 114.2 | 11.4 | 263.6 | 26.4 | 140.7 | 14.1 | 107.7 | 10.8 | 137.2 | 13.7 |
| steps_20000 | 107.7 | 10.8 | 247.4 | 24.7 | 129.5 | 12.9 | 102.4 | 10.2 | 136.8 | 13.7 |
| steps_30000 | 112.9 | 11.3 | 279.9 | 28.0 | 147.1 | 14.7 | 105.4 | 10.5 | 151.4 | 15.1 |
| steps_40000 | 107.6 | 10.8 | 255.2 | 25.5 | 136.5 | 13.7 | 106.1 | 10.6 | 143.3 | 14.3 |
| steps_50000 | 110.2 | 11.0 | 258.4 | 25.8 | 137.5 | 13.7 | 104.2 | 10.4 | 149.4 | 14.9 |
| steps_60000 | 110.6 | 11.1 | 265.3 | 26.5 | 138.0 | 13.8 | 104.9 | 10.5 | 150.9 | 15.1 |
| steps_70000 | 109.7 | 11.0 | 257.2 | 25.7 | 136.9 | 13.7 | 103.1 | 10.3 | 149.3 | 14.9 |
| steps_80000 | 108.9 | 10.9 | 259.1 | 25.9 | 136.1 | 13.6 | 102.6 | 10.3 | 148.7 | 14.9 |

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
| steps_5000 | 24% | 0% | 98% | 50% | 36% | 96% | 18% | 2% | 70% | 100% |
| steps_10000 | 68% | 44% | 32% | 100% | 94% | 88% | 66% | 60% | 80% | 72% |
| steps_20000 | 44% | 92% | 52% | 98% | 94% | 98% | 100% | 36% | 96% | 100% |
| steps_30000 | 98% | 64% | 82% | 98% | 100% | 86% | 78% | 66% | 94% | 100% |
| steps_40000 | 96% | 66% | 94% | 100% | 98% | 98% | 96% | 42% | 100% | 100% |
| steps_50000 | 92% | 80% | 88% | 100% | 94% | 94% | 98% | 82% | 98% | 100% |
| steps_60000 | 96% | 84% | 96% | 100% | 98% | 96% | 96% | 82% | 100% | 100% |
| steps_70000 | 100% | 84% | 98% | 100% | 98% | 94% | 98% | 90% | 100% | 100% |
| steps_80000 | 98% | 90% | 96% | 100% | 100% | 98% | 94% | 90% | 100% | 100% |

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
| steps_5000 | 0% | 0% | 26% | 0% | 16% | 14% | 0% | 0% | 2% | 44% |
| steps_10000 | 64% | 2% | 40% | 16% | 34% | 76% | 38% | 16% | 4% | 26% |
| steps_20000 | 82% | 14% | 74% | 80% | 82% | 88% | 18% | 74% | 14% | 42% |
| steps_30000 | 60% | 6% | 68% | 56% | 92% | 90% | 54% | 8% | 86% | 84% |
| steps_40000 | 92% | 12% | 90% | 86% | 98% | 30% | 64% | 38% | 68% | 86% |
| steps_50000 | 92% | 70% | 96% | 86% | 96% | 78% | 88% | 72% | 82% | 84% |
| steps_60000 | 84% | 60% | 86% | 86% | 96% | 98% | 74% | 70% | 84% | 86% |
| steps_70000 | 92% | 60% | 98% | 92% | 96% | 96% | 88% | 84% | 78% | 98% |
| steps_80000 | 90% | 76% | 98% | 84% | 96% | 88% | 82% | 76% | 90% | 84% |

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
| steps_5000 | 16% | 36% | 66% | 78% | 92% | 42% | 14% | 62% | 86% | 74% |
| steps_10000 | 48% | 70% | 98% | 90% | 94% | 78% | 96% | 96% | 96% | 86% |
| steps_20000 | 88% | 96% | 100% | 100% | 96% | 100% | 100% | 82% | 100% | 94% |
| steps_30000 | 100% | 96% | 96% | 74% | 86% | 100% | 98% | 92% | 100% | 74% |
| steps_40000 | 98% | 96% | 100% | 98% | 98% | 100% | 100% | 92% | 100% | 100% |
| steps_50000 | 100% | 94% | 98% | 100% | 96% | 100% | 100% | 80% | 100% | 90% |
| steps_60000 | 100% | 90% | 100% | 96% | 100% | 100% | 100% | 84% | 98% | 96% |
| steps_70000 | 100% | 96% | 100% | 98% | 100% | 100% | 100% | 98% | 100% | 98% |
| steps_80000 | 100% | 98% | 100% | 96% | 100% | 100% | 100% | 92% | 100% | 98% |

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
| steps_5000 | 72% | 86% | 2% | 58% | 16% | 0% | 60% | 42% | 64% | 20% |
| steps_10000 | 80% | 98% | 92% | 92% | 98% | 100% | 96% | 88% | 90% | 72% |
| steps_20000 | 100% | 84% | 96% | 94% | 100% | 98% | 96% | 82% | 94% | 98% |
| steps_30000 | 98% | 74% | 84% | 100% | 84% | 86% | 82% | 76% | 100% | 96% |
| steps_40000 | 100% | 100% | 94% | 96% | 100% | 98% | 100% | 86% | 96% | 94% |
| steps_50000 | 98% | 100% | 90% | 100% | 100% | 100% | 100% | 98% | 96% | 100% |
| steps_60000 | 100% | 98% | 98% | 100% | 98% | 92% | 90% | 88% | 90% | 96% |
| steps_70000 | 100% | 100% | 96% | 100% | 98% | 100% | 96% | 88% | 98% | 94% |
| steps_80000 | 98% | 100% | 98% | 96% | 88% | 100% | 100% | 96% | 92% | 100% |
