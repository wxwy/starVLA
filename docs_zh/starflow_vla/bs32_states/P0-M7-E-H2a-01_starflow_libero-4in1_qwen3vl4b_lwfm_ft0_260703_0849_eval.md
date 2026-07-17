# P0-M7-E-H2a-01: StarFlow LIBERO 4-in-1 Qwen3VL-4B future_tokens=0（bs32）state-fix 评估追踪

> **实验代号**: E-H2a-01 / P0-M7  
> **状态**: ✅ 评估完成  
> **run_id**: `P0-M7-E-H2a-01_starflow_libero-4in1_qwen3vl4b_lwfm_ft0_260703_0849`  
> **启动时间**: 2026-07-16 11:46 CST  
> **当前更新**: 2026-07-17 12:08 CST  
> **tmux 会话**: `test`  
> **配置来源**: `configs/starflow_vla/ablations/future_tokens_0.yaml`  
> **评测输出**: `playground/starflow_eval_result`

---

## 评估概述

本 run 为 StarFlowVLA `num_target_vision_tokens=0`（ft0）baseline。2026-07-16 修复 `eval_libero.py` 未传入 robot state 的问题后，使用 `playground/starflow_eval_plan.sh` 对全部关键 ckpt 进行重新评测，结果固定输出到 `playground/starflow_eval_result/`。

### 评测参数

| 参数 | 值 |
|------|-----|
| `RUN_ID` | `P0-M7-E-H2a-01_starflow_libero-4in1_qwen3vl4b_lwfm_ft0_260703_0849` |
| `workers` | 20 |
| `num_trials` | 50 |
| `suites` | libero_goal, libero_10, libero_object, libero_spatial |
| `ckpt 列表` | steps_5000, steps_10000, steps_20000, steps_30000, steps_40000, steps_50000, steps_60000, steps_70000, steps_80000 |
| `policy server` | 单 server，固定端口 6700-6708 |
| `server 优化` | server-side batching（max_batch=8, timeout=30ms） |

---

## 评估进度

> 最后更新：2026-07-17 12:08 CST

| ckpt | 状态 | 总 eps | 总 successes | 总 SR |
|------|------|--------|--------------|-------|
| steps_5000 | ✅ 完成 | 2000 / 2000 | 1032 | 51.6% |
| steps_10000 | ✅ 完成 | 2000 / 2000 | 1437 | 71.9% |
| steps_20000 | ✅ 完成 | 2000 / 2000 | 1646 | 82.3% |
| steps_30000 | ✅ 完成 | 2000 / 2000 | 1678 | 83.9% |
| steps_40000 | ✅ 完成 | 2000 / 2000 | 1757 | 87.8% |
| steps_50000 | ✅ 完成 | 2000 / 2000 | 1822 | 91.1% |
| steps_60000 | ✅ 完成 | 2000 / 2000 | 1840 | 92.0% |
| steps_70000 | ✅ 完成 | 2000 / 2000 | 1854 | 92.7% |
| steps_80000 | ✅ 完成 | 2000 / 2000 | 1843 | 92.2% |

### 已完成的 ckpt 分 suite 结果

#### steps_5000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 282 | 56.4% |
| libero_10 | 500 | 93 | 18.6% |
| libero_object | 500 | 346 | 69.2% |
| libero_spatial | 500 | 311 | 62.2% |
| **合计** | **2000** | **1032** | **51.6%** |

#### steps_10000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 359 | 71.8% |
| libero_10 | 500 | 182 | 36.4% |
| libero_object | 500 | 462 | 92.4% |
| libero_spatial | 500 | 434 | 86.8% |
| **合计** | **2000** | **1437** | **71.9%** |

#### steps_20000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 439 | 87.8% |
| libero_10 | 500 | 284 | 56.8% |
| libero_object | 500 | 477 | 95.4% |
| libero_spatial | 500 | 446 | 89.2% |
| **合计** | **2000** | **1646** | **82.3%** |

#### steps_30000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 446 | 89.2% |
| libero_10 | 500 | 266 | 53.2% |
| libero_object | 500 | 491 | 98.2% |
| libero_spatial | 500 | 475 | 95.0% |
| **合计** | **2000** | **1678** | **83.9%** |

#### steps_40000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 413 | 82.6% |
| libero_10 | 500 | 380 | 76.0% |
| libero_object | 500 | 492 | 98.4% |
| libero_spatial | 500 | 472 | 94.4% |
| **合计** | **2000** | **1757** | **87.8%** |

#### steps_50000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 451 | 90.2% |
| libero_10 | 500 | 395 | 79.0% |
| libero_object | 500 | 488 | 97.6% |
| libero_spatial | 500 | 488 | 97.6% |
| **合计** | **2000** | **1822** | **91.1%** |

#### steps_60000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 455 | 91.0% |
| libero_10 | 500 | 409 | 81.8% |
| libero_object | 500 | 491 | 98.2% |
| libero_spatial | 500 | 485 | 97.0% |
| **合计** | **2000** | **1840** | **92.0%** |

#### steps_70000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 455 | 91.0% |
| libero_10 | 500 | 424 | 84.8% |
| libero_object | 500 | 496 | 99.2% |
| libero_spatial | 500 | 479 | 95.8% |
| **合计** | **2000** | **1854** | **92.7%** |

#### steps_80000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 462 | 92.4% |
| libero_10 | 500 | 401 | 80.2% |
| libero_object | 500 | 495 | 99.0% |
| libero_spatial | 500 | 485 | 97.0% |
| **合计** | **2000** | **1843** | **92.2%** |

---

## state 修复前后对比

本次重新评测前，`playground/eval_results/` 中 ft0 的 steps_5000/steps_10000 为未传 state 的结果。对比如下：

| ckpt | 无 state（老） | 有 state（新） | 差异 |
|------|---------------|---------------|------|
| steps_5000 | 23.6% | 51.6% | +28.0% |
| steps_10000 | 47.5% | 71.9% | +24.4% |

---

## 备注

- 修复 commit：`b7af6758 fix(LIBERO): pass robot state to model during evaluation`
- 评测过程中使用 workers=20。
- 每个 ckpt 固定端口：steps_5000→6700，steps_10000→6701，...，steps_80000→6708。

---

---

## 每个 ckpt 的平均完成时长

> 单位：秒。基于 `video_duration_sec` 统计。

| ckpt | libero_goal | libero_10 | libero_object | libero_spatial | 总体平均 |
|------|-------------|-----------|---------------|----------------|----------|
| steps_5000 | 19.2 | 48.1 | 19.4 | 14.9 | 25.4 |
| steps_10000 | 16.7 | 43.2 | 15.5 | 12.4 | 21.9 |
| steps_20000 | 13.7 | 37.2 | 14.2 | 11.9 | 19.2 |
| steps_30000 | 13.6 | 40.1 | 14.3 | 10.9 | 19.8 |
| steps_40000 | 14.5 | 32.1 | 14.2 | 11.4 | 18.1 |
| steps_50000 | 13.0 | 31.4 | 14.0 | 10.6 | 17.3 |
| steps_60000 | 12.9 | 31.0 | 14.0 | 10.8 | 17.2 |
| steps_70000 | 12.8 | 30.1 | 13.7 | 10.9 | 16.9 |
| steps_80000 | 12.8 | 31.2 | 13.8 | 10.7 | 17.1 |

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
| steps_5000 | 16% | 6% | 86% | 64% | 50% | 96% | 70% | 6% | 72% | 98% |
| steps_10000 | 100% | 38% | 14% | 88% | 92% | 96% | 44% | 70% | 92% | 84% |
| steps_20000 | 94% | 64% | 72% | 100% | 98% | 92% | 90% | 70% | 98% | 100% |
| steps_30000 | 94% | 74% | 84% | 100% | 98% | 92% | 94% | 58% | 98% | 100% |
| steps_40000 | 66% | 80% | 50% | 100% | 100% | 98% | 88% | 46% | 98% | 100% |
| steps_50000 | 92% | 80% | 92% | 94% | 100% | 98% | 88% | 60% | 98% | 100% |
| steps_60000 | 84% | 84% | 92% | 98% | 100% | 96% | 90% | 66% | 100% | 100% |
| steps_70000 | 94% | 84% | 92% | 98% | 96% | 96% | 98% | 52% | 100% | 100% |
| steps_80000 | 94% | 88% | 98% | 96% | 96% | 96% | 90% | 66% | 100% | 100% |

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
| steps_5000 | 0% | 0% | 22% | 20% | 50% | 12% | 0% | 0% | 6% | 76% |
| steps_10000 | 36% | 0% | 48% | 22% | 76% | 20% | 22% | 38% | 28% | 74% |
| steps_20000 | 92% | 10% | 80% | 64% | 88% | 82% | 22% | 68% | 6% | 56% |
| steps_30000 | 26% | 44% | 52% | 62% | 84% | 84% | 26% | 20% | 52% | 82% |
| steps_40000 | 70% | 26% | 94% | 86% | 96% | 66% | 78% | 74% | 74% | 96% |
| steps_50000 | 88% | 72% | 88% | 70% | 94% | 70% | 66% | 82% | 72% | 88% |
| steps_60000 | 78% | 74% | 94% | 84% | 98% | 82% | 78% | 80% | 78% | 72% |
| steps_70000 | 92% | 72% | 94% | 74% | 100% | 92% | 84% | 72% | 80% | 88% |
| steps_80000 | 86% | 60% | 92% | 76% | 98% | 88% | 84% | 66% | 76% | 76% |

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
| steps_5000 | 56% | 42% | 90% | 78% | 98% | 46% | 50% | 72% | 90% | 70% |
| steps_10000 | 74% | 86% | 100% | 98% | 96% | 96% | 100% | 98% | 88% | 88% |
| steps_20000 | 76% | 100% | 96% | 84% | 100% | 100% | 100% | 100% | 100% | 98% |
| steps_30000 | 98% | 98% | 100% | 100% | 100% | 98% | 98% | 94% | 100% | 96% |
| steps_40000 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 88% | 96% | 100% |
| steps_50000 | 100% | 96% | 96% | 96% | 100% | 100% | 100% | 90% | 100% | 98% |
| steps_60000 | 100% | 100% | 98% | 100% | 100% | 100% | 98% | 90% | 100% | 96% |
| steps_70000 | 100% | 96% | 100% | 100% | 100% | 98% | 100% | 98% | 100% | 100% |
| steps_80000 | 100% | 100% | 100% | 96% | 100% | 98% | 100% | 100% | 100% | 96% |

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
| steps_5000 | 94% | 96% | 14% | 72% | 60% | 44% | 76% | 56% | 82% | 28% |
| steps_10000 | 98% | 96% | 56% | 90% | 86% | 100% | 96% | 94% | 92% | 60% |
| steps_20000 | 96% | 92% | 58% | 100% | 94% | 94% | 94% | 76% | 92% | 96% |
| steps_30000 | 100% | 100% | 96% | 98% | 94% | 96% | 98% | 92% | 84% | 92% |
| steps_40000 | 100% | 100% | 98% | 96% | 86% | 94% | 96% | 94% | 88% | 92% |
| steps_50000 | 100% | 100% | 98% | 96% | 100% | 94% | 100% | 92% | 96% | 100% |
| steps_60000 | 100% | 100% | 98% | 98% | 100% | 98% | 96% | 86% | 94% | 100% |
| steps_70000 | 100% | 98% | 92% | 94% | 94% | 98% | 100% | 90% | 96% | 96% |
| steps_80000 | 98% | 100% | 98% | 96% | 96% | 98% | 96% | 90% | 98% | 100% |

---

## 每个 ckpt 的成功 episode 平均完成时长

> 单位：秒。仅统计 `success=true` 的 episode，基于 `video_duration_sec` 计算。

| ckpt | libero_goal | libero_10 | libero_object | libero_spatial | 总体平均 |
|------|-------------|-----------|---------------|----------------|----------|
| steps_5000 | 10.9 | 30.8 | 15.6 | 10.6 | 14.2 |
| steps_10000 | 11.5 | 27.8 | 14.5 | 10.9 | 14.3 |
| steps_20000 | 11.4 | 25.9 | 13.5 | 10.6 | 14.3 |
| steps_30000 | 11.6 | 29.7 | 14.1 | 10.3 | 14.8 |
| steps_40000 | 11.3 | 25.8 | 14.0 | 10.8 | 15.1 |
| steps_50000 | 11.1 | 26.0 | 13.8 | 10.4 | 14.9 |
| steps_60000 | 11.2 | 26.4 | 13.7 | 10.5 | 15.1 |
| steps_70000 | 11.1 | 26.2 | 13.6 | 10.4 | 15.0 |
| steps_80000 | 11.3 | 26.0 | 13.7 | 10.4 | 14.9 |

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
| steps_5000 | 16% | 6% | 86% | 64% | 50% | 96% | 70% | 6% | 72% | 98% |
| steps_10000 | 100% | 38% | 14% | 88% | 92% | 96% | 44% | 70% | 92% | 84% |
| steps_20000 | 94% | 64% | 72% | 100% | 98% | 92% | 90% | 70% | 98% | 100% |
| steps_30000 | 94% | 74% | 84% | 100% | 98% | 92% | 94% | 58% | 98% | 100% |
| steps_40000 | 66% | 80% | 50% | 100% | 100% | 98% | 88% | 46% | 98% | 100% |
| steps_50000 | 92% | 80% | 92% | 94% | 100% | 98% | 88% | 60% | 98% | 100% |
| steps_60000 | 84% | 84% | 92% | 98% | 100% | 96% | 90% | 66% | 100% | 100% |
| steps_70000 | 94% | 84% | 92% | 98% | 96% | 96% | 98% | 52% | 100% | 100% |
| steps_80000 | 94% | 88% | 98% | 96% | 96% | 96% | 90% | 66% | 100% | 100% |

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
| steps_5000 | 0% | 0% | 22% | 20% | 50% | 12% | 0% | 0% | 6% | 76% |
| steps_10000 | 36% | 0% | 48% | 22% | 76% | 20% | 22% | 38% | 28% | 74% |
| steps_20000 | 92% | 10% | 80% | 64% | 88% | 82% | 22% | 68% | 6% | 56% |
| steps_30000 | 26% | 44% | 52% | 62% | 84% | 84% | 26% | 20% | 52% | 82% |
| steps_40000 | 70% | 26% | 94% | 86% | 96% | 66% | 78% | 74% | 74% | 96% |
| steps_50000 | 88% | 72% | 88% | 70% | 94% | 70% | 66% | 82% | 72% | 88% |
| steps_60000 | 78% | 74% | 94% | 84% | 98% | 82% | 78% | 80% | 78% | 72% |
| steps_70000 | 92% | 72% | 94% | 74% | 100% | 92% | 84% | 72% | 80% | 88% |
| steps_80000 | 86% | 60% | 92% | 76% | 98% | 88% | 84% | 66% | 76% | 76% |

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
| steps_5000 | 56% | 42% | 90% | 78% | 98% | 46% | 50% | 72% | 90% | 70% |
| steps_10000 | 74% | 86% | 100% | 98% | 96% | 96% | 100% | 98% | 88% | 88% |
| steps_20000 | 76% | 100% | 96% | 84% | 100% | 100% | 100% | 100% | 100% | 98% |
| steps_30000 | 98% | 98% | 100% | 100% | 100% | 98% | 98% | 94% | 100% | 96% |
| steps_40000 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 88% | 96% | 100% |
| steps_50000 | 100% | 96% | 96% | 96% | 100% | 100% | 100% | 90% | 100% | 98% |
| steps_60000 | 100% | 100% | 98% | 100% | 100% | 100% | 98% | 90% | 100% | 96% |
| steps_70000 | 100% | 96% | 100% | 100% | 100% | 98% | 100% | 98% | 100% | 100% |
| steps_80000 | 100% | 100% | 100% | 96% | 100% | 98% | 100% | 100% | 100% | 96% |

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
| steps_5000 | 94% | 96% | 14% | 72% | 60% | 44% | 76% | 56% | 82% | 28% |
| steps_10000 | 98% | 96% | 56% | 90% | 86% | 100% | 96% | 94% | 92% | 60% |
| steps_20000 | 96% | 92% | 58% | 100% | 94% | 94% | 94% | 76% | 92% | 96% |
| steps_30000 | 100% | 100% | 96% | 98% | 94% | 96% | 98% | 92% | 84% | 92% |
| steps_40000 | 100% | 100% | 98% | 96% | 86% | 94% | 96% | 94% | 88% | 92% |
| steps_50000 | 100% | 100% | 98% | 96% | 100% | 94% | 100% | 92% | 96% | 100% |
| steps_60000 | 100% | 100% | 98% | 98% | 100% | 98% | 96% | 86% | 94% | 100% |
| steps_70000 | 100% | 98% | 92% | 94% | 94% | 98% | 100% | 90% | 96% | 96% |
| steps_80000 | 98% | 100% | 98% | 96% | 96% | 98% | 96% | 90% | 98% | 100% |

---

## 每个 ckpt 成功 episode 的平均完成部署步数 / 时长

> `steps_executed` 为成功 episode 中实际执行的 env step 数；按 LIBERO 10Hz 控制频率换算为秒。仅统计 `success=true` 的 episode。

| ckpt | goal_steps | goal_sec | 10_steps | 10_sec | object_steps | object_sec | spatial_steps | spatial_sec | overall_steps | overall_sec |
|------|------------|----------|----------|--------|--------------|------------|---------------|-------------|---------------|-------------|
| steps_5000 | 108.4 | 10.8 | 307.1 | 30.7 | 155.0 | 15.5 | 105.2 | 10.5 | 141.0 | 14.1 |
| steps_10000 | 114.0 | 11.4 | 276.6 | 27.7 | 143.7 | 14.4 | 108.4 | 10.8 | 142.5 | 14.2 |
| steps_20000 | 113.3 | 11.3 | 257.8 | 25.8 | 134.2 | 13.4 | 105.3 | 10.5 | 142.1 | 14.2 |
| steps_30000 | 115.4 | 11.5 | 295.8 | 29.6 | 140.0 | 14.0 | 102.3 | 10.2 | 147.5 | 14.7 |
| steps_40000 | 111.6 | 11.2 | 257.5 | 25.7 | 138.7 | 13.9 | 107.1 | 10.7 | 149.5 | 15.0 |
| steps_50000 | 110.3 | 11.0 | 258.7 | 25.9 | 137.0 | 13.7 | 102.6 | 10.3 | 147.5 | 14.8 |
| steps_60000 | 111.2 | 11.1 | 262.7 | 26.3 | 136.3 | 13.6 | 103.6 | 10.4 | 149.6 | 15.0 |
| steps_70000 | 110.4 | 11.0 | 261.0 | 26.1 | 135.2 | 13.5 | 102.7 | 10.3 | 149.5 | 14.9 |
| steps_80000 | 112.5 | 11.2 | 259.3 | 25.9 | 135.7 | 13.6 | 102.5 | 10.3 | 148.0 | 14.8 |

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
| steps_5000 | 16% | 6% | 86% | 64% | 50% | 96% | 70% | 6% | 72% | 98% |
| steps_10000 | 100% | 38% | 14% | 88% | 92% | 96% | 44% | 70% | 92% | 84% |
| steps_20000 | 94% | 64% | 72% | 100% | 98% | 92% | 90% | 70% | 98% | 100% |
| steps_30000 | 94% | 74% | 84% | 100% | 98% | 92% | 94% | 58% | 98% | 100% |
| steps_40000 | 66% | 80% | 50% | 100% | 100% | 98% | 88% | 46% | 98% | 100% |
| steps_50000 | 92% | 80% | 92% | 94% | 100% | 98% | 88% | 60% | 98% | 100% |
| steps_60000 | 84% | 84% | 92% | 98% | 100% | 96% | 90% | 66% | 100% | 100% |
| steps_70000 | 94% | 84% | 92% | 98% | 96% | 96% | 98% | 52% | 100% | 100% |
| steps_80000 | 94% | 88% | 98% | 96% | 96% | 96% | 90% | 66% | 100% | 100% |

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
| steps_5000 | 0% | 0% | 22% | 20% | 50% | 12% | 0% | 0% | 6% | 76% |
| steps_10000 | 36% | 0% | 48% | 22% | 76% | 20% | 22% | 38% | 28% | 74% |
| steps_20000 | 92% | 10% | 80% | 64% | 88% | 82% | 22% | 68% | 6% | 56% |
| steps_30000 | 26% | 44% | 52% | 62% | 84% | 84% | 26% | 20% | 52% | 82% |
| steps_40000 | 70% | 26% | 94% | 86% | 96% | 66% | 78% | 74% | 74% | 96% |
| steps_50000 | 88% | 72% | 88% | 70% | 94% | 70% | 66% | 82% | 72% | 88% |
| steps_60000 | 78% | 74% | 94% | 84% | 98% | 82% | 78% | 80% | 78% | 72% |
| steps_70000 | 92% | 72% | 94% | 74% | 100% | 92% | 84% | 72% | 80% | 88% |
| steps_80000 | 86% | 60% | 92% | 76% | 98% | 88% | 84% | 66% | 76% | 76% |

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
| steps_5000 | 56% | 42% | 90% | 78% | 98% | 46% | 50% | 72% | 90% | 70% |
| steps_10000 | 74% | 86% | 100% | 98% | 96% | 96% | 100% | 98% | 88% | 88% |
| steps_20000 | 76% | 100% | 96% | 84% | 100% | 100% | 100% | 100% | 100% | 98% |
| steps_30000 | 98% | 98% | 100% | 100% | 100% | 98% | 98% | 94% | 100% | 96% |
| steps_40000 | 100% | 100% | 100% | 100% | 100% | 100% | 100% | 88% | 96% | 100% |
| steps_50000 | 100% | 96% | 96% | 96% | 100% | 100% | 100% | 90% | 100% | 98% |
| steps_60000 | 100% | 100% | 98% | 100% | 100% | 100% | 98% | 90% | 100% | 96% |
| steps_70000 | 100% | 96% | 100% | 100% | 100% | 98% | 100% | 98% | 100% | 100% |
| steps_80000 | 100% | 100% | 100% | 96% | 100% | 98% | 100% | 100% | 100% | 96% |

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
| steps_5000 | 94% | 96% | 14% | 72% | 60% | 44% | 76% | 56% | 82% | 28% |
| steps_10000 | 98% | 96% | 56% | 90% | 86% | 100% | 96% | 94% | 92% | 60% |
| steps_20000 | 96% | 92% | 58% | 100% | 94% | 94% | 94% | 76% | 92% | 96% |
| steps_30000 | 100% | 100% | 96% | 98% | 94% | 96% | 98% | 92% | 84% | 92% |
| steps_40000 | 100% | 100% | 98% | 96% | 86% | 94% | 96% | 94% | 88% | 92% |
| steps_50000 | 100% | 100% | 98% | 96% | 100% | 94% | 100% | 92% | 96% | 100% |
| steps_60000 | 100% | 100% | 98% | 98% | 100% | 98% | 96% | 86% | 94% | 100% |
| steps_70000 | 100% | 98% | 92% | 94% | 94% | 98% | 100% | 90% | 96% | 96% |
| steps_80000 | 98% | 100% | 98% | 96% | 96% | 98% | 96% | 90% | 98% | 100% |
