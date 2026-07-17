# P1-M1-E-H2b-02: StarFlow LIBERO 4-in-1 Qwen3VL-4B continuous_head（bs32）state-fix 评估追踪

> **实验代号**: E-H2b-02 / P1-M1  
> **状态**: ✅ 评估完成  
> **run_id**: `P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260702_2021`  
> **启动时间**: 2026-07-16 13:02 CST  
> **当前更新**: 2026-07-17 12:05 CST  
> **tmux 会话**: `test`  
> **配置来源**: `configs/starflow_vla/state/continuous_head.yaml`  
> **评测输出**: `playground/starflow_eval_result`

---

## 评估概述

本 run 为 StarFlowVLA `state_mode=continuous_head` 变体，基于 ft32 继续进行 LIBERO 4-in-1 训练。2026-07-16 修复 `eval_libero.py` 未传入 robot state 的问题后，使用 `playground/starflow_eval_plan.sh` 对全部关键 ckpt 进行重新评测，结果固定输出到 `playground/starflow_eval_result/`。

### 评测参数

| 参数 | 值 |
|------|-----|
| `RUN_ID` | `P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260702_2021` |
| `workers` | 20 |
| `num_trials` | 50 |
| `suites` | libero_goal, libero_10, libero_object, libero_spatial |
| `ckpt 列表` | steps_5000, steps_10000, steps_20000, steps_30000, steps_40000, steps_50000, steps_60000, steps_70000, steps_80000 |
| `policy server` | 单 server，固定端口 6740-6748 |
| `server 优化` | server-side batching（max_batch=8, timeout=30ms） |

---

## 评估进度

> 最后更新：2026-07-17 12:05 CST

| ckpt | 状态 | 总 eps | 总 successes | 总 SR |
|------|------|--------|--------------|-------|
| steps_5000 | ✅ 完成 | 2000 / 2000 | 1083 | 54.1% |
| steps_10000 | ✅ 完成 | 2000 / 2000 | 1427 | 71.4% |
| steps_20000 | ✅ 完成 | 2000 / 2000 | 1673 | 83.7% |
| steps_30000 | ✅ 完成 | 2000 / 2000 | 1659 | 83.0% |
| steps_40000 | ✅ 完成 | 2000 / 2000 | 1805 | 90.2% |
| steps_50000 | ✅ 完成 | 2000 / 2000 | 1800 | 90.0% |
| steps_60000 | ✅ 完成 | 2000 / 2000 | 1830 | 91.5% |
| steps_70000 | ✅ 完成 | 2000 / 2000 | 1855 | 92.8% |
| steps_80000 | ✅ 完成 | 2000 / 2000 | 1876 | 93.8% |

### 已完成的 ckpt 分 suite 结果

#### steps_5000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 321 | 64.2% |
| libero_10 | 500 | 94 | 18.8% |
| libero_object | 500 | 320 | 64.0% |
| libero_spatial | 500 | 348 | 69.6% |
| **合计** | **2000** | **1083** | **54.1%** |

#### steps_10000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 353 | 70.6% |
| libero_10 | 500 | 162 | 32.4% |
| libero_object | 500 | 446 | 89.2% |
| libero_spatial | 500 | 466 | 93.2% |
| **合计** | **2000** | **1427** | **71.4%** |

#### steps_20000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 427 | 85.4% |
| libero_10 | 500 | 301 | 60.2% |
| libero_object | 500 | 488 | 97.6% |
| libero_spatial | 500 | 457 | 91.4% |
| **合计** | **2000** | **1673** | **83.7%** |

#### steps_30000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 451 | 90.2% |
| libero_10 | 500 | 265 | 53.0% |
| libero_object | 500 | 476 | 95.2% |
| libero_spatial | 500 | 467 | 93.4% |
| **合计** | **2000** | **1659** | **83.0%** |

#### steps_40000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 475 | 95.0% |
| libero_10 | 500 | 359 | 71.8% |
| libero_object | 500 | 487 | 97.4% |
| libero_spatial | 500 | 484 | 96.8% |
| **合计** | **2000** | **1805** | **90.2%** |

#### steps_50000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 466 | 93.2% |
| libero_10 | 500 | 372 | 74.4% |
| libero_object | 500 | 480 | 96.0% |
| libero_spatial | 500 | 482 | 96.4% |
| **合计** | **2000** | **1800** | **90.0%** |

#### steps_60000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 473 | 94.6% |
| libero_10 | 500 | 390 | 78.0% |
| libero_object | 500 | 490 | 98.0% |
| libero_spatial | 500 | 477 | 95.4% |
| **合计** | **2000** | **1830** | **91.5%** |

#### steps_70000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 477 | 95.4% |
| libero_10 | 500 | 399 | 79.8% |
| libero_object | 500 | 490 | 98.0% |
| libero_spatial | 500 | 489 | 97.8% |
| **合计** | **2000** | **1855** | **92.8%** |

#### steps_80000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 472 | 94.4% |
| libero_10 | 500 | 427 | 85.4% |
| libero_object | 500 | 491 | 98.2% |
| libero_spatial | 500 | 486 | 97.2% |
| **合计** | **2000** | **1876** | **93.8%** |

---

## state 修复前后对比

本次重新评测前，`playground/eval_results/` 中 cont.ft32 的 steps_5000/steps_10000 为旧版结果。对比如下：

| ckpt | 旧版（eval_results） | 新版（starflow_eval_result） | 差异 |
|------|---------------------|----------------------------|------|
| steps_5000 | 54.1% | 54.1% | 0.0% |
| steps_10000 | 70.2% | 71.4% | +1.2% |
| steps_20000 | 83.9% | 83.7% | -0.2% |
| steps_30000 | 85.5% | 83.0% | -2.5% |
| steps_40000 | 90.2% | 90.2% | 0.0% |
| steps_50000 | 88.1% | 90.0% | +1.9% |
| steps_60000 | 92.7% | 91.5% | -1.2% |
| steps_70000 | 92.4% | 92.8% | +0.4% |
| steps_80000 | 93.3% | 93.8% | +0.5% |

---

## 备注

- 修复 commit：`b7af6758 fix(LIBERO): pass robot state to model during evaluation`
- 修复内容：系统安装 `libegl1` 解决 headless 渲染 EGL 初始化失败；`playground/eval_pool_manager.py` 增加单 task 最大重试次数，避免崩溃任务无限循环。
- 评测过程中使用 workers=20，speed 约 22-32 eps/min。
- 每个 ckpt 固定端口：steps_5000→6740，steps_10000→6741，...，steps_80000→6748。
- 日志文件：`starflow_eval_result/logs/starflow_eval_plan_YYYYMMDD_HHMMSS.log`。
- **2026-07-17 09:30 更新**：cont.ft32 全部 9 个 ckpt 评估已完成。最终 `steps_80000` SR 达到 **93.8%**。

---

---

## 每个 ckpt 的平均完成时长

> 单位：秒。基于 `video_duration_sec` 统计。

| ckpt | libero_goal | libero_10 | libero_object | libero_spatial | 总体平均 |
|------|-------------|-----------|---------------|----------------|----------|
| steps_5000 | 17.6 | 48.0 | 19.9 | 14.2 | 24.9 |
| steps_10000 | 17.1 | 44.3 | 15.4 | 12.3 | 22.3 |
| steps_20000 | 14.0 | 36.6 | 14.1 | 11.5 | 19.1 |
| steps_30000 | 13.3 | 38.8 | 15.0 | 11.4 | 19.6 |
| steps_40000 | 12.1 | 33.3 | 14.6 | 10.8 | 17.7 |
| steps_50000 | 12.5 | 32.9 | 14.8 | 10.9 | 17.8 |
| steps_60000 | 12.1 | 31.9 | 14.0 | 11.1 | 17.3 |
| steps_70000 | 12.0 | 30.8 | 13.8 | 10.7 | 16.8 |
| steps_80000 | 12.0 | 29.7 | 13.8 | 10.7 | 16.6 |

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
| steps_5000 | 38% | 2% | 94% | 92% | 94% | 84% | 66% | 0% | 72% | 100% |
| steps_10000 | 60% | 56% | 18% | 100% | 96% | 86% | 58% | 54% | 98% | 80% |
| steps_20000 | 100% | 90% | 18% | 100% | 98% | 96% | 96% | 62% | 94% | 100% |
| steps_30000 | 94% | 76% | 90% | 100% | 98% | 96% | 88% | 66% | 94% | 100% |
| steps_40000 | 98% | 84% | 92% | 100% | 96% | 98% | 98% | 86% | 98% | 100% |
| steps_50000 | 92% | 94% | 94% | 100% | 98% | 98% | 92% | 66% | 98% | 100% |
| steps_60000 | 92% | 84% | 96% | 100% | 100% | 96% | 96% | 82% | 100% | 100% |
| steps_70000 | 98% | 90% | 92% | 98% | 98% | 96% | 96% | 86% | 100% | 100% |
| steps_80000 | 98% | 86% | 92% | 96% | 100% | 98% | 98% | 76% | 100% | 100% |

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
| steps_5000 | 12% | 0% | 14% | 0% | 38% | 8% | 0% | 0% | 22% | 94% |
| steps_10000 | 72% | 0% | 26% | 26% | 28% | 8% | 60% | 58% | 12% | 34% |
| steps_20000 | 80% | 32% | 78% | 70% | 82% | 74% | 46% | 76% | 0% | 64% |
| steps_30000 | 78% | 2% | 76% | 66% | 92% | 68% | 28% | 44% | 0% | 76% |
| steps_40000 | 98% | 12% | 94% | 76% | 96% | 72% | 74% | 66% | 34% | 96% |
| steps_50000 | 94% | 42% | 96% | 78% | 100% | 60% | 50% | 62% | 64% | 98% |
| steps_60000 | 92% | 44% | 92% | 72% | 88% | 82% | 72% | 70% | 74% | 94% |
| steps_70000 | 94% | 42% | 96% | 62% | 96% | 74% | 90% | 74% | 80% | 90% |
| steps_80000 | 98% | 74% | 100% | 70% | 100% | 82% | 82% | 74% | 88% | 86% |

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
| steps_5000 | 26% | 70% | 90% | 74% | 80% | 50% | 68% | 40% | 84% | 58% |
| steps_10000 | 70% | 96% | 100% | 78% | 66% | 98% | 98% | 98% | 98% | 90% |
| steps_20000 | 96% | 90% | 100% | 98% | 98% | 100% | 98% | 98% | 100% | 98% |
| steps_30000 | 94% | 82% | 100% | 100% | 100% | 100% | 98% | 92% | 94% | 92% |
| steps_40000 | 98% | 100% | 98% | 92% | 100% | 100% | 98% | 94% | 100% | 94% |
| steps_50000 | 100% | 94% | 100% | 94% | 96% | 100% | 100% | 86% | 100% | 90% |
| steps_60000 | 98% | 90% | 100% | 100% | 100% | 100% | 100% | 98% | 98% | 96% |
| steps_70000 | 98% | 96% | 100% | 100% | 100% | 100% | 100% | 96% | 98% | 92% |
| steps_80000 | 100% | 98% | 100% | 100% | 100% | 100% | 100% | 96% | 98% | 90% |

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
| steps_5000 | 100% | 90% | 48% | 86% | 92% | 12% | 94% | 64% | 88% | 22% |
| steps_10000 | 90% | 100% | 88% | 100% | 94% | 98% | 96% | 96% | 88% | 82% |
| steps_20000 | 100% | 88% | 74% | 100% | 96% | 100% | 100% | 72% | 88% | 96% |
| steps_30000 | 98% | 96% | 94% | 96% | 84% | 96% | 96% | 92% | 98% | 84% |
| steps_40000 | 98% | 100% | 94% | 94% | 96% | 100% | 98% | 98% | 92% | 98% |
| steps_50000 | 98% | 100% | 94% | 96% | 92% | 96% | 98% | 94% | 96% | 100% |
| steps_60000 | 98% | 100% | 90% | 96% | 96% | 98% | 96% | 90% | 90% | 100% |
| steps_70000 | 100% | 96% | 100% | 98% | 100% | 100% | 100% | 96% | 90% | 98% |
| steps_80000 | 100% | 100% | 88% | 100% | 98% | 98% | 94% | 96% | 98% | 100% |

---

## 每个 ckpt 的成功 episode 平均完成时长

> 单位：秒。仅统计 `success=true` 的 episode，基于 `video_duration_sec` 计算。

| ckpt | libero_goal | libero_10 | libero_object | libero_spatial | 总体平均 |
|------|-------------|-----------|---------------|----------------|----------|
| steps_5000 | 10.8 | 30.9 | 15.4 | 10.8 | 13.9 |
| steps_10000 | 11.7 | 28.4 | 13.8 | 11.5 | 14.2 |
| steps_20000 | 11.2 | 26.4 | 13.8 | 10.6 | 14.5 |
| steps_30000 | 11.5 | 27.2 | 14.3 | 10.7 | 14.6 |
| steps_40000 | 11.1 | 25.9 | 14.2 | 10.4 | 14.7 |
| steps_50000 | 11.3 | 26.3 | 14.3 | 10.5 | 15.0 |
| steps_60000 | 11.1 | 26.2 | 13.7 | 10.5 | 14.9 |
| steps_70000 | 11.1 | 25.4 | 13.5 | 10.4 | 14.7 |
| steps_80000 | 11.0 | 25.9 | 13.6 | 10.4 | 14.9 |

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
| steps_5000 | 38% | 2% | 94% | 92% | 94% | 84% | 66% | 0% | 72% | 100% |
| steps_10000 | 60% | 56% | 18% | 100% | 96% | 86% | 58% | 54% | 98% | 80% |
| steps_20000 | 100% | 90% | 18% | 100% | 98% | 96% | 96% | 62% | 94% | 100% |
| steps_30000 | 94% | 76% | 90% | 100% | 98% | 96% | 88% | 66% | 94% | 100% |
| steps_40000 | 98% | 84% | 92% | 100% | 96% | 98% | 98% | 86% | 98% | 100% |
| steps_50000 | 92% | 94% | 94% | 100% | 98% | 98% | 92% | 66% | 98% | 100% |
| steps_60000 | 92% | 84% | 96% | 100% | 100% | 96% | 96% | 82% | 100% | 100% |
| steps_70000 | 98% | 90% | 92% | 98% | 98% | 96% | 96% | 86% | 100% | 100% |
| steps_80000 | 98% | 86% | 92% | 96% | 100% | 98% | 98% | 76% | 100% | 100% |

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
| steps_5000 | 12% | 0% | 14% | 0% | 38% | 8% | 0% | 0% | 22% | 94% |
| steps_10000 | 72% | 0% | 26% | 26% | 28% | 8% | 60% | 58% | 12% | 34% |
| steps_20000 | 80% | 32% | 78% | 70% | 82% | 74% | 46% | 76% | 0% | 64% |
| steps_30000 | 78% | 2% | 76% | 66% | 92% | 68% | 28% | 44% | 0% | 76% |
| steps_40000 | 98% | 12% | 94% | 76% | 96% | 72% | 74% | 66% | 34% | 96% |
| steps_50000 | 94% | 42% | 96% | 78% | 100% | 60% | 50% | 62% | 64% | 98% |
| steps_60000 | 92% | 44% | 92% | 72% | 88% | 82% | 72% | 70% | 74% | 94% |
| steps_70000 | 94% | 42% | 96% | 62% | 96% | 74% | 90% | 74% | 80% | 90% |
| steps_80000 | 98% | 74% | 100% | 70% | 100% | 82% | 82% | 74% | 88% | 86% |

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
| steps_5000 | 26% | 70% | 90% | 74% | 80% | 50% | 68% | 40% | 84% | 58% |
| steps_10000 | 70% | 96% | 100% | 78% | 66% | 98% | 98% | 98% | 98% | 90% |
| steps_20000 | 96% | 90% | 100% | 98% | 98% | 100% | 98% | 98% | 100% | 98% |
| steps_30000 | 94% | 82% | 100% | 100% | 100% | 100% | 98% | 92% | 94% | 92% |
| steps_40000 | 98% | 100% | 98% | 92% | 100% | 100% | 98% | 94% | 100% | 94% |
| steps_50000 | 100% | 94% | 100% | 94% | 96% | 100% | 100% | 86% | 100% | 90% |
| steps_60000 | 98% | 90% | 100% | 100% | 100% | 100% | 100% | 98% | 98% | 96% |
| steps_70000 | 98% | 96% | 100% | 100% | 100% | 100% | 100% | 96% | 98% | 92% |
| steps_80000 | 100% | 98% | 100% | 100% | 100% | 100% | 100% | 96% | 98% | 90% |

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
| steps_5000 | 100% | 90% | 48% | 86% | 92% | 12% | 94% | 64% | 88% | 22% |
| steps_10000 | 90% | 100% | 88% | 100% | 94% | 98% | 96% | 96% | 88% | 82% |
| steps_20000 | 100% | 88% | 74% | 100% | 96% | 100% | 100% | 72% | 88% | 96% |
| steps_30000 | 98% | 96% | 94% | 96% | 84% | 96% | 96% | 92% | 98% | 84% |
| steps_40000 | 98% | 100% | 94% | 94% | 96% | 100% | 98% | 98% | 92% | 98% |
| steps_50000 | 98% | 100% | 94% | 96% | 92% | 96% | 98% | 94% | 96% | 100% |
| steps_60000 | 98% | 100% | 90% | 96% | 96% | 98% | 96% | 90% | 90% | 100% |
| steps_70000 | 100% | 96% | 100% | 98% | 100% | 100% | 100% | 96% | 90% | 98% |
| steps_80000 | 100% | 100% | 88% | 100% | 98% | 98% | 94% | 96% | 98% | 100% |
