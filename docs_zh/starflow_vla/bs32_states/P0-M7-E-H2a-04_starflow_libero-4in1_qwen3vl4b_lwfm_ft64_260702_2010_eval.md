# P0-M7-E-H2a-04: StarFlow LIBERO 4-in-1 Qwen3VL-4B future_tokens=64（bs32）state-fix 评估追踪

> **实验代号**: E-H2a-04 / P0-M7
> **状态**: ✅ 评估全部完成
> **run_id**: `P0-M7-E-H2a-04_starflow_libero-4in1_qwen3vl4b_lwfm_ft64_260702_2010`
> **启动时间**: 2026-07-16 11:49 CST
> **当前更新**: 2026-07-17 10:41 CST
> **完成时间**: 2026-07-17 03:47 CST
> **tmux 会话**: `test`
> **配置来源**: `configs/starflow_vla/ablations/future_tokens_64.yaml`
> **评测输出**: `playground/starflow_eval_result`

---

## 评估概述

本 run 为 StarFlowVLA `num_target_vision_tokens=64`（ft64）。2026-07-16 修复 `eval_libero.py` 未传入 robot state 的问题后，使用 `playground/starflow_eval_plan.sh` 对全部关键 ckpt 进行重新评测，结果固定输出到 `playground/starflow_eval_result/`。

### 评测参数

| 参数 | 值 |
|------|-----|
| `RUN_ID` | `P0-M7-E-H2a-04_starflow_libero-4in1_qwen3vl4b_lwfm_ft64_260702_2010` |
| `workers` | 20 |
| `num_trials` | 50 |
| `suites` | libero_goal, libero_10, libero_object, libero_spatial |
| `ckpt 列表` | steps_5000, steps_10000, steps_20000, steps_30000, steps_40000, steps_50000, steps_60000, steps_70000, steps_80000 |
| `policy server` | 单 server，固定端口 6730-6738 |
| `server 优化` | server-side batching（max_batch=8, timeout=30ms） |

---

## 评估进度

> 最后更新：2026-07-17 10:41 CST

| ckpt | 状态 | 总 eps | 总 successes | 总 SR |
|------|------|--------|--------------|-------|
| steps_5000 | ✅ 完成 | 2000 / 2000 | 850 | 42.5% |
| steps_10000 | ✅ 完成 | 2000 / 2000 | 1403 | 70.2% |
| steps_20000 | ✅ 完成 | 2000 / 2000 | 1650 | 82.5% |
| steps_30000 | ✅ 完成 | 2000 / 2000 | 1646 | 82.3% |
| steps_40000 | ✅ 完成 | 2000 / 2000 | 1745 | 87.2% |
| steps_50000 | ✅ 完成 | 2000 / 2000 | 1822 | 91.1% |
| steps_60000 | ✅ 完成 | 2000 / 2000 | 1849 | 92.5% |
| steps_70000 | ✅ 完成 | 2000 / 2000 | 1879 | 94.0% |
| steps_80000 | ✅ 完成 | 2000 / 2000 | 1885 | 94.2% |

### 已完成的 ckpt 分 suite 结果

#### steps_5000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 325 | 65.0% |
| libero_10 | 500 | 60 | 12.0% |
| libero_object | 500 | 186 | 37.2% |
| libero_spatial | 500 | 279 | 55.8% |
| **合计** | **2000** | **850** | **42.5%** |

#### steps_10000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 360 | 72.0% |
| libero_10 | 500 | 190 | 38.0% |
| libero_object | 500 | 438 | 87.6% |
| libero_spatial | 500 | 415 | 83.0% |
| **合计** | **2000** | **1403** | **70.2%** |

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
| libero_goal | 500 | 414 | 82.8% |
| libero_10 | 500 | 280 | 56.0% |
| libero_object | 500 | 470 | 94.0% |
| libero_spatial | 500 | 482 | 96.4% |
| **合计** | **2000** | **1646** | **82.3%** |

#### steps_40000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 464 | 92.8% |
| libero_10 | 500 | 318 | 63.6% |
| libero_object | 500 | 478 | 95.6% |
| libero_spatial | 500 | 485 | 97.0% |
| **合计** | **2000** | **1745** | **87.2%** |

#### steps_50000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 481 | 96.2% |
| libero_10 | 500 | 393 | 78.6% |
| libero_object | 500 | 473 | 94.6% |
| libero_spatial | 500 | 475 | 95.0% |
| **合计** | **2000** | **1822** | **91.1%** |

#### steps_60000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 467 | 93.4% |
| libero_10 | 500 | 402 | 80.4% |
| libero_object | 500 | 491 | 98.2% |
| libero_spatial | 500 | 489 | 97.8% |
| **合计** | **2000** | **1849** | **92.5%** |

#### steps_70000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 476 | 95.2% |
| libero_10 | 500 | 432 | 86.4% |
| libero_object | 500 | 487 | 97.4% |
| libero_spatial | 500 | 484 | 96.8% |
| **合计** | **2000** | **1879** | **94.0%** |

#### steps_80000

| suite | eps | successes | SR |
|-------|-----|-----------|-----|
| libero_goal | 500 | 468 | 93.6% |
| libero_10 | 500 | 442 | 88.4% |
| libero_object | 500 | 489 | 97.8% |
| libero_spatial | 500 | 486 | 97.2% |
| **合计** | **2000** | **1885** | **94.2%** |

---

## state 修复前后对比

本次重新评测前，`playground/eval_results/` 中 ft64 的 steps_5000/steps_10000 为未传 state 的结果。对比如下：

| ckpt | 无 state（老） | 有 state（新） | 差异 |
|------|---------------|---------------|------|
| steps_5000 | 17.0% | 42.5% | +25.5% |
| steps_10000 | 31.6% | 70.2% | +38.5% |

### steps_5000 分 suite 对比

| suite | 无 state（老） | 有 state（新） | 差异 |
|-------|---------------|---------------|------|
| libero_goal | 44.8% | 65.0% | +20.2% |
| libero_10 | 2.6% | 12.0% | +9.4% |
| libero_object | 7.6% | 37.2% | +29.6% |
| libero_spatial | 12.8% | 55.8% | +43.0% |
| **合计** | **17.0%** | **42.5%** | **+25.5%** |

### steps_10000 分 suite 对比

| suite | 无 state（老） | 有 state（新） | 差异 |
|-------|---------------|---------------|------|
| libero_goal | 51.8% | 72.0% | +20.2% |
| libero_10 | 6.0% | 38.0% | +32.0% |
| libero_object | 29.4% | 87.6% | +58.2% |
| libero_spatial | 39.4% | 83.0% | +43.6% |
| **合计** | **31.6%** | **70.2%** | **+38.5%** |

---

## 备注

- 修复 commit：`b7af6758 fix(LIBERO): pass robot state to model during evaluation`
- 评测过程中使用 workers=20，speed 约 15-25 eps/min。
- 每个 ckpt 固定端口：steps_5000→6730，steps_10000→6731，...，steps_80000→6738。
- **异常记录**：`steps_20000` 于 2026-07-16 15:18 启动 policy server 时超时（日志显示 `❌ Server 启动超时`），评估脚本退出到 shell。
- **根因**：`playground/starflow_eval_pool.sh` 中 server 就绪等待时间仅 180 秒，ft64 的 10GB+ ckpt 加载时间波动大（steps_5000 实测约 100 秒，已接近阈值），steps_20000 加载时超过 180 秒导致超时。
- **修复**：已将 `starflow_eval_pool.sh` 中超时时间从 180 秒调整为 **600 秒**（`seq 1 90` → `seq 1 300`），并补充注释说明 ft64 大 ckpt 需要更长的加载时间。
- **00:47 更新**：`steps_60000` 已完成（SR 92.5%），`steps_70000` 已启动（端口 6737）。`steps_70000` 当前完成 714/2000（35.7%），速度约 25 trials/min，剩余 1286 trials 约需 51 分钟；后续 `steps_80000` 按约 80 分钟估算，预计全部完成时间为 **2026-07-17 02:55-03:10** 左右。
- **03:47 更新**：**ft64 全部 9 个 ckpt 评估已完成**。最终 `steps_80000` SR 达到 **94.2%**，为本次 ft64 评估中 SR 最高的 ckpt。state-fix 后的新结果全面高于老结果，提升幅度 25.5-38.5 个百分点。

---

---

## 每个 ckpt 的平均完成时长

> 单位：秒。基于 `video_duration_sec` 统计。

| ckpt | libero_goal | libero_10 | libero_object | libero_spatial | 总体平均 |
|------|-------------|-----------|---------------|----------------|----------|
| steps_5000 | 17.7 | 49.2 | 24.1 | 15.6 | 26.6 |
| steps_10000 | 16.3 | 41.9 | 15.5 | 13.0 | 21.7 |
| steps_20000 | 13.7 | 38.7 | 13.8 | 11.2 | 19.3 |
| steps_30000 | 14.6 | 38.5 | 16.1 | 11.0 | 20.0 |
| steps_40000 | 12.5 | 35.1 | 14.5 | 10.6 | 18.2 |
| steps_50000 | 11.9 | 31.7 | 15.1 | 10.9 | 17.4 |
| steps_60000 | 12.2 | 30.6 | 13.8 | 10.6 | 16.8 |
| steps_70000 | 11.9 | 29.9 | 14.2 | 10.8 | 16.7 |
| steps_80000 | 12.4 | 29.5 | 14.3 | 10.8 | 16.7 |

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
| steps_5000 | 76% | 0% | 36% | 88% | 88% | 88% | 72% | 26% | 82% | 94% |
| steps_10000 | 36% | 24% | 38% | 98% | 100% | 96% | 90% | 52% | 86% | 100% |
| steps_20000 | 98% | 68% | 68% | 100% | 100% | 100% | 92% | 74% | 84% | 100% |
| steps_30000 | 88% | 44% | 76% | 98% | 100% | 84% | 86% | 56% | 96% | 100% |
| steps_40000 | 100% | 76% | 90% | 100% | 100% | 98% | 98% | 68% | 98% | 100% |
| steps_50000 | 100% | 88% | 96% | 94% | 98% | 100% | 100% | 86% | 100% | 100% |
| steps_60000 | 96% | 76% | 84% | 100% | 98% | 100% | 94% | 86% | 100% | 100% |
| steps_70000 | 100% | 82% | 96% | 100% | 100% | 100% | 96% | 78% | 100% | 100% |
| steps_80000 | 100% | 80% | 96% | 100% | 96% | 96% | 96% | 74% | 98% | 100% |

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
| steps_5000 | 0% | 0% | 10% | 4% | 32% | 26% | 0% | 0% | 0% | 48% |
| steps_10000 | 86% | 2% | 48% | 22% | 56% | 92% | 12% | 6% | 8% | 48% |
| steps_20000 | 58% | 6% | 52% | 62% | 82% | 96% | 40% | 48% | 30% | 42% |
| steps_30000 | 74% | 28% | 74% | 62% | 94% | 56% | 40% | 74% | 12% | 46% |
| steps_40000 | 84% | 2% | 86% | 76% | 94% | 40% | 48% | 70% | 62% | 74% |
| steps_50000 | 82% | 50% | 92% | 76% | 94% | 72% | 84% | 60% | 80% | 96% |
| steps_60000 | 86% | 36% | 92% | 82% | 96% | 76% | 86% | 86% | 90% | 74% |
| steps_70000 | 84% | 82% | 90% | 86% | 98% | 88% | 80% | 74% | 90% | 92% |
| steps_80000 | 80% | 88% | 94% | 90% | 96% | 94% | 84% | 78% | 90% | 90% |

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
| steps_5000 | 26% | 12% | 20% | 64% | 98% | 18% | 4% | 32% | 36% | 62% |
| steps_10000 | 46% | 96% | 100% | 96% | 70% | 98% | 76% | 100% | 100% | 94% |
| steps_20000 | 96% | 100% | 100% | 100% | 100% | 98% | 98% | 80% | 100% | 98% |
| steps_30000 | 92% | 98% | 82% | 94% | 100% | 100% | 100% | 84% | 100% | 90% |
| steps_40000 | 84% | 86% | 100% | 100% | 100% | 100% | 98% | 94% | 100% | 94% |
| steps_50000 | 100% | 94% | 98% | 98% | 100% | 100% | 100% | 72% | 100% | 84% |
| steps_60000 | 96% | 100% | 94% | 98% | 100% | 100% | 100% | 98% | 100% | 96% |
| steps_70000 | 100% | 100% | 92% | 98% | 100% | 100% | 100% | 94% | 100% | 90% |
| steps_80000 | 100% | 96% | 98% | 100% | 100% | 100% | 100% | 96% | 98% | 90% |

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
| steps_5000 | 72% | 50% | 32% | 42% | 82% | 50% | 76% | 66% | 72% | 16% |
| steps_10000 | 96% | 96% | 80% | 76% | 96% | 82% | 90% | 64% | 90% | 60% |
| steps_20000 | 100% | 98% | 86% | 100% | 94% | 82% | 92% | 94% | 88% | 96% |
| steps_30000 | 100% | 100% | 96% | 98% | 96% | 96% | 94% | 88% | 98% | 98% |
| steps_40000 | 100% | 100% | 94% | 100% | 100% | 98% | 96% | 94% | 100% | 88% |
| steps_50000 | 98% | 94% | 96% | 96% | 100% | 94% | 96% | 86% | 94% | 96% |
| steps_60000 | 98% | 100% | 92% | 98% | 98% | 100% | 98% | 100% | 98% | 96% |
| steps_70000 | 100% | 100% | 94% | 98% | 98% | 100% | 94% | 92% | 98% | 94% |
| steps_80000 | 98% | 98% | 100% | 94% | 100% | 98% | 92% | 94% | 100% | 98% |
