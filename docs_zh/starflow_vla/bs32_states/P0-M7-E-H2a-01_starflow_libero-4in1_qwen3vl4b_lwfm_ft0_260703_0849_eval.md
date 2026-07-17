# P0-M7-E-H2a-01: StarFlow LIBERO 4-in-1 Qwen3VL-4B future_tokens=0（bs32）state-fix 评估追踪

> **实验代号**: E-H2a-01 / P0-M7  
> **状态**: ✅ 评估完成  
> **run_id**: `P0-M7-E-H2a-01_starflow_libero-4in1_qwen3vl4b_lwfm_ft0_260703_0849`  
> **启动时间**: 2026-07-16 11:46 CST  
> **当前更新**: 2026-07-17 09:29 CST  
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

> 最后更新：2026-07-17 09:29 CST

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
