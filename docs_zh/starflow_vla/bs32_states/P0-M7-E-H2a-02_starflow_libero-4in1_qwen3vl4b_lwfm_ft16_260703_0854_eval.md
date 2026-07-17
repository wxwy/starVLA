# P0-M7-E-H2a-02: StarFlow LIBERO 4-in-1 Qwen3VL-4B future_tokens=16（bs32）state-fix 评估追踪

> **实验代号**: E-H2a-02 / P0-M7  
> **状态**: ✅ 评估完成  
> **run_id**: `P0-M7-E-H2a-02_starflow_libero-4in1_qwen3vl4b_lwfm_ft16_260703_0854`  
> **启动时间**: 2026-07-16 12:51 CST  
> **当前更新**: 2026-07-17 09:30 CST  
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

> 最后更新：2026-07-17 09:30 CST

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
