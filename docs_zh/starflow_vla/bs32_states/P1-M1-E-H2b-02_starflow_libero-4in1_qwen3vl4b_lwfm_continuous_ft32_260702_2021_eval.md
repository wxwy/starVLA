# P1-M1-E-H2b-02: StarFlow LIBERO 4-in-1 Qwen3VL-4B continuous_head（bs32）state-fix 评估追踪

> **实验代号**: E-H2b-02 / P1-M1  
> **状态**: ✅ 评估完成  
> **run_id**: `P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260702_2021`  
> **启动时间**: 2026-07-16 13:02 CST  
> **当前更新**: 2026-07-17 09:30 CST  
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

> 最后更新：2026-07-17 09:30 CST

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
