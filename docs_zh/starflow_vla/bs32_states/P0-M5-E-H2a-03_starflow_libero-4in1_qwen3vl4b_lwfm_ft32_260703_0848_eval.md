# P0-M5-E-H2a-03: StarFlow LIBERO 4-in-1 Qwen3VL-4B future_tokens=32（bs32）state-fix 评估追踪

> **实验代号**: E-H2a-03 / P0-M5
> **状态**: 🟢 评估运行中
> **run_id**: `P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_260703_0848`
> **启动时间**: 2026-07-16 11:48 CST
> **当前更新**: 2026-07-16 18:33 CST
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
| `workers` | 20 |
| `num_trials` | 50 |
| `suites` | libero_goal, libero_10, libero_object, libero_spatial |
| `ckpt 列表` | steps_5000, steps_10000, steps_20000, steps_30000, steps_40000, steps_50000, steps_60000, steps_70000, steps_80000 |
| `policy server` | 单 server，固定端口 6720-6728 |
| `server 优化` | server-side batching（max_batch=8, timeout=30ms） |

---

## 评估进度

> 最后更新：2026-07-16 18:33 CST

| ckpt | 状态 | 总 eps | 总 successes | 总 SR |
|------|------|--------|--------------|-------|
| steps_5000 | ✅ 完成 | 2000 / 2000 | 933 | 46.7% |
| steps_10000 | ✅ 完成 | 2000 / 2000 | 1362 | 68.1% |
| steps_20000 | 🟡 运行中 | 1519 / 2000 | 1282 | 84.4% |
| steps_30000 | ⏳ 待测 | 0 / 2000 | — | — |
| steps_40000 | ⏳ 待测 | 0 / 2000 | — | — |
| steps_50000 | ⏳ 待测 | 0 / 2000 | — | — |
| steps_60000 | ⏳ 待测 | 0 / 2000 | — | — |
| steps_70000 | ⏳ 待测 | 0 / 2000 | — | — |
| steps_80000 | ⏳ 待测 | 0 / 2000 | — | — |

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

---

## state 修复前后对比

本次重新评测前，`playground/eval_results/` 中 ft32 的 steps_5000/steps_10000 为未传 state 的结果。对比如下：

| ckpt | 无 state（老） | 有 state（新） | 差异 |
|------|---------------|---------------|------|
| steps_5000 | 7.8% | 46.7% | +38.9% |
| steps_10000 | 40.9% | 68.1% | +27.2% |

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

---

## 备注

- 修复 commit：`b7af6758 fix(LIBERO): pass robot state to model during evaluation`（与 ft0 共用同一 state-fix）
- 评测过程中使用 workers=20，当前 speed 约 10-15 eps/min（尾部 task 较慢）。
- 每个 ckpt 固定端口：steps_5000→6720，steps_10000→6721，...，steps_80000→6728。
- 日志文件已调整为按秒命名格式：`starflow_eval_plan_YYYYMMDD_HHMMSS.log`，避免同一分钟内多次启动冲突。

---

## 中断说明

2026-07-16 15:15 左右，ft32 在启动 `steps_20000` 的 policy server 时遇到 **Server 启动超时**（日志：`[Thu Jul 16 03:15:08 PM CST 2026] ❌ Server 启动超时`）。后续尝试重启后再次在 `steps_20000` 启动阶段超时（`[Thu Jul 16 03:18:33 PM CST 2026] ❌ Server 启动超时`）。

当前状态：
- `test` tmux 会话仍在，之前评估脚本已退出，无 `672x` 端口监听。
- 无 `server_policy` / `eval_pool_manager` / `eval_libero` 进程在运行。
- `steps_5000` 和 `steps_10000` 结果已落盘；`steps_20000` 无有效 `eval_report_task_*.json` 输出。
- 需手动恢复后续评测，可从 `steps_20000` 重新启动 `playground/starflow_eval_plan.sh`。如果 server 启动超时反复出现，可能需要检查 GPU 显存/加载时间或调大 `starflow_eval_pool.sh` 中的 server 启动等待时间。

**17:34 更新**：发现一个新的 `server_policy.py` 进程（PID 431170）正在启动 `steps_20000`（端口 6722），CPU 占用约 188%，但端口尚未监听，可能仍在加载模型。状态已改为「评估恢复中」。

**18:33 更新**：`steps_20000` 评估继续进行中，当前已完成 1506/2000 trials，整体 SR 84.3%。
