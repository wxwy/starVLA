# MoWA 实验启动 Readiness Matrix

本文件用于回答一个具体工程问题：在真正启动各实验前，当前 checkout 还缺什么，哪些实验已经具备可执行路径，哪些仍停留在接口 / plan / smoke。

## 生成方式

使用现有 smoke / readiness / rollout JSON 汇总，不重新运行训练：

```bash
.venv/bin/python tools/mowa/experiment_launch_readiness_matrix.py \
  --repo-root . \
  --output docs_zh/mowa/mowa_experiment_launch_readiness_matrix.json
```

## 字段约定

- `can_start_now`
  - `true`：就当前仓库和现有 gate 而言，可以直接启动该实验。
  - `false`：仍有明确 blocker，或该项本来就不是可启动训练实验。
- `status`
  - `bounded_executable_but_full_launch_blocked`：短链路、bounded smoke 或 runtime 路径已经证明可执行，但不等于正式实验已放行。
  - `runtime_integrated_but_training_gated`：代码路径和可观测性已接上，但训练候选或 launch gate 未放开。
  - `interface_ready_but_*`：仅接口/plan 完成，未进入真实训练主干。
  - `plan_ready_only` / `not_started`：仅有计划，或还没有 readiness 证据。
- `blocking_items`
  - 直接列出当前最需要解决的缺口，优先来自已有 readiness / smoke 报告中的 `unresolved_items`。

## 当前用途

这份 matrix 不是实验结论，也不是新 Source-of-Truth。它只是把已经散落在：

- `mowa_e001_readiness_smoke.json`
- `mowa_e002_future_gated_heads_comparison_smoke.json`
- `mowa_future_latent_prior_interface_smoke.json`
- `mowa_hlc_gci_interface_smoke.json`
- `mowa_e004_hlc_gci_checkpoint_preflight_smoke.json`
- `mowa_e004_hlc_gci_policy_rollout_smoke.json`
- `mowa_e006_policy_rollout_smoke.json`

等文件里的状态，统一整理成“是否可启动”的单一视图，避免后续反复人工翻 report。
