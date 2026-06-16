# StarFlow-VLA 实验矩阵

## Scope

本文件记录 P0/P1/P2 阶段已建立的实验配置入口、验证状态、当前阻塞，以及与 `ALGORITHM_OPTIMIZATION_PLAN.md` 中 H2-a/H2-b 假设的映射关系。未在目标环境运行的训练、评测结果保持占位符，不填写推测值。

## P0 Matrix

| ID | 算法假设 ID | 配置 | 目标 | 当前状态 | 阻塞 |
| --- | --- | --- | --- | --- | --- |
| P0-M5-Stage1 | E-H2a-03 / E-H2b-01 | `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml` | StarFlowVLA + QwenPI_v3 native + LayerwiseFM 7DoF action / 8D state；`num_target_vision_tokens=32`、`state_mode=discretized_instruction` | 配置解析、`apply_config_compat()`、monkeypatch `build_framework()` dry-run 通过；LIBERO batch schema、真实 forward/backward、loss finite、single batch overfit 前置验证通过；真实 `train_starvla.py` 10 step 训练闭环通过；**正式长训进行中：`P0-M5-E-H2a-04_starflow_libero-goal_qwen3vl4b_lwfm_ft32_250615`（run_id 标签误写为 E-H2a-04，实际配置为 ft32，对应 E-H2a-03），当前约 steps_15500 / 100000** | 运行依赖当前环境需显式设置 `LD_LIBRARY_PATH`、`MASTER_ADDR`、`MASTER_PORT`、`RANK`、`LOCAL_RANK`、`WORLD_SIZE` |
| P0-M5-QwenPI-v3-Baseline | — | `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml` with `framework.name=QwenPI_v3` | QwenPI_v3 baseline compatibility | 真实 forward/backward、loss finite 通过 | 未运行 baseline overfit / checkpoint / 长训 |
| P0-M6-MLP | — | `configs/starflow_vla/stage2_mlp_baseline.yaml` | QwenOFT + MLP 7DoF action / 8D state baseline | 配置解析、`apply_config_compat()`、monkeypatch `build_framework()` dry-run 通过；真实 forward/backward、loss finite、single batch overfit 前置验证通过 | 未保存 baseline checkpoint / 未长训 |
| P0-M7-FutureTokens | E-H2a-01 / E-H2a-02 / E-H2a-03 / E-H2a-04 | `configs/starflow_vla/stage3_future_token_ablation.yaml`（单文件多值占位）+ `configs/starflow_vla/ablations/future_tokens_{0,16,32,64}.yaml` | `num_target_vision_tokens=0/16/32/64` 消融 | 4 组配置级派生与默认 dry-run 通过；4 组真实 forward/backward、loss finite、single batch overfit 前置验证通过；`ablations/` 下 4 个独立 yaml 已创建 | 未运行完整训练 / eval；`stage3` 单文件中的 `num_target_vision_tokens_values` 未被训练代码消费，需使用独立 yaml 或 sweep 脚本 |
| P0-M8-LIBERO-Batch | — | `tests/test_starflow_libero_batch.py` | LIBERO batch schema smoke | 真实 `libero_goal` batch 通过；action=7D，state=8D | 无 |
| P0-M9-Mapping | — | `tests/test_starflow_checkpoint_mapping.py` | checkpoint sidecar `starflow_mapping.json` | unittest 通过；lightweight checkpoint 已自动写入 `scaler.pt`、`config.yaml`、`config.full.yaml`、`starflow_mapping.json`；`steps_1` smoke checkpoint 已补齐 `scaler.pt`；`load_model_weights(..., strict=True)` 通过；bootstrap checkpoint + `resume 100 step` smoke 通过 | 无 |
| P0-M10-Eval | — | `tests/test_starflow_eval_preflight.py` | LIBERO eval smoke preflight | shell 语法检查通过；checkpoint mapping preflight 通过；`steps_1` 的 1 task × 1 trial rollout smoke 通过，success_rate=0.0；`steps_10` 真实训练产物已完成 `libero_goal` 全 10 task × 1 trial task sweep，success_rate=0.0，`timeout_no_success=10`；`eval_report.json` 已包含 `failure_category`、`checkpoint_hash`、`config_hash`、`data_version`、`starflow_mapping` | 未运行标准 50 trials/task 的完整 LIBERO 评测 / 细粒度 failure taxonomy；policy server 冷启动过长，quick regression 未端到端跑通 |

## P0 Matrix 与 H2-a/H2-b 映射

| P0 Matrix 项 | 对应 H2-a/H2-b exp_id | 说明 |
| --- | --- | --- |
| P0-M5-Stage1 | **E-H2b-01** + **E-H2a-03** | 默认 `state_mode=discretized_instruction`，`num_target_vision_tokens=32` |
| P0-M7-FutureTokens | **E-H2a-01/02/03/04** | `num_target_vision_tokens=0/16/32/64`，对应 4 个独立 ablation yaml |
| P0-M6-MLP | — | H1 口径 baseline，不在 H2-a/H2-b 矩阵内 |
| P0-M5-QwenPI-v3-Baseline | — | 基线复用验证，不在 H2-a/H2-b 矩阵内 |

## P1/P2 新增实验（尚未完整实施）

| ID | 算法假设 ID | 配置 | 目标 | 当前状态 | 阻塞 |
| --- | --- | --- | --- | --- | --- |
| P1-S1 | E-H2b-02 | `configs/starflow_vla/state/continuous_head.yaml` | continuous state encoder 路径对照 | 配置已创建，未训练 | 未实现 `state_mode=continuous_head` 运行时路径 |
| P2-S1 | E-H2b-03 | `configs/starflow_vla/state/hybrid_gated.yaml` | hybrid gated state conditioning | 配置已创建，未训练 | P2 计划 |
| P2-S2 | E-H2b-04 | `configs/starflow_vla/state/hybrid_gated_cross.yaml` | hybrid gated 跨 benchmark | 配置已创建，未训练 | P2 计划；依赖 RoboCasa/RoboTwin 数据 |

## Commands

```bash
# P0 基础 smoke
.venv/bin/python -m unittest tests.test_starflow_libero_batch -v
.venv/bin/python -m unittest tests.test_starflow_checkpoint_mapping -v
.venv/bin/python -m unittest tests.test_starflow_eval_preflight -v

# P0-M5-Stage1 / E-H2a-03 / E-H2b-01 长训（当前在 tmux starflow_train 中运行）
bash examples/LIBERO/train_files/run_starflow_train_ready.sh

# P0-M7 Future Tokens 消融 sweep（4 个独立 yaml）
for ft in 0 16 32 64; do
  RUN_ID="P0-M7-E-H2a-$(printf '%02d' $((ft/16+1)))_starflow_libero-goal_qwen3vl4b_lwfm_ft${ft}_$(date +%y%m%d)" \
  bash examples/LIBERO/train_files/run_starflow_train_ready.sh
done

# P1 State Conditioning 对照
RUN_ID="P1-S1-E-H2b-02_starflow_libero-goal_qwen3vl4b_lwfm_continuous_$(date +%y%m%d)" \
  bash examples/LIBERO/train_files/run_starflow_train_ready.sh
```

## Not Run

未运行标准 50 trials/task 的完整 LIBERO 评测、细粒度 failure taxonomy、长训、正式部署。
