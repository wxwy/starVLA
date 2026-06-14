# StarFlow-VLA 实验矩阵

## Scope

本文件只记录 P0 阶段已建立的实验配置入口、验证状态和当前阻塞，不记录未运行的训练、评测、成功率、显存或延迟结果。

## P0 Matrix

| ID | 配置 | 目标 | 当前状态 | 阻塞 |
| --- | --- | --- | --- | --- |
| P0-M5-Stage1 | `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml` | StarFlowVLA + QwenPI_v3 native + LayerwiseFM 7DoF action / 8D state | 配置解析、`apply_config_compat()`、monkeypatch `build_framework()` dry-run 通过；LIBERO batch schema、真实 forward/backward、loss finite、single batch overfit 前置验证通过 | 未运行完整训练 / checkpoint save-load |
| P0-M5-QwenPI-v3-Baseline | `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml` with `framework.name=QwenPI_v3` | QwenPI_v3 baseline compatibility | 真实 forward/backward、loss finite 通过 | 未运行 baseline overfit / checkpoint |
| P0-M6-MLP | `configs/starflow_vla/stage2_mlp_baseline.yaml` | QwenOFT + MLP 7DoF action / 8D state baseline | 配置解析、`apply_config_compat()`、monkeypatch `build_framework()` dry-run 通过；真实 forward/backward、loss finite、single batch overfit 前置验证通过 | 未保存 baseline checkpoint |
| P0-M7-FutureTokens | `configs/starflow_vla/stage3_future_token_ablation.yaml` | `num_target_vision_tokens=0/8/16/32/64` 消融 | 5 组配置级派生与默认 dry-run 通过；5 组真实 forward/backward、loss finite、single batch overfit 前置验证通过 | 未运行完整训练 / eval |
| P0-M8-LIBERO-Batch | `tests/test_starflow_libero_batch.py` | LIBERO batch schema smoke | 真实 `libero_goal` batch 通过；action=7D，state=8D | 无 |
| P0-M9-Mapping | `tests/test_starflow_checkpoint_mapping.py` | checkpoint sidecar `starflow_mapping.json` | unittest 通过；`steps_1` smoke checkpoint 已保存 mapping；`load_model_weights(..., strict=True)` 通过 | 未运行 resume 100 step |
| P0-M10-Eval | `tests/test_starflow_eval_preflight.py` | LIBERO eval smoke preflight | shell 语法检查通过；checkpoint mapping preflight 通过；1 task × 1 trial LIBERO rollout smoke 通过，success_rate=0.0 | 未运行完整 LIBERO suite / failure taxonomy |

## Commands

```bash
.venv/bin/python -m unittest tests.test_starflow_libero_batch -v
.venv/bin/python -m unittest tests.test_starflow_checkpoint_mapping -v
.venv/bin/python -m unittest tests.test_starflow_eval_preflight -v
```

## Not Run

未运行 resume 100 step、完整 LIBERO suite、failure taxonomy、完整训练、评测或部署。
