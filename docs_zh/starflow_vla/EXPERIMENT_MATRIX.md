# StarFlow-VLA 实验矩阵

## Scope

本文件只记录 P0 阶段已建立的实验配置入口、验证状态和当前阻塞，不记录未运行的训练、评测、成功率、显存或延迟结果。

## P0 Matrix

| ID | 配置 | 目标 | 当前状态 | 阻塞 |
| --- | --- | --- | --- | --- |
| P0-M5-Stage1 | `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml` | StarFlowVLA + QwenPI_v3 native + LayerwiseFM 7DoF | 配置解析、`apply_config_compat()`、monkeypatch `build_framework()` dry-run 通过 | 缺少 `playground/Datasets/LEROBOT_LIBERO_DATA`，未运行真实 batch / forward / overfit |
| P0-M6-MLP | `configs/starflow_vla/stage2_mlp_baseline.yaml` | QwenOFT + MLP 7DoF baseline | 配置解析、`apply_config_compat()`、monkeypatch `build_framework()` dry-run 通过 | 缺少 LIBERO 数据，未运行真实 baseline overfit |
| P0-M7-FutureTokens | `configs/starflow_vla/stage3_future_token_ablation.yaml` | `num_target_vision_tokens=0/8/16/32/64` 消融 | 5 组配置级派生与默认 dry-run 通过 | 缺少 LIBERO 数据，未运行真实 forward / overfit |
| P0-M8-LIBERO-Batch | `tests/test_starflow_libero_batch.py` | LIBERO batch schema smoke | 数据缺失时显式 skip | 缺少 `playground/Datasets/LEROBOT_LIBERO_DATA` |
| P0-M9-Mapping | `tests/test_starflow_checkpoint_mapping.py` | checkpoint sidecar `starflow_mapping.json` | unittest 通过 | 真实 checkpoint save/load 待 P0-M5 训练产物 |
| P0-M10-Eval | `tests/test_starflow_eval_preflight.py` | LIBERO eval smoke preflight | shell 语法检查通过，checkpoint 检查 skip | 缺少 P0 checkpoint 与 LIBERO 数据 |

## Commands

```bash
.venv/bin/python -m unittest tests.test_starflow_libero_batch -v
.venv/bin/python -m unittest tests.test_starflow_checkpoint_mapping -v
.venv/bin/python -m unittest tests.test_starflow_eval_preflight -v
```

## Not Run

未运行真实模型加载、LIBERO batch 读取、forward/backward、loss finite、single batch overfit、checkpoint 保存/加载、policy server、LIBERO rollout、训练、评测或部署。
