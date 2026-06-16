# StarFlow-VLA 实验矩阵

## Scope

本文件记录当前 P0/P1 可执行实验矩阵、H2-a/H2-b 算法子实验、工程兼容性验证和详细设计文档全量研究假设覆盖状态。

本文件不宣称完整覆盖详细设计文档中的 H1-H8 全部研究版图。当前矩阵主要覆盖：

- StarFlowVLA framework + QwenPI_v3 reuse
- LayerwiseFM 7DoF P0 闭环
- MLP baseline 作为 H2 总 baseline
- H2-a future_tokens 消融（`0/16/32/64`）
- H2-b state conditioning 的 P1 入口
- starflow_mapping / checkpoint / eval smoke 追踪

尚未完整覆盖：

- H1 Flow Matching vs ACT 正式对照
- H3 Data Mixture 最优比例
- RoboCasa/RoboTwin 完整 cross benchmark
- ACT / DP / OpenVLA / StarVLA 原版完整 baseline
- Data Scaling 25/50/75/100
- Leave-One-Benchmark-Out
- Sim2Real / 真实机器人部署结果

所有 Stage A / Stage B / 未验证边界均保留；smoke、dry-run、single batch overfit、1 trial eval 不等价于正式长训或完整评测结论。

---

## 1. P0/P1 Engineering Execution Matrix

| 工程任务 ID | 关联算法实验 ID | 配置/入口 | 目标 | 当前状态 | 阻塞/下一步 |
| --- | --- | --- | --- | --- | --- |
| **P0-M5-Stage1** | E-H2a-03 / E-H2b-01 | `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml` | StarFlowVLA + QwenPI_v3 native + LayerwiseFM 7DoF action / 8D state；`num_target_vision_tokens=32`、`state_mode=discretized_instruction` | 配置解析、`apply_config_compat()`、`build_framework()` dry-run 通过；LIBERO batch schema、真实 forward/backward、loss finite、single batch overfit 前置验证通过；真实 `train_starvla.py` 10 step 训练闭环通过；**正式长训进行中：`P0-M5-E-H2a-04_starflow_libero-goal_qwen3vl4b_lwfm_ft32_250615`（run_id 标签误写为 E-H2a-04，实际配置为 ft32，对应 E-H2a-03），当前约 steps_15500 / 100000，不代表最终成功率** | 运行依赖当前环境需显式设置 `LD_LIBRARY_PATH`、`MASTER_ADDR`、`MASTER_PORT`、`RANK`、`LOCAL_RANK`、`WORLD_SIZE`；policy server 冷启动过长，完整 LIBERO 评测未端到端跑通 |
| **P0-M5-QwenPI-v3-Compatibility** | — | `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml` with `framework.name=QwenPI_v3` | 工程兼容性验证：确认 StarFlowVLA facade 没有破坏原 QwenPI_v3 baseline 入口 | 真实 forward/backward、loss finite 通过 | 未运行 baseline overfit / checkpoint / 长训；不属于 H2-a/H2-b 算法实验 |
| **P0-M6-MLP** | H2-total-baseline | `configs/starflow_vla/stage2_mlp_baseline.yaml` | H2 总 baseline：future_tokens + cross-DiT 主路线与 MLP/OFT/VLA_AdapterHeader baseline 的受控对照；不属于 H2-a/H2-b 子消融 | 配置解析、`apply_config_compat()`、`build_framework()` dry-run 通过；真实 forward/backward、loss finite、single batch overfit 前置验证通过 | 未保存 baseline checkpoint / 未长训；OFT / VLA_AdapterHeader 可作为后续扩展 baseline |
| **P0-M7-FutureTokens** | E-H2a-01 / E-H2a-02 / E-H2a-04 | `configs/starflow_vla/ablations/future_tokens_0.yaml` / `future_tokens_16.yaml` / `future_tokens_64.yaml`（`ft=32` 由 P0-M5-Stage1 baseline 覆盖，不重复跑） | `num_target_vision_tokens=0/16/32/64` 消融；当前执行矩阵冻结为 `0/16/32/64`，`ft=8` 不作为当前 P0/P1 编号 | `ft=0/16/32/64` 4 组配置级派生与默认 dry-run 通过；4 组真实 forward/backward、loss finite、single batch overfit 前置验证通过；`ablations/` 下 3 个新增独立 yaml 已创建（ft=32 使用 stage1 baseline） | 未运行完整训练 / eval；`stage3_future_token_ablation.yaml` 单文件中的 `num_target_vision_tokens_values` 未被训练代码消费，需使用独立 yaml 或 sweep 脚本 |
| **P1-M1-StateContinuousHead** | E-H2b-02 | `configs/starflow_vla/state/continuous_head.yaml` | continuous state encoder 路径对照 | 配置入口已创建；未实现 `state_mode=continuous_head` 运行时路径；未训练 | 需先实现 runtime，再进入完整实验；runtime 实现前禁止写成已完成正式实验 |
| **P2-M1-StateHybridGated** | E-H2b-03 | `configs/starflow_vla/state/hybrid_gated.yaml` | hybrid gated state conditioning | P2 advanced 占位；配置已创建，未训练 | 不进入当前 P0/P1 执行矩阵 |
| **P2-M2-StateHybridGatedCross** | E-H2b-04 | `configs/starflow_vla/state/hybrid_gated_cross.yaml` | hybrid gated 跨 benchmark | P2 advanced 占位；配置已创建，未训练 | 不进入当前 P0/P1 执行矩阵；依赖 RoboCasa/RoboTwin 数据 |
| **P0-M8-LIBERO-Batch** | — | `tests/test_starflow_libero_batch.py` | LIBERO batch schema smoke | 真实 `libero_goal` batch 通过；action=7D，state=8D | 无 |
| **P0-M9-Mapping** | — | `tests/test_starflow_checkpoint_mapping.py` | checkpoint sidecar `starflow_mapping.json` | unittest 通过；lightweight checkpoint 已自动写入 `scaler.pt`、`config.yaml`、`config.full.yaml`、`starflow_mapping.json`；`steps_1` smoke checkpoint 已补齐 `scaler.pt`；`load_model_weights(..., strict=True)` 通过；bootstrap checkpoint + `resume 100 step` smoke 通过 | 无 |
| **P0-M10-Eval** | — | `tests/test_starflow_eval_preflight.py` | LIBERO eval smoke preflight | shell 语法检查通过；checkpoint mapping preflight 通过；`steps_1` 的 1 task × 1 trial rollout smoke 通过，success_rate=0.0；`steps_10` 真实训练产物已完成 `libero_goal` 全 10 task × 1 trial task sweep，success_rate=0.0，`timeout_no_success=10`；`eval_report.json` 已包含 `failure_category`、`checkpoint_hash`、`config_hash`、`data_version`、`starflow_mapping` | 未运行标准 50 trials/task 的完整 LIBERO 评测 / 细粒度 failure taxonomy；policy server 冷启动过长，quick regression 未端到端跑通 |

---

## 2. H2-a / H2-b Core Algorithm Sub-Matrix

| 实验 ID | 工程任务映射 | 变量 | 配置 | 是否新增训练 | 当前状态 | 说明 |
| --- | --- | --- | --- | --- | --- | --- |
| **E-H2a-01** | P0-M7 | `num_target_vision_tokens=0` | `configs/starflow_vla/ablations/future_tokens_0.yaml` | 是 | 配置已创建，前置 smoke 通过，待正式长训 | 验证无 future planning slots 的边界 |
| **E-H2a-02** | P0-M7 | `num_target_vision_tokens=16` | `configs/starflow_vla/ablations/future_tokens_16.yaml` | 是 | 配置已创建，前置 smoke 通过，待正式长训 | 低 token 预算 |
| **E-H2a-03** | P0-M5-Stage1 | `num_target_vision_tokens=32` | `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml` | 否 | 当前 baseline 长训进行中；run_id 命名标签误写为 E-H2a-04，实际对应 E-H2a-03 | 由 baseline 覆盖，不重复跑 |
| **E-H2a-04** | P0-M7 | `num_target_vision_tokens=64` | `configs/starflow_vla/ablations/future_tokens_64.yaml` | 是 | 配置已创建，前置 smoke 通过，待正式长训 | 高 token 预算，关注显存和延迟 |
| **E-H2b-01** | P0-M5-Stage1 | `state_mode=discretized_instruction` | `configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml` | 否 | 当前 baseline 默认路径 | 默认状态路径 |
| **E-H2b-02** | P1-M1 | `state_mode=continuous_head` | `configs/starflow_vla/state/continuous_head.yaml` | 是 | 配置入口已创建，`state_mode=continuous_head` 运行时路径待实现 | P1 对照实验 |
| **E-H2b-03** | P2-M1 | `state_mode=hybrid_gated` | `configs/starflow_vla/state/hybrid_gated.yaml` | 是 | P2 advanced 占位 | 不进入当前 P0/P1 执行矩阵 |
| **E-H2b-04** | P2-M2 | `state_mode=hybrid_gated` | `configs/starflow_vla/state/hybrid_gated_cross.yaml` | 是 | P2 advanced 占位 | 不进入当前 P0/P1 执行矩阵 |

> 说明：`ft=8` 曾为早期候选，当前 P0/P1 可执行矩阵已冻结为 `0/16/32/64`，不分配当前 E-H2a 编号；`ft=8` 仅可作为资源充足时的补充 dense-sweep，不进入当前执行矩阵。

---

## 3. Engineering and Baseline Appendix

| 项目 | 类型 | 是否进入 H2-a/H2-b 子矩阵 | 是否支撑详细设计文档 | 说明 |
| --- | --- | --- | --- | --- |
| **QwenPI_v3 baseline compatibility** | 工程兼容性验证 | 否 | 支撑 StarVLA-native 复用与上游兼容 | 验证 StarFlowVLA facade 不破坏原 baseline；不进入 H2-a/H2-b 算法子实验矩阵 |
| **MLP baseline** | H2 总 baseline | 否 | 支撑 H2 主线对照 | 用于 `future_tokens + cross-DiT` vs `MLP/OFT/VLA_AdapterHeader` 对照；当前只实现 MLP，OFT/VLA_AdapterHeader 可作为后续扩展 |
| **OFT / VLA_AdapterHeader** | 后续 baseline | 否 | 可进一步支撑 H2 | 当前未运行，标注为待扩展 baseline |
| **hybrid_gated** | P2 advanced | 否 | 仅支撑未来 H2-b 扩展 | 不进入当前 P0/P1 |
| **VGGT / geometry fusion** | P2 / 世界模型项目接口 | 否 | 不支撑本项目 P0/P1 主线 | 仅接口预留 |

---

## 4. Design Hypothesis Coverage Status

| 详细设计假设/模块 | 当前覆盖状态 | 当前支撑文件/实验 | 尚缺内容 |
| --- | --- | --- | --- |
| StarVLA-native framework | 已覆盖 | P0-M2/P0-M5、`MODULE_MAPPING.md`、`PATCH_MANIFEST.md` | 后续随上游更新 compatibility audit |
| P0 7DoF LayerwiseFM 闭环 | 部分覆盖 | P0-M5-Stage1 | 正式长训与完整 eval |
| H2-a future_tokens | 部分覆盖 | E-H2a-01 至 E-H2a-04 | 需完整训练、评测、显存/延迟统计 |
| H2-b state conditioning | 部分覆盖 | E-H2b-01/E-H2b-02 | `continuous_head` runtime 与正式实验 |
| H2 总 baseline | 部分覆盖 | P0-M6-MLP | OFT/VLA_AdapterHeader 可后续补充 |
| H1 FM vs ACT | 未覆盖 | 当前无完整 ACT 对照 | 需要 ACT baseline |
| H3 Data Mixture | 未覆盖 | 当前无三数据集混训比例矩阵 | LIBERO/RoboCasa/RoboTwin mixture |
| Cross Benchmark | 未完整覆盖 | eval smoke | RoboCasa/RoboTwin 完整评测 |
| Sim2Real / deployment | 未覆盖 | 接口与安全门设计 | 真实机器人验证 |

---

## Commands

```bash
# P0 基础 smoke
.venv/bin/python -m unittest tests.test_starflow_libero_batch -v
.venv/bin/python -m unittest tests.test_starflow_checkpoint_mapping -v
.venv/bin/python -m unittest tests.test_starflow_eval_preflight -v

# P0-M5-Stage1 / E-H2a-03 / E-H2b-01 长训
# 当前在 tmux starflow_train 中运行：NUM_PROCESSES=4，有效全局 batch=32
RUN_ID="P0-M5-E-H2a-03_starflow_libero-goal_qwen3vl4b_lwfm_ft32_$(date +%y%m%d)" \
MAX_TRAIN_STEPS=100000 \
SAVE_INTERVAL=500 \
LOGGING_FREQUENCY=100 \
EVAL_INTERVAL=1000 \
NUM_PROCESSES=4 \
GRADIENT_ACCUMULATION_STEPS=2 \
PER_DEVICE_BATCH_SIZE=4 \
NUM_WORKERS=1 \
CONFIG_YAML=configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml \
bash examples/LIBERO/train_files/run_starflow_train_ready.sh

# P0-M7 Future Tokens 消融 sweep（3 个新增独立 yaml；ft=32 由 P0-M5 baseline 覆盖）
for ft in 0 16 64; do
  idx=$((ft == 0 ? 1 : ft / 16 + 1))
  RUN_ID="P0-M7-E-H2a-$(printf '%02d' ${idx})_starflow_libero-goal_qwen3vl4b_lwfm_ft${ft}_$(date +%y%m%d)" \
  CONFIG_YAML="configs/starflow_vla/ablations/future_tokens_${ft}.yaml" \
  MAX_TRAIN_STEPS=100000 \
  SAVE_INTERVAL=500 \
  LOGGING_FREQUENCY=100 \
  EVAL_INTERVAL=1000 \
  NUM_PROCESSES=4 \
  GRADIENT_ACCUMULATION_STEPS=2 \
  PER_DEVICE_BATCH_SIZE=4 \
  NUM_WORKERS=1 \
  bash examples/LIBERO/train_files/run_starflow_train_ready.sh
done

# P1-M1 / E-H2b-02 State Conditioning 对照
# 注意：state_mode=continuous_head 运行时路径尚未实现，当前命令仅作为配置占位
RUN_ID="P1-M1-E-H2b-02_starflow_libero-goal_qwen3vl4b_lwfm_continuous_$(date +%y%m%d)" \
CONFIG_YAML=configs/starflow_vla/state/continuous_head.yaml \
MAX_TRAIN_STEPS=100000 \
SAVE_INTERVAL=500 \
LOGGING_FREQUENCY=100 \
EVAL_INTERVAL=1000 \
NUM_PROCESSES=4 \
GRADIENT_ACCUMULATION_STEPS=2 \
PER_DEVICE_BATCH_SIZE=4 \
NUM_WORKERS=1 \
bash examples/LIBERO/train_files/run_starflow_train_ready.sh
```

## Not Run

未运行标准 50 trials/task 的完整 LIBERO 评测、细粒度 failure taxonomy、H1/H3 完整研究实验、RoboCasa/RoboTwin 完整 cross benchmark、真实机器人部署。
