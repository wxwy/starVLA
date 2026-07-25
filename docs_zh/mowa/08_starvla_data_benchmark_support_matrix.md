# MoWA 数据 / Benchmark 支持矩阵（基于 StarVLA 当前仓库）

本矩阵的目标不是复述 `README.md` 的宣传口径，而是区分：

1. StarVLA 仓库公开声称支持什么；
2. 仓库里实际存在哪些训练脚本、评测脚本和数据 registry；
3. 这些支持项对 MoWA 的优先级是什么。

判定口径：

- `已接入`：仓库内同时存在明确的数据接入或训练脚本证据。
- `评测已接入`：存在 benchmark / env 评测脚本。
- `部分接入`：只有训练或评测一侧，或数据 registry 仍依赖外部实现。
- `MoWA 优先级`：只代表 MoWA 当前工程路线下的推荐程度，不改变 StarVLA 原项目定位。

## 一、StarVLA 已有支持概览

`README.md` 当前公开列出的 benchmark 集成为：SimplerEnv、LIBERO、LIBERO-plus、Robocasa-GR1、Robocasa365、RoboTwin 2.0、DOMINO、BEHAVIOR、Calvin；`RLBench` 与 `SO101` 仍未勾选。

仓库层面的自动数据接入通过 [registry.py](/gemini/code/starVLA/starVLA/dataloader/gr00t_lerobot/registry.py:1) 完成，会自动扫描 `examples/*/train_files/data_registry/` 并合并到全局 registry。

## 二、MoWA 支持矩阵

| 项目 | StarVLA 训练侧 | StarVLA 评测侧 | 数据 registry / mixture | 证据结论 | 与 MoWA 相关性 | MoWA 优先级 | 备注 |
|---|---|---|---|---|---|---|---|
| RoboCasa365 | 是 | 是 | 是 | 已接入 | 高 | P0/G0 第一优先 | 单臂移动厨房任务，最接近 MoWA 第一闭环。 |
| RoboCasa tabletop / GR1 | 是 | 是 | 是 | 已接入 | 中 | 辅助 | household manipulation 强，但不应当作移动操作主证据。 |
| LIBERO | 是 | 是 | 是 | 已接入 | 低 | 辅助 | 适合 action schema / sanity，不是移动操作主证据。 |
| LIBERO-plus | 否，未见独立训练入口 | 是 | 否 | 评测已接入 | 低 | 后置评测 | 更像 benchmark 扩展，不是 MoWA 主数据源。 |
| RoboTwin 2.0 | 是 | 是 | 是 | 已接入 | 低-中 | 辅助 | 适合操作泛化、双臂与 action-side sanity。 |
| DOMINO | 是 | 是 | 是 | 已接入 | 中 | 诊断 / 辅助 | 动态场景有价值，但不是移动操作主线。 |
| CALVIN | 是 | 是 | 有训练脚本，未见独立 registry 文件内容外部补充 | 已接入 | 低-中 | 辅助 | 长时序操作有参考价值，但非移动操作。 |
| BEHAVIOR | 是 | 是 | 部分 | 部分接入 | 中-高 | 后置高难验证 | 长时序 household 相关，但工程复杂度高。 |
| SimplerEnv | 是 | 是 | 复用 OXE mixture | 已接入 | 低 | 泛化评测 | 更适合通用 VLA 泛化，不是 MoWA 主证据。 |
| VLA-Arena | 是 | 是 | 是 | 已接入 | 中 | 诊断 / 外推 | 可做跨域评测追踪，不是首批训练域。 |
| OXE Bridge / RT-1 / DROID | 是 | 否，仓库内无对应独立 benchmark 入口 | 是 | 数据已接入 | 低 | 预训练 / 辅助 | 可作表征或辅训练域，不作为移动操作主证据。 |
| RLBench | 否 | 否 | 否 | 未接入 | 低 | 不纳入当前 MoWA | README 未勾选。 |
| SO101 | 否 | 否 | mixture 占位 | 未接入 | 低 | 不纳入当前 MoWA | README 未勾选，registry 仅占位。 |

## 三、关键证据

### 1. RoboCasa365

- [README.md](/gemini/code/starVLA/README.md:132) 将 `Robocasa365` 列为已支持 benchmark。
- [examples/Robocasa_365/README.md](/gemini/code/starVLA/examples/Robocasa_365/README.md:1) 提供完整 walk-through。
- [examples/Robocasa_365/train_files/data_registry/data_config.py](/gemini/code/starVLA/examples/Robocasa_365/train_files/data_registry/data_config.py:1) 定义了 `panda_omron_robocasa365` data config 与 `robocasa365_*` mixtures。
- [run_robocasa365.sh](/gemini/code/starVLA/examples/Robocasa_365/train_files/run_robocasa365.sh:1) 和 [run_robocasa365_all.sh](/gemini/code/starVLA/examples/Robocasa_365/train_files/run_robocasa365_all.sh:1) 提供单任务与全量训练入口。
- [model2robocasa365_interface.py](/gemini/code/starVLA/examples/Robocasa_365/eval_files/model2robocasa365_interface.py:1) 和 `run_eval.sh` 提供评测适配。

结论：`robocasa365` 不是纸面支持，而是 StarVLA 当前仓库里最完整的移动操作相关接入项之一。

### 2. LIBERO

- [README.md](/gemini/code/starVLA/README.md:132) 列为已支持。
- [examples/LIBERO/train_files/data_registry/data_config.py](/gemini/code/starVLA/examples/LIBERO/train_files/data_registry/data_config.py:1) 定义 `libero_all` / `libero_goal`。
- [examples/LIBERO/train_files/run_starflow_train_ready.sh](/gemini/code/starVLA/examples/LIBERO/train_files/run_starflow_train_ready.sh:1) 提供训练入口。
- [examples/LIBERO/eval_files/eval_libero.py](/gemini/code/starVLA/examples/LIBERO/eval_files/eval_libero.py:1) 提供评测入口。

结论：LIBERO 是当前最成熟的桌面操作基线，但对 MoWA 只能作为辅助 sanity，不宜当移动操作主证据。

### 3. RoboTwin / DOMINO / VLA-Arena / SimplerEnv

- RoboTwin 的 registry 位于 [examples/Robotwin/train_files/data_registry/data_config.py](/gemini/code/starVLA/examples/Robotwin/train_files/data_registry/data_config.py:1)，且同时有训练与评测脚本。
- DOMINO 的 registry 位于 [examples/DOMINO/train_files/data_registry/data_config.py](/gemini/code/starVLA/examples/DOMINO/train_files/data_registry/data_config.py:1)，同时有动态任务训练与评测脚本。
- VLA-Arena 有数据准备脚本 [data_preparation.sh](/gemini/code/starVLA/examples/VLA-Arena/data_preparation.sh:1)，也有 train/eval 入口。
- SimplerEnv 使用 OXE 训练配置 [starvla_cotrain_oxe.yaml](/gemini/code/starVLA/examples/SimplerEnv/train_files/starvla_cotrain_oxe.yaml:1)，评测通过 `model2simpler_interface.py`。

结论：这些项更适合 MoWA 的外推评测、泛化评测或动态诊断，不是 G0 主训练域第一选择。

### 4. BEHAVIOR / CALVIN

- BEHAVIOR 在 `README` 中被标记为已支持，且 `examples/Behavior/` 下有训练与评测脚本，但其 train registry 只有 `BEHAVIOR_challenge` mixture，说明仍依赖外部数据/机器人实现环境。
- CALVIN 在 `README` 中被标记为已支持，且 `examples/calvin/` 下有 train/eval 入口，但它仍是操作 benchmark，不是移动操作 benchmark。

结论：两者都可作为后置验证，但不建议拿来做 MoWA 第一个 G0 闭环。

## 四、MoWA 推荐使用顺序

| 顺序 | 数据 / benchmark | 用途 |
|---|---|---|
| 1 | RoboCasa365 | G0 与 P0 第一闭环候选。 |
| 2 | RoboCasa tabletop / GR1 | household manipulation 辅助闭环。 |
| 3 | LIBERO / RoboTwin | action schema、sampler、bridge、sanity。 |
| 4 | DOMINO / VLA-Arena / SimplerEnv | 动态诊断、跨域评测、外推检查。 |
| 5 | BEHAVIOR / CALVIN | 后置高难验证。 |

## 五、`robocasa365` 是否可以提前下载

可以，且值得优先准备，但建议按下面顺序做：

1. 先准备 `robocasa365` 环境和 kitchen assets。
2. 先下载最小闭环数据：`OpenDrawer` 的 `target/human`，用于 G0 schema / temporal profile / leakage / label feasibility。
3. G0 通过后，再扩展到 `robocasa365_target_human_all`。

原因：

- `robocasa365` 已有现成训练 mixture、评测接口和下载脚本，工程摩擦最低。
- 它比 LIBERO 更接近 MoWA 的移动厨房任务链。
- 但在 G0 前直接全量下载 50 个 target/human 任务，会把下载时间、存储和环境配置风险提前放大。

直接证据：

- [download_target_human.sh](/gemini/code/starVLA/examples/Robocasa_365/train_files/download_target_human.sh:1) 提供了全量 `target/human` 下载入口。
- [examples/Robocasa_365/README.md](/gemini/code/starVLA/examples/Robocasa_365/README.md:35) 先用单任务 `OpenDrawer` 演示，再扩展到更多任务。

## 六、MoWA 固定主对比数据配方

P0 / P1-b0 / P1-b1 / P2 diagnostic 的主对比必须使用同一数据域。G0 可以从单任务 smoke 开始，但进入主对比时必须固定 recipe，避免把数据差异误写成算法差异。

当前固定 recipe：

| 字段 | 值 |
|---|---|
| recipe_name | `mowa_robocasa365_target_human_atomic_core_v1` |
| split | `target` |
| source | `human` |
| task_type | `atomic` |
| 用途 | MoWA P0/P1/P2 主对比数据域 |
| G0 状态 | 下载可用性检查中；profile / label / leakage 仍为 Data Gate |

任务清单均来自 RoboCasa365 registry 中存在 `target/human` 路径的 atomic 任务：

| task | relative_path |
|---|---|
| OpenDrawer | `v1.0/target/atomic/OpenDrawer/20250816/lerobot` |
| OpenCabinet | `v1.0/target/atomic/OpenCabinet/20250813/lerobot` |
| CloseFridge | `v1.0/target/atomic/CloseFridge/20250816/lerobot` |
| CloseToasterOvenDoor | `v1.0/target/atomic/CloseToasterOvenDoor/20250818/lerobot` |
| CoffeeSetupMug | `v1.0/target/atomic/CoffeeSetupMug/20250813/lerobot` |
| NavigateKitchen | `v1.0/target/atomic/NavigateKitchen/20250821/lerobot` |
| PickPlaceCounterToCabinet | `v1.0/target/atomic/PickPlaceCounterToCabinet/20250811/lerobot` |
| PickPlaceToasterToCounter | `v1.0/target/atomic/PickPlaceToasterToCounter/20250817/lerobot` |
| PickPlaceSinkToCounter | `v1.0/target/atomic/PickPlaceSinkToCounter/20250813/lerobot` |
| TurnOnSinkFaucet | `v1.0/target/atomic/TurnOnSinkFaucet/20250812/lerobot` |

注意：`CloseDrawer`、`CloseCabinet`、`OpenFridge` 当前 RoboCasa365 registry 中没有 `target/human` 路径，不能放入该固定主对比 recipe。若后续使用它们，只能作为其他 split/source 的额外实验或待确认项，不能混入主对比。

### 6.1 10 项子集选择依据

该 recipe 是 MoWA 详细设计的执行补充，不改写第 1 章 Source-of-Truth。选择 10 个子集时采用以下约束：

1. 数据域固定：只选 `target/human/atomic`，用于 P0 / P1-b0 / P1-b1 / P2 diagnostic 的同域主对比，避免把 split、source 或 task_type 差异误写成算法差异。
2. Registry 可验证：每个任务必须在 RoboCasa365 registry 中存在明确 `target/human` lerobot 路径，并能被 `g0_recipe_smoke.py` 做目录、meta、data、videos 四项可用性检查。
3. 能力覆盖：任务组合覆盖厨房移动操作的基础原语，包括开关容器、导航接近、拿取放置、水槽/设备交互和简单准备类任务。
4. 工程规模受控：10 项作为 OpenDrawer 单任务 smoke 与全量 `target_human_all` 之间的核心子集，先支持 G0 / P0 / P1 的 profile、label coverage、leakage 和 latent manifest 闭环。
5. 排除规则明确：缺少 `target/human` 路径的任务不得混入该 recipe；若后续使用其他 split/source 或 composite 任务，必须作为额外数据域记录，不能替代主对比 recipe。

当前选择不是 RoboCasa365 官方 benchmark 划分，而是 MoWA 在三个月工程约束下的固定主对比数据配方。若后续 G0 发现某项任务字段不足、下载不可用或 label / latent / leakage 不通过，应记录为 Data Gate 风险，并通过新 recipe 版本处理，不在原 recipe 内静默替换。

下载完成后用以下命令检查 recipe 是否齐全：

```bash
.venv/bin/python tools/mowa/g0_recipe_smoke.py \
  --data-root playground/Datasets/robocasa365 \
  --output docs_zh/mowa/mowa_g0_robocasa365_atomic_core_recipe_smoke.json
```

## 七、MoWA 执行建议

下一步建议直接做两件事：

1. 在 G0 中把 `robocasa365` 设为 `PrimaryCandidate`，把 `LIBERO` 和 `RoboTwin` 设为 `AuxiliarySanity`。
2. 若要实际下载，先执行 `mowa_robocasa365_target_human_atomic_core_v1`，并用 `g0_recipe_smoke.py` 检查 10 个任务是否齐全。
