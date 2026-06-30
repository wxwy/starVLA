# MoWA 会话状态

## 当前阶段
- M1 G0 Data Verification Gate 数据准备阶段。

## 数据状态
- `.robocase` 环境已就绪：`/gemini/code/starVLA/.robocase`。
- `robocasa365` 数据集软链已建立：`playground/Datasets/robocasa365 -> /gemini/code/datasets/robocasa365`。
- 核心配方 `mowa_robocasa365_target_human_atomic_core_v1` 当前 10/10 任务可用：
  - OpenDrawer、OpenCabinet、CloseFridge、CloseToasterOvenDoor、CoffeeSetupMug、NavigateKitchen、PickPlaceCounterToCabinet、PickPlaceToasterToCounter、PickPlaceSinkToCounter、TurnOnSinkFaucet。
  - recipe availability smoke 已通过目录、meta、data、videos 四项可用性检查。
  - 10 项批量 profile / P0 label coverage / latent manifest smoke 已通过。
  - 10 项 P0 ConstructibleHeads label builder dry-run 已通过；当前仅生成 `task_progress` 和 `action_outcome_class` smoke targets，其他 head 继续 mask。
  - 10 项 batch 级 dataloader smoke 已通过：30 个 sampled windows，future action 未进入 inputs，ConstructibleHeads targets/masks 可随 window 对齐。
  - 10 项 metadata 级 leakage gate 已通过：5055 个 episode、15165 个 anchor windows，future_action / cross_episode leakage 失败数 0。
  - 10 项 temporal profile 已通过：5055 个 parquet、1342150 行，timestamp/frame_index 单调，state/action shape 为 `(16,)` / `(12,)`；WAM Hz/window 仍不冻结。
  - 真实 label builder 阈值 / class mapping、生产 dataloader workers、真实 latent cache 仍为 Data Gate。

## 进行中的任务
- 完整 `target/human` / `mimicgen` 下载进程仍在进行中；当前下载进程 PID 11968。核心 10 项 recipe 已齐，不再阻塞下一步 G0 smoke。

## 下一步
- 在不放开其他五类 head 的前提下，准备 P0 ConstructibleHeads 最小训练入口。
- 训练入口必须显式引用 G0 temporal profile，但不得把 WAM Hz/window 当作已冻结结论。
- 真实 Wan latent cache builder 仍需单独放行。
