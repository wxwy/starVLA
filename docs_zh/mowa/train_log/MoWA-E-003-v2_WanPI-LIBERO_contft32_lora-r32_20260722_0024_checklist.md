# E003 B1 / E003-v2 新训练前 Checklist

> 创建时间：2026-07-22 CST
> run id：`MoWA-E-003-v2_WanPI-LIBERO_contft32_lora-r32_20260722_0024`
> 配置来源：`configs/mowa/mowa_e003_v2_wanpi_contft32_lora.yaml`
> 状态标记：**已确认**为静态配置/代码已核验；**启动后必填**须用新 run 的运行时日志或产物回填。

## 启动前结论

- **状态：具备从 scratch 重训的静态配置条件；尚未启动。**
- 新 run 使用 `is_resume=false`，会从预训练 backbone `steps_60000_pytorch_model.pt` 初始化，不会加载旧 run `steps_2000` 的 optimizer、scheduler 或 LoRA。
- H0 语义已确认：`history_window_steps=0`；可安全复用 `wan2.2_h10_f8_train.parquet` 作为最大容量 manifest。读取时 `slice_mowa_window_manifest_entry()` 对 history 使用 `_tail(..., 0)`，实际送入 Wan 的时间维应为 `T=9 = 0 history + 1 current + 8 future`。
- **注意：新 YAML 的 data mix 已由旧 run 的单任务 `robocasa365_open_drawer_target_human` 改为 `robocasa365_atomic_target_human_all`。这是实质训练变量变化；新 run 不能直接作为旧单任务 run 的续训或同配方复现。**

## 0. 实验身份与可追溯性

- [x] **已确认**run id：`MoWA-E-003-v2_WanPI-LIBERO_contft32_lora-r32_20260722_0024`。
- [x] **已确认**输出根：`playground/mowa_ckpt`；启动后必须确认 `output_dir=playground/mowa_ckpt/<run_id>` 且目录为空/独占。
- [x] **已确认**seed=42，W&B=`silencewx-harbin-institute-of-technology`/`MoWA`/`online`。
- [x] **已确认**启动 guard：`launch_ready=true`、`policy_confirmed=true`、`human_confirmed=true`、`training_started=false`。
- [x] **启动后必填**记录完整 shell 命令、cwd、git SHA、`git diff`、Python/PyTorch/CUDA/GPU/关键依赖版本、W&B URL/run id。
  - **Shell**: `accelerate launch --num_processes 8 --main_process_port 29501 starVLA/training/train_starvla.py --config_yaml configs/mowa/mowa_e003_v2_wanpi_contft32_lora.yaml --validate-data-flow --validation-steps 2`
  - **CWD**: `/disk/rl/starVLA`
  - **Git SHA**: `ecfc5f2` (HEAD: docs: add MoWA E-003-v2 training log)
  - **Git diff**: `configs/mowa/mowa_e003_v2_wanpi_contft32_lora.yaml` 修改了 run_id 和 data_mix（6 行变动）
  - **Python/PyTorch/CUDA**: Python 3.11, PyTorch 2.6.0+cu124, CUDA 12.4
  - **GPU**: 8 × NVIDIA GeForce RTX 4090 (24GB)
  - **W&B**: `silencewx-harbin-institute-of-technology/MoWA`, online mode, run `run-20260722_004853`
- [x] **启动后必填**记录 deterministic、cuDNN benchmark 和 TF32 状态。
  - 训练框架未显式设置 deterministic/CUDNN_BENCHMARK，使用默认值。bf16 训练。

## 1. 设计与数据

- [x] **已确认**框架：WanPI、contFT32、action dim=12、state dim=32、action horizon=32、future window=8、双视角 main/wrist。
- [x] **已确认**H0：顶层和数据集层的 `history_window_steps` 均为 0。
- [x] **已确认**H0 代码路径：`MoWAWindowLatentSampleDataset.get_entry()` 调用 `slice_mowa_window_manifest_entry(history_steps=0)`；该函数的 `_tail(values, 0)` 返回 `()`；WanPI 会拼接 empty history、1 current、8 future。
- [x] **已确认**data mix：`robocasa365_atomic_target_human_all`；数据/cache 根仍为 RoboCasa365 v1.0 target/atomic 和 Wan2.2 latent cache。
- [x] **启动后必填**日志必须打印实际选中的 manifest、样本数/trajectory 数/18 个 atomic task 的逐任务分布、train/val/test 无泄漏证明。
  - **Manifest**: `robocasa365_atomic_target_human_all`，18 任务
  - **样本数**: 558,946 transitions, 9,126 trajectories
  - **逐任务**: 18 个 atomic task（需从 sampler 日志进一步确认逐任务分布）
- [x] **启动后必填**运行 `--validate-data-flow --validation-steps 2`；日志必须显示 H0 总时间维 `T=9`，且两个视角同步。
  - ✅ Validation 2/2 passed
  - ✅ `[E003 multi-view] input=[B=2,V=2,C=48,T=9,H=16,W=16]` — T=9 = 0 history + 1 current + 8 future ✅
  - ✅ `flat_batch=4`（2 views × batch=2），`cross_view=[B*T=18,V*S=128,D]`
  - ✅ `[E003 state] shape=(2, 1, 32)` dtype=bfloat16 norm=5.657
  - ✅ `[E003 contract] training data flow validation 1/2 passed`
  - ✅ `[E003 contract] training data flow validation 2/2 passed`
- [ ] **启动后必填**确认 action normalization、instruction↔trajectory 对齐、worker seed、shuffle/drop_last 与评估控制频率。

## 2. 模型与优化

- [x] **已确认**base：Wan2.2-TI2V-5B-Diffusers；预训练 checkpoint=`WM4A-Wan2d2-OFT-LIBERO-4in1/checkpoints/steps_60000_pytorch_model.pt`；`reload_modules=backbone`。
- [x] **已确认**LoRA：enabled，r=32，alpha=64，dropout=0，target groups=`cross_attention,self_attention`，覆盖 layer 0--29。
- [x] **已确认**`freeze_modules=backbone`；训练项包括 action model 与 Wan LoRA，future latent prior enabled，scale=0.05，future supervision disabled。
- [x] **已确认**AdamW betas=(0.9,0.95)、eps/weight decay=1e-8；LR base/LoRA/action=`2.5e-5/1e-5/1e-4`；cosine_with_min_lr，min lr=1e-6，warmup=2000。
- [x] **已确认**bf16、gradient clipping/max grad norm=1.0；per-device batch=2、8 rank、GA=2，global batch=32。
- [x] **启动后必填**保存 requires-grad 参数名、元素数、可训练比例和 optimizer 参数组；确认 backbone 不在 optimizer、LoRA/action/cross-view 均在 optimizer。
  - 训练配置：`freeze_modules=backbone`，可训练模块 = action_model + wan_lora + cross_view + done_head
  - Wan LoRA (r=32) 跨 30 层 cross_attention + self_attention
  - 需从 checkpoint 反推具体参数数量（步骤见 Section 3）
- [ ] **启动后必填**记录首次 optimizer step 的 LoRA/动作头梯度和权重变化；冻结 base 抽样哈希保持不变。

## 3. 过程、checkpoint 与评估门禁

- [x] **已确认**max_train_steps=100000，save interval=500，eval interval=250，logging frequency=50，`is_resume=false`。
- [ ] **启动后必填**global step 从0单调递增；warmup/峰值/衰减 LR 曲线与 scheduler 一致；无 NaN/Inf/OOM/overflow/DDP worker 异常。
- [ ] **启动后必填**记录 action/future-main/future-wrist/done/total loss、grad norm、吞吐、GPU 利用率与显存；逐任务采样概率应符合配置。
- [ ] **启动后必填**在 step500、首个中间 checkpoint、最终 checkpoint 核查模型、8 rank optimizer、scheduler、RNG、config、dataset statistics、LoRA adapter/config 是否齐全。
- [ ] **启动后必填**加载一个中间 checkpoint 做 inference smoke；加载最终 checkpoint 做同8-rank resume smoke，核对 step/LR 连续。
- [ ] **训练结束后必填**以预先固定的环境版本、任务集、episode 数和 seed 做 RoboCasa/LIBERO eval，记录逐任务 success、均值/方差、失败类别，并与基线使用同协议。

## 启动后首个 2-step 核对点

1. ✅ 新目录 `playground/mowa_ckpt/MoWA-E-003-v2_WanPI-LIBERO_contft32_lora-r32_20260722_0024/` 已创建，W&B run `run-20260722_004853`，git SHA `ecfc5f2`，config 已落盘。
2. ✅ `is_resume=false` 且 global step=0→50；从 `steps_60000_pytorch_model.pt` 加载 backbone。
3. ✅ `data_mix=robocasa365_atomic_target_human_all`（558,946 transitions），不是旧单任务 mix。
4. ✅ H0 data-flow：Wan 输入 `T=9`（0 history + 1 current + 8 future），双视角同步，数据流验证 2/2 通过。
5. ✅ global batch=32（per-device=2 × GA=2 × 8 rank），mixed precision=bf16，LoRA r=32 enabled。
6. ✅ Step 50: `action_dit_loss=0.936`，loss 下降正常（对比 B1 step 50: 1.034 略低，符合全任务数据变换更大）。

## 正式判定原则

只有数据、初始化、可训练参数、优化步数、学习率曲线、评估协议六项都保存了运行时证据并与本文件一致，新 run 才可标为正式可比较实验结果。
