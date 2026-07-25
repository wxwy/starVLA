# E003-v2 TurnOnElectricKettle 训练前 Checklist

> 创建时间：2026-07-22 CST
> run id：`MoWA-E-003-v2_WanPI-TurnOnElectricKettle_contft32_lora-r32_20260722_1326`
> 配置来源：`configs/mowa/mowa_e003_v2_wanpi_contft32_lora_TurnOnElectricKettle.yaml`
> 状态标记：**训练已启动**，record as we go。

## 启动前结论

- **状态：训练运行中**（启动于 13:26 CST，当前 step ~1250）。
- 新 run 使用 `is_resume=false`，从预训练 backbone `steps_60000_pytorch_model.pt` 初始化。
- 单任务 `robocasa365_turn_on_electric_kettle_target_human`，与全 18 任务 E003-v2 B3 为同配方单任务对照。

## 0. 实验身份与可追溯性

- [x] **已确认**run id：`MoWA-E-003-v2_WanPI-TurnOnElectricKettle_contft32_lora-r32_20260722_1326`。
- [x] **已确认**输出根：`playground/mowa_ckpt`；目录 `playground/mowa_ckpt/<run_id>` 已创建且包含 checkpoints/configs。
- [x] **已确认**seed=42，W&B=`silencewx-harbin-institute-of-technology`/`MoWA`/`online`。
- [x] **启动后已填**记录完整 shell 命令、cwd、git SHA、Python/PyTorch/CUDA 版本、W&B URL。
  - **Shell**: `accelerate launch --num_processes 8 --main_process_port 29501 starVLA/training/train_starvla.py --config_yaml configs/mowa/mowa_e003_v2_wanpi_contft32_lora_TurnOnElectricKettle.yaml --validate-data-flow --validation-steps 2`
  - **CWD**: `/disk/rl/starVLA`
  - **Git SHA**: `ecfc5f2` (HEAD: docs: add MoWA E-003-v2 training log)
  - **Python/PyTorch/CUDA**: Python 3.11, PyTorch 2.6.0+cu124, CUDA 12.4
  - **GPU**: 8 × NVIDIA GeForce RTX 4090 (24GB)

## 1. 设计与数据

- [x] **已确认**框架：WanPI、contFT32、action dim=12、state dim=32、action horizon=32、future window=8、双视角 main/wrist。
- [x] **已确认**H0：history_window_steps=0；T=9（0 history + 1 current + 8 future）。
- [x] **启动后已填**data mix：`robocasa365_turn_on_electric_kettle_target_human`（单任务）。
- [x] **启动后已填**data-flow validation 2/2 通过，`T=9` 确认。
- [ ] **待填**逐任务分布、样本数、trajectory 数的运行时日志。

## 2. 模型与优化

- [x] **已确认**base：Wan2.2-TI2V-5B-Diffusers；pretrained=`steps_60000_pytorch_model.pt`；`reload_modules=backbone`。
- [x] **已确认**LoRA：enabled，r=32，alpha=64，dropout=0，target groups=`cross_attention,self_attention`。
- [x] **已确认**`freeze_modules=backbone`；训练项包括 action model + Wan LoRA + cross_view + done_head。
- [x] **已确认**LR: base/LoRA/action=`2.5e-5/1e-5/1e-4`；cosine_with_min_lr，min lr=1e-6，warmup=2000。
- [x] **已确认**bf16、gradient clipping=1.0；per-device batch=2、8 rank、GA=2，global batch=32。
- [ ] **待填**记录 requires-grad 参数名、元素数、可训练比例。
- [ ] **待填**记录首次 optimizer step 的 LoRA/动作头梯度和权重变化。

## 3. 过程、checkpoint 与评估门禁

- [x] **已确认**max_train_steps=100000，save interval=500，eval interval=250，logging frequency=50，`is_resume=false`。
- [ ] **跟踪中**global step 单调递增；warmup/衰减 LR 曲线与 scheduler 一致；无 NaN/Inf/OOM。
- [ ] **跟踪中**记录 action/future-main/future-wrist/done/total loss、grad norm、吞吐、GPU 利用率。
- [ ] **待填**在 step500、step1000 等 checkpoint 核查模型完整性。
- [ ] **待填**加载中间 checkpoint 做 inference smoke。
- [ ] **训练结束后必填**做 RoboCasa/LIBERO eval，记录 success 率。

## 启动后首个核对点

1. ✅ 目录 `.../TurnOnElectricKettle_...1326/` 已创建，config 已落盘，checkpoints 已开始保存（steps_500, steps_1000）。
2. ✅ `is_resume=false`，global step 从 0 开始。
3. ✅ `data_mix=robocasa365_turn_on_electric_kettle_target_human`。
4. ✅ H0 data-flow validation 通过（T=9）。
5. ✅ Step 50: `action_dit_loss=0.885`，loss 下降正常。
6. ✅ Step 500: `action_dit_loss=0.058`，eval mse=0.0085。
