# E003 B1 / E003-v2 训练审计 Checklist

> 审计时间：2026-07-22 CST
> 审计对象：`MoWA-E-003-v2_WanPI-LIBERO_contft32_lora-r32_20260721_2150`
> 证据等级：**已确认**＝本地运行产物/日志直接证明；**不通过**＝发现明确不一致；**待确认**＝当前产物不足以证明；**不适用**＝本实验设计未启用。
> 审计范围：仅审计已有 run，不以当前工作区的 YAML 覆盖运行时保存配置。

## 审计结论

- **实验状态：不通过（不可作为正式 E003 B1 对照结果）**。
- 不通过原因：训练未完成（设计 100000 steps，本地日志仅至 2350，最后完整 checkpoint 为 2000），且没有闭环评估结果；因此不构成正式可比实验结果。
- **更正（代码审计，2026-07-22）**：`wan2.2_h10_f8_train.parquet` 是最大容量 manifest，不代表实际使用 H10。`slice_mowa_window_manifest_entry(..., history_steps=0)` 会将 history indices/masks 裁成空元组；旧 run 的 data-flow 日志输入总时间维 `T=9 = 0 history + 1 current + 8 future`，已直接证明实际为 H0。
- 训练已不在运行：本地日志最后记录到 step 2350（2026-07-22 00:15:19），没有结束、异常或人工停止原因；最后完整 checkpoint 为 `steps_2000`。
- step 1000 与 2000 的 LoRA 权重均已变化（480/480 tensors，最大绝对差 `0.005859375`），说明 LoRA 确实进入训练；但冻结 base 权重未被保存，无法对“base 未变”做权重级复核。
- 未找到针对该 run 的 LIBERO/RoboCasa 闭环评估产物；不能报告主成功率。

## 0. 实验身份与可追溯性

- [x] **已确认**实验名、run id、训练记录与输出目录：`MoWA-E-003-v2_WanPI-LIBERO_contft32_lora-r32_20260721_2150`；记录：[原训练记录](MoWA-E-003-v2_WanPI-LIBERO_contft32_lora-r32_20260721_2150.md)，输出：`playground/mowa_ckpt/MoWA-E-003-v2_WanPI-LIBERO_contft32_lora-r32_20260721_2150/`。
- [x] **已确认**实际启动入口：`starVLA/training/train_starvla.py --config_yaml configs/mowa/mowa_e003_v2_wanpi_contft32_lora.yaml --validate-data-flow --validation-steps 2`（W&B metadata `args`）。
- [x] **已确认**运行时完整配置：输出根目录 `config.full.yaml`；两个 checkpoint 亦各自保存 `config.yaml`/`config.full.yaml`。
- [x] **已确认**运行时代码提交：W&B metadata 为 `093fc65f4d9039f9e654728d95488eac82bde3d5`。
- [x] **已确认**审计时工作区提交：`ecfc5f28bfbfd7b5a2b970ede97c812e09b8e2cd`，分支 `merge-official-starvla-dev`；与运行时提交不同，不能以当前代码替代运行时代码复现。
- [x] **已确认**审计时工作区非干净：`configs/mowa/mowa_e003_v2_wanpi_contft32_lora.yaml` 有未提交修改；当前 YAML run id 已为 `…20260722_0024`，不是本 run 的实际配置。
- [x] **已确认**W&B 本地离线同步包：`wandb/run-20260721_224115-.../`，项目/entity=`MoWA`/`silencewx-harbin-institute-of-technology`，模式 `online`。
- [ ] **待确认**W&B 云端 URL/run id：本地 metadata 未保存可直接核验的云端 URL；应在 W&B 页面补填链接并核对最终 step。
- [x] **已确认**Python/硬件：CPython 3.11.9，CUDA 13.0，8× NVIDIA RTX 4090 24 GiB，主机 `bitahub-a20457231046275072812100`。
- [x] **已确认**关键依赖：torch 2.6.0、accelerate 1.5.2、diffusers 0.38.0、transformers 4.57.0、peft 0.17.0、wandb 0.27.2。
- [x] **已确认**随机种子：42（运行时 `config.full.yaml`）。
- [ ] **待确认**deterministic 开关及 cuDNN/TF32 状态：运行时配置和日志均无证据。
- [x] **已确认**非续训：`is_resume=false`，全局训练从 0 开始；加载的是预训练 backbone `steps_60000_pytorch_model.pt`，而不是 trainer/optimizer 状态。
- [ ] **待确认**运行期间是否有配置/数据替换：运行时产物仅能证明保存时的配置，无法证明训练期间文件未被外部修改。
- [ ] **不通过**checkpoint、日志、W&B 最终步数不能完全闭环：日志到 2350，最后完整 checkpoint/summary 到 2000，W&B 云端最终值未核验。

## 1. B1 设计定义

- [x] **已确认**已实现的设定：WanPI + contFT32（action horizon=32）+ Wan LoRA r=32 + 双视角 future latent prior；详情见运行时 `config.full.yaml`。
- [x] **已确认**可识别的训练变量：单任务 `robocasa365_open_drawer_target_human`、预训练 backbone step 60000、LoRA r=32/alpha=64、future-prior scale=0.05、双视角 main/wrist、action head/LoRA 训练且 backbone 冻结。
- [ ] **待确认**B1 的书面假设、唯一自变量、明确对照组、预注册成功阈值：原训练记录未给出可审计定义。
- [ ] **待确认**除 B1 自变量外是否与基线完全一致：缺少指定基线的冻结配置及差异清单。
- [x] **已确认**H0 设计与实际一致：虽复用 H10 容量 manifest，但加载器按 `history_window_steps=0` 裁切前序 history；data-flow 实测 `T=9=0+1+8`。

## 2. 启动命令与运行时参数一致性

- [x] **已确认**训练脚本为预期正式入口，且启动时运行了 2/2 data-flow contract validation。
- [x] **已确认**配置路径、输出目录、run id：均由 W&B metadata 和运行时配置记录。
- [x] **已确认**输出目录独占本 run；未覆盖其他 run 目录。
- [x] **已确认**从 scratch：`is_resume=false`；日志的 `resume_from_checkpoint`/`pretrained_checkpoint` 均指向 base 初始化权重，不是 optimizer/scheduler resume。
- [x] **已确认**total steps=100000、每卡 batch=2、GA=2、8 rank、global batch=32；训练日志直接打印。
- [x] **已确认**运行时最终 YAML 与 checkpoint 保存配置一致（`steps_1000` 与 `steps_2000`）。
- [ ] **待确认**命令行覆盖优先级：只发现启动参数 `--validate-data-flow --validation-steps 2`，未见完整 shell 启动脚本/环境变量记录。
- [ ] **待确认**启动 cwd/相对路径解析：路径均为相对路径，但未保存 cwd 打印。
- [ ] **不通过**当前正式 YAML 不可视为本 run 的实际 YAML：run id 和保存间隔已变化（当前为 `…20260722_0024`、save interval 500；运行时为 `…20260721_2150`、1000）。

## 3. 模型与可训练参数

- [x] **已确认**base model：Wan2.2-TI2V-5B-Diffusers；初始化 checkpoint：`/disk/rl/models/mowa_init_candidates/WM4A-Wan2d2-OFT-LIBERO-4in1/checkpoints/steps_60000_pytorch_model.pt`；`reload_modules=backbone`。
- [x] **已确认**架构：WanPI、LayerwiseFM action DiT（30 层，hidden 1024，16 heads）、action dim=12、state dim=32、horizon=32、双视角 main/wrist、cross-view gate init=1.0、done head weight=0.1。
- [x] **已确认**LoRA 已注入并保存：`wan_lora_config.json` 显示 0--29 层 self/cross attention 的 Q/K/V/out target modules；rank=32、alpha=64、dropout=0。
- [x] **已确认**LoRA checkpoint 完整：`wan_lora.safetensors` 含 480 tensors、47,185,920 elements（94,371,840 bytes），配套 `wan_lora_config.json` 存在。
- [x] **已确认**LoRA 发生更新：steps 1000→2000 共有 480 tensors，480/480 不相等，最大绝对差 0.005859375；日志 step 2350 仍有 LoRA `grad_norm=0.0249`、`delta_norm=4.5808`。
- [x] **已确认**backbone 配置为冻结：`freeze_modules=backbone`；checkpoint `trainer_state.json` 明确 `save_frozen_backbone=false`、省略 `backbone.`。
- [ ] **待确认**requires_grad=True 的完整参数名、参数量、占总参数比例：训练日志未保存清单。
- [ ] **待确认**动作 head、cross-view、norm/embedding 是否均进入 optimizer：日志显示 cross-view/LoRA 梯度非零，但无 optimizer 参数组全量清单。
- [ ] **待确认**冻结 base 权重前后逐张量一致：checkpoint 有意省略 base，无法做权重级比较。
- [ ] **待确认**DDP 未遗漏其他可训练参数：首个 220911 run 曾因未使用参数报错；224115 run 可训练至 2350，但仍缺完整参数/optimizer 清单。

## 4. 数据集、任务与采样

- [x] **已确认**实际 data mix：`robocasa365_open_drawer_target_human`；日志采样指标显示 `OpenDrawer` target/observed probability=1.0。
- [x] **已确认**数据根与缓存根：`playground/Datasets/robocasa365`、`playground/Datasets/robocasa365_wan2.2_latent/v1.0/target/atomic`。
- [x] **已确认**双视角字段：`observation.images.robot0_agentview_left`、`observation.images.robot0_eye_in_hand`；data-flow validation 通过的输入为 B=2、V=2、C=48、T=9、H=W=16。
- [x] **已确认**action/state：`delta_ee` 12D、连续 state 32D、action chunk=32、future window=8，运行时数据统计文件已随 run/checkpoint 保存。
- [x] **已确认**采样日志健康：单任务窗口和累计 observed-to-target ratio 均为 1.0；未见意外空任务。
- [x] **已确认**后 padding 已被记录：step 2350 窗口 back_pad=13.1875%，front/both=0；action/future valid ratio 分别约 0.9239/0.9295。
- [ ] **待确认**原训练记录所称 transitions 数（24,424 与 34,424 两种数值）哪个准确：运行时日志未保存样本/trajectory 总数，记录内部自相矛盾。
- [ ] **待确认**train/val/test trajectory、episode、task 无泄漏：无 split manifest 或哈希审计产物。
- [ ] **待确认**instruction 与 trajectory 对齐、action normalization 来源、dataloader shuffle/drop_last/worker seed：无专项证据。
- [ ] **待确认**训练/验证增强隔离：无验证数据流或增强开关日志。
- [x] **已确认**时序输入为 H0：`window_manifest.py:slice_mowa_window_manifest_entry()` 对 history 使用 `_tail(..., 0)`，返回空元组；WanPI data-flow 实测 `T=9=0 history+1 current+8 future`。H10 manifest 仅提供可裁切的最大窗口容量。
- [ ] **待确认**控制频率、action horizon 与后续环境评估一致：本 run 无评估产物。

## 5. 优化器、学习率与训练步数

- [x] **已确认**optimizer=AdamW，betas=(0.9,0.95)，eps=1e-8，weight_decay=1e-8。
- [x] **已确认**LR：base=2.5e-5、wan_lora=1e-5、action_model=1e-4；scheduler=`cosine_with_min_lr`，min_lr=1e-6，warmup=2000，max steps=100000。
- [x] **已确认**mixed precision=bf16；gradient clipping/max_grad_norm=1.0。
- [x] **已确认**global batch 推导正确：2（per-device）×8（world size）×2（GA）=32，与日志 `Total batch size=32` 一致。
- [x] **已确认**warmup LR 正常上升：step 100 训练记录为 base 1.25e-6 / LoRA 5e-7 / action 5e-6；step 2350 分别约 2.49994e-5 / 9.99978e-6 / 9.99978e-5，符合 warmup 已结束后的峰值附近预期。
- [x] **已确认**optimizer step 计数与日志一致至 2350；checkpoint 目录与 trainer state 分别一致于 1000/2000。
- [ ] **待确认**参数分组中 norm/bias 与动作头的 weight decay 策略：无实际 optimizer groups dump。
- [ ] **待确认**全程 LR 曲线、loss scaling/overflow：本地日志只有窗口点且 W&B 云端未核验。
- [ ] **不通过**设计总步数与已完成训练步数不一致：设计 100000，现有日志止于 2350（2.35%），不是完成训练结果。

## 6. 训练过程健康度

- [x] **已确认**第二次实际 run（224115）成功完成模型、数据、optimizer、scheduler、W&B 初始化并通过 data-flow validation 2/2。
- [x] **已确认**截至本地最后日志 step 2350，未出现 NaN、Inf、OOM、CUDA error、overflow 或 dataloader worker crash 文本。
- [x] **已确认**loss 在有限值范围：step 50 `action=1.0341/future=3.2514`；step 2350 `action=0.0839/future=0.9983/total=0.1538`，早期总体下降但非单调。
- [x] **已确认**组件 loss 与权重均已记录：action、future main/wrist、done、weighted future、total、aux/action ratio、padding 分桶、按任务损失。
- [x] **已确认**梯度非零：step 2350 cross-view grad norm=0.1946，LoRA grad norm=0.0249。
- [x] **已确认**吞吐无明显数据瓶颈：step 2350 data/model time≈0.00095/1.32147 s；原训练记录约 2.33 s/optimizer step。
- [ ] **待确认**2350 后训练为何停止：日志无完成、traceback、SIGTERM 或人工终止记录；当前没有对应进程。
- [ ] **待确认**多 rank 完整健康度：8 rank 能训练至 2350，但无各 rank 心跳/利用率/聚合审计。
- [ ] **待确认**checkpoint 保存是否不阻塞：仅观察到 step 1000/2000 保存成功。

## 7. W&B 指标核查

- [x] **已确认**W&B run 名可唯一定位，项目/entity 和本地同步目录明确；启动时间 2026-07-21 22:41 CST。
- [x] **已确认**本地 console 上报了 global step、loss、各子 loss、learning rate、epoch、LoRA/cross-view/采样/padding/时序指标。
- [x] **已确认**本地步数单调递增至 2350，window 日志与本地输出相符。
- [ ] **待确认**云端 W&B config 是否来自最终命令行覆盖，而非默认值：需打开云端 run 对比 `config.full.yaml`。
- [ ] **待确认**云端曲线连续性、GPU 系统指标、最终 step/loss/lr、是否多进程重复上报：本地 bundle不足以完整证明。
- [ ] **待确认**W&B URL/run id：应补填；本审计不猜测 URL。

## 8. Checkpoint 完整性与可恢复性

- [x] **已确认**checkpoint `steps_1000`、`steps_2000` 均存在，目录名与 `trainer_state.json.completed_steps` 一致。
- [x] **已确认**每个 checkpoint 包含模型 safetensors shard/index、8 个 rank-sharded optimizer state、scheduler、8 个 RNG state、scaler、配置、dataset statistics、LoRA adapter 与配置。
- [x] **已确认**step 2000 的主要文件大小合理：model shard 2,107,004,596 B；每个 optimizer rank state 4,197,654,366 B；LoRA 94,423,416 B。
- [x] **已确认**step 2000 LoRA 文件 SHA-256：`304ab102759877ca68514bfe5b0fb3287fbae9f789458069cfa8a0806435d77e`。
- [x] **已确认**轻量 checkpoint 的设计语义明确：冻结 backbone 不保存，LoRA 单独保存，同时保留 rank-sharded optimizer/scheduler/RNG 用于同 world size 恢复。
- [ ] **待确认**任一 checkpoint 实际推理加载：未执行加载 smoke，避免占用正在共享的 GPU 资源。
- [ ] **待确认**从 `steps_2000` 恢复训练后的 step、LR、loss 连续性：未执行 resume smoke。
- [ ] **待确认**动作 head/cross-view 权重未遗漏：model index 存在，但未做模型加载后键覆盖验证。
- [ ] **不通过**“最终 checkpoint”不存在：日志到 2350 而最后完整保存点为 2000；未完成 100000 步，也无 best-checkpoint 选择依据。

## 9. 验证与 LIBERO/RoboCasa 评估

- [ ] **待确认**评估 checkpoint、环境版本、任务集、episode 数、seed、图像预处理、语言模板、反归一化：本 run 目录和训练记录未找到评估报告。
- [ ] **待确认**model.eval()/dropout/随机增强关闭、每任务 success rate、总体均值/方差、失败分类：无评估产物。
- [ ] **待确认**与基线使用同协议：未指定基线评估记录。
- [ ] **不通过**主结果缺失：不能以训练 loss 代替 LIBERO/RoboCasa 闭环 success rate。

## 10. “设计与实际不一致”专项判定

- [x] **已确认一致**：H0 配置与 H10 容量 manifest 兼容；代码按 `history_window_steps=0` 裁切，实际输入已由 `T=9=0+1+8` 验证。
- [x] **不一致，阻断完整训练结论**：设计 100000 steps，实有日志仅至 2350，最后 checkpoint=2000。
- [x] **不一致，阻断当前 YAML 追溯**：当前 YAML run id/保存间隔与运行时保存 YAML 不同；只能使用 run 目录内配置审计。
- [x] **已确认一致**：LoRA r=32、alpha=64、dropout=0、self/cross-attention targets 与实际 adapter config 一致。
- [x] **已确认一致**：从 scratch（不恢复 optimizer/scheduler），global batch=32，warmup/学习率窗口值合理，冻结 backbone 配置与 checkpoint 省略策略一致。
- [ ] **待确认**backbone 权重未实际更新、动作头/MoWA 全部入 optimizer、W&B config 最终覆盖、评估协议：现有证据不够。

## 最终结论模板（本次填写）

- 实验状态：**不通过**。
- 代码版本：运行时 `093fc65f4d9039f9e654728d95488eac82bde3d5`；审计时 `ecfc5f28bfbfd7b5a2b970ede97c812e09b8e2cd`（不同）。
- 数据版本：RoboCasa365 v1.0 target/atomic Wan2.2 latent cache；复用容量 manifest `wan2.2_h10_f8_train.parquet`，运行时按 H0 裁切，已验证一致。
- 启动命令：见本文件“0. 实验身份与可追溯性”。
- 实际训练范围：8×RTX4090、bf16、per-device bs2、GA2、global batch32；日志至 step2350，完整 checkpoint 至 step2000。
- 可训练参数：backbone 配置冻结；LoRA 已实际更新（47,185,920 elements）；完整可训练参数清单缺失。
- 总优化步数与实际结束步数：设计100000；无正常结束记录，最后日志2350。
- 学习率策略核对：warmup2000 + cosine min 1e-6；本地窗口值一致，完整曲线待 W&B 核验。
- W&B：`MoWA` / `silencewx-harbin-institute-of-technology`，本地 run `run-20260721_224115-...`；云端 URL 待补。
- checkpoint 完整性：steps1000/2000 的训练状态文件齐全；尚未实际加载验证；无最终/best checkpoint。
- LIBERO 评估协议与主结果：未执行/未记录。

### 已确认一致项

- WanPI、contFT32、双视角、future-prior 开关与 scale、LoRA r32/alpha64/targets、base 初始化、batch/GA/world size、bf16、optimizer 和 warmup 配置。
- LoRA adapter 已保存且在 1000→2000 实际更新；checkpoint 训练状态组件存在。

### 发现的不一致项

- 当前 YAML 与该 run 的运行时 YAML 不同。
- 计划100000步但训练记录在2350步中断，最后 checkpoint 为2000。

### 对结果可信度的影响

- 不能把该 run 的 loss 或 checkpoint 作为正式可比结果；阻断原因是训练完成度、最终 checkpoint 和闭环评估缺失，不是历史窗口语义。

### 是否可与基线直接比较

- **不可以**：训练完成度、最终 checkpoint 和评估协议均未形成完整一致证据。

### 后续动作

1. 新 run 可继续复用 H10 容量 manifest；启动后保留 `T=9=0+1+8` 的 data-flow 日志作为 H0 证据。
2. 用运行时 commit 与 `config.full.yaml` 建立不可变启动包，保存完整 shell 命令、deterministic/TF32 开关、requires-grad/optimizer 参数清单。
3. 训练达到预设终点后，以固定 seed、每任务 episode 数和同基线协议完成 RoboCasa/LIBERO 闭环评估，并补 W&B 云端链接。
