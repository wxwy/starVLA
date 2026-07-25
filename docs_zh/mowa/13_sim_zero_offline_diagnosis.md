# MoWA 仿真 0% 根因定位报告与离线动作一致性检查方法

> 日期：2026-07-19。作者：Kimi（应用户要求落盘）。
> 适用读者：**任何接手 MoWA 训练/评测效果定位的 Agent 或工程师**。
> 一句话结论：E-001/E-003 各 checkpoint 仿真全 0% 的根因是**模型闭环质量不足**，而不是数据/动作/相机/状态/归一化等链路 bug——后者已逐一实证排除。再遇到"训了 N 步仿真还是 0%"，请先按本文第 6 节 SOP 做离线检查，不要直接烧仿真时间或怀疑管线。

---

## 0. 结论速览

| 问题 | 结论 | 关键证据 |
|---|---|---|
| 仿真 0% 是评测管线坏了吗？ | **不是**。数据↔仿真的动作/状态语义完全一致（expert 回放 EE 轨迹吻合到 ~1mm），client↔server↔env 动作布局一致，相机几何一致。 | §2 |
| 是 E-003（future_latent_prior_wo_history）方案特有的失败吗？ | **不是单一方案问题，但也不能跨方案互证**。E-001 baseline（StarFlow ft0，不同架构）离线拟合更好、仿真同样 0%。两个方案各自的失败要用各自证据解释。 | §3 |
| 那是"训练步数不够"吗？ | **部分是，且不全是**。E-003 离线拟合曲线 5k→30k 仍在上升（仅 ~1.2 epoch），继续训练合理；但两个模型共同的短板形态（幅度收缩、eef z 维预测差、语言调制弱）指向 h=0 时序信息瓶颈等结构性因素。 | §3、§7 |
| 下次 0% 怎么办？ | 先跑 `tools/mowa/e003_offline_action_check.py`（1 分钟），再按 §6 SOP 逐项二分。 | §5、§6 |

---

## 1. 问题现象

- `playground/mowa_eval_results/MoWA-E-003_*/steps_*`：9000 → 28500 步全部 checkpoint、全部 robocasa atomic 任务 success_rate=0（每任务仅 1~10 episode）。
- `playground/mowa_ckpt/MoWA-E-001_*/checkpoints/steps_*.eval/`：baseline 70000 步 0/2、50000 步 0/5；E-001-mowa 50000 步 0/5。
- 录像行为：机械臂不是不动，而是在灶台上方无目的漂移（E-003），或下潜到台面以下停住（E-001 baseline），均不完成任务。
- 训练 loss 正常收敛（action_dit_loss 0.305→0.069 缓降，无发散）。

## 2. 验证为"清白"的环节（不要重复排查）

| 环节 | 验证方法 | 证据 | 结论 |
|---|---|---|---|
| 动作维度布局 client↔server↔env | 代码逐级审查 + server meta 校验 | `PolicyNormProcessor` 按训练 `action_keys` 顺序输出；client `ACTION_SLICES` 同序切片；`RoboCasaGymEnv.step`→`unmap_action` 按**键名**消费，再按 part_name 填 robosuite 动作向量 | 一致 ✓ |
| 动作/状态语义（训练↔仿真） | **expert 动作回放**（决定性实验） | 数据集 OpenDrawer ep0 前 80 个 expert 动作逐步在仿真执行，EE 位移与数据集记录吻合到 ~1mm（sim 0.984 vs data 0.981 累计位移） | 完全一致 ✓ |
| 相机视角/几何 | 训练帧 vs 仿真帧并排对比 | 同为机器人肩上视角，机位/FOV 一致；场景差异是 layout 随机化 | 一致 ✓ |
| state 处理 | 代码对照 | 训练 transform 与 client 均为 sin/cos(16d)→32d；server 不二次处理 | 一致 ✓ |
| 动作反归一化 | 统计核对 | DataConfig 全部 action key 用 min_max，而 stats 的 min/max ≈ ±1，unnorm 实为恒等映射，训练/推理同一份 stats | 一致 ✓ |
| Wan 视觉 condition 构造（h=0 语义） | 代码对照 | 训练（cache）与推理（在线 VAE）都是"当前 4 帧 chunk 的 latent、无 history"；`required_frames=1+4*(history_steps+1)=5` | 一致 ✓ |
| 指令文本 | 数据集 tasks.jsonl vs 仿真 annotation | 同一字符串体系（如 "Open the left drawer."） | 一致 ✓ |
| 语言是否生效 | 同画面切换 left/right 指令测输出差 | 输出 RMS 调制：baseline ≈40%，E-003 ≈29% | 有效但偏弱（非 instruction-blind） ✓ |
| flow-matching 推理步数 | `--config_override framework.action_model.num_inference_timesteps=10` 对照 | 4 步 MSE 0.0547 vs 10 步 0.0604，无改善 | 4 步不是瓶颈 ✓ |

## 3. 模型侧的直接测量（核心证据）

工具：`tools/mowa/e003_offline_action_check.py` —— 把训练集真实片段按 eval client 完全相同的方式发给 policy server，逐维对比预测动作与 expert 动作（3 条 OpenDrawer episode × stride 15 × 前 8 步，与闭环 N_ACT=8 对齐；zero 基线 MSE=0.213）。

### 3.1 E-003 离线拟合曲线（学习曲线仍在上升）

| steps | 总 MSE | vs zero | eef_pos corr | eef_rot corr | grip corr |
|---|---|---|---|---|---|
| 5000  | 0.0911 | 0.427 | 0.297 | 0.262 | 0.691 |
| 10000 | 0.0765 | 0.359 | 0.352 | 0.225 | 0.724 |
| 20000 | 0.0532 | 0.250 | 0.458 | 0.407 | 0.822 |
| 30000 | 0.0557 | 0.261 | 0.508 | 0.375 | 0.830 |
| 36000 | 0.0608 | 0.285 | 0.486 | 0.449 | 0.778 |
| 36500 | 0.0506 | 0.237 | 0.535 | 0.394 | 0.839 |
| 37000 | 0.0493 | 0.231 | 0.580 | 0.529 | 0.827 |
| 40000 | 0.0480 | 0.225 | 0.601 | 0.534 | 0.827 |
| 45000 | 0.0536 | 0.251 | 0.581 | 0.547 | 0.833 |
| 46000 | 0.0583 | 0.273 | 0.518 | 0.536 | 0.792 |

注：单 checkpoint 波动约 ±0.005 MSE（54 次预测样本量），应看趋势不看单点——36000 相对 30000 的回落即属噪声。30k→40k 各指标仍在改善（MSE 0.0557→0.0480，pos_corr 0.508→0.601），但 **40k 之后进入平台/噪声区**（45k/46k 未再超过 40k，46k 回落 ~2σ），训练 loss 同期仅从 0.059 缓降至 0.055——"继续同 recipe 加步数"的边际收益已基本消失，40000 为当前最佳 checkpoint（pos_corr≈0.60，与 E-001 baseline 70k 的 0.62 同档）。

### 3.2 E-001 baseline（70k，不同方案，仅作环境公平性参照）

- 总 MSE 0.0293（ratio 0.137），eef_pos corr 0.615，grip corr 0.949——全面好于 E-003@30k。
- 用**修正后的 3 视角 client** 跑仿真 OpenDrawer 10 episode：仍 0/10。
- **注意**：baseline 是不同方案（Qwen-VL / 离散化 state / horizon 8），它的失败不能用来推断 E-003 的上限，只能证明仿真环境与公共链路对两个方案是公平的。

### 3.2b E-003@40000 仿真验证（2026-07-20，v1.0.1 对齐 horizon=750）

- 离线最佳 checkpoint steps_40000（pos_corr 0.601）跑 OpenDrawer 10 episode、MAX_STEPS=750：**0/10**。
- 至此两个独立证据（baseline pos_corr 0.62→0/10；E-003 pos_corr 0.60→0/10）钉死：**离线 pos_corr ≈0.6 这一档在闭环里不足以产生成功**。
- 失败末态一致：手臂坠到台面下方（z 维弱，corr 0.3~0.45），够不到台面高度的抽屉把手。
- 同时 45k/46k 离线指标未再超过 40k（46k 回落 ~2σ），训练 loss 同步走平——**当前 recipe 已收敛，继续加步数无边际收益**。后续实验应转向 recipe 杠杆（B2 history、binary gripper、state 归一化、q01/q99、EMA、图像增强、多任务/多本体数据），见 §7。

### 3.3 共同的短板形态

- **预测幅度收缩**：pred std 只有 expert 的 1/3~1/2（典型 mean-regression），闭环表现为"方向大致对、永远走不到位"。
- **eef z（垂直）维最弱**：corr 仅 0.25~0.48；仿真录像里两个模型都呈现 z 向病态行为（栽到台面下/悬停漂移）。
- **episode 起始帧 gripper 偶发符号翻转**（pred +0.93 vs expert -1.00），经 env 0.5 阈值后直接执行成反向夹爪指令。
- 样本量提醒：此前多数仿真评测只有 1~2 episode，无统计意义；协议应为每任务 ≥10 episode。

## 4. 已作废 / 不可引用的评测结果

1. **E-001 在 2026-07-16 之前的所有仿真结果作废**：当时的 eval client 只发送 `agentview_left` 单视角，而 E-001 按 3 视角（left/right/wrist 单帧 224²）训练。受污染的记录包括 baseline 70k 的 0/2、50k 的 0/5、E-001-mowa 50k 的 0/5。baseline 已由本文用修正 client 重测（0/10）；**E-001-mowa 至今仍无干净的仿真测量**。
2. **`steps_28500` checkpoint 已被轮转删除**：`mowa_ckpt/...260718_0039/checkpoints/` 下只剩 `.eval` 结果目录。以后要评测某个具体 step 需趁早，或先确认 checkpoint 保留策略。
3. 2026-07-19 中午部分 eval run 因 server 进程被杀（与训练/其他任务抢 GPU）而无结果，属工程问题。

## 5. 工具箱（均在 `tools/mowa/`，可直接复用）

| 工具 | 用途 | 典型用法 |
|---|---|---|
| `e003_offline_action_check.py` | 离线动作一致性检查：数据集观测→policy server→逐维对比 expert，输出 MSE/NMSE/corr/逐维明细/示例轨迹 | `.venv/bin/python tools/mowa/e003_offline_action_check.py --dataset-dir <lerobot目录> --ckpt-path <ckpt> --episodes 0 1 2 --stride 15 --compare-horizon 8 --payload-style wanpi --port 6791` |
| `e003_offline_action_curve.sh` | 多 checkpoint 批跑（学习曲线），自动逐个起停 server | `bash tools/mowa/e003_offline_action_curve.sh 5000 10000 20000 30000` |
| `e003_expert_replay_sim.py` | expert 动作仿真回放：判定动作/状态语义在训练↔仿真间是否一致（无需 GPU server，`.robocase` 环境运行） | `.robocase/bin/python tools/mowa/e003_expert_replay_sim.py --dataset-dir <lerobot目录> --episode 0 --n-replay 80` |
| `mowa_lang_sensitivity_probe.py` | 语言敏感度：同画面切指令，测输出调制幅度 | `.venv/bin/python tools/mowa/mowa_lang_sensitivity_probe.py --dataset-dir <lerobot目录> --episodes 4 5 0 1 --payload-style wanpi --port 6791` |
| eval client 双协议开关 | `examples/Robocasa_365/eval_files/`：`PolicyWarper(payload_style="wanpi"\|"starflow")` + `simulation_env.py --args.payload-style` | wanpi=E-003 双视角 5 帧历史；starflow=E-001 三视角单帧 224²。默认 wanpi，行为不变 |

前置：policy server 需先启动（`.venv` 环境、GPU）：
```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python deployment/model_server/server_policy.py \
  --ckpt_path <ckpt> --port 6791 --use_bf16 --batch_timeout_ms 100 --max_batch_size 8
```

## 6. 再遇仿真 0% 时的排查 SOP

1. **先跑离线动作检查**（1 分钟）：若离线拟合差（pos corr < ~0.6），直接归因模型质量，停止怀疑管线。
2. 离线拟合好但仿真 0% → 跑 **expert 回放**：数据集动作能否在仿真复现运动？不能 → 动作/状态语义链路有 bug（查 `convert_hdf5_lerobot` ↔ `gym_wrapper` ↔ client 布局）。
3. 回放正常 → 对比**训练帧 vs 仿真帧**（相机机位/翻转/resize）。
4. 再跑**语言敏感度探针**（排除 instruction-blind）。
5. 以上全过 → 才是模型闭环质量问题：看离线曲线是否随步数上升，决定继续训练还是改 recipe。
6. 评测协议：每任务 ≥10 episode，固定 client 版本与 payload-style；确认 server 不与训练抢卡。

## 7. 当前解释模型与下一步

**解释**：h=0（wo_history）下视觉条件只有当前 ~0.2s 快照 + state，模型看不到运动方向/速度/任务阶段，对"未来 1.6s 做什么"的最优猜测天然偏向条件均值——这解释了"方向对、幅度收缩、gripper（由位姿决定）学得好、eef z 最弱"的全部观察。叠加仅 ~1.2 epoch 的训练量，30k 步 0% 是符合模型当前水平的合理结果，不需要引入"管线 bug"或"recipe 封顶"假设。

**下一步实验优先级**：
1. E-003 继续训练 + **每个新 checkpoint 跑离线曲线**（`e003_offline_action_curve.sh`），盯 eef_pos/z 维 corr；不到 ~0.7 不烧仿真时间。
2. E-004（h>0 / HLC-GCI）出 ckpt 后用同一离线脚本做**同族对照**：corr 明显上升则坐实 history 是关键缺失；仍 ≈0.5 则问题在 action head 容量/数据噪声。
3. state 表示消融（需重训）：当前仅 sin/cos 无归一化，且 base_position 存在 |x|>π 的 sin/cos 混叠实例（数据实测 -4.905）；可对位置量改 min_max。注意 `PolicyNormProcessor` 会自动跟随 DataConfig 的 transform，管线保持一致。
4. 低成本消融：action_horizon 32→8/16（h=0 下长 chunk 难拟合）；LoRA rank 8→更大（Wan 主干适配容量）。
5. 已排除、不要再试：增加 flow-matching 推理步数（4→10 无改善，§2）。

## 8. 关键陷阱清单（后来者必看）

1. **parquet 动作列布局 ≠ 训练布局**：lerobot parquet 原始 `action` 列是 `[base_motion(4), control_mode(1), eef_pos(3), eef_rot(3), gripper(1)]`，训练/模型输出是 `[eef_pos, eef_rot, gripper, base_motion, control_mode]`。做任何离线数据分析必须按 `meta/modality.json` 重映射（`e003_offline_action_check.py` 的 `build_parquet_to_training_map` 已实现）。推理链路不接触 parquet，无此问题。
2. **E-001 与 E-003 的推理 payload 完全不同**：E-001（StarFlowVLA/Qwen）=`image`[left,right,wrist 单帧 224²]+离散化 state；E-003（WanPI）=`mowa_multi_view_images`[main,wrist 各 5 帧 256²]+sin/cos state。混用会静默退化或直接报错；用 §5 的 payload-style 开关。
3. **dataset_statistics 的 state 统计是 sin/cos 之前的 16 维**（物理单位），action 统计 min/max≈±1 使 min_max unnorm 成为恒等——离线对比时 server 输出可与数据集 action 直接比较。
4. **Wan 在线 condition 帧数公式**：`required_frames = 1+4*(history_window_steps+1)`，h=0 时为 5；client 的 `wan_history_frames` 必须满足 1+4k 且 ≥5。
5. **训练日志里 `epoch` 是真实 epoch 计数**（30k 步 ≈1.2 epoch），评估"欠训练 vs recipe 问题"时以它为准，不要用错数量级的估算。
6. **checkpoint 会被轮转删除**（见 §4.2）；评测脚本与训练共用 GPU，server 可能被杀导致 eval 无结果。

## 9. 对外部评审意见的勘误（2026-07-19）

对一份外部分析（基于代码/loss 静态阅读）的核实结果：

- 其"冻结 `proj_out_1/2` 削弱输出"为**误读**：`LayerwiseFM_ActionHeader.py:275-278` 冻结的是 LayerwiseFM 本就不使用的 DiT 封装输出层，属无害空操作；444M 参数 action DiT 可训练。
- 其"30k 步 ≈ 数据的 1/6~1/3"**算错数量级**：实际 ~1.2-1.7 epoch（训练日志自报 `epoch=1.19` @32.5k）。
- 其"4 步推理误差累积"已被实验否掉（§2 末行）。
- 其"state_encoder 接收原始物理尺度"不对（输入是 [-1,1] 的 sin/cos），但"state 无归一化"本身属实（§7.3）。
- 其主结论（非 pipeline bug、h=0 是结构瓶颈、建议 E-004 对照与离线监控）与本文一致。
