# EXT-B1 外部正对照设计方案：StarVLA 评测框架加载官方 GR00T N1.5 RoboCasa365 checkpoint

> 版本：v2（2026-07-20），合并 codex 与 claude(deepseek) 两轮评审的修订版。v1 评审结论：方向正确，细节 No-Go；本版按评审阻断项逐条修订。
> 定位：**评测-only 的外部正对照**，不是训练实验，**不占用也不变相新增实验编号**（应治理评审要求，由 v1 的 "E000-B1" 更名为非实验标识 **EXT-B1 / MoWA-External-GR00T-N1.5-B1**；工程产物统一 `tools/mowa/mowa_external_groot_n15_*` 命名）。
> 关联文档：`13_sim_zero_offline_diagnosis.md`、`README.md`。

---

## 0. 一句话方案

保留 StarVLA 的仿真 client、批量评测、结果统计与视频录制，**不改 StarVLA 框架代码**；通过"传输层代理 + 新 payload-style"把官方 GR00T N1.5 RoboCasa365 checkpoint 接到同一套评测链路上：先 artifact/env preflight 与官方组件 smoke，再固定噪声 parity，再 paired rollout，最后按官方协议出参照分。

## 1. 背景与目标

E-001 baseline 与 E-003 已有干净评测均为 OpenDrawer 0/10（E-001-mowa 尚无干净仿真测量），根因定位为模型闭环质量不足（`13_sim_zero_offline_diagnosis.md`）。expert 回放已把**共享的** simulator、动作执行、成功判定与统计链路验证到 ~1mm，但这不能证明 MoWA 专属的 wanpi/starflow 输入、checkpoint loader 与 server 路径无错——引入一个外部强策略正对照，可以同时校准"正常强策略"的行为/成功率量级，并对评测链路做最后一次端到端独立验证。

**非目标**：不重新训练；不把 GR00T 权重移植/转换成 StarVLA 内部格式；不合并两套依赖栈；不评价官方模型优劣。

## 2. 关键事实（代码调研核实，设计依据）

### 2.1 官方 GR00T N1.5 数据契约（`playground/Code/Isaac-GR00T`）

| 项 | 值 | 来源 |
|---|---|---|
| 相机 | 3 路 `agentview_left/agentview_right/eye_in_hand`；**官方 processor 独占预处理**：输入原始 uint8 256² → center crop 0.95 → resize 224²（eval 无颜色抖动；metadata 校验原始分辨率） | `gr00t/experiment/data_config.py:498-511`；`gr00t/data/transform/video.py:290` |
| 帧率/历史 | 20Hz 原样，**单帧观测（T=1），无历史堆叠** | `data_config.py:662-663` |
| state | 原始 16 维 → 两个 quat 转 **rotation_6d** → 20 维；min_max→[-1,1]（绝对旋转固定 ±1；min==max 维置 0） | `data_config.py:666-676`；`gr00t/data/transform/state_action.py:592-604,322-325` |
| action | **12 维，布局与我们逐维一致**：eef_pos(3)+eef_rot(3)+gripper_close(1)+base_motion(4)+control_mode(1)；delta EEF（base 系，±0.05m/±0.5rad）；eef/base 用 min_max，gripper/control_mode 用 binary | `robocasa/utils/lerobot_utils.py:64-70`；`data_config.py:653-659,677-683` |
| 指令 | `annotation.human.task_description`（episode 级） | `data_config.py:661` |
| 归一化元数据 | 随 checkpoint 存放 `experiment_cfg/metadata.json` | `gr00t/experiment/runner.py:72-91` |
| 架构 | Eagle2.5-VL（SigLIP 1152 + Qwen3-1.7B 2048，冻结）+ FlowmatchingActionHead；**全部预训练初始化**（`nvidia/GR00T-N1.5-3B`） | `gr00t_finetune.py:81,84-94`；`eagle2_hg_model/config.json` |
| horizon/推理 | **action horizon 16，全开环执行 16 步**（0.8s）再重规划；去噪 4 步 Euler；无 EMA | `data_config.py:663`；`scripts/run_eval.py:47,155-160` |
| **推理随机性** | flow-matching head **每次推理 `torch.randn` 生成初始噪声**（`do_sample=False` 对其无效）——parity 必须显式控制 RNG | `flow_matching_action_head.py:364` |
| 推理入口 | `Gr00tPolicy`（构造需 panda_omron 的 modality config + transform + embodiment tag）；`get_action()` 返回**按 action key 分段的 dict**，不是 (B,16,12) | `gr00t/model/policy.py:57`；`scripts/run_eval.py:36`；`data_config.py:653` |
| data_config 注册 key | `DATA_CONFIG_MAP["panda_omron"]`（**不是 "robocasa365"**） | `data_config.py:942` |
| 评测协议 | v1.0.1：各任务 horizon ×1.5（OpenDrawer 750）；每任务 50 episodes；multitask 榜单在 **pretrain 场景**（layout/style -2） | `robocasa365/README.md:108`；`robocasa/utils/env_utils.py:86-95`；`scripts/run_eval.py:76-86` |

### 2.2 官方成绩的两个口径（必须分开，不得混用）

| 口径 | 数字 | 来源 | 性质 |
|---|---|---|---|
| 本地官方文档（checkpoint-120000，split=pretrain，18 任务） | **Atomic-Seen 43.0%** | `robocasa365/docs/benchmarking/multitask_learning.md:27,131` | **Phase 4A 的可核验复现目标** |
| 线上 leaderboard（07-16 更新，1.5× horizon 复测） | 50.7% | robocasa.ai/leaderboard | 仅作参考背景，不作为复现目标； horizon/版本不同不可直接对齐 |

**禁止**：用 18 任务平均值推断 OpenDrawer 单任务水平；把 target split 的评测结果称为"官方成绩复现"。

### 2.3 我们评测链路侧

- 动作执行契约与官方**逐维一致**（`RoboCasaGymEnv.unmap_action` 的 0.5 阈值、base/torso 切分），动作侧零转换（expert 回放实证，13 号文档 §2）。
- 差异只在**输入侧**：官方要 3 视角原始 256² 单帧 + 原始 16 维 state（其 processor 内部转 6D+min_max），我们的现有 payload（wanpi/starflow）均不满足。
- client↔server 是自有 websocket 协议（`deployment/model_server/tools/websocket_policy_*.py`，依赖仅 websockets+msgpack+numpy，可独立 import）。
- **已知坑（评审发现）**：`websocket_policy_server.py:96` 对单请求 B>1 的批量拆包按请求数而非样本跨度处理，多请求并发时会只回第一条动作——**初版强制 N_ENVS=1**；`websocket_policy_server.py:223` 会把 adapter 返回值包进 wire response 的 `{"data": ...}`，adapter 若自己返回 `{"data": ...}` 会形成 `data.data.actions` 嵌套。
- `examples/Robocasa_365/eval_files/simulation_env.py:97` 当前硬编码 `split="target"`；`Args.seed` 存在但未传给 env reset——split/seed 需要显式接入。

## 3. 总体架构

```
RoboCasa365 sim (.robocase env)
  └─ examples/Robocasa_365/eval_files/simulation_env.py
       └─ PolicyWarper(payload_style="groot")            ← 新增第 3 种 payload
            │  websocket: examples=[{image:[3×uint8 256²], lang, state_raw:(1,16)}]
            ▼
EXT-B1 代理 server（独立 gr00t venv，GPU）
  └─ tools/mowa/mowa_external_groot_n15_proxy_server.py  ← 新增，薄壳
       ├─ starVLA websocket 协议端（复用 websocket_policy_server.py）
       └─ adapter（唯一新逻辑）：predict_action(examples)
            ├─ 观测映射为官方 observation dict（原样转发，不重写预处理）
            ├─ Gr00tPolicy.get_action() → per-key action dict
            └─ 按官方 action_keys 顺序拼接 (B,16,12) + 逐 key shape 断言
            → return {"actions": ndarray}                ← 不包 "data" 层
```

原则：**官方 processor、官方反归一化、官方推理路径一律不重写**；StarVLA 框架、trainer、`PolicyNormProcessor` 零改动。

## 4. 分阶段实施与验收（按评审修订的门禁顺序）

### Phase -1：治理与 artifact preflight

- 更名落地（EXT-B1）；记录下载/安装的人工授权范围（用户已授权 checkpoint 下载与 venv 安装）。
- checkpoint manifest 核对：5 个必需文件（2×safetensors 分片、index、config.json、`experiment_cfg/metadata.json`）+ SHA256；记录 Isaac-GR00T/robocasa365 的 commit hash 与依赖版本。
- **验收**：manifest 齐全、hash 一致；授权范围写入实施日志。
- 状态：checkpoint 用户下载中（本版评审时目录尚不存在）。

### Phase 0A：环境与官方组件 smoke

- 新建独立 venv `playground/.venvs/ext_b1_groot`（python 3.11，`pyproject.toml` 要求 >=3.10，兼容）。
- 安装：`pip install -e playground/Code/Isaac-GR00T` **不足以单独成事**——torch 在 optional extra 中，还需显式安装 torch、robosuite、robocasa（本地 `playground/Code/robocasa365`）、websockets、msgpack；flash-attn 当前代码**没有可靠的 sdpa/eager 回退**，列为风险 R1（首日验证）。
- 用**薄 runner**（官方组件组装）跑：checkpoint 加载 → 单步前向（有限动作、shape 断言）→ 单任务 native rollout smoke。
  - 注意：官方 `scripts/run_eval.py:63` 只接受 `TASK_SET_REGISTRY` 的整组 task set，不支持单任务参数；单任务 smoke 必须自组 runner，完整复现留给 Phase 4A。
- **验收**：加载成功、前向有限值、rollout 完整跑完。**0/N 不作为失败**（见 §6 统计分析）。
- Phase 0A 全部日志归档至 `playground/mowa_eval_results/ext_b1_groot_n15/phase0_native/`。

### Phase 1：代理 server + groot payload-style

改动清单（全部新增或向后兼容，默认行为不变）：

1. `examples/Robocasa_365/eval_files/model2robocasa365_interface.py`
   - `payload_style` 增加 `"groot"` 分支：**发送原始 uint8 256² ×3 视角、T=1，client 不做任何 resize/crop**（官方 processor 独占预处理；client 预 resize 会造成双重预处理并可能触发其 metadata 的 shape 断言）；
   - `state_raw` = 原始 16 维（**不做 sin/cos**），显式 schema 注释（顺序见 §2.3）；
   - `_validate_observations` 对 groot 分支**提前返回**，不与 wanpi/starflow 校验耦合；
   - `n_action_steps` 支持 16；跳过 wanpi 的 `wan_history_frames≥5` 校验。
2. `examples/Robocasa_365/eval_files/simulation_env.py`
   - groot 模式 `video_delta_indices=np.array([0])`；
   - **split 与 seed 显式接入**：`--args.split`（默认 target）、`--args.seed` 传入 `env.reset(seed=...)`；episode 级 seed manifest 落盘（供 Phase 3 paired rollout 复用）；
   - **N_ENVS 固定 1**（见 §2.3 websocket 拆包坑）。
3. `tools/mowa/mowa_external_groot_n15_proxy_server.py`（新增，gr00t venv 运行）
   - 复用 `websocket_policy_server.py` 起 starVLA 协议端口；
   - adapter 实现 `predict_action(examples, **kwargs) -> {"actions": np.ndarray}`（**不包 `"data"` 层**，wire response 由 server 统一包装）；
   - 加载：`Gr00tPolicy` + `DATA_CONFIG_MAP["panda_omron"]`（modality config + transform + embodiment tag），checkpoint 内 `experiment_cfg/metadata.json` 提供归一化元数据；
   - 观测映射伪代码：

     ```python
     # examples[i] = {"image": [left,right,wrist] uint8 256², "lang": str, "state_raw": (16,)}
     obs = {
         "video.robot0_agentview_left":  img_left[None],     # (1,256,256,3)
         "video.robot0_agentview_right": img_right[None],
         "video.robot0_eye_in_hand":     img_wrist[None],
         "state.end_effector_position_relative": s[None, 7:10],
         "state.end_effector_rotation_relative": s[None, 10:14],
         "state.gripper_qpos":                 s[None, 14:16],
         "state.base_position":                s[None, 0:3],
         "state.base_rotation":                s[None, 3:7],
         "annotation.human.task_description": [lang],
     }
     out = policy.get_action(obs)          # per-action-key dict
     actions = concat_in_official_key_order(out)   # 按 data_config.action_keys，逐 key 断言 shape
     return {"actions": actions[None]}             # (1,16,12)
     ```
   - server meta 报 `action_chunk_size=16`、action_keys 与 client 同序、`available_unnorm_keys=["external_groot_n15_raw"]`（仅供存在性校验；输出已是 env 空间，不做二次反归一化）。
4. `playground/mowa_ext_b1_eval.sh`（新增）：起代理（gr00t venv）+ 起 client（.robocase），默认 `N_ACT=16`、`MAX_STEPS=750`、split/seed 可配。

- **验收**：client 握手成功；单 episode 完整跑完；动作非恒定、无 NaN；无 `data.data` 嵌套。

### Phase 2：固定噪声数值 parity（强制门禁）

评审关键修正：官方 flow-matching head 每次推理都 `torch.randn`，`do_sample=False` 无效，**不控制 RNG 则直连与代理必然不一致**。

- `tools/mowa/mowa_external_groot_n15_parity.py`：
  1. 同一份原始观测（来自 Phase 0A 录制或数据集帧）；
  2. 每次调用前**恢复同一 CPU/CUDA RNG 状态**（或显式注入同一 action noise）；
  3. 分别核对：transform 后 tensor（bit 级）与反归一化后动作（<1e-5）；
  4. 直连分支**独立实现**观测映射，不复用代理的 helper，避免同错自证；
  5. 覆盖 **≥3 个任务**、每任务多帧（含 episode 观测序列，防 state 累积误差——claude 评审建议）；
  6. 附带验证 `VideoColorJitter` 在 eval 模式下确已禁用（同一观测两次调用、控制 RNG 后 bit-exact）。
- **验收**：全部观测对上；不通过不进 Phase 3。

### Phase 3：paired rollout 轨迹一致性

- OpenDrawer **10 个 paired episodes**：代理链路与官方原生使用**相同 env seed（seed manifest）、相同 flow-noise seed**；
- **验收**：success vector 一致；若确定性对齐成立，进一步要求 EE 轨迹一致（容差内）。统计说明见 §6。

### Phase 4A：官方口径复现（pretrain）

- `split=pretrain`（layout/style -2），18 个 atomic_seen 任务 × 50 episodes，v1.0.1 各任务 horizon（450~1050 步）。
- **验收**：复现 **43.0%**（本地官方文档口径，§2.2）；记录 CI 与逐任务分解。

### Phase 4B：MoWA 场景外部参考（target）

- `split=target`，同 18×50 协议；**单独报告为"外部场景参考"**，不得称官方成绩复现。
- MoWA 侧只有按**相同 seed/task/horizon** 重跑后，才允许做公平差值对比。

## 5. 环境方案

| 环境 | 用途 | 说明 |
|---|---|---|
| `.robocase` | 仿真 client | 不变 |
| `playground/.venvs/ext_b1_groot`（新建） | GR00T policy + 代理 server | `pip install -e playground/Code/Isaac-GR00T` + 显式补 torch、robosuite、本地 robocasa365、websockets、msgpack；与 starVLA `.venv` 完全隔离；GPU 可配 `CUDA_VISIBLE_DEVICES` |
| starVLA `.venv` | 不使用 | 本方案不经过 StarVLA 框架 |

- R1：flash-attn 可能装不上且当前代码无可靠 sdpa/eager 回退——Phase 0A 首日验证；若失败，选项为源码编译 flash-attn、改官方代码 attention 接口（记录为外部补丁，不回写 StarVLA）、或换用 GPU 驱动兼容的 wheel。
- R2：run_eval.py 仅接受整组 task set——单任务 smoke 用自组薄 runner。

## 6. 统计与判定准则

- **Phase 0A 不以 0/N 为失败**：真实 SR=43% 时，2 连败概率 32.5%、5 连败仍约 6%；小样本无权判死刑。
- 评审修正：v1 的"±1 episode 一致"无统计依据，废除。Phase 3 的判定依赖 paired seed/noise 的逐条对比（success vector / 轨迹），不靠 episode 数对齐。
- Phase 4 报告 95% CI（Wilson）；与官方 43.0% 的差值超过 CI 才判定不一致。
- 16 步全开环 = 0.8s 无重规划，早期步走偏则整 chunk 作废——这是官方协议固有性质，结果分析时需显式注明。

## 7. 测试计划

| 测试 | 入口 | 通过标准 |
|---|---|---|
| 静态 | `py_compile` 改动文件；`bash -n playground/mowa_ext_b1_eval.sh` | 通过 |
| 回归 | `pytest tests/mowa/test_multiview_wan.py -q` | 全过（wanpi 路径不受影响） |
| 契约 | groot payload 单步 mock：image 为 uint8 256²×3、state_raw 16 维、无 resize | 键/形状/dtype 正确 |
| parity | `mowa_external_groot_n15_parity.py`（固定 RNG，≥3 任务） | transform 后 bit 级一致；动作 <1e-5 |
| 闭环 | Phase 3 paired rollout | success vector 一致 |

## 8. 风险清单

| 风险 | 等级 | 缓解 |
|---|---|---|
| R1 flash-attn / 依赖安装失败 | 中 | Phase 0A 首日验证；备选 wheel/源码编译/外部补丁（不回写 StarVLA） |
| R2 run_eval.py 不支持单任务 | 低 | 自组薄 runner（Phase 0A） |
| R3 parity 确定性被官方噪声源破坏 | 已修订 | Phase 2 强制 RNG 控制 + 独立直连映射（§4） |
| R4 websocket B>1 拆包只回首条 | 已修订 | 初版 N_ENVS=1；后续如需并发另开 issue |
| R5 checkpoint 不完整/版本漂移 | 低 | Phase -1 manifest + SHA256 + commit 记录 |
| R6 官方文档 43.0% 与线上 50.7% 口径混淆 | 已修订 | §2.2 双口径声明；Phase 4A 只对 43.0% 负责 |

## 9. 治理声明

- 本方案为 eval-only 外部正对照（EXT-B1 / MoWA-External-GR00T-N1.5-B1），不占用也不变相新增实验编号；工程产物统一 `tools/mowa/mowa_external_groot_n15_*`；
- 所有代码改动为新增或向后兼容开关（默认 `payload_style="wanpi"` 行为不变）；不改写 00/01/02 核心 SOT；
- 下载/安装的人工授权范围记录于 Phase -1 与实施日志；
- 执行后按 AGENTS.md 规则更新 `07_implementation_log.md`、`TODO.md`；结果写回本文件 §10 并在 13 号文档增补参照行；
- 背景陈述已按评审修正为：E-001 baseline 与 E-003 已有干净评测均为 0/10；E-001-mowa 尚无干净仿真测量。

## 10. 结果记录（执行后填写）

| 阶段 | 任务 | 协议 | 结果 | 日期 |
|---|---|---|---|---|
| Phase 0A 官方组件 smoke | OpenDrawer（薄 runner） | native rollout，target，2 eps/750 步 | **SR=0.50（1/2），150s/2eps，官方路径完整可用** | 2026-07-20 |
| Phase 3 paired rollout | OpenDrawer | 10 paired eps | TBD | - |
| Phase 4A 官方口径复现 | 18 atomic | pretrain ×50 / v1.0.1 horizon | 目标 43.0%（§2.2） | - |
| Phase 4B 外部场景参考 | 18 atomic | target ×50 / 同协议 | TBD | - |

Phase 0A 附加记录：checkpoint（`gr00t_n1-5/multitask_learning/checkpoint-120000`，两 safetensors 分片字节数与官方一致，SHA256 见 `playground/mowa_eval_results/ext_b1_groot_n15/checkpoint_sha256.txt`）；环境 `playground/.venvs/ext_b1_groot`（torch 2.5.1+cu124、flash_attn 2.7.4、Isaac-GR00T editable、robocasa 1.0.1 + robosuite 源码版 + mujoco 3.3.1；lerobot 按评测路径不需要而跳过）；报告 `phase0_native/phase0a_native_smoke.json` + 视频。

## 11. FAQ：为什么不用 StarVLA 已有的 QwenGR00T 加载官方 checkpoint？

结论：**不能，兼容是结构/配方层面，不是权重层面**。StarVLA 的 `QwenGR00T` 是"Qwen-VL + GR00T 配方 flow-matching head"的自训框架——action head 代码同源自 GR00T N1.5（`GR00T_ActionHeader.py` 与官方 `flow_matching_action_head.py:60-69` 的 CategorySpecific 结构一致），但：

| 层 | StarVLA QwenGR00T | 官方 GR00T N1.5 |
|---|---|---|
| backbone | Qwen2.5/3-VL（hidden 2560 级） | Eagle2.5-VL = SigLIP（1152, 27 层）+ Qwen3-1.7B（2048, 28 层）（`Isaac-GR00T/gr00t/model/backbone/eagle2_hg_model/config.json`） |
| processor/数据管线 | StarVLA gr00t_lerobot transforms（本项目 sin/cos state 等） | 官方 modality transform（quat→rotation_6d、min_max、Eagle 图像预处理） |
| action head 条件维度 | 随 Qwen-VL（2560 级） | LLM 2048→1536 投影后进 DiT |

官方 backbone 权重装不进 QwenGR00T；action head 权重 shape 不匹配；processor 完全不同。要在 StarVLA 进程内跑官方模型，必须 import 官方整个模型栈——收益为零、只剩依赖合并成本。故选择代理：评测链路复用 StarVLA，模型本体跑官方代码。QwenGR00T 的用武之地是"自训 GR00T 配方 baseline"（层次 B，与 E-001 修正相关），与 EXT-B1 的"复现官方成绩"是两件事。

## 12. 修订记录

- v1（2026-07-20 早）：初稿。
- v2（2026-07-20）：合并 codex 评审（No-Go→修订：治理更名 EXT-B1；43.0% vs 50.7% 双口径；原始 256² 输入；Phase 0 重构为 -1/0A preflight；Gr00tPolicy+per-key dict adapter 契约与 wire 不嵌套；parity 强制 RNG 控制；split/seed/N_ENVS=1 与统计门禁；背景过宽陈述修正）与 claude(deepseek) 评审（DATA_CONFIG_MAP["panda_omron"] 显式化；VideoColorJitter eval 行为验证；adapter 伪代码；parity 覆盖 episode 序列；Phase 0 日志归档；`_validate_observations` groot 提前返回）。
