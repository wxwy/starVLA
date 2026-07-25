# WanOFT 推理回归事件与修复（2026-07-23，Kimi 落盘）

> 面向其他大模型协作者（codex / claude 等）的情况同步。读完应能回答：
> WM4A-Wan2d2-OFT-LIBERO-4in1 为什么一度仿真 0%，修了什么、改在哪些文件、
> 对 E-003-B1 / E-003-B1-v2 / 各类评测有什么影响、后续评测该怎么发 payload。

## TL;DR

- 7 月 MoWA 改动把 `Wan2.py` 的 VAE 图像输入预处理换成了 256²+4n+1 补帧+`mode()`，
  导致**所有 WanOFT 旧 ckpt（含 WM4A-Wan2d2-OFT-LIBERO-4in1）推理全坏**（latent 网格
  从 (1,30,52) 变 (2,16,16)，特征全毁），LIBERO 仿真 0/17。
- 同时段另一独立问题：07-16 起 LIBERO eval client 无条件给 payload 加 `state`，
  对训练时 `include_state` 未开的 WanOFT ckpt 是纯毒（修复后 A/B：带 state 0/15，不带 14/15）。
- 修复：`legacy_vae_input` 开关恢复 WanOFT 旧预处理；eval client 的 state 改为按开关发送。
- 验证：离线 pos_corr 0.08 → **0.987**；仿真 3 任务×5 轮 **14/15=93.3%**（官方 50 轮口径 95.4%）。
- E-003-B1（WanPI/RoboCasa 双视角长训）**零影响**；E-003-B1-v2 代码零影响，
  但存在"供体 backbone 是在旧网格上微调的"这一表示事实（见 §6）。

## 1. 背景与症状

任务：对 `/disk/rl/models/mowa_init_candidates/WM4A-Wan2d2-OFT-LIBERO-4in1`
（WanOFT 框架，Wan2.2-TI2V-5B backbone + MLP 头，steps_60000，官方报告
libero_goal 95.4%、4-suite avg 91.6%）做 LIBERO 仿真推理测试。

环境：server 用 repo `.venv`（torch 2.6.0）；client 借用 `/disk/rl/RLinf/.venv`
（Python 3.11，libero editable 安装自 `/disk/rl/RLinf/.venv/libero`，mujoco 3.8.0，
robosuite 1.4.1，tyro/websockets/imageio 齐全）。GPU 0（4090 24G）。

症状：烟测 libero_goal 3 任务×5 轮 = **0/15**（加上首轮共 0/17）。视频中机械臂
有目的性运动（语言条件部分生效），但始终无法完成任务。

## 2. 排查链（每步都有实证）

1. **排除运气**：95% 真实 SR 下 0/17 概率≈10⁻⁹。
2. **排除/隔离 state 假设**：发现 git log `b7af675`(2026-07-16) "fix(LIBERO): pass robot
   state to model during evaluation" 起 eval 才发 state；官方 04/26 评测日志
   （ckpt 仓库 `logs/libero_goal/*.log`）显示当时 args 无 state、SR=0.954。
   A/B（当时 VAE bug 仍在）：带/不带 state 均 0/15——VAE bug 占主导，state 影响被掩盖。
3. **离线对照定界**：新工具 `tools/mowa/wm4a_offline_action_check.py` 把 libero_goal
   专家轨迹观测（primary/wrist 视频帧+指令）逐帧发给 server，比较返回动作块与专家动作：
   **新 server pos_corr≈0.084**（预测幅度收缩、z 维负相关）→ 问题在模型服务侧，与仿真循环无关。
4. **旧版对照实锤**：`git worktree add /tmp/starvla_0426 342d3da`（官方评测当日代码），
   起旧 server（旧协议 `normalized_actions` + 客户端 min_max/clip/gripper 二值化 unnorm，
   工具 `wm4a_offline_action_check_oldproto.py`）：**pos_corr=0.987**。
   同一 ckpt、同一观测、同一 .venv → 回归在当前代码。
5. **diff 定位**：`git diff 342d3da HEAD -- starVLA/model/modules/world_model/Wan2.py`
   （340 行改动，MoWA 7 月系列提交）发现三处输入变化（见 §3）。

## 3. 根因（三处叠加，全在 `_encode_images_vae` 路径）

| 项 | 训练/官方评测（旧，342d3da） | 7 月 MoWA 改动后（HEAD） |
|---|---|---|
| 分辨率 | `preprocess_video(imgs, height=480, width=832)` → latent (1,30,52) | `prepare_wan_vae_frames` center-crop→256×256 → latent (2,16,16) |
| 时间维 | 不补帧，[main,wrist] 2 帧 → T_latent=1 | `wan_padded_frame_count` 补到 5 帧（wrist×3）→ T_latent=2 |
| VAE 编码 | `no_grad` + `latent_dist.sample()` | `inference_mode` + `latent_dist.mode()` |

WanOFT ckpt 在旧网格上全量微调，token 数/RoPE 位置全变 → 特征全毁。
注意 `inference_mode` 产出的 tensor 进 autograd 会报错，legacy 分支必须用 `no_grad`。

> 表中的 `[main,wrist]` 混帧只发生在 **WanOFT 路径**（WanOFT 把两个静态视角当一条
> 2 帧"视频"送进 `_encode_images_vae`）。WanPI/E-003-B1 不存在这种混帧：
> 每个视角的 RGB 序列独立编码，历史不足时首帧前补、尾部不足时末帧后补（见附录 A.2）。

**勘误（重要）**：codex 此前描述的"原生 WanOFT 训练把 [main,wrist] 补帧到 5 帧"
是**新代码的行为，不是训练时的真实格式**。训练真实格式：2 帧、480×832、T_latent=1。

## 4. 修复内容（未提交 git，工作区改动）

1. `starVLA/model/modules/world_model/Wan2.py`
   - `_Wan2_Interface.__init__` 新增 `self.legacy_vae_input = bool(wm_cfg.get("legacy_vae_input", False))`。
   - `_encode_images_vae` 增加 legacy 分支：480×832 resize、不补帧
     （`target_frames = num_frames or batch max`，不走 `wan_padded_frame_count`）、
     `no_grad + sample()` + `normalize_wan_vae_latents`。
   - 非 legacy 分支（MoWA/WanPI 的 256²+补帧+mode）**未做任何改动**。
2. `starVLA/model/framework/WM4A/WanOFT.py`
   - `WanOFTDefaultConfig.world_model` 增加 `"legacy_vae_input": True`。
   - `merge_framework_config` 是深合并（`OmegaConf.merge`），旧 ckpt 的 config.yaml
     不含此键 → 默认 True 生效，无需改 ckpt。
3. `examples/LIBERO/eval_files/eval_libero.py`
   - `example_dict` 不再无条件带 `state`：`payload_style=="wanpi"` 始终带
     （client wanpi 分支强制要求 (1,8) raw state，否则报错）；
     standard 时仅环境变量 `LIBERO_SEND_STATE=1` 才带。
4. 新工具：`tools/mowa/wm4a_offline_action_check.py`（新协议）、
   `tools/mowa/wm4a_offline_action_check_oldproto.py`（旧协议对照）。
5. `docs_zh/mowa/07_implementation_log.md` 登记 M7-018。

## 5. 验证结果

| 测试 | 修复前 | 修复后 |
|---|---|---|
| 离线 pos_corr（新 server，新协议） | 0.084 | **0.987**（与旧版 server 一致） |
| 仿真 3 任务×5 轮，不带 state | 0/15 | **14/15 = 93.3%** |
| 仿真 3 任务×5 轮，带 state（LIBERO_SEND_STATE=1） | 0/15 | **0/15** |

分任务（不带 state）：open middle drawer 5/5、put bowl on stove 5/5、
put wine bottle on cabinet 4/5。视频与 eval_report：
`playground/eval_results/libero_goal/WM4A-Wan2d2-OFT-LIBERO-4in1_steps_60000_fixed_3x5/`
（对照组 `_smoke` / `_smoke3x5` / `_nostate_3x5` / `_fixed_state_3x5` 同目录）。

结论：两个 bug 独立且都是致命的——VAE 输入回归 + state token 污染。
WM4A ckpt 训练时 `include_state` 未开（官方 run_libero_train.sh / config.yaml 均无），
`add_discretized_state_to_instruction` 追加的 `[STATE] ... [ACTION]` token 它从未见过。

## 6. 对各线的影响

- **E-003-B1 / E-003-B2（WanPI + RoboCasa 双视角长训）：零影响。**
  WanPI 框架不用 WanOFTDefaultConfig，`legacy_vae_input` 默认 False；
  训练走 `use_visual_cache: true`（预算 latent），不经过 `_encode_images_vae`；
  在线推理走非 legacy 分支（256²+补帧），与其 latent cache 构建方式一致。
  RoboCasa 仿真用 `simulation_env.py`，不经过 `eval_libero.py`。
- **E-003-B1-v2（WanPI contft32 LoRA，backbone 供体 = WM4A ckpt，`reload_modules: backbone`）：
  代码零影响。** 但有一个表示事实：供体 Wan backbone 的 LIBERO 微调是在
  480×832 / T_latent=1 网格上做的，v2 喂的是 256² / 补帧 / T_latent=2 的 latent。
  Wan2.2 基座预训练经 RoPE 对网格灵活，迁移 OK；WM4A 的 LIBERO 特定适配与旧网格绑定，
  跨网格残余价值未知（选型时即存在，非本次引入）。可选项（不急，不建议现在动）：
  用 legacy 格式重建 v2 latent cache 以最大化供体迁移。
- **E-001（QwenPI/Qwen3-VL）：完全无关**，不经过 Wan 代码。
- **其他 LIBERO 评测（starflow P0/P1、v2 LIBERO 线）**：eval client 默认行为变了——
  standard payload 现在默认**不发** state。对训练时带 state 的模型（starflow、v2），
  评测必须 `LIBERO_SEND_STATE=1`；wanpi payload 不受影响（始终带）。
  07-16 的 b7af675 对带 state 训练的模型是修复、对 WanOFT 官方 ckpt 是破坏，
  现在两条路由开关分别覆盖。

## 7. 复现/复测命令

```bash
# server（修复后，repo 根目录）
STARVLA_PYTHON=.venv/bin/python \
RUN_DIR=/disk/rl/models/mowa_init_candidates/WM4A-Wan2d2-OFT-LIBERO-4in1 \
CKPT_STEP=60000 GPU_ID=0 PORT=6694 USE_BF16=1 \
bash examples/LIBERO/eval_files/run_policy_server.sh

# 离线对照（专家观测 vs 预测，无需仿真）
.venv/bin/python tools/mowa/wm4a_offline_action_check.py --episode 0 --num-probes 12 --port 6694

# 仿真烟测（3 任务×5 轮）
RUN_DIR=/disk/rl/models/mowa_init_candidates/WM4A-Wan2d2-OFT-LIBERO-4in1 \
CKPT_STEP=60000 LIBERO_HOME=/disk/rl/RLinf/.venv/libero \
LIBERO_PYTHON=/disk/rl/RLinf/.venv/bin/python PORT=6694 \
TASK_SUITE_NAME=libero_goal NUM_TRIALS_PER_TASK=5 MAX_TASKS=3 \
VIDEO_OUT_PATH=playground/eval_results/libero_goal/<out_dir> \
bash examples/LIBERO/eval_files/eval_libero.sh
# 带 state 对照：前面加 LIBERO_SEND_STATE=1
```

旧版对照环境（如还需）：`git worktree add /tmp/starvla_0426 342d3da`，
旧 server 需 `PYTHONPATH=/tmp/starvla_0426` 且 cwd=repo 根（config 里 base_wm 是相对路径）。

## 8. 给协作者的注意事项

- 改 `Wan2.py` / `wan_vae_utils.py` 的共享预处理时，必须同时考虑两类消费者：
  WanOFT 旧 ckpt（legacy 网格）与 WanPI/MoWA（256²+补帧 cache 体系）。
  任何默认行为变化都是静默回归，仿真 0% 之外无其他告警。
- "模型在仿真里 0%"不等于"模型不行"：本次链路是 离线对照（corr≈0.08）→ 旧版对照
  （0.987）→ diff 定位。建议把 `wm4a_offline_action_check.py` 作为 Wan 系 ckpt
  服务端正对照的标配第一步。
- ckpt 仓库自带 `logs/`（官方逐 suite 评测日志，含当时 args）和 `config.yaml`，
  是判断"官方口径"的一手证据；遇到官方成绩无法复现先查它。
- 当前现场：6694 端口修复版 server 仍在后台（22.5GB 显存）；`/tmp/starvla_0426`
  worktree 可删；全部改动未提交 git。
- 全量 4-suite×50 轮评测未跑（用户指示不需要）；小样本 93.3% 不足以替代官方口径，
  引用成绩时请注明样本量。

---

## 附录 A：概念解读（每个东西意味着什么）

### A.1 VAE 与 latent 网格 —— 本次事故的核心

Wan 模型不直接"看"图片：先用 VAE 把图片压缩成 latent token 网格，DiT 主干在网格上做注意力。
模型训练时见到的网格尺寸是它的"母语格式"，推理时网格变了，token 数量与位置编码全对不上，
特征即毁。这就是"换预处理 = 模型变瞎"的机制。

- 旧格式（WanOFT 训练/官方评测）：480×832 → latent 网格 (1, 30, 52) = 1560 token
- 新格式（MoWA 7 月改动后）：256×256 → latent 网格 (2, 16, 16) = 512 token

大白话：训练一个人只看 A4 纸报表，突然给他一张名片大小的缩略版——不是不努力，是真看不懂。

### A.2 Wan VAE 时间维与 4n+1 补帧 —— 先把正确机制说清楚

Wan VAE 的时间压缩方案（diffusers `AutoencoderKLWan._encode` 实证）：**第 1 帧单独编码为第 1 个
latent**（`x[:,:,:1]`，因果上下文零填充），之后**每 4 帧编码为 1 个 latent**
（`x[:,:,1+4(i-1):1+4i]`），即 `T_latent = (T-1)/4 + 1`，要求输入帧数满足 `T = 1 + 4k`。
注意 VAE 内部不做"首帧复制 3 次凑第一组"——首帧前补发生在数据层，VAE 只看到已补好的序列。
本项目所有补帧都遵守两条约定：

1. **历史不足时向前（首帧方向）补**：重复**首帧**把序列左端填满，保证"当前帧"始终位于序列末尾。
   - 训练侧（window manifest，`boundary_padding=True`，`window_manifest.py:628-635`）：
     episode 开头 history 窗口左端 clamp 到首个有效帧，等价于重复首帧，并带 valid mask。
   - 在线推理 client（`model2robocasa365_interface.py:183`）：
     `[history[0]] * (wan_history_frames - len(history)) + list(history)`，同样是首帧前补。
2. **尾部不够 4 帧时向后（末帧方向）补**：重复**最后一帧**补齐到 `1+4k`
   （`wan_vae_utils.py: prepare_wan_vae_frames` → `prepared + [prepared[-1]] * k`）。
   episode 末尾的 future 窗口同理右 clamp 到最后一帧并带 valid mask。

**关键：每个视角的 RGB 序列各自独立走上述流程。** WanPI/E-003-B1 中 main 和 wrist
是两条独立的"视频"，各自编码成各自的 latent 序列，再以 sample 维度交错（2B）进主干。
**不存在** `[main, wrist, wrist, wrist]` 这种把两个视角混进同一条序列的补帧。

**E-003-B1 的进一步精确化（2026-07-23 逐行核实）**：

- 训练 cache：`episode_latent_store.py` 按 video_key 逐视角编码**整段 episode** 为连续视频
  （"frames must never be moved into the batch dimension"），存 `[T_z, C, h, w]` +
  源帧→latent 映射；window manifest 在该 regular 网格上采窗口，帧下标 clamp 到 `>=1`
  （第 0 帧语义特殊，排除在采样网格外）。
- 在线推理：`required_frames = 1 + 4*(history_steps+1)`（h=0 时 5 帧），client 取每视角最后
  `required_frames` 帧（不足时首帧前补），逐视角过 VAE 得 `1+4k` 个 latent。
- **首帧 latent 被丢弃**：`WanPI.py:703` `regular = encoded[:, :, 1:]`，模型只消费
  "4 帧组"的 regular latent（history = regular[:-1]，current = regular[-1]）。
  因此 episode 开头模型看到的第一个 current latent 实际由 4 个首帧副本编成——
  与"首帧前补"的数据层约定自洽。

`[main, wrist, wrist, wrist]` 只是 **WanOFT 路径**特有的现象：WanOFT 把两个静态视角
当作一条 2 帧"视频"喂给 `_encode_images_vae`：

- 旧代码（WanOFT 训练/官方评测）：不补帧，2 帧直接进 VAE → T_latent=1。
  wrist 帧基本被时间压缩"吃掉"，**WanOFT 实质≈单视角模型**
  （与其模型卡 "single 3rd-person view" 一致）。
- 新代码（7 月改动后）：`wan_padded_frame_count(2)=5`，在同一条序列里把 wrist
  复制 3 次凑成 5 帧 → T_latent=2，wrist 被显式编码为第二个 latent 帧。

这正是本次回归的一部分：新代码不仅改了网格，还无意中改变了 WanOFT 的视角语义。

### A.3 sample() vs mode()

VAE 输出的是分布。旧代码 `latent_dist.sample()`（随机采样）+ `no_grad`；
新代码 `latent_dist.mode()`（取峰值）+ `inference_mode`。
差异最小但修复时一并还原。注意 `inference_mode` 产出的 tensor 进 autograd 会报错，
训练可达路径必须用 `no_grad`。

### A.4 state token —— 第二个独立的坑

07-16（b7af675）起 LIBERO eval 把机械臂状态离散化成文字 token 拼进指令
（`add_discretized_state_to_instruction`，指令变成 `<任务> [STATE] <bins> [ACTION]`）。
对训练时 `include_state` 开启的模型（starflow、v2 各线）这是修复；
对 WM4A 这种训练时没见过该格式的 ckpt 是纯毒。修复后 A/B 实锤：带 state 0/15，不带 14/15。

大白话：跟一个只懂中文的人说话，每句后面硬加一串乱码英文，他直接懵了。

## 附录 B：修复后 WanOFT vs E-003-B1-v2 —— 两条路各自成立的格式

修复不是"二选一"，而是两条路**各自使用自己训练时的格式**，由 `legacy_vae_input` 开关分流：

| | WanOFT（WM4A 官方 ckpt） | E-003-B1-v2（WanPI contft32 LoRA） |
|---|---|---|
| 框架 | WanOFT（MLP 头，~170M） | WanPI（LayerwiseFM cont.ft32 头，~634M） |
| VAE 输入 | 480×832，不补帧，sample() | 256²；每视角独立序列，历史首帧前补、尾部末帧后补到 1+4k；mode() |
| latent 网格 | (1, 30, 52)，1560 token | (2+, 16, 16)，512+ token |
| 双视角用法 | [main, wrist] 当视频 2 帧（wrist 被压缩掉，≈单视角） | 两视角各自为独立样本交错过主干（2B），cross-view adapter 显式交换视角信息 |
| 图像来源 | 在线 raw 帧现场编码 | 训练用预计算 latent cache；在线推理才现场编码（与 cache 同格式） |
| 主干权重 | Wan2.2 基座 + LIBERO 全量微调（它自己） | 从 WM4A ckpt 借 backbone（冻结）+ LoRA r32 |
| 动作头 | MLP，chunk 8 | LayerwiseFM flow-matching，chunk 32 |
| state | 不用（训练未开 include_state） | 32 维 sin/cos 连续 state 进 action head |
| 数据/任务 | LIBERO 4 套件 | RoboCasa 18 atomic |

关键解读：

- WanOFT 是"成品老专家"，修复就是把它的眼镜（旧输入格式）还给它。
- E-003-B1-v2 从老专家借了眼睛（Wan backbone 权重），但它看的"报表格式"
  （256² 双视角补帧）与老专家当年学的（480×832 单帧）不同。Wan2.2 基座预训练能力
  经 RoPE 对网格尺寸不敏感，迁移 OK；WM4A 在 LIBERO 上的"专门手感"与旧网格绑定，
  跨网格后残余价值未知。**v2 借到的是基座能力，不是 WM4A 的 LIBERO 手感**，
  后者要靠 LoRA 在 RoboCasa 数据上重新学。
- 对 v2 代码零影响：它本就走非 legacy 分支，训练 cache 与在线推理格式一致。

## 附录 C：教训（给后续所有协作者）

1. "仿真 0%" ≠ "模型不行"。有效排查顺序：离线动作对照（corr）→ 旧版代码对照 → diff 定位。
   本次三个候选（state、unnorm、VAE 输入）里两个是真 bug，只看仿真结果分不开。
2. `Wan2.py` / `wan_vae_utils.py` 的共享预处理有两类消费者
   （WanOFT 旧 ckpt = legacy 网格；WanPI/MoWA = 256²+补帧 cache 体系）。
   改默认行为 = 静默回归，除了仿真 0% 没有任何告警。
3. 评测前先确认模型训练时吃没吃 state：吃了 → `LIBERO_SEND_STATE=1`（standard）
   或 wanpi payload（自动带）；没吃 → 默认（不发）。
4. ckpt 仓库自带 `logs/`（官方逐 suite 评测日志含当时 args）与 `config.yaml`，
   是"官方口径"的一手证据，复现不出官方成绩先查它。
