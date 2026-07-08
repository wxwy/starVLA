# MoWA Future Latent Cache Manifest 设计索引

本文件是当前仓库内 future latent cache manifest / contract smoke 的派生设计索引，不是核心 Source-of-Truth。若与 `00_project_proposal.md`、`01_technical_survey.md`、`02_detailed_design.md` 冲突，以核心 SOT 为准；若未来核心 SOT 再次缺失，也不得在本文件中补写或改写其结论。

## 1. 设计结论

默认采用 **Episode-level latent cache + Window manifest references** 作为 P1 / HLC-GCI 的 latent cache 与 window sampling 主线。

核心分工：

- **cache 只负责保存 episode-level 可复用事实**：每个 episode 一个 latent store 文件，只提前缓存高维视觉信息的 latent，不按 anchor 预生成 history/current/future artifact。
- **manifest 负责定义训练样本切片**：每一行定义一个训练 sample，包含 `episode_id`、`anchor_index`、`history_indices`、`current_index`、`future_indices`、`robot_state_indices`、`history_action_indices`、`action_chunk_indices` 等。
- **label sidecar 负责保存监督标签**：P0 labels、future latent target、action chunk 等监督信号与 visual latent store 分离。
- **dataloader 负责组装模型输入并执行 no-leakage validation**：根据 manifest 从 episode latent store 动态 gather visual latent，同时从同一 episode store 读取 robot state/action/language，组装成 `WindowLatentSample`。

这一分工确保：

- visual latent 与原始 episode frame 绑定，不与某个 anchor/window/sample 绑定；
- 同一个 episode latent 可被不同 `history_steps`、`stride`、`WAM Hz`、`future horizon` 复用；
- 后续改实验时只需重建 window manifest，不需要重新编码视觉 latent；
- 支持 Rec-HLC、event-aware keyframes、summary tokens 等扩展，而不改变底层 cache 文件。

当前实现入口：

- episode latent store builder: `tools/mowa/build_episode_latent_store.py`
- episode latent store reader / validator: `starVLA/dataloader/mowa/episode_latent_store.py`
- window manifest builder: `tools/mowa/build_window_manifest.py`
- window manifest loader: `starVLA/dataloader/mowa/window_manifest.py`
- dataloader / sample assembler: `starVLA/dataloader/mowa/window_latent_sample.py`
- label sidecar builder: `tools/mowa/build_label_sidecar.py`
- 核心单测: `tests/mowa/test_episode_level_latent_cache.py`
- 遗留 window-level artifact builder（方案 A 兼容）: `tools/mowa/build_future_latent_cache.py`
- 遗留 manifest / contract smoke: `tools/mowa/latent_cache_contract_smoke.py`
- E-003 config preview: `tools/mowa/e003_future_latent_prior_config_preview.py`
- E-003 train dry-run: `tools/mowa/e003_future_latent_prior_train_dry_run.py`
- E-004 HLC-GCI interface smoke: `tools/mowa/e004_hlc_gci_interface_smoke.py`

当前报告口径：

- `latent_shape_status`、`cache_hash_status`、`encoder_status` 未实测时保持 `Data Gate`。
- contract smoke 只检查 episode latent store 路径、window manifest schema 和 future-action-not-input 约束。
- 不得把 plan-only manifest 当作真实 latent cache 可用证据。
- 所有具体 latent shape 必须标注为 `Data Gate / profiling 后确认`，不得由模型推断或编造。

## 2. 三种 cache 方案对比表

| 方案 | 默认推荐 | 关键特征 | 适用场景 |
|---|---|---|---|
| **C：Episode-level latent cache + Window manifest references** | ✅ 默认主线 | 每 episode 一个 latent store；manifest 每行定义一个 window/sample；visual latent 与 anchor 解耦 | P1-b0 / P1-b1 / HLC-GCI / Rec-HLC / event-aware keyframes |
| **A：Role-aware 2D history artifact** | ⚠️ 短期兼容 | 每 `(episode, anchor, video_key, window_role)` 一个 `.pt` artifact；history 存 `[T, D]`，current/future 存 `[D]` | 已有 window_role artifact 体系时可短期兼容，但不作为默认主线 |
| **B：Per-step history artifacts** | ❌ 不推荐 | history 每帧单独存 1D artifact，加载时排序 stack | 文件数量多、需要 step_index、加载复杂、仍与 anchor 绑定 |

方案选择原因：

- **方案 C**：latent 与原始 episode frame 绑定，不与某个 anchor/window/sample 绑定；同一个 episode latent 可被不同 `history_steps`、`stride`、`WAM Hz`、`future horizon` 复用；后续改实验时只需重建 window manifest，不需要重新编码视觉 latent；支持 Rec-HLC、event-aware keyframes、summary tokens；避免 per-anchor 重复存 history window。
- **方案 A**：如果当前已有 window_role artifact 体系，可以作为短期兼容方案；但它是 sample/window-level cache，history/current/future 已经和 anchor 绑定；后续变更 history window 或 future horizon 时可能要重建 cache。
- **方案 B**：不推荐；它会显著增加文件数量；需要额外 `step_index` / `frame_index`；加载时还要排序再 stack；仍然与 anchor/window 绑定，既不如方案 C 通用，也不如方案 A 简洁。

## 3. Episode-level latent store schema

默认 latent cache 目录结构：

```text
latent_cache/
  ep_000001.h5
  ep_000002.h5
  ep_000003.h5
```

每个 episode 一个 HDF5 文件。文件内建议 schema：

```text
episode_000123.h5
├── /latents/
│   ├── agentview_rgb              # 视觉 latent
│   └── robot0_eye_in_hand         # 多相机视觉 latent
├── /timestamps/
│   ├── agentview_rgb
│   ├── robot_state
│   └── action
├── /indices/
│   ├── frame_indices
│   ├── robot_state_indices
│   └── action_indices
├── /valid/
│   ├── agentview_rgb
│   └── robot0_eye_in_hand
├── /robot/
│   ├── state
│   ├── action
│   ├── eef_pose
│   └── gripper_state
├── /language/
│   ├── instruction
│   └── input_ids                  # optional
├── /sim/                          # optional, for label generation / debug
│   ├── object_poses
│   ├── fixture_joint_states
│   ├── contact_flags
│   ├── task_predicates
│   └── success
└── attrs:
    episode_id
    task_name
    scene_id
    demo_id
    latent_model
    latent_model_version
    latent_type
    latent_shape_per_frame
    flatten_policy
    obs_fps
    action_hz
    wam_hz
    downsample_policy
    time_order = oldest_to_latest
    created_at
```

提前转 latent 的范围：

- **需要提前转 latent**：
  - RGB frame
  - multi-view RGB
  - future RGB target
- **不需要提前转 latent**：
  - robot state
  - action
  - gripper state
  - task id
  - language
  - predicate
  - success / failure
  - P0 labels

robot state / action 可以存到同一个 episode h5 的 `/robot/` 组中，但**不能**混入 `/latents/{video_key}`。它们应保留原始低维形式，由模型中的 projector / adapter 转成 robot history latent / condition tokens。

visual latent 的 shape 不要写死，允许三种形式：

```text
[T, C, h, w]   # VAE spatial latent
[T, N, D]      # patch/token latent
[T, D]         # pooled vector latent
```

必须在 metadata 中记录：

```text
latent_type
latent_shape_per_frame
flatten_policy
dtype
latent_model
latent_model_version
time_order
```

示例 attrs：

```json
{
  "latent_model": "Wan2.2-VAE",
  "latent_model_version": "TBD",
  "latent_type": "vae_spatial",
  "latent_shape_per_frame": ["C", "h", "w"],
  "flatten_policy": "none",
  "dtype": "float16",
  "time_order": "oldest_to_latest"
}
```

所有具体 shape 都必须标注为 `Data Gate / profiling 后确认`，不得由模型推断或编造。

## 4. Window manifest schema

window manifest 单独存放：

```text
window_manifest/
  train_windows.parquet
  val_windows.parquet
```

每一行对应一个训练 sample，建议字段：

```text
sample_id
episode_id
episode_latent_path
task_name
anchor_index
anchor_timestamp
video_key
history_indices
current_index
future_indices
robot_state_indices
history_action_indices
action_chunk_indices
label_index
label_sidecar_path
history_seconds
future_seconds
wam_hz
history_stride
split
status
```

示例：

```json
{
  "sample_id": "ep000123_a000008",
  "episode_id": "ep000123",
  "episode_latent_path": "latent_cache/ep_000123.h5",
  "task_name": "open_drawer",
  "anchor_index": 8,
  "anchor_timestamp": 2.0,
  "video_key": "agentview_rgb",

  "history_indices": [2, 4, 6, 8],
  "current_index": 8,
  "future_indices": [12, 13, 14, 15],

  "robot_state_indices": [40, 45, 50, 55, 60],
  "history_action_indices": [40, 45, 50, 55],
  "action_chunk_indices": [60, 61, 62, 63, 64, 65, 66, 67],

  "label_sidecar_path": "labels/ep_000123.jsonl",
  "label_index": 8,

  "history_seconds": 4.0,
  "future_seconds": 2.0,
  "wam_hz": 4,
  "history_stride": 2,
  "split": "train",
  "status": "target"
}
```

关键约定：

- `history_steps`、`stride`、`future_horizon`、`WAM Hz` 等实验参数**不写死**在 latent cache 中；它们由 manifest / config 决定。
- manifest 可以按实验需求重新生成，而无需重新编码视觉 latent。

## 5. Dataloader 数据流

dataloader 职责：

1. 读取 window manifest；
2. 根据 `episode_latent_path` 打开 episode latent store；
3. 根据 `history_indices` / `current_index` / `future_indices` gather visual latent；
4. 根据 `robot_state_indices` / `history_action_indices` / `action_chunk_indices` 读取低维状态与动作；
5. 读取 language instruction；
6. 读取 label sidecar 中的 P0 labels；
7. 执行 index validation / leakage validation；
8. 返回统一 `WindowLatentSample`。

建议输出结构：

```python
@dataclass
class WindowLatentSample:
    episode_id: str
    sample_id: str
    anchor_index: int

    current_latent: Tensor
    history_latents: Tensor
    future_latents: Tensor

    robot_state_history: Tensor
    history_actions: Tensor
    action_chunk: Tensor

    language: str | dict
    labels: dict
    metadata: dict
```

shape 允许：

```text
current_latent: [D] or [N, D] or [C, h, w]
history_latents: [H, D] or [H, N, D] or [H, C, h, w]
future_latents: [T_f, D] or [T_f, N, D] or [T_f, C, h, w]
```

具体 shape 由 `latent_type` 和 `flatten_policy` 决定，dataloader 只负责 gather 和 optional flatten，不做隐式 reshape 假设。

## 6. Index validation / leakage invariant

强制加入以下校验：

```python
assert max(history_indices) <= anchor_index
assert current_index == anchor_index
assert min(future_indices) > anchor_index
assert indices_are_monotonic(history_indices)
assert indices_are_monotonic(future_indices)
assert not crosses_episode_boundary(history_indices, future_indices)
assert all_indices_exist_in_latent_store(...)
```

如果最终设计决定 history 不包含 anchor 自身，则改成：

```python
assert max(history_indices) < anchor_index
```

但必须全文统一。

输入泄漏禁止：

- future action label 不得进入 WAM input；
- future predicate / future success / future failure flag 不得进入 WAM input；
- future frame / future latent 只能作为 supervision target；
- `action_chunk_indices` 只用于 action supervision，不得作为 WAM history input；
- P0 labels 只能作为 supervision label，不得作为 model input。

## 7. Label sidecar 边界

P0 labels 不放入 visual latent，也不作为模型输入。

P0 labels 包括：

```text
failure_risk
subgoal_feasibility
manipulation_readiness
```

建议存放位置：

```text
labels/
  ep_000123.jsonl
```

或统一 sidecar 表中。

这些 label 可以使用 future 信息生成，但训练输入不能使用 future 信息。

## 8. 与 HLC-GCI 的接口关系

Episode-level latent cache **不等于** HLC-GCI 压缩；它只是保存 per-frame visual latent。

HLC-GCI 的输入来自 dataloader 动态组装：

```text
visual history latent:
  history_latents

text condition tokens:
  language instruction 经 text encoder / tokenizer 得到

robot history latent:
  robot_state_history + history_actions 经 trainable projector / adapter 得到
```

HLC-GCI 输出：

```text
C_hist
h_hist
g_hist
```

注入路径仍然保持当前设计：

```text
C_wan_cond = concat(C_text, C_current, g_hist · Project(C_hist))
```

不要因为 cache 重构改变 P0 / P1-b0 / P1-b1 / P2 阶段定义。

## 9. 对当前文档中 Role-aware 2D history artifact 相关段落的替换建议

此前 `10_future_latent_cache_manifest_design.md` 中 M3-001 的 `MoWALatentCacheArtifact` schema、按 `(episode, anchor, video_key, window_role)` 存储 `.pt` artifact、以及 `MoWALatentCacheDataset` 按 sample key 分组回读 current/future/history 的设计，已被本设计的 **Episode-level latent store + Window manifest references** 替代。

需要替换或删除的当前代码/文档：

- `starVLA/dataloader/mowa/latent_cache_builder.py` 中按 window_role 生成 `.pt` artifact 的逻辑：改为生成每 episode 一个 HDF5 latent store。
- `starVLA/dataloader/mowa/latent_cache_dataset.py` 中按 `(episode, anchor, video_key)` 分组回读三个 artifact 的逻辑：改为读取 window manifest 后动态 gather。
- `tools/mowa/build_future_latent_cache.py`：改为 `tools/mowa/build_episode_latent_store.py`。
- `tools/mowa/validate_future_latent_cache.py`：改为校验 episode latent store + window manifest。
- `tests/mowa/test_mowa_latent_cache_builder.py` 中针对 role-aware 2D artifact 的测试：改为测试 episode store、window manifest、no-leakage validation。

保留但调整范围：

- `starVLA/dataloader/mowa/latent_cache_manifest.py` 中的 manifest/contract dataclass：保留作为过渡，但新增 `MoWAWindowManifest` / `MoWAWindowManifestEntry` 作为默认接口。
- E-003 / E-004 的 config preview 和 smoke 脚本：把 `cache_root/*.pt` 检查改为 `latent_cache/ep_*.h5` + `window_manifest/*.parquet` 检查。

## 10. 给 Codex 的实现任务清单

1. 新增 episode-level latent store 生成脚本 `tools/mowa/build_episode_latent_store.py`。
2. 使用每 episode 一个 HDF5 文件保存 visual latent，路径格式 `latent_cache/ep_{episode_id:06d}.h5`。
3. 保留 robot state/action/language 原始数据，不提前转 latent；存放到同一 h5 的 `/robot/` 和 `/language/` 组。
4. 新增 window manifest 生成脚本 `tools/mowa/build_window_manifest.py`，输出 `window_manifest/train_windows.parquet` / `val_windows.parquet`。
5. 实现 dataloader 动态 gather history/current/future latent，入口 `starVLA/dataloader/mowa/window_latent_sample.py`。
6. 实现 no-leakage validation，覆盖 history/current/future index 边界和 future action / label 不进入 input。
7. 接入 label sidecar，入口 `tools/mowa/build_label_sidecar.py`。
8. 增加 unit tests：
   - `history_indices <= anchor_index`
   - `current_index == anchor_index`
   - `future_indices > anchor_index`
   - `action_chunk_indices` 不进入 WAM input
   - label fields 不进入 WAM input
   - multi-view `video_key` 存在
   - latent shape 与 metadata 一致
   - valid mask 检查

实现时禁止：

- 重新讨论项目总体路线；
- 改 P0 / P1-b0 / P1-b1 / P2 阶段定义；
- 把 Role-aware 2D history artifact 继续写成默认主线；
- 提出每帧一个 `.npy` 小文件；
- 把 robot state/action/language 也提前转成 latent；
- 把 future action / future predicate / future success flag 放进模型 input；
- 未经 profiling 写死 Wan VAE latent shape。
