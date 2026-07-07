# MoWA Future Latent Cache Manifest 设计索引

本文件是当前仓库内 future latent cache manifest / contract smoke 的派生设计索引，不是核心 Source-of-Truth。若与 `00_project_proposal.md`、`01_technical_survey.md`、`02_detailed_design.md` 冲突，以核心 SOT 为准；若未来核心 SOT 再次缺失，也不得在本文件中补写或改写其结论。

## 当前冻结范围

- 当前只实现 latent cache manifest / contract smoke，不执行 Wan encoder、VAE 或真实 latent 写入。
- future frame / future latent 只能作为 target 或 cache artifact 规划，不得作为 WAM 输入。
- cache key 必须可复现，并与 dataset、task、episode、view、time window 绑定。

## 当前实现入口

- manifest helper: `starVLA/dataloader/mowa/latent_cache_manifest.py`
- manifest smoke: `tools/mowa/g0_latent_cache_manifest_smoke.py`
- future latent contract smoke: `tools/mowa/latent_cache_contract_smoke.py`
- future latent builder design smoke: `tools/mowa/latent_cache_builder_design_smoke.py`
- future latent prior interface smoke: `tools/mowa/future_latent_prior_interface_smoke.py`
- Data Gate 单测: `tests/mowa/test_mowa_data_gate.py`

## 当前报告口径

- `latent_shape_status`、`cache_hash_status`、`encoder_status` 未实测时保持 `Data Gate`。
- contract smoke 只检查路径、cache key、artifact 规划和 future-action-not-input 约束。
- builder design smoke 只在 contract 规划之上补一层 config/input-policy 审计，不执行真实 encoder/VAE/cache writer。
- 不得把 plan-only manifest 当作真实 latent cache 可用证据。

## 未解决项

- 真实 Wan latent cache builder 需要单独人工确认后才能执行。
- latent shape、编码耗时、磁盘预算和 cache 命中率仍为 `Data Gate`。
- 若进入 future latent prior 训练，必须先补真实 cache builder、cache manifest 校验和 resume/invalid-cache 处理。

## P1 实现框架总览

本节用于后续代码实现交接。当前只冻结实现框架、接口边界和验收顺序，不声明真实 latent cache 已经可用，不启动 E-003/E-004 训练。

P1 分三条独立链路推进：

| 链路 | 实验 | 目标 | 当前状态 | 进入训练前必须补齐 |
|---|---|---|---|---|
| latent cache builder | E-003 前置 | 把 current/future/history RGB 编码为可复用 Wan latent cache | plan / contract smoke | 真实 encoder adapter、cache writer、manifest 校验、回读测试 |
| P1-b0 future latent prior | E-003 | `current_latent + text -> predicted_future_latent` | interface smoke | 真实 cache dataloader、latent loss、训练配置、preview |
| P1-b1 HLC-GCI | E-004/E-005 | 引入 history RGB latent + robot history 条件 | module/interface smoke | history cache、state/action history sampler、framework 接入、shuffle sanity |

P1 与 P0 的关系：

- E-001 继续只使用两个已可构造 future heads，不等待 P1。
- P1 cache / prior / HLC-GCI 不依赖 P0 新增 head 标签。
- P1 可以复用 G0 的 atomic core 数据域、window sampler、leakage invariant 和 action bridge，但不能把 P0 未解 mask 的标签作为 P1 前置条件。

## P1 阶段划分

### M3-001：真实 latent cache builder

目标：把现有 plan-only manifest 推进为可落盘、可回读、可校验的 future latent cache。

建议新增或扩展文件：

| 文件 | 职责 |
|---|---|
| `starVLA/dataloader/mowa/latent_cache_manifest.py` | 保留 manifest / contract dataclass，增加真实 cache artifact metadata schema。 |
| `starVLA/dataloader/mowa/latent_cache_builder.py` | 新增真实 builder：读取视频窗口、调用 encoder adapter、写 cache、写 manifest。 |
| `starVLA/dataloader/mowa/latent_cache_dataset.py` | 新增只读 cache loader：按 sample key 回读 current/future/history latent。 |
| `tools/mowa/build_future_latent_cache.py` | CLI：构建 cache，默认 dry-run，真实写入需要显式 `--execute`。 |
| `tools/mowa/validate_future_latent_cache.py` | CLI：校验 manifest、cache 文件、shape、hash、缺失项和坏文件。 |
| `tests/mowa/test_mowa_latent_cache_builder.py` | 单测：fake encoder、fake video、cache write/read/hash/mask。 |

真实 builder 最小接口：

```python
@dataclass(frozen=True)
class MoWALatentCacheBuildConfig:
    dataset_path: Path
    cache_root: Path
    encoder_name: str
    video_keys: tuple[str, ...]
    current_window: MoWALatentWindowSpec
    future_window: MoWALatentWindowSpec
    history_window: MoWALatentWindowSpec | None
    overwrite: bool = False
    dry_run: bool = True

@dataclass(frozen=True)
class MoWALatentCacheArtifact:
    cache_key: str
    cache_path: Path
    episode_index: int
    anchor_index: int
    video_key: str
    split: str
    window_role: str  # current | future | history
    frame_indices: tuple[int, ...]
    latent_shape: tuple[int, ...]
    dtype: str
    encoder_name: str
    encoder_version: str
    source_video_sha256: str
    latent_sha256: str
```

cache 文件建议先用 `torch.save()` 写单 artifact dict，后续如规模过大再切 shard。artifact dict 至少包含：

```python
{
    "latent": Tensor,
    "metadata": MoWALatentCacheArtifact as dict,
}
```

cache key 必须包含：

- dataset absolute/resolved identity 或 stable dataset id
- episode index
- anchor index
- video key
- window role
- frame index tuple
- encoder name / version
- latent normalization version

### M3-002：P1-b0 future latent prior

目标：把当前 `MoWAFutureLatentPrior` 从 synthetic tensor interface smoke 推进到真实 cache supervision。

建议扩展文件：

| 文件 | 职责 |
|---|---|
| `starVLA/model/modules/mowa/future_latent_prior.py` | 保留禁止 `history_latent` 的 invariant；增加 mask、loss config、optional cosine loss。 |
| `configs/mowa/mowa_e003_future_latent_prior_candidate.yaml` | 新增 E-003 训练草案，launch guard 默认关闭。 |
| `tools/mowa/e003_future_latent_prior_config_preview.py` | 打印最终 cache root、latent shape、loss、batch、warmup、wandb、launch guard。 |
| `tools/mowa/e003_future_latent_prior_train_dry_run.py` | 只构建 batch 和 model forward，不启动正式训练。 |
| `tests/mowa/test_mowa_future_latent_prior.py` | 单测：shape、loss finite、history_latent reject、mask、config preview。 |

P1-b0 batch contract：

```python
{
    "mowa_current_latent": Tensor[B, ...],
    "mowa_text_hidden": Tensor[B, D_text],
    "mowa_future_latent_target": Tensor[B, ...],
    "mowa_future_latent_mask": Tensor[B],
    "mowa_latent_cache_keys": list[str],
    "mowa_anchor_indices": Tensor[B],
}
```

P1-b0 禁止字段：

- `mowa_history_latent`
- `future_action` 作为 WAM input
- `future_rgb` 作为 model input
- `future_done` / `future_reward` 作为 model input

P1-b0 loss 默认只开：

```text
L = L_action + lambda_future_latent * L_future_latent_mse
```

`cosine`、`latent_alignment`、`action_relevance_align` 只能作为 opt-in，且默认关闭。

### M4-001：P1-b1 HLC-GCI

目标：在 P1-b0 可训练后，再接 history latent 和 robot history 条件。E-004 之前不应把 HLC-GCI 接进 E-003。

建议扩展文件：

| 文件 | 职责 |
|---|---|
| `starVLA/model/modules/mowa/hlcgci.py` | `C_hist/h_hist/g_hist` 模块实现与 gate 初始化。 |
| `starVLA/model/modules/mowa/future_latent_action_bridge.py` | predicted future latent 到 action bridge features 的投影。 |
| `configs/mowa/mowa_e004_hlc_gci_candidate.yaml` | E-004 训练草案，依赖 E-003 cache/prior 证据。 |
| `tools/mowa/e004_hlc_gci_interface_smoke.py` | shape/gate/invariant smoke。 |
| `tests/mowa/test_mowa_hlc_gci.py` | 单测：gate range、history mask、future leakage reject、shuffle pair。 |

P1-b1 batch contract：

```python
{
    "mowa_current_latent": Tensor[B, ...],
    "mowa_text_hidden": Tensor[B, T_text, D],
    "mowa_visual_history_latent": Tensor[B, T_hist, ...],
    "mowa_robot_state_history": Tensor[B, T_hist, D_state],
    "mowa_robot_action_history": Tensor[B, T_hist, D_action],
    "mowa_history_mask": Tensor[B, T_hist],
    "mowa_future_latent_target": Tensor[B, ...],
    "mowa_future_latent_mask": Tensor[B],
}
```

P1-b1 输出 contract：

```python
{
    "predicted_future_latent": Tensor[B, ...],
    "C_hist": Tensor[B, K_hist, D],
    "h_hist": Tensor[B, D],
    "g_hist": Tensor[B, 1] or Tensor[B, K_hist, 1],
    "layerwise_condition_features": tuple[Tensor[B, N_bridge, D_action], ...],
}
```

gate 初始化要求：

- `g_hist` 初始均值接近 0，不应一开始覆盖 text/current condition。
- 单测只断言范围和初始化趋势，不写死训练后 gate 应打开。
- E-005 shuffle sanity 只在 E-004 有有效 checkpoint 后执行。

## Encoder Adapter 边界

真实 Wan encoder / VAE 接入必须通过 adapter 封装，不能把具体第三方模型调用散落在 builder、dataset 和训练脚本中。

建议接口：

```python
class MoWALatentEncoderAdapter(Protocol):
    encoder_name: str
    encoder_version: str
    output_dtype: torch.dtype

    def encode_video_window(
        self,
        frames: torch.Tensor,
        *,
        video_key: str,
        window_role: str,
    ) -> torch.Tensor:
        ...
```

第一版可以提供 `FakeLatentEncoderAdapter` 用于单测和 smoke：

- 输入 fake frames，输出确定性 tensor。
- 不依赖 Wan 权重。
- 用于测试 cache key、shape、hash、回读、invalid-cache 处理。

真实 Wan adapter 另行接入，且执行前需要人工确认，因为它可能加载大模型、占用显存并显著影响当前 E-001 训练机器。

## Cache Manifest 与 Resume 策略

manifest 建议使用 JSONL，便于追加和局部恢复：

```text
cache_root/
  manifest.jsonl
  artifacts/
    <cache_key>.pt
  reports/
    build_report.json
    validate_report.json
```

每条 manifest 必须能独立判断：

- source video 是否存在
- cache 文件是否存在
- latent hash 是否匹配
- latent shape 是否匹配当前 config
- encoder/version 是否匹配
- window spec 是否匹配

invalid-cache 处理：

| 情况 | 默认动作 |
|---|---|
| source video 缺失 | report failure，不写 cache |
| cache 缺失 | dry-run 记录 missing；execute 时构建 |
| latent shape 不匹配 | 标记 invalid，不直接复用 |
| encoder version 不匹配 | 标记 stale，不直接复用 |
| hash 不匹配 | 标记 corrupted，需要重建 |
| duplicate cache key | No-Go，必须修 key 规则 |

## Data Gate 与防泄漏检查

P1 进入 E-003 训练前必须通过以下 gate：

| gate | 必须检查 |
|---|---|
| cache path gate | 所有 sampled entries 的 source video 可定位，cache key 无重复。 |
| encoder gate | encoder name/version、latent dtype、latent shape、编码耗时有报告。 |
| cache artifact gate | cache 文件存在、可回读、hash 可复现、invalid-cache 可检测。 |
| batch whitelist gate | batch input keys 不包含 future rgb / future action / future reward / future done。 |
| boundary gate | history/current/future window 不跨 episode。 |
| train/infer consistency gate | WAM Hz、history window、future horizon、action chunk 对齐。 |

P1-b0 防泄漏单测必须覆盖：

- `future_latent_target` 只进入 loss target，不进入 model forward input。
- `history_latent` 传给 P1-b0 会报错。
- future action 只作为 action target 或 invariant 检查对象。
- batch keys whitelist 拒绝 future RGB input。

P1-b1 额外覆盖：

- robot history action 的最大时间戳 `<= anchor_t`。
- visual history latent 的最大 frame index `<= anchor_t`。
- shuffled robot history 可构造 pair，但不声明性能下降，性能下降留给 E-005。

## 最小实现顺序

建议后续模型按以下顺序实现，不要直接跳到 E-003 训练：

1. 补 `FakeLatentEncoderAdapter` 与真实 cache artifact schema。
2. 补 `build_future_latent_cache.py --dry-run/--execute`，先只在 1 个 episode、1 个 view 上用 fake encoder 写 cache。
3. 补 `validate_future_latent_cache.py`，覆盖 missing / stale / corrupted / duplicate。
4. 补 `MoWALatentCacheDataset` 或等价 loader，把真实 cache 转成 P1-b0 batch contract。
5. 扩展 `MoWAFutureLatentPrior.compute_loss()` 支持 mask 和 loss config，保留 history reject。
6. 新增 E-003 config preview 和 train dry-run，只做 forward/loss，不启动正式训练。
7. 只有 E-003 dry-run 通过后，再补 HLC-GCI 的真实 history cache 和 E-004 接口。

## 当前交接结论

- 当前 P1 不是训练 ready。
- P1-b0 的模型接口已存在，但只吃 synthetic tensor。
- latent cache 目前只有 manifest / contract / design smoke，没有真实 tensor artifact。
- 下一步最小可实现目标是：用 fake encoder 建立真实 cache writer/validator/loader 闭环；该闭环通过后，再替换为真实 Wan adapter。
