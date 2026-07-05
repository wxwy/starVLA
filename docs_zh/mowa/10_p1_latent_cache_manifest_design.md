# MoWA P1 Latent Cache Manifest 设计索引

本文件是当前仓库内 P1 latent cache manifest / contract smoke 的派生设计索引，不是核心 Source-of-Truth。若与 `00_project_proposal.md`、`01_technical_survey.md`、`02_detailed_design.md` 冲突，以核心 SOT 为准；当前 checkout 中三份核心 SOT 缺失时，不得在本文件中补写或改写其结论。

## 当前冻结范围

- 当前只实现 latent cache manifest / contract smoke，不执行 Wan encoder、VAE 或真实 latent 写入。
- future frame / future latent 只能作为 target 或 cache artifact 规划，不得作为 WAM 输入。
- cache key 必须可复现，并与 dataset、task、episode、view、time window 绑定。

## 当前实现入口

- manifest helper: `starVLA/dataloader/mowa/latent_cache_manifest.py`
- manifest smoke: `tools/mowa/g0_latent_cache_manifest_smoke.py`
- P1 contract smoke: `tools/mowa/p1_latent_cache_contract_smoke.py`
- Data Gate 单测: `tests/mowa/test_mowa_data_gate.py`

## 当前报告口径

- `latent_shape_status`、`cache_hash_status`、`encoder_status` 未实测时保持 `Data Gate`。
- contract smoke 只检查路径、cache key、artifact 规划和 future-action-not-input 约束。
- 不得把 plan-only manifest 当作真实 latent cache 可用证据。

## 未解决项

- 真实 Wan latent cache builder 需要单独人工确认后才能执行。
- latent shape、编码耗时、磁盘预算和 cache 命中率仍为 `Data Gate`。
- 若进入 P1-b0 训练，必须先补真实 cache builder、cache manifest 校验和 resume/invalid-cache 处理。
