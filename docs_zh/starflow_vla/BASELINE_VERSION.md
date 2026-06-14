# StarFlow-VLA 基线版本记录

## Task ID
P0-M1

## Scope
本文件记录当前仓库可见的 StarVLA 基线信息，用于 P0 前置锁定。不修改 git 分支，不回退 commit，不加载模型。

## Current Workspace Before Baseline Switch
| 字段 | 取值 |
| --- | --- |
| Git branch | `starVLA_dev` |
| Current HEAD | `f723d4b5629cd0f5fb8bc8a562dd744f7a79b552` |
| Current HEAD short | `f723d4b` |
| Current HEAD subject | `Add LIBERO eval pipeline to code walkthrough` |
| Current HEAD date | `2026-05-25 17:07:53 +0800` |
| Python package version | `1.0.1` |
| LIBERO config schema | `version_id: "0.21"` |

## Current Workspace After Baseline Switch
| 字段 | 取值 |
| --- | --- |
| Git state | detached HEAD |
| Current HEAD | `42170b2a4df3877ccf6581948e2198d37c363c7f` |
| Current HEAD short | `42170b2` |
| Current HEAD subject | `[fix eval] (Robocasa_tabletop): make state input and unnorm_key configurable in eval client (#374)` |
| Current HEAD date | `2026-06-12 11:40:16 +0800` |
| Fetched from | `https://github.com/starVLA/starVLA.git` `starVLA_dev` |
| Temporary fetched ref | `refs/remotes/starvla-official/starVLA_dev` |
| Preserved local change | `SESSION.md` saved in `stash@{0}` as `codex-preserve-session-before-42170b2` |

## Current Workspace After Official Merge
| 字段 | 取值 |
| --- | --- |
| Git branch | `merge-official-starvla-dev` |
| Current HEAD | `5d94274a2e6131a5dbbc49546fc662d2ccbe5117` |
| Current HEAD short | `5d94274` |
| Current HEAD subject | `Remove root TODO` |
| Current HEAD date | `2026-06-14 19:19:14 +0800` |
| Merge commit | `9a8f5057888faf45eab62fddb97a0e9b84c44a90` |
| Merge commit subject | `Merge official starVLA_dev while preserving fork changes` |
| Official upstream ref | `starvla-official/starVLA_dev` |
| Official upstream HEAD | `cdf5434438f4449cff85e3588956f7706a5c9cc3` |
| Official upstream subject | `[chore] Enhance citation details for StarVLA article (#377)` |
| Origin branch | `origin/merge-official-starvla-dev` |
| Preserved local untracked dirs | `.libero/`, `LIBERO/` |

## Design Anchor
| 字段 | 取值 |
| --- | --- |
| Design document anchor commit | `42170b2a4df3877ccf6581948e2198d37c363c7f` |
| Local availability | 已可解析并已切换到该 commit |
| Evidence | `git show -s 42170b2a4df3877ccf6581948e2198d37c363c7f` 返回目标 commit 信息 |

## Baseline Decision
P0 文档设计锚点仍为 `42170b2a4df3877ccf6581948e2198d37c363c7f`。当前代码工作区已在 `merge-official-starvla-dev` 上合并官方 `starVLA_dev` 最新 `cdf5434438f4449cff85e3588956f7706a5c9cc3`，并保留 fork 中已有 LIBERO 训练、checkpoint、eval 与文档资产。进入 P0-M2 代码实现前，应以当前合并分支为候选实现基线，并对与 P0 相关的 framework registry、QwenPI_v3、LayerwiseFM、checkpoint loader 和 dataloader 进行一次 compatibility audit。

## Commands
```bash
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git log -1 --format='%H%n%h %ci %s'
git show -s --format='%H%n%h %ci %s' 42170b2a4df3877ccf6581948e2198d37c363c7f
git fetch https://github.com/starVLA/starVLA.git starVLA_dev:refs/remotes/starvla-official/starVLA_dev
git switch --detach 42170b2a4df3877ccf6581948e2198d37c363c7f
git log -1 --format='%H%n%h %ci %s' starvla-official/starVLA_dev
git log -1 --format='%H%n%h %ci %s' origin/merge-official-starvla-dev
rg -n "^version =|version_id" pyproject.toml examples/LIBERO/train_files/starvla_cotrain_libero.yaml
```

## Not Run
未运行 StarFlowVLA 代码测试、真实模型加载、训练、评测或部署。
