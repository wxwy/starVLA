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

## Design Anchor
| 字段 | 取值 |
| --- | --- |
| Design document anchor commit | `42170b2a4df3877ccf6581948e2198d37c363c7f` |
| Local availability | 已可解析并已切换到该 commit |
| Evidence | `git show -s 42170b2a4df3877ccf6581948e2198d37c363c7f` 返回目标 commit 信息 |

## Baseline Decision
P0 实现基线已切换为设计文档锚点 `42170b2a4df3877ccf6581948e2198d37c363c7f`。进入 P0-M2 前建议基于该 commit 创建专用实现分支，避免在 detached HEAD 上直接累积代码实现。

## Commands
```bash
git rev-parse --abbrev-ref HEAD
git rev-parse HEAD
git log -1 --format='%H%n%h %ci %s'
git show -s --format='%H%n%h %ci %s' 42170b2a4df3877ccf6581948e2198d37c363c7f
git fetch https://github.com/starVLA/starVLA.git starVLA_dev:refs/remotes/starvla-official/starVLA_dev
git switch --detach 42170b2a4df3877ccf6581948e2198d37c363c7f
rg -n "^version =|version_id" pyproject.toml examples/LIBERO/train_files/starvla_cotrain_libero.yaml
```

## Not Run
未运行 StarFlowVLA 代码测试、真实模型加载、训练、评测或部署。
