# StarVLA 分支与发布策略

本文档描述 StarVLA 仓库中分支、发布和版本标签的管理方式。

## 分支模型

StarVLA 遵循受 GitHub Flow 启发的**双分支模型**：

| 分支 | 用途 | 稳定性 |
|--------|---------|-----------|
| `starVLA` | 稳定发布分支。包含已验证的生产就绪代码。 | ✅ 稳定 |
| `starVLA_dev` | 活跃开发分支。新功能和改进首先合入此处。 | ⚠️ 可能暂时不稳定 |

```
feature/xxx ──► starVLA_dev ──► starVLA (稳定发布)
   fix/xxx ──┘                      │
                                    ▼
                               tag: vX.Y.Z
```

### 为什么用双分支（而非 GitFlow）？

- **简单**：贡献者只需面向一个分支（`starVLA_dev`）。
- **快速迭代**：功能通过 `starVLA_dev` 快速触达开发者。
- **稳定基线**：需要可复现结果的用户可以始终使用 `starVLA`。

我们刻意避免 GitFlow（`develop`、`release/*`、`staging` 等）的复杂性——它会增加与我们的发布节奏不匹配的额外开销。

## 合并流程

### 1. 功能 / 修复开发

所有贡献从 `starVLA_dev` 分叉的**功能分支**开始：

```bash
git checkout starVLA_dev
git pull origin starVLA_dev
git checkout -b feat/my-new-feature
```

### 2. Pull Request → `starVLA_dev`

- 提交 PR，目标分支为 `starVLA_dev`。
- 在修改的文件上通过 Black + Ruff 检查（全仓库 `make check` 目前因历史 lint 积压预期会失败；参见 [PR_readme.md](PR_readme.md#4-提交前检查清单)）。
- 获得至少**一位维护者批准**。
- 通过 **Squash Merge** 合并以保持提交历史整洁。

### 3. Hotfix

针对稳定分支上的紧急 bug：

```bash
git checkout starVLA
git checkout -b hotfix/fix-critical-bug
# ... 修复并测试 ...
# PR → starVLA（直接）
# 然后 cherry-pick 或合并回 starVLA_dev
git checkout starVLA_dev
git cherry-pick <hotfix-commit>
```

## 分支命名规范

| 前缀 | 用途 | 示例 |
|--------|----------|---------|
| `feat/` | 新功能或能力 | `feat/cosmos-world-model` |
| `fix/` | Bug 修复 | `fix/oom-in-gr00t-training` |
| `docs/` | 仅文档 | `docs/add-libero-tutorial` |
| `refactor/` | 代码重构（行为不变） | `refactor/dataloader-registry` |
| `exp/` | 实验 / 研究分支 | `exp/diffusion-policy-head` |
| `hotfix/` | 稳定分支紧急修复 | `hotfix/checkpoint-loading-crash` |

**规则：**
- 小写加连字符：`feat/my-feature`（而非 `feat/MyFeature`）。
- 名称简短但有描述性。
- 适用时包含 issue 编号：`fix/192-action-stats-cache`。

## 社区 PR 指南

> 核心原则：**最小化变更，聚焦目标，确保可验证。**

### 1. 文件隔离

StarVLA 的架构设计为**文件级隔离**——新的框架 / 基准 / 数据集应自包含在各自的目录中，避免修改共享模块。

| 场景 | 推荐做法 | 不允许 |
|----------|-------------|-------------|
| 新框架 | 在 `starVLA/model/framework/` 下创建新模块 | 直接修改现有框架的核心逻辑 |
| 新基准 | 在 `examples/<benchmark>/` 下创建新目录 | 将配置分散到多个现有目录 |
| 新数据集 | 在 `examples/<benchmark>/train_files/data_registry/` 中注册 | 修改共享的 dataloader 接口（除非有充分理由） |
| Bug 修复 | 精确修改受影响文件，包含单元测试 | 顺便重构无关代码 |

如果你确实需要修改共享模块（dataloader、config、trainer），请在 PR 描述中**清楚说明原因和影响范围**。

### 2. 基准 / 框架贡献必须包含验证

以下类型的 PR **必须**提供验证材料：

- 添加或修改训练框架
- 添加或修改基准集成
- 修改核心 dataloader / 训练循环逻辑

**所需材料：**

| 材料 | 要求 |
|----------|-------------|
| **基准结果** | 至少在某个基准上的定量评估结果（如 LIBERO 成功率、SimplerEnv 分数），以表格或截图形式呈现在 PR 描述中 |
| **检查点** | 训练好的检查点上传至贡献者自己的 Hugging Face 账号并设为**公开**，在 PR 中提供链接 |
| **训练配置** | 完整的训练配置 YAML，放置在对应的 `examples/` 目录下 |
| **复现说明** | 简要说明如何使用提供的配置 + 检查点复现评估结果 |

PR 描述中的示例格式：

```markdown
## 验证

| 基准 | 指标 | 结果 |
|-----------|--------|--------|
| LIBERO-Goal | 成功率（3 种子均值） | 78.5% |

- 检查点：https://huggingface.co/<your-username>/starvla-xxx
- 配置：`examples/LIBERO/train_files/xxx.yaml`
- 复现：`bash examples/LIBERO/eval.sh --ckpt <hf-path>`
```

### 3. 测试

- **新功能**：必须包含至少一个测试脚本或用例，放置在 PR 对应的 `tmp/` 或 `examples/` 目录中。
- **Bug 修复**：描述复现步骤，最好提供一个能触发 bug 的最小测试。
- **重构**：修改的文件必须通过 Black + Ruff；如涉及行为变更，需补充测试。

### 4. PR 范围控制

| ✅ 推荐 | ❌ 避免 |
|----------------|----------|
| 一个 PR 做一件事 | 在一个 PR 中混合新框架 + Bug 修复 + 文档更新 |
| Diff < 500 行 | 过大 PR（>1000 行），除非添加独立模块 |
| 仅修改相关文件 | 顺便格式化 / 重构无关代码 |
| 先开 Issue 讨论 | 直接提交大型变更 PR |

> 详细的 PR 提交流程、提交信息规范和审查流程，参见 [docs/PR_readme.md](PR_readme.md)。

## 总结

```
                  hotfix/xxx
                     │
                     ▼
  ┌──────────────────────────────────────┐
  │           starVLA (稳定)             │  ◄── 标签: v0.1.0, v0.2.0, ...
  └──────────────┬───────────────────────┘
                 │ merge (里程碑)
                 │
  ┌──────────────▼───────────────────────┐
  │         starVLA_dev (活跃)           │  ◄── PR 合入此处
  └──┬───────────┬───────────┬───────────┘
     │           │           │
  feat/a      fix/b      docs/c
```

**一句话总结**：贡献者从 `starVLA_dev` 分支，提交 PR 回到 `starVLA_dev`，维护者定期将稳定快照提升到 `starVLA` 并打版本标签。
