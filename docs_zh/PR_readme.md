# Pull Request 指南

感谢你为 StarVLA 做贡献！为确保 StarVLA 的稳定发展，我们再次尝试建立 PR 规范。本文档描述如何提交高质量的 PR，以便高效地通过审查和合并。

> **太长不看版**：先开 Issue → 从 `starVLA_dev` 分支 → 提交 PR 并附上清晰描述 → 处理审查反馈。

## 开始之前

1. **检查已有工作** — 搜索开放中的 PR 和 Issue，确保你的想法尚未在进行中。

2. **阅读分支策略** — 参见 [docs/branching_strategy.md](branching_strategy.md) 了解分支组织方式。

## PR 生命周期

```
1. Issue / Discussion       与维护者对齐范围
        │
2. 创建分支                 git checkout -b feat/xxx starVLA_dev
        │
3. 开发与测试               编写代码，运行 make check
        │
4. 提交 PR                  目标分支: starVLA_dev
        │
5. 代码审查                 处理审查者反馈
        │
6. 合并（Squash）           维护者在批准后合并
```

## 分步说明

### 1. Fork & Clone（外部贡献者）

```bash
# 在 GitHub 上 Fork 仓库，然后：
git clone https://github.com/<your-username>/starVLA.git
cd starVLA
git remote add upstream https://github.com/starVLA/starVLA.git
```

### 2. 创建功能分支

始终从最新的 `starVLA_dev` 分支：

```bash
git fetch upstream
git checkout -b feat/my-feature upstream/starVLA_dev
```

命名规范参见 [branching_strategy.md](branching_strategy.md)（`feat/`、`fix/`、`docs/` 等）。

### 3. 写代码

- 遵循现有代码风格（Black + Ruff）。
- 保持变更聚焦——一个 PR 一个逻辑变更。
- 适当添加或更新 docstring。
- 如果添加新的训练框架或数据集，请在 `examples/` 中包含示例配置。

### 4. 提交前检查清单

> **关于 `make check` 的说明**：仓库目前存在历史 Black / Ruff 积压，因此对整个仓库运行 `make check` 预期会失败。在积压清理完毕前，**只检查你的 PR 实际触及的文件**。混入无关文件的格式化修复违反"不修改无关代码"规则。

推荐的本地检查（限定在 PR 范围内）：

```bash
# 本 PR 中修改或新增的文件（与 starVLA_dev 比较）
FILES=$(git diff --name-only --diff-filter=ACMR origin/starVLA_dev | grep -E '\.py$')

# 仅对这些文件进行格式化和 lint
black $FILES
python -m ruff check --fix $FILES

# 验证
black --check $FILES
python -m ruff check $FILES
```

全仓库入口命令仅供参考：

```bash
make check        # 对整个仓库验证格式（Black）和 lint（Ruff）
make autoformat   # 对整个仓库自动修复格式问题（不要提交无关的格式化噪音）
```

自查清单：

- [ ] `black --check` 和 `ruff check` 在**本 PR 触及的文件**上通过
- [ ] 无无关变更（调试打印、无关重构、顺便格式化）
- [ ] 新功能有示例配置或文档
- [ ] 未提交密钥、API 密钥或大型二进制文件
- [ ] 配置 YAML 变更向后兼容（或明确标注为 breaking）

### 5. 提交信息

我们遵循 [Conventional Commits](https://www.conventionalcommits.org/)：

```
<type>(<scope>): <简短描述>

[可选正文]

[可选脚注]
```

**类型：**

| 类型 | 描述 |
|------|-------------|
| `feat` | 新功能 |
| `fix` | Bug 修复 |
| `docs` | 仅文档 |
| `refactor` | 代码变更，既非修复 bug 也非新增功能 |
| `perf` | 性能优化 |
| `test` | 添加或更新测试 |
| `chore` | 构建过程、CI 或工具链变更 |

**示例：**

```
feat(dataloader): 使用 Welford 算法添加流式统计
fix(training): 移除 save_full_config 中无效的 resolve 参数 (#192)
docs(examples): 添加 LIBERO 4合1 训练教程
refactor(model): 将 QwenOFT 移至 framework/VLM4A/
```

**规则：**
- 使用英文编写提交信息。
- 主题行控制在 72 字符以内。
- 引用相关 Issue：`Closes #42`、`Fixes #108`。

### 6. 提交 Pull Request

目标分支：**`starVLA_dev`**（非 `starVLA`）。

请使用以下模板填写 PR 描述：

---

#### PR 描述模板

```markdown
## 动机

<!-- 为什么需要这个变更？链接到相关 Issue。 -->
Closes #<issue-number>

## 变更内容

<!-- 这个 PR 做了什么？列出关键变更。 -->
- 添加了 ...
- 修复了 ...
- 重构了 ...

## 测试

<!-- 如何测试的？ -->
- [ ] 在本 PR 触及的文件上运行 `black --check` / `ruff check` — 通过
- [ ] 在 [数据集/框架/环境] 上测试：...
- [ ] 训练运行 N 步无错误

## Breaking Changes

<!-- 此 PR 是否破坏向后兼容性？如是，请描述。 -->
无 / 是：...

## 截图 / 日志（可选）

<!-- 如有，附上训练曲线、评估结果或相关日志。 -->
```

---

### 7. 代码审查

- 会有一位维护者被分配来审查你的 PR。
- 处理所有审查意见并推送后续提交。
- 做出要求的修改后解决对话。
- 如果 PR 超过 14 天无响应，可能会被关闭。

### 8. 合并

- PR 通过 **Squash Merge** 合并以保持主分支历史整洁。
- 维护者将基于你的 PR 标题编写一个简洁的 squash 提交信息。
- 合并后，你的功能分支可以删除。

## 好的 PR 长什么样

| ✅ 应该做 | ❌ 不要做 |
|-------|----------|
| 一个 PR 一个逻辑变更 | 在一个 PR 中混合无关变更 |
| 清晰、有描述性的标题 | 模糊的标题如"更新代码" |
| 引用相关 Issue | 未经事先讨论直接提交 |
| 包含前后对比 | 让审查者猜测影响 |
| 尽量保持 diff 小（理想 <500 行） | 提交 3000 行 PR 且无上下文 |
| 推送前在修改的文件上运行 Black + Ruff | 推送在你触及的行上 lint/format 失败的代码 |

## 特殊情况

### 添加新的训练框架

1. 在 `starVLA/model/framework/` 下创建框架模块。
2. 在框架配置系统中注册。
3. 添加默认配置 dataclass（如 `QwenOFTDefaultConfig`）。
4. 在 `examples/<benchmark>/train_files/` 下添加示例配置 YAML。
5. 在 `docs/model_zoo.md` 中包含简要说明。
6. **提供至少一项基准结果 + 公开 HF 检查点**（参见 [branching_strategy.md § 社区 PR 指南](branching_strategy.md#社区-pr-指南)）。

### 添加新数据集 / 基准

1. 在 `examples/<benchmark>/train_files/data_registry/` 下创建数据配置。
2. 确保数据集为 LeRobot 格式。
3. 在 `examples/<benchmark>/` 下添加示例训练脚本。
4. 记录任何特殊设置步骤。
5. **提供定量评估结果 + 公开 HF 检查点**（参见 [branching_strategy.md § 社区 PR 指南](branching_strategy.md#社区-pr-指南)）。

### 纯文档 PR

- 正常以 `starVLA_dev` 为目标。
- 无需大量测试，但请验证链接和格式。
- 这类 PR 始终欢迎！

## 需要帮助？

- **办公时间**：每周五下午 — 填写[合作表单](https://forms.gle/R4VvgiVveULibTCCA)。
- **讨论**：通过 [GitHub Issues](https://github.com/starVLA/starVLA/issues) 提问技术问题。
- **快速问题**：在 PR 评论中 @ 维护者。

---

*最后更新：2026 年 4 月*
