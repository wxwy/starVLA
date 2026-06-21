#!/usr/bin/env bash
# 每 30 分钟运行一次：监控活跃训练进程，更新训练记录 markdown，并 push。
set -euo pipefail

cd /disk/rl/starVLA

python3 docs_zh/starflow_vla/.monitor_active_runs.py

# Only commit/push if there are changes in docs.
if git diff --quiet -- docs_zh/starflow_vla/; then
    echo "No training tracker changes to commit."
    exit 0
fi

git add docs_zh/starflow_vla/
git commit -m "docs(starflow_vla): auto-update training tracker at $(date '+%Y-%m-%d %H:%M:%S %Z')

Co-Authored-By: Claude <noreply@anthropic.com>"

# Pull remote changes to avoid push rejection. --autostash handles any unrelated
# local modifications (e.g. config YAML edits) without committing them.
git pull --rebase --autostash origin merge-official-starvla-dev || true
git push origin merge-official-starvla-dev
