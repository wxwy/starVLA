#!/usr/bin/env bash
# 每 2 小时运行一次：监控非 tmux 环境下的训练进程，更新训练记录 markdown，并 push。
set -euo pipefail

REPO_DIR="/disk/rl/starVLA"
TRACKER_DIR="${REPO_DIR}/docs_zh/starflow_vla"

cd "${REPO_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running no-tmux monitor..."
python3 "${TRACKER_DIR}/.monitor_no_tmux.py"

# Check if there are any changes in markdown trackers
if git diff --quiet -- docs_zh/starflow_vla/*.md; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No training tracker changes to commit."
    exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Changes detected, committing and pushing..."
git add docs_zh/starflow_vla/*.md
git commit -m "docs(starflow_vla): auto-update training tracker at $(date '+%Y-%m-%d %H:%M:%S %Z')

🤖 Generated with [Claude Code](https://claude.com/code)"

# Pull remote changes to avoid push rejection
git pull --rebase --autostash origin merge-official-starvla-dev || true
git push origin merge-official-starvla-dev

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done."
