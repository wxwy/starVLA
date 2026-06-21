#!/usr/bin/env bash
# 每 30 分钟运行一次：更新 P1-M1-E-H2b-02 tracker 并 push。
set -euo pipefail

REPO_DIR="/disk/rl/starVLA"
TRACKER_DIR="${REPO_DIR}/docs_zh/starflow_vla"
RUN_ID="P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260621_1901"
MD_FILE="${TRACKER_DIR}/${RUN_ID}.md"

cd "${REPO_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Updating P1-M1 tracker..."
python3 "${TRACKER_DIR}/.update_p1m1_tracker.py"

# Check if there are changes to commit
if git diff --quiet -- "${MD_FILE}" 2>/dev/null; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No changes to ${MD_FILE}. Skipping push."
    exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Committing and pushing updated tracker..."
git add "${MD_FILE}"
git commit -m "docs(starflow_vla): update ${RUN_ID} training tracker at $(date '+%Y%m%d_%H%M%S')

🤖 Generated with [Claude Code](https://claude.com/claude-code)"
git pull --rebase --autostash origin merge-official-starvla-dev || true
git push origin merge-official-starvla-dev

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Push complete."
