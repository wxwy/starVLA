#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/disk/rl/starVLA"
TRACKER_DIR="${REPO_DIR}/docs_zh/starflow_vla"
RUN_ID="P0-M7-E-H2a-04_starflow_libero-4in1_qwen3vl4b_lwfm_ft64_260620_1652"
MD_FILE="${TRACKER_DIR}/${RUN_ID}.md"

cd "${REPO_DIR}"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Updating training tracker..."
python3 "${TRACKER_DIR}/.update_training_tracker.py"

# Check if there are changes to commit
if git diff --quiet -- "${MD_FILE}" 2>/dev/null; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] No changes to ${MD_FILE}. Skipping push."
    exit 0
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Committing and pushing updated tracker..."
git add "${MD_FILE}"
git commit -m "docs(starflow_vla): update ${RUN_ID} training tracker at $(date '+%Y%m%d_%H%M%S')

🤖 Generated with [Claude Code](https://claude.com/code)"
git pull --rebase origin "$(git branch --show-current)"
git push origin HEAD

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Push complete."
