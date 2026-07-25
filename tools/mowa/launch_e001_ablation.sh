#!/usr/bin/env bash
# Launch E-001 ablations: 2-head, 4-head, 5-head (+NBV), 6-head (+NBV+OVF).
# Run from the repo root inside the `starVLA` conda/venv env.
set -euo pipefail

# Select which variant to run via VARIANT=<name>.
VARIANT=${VARIANT:-4head}

if [[ "${VARIANT}" == "2head" ]]; then
  CONFIG="configs/mowa/mowa_e001_starflow_ft0_2head_baseline_ablation.yaml"
  RUN_ID="MoWA-E-001_starflow_ft0_2head_baseline_ablation_260707"
elif [[ "${VARIANT}" == "4head" ]]; then
  CONFIG="configs/mowa/mowa_e001_starflow_ft0_4head_ablation.yaml"
  RUN_ID="MoWA-E-001_starflow_ft0_4head_ablation_260707"
elif [[ "${VARIANT}" == "5head_nbv" ]]; then
  CONFIG="configs/mowa/mowa_e001_starflow_ft0_5head_nbv_ablation.yaml"
  RUN_ID="MoWA-E-001_starflow_ft0_5head_nbv_ablation_260708"
elif [[ "${VARIANT}" == "6head" ]]; then
  CONFIG="configs/mowa/mowa_e001_starflow_ft0_6head_ablation.yaml"
  RUN_ID="MoWA-E-001_starflow_ft0_6head_ablation_260708"
else
  echo "Usage: VARIANT=2head|4head|5head_nbv|6head $0"
  exit 1
fi

# Ensure future-label sidecars have been precomputed.  The dataloader merges
# subgoal_feasibility / manipulation_readiness only when sidecars exist.
if ! .venv/bin/python -c "
from pathlib import Path
from tools.mowa.report_future_label_distribution import discover_task_sidecar_dirs
assert len(discover_task_sidecar_dirs(Path('playground/Datasets/robocasa365'))) >= 16
" 2>/dev/null; then
  echo "Warning: future-label sidecars appear incomplete. Run:"
  echo "  python tools/mowa/precompute_all_atomic_task_future_labels.py"
  echo "Continuing anyway..."
fi

echo "Launching ${VARIANT} ablation: ${RUN_ID}"
exec .venv/bin/python starVLA/training/train_starvla.py \
  --config_yaml "${CONFIG}"
