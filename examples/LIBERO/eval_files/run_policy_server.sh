#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
STARVLA_DIR=${STARVLA_DIR:-${REPO_ROOT}}
STARVLA_PYTHON=${STARVLA_PYTHON:-python}
RUN_DIR=${RUN_DIR:-${STARVLA_DIR}/playground/trained_model/starVLA_QwenGR00T_libero4in1_qwen3_dit}
CKPT_STEP=${CKPT_STEP:-40000}
GPU_ID=${GPU_ID:-0}
PORT=${PORT:-6694}
USE_BF16=${USE_BF16:-1}

cd "${STARVLA_DIR}"
export PYTHONPATH="${STARVLA_DIR}:${PYTHONPATH:-}"

DEFAULT_CKPT="${RUN_DIR}/checkpoints/steps_${CKPT_STEP}"
DIRECT_PT_CKPT="${RUN_DIR}/checkpoints/steps_${CKPT_STEP}_pytorch_model.pt"
LEGACY_DS_MODEL_STATE="${RUN_DIR}/checkpoints/steps_${CKPT_STEP}/pytorch_model/mp_rank_00_model_states.pt"
if [[ -z "${CKPT:-}" ]]; then
    if [[ -f "${DIRECT_PT_CKPT}" ]]; then
        CKPT="${DIRECT_PT_CKPT}"
    elif [[ -f "${LEGACY_DS_MODEL_STATE}" ]]; then
        CKPT="${LEGACY_DS_MODEL_STATE}"
    else
        CKPT="${DEFAULT_CKPT}"
    fi
fi

echo "=== Policy Server Config ==="
echo "CKPT=${CKPT}"
echo "GPU_ID=${GPU_ID}"
echo "PORT=${PORT}"

CMD=(
    "${STARVLA_PYTHON}" deployment/model_server/server_policy.py
    --ckpt_path "${CKPT}"
    --port "${PORT}"
)

if [[ "${USE_BF16}" == "1" ]]; then
    CMD+=(--use_bf16)
fi

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${CMD[@]}"
