#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
STARVLA_DIR=${REPO_ROOT}
STARVLA_PYTHON=${STARVLA_PYTHON:-${REPO_ROOT}/.venv/bin/python}
RUN_DIR=${RUN_DIR:-${STARVLA_DIR}/playground/trained_model/starVLA_QwenGR00T_libero4in1_qwen3_dit}
CKPT_STEP=${CKPT_STEP:-40000}

# === Checkpoint ===
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

export star_vla_python=${STARVLA_PYTHON}
your_ckpt=${CKPT}   
gpu_id=${GPU_ID:-0}
port=${PORT:-6694}
################# star Policy Server ######################

# export DEBUG=true
echo "=== Policy Server Config ==="
echo "CKPT=${your_ckpt}"
echo "GPU_ID=${gpu_id}"
echo "PORT=${port}"
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
