#!/bin/bash
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
STARVLA_DIR=${REPO_ROOT}
STARVLA_PYTHON=${STARVLA_PYTHON:-${REPO_ROOT}/.venv/bin/python}

# === Checkpoint ===
CKPT=${CKPT:-${STARVLA_DIR}/playground/trained_model/starVLA_QwenGR00T_libero4in1_qwen3_dit/steps_23000}

export star_vla_python=${STARVLA_PYTHON}
your_ckpt=${CKPT}   
gpu_id=0
port=6694
################# star Policy Server ######################

# export DEBUG=true
CUDA_VISIBLE_DEVICES=$gpu_id ${star_vla_python} deployment/model_server/server_policy.py \
    --ckpt_path ${your_ckpt} \
    --port ${port} \
    --use_bf16

# #################################
