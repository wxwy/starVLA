#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
STARVLA_DIR=${STARVLA_DIR:-${REPO_ROOT}}
RUN_DIR=${RUN_DIR:-${STARVLA_DIR}/playground/trained_model/starVLA_QwenGR00T_libero4in1_qwen3_dit}
CKPT_STEP=${CKPT_STEP:-40000}

HOST=${HOST:-127.0.0.1}
PORT=${PORT:-${BASE_PORT:-6694}}
TASK_SUITE_NAME=${TASK_SUITE_NAME:-libero_goal}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-50}
MAX_TASKS=${MAX_TASKS:-}
REPLAN_INTERVAL=${REPLAN_INTERVAL:-}
PAYLOAD_STYLE=${PAYLOAD_STYLE:-standard}
WAN_HISTORY_FRAMES=${WAN_HISTORY_FRAMES:-5}
RESUME_EVAL=${RESUME_EVAL:-}
MUJOCO_GL_VALUE=${MUJOCO_GL_VALUE:-egl}
PYOPENGL_PLATFORM_VALUE=${PYOPENGL_PLATFORM_VALUE:-egl}

cd "${STARVLA_DIR}"

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

if [[ -z "${LIBERO_HOME:-}" ]]; then
    if [[ -d "${REPO_ROOT}/LIBERO" ]]; then
        export LIBERO_HOME="${REPO_ROOT}/LIBERO"
    elif [[ -d "${REPO_ROOT}/../LIBERO" ]]; then
        export LIBERO_HOME="${REPO_ROOT}/../LIBERO"
    else
        echo "LIBERO_HOME is required."
        echo "Example: LIBERO_HOME=/path/to/LIBERO LIBERO_PYTHON=/path/to/python bash $0"
        exit 1
    fi
fi

export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}
export LIBERO_PYTHON=${LIBERO_PYTHON:-$(command -v python)}
export PYTHONPATH="${PYTHONPATH:-}:${LIBERO_HOME}:${STARVLA_DIR}"
export MUJOCO_GL="${MUJOCO_GL_VALUE}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM_VALUE}"

if [[ "${CKPT}" == *"/checkpoints/steps_"*_pytorch_model.pt ]]; then
    ckpt_file=$(basename "${CKPT}")
    FOLDER_NAME="${ckpt_file%_pytorch_model.pt}"
elif [[ "${CKPT}" == *"/checkpoints/"* ]]; then
    FOLDER_NAME=$(echo "${CKPT}" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
else
    ckpt_name=$(basename "${CKPT}")
    parent_name=$(basename "$(dirname "${CKPT}")")
    FOLDER_NAME="${parent_name}_${ckpt_name}"
fi

VIDEO_OUT_PATH=${VIDEO_OUT_PATH:-${STARVLA_DIR}/playground/eval_results/${TASK_SUITE_NAME}/${FOLDER_NAME}}

echo "=== Eval Config ==="
echo "CKPT=${CKPT}"
echo "TASK_SUITE_NAME=${TASK_SUITE_NAME}"
echo "NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}"
echo "MAX_TASKS=${MAX_TASKS:-<all>}"
echo "REPLAN_INTERVAL=${REPLAN_INTERVAL:-<full_chunk>}"
echo "PAYLOAD_STYLE=${PAYLOAD_STYLE}"
echo "WAN_HISTORY_FRAMES=${WAN_HISTORY_FRAMES}"
echo "RESUME_EVAL=${RESUME_EVAL:-false}"
echo "VIDEO_OUT_PATH=${VIDEO_OUT_PATH}"

CMD=(
    "${LIBERO_PYTHON}" ./examples/LIBERO/eval_files/eval_libero.py
    --args.pretrained-path "${CKPT}"
    --args.host "${HOST}"
    --args.port "${PORT}"
    --args.task-suite-name "${TASK_SUITE_NAME}"
    --args.num-trials-per-task "${NUM_TRIALS_PER_TASK}"
    --args.video-out-path "${VIDEO_OUT_PATH}"
    --args.payload-style "${PAYLOAD_STYLE}"
    --args.wan-history-frames "${WAN_HISTORY_FRAMES}"
)

if [[ -n "${MAX_TASKS}" ]]; then
    CMD+=(--args.max-tasks "${MAX_TASKS}")
fi

if [[ -n "${REPLAN_INTERVAL}" ]]; then
    CMD+=(--args.replan-interval "${REPLAN_INTERVAL}")
fi

if [[ -n "${RESUME_EVAL}" ]]; then
    case "${RESUME_EVAL}" in
        1|true|TRUE|True|yes|YES|Yes|on|ON|On)
            CMD+=(--args.resume-eval)
            ;;
        0|false|FALSE|False|no|NO|No|off|OFF|Off)
            ;;
        *)
            echo "Invalid RESUME_EVAL value: ${RESUME_EVAL}"
            exit 1
            ;;
    esac
fi

"${CMD[@]}"
