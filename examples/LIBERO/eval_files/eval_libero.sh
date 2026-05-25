#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
STARVLA_DIR=${STARVLA_DIR:-${REPO_ROOT}}
RUN_DIR=${RUN_DIR:-${STARVLA_DIR}/playground/trained_model/starVLA_QwenGR00T_libero4in1_qwen3_dit}
CKPT_STEP=${CKPT_STEP:-40000}

cd ${STARVLA_DIR}
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

###########################################################################################
# === Please modify the following paths according to your environment ===
if [[ -z "${LIBERO_HOME}" ]]; then
    if [[ -d "${REPO_ROOT}/LIBERO" ]]; then
        export LIBERO_HOME="${REPO_ROOT}/LIBERO"
    elif [[ -d "${REPO_ROOT}/../LIBERO" ]]; then
        export LIBERO_HOME="${REPO_ROOT}/../LIBERO"
    else
        export LIBERO_HOME="/gemini/code/LIBERO"
    fi
fi
export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}
export LIBERO_PYTHON=${LIBERO_PYTHON:-$(which python)}

export PYTHONPATH=$PYTHONPATH:${LIBERO_HOME} # let eval_libero find the LIBERO tools
export PYTHONPATH=$(pwd):${PYTHONPATH} # let LIBERO find the websocket tools from main repo

export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

host=${HOST:-127.0.0.1}
base_port=${BASE_PORT:-6694}
unnorm_key="franka"
your_ckpt=${CKPT}

# export DEBUG=true

if [[ "${your_ckpt}" == *"/checkpoints/steps_"*_pytorch_model.pt ]]; then
    ckpt_file=$(basename "${your_ckpt}")
    folder_name="${ckpt_file%_pytorch_model.pt}"
    model_root="${your_ckpt%%/checkpoints/*}"
elif [[ "${your_ckpt}" == *"/checkpoints/"* ]]; then
    folder_name=$(echo "$your_ckpt" | awk -F'/' '{print $(NF-2)"_"$(NF-1)"_"$NF}')
    model_root="${your_ckpt%%/checkpoints/*}"
else
    ckpt_name=$(basename "${your_ckpt}")
    parent_name=$(basename "$(dirname "${your_ckpt}")")
    folder_name="${parent_name}_${ckpt_name}"
    model_root=$(dirname "${your_ckpt}")
fi
# === End of environment variable configuration ===
###########################################################################################

task_suite_name=${TASK_SUITE_NAME:-libero_goal}
num_trials_per_task=${NUM_TRIALS_PER_TASK:-50}
video_out_path="${STARVLA_DIR}/playground/eval_results/${task_suite_name}/${folder_name}"

echo "=== Eval Config ==="
echo "CKPT=${your_ckpt}"
echo "TASK_SUITE_NAME=${task_suite_name}"
echo "NUM_TRIALS_PER_TASK=${num_trials_per_task}"
echo "VIDEO_OUT_PATH=${video_out_path}"

${LIBERO_PYTHON} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path"
