#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
STARVLA_DIR=${STARVLA_DIR:-${REPO_ROOT}}

cd ${STARVLA_DIR}
# === Checkpoint ===
CKPT=${CKPT:-${STARVLA_DIR}/playground/trained_model/starVLA_QwenGR00T_libero4in1_qwen3_dit/steps_23000}

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

host="127.0.0.1"
base_port=6694
unnorm_key="franka"
your_ckpt=${CKPT}

# export DEBUG=true

if [[ "${your_ckpt}" == *"/checkpoints/"* ]]; then
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

task_suite_name=libero_goal
num_trials_per_task=50
video_out_path="${model_root}/results/${task_suite_name}/${folder_name}"

${LIBERO_PYTHON} ./examples/LIBERO/eval_files/eval_libero.py \
    --args.pretrained-path ${your_ckpt} \
    --args.host "$host" \
    --args.port $base_port \
    --args.task-suite-name "$task_suite_name" \
    --args.num-trials-per-task "$num_trials_per_task" \
    --args.video-out-path "$video_out_path"
