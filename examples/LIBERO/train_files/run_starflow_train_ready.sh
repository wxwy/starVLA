#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
STARVLA_DIR=${STARVLA_DIR:-${REPO_ROOT}}
STARVLA_PYTHON=${STARVLA_PYTHON:-${STARVLA_DIR}/.venv/bin/python}

RUN_ROOT_DIR=${RUN_ROOT_DIR:-${STARVLA_DIR}/playground/Checkpoints}
RUN_ID=${RUN_ID:-starflow_vla_stage1_train_ready}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-10}
SAVE_INTERVAL=${SAVE_INTERVAL:-10}
LOGGING_FREQUENCY=${LOGGING_FREQUENCY:-10}
EVAL_INTERVAL=${EVAL_INTERVAL:-1000}
WANDB_PROJECT=${WANDB_PROJECT:-starVLA_Libero}
WANDB_ENTITY=${WANDB_ENTITY:-silencewx-harbin-institute-of-technology}
BASE_VLM=${BASE_VLM:-${STARVLA_DIR}/playground/Pretrained_models/Qwen3-VL-4B-Instruct}
LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT:-${STARVLA_DIR}/playground/Datasets/LEROBOT_LIBERO_DATA}
DATA_MIX=${DATA_MIX:-libero_goal}
FRAMEWORK_NAME=${FRAMEWORK_NAME:-StarFlowVLA}
FREEZE_MODULES=${FREEZE_MODULES:-qwen_vl_interface}
NUM_WORKERS=${NUM_WORKERS:-0}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-1}
NUM_PROCESS=${NUM_PROCESS:-1}

cd "${STARVLA_DIR}"

if [[ ! -d "${LIBERO_DATA_ROOT}" ]]; then
    echo "LIBERO data root not found: ${LIBERO_DATA_ROOT}"
    exit 1
fi

if [[ ! -e "${BASE_VLM}" ]]; then
    echo "Base VLM not found: ${BASE_VLM}"
    exit 1
fi

for candidate in /usr/lib/x86_64-linux-gnu /usr/local/cuda-12.3/compat; do
    if [[ -d "${candidate}" ]]; then
        if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
            case ":${LD_LIBRARY_PATH}:" in
                *:"${candidate}":*) ;;
                *) export LD_LIBRARY_PATH="${candidate}:${LD_LIBRARY_PATH}" ;;
            esac
        else
            export LD_LIBRARY_PATH="${candidate}"
        fi
    fi
done

export WANDB_MODE=${WANDB_MODE:-disabled}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29621}
export RANK=${RANK:-0}
export LOCAL_RANK=${LOCAL_RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}
export PYTHONUNBUFFERED=1

OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"

echo "=== StarFlow Train Ready ==="
echo "STARVLA_DIR=${STARVLA_DIR}"
echo "RUN_ROOT_DIR=${RUN_ROOT_DIR}"
echo "RUN_ID=${RUN_ID}"
echo "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "WORLD_SIZE=${WORLD_SIZE}"
echo "LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT}"
echo "BASE_VLM=${BASE_VLM}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"

"${STARVLA_PYTHON}" -u starVLA/training/train_starvla.py \
    --config_yaml configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml \
    --framework.name "${FRAMEWORK_NAME}" \
    --framework.qwenvl.base_vlm "${BASE_VLM}" \
    --datasets.vla_data.data_root_dir "${LIBERO_DATA_ROOT}" \
    --datasets.vla_data.data_mix "${DATA_MIX}" \
    --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --datasets.vla_data.num_workers "${NUM_WORKERS}" \
    --trainer.freeze_modules "${FREEZE_MODULES}" \
    --trainer.max_train_steps "${MAX_TRAIN_STEPS}" \
    --trainer.save_interval "${SAVE_INTERVAL}" \
    --trainer.logging_frequency "${LOGGING_FREQUENCY}" \
    --trainer.eval_interval "${EVAL_INTERVAL}" \
    --trainer.save_checkpoint_as_directory True \
    --trainer.save_with_training_state False \
    --trainer.save_format safetensors \
    --run_root_dir "${RUN_ROOT_DIR}" \
    --run_id "${RUN_ID}" \
    --wandb_project "${WANDB_PROJECT}" \
    --wandb_entity "${WANDB_ENTITY}" \
    --trainer.is_resume False
