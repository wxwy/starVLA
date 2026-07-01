#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
STARVLA_DIR=${STARVLA_DIR:-${REPO_ROOT}}
STARVLA_PYTHON=${STARVLA_PYTHON:-${STARVLA_DIR}/.venv/bin/python}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-starVLA/config/deepseeds/deepspeed_zero2.yaml}

RUN_ROOT_DIR=${RUN_ROOT_DIR:-${STARVLA_DIR}/playground/Checkpoints}
RUN_ID=${RUN_ID:-P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_250618_r3}
MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS:-80000}
SAVE_INTERVAL=${SAVE_INTERVAL:-125}
LOGGING_FREQUENCY=${LOGGING_FREQUENCY:-50}
EVAL_INTERVAL=${EVAL_INTERVAL:-250}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
WANDB_PROJECT=${WANDB_PROJECT:-starflow_vla}
WANDB_ENTITY=${WANDB_ENTITY:-silencewx-harbin-institute-of-technology}
WANDB_RUN_ID=${WANDB_RUN_ID:-${RUN_ID}}
WANDB_NAME=${WANDB_NAME:-${RUN_ID}}
BASE_VLM=${BASE_VLM:-${STARVLA_DIR}/playground/Pretrained_models/Qwen3-VL-4B-Instruct}
LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT:-${STARVLA_DIR}/playground/Datasets/LEROBOT_LIBERO_DATA}
DATA_MIX=${DATA_MIX:-libero_all}
CONFIG_YAML=${CONFIG_YAML:-configs/starflow_vla/stage1_starflow_qwenpi_v3_native.yaml}
FRAMEWORK_NAME=${FRAMEWORK_NAME:-}
FREEZE_MODULES=${FREEZE_MODULES:-qwen_vl_interface}
NUM_WORKERS=${NUM_WORKERS:-1}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
NUM_PROCESSES=${NUM_PROCESSES:-${NUM_PROCESS:-1}}
ENABLE_LOCAL_CHECKPOINT_STAGING=${ENABLE_LOCAL_CHECKPOINT_STAGING:-True}
LOCAL_CHECKPOINT_ROOT=${LOCAL_CHECKPOINT_ROOT:-/root/temp}
LOCAL_CHECKPOINT_KEEP_COUNT=${LOCAL_CHECKPOINT_KEEP_COUNT:-1}
IS_RESUME=${IS_RESUME:-False}

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

export WANDB_MODE=${WANDB_MODE:-online}
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-29621}
export PYTHONUNBUFFERED=1
export ACCELERATE_GRADIENT_ACCUMULATION_STEPS=${ACCELERATE_GRADIENT_ACCUMULATION_STEPS:-${GRADIENT_ACCUMULATION_STEPS}}

WORLD_SIZE=${WORLD_SIZE:-${NUM_PROCESSES}}

if [[ "${NUM_PROCESSES}" -gt 1 ]]; then
    export RANK=${RANK:-0}
    export LOCAL_RANK=${LOCAL_RANK:-0}
    export WORLD_SIZE
    if ip link show bond0 &>/dev/null; then
        export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond0}
        export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_2,mlx5_3}
    elif ip link show eth0 &>/dev/null; then
        export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
    else
        export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
    fi
    export NCCL_BLOCKING_WAIT=${NCCL_BLOCKING_WAIT:-1}
    export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
    export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10000}
    export NCCL_SOCKET_TIMEOUT_MS=${NCCL_SOCKET_TIMEOUT_MS:-360000}
fi

OUTPUT_DIR="${RUN_ROOT_DIR}/${RUN_ID}"
mkdir -p "${OUTPUT_DIR}"
if [[ "${ENABLE_LOCAL_CHECKPOINT_STAGING}" == "True" || "${ENABLE_LOCAL_CHECKPOINT_STAGING}" == "true" ]]; then
    mkdir -p "${LOCAL_CHECKPOINT_ROOT}"
fi

# Allow users to force accelerate + DeepSpeed launch even with a single process.
# This is useful when resuming a DeepSpeed checkpoint and you want to stay in DeepSpeed.
FORCE_DEEPSPEED=${FORCE_DEEPSPEED:-false}

echo "=== StarFlow Train Ready ==="
echo "STARVLA_DIR=${STARVLA_DIR}"
echo "RUN_ROOT_DIR=${RUN_ROOT_DIR}"
echo "RUN_ID=${RUN_ID}"
echo "CONFIG_YAML=${CONFIG_YAML}"
echo "MAX_TRAIN_STEPS=${MAX_TRAIN_STEPS}"
echo "NUM_PROCESSES=${NUM_PROCESSES}"
echo "GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS}"
echo "ACCELERATE_GRADIENT_ACCUMULATION_STEPS=${ACCELERATE_GRADIENT_ACCUMULATION_STEPS}"
echo "PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "WANDB_RUN_ID=${WANDB_RUN_ID}"
echo "WANDB_NAME=${WANDB_NAME}"
echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "WORLD_SIZE=${WORLD_SIZE}"
echo "LIBERO_DATA_ROOT=${LIBERO_DATA_ROOT}"
echo "BASE_VLM=${BASE_VLM}"
echo "OUTPUT_DIR=${OUTPUT_DIR}"
echo "ENABLE_LOCAL_CHECKPOINT_STAGING=${ENABLE_LOCAL_CHECKPOINT_STAGING}"
echo "LOCAL_CHECKPOINT_ROOT=${LOCAL_CHECKPOINT_ROOT}"
echo "LOCAL_CHECKPOINT_KEEP_COUNT=${LOCAL_CHECKPOINT_KEEP_COUNT}"
echo "IS_RESUME=${IS_RESUME}"
echo "FORCE_DEEPSPEED=${FORCE_DEEPSPEED}"
if [[ "$#" -gt 0 ]]; then
    echo "EXTRA_ARGS=$*"
fi

TRAIN_ARGS=(
    starVLA/training/train_starvla.py
    --config_yaml "${CONFIG_YAML}"
    --framework.qwenvl.base_vlm "${BASE_VLM}"
    --datasets.vla_data.data_root_dir "${LIBERO_DATA_ROOT}"
    --datasets.vla_data.data_mix "${DATA_MIX}"
    --datasets.vla_data.per_device_batch_size "${PER_DEVICE_BATCH_SIZE}"
    --datasets.vla_data.num_workers "${NUM_WORKERS}"
    --trainer.freeze_modules "${FREEZE_MODULES}"
    --trainer.max_train_steps "${MAX_TRAIN_STEPS}"
    --trainer.save_interval "${SAVE_INTERVAL}"
    --trainer.logging_frequency "${LOGGING_FREQUENCY}"
    --trainer.eval_interval "${EVAL_INTERVAL}"
    --trainer.gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --trainer.checkpoint_format lightweight
    --trainer.enable_local_checkpoint_staging "${ENABLE_LOCAL_CHECKPOINT_STAGING}"
    --trainer.local_checkpoint_root "${LOCAL_CHECKPOINT_ROOT}"
    --trainer.local_checkpoint_keep_count "${LOCAL_CHECKPOINT_KEEP_COUNT}"
    --trainer.save_checkpoint_as_directory True
    --trainer.save_with_training_state False
    --trainer.save_format safetensors
    --run_root_dir "${RUN_ROOT_DIR}"
    --run_id "${RUN_ID}"
    --wandb_run_id "${WANDB_RUN_ID}"
    --wandb_name "${WANDB_NAME}"
    --wandb_project "${WANDB_PROJECT}"
    --wandb_entity "${WANDB_ENTITY}"
    --trainer.is_resume "${IS_RESUME}"
    "$@"
)

if [[ -n "${FRAMEWORK_NAME}" ]]; then
    TRAIN_ARGS+=(--framework.name "${FRAMEWORK_NAME}")
fi

if [[ "${NUM_PROCESSES}" -gt 1 || "${FORCE_DEEPSPEED}" == "true" ]]; then
    accelerate launch \
        --config_file "${DEEPSPEED_CONFIG}" \
        --num_processes "${NUM_PROCESSES}" \
        --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
        "${TRAIN_ARGS[@]}"
else
    "${STARVLA_PYTHON}" -u "${TRAIN_ARGS[@]}"
fi
