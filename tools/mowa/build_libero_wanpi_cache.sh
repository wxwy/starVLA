#!/usr/bin/env bash
set -euo pipefail

# Build the real Wan2.2 episode cache and 5 Hz window manifests required by
# configs/mowa/mowa_e003_v2_wanpi_libero4in1_contft32_lora.yaml.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/playground/Datasets/LEROBOT_LIBERO_DATA}"
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/playground/Datasets/libero_wan2.2_latent}"
WAN_MODEL="${WAN_MODEL:-${REPO_ROOT}/playground/Pretrained_models/Wan-AI/Wan2.2-TI2V-5B-Diffusers}"
TEXT_CACHE="${TEXT_CACHE:-${CACHE_ROOT}/instruction_text_latents.pt}"
NUM_WORKERS="${NUM_WORKERS:-32}"

DATASETS=(
  libero_object_no_noops_1.0.0_lerobot
  libero_goal_no_noops_1.0.0_lerobot
  libero_spatial_no_noops_1.0.0_lerobot
  libero_10_no_noops_1.0.0_lerobot
)

for dataset_name in "${DATASETS[@]}"; do
  dataset_path="${DATA_ROOT}/${dataset_name}"
  dataset_cache="${CACHE_ROOT}/${dataset_name}"
  "${PYTHON_BIN}" "${REPO_ROOT}/tools/mowa/build_episode_latent_store.py" \
    --dataset-path "${dataset_path}" \
    --cache-root "${dataset_cache}" \
    --video-key observation.images.image \
    --video-key observation.images.wrist_image \
    --encoder-kind wan2.2-vae \
    --encoder-model-path "${WAN_MODEL}" \
    --encoder-name Wan-AI/Wan2.2-TI2V-5B-Diffusers \
    --encoder-version wan2.2-vae-v1 \
    --latent-type vae_spatial \
    --dtype float16 \
    --video-backend pyav \
    --num-workers "${NUM_WORKERS}" \
    --execute

  "${PYTHON_BIN}" "${REPO_ROOT}/tools/mowa/build_window_manifest.py" \
    --cache-root "${dataset_cache}" \
    --output-path "${dataset_cache}/window_manifest.parquet" \
    --history-steps 0 \
    --future-steps 8 \
    --action-chunk-steps 32 \
    --anchor-video-key observation.images.image \
    --history-stride 1 \
    --split train
done

"${PYTHON_BIN}" "${REPO_ROOT}/tools/mowa/build_instruction_text_latent_cache.py" \
  --dataset-root "${DATA_ROOT}" \
  --output "${TEXT_CACHE}" \
  --encoder-model-path "${WAN_MODEL}" \
  --dtype float16

echo "LIBERO WanPI cache ready: ${CACHE_ROOT}"
