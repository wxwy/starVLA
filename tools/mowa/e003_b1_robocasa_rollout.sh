#!/usr/bin/env bash
# E003-B1 checkpoint RoboCasa 闭环评测：server/client 分别在训练与仿真环境执行。
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
CKPT="${CKPT:-${REPO_ROOT}/playground/mowa_ckpt/MoWA-E-003_future_latent_prior_wo_history/MoWA-E-003_future_latent_prior_wo_history/checkpoints/steps_1000}"
PORT="${PORT:-5678}"
GPU_ID="${GPU_ID:-0}"
ENV_NAME="${ENV_NAME:-robocasa/OpenDrawer}"
N_EPISODES="${N_EPISODES:-1}"
N_ENVS="${N_ENVS:-1}"
MAX_STEPS="${MAX_STEPS:-500}"
N_ACT="${N_ACT:-8}"
WAN_HISTORY_FRAMES="${WAN_HISTORY_FRAMES:-5}"
ROBOCASA_PYTHON="${ROBOCASA_PYTHON:-${REPO_ROOT}/.robocase/bin/python}"
VIDEO_OUT_PATH="${VIDEO_OUT_PATH:-${CKPT}.eval/videos}"

if [[ ! -d "${CKPT}" ]]; then
    echo "[e003-b1-rollout] checkpoint directory not found: ${CKPT}" >&2
    exit 1
fi

case "${1:-}" in
    server)
        exec env CUDA_VISIBLE_DEVICES="${GPU_ID}" \
            "${REPO_ROOT}/.venv/bin/python" "${REPO_ROOT}/deployment/model_server/server_policy.py" \
            --ckpt_path "${CKPT}" \
            --port "${PORT}" \
            --use_bf16
        ;;
    client)
        if [[ ! -x "${ROBOCASA_PYTHON}" ]]; then
            echo "[e003-b1-rollout] RoboCasa Python not executable: ${ROBOCASA_PYTHON}" >&2
            exit 1
        fi
        exec env \
            PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
            MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-${GPU_ID}}" \
            MUJOCO_GL="${MUJOCO_GL:-egl}" \
            PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}" \
            "${ROBOCASA_PYTHON}" -m examples.Robocasa_365.eval_files.simulation_env \
            --args.pretrained-path "${CKPT}" \
            --args.env-name "${ENV_NAME}" \
            --args.port "${PORT}" \
            --args.n-episodes "${N_EPISODES}" \
            --args.n-envs "${N_ENVS}" \
            --args.max-episode-steps "${MAX_STEPS}" \
            --args.n-action-steps "${N_ACT}" \
            --args.wan-history-frames "${WAN_HISTORY_FRAMES}" \
            --args.video-out-path "${VIDEO_OUT_PATH}"
        ;;
    *)
        cat <<USAGE
Usage:
  # terminal 1: start policy server in the starVLA environment
  bash tools/mowa/e003_b1_robocasa_rollout.sh server

  # terminal 2: start RoboCasa client in the .robocase environment
  bash tools/mowa/e003_b1_robocasa_rollout.sh client

Environment overrides:
  CKPT, PORT, GPU_ID, ENV_NAME, N_EPISODES, N_ENVS, MAX_STEPS, N_ACT,
  WAN_HISTORY_FRAMES, ROBOCASA_PYTHON, VIDEO_OUT_PATH.
USAGE
        exit 2
        ;;
esac
