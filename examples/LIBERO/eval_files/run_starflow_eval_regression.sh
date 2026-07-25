#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
STARVLA_DIR=${STARVLA_DIR:-${REPO_ROOT}}

STARVLA_PYTHON=${STARVLA_PYTHON:-${STARVLA_DIR}/.venv/bin/python}
LIBERO_PYTHON=${LIBERO_PYTHON:-${STARVLA_DIR}/.libero/bin/python}
HOST=${HOST:-127.0.0.1}
PORT=${PORT:-6696}
GPU_ID=${GPU_ID:-0}
USE_BF16=${USE_BF16:-1}
TASK_SUITE_NAME=${TASK_SUITE_NAME:-libero_goal}
NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK:-1}
MAX_TASKS=${MAX_TASKS:-1}
SERVER_READY_TIMEOUT=${SERVER_READY_TIMEOUT:-900}
SERVER_READY_POLL_INTERVAL=${SERVER_READY_POLL_INTERVAL:-2}

cd "${STARVLA_DIR}"

if [[ -z "${LIBERO_HOME:-}" ]]; then
    if [[ -d "${STARVLA_DIR}/LIBERO" ]]; then
        export LIBERO_HOME="${STARVLA_DIR}/LIBERO"
    else
        echo "LIBERO_HOME is required."
        echo "Example: LIBERO_HOME=/path/to/LIBERO bash $0"
        exit 1
    fi
fi

export LIBERO_CONFIG_PATH=${LIBERO_CONFIG_PATH:-${LIBERO_HOME}/libero}

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

RUN_DIR=${RUN_DIR:-${STARVLA_DIR}/playground/trained_model/starVLA_QwenGR00T_libero4in1_qwen3_dit}
CKPT_STEP=${CKPT_STEP:-40000}
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

if [[ ! -e "${CKPT}" ]]; then
    echo "Checkpoint not found: ${CKPT}"
    exit 1
fi

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

VIDEO_OUT_PATH=${VIDEO_OUT_PATH:-${STARVLA_DIR}/playground/eval_results/${TASK_SUITE_NAME}/${FOLDER_NAME}_regression}
SERVER_LOG_PATH=${SERVER_LOG_PATH:-${VIDEO_OUT_PATH}/policy_server.log}
mkdir -p "${VIDEO_OUT_PATH}"
rm -f "${SERVER_LOG_PATH}"

SERVER_PID=""
cleanup() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill "${SERVER_PID}" 2>/dev/null || true
        wait "${SERVER_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

echo "=== StarFlow Eval Regression ==="
echo "CKPT=${CKPT}"
echo "TASK_SUITE_NAME=${TASK_SUITE_NAME}"
echo "NUM_TRIALS_PER_TASK=${NUM_TRIALS_PER_TASK}"
echo "MAX_TASKS=${MAX_TASKS}"
echo "VIDEO_OUT_PATH=${VIDEO_OUT_PATH}"
echo "SERVER_LOG_PATH=${SERVER_LOG_PATH}"
echo "PORT=${PORT}"

PYTHONUNBUFFERED=1 \
CKPT="${CKPT}" \
STARVLA_DIR="${STARVLA_DIR}" \
STARVLA_PYTHON="${STARVLA_PYTHON}" \
GPU_ID="${GPU_ID}" \
PORT="${PORT}" \
USE_BF16="${USE_BF16}" \
bash "${SCRIPT_DIR}/run_policy_server.sh" >"${SERVER_LOG_PATH}" 2>&1 &
SERVER_PID=$!

"${STARVLA_PYTHON}" - "${HOST}" "${PORT}" "${SERVER_READY_TIMEOUT}" "${SERVER_READY_POLL_INTERVAL}" "${SERVER_LOG_PATH}" "${SERVER_PID}" <<'PY'
import pathlib
import socket
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
timeout_s = float(sys.argv[3])
poll_interval_s = float(sys.argv[4])
log_path = pathlib.Path(sys.argv[5])
server_pid = int(sys.argv[6])
deadline = time.time() + timeout_s
last_error = None

while time.time() < deadline:
    try:
        with socket.create_connection((host, port), timeout=2):
            sys.exit(0)
    except OSError as exc:
        last_error = exc

    try:
        import os

        os.kill(server_pid, 0)
    except OSError:
        log_tail = ""
        if log_path.exists():
            lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
            log_tail = "\n".join(lines[-40:])
        raise SystemExit(
            f"policy server exited before becoming ready on {host}:{port}\n"
            f"last_error={last_error}\n"
            f"log_tail:\n{log_tail}"
        )

    time.sleep(poll_interval_s)

log_tail = ""
if log_path.exists():
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    log_tail = "\n".join(lines[-40:])
raise SystemExit(
    f"policy server did not become ready within {timeout_s:.0f}s on {host}:{port}\n"
    f"last_error={last_error}\n"
    f"log_tail:\n{log_tail}"
)
PY

CKPT="${CKPT}" \
STARVLA_DIR="${STARVLA_DIR}" \
LIBERO_HOME="${LIBERO_HOME}" \
LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH}" \
LIBERO_PYTHON="${LIBERO_PYTHON}" \
HOST="${HOST}" \
PORT="${PORT}" \
TASK_SUITE_NAME="${TASK_SUITE_NAME}" \
NUM_TRIALS_PER_TASK="${NUM_TRIALS_PER_TASK}" \
MAX_TASKS="${MAX_TASKS}" \
VIDEO_OUT_PATH="${VIDEO_OUT_PATH}" \
bash "${SCRIPT_DIR}/eval_libero.sh"

REPORT_PATH="${VIDEO_OUT_PATH}/eval_report.json"
if [[ ! -f "${REPORT_PATH}" ]]; then
    echo "Missing eval report: ${REPORT_PATH}"
    exit 1
fi

"${STARVLA_PYTHON}" - "${REPORT_PATH}" <<'PY'
import json
import pathlib
import sys

report_path = pathlib.Path(sys.argv[1])
report = json.loads(report_path.read_text(encoding="utf-8"))
print("=== Eval Regression Summary ===")
print(f"report={report_path}")
print(f"task_suite_name={report.get('task_suite_name')}")
print(f"total_episodes={report.get('total_episodes')}")
print(f"total_successes={report.get('total_successes')}")
print(f"success_rate={report.get('success_rate')}")
print(f"failure_category={report.get('failure_category')}")
PY
