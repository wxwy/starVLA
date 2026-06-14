#!/usr/bin/env bash
# Install LIBERO environment for evaluation.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)

LIBERO_CONDA_ENV=${LIBERO_CONDA_ENV:-libero}
LIBERO_PARENT_DIR=${LIBERO_PARENT_DIR:-${REPO_ROOT}}
LIBERO_DIR=${LIBERO_DIR:-${LIBERO_PARENT_DIR}/LIBERO}
VENV_DIR=${VENV_DIR:-${REPO_ROOT}/.libero}
SKIP_CONDA_ACTIVATE=${SKIP_CONDA_ACTIVATE:-0}
USE_LOCAL_VENV=${USE_LOCAL_VENV:-auto}

echo "=== Step 1: Activate LIBERO Python environment ==="
if [[ "${USE_LOCAL_VENV}" == "1" || "${USE_LOCAL_VENV}" == "auto" && -f "${VENV_DIR}/bin/activate" ]]; then
    source "${VENV_DIR}/bin/activate"
elif [[ "${SKIP_CONDA_ACTIVATE}" != "1" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda not found. Either initialize conda first, create ${VENV_DIR}, or run with SKIP_CONDA_ACTIVATE=1."
        exit 1
    fi
    eval "$(conda shell.bash hook)"
    conda activate "${LIBERO_CONDA_ENV}"
else
    echo "Skipping environment activation; using current python: $(command -v python)"
fi

echo "=== Step 2: Install MuJoCo and eval dependencies ==="
python -m pip install mujoco==3.2.3
python -m pip install tyro matplotlib mediapy websockets msgpack
python -m pip install numpy==1.24.4

echo "=== Step 3: Prepare LIBERO source ==="
mkdir -p "${LIBERO_PARENT_DIR}"
if [[ ! -d "${LIBERO_DIR}" ]]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "${LIBERO_DIR}"
else
    echo "LIBERO already exists at ${LIBERO_DIR}"
fi

cd "${LIBERO_DIR}"

echo "=== Step 4: Install LIBERO dependencies ==="
python -m pip install -r requirements.txt

echo "=== Step 5: Install LIBERO (editable) ==="
python -m pip install -e .

echo "=== Step 6: Verify installation ==="
python -c "from libero.libero import benchmark; print('LIBERO OK:', benchmark)"
python -c "import mujoco; print('MuJoCo OK:', mujoco.__version__)"
python -c "import robosuite; print('robosuite OK:', robosuite.__version__)"
python -c "import bddl; print('bddl OK:', getattr(bddl, '__version__', 'unknown'))"
python -c "import tyro; print('tyro OK')"
python -c "import websockets; print('websockets OK')"

echo "=== ALL DONE ==="
echo "LIBERO_DIR=${LIBERO_DIR}"
