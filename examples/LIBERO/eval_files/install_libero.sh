#!/bin/bash
# Install LIBERO environment for evaluation
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/../../.." && pwd)
VENV_DIR="${REPO_ROOT}/.libero"
LIBERO_DIR="${LIBERO_DIR:-${REPO_ROOT}/LIBERO}"

echo "=== Step 1: Activate libero venv ==="
if [ ! -f "${VENV_DIR}/bin/activate" ]; then
    echo "Missing virtualenv: ${VENV_DIR}"
    exit 1
fi
source "${VENV_DIR}/bin/activate"

echo "=== Step 2: Install mujoco ==="
python -m pip install mujoco==3.2.3

echo "=== Step 3: Prepare LIBERO source ==="
if [ ! -d "${LIBERO_DIR}" ]; then
    echo "LIBERO source not found at ${LIBERO_DIR}"
    echo "Please clone LIBERO there first, or export LIBERO_DIR=/path/to/LIBERO"
    exit 1
fi
cd "${LIBERO_DIR}"

echo "=== Step 4: Install LIBERO dependencies ==="
python -m pip install -r requirements.txt

echo "=== Step 5: Install LIBERO (editable) ==="
python -m pip install -e .

echo "=== Step 6: Install additional eval deps ==="
python -m pip install tyro matplotlib mediapy websockets msgpack
python -m pip install numpy==1.24.4

echo "=== Step 7: Verify installation ==="
python -c "from libero.libero import benchmark; print('LIBERO OK:', benchmark)"
python -c "import mujoco; print('MuJoCo OK:', mujoco.__version__)"
python -c "import robosuite; print('robosuite OK:', robosuite.__version__)"
python -c "import bddl; print('bddl OK:', getattr(bddl, '__version__', 'unknown'))"
python -c "import tyro; print('tyro OK')"
python -c "import websockets; print('websockets OK')"

echo "=== ALL DONE ==="
