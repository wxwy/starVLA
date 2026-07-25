#!/usr/bin/env bash
# StarFlow state-fix 评估 pool
# 一个 policy server + N workers，输出到 playground/starflow_eval_result
# 用法: bash playground/starflow_eval_pool.sh CKPT_DIR STEP_NAME PORT WORKERS [NUM_TRIALS]

set -euo pipefail

STARVLA_PY=/opt/conda/envs/starVLA/bin/python
LIBERO_PY=/disk/rl/starVLA/.libero/bin/python

CKPT_DIR="$1"
STEP_NAME="$2"
PORT="${3:-6694}"
WORKERS="${4:-20}"
NUM_TRIALS="${5:-50}"

EXP_NAME=$(basename "$(dirname "$(dirname "$CKPT_DIR")")")
EVAL_ROOT="/disk/rl/starVLA/playground/starflow_eval_result"

export LIBERO_CONFIG_PATH=/disk/rl/starVLA/LIBERO/libero/libero
export MUJOCO_GL=glfw
export PYTHONPATH="/disk/rl/starVLA/LIBERO:/disk/rl/starVLA:${PYTHONPATH:-}"
export LIBERO_HOME=/disk/rl/starVLA/LIBERO

echo "[$(date)] 🚀 启动: ${EXP_NAME} | ${STEP_NAME} | port ${PORT} | workers ${WORKERS} | output ${EVAL_ROOT}"

# 1. 启动 server
CUDA_VISIBLE_DEVICES=0 $STARVLA_PY /disk/rl/starVLA/deployment/model_server/server_policy.py \
  --ckpt_path "$CKPT_DIR" \
  --port $PORT \
  --use_bf16 &
SERVER_PID=$!
echo "[$(date)] Server PID: ${SERVER_PID}"

# 2. 等待 server 就绪
for i in $(seq 1 90); do
    sleep 2
    if ss -tlnp 2>/dev/null | grep -q ":$PORT "; then
        echo "[$(date)] Server 就绪 (${i}x2s)"
        break
    fi
    if [ $i -eq 90 ]; then
        echo "[$(date)] ❌ Server 启动超时"
        kill $SERVER_PID 2>/dev/null || true
        exit 1
    fi
done

# 3. 硬检查门禁
SERVER_PID_ON_PORT=$(ss -tlnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1)
SERVER_CKPT=$(cat /proc/$SERVER_PID_ON_PORT/cmdline 2>/dev/null | tr '\0' ' ' | tr ' ' '\n' | grep -A1 ckpt_path | tail -1 || echo "")
EXPECTED_CKPT=$(cd "$CKPT_DIR" && pwd)
if [ "$SERVER_CKPT" != "$EXPECTED_CKPT" ]; then
    echo "[$(date)] ❌ 门禁失败"
    echo "  预期: ${EXPECTED_CKPT}"
    echo "  实际: ${SERVER_CKPT}"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi
echo "[$(date)] 🔒 门禁通过 ✅"

# 4. 运行多 worker 任务池
echo "[$(date)] 🏊 启动任务池: ${WORKERS} workers..."
cd /disk/rl/starVLA

# 新目录下若已有部分结果，支持 resume
RESUME_FLAG=""
for SUITE in libero_goal libero_10 libero_object libero_spatial; do
    if ls "${EVAL_ROOT}/${SUITE}/${EXP_NAME}/${STEP_NAME}"/eval_report_task_*.json 2>/dev/null | head -1 | grep -q .; then
        RESUME_FLAG="--resume"
        echo "[$(date)] 🔄 检测到已有任务报告，启用 resume 模式"
        break
    fi
done

$LIBERO_PY playground/eval_pool_manager.py \
  --ckpt-path "$CKPT_DIR" \
  --port $PORT \
  --workers $WORKERS \
  --video-root "$EVAL_ROOT" \
  --num-trials $NUM_TRIALS $RESUME_FLAG 2>&1

POOL_EXIT=$?
if [ $POOL_EXIT -ne 0 ]; then
    echo "[$(date)] ⚠️ 任务池退出码: ${POOL_EXIT}"
fi

# 5. 输出汇总
echo "[$(date)] ====== 评估汇总 ======"
for SUITE in libero_goal libero_10 libero_object libero_spatial; do
    EVAL_REPORT="${EVAL_ROOT}/${SUITE}/${EXP_NAME}/${STEP_NAME}/eval_report.json"
    if [ -f "$EVAL_REPORT" ]; then
        SR=$(python3 -c "import json; r=json.load(open('$EVAL_REPORT')); print(f'{r[\"success_rate\"]*100:.1f}%')" 2>/dev/null || echo "?")
        echo "[$(date)]   ${SUITE}: SR=${SR}"
    fi
done
echo "[$(date)] ====================="

# 6. 关闭 server
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "[$(date)] 🏁 完成"
