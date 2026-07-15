#!/usr/bin/env bash
# nohup 多 Worker 任务池包装脚本
# 启动 1 个 server → 硬检查门禁 → 多 Worker 并发评估全部 suite → 关闭 server
# 用法: bash nohup_eval_pool.sh CKPT_DIR STEP_NAME PORT WORKERS > log.log 2>&1 &

set -euo pipefail

STARVLA_PY=/opt/conda/envs/starVLA/bin/python
LIBERO_PY=/disk/rl/starVLA/.libero/bin/python

CKPT_DIR="$1"
STEP_NAME="$2"
PORT="${3:-6694}"
WORKERS="${4:-2}"
NUM_TRIALS="${5:-50}"

EXP_NAME=$(basename "$(dirname "$(dirname "$CKPT_DIR")")")
EVAL_ROOT="/disk/rl/starVLA/playground/eval_results"

export LIBERO_CONFIG_PATH=/disk/rl/starVLA/LIBERO/libero/libero
export MUJOCO_GL=egl
export PYTHONPATH="/disk/rl/starVLA/LIBERO:/disk/rl/starVLA:${PYTHONPATH:-}"
export LIBERO_HOME=/disk/rl/starVLA/LIBERO

echo "[$(date)] 🚀 多 Worker 池启动: ${EXP_NAME} | ${STEP_NAME} | port ${PORT} | workers ${WORKERS}"

# 0. 🔒 互斥门禁：检查所有 4 个 suite 的结果主目录
ALL_DONE=true
for SUITE in libero_goal libero_10 libero_object libero_spatial; do
    SUITE_DIR="${EVAL_ROOT}/${SUITE}/${EXP_NAME}/${STEP_NAME}"
    EVAL_REPORT="${SUITE_DIR}/eval_report.json"
    if [ -f "$EVAL_REPORT" ]; then
        SUITE_EXPECTED=$(( NUM_TRIALS * 10 ))
        COMPLETE=$(python3 -c "
import json
r=json.load(open('$EVAL_REPORT'))
ep=r.get('total_episodes',0)
print('yes' if ep >= $SUITE_EXPECTED else 'no')
" 2>/dev/null || echo "no")
        if [ "$COMPLETE" = "yes" ]; then
            echo "[$(date)] ✅ ${SUITE} 已完成，跳过"
            continue
        fi
        echo "[$(date)] 🔄 ${SUITE} 有局部进度，会通过 resume 继续"
        ALL_DONE=false
    elif [ -d "$SUITE_DIR" ]; then
        # 无报告但有目录 → 检查扁平格式的 task 报告
        FLAT_TASK_COUNT=$(find "$SUITE_DIR" -maxdepth 1 -name "eval_report_task_*.json" 2>/dev/null | wc -l)
        TASK_DIR_COUNT=$(find "$SUITE_DIR" -maxdepth 1 -type d -name "task_*" 2>/dev/null | wc -l)
        if [ "$FLAT_TASK_COUNT" -gt 0 ] || [ "$TASK_DIR_COUNT" -gt 0 ]; then
            echo "[$(date)] ⚠️  ${SUITE} 有 ${FLAT_TASK_COUNT} 个扁平报告 / ${TASK_DIR_COUNT} 个 task 子目录，可能被其他机器占用"
            # 统计已完成 task 数
            DONE_TASKS=$(python3 -c "
import json, glob, sys
path='${SUITE_DIR}/eval_report_task_*.json'
files=glob.glob(path)
done=0
for f in files:
    try:
        r=json.load(open(f))
        ep=r.get('total_episodes',0)
        if ep>=${NUM_TRIALS}:
            done+=1
    except:
        pass
print(done)
" 2>/dev/null || echo "0")
            echo "[$(date)]   ${DONE_TASKS}/10 tasks 已完成，进入 resume 流程"
            ALL_DONE=false
        else
            # 空目录 → 视为未启动，删除重建
            rmdir "$SUITE_DIR" 2>/dev/null || true
            echo "[$(date)] 🆕 ${SUITE} 目录为空，重置为未启动状态"
            ALL_DONE=false
        fi
        ALL_DONE=false
    else
        ALL_DONE=false
    fi
done

if [ "$ALL_DONE" = true ]; then
    echo "[$(date)] ✅ 所有 4 个 suite 均已完成，退出"
    exit 0
fi

# 1. 启动 server（只启动 1 个）
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

# 3. 🔒 硬检查门禁
SERVER_PID_ON_PORT=$(ss -tlnp 2>/dev/null | grep ":$PORT " | grep -oP 'pid=\K[0-9]+' | head -1)
SERVER_CKPT=$(cat /proc/$SERVER_PID_ON_PORT/cmdline 2>/dev/null | tr '\0' ' ' | tr ' ' '\n' | grep -A1 ckpt_path | tail -1 || echo "")
EXPECTED_CKPT=$(cd "$CKPT_DIR" && pwd)
if [ "$SERVER_CKPT" != "$EXPECTED_CKPT" ]; then
    echo "[$(date)] ❌❌❌ 门禁失败！"
    echo "  预期: ${EXPECTED_CKPT}"
    echo "  实际: ${SERVER_CKPT}"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi
echo "[$(date)] 🔒 门禁通过 ✅"

# 4. 运行多 Worker 任务池
echo "[$(date)] 🏊 启动任务池: ${WORKERS} workers..."
cd /disk/rl/starVLA

# 自动检测是否已有部分结果 → 启用 resume
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
    echo "[$(date)] ⚠️  任务池退出码: ${POOL_EXIT}"
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
echo "[$(date)] 🏁 全部完成"
