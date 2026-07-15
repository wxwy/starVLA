#!/usr/bin/env bash
# nohup 包装脚本：启动 server → 硬检查门禁 → 运行 eval → 关闭 server
# 完全脱离 shell 会话运行，不受上下文切换影响
# 用法: bash nohup_eval.sh CKPT_DIR STEP_NAME PORT SUITE_NAME > log.log 2>&1 &

set -euo pipefail

STARVLA_PY=/opt/conda/envs/starVLA/bin/python
LIBERO_PY=/disk/rl/starVLA/.libero/bin/python

CKPT_DIR="$1"
STEP_NAME="$2"
PORT="${3:-6694}"
SUITE_NAME="$4"

EXP_NAME=$(basename "$(dirname "$(dirname "$CKPT_DIR")")")
VIDEO_DIR="/disk/rl/starVLA/playground/eval_results/${SUITE_NAME}/${EXP_NAME}/${STEP_NAME}"

export LIBERO_CONFIG_PATH=/disk/rl/starVLA/LIBERO/libero/libero
export MUJOCO_GL=egl
export PYTHONPATH="/disk/rl/starVLA/LIBERO:/disk/rl/starVLA:${PYTHONPATH:-}"
export LIBERO_HOME=/disk/rl/starVLA/LIBERO

echo "[$(date)] 🚀 启动: ${EXP_NAME} | ${STEP_NAME} | ${SUITE_NAME} | port ${PORT}"
echo "[$(date)] 📁 输出: ${VIDEO_DIR}"

# 0. 🔒 互斥门禁：检查结果目录防止多机冲突
if [ -d "$VIDEO_DIR" ]; then
    EVAL_REPORT="${VIDEO_DIR}/eval_report.json"
    if [ -f "$EVAL_REPORT" ]; then
        # 有报告文件 → 检查是否已完成
        COMPLETE=$(python3 -c "
import json
r=json.load(open('$EVAL_REPORT'))
print('yes' if r.get('total_episodes',0) >= 500 else 'no')
" 2>/dev/null || echo "no")
        if [ "$COMPLETE" = "yes" ]; then
            echo "[$(date)] ✅ ${SUITE_NAME} 已完成 (${EVAL_REPORT})，跳过"
            exit 0
        fi
        # 未完成 → resume 模式（由后续 resume 逻辑接管）
        echo "[$(date)] 🔄 发现已有进度，进入 resume 流程"
    else
        # 目录存在但无报告 → 可能是另一台机器正在初始化，冲突风险
        echo "[$(date)] ❌ 结果目录已存在但无报告文件: ${VIDEO_DIR}"
        echo "[$(date)] ❌ 可能被其他机器占用。若确认安全请手动删除目录后重试。"
        exit 1
    fi
fi

# 1. 启动 server
CUDA_VISIBLE_DEVICES=0 $STARVLA_PY deployment/model_server/server_policy.py \
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

# 4. 运行 eval（支持 resume）
echo "[$(date)] 开始评估..."
mkdir -p "$VIDEO_DIR"
cd /disk/rl/starVLA

RESUME_FLAG=""
EVAL_REPORT="${VIDEO_DIR}/eval_report.json"
if [ -f "$EVAL_REPORT" ]; then
    # 有已有结果，检查 ckpt 是否一致
    EXISTING_CKPT=$(python3 -c "import json; print(json.load(open('$EVAL_REPORT')).get('checkpoint_path',''))" 2>/dev/null || echo "")
    EXPECTED_CKPT_PATH=$(cd "$CKPT_DIR" && pwd)
    if [ "$EXISTING_CKPT" = "$EXPECTED_CKPT_PATH" ]; then
        RESUME_FLAG="--args.resume-eval"
        echo "[$(date)] 🔄 发现已有结果，启用 resume 模式"
    else
        echo "[$(date)] ⚠️  已有结果 ckpt 不匹配(=$EXISTING_CKPT)，覆盖重跑"
        rm -f "$EVAL_REPORT"
    fi
fi

$LIBERO_PY examples/LIBERO/eval_files/eval_libero.py \
  --args.pretrained-path "$CKPT_DIR" \
  --args.host 127.0.0.1 \
  --args.port $PORT \
  --args.task-suite-name "$SUITE_NAME" \
  --args.num-trials-per-task 50 \
  --args.video-out-path "$VIDEO_DIR" \
  $RESUME_FLAG 2>&1

# 5. 输出结果
EVAL_REPORT="${VIDEO_DIR}/eval_report.json"
if [ -f "$EVAL_REPORT" ]; then
    SR=$(python3 -c "import json; r=json.load(open('$EVAL_REPORT')); print(f'{r[\"success_rate\"]*100:.1f}%')")
    TS=$(python3 -c "import json; r=json.load(open('$EVAL_REPORT')); print(r['total_successes'])")
    TE=$(python3 -c "import json; r=json.load(open('$EVAL_REPORT')); print(r['total_episodes'])")
    echo "[$(date)] ✅ 完成: ${SUITE_NAME} | ${EXP_NAME} | ${STEP_NAME} | SR=${SR} (${TS}/${TE})"
else
    echo "[$(date)] ❌ 未找到评估报告!"
fi

# 6. 关闭 server
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "[$(date)] 🏁 全部完成"
