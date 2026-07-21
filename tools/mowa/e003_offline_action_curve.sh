#!/usr/bin/env bash
# E-003 离线动作一致性检查的多 checkpoint 批跑器（学习曲线）。
# 逐个启动 policy server、运行 tools/mowa/e003_offline_action_check.py、关闭 server。
#
# 用法:
#   bash tools/mowa/e003_offline_action_curve.sh 5000 10000 20000 30000
#
# 环境变量覆盖:
#   CKPT_ROOT   checkpoint 根目录（默认 E-003 260718_0039 的 checkpoints/）
#   DATA        lerobot 数据集目录（默认 OpenDrawer target/human）
#   OUT         报告输出目录（默认 playground/tmp/mowa_offline_curve）
#   PORT        server 端口（默认 6791）
#   EPISODES    回放 episode 列表（默认 "0 1 2"）
#   STRIDE      采样间隔帧数（默认 15）
#   HORIZON     对比的 chunk 步数（默认 8，与闭环 N_ACT 对齐）
set -u
REPO=/disk/rl/starVLA
CKPT_ROOT=${CKPT_ROOT:-$REPO/playground/mowa_ckpt/MoWA-E-003_future_latent_prior_wo_history_lora_260718_0039/checkpoints}
DATA=${DATA:-$REPO/playground/Datasets/robocasa365/v1.0/target/atomic/OpenDrawer/20250816/lerobot}
OUT=${OUT:-$REPO/playground/tmp/mowa_offline_curve}
PORT=${PORT:-6791}
EPISODES=${EPISODES:-"0 1 2"}
STRIDE=${STRIDE:-15}
HORIZON=${HORIZON:-8}
mkdir -p "$OUT"

for STEP in "$@"; do
  CKPT=$CKPT_ROOT/steps_$STEP
  echo "===== steps_$STEP ====="
  if [[ ! -d "$CKPT" ]]; then echo "skip: $CKPT missing"; continue; fi
  CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONUNBUFFERED=1 "$REPO/.venv/bin/python" \
    "$REPO/deployment/model_server/server_policy.py" \
    --ckpt_path "$CKPT" --port $PORT --use_bf16 \
    --batch_timeout_ms 100 --max_batch_size 8 \
    > "$OUT/server_$STEP.log" 2>&1 &
  SPID=$!
  READY=0
  for _ in $(seq 1 90); do
    if grep -q "server running" "$OUT/server_$STEP.log" 2>/dev/null; then READY=1; break; fi
    if ! kill -0 $SPID 2>/dev/null; then echo "server died"; break; fi
    sleep 2
  done
  if [[ $READY == 1 ]]; then
    PYTHONUNBUFFERED=1 PYTHONPATH=$REPO "$REPO/.venv/bin/python" \
      "$REPO/tools/mowa/e003_offline_action_check.py" \
      --dataset-dir "$DATA" --ckpt-path "$CKPT" \
      --episodes $EPISODES --stride "$STRIDE" --compare-horizon "$HORIZON" \
      --port $PORT --output "$OUT/offline_$STEP.json" 2>&1 | tail -6
  fi
  kill $SPID 2>/dev/null
  for _ in $(seq 1 20); do kill -0 $SPID 2>/dev/null || break; sleep 0.5; done
  kill -9 $SPID 2>/dev/null
  wait $SPID 2>/dev/null
  sleep 3
done
echo "ALL DONE -> $OUT"
