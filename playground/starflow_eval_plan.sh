#!/usr/bin/env bash
# StarFlow state-fix 通用评估计划脚本
# 按策略：单 policy + N workers，一个 ckpt 跑完 4 suites 再换下一个
#
# 用法示例：
#   # 当前机器：ft16 / ft32 / cont.ft32
#   bash playground/starflow_eval_plan.sh \
#     --exp ft16,ft32,cont.ft32 \
#     --steps 5000,10000,20000,30000,40000,50000,60000,70000,80000
#
#   # 另一台机器：ft0 / ft64
#   bash playground/starflow_eval_plan.sh \
#     --exp ft0,ft64 \
#     --steps 5000,10000,20000,30000,40000,50000,60000,70000,80000
#
# 可选参数：
#   --workers N   并发 worker 数（默认 20）
#   --trials N    每个 task 的 trial 数（默认 50）
#   --ckpt-base   checkpoint 根目录（默认 playground/Checkpoints）
#
# 端口规划（按实验+step固定分配，避免串行复用，便于核对）：
#   ft0:       6700-6709  (base 6700 + step_index)
#   ft16:      6710-6719  (base 6710 + step_index)
#   ft32:      6720-6729  (base 6720 + step_index)
#   ft64:      6730-6739  (base 6730 + step_index)
#   cont.ft32: 6740-6749  (base 6740 + step_index)

set -euo pipefail

# 实验简称 -> 完整目录名映射
declare -A EXP_DIRS
EXP_DIRS[ft0]="P0-M7-E-H2a-01_starflow_libero-4in1_qwen3vl4b_lwfm_ft0_260703_0849"
EXP_DIRS[ft16]="P0-M7-E-H2a-02_starflow_libero-4in1_qwen3vl4b_lwfm_ft16_260703_0854"
EXP_DIRS[ft32]="P0-M5-E-H2a-03_starflow_libero-4in1_qwen3vl4b_lwfm_ft32_260703_0848"
EXP_DIRS[ft64]="P0-M7-E-H2a-04_starflow_libero-4in1_qwen3vl4b_lwfm_ft64_260702_2010"
EXP_DIRS[cont.ft32]="P1-M1-E-H2b-02_starflow_libero-4in1_qwen3vl4b_lwfm_continuous_ft32_260702_2021"

# 实验基础端口映射
declare -A EXP_BASE_PORTS
EXP_BASE_PORTS[ft0]=6700
EXP_BASE_PORTS[ft16]=6710
EXP_BASE_PORTS[ft32]=6720
EXP_BASE_PORTS[ft64]=6730
EXP_BASE_PORTS[cont.ft32]=6740

# step -> 固定索引映射（保证即使 step 列表乱序，端口也固定）
declare -A STEP_INDEX
STEP_INDEX[5000]=0
STEP_INDEX[10000]=1
STEP_INDEX[20000]=2
STEP_INDEX[30000]=3
STEP_INDEX[40000]=4
STEP_INDEX[50000]=5
STEP_INDEX[60000]=6
STEP_INDEX[70000]=7
STEP_INDEX[80000]=8
STEP_INDEX[75500]=9

# 默认值
EXP_LIST=""
STEP_LIST=""
WORKERS=10
NUM_TRIALS=50
CKPT_BASE="/disk/rl/starVLA/playground/Checkpoints"

# 解析参数
while [[ $# -gt 0 ]]; do
  case $1 in
    --exp)
      EXP_LIST="$2"; shift 2 ;;
    --steps)
      STEP_LIST="$2"; shift 2 ;;
    --workers)
      WORKERS="$2"; shift 2 ;;
    --trials)
      NUM_TRIALS="$2"; shift 2 ;;
    --ckpt-base)
      CKPT_BASE="$2"; shift 2 ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: bash $0 --exp ft16,ft32 --steps 5000,10000,..."
      exit 1
      ;;
  esac
done

if [[ -z "$EXP_LIST" || -z "$STEP_LIST" ]]; then
  echo "Usage: bash $0 --exp ft16,ft32 --steps 5000,10000,..."
  exit 1
fi

# 逗号分割
IFS=',' read -ra EXPS <<< "$EXP_LIST"
IFS=',' read -ra STEPS <<< "$STEP_LIST"

# 端口计算函数
get_port() {
    local label=$1
    local step_num=$2
    local base_port=${EXP_BASE_PORTS[$label]:-}
    local idx=${STEP_INDEX[$step_num]:-}
    if [[ -z "$base_port" || -z "$idx" ]]; then
        echo ""
        return
    fi
    echo $((base_port + idx))
}

EVAL_ROOT="/disk/rl/starVLA/playground/starflow_eval_result"
mkdir -p "$EVAL_ROOT/logs"
LOG="$EVAL_ROOT/logs/starflow_eval_plan_$(date +%m%d_%H%M).log"
echo "[$(date)] 🚀 评估计划启动" | tee -a "$LOG"
echo "[$(date)] 负责实验: ${EXP_LIST}" | tee -a "$LOG"
echo "[$(date)] 待测 steps: ${STEP_LIST}" | tee -a "$LOG"
echo "[$(date)] workers: ${WORKERS}, trials: ${NUM_TRIALS}" | tee -a "$LOG"
echo "[$(date)] ckpt base: ${CKPT_BASE}" | tee -a "$LOG"
echo "[$(date)] eval root: $EVAL_ROOT" | tee -a "$LOG"

# 打印端口分配表
echo "[$(date)] 📋 端口分配表：" | tee -a "$LOG"
for label in "${EXPS[@]}"; do
  base_port=${EXP_BASE_PORTS[$label]:-}
  [[ -z "$base_port" ]] && continue
  echo "[$(date)]   $label (base $base_port):" | tee -a "$LOG"
  for step_num in "${STEPS[@]}"; do
    port=$(get_port "$label" "$step_num")
    [[ -z "$port" ]] && continue
    echo "[$(date)]     steps_$step_num -> port $port" | tee -a "$LOG"
  done
done

for label in "${EXPS[@]}"; do
  exp_dir=${EXP_DIRS[$label]:-}
  if [[ -z "$exp_dir" ]]; then
    echo "[$(date)] ❌ 未知实验: $label，跳过" | tee -a "$LOG"
    continue
  fi
  echo "[$(date)] ====== 实验: $label ($exp_dir) ======" | tee -a "$LOG"
  for step_num in "${STEPS[@]}"; do
    step="steps_${step_num}"
    ckpt_dir="$CKPT_BASE/$exp_dir/checkpoints/$step"
    port=$(get_port "$label" "$step_num")
    if [[ -z "$port" ]]; then
      echo "[$(date)] ❌ $label $step 无端口映射，跳过" | tee -a "$LOG"
      continue
    fi
    if [ ! -d "$ckpt_dir" ]; then
      echo "[$(date)] ⚠️ $label $step 不存在，跳过 (port $port)" | tee -a "$LOG"
      continue
    fi
    echo "[$(date)] 🚀 开始: $label $step | server/client port=${port} | workers=${WORKERS}" | tee -a "$LOG"
    bash /disk/rl/starVLA/playground/starflow_eval_pool.sh \
      "$ckpt_dir" \
      "$step" \
      "$port" \
      "$WORKERS" \
      "$NUM_TRIALS" \
      2>&1 | tee -a "$LOG"
    echo "[$(date)] ✅ 完成: $label $step (port ${port})" | tee -a "$LOG"
  done
done

echo "[$(date)] 🏁 全部计划完成" | tee -a "$LOG"
