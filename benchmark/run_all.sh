#!/bin/bash
# ===========================================================================
# AD vs LBD Benchmark — 一键运行全部实验（嵌套5折CV，单 seed）
# 用法:
#   bash run_all.sh                    # 全部方法 × 单seed42（嵌套5折，报告test）
#   bash run_all.sh --quick            # 快速测试 (2-fold, 3-epoch)
#   bash run_all.sh --pure             # 纯5折CV（不做holdout拆分）
#   bash run_all.sh --group A          # 只跑 Group A（元学习策略）
#   bash run_all.sh --group B          # 只跑 Group B（骨干网络）
# ===========================================================================
set -e

CONDA_ENV="${CONDA_ENV:-cnn}"
GPU="${GPU:-0}"
SEEDS="42"                     # 单 seed（重跑要求，不做多 seed）
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# 激活环境
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$CONDA_ENV"

cd "$SCRIPT_DIR"

ARGS="--gpus $GPU --seeds $SEEDS --resume --holdout-test 0.2"

# 解析参数
MODE="nested"
for arg in "$@"; do
    case $arg in
        --quick)   ARGS="$ARGS --folds 2 --epochs 3"; MODE="quick" ;;
        --pure)    ARGS="$ARGS --holdout-test 0.0"; MODE="pure_cv" ;;
        --group)   GROUP_MODE="group" ;;
        A)         METHOD_FILTER="--group A"; RUN_TAG="GroupA" ;;
        B)         METHOD_FILTER="--group B"; RUN_TAG="GroupB" ;;
    esac
done

# ===========================================================================
# 主流程
# ===========================================================================
echo "=============================================="
echo "  AD vs LBD Benchmark — $TIMESTAMP"
echo "  Mode: $MODE | GPU: $GPU | Seeds: $SEEDS"
echo "  Logs: $LOG_DIR"
echo "=============================================="

if [ -n "$METHOD_FILTER" ]; then
    # 只跑指定组
    LOGFILE="$LOG_DIR/${RUN_TAG}_${TIMESTAMP}.log"
    echo "[$(date)] Running $RUN_TAG..." | tee -a "$LOGFILE"
    python scripts/run_batch.py $METHOD_FILTER $ARGS 2>&1 | tee -a "$LOGFILE"
else
    # 跑全部：先 Group A 再 Group B
    LOGFILE_A="$LOG_DIR/GroupA_${TIMESTAMP}.log"
    LOGFILE_B="$LOG_DIR/GroupB_${TIMESTAMP}.log"

    echo "[$(date)] Running Group A (meta-learning strategies)..." | tee -a "$LOGFILE_A"
    python scripts/run_batch.py --group A $ARGS 2>&1 | tee -a "$LOGFILE_A"

    echo "[$(date)] Running Group B (backbone architectures)..." | tee -a "$LOGFILE_B"
    python scripts/run_batch.py --group B $ARGS 2>&1 | tee -a "$LOGFILE_B"
fi

# 生成报告
echo "[$(date)] Generating comparison report..."
python scripts/generate_report.py --input ./results --output ./reports 2>&1 | tee -a "$LOG_DIR/report_${TIMESTAMP}.log"

echo ""
echo "=============================================="
echo "  All experiments completed!"
echo "  Results: $SCRIPT_DIR/results/"
echo "  Reports: $SCRIPT_DIR/reports/"
echo "  Logs:    $LOG_DIR/"
echo "=============================================="
