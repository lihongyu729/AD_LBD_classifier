#!/bin/bash
# run_ablation.sh
# 三路消融实验：Conv-only vs Mamba-only vs Dual-branch
# 证明 Dual-branch 的价值：消除单支路在 AUC-BAC 之间的取舍困境
#
# 实验设计：
#   Conv-only：  提升 BAC（当前崩塌导致 BAC≈0.72）
#   Mamba-only： 提升 AUC（当前天花板 AUC≈0.76）
#   Dual-branch：不做任一支路优化，看是否能同时获得 conv 的 AUC + mamba 的 BAC

export CUDA_VISIBLE_DEVICES=0,1

LOG_DIR="meta/out/ablation_logs"
mkdir -p $LOG_DIR

source ~/miniconda3/bin/activate cnn

echo "============================================================"
echo "  消融实验：Conv-only vs Mamba-only vs Dual-branch"
echo "  假设：单支路存在 AUC-BAC 取舍，双支路可同时优化两者"
echo "============================================================"
echo ""

# ============================================================
# 实验1: Conv-only — 优化 BAC
# ============================================================
echo ">> [$(date +'%Y-%m-%d %H:%M:%S')] 实验1: Conv-only (优化BAC)"
python meta/train_classifier.py \
    --config meta/config_ablation_conv.yaml \
    --run-folds 5 \
    > $LOG_DIR/ablation_conv_$(date +'%Y%m%d_%H%M%S').log 2>&1

# ============================================================
# 实验2: Mamba-only — 优化 AUC
# ============================================================
echo ">> [$(date +'%Y-%m-%d %H:%M:%S')] 实验2: Mamba-only (优化AUC)"
python meta/train_classifier.py \
    --config meta/config_ablation_mamba.yaml \
    --run-folds 5 \
    > $LOG_DIR/ablation_mamba_$(date +'%Y%m%d_%H%M%S').log 2>&1

# ============================================================
# 实验3: Dual-branch — 不做优化，验证取舍消解假说
# ============================================================
echo ">> [$(date +'%Y-%m-%d %H:%M:%S')] 实验3: Dual-branch (conv+mamba)"
python meta/train_classifier.py \
    --config meta/config_ablation_dual.yaml \
    --run-folds 5 \
    > $LOG_DIR/ablation_dual_$(date +'%Y%m%d_%H%M%S').log 2>&1

echo ""
echo "============================================================"
echo "  消融实验全部完成！"
echo "  预期结果对比："
echo "    Conv-only:  高 AUC / 低 BAC（崩塌问题）"
echo "    Mamba-only: 低 AUC / 高 BAC（天花板问题）"
echo "    Dual-branch: 高 AUC + 高 BAC（消除取舍）"
echo "============================================================"
