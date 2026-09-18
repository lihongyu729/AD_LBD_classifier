#!/bin/bash
# run_ablation_final.sh
# 消融实验 — 分支架构对比（论文 Table X: Branch Architecture）
# Conv-only vs Mamba-only vs Dual-branch
# 验证双支路设计的价值：消解 AUC-BAC 取舍困境

export CUDA_VISIBLE_DEVICES=0,1
LOG_DIR="meta/out/ablation_logs"
mkdir -p $LOG_DIR
source ~/miniconda3/bin/activate cnn

echo "============================================================"
echo "  消融实验：分支架构对比"
echo "  Conv-only (优化AUC) vs Mamba-only (优化BAC) vs Dual-branch"
echo "============================================================"

# ===== 实验1: Conv-only =====
echo ">> [$(date +'%Y-%m-%d %H:%M:%S')] Conv-only"
python meta/train_classifier.py \
    --config meta/config_ablation_final_conv.yaml \
    --run-folds 5 \
    > $LOG_DIR/ablation_conv_final_$(date +'%Y%m%d_%H%M%S').log 2>&1

# ===== 实验2: Mamba-only =====
echo ">> [$(date +'%Y-%m-%d %H:%M:%S')] Mamba-only"
python meta/train_classifier.py \
    --config meta/config_ablation_final_mamba.yaml \
    --run-folds 5 \
    > $LOG_DIR/ablation_mamba_final_$(date +'%Y%m%d_%H%M%S').log 2>&1

# ===== 实验3: Dual-branch =====
echo ">> [$(date +'%Y-%m-%d %H:%M:%S')] Dual-branch (conv+mamba)"
python meta/train_classifier.py \
    --config meta/config_ablation_final_dual.yaml \
    --run-folds 5 \
    > $LOG_DIR/ablation_dual_final_$(date +'%Y%m%d_%H%M%S').log 2>&1

echo "============================================================"
echo "  消融实验全部完成！"
echo "  运行命令: bash run_ablation_final.sh"
echo "============================================================"
