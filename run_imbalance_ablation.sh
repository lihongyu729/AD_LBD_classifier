#!/bin/bash
# run_imbalance_ablation.sh
# 训练范式对比：Episodic (ANIL) vs Standard (class_weight / resample / none)
# 验证 Episodic 均衡采样的独立贡献
#
# 4 组实验：
#   A: ANIL (episodic, dual-branch)     — 已有数据，可直接引用
#   B: Standard + class weight          — 类加权 Focal Loss
#   C: Standard + resampling            — WeightedRandomSampler + 类加权 Focal Loss
#   D: Standard (no mitigation)         — 标准 CrossEntropy，不做任何处理

export CUDA_VISIBLE_DEVICES=0
LOG_DIR="meta/out/imbalance_logs"
mkdir -p $LOG_DIR
source ~/miniconda3/bin/activate cnn

echo "============================================================"
echo "  训练范式对比：Episodic vs Standard"
echo "============================================================"

# A: ANIL (episodic, dual-branch) — 如果已有结果可跳过
echo ">> [$(date +'%Y-%m-%d %H:%M:%S')] A: ANIL (episodic, dual-branch)"
python meta/train_classifier.py \
    --config meta/config_ablation_final_dual.yaml \
    --run-folds 5 \
    > $LOG_DIR/imbalance_anil_$(date +'%Y%m%d_%H%M%S').log 2>&1

# B: Standard + class weight
echo ">> [$(date +'%Y-%m-%d %H:%M:%S')] B: Standard + class weight"
python meta/train_classifier_baseline.py \
    --config meta/config_imbalance_standard_weighted.yaml \
    --run-folds 5 \
    > $LOG_DIR/imbalance_std_weighted_$(date +'%Y%m%d_%H%M%S').log 2>&1

# C: Standard + resampling
echo ">> [$(date +'%Y-%m-%d %H:%M:%S')] C: Standard + resampling"
python meta/train_classifier_baseline.py \
    --config meta/config_imbalance_standard_resample.yaml \
    --run-folds 5 \
    > $LOG_DIR/imbalance_std_resample_$(date +'%Y%m%d_%H%M%S').log 2>&1

# D: Standard (no mitigation)
echo ">> [$(date +'%Y-%m-%d %H:%M:%S')] D: Standard (no mitigation)"
python meta/train_classifier_baseline.py \
    --config meta/config_imbalance_standard_none.yaml \
    --run-folds 5 \
    > $LOG_DIR/imbalance_std_none_$(date +'%Y%m%d_%H%M%S').log 2>&1

echo "============================================================"
echo "  全部完成！"
echo "  日志目录: $LOG_DIR"
echo "============================================================"
