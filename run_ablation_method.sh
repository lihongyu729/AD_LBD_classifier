#!/bin/bash
# run_ablation_method.sh
# 方法论消融实验 — 对应论文 §4.3
# 验证 SS3D / MAE预训练 / 三层不平衡缓解 的独立贡献
#
# 5个实验：
#   1) Full MedMambaSS3D（基线，复用已有 config_ablation_imbalance_base.yaml）
#   2) −SS3D（单方向纯Conv，n_dirs=1）
#   3) −MAE预训练（随机初始化，config_ablation_no_pretrain.yaml）
#   4) −差分学习率（backbone_lr_ratio=1.0）
#   5) −加权损失（use_class_weights=false）
#   6) −Episodic平衡采样（vanilla 非元学习）

export CUDA_VISIBLE_DEVICES=0
LOG_DIR="meta/out/ablation_logs"
mkdir -p $LOG_DIR
source ~/miniconda3/bin/activate cnn

BASE_CONFIG="meta/config_ablation_imbalance_base.yaml"
NO_PRETRAIN_CONFIG="meta/config_ablation_no_pretrain.yaml"

echo "============================================================"
echo "  方法论消融实验 — 论文 §4.3"
echo "============================================================"

# ===== 实验1：Full MedMambaSS3D 基线 =====
echo ">> [$(date +'%Y-%m-%d %H:%M:%S')] 实验1: Full MedMambaSS3D (baseline)"
python meta/train_classifier.py --config $BASE_CONFIG --run-folds 5 \
    > $LOG_DIR/ablation_full_$(date +'%Y%m%d_%H%M%S').log 2>&1

# ===== 实验2：−SS3D（替换为单方向纯Conv SSM） =====
# 关闭8方向扫描 + 两支路均替换为标准Conv-SSM（无选择性扫描）
echo ">> [$(date +'%Y%m%d_%H%M%S')] 实验2: -SS3D (single-dir conv SSM)"
python meta/train_classifier.py --config $BASE_CONFIG --run-folds 5 \
    ss3m.branch_types="['conv','conv']" \
    classifier.backbone_use_dual_branch=False \
    classifier.backbone_n_dirs_train=1 \
    paths.out_dir=~/MRI/pretrain/code/meta/out/ablation/no_ss3d \
    > $LOG_DIR/ablation_no_ss3d_$(date +'%Y%m%d_%H%M%S').log 2>&1

# ===== 实验3：−MAE预训练（随机初始化） =====
echo ">> [$(date +'%Y%m%d_%H%M:%S')] 实验3: -MAE pretrain (random init)"
python meta/train_classifier.py --config $NO_PRETRAIN_CONFIG --run-folds 5 \
    > $LOG_DIR/ablation_no_pretrain_$(date +'%Y%m%d_%H%M%S').log 2>&1

# ===== 实验4：−差分学习率（backbone与head同lr） =====
echo ">> [$(date +'%Y%m%d_%H%M:%S')] 实验4: -DiffLR (backbone_lr_ratio=1.0)"
python meta/train_classifier.py --config $BASE_CONFIG --run-folds 5 \
    classifier.backbone_lr_ratio=1.0 \
    paths.out_dir=~/MRI/pretrain/code/meta/out/ablation/no_difflr \
    > $LOG_DIR/ablation_no_difflr_$(date +'%Y%m%d_%H%M%S').log 2>&1

# ===== 实验5：−加权损失（标准交叉熵） =====
echo ">> [$(date +'%Y%m%d_%H%M:%S')] 实验5: -WeightedLoss (standard CE)"
python meta/train_classifier.py --config $BASE_CONFIG --run-folds 5 \
    classifier.use_class_weights=False \
    paths.out_dir=~/MRI/pretrain/code/meta/out/ablation/no_weighted_loss \
    > $LOG_DIR/ablation_no_weightloss_$(date +'%Y%m%d_%H%M%S').log 2>&1

# ===== 实验6：−Episodic平衡采样（标准随机采样） =====
echo ">> [$(date +'%Y%m%d_%H%M:%S')] 实验6: -Episodic (vanilla fine-tuning)"
python meta/train_classifier.py --config $BASE_CONFIG --run-folds 5 \
    meta_learning.three_loyal_strategies.optimizer_strategy.name='none' \
    paths.out_dir=~/MRI/pretrain/code/meta/out/ablation/no_episodic \
    > $LOG_DIR/ablation_no_episodic_$(date +'%Y%m%d_%H%M%S').log 2>&1

echo ""
echo "============================================================"
echo "  方法论消融实验全部完成！"
echo "  6个变体 × 5折 = 30次训练"
echo "============================================================"
