#!/bin/bash
# ===========================================================================
# AD vs LBD —— 完整重跑实验矩阵（单 seed 42，嵌套5折，报告 test）
#
# 共享配置：configs/base_config.yaml + configs/protocol_rerun.yaml
#   （protocol 统一 epochs=160/调度/早停/增强/CV；lr 按方法保留）
#
# 用法:
#   bash run_all_exp.sh                      # 跑全部
#   DRY=1 bash run_all_exp.sh                # 只打印命令不执行
#   SKIP_GROUPS="IV,VII,VIII" bash run_all_exp.sh   # 跳过部分
#   单个 section: bash -c 'source <(sed -n "/^# SECTION: I/,/^# SECTION: II/p" run_all_exp.sh)'
#
# 泛化实验用 eval_external.py 单独跑（见文件末尾注释）。
# ===========================================================================
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-cnn}"
GPU="${GPU:-0}"
SEED="${SEED:-42}"
DRY="${DRY:-0}"
SKIP_GROUPS="${SKIP_GROUPS:-}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate "$CONDA_ENV" 2>/dev/null || true

run() {
  if [ "$DRY" = "1" ]; then
    echo "DRY> python scripts/run_single.py $*"
    return 0
  fi
  echo ">>> python scripts/run_single.py $*"
  python scripts/run_single.py "$@"
}

in_skip() { case ",${SKIP_GROUPS}," in *",$1,"*) return 0;; *) return 1;; esac; }

echo "=================================================="
echo " AD vs LBD rerun matrix | seed=$SEED gpu=$GPU"
echo " DRY=$DRY  SKIP_GROUPS=$SKIP_GROUPS"
echo "=================================================="

# ===========================================================================
# SECTION: I — 元学习方法对比 (Group A, 同一 MedMambaSS3M 骨干)
# ===========================================================================
if ! in_skip I; then
  for m in vanilla anil protonet maml hybrid; do
    run --method "$m" --seed "$SEED" --gpu "$GPU" --holdout-test 0.2 --note "meta_${m}"
  done
fi

# ===========================================================================
# SECTION: II — 融合策略 (anil, softmax 为基线默认)
# ===========================================================================
if ! in_skip II; then
  run --method anil --seed "$SEED" --gpu "$GPU" --holdout-test 0.2 --note "merge_avg" \
      --set ss3m.merge_type=avg
fi

# ===========================================================================
# SECTION: III — 方向数 × 元学习策略 交叉 (n_dirs=8 即各方法默认)
# ===========================================================================
if ! in_skip III; then
  for nd in 2 4; do
    for m in vanilla anil; do
      run --method "$m" --seed "$SEED" --gpu "$GPU" --holdout-test 0.2 \
          --note "n_dirs_${nd}" --set ss3m.n_dirs_train="$nd"
    done
  done
fi

# ===========================================================================
# SECTION: IV — embed_dim 选择 (medmamba_ss3m 标准训练；768 为默认基线)
#   注意: 768 维 MAE 预训练权重仅匹配 embed_dim=768，384/1152 为随机初始化
# ===========================================================================
if ! in_skip IV; then
  for dim in 384 1152; do
    run --method medmamba_ss3m --seed "$SEED" --gpu "$GPU" --holdout-test 0.2 \
        --note "embed_${dim}" --set model.embed_dim="$dim" --set ss3m.embed_dim="$dim"
  done
fi

# ===========================================================================
# SECTION: V — 别人方法对比 (Group B 骨干, 标准训练)
#   swin3d / crossformer3d 已剔除（当前实现不兼容 112³）
# ===========================================================================
if ! in_skip V; then
  for m in medmamba_ss3m resnet3d densenet3d cnn3d_baseline convnext3d medmamba3d; do
    run --method "$m" --seed "$SEED" --gpu "$GPU" --holdout-test 0.2 --note "bb_${m}"
  done
fi

# ===========================================================================
# SECTION: VI — 消融 (anil / MedMambaSS3M)
# ===========================================================================
if ! in_skip VI; then
  # 单双支路（双分支为默认基线）
  run --method anil --seed "$SEED" --gpu "$GPU" --holdout-test 0.2 --note "ab_conv_only" \
      --set ss3m.use_dual_branch=false --set ss3m.branch_types='["conv","conv"]'
  run --method anil --seed "$SEED" --gpu "$GPU" --holdout-test 0.2 --note "ab_mamba_only" \
      --set ss3m.use_dual_branch=false --set ss3m.branch_types='["mamba","mamba"]'
  # 有无 MAE 预训练
  run --method anil --seed "$SEED" --gpu "$GPU" --holdout-test 0.2 --note "ab_no_pretrain" \
      --set model.pretrained_path=""
fi

# ===========================================================================
# SECTION: VII — 均衡采样 (标准训练 medmamba_ss3m；standard 为默认基线)
# ===========================================================================
if ! in_skip VII; then
  run --method medmamba_ss3m --seed "$SEED" --gpu "$GPU" --holdout-test 0.2 \
      --note "bal_class_weight" --set training.class_weights='[1.0,8.0]'
  run --method medmamba_ss3m --seed "$SEED" --gpu "$GPU" --holdout-test 0.2 \
      --note "bal_resampling" --set training.resampling=true
fi

# ===========================================================================
# SECTION: VIII — 三个交叉实验 (2×2×2 on 标准训练 medmamba_ss3m)
#   支路(双/单mamba) × 预训练(有/无) × 均衡(std/class-weight) = 8 配置
# ===========================================================================
if ! in_skip VIII; then
  for branch in dual mamba; do
    for pt in on off; do
      for bal in std cw; do
        ARGS="--method medmamba_ss3m --seed $SEED --gpu $GPU --holdout-test 0.2 --note x_${branch}_${pt}_${bal}"
        if [ "$branch" = "mamba" ]; then
          ARGS="$ARGS --set ss3m.use_dual_branch=false --set ss3m.branch_types='[\"mamba\",\"mamba\"]'"
        fi
        if [ "$pt" = "off" ]; then
          ARGS="$ARGS --set model.pretrained_path=\"\""
        fi
        if [ "$bal" = "cw" ]; then
          ARGS="$ARGS --set training.class_weights='[1.0,8.0]'"
        fi
        # shellcheck disable=SC2086
        run $ARGS
      done
    done
  done
fi

echo ""
echo "=================================================="
echo " Matrix done. Generate report:"
echo "   python scripts/generate_report.py --input ./results --output ./reports"
echo "=================================================="
