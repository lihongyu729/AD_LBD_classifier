#!/bin/bash
# run_all_meta.sh
# 此脚本用于一次性运行5种不同配置的5折交叉验证
# 限制：每批最多同时运行2个任务（通过后台任务 & 和 wait 命令配合实现）

export CUDA_VISIBLE_DEVICES=0,1  # 如果您有两张卡，可以在这里配置

CONFIG="meta/config.yaml"
LOG_DIR="meta/logs"
mkdir -p $LOG_DIR
# --- 在这里激活环境 ---
source ~/miniconda3/bin/activate cnn
# ---------------------------------------------------------
# 批量定义各种元学习的配置覆盖项 (键=值 格式)
# ---------------------------------------------------------
# 1. 无元学习 (No-Meta)
OPTS_NOMETA="meta_learning.three_loyal_strategies.optimizer_strategy.enable=False \
             meta_learning.three_loyal_strategies.metric_strategy.enable=False \
             meta_learning.three_loyal_strategies.hybrid_strategy.enable=False \
             paths.out_dir=~/MRI/pretrain/code/meta/out/nometa"

# 2. MAML
OPTS_MAML="meta_learning.three_loyal_strategies.optimizer_strategy.enable=True \
           meta_learning.three_loyal_strategies.optimizer_strategy.name='maml' \
           meta_learning.three_loyal_strategies.metric_strategy.enable=False \
           meta_learning.three_loyal_strategies.hybrid_strategy.enable=False \
           paths.out_dir=~/MRI/pretrain/code/meta/out/maml"

# 3. ANIL
OPTS_ANIL="meta_learning.three_loyal_strategies.optimizer_strategy.enable=True \
           meta_learning.three_loyal_strategies.optimizer_strategy.name='anil' \
           meta_learning.three_loyal_strategies.metric_strategy.enable=False \
           meta_learning.three_loyal_strategies.hybrid_strategy.enable=False \
           paths.out_dir=~/MRI/pretrain/code/meta/out/anil"

# 4. ProtoNet (Metric Learning)
OPTS_PROTO="meta_learning.three_loyal_strategies.optimizer_strategy.enable=False \
            meta_learning.three_loyal_strategies.metric_strategy.enable=True \
            meta_learning.three_loyal_strategies.metric_strategy.name='protonet' \
            meta_learning.three_loyal_strategies.hybrid_strategy.enable=False \
            paths.out_dir=~/MRI/pretrain/code/meta/out/proto"

# 5. Hybrid (ANIL + ProtoNet = MAML + Metric)
OPTS_HYBRID="meta_learning.three_loyal_strategies.optimizer_strategy.enable=True \
             meta_learning.three_loyal_strategies.optimizer_strategy.name='anil' \
             meta_learning.three_loyal_strategies.metric_strategy.enable=True \
             meta_learning.three_loyal_strategies.metric_strategy.name='protonet' \
             meta_learning.three_loyal_strategies.hybrid_strategy.enable=True \
             paths.out_dir=~/MRI/pretrain/code/meta/out/hybrid"

# -------- 执行函数 --------
function run_task() {
    local task_name=$1
    local opts=$2
    echo ">> [$(date +'%Y-%m-%d %H:%M:%S')] 开始运行任务: ${task_name}"
    
    # opts 参数不需要引号，会自动被 python 脚本的 argparse.REMAINDER 解析
    python meta/train_classifier.py --config $CONFIG --run-folds 5 $opts > $LOG_DIR/${task_name}_$(date +'%Y%m%d_%H%M%S').log 2>&1
    
    echo "<< [$(date +'%Y-%m-%d %H:%M:%S')] 任务结束: ${task_name}"
}

echo "=== 准备启动 5 种元学习版本的完整对比训练 ==="
echo "策略限制：服务器一次最多跑2个任务，我们将按批次提交运行。"

# --- 第1批 ---
run_task "nometa" "$OPTS_NOMETA" &
run_task "maml" "$OPTS_MAML" &
wait

# --- 第2批 ---
run_task "anil" "$OPTS_ANIL" &
run_task "proto" "$OPTS_PROTO" &
wait

# --- 第3批 ---
run_task "hybrid" "$OPTS_HYBRID" &
wait

echo "=== 所有版本 5 折验证运行完成！ ==="
