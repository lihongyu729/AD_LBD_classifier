# AD vs LBD —— 重跑指南（.npy + 128³→112³ + 嵌套5折 + 报告test）

> 2026-08 版本。所有实验：**嵌套5折（20% holdout）、单 seed 42、报告 holdout-test 数字**。
> 数据已离线转成 112³ `.npy`（`Dataset_112_npy/{AD,LBD}`，AD 295 / LBD 38，脑区 Z-score、背景=0、文件名即受试者 ID）。

## 一、本地已完成的准备（无需重复）
1. `scripts/prepare_npy.py` 已把 128³ `.npy` → 112³（trilinear + mask 重零背景）→ `Dataset_112_npy/`。
2. 代码已支持 `.npy`（`data.normalized: true` 跳过重复 z-score）、`patient_ids` 输出、单层 fold 目录、ROC、训练侧阈值、holdout 优先的报告。

## 二、部署到服务器
```bash
# 1) 上传项目（含 112 npy 数据集与预训练权重）
scp -P 20954 -r D:\py_project\MRI\code\benchmark\  root@10.2.132.202:~/MRI/pretrain/code/
scp -P 20954 -r D:\py_project\MRI\code\meta\            root@10.2.132.202:~/MRI/pretrain/code/
scp -P 20954 D:\py_project\MRI\code\medmamba_ss3m.py    root@10.2.132.202:~/MRI/pretrain/code/
scp -P 20954 D:\py_project\MRI\code\medmamba3d.py      root@10.2.132.202:~/MRI/pretrain/code/
scp -P 20954 D:\py_project\MRI\code\weights\best_mae_8dir_768dim.pth root@10.2.132.202:~/MRI/pretrain/code/weights/
```

2) 检查 `configs/base_config.yaml`：
   - `data.label_roots` 指向服务器上的 `Dataset_112_npy/{AD,LBD}`（若路径不同则改）
   - `model.pretrained_path` 指向服务器上的 768 维权重（medmamba_ss3m 系方法必须有）
   - `dataset.cache_dir: ./cache_112_npy`（新缓存）

## 三、验证
```bash
conda activate cnn
cd ~/MRI/pretrain/code/benchmark
python scripts/check_data.py                     # 应输出 AD=295 LBD=38, 共 333
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 四、冒烟（可选，几分钟）
```bash
python scripts/run_single.py --method cnn3d_baseline --seed 42 --folds 2 --epochs 3 --holdout-test 0.2 --gpu 0
```
检查产物：`results/<method>/seed_42/<ts>/` 下应出现
`fold_0/{metrics.json,cm.png,predictions.csv,best_model.pth,roc_curve.png,roc_curve.csv}`（单层）
`holdout_test/{metrics.json,roc_curve.png,predictions.csv,split_info.json,retrain/}`
`roc_combined.png`、`cv_summary.json`（含 holdout_test + fold_thresholds）、`run_info.json`（含 param_count/elapsed）。

## 五、正式运行（全部实验 —— 完整矩阵 31 个 run）
```bash
screen -S rerun
conda activate cnn
cd ~/MRI/pretrain/code/benchmark
DRY=1 bash run_all_exp.sh    # 先预览 31 条命令
bash run_all_exp.sh          # 正式跑（单 seed 42，嵌套5折，报告 test）
# Ctrl+A D 断开；screen -r rerun 重连
```
矩阵包含（共享 protocol_rerun.yaml，统一 epochs=160/调度/早停/CV，lr 按方法保留）：
| 组 | 实验 | run 数 |
|---|---|---|
| I | 元学习方法对比（vanilla/anil/protonet/maml/hybrid） | 5 |
| II | 融合策略 avg（softmax 为基线） | 1 |
| III | 方向数 {2,4} × 策略 {vanilla,anil}（8 为默认） | 4 |
| IV | embed_dim {384,1152}（768 为基线；非 768 无对应预训练=随机初始化） | 2 |
| V | 骨干对比（medmamba_ss3m/resnet3d/densenet3d/cnn3d_baseline/convnext3d/medmamba3d）| 6 |
| VI | 消融：单conv/单mamba 支路、无预训练（anil） | 3 |
| VII | 均衡采样：class-weight、resampling（标准 medmamba_ss3m） | 2 |
| VIII | 三个交叉实验 2×2×2（支路双/单mamba × 预训练有/无 × 均衡std/class-weight） | 8 |

> `SKIP_GROUPS="IV,VII,VIII" bash run_all_exp.sh` 可跳过部分组。
> 注意：改了数据/代码后重跑，务必先 `rm -rf results/ reports/ cache_112_npy/`，否则旧结果会混入。

## 五bis、泛化实验（跨场强/跨诊断外部验证）
训练用上面的嵌套 CV（本数据集），泛化用 `eval_external.py` 在**外部数据**上评估：
```bash
# 外部数据必须是 112³ 归一化 .npy（先用 prepare_npy.py 转换）
python scripts/prepare_npy.py --source-root <外部数据根/images> --output-root <外部112根>

# 二元外部（如 AD/LBD @1.5T，跨场强）
python scripts/eval_external.py --run-dir results/anil/seed_42/<ts> \
    --data-root <外部112根> --map "AD=0,LBD=1" --out ./reports/gen_1p5t

# 单类外部（如 MCI —— 无 LBD/AD 标签，报告 P(LBD)/P(AD) 分布）
python scripts/eval_external.py --run-dir results/medmamba_ss3m/seed_42/<ts> \
    --data-root <外部112根>/MCI --map "MCI=-1" --out ./reports/gen_mci

# 哪个效果好报告哪个（对比各外部数据集的 summary.json）
```
> 外部数据：本机 E:\...\preprocessed\images 有 MCI(201)/NC(430)；3T/1.5T 场强需你提供（metadata 无场强列）。

## 六、消融（对 anil / MedMambaSS3M，嵌套5折）
```bash
python scripts/run_single.py --method anil --seed 42 --gpu 0 --holdout-test 0.2 \
    --note conv_only --set ss3m.use_dual_branch=false --set ss3m.branch_types='["conv","conv"]'
python scripts/run_single.py --method anil --seed 42 --gpu 0 --holdout-test 0.2 \
    --note mamba_only --set ss3m.use_dual_branch=false --set ss3m.branch_types='["mamba","mamba"]'
python scripts/run_single.py --method anil --seed 42 --gpu 0 --holdout-test 0.2 \
    --note no_pretrain --set model.pretrained_path=""
python scripts/run_single.py --method anil --seed 42 --gpu 0 --holdout-test 0.2 \
    --note avg_fusion --set ss3m.merge_type=avg
python scripts/run_single.py --method anil --seed 42 --gpu 0 --holdout-test 0.2 \
    --note n_dirs_4 --set ss3m.n_dirs_train=4
```

## 七、报告（holdout test 优先）
```bash
python scripts/generate_report.py --input ./results --output ./reports
# 输出：
#   reports/comparison_table.csv          每方法×每seed×每run 明细（含 holdout_* 列）
#   reports/comparison_aggregated.csv     按方法聚合（holdout AUC 排序）
#   reports/method_ranking.txt            TEST AUC 排名
#   reports/comparison_plot.png           test(主)+CV(次) AUC 柱状图
python scripts/run_batch.py --group all --seeds 42 --gpus 0 --dry-run   # 预览任务
```

## 八、论文可直接用的产物
- 每方法 holdout 测试指标（AUC/BAC/Sens/Spec/F1）：`cv_summary.json` 的 `holdout_test` + 汇总表
- ROC 曲线：`fold_0..N/roc_curve.png`、`roc_combined.png`、`holdout_test/roc_curve.png` + 对应 `.csv`（fpr/tpr/threshold，可重绘）
- 混淆矩阵：每折 + 合并 + holdout 的 `cm.png` / `cm_raw.json`
- 逐受试者预测：`predictions.csv`（真实 NACC ID + 阈值一致 y_pred）
- 数据拆分可复现：`holdout_test/split_info.json`（含 train_val/holdout 的索引与受试者 ID）
- 训练曲线：`metrics_fold_*.png`（loss/acc/AUC）、`train_log_all.csv`
- 模型权重：`fold_0..N/best_model.pth`、`holdout_test/retrain/best_model.pth`
- 配置/运行元数据：`config_snapshot.yaml`、`run_info.json`（param_count、耗时、test 阈值）
- 数据表素材：`E:\nifti\proprecessed\preprocessed\metadata\{labels,statistics}.csv`

## 九、局限（写论文注意）
- 单 seed 42 → holdout 只有 1 个测试数字，无误差棒/显著性检验（DeLong/McNemar 需要重复或多 seed）。
- 测试阈值 = 内层 CV 各折 val 阈值的均值（`mean_inner_cv_val`）；若折数<2 或阈值缺失则回退 0.5（`threshold_source` 已记录）。
