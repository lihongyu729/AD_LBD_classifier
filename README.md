# MRI MedMamba3D/SS3M 训练与HPO说明

## 项目简介
本项目包含 3D MRI 的预训练（MAE）、对比学习和下游分类训练流程，并支持基于交叉验证的超参数优化（HPO）。分类阶段已与预训练阶段统一为单通道 3D 输入，确保输入规范一致。

## 环境配置
- Python 3.8+
- PyTorch + CUDA（如有 GPU）
- NIfTI 读写依赖：nibabel
- 其他依赖以项目环境为准

## 运行步骤
1) 配置参数  
   - 编辑 [config.yaml](file:///d:/py_project/MRI/code/config.yaml)  
   - 关键项：`paths.*`、`dataset.allowed_labels`、`classifier.backbone`、`hpo.*`

2) 预训练  
   - 运行 `train_pretrain_mae.py`

3) 分类训练  
   - 运行 `train_classifier.py`

4) 超参数优化（HPO）  
   - 保持 `hpo.enabled: true`  
   - 运行 `train_classifier.py`

## 参数说明
### 训练日志输出
- 训练日志由 `print_every` 控制频率  
- HPO 模式下会输出每个 trial 的结果与最终摘要  
- 如需更多过程日志：  
  - 提高 `classifier.print_every`（例如 50 或 100）  
  - 确保 `hpo.use_tensorboard: true`，日志会写入 `runs/exp_*`

### HPO 关键参数
- `hpo.algorithm`: 采样策略（three_stage/hyperband 等）  
- `hpo.trials`: 最大试验次数  
- `hpo.max_epochs`: 每个 trial 的最大 epoch  
- `hpo.early_stop_*`: 早停策略  
- `hpo.search_space`: 搜索空间  

## 输出结果解读
- `mean_bal_acc`（平均平衡准确率）  
  - 定义：各类别召回率的均值  
  - 用于类别不平衡场景  
- `mean_acc`（平均准确率）  
  - 定义：正确预测样本 / 总样本  
- `mean_auc`（平均 AUC）  
  - 定义：ROC 曲线下面积  
- `baseline mean_acc`  
  - 定义：基线超参数在交叉验证上的平均准确率  
- `improve_ratio`  
  - 定义：(best.mean_acc - baseline.mean_acc) / baseline.mean_acc  
  - 为负值表示最优结果不如基线  
- `target`  
  - 定义：期望的提升阈值（如 0.150）

## HPO 结果保存路径
- HPO 最佳参数文件：  
  - `~/MRI/classifer/out/runs/exp_20260301_152236_b73504f4/hpo_best.json`
- 其他输出：  
  - `hpo_results.csv`：每个 trial 的详细结果  
  - `report_*.html / report_*.pdf`：可视化报告  
  - `runs/exp_*`：TensorBoard 日志

## 常见问题排查
1) 训练过程无日志输出  
   - 检查 `classifier.print_every` 是否过大  
   - 检查 HPO 是否开启，HPO 只打印摘要  
   - 可开启 TensorBoard 输出以查看曲线

2) 权重加载缺失字段  
   - `head.*` 缺失属于正常现象  
   - `blocks.*.ssm_b.*` 缺失可能来自单分支预训练  
   - 可通过 `weight_diff_report.py` 生成缺失清单

3) JSON 序列化失败  
   - 已在 `_run_hpo` 中统一转换 numpy 标量  
   - 若仍报错，检查是否有新字段引入 numpy 类型

## 输出示例指标解读
- mean_bal_acc = 0.6051：类别召回均值  
- mean_acc = 0.3801：整体准确率  
- mean_auc = 0.7143：区分能力尚可  
- baseline mean_acc = 0.8719：基线远高于当前最优  
- improve_ratio = -0.564：相比基线下降 56.4%  
- target = 0.150：期望提升 15%，当前未达标
