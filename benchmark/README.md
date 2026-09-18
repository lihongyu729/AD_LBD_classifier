# AD vs LBD 多方法对比实验框架

## 概述

本框架用于系统对比不同深度学习方法在 **AD（阿尔茨海默病）vs LBD（路易体痴呆）** 3D MRI 二分类任务上的性能。

## 目录结构

```
benchmark/
├── configs/                 # 配置文件
│   ├── base_config.yaml     # 全局默认配置
│   └── methods/             # 每个方法的特定配置
├── core/                    # 核心模块
│   ├── base_method.py       # 方法抽象基类
│   ├── dataset.py           # 数据加载、增强、episodic采样
│   ├── metrics.py           # 统一指标计算
│   ├── trainer.py           # 双模式训练器
│   ├── evaluator.py         # CV评估器
│   └── config_loader.py     # 配置加载
├── methods/                 # 方法实现
│   ├── meta_strategies/     # 元学习策略 (ANIL/MAML/ProtoNet/Hybrid/Vanilla)
│   └── backbones/           # 骨干网络 (MedMambaSS3M/3DCNN/ResNet/CrossFormer)
├── scripts/                 # 运行脚本
│   ├── run_single.py        # 单次实验
│   ├── run_batch.py         # 批量运行
│   ├── check_data.py        # 数据检查
│   └── generate_report.py   # 报告生成
├── results/                 # 实验结果（自动生成）
└── reports/                 # 对比报告（自动生成）
```

## 快速开始

### 1. 配置数据路径

编辑 `configs/base_config.yaml`，设置你的数据路径：

```yaml
data:
  label_roots:
    AD: "D:/data/split_1mm_112/AD"
    LBD: "D:/data/split_1mm_112/LBD"
```

### 2. 检查数据

```bash
python scripts/check_data.py
```

### 3. 快速测试（2 fold, 5 epochs）

```bash
python scripts/run_single.py --method vanilla --seed 42 --folds 2 --epochs 5
```

### 4. 正式运行单个方法

```bash
python scripts/run_single.py --method anil --seed 42 --gpu 0
```

### 5. 批量运行所有方法

```bash
# 所有方法 × 3个种子, 使用GPU 0
python scripts/run_batch.py --all --seeds 42,123,456 --gpus 0

# 仅运行元学习方法
python scripts/run_batch.py --methods vanilla,anil,protonet,maml,hybrid --seeds 42,123,456 --gpus 0,1 --resume

# 使用多GPU并行
python scripts/run_batch.py --all --seeds 42,123 --gpus 0,1,2,3 --jobs 2
```

### 6. 生成对比报告

```bash
python scripts/generate_report.py --input ./results --output ./reports
```

## 包含的方法 (10个)

### 元学习策略（均使用 MedMambaSS3M 骨干）
| 方法 | 说明 |
|------|------|
| `vanilla` | 标准监督学习（基线） |
| `anil` | Almost No Inner Loop — 仅适配分类头 |
| `protonet` | Prototypical Networks — 原型度量学习 |
| `maml` | Model-Agnostic Meta-Learning — 全模型适配 |
| `hybrid` | ANIL/MAML + ProtoNet 组合 |

### 替代骨干网络（标准训练）
| 方法 | 来源 | 说明 |
|------|------|------|
| `cnn3d_baseline` | CNN_design_for_AD | 3D CNN (InstanceNorm) |
| `resnet3d` | DeepSPARE | 3D ResNet-18 (GroupNorm) |
| `crossformer3d` | MML-3DCrossFormer | 3D CrossFormer (Tiny) |
| `medmamba3d` | meta | MedMamba3D 简化版 |
| `medmamba_ss3m` | meta | MedMambaSS3M 参考基线 |

## 训练模式

- **标准模式**：`cnn3d_baseline`, `resnet3d`, `crossformer3d`, `medmamba3d`, `medmamba_ss3m`（以及 `vanilla`）
  - Batch → CE/Focal Loss → Backward
- **元学习模式**：`anil`, `protonet`, `maml`, `hybrid`
  - Episodic tasks → Strategy.core_step(support, query) → Outer loss → Backward

## 评估指标

- AUC-ROC（主要指标）
- Balanced Accuracy (BAC)
- Sensitivity / Specificity
- F1 Score
- Confusion Matrix

所有指标均通过 5-fold 分层交叉验证计算，输出 mean ± std。

## 添加新方法

1. 创建 `methods/backbones/your_backbone.py`，继承 `BaseMethod`
2. 实现 `build_model()`, `forward_encoder()`, `forward_classifier()`, `get_optimizer_param_groups()`
3. 调用 `register_method("your_method", YourMethodClass)` 注册
4. 创建 `configs/methods/your_method.yaml`
5. 运行：`python scripts/run_single.py --method your_method --seed 42`

## 依赖

```
torch >= 2.0
nibabel
numpy
scikit-learn
pyyaml
matplotlib
scipy (可选, 用于曲线平滑)
higher (可选, 用于 ANIL/MAML 元学习)
timm (可选, 用于 CrossFormer)
```

## 关键复用

本框架复用以下项目的代码：
- `D:\py_project\MRI\code\meta` — MedMambaSS3M 骨干 + 元学习策略
- `D:\桌面\deep fusion\code\CNN_design_for_AD` — 3D CNN 骨干
- `D:\桌面\deep fusion\code\DeepSPARE` — 3D ResNet 骨干
- `D:\桌面\deep fusion\code\MML-3DCrossFormer` — CrossFormer 参考
