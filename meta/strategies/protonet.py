import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from .base import BaseMetaStrategy

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, target):
        logpt = F.log_softmax(logits, dim=1)
        pt = logpt.exp()
        logpt_t = logpt.gather(1, target.view(-1, 1)).squeeze(1)
        pt_t = pt.gather(1, target.view(-1, 1)).squeeze(1)
        
        # 【核心修复】：为 Focal Loss 的底数添加 epsilon，避免 gamma < 1 时 0.0 ** (gamma - 1) 导致 NaN 梯度
        eps = 1e-7
        p_diff = (1.0 - pt_t).clamp(min=eps)

        # 抛弃 class_weight，纯粹依靠 Gamma 压制易分样本
        loss = -(p_diff ** self.gamma) * logpt_t
        return loss.mean()

class ProtoNetStrategy(BaseMetaStrategy):
    """
    Prototypical Networks 实现。
    通过距离度量进行分类，无需内层梯度更新。
    """
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        self.params = config.get("params", {})
        self.metric = self.params.get("metric", "euclidean")
        self.proj_dim = self.params.get("proj_dim", 128)
        self.label_smoothing = float(self.params.get("label_smoothing", 0.0))
        self.gamma = float(self.params.get("gamma", 2.0))
        # 将温度系数作为类属性保存
        self.temperature = float(self.params.get("temperature", 10.0))
        
    def _get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, 'forward_encoder'):
            z, _ = self.model.forward_encoder(x)
        else:
            z = self.model(x)
        
        if z.dim() == 3:
            z = z.mean(dim=1)
        
        if hasattr(self.model, 'forward_head') and self.proj_dim > 0:
            z = self.model.forward_head(z)
        
        if self.metric == 'cosine':
            z = F.normalize(z, dim=1)
        return z

    def core_step(self, 
                  support_x: torch.Tensor, 
                  support_y: torch.Tensor, 
                  query_x: torch.Tensor, 
                  query_y: torch.Tensor,
                  optimizer: Optional[torch.optim.Optimizer] = None,
                  inner_optimizer: Optional[torch.optim.Optimizer] = None,
                  is_training: bool = True) -> Dict[str, torch.Tensor]:
        
        sup_emb = self._get_embedding(support_x)
        que_emb = self._get_embedding(query_x)
        
        # torch.unique 会自动排序，保证 Prototype 的顺序与类别大小顺序一致
        unique_labels = torch.unique(support_y)
        
        prototypes = []
        for c in unique_labels:
            mask = (support_y == c)
            proto_c = sup_emb[mask].mean(dim=0)
            prototypes.append(proto_c)
        prototypes = torch.stack(prototypes)
        
        # 【逻辑修复 1】：如果使用余弦相似度，求平均后的 Prototype 必须再次 L2 归一化
        if self.metric == 'cosine':
            prototypes = F.normalize(prototypes, dim=1)
        
        if que_emb.dim() > 2:
            que_emb = que_emb.view(que_emb.size(0), -1)
        if prototypes.dim() > 2:
            prototypes = prototypes.view(prototypes.size(0), -1)
            
        if que_emb.dim() == 2 and prototypes.dim() == 2:
            if self.metric == 'euclidean':
                distances = torch.cdist(que_emb, prototypes)
                logits = -distances
                # 【逻辑修复 2】：欧式距离需要缩小 (除以 temperature)
                scaled_logits = logits / self.temperature
            elif self.metric == 'cosine':
                logits = torch.mm(que_emb, prototypes.t())
                # 【逻辑修复 3】：余弦相似度需要放大 (乘以 temperature)
                scaled_logits = logits * self.temperature
            else:
                raise NotImplementedError(f"Metric {self.metric} not implemented")
        else:
             raise RuntimeError(f"Dimension mismatch in ProtoNet: que_emb {que_emb.shape}, prototypes {prototypes.shape}")
            
        # 映射局部标签 (Meta-Learning 标准做法)
        local_labels = torch.zeros_like(query_y)
        for i, c in enumerate(unique_labels):
            mask = (query_y == c)
            local_labels[mask] = i
            
        # 【语法修复 1】：使用定义好的 focal_criterion，并传入正确的 local_labels
        focal_criterion = FocalLoss(gamma=self.gamma)
        loss = focal_criterion(scaled_logits, local_labels)
        
        with torch.no_grad():
            preds = scaled_logits.argmax(dim=1)
            acc = (preds == local_labels).float().mean()
            
        return {
            "loss": loss,
            "acc": acc,
            "sup_loss": scaled_logits.new_tensor(0.0),
            # 【逻辑修复 4】：统一向外输出缩放后的 scaled_logits，保证外层计算 AUC 正常
            "logits": scaled_logits, 
            # 传出原始的 query_y 以便外层的 train_meta 可以计算全局指标
            "y_true": query_y 
        }
