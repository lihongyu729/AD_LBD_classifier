
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Tuple, Optional

class BaseMetaStrategy(ABC, nn.Module):
    """
    元学习策略基类。
    所有具体策略（如 MAML, ProtoNet 等）必须继承此类并实现 core_step 方法。
    """
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__()
        self.model = model
        self.config = config
        self.strategy_name = config.get("name", "unknown")
        
    @abstractmethod
    def core_step(self, 
                  support_x: torch.Tensor, 
                  support_y: torch.Tensor, 
                  query_x: torch.Tensor, 
                  query_y: torch.Tensor,
                  optimizer: Optional[torch.optim.Optimizer] = None,
                  inner_optimizer: Optional[torch.optim.Optimizer] = None,
                  is_training: bool = True) -> Dict[str, torch.Tensor]:
        """
        核心元学习步骤。
        Args:
            support_x: 支持集数据 [N_way * K_shot, C, D, H, W]
            support_y: 支持集标签
            query_x: 查询集数据 [N_way * Q_query, C, D, H, W]
            query_y: 查询集标签
            optimizer: 外层优化器 (Meta-Optimizer)
            inner_optimizer: 内层优化器 (Inner-Optimizer, 仅 MAML 类需要)
            is_training: 是否处于元训练阶段
            
        Returns:
            Dict: 包含 loss, acc 等指标的字典。必须包含 "loss" 键用于反向传播。
        """
        pass

    def forward(self, *args, **kwargs):
        return self.core_step(*args, **kwargs)
