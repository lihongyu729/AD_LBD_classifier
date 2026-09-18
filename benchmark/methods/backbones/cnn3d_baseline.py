"""
3D CNN Baseline — adapted from D:\桌面\deep fusion\code\CNN_design_for_AD\models\models.py

Removes AgeEncoding, adds forward_encoder/forward_classifier separation.
Uses InstanceNorm3d for small-batch stability.
"""
import sys
import os
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import math

from core.base_method import BaseMethod


class CNN3DBackbone(nn.Module):
    """
    3D CNN backbone adapted from NetWork (CNN_design_for_AD).

    Architecture:
        Conv3d(1→4e, k1) → IN → ReLU → MaxPool(s2)
        Conv3d(4e→32e, k3,d2) → IN → ReLU → MaxPool(s2)
        Conv3d(32e→64e, k5,p2,d2) → IN → ReLU → MaxPool(s2)
        Conv3d(64e→64e, k3,p1,d2) → IN → ReLU → MaxPool(s2)
        AdaptiveAvgPool3d → FC(feat_dim) → FC(num_classes)
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        feat_dim: int = 1024,
        expansion: int = 4,
        norm_type: str = "Instance",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.feat_dim = feat_dim
        self.expansion = expansion

        NormLayer = nn.InstanceNorm3d if norm_type == "Instance" else nn.BatchNorm3d

        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, 4 * expansion, kernel_size=1),
            NormLayer(4 * expansion),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),  # 112→55

            nn.Conv3d(4 * expansion, 32 * expansion, kernel_size=3, padding=0, dilation=2),
            NormLayer(32 * expansion),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),  # 55→27

            nn.Conv3d(32 * expansion, 64 * expansion, kernel_size=5, padding=2, dilation=2),
            NormLayer(64 * expansion),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),  # 27→13

            nn.Conv3d(64 * expansion, 64 * expansion, kernel_size=3, padding=1, dilation=2),
            NormLayer(64 * expansion),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=5, stride=2),  # 13→5
        )

        self.pool = nn.AdaptiveAvgPool3d((5, 5, 5))
        self.fc_feat = nn.Linear(64 * expansion * 5 * 5 * 5, feat_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(feat_dim, num_classes)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features: [B, C, D, H, W] → [B, feat_dim]"""
        z = self.conv(x)
        z = self.pool(z)
        z = z.view(z.size(0), -1)
        z = self.fc_feat(z)
        z = self.dropout(z)
        return z

    def forward_classifier(self, x: torch.Tensor) -> torch.Tensor:
        """Classification: [B, C, D, H, W] → [B, num_classes]"""
        z = self.forward_encoder(x)
        return self.classifier(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_classifier(x)


class CNN3DBaselineMethod(BaseMethod):
    """
    3D CNN baseline method wrapping CNN3DBackbone.
    Standard supervised training (no meta-learning).
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def build_model(self) -> nn.Module:
        m_cfg = self.config.get("model", {})
        self.model = CNN3DBackbone(
            in_channels=int(m_cfg.get("in_channels", 1)),
            num_classes=int(m_cfg.get("num_classes", 2)),
            feat_dim=int(m_cfg.get("feat_dim", 1024)),
            expansion=int(m_cfg.get("expansion", 4)),
            norm_type=str(m_cfg.get("norm_type", "Instance")),
            dropout=float(m_cfg.get("dropout", 0.3)),
        )
        return self.model

    def forward_encoder(self, x):
        return self.model.forward_encoder(x)

    def forward_classifier(self, x):
        return self.model.forward_classifier(x)

    def get_optimizer_param_groups(self) -> List[Dict]:
        backbone_lr_ratio = float(self.config.get("training", {}).get("backbone_lr_ratio", 0.1))
        backbone_params = []
        head_params = []
        head_module = self.model.classifier
        head_ids = {id(p) for p in head_module.parameters()}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in head_ids:
                head_params.append(param)
            else:
                backbone_params.append(param)
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr_ratio": backbone_lr_ratio})
        if head_params:
            groups.append({"params": head_params, "lr_ratio": 1.0})
        return groups


# Register
from methods.method_registry import register_method
register_method("cnn3d_baseline", CNN3DBaselineMethod)
