"""
3D DenseNet backbone for 112³ MRI classification.

Implements a simplified 3D DenseNet using dense blocks with
concatenation-based feature reuse. Uses GroupNorm for small-batch stability.

If monai is available, monai.networks.nets.DenseNet is preferred.
This is a standalone implementation that works without monai.
"""
import sys
import os
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import math

from core.base_method import BaseMethod


# ---------------------------------------------------------------------------
# DenseNet3D components
# ---------------------------------------------------------------------------

class DenseLayer3D(nn.Module):
    """Single dense layer: BN → ReLU → Conv3d(1×1×1, bottleneck) → BN → ReLU → Conv3d(3×3×3, growth_rate)."""

    def __init__(self, in_channels: int, growth_rate: int, bn_size: int = 4, dropout: float = 0.0):
        super().__init__()
        inter_channels = bn_size * growth_rate

        self.norm1 = nn.GroupNorm(min(8, in_channels), in_channels) if in_channels >= 8 else nn.InstanceNorm3d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels, inter_channels, kernel_size=1, bias=False)

        self.norm2 = nn.GroupNorm(min(8, inter_channels), inter_channels) if inter_channels >= 8 else nn.InstanceNorm3d(inter_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(inter_channels, growth_rate, kernel_size=3, padding=1, bias=False)

        self.dropout = nn.Dropout3d(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        out = self.conv1(self.relu1(self.norm1(x)))
        out = self.conv2(self.relu2(self.norm2(out)))
        out = self.dropout(out)
        return torch.cat([x, out], dim=1)


class DenseBlock3D(nn.Module):
    """Stack of DenseLayer3D modules."""

    def __init__(self, num_layers: int, in_channels: int, growth_rate: int, bn_size: int = 4, dropout: float = 0.0):
        super().__init__()
        self.layers = nn.ModuleList()
        current_channels = in_channels
        for i in range(num_layers):
            self.layers.append(DenseLayer3D(current_channels, growth_rate, bn_size, dropout))
            current_channels += growth_rate

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class Transition3D(nn.Module):
    """Transition layer: BN → ReLU → Conv3d(1×1) → AvgPool3d(2×2×2)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(min(8, in_channels), in_channels) if in_channels >= 8 else nn.InstanceNorm3d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(self.relu(self.norm(x)))
        return self.pool(x)


class DenseNet3DBackbone(nn.Module):
    """
    3D DenseNet for volumetric MRI.

    Args:
        in_channels: Input channels (1 for grayscale MRI).
        num_classes: Output classes (2 for AD/LBD).
        init_channels: Channels after stem conv.
        growth_rate: Channels added per dense layer (k).
        block_config: Tuple of layer counts per dense block.
        bn_size: Bottleneck multiplier.
        dropout: Dropout rate.
        compression: Transition compression factor (0.5 = halve channels).
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        init_channels: int = 32,
        growth_rate: int = 16,
        block_config: tuple = (6, 12, 12, 6),
        bn_size: int = 4,
        dropout: float = 0.1,
        compression: float = 0.5,
    ):
        super().__init__()

        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, init_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(min(8, init_channels), init_channels) if init_channels >= 8 else nn.InstanceNorm3d(init_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        # Dense blocks + transitions
        num_features = init_channels
        self.blocks = nn.ModuleList()
        self.transitions = nn.ModuleList()

        for i, num_layers in enumerate(block_config):
            block = DenseBlock3D(num_layers, num_features, growth_rate, bn_size, dropout)
            self.blocks.append(block)
            num_features += num_layers * growth_rate

            if i < len(block_config) - 1:
                out_features = int(num_features * compression)
                transition = Transition3D(num_features, out_features)
                self.transitions.append(transition)
                num_features = out_features

        # Final norm + pooling
        self.final_norm = nn.GroupNorm(min(8, num_features), num_features) if num_features >= 8 else nn.InstanceNorm3d(num_features)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_features, num_classes)

        # Init weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, D, H, W] → [B, num_features]"""
        x = self.stem(x)
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i < len(self.transitions):
                x = self.transitions[i](x)
        x = self.relu(self.final_norm(x))
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

    def forward_classifier(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, D, H, W] → [B, num_classes]"""
        z = self.forward_encoder(x)
        z = self.dropout(z)
        return self.classifier(z)

    def forward(self, x):
        return self.forward_classifier(x)


# ---------------------------------------------------------------------------
# Factory for paper's DenseNet-121 variant
# ---------------------------------------------------------------------------

def densenet121_3d(**kwargs) -> DenseNet3DBackbone:
    return DenseNet3DBackbone(
        init_channels=64, growth_rate=32, block_config=(6, 12, 24, 16), **kwargs
    )


# ---------------------------------------------------------------------------
# Benchmark Method wrapper
# ---------------------------------------------------------------------------

class DenseNet3DMethod(BaseMethod):
    """3D DenseNet method for the benchmark."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def build_model(self) -> nn.Module:
        m_cfg = self.config.get("model", {})
        variant = str(m_cfg.get("variant", "custom"))

        if variant == "121":
            self.model = densenet121_3d(
                in_channels=int(m_cfg.get("in_channels", 1)),
                num_classes=int(m_cfg.get("num_classes", 2)),
                dropout=float(m_cfg.get("dropout", 0.1)),
            )
        else:
            self.model = DenseNet3DBackbone(
                in_channels=int(m_cfg.get("in_channels", 1)),
                num_classes=int(m_cfg.get("num_classes", 2)),
                init_channels=int(m_cfg.get("init_channels", 32)),
                growth_rate=int(m_cfg.get("growth_rate", 16)),
                block_config=tuple(m_cfg.get("block_config", (6, 12, 12, 6))),
                dropout=float(m_cfg.get("dropout", 0.1)),
                compression=float(m_cfg.get("compression", 0.5)),
            )

        # Calculate params
        total = sum(p.numel() for p in self.model.parameters())
        print(f"[DenseNet3D] variant={variant}, params={total/1e6:.2f}M", flush=True)
        return self.model

    def forward_encoder(self, x):
        return self.model.forward_encoder(x)

    def forward_classifier(self, x):
        return self.model.forward_classifier(x)

    def get_optimizer_param_groups(self) -> List[Dict]:
        backbone_lr_ratio = float(self.config.get("training", {}).get("backbone_lr_ratio", 0.1))
        head_params, backbone_params = [], []
        head_ids = {id(p) for p in self.model.classifier.parameters()}
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
register_method("densenet3d", DenseNet3DMethod)
