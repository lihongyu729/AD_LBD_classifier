"""
3D ResNet — adapted from D:\...\deep fusion\code\DeepSPARE\mytools\resnet3d.py

Key changes:
- BatchNorm3d → GroupNorm for small-batch stability (configurable).
- Adds forward_encoder / forward_classifier interface.
- Supports BasicBlock and Bottleneck.
- Configurable depth: ResNet10, ResNet18, ResNet34.
"""
import sys
import os
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from core.base_method import BaseMethod


# ---------------------------------------------------------------------------
# Blocks
# ---------------------------------------------------------------------------

def get_norm_layer(norm_type: str, num_features: int, num_groups: int = 8):
    """Get normalization layer based on config."""
    if norm_type == "Batch":
        return nn.BatchNorm3d(num_features)
    elif norm_type == "Group":
        groups = min(num_groups, num_features)
        if num_features % groups != 0:
            groups = 1
        return nn.GroupNorm(groups, num_features)
    else:
        return nn.InstanceNorm3d(num_features)


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, conv_kernel=3, stride=1,
                 downsample=None, norm_type="Group", num_groups=8):
        super().__init__()
        padding = {3: 1, 5: 2, 7: 3}.get(conv_kernel, 1)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=conv_kernel,
                               stride=stride, padding=padding, bias=False)
        self.norm1 = get_norm_layer(norm_type, out_channels, num_groups)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=conv_kernel,
                               stride=1, padding=padding, bias=False)
        self.norm2 = get_norm_layer(norm_type, out_channels, num_groups)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, conv_kernel=3, stride=1,
                 downsample=None, norm_type="Group", num_groups=8):
        super().__init__()
        padding = {3: 1, 5: 2, 7: 3}.get(conv_kernel, 1)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm1 = get_norm_layer(norm_type, out_channels, num_groups)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=conv_kernel,
                               stride=stride, padding=padding, bias=False)
        self.norm2 = get_norm_layer(norm_type, out_channels, num_groups)
        self.conv3 = nn.Conv3d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.norm3 = get_norm_layer(norm_type, out_channels * self.expansion, num_groups)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


# ---------------------------------------------------------------------------
# ResNet
# ---------------------------------------------------------------------------

class ResNet3DBackbone(nn.Module):
    """
    3D ResNet with configurable depth, block type, and normalization.

    Args:
        block: BasicBlock3D or Bottleneck3D.
        layers: List of block counts per stage (e.g., [2,2,2,2] for ResNet18).
        in_channels: Input channels (1 for grayscale MRI).
        num_classes: Number of output classes.
        conv_kernel: Kernel size for conv layers (3, 5, or 7).
        block_channels: List of output channels per stage.
        norm_type: "Batch", "Group", or "Instance".
        num_groups: Groups for GroupNorm.
        dropout: Dropout rate before classifier.
    """

    def __init__(
        self,
        block,
        layers: List[int],
        in_channels: int = 1,
        num_classes: int = 2,
        conv_kernel: int = 3,
        block_channels: List[int] = None,
        norm_type: str = "Group",
        num_groups: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        if block_channels is None:
            block_channels = [64, 128, 256, 512]

        self.in_planes = block_channels[0]
        self.norm_type = norm_type
        self.num_groups = num_groups

        # Stem
        self.conv1 = nn.Conv3d(in_channels, self.in_planes, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.norm1 = get_norm_layer(norm_type, self.in_planes, num_groups)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Stages
        self.layer1 = self._make_layer(block, block_channels[0], layers[0], conv_kernel, stride=1)
        self.layer2 = self._make_layer(block, block_channels[1], layers[1], conv_kernel, stride=2)
        self.layer3 = self._make_layer(block, block_channels[2], layers[2], conv_kernel, stride=2)
        self.layer4 = self._make_layer(block, block_channels[3], layers[3], conv_kernel, stride=2)

        final_channels = block_channels[3] * block.expansion
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(final_channels, num_classes)

        # Init
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, out_channels, num_blocks, conv_kernel, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.in_planes, out_channels * block.expansion,
                         kernel_size=1, stride=stride, bias=False),
                get_norm_layer(self.norm_type, out_channels * block.expansion, self.num_groups),
            )

        layers = [block(self.in_planes, out_channels, conv_kernel, stride, downsample,
                       self.norm_type, self.num_groups)]
        self.in_planes = out_channels * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, out_channels, conv_kernel, 1, None,
                              self.norm_type, self.num_groups))
        return nn.Sequential(*layers)

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features: [B, C, D, H, W] → [B, feat_dim]"""
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x.view(x.size(0), -1)

    def forward_classifier(self, x: torch.Tensor) -> torch.Tensor:
        """Classification: [B, C, D, H, W] → [B, num_classes]"""
        z = self.forward_encoder(x)
        z = self.dropout(z)
        return self.classifier(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_classifier(x)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def resnet10_3d(**kwargs) -> ResNet3DBackbone:
    return ResNet3DBackbone(BasicBlock3D, [1, 1, 1, 1], **kwargs)

def resnet18_3d(**kwargs) -> ResNet3DBackbone:
    return ResNet3DBackbone(BasicBlock3D, [2, 2, 2, 2], **kwargs)

def resnet34_3d(**kwargs) -> ResNet3DBackbone:
    return ResNet3DBackbone(BasicBlock3D, [3, 4, 6, 3], **kwargs)


# ---------------------------------------------------------------------------
# Benchmark Method wrapper
# ---------------------------------------------------------------------------

class ResNet3DMethod(BaseMethod):
    """3D ResNet method for the benchmark."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def build_model(self) -> nn.Module:
        m_cfg = self.config.get("model", {})
        depth = str(m_cfg.get("depth", "18"))
        factory = {
            "10": resnet10_3d, "18": resnet18_3d, "34": resnet34_3d,
        }
        build_fn = factory.get(depth, resnet18_3d)
        self.model = build_fn(
            in_channels=int(m_cfg.get("in_channels", 1)),
            num_classes=int(m_cfg.get("num_classes", 2)),
            conv_kernel=int(m_cfg.get("conv_kernel", 3)),
            norm_type=str(m_cfg.get("norm_type", "Group")),
            num_groups=int(m_cfg.get("num_groups", 8)),
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
register_method("resnet3d", ResNet3DMethod)
