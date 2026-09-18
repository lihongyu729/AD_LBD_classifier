"""
ConvNeXt 3D — Modern CNN with design elements from Transformers.

Adapted from ConvNeXt (Liu et al. 2022) for 3D volumetric MRI.
Key features over standard ResNet:
  - 7x7x7 Depthwise Conv3d (large kernel, similar to window attention)
  - Inverted bottleneck (wider middle, 4x expansion)
  - LayerNorm instead of BatchNorm
  - GELU instead of ReLU
  - Fewer normalization layers
  - Patchify stem (Conv3d k=4,s=4 instead of 7x7+MaxPool)

Configs:
  Tiny:   depths=[3,3,9,3],  dims=[96,192,384,768]
  Small:  depths=[3,3,27,3], dims=[96,192,384,768]
  Base:   depths=[3,3,27,3], dims=[128,256,512,1024]
"""
import sys, os
from typing import Any, Dict, List, Optional, Tuple
import torch, torch.nn as nn, torch.nn.functional as F, math

from core.base_method import BaseMethod


class LayerNorm3D(nn.Module):
    """LayerNorm over channel dimension for [B,C,D,H,W]."""
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight.view(1,-1,1,1,1) * x + self.bias.view(1,-1,1,1,1)


class ConvNeXtBlock3D(nn.Module):
    """
    ConvNeXt block:
        7x7x7 depthwise conv -> LayerNorm -> 1x1x1 (expand 4x) -> GELU -> 1x1x1 (shrink)
        + residual connection
    """
    def __init__(self, dim, drop_path=0., layer_scale_init=1e-6):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm3D(dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        self.drop_path = nn.Identity()

    def forward(self, x):
        # x: [B, C, D, H, W]
        skip = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma.view(1, -1, 1, 1, 1) * x
        x = skip + self.drop_path(x)
        return x


class ConvNeXt3DBackbone(nn.Module):
    """
    ConvNeXt 3D backbone with hierarchical stages.

    Args:
        in_chans: input channels (1 for MRI)
        num_classes: output classes (2 for AD/LBD)
        depths: number of blocks per stage [3,3,9,3] for Tiny
        dims: channel dimensions per stage [96,192,384,768] for Tiny
        drop_path_rate: stochastic depth rate
    """
    def __init__(self, in_chans=1, num_classes=2,
                 depths=(3,3,9,3), dims=(96,192,384,768),
                 drop_path_rate=0.1):
        super().__init__()

        # Stem — patchify with 4x4x4 stride
        self.stem = nn.Sequential(
            nn.Conv3d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm3D(dims[0]))

        # Stages
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        cur = 0
        for i in range(len(depths)):
            stage = nn.Sequential(*[
                ConvNeXtBlock3D(dims[i], drop_path=dp_rates[cur + j])
                for j in range(depths[i])])
            self.stages.append(stage)
            cur += depths[i]

            if i < len(depths) - 1:
                self.downsamples.append(nn.Sequential(
                    LayerNorm3D(dims[i]),
                    nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2)))

        # Head
        self.norm = LayerNorm3D(dims[-1])
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.classifier = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv3d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, D, H, W] → [B, dims[-1]]"""
        x = self.stem(x)
        for i, stage in enumerate(self.stages):
            x = stage(x)
            if i < len(self.downsamples):
                x = self.downsamples[i](x)
        x = self.norm(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)

    def forward_classifier(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.forward_encoder(x))
    def forward(self, x):
        return self.forward_classifier(x)


class ConvNeXt3DMethod(BaseMethod):
    VARIANTS = {
        "Tiny":   {"depths": (3,3,9,3),  "dims": (96,192,384,768)},
        "Small":  {"depths": (3,3,27,3), "dims": (96,192,384,768)},
        "Base":   {"depths": (3,3,27,3), "dims": (128,256,512,1024)},
    }

    def __init__(self, config): super().__init__(config)

    def build_model(self):
        c = self.config.get("model", {})
        variant = str(c.get("variant", "Tiny"))
        v = self.VARIANTS.get(variant, self.VARIANTS["Tiny"])
        self.model = ConvNeXt3DBackbone(
            in_chans=int(c.get("in_channels", 1)),
            num_classes=int(c.get("num_classes", 2)),
            depths=tuple(c.get("depths", v["depths"])),
            dims=tuple(c.get("dims", v["dims"])),
            drop_path_rate=float(c.get("drop_path_rate", 0.1)))
        n = sum(p.numel() for p in self.model.parameters())
        print(f"[ConvNeXt3D] variant={variant}, params={n/1e6:.2f}M", flush=True)
        return self.model

    def forward_encoder(self, x): return self.model.forward_encoder(x)
    def forward_classifier(self, x): return self.model.forward_classifier(x)

    def get_optimizer_param_groups(self):
        r = float(self.config.get("training", {}).get("backbone_lr_ratio", 0.1))
        hid = {id(p) for p in self.model.classifier.parameters()}
        h, bk = [], []
        for _, p in self.model.named_parameters():
            if not p.requires_grad: continue
            (h if id(p) in hid else bk).append(p)
        g = []
        if bk: g.append({"params": bk, "lr_ratio": r})
        if h:  g.append({"params": h, "lr_ratio": 1.0})
        return g

from methods.method_registry import register_method
register_method("convnext3d", ConvNeXt3DMethod)
