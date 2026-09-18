"""
MedMamba3D wrapper — simpler Mamba variant (single branch, dwconv-based).

Source: D:\py_project\MRI\code\medmamba3d.py
"""
import sys
import os
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

_benchmark_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_parent_dir = os.path.dirname(_benchmark_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from core.base_method import BaseMethod


class MedMamba3DMethod(BaseMethod):
    """
    MedMamba3D standalone method.
    Simpler than SS3M: single branch, depthwise-conv-based Mamba.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def build_model(self) -> nn.Module:
        # Try to import MedMamba3D
        try:
            from medmamba3d import MedMamba3D
        except ImportError:
            print("[MedMamba3DMethod] Could not import MedMamba3D. "
                  "Using MedMambaSS3M with single branch as fallback.", flush=True)
            from medmamba_ss3m import MedMambaSS3M
            m_cfg = self.config.get("model", {})
            self.model = MedMambaSS3M(
                in_channels=1, num_classes=2,
                embed_dim=int(m_cfg.get("embed_dim", 384)),
                depth=int(m_cfg.get("depth", 6)),
                patch_size=tuple(m_cfg.get("patch_size", [16, 16, 16])),
                use_dual_branch=False,
                branch_types=("conv", "conv"),
                n_dirs_train=int(m_cfg.get("n_dirs_train", 4)),
            )
            return self.model

        m_cfg = self.config.get("model", {})
        self.model = MedMamba3D(
            in_chans=1, num_classes=2,
            embed_dim=int(m_cfg.get("embed_dim", 384)),
            depth=int(m_cfg.get("depth", 6)),
            patch_size=tuple(m_cfg.get("patch_size", [16, 16, 16])),
        )
        return self.model

    def forward_encoder(self, x):
        if hasattr(self.model, "forward_encoder"):
            tokens, _ = self.model.forward_encoder(x)
            if isinstance(tokens, tuple):
                tokens = tokens[0]
            if tokens.dim() == 4:
                tokens = tokens.mean(dim=1)
            return tokens
        return self.model.forward_encoder(x)

    def forward_classifier(self, x):
        if hasattr(self.model, "forward_classifier"):
            return self.model.forward_classifier(x)
        return self.model(x)

    def get_optimizer_param_groups(self) -> List[Dict]:
        backbone_lr_ratio = float(self.config.get("training", {}).get("backbone_lr_ratio", 0.05))
        head_params, backbone_params = [], []
        head_ids = set()
        head_mod = getattr(self.model, "head", getattr(self.model, "classifier", None))
        if head_mod is not None:
            head_ids = {id(p) for p in head_mod.parameters()}
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
register_method("medmamba3d", MedMamba3DMethod)
