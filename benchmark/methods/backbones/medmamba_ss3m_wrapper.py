"""
MedMambaSS3M backbone wrapper — integrates the existing MedMambaSS3M model
into the benchmark's BaseMethod interface.

Source: ~/MRI/code/medmamba_ss3m.py
"""
import sys
import os
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

# Ensure parent is importable
_benchmark_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_parent_dir = os.path.dirname(_benchmark_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from core.base_method import BaseMethod


class MedMambaSS3MWrapper(BaseMethod):
    """
    Wraps MedMambaSS3M for the benchmark framework.

    Config keys (under 'model'):
        embed_dim, depth, patch_size, branch_types, use_checkpoint,
        dropout, use_pos_emb, merge_type, n_dirs_train, use_dual_branch,
        pretrained_path, head_type, head_hidden_dim, head_dropout, head_activation.
    """

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self._backbone = None
        self._head = None

    def build_model(self) -> nn.Module:
        from medmamba_ss3m import MedMambaSS3M

        m_cfg = self.config.get("model", {})
        ss3m_cfg = self.config.get("ss3m", {})

        embed_dim = int(m_cfg.get("embed_dim", ss3m_cfg.get("embed_dim", 768)))
        depth = int(m_cfg.get("depth", ss3m_cfg.get("depth", 6)))
        patch_size = tuple(m_cfg.get("patch_size", ss3m_cfg.get("patch_size", [16, 16, 16])))
        branch_types = tuple(m_cfg.get("branch_types", ss3m_cfg.get("branch_types", ["conv", "mamba"])))
        use_checkpoint = bool(m_cfg.get("use_checkpoint", ss3m_cfg.get("use_checkpoint", True)))
        dropout = float(m_cfg.get("dropout", ss3m_cfg.get("dropout", 0.0)))
        use_pos_emb = bool(m_cfg.get("use_pos_emb", ss3m_cfg.get("use_pos_emb", True)))
        merge_type = str(m_cfg.get("merge_type", ss3m_cfg.get("merge_type", "softmax")))
        n_dirs_train = int(m_cfg.get("backbone_n_dirs_train", ss3m_cfg.get("n_dirs_train", 8)))
        use_dual_branch = bool(m_cfg.get("backbone_use_dual_branch", ss3m_cfg.get("use_dual_branch", True)))

        self.model = MedMambaSS3M(
            in_channels=1,
            num_classes=2,
            embed_dim=embed_dim,
            depth=depth,
            patch_size=patch_size,
            branch_types=branch_types,
            use_checkpoint=use_checkpoint,
            dropout=dropout,
            use_pos_emb=use_pos_emb,
            merge_type=merge_type,
            n_dirs_train=n_dirs_train,
            use_dual_branch=use_dual_branch,
        )
        return self.model

    def forward_encoder(self, x):
        return self.model.forward_encoder(x)

    def forward_classifier(self, x):
        return self.model.forward_classifier(x)

    def get_optimizer_param_groups(self) -> List[Dict]:
        """Differential LR: backbone gets lr * backbone_lr_ratio, head gets base lr."""
        backbone_lr_ratio = float(self.config.get("training", {}).get("backbone_lr_ratio", 0.05))

        head_params = []
        backbone_params = []

        head_module = getattr(self.model, "head", getattr(self.model, "classifier", None))
        if head_module is not None:
            head_ids = {id(p) for p in head_module.parameters()}
        else:
            head_ids = set()

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


# Register the method so build_method("medmamba_ss3m", cfg) works.
from methods.method_registry import register_method
register_method("medmamba_ss3m", MedMambaSS3MWrapper)
