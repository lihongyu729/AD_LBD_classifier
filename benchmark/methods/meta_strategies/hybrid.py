"""
Hybrid Method: Combined optimizer-based (ANIL/MAML) + metric-based (ProtoNet) meta-learning.

The hybrid strategy runs both branches and combines their losses:
    loss = opt_weight * loss_opt + metric_weight * loss_metric

Uses HybridStrategy from meta/strategies/hybrid.py.
"""
import sys
import os
from typing import Any, Dict, List, Optional

_meta_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "meta")
_meta_dir = os.path.abspath(_meta_dir)
if _meta_dir not in sys.path:
    sys.path.insert(0, _meta_dir)

from core.base_method import BaseMethod
from methods.backbones.medmamba_ss3m_wrapper import MedMambaSS3MWrapper


class HybridMethod(MedMambaSS3MWrapper):
    """Hybrid meta-learning method combining optimizer-based and metric-based strategies."""

    def get_strategy_config(self) -> Optional[Dict]:
        meta_cfg = self.config.get("meta", {})
        strategy_cfg = meta_cfg.get("strategy", {})
        hybrid_cfg = self.config.get("meta_learning", {}).get("three_loyal_strategies", {}).get("hybrid_strategy", {})
        params = hybrid_cfg.get("params", strategy_cfg)

        return {
            "name": "hybrid",
            "params": {
                "opt_weight": float(params.get("opt_weight", 0.5)),
                "metric_weight": float(params.get("metric_weight", 0.5)),
                "scheduler": str(params.get("scheduler", "adaptive")),
            },
            "optimizer_strategy": {
                "name": "maml",
                "params": {
                    "inner_lr": 0.01, "inner_steps": 3, "first_order": True,
                    "focal_weight": 0.20, "focal_gamma": 1.0,
                    "label_smoothing": 0.04, "inner_label_smoothing": 0.03,
                }
            },
            "metric_strategy": {
                "name": "protonet",
                "params": {
                    "metric": "cosine", "proj_dim": 32,
                    "temperature": 12.0, "label_smoothing": 0.04,
                }
            },
        }


# Register
from methods.method_registry import register_method
register_method("hybrid", HybridMethod)
