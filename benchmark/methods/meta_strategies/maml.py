"""
MAML Method: Model-Agnostic Meta-Learning with full inner-loop adaptation.

MAML adapts the entire model (backbone + head) in the inner loop.
Supports first-order (FOMAML) approximation for efficiency.

Uses MAMLStrategy from meta/strategies/maml.py.
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


class MAMLMethod(MedMambaSS3MWrapper):
    """MAML full-model meta-learning method."""

    def get_strategy_config(self) -> Optional[Dict]:
        meta_cfg = self.config.get("meta", {})
        strategy_cfg = meta_cfg.get("strategy", {})
        optimizer_cfg = self.config.get("meta_learning", {}).get("three_loyal_strategies", {}).get("optimizer_strategy", {})
        params = optimizer_cfg.get("params", strategy_cfg)

        return {
            "name": "maml",
            "params": {
                "inner_lr": float(params.get("inner_lr", 0.01)),
                "inner_steps": int(params.get("inner_steps", 3)),
                "first_order": bool(params.get("first_order", True)),
                "dropout": float(params.get("dropout", 0.06)),
                "label_smoothing": float(params.get("label_smoothing", 0.04)),
                "inner_label_smoothing": float(params.get("inner_label_smoothing", 0.03)),
                "focal_weight": float(params.get("focal_weight", 0.20)),
                "focal_gamma": float(params.get("focal_gamma", 1.0)),
                "use_class_weights": bool(params.get("use_class_weights", False)),
                "max_inner_lr": float(params.get("max_inner_lr", 0.1)),
            }
        }


# Register
from methods.method_registry import register_method
register_method("maml", MAMLMethod)
