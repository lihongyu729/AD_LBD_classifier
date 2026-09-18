"""
ANIL Method: Almost No Inner Loop meta-learning with MedMambaSS3M.

ANIL adapts only the classifier head in the inner loop, keeping the
feature extractor (backbone) frozen during inner-loop adaptation.

Uses ANILStrategy from meta/strategies/anil.py.
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


class ANILMethod(MedMambaSS3MWrapper):
    """ANIL meta-learning method."""

    def get_strategy_config(self) -> Optional[Dict]:
        meta_cfg = self.config.get("meta", {})
        strategy_cfg = meta_cfg.get("strategy", {})
        return {
            "name": "anil",
            "params": {
                "inner_lr": float(strategy_cfg.get("inner_lr", 0.01)),
                "inner_steps": int(strategy_cfg.get("inner_steps", 3)),
                "first_order": bool(strategy_cfg.get("first_order", True)),
                "dropout": float(strategy_cfg.get("dropout", 0.06)),
                "label_smoothing": float(strategy_cfg.get("label_smoothing", 0.04)),
                "inner_label_smoothing": float(strategy_cfg.get("inner_label_smoothing", 0.03)),
                "focal_weight": float(strategy_cfg.get("focal_weight", 0.20)),
                "focal_gamma": float(strategy_cfg.get("focal_gamma", 1.0)),
                "use_class_weights": bool(strategy_cfg.get("use_class_weights", False)),
                "inner_use_class_weights": bool(strategy_cfg.get("inner_use_class_weights", True)),
                "max_inner_lr": float(strategy_cfg.get("max_inner_lr", 0.1)),
                "supcon_weight": float(strategy_cfg.get("supcon_weight", 0.25)),
                "supcon_temperature": float(strategy_cfg.get("supcon_temperature", 0.06)),
                "inner_weight_decay": float(strategy_cfg.get("inner_weight_decay", 1e-4)),
                "use_compile": bool(strategy_cfg.get("use_compile", False)),
            }
        }


# Register
from methods.method_registry import register_method
register_method("anil", ANILMethod)
