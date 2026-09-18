"""
Vanilla Method: Standard supervised training with MedMambaSS3M backbone.

Uses VanillaStrategy from meta/strategies/ — standard CE + Focal loss,
no meta-learning inner loop.
"""
import sys
import os
from typing import Any, Dict, List, Optional

# Ensure meta strategies are importable
_meta_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "meta")
_meta_dir = os.path.abspath(_meta_dir)
if _meta_dir not in sys.path:
    sys.path.insert(0, _meta_dir)

from core.base_method import BaseMethod
from methods.backbones.medmamba_ss3m_wrapper import MedMambaSS3MWrapper


class VanillaMethod(MedMambaSS3MWrapper):
    """
    Vanilla (standard supervised) method.

    Inherits the backbone wrapper; adds strategy config for the trainer.
    The trainer uses standard mode when no strategy config is returned,
    OR uses VanillaStrategy for episodic compatible training.
    """

    def get_strategy_config(self) -> Optional[Dict]:
        meta_cfg = self.config.get("meta", {})
        strategy_cfg = meta_cfg.get("strategy", {})
        if strategy_cfg.get("name") == "vanilla":
            return strategy_cfg
        # Return vanilla strategy even for standard training to enable
        # episodic-compatible evaluation
        return {
            "name": "vanilla",
            "params": {
                "label_smoothing": float(self.config.get("training", {}).get("label_smoothing", 0.05)),
                "focal_gamma": 1.0,
                "focal_weight": 0.20,
                "use_class_weights": False,
                "use_support_in_loss": True,
            }
        }


# Register
from methods.method_registry import register_method
register_method("vanilla", VanillaMethod)
