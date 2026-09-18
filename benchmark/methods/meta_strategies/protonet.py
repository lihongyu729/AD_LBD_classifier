"""
ProtoNet Method: Prototypical Networks for metric-based meta-learning.

ProtoNet computes per-class prototypes from support embeddings, then
classifies queries by distance to prototypes (Euclidean or Cosine).

Uses ProtoNetStrategy from meta/strategies/protonet.py.
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


class ProtoNetMethod(MedMambaSS3MWrapper):
    """ProtoNet metric-based meta-learning method."""

    def get_strategy_config(self) -> Optional[Dict]:
        meta_cfg = self.config.get("meta", {})
        strategy_cfg = meta_cfg.get("strategy", {})
        return {
            "name": "protonet",
            "params": {
                "metric": str(strategy_cfg.get("metric", "cosine")),
                "proj_dim": int(strategy_cfg.get("proj_dim", 32)),
                "gamma": float(strategy_cfg.get("gamma", 0.1)),
                "temperature": float(strategy_cfg.get("temperature", 12.0)),
                "label_smoothing": float(strategy_cfg.get("label_smoothing", 0.04)),
            }
        }


# Register
from methods.method_registry import register_method
register_method("protonet", ProtoNetMethod)
