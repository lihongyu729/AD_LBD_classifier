"""
Method registry: registration + factory pattern for all benchmark methods.

Usage:
    from methods.method_registry import register_method, build_method, list_methods
    method = build_method("vanilla", config)
"""
import sys
import os
from typing import Dict, Any, Type

# Ensure benchmark/ itself is on path so 'core' and 'methods' are top-level packages
_benchmark_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _benchmark_dir not in sys.path:
    sys.path.insert(0, _benchmark_dir)
# Also add code/ parent for importing medmamba_ss3m and meta project
_parent_dir = os.path.dirname(_benchmark_dir)
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

_registry: Dict[str, Type] = {}


def register_method(name: str, cls: Type):
    """Register a method class."""
    _registry[name] = cls


def build_method(name: str, config: Dict[str, Any]):
    """
    Build a method instance by name.

    Args:
        name: Method name (e.g. 'vanilla', 'anil', 'cnn3d_baseline').
        config: Full config dict.

    Returns:
        BaseMethod instance.
    """
    if name not in _registry:
        # Auto-import on first use
        _auto_import(name)
    if name not in _registry:
        raise ValueError(f"Unknown method: {name}. Available: {list(_registry.keys())}")
    return _registry[name](config)


def list_methods() -> list:
    """List all registered method names."""
    # Auto-discover
    _auto_import_all()
    return sorted(_registry.keys())


def _auto_import(name: str):
    """Lazy import a method module by name."""
    import importlib

    # Explicit mapping: method_name → (module_path,)
    META_STRATEGIES = ("vanilla", "anil", "protonet", "maml", "hybrid")
    BACKBONE_MODULES = {
        "cnn3d_baseline":   "methods.backbones.cnn3d_baseline",
        "convnext3d":       "methods.backbones.convnext3d",
        "densenet3d":       "methods.backbones.densenet3d",
        "resnet3d":         "methods.backbones.resnet3d",
        "medmamba3d":       "methods.backbones.medmamba3d_wrapper",
        "medmamba_ss3m":    "methods.backbones.medmamba_ss3m_wrapper",
    }

    try:
        if name in META_STRATEGIES:
            mod = importlib.import_module(f"methods.meta_strategies.{name}")
        elif name in BACKBONE_MODULES:
            mod = importlib.import_module(BACKBONE_MODULES[name])
        else:
            # Try both locations as fallback
            for loc in ["meta_strategies", "backbones"]:
                try:
                    mod = importlib.import_module(f"methods.{loc}.{name}")
                    break
                except ImportError:
                    continue
    except ImportError as e:
        print(f"[Registry] Could not import method '{name}': {e}", flush=True)


def _auto_import_all():
    """Try to import all known method modules."""
    known = [
        "vanilla", "anil", "protonet", "maml", "hybrid",
        "cnn3d_baseline", "convnext3d", "densenet3d",
        "resnet3d", "medmamba3d", "medmamba_ss3m",
    ]
    for name in known:
        if name not in _registry:
            try:
                _auto_import(name)
            except Exception:
                pass
