"""
YAML configuration loader with deep-merge and CLI override support.

Usage:
    cfg = load_config("configs/base_config.yaml", "configs/methods/anil.yaml")
    cfg = load_config("configs/base_config.yaml", cli_overrides=["training.epochs=200", "seed=123"])
"""
import os
import sys
import copy
import platform
import yaml
from typing import Any, Dict, List, Optional


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override dict into base dict. Returns merged dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _set_nested_key(d: Dict[str, Any], key_path: str, value: Any):
    """Set a nested key in dict using dot-notation path, e.g. 'training.epochs'."""
    keys = key_path.split(".")
    for k in keys[:-1]:
        if k not in d or not isinstance(d[k], dict):
            d[k] = {}
        d = d[k]
    # Try to infer type from existing value
    existing = d.get(keys[-1])
    if existing is not None and not isinstance(value, type(existing)):
        try:
            if isinstance(existing, bool):
                value = value.lower() in ("true", "1", "yes")
            elif isinstance(existing, int):
                value = int(float(value))
            elif isinstance(existing, float):
                value = float(value)
            elif isinstance(existing, list):
                value = yaml.safe_load(value)
        except (ValueError, TypeError):
            pass
    d[keys[-1]] = value


def load_config(*config_paths: str, cli_overrides: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Load and merge multiple YAML config files in order.
    Later files override earlier ones. CLI overrides are applied last.

    Args:
        *config_paths: One or more YAML config file paths.
        cli_overrides: List of "key.path=value" strings from CLI.

    Returns:
        Merged configuration dictionary.
    """
    merged: Dict[str, Any] = {}

    for path in config_paths:
        if path and os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
            merged = _deep_merge(merged, cfg)

    # Apply CLI overrides (key=value format)
    if cli_overrides:
        for override in cli_overrides:
            if "=" in override:
                key, value = override.split("=", 1)
                _set_nested_key(merged, key.strip(), value.strip())

    return merged


def save_config(cfg: Dict[str, Any], path: str):
    """Save configuration dict to YAML file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


def resolve_data_paths(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Resolve and validate data paths in config.
    On Windows, auto-convert Linux-style paths.
    """
    data_cfg = cfg.get("data", {})
    label_roots = data_cfg.get("label_roots", {})

    resolved_roots = {}
    for label, root in label_roots.items():
        roots = root if isinstance(root, (list, tuple)) else [root]
        resolved = []
        for r in roots:
            r = os.path.expanduser(r)
            r = os.path.normpath(r)
            if os.path.isdir(r):
                resolved.append(r)
            else:
                # Auto-fallback: try swapping /home/... ↔ D:/... for cross-platform use
                alt = _cross_platform_fallback(r)
                if alt and os.path.isdir(alt):
                    print(f"[Config] Auto-switched data path: {r} → {alt}", flush=True)
                    resolved.append(alt)
                else:
                    print(f"[Config][warn] Data path not found: {r} (label={label})", flush=True)
        if resolved:
            resolved_roots[label] = resolved if len(resolved) > 1 else resolved[0]

    cfg = copy.deepcopy(cfg)
    cfg["data"]["label_roots"] = resolved_roots
    return cfg


def _cross_platform_fallback(path: str) -> Optional[str]:
    """Try swapping Linux /home/... path to Windows D:/... or vice versa."""
    import platform
    if platform.system() == "Windows":
        # ~/split_1mm_112/AD → D:/data/split_1mm_112/AD
        if path.startswith("/home/"):
            parts = path.split("/")
            # Find the meaningful suffix after username
            for i, p in enumerate(parts):
                if p in ("split_1mm_112", "MRI", "data"):
                    fallback = "D:/data/" + "/".join(parts[i:])
                    if os.path.isdir(fallback):
                        return fallback
            # Generic fallback
            fallback = "D:/data/" + "/".join(parts[3:])
            return fallback
    else:
        # D:/data/split_1mm_112/AD → ~/split_1mm_112/AD
        if path.startswith("D:/") or path.startswith("C:/"):
            drive, rest = path.split(":", 1)
            rest = rest.lstrip("/").lstrip("\\")
            # Try common server prefixes
            for prefix in ["~", "~/MRI"]:
                fallback = os.path.join(prefix, rest)
                if os.path.isdir(fallback):
                    return fallback
            fallback = "~/" + rest
            return fallback
    return None
