# -*- coding: utf-8 -*-
"""
Baseline wrapper for meta/train_classifier.py.

Goal: keep the same data pipeline, augmentation, model, and schedule as meta training,
but disable meta-learning strategies so only the algorithm changes.
"""

import train_classifier as meta_tc


def _apply_baseline_overrides(cfg: dict) -> dict:
    cls_cfg = cfg.setdefault("classifier", {})
    cls_cfg["strategy"] = "none"

    meta_cfg = cfg.setdefault("meta_learning", {}).setdefault("three_loyal_strategies", {})
    for key in ("metric_strategy", "hybrid_strategy"):
        if not isinstance(meta_cfg.get(key), dict):
            meta_cfg[key] = {}
        meta_cfg[key]["enable"] = False

    opt_cfg = meta_cfg.get("optimizer_strategy")
    if not isinstance(opt_cfg, dict):
        opt_cfg = {}
        meta_cfg["optimizer_strategy"] = opt_cfg
    opt_cfg["name"] = "none"

    return cfg


def _patch_load_config() -> None:
    original_load_config = meta_tc.load_config

    def _wrapped(default_path: str):
        cfg = original_load_config(default_path)
        return _apply_baseline_overrides(cfg)

    meta_tc.load_config = _wrapped


def main() -> None:
    _patch_load_config()
    meta_tc.train_meta()


if __name__ == "__main__":
    main()
