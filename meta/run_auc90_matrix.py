import argparse
import copy
import csv
import json
import math
import os
import re
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import yaml


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_yaml(path: str, data) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def _set_nested(cfg: dict, dotted_key: str, value) -> None:
    keys = dotted_key.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    cur[keys[-1]] = value


def _apply_overrides(base_cfg: dict, overrides: dict) -> dict:
    cfg = copy.deepcopy(base_cfg)
    for key, value in overrides.items():
        _set_nested(cfg, key, value)
    return cfg


def _matrix_definitions():
    # High-yield first batch for AUC uplift on imbalanced AD/LBD.
    return [
        {
            "name": "m00_baseline_ref",
            "overrides": {},
            "notes": "Reference from current config.",
        },
        {
            "name": "m01_focal50_gamma20",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.5,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.label_smoothing": 0.10,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_label_smoothing": 0.10,
            },
            "notes": "Enable focal + stronger smoothing.",
        },
        {
            "name": "m02_focal50_gamma25",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.5,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.5,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.label_smoothing": 0.10,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_label_smoothing": 0.10,
            },
            "notes": "Harder minority focus than m01.",
        },
        {
            "name": "m03_focal75_gamma20",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.75,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.label_smoothing": 0.10,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_label_smoothing": 0.10,
            },
            "notes": "Aggressive focal branch.",
        },
        {
            "name": "m04_focal50_cov90_k4q4",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.5,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 0.90,
                "task.max_train_episodes": 120,
                "task.min_train_episodes": 24,
                "task.max_minority_reuse_per_epoch": 4.0,
                "task.val_episodes_meta": 100,
            },
            "notes": "Coverage-oriented with stable task size.",
        },
        {
            "name": "m05_focal50_cov90_k5q5",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.5,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "task.k_shot": 5,
                "task.q_query": 5,
                "task.min_epoch_coverage_ratio": 0.90,
                "task.max_train_episodes": 120,
                "task.min_train_episodes": 24,
                "task.max_minority_reuse_per_epoch": 4.0,
                "task.val_episodes_meta": 100,
            },
            "notes": "Larger episode tasks; may increase variance.",
        },
        {
            "name": "m06_focal50_cov90_val150",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.5,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 0.90,
                "task.max_train_episodes": 140,
                "task.min_train_episodes": 24,
                "task.max_minority_reuse_per_epoch": 4.0,
                "task.val_episodes_meta": 150,
            },
            "notes": "Higher validation episodes for lower noise.",
        },
    ]


def _matrix_definitions_v2():
    # Second-batch matrix: expand from loss/coverage-only to optimizer, LR ratio,
    # and augmentation strength while keeping runtime manageable.
    return [
        {
            "name": "v2_00_baseline_ref",
            "overrides": {},
            "notes": "Reference from current config.",
        },
        {
            "name": "v2_01_focal50_innerlr3e3",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.5,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.003,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 3,
                "classifier.backbone_lr_ratio": 0.06,
            },
            "notes": "Lower inner LR + focal to reduce oscillation.",
        },
        {
            "name": "v2_02_focal50_innersteps5",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.5,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.004,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 5,
                "classifier.backbone_lr_ratio": 0.05,
            },
            "notes": "More adaptation steps, slightly lower backbone LR ratio.",
        },
        {
            "name": "v2_03_cov90_k4q4_val120",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.5,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 0.90,
                "task.max_train_episodes": 140,
                "task.min_train_episodes": 24,
                "task.max_minority_reuse_per_epoch": 4.0,
                "task.val_episodes_meta": 120,
            },
            "notes": "Coverage-focused stable episode setting.",
        },
        {
            "name": "v2_04_cov90_k5q5_val120",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.5,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "task.k_shot": 5,
                "task.q_query": 5,
                "task.min_epoch_coverage_ratio": 0.90,
                "task.max_train_episodes": 120,
                "task.min_train_episodes": 24,
                "task.max_minority_reuse_per_epoch": 4.0,
                "task.val_episodes_meta": 120,
            },
            "notes": "Larger episode tasks to test representation limits.",
        },
        {
            "name": "v2_05_aug_mid",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.5,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "augment.noise_std": 0.05,
                "augment.gamma_jitter": 0.15,
                "augment.cutout_p": 0.30,
                "augment.cutout_frac": 0.15,
            },
            "notes": "Mid-strength augmentation for minority generalization.",
        },
        {
            "name": "v2_06_aug_high",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.5,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "augment.noise_std": 0.06,
                "augment.gamma_jitter": 0.18,
                "augment.cutout_p": 0.40,
                "augment.cutout_frac": 0.15,
            },
            "notes": "High-strength augmentation for robustness stress test.",
        },
        {
            "name": "v2_07_combo_balanced",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.5,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.5,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.003,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 5,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.label_smoothing": 0.12,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_label_smoothing": 0.12,
                "classifier.backbone_lr_ratio": 0.04,
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 0.90,
                "task.max_train_episodes": 140,
                "task.min_train_episodes": 24,
                "task.val_episodes_meta": 120,
                "augment.noise_std": 0.05,
                "augment.gamma_jitter": 0.15,
                "augment.cutout_p": 0.30,
                "augment.cutout_frac": 0.15,
            },
            "notes": "Balanced combo expected to improve mean and worst-fold AUC.",
        },
        {
            "name": "v2_08_combo_conservative",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.35,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.004,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 3,
                "classifier.backbone_lr_ratio": 0.03,
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 0.90,
                "task.max_train_episodes": 120,
                "task.min_train_episodes": 24,
                "task.val_episodes_meta": 100,
                "augment.noise_std": 0.04,
                "augment.gamma_jitter": 0.12,
                "augment.cutout_p": 0.20,
                "augment.cutout_frac": 0.12,
            },
            "notes": "Conservative combo for stability-first runs.",
        },
        {
            "name": "v2_09_combo_aggressive",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.75,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.5,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.0025,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 5,
                "classifier.backbone_lr_ratio": 0.02,
                "task.k_shot": 5,
                "task.q_query": 5,
                "task.min_epoch_coverage_ratio": 0.90,
                "task.max_train_episodes": 140,
                "task.min_train_episodes": 24,
                "task.val_episodes_meta": 150,
                "augment.noise_std": 0.06,
                "augment.gamma_jitter": 0.18,
                "augment.cutout_p": 0.40,
                "augment.cutout_frac": 0.15,
            },
            "notes": "Aggressive exploration for higher ceiling.",
        },
    ]


def _matrix_definitions_v3():
    # Third-batch matrix: stability-first search space after v2 underperformed.
    # Goal: improve raw_auc mean and worst-fold by reducing high-variance combinations.
    return [
        {
            "name": "v3_00_baseline_ref",
            "overrides": {},
            "notes": "Reference from current config.",
        },
        {
            "name": "v3_01_core_stable",
            "overrides": {
                "imbalance.mode": "class_weight_focal",
                "classifier.use_class_weights": True,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.40,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.003,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 4,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.label_smoothing": 0.08,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_label_smoothing": 0.08,
                "classifier.backbone_lr_ratio": 0.03,
                "early_stopping.monitor": "val_raw_auc",
                "thresholding.allow_tuned_early_stop": False,
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 1.00,
                "task.max_train_episodes": 140,
                "task.min_train_episodes": 24,
                "task.max_minority_reuse_per_epoch": 3.0,
                "task.val_episodes_meta": 120,
                "augment.noise_std": 0.03,
                "augment.gamma_jitter": 0.10,
                "augment.cutout_p": 0.15,
                "augment.cutout_frac": 0.10,
            },
            "notes": "Core stable anchor with low-variance settings.",
        },
        {
            "name": "v3_02_core_cov120",
            "overrides": {
                "imbalance.mode": "class_weight_focal",
                "classifier.use_class_weights": True,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.40,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.003,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 4,
                "classifier.backbone_lr_ratio": 0.03,
                "early_stopping.monitor": "val_raw_auc",
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 1.20,
                "task.max_train_episodes": 160,
                "task.min_train_episodes": 30,
                "task.max_minority_reuse_per_epoch": 3.2,
                "task.val_episodes_meta": 140,
                "augment.noise_std": 0.03,
                "augment.gamma_jitter": 0.10,
                "augment.cutout_p": 0.15,
                "augment.cutout_frac": 0.10,
            },
            "notes": "Increase coverage while keeping augmentation conservative.",
        },
        {
            "name": "v3_03_core_k3q3",
            "overrides": {
                "imbalance.mode": "class_weight_focal",
                "classifier.use_class_weights": True,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.40,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.003,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 4,
                "classifier.backbone_lr_ratio": 0.03,
                "early_stopping.monitor": "val_raw_auc",
                "task.k_shot": 3,
                "task.q_query": 3,
                "task.min_epoch_coverage_ratio": 1.10,
                "task.max_train_episodes": 180,
                "task.min_train_episodes": 36,
                "task.max_minority_reuse_per_epoch": 3.0,
                "task.val_episodes_meta": 150,
                "augment.noise_std": 0.03,
                "augment.gamma_jitter": 0.10,
                "augment.cutout_p": 0.10,
                "augment.cutout_frac": 0.08,
            },
            "notes": "Smaller episode size to reduce minority exhaustion.",
        },
        {
            "name": "v3_04_focal55_gamma18",
            "overrides": {
                "imbalance.mode": "class_weight_focal",
                "classifier.use_class_weights": True,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.55,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 1.8,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.003,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 4,
                "classifier.backbone_lr_ratio": 0.03,
                "early_stopping.monitor": "val_raw_auc",
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 1.00,
                "task.max_train_episodes": 140,
                "task.min_train_episodes": 24,
                "task.val_episodes_meta": 120,
                "augment.noise_std": 0.03,
                "augment.gamma_jitter": 0.10,
                "augment.cutout_p": 0.15,
                "augment.cutout_frac": 0.10,
            },
            "notes": "Higher focal weight with softer gamma for recall/precision balance.",
        },
        {
            "name": "v3_05_focal30_gamma22",
            "overrides": {
                "imbalance.mode": "class_weight_focal",
                "classifier.use_class_weights": True,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.30,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.2,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.003,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 4,
                "classifier.backbone_lr_ratio": 0.03,
                "early_stopping.monitor": "val_raw_auc",
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 1.00,
                "task.max_train_episodes": 140,
                "task.min_train_episodes": 24,
                "task.val_episodes_meta": 120,
                "augment.noise_std": 0.03,
                "augment.gamma_jitter": 0.10,
                "augment.cutout_p": 0.15,
                "augment.cutout_frac": 0.10,
            },
            "notes": "Lower focal branch to avoid over-focusing hard minority samples.",
        },
        {
            "name": "v3_06_innerlr25e3_steps5",
            "overrides": {
                "imbalance.mode": "class_weight_focal",
                "classifier.use_class_weights": True,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.40,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.0025,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 5,
                "classifier.backbone_lr_ratio": 0.03,
                "early_stopping.monitor": "val_raw_auc",
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 1.00,
                "task.max_train_episodes": 140,
                "task.min_train_episodes": 24,
                "task.val_episodes_meta": 120,
                "augment.noise_std": 0.03,
                "augment.gamma_jitter": 0.10,
                "augment.cutout_p": 0.15,
                "augment.cutout_frac": 0.10,
            },
            "notes": "More adaptation steps with lower inner LR.",
        },
        {
            "name": "v3_07_backbone_ratio02",
            "overrides": {
                "imbalance.mode": "class_weight_focal",
                "classifier.use_class_weights": True,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.40,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.003,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 4,
                "classifier.backbone_lr_ratio": 0.02,
                "early_stopping.monitor": "val_raw_auc",
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 1.00,
                "task.max_train_episodes": 140,
                "task.min_train_episodes": 24,
                "task.val_episodes_meta": 120,
                "augment.noise_std": 0.03,
                "augment.gamma_jitter": 0.10,
                "augment.cutout_p": 0.15,
                "augment.cutout_frac": 0.10,
            },
            "notes": "More conservative backbone updates for transfer stability.",
        },
        {
            "name": "v3_08_wd3e3",
            "overrides": {
                "imbalance.mode": "class_weight_focal",
                "classifier.use_class_weights": True,
                "classifier.weight_decay": 0.003,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.40,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.003,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 4,
                "classifier.backbone_lr_ratio": 0.03,
                "early_stopping.monitor": "val_raw_auc",
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 1.00,
                "task.max_train_episodes": 140,
                "task.min_train_episodes": 24,
                "task.val_episodes_meta": 120,
                "augment.noise_std": 0.03,
                "augment.gamma_jitter": 0.10,
                "augment.cutout_p": 0.15,
                "augment.cutout_frac": 0.10,
            },
            "notes": "Slightly stronger regularization for noisy folds.",
        },
        {
            "name": "v3_09_patience35",
            "overrides": {
                "imbalance.mode": "class_weight_focal",
                "classifier.use_class_weights": True,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.40,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.003,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 4,
                "classifier.backbone_lr_ratio": 0.03,
                "early_stopping.monitor": "val_raw_auc",
                "early_stopping.patience": 35,
                "early_stopping.min_delta": 0.0005,
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 1.00,
                "task.max_train_episodes": 140,
                "task.min_train_episodes": 24,
                "task.val_episodes_meta": 120,
                "augment.noise_std": 0.03,
                "augment.gamma_jitter": 0.10,
                "augment.cutout_p": 0.15,
                "augment.cutout_frac": 0.10,
            },
            "notes": "Allow slower but steadier improvement before early stop.",
        },
        {
            "name": "v3_10_seed43",
            "overrides": {
                "imbalance.mode": "class_weight_focal",
                "classifier.use_class_weights": True,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.40,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.003,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 4,
                "classifier.backbone_lr_ratio": 0.03,
                "early_stopping.monitor": "val_raw_auc",
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 1.00,
                "task.max_train_episodes": 140,
                "task.min_train_episodes": 24,
                "task.val_episodes_meta": 120,
                "augment.noise_std": 0.03,
                "augment.gamma_jitter": 0.10,
                "augment.cutout_p": 0.15,
                "augment.cutout_frac": 0.10,
                "cv.cv_random_state": 43,
            },
            "notes": "Seed repeat for robustness check.",
        },
        {
            "name": "v3_11_seed44",
            "overrides": {
                "imbalance.mode": "class_weight_focal",
                "classifier.use_class_weights": True,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.40,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.003,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 4,
                "classifier.backbone_lr_ratio": 0.03,
                "early_stopping.monitor": "val_raw_auc",
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 1.00,
                "task.max_train_episodes": 140,
                "task.min_train_episodes": 24,
                "task.val_episodes_meta": 120,
                "augment.noise_std": 0.03,
                "augment.gamma_jitter": 0.10,
                "augment.cutout_p": 0.15,
                "augment.cutout_frac": 0.10,
                "cv.cv_random_state": 44,
            },
            "notes": "Seed repeat for robustness check.",
        },
    ]


def _matrix_definitions_v4():
    # Fourth-batch matrix: fix two practical issues observed in logs.
    # 1) Coverage ceiling caused by low max_minority_reuse_per_epoch.
    # 2) No-op optimizer-parameter tuning when optimizer strategy is disabled.
    return [
        {
            "name": "v4_00_protonet_ref",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.enable": False,
                "meta_learning.three_loyal_strategies.metric_strategy.enable": True,
                "meta_learning.three_loyal_strategies.hybrid_strategy.enable": False,
                "early_stopping.monitor": "val_raw_auc",
            },
            "notes": "ProtoNet reference with explicit strategy flags.",
        },
        {
            "name": "v4_01_protonet_temp12_proj32",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.enable": False,
                "meta_learning.three_loyal_strategies.metric_strategy.enable": True,
                "meta_learning.three_loyal_strategies.hybrid_strategy.enable": False,
                "meta_learning.three_loyal_strategies.metric_strategy.params.metric": "cosine",
                "meta_learning.three_loyal_strategies.metric_strategy.params.temperature": 12.0,
                "meta_learning.three_loyal_strategies.metric_strategy.params.proj_dim": 32,
                "meta_learning.three_loyal_strategies.metric_strategy.params.label_smoothing": 0.06,
                "early_stopping.monitor": "val_raw_auc",
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 0.95,
                "task.max_minority_reuse_per_epoch": 5.5,
                "task.max_train_episodes": 96,
                "task.min_train_episodes": 40,
                "task.val_episodes_meta": 120,
                "task.allow_unsafe_episode_override": True,
            },
            "notes": "ProtoNet with effective metric-space tuning and achievable >=0.9 coverage.",
        },
        {
            "name": "v4_02_protonet_temp10_proj48_cov100",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.enable": False,
                "meta_learning.three_loyal_strategies.metric_strategy.enable": True,
                "meta_learning.three_loyal_strategies.hybrid_strategy.enable": False,
                "meta_learning.three_loyal_strategies.metric_strategy.params.metric": "cosine",
                "meta_learning.three_loyal_strategies.metric_strategy.params.temperature": 10.0,
                "meta_learning.three_loyal_strategies.metric_strategy.params.proj_dim": 48,
                "meta_learning.three_loyal_strategies.metric_strategy.params.label_smoothing": 0.08,
                "early_stopping.monitor": "val_raw_auc",
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 1.00,
                "task.max_minority_reuse_per_epoch": 6.0,
                "task.max_train_episodes": 104,
                "task.min_train_episodes": 48,
                "task.val_episodes_meta": 140,
                "task.allow_unsafe_episode_override": True,
            },
            "notes": "ProtoNet high-coverage variant with lower temperature.",
        },
        {
            "name": "v4_03_anil_ref",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.enable": True,
                "meta_learning.three_loyal_strategies.metric_strategy.enable": False,
                "meta_learning.three_loyal_strategies.hybrid_strategy.enable": False,
                "meta_learning.three_loyal_strategies.optimizer_strategy.name": "anil",
                "imbalance.mode": "class_weight_focal",
                "classifier.use_class_weights": True,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.40,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.003,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 4,
                "classifier.backbone_lr_ratio": 0.03,
                "early_stopping.monitor": "val_raw_auc",
            },
            "notes": "ANIL reference with actually effective optimizer-side tuning.",
        },
        {
            "name": "v4_04_anil_cov95",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.enable": True,
                "meta_learning.three_loyal_strategies.metric_strategy.enable": False,
                "meta_learning.three_loyal_strategies.hybrid_strategy.enable": False,
                "meta_learning.three_loyal_strategies.optimizer_strategy.name": "anil",
                "imbalance.mode": "class_weight_focal",
                "classifier.use_class_weights": True,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.45,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.003,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 4,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.label_smoothing": 0.08,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_label_smoothing": 0.08,
                "classifier.backbone_lr_ratio": 0.03,
                "early_stopping.monitor": "val_raw_auc",
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 0.95,
                "task.max_minority_reuse_per_epoch": 5.5,
                "task.max_train_episodes": 96,
                "task.min_train_episodes": 40,
                "task.val_episodes_meta": 120,
                "task.allow_unsafe_episode_override": True,
            },
            "notes": "ANIL with feasible high coverage and moderate focal.",
        },
        {
            "name": "v4_05_anil_cov105_steps5",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.enable": True,
                "meta_learning.three_loyal_strategies.metric_strategy.enable": False,
                "meta_learning.three_loyal_strategies.hybrid_strategy.enable": False,
                "meta_learning.three_loyal_strategies.optimizer_strategy.name": "anil",
                "imbalance.mode": "class_weight_focal",
                "classifier.use_class_weights": True,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.40,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.2,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.0025,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 5,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.label_smoothing": 0.08,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_label_smoothing": 0.08,
                "classifier.backbone_lr_ratio": 0.025,
                "early_stopping.monitor": "val_raw_auc",
                "early_stopping.patience": 35,
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 1.05,
                "task.max_minority_reuse_per_epoch": 6.2,
                "task.max_train_episodes": 112,
                "task.min_train_episodes": 48,
                "task.val_episodes_meta": 140,
                "task.allow_unsafe_episode_override": True,
            },
            "notes": "ANIL higher coverage with cautious LR and more inner adaptation.",
        },
        {
            "name": "v4_06_anil_seed43",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.enable": True,
                "meta_learning.three_loyal_strategies.metric_strategy.enable": False,
                "meta_learning.three_loyal_strategies.hybrid_strategy.enable": False,
                "meta_learning.three_loyal_strategies.optimizer_strategy.name": "anil",
                "imbalance.mode": "class_weight_focal",
                "classifier.use_class_weights": True,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.45,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.003,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 4,
                "classifier.backbone_lr_ratio": 0.03,
                "early_stopping.monitor": "val_raw_auc",
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 0.95,
                "task.max_minority_reuse_per_epoch": 5.5,
                "task.max_train_episodes": 96,
                "task.min_train_episodes": 40,
                "task.val_episodes_meta": 120,
                "task.allow_unsafe_episode_override": True,
                "cv.cv_random_state": 43,
            },
            "notes": "ANIL seed repeat to verify robustness.",
        },
        {
            "name": "v4_07_anil_seed44",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.enable": True,
                "meta_learning.three_loyal_strategies.metric_strategy.enable": False,
                "meta_learning.three_loyal_strategies.hybrid_strategy.enable": False,
                "meta_learning.three_loyal_strategies.optimizer_strategy.name": "anil",
                "imbalance.mode": "class_weight_focal",
                "classifier.use_class_weights": True,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.45,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.003,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 4,
                "classifier.backbone_lr_ratio": 0.03,
                "early_stopping.monitor": "val_raw_auc",
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 0.95,
                "task.max_minority_reuse_per_epoch": 5.5,
                "task.max_train_episodes": 96,
                "task.min_train_episodes": 40,
                "task.val_episodes_meta": 120,
                "task.allow_unsafe_episode_override": True,
                "cv.cv_random_state": 44,
            },
            "notes": "ANIL seed repeat to verify robustness.",
        },
    ]


def _matrix_definitions_v5():
    # Fifth-batch matrix: keep best v4 ANIL anchors and explicitly test embed_dim=786.
    # Includes both partial-checkpoint transfer and no-pretrain controls.
    return [
        {
            "name": "v5_00_anil_cov95_ref",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.enable": True,
                "meta_learning.three_loyal_strategies.metric_strategy.enable": False,
                "meta_learning.three_loyal_strategies.hybrid_strategy.enable": False,
                "meta_learning.three_loyal_strategies.optimizer_strategy.name": "anil",
                "imbalance.mode": "class_weight_focal",
                "classifier.use_class_weights": True,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.45,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.003,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 4,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.label_smoothing": 0.08,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_label_smoothing": 0.08,
                "classifier.backbone_lr_ratio": 0.03,
                "early_stopping.monitor": "val_raw_auc",
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 0.95,
                "task.max_minority_reuse_per_epoch": 5.5,
                "task.max_train_episodes": 96,
                "task.min_train_episodes": 40,
                "task.val_episodes_meta": 120,
                "task.allow_unsafe_episode_override": True,
            },
            "notes": "Reference anchor from v4_04 for fair comparison.",
        },
        {
            "name": "v5_01_anil_cov105_steps5_ref",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.enable": True,
                "meta_learning.three_loyal_strategies.metric_strategy.enable": False,
                "meta_learning.three_loyal_strategies.hybrid_strategy.enable": False,
                "meta_learning.three_loyal_strategies.optimizer_strategy.name": "anil",
                "imbalance.mode": "class_weight_focal",
                "classifier.use_class_weights": True,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.40,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.2,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.0025,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 5,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.label_smoothing": 0.08,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_label_smoothing": 0.08,
                "classifier.backbone_lr_ratio": 0.025,
                "early_stopping.monitor": "val_raw_auc",
                "early_stopping.patience": 35,
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 1.05,
                "task.max_minority_reuse_per_epoch": 6.2,
                "task.max_train_episodes": 112,
                "task.min_train_episodes": 48,
                "task.val_episodes_meta": 140,
                "task.allow_unsafe_episode_override": True,
            },
            "notes": "Reference anchor from v4_05 for fair comparison.",
        },
        {
            "name": "v5_02_anil_ed786_partialckpt_cov95",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.enable": True,
                "meta_learning.three_loyal_strategies.metric_strategy.enable": False,
                "meta_learning.three_loyal_strategies.hybrid_strategy.enable": False,
                "meta_learning.three_loyal_strategies.optimizer_strategy.name": "anil",
                "imbalance.mode": "class_weight_focal",
                "classifier.use_class_weights": True,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.45,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.003,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 4,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.label_smoothing": 0.08,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_label_smoothing": 0.08,
                "ss3m.embed_dim": 786,
                "model.embed_dim": 786,
                "classifier.backbone_lr_ratio": 0.02,
                "classifier.pretrained_failfast_min_ratio": 0.0,
                "classifier.pretrained_warn_ratio": 0.05,
                "early_stopping.monitor": "val_raw_auc",
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 0.95,
                "task.max_minority_reuse_per_epoch": 5.5,
                "task.max_train_episodes": 96,
                "task.min_train_episodes": 40,
                "task.val_episodes_meta": 120,
                "task.allow_unsafe_episode_override": True,
            },
            "notes": "embed_dim=786 with partial checkpoint loading allowed.",
        },
        {
            "name": "v5_03_anil_ed786_partialckpt_cov105_steps5",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.enable": True,
                "meta_learning.three_loyal_strategies.metric_strategy.enable": False,
                "meta_learning.three_loyal_strategies.hybrid_strategy.enable": False,
                "meta_learning.three_loyal_strategies.optimizer_strategy.name": "anil",
                "imbalance.mode": "class_weight_focal",
                "classifier.use_class_weights": True,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.40,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.2,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.0025,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 5,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.label_smoothing": 0.08,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_label_smoothing": 0.08,
                "ss3m.embed_dim": 786,
                "model.embed_dim": 786,
                "classifier.backbone_lr_ratio": 0.02,
                "classifier.pretrained_failfast_min_ratio": 0.0,
                "classifier.pretrained_warn_ratio": 0.05,
                "early_stopping.monitor": "val_raw_auc",
                "early_stopping.patience": 35,
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 1.05,
                "task.max_minority_reuse_per_epoch": 6.2,
                "task.max_train_episodes": 112,
                "task.min_train_episodes": 48,
                "task.val_episodes_meta": 140,
                "task.allow_unsafe_episode_override": True,
            },
            "notes": "embed_dim=786 with partial checkpoint loading and higher coverage.",
        },
        {
            "name": "v5_04_anil_ed786_nopretrain_cov95",
            "overrides": {
                "meta_learning.three_loyal_strategies.optimizer_strategy.enable": True,
                "meta_learning.three_loyal_strategies.metric_strategy.enable": False,
                "meta_learning.three_loyal_strategies.hybrid_strategy.enable": False,
                "meta_learning.three_loyal_strategies.optimizer_strategy.name": "anil",
                "imbalance.mode": "class_weight_focal",
                "classifier.use_class_weights": True,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_weight": 0.45,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.focal_gamma": 2.0,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_lr": 0.003,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_steps": 4,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.label_smoothing": 0.08,
                "meta_learning.three_loyal_strategies.optimizer_strategy.params.inner_label_smoothing": 0.08,
                "ss3m.embed_dim": 786,
                "model.embed_dim": 786,
                "classifier.pretrained_path": "",
                "classifier.pretrained_failfast_min_ratio": 0.0,
                "classifier.pretrained_warn_ratio": 0.0,
                "classifier.backbone_lr_ratio": 0.02,
                "early_stopping.monitor": "val_raw_auc",
                "task.k_shot": 4,
                "task.q_query": 4,
                "task.min_epoch_coverage_ratio": 0.95,
                "task.max_minority_reuse_per_epoch": 5.5,
                "task.max_train_episodes": 96,
                "task.min_train_episodes": 40,
                "task.val_episodes_meta": 120,
                "task.allow_unsafe_episode_override": True,
            },
            "notes": "embed_dim=786 no-pretrain control to verify transfer contribution.",
        },
    ]


def _warn_noop_overrides(name: str, cfg: dict, overrides: dict) -> None:
    # Warn when optimizer params are tuned but optimizer strategy is disabled.
    if not isinstance(overrides, dict):
        return
    has_opt_param_override = any(
        k.startswith("meta_learning.three_loyal_strategies.optimizer_strategy.params.")
        for k in overrides.keys()
    )
    if not has_opt_param_override:
        return

    three = cfg.get("meta_learning", {}).get("three_loyal_strategies", {})
    opt_on = bool(three.get("optimizer_strategy", {}).get("enable", False))
    hybrid_on = bool(three.get("hybrid_strategy", {}).get("enable", False))
    if (not opt_on) and (not hybrid_on):
        _log(
            f"[Warn][NoOp] {name}: optimizer_strategy.params were overridden but optimizer/hybrid strategy is disabled; "
            "these overrides are likely ineffective under current strategy.")


def _extract_float(line: str, key: str):
    m = re.search(rf"{re.escape(key)}=([0-9]+(?:\.[0-9]+)?|nan)", line)
    if not m:
        return float("nan")
    token = m.group(1)
    if token == "nan":
        return float("nan")
    return float(token)


def _parse_fold_best_lines(log_path: str):
    fold_raw_auc = {}
    fold_auc = {}
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "best_epoch=" in line and "[CV " in line:
                m = re.search(r"\[CV\s+(\d+)\]", line)
                if not m:
                    continue
                fold = int(m.group(1))
                raw_auc = _extract_float(line, "raw_auc")
                auc = _extract_float(line, "auc")
                fold_raw_auc[fold] = raw_auc
                fold_auc[fold] = auc
                continue

            if not line.startswith("[TEST "):
                continue

            m = re.search(r"\[TEST\s+(\d+)\]", line)
            if not m:
                continue
            fold = int(m.group(1))
            raw_auc = _extract_float(line, "test_auc")
            auc = raw_auc
            fold_raw_auc[fold] = raw_auc
            fold_auc[fold] = auc
    return fold_raw_auc, fold_auc


def _stats(values):
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "worst": float("nan"), "n": 0}
    mean_v = float(statistics.mean(vals))
    std_v = float(statistics.pstdev(vals)) if len(vals) > 1 else 0.0
    return {"mean": mean_v, "std": std_v, "worst": float(min(vals)), "n": len(vals)}


def _run_one(
    train_script: str,
    cfg_path: str,
    folds: int,
    run_folds: int,
    epochs: int,
    out_log: str,
    split_mode: str = "nested_cv",
    train_ratio: float | None = None,
    val_ratio: float | None = None,
    test_ratio: float | None = None,
    env_overrides: dict | None = None,
    stream_stdout: bool = True,
) -> int:
    cmd = [sys.executable, "-u", train_script, "--config", cfg_path]
    cmd.extend(["--split-mode", "nested_cv", "--folds", "5", "--run-folds", "5", "--outer-folds", "5"])
    if epochs > 0:
        cmd.extend(["--epochs", str(epochs)])

    os.makedirs(os.path.dirname(out_log), exist_ok=True)
    _log("[Run] " + " ".join(cmd))

    env = os.environ.copy()
    if env_overrides:
        env.update(env_overrides)

    with open(out_log, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(train_script),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
            env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            if stream_stdout:
                print(line, end="")
            lf.write(line)
        proc.wait()
        return proc.returncode


def _run_and_collect(
    idx: int,
    row: dict,
    run_log: str,
    train_script: str,
    folds: int,
    run_folds: int,
    epochs: int,
    resume: bool,
    split_mode: str,
    train_ratio: float | None,
    val_ratio: float | None,
    test_ratio: float | None,
    env_overrides: dict | None,
    stream_stdout: bool,
) -> dict:
    name = row["name"]
    if resume and os.path.exists(run_log):
        _log(f"[Skip] Existing log found for {name}: {run_log}")
        code = 0
    else:
        code = _run_one(
            train_script,
            row["config"],
            folds,
            run_folds,
            epochs,
            run_log,
            split_mode,
            train_ratio,
            val_ratio,
            test_ratio,
            env_overrides=env_overrides,
            stream_stdout=stream_stdout,
        )

    raw_auc_by_fold, auc_by_fold = _parse_fold_best_lines(run_log)
    raw_auc_stats = _stats(list(raw_auc_by_fold.values()))
    auc_stats = _stats(list(auc_by_fold.values()))

    result = {
        "idx": idx,
        "name": name,
        "return_code": code,
        "log": run_log,
        "raw_auc_mean": raw_auc_stats["mean"],
        "raw_auc_std": raw_auc_stats["std"],
        "raw_auc_worst": raw_auc_stats["worst"],
        "raw_auc_n": raw_auc_stats["n"],
        "auc_mean": auc_stats["mean"],
        "auc_std": auc_stats["std"],
        "auc_worst": auc_stats["worst"],
        "auc_n": auc_stats["n"],
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Generate and optionally run AUC>=0.90 experiment matrix.")
    parser.add_argument("--base-config", type=str, default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--train-script", type=str, default=os.path.join(os.path.dirname(__file__), "train_classifier.py"))
    parser.add_argument("--out-dir", type=str, default=os.path.join(os.path.dirname(__file__), "out", "auc90_matrix"))
    parser.add_argument("--matrix-version", type=str, default="v5", choices=["v1", "v2", "v3", "v4", "v5"], help="Choose matrix set")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--run-folds", type=int, default=5, help="Fixed to 5 in nested_cv mode")
    parser.add_argument("--epochs", type=int, default=0, help="0 means keep config value")
    parser.add_argument("--split-mode", type=str, default="nested_cv", choices=["nested_cv"])
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--locked-test-ratio", type=float, default=0.2, help="Locked test ratio for nested_cv")
    parser.add_argument("--patient-id-mode", type=str, default="parent_dir", help="Patient ID mode for nested_cv")
    parser.add_argument("--run", action="store_true", help="Run generated configs")
    parser.add_argument("--max-runs", type=int, default=0, help="0 means all generated configs")
    parser.add_argument("--only", type=str, default="", help="Comma-separated names to run")
    parser.add_argument("--resume", action="store_true", help="Skip run if log file already exists")
    parser.add_argument("--jobs", type=int, default=1, help="Number of configs to run concurrently")
    parser.add_argument("--gpu-ids", type=str, default="", help="Comma-separated GPU IDs for worker slots, e.g. 0,1")
    args = parser.parse_args()
    if int(args.folds) != 5 or int(args.run_folds) != 5:
        raise RuntimeError("This project is locked to nested_cv 5-fold only. Please use --folds 5 --run-folds 5.")

    base_cfg = _load_yaml(args.base_config)
    if args.matrix_version == "v5":
        matrix = _matrix_definitions_v5()
    elif args.matrix_version == "v4":
        matrix = _matrix_definitions_v4()
    elif args.matrix_version == "v3":
        matrix = _matrix_definitions_v3()
    elif args.matrix_version == "v2":
        matrix = _matrix_definitions_v2()
    else:
        matrix = _matrix_definitions()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    exp_root = os.path.join(args.out_dir, f"{args.matrix_version}_{ts}")
    cfg_dir = os.path.join(exp_root, "configs")
    run_dir = os.path.join(exp_root, "runs")
    os.makedirs(cfg_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)

    selected = None
    if args.only.strip():
        selected = {x.strip() for x in args.only.split(",") if x.strip()}

    registry = []
    for item in matrix:
        name = item["name"]
        if selected is not None and name not in selected:
            continue
        cfg = _apply_overrides(base_cfg, item["overrides"])
        cfg.setdefault("cv", {})["split_mode"] = "nested_cv"
        cfg["cv"].setdefault("nested_cv", {})["locked_test_ratio"] = float(args.locked_test_ratio)
        cfg["cv"].setdefault("nested_cv", {})["patient_id_mode"] = str(args.patient_id_mode)
        cfg["cv"]["cv_n_splits"] = 5
        cfg["cv"]["run_folds"] = 5
        cfg["cv"]["nested_cv"]["outer_folds"] = 5
        _warn_noop_overrides(name, cfg, item["overrides"])
        cfg_path = os.path.join(cfg_dir, f"{name}.yaml")
        _save_yaml(cfg_path, cfg)
        registry.append(
            {
                "name": name,
                "config": cfg_path,
                "notes": item.get("notes", ""),
                "overrides": item["overrides"],
            }
        )

    with open(os.path.join(exp_root, "matrix_registry.json"), "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)

    _log(f"[Generate] {len(registry)} configs generated under: {exp_root}")

    if not args.run:
        _log("[Generate] Done. Add --run to execute the matrix.")
        return

    effective_folds = 5
    effective_run_folds = 5

    limit = args.max_runs if args.max_runs > 0 else len(registry)
    jobs = max(1, int(args.jobs))
    gpu_ids = [x.strip() for x in str(args.gpu_ids).split(",") if x.strip()]
    if gpu_ids and jobs > len(gpu_ids):
        _log(
            f"[Run][Warn] jobs={jobs} > gpu_slots={len(gpu_ids)}; some workers will share GPUs and may reduce speed."
        )

    tasks = []
    for idx, row in enumerate(registry[:limit], start=1):
        name = row["name"]
        run_log = os.path.join(run_dir, f"{idx:02d}_{name}.log")
        slot = (idx - 1) % jobs
        env_overrides = None
        gpu_note = "default"
        if gpu_ids:
            gpu = gpu_ids[slot % len(gpu_ids)]
            env_overrides = {"CUDA_VISIBLE_DEVICES": gpu}
            gpu_note = gpu
        tasks.append(
            {
                "idx": idx,
                "row": row,
                "run_log": run_log,
                "env_overrides": env_overrides,
                "gpu_note": gpu_note,
            }
        )

    rows = []
    if jobs == 1:
        for t in tasks:
            _log(f"[Run][Start] #{t['idx']:02d} {t['row']['name']} gpu={t['gpu_note']}")
            result = _run_and_collect(
                t["idx"],
                t["row"],
                t["run_log"],
                args.train_script,
                effective_folds,
                effective_run_folds,
                args.epochs,
                args.resume,
                    args.split_mode,
                    args.train_ratio,
                    args.val_ratio,
                    args.test_ratio,
                t["env_overrides"],
                True,
            )
            rows.append(result)
            _log(
                "[Summary] "
                f"{result['name']} code={result['return_code']} raw_auc(mean/std/worst)="
                f"{result['raw_auc_mean']:.4f}/{result['raw_auc_std']:.4f}/{result['raw_auc_worst']:.4f}"
            )
    else:
        _log(f"[Run] Parallel mode enabled: jobs={jobs}")
        for t in tasks:
            _log(f"[Run][Queue] #{t['idx']:02d} {t['row']['name']} gpu={t['gpu_note']}")

        with ThreadPoolExecutor(max_workers=jobs) as ex:
            futures = {
                ex.submit(
                    _run_and_collect,
                    t["idx"],
                    t["row"],
                    t["run_log"],
                    args.train_script,
                    effective_folds,
                    effective_run_folds,
                    args.epochs,
                    args.resume,
                    args.split_mode,
                    args.train_ratio,
                    args.val_ratio,
                    args.test_ratio,
                    t["env_overrides"],
                    False,
                ): t
                for t in tasks
            }
            for fut in as_completed(futures):
                result = fut.result()
                rows.append(result)
                _log(
                    "[Summary] "
                    f"#{result['idx']:02d} {result['name']} code={result['return_code']} raw_auc(mean/std/worst)="
                    f"{result['raw_auc_mean']:.4f}/{result['raw_auc_std']:.4f}/{result['raw_auc_worst']:.4f}"
                )

    rows.sort(key=lambda x: x["idx"])
    for r in rows:
        r.pop("idx", None)

    summary_json = os.path.join(exp_root, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    summary_csv = os.path.join(exp_root, "summary.csv")
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["name"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    _log(f"[Done] Summary JSON: {summary_json}")
    _log(f"[Done] Summary CSV:  {summary_csv}")


if __name__ == "__main__":
    main()
