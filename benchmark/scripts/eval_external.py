#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
External / cross-cohort generalization evaluation.

Load a trained model from a completed nested-CV run and evaluate it on an
EXTERNAL dataset (e.g. MCI, or AD/LBD from a different field strength / site).

The external data must be the same 112^3 normalized .npy format produced by
scripts/prepare_npy.py (convert raw 128^3 npy first with prepare_npy.py).

Usage:
  # Binary external (cross field strength: AD & LBD at 1.5T)
  python scripts/eval_external.py --run-dir results/anil/seed_42/<ts> \\
      --data-root ./Dataset_112_npy_ext --map "AD=0,LBD=1" --out ./reports/gen_1p5t

  # Single-class external (MCI — no LBD/AD ground truth; report P(LBD) dist.)
  python scripts/eval_external.py --run-dir results/medmamba_ss3m/seed_42/<ts> \\
      --data-root ./Dataset_112_npy_ext/MCI --map "MCI=-1" --out ./reports/gen_mci

Output: metrics.json (+ roc_curve.png/csv when binary), predictions.csv with
real subject ids, and a summary of the P(LBD)/P(AD) distribution.
"""
import os
import sys
import json
import csv
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_loader import load_config
from core.utils import set_seed, detect_gpu, autocast_ctx
from core.dataset import MRIVolumeFolderDataset, make_collate_filter_none
from core.metrics import compute_metrics_from_raw, save_roc_curve, save_predictions_csv
from methods.method_registry import build_method


def _parse_map(map_str):
    """'AD=0,LBD=1,MCI=-1' -> {'AD': 0, 'LBD': 1, 'MCI': -1}."""
    out = {}
    for item in map_str.split(","):
        if not item.strip():
            continue
        k, _, v = item.strip().partition("=")
        out[k.strip()] = int(float(v.strip()))
    return out


def main():
    parser = argparse.ArgumentParser(description="External generalization evaluation")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Completed run dir, e.g. results/<method>/seed_42/<ts>")
    parser.add_argument("--data-root", type=str, required=True,
                        help="External dataset root (label subfolders) or a single-class dir")
    parser.add_argument("--map", type=str, required=True,
                        help="Label mapping 'AD=0,LBD=1' (binary) or 'MCI=-1' (unlabeled)")
    parser.add_argument("--out", type=str, default="./reports/external",
                        help="Output directory")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint path (default: run_dir/holdout_test/retrain/best_model.pth)")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device, device_type, _ = detect_gpu()
    os.makedirs(args.out, exist_ok=True)

    # --- Load trained model from the run's config snapshot ---
    cfg_path = os.path.join(args.run_dir, "config_snapshot.yaml")
    if not os.path.isfile(cfg_path):
        print(f"[eval_external] config_snapshot.yaml not found: {cfg_path}")
        return 1
    cfg = load_config(cfg_path)
    method_name = cfg.get("experiment", {}).get("method", "?")
    print(f"[eval_external] method={method_name}, run={args.run_dir}")

    if args.checkpoint:
        ckpt_path = args.checkpoint
    else:
        ckpt_path = os.path.join(args.run_dir, "holdout_test", "retrain", "best_model.pth")
    if not os.path.isfile(ckpt_path):
        print(f"[eval_external][ERROR] checkpoint not found: {ckpt_path}")
        print("  (checkpoint is written only when training improved past min_epochs)")
        return 1

    method = build_method(method_name, cfg)
    method.build_model()
    method.prepare_for_training(device, pretrained_path=None)
    sd = torch.load(ckpt_path, map_location="cpu")
    missing, unexpected = method.model.load_state_dict(sd, strict=False)
    print(f"[eval_external] loaded checkpoint: missing={len(missing)}, unexpected={len(unexpected)}")
    method.model.to(device)
    method.eval()

    # --- Build external dataset ---
    label_map = _parse_map(args.map)
    roots = {}
    for lab in label_map:
        p = os.path.join(args.data_root, lab)
        if os.path.isdir(p):
            roots[lab] = p
    if not roots:
        # single-class dir directly under data_root
        for lab in label_map:
            if os.path.isdir(args.data_root):
                roots = {lab: args.data_root}
                break
    if not roots:
        print(f"[eval_external][ERROR] no external label dirs found under {args.data_root}")
        return 1
    print(f"[eval_external] external roots: {roots}, label_map={label_map}")

    ds = MRIVolumeFolderDataset(
        label_roots=roots,
        label_map=label_map,
        target_shape=tuple(cfg.get("data", {}).get("target_shape", [112, 112, 112])),
        file_exts=(".npy", ".nii", ".nii.gz"),
        validate_nifti=False,
        cache_enabled=False,
        normalized=bool(cfg.get("data", {}).get("normalized", False)),
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=0, pin_memory=True,
                        collate_fn=make_collate_filter_none())

    # --- Forward ---
    all_y_true, all_y_prob, all_pids = [], [], []
    amp = bool(cfg.get("training", {}).get("use_amp", True)) and device_type == "cuda"
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            x = x.to(device)
            with autocast_ctx(enabled=amp, dtype=torch.bfloat16):
                logits = method.forward_classifier(x)
            probs = torch.softmax(logits.float(), dim=1).cpu().numpy()
            all_y_true.extend(y.cpu().numpy().tolist())
            all_y_prob.append(probs)
    all_y_prob = np.concatenate(all_y_prob, axis=0)
    all_y_true = np.array(all_y_true, dtype=np.int64)
    all_pids = list(ds.patient_ids)

    # Test threshold: use the run's recorded training-side threshold if available
    test_thr = 0.5
    ri_path = os.path.join(args.run_dir, "run_info.json")
    if os.path.isfile(ri_path):
        ri = json.load(open(ri_path, encoding="utf-8"))
        test_thr = ri.get("test_threshold") if ri.get("test_threshold") is not None else 0.5

    labeled = (all_y_true >= 0)
    n_binary = (all_y_true == 0).sum() > 0 and (all_y_true == 1).sum() > 0

    # --- Predictions CSV ---
    save_predictions_csv(all_y_true, all_y_prob,
                         os.path.join(args.out, "predictions.csv"),
                         patient_ids=all_pids, fold_label="external",
                         threshold=test_thr)

    summary = {
        "run_dir": args.run_dir,
        "method": method_name,
        "external_root": args.data_root,
        "label_map": label_map,
        "n_samples": int(len(all_y_true)),
        "test_threshold": test_thr,
        "p_lbd_mean": float(all_y_prob[:, 1].mean()),
        "p_lbd_std": float(all_y_prob[:, 1].std()),
        "n_predicted_lbd": int((all_y_prob[:, 1] >= test_thr).sum()),
        "n_predicted_ad": int((all_y_prob[:, 1] < test_thr).sum()),
    }

    if n_binary:
        metrics = compute_metrics_from_raw(all_y_true[labeled], all_y_prob[labeled], 2, test_thr)
        summary.update({
            "auc": float(metrics["auc"]), "bal_acc": float(metrics["bal_acc"]),
            "acc": float(metrics["acc"]), "sens": float(metrics["sens"]),
            "spec": float(metrics["spec"]), "f1": float(metrics["f1"]),
            "threshold_source": "run_test_threshold",
        })
        save_roc_curve(all_y_true[labeled], all_y_prob[labeled],
                       os.path.join(args.out, "roc_curve.png"),
                       title=f"External generalization — {method_name}")
        print(f"[eval_external] EXTERNAL AUC={metrics['auc']:.4f} BAC={metrics['bal_acc']:.4f} "
              f"sens={metrics['sens']:.4f} spec={metrics['spec']:.4f}")
    else:
        summary["mode"] = "unlabeled_distribution"
        print(f"[eval_external] single-class external ({n_binary=}): "
              f"P(LBD) mean={summary['p_lbd_mean']:.4f}±{summary['p_lbd_std']:.4f}, "
              f"pred-LBD={summary['n_predicted_lbd']}/{summary['n_samples']}")

    with open(os.path.join(args.out, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[eval_external] summary -> {os.path.join(args.out, 'summary.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
