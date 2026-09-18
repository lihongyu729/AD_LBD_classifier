#!/usr/bin/env python
"""
Generate comparison tables and plots from experiment results.

Reports the holdout TEST metrics as the primary numbers (the paper's values),
with the inner-CV metrics as secondary reference.

Usage:
    python scripts/generate_report.py --input ./results --output ./reports
    python scripts/generate_report.py --input ./results --output ./reports --format csv,png
"""
import os
import sys
import json
import argparse
import csv
import numpy as np
from typing import Dict, List, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("[Report] matplotlib not available — skipping plots")

METRIC_KEYS = ["auc", "bal_acc", "acc", "sens", "spec", "f1"]
HOLDOUT_KEYS = [f"holdout_{k}" for k in METRIC_KEYS]


def collect_all_results(results_root: str) -> List[Dict]:
    """Walk results/ and collect all cv_summary.json files.

    Handles the timestamped layout: results/{method}/seed_{N}/{ts}/cv_summary.json
    (mirrors run_batch.collect_results).
    """
    all_results = []
    if not os.path.isdir(results_root):
        print(f"[Report] Results directory not found: {results_root}")
        return all_results

    for method_dir in sorted(os.listdir(results_root)):
        method_path = os.path.join(results_root, method_dir)
        if not os.path.isdir(method_path):
            continue
        for seed_dir in sorted(os.listdir(method_path)):
            seed_path = os.path.join(method_path, seed_dir)
            if not os.path.isdir(seed_path):
                continue
            # Timestamped run subdirectories
            for ts_dir in sorted(os.listdir(seed_path)):
                ts_path = os.path.join(seed_path, ts_dir)
                summary_path = os.path.join(ts_path, "cv_summary.json")
                if os.path.isfile(summary_path):
                    try:
                        with open(summary_path, "r") as f:
                            d = json.load(f)
                        d["_method"] = method_dir
                        d["_seed"] = seed_dir.replace("seed_", "")
                        d["_ts"] = ts_dir
                        d["_holdout"] = d.get("holdout_test")
                        all_results.append(d)
                    except Exception as e:
                        print(f"[Report] Error reading {summary_path}: {e}")
    return all_results


def _row_from_result(r: Dict) -> Dict:
    """Extract CV + holdout metric means from one cv_summary entry."""
    row = {"method": r.get("_method", r.get("method", "?")),
           "seed": r.get("_seed", r.get("seed", "?")),
           "ts": r.get("_ts", ""),
           "num_folds": r.get("num_folds", "")}
    for mk in METRIC_KEYS:
        m = r.get(mk, {})
        row[f"{mk}_mean"] = m.get("mean", "") if isinstance(m, dict) else ""
        row[f"{mk}_std"] = m.get("std", "") if isinstance(m, dict) else ""
    ho = r.get("_holdout") or {}
    for hk in HOLDOUT_KEYS:
        row[hk] = ho.get(hk[len("holdout_"):], "") if isinstance(ho, dict) else ""
    return row


def aggregate_across_seeds(results: List[Dict]) -> List[Dict]:
    """Aggregate metrics across seeds for each method (CV + holdout)."""
    methods = {}
    for r in results:
        method = r.get("_method", r.get("method", "?"))
        methods.setdefault(method, []).append(r)

    aggregated = []
    for method, entries in methods.items():
        row = {"method": method, "num_seeds": len(entries)}
        for mk in METRIC_KEYS:
            vals = []
            for e in entries:
                m = e.get(mk, {})
                mean = m.get("mean", float("nan")) if isinstance(m, dict) else float("nan")
                if not (isinstance(mean, float) and np.isnan(mean)):
                    vals.append(mean)
            row[f"{mk}_mean"] = float(np.mean(vals)) if vals else float("nan")
            row[f"{mk}_std"] = float(np.std(vals)) if vals else float("nan")
        for hk in HOLDOUT_KEYS:
            vals = []
            for e in entries:
                ho = e.get("_holdout") or {}
                v = ho.get(hk[len("holdout_"):])
                if v is not None and not (isinstance(v, float) and np.isnan(v)):
                    vals.append(float(v))
            row[hk] = float(np.mean(vals)) if vals else float("nan")
        aggregated.append(row)

    # Sort by holdout AUC mean descending (fallback CV AUC)
    aggregated.sort(
        key=lambda r: r.get("holdout_auc", float("nan")) if not (
            isinstance(r.get("holdout_auc"), float) and np.isnan(r.get("holdout_auc")))
        else r.get("auc_mean", float("-inf")),
        reverse=True,
    )
    return aggregated


def save_comparison_csv(results: List[Dict], aggregated: List[Dict], output_dir: str):
    """Save per-seed detail and aggregated CSV tables."""
    os.makedirs(output_dir, exist_ok=True)

    # Per-seed detail
    detail_path = os.path.join(output_dir, "comparison_table.csv")
    fieldnames = ["method", "seed", "ts", "num_folds"] \
        + [f"{mk}_mean" for mk in METRIC_KEYS] + [f"{mk}_std" for mk in METRIC_KEYS] \
        + HOLDOUT_KEYS
    with open(detail_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            writer.writerow(_row_from_result(r))
    print(f"[Report] Detail table saved to {detail_path}")

    # Aggregated
    agg_path = os.path.join(output_dir, "comparison_aggregated.csv")
    agg_fields = ["method", "num_seeds"] \
        + [f"{mk}_mean" for mk in METRIC_KEYS] + [f"{mk}_std" for mk in METRIC_KEYS] \
        + HOLDOUT_KEYS
    with open(agg_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=agg_fields, extrasaction="ignore")
        writer.writeheader()
        for r in aggregated:
            writer.writerow(r)
    print(f"[Report] Aggregated table saved to {agg_path}")

    # Ranking (by holdout AUC)
    rank_path = os.path.join(output_dir, "method_ranking.txt")
    with open(rank_path, "w", encoding="utf-8") as f:
        f.write("Method Ranking by TEST (holdout) AUC\n")
        f.write("=" * 64 + "\n")
        for i, r in enumerate(aggregated):
            f.write(f"{i+1}. {r['method']}: TEST AUC={r.get('holdout_auc', float('nan')):.4f}, "
                    f"TEST BAC={r.get('holdout_bal_acc', float('nan')):.4f}, "
                    f"(CV AUC={r.get('auc_mean', float('nan')):.4f}±{r.get('auc_std', float('nan')):.4f})\n")
    print(f"[Report] Ranking saved to {rank_path}")


def save_comparison_plot(aggregated: List[Dict], output_dir: str):
    """Bar chart: holdout (TEST) AUC primary, CV AUC secondary."""
    if not HAS_MPL:
        return

    os.makedirs(output_dir, exist_ok=True)
    methods = [r["method"] for r in aggregated]
    test_aucs = [r.get("holdout_auc", float("nan")) for r in aggregated]
    cv_aucs = [r.get("auc_mean", float("nan")) for r in aggregated]
    cv_stds = [r.get("auc_std", float("nan")) for r in aggregated]

    valid = [(m, t, c, s) for m, t, c, s in zip(methods, test_aucs, cv_aucs, cv_stds)
             if not np.isnan(t)]
    if not valid:
        print("[Report] No valid holdout AUCs to plot.")
        return

    methods, test_aucs, cv_aucs, cv_stds = zip(*valid)
    x = np.arange(len(methods))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    b1 = ax.bar(x - width / 2, test_aucs, width, color="steelblue", edgecolor="navy", label="Test (holdout) AUC")
    b2 = ax.bar(x + width / 2, cv_aucs, width, yerr=cv_stds, capsize=3, color="lightsteelblue",
                edgecolor="slategray", label="CV AUC (mean±std)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("AD vs LBD Classification — Test (holdout) vs CV")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()

    for bar, val in zip(b1, test_aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "comparison_plot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Report] Comparison plot saved to {os.path.join(output_dir, 'comparison_plot.png')}")


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark comparison report")
    parser.add_argument("--input", type=str, default="./results",
                        help="Results directory")
    parser.add_argument("--output", type=str, default="./reports",
                        help="Output directory for reports")
    args = parser.parse_args()

    print(f"[Report] Collecting results from {args.input}...")
    all_results = collect_all_results(args.input)
    print(f"[Report] Found {len(all_results)} experiment results")

    if not all_results:
        print("[Report] No results found. Run experiments first!")
        return 1

    aggregated = aggregate_across_seeds(all_results)
    save_comparison_csv(all_results, aggregated, args.output)
    save_comparison_plot(aggregated, args.output)

    print("\n" + "=" * 64)
    print("Method Ranking (TEST/holdout AUC):")
    print("=" * 64)
    for i, r in enumerate(aggregated):
        print(f"  {i+1}. {r['method']}: TEST AUC={r.get('holdout_auc', float('nan')):.4f}, "
              f"TEST BAC={r.get('holdout_bal_acc', float('nan')):.4f} | "
              f"CV AUC={r.get('auc_mean', float('nan')):.4f}±{r.get('auc_std', float('nan')):.4f}")
    print("=" * 64)

    return 0


if __name__ == "__main__":
    sys.exit(main())
