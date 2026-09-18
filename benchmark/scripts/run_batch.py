#!/usr/bin/env python
"""
Batch runner: all methods × all seeds with GPU management.

Usage:
    python scripts/run_batch.py --methods vanilla,anil,protonet --seeds 42,123,456 --gpus 0,1
    python scripts/run_batch.py --all --seeds 42,123,456 --gpus 0
    python scripts/run_batch.py --all --seeds 42 --gpus 0,1 --resume --jobs 2

Pattern adapted from meta/run_auc90_matrix.py: subprocess-based execution
with GPU slot assignment for clean CUDA state management.
"""
import os
import sys
import json
import time
import argparse
import subprocess
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.utils import format_time


ALL_METHODS = [
    "vanilla", "anil", "protonet", "maml", "hybrid",
    "cnn3d_baseline", "convnext3d", "densenet3d",
    "resnet3d", "medmamba3d", "medmamba_ss3m",
]

# Experiment groups aligned with paper
# NOTE: swin3d / crossformer3d are intentionally excluded — their current
# implementations do not run at 112^3 input (window/downsample incompatibility).
GROUP_A_METHODS = ["vanilla", "anil", "protonet", "maml", "hybrid"]
GROUP_B_METHODS = ["medmamba_ss3m", "resnet3d", "densenet3d", "cnn3d_baseline",
                    "convnext3d", "medmamba3d"]

METHOD_GROUPS = {
    "A": {
        "description": "Meta-learning strategy comparison (same MedMambaSS3M backbone)",
        "methods": GROUP_A_METHODS,
    },
    "B": {
        "description": "Backbone architecture comparison (standard training)",
        "methods": GROUP_B_METHODS,
    },
    "C": {
        "description": "Ablation experiment candidates (use --set to vary config)",
        "methods": ["anil"],
    },
    "all": {
        "description": "All methods (Group A + Group B)",
        "methods": list(set(GROUP_A_METHODS + GROUP_B_METHODS)),
    },
}


def parse_gpu_list(gpu_str: str) -> List[int]:
    """Parse GPU list like '0,1,2' or '0'."""
    return [int(x.strip()) for x in gpu_str.split(",") if x.strip()]


def parse_list(s: str) -> List[str]:
    """Parse comma-separated string into list."""
    return [x.strip() for x in s.split(",") if x.strip()]


def check_completed(output_root: str, method: str, seed: int) -> bool:
    """Check if a method × seed combination has already been completed.
    Looks inside timestamped subdirectories under seed_{N}/."""
    seed_dir = os.path.join(output_root, method, f"seed_{seed}")
    if not os.path.isdir(seed_dir):
        return False
    # Search for any cv_summary.json inside timestamp subdirectories
    for ts_dir in os.listdir(seed_dir):
        p = os.path.join(seed_dir, ts_dir, "cv_summary.json")
        if os.path.isfile(p):
            return True
    return False


def _get_python():
    """Get a working Python executable path (sys.executable can be empty)."""
    if sys.executable and os.path.isfile(sys.executable):
        return sys.executable
    import shutil
    for name in ("python3", "python"):
        p = shutil.which(name)
        if p:
            return p
    raise RuntimeError("Cannot find a Python interpreter")


def _benchmark_root():
    """Return absolute path to benchmark/ root (works regardless of cwd)."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_task(method: str, seed: int, gpu_id: int, extra_args: List[str]) -> Tuple[str, int, int, bool, str]:
    """Run a single method × seed task as a subprocess."""
    root = _benchmark_root()
    cmd = [
        _get_python(), "-u",
        os.path.join(root, "scripts", "run_single.py"),
        "--method", method,
        "--seed", str(seed),
        "--gpu", str(gpu_id),
    ] + extra_args

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["PYTHONUNBUFFERED"] = "1"
    env["TERM"] = "xterm-256color"  # clean terminal type

    print(f"[Batch] Starting: method={method}, seed={seed}, gpu={gpu_id}", flush=True)
    print(f"[Batch] CMD: {' '.join(cmd[:4])}...", flush=True)
    print(f"[Batch] CWD: {root}", flush=True)

    t0 = time.time()
    try:
        # Stream output in real-time, but capture stderr for error reporting
        proc = subprocess.Popen(
            cmd, env=env,
            cwd=root,
            stdout=None,  # inherit parent stdout
            stderr=subprocess.PIPE,
        )
        stderr_chunks = []
        for line in proc.stderr:
            stderr_chunks.append(line)
            sys.stderr.buffer.write(line)
            sys.stderr.buffer.flush()
        proc.wait(timeout=86400)
        elapsed = time.time() - t0

        if proc.returncode == 0:
            print(f"[Batch] OK: {method} seed={seed} ({format_time(elapsed)})", flush=True)
            return method, seed, proc.returncode, True, ""
        else:
            stderr_text = b"".join(stderr_chunks).decode("utf-8", errors="replace")[-2000:]
            print(f"[Batch] FAIL: {method} seed={seed} (exit={proc.returncode})", flush=True)
            if stderr_text.strip():
                print(f"[Batch][STDERR] {method} seed={seed}:\n{stderr_text}", flush=True)
            err_detail = stderr_text.strip().split("\n")[-1] if stderr_text.strip() else f"exit={proc.returncode}"
            return method, seed, proc.returncode, False, err_detail
    except subprocess.TimeoutExpired:
        proc.kill()
        print(f"[Batch] TIMEOUT: {method} seed={seed} (>24h)", flush=True)
        return method, seed, -1, False, "Timeout (>24h)"
    except Exception as e:
        import traceback
        print(f"[Batch] EXCEPTION: {method} seed={seed}: {e}", flush=True)
        traceback.print_exc()
        return method, seed, -1, False, str(e)


def collect_results(output_root: str) -> List[Dict]:
    """Collect all cv_summary.json files into a list (handles timestamped subdirs)."""
    results = []
    for method_dir in os.listdir(output_root):
        method_path = os.path.join(output_root, method_dir)
        if not os.path.isdir(method_path):
            continue
        for seed_dir in os.listdir(method_path):
            seed_path = os.path.join(method_path, seed_dir)
            if not os.path.isdir(seed_path):
                continue
            # Look inside timestamp subdirectories
            for ts_dir in os.listdir(seed_path):
                ts_path = os.path.join(seed_path, ts_dir)
                summary_path = os.path.join(ts_path, "cv_summary.json")
                if os.path.isfile(summary_path):
                    try:
                        with open(summary_path, "r") as f:
                            d = json.load(f)
                        d["_method"] = d.get("method", method_dir)
                        d["_seed"] = seed_dir.replace("seed_", "")
                        d["_holdout"] = d.get("holdout_test")
                        results.append(d)
                    except Exception:
                        pass
    return results


def save_summary_csv(results: List[Dict], path: str):
    """Save aggregated results as CSV."""
    if not results:
        return
    keys = ["method", "seed", "num_folds"]
    metric_keys = ["auc", "bal_acc", "acc", "sens", "spec", "f1"]
    for mk in metric_keys:
        keys.append(f"{mk}_mean")
        keys.append(f"{mk}_std")
    # Holdout (TEST) metrics — the paper's primary numbers
    holdout_keys = ["auc", "bal_acc", "acc", "sens", "spec", "f1"]
    for mk in holdout_keys:
        keys.append(f"holdout_{mk}")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row = {k: r.get(k, "") for k in ["method", "seed", "num_folds"]}
            for mk in metric_keys:
                m = r.get(mk, {})
                row[f"{mk}_mean"] = m.get("mean", "") if isinstance(m, dict) else ""
                row[f"{mk}_std"] = m.get("std", "") if isinstance(m, dict) else ""
            ho = r.get("_holdout") or {}
            for mk in holdout_keys:
                row[f"holdout_{mk}"] = ho.get(mk, "") if isinstance(ho, dict) else ""
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Batch benchmark runner for AD vs LBD comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Experiment groups (from paper):
  A: Meta-learning strategies — {', '.join(GROUP_A_METHODS)}
  B: Backbone architectures — {', '.join(GROUP_B_METHODS)}
  C: Ablation candidates — anil
  all: All methods combined

Usage examples:
  python run_batch.py --group A --seeds 42,123 --gpus 0
  python run_batch.py --methods anil,vanilla --seeds 42 --gpus 0,1
  python run_batch.py --group all --seeds 42,123,456 --gpus 0 --resume
  python run_batch.py --list-methods
        """
    )
    parser.add_argument("--methods", type=str, default=None,
                        help=f"Comma-separated method names")
    parser.add_argument("--group", type=str, default=None,
                        choices=["A", "B", "C", "all"],
                        help="Experiment group (A=meta-strategies, B=backbones, C=ablation, all=everything)")
    parser.add_argument("--all", action="store_true", help="Run all registered methods")
    parser.add_argument("--list-methods", action="store_true",
                        help="List all available methods and groups, then exit")
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated random seeds")
    parser.add_argument("--gpus", type=str, default="0",
                        help="Comma-separated GPU IDs (e.g., '0,1,2')")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                        help="Path to base config")
    parser.add_argument("--output", type=str, default="./results",
                        help="Output root directory")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-completed tasks")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Max parallel jobs (per GPU)")
    parser.add_argument("--folds", type=int, default=None,
                        help="Override CV folds for quick testing")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs for quick testing")
    parser.add_argument("--holdout-test", type=float, default=0.2,
                        help="Fraction locked as independent test set (0.2 = 80/20, 0.0 = pure CV)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print task grid without running")
    args = parser.parse_args()

    # --list-methods
    if args.list_methods:
        print("\n" + "=" * 65)
        print("Available Methods for AD vs LBD Benchmark")
        print("=" * 65)
        for group_name in ["A", "B", "C", "all"]:
            g = METHOD_GROUPS[group_name]
            print(f"\n  Group {group_name}: {g['description']}")
            for m in g["methods"]:
                tags = []
                if m in GROUP_A_METHODS and group_name == "A":
                    tags.append("meta")
                elif m in GROUP_B_METHODS:
                    tags.append("standard")
                tag_str = f" [{', '.join(tags)}]" if tags else ""
                print(f"    - {m}{tag_str}")
        print("\n" + "=" * 65)
        print("Quick start:")
        print("  python run_batch.py --group A --seeds 42 --gpus 0      # meta strategies")
        print("  python run_batch.py --group B --seeds 42 --gpus 0      # backbones")
        print("  python run_batch.py --group all --seeds 42,123 --gpus 0,1  # everything")
        print("=" * 65 + "\n")
        return 0

    # Resolve methods
    if args.all:
        methods = ALL_METHODS
    elif args.group:
        methods = METHOD_GROUPS[args.group]["methods"]
        print(f"[Batch] Group {args.group}: {METHOD_GROUPS[args.group]['description']}")
        print(f"[Batch] Methods: {methods}")
    elif args.methods:
        methods = parse_list(args.methods)
    else:
        print("[ERROR] Specify --methods, --group, or --all")
        print("Use --list-methods to see available options.")
        return 1

    seeds = [int(x) for x in parse_list(args.seeds)]
    gpus = parse_gpu_list(args.gpus)

    # Build task grid
    tasks = []
    skipped = 0
    for method in methods:
        for seed in seeds:
            if args.resume and check_completed(args.output, method, seed):
                skipped += 1
                print(f"[Batch] Skipping completed: {method} seed={seed}")
                continue
            gpu_id = gpus[len(tasks) % len(gpus)]
            tasks.append((method, seed, gpu_id))

    if skipped:
        print(f"[Batch] Skipped {skipped} completed tasks")
    print(f"[Batch] Task grid: {len(methods)} methods × {len(seeds)} seeds = {len(tasks)} tasks")
    print(f"[Batch] GPUs: {gpus}, Jobs/GPU: {args.jobs}")

    if args.dry_run:
        print("[Batch] DRY RUN — task grid (not executing):")
        for i, (method, seed, gpu) in enumerate(tasks):
            print(f"  {i+1}. method={method}, seed={seed}, gpu={gpu}")
        return 0

    if not tasks:
        print("[Batch] All tasks already completed!")
        # Collect and save aggregate
        results = collect_results(args.output)
        save_summary_csv(results, os.path.join(args.output, "summary.csv"))
        return 0

    # Build extra args
    extra_args = ["--config", args.config, "--output", args.output]
    if args.folds:
        extra_args.extend(["--folds", str(args.folds)])
    if args.epochs:
        extra_args.extend(["--epochs", str(args.epochs)])
    extra_args.extend(["--holdout-test", str(args.holdout_test)])

    # Run tasks with concurrency limited by --jobs
    max_workers = min(args.jobs * len(gpus), len(tasks))
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(run_task, m, s, g, extra_args): (m, s)
            for m, s, g in tasks
        }
        for future in as_completed(futures):
            method, seed, code, ok, err = future.result()
            if not ok:
                print(f"[Batch][ERROR] {method} seed={seed}: exit={code}, {err[:200]}")

    elapsed = time.time() - t0
    print(f"\n[Batch] All tasks completed in {format_time(elapsed)}")

    # Collect and save aggregate
    results = collect_results(args.output)
    save_summary_csv(results, os.path.join(args.output, "summary.csv"))
    print(f"[Batch] Summary saved to {os.path.join(args.output, 'summary.csv')}")

    # Quick ranking — by holdout (TEST) AUC when present, else CV AUC
    if results:
        print("\n[Batch] Quick ranking by TEST (holdout) AUC:")
        ranked = sorted(
            results,
            key=lambda r: (r.get("_holdout") or {}).get("auc", float("-inf")) if (r.get("_holdout") or {}).get("auc") is not None
            else r.get("auc", {}).get("mean", float("-inf")),
            reverse=True,
        )
        for i, r in enumerate(ranked):
            ho = r.get("_holdout") or {}
            auc_m = ho.get("auc") if ho.get("auc") is not None else r.get("auc", {}).get("mean", float("nan"))
            auc_s = r.get("auc", {}).get("std", float("nan"))
            method = r.get("method", "?")
            seed = r.get("seed", "?")
            print(f"  {i+1}. {method} (seed={seed}): TEST AUC={auc_m:.4f} (CV {auc_s:.4f}±std)")

    return 0


if __name__ == "__main__":
    sys.exit(main())
