#!/usr/bin/env python
"""
Run a single method × seed experiment.

Modes:
  Nested CV (default):     lock 20% test set → inner 5-fold CV → holdout eval
  Pure CV (--holdout 0.0): 5-fold CV on full dataset → fast method comparison

Usage:
  python scripts/run_single.py --method anil --seed 42
  python scripts/run_single.py --method vanilla --seed 42 --folds 2 --epochs 5
  python scripts/run_single.py --method anil --seed 42 --holdout-test 0.2
  python scripts/run_single.py --list-methods

Output goes to: results/{method}/seed_{N}/{YYYYMMDD_HHMMSS}/
Each run creates a timestamped subdirectory — no overwriting.
"""
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import json
import argparse
import copy

# Add benchmark root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_loader import load_config, resolve_data_paths, save_config
from core.utils import set_seed, detect_gpu, setup_logging, format_time
from core.dataset import build_dataset
from core.evaluator import CrossValidator
from methods.method_registry import build_method, list_methods


def main():
    parser = argparse.ArgumentParser(
        description="Run single benchmark experiment for AD vs LBD",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available methods (from paper):
  Meta-learning strategies: vanilla, anil, protonet, maml, hybrid
  Backbone architectures: medmamba_ss3m, resnet3d, densenet3d,
                           cnn3d_baseline, convnext3d, swin3d,
                           crossformer3d, medmamba3d

Quick examples:
  python run_single.py --method vanilla --seed 42 --folds 2 --epochs 5        # quick test
  python run_single.py --method anil --seed 42                                # nested CV (default)
  python run_single.py --method anil --seed 42 --holdout-test 0.0             # pure 5-fold CV
  python run_single.py --list-methods
        """
    )
    parser.add_argument("--method", type=str, default=None,
                        help="Method name (use --list-methods to see all)")
    parser.add_argument("--list-methods", action="store_true",
                        help="List all available methods and exit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                        help="Path to base config")
    parser.add_argument("--folds", type=int, default=None,
                        help="Override number of CV folds (for quick testing)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs (for quick testing)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--note", type=str, default="", help="Experiment note")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory override")
    parser.add_argument("--pretrained", type=str, default=None,
                        help="Pretrained weights path override")
    parser.add_argument("--set", type=str, action="append", default=None, dest="set_overrides",
                        help="Config override as key=value (repeatable). E.g. --set ss3m.use_dual_branch=false")
    parser.add_argument("--holdout-test", type=float, default=0.2,
                        help="Holdout test fraction for nested CV (0.2 = 80/20, 0.0 = pure CV)")
    parser.add_argument("--protocol", type=str, default="configs/protocol_rerun.yaml",
                        help="Shared protocol config loaded after the method config "
                             "(set to '' to disable)")
    parser.add_argument("key_value", nargs="*", help="Additional config overrides as key=value")
    args = parser.parse_args()

    # --list-methods
    if args.list_methods:
        from core.adapters import print_adapter_info
        print("\n" + "=" * 65)
        print("Available Methods for AD vs LBD Benchmark")
        print("=" * 65)
        print("\n  Meta-learning strategies (Group A):")
        print("    vanilla  — Standard supervised (No-meta baseline)")
        print("    anil     — Almost No Inner Loop (default)")
        print("    protonet — Prototypical Networks")
        print("    maml     — Model-Agnostic Meta-Learning")
        print("    hybrid   — ANIL + ProtoNet combined")
        print("\n  Backbone architectures (Group B):")
        print("    medmamba_ss3m   — MedMambaSS3D (our method, reference)")
        print("    resnet3d        — 3D ResNet-18 (DeepSPARE)")
        print("    densenet3d      — 3D DenseNet")
        print("    cnn3d_baseline  — 3D CNN (CNN_design_for_AD)")
        print("    convnext3d      — ConvNeXt 3D (modern CNN)")
        print("    medmamba3d      — MedMamba3D (simplified)")
        print("\n" + "=" * 65)
        print("Usage modes:")
        print("  Pure CV:      python run_single.py --method anil --seed 42")
        print("  Nested CV:    python run_single.py --method anil --seed 42 --holdout-test 0.2")
        print("  Quick test:   python run_single.py --method vanilla --seed 42 --folds 2 --epochs 5")
        print("=" * 65)
        print_adapter_info()
        return 0

    if not args.method:
        print("[ERROR] --method is required. Use --list-methods to see available methods.")
        return 1

    # Load configs (base → method → shared rerun protocol → CLI overrides)
    base_cfg = args.config
    method_cfg = f"configs/methods/{args.method}.yaml"
    configs = [base_cfg]
    if os.path.isfile(method_cfg):
        configs.append(method_cfg)
    if args.protocol and os.path.isfile(args.protocol):
        configs.append(args.protocol)
    elif args.protocol and not args.protocol.startswith("configs/methods"):
        # protocol file missing but requested — warn (not fatal)
        print(f"[WARN] Protocol config not found: {args.protocol}", flush=True)

    cli_overrides = list(args.key_value) if args.key_value else []
    if args.set_overrides:
        cli_overrides.extend(args.set_overrides)
    if args.epochs:
        cli_overrides.append(f"training.epochs={args.epochs}")
    if args.folds:
        cli_overrides.append(f"cv.n_splits={args.folds}")
        cli_overrides.append(f"cv.run_folds={args.folds}")
    if args.note:
        cli_overrides.append(f"experiment.note={args.note}")
    if args.pretrained:
        cli_overrides.append(f"model.pretrained_path={args.pretrained}")

    cfg = load_config(*configs, cli_overrides=cli_overrides)
    cfg = resolve_data_paths(cfg)
    cfg["experiment"]["seed"] = args.seed

    # Setup
    set_seed(args.seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device, device_type, gpu_count = detect_gpu()
    logger = setup_logging()

    if device_type != "cuda":
        logger.warning("CUDA not available — running on CPU (will be slow!)")

    # ——— Output directory with timestamp ———
    method_name = cfg["experiment"]["method"]
    output_root = args.output or cfg.get("output", {}).get("root_dir", "./results")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    holdout_tag = f"_h{args.holdout_test}" if args.holdout_test > 0 else ""
    note_tag = f"_{args.note}" if args.note else ""
    run_dir_name = f"{timestamp}{holdout_tag}{note_tag}"
    output_dir = os.path.join(output_root, method_name, f"seed_{args.seed}", run_dir_name)
    os.makedirs(output_dir, exist_ok=True)

    mode_str = "nested_cv" if args.holdout_test > 0 else "pure_cv"
    logger.info(f"Method: {method_name}, Seed: {args.seed}, Mode: {mode_str}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Device: {device}")

    # Save a run_info.json
    with open(os.path.join(output_dir, "run_info.json"), "w") as f:
        json.dump({
            "method": method_name, "seed": args.seed, "mode": mode_str,
            "holdout_fraction": args.holdout_test,
            "folds": args.folds or cfg.get("cv", {}).get("n_splits", 5),
            "epochs": args.epochs or cfg.get("training", {}).get("epochs", 150),
            "gpu": args.gpu, "note": args.note,
            "timestamp": timestamp,
        }, f, indent=2)

    # Build dataset
    logger.info("Building dataset...")
    dataset = build_dataset(cfg)
    logger.info(f"Dataset: {len(dataset)} samples, {dataset.in_channels} channels")

    # Build method and run CV
    logger.info(f"Building method: {method_name}")
    method = build_method(method_name, cfg)
    method.build_model()
    param_count = sum(p.numel() for p in method.parameters())
    logger.info(f"Model parameters: {param_count:,}")

    cv = CrossValidator(method, cfg, device)
    t0 = time.time()

    cv_summary = cv.run(
        dataset,
        output_dir,
        seed=args.seed,
        quick_folds=args.folds,
        quick_epochs=args.epochs,
        holdout_fraction=args.holdout_test,
    )

    elapsed = time.time() - t0
    logger.info(f"Done in {format_time(elapsed)}")
    logger.info(f"Results saved to {output_dir}")

    # Enrich run_info.json with runtime facts (params, wall time, holdout summary)
    try:
        with open(os.path.join(output_dir, "run_info.json"), "r") as f:
            _ri = json.load(f)
        _ri["param_count"] = param_count
        _ri["elapsed_seconds"] = round(elapsed, 1)
        _ri["elapsed_human"] = format_time(elapsed)
        ht = cv_summary.get("holdout_test") or {}
        if ht:
            _ri["test_auc"] = ht.get("auc")
            _ri["test_bal_acc"] = ht.get("bal_acc")
            _ri["test_sens"] = ht.get("sens")
            _ri["test_spec"] = ht.get("spec")
            _ri["test_threshold"] = ht.get("best_threshold")
            _ri["test_threshold_source"] = ht.get("threshold_source")
        with open(os.path.join(output_dir, "run_info.json"), "w") as f:
            json.dump(_ri, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not enrich run_info.json: {e}")

    # Print final summary
    print(f"\n{'='*60}")
    print(f"Method: {method_name} | Seed: {args.seed} | Mode: {mode_str}")
    print(f"Output: {output_dir}")
    for metric in ["auc", "bal_acc", "acc", "sens", "spec", "f1"]:
        m = cv_summary.get(metric, {})
        print(f"  {metric}: {m.get('mean', float('nan')):.4f} ± {m.get('std', float('nan')):.4f}")
    if "holdout_test" in cv_summary:
        ht = cv_summary["holdout_test"]
        print(f"  [Holdout] AUC={ht.get('auc','N/A'):.4f} BAC={ht.get('bal_acc','N/A'):.4f}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
