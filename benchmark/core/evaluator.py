"""
Cross-validation evaluator: supports pure K-fold CV (default) and
nested CV with holdout test set (--holdout-test mode).

Pure CV (holdout_fraction=0):
    Full dataset → 5-fold stratified CV → cv_summary.json

Nested CV (holdout_fraction>0, e.g. 0.2):
    Full dataset → StratifiedShuffleSplit(80% train_val, 20% holdout_test)
        train_val → 5-fold CV → cv_summary.json
        After CV: retrain best config on full train_val → evaluate on holdout_test
"""
import os
import json
import csv
import copy
import time
import numpy as np
import torch
from torch.utils.data import Subset
from typing import Dict, Any, List, Optional, Tuple

from .metrics import (
    aggregate_cv_metrics, save_cm_png, save_metrics_curves, save_predictions_csv,
    save_combined_cm_png, save_cv_table_csv, compute_metrics_from_raw,
    evaluate_loader, save_roc_curve,
)
from .trainer import Trainer
from .config_loader import save_config
from .utils import autocast_ctx


class CrossValidator:
    """
    Run K-fold or nested cross-validation for one method x seed combination.

    Usage:
        cv = CrossValidator(method, config, device)

        # Pure 5-fold CV
        summary = cv.run(dataset, output_dir, seed=42)

        # Nested CV with 20% holdout
        summary = cv.run(dataset, output_dir, seed=42, holdout_fraction=0.2)
    """

    def __init__(self, method, config: Dict[str, Any], device: torch.device):
        self.method = method
        self.config = config
        self.device = device
        self.trainer = Trainer(method, config, device)

        cv_cfg = config.get("cv", {})
        self.n_splits = int(cv_cfg.get("n_splits", 5))
        self.run_folds = int(cv_cfg.get("run_folds", 5))
        self.shuffle = bool(cv_cfg.get("shuffle", True))
        self.stratified = bool(cv_cfg.get("stratified", True))
        self.sample_fraction = float(cv_cfg.get("sample_fraction", 1.0))

    # =====================================================================
    # Public API — main entry point
    # =====================================================================

    def run(
        self,
        dataset,
        output_dir: str,
        seed: int = 42,
        quick_folds: Optional[int] = None,
        quick_epochs: Optional[int] = None,
        holdout_fraction: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Run CV evaluation. If holdout_fraction > 0, uses nested CV.

        Args:
            dataset: MRIVolumeFolderDataset instance.
            output_dir: Base directory.
            seed: Random seed.
            quick_folds: Override n_splits for quick testing.
            quick_epochs: Override epochs for quick testing.
            holdout_fraction: If >0, fraction locked as independent test set.

        Returns:
            cv_summary dict with aggregated metrics.
        """
        os.makedirs(output_dir, exist_ok=True)
        save_config(self.config, os.path.join(output_dir, "config_snapshot.yaml"))

        # Collect labels
        all_labels = self._collect_labels(dataset)
        print(f"[CV] Dataset: {len(all_labels)} samples", flush=True)

        n_folds = quick_folds if quick_folds else self.n_splits
        run_folds_val = min(quick_folds or self.run_folds, n_folds)

        # ——— Nested CV path ———
        if holdout_fraction > 0:
            return self._run_nested(
                dataset, all_labels, output_dir, seed,
                n_folds, run_folds_val, holdout_fraction,
                quick_epochs,
            )

        # ——— Pure CV path ———
        return self._run_pure_cv(
            dataset, all_labels, output_dir, seed,
            n_folds, run_folds_val, quick_epochs,
        )

    # =====================================================================
    # Pure CV
    # =====================================================================

    def _run_pure_cv(
        self, dataset, all_labels, output_dir, seed,
        n_folds, run_folds_val, quick_epochs,
    ) -> Dict[str, Any]:
        """Standard K-fold CV on the full dataset."""
        splits = self._kfold_split(all_labels, n_folds, seed)
        print(f"[CV] Pure {n_folds}-fold CV (running {run_folds_val} folds), seed={seed}", flush=True)

        return self._run_folds(
            dataset, dataset, output_dir, seed, splits,
            run_folds_val, quick_epochs, mode="pure_cv",
        )

    # =====================================================================
    # Nested CV
    # =====================================================================

    def _run_nested(
        self, dataset, all_labels, output_dir, seed,
        n_folds, run_folds_val, holdout_fraction, quick_epochs,
    ) -> Dict[str, Any]:
        """
        Nested CV:
          1. Stratified split → train_val + holdout_test
          2. Inner K-fold CV on train_val
          3. Retrain on full train_val → evaluate on holdout_test once
        """
        from sklearn.model_selection import StratifiedShuffleSplit

        print(f"\n{'='*60}")
        print(f"[Nested CV] Holdout fraction: {holdout_fraction}")
        print(f"{'='*60}")

        # Step 1: Holdout split by patient if possible
        all_indices = np.arange(len(all_labels))
        sss = StratifiedShuffleSplit(
            n_splits=1, test_size=holdout_fraction, random_state=seed,
        )
        train_val_idx, holdout_idx = next(sss.split(all_indices, all_labels))
        train_val_idx = train_val_idx.tolist()
        holdout_idx = holdout_idx.tolist()

        holdout_labels = all_labels[holdout_idx]
        holdout_ad = int((holdout_labels == 0).sum())
        holdout_lbd = int((holdout_labels == 1).sum())

        print(f"[Nested CV] train_val: {len(train_val_idx)} samples "
              f"(AD={int((all_labels[train_val_idx]==0).sum())}, LBD={int((all_labels[train_val_idx]==1).sum())})")
        print(f"[Nested CV] holdout_test: {len(holdout_idx)} samples "
              f"(AD={holdout_ad}, LBD={holdout_lbd})", flush=True)

        # Save holdout info (includes full split indices + subject ids for reproducibility)
        holdout_dir = os.path.join(output_dir, "holdout_test")
        os.makedirs(holdout_dir, exist_ok=True)
        split_info = {
            "holdout_fraction": holdout_fraction,
            "train_val_size": len(train_val_idx),
            "holdout_size": len(holdout_idx),
            "holdout_AD": holdout_ad, "holdout_LBD": holdout_lbd,
            "seed": seed,
            "train_val_indices": train_val_idx,
            "holdout_indices": holdout_idx,
        }
        if getattr(dataset, "patient_ids", None):
            split_info["train_val_patient_ids"] = [dataset.patient_ids[i] for i in train_val_idx]
            split_info["holdout_patient_ids"] = [dataset.patient_ids[i] for i in holdout_idx]
        with open(os.path.join(holdout_dir, "split_info.json"), "w") as f:
            json.dump(split_info, f, indent=2)

        # Step 2: Inner K-fold CV on train_val
        tv_labels = all_labels[train_val_idx]
        inner_splits = self._kfold_split(tv_labels, n_folds, seed + 1000)
        # Remap inner indices to train_val subset
        inner_splits = [
            ([train_val_idx[i] for i in tr], [train_val_idx[i] for i in va])
            for tr, va in inner_splits
        ]

        print(f"\n[Nested CV] Inner {n_folds}-fold CV on train_val...", flush=True)

        cv_summary = self._run_folds(
            dataset, dataset, output_dir, seed, inner_splits,
            run_folds_val, quick_epochs, mode="nested_cv",
        )

        # Step 3: Retrain on full train_val → evaluate on holdout_test
        print(f"\n{'='*60}")
        print(f"[Nested CV] Retraining on full train_val → holdout test evaluation")
        print(f"{'='*60}")

        # Training-side threshold for the test set: prefer the mean of the
        # inner-CV fold val thresholds (far more val samples to tune on than
        # the retrain's tiny internal val split — with ~30 LBD, the internal
        # val has only ~3 LBD and calibration tuning falls back to 0.5).
        _fold_ths = cv_summary.get("fold_thresholds") or []
        _thr_cands = [t for t in _fold_ths if t is not None]
        _test_thr = float(np.mean(_thr_cands)) if len(_thr_cands) >= 2 else None

        holdout_metrics = self._retrain_evaluate_holdout(
            dataset, train_val_idx, holdout_idx,
            output_dir, holdout_dir, seed, quick_epochs,
            test_threshold=_test_thr,
        )

        # Merge
        cv_summary["mode"] = "nested_cv"
        cv_summary["holdout_fraction"] = holdout_fraction
        cv_summary["train_val_samples"] = len(train_val_idx)
        cv_summary["holdout_test_samples"] = len(holdout_idx)
        cv_summary["holdout_AD"] = holdout_ad
        cv_summary["holdout_LBD"] = holdout_lbd
        cv_summary["holdout_test"] = holdout_metrics

        with open(os.path.join(output_dir, "cv_summary.json"), "w") as f:
            json.dump(cv_summary, f, indent=2)

        return cv_summary

    # =====================================================================
    # Fold execution loop (shared by pure and nested CV)
    # =====================================================================

    def _run_folds(
        self, train_dataset, val_dataset, output_dir, seed,
        splits, run_folds_val, quick_epochs, mode,
    ) -> Dict[str, Any]:
        """Execute the fold training loop, collect metrics and CSVs."""
        fold_metrics = []
        fold_histories = []
        fold_cms = []
        all_predictions = []  # (y_true, y_prob) per fold

        # Open combined train log
        log_all_path = os.path.join(output_dir, "train_log_all.csv")
        log_all_file = open(log_all_path, "w", newline="", encoding="utf-8")
        log_writer = csv.writer(log_all_file)
        log_writer.writerow(["fold", "epoch", "train_loss", "val_loss", "train_acc", "val_acc",
                            "train_auc", "val_auc", "val_bal_acc", "val_sens", "val_spec"])
        log_all_file.flush()

        for fold_idx in range(run_folds_val):
            train_idx, val_idx = splits[fold_idx]
            train_idx = train_idx.tolist() if not isinstance(train_idx, list) else train_idx
            val_idx = val_idx.tolist() if not isinstance(val_idx, list) else val_idx

            print(f"\n{'='*50}")
            print(f"[CV] Fold {fold_idx + 1}/{run_folds_val}", flush=True)
            print(f"{'='*50}")

            # Rebuild model
            self.method.build_model()
            self.method.prepare_for_training(
                self.device,
                pretrained_path=self.config.get("model", {}).get("pretrained_path"),
            )
            if quick_epochs:
                self.config["training"]["epochs"] = quick_epochs

            fold_dir = os.path.join(output_dir, f"fold_{fold_idx}")
            os.makedirs(fold_dir, exist_ok=True)

            # Train fold
            fold_result = self.trainer.train_fold(
                train_dataset, val_dataset, fold_idx, fold_dir,
                train_indices=train_idx, val_indices=val_idx,
            )

            # Extract metrics
            metrics_clean = {
                "acc": fold_result.get("acc", 0),
                "bal_acc": fold_result.get("bal_acc", 0),
                "auc": fold_result.get("auc", float("nan")),
                "sens": fold_result.get("sens", float("nan")),
                "spec": fold_result.get("spec", float("nan")),
                "f1": fold_result.get("f1", float("nan")),
            }
            metrics_clean["best_threshold"] = fold_result.get("best_threshold")
            fold_metrics.append(metrics_clean)
            fold_histories.append(fold_result.get("history", {}))

            # Collect CM
            cm = fold_result.get("cm")
            if cm is not None:
                fold_cms.append(cm)

            # Collect predictions for CSV
            y_true_arr = fold_result.get("_y_true")
            y_prob_arr = fold_result.get("_y_prob")
            if y_true_arr is not None and y_prob_arr is not None:
                all_predictions.append((y_true_arr, y_prob_arr))
                # Real subject IDs (dataset.patient_ids aligned with global indices)
                _pids = None
                if getattr(train_dataset, "patient_ids", None):
                    _pids = [train_dataset.patient_ids[i] for i in val_idx]
                # Save per-fold predictions CSV (threshold-consistent y_pred)
                save_predictions_csv(
                    y_true_arr, y_prob_arr,
                    os.path.join(fold_dir, "predictions.csv"),
                    patient_ids=_pids,
                    fold_label=str(fold_idx),
                    threshold=fold_result.get("best_threshold"),
                )
                # Per-fold ROC curve
                save_roc_curve(
                    y_true_arr, y_prob_arr,
                    os.path.join(fold_dir, "roc_curve.png"),
                    title=f"Fold {fold_idx + 1} ROC",
                )

            # Save per-fold train log CSV
            self._save_fold_train_log(fold_result, fold_dir)

            # Append to combined log
            self._append_combined_log(log_writer, fold_result, fold_idx)

            # Plot metrics curves
            hist = fold_result.get("history", {})
            if hist:
                save_metrics_curves(fold_dir, fold_idx, hist)

        log_all_file.close()

        # Combined ROC curve across all folds
        if all_predictions:
            all_yt = np.concatenate([t for t, _ in all_predictions])
            all_yp = np.concatenate([p for _, p in all_predictions])
            save_roc_curve(
                all_yt, all_yp,
                os.path.join(output_dir, "roc_combined.png"),
                title=f"{self.config.get('experiment', {}).get('method', 'method')} — Combined ROC (all folds)",
            )

        # Save combined confusion matrix
        if fold_cms:
            all_cm_sum = sum(fold_cms)
            combined_cm_path = os.path.join(output_dir, "combined_cm.png")
            save_combined_cm_png(fold_cms, combined_cm_path)
            with open(os.path.join(output_dir, "combined_cm_raw.json"), "w") as f:
                json.dump(all_cm_sum.tolist(), f, indent=2)

        # Save cv_table.csv
        if fold_metrics:
            save_cv_table_csv(fold_metrics, os.path.join(output_dir, "cv_table.csv"))

        # Aggregate
        cv_summary = aggregate_cv_metrics(fold_metrics)
        cv_summary["method"] = self.config.get("experiment", {}).get("method", "unknown")
        cv_summary["seed"] = seed
        cv_summary["num_folds"] = run_folds_val
        cv_summary["mode"] = mode
        cv_summary["fold_thresholds"] = [m.get("best_threshold") for m in fold_metrics]

        with open(os.path.join(output_dir, "cv_summary.json"), "w") as f:
            json.dump(cv_summary, f, indent=2)

        # Save fold histories
        if fold_histories:
            with open(os.path.join(output_dir, "fold_histories.json"), "w") as f:
                json.dump(fold_histories, f, indent=2, default=str)

        # Print summary
        self._print_summary(cv_summary)

        return cv_summary

    # =====================================================================
    # Holdout test: retrain on full train_val → single evaluation
    # =====================================================================

    def _retrain_evaluate_holdout(
        self, dataset, train_val_idx, holdout_idx,
        output_dir, holdout_dir, seed, quick_epochs,
        test_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Retrain on the full train_val set, then evaluate ONCE on holdout_test.
        Uses the same training config as CV folds.

        Args:
            test_threshold: Training-side threshold for the test set (mean of
                the inner-CV fold thresholds). If None, falls back to the
                retrain's own internal-val threshold.
        """
        train_val_idx = train_val_idx if isinstance(train_val_idx, list) else list(train_val_idx)
        holdout_idx = holdout_idx if isinstance(holdout_idx, list) else list(holdout_idx)

        # Rebuild model
        self.method.build_model()
        self.method.prepare_for_training(
            self.device,
            pretrained_path=self.config.get("model", {}).get("pretrained_path"),
        )
        if quick_epochs:
            self.config["training"]["epochs"] = quick_epochs

        # Train on train_val with a simple train/val split from within train_val
        # Use 10% of train_val as internal validation for early stopping
        from sklearn.model_selection import train_test_split
        tr_idx, va_idx = train_test_split(
            train_val_idx, test_size=0.1, random_state=seed + 9999,
            stratify=[dataset[i][1].item() for i in train_val_idx],
        )
        tr_idx = tr_idx.tolist() if not isinstance(tr_idx, list) else list(tr_idx)
        va_idx = va_idx.tolist() if not isinstance(va_idx, list) else list(va_idx)

        print(f"[Holdout] Retrain: train={len(tr_idx)}, internal_val={len(va_idx)}", flush=True)

        # Use a temp fold dir for retraining
        retrain_dir = os.path.join(holdout_dir, "retrain")
        os.makedirs(retrain_dir, exist_ok=True)

        fold_result = self.trainer.train_fold(
            dataset, dataset, 0, retrain_dir,
            train_indices=tr_idx, val_indices=va_idx,
        )

        # Evaluate on locked holdout test using a TRAINING-side threshold
        # (never tuned on the test set). Prefer the mean inner-CV val threshold;
        # else use the retrain's internal-val threshold.
        print(f"[Holdout] Evaluating on locked test set ({len(holdout_idx)} samples)...", flush=True)
        if test_threshold is not None:
            retrain_thr = test_threshold
        else:
            retrain_thr = fold_result.get("best_threshold")
        holdout_metrics = self._eval_on_holdout(dataset, holdout_idx, holdout_dir, threshold=retrain_thr)
        holdout_metrics["threshold_source"] = (
            "mean_inner_cv_val" if test_threshold is not None else "train_val_internal_val"
        )

        # Save holdout results (strip internal arrays)
        _holdout_array_keys = {"_y_true", "_y_prob", "cm"}
        metrics_serializable = {}
        for k, v in holdout_metrics.items():
            if k in _holdout_array_keys:
                continue
            if isinstance(v, torch.Tensor):
                metrics_serializable[k] = v.tolist()
            elif isinstance(v, np.ndarray):
                metrics_serializable[k] = v.tolist()
            elif isinstance(v, (int, float, str, bool, list, dict)) or v is None:
                metrics_serializable[k] = v
            else:
                metrics_serializable[k] = str(v)

        with open(os.path.join(holdout_dir, "metrics.json"), "w") as f:
            json.dump(metrics_serializable, f, indent=2)

        # Confusion matrix
        cm = holdout_metrics.get("cm")
        if cm is not None:
            save_cm_png(cm, ["AD", "LBD"], os.path.join(holdout_dir, "cm.png"))
            if isinstance(cm, torch.Tensor):
                cm_list = cm.tolist()
            else:
                cm_list = cm
            with open(os.path.join(holdout_dir, "cm_raw.json"), "w") as f:
                json.dump(cm_list, f, indent=2)

        # Save predictions CSV (real subject ids + threshold-consistent y_pred)
        ht_y_true = holdout_metrics.get("_y_true")
        ht_y_prob = holdout_metrics.get("_y_prob")
        if ht_y_true is not None and ht_y_prob is not None:
            _hpids = None
            if getattr(dataset, "patient_ids", None):
                _hpids = [dataset.patient_ids[i] for i in holdout_idx]
            save_predictions_csv(
                ht_y_true, ht_y_prob,
                os.path.join(holdout_dir, "predictions.csv"),
                patient_ids=_hpids,
                fold_label="holdout",
                threshold=holdout_metrics.get("best_threshold"),
            )
            save_roc_curve(
                ht_y_true, ht_y_prob,
                os.path.join(holdout_dir, "roc_curve.png"),
                title="Holdout Test ROC",
            )

        # Clean up raw arrays from JSON
        clean = {k: v for k, v in metrics_serializable.items() if not k.startswith("_")}
        print(f"[Holdout] AUC={clean.get('auc', 'N/A'):.4f}, "
              f"BAC={clean.get('bal_acc', 'N/A'):.4f}, "
              f"Sens={clean.get('sens', 'N/A'):.4f}, "
              f"Spec={clean.get('spec', 'N/A'):.4f}", flush=True)
        return clean

    def _eval_on_holdout(
        self, dataset, holdout_indices, holdout_dir,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Evaluate best model on holdout test set.

        Args:
            threshold: Decision threshold for class 1, taken from the TRAINING
                side (retrain's internal val split). If None, falls back to 0.5.
                Never tuned on the test set.
        """
        from torch.utils.data import DataLoader
        from .dataset import make_collate_filter_none

        t_cfg = self.config.get("training", {})
        batch_size = int(t_cfg.get("batch_size", 4))
        num_workers = int(self.config.get("dataset", {}).get("num_workers", 4))
        amp_enabled = self.use_amp() and self.device.type == "cuda"

        holdout_ds = Subset(dataset, holdout_indices)
        loader = DataLoader(
            holdout_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            collate_fn=make_collate_filter_none(),
        )

        self.method.eval()
        all_y_true = []
        all_y_prob = []
        import torch.nn.functional as F

        with torch.no_grad():
            for batch in loader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                with autocast_ctx(enabled=amp_enabled, dtype=torch.bfloat16):
                    logits = self.method.forward_classifier(x)
                probs = F.softmax(logits.to(torch.float32), dim=1).detach().cpu().numpy()
                all_y_true.extend(y.cpu().numpy().tolist())
                all_y_prob.append(probs)

        if not all_y_true:
            return {"auc": float("nan"), "bal_acc": float("nan")}

        all_y_true = np.array(all_y_true, dtype=np.int64)
        all_y_prob = np.concatenate(all_y_prob, axis=0)

        # Evaluate on the FULL holdout with a training-side threshold.
        # (AUC is threshold-independent; BAC/sens/spec use the given threshold.)
        thr = 0.5 if threshold is None else threshold
        metrics = compute_metrics_from_raw(all_y_true, all_y_prob, 2, thr)
        metrics["best_threshold"] = thr
        metrics["threshold_source"] = "train_val_internal_val" if threshold is not None else "default_0.5"
        metrics["_y_true"] = all_y_true
        metrics["_y_prob"] = all_y_prob
        return metrics

    # =====================================================================
    # Helpers
    # =====================================================================

    def use_amp(self) -> bool:
        return bool(self.config.get("training", {}).get("use_amp", True))

    def _collect_labels(self, dataset) -> np.ndarray:
        all_labels = []
        for i in range(len(dataset)):
            _, y = dataset[i]
            if y is not None:
                all_labels.append(y.item() if isinstance(y, torch.Tensor) else y)
        return np.array(all_labels)

    def _kfold_split(self, labels, n_folds, seed):
        if self.stratified:
            from sklearn.model_selection import StratifiedKFold
            splitter = StratifiedKFold(n_splits=n_folds, shuffle=self.shuffle, random_state=seed)
        else:
            from sklearn.model_selection import KFold
            splitter = KFold(n_splits=n_folds, shuffle=self.shuffle, random_state=seed)
        return list(splitter.split(np.arange(len(labels)), labels))

    def _save_fold_train_log(self, fold_result, fold_dir):
        hist = fold_result.get("history", {})
        if not hist or "train_loss" not in hist:
            return
        log_path = os.path.join(fold_dir, "train_log.csv")
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            keys = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc",
                    "train_auc", "val_auc"]
            writer.writerow(keys)
            n_epochs = len(hist.get("train_loss", []))
            for e in range(n_epochs):
                writer.writerow([
                    e + 1,
                    hist["train_loss"][e] if e < len(hist.get("train_loss", [])) else "",
                    hist["val_loss"][e] if e < len(hist.get("val_loss", [])) else "",
                    hist["train_acc"][e] if e < len(hist.get("train_acc", [])) else "",
                    hist["val_acc"][e] if e < len(hist.get("val_acc", [])) else "",
                    hist["train_auc"][e] if e < len(hist.get("train_auc", [])) else "",
                    hist["val_auc"][e] if e < len(hist.get("val_auc", [])) else "",
                ])

    def _append_combined_log(self, writer, fold_result, fold_idx):
        hist = fold_result.get("history", {})
        if not hist or "train_loss" not in hist:
            return
        n_epochs = len(hist.get("train_loss", []))
        for e in range(n_epochs):
            writer.writerow([
                fold_idx, e + 1,
                hist["train_loss"][e] if e < len(hist.get("train_loss", [])) else "",
                hist["val_loss"][e] if e < len(hist.get("val_loss", [])) else "",
                hist["train_acc"][e] if e < len(hist.get("train_acc", [])) else "",
                hist["val_acc"][e] if e < len(hist.get("val_acc", [])) else "",
                hist.get("train_auc", [None] * n_epochs)[e] if e < len(hist.get("train_auc", [])) else "",
                hist["val_auc"][e] if e < len(hist.get("val_auc", [])) else "",
                hist.get("val_bal_acc", [None] * n_epochs)[e] if e < len(hist.get("val_bal_acc", [])) else "",
                hist.get("val_sens", [None] * n_epochs)[e] if e < len(hist.get("val_sens", [])) else "",
                hist.get("val_spec", [None] * n_epochs)[e] if e < len(hist.get("val_spec", [])) else "",
            ])

    def _print_summary(self, cv_summary):
        print(f"\n[CV Summary] {cv_summary['method']} (seed={cv_summary['seed']}, "
              f"mode={cv_summary.get('mode', '?')}):")
        for metric in ["auc", "bal_acc", "acc", "sens", "spec", "f1"]:
            m = cv_summary.get(metric, {})
            print(f"  {metric}: {m.get('mean', float('nan')):.4f} ± {m.get('std', float('nan')):.4f}")
        if "holdout_test" in cv_summary:
            ht = cv_summary["holdout_test"]
            print(f"  [Holdout Test] AUC={ht.get('auc', 'N/A'):.4f}, "
                  f"BAC={ht.get('bal_acc', 'N/A'):.4f}")
