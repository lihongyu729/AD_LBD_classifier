"""
Dual-mode trainer: standard supervised and meta-learning (episodic).

Standard mode: batch → forward → loss → backward
Meta mode: episodic tasks → strategy.core_step(support, query) → outer_loss → backward

Adapted from D:\py_project\MRI\code\meta\train_classifier.py (train_meta function).
"""
import os
import sys
import csv
import time
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, Subset
from typing import Dict, Any, Optional, List, Tuple, Callable

from .metrics import (
    compute_metrics_from_raw,
    evaluate_loader,
    find_best_threshold_bal_acc,
    split_calib_eval_indices,
    save_metrics_curves,
    save_cm_png,
)
from .dataset import (
    augment_3d_batch,
    EpisodicBatchSampler,
    split_task_data,
    make_collate_filter_none,
)
from .utils import autocast_ctx


class Trainer:
    """
    Unified trainer supporting both standard supervised and meta-learning.

    Args:
        method: BaseMethod instance (model + strategy).
        config: Full config dict.
        device: torch.device.

    Usage:
        trainer = Trainer(method, config, device)
        fold_results = trainer.train_fold(train_loader, val_loader, val_dataset, fold_idx, out_dir)
    """

    def __init__(self, method, config: Dict[str, Any], device: torch.device):
        self.method = method
        self.config = config
        self.device = device

        # Training params
        t_cfg = config.get("training", {})
        self.epochs = int(t_cfg.get("epochs", 150))
        self.base_lr = float(t_cfg.get("lr", 5e-5))
        self.weight_decay = float(t_cfg.get("weight_decay", 0.008))
        self.backbone_lr_ratio = float(t_cfg.get("backbone_lr_ratio", 0.05))
        self.warmup_epochs = int(t_cfg.get("warmup_epochs", 10))
        self.lr_scheduler_type = str(t_cfg.get("lr_scheduler", "cosine"))
        self.label_smoothing = float(t_cfg.get("label_smoothing", 0.05))
        self.ema_decay = float(t_cfg.get("ema_decay", 0.99))
        self.max_grad_norm = float(t_cfg.get("max_grad_norm", 0.7))
        self.grad_accum_steps = int(t_cfg.get("grad_accum_steps", 1))
        self.use_amp = bool(t_cfg.get("use_amp", True))
        self.amp_dtype = str(t_cfg.get("amp_dtype", "bf16"))

        # Early stopping (aligned with paper §4.2: patience=28 on val_auc)
        es_cfg = config.get("early_stopping", {})
        self.es_patience = int(es_cfg.get("patience", 28))
        self.es_monitor = str(es_cfg.get("monitor", "val_auc"))
        self.es_min_epochs = int(es_cfg.get("min_epochs", 25))
        self.es_min_lbd_sens = float(es_cfg.get("min_lbd_sens", 0.50))
        self.es_min_lbd_spec = float(es_cfg.get("min_lbd_spec", 0.25))
        self.es_gate_blocking = bool(es_cfg.get("gate_blocking", False))

        # Class-imbalance handling for STANDARD training
        #   training.class_weights: [w_AD, w_LBD] -> weighted CE/Focal loss
        #   training.resampling: true -> WeightedRandomSampler (oversample minority)
        cw = t_cfg.get("class_weights")
        if isinstance(cw, str):  # --set passes strings; parse [1.0,8.0]
            try:
                cw = yaml.safe_load(cw)
            except Exception:
                cw = None
        if cw is not None and len(cw) > 0:
            self.class_weights = torch.tensor([float(x) for x in cw], dtype=torch.float32)
        else:
            self.class_weights = None
        rv = t_cfg.get("resampling", False)
        if isinstance(rv, str):
            rv = rv.lower() in ("true", "1", "yes")
        self.use_resampling = bool(rv)

        # Augmentation
        aug_cfg = config.get("augmentation", {})
        self.do_augment = aug_cfg.get("enable", True)
        self.augment_params = aug_cfg

        # Thresholding
        th_cfg = config.get("thresholding", {})
        self.calib_fraction = float(th_cfg.get("calib_fraction", 0.4))
        self.calib_seed = int(th_cfg.get("split_seed", 42))

        # Meta-learning
        meta_cfg = config.get("meta", {})
        self.n_way = int(meta_cfg.get("n_way", 2))
        self.k_shot = int(meta_cfg.get("k_shot", 4))
        self.q_query = int(meta_cfg.get("q_query", 4))
        self.train_episodes = int(meta_cfg.get("train_episodes", 80))
        self.val_episodes = int(meta_cfg.get("val_episodes", 120))

        # Misc
        self.use_compile = bool(config.get("classifier", {}).get("use_compile", False))

    # =====================================================================
    # Public API
    # =====================================================================

    def train_fold(
        self,
        train_dataset,
        val_dataset,
        fold_idx: int,
        output_dir: str,
        train_indices: Optional[List[int]] = None,
        val_indices: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Train one CV fold.

        Args:
            train_dataset: Full training dataset (will be subset if indices given).
            val_dataset: Full validation dataset.
            fold_idx: 0-based fold index.
            output_dir: Directory to save results.
            train_indices: Subset indices for training (or None).
            val_indices: Subset indices for validation (or None).

        Returns:
            Dict with best metrics for this fold.
        """
        strategy_config = self.method.get_strategy_config()
        is_meta = strategy_config is not None

        if is_meta:
            return self._train_meta(
                train_dataset, val_dataset, fold_idx, output_dir,
                train_indices, val_indices, strategy_config,
            )
        else:
            return self._train_standard(
                train_dataset, val_dataset, fold_idx, output_dir,
                train_indices, val_indices,
            )

    # =====================================================================
    # Standard training
    # =====================================================================

    def _train_standard(
        self,
        train_dataset,
        val_dataset,
        fold_idx: int,
        output_dir: str,
        train_indices: Optional[List[int]],
        val_indices: Optional[List[int]],
    ) -> Dict[str, Any]:
        """Standard supervised training loop."""
        t_cfg = self.config.get("training", {})
        batch_size = int(t_cfg.get("batch_size", 4))
        num_workers = int(self.config.get("dataset", {}).get("num_workers", 4))

        # Subset datasets
        train_ds = Subset(train_dataset, train_indices) if train_indices else train_dataset
        val_ds = Subset(val_dataset, val_indices) if val_indices else val_dataset

        # Class-balanced resampling (oversample minority class) if enabled
        train_sampler = None
        if self.use_resampling:
            from torch.utils.data import WeightedRandomSampler
            _idx = list(train_indices) if train_indices is not None else list(range(len(train_dataset)))
            _lbls = [int(train_dataset[i][1]) for i in _idx]
            _counts = torch.bincount(torch.tensor(_lbls, dtype=torch.long), minlength=2).float()
            _counts = _counts.clamp(min=1.0)
            _weights = torch.tensor([1.0 / _counts[l].item() for l in _lbls], dtype=torch.float32)
            train_sampler = WeightedRandomSampler(_weights, num_samples=len(train_ds), replacement=True)
            print(f"[Trainer] Class-balanced resampling ON "
                  f"(n={len(train_ds)}, counts={_counts.tolist()})", flush=True)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers, pin_memory=True,
            collate_fn=make_collate_filter_none(), drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            collate_fn=make_collate_filter_none(),
        )

        # Build optimizer
        optimizer = self._build_optimizer()

        # Scheduler
        total_steps = len(train_loader) * self.epochs
        warmup_steps = len(train_loader) * self.warmup_epochs
        scheduler = self._build_scheduler(optimizer, total_steps, warmup_steps)

        # EMA
        ema_model = None
        if self.ema_decay > 0:
            ema_model = copy.deepcopy(self.method.model)
            ema_model.eval()

        # AMP
        amp_enabled = self.use_amp and self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if self.amp_dtype == "bf16" else torch.float16

        # CE + Focal loss (optionally class-weighted for imbalance)
        ce_loss_fn = nn.CrossEntropyLoss(
            label_smoothing=self.label_smoothing,
            weight=self.class_weights.to(self.device) if self.class_weights is not None else None,
        )
        focal_loss_fn = FocalLoss(gamma=1.0, weight=self.class_weights)

        # History
        history = {
            "train_loss": [], "train_acc": [], "train_auc": [],
            "val_loss": [], "val_acc": [], "val_auc": [],
            "val_bal_acc": [], "val_sens": [], "val_spec": [],
        }

        best_monitor = -float("inf")
        best_epoch = 0
        best_state = None
        best_val_metrics = None
        no_improve = 0

        for epoch in range(self.epochs):
            # --- Train ---
            self.method.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for step, batch in enumerate(train_loader):
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                if self.do_augment:
                    x = augment_3d_batch(x, self.augment_params)

                with autocast_ctx(enabled=amp_enabled, dtype=amp_dtype):
                    logits = self.method.forward_classifier(x)
                    ce = ce_loss_fn(logits, y)
                    focal = focal_loss_fn(logits, y)
                    loss = 0.8 * ce + 0.2 * focal

                loss = loss / self.grad_accum_steps
                loss.backward()

                if (step + 1) % self.grad_accum_steps == 0:
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.method.parameters(), self.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None and self.lr_scheduler_type == "cosine":
                        scheduler.step()
                    if ema_model is not None:
                        self._update_ema(ema_model, self.method.model)

                epoch_loss += loss.item() * self.grad_accum_steps
                preds = logits.argmax(dim=1)
                epoch_correct += (preds == y).sum().item()
                epoch_total += y.size(0)

            avg_train_loss = epoch_loss / max(len(train_loader), 1)
            avg_train_acc = epoch_correct / max(epoch_total, 1)

            # --- Validate ---
            val_metrics = self._collect_val_metrics(val_loader)
            val_loss = self._compute_val_loss(val_loader, ce_loss_fn, focal_loss_fn, amp_enabled, amp_dtype)

            history["train_loss"].append(avg_train_loss)
            history["train_acc"].append(avg_train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_metrics.get("acc", 0.0))
            history["val_auc"].append(val_metrics.get("auc", float("nan")))
            history["val_bal_acc"].append(val_metrics.get("bal_acc", float("nan")))
            history["val_sens"].append(val_metrics.get("sens", float("nan")))
            history["val_spec"].append(val_metrics.get("spec", float("nan")))

            # Store raw predictions from best epoch for CSV export
            val_y_true = val_metrics.get("_y_true")
            val_y_prob = val_metrics.get("_y_prob")
            monitor_val = val_metrics.get("auc", 0.0)
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(
                    f"[Fold {fold_idx + 1}] Epoch {epoch + 1}/{self.epochs} | "
                    f"Loss: {avg_train_loss:.4f}/{val_loss:.4f} | "
                    f"Acc: {avg_train_acc:.3f}/{val_metrics.get('acc', 0):.3f} | "
                    f"AUC: {val_metrics.get('auc', float('nan')):.4f} | "
                    f"BAC: {val_metrics.get('bal_acc', 0):.4f}",
                    flush=True,
                )

            # Early stopping with sens/spec quality gates (aligned with paper §4.2)
            if epoch >= self.es_min_epochs - 1:
                gated = self._check_quality_gates(val_metrics, epoch, fold_idx)
                improved = (not gated and not math.isnan(monitor_val) and
                          monitor_val > best_monitor + 1e-6)
            else:
                improved = False

            if improved:
                best_monitor = monitor_val
                best_epoch = epoch + 1
                best_state = copy.deepcopy(self.method.model.state_dict())
                best_val_metrics = val_metrics
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.es_patience and epoch >= self.es_min_epochs:
                print(f"[Fold {fold_idx + 1}] Early stopping at epoch {epoch + 1}", flush=True)
                break

            # Step scheduler (non-cosine)
            if scheduler is not None and self.lr_scheduler_type != "cosine":
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(monitor_val)
                else:
                    scheduler.step()

        # Save best model
        if best_state is not None:
            self.method.model.load_state_dict(best_state)

        # output_dir is already the fold-specific dir (evaluator passes it);
        # save into it directly to avoid double fold_/ nesting.
        fold_dir = output_dir
        os.makedirs(fold_dir, exist_ok=True)

        # Fallback for short runs (epochs < min_epochs): use last val if no best
        _fallback = val_metrics if (best_val_metrics is None and val_metrics is not None) else None
        _save_metrics = _fallback or best_val_metrics or {}

        from .metrics import save_fold_results
        save_fold_results(fold_dir, _save_metrics, model_state=best_state)

        # Save curves
        save_metrics_curves(fold_dir, fold_idx, history)

        bvm = (_fallback or best_val_metrics) or {}
        print(f"[Fold {fold_idx + 1}] Best epoch={best_epoch}, "
              f"AUC={bvm.get('auc', float('nan')):.4f}, "
              f"BAC={bvm.get('bal_acc', 0):.4f}",
              flush=True)

        return {
            **{
                "best_epoch": best_epoch,
                "history": history,
            },
            **bvm,
        }

    def _compute_metrics_with_threshold(self, yt: np.ndarray, yp: np.ndarray) -> Dict[str, Any]:
        """Tune a threshold on a calibration split (if possible) and compute metrics.

        Returns a result dict that additionally carries ``best_threshold``
        (falls back to 0.5 when the calibration split is unavailable) plus the
        raw ``_y_true`` / ``_y_prob`` arrays.
        """
        calib_idx, eval_idx = split_calib_eval_indices(yt, self.calib_fraction, self.calib_seed)
        if calib_idx is not None:
            best_thr, _, _ = find_best_threshold_bal_acc(yt[calib_idx], yp[calib_idx, 1])
            result = compute_metrics_from_raw(yt[eval_idx], yp[eval_idx], 2, best_thr)
        else:
            best_thr = 0.5
            result = compute_metrics_from_raw(yt, yp, 2, 0.5)
        result["_y_true"] = yt
        result["_y_prob"] = yp
        result["best_threshold"] = best_thr
        return result

    def _collect_val_metrics(self, val_loader) -> Dict[str, Any]:
        """Collect all predictions from val loader and compute metrics.
        Also returns raw y_true/y_prob arrays for CSV export."""
        all_y_true = []
        all_y_prob = []
        self.method.eval()
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x = x.to(self.device)
                with autocast_ctx(enabled=self.use_amp,
                                        dtype=torch.bfloat16 if self.amp_dtype == "bf16" else torch.float16):
                    logits = self.method.forward_classifier(x)
                probs = F.softmax(logits.to(torch.float32), dim=1).detach().cpu().numpy()
                all_y_true.extend(y.cpu().numpy().tolist())
                all_y_prob.append(probs)
        if not all_y_true:
            return {"_y_true": np.array([]), "_y_prob": np.array([])}
        all_y_true_np = np.array(all_y_true, dtype=np.int64)
        all_y_prob_np = np.concatenate(all_y_prob, axis=0)
        return self._compute_metrics_with_threshold(all_y_true_np, all_y_prob_np)

    def _compute_val_loss(self, val_loader, ce_fn, focal_fn, amp_enabled, amp_dtype):
        """Compute average validation loss."""
        self.method.eval()
        total_loss = 0.0
        count = 0
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                with autocast_ctx(enabled=amp_enabled, dtype=amp_dtype):
                    logits = self.method.forward_classifier(x)
                    loss = 0.8 * ce_fn(logits, y) + 0.2 * focal_fn(logits, y)
                total_loss += loss.item()
                count += 1
        return total_loss / max(count, 1)

    # =====================================================================
    # Meta-learning training
    # =====================================================================

    def _train_meta(
        self,
        train_dataset,
        val_dataset,
        fold_idx: int,
        output_dir: str,
        train_indices: Optional[List[int]],
        val_indices: Optional[List[int]],
        strategy_config: Dict,
    ) -> Dict[str, Any]:
        """Meta-learning episodic training loop."""
        t_cfg = self.config.get("training", {})
        batch_size = int(t_cfg.get("batch_size", self.n_way * (self.k_shot + self.q_query)))
        num_workers = int(self.config.get("dataset", {}).get("num_workers", 4))

        # Subset
        train_ds = Subset(train_dataset, train_indices) if train_indices else train_dataset
        val_ds = Subset(val_dataset, val_indices) if val_indices else val_dataset

        # Collect labels for episodic sampler
        train_labels = [train_dataset[i][1].item() if isinstance(train_dataset[i], tuple) else train_dataset[i][1]
                       for i in (train_indices or range(len(train_dataset)))]
        val_labels = [val_dataset[i][1].item() if isinstance(val_dataset[i], tuple) else val_dataset[i][1]
                     for i in (val_indices or range(len(val_dataset)))]

        # Guard: training episodes must fit the smallest class in this fold.
        # (Inner-CV train folds keep >=24 LBD at 4-shot/4-query, so this never
        # fires in the current protocol; it is a clear diagnostic if it does.)
        _min_tr = int(np.unique(train_labels, return_counts=True)[1].min())
        if _min_tr < self.k_shot + self.q_query:
            raise RuntimeError(
                f"Meta train fold: minority class has only {_min_tr} samples, "
                f"needs {self.k_shot + self.q_query} for {self.k_shot}-shot/"
                f"{self.q_query}-query episodes. Reduce meta.k_shot / meta.q_query "
                f"or re-balance the split."
            )

        # Adaptive VALIDATION episode size. Inner-CV val folds hold only ~6 LBD
        # and the holdout retrain's internal val only ~3 — both below the
        # config's k_shot+q_query=8, so the old code raised ValueError there.
        # Episodic val metrics are diagnostic only (early stopping selects on
        # the raw full-batch val AUC), so clamp the val episode to the smallest
        # class available instead of failing the run.
        _min_val = int(np.unique(val_labels, return_counts=True)[1].min())
        if _min_val >= 2:
            val_k = min(self.k_shot, _min_val // 2)
            val_q = min(self.q_query, _min_val - val_k)
        else:
            val_k, val_q = 0, 0
            print(f"[Trainer][WARN] val fold has a class with only {_min_val} "
                  f"sample(s) — episodic validation disabled (reports nan).", flush=True)

        # Episodic samplers
        train_sampler = EpisodicBatchSampler(train_labels, self.n_way, self.k_shot, self.q_query, self.train_episodes)
        train_loader = DataLoader(train_ds, batch_sampler=train_sampler,
                                  num_workers=num_workers, pin_memory=True,
                                  collate_fn=make_collate_filter_none())

        val_ep_loader = None
        if val_k and val_q:
            val_sampler_ep = EpisodicBatchSampler(val_labels, self.n_way, val_k, val_q, self.val_episodes)
            val_ep_loader = DataLoader(val_ds, batch_sampler=val_sampler_ep,
                                       num_workers=num_workers, pin_memory=True,
                                       collate_fn=make_collate_filter_none())

        # Raw val loader for threshold tuning
        val_raw_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                                    num_workers=num_workers, pin_memory=True,
                                    collate_fn=make_collate_filter_none())

        # Get strategy class from config
        strategy_name = strategy_config.get("name", "anil")
        strategy_params = strategy_config.get("params", {})

        # Import strategy dynamically
        strategy = self._build_strategy(strategy_name, strategy_params)

        # Build optimizer
        optimizer = self._build_optimizer()

        # Scheduler
        total_steps = self.train_episodes * self.epochs
        warmup_steps = self.train_episodes * self.warmup_epochs
        scheduler = self._build_scheduler(optimizer, total_steps, warmup_steps) if self.lr_scheduler_type == "cosine" else None

        # EMA
        ema_model = None
        if self.ema_decay > 0:
            ema_model = copy.deepcopy(self.method.model)
            ema_model.eval()

        amp_enabled = self.use_amp and self.device.type == "cuda"
        amp_dtype = torch.bfloat16 if self.amp_dtype == "bf16" else torch.float16

        head_module = getattr(self.method.model, "head", getattr(self.method.model, "classifier", None))
        inner_weight_decay = float(strategy_params.get("inner_weight_decay", 1e-4))
        inner_optimizer = torch.optim.SGD(
            head_module.parameters() if head_module else self.method.model.parameters(),
            lr=strategy_params.get("inner_lr", 0.01),
            weight_decay=inner_weight_decay,
        )

        history = {
            "train_loss": [], "train_acc": [], "train_auc": [],
            "val_loss": [], "val_acc": [], "val_auc": [],
            "val_bal_acc": [], "val_sens": [], "val_spec": [],
        }

        best_monitor = -float("inf")
        best_epoch = 0
        best_state = None
        best_val_metrics = None
        no_improve = 0

        for epoch in range(self.epochs):
            # --- Meta-train ---
            self.method.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            n_batches = 0

            for batch in train_loader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)

                if self.do_augment:
                    x = augment_3d_batch(x, self.augment_params)

                # Split into support/query
                try:
                    s_x, s_y, q_x, q_y = split_task_data(x, y, self.n_way, self.k_shot, self.q_query)
                except ValueError:
                    continue

                with autocast_ctx(enabled=amp_enabled, dtype=amp_dtype):
                    result = strategy.core_step(
                        s_x, s_y, q_x, q_y,
                        optimizer=optimizer,
                        inner_optimizer=inner_optimizer,
                        is_training=True,
                    )

                loss = result["loss"] / self.grad_accum_steps
                loss.backward()

                if (n_batches + 1) % self.grad_accum_steps == 0:
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.method.parameters(), self.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
                    if ema_model is not None:
                        self._update_ema(ema_model, self.method.model)

                epoch_loss += result["loss"].item()
                logits_q = result.get("logits")
                if logits_q is not None and logits_q.dim() >= 2:
                    preds = logits_q.argmax(dim=1)
                    q_y_dev = q_y.to(preds.device)
                    epoch_correct += (preds == q_y_dev).sum().item()
                    epoch_total += q_y.size(0)
                n_batches += 1

            avg_train_loss = epoch_loss / max(n_batches, 1)
            avg_train_acc = epoch_correct / max(epoch_total, 1)

            # --- Meta-val (episodic, diagnostic only) ---
            if val_ep_loader is not None:
                val_loss, val_acc = self._validate_episodic(
                    val_ep_loader, strategy, amp_enabled, amp_dtype, val_k, val_q)
            else:
                val_loss, val_acc = float("nan"), float("nan")

            # --- Raw val (for AUC/early stopping) ---
            # forward_fn must be (model, x) → logits; use a lambda to avoid bound-method arg mismatch
            val_raw_metrics = evaluate_loader(
                self.method.model, val_raw_loader, self.device,
                num_classes=2, forward_fn=lambda _m, x: self.method.forward_classifier(x),
            )
            # Thread a training-side tuned threshold (consistent with the
            # standard path) so meta results also carry best_threshold.
            if val_raw_metrics.get("_y_true") is not None and len(val_raw_metrics.get("_y_true", [])) > 0:
                val_raw_metrics = self._compute_metrics_with_threshold(
                    val_raw_metrics["_y_true"], val_raw_metrics["_y_prob"],
                )

            history["train_loss"].append(avg_train_loss)
            history["train_acc"].append(avg_train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["val_auc"].append(val_raw_metrics.get("auc", float("nan")))
            history["val_bal_acc"].append(val_raw_metrics.get("bal_acc", float("nan")))
            history["val_sens"].append(val_raw_metrics.get("sens", float("nan")))
            history["val_spec"].append(val_raw_metrics.get("spec", float("nan")))

            monitor_val = val_raw_metrics.get("auc", 0.0)
            if epoch % 10 == 0 or epoch == self.epochs - 1:
                print(
                    f"[Fold {fold_idx + 1}] Epoch {epoch + 1}/{self.epochs} | "
                    f"Loss: {avg_train_loss:.4f}/{val_loss:.4f} | "
                    f"Acc: {avg_train_acc:.3f}/{val_acc:.3f} | "
                    f"AUC(raw): {val_raw_metrics.get('auc', float('nan')):.4f} | "
                    f"BAC: {val_raw_metrics.get('bal_acc', 0):.4f}",
                    flush=True,
                )

            # Early stopping with sens/spec quality gates (aligned with paper §4.2)
            if epoch >= self.es_min_epochs - 1:
                gated = self._check_quality_gates(val_raw_metrics, epoch, fold_idx)
                improved = (not gated and not math.isnan(monitor_val) and
                          monitor_val > best_monitor + 1e-6)
            else:
                improved = False

            if improved:
                best_monitor = monitor_val
                best_epoch = epoch + 1
                best_state = copy.deepcopy(self.method.model.state_dict())
                best_val_metrics = val_raw_metrics
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.es_patience and epoch >= self.es_min_epochs:
                print(f"[Fold {fold_idx + 1}] Early stopping at epoch {epoch + 1}", flush=True)
                break

        # Save best
        if best_state is not None:
            self.method.model.load_state_dict(best_state)

        # Fallback for short runs
        _fallback_m = val_raw_metrics if (best_val_metrics is None and val_raw_metrics is not None) else None
        _save_metrics_m = _fallback_m or best_val_metrics or {}

        # output_dir is already the fold-specific dir (evaluator passes it)
        from .metrics import save_fold_results
        save_fold_results(output_dir, _save_metrics_m, model_state=best_state)
        save_metrics_curves(output_dir, fold_idx, history)

        bvm = (_fallback_m or best_val_metrics) or {}
        print(f"[Fold {fold_idx + 1}] Best epoch={best_epoch}, "
              f"AUC={bvm.get('auc', float('nan')):.4f}, "
              f"BAC={bvm.get('bal_acc', 0):.4f}",
              flush=True)

        return {
            **{
                "best_epoch": best_epoch,
                "history": history,
            },
            **bvm,
        }

    def _validate_episodic(self, val_loader, strategy, amp_enabled, amp_dtype,
                           k_shot: Optional[int] = None, q_query: Optional[int] = None) -> Tuple[float, float]:
        """Validate on episodic tasks (diagnostic loss/acc; not used for early stopping)."""
        if k_shot is None:
            k_shot = self.k_shot
        if q_query is None:
            q_query = self.q_query
        self.method.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        n = 0

        with torch.no_grad():
            for batch in val_loader:
                x, y = batch
                x, y = x.to(self.device), y.to(self.device)
                try:
                    s_x, s_y, q_x, q_y = split_task_data(x, y, self.n_way, k_shot, q_query)
                except ValueError:
                    continue

                with autocast_ctx(enabled=amp_enabled, dtype=amp_dtype):
                    result = strategy.core_step(s_x, s_y, q_x, q_y, is_training=False)

                total_loss += result["loss"].item()
                logits = result.get("logits")
                if logits is not None and logits.dim() >= 2:
                    preds = logits.argmax(dim=1)
                    q_y_dev = q_y.to(preds.device)
                    total_correct += (preds == q_y_dev).sum().item()
                    total_samples += q_y.size(0)
                n += 1

        return total_loss / max(n, 1), total_correct / max(total_samples, 1)

    # =====================================================================
    # Helpers
    # =====================================================================

    def _check_quality_gates(self, metrics: Dict, epoch: int, fold_idx: int) -> bool:
        """
        Check sens/spec quality gates from paper.
        Returns True if model is "gated" (should NOT be selected as best),
        i.e., sensitivity or specificity is below the minimum threshold.

        When gate_blocking=True, returns True to prevent selecting a low-quality model.
        When gate_blocking=False (default), only logs a warning.
        """
        sens = metrics.get("sens", 1.0)
        spec = metrics.get("spec", 1.0)
        sens_ok = (not math.isnan(sens)) and sens >= self.es_min_lbd_sens
        spec_ok = (not math.isnan(spec)) and spec >= self.es_min_lbd_spec

        if not sens_ok or not spec_ok:
            flag = "BLOCKED" if self.es_gate_blocking else "WARN"
            print(
                f"[Fold {fold_idx + 1}][{flag}] Epoch {epoch + 1}: "
                f"Sens={sens:.4f}(min={self.es_min_lbd_sens}) "
                f"Spec={spec:.4f}(min={self.es_min_lbd_spec}) "
                f"— quality gate {'not met' if self.es_gate_blocking else 'warning only'}",
                flush=True,
            )
            return self.es_gate_blocking
        return False

    # =====================================================================
    # Optimizer, Scheduler, EMA helpers
    # =====================================================================

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build AdamW optimizer with differential learning rates."""
        param_groups = self.method.get_optimizer_param_groups()

        # Apply learning rates
        for group in param_groups:
            if "lr_ratio" in group:
                group["lr"] = self.base_lr * group.pop("lr_ratio")
            else:
                group.setdefault("lr", self.base_lr)
            group.setdefault("weight_decay", self.weight_decay)

        return torch.optim.AdamW(param_groups, lr=self.base_lr, weight_decay=self.weight_decay)

    def _build_scheduler(self, optimizer, total_steps, warmup_steps):
        """Build learning rate scheduler."""
        if self.lr_scheduler_type == "cosine":
            def lr_lambda(step):
                if step < warmup_steps:
                    return float(step) / max(warmup_steps, 1)
                progress = float(step - warmup_steps) / max(total_steps - warmup_steps, 1)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        elif self.lr_scheduler_type == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.5, patience=8,
            )
        return None

    def _update_ema(self, ema_model, model):
        """Update exponential moving average of model parameters.
        Gracefully handles shape mismatches (model rebuild between folds)."""
        try:
            with torch.no_grad():
                for ema_p, p in zip(ema_model.parameters(), model.parameters()):
                    if ema_p.shape != p.shape:
                        continue  # skip mismatched layers
                    ema_p.data.mul_(self.ema_decay).add_(p.data, alpha=1 - self.ema_decay)
        except Exception:
            pass  # EMA is optional; skip if parameter enumeration fails

    def _build_strategy(self, name: str, params: Dict):
        """Build meta-learning strategy instance from meta/strategies/.

        We add meta/ to sys.path so ``from strategies.xxx import ...`` works
        as a package import (avoids relative-import failures)."""
        _trainer_dir = os.path.dirname(os.path.abspath(__file__))
        _meta_dir = os.path.normpath(os.path.join(
            _trainer_dir, "..", "..", "meta"
        ))
        if _meta_dir not in sys.path:
            sys.path.insert(0, _meta_dir)

        if name == "anil":
            from strategies.anil import ANILStrategy
            return ANILStrategy(self.method.model, {"name": "anil", "params": params})
        elif name == "protonet":
            from strategies.protonet import ProtoNetStrategy
            return ProtoNetStrategy(self.method.model, {"name": "protonet", "params": params})
        elif name == "maml":
            from strategies.maml import MAMLStrategy
            return MAMLStrategy(self.method.model, {"name": "maml", "params": params})
        elif name == "hybrid":
            from strategies.hybrid import HybridStrategy
            return HybridStrategy(self.method.model, {
                "name": "hybrid",
                "params": params,
                "optimizer_strategy": {"name": "maml", "params": params},
                "metric_strategy": {"name": "protonet", "params": params},
            })
        elif name == "vanilla":
            from strategies.vanilla import VanillaStrategy
            return VanillaStrategy(self.method.model, {"name": "vanilla", "params": params})
        else:
            raise ValueError(f"Unknown meta strategy: {name}")


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits, target):
        logpt = F.log_softmax(logits, dim=1)
        pt = logpt.exp()
        logpt_t = logpt.gather(1, target.view(-1, 1)).squeeze(1)
        pt_t = pt.gather(1, target.view(-1, 1)).squeeze(1)
        eps = 1e-7
        p_diff = (1.0 - pt_t).clamp(min=eps)
        if self.weight is not None:
            alpha_t = self.weight.to(logits.device).gather(0, target)
            loss = -alpha_t * (p_diff ** self.gamma) * logpt_t
        else:
            loss = -(p_diff ** self.gamma) * logpt_t
        return loss.mean()
