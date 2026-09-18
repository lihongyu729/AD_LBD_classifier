"""
Unified metrics computation for AD vs LBD binary classification.

Provides: confusion matrix, accuracy, balanced accuracy, AUC, sensitivity,
specificity, F1, threshold tuning, and visualization helpers.
"""
import os
import json
import numpy as np
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_confusion_and_metrics(
    y_true: List[int],
    y_pred: List[int],
    num_classes: int,
) -> Tuple[torch.Tensor, float, float]:
    """Compute confusion matrix, accuracy, and balanced accuracy."""
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    per_class_recall = []
    for c in range(num_classes):
        tp = cm[c, c].item()
        fn = int(cm[c, :].sum().item()) - tp
        recall = tp / max(tp + fn, 1)
        per_class_recall.append(recall)
    bal_acc = float(sum(per_class_recall) / max(num_classes, 1))
    acc = float(sum(int(t == p) for t, p in zip(y_true, y_pred)) / max(len(y_true), 1))
    return cm, acc, bal_acc


def auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Binary ROC-AUC using Mann-Whitney U statistic (no sklearn dependency)."""
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    x = np.concatenate([neg, pos])
    order = np.argsort(x)
    rank = np.zeros_like(order, dtype=np.float64)
    for i, o in enumerate(order):
        rank[o] = i + 1
    pos_ranks = rank[len(neg):]
    n_pos = pos.size
    n_neg = neg.size
    auc = (pos_ranks.sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return float(auc)


def compute_sensitivity_specificity(cm: torch.Tensor) -> Tuple[float, float]:
    """Compute sensitivity (TPR) and specificity (TNR) from confusion matrix."""
    tn = cm[0, 0].item()
    fp = cm[0, 1].item()
    fn = cm[1, 0].item()
    tp = cm[1, 1].item()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return sens, spec


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute macro F1 score."""
    cm = torch.zeros((2, 2), dtype=torch.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    f1s = []
    for c in range(2):
        tp = cm[c, c].item()
        fp = cm[:, c].sum().item() - tp
        fn = cm[c, :].sum().item() - tp
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1s.append(2 * prec * rec / max(prec + rec, 1e-8))
    return float(np.mean(f1s))


# ---------------------------------------------------------------------------
# High-level evaluation
# ---------------------------------------------------------------------------

def compute_metrics_from_raw(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute all metrics from pre-collected y_true and y_prob arrays.

    Args:
        y_true: [N] integer labels
        y_prob: [N, C] predicted probabilities
        num_classes: Number of classes
        threshold: Decision threshold for class 1 (binary only)

    Returns:
        Dict with keys: cm, acc, bal_acc, auc, sens, spec, f1
    """
    if y_true.size == 0:
        cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        return {"cm": cm, "acc": 0.0, "bal_acc": 0.0, "auc": float("nan"),
                "sens": float("nan"), "spec": float("nan"), "f1": float("nan")}

    if num_classes == 2 and threshold is not None:
        y_pred = (y_prob[:, 1] >= threshold).astype(np.int64)
    else:
        y_pred = y_prob.argmax(axis=1).astype(np.int64)

    cm, acc, bal_acc = compute_confusion_and_metrics(list(y_true), list(y_pred), num_classes)
    f1 = compute_f1(y_true, y_pred)

    if num_classes == 2:
        auc = auc_binary(y_true.astype(np.int64), y_prob[:, 1])
        sens, spec = compute_sensitivity_specificity(cm)
    else:
        auc = float("nan")
        sens, spec = float("nan"), float("nan")

    return {
        "cm": cm,
        "acc": acc,
        "bal_acc": bal_acc,
        "auc": auc,
        "sens": sens,
        "spec": spec,
        "f1": f1,
    }


@torch.no_grad()
def evaluate_loader(
    model,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int = 2,
    threshold: Optional[float] = None,
    forward_fn=None,
) -> Dict[str, Any]:
    """
    Evaluate a model on a DataLoader, collecting y_true and y_prob for metrics.

    Args:
        model: nn.Module
        loader: DataLoader
        device: torch.device
        num_classes: Number of classes
        threshold: Optional decision threshold
        forward_fn: Optional custom forward function (model, x) → logits.
                    Defaults to model(x).

    Returns:
        Metrics dict from compute_metrics_from_raw.
    """
    model.eval()
    all_y_true = []
    all_y_prob = []

    for batch in loader:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            x, y = batch
        else:
            continue
        x = x.to(device)
        if forward_fn is not None:
            logits = forward_fn(model, x)
        elif hasattr(model, "forward_classifier"):
            logits = model.forward_classifier(x)
        else:
            logits = model(x)
        probs = F.softmax(logits.to(torch.float32), dim=1).detach().cpu().numpy()
        all_y_true.extend(y.cpu().numpy().tolist())
        all_y_prob.append(probs)

    if not all_y_true:
        result = compute_metrics_from_raw(np.array([]), np.array([]), num_classes, threshold)
        result["_y_true"] = np.array([])
        result["_y_prob"] = np.array([])
        return result

    all_y_true = np.array(all_y_true, dtype=np.int64)
    all_y_prob = np.concatenate(all_y_prob, axis=0)
    result = compute_metrics_from_raw(all_y_true, all_y_prob, num_classes, threshold)
    result["_y_true"] = all_y_true
    result["_y_prob"] = all_y_prob
    return result


# ---------------------------------------------------------------------------
# Threshold tuning
# ---------------------------------------------------------------------------

def find_best_threshold_bal_acc(
    y_true_np: np.ndarray,
    y_prob_pos_np: np.ndarray,
) -> Tuple[float, float, Optional[torch.Tensor]]:
    """Find threshold that maximizes balanced accuracy."""
    y_prob_pos_np = np.asarray(y_prob_pos_np, dtype=np.float32)
    if y_prob_pos_np.size == 0:
        return 0.5, 0.0, None
    unique_probs = np.unique(np.clip(y_prob_pos_np, 0.0, 1.0))
    thrs = np.unique(np.concatenate((
        np.array([0.0], dtype=np.float32),
        unique_probs,
        np.array([1.0], dtype=np.float32)
    )))
    best_thr, best_bal_acc, best_cm = 0.5, -1.0, None
    for t in thrs:
        preds = (y_prob_pos_np >= t).astype(np.int64)
        cm, _, bal_acc = compute_confusion_and_metrics(list(y_true_np), list(preds), num_classes=2)
        if bal_acc > best_bal_acc:
            best_thr, best_bal_acc, best_cm = float(t), float(bal_acc), cm
    return best_thr, best_bal_acc, best_cm


def split_calib_eval_indices(
    y_true_np: np.ndarray,
    calib_fraction: float,
    seed: int,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Stratified split into calibration/evaluation subsets for threshold tuning."""
    y_true_np = np.asarray(y_true_np, dtype=np.int64)
    if y_true_np.size < 8:
        return None, None

    classes = np.unique(y_true_np)
    rng = np.random.RandomState(seed)
    calib_idx = []
    eval_idx = []
    for c in classes:
        idx = np.where(y_true_np == c)[0]
        if idx.size < 4:
            return None, None
        rng.shuffle(idx)
        n_calib = int(round(idx.size * calib_fraction))
        n_calib = max(2, min(idx.size - 2, n_calib))
        calib_idx.extend(idx[:n_calib].tolist())
        eval_idx.extend(idx[n_calib:].tolist())

    if len(calib_idx) == 0 or len(eval_idx) == 0:
        return None, None
    return np.array(calib_idx, dtype=np.int64), np.array(eval_idx, dtype=np.int64)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def save_cm_png(cm: torch.Tensor, class_names: List[str], out_path: str):
    """Save confusion matrix as a PNG image."""
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)
    im = ax.imshow(cm.numpy(), cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    for i in range(cm.size(0)):
        for j in range(cm.size(1)):
            ax.text(j, i, int(cm[i, j].item()), va="center", ha="center", color="black", fontsize=9)
    ax.set_xlabel("Pred")
    ax.set_ylabel("True")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def save_roc_curve(
    y_true,
    y_prob,
    output_path: str,
    title: str = "",
    class_names: List[str] = None,
):
    """
    Save an ROC curve PNG + CSV (fpr, tpr, threshold) for a binary task.

    Args:
        y_true: [N] integer labels (0/1).
        y_prob: [N, C] probabilities, or [N] probability of class 1.
        output_path: e.g. '<run>/roc_curve.png' — CSV written as same stem.
        title: Optional title override.
        class_names: [neg, pos] labels for the legend.

    Single-class guard: writes '<stem>_meta.csv' with n_pos/n_neg/auc=nan
    and an empty fpr csv; no PNG (prevents sklearn crash on degenerate y).
    """
    import csv as _csv
    if class_names is None:
        class_names = ["AD", "LBD"]
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pos = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())

    base = os.path.splitext(output_path)[0]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if n_pos == 0 or n_neg == 0:
        with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
            _csv.writer(f).writerow(["fpr", "tpr", "threshold"])
        with open(base + "_meta.csv", "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["n_pos", "n_neg", "auc"])
            w.writerow([n_pos, n_neg, float("nan")])
        return

    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thr = roc_curve(y_true, y_pos)
    roc_auc = auc(fpr, tpr)

    with open(base + ".csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["fpr", "tpr", "threshold"])
        for a, b, c in zip(fpr, tpr, thr):
            w.writerow([float(a), float(b), float(c)])
    with open(base + "_meta.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["n_pos", "n_neg", "auc"])
        w.writerow([n_pos, n_neg, float(roc_auc)])

    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.plot(fpr, tpr, color="steelblue", lw=2, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", ls="--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title or f"ROC ({class_names[0]} vs {class_names[1]})")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_metrics_curves(out_dir: str, fold_idx: int, hist: Dict[str, List[float]]):
    """Plot training/validation curves (loss, accuracy, AUC)."""
    import numpy as np
    try:
        from scipy.signal import savgol_filter
        def smooth(data, wl=5, po=2):
            if len(data) < wl:
                return data
            try:
                return savgol_filter(data, wl, po)
            except Exception:
                return data
    except ImportError:
        def smooth(data, wl=5, po=2):
            return data

    epochs = range(1, len(hist.get("train_loss", [])) + 1)
    if not epochs:
        return

    fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=300)

    # Loss
    if "train_loss" in hist and "val_loss" in hist:
        axs[0].plot(epochs, hist["train_loss"], alpha=0.3, color='tab:blue', label="Train Loss (Raw)")
        axs[0].plot(epochs, smooth(hist["train_loss"]), color='tab:blue', linewidth=2, label="Train Loss")
        axs[0].plot(epochs, hist["val_loss"], alpha=0.3, color='tab:orange', label="Val Loss (Raw)")
        axs[0].plot(epochs, smooth(hist["val_loss"]), color='tab:orange', linewidth=2, label="Val Loss")
    axs[0].set_title("Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Accuracy
    if "train_acc" in hist and "val_acc" in hist:
        axs[1].plot(epochs, hist["train_acc"], alpha=0.3, color='tab:green', label="Train Acc (Raw)")
        axs[1].plot(epochs, smooth(hist["train_acc"]), color='tab:green', linewidth=2, label="Train Acc")
        axs[1].plot(epochs, hist["val_acc"], alpha=0.3, color='tab:red', label="Val Acc (Raw)")
        axs[1].plot(epochs, smooth(hist["val_acc"]), color='tab:red', linewidth=2, label="Val Acc")
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # AUC — train_auc may be empty for standard training
    has_train_auc = "train_auc" in hist and len(hist.get("train_auc", [])) > 0
    has_val_auc = "val_auc" in hist and len(hist.get("val_auc", [])) > 0
    if has_train_auc:
        axs[2].plot(epochs, hist["train_auc"], alpha=0.3, color='tab:purple', label="Train AUC (Raw)")
        axs[2].plot(epochs, smooth(hist["train_auc"]), color='tab:purple', linewidth=2, label="Train AUC")
    if has_val_auc:
        axs[2].plot(epochs, hist["val_auc"], alpha=0.3, color='tab:brown', label="Val AUC (Raw)")
        axs[2].plot(epochs, smooth(hist["val_auc"]), color='tab:brown', linewidth=2, label="Val AUC")
        if len(hist["val_auc"]) > 0:
            best_idx = np.argmax(hist["val_auc"])
            best_val = hist["val_auc"][best_idx]
            axs[2].scatter(best_idx + 1, best_val, color='red', s=50, zorder=5)
            axs[2].text(best_idx + 1, best_val, f"Best: {best_val:.3f}", fontsize=9, verticalalignment='bottom')
    axs[2].set_title("AUC")
    axs[2].set_xlabel("Epoch")
    axs[2].legend()
    axs[2].grid(True, alpha=0.3)

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(os.path.join(out_dir, f"metrics_fold_{fold_idx + 1}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# CV aggregation
# ---------------------------------------------------------------------------

def aggregate_cv_metrics(fold_metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate per-fold metrics into mean±std summary.

    Args:
        fold_metrics: List of dicts from compute_metrics_from_raw.

    Returns:
        Dict with per-metric mean, std, per_fold values.
    """
    keys = ["acc", "bal_acc", "auc", "sens", "spec", "f1"]
    summary = {}
    for key in keys:
        values = [m.get(key, float("nan")) for m in fold_metrics]
        # Filter NaN for mean/std
        valid = [v for v in values if not (isinstance(v, float) and np.isnan(v))]
        summary[key] = {
            "mean": float(np.mean(valid)) if valid else float("nan"),
            "std": float(np.std(valid)) if valid else float("nan"),
            "per_fold": values,
        }
    return summary


def save_fold_results(
    fold_dir: str,
    metrics: Dict[str, Any],
    model_state: Optional[Dict] = None,
    class_names: List[str] = None,
):
    """Save per-fold metrics, confusion matrix, and checkpoint.

    ``fold_dir`` is the fold's own directory (single level) — callers pass
    the already-fold-specific path (e.g. ``<run>/fold_0``), so nothing is
    appended here. The caller is responsible for creating it.
    """
    if class_names is None:
        class_names = ["AD", "LBD"]

    os.makedirs(fold_dir, exist_ok=True)

    # Save metrics JSON (strip numpy arrays that aren't JSON-serializable)
    _array_keys = {"_y_true", "_y_prob", "cm"}
    metrics_serializable = {}
    for k, v in metrics.items():
        if k in _array_keys:
            continue  # saved separately as cm_raw.json / predictions.csv
        if isinstance(v, torch.Tensor):
            metrics_serializable[k] = v.tolist()
        elif isinstance(v, np.ndarray):
            metrics_serializable[k] = v.tolist()
        elif isinstance(v, (int, float, str, bool, list, dict)) or v is None:
            metrics_serializable[k] = v
        else:
            metrics_serializable[k] = str(v)
    with open(os.path.join(fold_dir, "metrics.json"), "w") as f:
        json.dump(metrics_serializable, f, indent=2)

    # Save confusion matrix
    cm = metrics.get("cm")
    if cm is not None:
        save_cm_png(cm, class_names, os.path.join(fold_dir, "cm.png"))
        if isinstance(cm, torch.Tensor):
            cm_list = cm.tolist()
        else:
            cm_list = cm
        with open(os.path.join(fold_dir, "cm_raw.json"), "w") as f:
            json.dump(cm_list, f, indent=2)

    # Save checkpoint
    if model_state is not None:
        torch.save(model_state, os.path.join(fold_dir, "best_model.pth"))


# ---------------------------------------------------------------------------
# Predictions CSV export (for custom plotting in R/matplotlib/seaborn)
# ---------------------------------------------------------------------------

def save_predictions_csv(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    output_path: str,
    patient_ids: Optional[List[str]] = None,
    fold_label: str = "0",
    threshold: Optional[float] = None,
):
    """
    Export predictions to CSV for custom plotting.

    CSV columns: patient_id, fold, y_true, y_pred, y_prob_0, y_prob_1

    Args:
        y_true: [N] integer labels
        y_prob: [N, C] predicted probabilities
        output_path: CSV file path
        patient_ids: Optional list of patient IDs
        fold_label: Fold identifier string
        threshold: Decision threshold for class 1. If None, uses argmax.
    """
    import csv
    if threshold is not None:
        y_pred = (y_prob[:, 1] >= threshold).astype(np.int64)
    else:
        y_pred = y_prob.argmax(axis=1) if y_prob.ndim > 1 else (y_prob >= 0.5).astype(np.int64)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["patient_id", "fold", "y_true", "y_pred", "y_prob_AD", "y_prob_LBD"])
        for i in range(len(y_true)):
            pid = patient_ids[i] if patient_ids else f"sample_{i}"
            writer.writerow([
                pid, fold_label,
                int(y_true[i]), int(y_pred[i]),
                float(y_prob[i, 0]) if y_prob.ndim > 1 else float(1 - y_prob[i]),
                float(y_prob[i, 1]) if y_prob.ndim > 1 else float(y_prob[i]),
            ])


# ---------------------------------------------------------------------------
# Combined confusion matrix (multi-fold)
# ---------------------------------------------------------------------------

def save_combined_cm_png(
    cms: List[torch.Tensor],
    output_path: str,
    class_names: List[str] = None,
    fold_labels: List[str] = None,
):
    """
    Save a combined confusion matrix figure from multiple folds.

    Args:
        cms: List of confusion matrix tensors [num_classes, num_classes].
        output_path: PNG output path.
        class_names: ["AD", "LBD"] default.
        fold_labels: Labels for each subplot title.
    """
    if class_names is None:
        class_names = ["AD", "LBD"]
    if fold_labels is None:
        fold_labels = [f"Fold {i+1}" for i in range(len(cms))]

    n = len(cms)
    if n == 0:
        return

    cols = min(n, 5)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4), dpi=150)
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()

    # Plot sum CM first
    sum_cm = sum(cms)
    sum_cm = sum_cm / sum_cm.sum() * 100  # normalized to percentages

    for idx, (cm, label) in enumerate(zip(cms, fold_labels)):
        ax = axes[idx]
        im = ax.imshow(cm.numpy(), cmap="Blues", vmin=0)
        ax.set_title(label)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha="right")
        ax.set_yticklabels(class_names)
        for i in range(cm.size(0)):
            for j in range(cm.size(1)):
                ax.text(j, i, int(cm[i, j].item()), va="center", ha="center", fontsize=10)
        ax.set_xlabel("Pred")

    # Hide unused subplots
    for idx in range(n, len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Confusion Matrices (AD vs LBD)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CV table CSV (for Excel plotting)
# ---------------------------------------------------------------------------

def save_cv_table_csv(fold_metrics: List[Dict[str, Any]], output_path: str):
    """
    Save per-fold metrics as a flat CSV table for easy Excel/matplotlib import.

    Columns: fold, auc, bal_acc, acc, sens, spec, f1
    """
    import csv
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    keys = ["auc", "bal_acc", "acc", "sens", "spec", "f1"]
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["fold"] + keys)
        for i, m in enumerate(fold_metrics):
            row = [f"fold_{i}"] + [m.get(k, "") for k in keys]
            writer.writerow(row)
        # Add mean row
        means = ["mean"]
        for k in keys:
            vals = [m.get(k, float("nan")) for m in fold_metrics]
            valid = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            means.append(float(np.mean(valid)) if valid else float("nan"))
        writer.writerow(means)
        # Add std row
        stds = ["std"]
        for k in keys:
            vals = [m.get(k, float("nan")) for m in fold_metrics]
            valid = [v for v in vals if not (isinstance(v, float) and np.isnan(v))]
            stds.append(float(np.std(valid)) if valid else float("nan"))
        writer.writerow(stds)
