# 顶部导入
import os
import yaml
import json
import math
import re
import uuid
import argparse
import itertools
from datetime import datetime
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from medmamba3d import MedMamba3D
from medmamba_ss3m import MedMambaSS3M
from torch.utils.data._utils.collate import default_collate
import contextlib
from dataset_mri3d import ensure_3d_volume, zscore_normalize, center_pad_or_crop
from typing import Optional
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# EMA (Exponential Moving Average) for smoothing metrics
class EMA:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.value = None

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

def smooth_curve(data, window_length=5, polyorder=2):
    """Apply Savitzky-Golay filter for smoothing."""
    if len(data) < window_length:
        return data
    return savgol_filter(data, window_length, polyorder)

def plot_training_curves(log_data: list, output_path: str):
    """
    绘制高质量的训练曲线图，包含 Loss, Accuracy, AUC。
    log_data: List[dict] with keys 'epoch', 'train_loss', 'train_acc', 'val_acc', 'train_auc', 'val_auc'
    """
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    epochs = [d['epoch'] for d in log_data]
    train_loss = [d['train_loss'] for d in log_data]
    
    train_acc = [d['train_acc'] for d in log_data]
    val_acc = [d['val_acc'] for d in log_data]
    
    train_auc = [d.get('train_auc', 0) for d in log_data]
    val_auc = [d.get('val_auc', 0) for d in log_data]

    # Smooth data
    train_loss_smooth = smooth_curve(train_loss)
    train_acc_smooth = smooth_curve(train_acc)
    val_acc_smooth = smooth_curve(val_acc)
    train_auc_smooth = smooth_curve(train_auc)
    val_auc_smooth = smooth_curve(val_auc)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Loss
    axes[0].plot(epochs, train_loss, alpha=0.3, color='tab:blue', label='Train Loss (Raw)')
    axes[0].plot(epochs, train_loss_smooth, color='tab:blue', linewidth=2, label='Train Loss (Smooth)')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # 2. Accuracy
    axes[1].plot(epochs, train_acc, alpha=0.3, color='tab:blue', label='Train Acc (Raw)')
    axes[1].plot(epochs, train_acc_smooth, color='tab:blue', linewidth=2, label='Train Acc (Smooth)')
    axes[1].plot(epochs, val_acc, alpha=0.3, color='tab:orange', label='Val Acc (Raw)')
    axes[1].plot(epochs, val_acc_smooth, color='tab:orange', linewidth=2, label='Val Acc (Smooth)')
    
    # Mark best Val Acc
    best_acc_idx = np.argmax(val_acc)
    axes[1].scatter(epochs[best_acc_idx], val_acc[best_acc_idx], color='red', s=50, zorder=5)
    axes[1].annotate(f'Best: {val_acc[best_acc_idx]:.4f}', 
                     (epochs[best_acc_idx], val_acc[best_acc_idx]),
                     xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)

    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.5)

    # 3. AUC
    axes[2].plot(epochs, train_auc, alpha=0.3, color='tab:blue', label='Train AUC (Raw)')
    axes[2].plot(epochs, train_auc_smooth, color='tab:blue', linewidth=2, label='Train AUC (Smooth)')
    axes[2].plot(epochs, val_auc, alpha=0.3, color='tab:orange', label='Val AUC (Raw)')
    axes[2].plot(epochs, val_auc_smooth, color='tab:orange', linewidth=2, label='Val AUC (Smooth)')

    # Mark best Val AUC
    best_auc_idx = np.argmax(val_auc)
    axes[2].scatter(epochs[best_auc_idx], val_auc[best_auc_idx], color='red', s=50, zorder=5)
    axes[2].annotate(f'Best: {val_auc[best_auc_idx]:.4f}', 
                     (epochs[best_auc_idx], val_auc[best_auc_idx]),
                     xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)

    axes[2].set_title('AUC')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('AUC')
    axes[2].legend()
    axes[2].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"[Plot] Saved metrics plot to {output_path}")

try:
    import wandb
except ImportError:
    wandb = None


class MRIVolumeLabeledDataset(torch.utils.data.Dataset):
    """
    简化版：读取 (path, label) 列，统一到目标尺寸。这里假设数据已为 1mm
    """
    def __init__(self, manifest_csv: str, path_column: str, label_column: str, label_map: dict, target_shape=(112, 112, 112)):
        import csv
        import nibabel as nib
        import numpy as np

        self.target_shape = target_shape
        self.items = []
        with open(manifest_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = row.get(path_column, '')
                lab = row.get(label_column, '')
                if p and os.path.isfile(p) and lab in label_map:
                    self.items.append((p, label_map[lab]))
        self.nib = nib
        self.np = np

        if not self.items:
            raise RuntimeError("No labeled items found.")

    def zscore(self, v):
        p1, p99 = self.np.percentile(v, 1.0), self.np.percentile(v, 99.0)
        v = self.np.clip(v, p1, p99)
        m, s = v.mean(), v.std() + 1e-6
        return (v - m) / s

    def padcrop(self, vol):
        D, H, W = vol.shape
        tD, tH, tW = self.target_shape
        import numpy as np
        out = vol
        pad_d = max(tD - D, 0)
        pad_h = max(tH - H, 0)
        pad_w = max(tW - W, 0)
        if pad_d or pad_h or pad_w:
            out = np.pad(out, ((pad_d // 2, pad_d - pad_d // 2),
                               (pad_h // 2, pad_h - pad_h // 2),
                               (pad_w // 2, pad_w - pad_w // 2)), mode='constant')
        D, H, W = out.shape
        sD = (D - tD) // 2 if D > tD else 0
        sH = (H - tH) // 2 if H > tH else 0
        sW = (W - tW) // 2 if W > tW else 0
        out = out[sD:sD + tD, sH:sH + tH, sW:sW + tW]
        return out

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, y = self.items[idx]
        vol = self.nib.load(p).get_fdata().astype(self.np.float32)
        vol = self.zscore(vol)
        vol = self.padcrop(vol)
        vol = self.np.expand_dims(vol, 0)
        ten = torch.from_numpy(vol).float()  # 显式转 float32
        return ten, torch.tensor(y, dtype=torch.long)

class MRIVolumeFolderDataset(torch.utils.data.Dataset):
    def __init__(self, label_roots: dict, label_map: dict, target_shape=(112, 112, 112), validate_nifti: bool = True, file_exts: tuple = (".nii", ".nii.gz"), require_name_substring: Optional[str] = None):
        import nibabel as nib
        import numpy as np
        self.nib = nib
        self.np = np
        self.items = []
        self.target_shape = target_shape

        # 收集文件
        for lab_name, root in label_roots.items():
            y = label_map.get(lab_name, None)
            roots = root if isinstance(root, (list, tuple)) else [root]
            if y is None:
                continue
            for one_root in roots:
                if not isinstance(one_root, str) or (not os.path.isdir(one_root)):
                    continue
                for dirpath, _, filenames in os.walk(one_root):
                    for fn in filenames:
                        if not fn.lower().endswith(file_exts):
                            continue
                        if require_name_substring and (require_name_substring not in fn):
                            continue
                        p = os.path.join(dirpath, fn)
                        if os.path.isfile(p):
                            self.items.append((p, y))

        if not self.items:
            raise RuntimeError("No labeled items found from label_roots.")

        # 可选：校验 .nii 文件字节数是否满足头信息声明的大小（跳过损坏样本）
        if validate_nifti:
            valid, invalid = [], []
            for p, y in self.items:
                ext = os.path.splitext(p)[1].lower()
                if ext == ".nii":
                    try:
                        img = self.nib.load(p)
                        shape = img.header.get_data_shape()
                        dtype = img.get_data_dtype()
                        offset = int(float(img.header.get("vox_offset", 0)))
                        expected = offset + int(self.np.prod(shape)) * self.np.dtype(dtype).itemsize
                        actual = os.path.getsize(p)
                        if actual < expected:
                            invalid.append(p)
                        else:
                            valid.append((p, y))
                    except Exception:
                        invalid.append(p)
                else:
                    # .nii.gz 不做字节比对
                    valid.append((p, y))
            self.items = valid
            if invalid:
                print(f"[MRIVolumeFolderDataset] skipped {len(invalid)} damaged .nii files (size < expected).")

        if not self.items:
            raise RuntimeError("All labeled items are invalid after validation.")

        self.in_channels = 1
        print(f"[MRIVolumeFolderDataset] samples={len(self.items)}, inferred in_channels={self.in_channels}")

    def _to_channel_first_3d(self, arr):
        # arr: (X,Y,Z,ExtraDims...), 折叠额外维为通道 C
        spatial = arr.shape[:3]
        extra = arr.shape[3:] if arr.ndim > 3 else ()
        C = int(self.np.prod(extra)) if extra else 1
        arr = arr.reshape(spatial[0], spatial[1], spatial[2], C)  # (D,H,W,C)
        arr = self.np.transpose(arr, (3, 0, 1, 2))               # (C,D,H,W)
        return arr

    def zscore_channels(self, v):
        # v: (C,D,H,W)，按通道剪裁到1/99分位并做z-score
        C = v.shape[0]
        flat = v.reshape(C, -1)
        p1 = self.np.percentile(flat, 1.0, axis=1).reshape(C, 1, 1, 1)
        p99 = self.np.percentile(flat, 99.0, axis=1).reshape(C, 1, 1, 1)
        v = self.np.clip(v, p1, p99)
        m = v.mean(axis=(1, 2, 3), keepdims=True)
        s = v.std(axis=(1, 2, 3), keepdims=True) + 1e-6
        return (v - m) / s

    def padcrop_ch(self, v):
        # v: (C,D,H,W)，仅对空间维做居中 pad/crop
        C, D, H, W = v.shape
        tD, tH, tW = self.target_shape
        pad_d = max(tD - D, 0)
        pad_h = max(tH - H, 0)
        pad_w = max(tW - W, 0)
        if pad_d or pad_h or pad_w:
            v = self.np.pad(v, ((0, 0),
                                (pad_d // 2, pad_d - pad_d // 2),
                                (pad_h // 2, pad_h - pad_h // 2),
                                (pad_w // 2, pad_w - pad_w // 2)),
                             mode='constant')
        _, D, H, W = v.shape
        sD = (D - tD) // 2 if D > tD else 0
        sH = (H - tH) // 2 if H > tH else 0
        sW = (W - tW) // 2 if W > tW else 0
        v = v[:, sD:sD + tD, sH:sH + tH, sW:sW + tW]
        return v

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, y = self.items[idx]
        try:
            arr = self.nib.load(p).get_fdata().astype(self.np.float32)
        except Exception as e:
            # 读取失败（损坏或其他异常）→ 返回 None，让 collate_fn 过滤
            print(f"[MRIVolumeFolderDataset] read error, skip: {p} | {e}")
            return None
        v = ensure_3d_volume(arr, reduce_strategy='first')
        v = zscore_normalize(v)
        v = center_pad_or_crop(v, self.target_shape)
        v = self.np.expand_dims(v, 0)
        ten = torch.from_numpy(v).float()
        return ten, torch.tensor(y, dtype=torch.long)

def make_safe_collate(in_chans, target_shape):
    def _collate(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            x = torch.empty((0, in_chans, target_shape[0], target_shape[1], target_shape[2]), dtype=torch.float32)
            y = torch.empty((0,), dtype=torch.long)
            return x, y
        return default_collate(batch)
    return _collate

# ===== 新增：分层K折、AUC与丰富指标评估工具 =====
def stratified_kfold_indices(items, num_classes: int, n_splits: int = 10, seed: int = 42):
    import numpy as np
    # items: List[(path, y)]
    per_class = [[] for _ in range(num_classes)]
    for idx, (_, y) in enumerate(items):
        if 0 <= y < num_classes:
            per_class[y].append(idx)
    folds = [set() for _ in range(n_splits)]
    for c in range(num_classes):
        idxs = per_class[c]
        if not idxs:
            continue
        rng = np.random.default_rng(seed + c)
        idxs = rng.permutation(idxs).tolist()
        chunks = np.array_split(idxs, n_splits)
        for f in range(n_splits):
            folds[f].update(int(i) for i in chunks[f].tolist())
    # 返回每折的验证索引列表
    return [sorted(list(s)) for s in folds]

def auc_binary_np(y_true, y_score):
    import numpy as np
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score).astype(np.float64)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    x = np.concatenate([neg, pos])
    order = np.argsort(x)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(order.size) + 1
    R_pos = ranks[neg.size:].sum()
    P, N = float(pos.size), float(neg.size)
    return float((R_pos - P * (P + 1) / 2) / (P * N))

def auc_multiclass_macro_np(y_true, y_prob, num_classes: int):
    import numpy as np
    y_true = np.asarray(y_true).astype(np.int32)
    y_prob = np.asarray(y_prob).astype(np.float64)
    aucs = []
    for c in range(num_classes):
        y_bin = (y_true == c).astype(np.float32)
        auc_c = auc_binary_np(y_bin, y_prob[:, c])
        if not np.isnan(auc_c):
            aucs.append(auc_c)
    if not aucs:
        return float("nan")
    return float(sum(aucs) / len(aucs))

def metrics_from_preds(y_true_list, y_pred_list, num_classes: int):
    import numpy as np
    y_true = np.asarray(y_true_list, dtype=np.int64)
    y_pred = np.asarray(y_pred_list, dtype=np.int64)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    total = cm.sum()
    acc = float(np.trace(cm)) / max(total, 1)
    sensitivities, specificities = [], []
    for c in range(num_classes):
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = cm[:, c].sum() - tp
        tn = total - tp - fn - fp
        sens = tp / max(tp + fn, 1)           # 召回/敏感性
        spec = tn / max(tn + fp, 1)           # 特异性
        sensitivities.append(float(sens))
        specificities.append(float(spec))
    bal_acc = float(sum(sensitivities) / max(num_classes, 1))
    spec_macro = float(sum(specificities) / max(num_classes, 1))
    return cm, acc, bal_acc, sensitivities, specificities, spec_macro

def _f1_macro_from_cm(cm: np.ndarray):
    num_classes = cm.shape[0]
    f1s = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = (2 * tp + fp + fn)
        f1 = 0.0 if denom == 0 else (2 * tp / denom)
        f1s.append(float(f1))
    return float(sum(f1s) / max(num_classes, 1))

def _cohen_kappa(cm: np.ndarray):
    total = float(cm.sum())
    if total <= 0:
        return float("nan")
    po = float(np.trace(cm)) / total
    row = cm.sum(axis=1)
    col = cm.sum(axis=0)
    pe = float(np.sum(row * col)) / (total * total)
    denom = 1.0 - pe
    if denom <= 0:
        return float("nan")
    return float((po - pe) / denom)

def _mcc_from_cm(cm: np.ndarray):
    s = float(cm.sum())
    if s <= 0:
        return float("nan")
    row = cm.sum(axis=1).astype(np.float64)
    col = cm.sum(axis=0).astype(np.float64)
    c = float(np.trace(cm))
    numerator = (c * s) - float(np.dot(row, col))
    denom = (s * s - float(np.dot(col, col))) * (s * s - float(np.dot(row, row)))
    if denom <= 0:
        return float("nan")
    return float(numerator / np.sqrt(denom))

def _pr_curve_binary(y_true: np.ndarray, y_score: np.ndarray, num_thresholds: int = 101):
    thresholds = np.linspace(0.0, 1.0, num_thresholds)
    precision, recall = [], []
    for t in thresholds:
        y_pred = (y_score >= t).astype(np.int64)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        precision.append(float(prec))
        recall.append(float(rec))
    return {"thresholds": thresholds.tolist(), "precision": precision, "recall": recall}

def _class_weights_from_labels(labels, num_classes: int):
    labels = np.asarray(labels, dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    total = float(counts.sum())
    weights = []
    for c in range(num_classes):
        weights.append(0.0 if counts[c] == 0 else total / (num_classes * counts[c]))
    return np.asarray(weights, dtype=np.float32)

def _infer_num_classes_from_items(items, default_num_classes: int):
    if not items:
        return int(default_num_classes)
    labels = [y for _, y in items]
    if not labels:
        return int(default_num_classes)
    max_label = int(max(labels))
    min_label = int(min(labels))
    if min_label < 0:
        raise RuntimeError(f"存在负标签: min_label={min_label}")
    return max(int(default_num_classes), max_label + 1)

def _to_py(obj):
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {str(_to_py(k)) if not isinstance(k, str) else k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_py(v) for v in obj]
    return str(obj)

def _focal_loss(logits: torch.Tensor, targets: torch.Tensor, gamma: float = 2.0, alpha: Optional[torch.Tensor] = None):
    logp = F.log_softmax(logits, dim=1)
    p = torch.exp(logp)
    logp_t = logp.gather(1, targets.view(-1, 1)).squeeze(1)
    p_t = p.gather(1, targets.view(-1, 1)).squeeze(1)
    loss = -((1 - p_t) ** gamma) * logp_t
    if alpha is not None:
        loss = loss * alpha.gather(0, targets)
    return loss.mean()

def _dice_loss(logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-6):
    probs = torch.softmax(logits, dim=1)
    num_classes = probs.size(1)
    y_onehot = F.one_hot(targets, num_classes=num_classes).float()
    y_onehot = y_onehot.to(probs.device)
    intersect = torch.sum(probs * y_onehot, dim=0)
    denom = torch.sum(probs + y_onehot, dim=0)
    dice = (2 * intersect + smooth) / (denom + smooth)
    return 1.0 - dice.mean()

def _tversky_loss(logits: torch.Tensor, targets: torch.Tensor, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1e-6):
    probs = torch.softmax(logits, dim=1)
    num_classes = probs.size(1)
    y_onehot = F.one_hot(targets, num_classes=num_classes).float()
    y_onehot = y_onehot.to(probs.device)
    tp = torch.sum(probs * y_onehot, dim=0)
    fp = torch.sum(probs * (1 - y_onehot), dim=0)
    fn = torch.sum((1 - probs) * y_onehot, dim=0)
    tversky = (tp + smooth) / (tp + alpha * fp + beta * fn + smooth)
    return 1.0 - tversky.mean()

def _apply_smote_like(x: torch.Tensor, y: torch.Tensor, mode: str):
    if x.size(0) < 2:
        return x, y
    device = x.device
    y_cpu = y.detach().cpu().tolist()
    index_by_class = {}
    for idx, cls in enumerate(y_cpu):
        index_by_class.setdefault(cls, []).append(idx)
    lam_dist = torch.distributions.Beta(0.5, 0.5) if mode == "adasyn" else None
    x_out = x.clone()
    for cls, idxs in index_by_class.items():
        if len(idxs) < 2:
            continue
        idxs_tensor = torch.tensor(idxs, device=device)
        perm = idxs_tensor[torch.randperm(len(idxs_tensor))]
        lam = torch.rand(len(idxs_tensor), device=device) if lam_dist is None else lam_dist.sample((len(idxs_tensor),)).to(device)
        lam = lam.view(-1, 1, 1, 1, 1)
        x_out[idxs_tensor] = lam * x[idxs_tensor] + (1 - lam) * x[perm]
    return x_out, y

def augment_3d_batch(x, cfg):
    # x: (B, C, D, H, W)
    p_flip = float(cfg.get("flip_p", 0.5))
    if p_flip > 0.0:
        if torch.rand(()) < p_flip:
            x = x.flip(-1)
        if torch.rand(()) < p_flip:
            x = x.flip(-2)
        if torch.rand(()) < p_flip:
            x = x.flip(-3)
    noise_std = float(cfg.get("noise_std", 0.02))
    if noise_std > 0.0:
        x = x + torch.randn_like(x) * noise_std
    gamma_jitter = float(cfg.get("gamma_jitter", 0.1))
    if gamma_jitter > 0.0:
        g = torch.empty((x.size(0), 1, 1, 1, 1), device=x.device).uniform_(1.0 - gamma_jitter, 1.0 + gamma_jitter)
        x = torch.sign(x) * (torch.abs(x) ** g)
    c_p = float(cfg.get("cutout_p", 0.3))
    c_frac = float(cfg.get("cutout_frac", 0.15))
    if c_p > 0.0 and torch.rand(()) < c_p:
        d = int(max(1, c_frac * x.size(-3)))
        h = int(max(1, c_frac * x.size(-2)))
        w = int(max(1, c_frac * x.size(-1)))
        sd = torch.randint(0, x.size(-3) - d + 1, (1,)).item()
        sh = torch.randint(0, x.size(-2) - h + 1, (1,)).item()
        sw = torch.randint(0, x.size(-1) - w + 1, (1,)).item()
        x[:, :, sd:sd + d, sh:sh + h, sw:sw + w] = 0
    return x

def _mean_std(values):
    if not values:
        return float("nan"), float("nan")
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std())

def _fmt_mean_std(mean_val, std_val):
    if mean_val is None or std_val is None:
        return "nan"
    if isinstance(mean_val, float) and (np.isnan(mean_val) or np.isnan(std_val)):
        return "nan"
    return f"{mean_val:.4f}±{std_val:.4f}"

def _summarize_fold_metrics(fold_metrics: list) -> dict:
    def _extract(key, index=None):
        vals = []
        for m in fold_metrics:
            v = m.get(key)
            if isinstance(v, (list, tuple)):
                if index is not None and len(v) > index:
                    v = v[index]
                else:
                    v = None
            if isinstance(v, (int, float, np.integer, np.floating)):
                vals.append(float(v))
        return vals

    stats = {}
    for key in ("acc", "bal_acc", "auc", "f1", "kappa", "mcc", "spec_macro"):
        mean_v, std_v = _mean_std(_extract(key))
        stats[f"mean_{key}"] = mean_v
        stats[f"std_{key}"] = std_v

    sens_pos_vals = _extract("sens", index=1)
    spec_pos_vals = _extract("spec", index=1)
    sens_macro_vals = _extract("bal_acc")
    spec_macro_vals = _extract("spec_macro")

    stats["mean_sens_pos"], stats["std_sens_pos"] = _mean_std(sens_pos_vals)
    stats["mean_spec_pos"], stats["std_spec_pos"] = _mean_std(spec_pos_vals)
    stats["mean_sens_macro"], stats["std_sens_macro"] = _mean_std(sens_macro_vals)
    stats["mean_spec_macro"], stats["std_spec_macro"] = _mean_std(spec_macro_vals)
    return stats

@torch.no_grad()
def evaluate_loader(model: torch.nn.Module, loader: DataLoader, device: torch.device, num_classes: int, threshold: Optional[float] = None, return_pr: bool = False, use_tta: bool = False):
    """
    作用：统一评估接口，兼容 MedMamba3D 与 SS3M 两种骨干的 forward_classifier。
    改动原因：原签名限定为 MedMamba3D，会在选择 SS3M 时造成类型提示不匹配；改为 nn.Module 更通用。
    新增 use_tta 参数，开启测试时增强 (Test Time Augmentation)，对输入进行翻转并取平均预测。
    """
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    for x, y in loader:
        if y.numel() == 0:
            continue
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        y = y.to(device, non_blocking=True)
        
        if use_tta:
            # TTA: Forward pass with original and flipped inputs
            logits_list = []
            # 1. Original
            if hasattr(model, "forward_classifier"):
                logits_list.append(model.forward_classifier(x))
            else:
                logits_list.append(model(x))
            
            # 2. Flip D (dim 2)
            x_flip_d = torch.flip(x, [2])
            if hasattr(model, "forward_classifier"):
                logits_list.append(model.forward_classifier(x_flip_d))
            else:
                logits_list.append(model(x_flip_d))
                
            # 3. Flip H (dim 3)
            x_flip_h = torch.flip(x, [3])
            if hasattr(model, "forward_classifier"):
                logits_list.append(model.forward_classifier(x_flip_h))
            else:
                logits_list.append(model(x_flip_h))
                
            # 4. Flip W (dim 4)
            x_flip_w = torch.flip(x, [4])
            if hasattr(model, "forward_classifier"):
                logits_list.append(model.forward_classifier(x_flip_w))
            else:
                logits_list.append(model(x_flip_w))
            
            # Average probabilities (more stable than averaging logits)
            probs_list = [torch.softmax(l, dim=1) for l in logits_list]
            probs = torch.stack(probs_list).mean(dim=0)
            # Logits are less meaningful after averaging probs, but argmax works on probs
            logits = torch.log(probs + 1e-9) # Approximate logits for consistency
        else:
            # `MedMambaSS3M` 没有 forward_classifier 方法，故先尝试调用，失败则退回标准 forward
            if hasattr(model, "forward_classifier"):
                logits = model.forward_classifier(x)
            else:
                logits = model(x)
            probs = torch.softmax(logits, dim=1)
            
        y_true.extend(y.detach().cpu().tolist())
        y_prob.extend(probs.detach().cpu().tolist())
        if num_classes == 2 and threshold is not None:
            y_pred.extend((probs[:, 1] >= threshold).detach().cpu().int().tolist())
        else:
            y_pred.extend(probs.argmax(dim=1).detach().cpu().tolist())
    cm, acc, bal_acc, sens, spec, spec_macro = metrics_from_preds(y_true, y_pred, num_classes)
    if len(y_true) > 0:
        if num_classes == 2:
            auc = auc_binary_np(np.array(y_true), np.array(y_prob)[:, 1])
        else:
            auc = auc_multiclass_macro_np(np.array(y_true), np.array(y_prob), num_classes)
    else:
        auc = float("nan")
    kappa = _cohen_kappa(cm) if len(y_true) > 0 else float("nan")
    mcc = _mcc_from_cm(cm) if len(y_true) > 0 else float("nan")
    f1_macro = _f1_macro_from_cm(cm) if len(y_true) > 0 else float("nan")
    sens_pos = float("nan")
    spec_pos = float("nan")
    if num_classes == 2 and isinstance(sens, (list, tuple)) and len(sens) > 1:
        sens_pos = float(sens[1])
    if num_classes == 2 and isinstance(spec, (list, tuple)) and len(spec) > 1:
        spec_pos = float(spec[1])
    sens_macro = bal_acc
    pr_curve = None
    raw = None
    if return_pr and num_classes == 2 and len(y_true) > 0:
        y_true_np = np.asarray(y_true, dtype=np.int64)
        y_prob_np = np.asarray(y_prob, dtype=np.float64)
        pr_curve = _pr_curve_binary(y_true_np, y_prob_np[:, 1])
        raw = {"y_true": y_true_np.tolist(), "y_prob": y_prob_np.tolist()}
    return {
        "cm": cm, "acc": acc, "bal_acc": bal_acc,
        "sens": sens, "spec": spec, "spec_macro": spec_macro,
        "auc": auc, "count": len(y_true),
        "kappa": kappa, "mcc": mcc, "f1": f1_macro, "pr_curve": pr_curve, "raw": raw,
        "sens_pos": sens_pos, "spec_pos": spec_pos, "sens_macro": sens_macro,
    }

def _build_folds(items, num_classes: int, n_splits: int, seed: int):
    val_folds = stratified_kfold_indices(items, num_classes=num_classes, n_splits=n_splits, seed=seed)
    all_idx = list(range(len(items)))
    folds = []
    all_set = set(all_idx)
    for val_idx in val_folds:
        val_set = set(val_idx)
        train_idx = [i for i in all_set if i not in val_set]
        folds.append((train_idx, val_idx))
    return folds

def _sample_hpo_params(space: dict, rng: np.random.Generator):
    params = {}
    for k, v in space.items():
        if isinstance(v, dict):
            v_type = v.get("type", "")
            if v_type == "loguniform":
                vmin, vmax = float(v["min"]), float(v["max"])
                params[k] = float(10 ** rng.uniform(np.log10(vmin), np.log10(vmax)))
            elif v_type == "uniform":
                vmin, vmax = float(v["min"]), float(v["max"])
                params[k] = float(rng.uniform(vmin, vmax))
            elif v_type == "choice":
                params[k] = rng.choice(v.get("values", []))
            else:
                params[k] = v.get("value")
        elif isinstance(v, (list, tuple)):
            if k in ("lr", "weight_decay") and len(v) == 2:
                params[k] = float(10 ** rng.uniform(np.log10(v[0]), np.log10(v[1])))
            elif k in ("dropout",) and len(v) == 2:
                params[k] = float(rng.uniform(v[0], v[1]))
            else:
                params[k] = rng.choice(list(v))
        else:
            params[k] = v
    return params

def _params_to_vector(space: dict, params: dict):
    vec = []
    for k, v in space.items():
        if isinstance(v, dict):
            v_type = v.get("type", "")
            if v_type == "loguniform":
                vmin, vmax = float(v["min"]), float(v["max"])
                pv = float(params.get(k, vmin))
                vec.append((np.log10(pv) - np.log10(vmin)) / max(np.log10(vmax) - np.log10(vmin), 1e-9))
            elif v_type == "uniform":
                vmin, vmax = float(v["min"]), float(v["max"])
                pv = float(params.get(k, vmin))
                vec.append((pv - vmin) / max(vmax - vmin, 1e-9))
            elif v_type == "choice":
                choices = v.get("values", [])
                pv = params.get(k, choices[0] if choices else 0)
                idx = choices.index(pv) if pv in choices else 0
                vec.append(idx / max(len(choices) - 1, 1))
            else:
                vec.append(float(params.get(k, 0.0)))
        elif isinstance(v, (list, tuple)):
            if k in ("lr", "weight_decay") and len(v) == 2:
                vmin, vmax = float(v[0]), float(v[1])
                pv = float(params.get(k, vmin))
                vec.append((np.log10(pv) - np.log10(vmin)) / max(np.log10(vmax) - np.log10(vmin), 1e-9))
            elif len(v) == 2 and isinstance(v[0], (int, float)) and isinstance(v[1], (int, float)) and k in ("dropout", "threshold"):
                vmin, vmax = float(v[0]), float(v[1])
                pv = float(params.get(k, vmin))
                vec.append((pv - vmin) / max(vmax - vmin, 1e-9))
            else:
                choices = list(v)
                pv = params.get(k, choices[0] if choices else 0)
                idx = choices.index(pv) if pv in choices else 0
                vec.append(idx / max(len(choices) - 1, 1))
        else:
            vec.append(float(params.get(k, 0.0)))
    return np.asarray(vec, dtype=np.float64)

def _propose_bayes(space: dict, observations: list, rng: np.random.Generator, n_candidates: int = 64):
    if len(observations) < 3:
        return _sample_hpo_params(space, rng)
    X = np.stack([_params_to_vector(space, p) for p, _ in observations], axis=0)
    y = np.asarray([s for _, s in observations], dtype=np.float64)
    X_mean = X.mean(axis=0, keepdims=True)
    X_std = X.std(axis=0, keepdims=True) + 1e-6
    Xn = (X - X_mean) / X_std
    def rbf(a, b, lengthscale=1.0):
        d2 = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
        return np.exp(-0.5 * d2 / (lengthscale ** 2))
    K = rbf(Xn, Xn) + 1e-6 * np.eye(len(Xn))
    K_inv = np.linalg.solve(K, np.eye(len(Xn)))
    best_y = np.max(y)
    candidates = [_sample_hpo_params(space, rng) for _ in range(n_candidates)]
    Xc = np.stack([_params_to_vector(space, p) for p in candidates], axis=0)
    Xc = (Xc - X_mean) / X_std
    K_s = rbf(Xn, Xc)
    K_ss = np.ones((len(Xc),), dtype=np.float64)
    mu = K_s.T @ K_inv @ y
    var = K_ss - np.sum(K_s.T @ K_inv * K_s.T, axis=1)
    var = np.clip(var, 1e-9, None)
    sigma = np.sqrt(var)
    z = (mu - best_y) / sigma
    from math import erf, sqrt, exp, pi
    cdf = 0.5 * (1 + np.array([erf(v / sqrt(2)) for v in z]))
    pdf = (1 / sqrt(2 * pi)) * np.exp(-0.5 * z ** 2)
    ei = (mu - best_y) * cdf + sigma * pdf
    best_idx = int(np.argmax(ei))
    return candidates[best_idx]

def _set_backbone_frozen(model, backbone_choice, frozen: bool):
    # 先对所有参数统一设置
    for param in model.parameters():
        param.requires_grad = not frozen
    
    # 如果是冻结状态，必须保证分类头（head/classifier）是可训练的
    if frozen:
        if hasattr(model, "head"):
             for param in model.head.parameters():
                 param.requires_grad = True
        elif hasattr(model, "classifier"):
             for param in model.classifier.parameters():
                 param.requires_grad = True


def _extract_state_dict_container(raw_state):
    container = "raw"
    sd = raw_state
    if isinstance(raw_state, dict):
        for key in ("state_dict", "model_state", "model"):
            v = raw_state.get(key)
            if isinstance(v, dict):
                sd = v
                container = key
                break
    return sd, container


def _infer_ss3m_hparams_from_ckpt(pretrained_path: Optional[str]) -> dict:
    if not pretrained_path or (not os.path.isfile(pretrained_path)):
        return {}
    try:
        raw = torch.load(pretrained_path, map_location="cpu")
    except Exception:
        return {}
    sd, container = _extract_state_dict_container(raw)
    if not isinstance(sd, dict):
        return {}

    info = {"container": container}
    patch_candidates = (
        "encoder.patch.proj.weight",
        "encoder.patch_embed.proj.weight",
        "patch.proj.weight",
        "patch_embed.proj.weight",
    )
    patch_key = next((k for k in patch_candidates if k in sd and hasattr(sd[k], "shape")), None)
    if patch_key is not None:
        w = sd[patch_key]
        if hasattr(w, "shape") and len(w.shape) == 5:
            info["patch_key"] = patch_key
            info["embed_dim"] = int(w.shape[0])
            info["in_channels"] = int(w.shape[1])
            info["patch_size"] = (int(w.shape[2]), int(w.shape[3]), int(w.shape[4]))

    max_blk = -1
    pat = re.compile(r"(?:^|\.)blocks\.(\d+)\.")
    for k in sd.keys():
        if isinstance(k, str):
            m = pat.search(k)
            if m:
                max_blk = max(max_blk, int(m.group(1)))
    if max_blk >= 0:
        info["depth"] = max_blk + 1

    return info


def _resolve_ss3m_arch(cfg: dict, params: Optional[dict], in_chans: int) -> dict:
    params = params or {}
    ss3m_cfg = cfg.get("ss3m", {}) if isinstance(cfg.get("ss3m", {}), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    pretrained_path = cfg.get("classifier", {}).get("pretrained_path")
    strong_align = bool(cfg.get("classifier", {}).get("strong_align_pretrained", True))
    inferred = _infer_ss3m_hparams_from_ckpt(pretrained_path) if strong_align else {}

    embed_dim = int(params.get("embed_dim", ss3m_cfg.get("embed_dim", inferred.get("embed_dim", model_cfg.get("embed_dim", 96)))))
    depth = int(params.get("depth", ss3m_cfg.get("depth", inferred.get("depth", model_cfg.get("depth", 4)))))
    patch_size = tuple(params.get("patch_size", ss3m_cfg.get("patch_size", inferred.get("patch_size", model_cfg.get("patch_size", (16, 16, 16))))))
    dropout = float(params.get("dropout", ss3m_cfg.get("dropout", 0.0)))

    return {
        "embed_dim": embed_dim,
        "depth": depth,
        "patch_size": patch_size,
        "dropout": dropout,
        "in_channels": in_chans,
        "inferred": inferred,
        "strong_align": strong_align,
    }


def _resolve_imbalance_mode(cfg: dict, sampler_strategy: str, use_class_weights: bool) -> str:
    explicit = str(cfg.get("classifier", {}).get("imbalance_mode", "")).strip().lower()
    if explicit in ("none", "sampler", "class_weight"):
        return explicit
    if sampler_strategy in ("weighted", "oversample", "smote", "adasyn"):
        return "sampler"
    if use_class_weights:
        return "class_weight"
    return "none"

def _build_classifier_model(backbone_choice: str, cfg: dict, in_chans: int, num_classes: int, params: dict):
    if backbone_choice == "medmamba3d":
        embed_dim = int(params.get("embed_dim", cfg.get("model", {}).get("embed_dim", 128)))
        depth = int(params.get("depth", cfg.get("model", {}).get("depth", 12)))
        patch_size = tuple(cfg.get("model", {}).get("patch_size", (16, 16, 16)))
        return MedMamba3D(
            in_chans=in_chans,
            embed_dim=embed_dim,
            depth=depth,
            patch_size=patch_size,
            num_classes=num_classes
        )
    ss3m_arch = _resolve_ss3m_arch(cfg, params, in_chans)
    inferred = ss3m_arch.get("inferred", {})
    if inferred:
        print(
            f"[model][diag] ss3m align from ckpt container={inferred.get('container', 'na')} "
            f"patch_key={inferred.get('patch_key', 'na')} embed_dim={ss3m_arch['embed_dim']} "
            f"depth={ss3m_arch['depth']} patch_size={ss3m_arch['patch_size']}",
            flush=True,
        )
    return MedMambaSS3M(
        in_channels=ss3m_arch["in_channels"],
        embed_dim=ss3m_arch["embed_dim"],
        depth=ss3m_arch["depth"],
        patch_size=ss3m_arch["patch_size"],
        num_classes=num_classes,
        dropout=ss3m_arch["dropout"]
    )

def _train_one_trial_cv(cfg, backbone_choice: str, ds, folds, in_chans: int, target_shape, device, device_type, params, epochs: int, trial_ctx: Optional[dict] = None):
    num_workers = compute_num_workers(cfg)
    collate_fn = make_safe_collate(in_chans, target_shape)
    num_classes_cfg = int(cfg.get('dataset', {}).get('num_classes', 2))
    num_classes = _infer_num_classes_from_items(getattr(ds, "items", []), num_classes_cfg)
    if num_classes != num_classes_cfg:
        print(f"[HPO] num_classes adjusted {num_classes_cfg} -> {num_classes}", flush=True)
    grad_accum = int(cfg.get('classifier', {}).get('grad_accum_steps', 1))
    batch_size = int(params.get("batch_size", cfg.get('classifier', {}).get('batch_size', 1)))
    lr = float(params.get("lr", cfg.get('classifier', {}).get('lr', 1e-4)))
    weight_decay = float(params.get("weight_decay", cfg.get('classifier', {}).get('weight_decay', 0.0)))
    loss_type = str(params.get("loss", cfg.get('classifier', {}).get('loss', 'ce'))).lower()
    sampler_strategy = str(params.get("sampler", cfg.get('classifier', {}).get('sampler', 'none'))).lower()
    use_class_weights = bool(params.get("use_class_weights", cfg.get('classifier', {}).get('use_class_weights', False)))
    imbalance_mode = _resolve_imbalance_mode(cfg, sampler_strategy, use_class_weights)
    if imbalance_mode != "sampler" and sampler_strategy in ("weighted", "oversample", "smote", "adasyn"):
        sampler_strategy = "none"
    aug_cfg = cfg.get("augment", {}) if isinstance(cfg.get("augment", {}), dict) else {}
    do_augment = bool(aug_cfg.get("enable", False) or aug_cfg.get("enabled", False))
    augment_params = aug_cfg if do_augment else {}
    threshold = params.get("threshold", cfg.get('classifier', {}).get('eval_threshold'))
    scheduler_type = str(params.get("scheduler", cfg.get("classifier", {}).get("scheduler", "plateau"))).lower()
    eval_cfg = cfg.get("eval", {}) if isinstance(cfg.get("eval", {}), dict) else {}
    eval_use_tta = bool(eval_cfg.get("use_tta", False))
    aug_cfg = cfg.get("augment", {}) if isinstance(cfg.get("augment", {}), dict) else {}
    do_augment = bool(aug_cfg.get("enable", False) or aug_cfg.get("enabled", False))
    augment_params = aug_cfg if do_augment else {}
    optim_target = "auc"
    early_stop_patience = 0
    early_stop_min_epochs = 1
    early_stop_min_delta = 0.0
    writer = None
    trial_dir = None
    run_id = None
    trial_id = None
    if trial_ctx:
        optim_target = str(trial_ctx.get("optim_target", "auc")).lower()
        early_stop_patience = int(trial_ctx.get("early_stop_patience", 0))
        early_stop_min_epochs = int(trial_ctx.get("early_stop_min_epochs", 1))
        early_stop_min_delta = float(trial_ctx.get("early_stop_min_delta", 0.0))
        writer = trial_ctx.get("writer")
        trial_dir = trial_ctx.get("trial_dir")
        run_id = trial_ctx.get("run_id")
        trial_id = trial_ctx.get("trial_id")
    freeze_epochs = int(cfg.get('classifier', {}).get('freeze_backbone_epochs', 0))
    fold_metrics = []
    auc_traj_all = []
    pr_epochs_all = []
    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        try:
            ds_train = torch.utils.data.Subset(ds, train_idx)
            ds_val = torch.utils.data.Subset(ds, val_idx)
            labels_train = [getattr(ds, 'items', [])[i][1] for i in train_idx]
            class_weights = _class_weights_from_labels(labels_train, num_classes)
            ce_weight = torch.tensor(class_weights, dtype=torch.float32, device=device) if imbalance_mode == "class_weight" else None
            sampler = None
            if imbalance_mode == "sampler" and sampler_strategy in ("weighted", "oversample"):
                sample_weights = [class_weights[y] for y in labels_train]
                sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
            if fold_idx == 0:
                print(f"[Imbalance] mode={imbalance_mode} sampler={sampler_strategy} use_class_weights={imbalance_mode == 'class_weight'}", flush=True)
            train_loader = DataLoader(
                ds_train, batch_size=batch_size, shuffle=(sampler is None),
                sampler=sampler,
                num_workers=num_workers, pin_memory=(device_type == 'cuda'),
                persistent_workers=(num_workers > 0), prefetch_factor=2, collate_fn=collate_fn
            )
            # Enable augmentation for training set in this fold
            if hasattr(ds_train.dataset, 'augment'):
                 ds_train.dataset.augment = True
            
            val_loader = DataLoader(
                ds_val, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=(device_type == 'cuda'),
                persistent_workers=(num_workers > 0), prefetch_factor=2, collate_fn=collate_fn
            )
            model = _build_classifier_model(backbone_choice, cfg, in_chans, num_classes, params).to(device)
            if cfg.get('classifier', {}).get('pretrained_path') and os.path.isfile(cfg['classifier']['pretrained_path']):
                load_pretrained_partial(model, cfg['classifier']['pretrained_path'], in_chans, keep_classifier_if_shape_match=True)
            
            # Linear Probing: Freeze backbone if requested
            if freeze_epochs > 0:
                 _set_backbone_frozen(model, backbone_choice, True)
                 print(f"[Fold{fold_idx+1}] Backbone frozen for {freeze_epochs} epochs.")
            
            opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
            
            # Scheduler initialization
            sched = None
            if scheduler_type == "cosine":
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)
            elif scheduler_type == "plateau":
                sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max' if optim_target in ('auc', 'acc', 'bal_acc', 'f1') else 'min', patience=max(1, early_stop_patience // 2), factor=0.5, verbose=True)

            scaler = torch.amp.GradScaler('cuda') if device_type == 'cuda' else None
            best_score = None
            bad_epochs = 0
            auc_traj = []
            pr_epochs = []
            
            # Metric logger for plotting
            epoch_logs = []
            
            best_fold_score = -float('inf')
            best_fold_metrics = None
            
            # W&B init per fold (optional, might be noisy, better per trial)
            if wandb and wandb.run is not None:
                 wandb.define_metric(f"fold{fold_idx+1}/epoch")
                 wandb.define_metric(f"fold{fold_idx+1}/*", step_metric=f"fold{fold_idx+1}/epoch")

            for epoch in range(epochs):
                # Unfreeze check
                if freeze_epochs > 0 and epoch == freeze_epochs:
                    _set_backbone_frozen(model, backbone_choice, False)
                    print(f"[Fold{fold_idx+1}] Unfreezing backbone at epoch {epoch+1}")
                    # Re-create optimizer with all parameters
                    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
                    # Re-create scheduler
                    if scheduler_type == "cosine":
                        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs - epoch), eta_min=1e-6)
                    elif scheduler_type == "plateau":
                        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max' if optim_target in ('auc', 'acc', 'bal_acc', 'f1') else 'min', patience=max(1, early_stop_patience // 2), factor=0.5, verbose=True)

                model.train()
                epoch_loss = 0.0
                epoch_samples = 0
                current_lr = opt.param_groups[0]['lr']
                opt.zero_grad(set_to_none=True)
                
                for step, batch in enumerate(train_loader):
                    x, y = batch
                    if y.numel() == 0:
                        continue
                    if imbalance_mode == "sampler" and sampler_strategy in ("smote", "adasyn"):
                        x, y = _apply_smote_like(x, y, sampler_strategy)
                    x = x.to(device, dtype=torch.float32, non_blocking=True)
                    y = y.to(device, non_blocking=True)
                    if do_augment:
                        x = augment_3d_batch(x, augment_params)
                    amp_ctx = torch.amp.autocast('cuda', dtype=torch.float16) if device_type == 'cuda' else contextlib.nullcontext()
                    with amp_ctx:
                        logits = model.forward_classifier(x) if hasattr(model, "forward_classifier") else model(x)
                        if loss_type in ("focal", "focal_loss"):
                            loss = _focal_loss(logits, y, gamma=float(params.get("focal_gamma", 2.0)), alpha=ce_weight)
                        elif loss_type == "dice":
                            loss = _dice_loss(logits, y)
                        elif loss_type == "tversky":
                            loss = _tversky_loss(logits, y, alpha=float(params.get("tversky_alpha", 0.5)), beta=float(params.get("tversky_beta", 0.5)))
                        else:
                            loss = F.cross_entropy(logits, y, weight=ce_weight)
                    loss_for_backward = loss / max(1, grad_accum)
                    if scaler is not None:
                        scaler.scale(loss_for_backward).backward()
                        if ((step + 1) % grad_accum == 0) or ((step + 1) == len(train_loader)):
                            scaler.step(opt)
                            scaler.update()
                            opt.zero_grad(set_to_none=True)
                    else:
                        loss_for_backward.backward()
                        if ((step + 1) % grad_accum == 0) or ((step + 1) == len(train_loader)):
                            opt.step()
                            opt.zero_grad(set_to_none=True)
                    
                    epoch_loss += loss.item() * y.size(0)
                    epoch_samples += y.size(0)
                
                avg_train_loss = epoch_loss / max(epoch_samples, 1)
                train_aug_prev = None
                if hasattr(ds_train.dataset, 'augment'):
                    train_aug_prev = bool(ds_train.dataset.augment)
                    ds_train.dataset.augment = False
                train_metrics = evaluate_loader(model, train_loader, device, num_classes, threshold=threshold, return_pr=False, use_tta=False)
                if train_aug_prev is not None:
                    ds_train.dataset.augment = train_aug_prev
                # TTA is only applied to validation to boost performance without training cost
                val_metrics = evaluate_loader(model, val_loader, device, num_classes, threshold=threshold, return_pr=True, use_tta=eval_use_tta)
                
                # Log metrics for plotting
                epoch_logs.append({
                    'epoch': epoch + 1,
                    'train_loss': avg_train_loss,
                    'train_acc': train_metrics['acc'],
                    'val_acc': val_metrics['acc'],
                    'train_auc': train_metrics['auc'],
                    'val_auc': val_metrics['auc']
                })
                
                # Scheduler Step
                if sched:
                    if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        if optim_target == "f1":
                            sched.step(val_metrics["f1"])
                        elif optim_target == "bal_acc":
                            sched.step(val_metrics["bal_acc"])
                        else:
                            sched.step(val_metrics["auc"])
                    else:
                        sched.step()

                print(
                    f"[Fold{fold_idx+1} Epoch {epoch+1}/{epochs}] lr={current_lr:.2e} loss={avg_train_loss:.4f} "
                    f"train_acc={train_metrics['acc']:.4f} train_bac={train_metrics['bal_acc']:.4f} "
                    f"val_acc={val_metrics['acc']:.4f} val_bac={val_metrics['bal_acc']:.4f} val_auc={val_metrics['auc']:.4f} "
                    f"val_sens_pos={val_metrics.get('sens_pos', float('nan')):.4f} val_spec_pos={val_metrics.get('spec_pos', float('nan')):.4f}",
                    flush=True,
                )

                auc_traj.append({
                    "fold": fold_idx + 1,
                    "epoch": epoch + 1,
                    "train_auc": train_metrics["auc"],
                    "val_auc": val_metrics["auc"]
                })
                if val_metrics.get("pr_curve") is not None:
                    pr_epochs.append({
                        "fold": fold_idx + 1,
                        "epoch": epoch + 1,
                        "pr_curve": val_metrics["pr_curve"]
                    })
                
                if writer:
                    writer.add_scalar(f"{loss_type}/fold{fold_idx+1}/lr", current_lr, epoch + 1)
                    writer.add_scalar(f"{loss_type}/fold{fold_idx+1}/train_loss", avg_train_loss, epoch + 1)
                    writer.add_scalar(f"{loss_type}/fold{fold_idx+1}/train_acc", train_metrics["acc"], epoch + 1)
                    writer.add_scalar(f"{loss_type}/fold{fold_idx+1}/val_acc", val_metrics["acc"], epoch + 1)
                    writer.add_scalar(f"{loss_type}/fold{fold_idx+1}/val_auc", val_metrics["auc"], epoch + 1)
                    if val_metrics.get("raw") is not None and num_classes == 2:
                         y_true_np = np.asarray(val_metrics["raw"]["y_true"], dtype=np.int64)
                         y_prob_np = np.asarray(val_metrics["raw"]["y_prob"], dtype=np.float64)
                         if y_prob_np.ndim == 2:
                            writer.add_pr_curve(
                                f"{loss_type}/fold{fold_idx+1}/pr",
                                torch.tensor(y_true_np, dtype=torch.int64),
                                torch.tensor(y_prob_np[:, 1], dtype=torch.float32),
                                epoch + 1
                            )

                if wandb and wandb.run is not None:
                    wandb.log({
                        f"fold{fold_idx+1}/epoch": epoch + 1,
                        f"fold{fold_idx+1}/lr": current_lr,
                        f"fold{fold_idx+1}/train_loss": avg_train_loss,
                        f"fold{fold_idx+1}/train_acc": train_metrics["acc"],
                        f"fold{fold_idx+1}/val_acc": val_metrics["acc"],
                        f"fold{fold_idx+1}/val_auc": val_metrics["auc"]
                    })

                if optim_target == "f1":
                    score = val_metrics["f1"]
                elif optim_target == "bal_acc":
                    score = val_metrics["bal_acc"]
                else:
                    score = val_metrics["auc"]
                
                # Update best fold metrics
                if score > best_fold_score:
                    best_fold_score = score
                    best_fold_metrics = val_metrics
                
                if best_score is None or (score - best_score) > early_stop_min_delta:
                    best_score = score
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                if early_stop_patience > 0 and (epoch + 1) >= early_stop_min_epochs and bad_epochs >= early_stop_patience:
                    print(f"[Fold{fold_idx+1}] Early stopping at epoch {epoch+1} (best_score={best_score:.4f})", flush=True)
                    break
            if auc_traj:
                auc_traj_all.extend(auc_traj)
            if pr_epochs:
                pr_epochs_all.extend(pr_epochs)
            
            # Generate plots for this fold
            if epoch_logs:
                plot_path = os.path.join(trial_dir if trial_dir else ".", f"metrics_fold_{fold_idx + 1}.png")
                plot_training_curves(epoch_logs, plot_path)

            # Use best metrics recorded during training instead of last epoch metrics
            fold_metrics.append(best_fold_metrics if best_fold_metrics else val_metrics)
        except RuntimeError as e:
            msg = str(e).lower()
            if "cublas_status_alloc_failed" in msg or "out of memory" in msg or "cuda error" in msg:
                if device_type == "cuda":
                    torch.cuda.empty_cache()
                return float("nan"), float("nan"), float("nan"), []
            raise
    if trial_dir and run_id is not None and trial_id is not None:
        loss_tag = loss_type
        _write_trial_curves(trial_dir, loss_tag, run_id, trial_id, pr_epochs_all, auc_traj_all)
    mean_acc = float(np.mean([m["acc"] for m in fold_metrics])) if fold_metrics else float("nan")
    mean_bal = float(np.mean([m["bal_acc"] for m in fold_metrics])) if fold_metrics else float("nan")
    mean_auc = float(np.mean([m["auc"] for m in fold_metrics])) if fold_metrics else float("nan")
    return mean_acc, mean_bal, mean_auc, fold_metrics

def _train_final_model(cfg, backbone_choice: str, ds, in_chans: int, target_shape, device, device_type, params, epochs: int, out_path: str):
    num_workers = compute_num_workers(cfg)
    collate_fn = make_safe_collate(in_chans, target_shape)
    num_classes_cfg = int(cfg.get('dataset', {}).get('num_classes', 2))
    num_classes = _infer_num_classes_from_items(getattr(ds, "items", []), num_classes_cfg)
    if num_classes != num_classes_cfg:
        print(f"[Train] num_classes adjusted {num_classes_cfg} -> {num_classes}", flush=True)
    grad_accum = int(cfg.get('classifier', {}).get('grad_accum_steps', 1))
    batch_size = int(params.get("batch_size", cfg.get('classifier', {}).get('batch_size', 1)))
    lr = float(params.get("lr", cfg.get('classifier', {}).get('lr', 1e-4)))
    weight_decay = float(params.get("weight_decay", cfg.get('classifier', {}).get('weight_decay', 0.0)))
    loss_type = str(params.get("loss", cfg.get('classifier', {}).get('loss', 'ce'))).lower()
    sampler_strategy = str(params.get("sampler", cfg.get('classifier', {}).get('sampler', 'none'))).lower()
    use_class_weights = bool(params.get("use_class_weights", cfg.get('classifier', {}).get('use_class_weights', False)))
    imbalance_mode = _resolve_imbalance_mode(cfg, sampler_strategy, use_class_weights)
    if imbalance_mode != "sampler" and sampler_strategy in ("weighted", "oversample", "smote", "adasyn"):
        sampler_strategy = "none"
    labels_all = [y for _, y in getattr(ds, 'items', [])]
    class_weights = _class_weights_from_labels(labels_all, num_classes)
    ce_weight = torch.tensor(class_weights, dtype=torch.float32, device=device) if imbalance_mode == "class_weight" else None
    sampler = None
    if imbalance_mode == "sampler" and sampler_strategy in ("weighted", "oversample"):
        sample_weights = [class_weights[y] for y in labels_all]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    print(f"[Imbalance] mode={imbalance_mode} sampler={sampler_strategy} use_class_weights={imbalance_mode == 'class_weight'}", flush=True)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler,
        num_workers=num_workers, pin_memory=(device_type == 'cuda'),
        persistent_workers=(num_workers > 0), prefetch_factor=2, collate_fn=collate_fn
    )
    model = _build_classifier_model(backbone_choice, cfg, in_chans, num_classes, params).to(device)
    if cfg.get('classifier', {}).get('pretrained_path') and os.path.isfile(cfg['classifier']['pretrained_path']):
        load_pretrained_partial(model, cfg['classifier']['pretrained_path'], in_chans, keep_classifier_if_shape_match=True)
    
    freeze_epochs = int(cfg.get('classifier', {}).get('freeze_backbone_epochs', 0))
    if freeze_epochs > 0:
        _set_backbone_frozen(model, backbone_choice, True)
        print(f"[Train] Backbone frozen for {freeze_epochs} epochs.")

    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler('cuda') if device_type == 'cuda' else None
    for epoch in range(epochs):
        if freeze_epochs > 0 and epoch == freeze_epochs:
            _set_backbone_frozen(model, backbone_choice, False)
            print(f"[Train] Unfreezing backbone at epoch {epoch+1}")
            opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        model.train()
        opt.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            x, y = batch
            if y.numel() == 0:
                continue
            if imbalance_mode == "sampler" and sampler_strategy in ("smote", "adasyn"):
                x, y = _apply_smote_like(x, y, sampler_strategy)
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if do_augment:
                x = augment_3d_batch(x, augment_params)
            amp_ctx = torch.amp.autocast('cuda', dtype=torch.float16) if device_type == 'cuda' else contextlib.nullcontext()
            with amp_ctx:
                logits = model.forward_classifier(x) if hasattr(model, "forward_classifier") else model(x)
                if loss_type in ("focal", "focal_loss"):
                    loss = _focal_loss(logits, y, gamma=float(params.get("focal_gamma", 2.0)), alpha=ce_weight)
                elif loss_type == "dice":
                    loss = _dice_loss(logits, y)
                elif loss_type == "tversky":
                    loss = _tversky_loss(logits, y, alpha=float(params.get("tversky_alpha", 0.5)), beta=float(params.get("tversky_beta", 0.5)))
                else:
                    loss = F.cross_entropy(logits, y, weight=ce_weight)
            loss_for_backward = loss / max(1, grad_accum)
            if scaler is not None:
                scaler.scale(loss_for_backward).backward()
                if ((step + 1) % grad_accum == 0) or ((step + 1) == len(loader)):
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
            else:
                loss_for_backward.backward()
                if ((step + 1) % grad_accum == 0) or ((step + 1) == len(loader)):
                    opt.step()
                    opt.zero_grad(set_to_none=True)
    torch.save({"state_dict": model.state_dict(), "epoch": epochs}, out_path)
    return out_path

def _write_trial_curves(trial_dir: str, loss_name: str, run_id: str, trial_id: int, pr_epochs: list, auc_traj: list):
    os.makedirs(trial_dir, exist_ok=True)
    base = f"{loss_name}_{run_id}_trial{trial_id}"
    pr_json_path = os.path.join(trial_dir, f"pr_curve_{base}.json")
    pr_csv_path = os.path.join(trial_dir, f"pr_curve_{base}.csv")
    auc_json_path = os.path.join(trial_dir, f"auc_traj_{base}.json")
    auc_csv_path = os.path.join(trial_dir, f"auc_traj_{base}.csv")
    with open(pr_json_path, "w", encoding="utf-8") as f:
        json.dump(pr_epochs, f, ensure_ascii=False, indent=2)
    with open(auc_json_path, "w", encoding="utf-8") as f:
        json.dump(auc_traj, f, ensure_ascii=False, indent=2)
    with open(pr_csv_path, "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["fold", "epoch", "threshold", "precision", "recall"])
        for item in pr_epochs:
            fold = item.get("fold")
            epoch = item.get("epoch")
            pr = item.get("pr_curve", {})
            th = pr.get("thresholds", [])
            prec = pr.get("precision", [])
            rec = pr.get("recall", [])
            for t, p, r in zip(th, prec, rec):
                w.writerow([fold, epoch, t, p, r])
    with open(auc_csv_path, "w", newline="", encoding="utf-8") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["fold", "epoch", "train_auc", "val_auc"])
        for item in auc_traj:
            w.writerow([item.get("fold"), item.get("epoch"), item.get("train_auc"), item.get("val_auc")])

def _generate_grid_candidates(space: dict):
    # 将search_space展开为网格
    # 对于 continuous (uniform, loguniform)，如果未提供步长，我们默认采样3个点 (min, mid, max)
    # 对于 choice，直接展开
    keys = sorted(list(space.keys()))
    values_list = []
    for k in keys:
        v = space[k]
        if isinstance(v, dict):
            v_type = v.get("type", "")
            if v_type == "choice":
                values_list.append(v.get("values", []))
            elif v_type in ("uniform", "loguniform"):
                # 简单离散化：取 min, (min+max)/2, max
                vmin, vmax = float(v["min"]), float(v["max"])
                if v_type == "loguniform":
                    # log scale: min, sqrt(min*max), max
                    mid = math.sqrt(vmin * vmax)
                    values_list.append([vmin, mid, vmax])
                else:
                    mid = (vmin + vmax) / 2.0
                    values_list.append([vmin, mid, vmax])
            else:
                values_list.append([v.get("value")])
        elif isinstance(v, (list, tuple)):
             # 列表/元组视为 choice
             values_list.append(list(v))
        else:
             values_list.append([v])
    
    candidates = []
    for combination in itertools.product(*values_list):
        params = dict(zip(keys, combination))
        candidates.append(params)
    return candidates

def _write_text_report(path: str, run_id: str, summary: dict, fold_reports: list):
    lines = []
    lines.append(f"====== Training Report {run_id} ======")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("-" * 40)
    lines.append("Best Hyperparameters:")
    best = summary.get("best", {})
    if best.get("params"):
        for k, v in best["params"].items():
            lines.append(f"  {k}: {v}")
    lines.append("-" * 40)
    lines.append("Performance Metrics (Best Model):")
    lines.append(f"  ACC: {_fmt_mean_std(best.get('mean_acc'), best.get('std_acc'))}")
    lines.append(f"  BAC: {_fmt_mean_std(best.get('mean_bal_acc'), best.get('std_bal_acc'))}")
    lines.append(f"  AUC: {_fmt_mean_std(best.get('mean_auc'), best.get('std_auc'))}")
    lines.append(f"  Sens (pos=1): {_fmt_mean_std(best.get('mean_sens_pos'), best.get('std_sens_pos'))}")
    lines.append(f"  Spec (pos=1): {_fmt_mean_std(best.get('mean_spec_pos'), best.get('std_spec_pos'))}")
    lines.append(f"  Sens (macro): {_fmt_mean_std(best.get('mean_sens_macro'), best.get('std_sens_macro'))}")
    lines.append(f"  Spec (macro): {_fmt_mean_std(best.get('mean_spec_macro'), best.get('std_spec_macro'))}")
    lines.append(f"  F1-Score: {_fmt_mean_std(best.get('mean_f1'), best.get('std_f1'))}")
    lines.append(f"  Kappa: {_fmt_mean_std(best.get('mean_kappa'), best.get('std_kappa'))}")
    lines.append(f"  MCC: {_fmt_mean_std(best.get('mean_mcc'), best.get('std_mcc'))}")
    lines.append("-" * 40)
    lines.append("Baseline Comparison:")
    baseline = summary.get("baseline", {})
    lines.append(f"  Baseline ACC: {_fmt_mean_std(baseline.get('mean_acc'), baseline.get('std_acc'))}")
    lines.append(f"  Baseline BAC: {_fmt_mean_std(baseline.get('mean_bal_acc'), baseline.get('std_bal_acc'))}")
    lines.append(f"  Baseline AUC: {_fmt_mean_std(baseline.get('mean_auc'), baseline.get('std_auc'))}")
    lines.append(f"  Improvement (Acc): {summary.get('improve_ratio', 'N/A')}")
    lines.append("-" * 40)
    
    if fold_reports:
        lines.append("Detailed Fold Reports (with TTA):")
        # Collect AUCs for variance analysis
        aucs = []
        for rep in fold_reports:
            if rep.get("auc") is not None:
                aucs.append(rep["auc"])
            elif rep.get("val_auc") is not None:
                aucs.append(rep["val_auc"])
        
        if aucs:
            mean_auc = np.mean(aucs)
            std_auc = np.std(aucs)
            lines.append(f"Fold AUC Stats: Mean={mean_auc:.4f}, Std={std_auc:.4f}, Range=[{min(aucs):.4f}, {max(aucs):.4f}]")
            # Simple ASCII Box Plot
            lines.append(f"AUC Box Plot: |-{min(aucs):.2f}--[{mean_auc-std_auc:.2f}=={mean_auc:.2f}=={mean_auc+std_auc:.2f}]--{max(aucs):.2f}-|")
            lines.append("-" * 20)

        for i, rep in enumerate(fold_reports):
            lines.append(f"  Fold {i+1}:")
            cm = rep.get("cm")
            if cm:
                lines.append("    Confusion Matrix:")
                for row in cm:
                    lines.append(f"      {row}")
            lines.append(f"    Kappa: {rep.get('kappa')}")
            lines.append(f"    MCC: {rep.get('mcc')}")
            if rep.get("sens_pos") is not None:
                lines.append(f"    Sens (pos=1): {rep.get('sens_pos')}")
            if rep.get("spec_pos") is not None:
                lines.append(f"    Spec (pos=1): {rep.get('spec_pos')}")
            if rep.get("sens_macro") is not None:
                lines.append(f"    Sens (macro): {rep.get('sens_macro')}")
            if rep.get("spec_macro") is not None:
                lines.append(f"    Spec (macro): {rep.get('spec_macro')}")
            # Try to get AUC from report or infer from context if available
            if rep.get("auc"):
                 lines.append(f"    AUC: {rep.get('auc')}")
    
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return path

def _svg_line(points, width=320, height=220, pad=10):
    if not points:
        return ""
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    dx = max(max_x - min_x, 1e-9)
    dy = max(max_y - min_y, 1e-9)
    coords = []
    for x, y in points:
        px = pad + (x - min_x) / dx * (width - 2 * pad)
        py = height - pad - (y - min_y) / dy * (height - 2 * pad)
        coords.append(f"{px:.2f},{py:.2f}")
    poly = " ".join(coords)
    return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"><polyline fill="none" stroke="#2c7" stroke-width="2" points="{poly}"/></svg>'

def _write_html_report(exp_dir: str, run_id: str, optim_target: str):
    results_path = os.path.join(exp_dir, "hpo_results.csv")
    summary_path = os.path.join(exp_dir, "hpo_best.json")
    rows = []
    if os.path.isfile(results_path):
        import csv
        with open(results_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    summary = {}
    if os.path.isfile(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
    best_by_loss = {}
    for r in rows:
        try:
            params = json.loads(r.get("params", "{}"))
        except Exception:
            params = {}
        loss = str(params.get("loss", "ce"))
        if optim_target == "f1":
            score = float(r.get("mean_f1", "nan"))
        elif optim_target == "bal_acc":
            score = float(r.get("mean_bal_acc", "nan"))
        else:
            score = float(r.get("mean_auc", "nan"))
        cur = best_by_loss.get(loss)
        if cur is None or score > cur["score"]:
            best_by_loss[loss] = {"row": r, "score": score}
    sections = []
    for loss, info in best_by_loss.items():
        row = info["row"]
        trial_id = row.get("trial_id", "0")
        trial_dir = os.path.join(exp_dir, f"trial_{int(trial_id):03d}")
        auc_files = [p for p in os.listdir(trial_dir) if p.startswith(f"auc_traj_{loss}_{run_id}") and p.endswith(".json")] if os.path.isdir(trial_dir) else []
        pr_files = [p for p in os.listdir(trial_dir) if p.startswith(f"pr_curve_{loss}_{run_id}") and p.endswith(".json")] if os.path.isdir(trial_dir) else []
        auc_svg = ""
        pr_svg = ""
        if auc_files:
            with open(os.path.join(trial_dir, auc_files[0]), "r", encoding="utf-8") as f:
                auc_data = json.load(f)
            pts = [(d.get("epoch", 0), d.get("val_auc", 0.0)) for d in auc_data]
            auc_svg = _svg_line(pts)
        if pr_files:
            with open(os.path.join(trial_dir, pr_files[0]), "r", encoding="utf-8") as f:
                pr_data = json.load(f)
            if pr_data:
                pr = pr_data[-1].get("pr_curve", {})
                pts = list(zip(pr.get("recall", []), pr.get("precision", [])))
                pr_svg = _svg_line(pts)
        sections.append(f"<h3>loss={loss}</h3><div>trial={trial_id}</div>{auc_svg}{pr_svg}")
    html = f"""<html><head><meta charset="utf-8"></head><body>
<h1>HPO Report {run_id}</h1>
<pre>{json.dumps(summary, ensure_ascii=False, indent=2)}</pre>
{''.join(sections)}
<button onclick="window.print()">Export PDF</button>
</body></html>"""
    html_path = os.path.join(exp_dir, f"report_{run_id}.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)
    return html_path

def _write_simple_pdf(path: str, lines: list):
    def esc(s):
        return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    y = 760
    content_lines = []
    for line in lines:
        content_lines.append(f"72 {y} Td ({esc(line)}) Tj")
        y -= 16
    content = "BT /F1 12 Tf " + " ".join(content_lines) + " ET"
    objects = []
    objects.append("1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj")
    objects.append("2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj")
    objects.append("3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 5 0 R /Resources << /Font << /F1 4 0 R >> >> >> endobj")
    objects.append("4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj")
    objects.append(f"5 0 obj << /Length {len(content)} >> stream {content} endstream endobj")
    xref = ["xref", "0 6", "0000000000 65535 f "]
    offset = 0
    body = ""
    for obj in objects:
        xref.append(f"{offset:010d} 00000 n ")
        body += obj + "\n"
        offset += len(obj) + 1
    trailer = f"trailer << /Size 6 /Root 1 0 R >> startxref {offset} %%EOF"
    pdf = "%PDF-1.4\n" + body + "\n".join(xref) + "\n" + trailer
    with open(path, "wb") as f:
        f.write(pdf.encode("latin-1"))

def _run_hpo(cfg, backbone_choice: str, ds, in_chans: int, target_shape, device, device_type):
    hpo_cfg = cfg.get("hpo", {})
    out_dir = cfg.get("paths", {}).get("out_dir", ".")
    os.makedirs(out_dir, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    runs_root = os.path.join(out_dir, str(hpo_cfg.get("runs_dir", "runs")))
    exp_dir = os.path.join(runs_root, f"exp_{run_id}")
    os.makedirs(exp_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=exp_dir) if bool(hpo_cfg.get("use_tensorboard", True)) else None
    num_classes_cfg = int(cfg.get("dataset", {}).get("num_classes", 2))
    num_classes = _infer_num_classes_from_items(getattr(ds, "items", []), num_classes_cfg)
    if num_classes != num_classes_cfg:
        print(f"[HPO] num_classes adjusted {num_classes_cfg} -> {num_classes}", flush=True)
    folds = _build_folds(getattr(ds, "items", []), num_classes=num_classes,
                         n_splits=int(hpo_cfg.get("folds", 5)), seed=int(hpo_cfg.get("seed", 42)))
    search_space = hpo_cfg.get("search_space", {})
    if "loss" not in search_space and hpo_cfg.get("losses"):
        search_space["loss"] = hpo_cfg.get("losses")
    trials = int(hpo_cfg.get("trials", 10))
    max_epochs = int(hpo_cfg.get("max_epochs", cfg.get("classifier", {}).get("epochs", 10)))
    eta = int(hpo_cfg.get("eta", 3))
    rng = np.random.default_rng(int(hpo_cfg.get("seed", 42)))
    results_path = os.path.join(exp_dir, "hpo_results.csv")
    if not os.path.isfile(results_path):
        with open(results_path, "w", newline="", encoding="utf-8") as f:
            import csv
            w = csv.writer(f)
            w.writerow([
                "trial_id", "stage", "epochs",
                "mean_acc", "std_acc", "mean_bal_acc", "std_bal_acc", "mean_auc", "std_auc",
                "mean_f1", "std_f1", "mean_kappa", "std_kappa", "mean_mcc", "std_mcc",
                "mean_sens_pos", "std_sens_pos", "mean_spec_pos", "std_spec_pos",
                "mean_sens_macro", "std_sens_macro", "mean_spec_macro", "std_spec_macro",
                "params", "fold_reports"
            ])
    best = {
        "score": -1.0, "params": None,
        "mean_acc": None, "std_acc": None,
        "mean_bal_acc": None, "std_bal_acc": None,
        "mean_auc": None, "std_auc": None,
        "mean_f1": None, "std_f1": None,
        "mean_kappa": None, "std_kappa": None,
        "mean_mcc": None, "std_mcc": None,
        "mean_sens_pos": None, "std_sens_pos": None,
        "mean_spec_pos": None, "std_spec_pos": None,
        "mean_sens_macro": None, "std_sens_macro": None,
        "mean_spec_macro": None, "std_spec_macro": None,
    }
    algorithm = str(hpo_cfg.get("algorithm", "hyperband")).lower()
    optim_target = str(hpo_cfg.get("optim_target", "auc")).lower()
    early_stop_patience = int(hpo_cfg.get("early_stop_patience", 0))
    early_stop_min_epochs = int(hpo_cfg.get("early_stop_min_epochs", 1))
    early_stop_min_delta = float(hpo_cfg.get("early_stop_min_delta", 0.0))
    def record_trial(trial_id, stage, epochs, params, mean_acc, mean_bal, mean_auc, fold_metrics):
        stats = _summarize_fold_metrics(fold_metrics)
        with open(results_path, "a", newline="", encoding="utf-8") as f:
            import csv
            w = csv.writer(f)
            fold_reports = []
            for m in fold_metrics:
                fold_reports.append({
                    "cm": m.get("cm").tolist() if isinstance(m.get("cm"), np.ndarray) else m.get("cm"),
                    "kappa": m.get("kappa"),
                    "mcc": m.get("mcc"),
                    "pr_curve": m.get("pr_curve"),
                    "sens_pos": m.get("sens_pos"),
                    "spec_pos": m.get("spec_pos"),
                    "sens_macro": m.get("sens_macro"),
                    "spec_macro": m.get("spec_macro"),
                })
            params_safe = _to_py(copy.deepcopy(params))
            fold_reports_safe = _to_py(copy.deepcopy(fold_reports))
            w.writerow([
                trial_id, stage, epochs,
                stats.get("mean_acc"), stats.get("std_acc"),
                stats.get("mean_bal_acc"), stats.get("std_bal_acc"),
                stats.get("mean_auc"), stats.get("std_auc"),
                stats.get("mean_f1"), stats.get("std_f1"),
                stats.get("mean_kappa"), stats.get("std_kappa"),
                stats.get("mean_mcc"), stats.get("std_mcc"),
                stats.get("mean_sens_pos"), stats.get("std_sens_pos"),
                stats.get("mean_spec_pos"), stats.get("std_spec_pos"),
                stats.get("mean_sens_macro"), stats.get("std_sens_macro"),
                stats.get("mean_spec_macro"), stats.get("std_spec_macro"),
                json.dumps(params_safe, ensure_ascii=False), json.dumps(fold_reports_safe, ensure_ascii=False)
            ])
    baseline_epochs = int(hpo_cfg.get("baseline_epochs", max_epochs))
    if backbone_choice == "medmamba3d":
        baseline_depth = int(cfg.get("model", {}).get("depth", 12))
        baseline_embed_dim = int(cfg.get("model", {}).get("embed_dim", 128))
    else:
        baseline_depth = int(cfg.get("ss3m", {}).get("depth", 4))
        baseline_embed_dim = int(cfg.get("ss3m", {}).get("embed_dim", 96))
    baseline_params = {
        "lr": float(cfg.get("classifier", {}).get("lr", 1e-4)),
        "weight_decay": float(cfg.get("classifier", {}).get("weight_decay", 0.0)),
        "batch_size": int(cfg.get("classifier", {}).get("batch_size", 1)),
        "depth": baseline_depth,
        "embed_dim": baseline_embed_dim,
        "dropout": float(cfg.get("ss3m", {}).get("dropout", 0.0)),
        "loss": str(cfg.get("classifier", {}).get("loss", "ce")),
        "focal_gamma": float(cfg.get("classifier", {}).get("focal_gamma", 2.0)),
        "tversky_alpha": float(cfg.get("classifier", {}).get("tversky_alpha", 0.5)),
        "tversky_beta": float(cfg.get("classifier", {}).get("tversky_beta", 0.5)),
        "use_class_weights": bool(cfg.get("classifier", {}).get("use_class_weights", False)),
        "sampler": str(cfg.get("classifier", {}).get("sampler", "none")),
        "threshold": cfg.get("classifier", {}).get("eval_threshold")
    }
    baseline_ctx = {
        "writer": writer, "trial_dir": os.path.join(exp_dir, "trial_-01"),
        "run_id": run_id, "trial_id": -1, "optim_target": optim_target,
        "early_stop_patience": early_stop_patience, "early_stop_min_epochs": early_stop_min_epochs,
        "early_stop_min_delta": early_stop_min_delta
    }
    base_acc, base_bal, base_auc, base_fold = _train_one_trial_cv(
        cfg, backbone_choice, ds, folds, in_chans, target_shape, device, device_type, baseline_params, baseline_epochs, baseline_ctx
    )
    base_stats = _summarize_fold_metrics(base_fold)
    record_trial(-1, "baseline", baseline_epochs, baseline_params, base_acc, base_bal, base_auc, base_fold)
    target_improve = float(hpo_cfg.get("target_improve", 0.15))
    observations = []
    trial_id = 0
    def update_best(params, stats, fold_metrics):
        mean_f1 = stats.get("mean_f1", float("nan"))
        mean_bal = stats.get("mean_bal_acc", float("nan"))
        mean_auc = stats.get("mean_auc", float("nan"))
        if optim_target == "f1":
            score = mean_f1
        elif optim_target == "bal_acc":
            score = mean_bal
        else:
            score = mean_auc
        nonlocal best
        if score > best["score"]:
            best = {
                "score": score, "params": params,
                "mean_acc": stats.get("mean_acc"), "std_acc": stats.get("std_acc"),
                "mean_bal_acc": stats.get("mean_bal_acc"), "std_bal_acc": stats.get("std_bal_acc"),
                "mean_auc": stats.get("mean_auc"), "std_auc": stats.get("std_auc"),
                "mean_f1": stats.get("mean_f1"), "std_f1": stats.get("std_f1"),
                "mean_kappa": stats.get("mean_kappa"), "std_kappa": stats.get("std_kappa"),
                "mean_mcc": stats.get("mean_mcc"), "std_mcc": stats.get("std_mcc"),
                "mean_sens_pos": stats.get("mean_sens_pos"), "std_sens_pos": stats.get("std_sens_pos"),
                "mean_spec_pos": stats.get("mean_spec_pos"), "std_spec_pos": stats.get("std_spec_pos"),
                "mean_sens_macro": stats.get("mean_sens_macro"), "std_sens_macro": stats.get("std_sens_macro"),
                "mean_spec_macro": stats.get("mean_spec_macro"), "std_spec_macro": stats.get("std_spec_macro"),
            }
    def target_met():
        if base_bal <= 0 or base_auc <= 0:
            return False
        improve_bal = (best["mean_bal_acc"] - base_bal) / base_bal
        improve_auc = (best["mean_auc"] - base_auc) / base_auc
        return improve_bal >= target_improve and improve_auc >= target_improve
    def run_trial(params, stage, epochs):
        nonlocal trial_id
        trial_dir = os.path.join(exp_dir, f"trial_{trial_id:03d}")
        trial_ctx = {
            "writer": writer, "trial_dir": trial_dir, "run_id": run_id, "trial_id": trial_id,
            "optim_target": optim_target, "early_stop_patience": early_stop_patience,
            "early_stop_min_epochs": early_stop_min_epochs, "early_stop_min_delta": early_stop_min_delta
        }
        mean_acc, mean_bal, mean_auc, fold_metrics = _train_one_trial_cv(
            cfg, backbone_choice, ds, folds, in_chans, target_shape, device, device_type, params, epochs, trial_ctx
        )
        stats = _summarize_fold_metrics(fold_metrics)
        record_trial(trial_id, stage, epochs, params, mean_acc, mean_bal, mean_auc, fold_metrics)
        if optim_target == "f1":
            obs_score = stats.get("mean_f1", float("nan"))
        elif optim_target == "bal_acc":
            obs_score = stats.get("mean_bal_acc", float("nan"))
        else:
            obs_score = stats.get("mean_auc", float("nan"))
        observations.append((params, obs_score))
        update_best(params, stats, fold_metrics)
        trial_id += 1
        return target_met()
    if algorithm == "grid":
        candidates = _generate_grid_candidates(search_space)
        print(f"[HPO] Grid Search: {len(candidates)} candidates generated.")
        for params in candidates:
            if trial_id >= trials:
                break
            if run_trial(params, "grid", max_epochs):
                break

    if algorithm in ("hyperband", "three_stage"):
        s_max = int(math.log(max_epochs, eta)) if max_epochs > 1 else 0
        B = (s_max + 1) * max_epochs
        for s in reversed(range(s_max + 1)):
            n = int(math.ceil(B / max_epochs / (s + 1) * eta ** s))
            r = int(max_epochs * eta ** (-s))
            configs = [_sample_hpo_params(search_space, rng) for _ in range(min(n, trials))]
            for i in range(s + 1):
                n_i = int(max(1, math.floor(n * eta ** (-i))))
                r_i = int(max(1, r * eta ** i))
                scores = []
                for params in configs[:n_i]:
                    if trial_id >= trials:
                        break
                    if run_trial(params, f"hb_s{s}_i{i}", r_i):
                        break
                    scores.append((observations[-1][1], params))
                scores.sort(key=lambda x: x[0], reverse=True)
                keep = max(1, int(len(scores) / eta))
                configs = [p for _, p in scores[:keep]]
                if trial_id >= trials or target_met():
                    break
            if trial_id >= trials or target_met():
                break
    if algorithm in ("bayes", "three_stage") and not target_met():
        bayes_trials = int(hpo_cfg.get("bayes_trials", max(1, trials // 3)))
        for _ in range(bayes_trials):
            if trial_id >= trials:
                break
            params = _propose_bayes(search_space, observations, rng)
            if run_trial(params, "bayes", max_epochs):
                break
    if algorithm in ("genetic", "three_stage") and not target_met():
        ga_trials = int(hpo_cfg.get("ga_trials", max(1, trials // 3)))
        population = [p for p, _ in observations[-max(4, ga_trials):]]
        while len(population) < 4:
            population.append(_sample_hpo_params(search_space, rng))
        for _ in range(ga_trials):
            if trial_id >= trials:
                break
            parents = rng.choice(len(population), size=2, replace=False)
            p1, p2 = population[parents[0]], population[parents[1]]
            child = {}
            for k in search_space.keys():
                child[k] = p1.get(k) if rng.random() < 0.5 else p2.get(k)
                if rng.random() < float(hpo_cfg.get("ga_mutate_prob", 0.2)):
                    child[k] = _sample_hpo_params({k: search_space[k]}, rng)[k]
            if run_trial(child, "genetic", max_epochs):
                break
            population.append(child)
            population = population[-max(6, len(population))]
    improve_ratio = None if base_acc <= 0 else float((best["mean_acc"] - base_acc) / base_acc)
    improve_bal = None if base_bal <= 0 else float((best["mean_bal_acc"] - base_bal) / base_bal)
    improve_auc = None if base_auc <= 0 else float((best["mean_auc"] - base_auc) / base_auc)
    summary = {
        "run_id": run_id,
        "exp_dir": exp_dir,
        "optim_target": optim_target,
        "best": best,
        "baseline": {
            "mean_acc": base_stats.get("mean_acc"), "std_acc": base_stats.get("std_acc"),
            "mean_bal_acc": base_stats.get("mean_bal_acc"), "std_bal_acc": base_stats.get("std_bal_acc"),
            "mean_auc": base_stats.get("mean_auc"), "std_auc": base_stats.get("std_auc"),
            "mean_sens_pos": base_stats.get("mean_sens_pos"), "std_sens_pos": base_stats.get("std_sens_pos"),
            "mean_spec_pos": base_stats.get("mean_spec_pos"), "std_spec_pos": base_stats.get("std_spec_pos"),
            "mean_sens_macro": base_stats.get("mean_sens_macro"), "std_sens_macro": base_stats.get("std_sens_macro"),
            "mean_spec_macro": base_stats.get("mean_spec_macro"), "std_spec_macro": base_stats.get("std_spec_macro"),
        },
        "target_improve": target_improve,
        "improve_ratio": improve_ratio,
        "improve_bal": improve_bal,
        "improve_auc": improve_auc,
        "target_met": (improve_bal is not None and improve_auc is not None and improve_bal >= target_improve and improve_auc >= target_improve)
    }
    summary_path = os.path.join(exp_dir, "hpo_best.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        summary_safe = _to_py(copy.deepcopy(summary))
        json.dump(summary_safe, f, ensure_ascii=False, indent=2)
    print(f"[HPO] best params saved -> {summary_path}", flush=True)
    print(
        "[HPO] best "
        f"ACC={_fmt_mean_std(best.get('mean_acc'), best.get('std_acc'))} "
        f"BAC={_fmt_mean_std(best.get('mean_bal_acc'), best.get('std_bal_acc'))} "
        f"AUC={_fmt_mean_std(best.get('mean_auc'), best.get('std_auc'))} "
        f"Sens(pos)= {_fmt_mean_std(best.get('mean_sens_pos'), best.get('std_sens_pos'))} "
        f"Spec(pos)= {_fmt_mean_std(best.get('mean_spec_pos'), best.get('std_spec_pos'))} "
        f"Sens(macro)= {_fmt_mean_std(best.get('mean_sens_macro'), best.get('std_sens_macro'))} "
        f"Spec(macro)= {_fmt_mean_std(best.get('mean_spec_macro'), best.get('std_spec_macro'))}"
        ,
        flush=True,
    )
    if improve_ratio is not None:
        print(
            "[HPO] baseline "
            f"ACC={_fmt_mean_std(base_stats.get('mean_acc'), base_stats.get('std_acc'))} "
            f"BAC={_fmt_mean_std(base_stats.get('mean_bal_acc'), base_stats.get('std_bal_acc'))} "
            f"AUC={_fmt_mean_std(base_stats.get('mean_auc'), base_stats.get('std_auc'))} "
            f"improve_ratio={improve_ratio:.3f} target={target_improve:.3f}",
            flush=True,
        )
    if best["params"] is not None:
        final_folds = int(hpo_cfg.get("final_folds", 10))
        final_epochs = int(hpo_cfg.get("final_epochs", cfg.get("classifier", {}).get("epochs", max_epochs)))
        if final_folds > 1:
            final_fold_list = _build_folds(getattr(ds, "items", []), num_classes=num_classes,
                                           n_splits=final_folds, seed=int(hpo_cfg.get("seed", 42)))
            f_acc, f_bal, f_auc, f_fold = _train_one_trial_cv(
                cfg, backbone_choice, ds, final_fold_list, in_chans, target_shape, device, device_type, best["params"], final_epochs
            )
            f_stats = _summarize_fold_metrics(f_fold)
            final_report = {
                "mean_acc": f_stats.get("mean_acc"), "std_acc": f_stats.get("std_acc"),
                "mean_bal_acc": f_stats.get("mean_bal_acc"), "std_bal_acc": f_stats.get("std_bal_acc"),
                "mean_auc": f_stats.get("mean_auc"), "std_auc": f_stats.get("std_auc"),
                "mean_sens_pos": f_stats.get("mean_sens_pos"), "std_sens_pos": f_stats.get("std_sens_pos"),
                "mean_spec_pos": f_stats.get("mean_spec_pos"), "std_spec_pos": f_stats.get("std_spec_pos"),
                "mean_sens_macro": f_stats.get("mean_sens_macro"), "std_sens_macro": f_stats.get("std_sens_macro"),
                "mean_spec_macro": f_stats.get("mean_spec_macro"), "std_spec_macro": f_stats.get("std_spec_macro"),
                "fold_reports": [
                    {"cm": m.get("cm").tolist() if isinstance(m.get("cm"), np.ndarray) else m.get("cm"),
                     "kappa": m.get("kappa"), "mcc": m.get("mcc"), "pr_curve": m.get("pr_curve"), "auc": m.get("auc"),
                     "sens_pos": m.get("sens_pos"), "spec_pos": m.get("spec_pos"),
                     "sens_macro": m.get("sens_macro"), "spec_macro": m.get("spec_macro")}
                    for m in f_fold
                ]
            }
            final_path = os.path.join(exp_dir, "hpo_best_cv.json")
            with open(final_path, "w", encoding="utf-8") as f:
                json.dump(final_report, f, ensure_ascii=False, indent=2)
            print(f"[HPO] final CV report saved -> {final_path}", flush=True)
        if bool(hpo_cfg.get("save_best_weights", True)):
            best_path = os.path.join(exp_dir, "hpo_best_model.pth")
            _train_final_model(cfg, backbone_choice, ds, in_chans, target_shape, device, device_type, best["params"], final_epochs, best_path)
            print(f"[HPO] best model saved -> {best_path}", flush=True)
    
    report_path = _write_html_report(exp_dir, run_id, optim_target)
    
    # Write text report to specified location
    report_txt_path = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "report.txt"))
    # Or use a CLI argument if available, but for now default to project root as requested
    if cfg.get("paths", {}).get("report_file"):
        report_txt_path = cfg["paths"]["report_file"]
    
    # Collect best fold reports
    best_fold_reports = []
    if os.path.isfile(os.path.join(exp_dir, "hpo_best_cv.json")):
         with open(os.path.join(exp_dir, "hpo_best_cv.json"), "r", encoding="utf-8") as f:
             best_fold_reports = json.load(f).get("fold_reports", [])

    _write_text_report(report_txt_path, run_id, summary, best_fold_reports)
    print(f"[HPO] text report saved -> {report_txt_path}", flush=True)

    if bool(hpo_cfg.get("export_pdf", True)):
        pdf_path = os.path.join(exp_dir, f"report_{run_id}.pdf")
        _write_simple_pdf(pdf_path, [f"HPO Report {run_id}", f"optim_target={optim_target}", f"summary={summary_path}", f"html={report_path}"])
        print(f"[HPO] report pdf saved -> {pdf_path}", flush=True)
    if writer is not None:
        writer.close()


class MRIVolumeLabeledDataset(torch.utils.data.Dataset):
    """
    简化版：读取 (path, label) 列，统一到目标尺寸。这里假设数据已为 1mm, 
    """
    def __init__(self, manifest_csv: str, path_column: str, label_column: str, label_map: dict, target_shape=(112, 112, 112)):
        import csv
        import nibabel as nib
        import numpy as np

        self.target_shape = target_shape
        self.items = []
        with open(manifest_csv, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                p = row.get(path_column, '')
                lab = row.get(label_column, '')
                if p and os.path.isfile(p) and lab in label_map:
                    self.items.append((p, label_map[lab]))
        self.nib = nib
        self.np = np

        if not self.items:
            raise RuntimeError("No labeled items found.")

    def zscore(self, v):
        p1, p99 = self.np.percentile(v, 1.0), self.np.percentile(v, 99.0)
        v = self.np.clip(v, p1, p99)
        m, s = v.mean(), v.std() + 1e-6
        return (v - m) / s

    def padcrop(self, vol):
        D, H, W = vol.shape
        tD, tH, tW = self.target_shape
        import numpy as np
        out = vol
        pad_d = max(tD - D, 0)
        pad_h = max(tH - H, 0)
        pad_w = max(tW - W, 0)
        if pad_d or pad_h or pad_w:
            out = np.pad(out, ((pad_d // 2, pad_d - pad_d // 2),
                               (pad_h // 2, pad_h - pad_h // 2),
                               (pad_w // 2, pad_w - pad_w // 2)), mode='constant')
        D, H, W = out.shape
        sD = (D - tD) // 2 if D > tD else 0
        sH = (H - tH) // 2 if H > tH else 0
        sW = (W - tW) // 2 if W > tW else 0
        out = out[sD:sD + tD, sH:sH + tH, sW:sW + tW]
        return out

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, y = self.items[idx]
        vol = self.nib.load(p).get_fdata().astype(self.np.float32)
        vol = self.zscore(vol)
        vol = self.padcrop(vol)
        vol = self.np.expand_dims(vol, 0)
        ten = torch.from_numpy(vol).float()  # 显式转 float32
        return ten, torch.tensor(y, dtype=torch.long)

class MRIVolumeFolderDataset(torch.utils.data.Dataset):
    def __init__(self, label_roots: dict, label_map: dict, target_shape=(112,112,112), validate_nifti: bool = True, file_exts: tuple = (".nii", ".nii.gz"), require_name_substring: Optional[str] = None, augment: bool = False):
        import nibabel as nib
        import numpy as np
        self.nib = nib
        self.np = np
        self.items = []
        self.target_shape = target_shape
        self.augment = augment

        # 收集文件
        for lab_name, root in label_roots.items():
            y = label_map.get(lab_name, None)
            if y is None or not os.path.isdir(root):
                continue
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if not fn.lower().endswith(file_exts):
                        continue
                    if require_name_substring and (require_name_substring not in fn):
                        continue
                    p = os.path.join(dirpath, fn)
                    if os.path.isfile(p):
                        self.items.append((p, y))

        if not self.items:
            raise RuntimeError("No labeled items found from label_roots.")

        # 可选：校验 .nii 文件字节数是否满足头信息声明的大小（跳过损坏样本）
        if validate_nifti:
            valid, invalid = [], []
            for p, y in self.items:
                ext = os.path.splitext(p)[1].lower()
                if ext == ".nii":
                    try:
                        img = self.nib.load(p)
                        shape = img.header.get_data_shape()
                        dtype = img.get_data_dtype()
                        offset = int(float(img.header.get("vox_offset", 0)))
                        expected = offset + int(self.np.prod(shape)) * self.np.dtype(dtype).itemsize
                        actual = os.path.getsize(p)
                        if actual < expected:
                            invalid.append(p)
                        else:
                            valid.append((p, y))
                    except Exception:
                        invalid.append(p)
                else:
                    # .nii.gz 不做字节比对
                    valid.append((p, y))
            self.items = valid
            if invalid:
                print(f"[MRIVolumeFolderDataset] skipped {len(invalid)} damaged .nii files (size < expected).")

        if not self.items:
            raise RuntimeError("All labeled items are invalid after validation.")

        # 扫描形状，推断最大通道数（保留所有额外维为通道）
        max_c = 1
        for p, _ in self.items:
            try:
                img = self.nib.load(p)
                shp = tuple(img.shape)
                extra = shp[3:] if len(shp) > 3 else ()
                c = int(self.np.prod(extra)) if extra else 1
                max_c = max(max_c, c)
            except Exception:
                pass
        self.in_channels = int(max_c)
        print(f"[MRIVolumeFolderDataset] samples={len(self.items)}, inferred in_channels={self.in_channels}")

    def _to_channel_first_3d(self, arr):
        # arr: (X,Y,Z,ExtraDims...), 折叠额外维为通道 C
        spatial = arr.shape[:3]
        extra = arr.shape[3:] if arr.ndim > 3 else ()
        C = int(self.np.prod(extra)) if extra else 1
        arr = arr.reshape(spatial[0], spatial[1], spatial[2], C)  # (D,H,W,C)
        arr = self.np.transpose(arr, (3, 0, 1, 2))               # (C,D,H,W)
        return arr

    def zscore_channels(self, v):
        # v: (C,D,H,W)，按通道剪裁到1/99分位并做z-score
        C = v.shape[0]
        flat = v.reshape(C, -1)
        p1 = self.np.percentile(flat, 1.0, axis=1).reshape(C, 1, 1, 1)
        p99 = self.np.percentile(flat, 99.0, axis=1).reshape(C, 1, 1, 1)
        v = self.np.clip(v, p1, p99)
        m = v.mean(axis=(1, 2, 3), keepdims=True)
        s = v.std(axis=(1, 2, 3), keepdims=True) + 1e-6
        return (v - m) / s

    def padcrop_ch(self, v):
        # v: (C,D,H,W)，仅对空间维做居中 pad/crop
        C, D, H, W = v.shape
        tD, tH, tW = self.target_shape
        pad_d = max(tD - D, 0)
        pad_h = max(tH - H, 0)
        pad_w = max(tW - W, 0)
        if pad_d or pad_h or pad_w:
            v = self.np.pad(v, ((0, 0),
                                (pad_d // 2, pad_d - pad_d // 2),
                                (pad_h // 2, pad_h - pad_h // 2),
                                (pad_w // 2, pad_w - pad_w // 2)),
                             mode='constant')
        _, D, H, W = v.shape
        sD = (D - tD) // 2 if D > tD else 0
        sH = (H - tH) // 2 if H > tH else 0
        sW = (W - tW) // 2 if W > tW else 0
        v = v[:, sD:sD + tD, sH:sH + tH, sW:sW + tW]
        return v

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, y = self.items[idx]
        try:
            arr = self.nib.load(p).get_fdata().astype(self.np.float32)
        except Exception as e:
            # 读取失败（损坏或其他异常）→ 返回 None，让 collate_fn 过滤
            print(f"[MRIVolumeFolderDataset] read error, skip: {p} | {e}")
            return None
        v = self._to_channel_first_3d(arr)  # (C,D,H,W)
        C = v.shape[0]
        if C < self.in_channels:
            v = self.np.pad(v, ((0, self.in_channels - C), (0, 0), (0, 0), (0, 0)), mode='constant')
        elif C > self.in_channels:
            raise RuntimeError(f"Sample channels {C} exceed dataset in_channels {self.in_channels}. Rebuild to match.")
        
        # Apply elastic transform with 50% probability during training if enabled
        if hasattr(self, 'augment') and self.augment and self.np.random.rand() < 0.5:
            # Import here to avoid circular import issues if placed at top
            from dataset_mri3d import apply_elastic_transform
            v = apply_elastic_transform(v, alpha=15.0, sigma=3.0)

        v = self.zscore_channels(v)
        v = self.padcrop_ch(v)
        ten = torch.from_numpy(v).float()  # 确保 float32，避免 double 混入
        return ten, torch.tensor(y, dtype=torch.long)

def make_safe_collate(in_chans, target_shape):
    def _collate(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            x = torch.empty((0, in_chans, target_shape[0], target_shape[1], target_shape[2]), dtype=torch.float32)
            y = torch.empty((0,), dtype=torch.long)
            return x, y
        return default_collate(batch)
    return _collate

# 在文件中定义的权重加载函数（方法/类：顶层函数）
def load_pretrained_partial(model, ckpt_path, in_chans_expected, keep_classifier_if_shape_match=True):
    state = torch.load(ckpt_path, map_location='cpu')
    # 兼容不同保存格式
    if isinstance(state, dict):
        if 'state_dict' in state and isinstance(state['state_dict'], dict):
            state = state['state_dict']
        elif 'model_state' in state and isinstance(state['model_state'], dict):
            state = state['model_state']
        elif 'model' in state and isinstance(state['model'], dict):
            state = state['model']

    tgt = model.state_dict()
    new_state = {}
    sample_target_keys = list(tgt.keys())[:8]

    for k, v in state.items():
        candidates = [k]
        if k.startswith("module."):
            candidates.append(k[7:])
        if k.startswith("encoder."):
            candidates.append(k[len("encoder."):])
        if k.startswith("backbone."):
            candidates.append(k[len("backbone."):])
        if k.startswith("model."):
            candidates.append(k[len("model."):])
        # Alias for legacy SS3M pretrain checkpoints: patch.* -> patch_embed.*
        expanded = []
        for cand in candidates:
            expanded.append(cand)
            if isinstance(cand, str) and cand.startswith("patch."):
                expanded.append("patch_embed." + cand[len("patch."):])
        candidates = expanded
        k2 = None
        for cand in candidates:
            if cand in tgt:
                k2 = cand
                break
        if k2 is None:
            continue

        # 形状完全一致 → 直接加载；可选地保留分类头
        if tgt[k2].shape == v.shape:
            if not keep_classifier_if_shape_match and k2.startswith('classifier.'):
                continue
            new_state[k2] = v
            continue

        # 首层卷积权重：适配输入通道数变化（例如 1 → in_chans_expected）
        if k2 in ('encoder.patch.proj.weight', 'patch_embed.proj.weight', 'patch.proj.weight'):
            try:
                out_c, in_c_old, kd, kh, kw = v.shape
                in_c_new = tgt[k2].shape[1]
                tgt_shape = tgt[k2].shape
                if in_c_old == 1 and in_c_new == in_chans_expected and out_c == tgt_shape[0] and (kd, kh, kw) == tuple(tgt_shape[2:]):
                    w_exp = v.repeat(1, in_c_new, 1, 1, 1) / float(in_c_new)
                    new_state[k2] = w_exp
                    print(f"[pretrained] adapted patch.proj.weight {tuple(v.shape)} -> {tuple(w_exp.shape)}")
                else:
                    print(f"[pretrained] skip patch.proj.weight due to shape mismatch {tuple(v.shape)} -> {tuple(tgt[k2].shape)}")
            except Exception as e:
                print(f"[pretrained] adapt patch.proj.weight failed: {e}")
            continue

        # 其它形状不匹配的键跳过
        # 也可能是 embed_dim 或 depth 
        # print(f"[pretrained] skip {k2} shape {tuple(v.shape)} -> {tuple(tgt[k2].shape)}")
        pass

    for k in tgt.keys():
        if ".ssm_b." in k and k not in new_state:
            k_a = k.replace(".ssm_b.", ".ssm_a.")
            if k_a in new_state and tgt[k].shape == new_state[k_a].shape:
                new_state[k] = new_state[k_a].clone()

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(
        f"[pretrained] path={ckpt_path} loaded_keys={len(new_state)}/{len(tgt)} "
        f"missing={len(missing)} unexpected={len(unexpected)} sample_model_keys={sample_target_keys}",
        flush=True,
    )

def load_config(cfg_path: str):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def compute_num_workers(cfg: dict) -> int:
    """
    函数用途：
    - 根据当前机器的 CPU 核心数自动计算用于 DataLoader 的 num_workers。
      默认保留 reserve_cores（配置项，默认为 2）给系统与其他任务，其余核心全部用于数据加载。
    设计原因：
    - 充分利用机器 CPU 性能，同时避免占满所有核心导致系统卡顿或其他任务性能下降。
    - 避免在不同位置重复写 num_workers 逻辑，也可减少“变量未定义”的风险。
    
    使用说明：
    - 当 cfg['dataset']['auto_num_workers'] 为 False 时，回退使用 cfg['dataset']['num_workers'] 的固定值；
      否则使用 os.cpu_count() - reserve_cores 的策略，且返回值至少为 1。
    """
    import os
    ds_cfg = cfg.get('dataset', {}) if isinstance(cfg, dict) else {}
    auto = bool(ds_cfg.get('auto_num_workers', True))
    if not auto:
        fixed = int(ds_cfg.get('num_workers', 2))
        return max(1, fixed)
    cpu_total = os.cpu_count() or 1
    reserve = int(ds_cfg.get('reserve_cores', 2))
    return max(1, cpu_total - reserve)

def main():
    """
    主训练入口
    作用：
        - 解析命令行参数
        - 构造数据与模型
        - 执行训练与评估
    改动原因：
        - 新增对“train/val/test 三层目录结构”的直接支持；
        - 将默认输入尺寸改为 112^3；
        - 将步级打印频率默认提升到每1000步；
        - 增加测试集预测与CSV输出，完成分类预测任务闭环。
    """
    parser = argparse.ArgumentParser(description="MRI Classifier Training & HPO")
    parser.add_argument("--config", type=str, default=os.path.join(os.path.dirname(__file__), 'config.yaml'), help="Path to config file")
    parser.add_argument("--train_roots", type=str, help="Comma separated paths for training roots (e.g. AD=path1,MCI=path2)")
    parser.add_argument("--val_roots", type=str, help="Comma separated paths for validation roots")
    parser.add_argument("--test_roots", type=str, help="Comma separated paths for testing roots")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--epochs", type=int, help="Number of epochs")
    parser.add_argument("--hpo", action="store_true", help="Enable HPO")
    parser.add_argument("--algorithm", type=str, help="HPO algorithm (grid, random, bayes)")
    parser.add_argument("--cv_folds", type=int, help="Run non-HPO K-fold CV (e.g., 5)")
    parser.add_argument("--report_file", type=str, default=None, help="Path to output report.txt")
    parser.add_argument("--wandb_project", type=str, help="W&B project name")
    
    args = parser.parse_args()
    
    cfg = load_config(args.config)
    
    # Override config with CLI args
    if args.batch_size:
        cfg.setdefault('classifier', {})['batch_size'] = args.batch_size
    if args.lr:
        cfg.setdefault('classifier', {})['lr'] = args.lr
    if args.epochs:
        cfg.setdefault('classifier', {})['epochs'] = args.epochs
    if args.hpo:
        cfg.setdefault('hpo', {})['enabled'] = True
    if args.algorithm:
        cfg.setdefault('hpo', {})['algorithm'] = args.algorithm
    if args.cv_folds is not None:
        cfg.setdefault('cv', {})['folds'] = int(args.cv_folds)
    if args.report_file:
        cfg.setdefault('paths', {})['report_file'] = args.report_file
    else:
        # Default report file path if not provided via CLI
        cfg.setdefault('paths', {})['report_file'] = os.path.abspath(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "report.txt"))

    if args.wandb_project and wandb:
        wandb.init(project=args.wandb_project, config=cfg)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 读取骨干选择（配置优先，未设置则默认旧骨干）
    backbone_choice = str(cfg.get('classifier', {}).get('backbone', 'medmamba3d')).lower()
    folder_label_map = cfg.get('dataset', {}).get('folder_label_map', {'AD': 0, 'LBD': 1, 'MCI': 2})
    allowed_labels = cfg.get('dataset', {}).get('allowed_labels', None)
    if allowed_labels is not None:
        if not isinstance(allowed_labels, (list, tuple)) or len(allowed_labels) not in (2, 3):
            raise RuntimeError(f"dataset.allowed_labels 需为长度2或3的列表，当前={allowed_labels}")
        folder_label_map = {name: idx for idx, name in enumerate(allowed_labels)}
    train_roots = cfg.get('paths', {}).get('train_label_roots', None)
    val_roots = cfg.get('paths', {}).get('val_label_roots', None)
    test_roots = cfg.get('paths', {}).get('test_label_roots', None)
    if allowed_labels is not None and len(allowed_labels) == 2:
        if isinstance(train_roots, dict):
            train_roots = {k: v for k, v in train_roots.items() if k in allowed_labels}
        if isinstance(val_roots, dict):
            val_roots = {k: v for k, v in val_roots.items() if k in allowed_labels}
        if isinstance(test_roots, dict):
            test_roots = {k: v for k, v in test_roots.items() if k in allowed_labels}
    label_roots_all = cfg.get('paths', {}).get('label_roots', None)
    if allowed_labels is not None and len(allowed_labels) == 2 and isinstance(label_roots_all, dict):
        label_roots_all = {k: v for k, v in label_roots_all.items() if k in allowed_labels}

    # === 配置驱动的常用参数初始化 ===
    # 卷积输入目标尺寸（D,H,W）
    target_shape = tuple(cfg.get('input', {}).get('shape_dhw', (112, 112, 112)))
    cls_cfg = cfg.get('classifier', {})
    ds_cfg = cfg.get('dataset', {})

    # 训练循环需要的几个变量，提前从配置读取并提供默认值
    batch_size   = int(cls_cfg.get('batch_size', 1))
    num_classes  = int(ds_cfg.get('num_classes', len(set(folder_label_map.values()))))
    grad_accum   = int(cls_cfg.get('grad_accum_steps', 1))
    print_every  = int(cls_cfg.get('print_every', 1000))
    cv_cfg = cfg.get('cv', {}) if isinstance(cfg.get('cv', {}), dict) else {}
    cv_folds = int(cv_cfg.get('folds', cv_cfg.get('cv_n_splits', cv_cfg.get('num_folds', 5))))
    cv_seed = int(cv_cfg.get('cv_random_state', cv_cfg.get('seed', 42)))
    print(
        f"[Mode] hpo_enabled={bool(cfg.get('hpo', {}).get('enabled', False))} cv_folds={cv_folds} "
        f"has_label_roots={isinstance(label_roots_all, dict)} has_train_val_roots={isinstance(train_roots, dict) and isinstance(val_roots, dict)}",
        flush=True,
    )

    def _build_cv_roots():
        if isinstance(label_roots_all, dict) and label_roots_all:
            return label_roots_all
        if isinstance(train_roots, dict) and isinstance(val_roots, dict):
            merged = {}
            for k in folder_label_map.keys():
                paths = []
                if k in train_roots:
                    paths.append(train_roots[k])
                if k in val_roots:
                    paths.append(val_roots[k])
                if paths:
                    merged[k] = paths
            return merged if merged else None
        return None

    if bool(cfg.get("hpo", {}).get("enabled", False)) and not (isinstance(train_roots, dict) and isinstance(val_roots, dict)):
        hpo_roots = _build_cv_roots()
        if isinstance(hpo_roots, dict) and hpo_roots:
            ds_hpo = MRIVolumeFolderDataset(
                label_roots=hpo_roots,
                label_map=folder_label_map,
                target_shape=target_shape,
                validate_nifti=True,
                file_exts=(".nii", ".nii.gz"),
                require_name_substring=None,
                augment=False,
            )
            inferred_in_chans = int(getattr(ds_hpo, 'in_channels', 1))
            _run_hpo(cfg, backbone_choice, ds_hpo, inferred_in_chans, target_shape, device, device_type)
            return
        print("[HPO][warn] hpo enabled but no usable label_roots found; falling back to other modes.", flush=True)

    if (not bool(cfg.get("hpo", {}).get("enabled", False))) and cv_folds >= 2:
        cv_roots = _build_cv_roots()
        if isinstance(cv_roots, dict) and cv_roots:
            ds_cv = MRIVolumeFolderDataset(
                label_roots=cv_roots,
                label_map=folder_label_map,
                target_shape=target_shape,
                validate_nifti=True,
                file_exts=(".nii", ".nii.gz"),
                require_name_substring=None,
                augment=False,
            )
            inferred_in_chans = int(getattr(ds_cv, 'in_channels', 1))
            num_classes_cfg = int(cfg.get('dataset', {}).get('num_classes', len(set(folder_label_map.values()))))
            num_classes = _infer_num_classes_from_items(getattr(ds_cv, "items", []), num_classes_cfg)
            folds = _build_folds(getattr(ds_cv, "items", []), num_classes=num_classes, n_splits=cv_folds, seed=cv_seed)
            params = {
                "lr": float(cls_cfg.get("lr", 1e-4)),
                "weight_decay": float(cls_cfg.get("weight_decay", 0.0)),
                "batch_size": int(cls_cfg.get("batch_size", 1)),
                "depth": int(cfg.get("ss3m", {}).get("depth", cfg.get("model", {}).get("depth", 4))),
                "embed_dim": int(cfg.get("ss3m", {}).get("embed_dim", cfg.get("model", {}).get("embed_dim", 96))),
                "dropout": float(cfg.get("ss3m", {}).get("dropout", 0.0)),
                "loss": str(cls_cfg.get("loss", "ce")),
                "focal_gamma": float(cls_cfg.get("focal_gamma", 2.0)),
                "tversky_alpha": float(cls_cfg.get("tversky_alpha", 0.5)),
                "tversky_beta": float(cls_cfg.get("tversky_beta", 0.5)),
                "sampler": str(cls_cfg.get("sampler", "none")),
                "use_class_weights": bool(cls_cfg.get("use_class_weights", False)),
                "threshold": cls_cfg.get("eval_threshold"),
            }
            trial_ctx = {
                "optim_target": str(cv_cfg.get("optim_target", "auc")).lower(),
                "early_stop_patience": int(cls_cfg.get("early_stop_patience", 20)),
                "early_stop_min_epochs": int(cls_cfg.get("early_stop_min_epochs", 10)),
                "early_stop_min_delta": float(cls_cfg.get("early_stop_min_delta", 0.0)),
            }
            epochs = int(cls_cfg.get('epochs', 100))
            mean_acc, mean_bal, mean_auc, fold_metrics = _train_one_trial_cv(
                cfg, backbone_choice, ds_cv, folds, inferred_in_chans, target_shape, device, device_type, params, epochs, trial_ctx
            )
            out_dir = cfg.get('paths', {}).get('out_dir', '.')
            os.makedirs(out_dir, exist_ok=True)
            cv_summary_path = os.path.join(out_dir, f"cv{cv_folds}_summary.json")
            cv_summary = {
                "folds": cv_folds,
                "mean_acc": mean_acc,
                "mean_bal_acc": mean_bal,
                "mean_auc": mean_auc,
                "fold_metrics": [
                    {
                        "acc": m.get("acc"),
                        "bal_acc": m.get("bal_acc"),
                        "auc": m.get("auc"),
                        "kappa": m.get("kappa"),
                        "mcc": m.get("mcc"),
                    } for m in fold_metrics
                ],
            }
            with open(cv_summary_path, "w", encoding="utf-8") as f:
                json.dump(cv_summary, f, ensure_ascii=False, indent=2)
            print(f"[CV] {cv_folds}-fold summary saved -> {cv_summary_path}", flush=True)
            return
        else:
            print("[CV][warn] cv_folds>=2 但未找到可用 roots，回退到 train/val 直训。", flush=True)

    if print_every < 1:
        print(f"[Train] configured print_every={print_every} too small, using 1 instead")
        print_every = 1
    if isinstance(train_roots, dict) and isinstance(val_roots, dict):
        # 构建数据集与加载器（train/val）
        ds_train = MRIVolumeFolderDataset(
            label_roots=train_roots,
            label_map=folder_label_map,
            target_shape=target_shape,
            validate_nifti=True,                # .nii.gz 不做字节比对，但会做体素可读性验证
            file_exts=(".nii", ".nii.gz"),
            require_name_substring=None
        )
        ds_val = MRIVolumeFolderDataset(
            label_roots=val_roots,
            label_map=folder_label_map,
            target_shape=target_shape,
            validate_nifti=True,
            file_exts=(".nii", ".nii.gz"),
            require_name_substring=None
        )

        # 新增：推断输入通道并构造 collate_fn；计算 num_workers
        inferred_in_chans = int(getattr(ds_train, 'in_channels', 1))
        collate_fn = make_safe_collate(inferred_in_chans, target_shape)
        num_workers = compute_num_workers(cfg)

        if bool(cfg.get("hpo", {}).get("enabled", False)):
            _run_hpo(cfg, backbone_choice, ds_train, inferred_in_chans, target_shape, device, device_type)
            return

        # 新增：提前读取 batch_size、num_classes，避免后续使用时未定义
        batch_size = int(cfg.get('classifier', {}).get('batch_size', 1))
        num_classes_cfg = int(cfg.get('dataset', {}).get('num_classes', len(set(folder_label_map.values()))))
        num_classes = _infer_num_classes_from_items(getattr(ds_train, "items", []), num_classes_cfg)
        if num_classes != num_classes_cfg:
            print(f"[Train] num_classes adjusted {num_classes_cfg} -> {num_classes}", flush=True)

        loss_type = str(cls_cfg.get("loss", "ce")).lower()
        sampler_strategy = str(cls_cfg.get("sampler", "none")).lower()
        use_class_weights = bool(cls_cfg.get("use_class_weights", False))
        imbalance_mode = _resolve_imbalance_mode(cfg, sampler_strategy, use_class_weights)
        if imbalance_mode != "sampler" and sampler_strategy in ("weighted", "oversample", "smote", "adasyn"):
            sampler_strategy = "none"
        eval_threshold = cls_cfg.get("eval_threshold")
        labels_train = [y for _, y in getattr(ds_train, "items", [])]
        class_weights = _class_weights_from_labels(labels_train, num_classes)
        ce_weight = torch.tensor(class_weights, dtype=torch.float32, device=device) if imbalance_mode == "class_weight" else None
        sampler = None
        if imbalance_mode == "sampler" and sampler_strategy in ("weighted", "oversample"):
            sample_weights = [class_weights[y] for y in labels_train]
            sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        print(f"[Imbalance] mode={imbalance_mode} sampler={sampler_strategy} use_class_weights={imbalance_mode == 'class_weight'}", flush=True)
        train_loader = DataLoader(
            ds_train, batch_size=batch_size, shuffle=(sampler is None),
            sampler=sampler,
            num_workers=num_workers, pin_memory=(device_type == 'cuda'),
            persistent_workers=(num_workers > 0), prefetch_factor=2, collate_fn=collate_fn
        )
        # Enable augmentation for training set (direct training mode)
        if hasattr(ds_train, 'augment'):
             ds_train.augment = True
        
        val_loader = DataLoader(
            ds_val, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=(device_type == 'cuda'),
            persistent_workers=(num_workers > 0), prefetch_factor=2, collate_fn=collate_fn
        )

        # 根据骨干选择构建模型（MedMamba3D 或 SS3M）
        if backbone_choice == "medmamba3d":
            model = MedMamba3D(
                in_chans=inferred_in_chans,
                embed_dim=cfg['model']['embed_dim'],
                depth=cfg['model']['depth'],
                patch_size=tuple(cfg['model']['patch_size']),
                num_classes=num_classes
            ).to(device)
        else:
            ss3m_arch = _resolve_ss3m_arch(cfg, {}, inferred_in_chans)
            inferred = ss3m_arch.get("inferred", {})
            if inferred:
                print(
                    f"[model][diag] ss3m align from ckpt container={inferred.get('container', 'na')} "
                    f"patch_key={inferred.get('patch_key', 'na')} embed_dim={ss3m_arch['embed_dim']} "
                    f"depth={ss3m_arch['depth']} patch_size={ss3m_arch['patch_size']}",
                    flush=True,
                )
            model = MedMambaSS3M(
                in_channels=inferred_in_chans,
                embed_dim=ss3m_arch['embed_dim'],
                depth=ss3m_arch['depth'],
                patch_size=tuple(ss3m_arch['patch_size']),
                num_classes=num_classes,
                dropout=float(ss3m_arch['dropout'])
            ).to(device)

        # 可选加载预训练权重（部分加载并适配输入通道）
        if cfg['classifier'].get('pretrained_path') and os.path.isfile(cfg['classifier']['pretrained_path']):
            load_pretrained_partial(model, cfg['classifier']['pretrained_path'], inferred_in_chans, keep_classifier_if_shape_match=True)
            print("[Classifier] pretrained partial load finished")

        # 优化器与 AMP
        opt = torch.optim.AdamW(model.parameters(), lr=cfg['classifier']['lr'], weight_decay=cfg['classifier']['weight_decay'])
        scaler = torch.amp.GradScaler('cuda') if device_type == 'cuda' else None

        epochs = int(cfg['classifier']['epochs'])
        eval_cfg = cfg.get("eval", {}) if isinstance(cfg.get("eval", {}), dict) else {}
        eval_use_tta = bool(eval_cfg.get("use_tta", False))
        if epochs <= 0:
            raise RuntimeError("classifier.epochs 必须大于 0，以保证先训练再验证。")
        best_bal_acc = -1.0
        for epoch in range(epochs):
            epoch_start_time = time.time()
            lr_now = opt.param_groups[0]['lr'] if opt.param_groups else cfg['classifier']['lr']
            print(f"[Train] Epoch {epoch+1}/{epochs} start lr={lr_now:.6g}", flush=True)
            model.train()
            running_loss, seen, correct = 0.0, 0, 0
            cm_train = torch.zeros((num_classes, num_classes), dtype=torch.int64)
            opt.zero_grad(set_to_none=True)

            for step, batch in enumerate(train_loader):
                x, y = batch
                if y.numel() == 0:
                    continue
                if imbalance_mode == "sampler" and sampler_strategy in ("smote", "adasyn"):
                    x, y = _apply_smote_like(x, y, sampler_strategy)
                x = x.to(device, dtype=torch.float32, non_blocking=True)
                y = y.to(device, non_blocking=True)
                amp_ctx = torch.amp.autocast('cuda', dtype=torch.float16) if device_type == 'cuda' else contextlib.nullcontext()
                with amp_ctx:
                    logits = model.forward_classifier(x) if hasattr(model, "forward_classifier") else model(x)
                    if loss_type in ("focal", "focal_loss"):
                        loss = _focal_loss(logits, y, gamma=float(cls_cfg.get("focal_gamma", 2.0)), alpha=ce_weight)
                    elif loss_type == "dice":
                        loss = _dice_loss(logits, y)
                    elif loss_type == "tversky":
                        loss = _tversky_loss(logits, y, alpha=float(cls_cfg.get("tversky_alpha", 0.5)), beta=float(cls_cfg.get("tversky_beta", 0.5)))
                    else:
                        loss = F.cross_entropy(logits, y, weight=ce_weight)

                loss_for_backward = loss / max(1, grad_accum)
                if scaler is not None:
                    scaler.scale(loss_for_backward).backward()
                    if ((step + 1) % grad_accum == 0) or ((step + 1) == len(train_loader)):
                        scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True)
                else:
                    loss_for_backward.backward()
                    if ((step + 1) % grad_accum == 0) or ((step + 1) == len(train_loader)):
                        opt.step(); opt.zero_grad(set_to_none=True)

                # 训练指标累计
                running_loss += loss.detach().item() * y.size(0)
                seen += y.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                for i, j in zip(y.detach().cpu().tolist(), preds.detach().cpu().tolist()):
                    cm_train[i, j] += 1

                # 步级打印（每1000步）
                if (step + 1) % max(print_every, 1) == 0:
                    avg_loss = running_loss / max(seen, 1)
                    avg_acc = correct / max(seen, 1)
                    print(f"[Train] Epoch {epoch+1} [{step+1}/{len(train_loader)}] "
                          f"loss={avg_loss:.4f} acc={avg_acc:.3f}", flush=True)

            avg_loss = running_loss / max(seen, 1)
            avg_acc = correct / max(seen, 1)
            train_time_sec = time.time() - epoch_start_time
            print(f"[Train] Epoch {epoch+1}/{epochs} done "
                  f"loss={avg_loss:.4f} acc={avg_acc:.3f} seen={seen} lr={lr_now:.6g} train_time={train_time_sec:.1f}s", flush=True)

            val_metrics = evaluate_loader(model, val_loader, device, num_classes, threshold=eval_threshold, return_pr=False, use_tta=eval_use_tta)
            val_acc = val_metrics["acc"]
            val_bal_acc = val_metrics["bal_acc"]
            val_auc = val_metrics["auc"]
            val_kappa = val_metrics["kappa"]
            val_mcc = val_metrics["mcc"]
            epoch_time_sec = time.time() - epoch_start_time
            
            print(f"[Val] Epoch {epoch+1}/{epochs} acc={val_acc:.3f} bal_acc={val_bal_acc:.3f} auc={val_auc:.3f} kappa={val_kappa:.3f} mcc={val_mcc:.3f}  epoch_time={epoch_time_sec:.1f}s", flush=True)
            if val_bal_acc > best_bal_acc:
                best_bal_acc = val_bal_acc
                out_path = os.path.join(cfg['paths']['out_dir'], "classifier_best.pth")
                torch.save({"state_dict": model.state_dict(), "epoch": epoch + 1}, out_path)
                print(f"[Ckpt] best model saved -> {out_path} (bal_acc={best_bal_acc:.3f})", flush=True)

        # 测试集预测（若提供 test_label_roots）
        if isinstance(test_roots, dict):
            ds_test = MRIVolumeFolderDataset(
                label_roots=test_roots,
                label_map=folder_label_map,
                target_shape=target_shape,
                validate_nifti=True,
                file_exts=(".nii", ".nii.gz"),
                require_name_substring=None
            )
            collate_fn_t = make_safe_collate(inferred_in_chans, target_shape)
            num_workers = compute_num_workers(cfg)
            test_loader = DataLoader(
                ds_test, batch_size=batch_size, shuffle=False,
                num_workers=num_workers, pin_memory=(device_type == 'cuda'),
                persistent_workers=(num_workers > 0), prefetch_factor=2, collate_fn=collate_fn_t
            )

            # 载入最佳权重并输出预测到 CSV
            ck = torch.load(os.path.join(cfg['paths']['out_dir'], "classifier_best.pth"), map_location='cpu')
            if isinstance(ck, dict) and 'state_dict' in ck:
                model.load_state_dict(ck['state_dict'], strict=False)
            else:
                model.load_state_dict(ck, strict=False)

            model.eval()
            import csv
            out_csv = os.path.join(cfg['paths']['out_dir'], "test_predictions.csv")
            with open(out_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                # 写表头：由于当前 Dataset 不返回路径，这里暂不包含 path 列
                w.writerow(["label_true", "label_pred"] + [f"prob_{c}" for c in range(num_classes)])
                with torch.no_grad():
                    for x, y in test_loader:
                        if y.numel() == 0:
                            continue
                        x = x.to(device, dtype=torch.float32, non_blocking=True)
                        logits = model.forward_classifier(x) if hasattr(model, "forward_classifier") else model(x)
                        probs = torch.softmax(logits, dim=1).detach().cpu()
                        preds = probs.argmax(dim=1)
                        for yi, pi, prob in zip(y.detach().cpu().tolist(), preds.tolist(), probs.tolist()):
                            w.writerow([yi, pi] + list(map(float, prob)))
            print(f"[Predict] 测试集预测已输出到 {out_csv}")

        # 使用 train/val/test 方式已完整结束 → 跳过原有 K 折流程
        return

    raise RuntimeError(
        "未找到可用的数据加载模式：\n"
        "1) 非HPO 5折请设置 hpo.enabled=false 且 cv.folds>=2，并提供 paths.label_roots={AD,LBD,...};\n"
        "2) 固定划分请提供 paths.train_label_roots 与 paths.val_label_roots。"
    )

    
if __name__ == "__main__":
    main()