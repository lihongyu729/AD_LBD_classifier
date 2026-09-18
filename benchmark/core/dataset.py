"""
Dataset, augmentation, and episodic sampling — extracted and adapted from
D:\py_project\MRI\code\meta\train_classifier.py.

Key components:
    - MRIVolumeFolderDataset: Scans directories for .nii/.nii.gz, validates,
      normalizes (z-score per channel), pad/crops to target shape.
    - augment_3d_batch: Random flip, Gaussian noise, Gamma jitter, 3D Cutout.
    - EpisodicBatchSampler: N-way K-shot Q-query task sampler for meta-learning.
    - split_task_data: Split sampled batch into support and query sets.
"""
import os
import hashlib
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Dict, List, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _stem(file_path: str) -> str:
    """Basename without extension (handles .nii.gz / .npy / .npz)."""
    name = os.path.basename(file_path)
    for ext in (".npy", ".nii.gz", ".nii", ".npz"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return name


def _cache_path(cache_dir: str, file_path: str) -> str:
    """Derive a cache filename from the source file path."""
    try:
        import xxhash
        h = xxhash.xxh64()
        h.update(os.path.abspath(file_path).encode("utf-8"))
        h.update(str(os.path.getsize(file_path)).encode("utf-8"))
        h.update(str(int(os.path.getmtime(file_path))).encode("utf-8"))
        key = h.hexdigest()
    except ImportError:
        blob = f"{os.path.abspath(file_path)}|{os.path.getsize(file_path)}|{int(os.path.getmtime(file_path))}"
        key = hashlib.blake2b(blob.encode("utf-8"), digest_size=16).hexdigest()
    return os.path.join(cache_dir, f"{key}.pt")


# ---------------------------------------------------------------------------
# MRIVolumeFolderDataset
# ---------------------------------------------------------------------------

class MRIVolumeFolderDataset(Dataset):
    """
    Folder-based 3D MRI dataset.

    Scans label root directories for .nii/.nii.gz files, validates NIfTI
    integrity, infers in_channels, and applies z-score normalization +
    pad/crop to target shape on load.

    Args:
        label_roots: Dict mapping label name → directory path(s).
        label_map: Dict mapping label name → integer class index.
        target_shape: (D, H, W) tuple for output volume shape.
        validate_nifti: Check NIfTI byte counts to skip damaged files.
        file_exts: Allowed file extensions (including .npy).
        require_name_substring: Optional substring filter on filenames.
        cache_dir: Directory for preprocessed cache.
        cache_enabled: Enable disk cache.
        normalized: If True, input volumes are already normalized (e.g.
            brain-mask Z-scored .npy) — skip the on-load z-score step.
    """

    def __init__(
        self,
        label_roots: dict,
        label_map: dict,
        target_shape: Tuple[int, int, int] = (112, 112, 112),
        validate_nifti: bool = True,
        file_exts: tuple = (".nii", ".nii.gz", ".npy"),
        require_name_substring: Optional[str] = None,
        cache_dir: Optional[str] = None,
        cache_enabled: bool = False,
        normalized: bool = False,
    ):
        self.items: List[Tuple[str, int]] = []
        self.patient_ids: List[str] = []
        self.normalized = normalized
        self.scan_stats: List[Tuple] = []
        self.target_shape = target_shape
        self.cache_dir = cache_dir if cache_enabled and cache_dir else None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Scan directories
        for lab_name, root in label_roots.items():
            y = label_map.get(lab_name, None)
            if y is None:
                self.scan_stats.append((lab_name, root, False, 0, "label_not_in_label_map"))
                continue
            roots = root if isinstance(root, (list, tuple)) else [root]
            for r in roots:
                if not isinstance(r, str):
                    self.scan_stats.append((lab_name, r, False, 0, "root_not_string"))
                    continue
                if not os.path.isdir(r):
                    self.scan_stats.append((lab_name, r, False, 0, "root_not_found"))
                    continue
                count_for_root = 0
                for dirpath, _, filenames in os.walk(r):
                    for fn in filenames:
                        if not fn.lower().endswith(file_exts):
                            continue
                        if require_name_substring and (require_name_substring not in fn):
                            continue
                        p = os.path.join(dirpath, fn)
                        if os.path.isfile(p):
                            self.items.append((p, y))
                            self.patient_ids.append(_stem(p))
                            count_for_root += 1
                self.scan_stats.append((lab_name, r, True, count_for_root, "ok"))

        if not self.items:
            print("[MRIVolumeFolderDataset][Diag] label_roots scan summary:", flush=True)
            for lab_name, root_path, exists, count, reason in self.scan_stats:
                print(f"  - label={lab_name} exists={exists} count={count} reason={reason} root={root_path}", flush=True)
            raise RuntimeError(
                f"No labeled items found from label_roots. "
                f"label_map_keys={list(label_map.keys())}, file_exts={list(file_exts)}"
            )

        # Validate NIfTI integrity
        if validate_nifti:
            valid, invalid = [], []
            print(f"[Dataset] Validating {len(self.items)} NIfTI files...", flush=True)
            for p, y in self.items:
                ext = os.path.splitext(p)[1].lower()
                if ext == ".nii":
                    try:
                        img = nib.load(p)
                        shape = img.header.get_data_shape()
                        dtype = img.get_data_dtype()
                        offset = int(float(img.header.get("vox_offset", 0)))
                        expected = offset + int(np.prod(shape)) * np.dtype(dtype).itemsize
                        actual = os.path.getsize(p)
                        (valid if actual >= expected else invalid).append((p, y))
                    except Exception:
                        invalid.append((p, y))
                else:
                    valid.append((p, y))
            self.items = valid
            if invalid:
                print(f"[MRIVolumeFolderDataset] skipped {len(invalid)} damaged .nii files.", flush=True)

        # Infer in_channels
        max_c = 1
        check_limit = 100 if not validate_nifti else len(self.items)
        checked = 0
        for p, _ in self.items:
            try:
                if p.lower().endswith(".npy"):
                    shp = tuple(np.load(p).shape)
                else:
                    img = nib.load(p)
                    shp = tuple(img.shape)
                extra = shp[3:] if len(shp) > 3 else ()
                c = int(np.prod(extra)) if extra else 1
                max_c = max(max_c, c)
                checked += 1
                if checked >= check_limit:
                    break
            except Exception:
                pass
        self.in_channels = int(max_c)
        print(f"[MRIVolumeFolderDataset] samples={len(self.items)}, inferred in_channels={self.in_channels}", flush=True)

    # ---- Preprocessing helpers ----
    def _to_channel_first_3d(self, arr):
        spatial = arr.shape[:3]
        extra = arr.shape[3:] if arr.ndim > 3 else ()
        C = int(np.prod(extra)) if extra else 1
        arr = arr.reshape(spatial[0], spatial[1], spatial[2], C)
        arr = np.transpose(arr, (3, 0, 1, 2))  # (C, D, H, W)
        return arr

    def zscore_channels(self, v):
        C = v.shape[0]
        flat = v.reshape(C, -1)
        p1 = np.percentile(flat, 1.0, axis=1).reshape(C, 1, 1, 1)
        p99 = np.percentile(flat, 99.0, axis=1).reshape(C, 1, 1, 1)
        v = np.clip(v, p1, p99)
        m = v.mean(axis=(1, 2, 3), keepdims=True)
        s = v.std(axis=(1, 2, 3), keepdims=True) + 1e-6
        return (v - m) / s

    def padcrop_ch(self, v):
        C, D, H, W = v.shape
        tD, tH, tW = self.target_shape
        pad_d = max(tD - D, 0)
        pad_h = max(tH - H, 0)
        pad_w = max(tW - W, 0)
        if pad_d or pad_h or pad_w:
            v = np.pad(v, ((0, 0), (pad_d // 2, pad_d - pad_d // 2),
                            (pad_h // 2, pad_h - pad_h // 2),
                            (pad_w // 2, pad_w - pad_w // 2)), mode='constant')
        _, D, H, W = v.shape
        sD = (D - tD) // 2 if D > tD else 0
        sH = (H - tH) // 2 if H > tH else 0
        sW = (W - tW) // 2 if W > tW else 0
        return v[:, sD:sD + tD, sH:sH + tH, sW:sW + tW]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, y = self.items[idx]

        # Cache check
        if self.cache_dir:
            cp = _cache_path(self.cache_dir, p)
            if os.path.isfile(cp):
                try:
                    cached = torch.load(cp, map_location="cpu")
                    if isinstance(cached, dict) and "x" in cached and "y" in cached:
                        x_cached = cached["x"]
                        if not torch.isfinite(x_cached).all():
                            print(f"[Dataset][warn] cache contains NaN/Inf, recompute: {os.path.basename(p)}", flush=True)
                        else:
                            return x_cached, cached["y"]
                except Exception:
                    pass

        try:
            if p.lower().endswith(".npy"):
                arr = np.load(p).astype(np.float32)
            else:
                arr = nib.load(p).get_fdata().astype(np.float32)
        except Exception as e:
            print(f"[MRIVolumeFolderDataset] read error, skip: {p} | {e}", flush=True)
            return None

        v = self._to_channel_first_3d(arr)
        C = v.shape[0]
        if C < self.in_channels:
            v = np.pad(v, ((0, self.in_channels - C), (0, 0), (0, 0), (0, 0)), mode='constant')
        elif C > self.in_channels:
            v = v[:self.in_channels]

        if not self.normalized:
            v = self.zscore_channels(v)
        v = self.padcrop_ch(v)
        v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        x_t = torch.from_numpy(v).float()
        y_t = torch.tensor(y, dtype=torch.long)

        # Cache write
        if self.cache_dir:
            cp = _cache_path(self.cache_dir, p)
            try:
                with open(cp, "wb") as f:
                    torch.save({"x": x_t, "y": y_t}, f)
            except Exception:
                pass

        return x_t, y_t


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------

def augment_3d_batch(x: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    """
    Apply 3D data augmentation to a batch.

    Augmentations: random flip, Gaussian noise, Gamma jitter, 3D Cutout.
    Args:
        x: [B, C, D, H, W] tensor
        cfg: augmentation config dict with keys:
            flip_p, noise_std, gamma_jitter, cutout_p, cutout_frac
    Returns:
        Augmented tensor (same shape).
    """
    if isinstance(cfg, dict) and not cfg.get("enable", True):
        return x

    p_flip = float(cfg.get('flip_p', 0.3))
    if p_flip > 0.0:
        if torch.rand(()) < p_flip:
            x = x.flip(-1)  # W
        if torch.rand(()) < p_flip:
            x = x.flip(-2)  # H
        if torch.rand(()) < p_flip:
            x = x.flip(-3)  # D

    noise_std = float(cfg.get('noise_std', 0.02))
    if noise_std > 0.0:
        x = x + torch.randn_like(x) * noise_std

    gamma_jitter = float(cfg.get('gamma_jitter', 0.08))
    if gamma_jitter > 0.0:
        g = torch.empty((x.size(0), 1, 1, 1, 1), device=x.device).uniform_(1.0 - gamma_jitter, 1.0 + gamma_jitter)
        x = torch.sign(x) * (torch.abs(x) ** g)

    c_p = float(cfg.get('cutout_p', 0.08))
    c_frac = float(cfg.get('cutout_frac', 0.08))
    if c_p > 0.0 and torch.rand(()) < c_p:
        d = int(max(1, c_frac * x.size(-3)))
        h = int(max(1, c_frac * x.size(-2)))
        w = int(max(1, c_frac * x.size(-1)))
        sd = torch.randint(0, x.size(-3) - d + 1, (1,)).item()
        sh = torch.randint(0, x.size(-2) - h + 1, (1,)).item()
        sw = torch.randint(0, x.size(-1) - w + 1, (1,)).item()
        x[:, :, sd:sd + d, sh:sh + h, sw:sw + w] = 0

    return x


# ---------------------------------------------------------------------------
# Episodic Sampler (for meta-learning)
# ---------------------------------------------------------------------------

class EpisodicBatchSampler(Sampler):
    """
    N-way K-shot Q-query episodic batch sampler for meta-learning.

    Each iteration samples N classes, then K+Q instances per class (without
    replacement), yielding a list of flat indices.

    Args:
        labels: List of integer labels for each sample in the dataset.
        n_way: Number of classes per episode.
        k_shot: Number of support samples per class.
        q_query: Number of query samples per class.
        n_episodes: Number of episodes per epoch.
        class_sampling_weights: Optional per-class sampling probability dict.
    """

    def __init__(
        self,
        labels: List[int],
        n_way: int,
        k_shot: int,
        q_query: int,
        n_episodes: int,
        class_sampling_weights: Optional[Dict[int, float]] = None,
    ):
        self.labels = np.array(labels)
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.n_episodes = n_episodes

        self.classes = sorted(list(set(labels)))
        self.class_indices = {c: np.where(self.labels == c)[0] for c in self.classes}
        self.class_sampling_probs = None
        if class_sampling_weights:
            probs = np.array([max(0.0, float(class_sampling_weights.get(c, 0.0))) for c in self.classes], dtype=np.float64)
            s = float(probs.sum())
            if s > 0:
                self.class_sampling_probs = probs / s

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for _ in range(self.n_episodes):
            batch = []
            if len(self.classes) < self.n_way:
                selected_classes = self.classes
            else:
                if self.class_sampling_probs is not None:
                    selected_classes = np.random.choice(self.classes, self.n_way, replace=False, p=self.class_sampling_probs)
                else:
                    selected_classes = np.random.choice(self.classes, self.n_way, replace=False)

            for c in selected_classes:
                indices = self.class_indices[c]
                if len(indices) < (self.k_shot + self.q_query):
                    raise ValueError(
                        f"Class {c} has {len(indices)} samples, needs {self.k_shot + self.q_query}. "
                        f"Reduce k_shot/q_query or add more data."
                    )
                selected_indices = np.random.choice(indices, self.k_shot + self.q_query, replace=False)
                batch.extend(selected_indices)
            yield batch


def split_task_data(
    x: torch.Tensor,
    y: torch.Tensor,
    n_way: int,
    k_shot: int,
    q_query: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Split an episodic batch into support and query sets.

    Assumes batch is organized as [C1_all, C2_all, ...] where each class has
    K+Q samples.

    Returns:
        support_x, support_y, query_x, query_y
    """
    expected_size = n_way * (k_shot + q_query)
    if x.size(0) != expected_size:
        raise ValueError(f"Expected batch size {expected_size}, got {x.size(0)}")

    x_reshaped = x.view(n_way, k_shot + q_query, *x.shape[1:])
    y_reshaped = y.view(n_way, k_shot + q_query)

    support_x = x_reshaped[:, :k_shot].reshape(n_way * k_shot, *x.shape[1:])
    support_y = y_reshaped[:, :k_shot].reshape(n_way * k_shot)
    query_x = x_reshaped[:, k_shot:].reshape(n_way * q_query, *x.shape[1:])
    query_y = y_reshaped[:, k_shot:].reshape(n_way * q_query)

    return support_x, support_y, query_x, query_y


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def _collate_filter_none_impl(batch, collate_fn):
    """Module-level impl so the returned collate is picklable (Windows spawn)."""
    batch = [b for b in batch if b is not None]
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long)
    return collate_fn(batch)


def make_collate_filter_none(collate_fn=None):
    """Wrap a collate function to filter out None samples.

    Returns a picklable function (functools.partial over a module-level impl)
    so DataLoader works under spawn-based multiprocessing (Windows)."""
    from functools import partial
    if collate_fn is None:
        from torch.utils.data.dataloader import default_collate
        collate_fn = default_collate
    return partial(_collate_filter_none_impl, collate_fn=collate_fn)


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(cfg: Dict[str, Any]) -> MRIVolumeFolderDataset:
    """
    Build an MRIVolumeFolderDataset from a config dict.

    Expected config keys:
        data.label_roots: {label_name: path}
        data.target_shape: [D, H, W]
        data.folder_file_exts: [".nii", ...]
        dataset.cache_enabled: bool
        dataset.cache_dir: str
    """
    data_cfg = cfg.get("data", {})
    ds_cfg = cfg.get("dataset", {})

    label_map = {"AD": 0, "LBD": 1}
    target_shape = tuple(data_cfg.get("target_shape", [112, 112, 112]))
    file_exts = tuple(data_cfg.get("folder_file_exts", [".nii", ".nii.gz", ".npy"]))
    cache_enabled = ds_cfg.get("cache_enabled", False)
    cache_dir = ds_cfg.get("cache_dir", None)
    normalized = data_cfg.get("normalized", False)

    return MRIVolumeFolderDataset(
        label_roots=data_cfg["label_roots"],
        label_map=label_map,
        target_shape=target_shape,
        file_exts=file_exts,
        cache_enabled=cache_enabled,
        cache_dir=cache_dir,
        normalized=normalized,
    )
