# -*- coding: utf-8 -*-
import os
import sys
import yaml
import math
import time
import csv
import argparse
import copy
import json
import re
import subprocess
import contextlib
import tracemalloc
import hashlib
import random
from typing import Optional, Tuple, List
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, ReduceLROnPlateau
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data._utils.collate import default_collate
from torch.utils.data import WeightedRandomSampler
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False
try:
    from sklearn.model_selection import StratifiedKFold, KFold, StratifiedShuffleSplit
    _HAS_SKLEARN_MS = True
except Exception:
    _HAS_SKLEARN_MS = False


# 开启行缓冲，尽量让打印立即输出（在非交互式环境也更稳）
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# 依赖现有模型实现
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from medmamba3d import MedMamba3D
from medmamba_ss3m import MedMambaSS3M
from medmamba_ss3m_2dscan import MedMambaSS3M2DScan


class NormedLinearHead(torch.nn.Module):
    def __init__(self, in_dim: int, out_dim: int, temperature: float = 1.0):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(out_dim, in_dim))
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        self.temperature = max(float(temperature), 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.normalize(x, dim=1)
        w = F.normalize(self.weight, dim=1)
        return F.linear(x, w) / self.temperature


def _build_head_activation(name: str):
    n = str(name or "gelu").lower()
    if n == "silu":
        return torch.nn.SiLU()
    if n == "leaky_relu":
        return torch.nn.LeakyReLU(negative_slope=0.1, inplace=True)
    return torch.nn.GELU()


def build_classifier_head(in_dim: int, num_classes: int, cfg: dict):
    cls_cfg = cfg.get("classifier", {}) if isinstance(cfg, dict) else {}
    head_type = str(cls_cfg.get("head_type", "linear")).lower()

    if head_type == "normed":
        temp = float(cls_cfg.get("head_temperature", 1.0))
        return NormedLinearHead(in_dim, num_classes, temperature=temp), head_type

    if head_type == "mlp":
        hidden = int(cls_cfg.get("head_hidden_dim", max(in_dim // 2, 16)))
        drop = float(cls_cfg.get("head_dropout", 0.3))
        act = _build_head_activation(cls_cfg.get("head_activation", "gelu"))
        return torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden),
            act,
            torch.nn.Dropout(drop),
            torch.nn.Linear(hidden, num_classes),
        ), head_type

    return torch.nn.Linear(in_dim, num_classes), "linear"


def _normalize_site_mode(site_mode):
    mode = str(site_mode or "both").strip().lower()
    if mode in ("both", "all", "mix", "mixed", "combined"):
        return "both"
    if mode in ("1.5t", "1p5t", "1_5t", "15t", "1.5", "1p5"):
        return "1.5T"
    if mode in ("3t", "3"):
        return "3T"
    return mode


def _filter_label_roots_by_site_mode(label_roots, site_mode):
    normalized = _normalize_site_mode(site_mode)
    if not isinstance(label_roots, dict) or normalized == "both":
        return label_roots

    site_key = "1.5t" if normalized == "1.5T" else "3t" if normalized == "3T" else None
    if site_key is None:
        return label_roots

    filtered = {}
    for label_name, root in label_roots.items():
        roots = root if isinstance(root, (list, tuple)) else [root]
        selected = [r for r in roots if isinstance(r, str) and site_key in r.lower()]
        if selected:
            filtered[label_name] = selected
    return filtered


def apply_backbone_and_head_optimizations(model: torch.nn.Module, cfg: dict, num_classes: int):
    cls_cfg = cfg.get("classifier", {}) if isinstance(cfg, dict) else {}

    # 1) Backbone normalization replacement for small-batch stability.
    norm_type = str(cls_cfg.get("backbone_norm", "batchnorm")).lower()
    if hasattr(model, "norm") and isinstance(model.norm, torch.nn.BatchNorm3d):
        ch = int(model.norm.num_features)
        if norm_type == "groupnorm":
            groups = int(cls_cfg.get("backbone_norm_groups", 8))
            groups = max(1, min(groups, ch))
            while ch % groups != 0 and groups > 1:
                groups -= 1
            model.norm = torch.nn.GroupNorm(groups, ch)
            print(f"[BackboneOpt] norm: BatchNorm3d -> GroupNorm(groups={groups}, channels={ch})", flush=True)
        elif norm_type == "layernorm":
            model.norm = torch.nn.GroupNorm(1, ch)
            print(f"[BackboneOpt] norm: BatchNorm3d -> GroupNorm(groups=1, channels={ch})", flush=True)

    # 2) Classifier head replacement.
    if hasattr(model, "head"):
        in_dim = None
        if hasattr(model.head, "in_features"):
            in_dim = int(model.head.in_features)
        elif hasattr(model, "classifier") and hasattr(model.classifier, "in_features"):
            in_dim = int(model.classifier.in_features)
        if in_dim is not None:
            new_head, head_type = build_classifier_head(in_dim, num_classes, cfg)
            model.head = new_head
            model.classifier = new_head
            print(f"[HeadOpt] head_type={head_type} in_dim={in_dim} num_classes={num_classes}", flush=True)

    return model


def resolve_device(cfg: dict):
    device_cfg = cfg.get("device", {}) if isinstance(cfg, dict) else {}
    force = str(device_cfg.get("force", "")).lower()
    
    cuda_available = torch.cuda.is_available()
    print(f"[Device] torch.cuda.is_available() = {cuda_available}")
    if cuda_available:
        print(f"[Device] Device count: {torch.cuda.device_count()}")
        print(f"[Device] Current device: {torch.cuda.current_device()}")
        print(f"[Device] Device name: {torch.cuda.get_device_name(0)}")
    else:
        print("[Device] WARNING: CUDA is NOT available. PyTorch cannot see any GPU.")

    if force == "cuda":
        if not cuda_available:
            raise RuntimeError("Config requests 'force: cuda' but torch.cuda.is_available() is False! Check your environment/driver.")
        
        requested = int(device_cfg.get("cuda_device", 0))
        if requested < 0: requested = 0
        torch.cuda.set_device(requested)
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        print(f"[Device] Resolved to {device} (Forced)")
        return device, "cuda"

    if force == "cpu":
        print(f"[Device] Resolved to cpu (Forced)")
        return torch.device("cpu"), "cpu"
        
    device_type = 'cuda' if cuda_available else 'cpu'
    if device_type == 'cuda':
        requested = int(device_cfg.get("cuda_device", 0))
        if requested < 0: requested = 0
        torch.cuda.set_device(requested)
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        print(f"[Device] Resolved to {device} (Auto)")
        return device, device_type
        
    print(f"[Device] Resolved to cpu (Auto - CUDA unavailable)")
    return torch.device("cpu"), device_type

def _normalize_path(path: str) -> str:
    return os.path.realpath(os.path.normpath(path))

def _cache_key_for_path(path: str) -> str:
    norm = _normalize_path(path)
    try:
        st = os.stat(norm)
        # Use st_mtime (int) for stability, plus size.
        # We omit inode to avoid issues on network drives, but keep it if local is preferred.
        # For safety/performance trade-off, path+mtime+size is usually good enough.
        key = f"{norm}|{st.st_size}|{int(st.st_mtime)}"
    except Exception:
        # If file not found (weird), just use path
        key = norm
    
    try:
        import xxhash
        return xxhash.xxh64(key, seed=0).hexdigest()
    except Exception:
        return hashlib.blake2b(key.encode("utf-8"), digest_size=16).hexdigest()

def _cache_path(cache_dir: str, path: str) -> str:
    return os.path.join(cache_dir, f"{_cache_key_for_path(path)}.pt")


def read_nvidia_smi():
    cmd = [
        "nvidia-smi",
        "--query-gpu=timestamp,power.draw,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits"
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None
    if not out:
        return None
    line = out.splitlines()[0]
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 5:
        return None
    return {
        "timestamp": parts[0],
        "power_draw": parts[1],
        "util_gpu": parts[2],
        "mem_used": parts[3],
        "mem_total": parts[4]
    }

def init_gpu_log(out_dir: str):
    log_path = os.path.join(out_dir, "gpu_monitor.csv")
    header = [
        "epoch", "step", "iter_time_s", "data_time_s", "gpu_time_s",
        "cuda_device", "cuda_name", "mem_alloc_bytes", "mem_reserved_bytes",
        "smi_timestamp", "smi_power_w", "smi_util_gpu_pct", "smi_mem_used_mb", "smi_mem_total_mb"
    ]
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        f.flush()
    return log_path

def append_gpu_log(log_path: str, row: list):
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.flush()

def init_benchmark_log(out_dir: str):
    log_path = os.path.join(out_dir, "benchmark_report.csv")
    header = [
        "mode", "steps", "batch_size", "shape_dhw", "total_time_s", "avg_step_time_s",
        "cpu_mem_peak_bytes", "cuda_mem_peak_bytes"
    ]
    if not os.path.isfile(log_path):
        with open(log_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            f.flush()
    return log_path

def run_synthetic_benchmark(cfg: dict, bench_cfg: dict):
    out_dir_cfg = cfg['paths']['out_dir']
    if os.name == "nt" and isinstance(out_dir_cfg, str) and out_dir_cfg.startswith("/"):
        out_dir = os.path.join(os.path.dirname(__file__), "out")
    else:
        out_dir = out_dir_cfg
    os.makedirs(out_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    cv_log_dir = os.path.join(out_dir, "logs", f"cv_10fold_{timestamp}")
    os.makedirs(cv_log_dir, exist_ok=True)
    with open(os.path.join(cv_log_dir, "config_snapshot.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    benchmark_seed = int(bench_cfg.get("seed", cfg.get("cv", {}).get("cv_random_state", cfg.get("cv", {}).get("seed", 42))))
    with open(os.path.join(cv_log_dir, "seed.txt"), "w", encoding="utf-8") as f:
        f.write(f"seed={benchmark_seed}\n")
    steps = int(bench_cfg.get("steps", 20))
    batch_size = int(bench_cfg.get("batch_size", 2))
    in_chans = int(bench_cfg.get("in_chans", 1))
    shape = tuple(bench_cfg.get("shape_dhw", cfg['input']['shape_dhw']))
    device_cfg = {
        "device": {
            "force": bench_cfg.get("force_device", ""),
            "cuda_device": bench_cfg.get("cuda_device", 0)
        }
    }
    device, device_type = resolve_device(device_cfg)
    model = torch.nn.Sequential(
        torch.nn.Conv3d(in_chans, 8, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.AdaptiveAvgPool3d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(8, 2)
    ).to(device)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    if device_type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()
    step_times = []
    sync_cuda = bool(bench_cfg.get("sync_cuda", False))
    start = time.time()
    for _ in range(steps):
        if device_type == 'cuda' and sync_cuda:
            torch.cuda.synchronize() 
            gpu_start = time.time()
        step_start = time.time()
        x = torch.randn((batch_size, in_chans, *shape), device=device)
        y = torch.randint(0, 2, (batch_size,), device=device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        opt.step()
        opt.zero_grad()
        if device_type == 'cuda' and sync_cuda:
            torch.cuda.synchronize()
            gpu_time = time.time() - gpu_start
        step_times.append(time.time() - step_start)
        
    total_time = time.time() - start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    cuda_peak = torch.cuda.max_memory_allocated() if device_type == 'cuda' else 0
    log_path = init_benchmark_log(out_dir)
    row = [
        device_type, steps, batch_size, "x".join([str(v) for v in shape]),
        f"{total_time:.6f}", f"{(sum(step_times) / max(1, len(step_times))):.6f}",
        peak_mem, cuda_peak
    ]
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(row)
        f.flush()
    print(f"[Benchmark] mode={device_type} total_time_s={total_time:.3f} avg_step_s={sum(step_times) / max(1, len(step_times)):.3f}", flush=True)


def load_config(default_path: str) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_path)
    parser.add_argument("--metrics", type=str, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--min-delta", type=float, default=None)
    parser.add_argument("--lr-scheduler", type=str, default=None)
    parser.add_argument("--lr-factor", type=float, default=None)
    parser.add_argument("--lr-patience", type=int, default=None)
    parser.add_argument("--monitor", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--folds", "--cv_n_splits", dest="folds", type=int, default=None)
    parser.add_argument("--run-folds", "--run_folds", dest="run_folds", type=int, default=None)
    parser.add_argument("--split-mode", type=str, default=None, help="Split mode: nested_cv (fixed)")
    parser.add_argument("--train-ratio", type=float, default=None, help="Deprecated for fixed nested_cv mode")
    parser.add_argument("--val-ratio", type=float, default=None, help="Deprecated for fixed nested_cv mode")
    parser.add_argument("--test-ratio", type=float, default=None, help="Deprecated for fixed nested_cv mode")
    parser.add_argument("--locked-test-ratio", type=float, default=None, help="Locked test ratio for nested_cv")
    parser.add_argument("--outer-folds", type=int, default=None, help="Number of CV folds for nested_cv")
    parser.add_argument("--patient-id-mode", type=str, default=None, help="Patient ID extraction mode: parent_dir or regex")
    parser.add_argument("--patient-id-regex", type=str, default=None, help="Regex to extract patient ID from paths")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--synthetic-benchmark", action="store_true")
    parser.add_argument("--benchmark-steps", type=int, default=20)
    parser.add_argument("--benchmark-batch", type=int, default=2)
    parser.add_argument("--benchmark-device", type=str, default=None)
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="Override nested config parameters: param1=val1 param2=val2")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.opts:
        import ast
        for opt in args.opts:
            if "=" not in opt:
                continue
            key_path, val_str = opt.split("=", 1)
            # 尝试解析布尔、数字等类型
            try:
                val = ast.literal_eval(val_str)
            except Exception:
                # 若无法解析（如单纯是字符串 比如 "anil"），则保持为字符串
                val = val_str
                
            keys = key_path.split(".")
            cur = cfg
            for k in keys[:-1]:
                if k not in cur or not isinstance(cur[k], dict):
                    cur[k] = {}
                cur = cur[k]
            cur[keys[-1]] = val
            print(f"[Config Override] {key_path} = {val}")

    if args.metrics:
        cfg["metrics"] = [m.strip() for m in args.metrics.split(",") if m.strip()]
    if args.patience is not None:
        cfg.setdefault("early_stopping", {})["patience"] = int(args.patience)
    if args.min_delta is not None:
        cfg.setdefault("early_stopping", {})["min_delta"] = float(args.min_delta)
    if args.lr_scheduler is not None:
        cfg.setdefault("classifier", {})["lr_scheduler"] = args.lr_scheduler
    if args.lr_factor is not None:
        cfg.setdefault("classifier", {})["lr_factor"] = float(args.lr_factor)
    if args.lr_patience is not None:
        cfg.setdefault("classifier", {})["lr_patience"] = int(args.lr_patience)
    if args.monitor is not None:
        cfg.setdefault("early_stopping", {})["monitor"] = args.monitor
    if args.epochs is not None:
        cfg.setdefault("classifier", {})["epochs"] = int(args.epochs)
    requested_split_mode = str(args.split_mode).strip().lower() if args.split_mode is not None else None
    if requested_split_mode is not None and requested_split_mode != "nested_cv":
        raise RuntimeError("This project is locked to nested_cv validation only.")
    if args.folds is not None and int(args.folds) != 5:
        raise RuntimeError("Only 5-fold nested_cv is allowed. Please set --folds 5.")
    if args.run_folds is not None and int(args.run_folds) != 5:
        raise RuntimeError("Only full 5-fold execution is allowed. Please set --run-folds 5.")
    if args.outer_folds is not None and int(args.outer_folds) != 5:
        raise RuntimeError("Only nested_cv.outer_folds=5 is allowed.")
    if args.train_ratio is not None or args.val_ratio is not None or args.test_ratio is not None:
        print("[CV Config][Info] --train-ratio/--val-ratio/--test-ratio are ignored in fixed nested_cv mode.", flush=True)

    cv_cfg = cfg.setdefault("cv", {})
    cv_cfg["split_mode"] = "nested_cv"
    cv_cfg["folds"] = 5
    cv_cfg["cv_n_splits"] = 5
    cv_cfg["run_folds"] = 5

    if args.locked_test_ratio is not None:
        cv_cfg.setdefault("nested_cv", {})["locked_test_ratio"] = float(args.locked_test_ratio)
    cv_cfg.setdefault("nested_cv", {})["outer_folds"] = 5
    if args.patient_id_mode is not None:
        cv_cfg.setdefault("nested_cv", {})["patient_id_mode"] = str(args.patient_id_mode)
    if args.patient_id_regex is not None:
        cv_cfg.setdefault("nested_cv", {})["patient_id_regex"] = str(args.patient_id_regex)
    if args.debug:
        cfg.setdefault("debug", {})["enabled"] = True
    if args.synthetic_benchmark:
        cfg.setdefault("benchmark", {})["enabled"] = True
        cfg["benchmark"]["steps"] = int(args.benchmark_steps)
        cfg["benchmark"]["batch_size"] = int(args.benchmark_batch)
        if args.benchmark_device:
            cfg["benchmark"]["force_device"] = args.benchmark_device
    return cfg


def _best_effort_load_pretrained(
    model: torch.nn.Module,
    raw_state,
    dry_run: bool = False,
    verbose: bool = True,
) -> dict:
    """
    Load the maximum number of shape-compatible parameters from a checkpoint.
    """
    container_key = "raw"
    if isinstance(raw_state, dict):
        # Support common checkpoint containers used across this repo.
        for key in ('state_dict', 'model_state', 'model', 'student', 'teacher', 'ema', 'backbone'):
            v = raw_state.get(key)
            if isinstance(v, dict):
                raw_state = v
                container_key = key
                break

    if not isinstance(raw_state, dict):
        return {"loaded": 0, "total_model": len(model.state_dict()), "note": "invalid_state_dict"}

    model_sd = model.state_dict()

    def _strip_prefixes(sd: dict, prefixes: Tuple[str, ...]) -> dict:
        out = {}
        for k, v in sd.items():
            nk = k
            changed = True
            while changed:
                changed = False
                for p in prefixes:
                    if nk.startswith(p):
                        nk = nk[len(p):]
                        changed = True
            out[nk] = v
        return out

    def _alias_patch_to_patch_embed(sd: dict) -> dict:
        out = {}
        for k, v in sd.items():
            if not isinstance(k, str):
                continue
            nk = k
            if nk.startswith("patch."):
                nk = "patch_embed." + nk[len("patch."):]
            out[nk] = v
        return out

    def _prefix_hit_stats(keys, prefixes: Tuple[str, ...]) -> str:
        stats = []
        total = max(1, len(keys))
        for p in prefixes:
            c = sum(1 for k in keys if isinstance(k, str) and k.startswith(p))
            if c > 0:
                stats.append(f"{p}:{c}/{total}")
        return ", ".join(stats) if stats else "none"

    def _top_prefix_buckets(keys, max_parts: int = 2, topk: int = 8) -> str:
        counter = {}
        for k in keys:
            if not isinstance(k, str):
                continue
            parts = k.split(".")
            prefix = ".".join(parts[:max_parts]) if parts else k
            counter[prefix] = counter.get(prefix, 0) + 1
        if not counter:
            return "none"
        ordered = sorted(counter.items(), key=lambda x: (-x[1], x[0]))[:topk]
        return ", ".join(f"{k}:{v}" for k, v in ordered)

    prefixes = ("module.", "model.", "encoder.", "backbone.", "student.", "teacher.", "ema.")
    stripped_state = _strip_prefixes(raw_state, prefixes)

    # Special path for SS3M-MAE checkpoints: keep only encoder weights and strip wrapper prefixes.
    encoder_only_state = {}
    for k, v in raw_state.items():
        if not isinstance(k, str):
            continue
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        if nk.startswith("encoder."):
            nk = nk[len("encoder."):]
            # Skip decoder/head weights from pretraining checkpoints.
            if nk.startswith("decoder.") or nk.startswith("head.") or nk.startswith("classifier."):
                continue
            encoder_only_state[nk] = v

    stripped_patch_alias_state = _alias_patch_to_patch_embed(stripped_state)
    encoder_patch_alias_state = _alias_patch_to_patch_embed(encoder_only_state)

    # ------------------------------------------------------------------
    # ssm_b → ssm_a remapping for single-branch ablation
    #
    # During dual-branch MAE pretraining, ssm_b (mamba) received the same
    # input `feat_seqs_batch` as ssm_a (conv) and was trained via real
    # reconstruction gradients backpropagated through the fused output.
    # When running a single-branch (mamba-only) ablation, the pretrained
    # mamba weights in ssm_b.* are semantically valid for ssm_a.* —
    # same input format, same SeqSSM(mamba) architecture, same data
    # distribution. This remap lifts pretrained coverage from 38/96 (40%)
    # to ~88/96 (92%), matching the conv-only single-branch baseline.
    # ------------------------------------------------------------------
    def _remap_ssm_b_to_ssm_a(sd: dict) -> dict:
        """Return a copy of *sd* with every ``ssm_b.`` key renamed to ``ssm_a.``."""
        out = {}
        for k, v in sd.items():
            if isinstance(k, str) and ".ssm_b." in k:
                out[k.replace(".ssm_b.", ".ssm_a.")] = v
            else:
                out[k] = v
        return out

    encoder_ssm_remap_state = _remap_ssm_b_to_ssm_a(encoder_only_state)
    candidates = [
        raw_state,
        stripped_state,
        encoder_only_state,
        stripped_patch_alias_state,
        encoder_patch_alias_state,
        encoder_ssm_remap_state,                         # ← new: ssm_b→ssm_a
    ]

    ckpt_keys = list(raw_state.keys())
    model_keys = list(model_sd.keys())
    if verbose:
        print(
            f"[pretrained][diag] container={container_key} ckpt_keys={len(ckpt_keys)} model_keys={len(model_keys)} "
            f"ckpt_prefix_hits=({_prefix_hit_stats(ckpt_keys, prefixes)})",
            flush=True,
        )
        print(
            f"[pretrained][diag] sample_ckpt_keys={ckpt_keys[:8]} sample_model_keys={model_keys[:8]}",
            flush=True,
        )

    best_matched = {}
    best_candidate = raw_state
    best_candidate_name = "raw"
    candidate_names = [
        "raw", "stripped", "encoder",
        "stripped_patch_alias", "encoder_patch_alias",
        "encoder_ssm_remap",                             # ← new: ssm_b→ssm_a
    ]
    candidate_match_counts = []
    for cand_name, cand in zip(candidate_names, candidates):
        matched = {}
        for k, v in cand.items():
            if k in model_sd and hasattr(v, "shape") and model_sd[k].shape == v.shape:
                matched[k] = v
        candidate_match_counts.append(len(matched))
        if len(matched) > len(best_matched):
            best_matched = matched
            best_candidate = cand
            best_candidate_name = cand_name

    if verbose:
        print(
            "[pretrained][diag] candidate_matches="
            f"raw:{candidate_match_counts[0]} "
            f"stripped:{candidate_match_counts[1]} "
            f"encoder:{candidate_match_counts[2]} "
            f"stripped_patch_alias:{candidate_match_counts[3]} "
            f"encoder_patch_alias:{candidate_match_counts[4]} "
            f"ssm_remap:{candidate_match_counts[5]}",
            flush=True,
        )

    shape_conflict_keys = [
        k for k, v in best_candidate.items()
        if (k in model_sd) and hasattr(v, "shape") and hasattr(model_sd[k], "shape") and (model_sd[k].shape != v.shape)
    ]
    unmatched_model_keys = [k for k in model_sd.keys() if k not in best_matched]
    if verbose:
        print(
            f"[pretrained][diag2] best_candidate={best_candidate_name} "
            f"shape_conflicts={len(shape_conflict_keys)} unmatched_model_keys={len(unmatched_model_keys)} "
            f"shape_conflict_prefixes=({_top_prefix_buckets(shape_conflict_keys)}) "
            f"unmatched_model_prefixes=({_top_prefix_buckets(unmatched_model_keys)})",
            flush=True,
        )

    if not best_matched:
        return {
            "loaded": 0,
            "total_model": len(model_sd),
            "note": "no_shape_matched_keys",
            "best_candidate": best_candidate_name,
        }

    if dry_run:
        return {
            "loaded": len(best_matched),
            "total_model": len(model_sd),
            "missing": 0,
            "unexpected": 0,
            "note": f"dry_run:{best_candidate_name}",
            "best_candidate": best_candidate_name,
        }

    updated = dict(model_sd)
    updated.update(best_matched)
    missing, unexpected = model.load_state_dict(updated, strict=False)
    return {
        "loaded": len(best_matched),
        "total_model": len(model_sd),
        "missing": len(missing),
        "unexpected": len(unexpected),
        "note": f"ok:{best_candidate_name}",
        "best_candidate": best_candidate_name,
    }


def _resolve_pretrained_checkpoint(model: torch.nn.Module, cfg: dict):
    cls_cfg = cfg.get('classifier', {}) if isinstance(cfg.get('classifier', {}), dict) else {}
    primary = cls_cfg.get('pretrained_path')
    candidates = []
    prefer_mae_names = bool(cls_cfg.get('pretrained_prefer_mae_names', True))

    def _infer_ckpt_family(state_obj) -> str:
        sd, _ = _extract_state_dict_container(state_obj)
        if not isinstance(sd, dict):
            return "unknown"
        keys = [k for k in sd.keys() if isinstance(k, str)]
        if not keys:
            return "unknown"
        if any("ssm_a." in k or "ssm_b." in k or "branch_logits" in k or "dir_logits" in k for k in keys):
            return "ss3m"
        if any("dwconv." in k or ".fc1." in k or ".fc2." in k for k in keys):
            return "medmamba3d"
        return "unknown"

    def _infer_model_family(model_obj: torch.nn.Module) -> str:
        mkeys = [k for k in model_obj.state_dict().keys() if isinstance(k, str)]
        if any("ssm_a." in k or "ssm_b." in k or "branch_logits" in k or "dir_logits" in k for k in mkeys):
            return "ss3m"
        if any("dwconv." in k or ".fc1." in k or ".fc2." in k for k in mkeys):
            return "medmamba3d"
        return "unknown"

    if isinstance(primary, str) and primary:
        candidates.append(primary)

    extra = cls_cfg.get('pretrained_path_candidates', [])
    if isinstance(extra, (list, tuple)):
        for p in extra:
            if isinstance(p, str) and p:
                candidates.append(p)

    # Auto-scan sibling checkpoint files so users can provide one path and still leverage nearby weights.
    auto_scan = bool(cls_cfg.get('pretrained_auto_scan_siblings', True))
    max_auto = max(0, int(cls_cfg.get('pretrained_auto_scan_max_files', 12)))
    if auto_scan and isinstance(primary, str) and primary:
        try:
            pdir = os.path.dirname(primary)
            if os.path.isdir(pdir):
                siblings = []
                for fn in os.listdir(pdir):
                    low = fn.lower()
                    if not low.endswith('.pth'):
                        continue
                    if not any(tag in low for tag in ('best', 'mae', 'contrast', 'epoch')):
                        continue
                    siblings.append(os.path.join(pdir, fn))
                siblings = sorted(siblings)[:max_auto]
                candidates.extend(siblings)
        except Exception:
            pass

    dedup_paths = []
    seen = set()
    for p in candidates:
        if p not in seen:
            dedup_paths.append(p)
            seen.add(p)

    model_family = _infer_model_family(model)
    print(f"[pretrained][diag] target_model_family={model_family} candidate_count={len(dedup_paths)}", flush=True)

    best = None
    for p in dedup_paths:
        if not os.path.isfile(p):
            continue
        try:
            state = torch.load(p, map_location='cpu')
            probe = _best_effort_load_pretrained(model, state, dry_run=True, verbose=False)
            loaded = int(probe.get('loaded', 0))
            total = int(probe.get('total_model', len(model.state_dict())))
            note = str(probe.get('note', 'na'))
            ckpt_family = _infer_ckpt_family(state)
            family_match = int(ckpt_family == model_family and ckpt_family != 'unknown')
            basename = os.path.basename(p).lower()
            if prefer_mae_names:
                if ('best_mae' in basename) or ('mae_epoch' in basename):
                    name_pref = 1
                elif basename.startswith('classifier_') or ('best_model' in basename):
                    name_pref = -1
                else:
                    name_pref = 0
            else:
                name_pref = 0
            print(
                f"[pretrained][candidate] path={p} family={ckpt_family} family_match={family_match} "
                f"name_pref={name_pref} probe_loaded={loaded}/{total} note={note}",
                flush=True,
            )
            score = (loaded, family_match, name_pref, -dedup_paths.index(p))
            if (best is None) or (score > best['score']):
                best = {'path': p, 'state': state, 'probe': probe, 'score': score, 'ckpt_family': ckpt_family}
        except Exception as e:
            print(f"[pretrained][candidate][warn] path={p} probe_failed={e}", flush=True)

    if best is None:
        return None, None

    chosen = best['path']
    if int(best['probe'].get('loaded', 0)) == 0:
        print(
            f"[pretrained][diag] chosen checkpoint still has zero compatible keys; "
            f"target_family={model_family}, chosen_family={best.get('ckpt_family', 'unknown')}. "
            "Likely model/checkpoint family mismatch.",
            flush=True,
        )
    if isinstance(primary, str) and primary and chosen != primary:
        print(f"[pretrained][select] use_better_checkpoint={chosen} instead_of={primary}", flush=True)
    else:
        print(f"[pretrained][select] use_checkpoint={chosen}", flush=True)
    return chosen, best['state']


def _extract_state_dict_container(raw_state):
    container_key = "raw"
    sd = raw_state
    if isinstance(raw_state, dict):
        for key in ("state_dict", "model_state", "model", "student", "teacher", "ema", "backbone"):
            v = raw_state.get(key)
            if isinstance(v, dict):
                sd = v
                container_key = key
                break
    return sd, container_key


def _infer_ss3m_hparams_from_checkpoint(pretrained_path: str) -> dict:
    """
    Infer SS3M-compatible embed_dim/depth/patch_size from checkpoint keys/shapes.
    Returns empty dict on failure.
    """
    if not pretrained_path or not os.path.isfile(pretrained_path):
        return {}
    try:
        raw = torch.load(pretrained_path, map_location="cpu")
    except Exception:
        return {}

    sd, container_key = _extract_state_dict_container(raw)
    if not isinstance(sd, dict):
        return {}

    info = {"container": container_key}
    key_candidates = (
        "encoder.patch_embed.proj.weight",
        "patch_embed.proj.weight",
        "encoder.patch.proj.weight",
        "patch.proj.weight",
        "encoder.patch_embed.weight",
        "patch_embed.weight",
        "encoder.patch.weight",
        "patch.weight",
    )
    patch_key = next((k for k in key_candidates if k in sd and hasattr(sd[k], "shape")), None)
    if patch_key is not None:
        w = sd[patch_key]
        if hasattr(w, "shape") and len(w.shape) == 5:
            info["embed_dim"] = int(w.shape[0])
            info["in_channels"] = int(w.shape[1])
            info["patch_size"] = (int(w.shape[2]), int(w.shape[3]), int(w.shape[4]))
            info["patch_key"] = patch_key

    max_block_idx = -1
    pat = re.compile(r"(?:^|\.)blocks\.(\d+)\.")
    for k in sd.keys():
        if not isinstance(k, str):
            continue
        m = pat.search(k)
        if m:
            max_block_idx = max(max_block_idx, int(m.group(1)))
    if max_block_idx >= 0:
        info["depth"] = max_block_idx + 1

    return info


class MRIVolumeLabeledDataset(torch.utils.data.Dataset):
    """
    清单版数据集：(path, label) 列，统一到目标尺寸；假设数据为 1mm, 256x256x192。
    """
    def __init__(self, manifest_csv: str, path_column: str, label_column: str, label_map: dict,
                 target_shape: Tuple[int, int, int] = (112, 112, 112), validate_nifti: bool = True,
                 cache_dir: Optional[str] = None, cache_enabled: bool = False):
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
        self.cache_dir = cache_dir if cache_enabled and cache_dir else None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        if not self.items:
            raise RuntimeError("No labeled items found.")

    def zscore(self, v):
        p1, p99 = np.percentile(v, 1.0), np.percentile(v, 99.0)
        v = np.clip(v, p1, p99)
        m, s = v.mean(), v.std() + 1e-6
        return (v - m) / s

    def padcrop(self, vol):
        D, H, W = vol.shape
        tD, tH, tW = self.target_shape
        pad_d = max(tD - D, 0)
        pad_h = max(tH - H, 0)
        pad_w = max(tW - W, 0)
        if pad_d or pad_h or pad_w:
            vol = self.np.pad(vol, ((pad_d // 2, pad_d - pad_d // 2),
                                    (pad_h // 2, pad_h - pad_h // 2),
                                    (pad_w // 2, pad_w - pad_w // 2)), mode='constant')
        D, H, W = vol.shape
        sD = (D - tD) // 2 if D > tD else 0
        sH = (H - tH) // 2 if H > tH else 0
        sW = (W - tW) // 2 if W > tW else 0
        return vol[sD:sD+tD, sH:sH+tH, sW:sW+tW]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        """
        根据索引获取一个数据样本。
        
        参数:
            idx (int): 数据在列表中的索引值。
            
        返回:
            tuple: 包含预处理后的张量数据 x_t 和对应的标签 y_t (x_t, y_t)。如果读取失败则返回 None。
        """
        p, y = self.items[idx]
        if self.cache_dir:
            cp = _cache_path(self.cache_dir, p)
            if os.path.isfile(cp):
                try:
                    cached = torch.load(cp, map_location="cpu")
                    if isinstance(cached, dict) and "x" in cached and "y" in cached:
                        return cached["x"], cached["y"]
                except Exception:
                    # corrupted cache?
                    pass

        vol = self.nib.load(p).get_fdata().astype(self.np.float32)
        vol = self.zscore(vol)
        vol = self.padcrop(vol)
        vol = self.np.expand_dims(vol, 0)  # (1, D, H, W)
        x_t = torch.from_numpy(vol).float()
        y_t = torch.tensor(y, dtype=torch.long)
        
        if self.cache_dir:
            cp = _cache_path(self.cache_dir, p)
            try:
                # Use xb to ensure we don't overwrite if another process just wrote it (race condition is fine)
                # But to update cache if key changed, we might want 'wb'.
                # Since key depends on mtime, if file changed, key changed, so new file.
                # So 'wb' is fine, or 'xb'.
                # But 'xb' might fail if key collision (unlikely).
                with open(cp, "wb") as f:
                    torch.save({"x": x_t, "y": y_t}, f)
            except Exception:
                pass
        return x_t, y_t


class MRIVolumeFolderDataset(torch.utils.data.Dataset):
    """
    文件夹版数据集：扫描根目录，按 label_map 做标注，推断 in_channels（将额外维折叠为通道）。
    """
    def __init__(self, label_roots: dict, label_map: dict, target_shape: Tuple[int, int, int] = (112, 112, 112),
                 validate_nifti: bool = True, file_exts: tuple = (".nii", ".nii.gz"),
                 require_name_substring: Optional[str] = None,
                 cache_dir: Optional[str] = None, cache_enabled: bool = False):
        import nibabel as nib
        import numpy as np
        self.nib = nib
        self.np = np
        self.items = []
        self.scan_stats = []
        self.target_shape = target_shape
        self.cache_dir = cache_dir if cache_enabled and cache_dir else None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

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
                            count_for_root += 1
                self.scan_stats.append((lab_name, r, True, count_for_root, "ok"))

        if not self.items:
            print("[MRIVolumeFolderDataset][Diag] label_roots scan summary:", flush=True)
            for lab_name, root_path, exists, count, reason in self.scan_stats:
                print(
                    f"  - label={lab_name} exists={exists} count={count} reason={reason} root={root_path}",
                    flush=True
                )
            raise RuntimeError(
                "No labeled items found from label_roots. "
                f"label_map_keys={list(label_map.keys())}, "
                f"file_exts={list(file_exts)}, "
                f"require_name_substring={require_name_substring}"
            )

        # 可选：校验 .nii 文件字节数（跳过损坏样本）
        if validate_nifti:
            valid, invalid = [], []
            print(f"[Dataset] Validating {len(self.items)} NIfTI files (this may take a while)...")
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
                        (valid if actual >= expected else invalid).append((p, y))
                    except Exception:
                        invalid.append((p, y))
                else:
                    valid.append((p, y))
            self.items = valid
            if invalid:
                print(f"[MRIVolumeFolderDataset] skipped {len(invalid)} damaged .nii files.")

        # 推断 in_channels
        max_c = 1
        check_limit = 100 if not validate_nifti else len(self.items)
        checked = 0
        for p, _ in self.items:
            try:
                img = self.nib.load(p)
                shp = tuple(img.shape)
                extra = shp[3:] if len(shp) > 3 else ()
                c = int(self.np.prod(extra)) if extra else 1
                max_c = max(max_c, c)
                checked += 1
                if checked >= check_limit:
                    break
            except Exception:
                pass
        self.in_channels = int(max_c)
        print(f"[MRIVolumeFolderDataset] samples={len(self.items)}, inferred in_channels={self.in_channels}")

    def _to_channel_first_3d(self, arr):
        spatial = arr.shape[:3]
        extra = arr.shape[3:] if arr.ndim > 3 else ()
        C = int(self.np.prod(extra)) if extra else 1
        arr = arr.reshape(spatial[0], spatial[1], spatial[2], C)
        arr = self.np.transpose(arr, (3, 0, 1, 2))  # (C,D,H,W)
        return arr

    def zscore_channels(self, v):
        C = v.shape[0]
        flat = v.reshape(C, -1)
        p1 = self.np.percentile(flat, 1.0, axis=1).reshape(C, 1, 1, 1)
        p99 = self.np.percentile(flat, 99.0, axis=1).reshape(C, 1, 1, 1)
        v = self.np.clip(v, p1, p99)
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
            v = self.np.pad(v, ((0, 0),
                                (pad_d // 2, pad_d - pad_d // 2),
                                (pad_h // 2, pad_h - pad_h // 2),
                                (pad_w // 2, pad_w - pad_w // 2)), mode='constant')
        _, D, H, W = v.shape
        sD = (D - tD) // 2 if D > tD else 0
        sH = (H - tH) // 2 if H > tH else 0
        sW = (W - tW) // 2 if W > tW else 0
        return v[:, sD:sD+tD, sH:sH+tH, sW:sW+tW]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        """
        根据索引获取一个数据样本（含 3D 体数据预处理、Z-score 和中心裁剪等）。
        
        参数:
            idx (int): 样本所在的列表索引。
            
        返回:
            tuple: 返回处理后的 (x_t, y_t) 对应数据张量和标签张量。若加载出错则返回 None。
        """
        p, y = self.items[idx]
        
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
                            self._logged_cache_hit = True
                            return x_cached, cached["y"]
                except Exception:
                    pass

        try:
            arr = self.nib.load(p).get_fdata().astype(self.np.float32)
        except Exception as e:
            print(f"[MRIVolumeFolderDataset] read error, skip: {p} | {e}")
            return None
        v = self._to_channel_first_3d(arr)
        C = v.shape[0]
        if C < self.in_channels:
            v = self.np.pad(v, ((0, self.in_channels - C), (0, 0), (0, 0), (0, 0)), mode='constant')
        elif C > self.in_channels:
            v = v[:self.in_channels]
            
        v = self.zscore_channels(v)
        v = self.padcrop_ch(v)
        v = self.np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
        x_t = torch.from_numpy(v).float()
        y_t = torch.tensor(y, dtype=torch.long)
        
        if self.cache_dir:
            cp = _cache_path(self.cache_dir, p)
            try:
                with open(cp, "wb") as f:
                    torch.save({"x": x_t, "y": y_t}, f)
                self._logged_cache_write = True
            except Exception:
                pass
                
        return x_t, y_t


class MRIVolumeItemsDataset(MRIVolumeFolderDataset):
    def __init__(self, items: list, target_shape: Tuple[int, int, int] = (112, 112, 112),
                 validate_nifti: bool = True,
                 cache_dir: Optional[str] = None, cache_enabled: bool = False):
        import nibabel as nib
        import numpy as np
        self.nib = nib
        self.np = np
        self.items = list(items)
        self.target_shape = target_shape
        self.cache_dir = cache_dir if cache_enabled and cache_dir else None
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        if not self.items:
            raise RuntimeError("No labeled items found from items list.")

        if validate_nifti:
            valid, invalid = [], []
            print(f"[Dataset] Validating {len(self.items)} NIfTI files (this may take a while)...")
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
                        (valid if actual >= expected else invalid).append((p, y))
                    except Exception:
                        invalid.append((p, y))
                else:
                    valid.append((p, y))
            self.items = valid
            if invalid:
                print(f"[MRIVolumeItemsDataset] skipped {len(invalid)} damaged .nii files.")

        max_c = 1
        check_limit = 100 if not validate_nifti else len(self.items)
        checked = 0
        for p, _ in self.items:
            try:
                img = self.nib.load(p)
                shp = tuple(img.shape)
                extra = shp[3:] if len(shp) > 3 else ()
                c = int(self.np.prod(extra)) if extra else 1
                max_c = max(max_c, c)
                checked += 1
                if checked >= check_limit:
                    break
            except Exception:
                pass
        self.in_channels = int(max_c)
        print(f"[MRIVolumeItemsDataset] samples={len(self.items)}, inferred in_channels={self.in_channels}")


def make_safe_collate(in_chans, target_shape):
    def _collate(batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            x = torch.empty((0, in_chans, target_shape[0], target_shape[1], target_shape[2]), dtype=torch.float32)
            y = torch.empty((0,), dtype=torch.long)
            return x, y
        return default_collate(batch)
    return _collate

# 在文件中合适位置添加这些工具函数（例如模型与数据集定义之后）
def augment_3d_batch(x, cfg):
    # x: (B, C, D, H, W)
    p_flip = float(cfg.get('flip_p', 0.5))
    if p_flip > 0.0:
        if torch.rand(()) < p_flip: x = x.flip(-1)  # W
        if torch.rand(()) < p_flip: x = x.flip(-2)  # H
        if torch.rand(()) < p_flip: x = x.flip(-3)  # D
    noise_std = float(cfg.get('noise_std', 0.02))
    if noise_std > 0.0:
        x = x + torch.randn_like(x) * noise_std
    gamma_jitter = float(cfg.get('gamma_jitter', 0.1))
    if gamma_jitter > 0.0:
        g = torch.empty((x.size(0), 1, 1, 1, 1), device=x.device).uniform_(1.0 - gamma_jitter, 1.0 + gamma_jitter)
        x = torch.sign(x) * (torch.abs(x) ** g)
    c_p = float(cfg.get('cutout_p', 0.3))
    c_frac = float(cfg.get('cutout_frac', 0.15))
    if c_p > 0.0 and torch.rand(()) < c_p:
        d = int(max(1, c_frac * x.size(-3)))
        h = int(max(1, c_frac * x.size(-2)))
        w = int(max(1, c_frac * x.size(-1)))
        sd = torch.randint(0, x.size(-3) - d + 1, (1,)).item()
        sh = torch.randint(0, x.size(-2) - h + 1, (1,)).item()
        sw = torch.randint(0, x.size(-1) - w + 1, (1,)).item()
        x[:, :, sd:sd + d, sh:sh + h, sw:sw + w] = 0
    return x

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
    def forward(self, logits, target):
        logpt = F.log_softmax(logits, dim=1)
        pt = logpt.exp()
        logpt_t = logpt.gather(1, target.view(-1, 1)).squeeze(1)
        pt_t = pt.gather(1, target.view(-1, 1)).squeeze(1)
        
        # Add epsilon to prevent NaN gradient when pt_t == 1.0 and gamma < 1.0
        eps = 1e-7
        p_diff = (1.0 - pt_t).clamp(min=eps)

        if self.weight is not None:
            alpha_t = self.weight.to(logits.device).gather(0, target)
            loss = -alpha_t * (p_diff ** self.gamma) * logpt_t
        else:
            loss = -(p_diff ** self.gamma) * logpt_t
        return loss.mean()

def find_best_threshold_bal_acc(y_true_np, y_prob_pos_np):
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


def split_calib_eval_indices(y_true_np: np.ndarray, calib_fraction: float, seed: int):
    """
    Stratified split indices into calibration/evaluation subsets for threshold tuning.
    Returns (calib_idx, eval_idx) or (None, None) if split is not feasible.
    """
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




class EpisodicBatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        labels: List[int],
        n_way: int,
        k_shot: int,
        q_query: int,
        n_episodes: int,
        class_sampling_weights: Optional[dict[int, float]] = None,
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
            # 1. Sample N classes
            # 如果类别数不够 N_way，就全选（或者重复选）
            if len(self.classes) < self.n_way:
                # 这种情况下其实做不了标准的 N-way，这里简化处理：全选
                selected_classes = self.classes
            else:
                if self.class_sampling_probs is not None:
                    selected_classes = np.random.choice(self.classes, self.n_way, replace=False, p=self.class_sampling_probs)
                else:
                    selected_classes = np.random.choice(self.classes, self.n_way, replace=False)
            
            for c in selected_classes:
                indices = self.class_indices[c]
                # 2. Sample K+Q instances per class
                if len(indices) < (self.k_shot + self.q_query):
                    # 【修复】绝不允许有放回采样，这会导致 Support 和 Query 数据泄露！
                    raise ValueError(f"严重错误：类别 {c} 的样本数 ({len(indices)}) "
                                     f"少于单次 Episode 需求的 K+Q ({self.k_shot + self.q_query})。"
                                     "请减小 k_shot/q_query，或扩充该类别的样本数据！")
                else:
                    selected_indices = np.random.choice(indices, self.k_shot + self.q_query, replace=False)
                batch.extend(selected_indices)
            yield batch

def split_task_data(x: torch.Tensor, y: torch.Tensor, n_way: int, k_shot: int, q_query: int):
    """
    将 Batch 数据拆分为 Support 和 Query。
    假设 DataLoader 输出的 batch 是按类别顺序排列的：
    [C1_S1..Sk, C1_Q1..Qq, C2_S1..Sk, C2_Q1..Qq, ...]
    """
    # 重新排列：EpisodicBatchSampler 输出的是 [C1_all, C2_all...]
    # 每个类有 K+Q 个样本
    # 总共有 N_way 个类
    
    # 校验 batch size
    expected_size = n_way * (k_shot + q_query)
    if x.size(0) != expected_size:
        # 可能是最后一个 batch 或者 sampler 逻辑差异，这里做个兼容或者报错
        # 为简单起见，这里假设 batch 结构严格匹配
        raise ValueError(f"Expected batch size {expected_size}, got {x.size(0)}")
        
    # Reshape: [N_way, K+Q, ...]
    # 注意：这里假设 x 的第 0 维是 batch
    x_reshaped = x.view(n_way, k_shot + q_query, *x.shape[1:])
    y_reshaped = y.view(n_way, k_shot + q_query)
    
    # Split
    support_x = x_reshaped[:, :k_shot].reshape(n_way * k_shot, *x.shape[1:])
    support_y = y_reshaped[:, :k_shot].reshape(n_way * k_shot) # 这里 y 是原始 label
    
    query_x = x_reshaped[:, k_shot:].reshape(n_way * q_query, *x.shape[1:])
    query_y = y_reshaped[:, k_shot:].reshape(n_way * q_query)
    
    # 重要：为了让 ProtoNet/MAML 能够计算 Loss，我们需要将 Label 映射到 0..N-1
    # 并在 meta-step 内部使用 local label。
    # 但如果是在 meta-val 算全局 accuracy，可能需要保留 global label。
    # 这里我们返回原始 label，策略内部负责 mapping。
    
    return support_x, support_y, query_x, query_y



def compute_confusion_and_metrics(y_true: List[int], y_pred: List[int], num_classes: int):
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


def compute_class_weights_from_counts(
    class_counts: List[int],
    mode: str = "effective_num",
    beta: float = 0.995,
) -> List[float]:
    """
    根据每类样本数生成更平滑的类别权重。

    改动原因：
    - 原实现直接使用逆频率，在 AD/LBD 约 10:1 的场景下会把少数类权重放大到约 10 倍，
      再叠加 episodic 平衡采样与 focal 后，容易把模型推向“过度追少数类”的状态。
    - 这里默认改为 effective number 权重，让类别补偿仍然存在，但不会过激。
    """
    counts = [max(0, int(c)) for c in class_counts]
    valid_counts = [c for c in counts if c > 0]
    if not valid_counts:
        return [1.0 for _ in counts]

    mode = str(mode or "effective_num").lower()
    weights = []
    if mode == "inverse":
        total = float(sum(valid_counts))
        num_classes = max(1, len(counts))
        for c in counts:
            weights.append(0.0 if c <= 0 else total / (num_classes * float(c)))
    else:
        beta = min(max(float(beta), 0.0), 0.99999)
        for c in counts:
            if c <= 0:
                weights.append(0.0)
                continue
            if beta <= 0.0:
                weights.append(1.0)
                continue
            effective_num = 1.0 - (beta ** float(c))
            weights.append((1.0 - beta) / max(effective_num, 1e-12))

    positive = [w for w in weights if w > 0.0]
    if not positive:
        return [1.0 for _ in counts]
    mean_w = float(sum(positive) / len(positive))
    normalized = [(w / mean_w) if w > 0.0 else 0.0 for w in weights]
    return normalized


def resolve_meta_imbalance_mode(
    requested_mode: str,
    sampler_effective: bool,
    n_way: int,
    unique_cls_in_fold: int,
    respect_episode_balance: bool,
    force_mode: str = "",
) -> Tuple[str, bool]:
    """
    结合 episodic 采样结构，解析本 fold 实际应使用的不均衡补偿模式。
    新增 force_mode：当用户显式传入时，跳过 auto 与 episode_balanced 覆写，
    直接使用 force_mode，供消融实验控制变量使用。
    """
    if force_mode:
        print(f"[balance] force_mode={force_mode} — skipping auto-resolution", flush=True)
        return force_mode, False

    requested_mode = str(requested_mode or "auto").lower()
    if requested_mode not in ("auto", "sampler", "class_weight", "focal", "class_weight_focal"):
        requested_mode = "auto"

    episode_balanced = bool(respect_episode_balance and unique_cls_in_fold > 1 and n_way >= unique_cls_in_fold)
    if requested_mode == "auto":
        if episode_balanced:
            return "episode_balanced", episode_balanced
        if sampler_effective:
            return "sampler", episode_balanced
        return "class_weight_focal", episode_balanced

    if episode_balanced and requested_mode in ("class_weight", "class_weight_focal"):
        return "episode_balanced", episode_balanced
    if requested_mode == "sampler" and not sampler_effective:
        return "class_weight_focal", episode_balanced
    return requested_mode, episode_balanced


def auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    二分类 ROC-AUC（无依赖）：使用秩统计近似（Mann–Whitney U）。
    y_true: 0/1；y_score: 正类概率或分数。
    """
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    x = np.concatenate([neg, pos])
    order = np.argsort(x)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(order.size) + 1  # 1-based ranks
    R_pos = ranks[neg.size:].sum()
    P, N = float(pos.size), float(neg.size)
    auc = (R_pos - P * (P + 1) / 2) / (P * N)
    # print(f"[DEBUG AUC] P={P} N={N} R_pos={R_pos} AUC={auc:.4f}")
    return float(auc)


def auc_multiclass_macro(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int) -> float:
    """
    多类宏平均 AUC：One-vs-Rest，使用二分类 AUC。
    y_true: shape (N,), 值为 [0..C-1]
    y_prob: shape (N,C), 为每类概率
    """
    aucs = []
    for c in range(num_classes):
        y_bin = (y_true == c).astype(np.float32)
        auc_c = auc_binary(y_bin, y_prob[:, c])
        if not np.isnan(auc_c):
            aucs.append(auc_c)
    if not aucs:
        return float("nan")
    return float(sum(aucs) / len(aucs))


def compute_metrics_from_logits(y_true: np.ndarray, logits: np.ndarray, num_classes: int, metric_names: List[str], is_distance: bool = False):
    """
    计算评估指标。
    参数 is_distance: 如果 logits 实际上是距离 (如 ProtoNet 返回的 Euclidean Distance)，
                      必须设为 True，将其取负后再进入 Softmax。
    """
    if logits.ndim == 1:
        logits = logits.reshape(-1, 1)
        
    logits_tensor = torch.tensor(logits)
    
    # 【核心修复 1】：处理 ProtoNet 的距离倒置问题
    if is_distance:
        logits_tensor = -logits_tensor

    # 计算概率
    if num_classes > 1:
        prob = torch.softmax(logits_tensor, dim=1).numpy()
    else:
        prob = torch.sigmoid(logits_tensor).numpy()
        prob = np.stack([1 - prob, prob], axis=1)
        num_classes = 2
        
    y_pred = prob.argmax(axis=1)
    out = {}
    
    # 安全地计算各指标 (使用传入的 metric_names)
    if "acc" in metric_names:
        out["acc"] = float((y_pred == y_true).mean())
    if "f1_macro" in metric_names and _HAS_SKLEARN:
        out["f1_macro"] = float(f1_score(y_true, y_pred, average="macro"))
    if "f1_weighted" in metric_names and _HAS_SKLEARN:
        out["f1_weighted"] = float(f1_score(y_true, y_pred, average="weighted"))
    if "precision_macro" in metric_names and _HAS_SKLEARN:
        out["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    if "recall_macro" in metric_names and _HAS_SKLEARN:
        out["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        
    if "auc" in metric_names:
        out["auc"] = float("nan") # Default to NaN
        if num_classes == 2:
            try:
                out["auc"] = float(roc_auc_score(y_true, prob[:, 1])) if _HAS_SKLEARN else auc_binary(y_true.astype(np.int64), prob[:, 1])
            except Exception:
                pass
        else:
            if _HAS_SKLEARN:
                try:
                    out["auc"] = float(roc_auc_score(y_true, prob, multi_class="ovr", average="macro"))
                except Exception:
                    pass
            else:
                out["auc"] = auc_multiclass_macro(y_true.astype(np.int64), prob, num_classes)
                
    if "confusion_matrix" in metric_names and _HAS_SKLEARN:
        out["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
        
    return out


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = "max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.bad_epochs = 0

    def step(self, value: float):
        if self.best is None:
            self.best = value
            return False
        improved = (value - self.best) > self.min_delta if self.mode == "max" else (self.best - value) > self.min_delta
        if improved:
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def stratified_kfold_indices(items, n_splits: int = 10, seed: int = 42):
    # items: [(path, label), ...]
    labels = [y for _, y in items]
    uniq = sorted(set(labels))
    rng = np.random.RandomState(seed)
    per_class_indices = {c: [] for c in uniq}
    for i, y in enumerate(labels):
        per_class_indices[y].append(i)
    for c in uniq:
        rng.shuffle(per_class_indices[c])
    folds = [[] for _ in range(n_splits)]
    for c in uniq:
        for j, idx in enumerate(per_class_indices[c]):
            folds[j % n_splits].append(idx)
    splits = []
    for k in range(n_splits):
        val_idx = sorted(folds[k])
        train_idx = sorted([i for j in range(n_splits) if j != k for i in folds[j]])
        splits.append((train_idx, val_idx))
    return splits


def stratified_train_val_test_indices(
    items,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
):
    """Create a single stratified train/val/test split."""
    ratio_sum = float(train_ratio) + float(val_ratio) + float(test_ratio)
    if not math.isclose(ratio_sum, 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise RuntimeError(f"train/val/test ratios must sum to 1.0, got {ratio_sum:.6f}")

    labels = [y for _, y in items]
    if not labels:
        raise RuntimeError("Cannot create train/val/test split from an empty dataset.")

    rng = np.random.RandomState(seed)
    per_class_indices = {}
    for sample_idx, label in enumerate(labels):
        per_class_indices.setdefault(label, []).append(sample_idx)

    train_indices, val_indices, test_indices = [], [], []
    for label, indices in per_class_indices.items():
        class_indices = list(indices)
        rng.shuffle(class_indices)

        class_size = len(class_indices)
        if class_size < 3:
            raise RuntimeError(
                f"Class {label} has only {class_size} samples; cannot form train/val/test split with three partitions."
            )

        desired_counts = np.array([
            class_size * float(train_ratio),
            class_size * float(val_ratio),
            class_size * float(test_ratio),
        ], dtype=np.float64)
        counts = np.floor(desired_counts).astype(int)
        remainder = int(class_size - counts.sum())
        if remainder > 0:
            fractional_parts = desired_counts - counts
            allocation_order = np.argsort(-fractional_parts)
            for part_idx in allocation_order[:remainder]:
                counts[part_idx] += 1

        for part_idx in range(3):
            if counts[part_idx] == 0:
                donor_idx = int(np.argmax(counts))
                if counts[donor_idx] <= 1:
                    raise RuntimeError(
                        f"Class {label} is too small for a stable 7:1.5:1.5 split after rounding."
                    )
                counts[donor_idx] -= 1
                counts[part_idx] += 1

        train_end = counts[0]
        val_end = counts[0] + counts[1]
        test_end = counts[0] + counts[1] + counts[2]
        train_indices.extend(class_indices[:train_end])
        val_indices.extend(class_indices[train_end:val_end])
        test_indices.extend(class_indices[val_end:test_end])

        realized = np.array([counts[0], counts[1], counts[2]], dtype=np.float64) / max(class_size, 1)
        target = np.array([train_ratio, val_ratio, test_ratio], dtype=np.float64)
        if np.max(np.abs(realized - target)) > 0.05:
            print(
                f"[Split][warn] class={label} size={class_size} target=({train_ratio:.3f},{val_ratio:.3f},{test_ratio:.3f}) "
                f"realized=({realized[0]:.3f},{realized[1]:.3f},{realized[2]:.3f})",
                flush=True,
            )

    return sorted(train_indices), sorted(val_indices), sorted(test_indices)

def extract_patient_ids(items, patient_id_mode="parent_dir", patient_id_regex=None):
    """
    从文件路径提取患者ID。
    - parent_dir模式: 取父目录名作为患者ID，若父目录名为类别名则取祖父目录
    - regex模式: 用正则提取，需有1个捕获组
    返回: list[str], 长度 == len(items)
    """
    _KNOWN_LABEL_DIRS = {"ad", "lbd", "mci", "normal", "ctrl", "control", "healthy"}
    patient_ids = []
    mode = str(patient_id_mode or "parent_dir").strip().lower()

    for path, _ in items:
        if not isinstance(path, str) or not path:
            patient_ids.append(f"unknown_{len(patient_ids)}")
            continue

        if mode == "regex" and patient_id_regex:
            m = re.search(patient_id_regex, path)
            if m:
                patient_ids.append(m.group(1))
                continue
            print(f"[PatientID][warn] regex did not match: {path}", flush=True)

        parent = os.path.basename(os.path.dirname(path))
        if parent.lower() in _KNOWN_LABEL_DIRS:
            grandparent = os.path.basename(os.path.dirname(os.path.dirname(path)))
            if grandparent:
                patient_ids.append(grandparent)
            else:
                patient_ids.append(parent)
        else:
            patient_ids.append(parent)

    unique_pids = set(patient_ids)
    if len(unique_pids) <= 1:
        raise RuntimeError(
            f"[nested_cv] 所有样本的患者ID都相同 ('{next(iter(unique_pids))}')，"
            "无法做患者级分层划分。请检查数据目录结构，或配置 patient_id_mode/ patient_id_regex。"
        )
    print(f"[PatientID] extracted {len(unique_pids)} unique patients from {len(items)} samples", flush=True)
    return patient_ids


def nested_cv_patient_split(items, patient_ids, test_ratio=0.2, seed=42):
    """
    按患者级别分层划分 dev / locked_test。
    返回: (dev_indices, locked_test_indices, patient_split_info)
    """
    patient_to_indices = {}
    patient_to_label = {}
    for idx, (pid, (_, label)) in enumerate(zip(patient_ids, items)):
        patient_to_indices.setdefault(pid, []).append(idx)
        patient_to_label.setdefault(pid, []).append(label)

    patient_list = sorted(patient_to_indices.keys())
    patient_labels = []
    for pid in patient_list:
        labels_for_patient = patient_to_label[pid]
        from collections import Counter
        majority = Counter(labels_for_patient).most_common(1)[0][0]
        patient_labels.append(majority)

    patient_labels_arr = np.array(patient_labels, dtype=np.int64)
    n_patients = len(patient_list)
    n_test_patients = max(1, int(round(n_patients * test_ratio)))

    if n_test_patients >= n_patients:
        raise RuntimeError(
            f"[nested_cv] test_ratio={test_ratio} 导致测试患者数({n_test_patients}) >= 总患者数({n_patients})。"
        )

    rng = np.random.RandomState(seed)
    per_class_patients = {}
    for i, (pid, plabel) in enumerate(zip(patient_list, patient_labels)):
        per_class_patients.setdefault(plabel, []).append(i)

    test_patient_indices = []
    for plabel, pidx_list in per_class_patients.items():
        rng.shuffle(pidx_list)
        n_class_test = max(1, int(round(len(pidx_list) * test_ratio)))
        n_class_test = min(n_class_test, len(pidx_list) - 1)
        test_patient_indices.extend(pidx_list[:n_class_test])

    test_patient_set = set(test_patient_indices)
    dev_indices = []
    locked_test_indices = []
    patient_split_info = {}

    for i, pid in enumerate(patient_list):
        sample_indices = patient_to_indices[pid]
        if i in test_patient_set:
            locked_test_indices.extend(sample_indices)
            patient_split_info[pid] = "test"
        else:
            dev_indices.extend(sample_indices)
            patient_split_info[pid] = "dev"

    dev_indices = sorted(dev_indices)
    locked_test_indices = sorted(locked_test_indices)

    dev_labels = [items[i][1] for i in dev_indices]
    test_labels = [items[i][1] for i in locked_test_indices]
    print(
        f"[nested_cv] dev={len(dev_indices)} samples ({len(dev_labels)-len(test_labels)+len(test_labels)} total), "
        f"locked_test={len(locked_test_indices)} samples, "
        f"dev_patients={n_patients - len(test_patient_set)}, test_patients={len(test_patient_set)}",
        flush=True,
    )
    if dev_labels:
        from collections import Counter
        dev_dist = Counter(dev_labels)
        print(f"[nested_cv] dev class distribution: {dict(dev_dist)}", flush=True)
    if test_labels:
        from collections import Counter
        test_dist = Counter(test_labels)
        print(f"[nested_cv] locked_test class distribution: {dict(test_dist)}", flush=True)

    return dev_indices, locked_test_indices, patient_split_info


def compute_sensitivity_specificity(cm: torch.Tensor):
    if cm.size(0) != 2 or cm.size(1) != 2:
        return float("nan"), float("nan")
    tn = cm[0, 0].item()
    fp = cm[0, 1].item()
    tp = cm[1, 1].item()
    fn = cm[1, 0].item()
    sens = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    return float(sens), float(spec)

def forward_eval_logits(model, x: torch.Tensor, strategy: str):
    if strategy == 'fine_tuning' and hasattr(model, 'forward_classifier'):
        return model.forward_classifier(x)
    if hasattr(model, 'forward_classifier'):
        return model.forward_classifier(x)
    if hasattr(model, 'forward_encoder'):
        tokens, _ = model.forward_encoder(x)
        if tokens.dim() == 3:
            feats = tokens.mean(dim=1)
        else:
            feats = tokens
        if hasattr(model, 'classifier'):
            return model.classifier(feats)
        if hasattr(model, 'head'):
            return model.head(feats)
    logits = model(x)
    if logits.dim() > 2:
        logits = logits.view(logits.size(0), -1)
    return logits

def evaluate_loader_raw(model, loader, device, num_classes: int, strategy: str):
    # 返回原始 y_true, y_prob 以便进行阈值搜索
    model.eval()
    y_true_all, y_prob_all = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            if y.numel() == 0:
                continue
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = forward_eval_logits(model, x, strategy)
            prob = torch.softmax(logits, dim=1)
            y_true_all.extend(y.detach().cpu().tolist())
            y_prob_all.extend(prob.detach().cpu().numpy().tolist())
    return {"y_true": np.asarray(y_true_all, dtype=np.int64),
            "y_prob": np.asarray(y_prob_all, dtype=np.float32)}


def evaluate_loader_raw_loss(model, loader, device, strategy: str) -> float:
    """Compute mean cross-entropy loss on a plain loader."""
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            if y.numel() == 0:
                continue
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = forward_eval_logits(model, x, strategy)
            loss = F.cross_entropy(logits, y)
            batch_size = int(y.size(0))
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size
    return total_loss / max(total_count, 1)

def evaluate_loader(model, loader, device, num_classes: int, strategy: str, threshold: Optional[float] = None):
    """
    函数用途：
    - 评估给定数据加载器上的分类性能，并返回混淆矩阵、准确率、平衡准确率、AUC、敏感性/特异性（在二分类时）。
    关键改动说明：
    - 新增 threshold 参数（仅二分类有效）。当提供阈值时，使用该阈值对正类概率进行二值化预测，而不是使用 argmax。
      这样可以在不改变训练的前提下，报告“阈值优化版”的 bal_acc，提升对不均衡数据的衡量准确性。
    """
    model.eval()
    y_true_all, y_prob_all = [], []
    with torch.no_grad():
        for batch in loader:
            x, y = batch
            if y.numel() == 0:
                continue
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = forward_eval_logits(model, x, strategy)
            prob = torch.softmax(logits, dim=1)
            y_true_all.extend(y.detach().cpu().tolist())
            y_prob_all.extend(prob.detach().cpu().numpy().tolist())
    if not y_true_all:
        cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        return {"cm": cm, "acc": 0.0, "bal_acc": 0.0, "auc": float("nan"), "sens": float("nan"), "spec": float("nan")}
    y_true_np = np.asarray(y_true_all, dtype=np.int64)
    y_prob_np = np.asarray(y_prob_all, dtype=np.float32)

    # 关键改动：在二分类且提供threshold时，使用阈值进行预测；否则使用argmax
    if num_classes == 2 and threshold is not None:
        y_pred = (y_prob_np[:, 1] >= threshold).astype(np.int64).tolist()
    else:
        y_pred = y_prob_np.argmax(axis=1).astype(np.int64).tolist()

    cm, acc, bal_acc = compute_confusion_and_metrics(list(y_true_np), list(y_pred), num_classes)
    if num_classes == 2:
        pos_prob = y_prob_np[:, 1]
        auc = auc_binary(y_true_np.astype(np.int64), pos_prob)
        sens, spec = compute_sensitivity_specificity(cm)
    else:
        auc = auc_multiclass_macro(y_true_np.astype(np.int64), y_prob_np, num_classes)
        sens, spec = float("nan"), float("nan")
    return {"cm": cm, "acc": acc, "bal_acc": bal_acc, "auc": auc, "sens": sens, "spec": spec}

def compute_metrics_from_raw(y_true: np.ndarray, y_prob: np.ndarray, num_classes: int, threshold: Optional[float] = None):
    """Compute all metrics from already-collected y_true and y_prob (no model forward pass)."""
    if y_true.size == 0:
        cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
        return {"cm": cm, "acc": 0.0, "bal_acc": 0.0, "auc": float("nan"), "sens": float("nan"), "spec": float("nan")}
    if num_classes == 2 and threshold is not None:
        y_pred = (y_prob[:, 1] >= threshold).astype(np.int64).tolist()
    else:
        y_pred = y_prob.argmax(axis=1).astype(np.int64).tolist()
    cm, acc, bal_acc = compute_confusion_and_metrics(list(y_true), list(y_pred), num_classes)
    if num_classes == 2:
        auc = auc_binary(y_true.astype(np.int64), y_prob[:, 1])
        sens, spec = compute_sensitivity_specificity(cm)
    else:
        auc = auc_multiclass_macro(y_true.astype(np.int64), y_prob, num_classes)
        sens, spec = float("nan"), float("nan")
    return {"cm": cm, "acc": acc, "bal_acc": bal_acc, "auc": auc, "sens": sens, "spec": spec}


def save_cm_png(cm: torch.Tensor, class_names, out_path: str):
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
    fig.savefig(out_path)
    plt.close(fig)

def plot_curves(out_dir: str, fold_idx: int, hist: dict):
    """
    绘制训练/验证各项指标曲线（300 DPI，带平滑处理）。
    """
    import numpy as np
    from scipy.signal import savgol_filter

    def smooth_curve(data, window_length=5, polyorder=2):
        if len(data) < window_length:
            return data
        try:
            return savgol_filter(data, window_length, polyorder)
        except Exception:
            return data

    epochs = range(1, len(hist["train_loss"]) + 1)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5), dpi=300)
    
    # Loss
    axs[0].plot(epochs, hist["train_loss"], alpha=0.3, color='tab:blue', label="Train Loss (Raw)")
    axs[0].plot(epochs, smooth_curve(hist["train_loss"]), color='tab:blue', linewidth=2, label="Train Loss (Smooth)")
    axs[0].plot(epochs, hist["val_loss"], alpha=0.3, color='tab:orange', label="Val Loss (Raw)")
    axs[0].plot(epochs, smooth_curve(hist["val_loss"]), color='tab:orange', linewidth=2, label="Val Loss (Smooth)")
    axs[0].set_title("Loss")
    axs[0].set_xlabel("Epoch")
    axs[0].legend()
    axs[0].grid(True, alpha=0.3)

    # Accuracy
    axs[1].plot(epochs, hist["train_acc"], alpha=0.3, color='tab:green', label="Train Acc (Raw)")
    axs[1].plot(epochs, smooth_curve(hist["train_acc"]), color='tab:green', linewidth=2, label="Train Acc (Smooth)")
    axs[1].plot(epochs, hist["val_acc"], alpha=0.3, color='tab:red', label="Val Acc (Raw)")
    axs[1].plot(epochs, smooth_curve(hist["val_acc"]), color='tab:red', linewidth=2, label="Val Acc (Smooth)")
    axs[1].set_title("Accuracy")
    axs[1].set_xlabel("Epoch")
    axs[1].legend()
    axs[1].grid(True, alpha=0.3)

    # AUC
    axs[2].plot(epochs, hist["train_auc"], alpha=0.3, color='tab:purple', label="Train AUC (Raw)")
    axs[2].plot(epochs, smooth_curve(hist["train_auc"]), color='tab:purple', linewidth=2, label="Train AUC (Smooth)")
    axs[2].plot(epochs, hist["val_auc"], alpha=0.3, color='tab:brown', label="Val AUC (Raw)")
    axs[2].plot(epochs, smooth_curve(hist["val_auc"]), color='tab:brown', linewidth=2, label="Val AUC (Smooth)")
    
    # 标记最佳 Val AUC
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
    fig.savefig(os.path.join(out_dir, f"metrics_fold_{fold_idx+1}.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)

def set_bn_eval(model: torch.nn.Module):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

def train_meta(config_path: Optional[str] = None, cv_n_splits_override: Optional[int] = None, run_folds_override: Optional[int] = None):
    """
    函数用途：
    - 进行元训练：包含数据加载、K 折训练、损失与指标记录等。
    改动原因：
    - 集成多策略元学习框架（MAML/ProtoNet/Hybrid）。
    """
    if config_path is None:
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "config.yaml"))
    cfg = load_config(os.path.abspath(config_path))
    print("=== [Config] ===", flush=True)
    print(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), flush=True)
    print("=== [Config End] ===", flush=True)
    device, device_type = resolve_device(cfg)
    
    # 策略判断：虽然 config 提供了详细开关，但为了兼容旧逻辑，我们默认开启 'meta_learning'
    # 如果用户想用旧的 fine_tuning，可以在 classifier.strategy 设置
    # 但这里是 meta 目录，默认跑元学习
    
    loss_type = str(cfg.get('classifier', {}).get('loss', 'ce')).lower()
    do_augment = bool(cfg.get('augment', {}).get('enable', False) or cfg.get('augment', {}).get('enabled', False))
    augment_params = cfg.get('augment', {'flip_p': 0.5, 'noise_std': 0.02, 'gamma_jitter': 0.1, 'cutout_p': 0.3, 'cutout_frac': 0.15})
    out_dir_cfg = cfg['paths']['out_dir']
    site_mode = _normalize_site_mode(cfg.get('paths', {}).get('site_mode', 'both'))
    if os.name == "nt" and isinstance(out_dir_cfg, str) and out_dir_cfg.startswith("/"):
        out_dir = os.path.join(os.path.dirname(__file__), "out")
        cfg.setdefault("paths", {})["out_dir"] = out_dir
    else:
        out_dir = out_dir_cfg
    os.makedirs(out_dir, exist_ok=True)
    cv_cfg = cfg.get('cv', {})
    cv_n_splits = 5
    cv_shuffle = bool(cv_cfg.get('cv_shuffle', True))
    cv_random_state = int(cv_cfg.get('cv_random_state', cv_cfg.get('seed', 42)))
    cv_stratified = bool(cv_cfg.get('stratified', True))
    cv_sample_fraction = float(cv_cfg.get('sample_fraction', 1.0))
    cv_quick_folds = int(cv_cfg.get('quick_folds', 3))
    cv_run_folds = 5
    split_mode = "nested_cv"
    train_ratio = float(cv_cfg.get('train_ratio', 0.7))
    val_ratio = float(cv_cfg.get('val_ratio', 0.15))
    test_ratio = float(cv_cfg.get('test_ratio', 0.15))
    nested_cv_cfg = cv_cfg.get('nested_cv', {}) if isinstance(cv_cfg.get('nested_cv', {}), dict) else {}
    nested_outer_folds = 5
    nested_locked_test_ratio = float(nested_cv_cfg.get('locked_test_ratio', 0.2))
    nested_patient_id_mode = str(nested_cv_cfg.get('patient_id_mode', 'parent_dir'))
    nested_patient_id_regex = nested_cv_cfg.get('patient_id_regex', None)
    if cv_n_splits_override is not None and int(cv_n_splits_override) != 5:
        raise RuntimeError("Only 5-fold nested_cv is allowed. --cv_n_splits must be 5.")
    if run_folds_override is not None and int(run_folds_override) != 5:
        raise RuntimeError("Only full 5-fold execution is allowed. --run_folds must be 5.")
    cv_n_splits = nested_outer_folds
    cv_run_folds = nested_outer_folds
    print(
        f"[CV Config] split_mode={split_mode}, cv_n_splits={cv_n_splits}, run_folds={cv_run_folds}, "
        f"ratios=train:{train_ratio:.3f} val:{val_ratio:.3f} test:{test_ratio:.3f}",
        flush=True,
    )
    print(
        f"[CV Config] nested_cv: locked_test_ratio={nested_locked_test_ratio:.3f}, "
        f"patient_id_mode={nested_patient_id_mode}",
        flush=True,
    )
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_tag = f"nested_cv_{nested_outer_folds}fold"
    cv_log_dir = os.path.join(out_dir, "logs", f"{log_tag}_{timestamp}")
    os.makedirs(cv_log_dir, exist_ok=True)
    with open(os.path.join(cv_log_dir, "config_snapshot.yaml"), "w", encoding="utf-8") as f:
        f.write(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
    with open(os.path.join(cv_log_dir, "seed.txt"), "w", encoding="utf-8") as f:
        f.write(f"seed={cv_random_state}\n")
    stability_cfg = cfg.get("stability", {})
    safe_mode = bool(stability_cfg.get("safe_mode", False))
    if device_type == 'cuda':
        torch.backends.cudnn.benchmark = bool(stability_cfg.get("cudnn_benchmark", True))
        torch.backends.cudnn.deterministic = bool(stability_cfg.get("cudnn_deterministic", False))
    if safe_mode:
        # Issue 4 Fix: Don't limit threads aggressively.
        # torch.set_num_threads(1)
        # torch.set_num_interop_threads(1)
        print("[Performance] Safe mode enabled but thread limitation disabled for performance.")

    metrics = cfg.get('metrics')
    if not metrics:
        metrics = ["acc", "auc", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    metrics = [m for m in metrics if isinstance(m, str)]
    metrics = list(dict.fromkeys(metrics))
    log_csv_path = os.path.join(cv_log_dir, "train_log.csv")
    log_header = [
        "epoch", "train_loss", "val_loss",
        "raw_val_acc", "raw_val_auc", "raw_val_bal_acc", "raw_val_bal_acc_tuned",
        "raw_val_sens", "raw_val_spec", "raw_val_threshold"
    ]
    for m in metrics:
        if m != "confusion_matrix":
            log_header.append(f"train_{m}")
            log_header.append(f"val_{m}")
    try:
        with open(log_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f); writer.writerow(log_header); f.flush()
    except Exception as e:
        print(f"[log] failed to init CSV: {e}", flush=True)

    logging_cfg = cfg.get("logging", {})
    use_tb = bool(logging_cfg.get("tensorboard", True))
    writer = None
    if use_tb:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(cv_log_dir)

    bench_cfg = cfg.get("benchmark", {})
    if bool(bench_cfg.get("enabled", False)):
        run_synthetic_benchmark(cfg, bench_cfg)
        return

    use_folder = bool(cfg['paths'].get('label_roots') or cfg['paths'].get('train_label_roots'))
    target_shape = tuple(cfg['input']['shape_dhw'])
    allowed_labels = cfg.get('dataset', {}).get('allowed_labels')
    use_weighted_sampler = bool(cfg.get('dataset', {}).get('use_weighted_sampler', False))
    
    # Issue 2 Fix: Auto-configure num_workers
    num_workers_cfg = cfg.get('dataset', {}).get('num_workers')
    if num_workers_cfg is None:
        import multiprocessing
        total_cores = multiprocessing.cpu_count()
        num_workers = max(1, min(total_cores - 2, 16))
        print(f"[Config] Auto-setting num_workers={num_workers} (System cores={total_cores})")
    else:
        num_workers = max(1, int(num_workers_cfg))
    prefetch_factor = int(cfg.get('dataset', {}).get('prefetch_factor', 2))

    # Issue 1 Fix: Cache config
    cache_cfg = cfg.get("preprocess_cache", {})
    cache_dir = cache_cfg.get("cache_dir")
    # Default enabled if dir is provided, unless explicitly disabled
    cache_enabled = bool(cache_cfg.get("enabled", True)) if cache_dir else False
    if cache_enabled and cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[Cache] Enabled. Dir: {cache_dir}")

    label_smoothing = float(cfg['classifier'].get('label_smoothing', 0.0))
    warmup_epochs = int(cfg['classifier'].get('warmup_epochs', 3))
    max_grad_norm = float(cfg['classifier'].get('max_grad_norm', 1.0))
    ema_decay = float(cfg['classifier'].get('ema_decay', 0.999))
    epochs = int(cfg['classifier']['epochs'])
    grad_accum = int(cfg['classifier'].get('grad_accum_steps', 1))
    if grad_accum < 1: grad_accum = 1
    
    # Issue 5 Fix: AMP default
    use_amp_cfg = cfg['classifier'].get('use_amp')
    if use_amp_cfg is None:
        use_amp = (device_type == 'cuda')
    else:
        use_amp = bool(use_amp_cfg) and (device_type == 'cuda')
    amp_dtype_cfg = str(cfg['classifier'].get('amp_dtype', 'fp16')).lower()
    if amp_dtype_cfg in ('bf16', 'bfloat16'):
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    use_amp_scaler = bool(use_amp and amp_dtype == torch.float16)
    if use_amp:
        print(f"[AMP] enabled=True dtype={amp_dtype} scaler={'on' if use_amp_scaler else 'off'}", flush=True)
    else:
        print("[AMP] enabled=False", flush=True)

    perf_cfg = cfg.get("performance", {})
    collect_metrics_every = int(perf_cfg.get("collect_metrics_every", 1))
    collect_metrics_every = max(1, collect_metrics_every)
    monitor_gpu = bool(perf_cfg.get("monitor_gpu", False)) and device_type == 'cuda'
    monitor_smi = bool(perf_cfg.get("monitor_smi", False)) and device_type == 'cuda'
    monitor_interval = int(perf_cfg.get("monitor_interval", 50))
    monitor_interval = max(1, monitor_interval)
    scheduler_type = str(cfg['classifier'].get('lr_scheduler', 'cosine')).lower()
    lr_factor = float(cfg['classifier'].get('lr_factor', 0.5))
    lr_patience = int(cfg['classifier'].get('lr_patience', 5))
    freeze_bn = bool(cfg['classifier'].get('freeze_bn', False))
    early_cfg = cfg.get("early_stopping", {})
    early_patience = int(early_cfg.get("patience", 10))
    early_min_delta = float(early_cfg.get("min_delta", 0.0))
    early_monitor = str(early_cfg.get("monitor", "val_acc"))
    early_min_epochs = int(early_cfg.get("min_epochs", 25))  # 防止 val_loss 噪声导致过早停止
    threshold_cfg = cfg.get("thresholding", {}) if isinstance(cfg.get("thresholding", {}), dict) else {}
    allow_tuned_early_stop = bool(threshold_cfg.get("allow_tuned_early_stop", False))
    if (early_monitor in ("val_raw_bal_acc_tuned", "raw_bal_acc_tuned", "val_bal_acc_tuned", "bal_acc_tuned")) and (not allow_tuned_early_stop):
        print(
            f"[EarlyStop][Guard] monitor={early_monitor} uses tuned-threshold metric and may leak validation information; "
            "fallback to val_raw_auc for robust model selection. "
            "Set thresholding.allow_tuned_early_stop=true to override.",
            flush=True,
        )
        early_monitor = "val_raw_auc"
    min_lbd_sens = float(early_cfg.get("min_lbd_sens", -1.0))
    min_lbd_spec = float(early_cfg.get("min_lbd_spec", -1.0))
    early_gate_blocking = bool(early_cfg.get("gate_blocking", False))
    early_mode = "min" if early_monitor == "val_loss" else "max"
    debug_cfg = cfg.get("debug", {})
    if bool(debug_cfg.get("enabled", False)):
        epochs = min(epochs, int(debug_cfg.get("epochs", 1)))
    
    gpu_log_path = None
    if monitor_gpu:
        gpu_log_path = init_gpu_log(out_dir)

    random.seed(cv_random_state)
    np.random.seed(cv_random_state)
    torch.manual_seed(cv_random_state)
    if device_type == 'cuda':
        torch.cuda.manual_seed_all(cv_random_state)

    if use_folder:
        label_map_eff = ({name: idx for idx, name in enumerate(allowed_labels)} if allowed_labels else cfg['dataset']['folder_label_map'])
        eff_class_count = len(label_map_eff)
        num_classes_override = int(cfg.get('classifier', {}).get('num_classes', 0))
        class_names = (allowed_labels if allowed_labels else list(sorted(label_map_eff.keys(), key=lambda k: label_map_eff[k])))
        print(f"[DataConfig] allowed_labels={allowed_labels}", flush=True)
        print(f"[DataConfig] effective label_map={label_map_eff}", flush=True)
        print(f"[DataConfig] site_mode={site_mode}", flush=True)
        raw_label_roots = cfg['paths'].get('label_roots')
        label_roots = _filter_label_roots_by_site_mode(raw_label_roots, site_mode)
        print(f"[DataConfig] paths.label_roots={raw_label_roots}", flush=True)
        print(f"[DataConfig] active_label_roots={label_roots}", flush=True)
        
        datasets = []
        if label_roots:
            datasets.append(MRIVolumeFolderDataset(
                label_roots=label_roots,
                label_map=label_map_eff,
                target_shape=target_shape,
                validate_nifti=False,
                file_exts=tuple(cfg['dataset'].get('folder_file_exts', [".nii", ".nii.gz"])),
                require_name_substring=cfg['dataset'].get('folder_require_substring'),
                cache_dir=cache_dir,
                cache_enabled=cache_enabled
            ))
        else:
            for roots_key in ("train_label_roots", "val_label_roots", "test_label_roots"):
                if cfg['paths'].get(roots_key):
                    datasets.append(MRIVolumeFolderDataset(
                        label_roots=cfg['paths'][roots_key],
                        label_map=label_map_eff,
                        target_shape=target_shape,
                        validate_nifti=False,
                        file_exts=tuple(cfg['dataset'].get('folder_file_exts', [".nii", ".nii.gz"])),
                        require_name_substring=cfg['dataset'].get('folder_require_substring'),
                        cache_dir=cache_dir,
                        cache_enabled=cache_enabled
                    ))
        if not datasets:
            raise RuntimeError("No valid label_roots found in config.")

        if len(datasets) == 1:
            ds = datasets[0]
        else:
            merged_items = []
            for d in datasets:
                merged_items.extend(getattr(d, "items", []))
            ds = MRIVolumeItemsDataset(
                items=merged_items,
                target_shape=target_shape,
                validate_nifti=False,
                cache_dir=cache_dir,
                cache_enabled=cache_enabled
            )

        collate_fn = make_safe_collate(getattr(ds, 'in_channels', 1), target_shape)
        labels_all = [y for _, y in getattr(ds, "items", [])]
        if cv_sample_fraction <= 0.0 or cv_sample_fraction > 1.0:
            raise RuntimeError("cv.sample_fraction must be in (0, 1].")
        if cv_sample_fraction < 1.0:
            if not _HAS_SKLEARN_MS:
                raise RuntimeError("sklearn is required for sample_fraction.")
            splitter = StratifiedShuffleSplit(n_splits=1, train_size=cv_sample_fraction, random_state=cv_random_state)
            sub_idx, _ = next(splitter.split(np.zeros(len(labels_all)), labels_all))
            sub_items = [getattr(ds, "items", [])[i] for i in sub_idx]
            ds = MRIVolumeItemsDataset(
                items=sub_items,
                target_shape=target_shape,
                validate_nifti=False,
                cache_dir=cache_dir,
                cache_enabled=cache_enabled
            )
            labels_all = [y for _, y in getattr(ds, "items", [])]

        patient_ids = extract_patient_ids(
            getattr(ds, "items", []),
            patient_id_mode=nested_patient_id_mode,
            patient_id_regex=nested_patient_id_regex,
        )
        dev_idx, locked_test_idx, patient_info = nested_cv_patient_split(
            getattr(ds, "items", []),
            patient_ids,
            test_ratio=nested_locked_test_ratio,
            seed=cv_random_state,
        )
        split_save_path = os.path.join(cv_log_dir, "nested_cv_split.json")
        with open(split_save_path, "w", encoding="utf-8") as f:
            json.dump({
                "dev_indices": dev_idx,
                "locked_test_indices": locked_test_idx,
                "patient_info": patient_info,
                "n_dev": len(dev_idx),
                "n_test": len(locked_test_idx),
            }, f, ensure_ascii=False, indent=2)
        print(f"[nested_cv] split info saved to {split_save_path}", flush=True)

        dev_labels = [labels_all[i] for i in dev_idx]
        if not _HAS_SKLEARN_MS:
            raise RuntimeError("sklearn is required for nested_cv StratifiedKFold.")
        if cv_stratified and dev_labels:
            dev_class_counts = torch.bincount(torch.tensor(dev_labels, dtype=torch.int64)).tolist()
            if min(dev_class_counts) < cv_n_splits:
                raise RuntimeError(f"nested_cv 5-fold is larger than dev smallest class count {min(dev_class_counts)}.")
        if cv_stratified:
            kfold_dev = StratifiedKFold(n_splits=cv_n_splits, shuffle=cv_shuffle, random_state=cv_random_state)
            local_splits = list(kfold_dev.split(np.zeros(len(dev_labels)), dev_labels))
        else:
            kfold_dev = KFold(n_splits=cv_n_splits, shuffle=cv_shuffle, random_state=cv_random_state)
            local_splits = list(kfold_dev.split(np.zeros(len(dev_labels))))
        dev_idx_arr = np.asarray(dev_idx, dtype=np.intp)
        splits = [(dev_idx_arr[train_local].tolist(), dev_idx_arr[val_local].tolist()) for train_local, val_local in local_splits]
        cv_n_splits = len(splits)

    else:
        # CSV 模式 (略，假设都用 Folder 模式)
        ds = MRIVolumeLabeledDataset(...) 
        # ... (保持原样)
        collate_fn = None
        splits = stratified_kfold_indices(...)
        label_map_eff = ({name: idx for idx, name in enumerate(allowed_labels)} if allowed_labels else cfg['dataset']['label_map'])
        eff_class_count = len(label_map_eff)
        num_classes_override = int(cfg.get('classifier', {}).get('num_classes', 0))
        class_names = (allowed_labels if allowed_labels else list(sorted(label_map_eff.keys(), key=lambda k: label_map_eff[k])))

    # ... (中间代码) ...

    fold_results = []
    for fold_idx, split_item in enumerate(splits):
        if fold_idx >= cv_run_folds:
            print(f"[CV] Stopping after {cv_run_folds} folds as requested.")
            break
        train_idx, val_idx = split_item
        test_idx = None
        split_size_msg = f"train={len(train_idx)} val={len(val_idx)}"
        if test_idx is not None:
            split_size_msg += f" test={len(test_idx)}"
        print(f"[CV {fold_idx+1}/{cv_n_splits}] {split_size_msg}", flush=True)
        ds_train = torch.utils.data.Subset(ds, train_idx)
        ds_val = torch.utils.data.Subset(ds, val_idx)
        ds_test = torch.utils.data.Subset(ds, test_idx) if test_idx is not None else None
        inferred_in_chans = getattr(ds, 'in_channels', 1)

        # 获取 labels 用于采样器
        labels_train = [labels_all[i] for i in train_idx]
        labels_val = [labels_all[i] for i in val_idx]

        # ... (后续代码) ...
        
        # class_counts 计算需要调整
        labels_t = torch.tensor(labels_train, dtype=torch.int64)
        
        # 确保 num_classes 已定义
        # 从 dataset 属性获取，或者从配置获取，或者自动推断
        if 'num_classes' not in locals():
            if 'num_classes_override' in locals() and num_classes_override > 0:
                num_classes = num_classes_override
            elif 'eff_class_count' in locals():
                num_classes = eff_class_count
            else:
                # Fallback: max label + 1
                num_classes = int(labels_t.max().item()) + 1 if len(labels_t) > 0 else 2
        
        class_counts = torch.bincount(labels_t, minlength=num_classes).tolist()
        # ...
        
        # ...
        
        # Validation Sampler
        # ...
        labels_t = torch.tensor(labels_train, dtype=torch.int64)
        class_counts = torch.bincount(labels_t, minlength=num_classes).tolist()
        task_cfg = cfg.get('task', {})
        n_way = int(task_cfg.get('n_way', 2))
        imbalance_cfg = cfg.get("imbalance", {}) if isinstance(cfg.get("imbalance", {}), dict) else {}
        imbalance_mode_req = str(imbalance_cfg.get("mode", "auto")).lower()
        class_weight_mode = str(imbalance_cfg.get("class_weight_mode", "effective_num")).lower()
        class_weight_beta = float(imbalance_cfg.get("class_weight_beta", 0.995))
        respect_episode_balance = bool(imbalance_cfg.get("respect_episode_balance", True))
        balanced_episode_focal_weight = float(imbalance_cfg.get("balanced_episode_focal_weight", 0.10))
        class_weights = compute_class_weights_from_counts(
            class_counts,
            mode=class_weight_mode,
            beta=class_weight_beta,
        )
        ce_weight_fold = torch.tensor(class_weights, dtype=torch.float32, device=device) if loss_type in ('ce', 'focal') else None
        class_sampling_weights = {int(cls_idx): float(class_weights[cls_idx]) for cls_idx in range(len(class_weights))}
        unique_cls_in_fold = len(set(labels_train))
        sampler_effective = bool(use_weighted_sampler and (n_way < unique_cls_in_fold))
        imbalance_force_mode = str(imbalance_cfg.get("force_mode", "")).strip()
        imbalance_mode, episode_balanced_meta = resolve_meta_imbalance_mode(
            requested_mode=imbalance_mode_req,
            sampler_effective=sampler_effective,
            n_way=n_way,
            unique_cls_in_fold=unique_cls_in_fold,
            respect_episode_balance=respect_episode_balance,
            force_mode=imbalance_force_mode,
        )
        if imbalance_mode_req == "sampler" and not sampler_effective:
            print(
                f"[balance][fold {fold_idx+1}] requested sampler but ineffective for n_way={n_way}, classes={unique_cls_in_fold}; "
                "fallback to class_weight_focal.",
                flush=True,
            )

        sampler_enabled_fold = bool(imbalance_mode == "sampler")
        print(
            f"[balance][fold {fold_idx+1}] counts={class_counts} weights={class_weights} "
            f"use_weighted_sampler_cfg={use_weighted_sampler} sampler_effective={sampler_effective} "
            f"imbalance_mode={imbalance_mode} episode_balanced_meta={episode_balanced_meta} "
            f"class_weight_mode={class_weight_mode}",
            flush=True,
        )

        # Prepare Task Config
        task_cfg = cfg.get('task', {})
        n_way = int(task_cfg.get('n_way', 2))
        k_shot = task_cfg.get('k_shot', 5)
        train_q_query = task_cfg.get('q_query', 5) # 训练时使用较大的 q_query 稳定梯度
        train_class_counts = torch.bincount(torch.tensor(labels_train, dtype=torch.int64)).tolist()
        minority_class_count = min(train_class_counts)
        minority_samples_per_episode = max(1, k_shot + train_q_query)
        max_minority_reuse = float(task_cfg.get('max_minority_reuse_per_epoch', 1.5))
        safe_episodes_limit = max(1, int((minority_class_count * max_minority_reuse) // minority_samples_per_episode))
        min_train_episodes = max(1, int(task_cfg.get('min_train_episodes', 10)))
        max_train_episodes = int(task_cfg.get('max_train_episodes', 0))
        train_episodes_override = int(task_cfg.get('train_episodes_override', 0))
        allow_unsafe_override = bool(task_cfg.get('allow_unsafe_episode_override', False))
        min_epoch_coverage_ratio = float(task_cfg.get('min_epoch_coverage_ratio', 0.0))
        auto_train_episodes = safe_episodes_limit
        if max_train_episodes > 0:
            auto_train_episodes = min(auto_train_episodes, max_train_episodes)

        if min_epoch_coverage_ratio > 0.0:
            episodes_for_target = int(math.ceil(
                (min_epoch_coverage_ratio * max(1, len(labels_train))) / max(1, n_way * (k_shot + train_q_query))
            ))
            if episodes_for_target > auto_train_episodes:
                # Never exceed safe limit unless explicitly allowed.
                upper = safe_episodes_limit if not allow_unsafe_override else max(safe_episodes_limit, episodes_for_target)
                auto_train_episodes = min(max(1, episodes_for_target), max(1, upper))
                if max_train_episodes > 0:
                    auto_train_episodes = min(auto_train_episodes, max_train_episodes)
            print(
                f"[Meta-Sampler][Target] min_epoch_coverage_ratio={min_epoch_coverage_ratio:.2f} "
                f"episodes_for_target={episodes_for_target} auto_train_episodes={auto_train_episodes}",
                flush=True,
            )

        if train_episodes_override > 0:
            requested = train_episodes_override
            if (requested > safe_episodes_limit) and (not allow_unsafe_override):
                train_episodes = safe_episodes_limit
                print(
                    f"[Meta-Sampler][Guard] override={requested} exceeds safe_limit={safe_episodes_limit}; "
                    f"capped to safe_limit to avoid minority over-reuse.",
                    flush=True
                )
            else:
                train_episodes = requested
                print(f"[Meta-Sampler] Using train_episodes_override={train_episodes}")
        else:
            train_episodes = auto_train_episodes
            if min_train_episodes > safe_episodes_limit:
                print(
                    f"[Meta-Sampler][Info] min_train_episodes={min_train_episodes} > safe_limit={safe_episodes_limit}; "
                    f"using safe_limit to reduce overfitting risk.",
                    flush=True
                )
        minority_reuse_ratio = (train_episodes * minority_samples_per_episode) / max(1, minority_class_count)
        epoch_effective_samples = int(train_episodes * n_way * (k_shot + train_q_query))
        epoch_coverage_ratio = float(epoch_effective_samples / max(1, len(labels_train)))
        print(f"[Meta-Sampler] Safe train_episodes limit: {safe_episodes_limit}")
        print(f"[Meta-Sampler] Set train_episodes to: {train_episodes} (minority reuse≈{minority_reuse_ratio:.2f}x/epoch)")
        print(f"[Meta-Sampler] Minority class count: {minority_class_count}")
        print(
            f"[Meta-Sampler] epoch_effective_samples={epoch_effective_samples} "
            f"coverage≈{epoch_coverage_ratio:.2%}",
            flush=True,
        )
        # 判断是否为非元学习模式（vanilla / baseline），若是则跳过 Episodic sampling
        is_vanilla_mode = (
            str(cfg.get('meta_learning', {}).get('three_loyal_strategies', {}).get('optimizer_strategy', {}).get('name', 'anil')).lower()
            in ("none", "no_meta", "no-meta", "vanilla", "fine_tuning", "finetune", "baseline")
        )
        if is_vanilla_mode:
            from torch.utils.data import BatchSampler, RandomSampler
            n_per_batch = n_way * (k_shot + train_q_query)
            train_sampler = BatchSampler(
                RandomSampler(ds_train, replacement=True, num_samples=epoch_effective_samples),
                batch_size=n_per_batch,
                drop_last=False,
            )
            print(f"[DataLoader] vanilla mode — using RandomSampler + BatchSampler (no episodic sampling)", flush=True)
        else:
            train_sampler = EpisodicBatchSampler(
                labels_train,
                n_way,
                k_shot,
                train_q_query,
                n_episodes=train_episodes,
                class_sampling_weights=(class_sampling_weights if sampler_enabled_fold else None),
            )
        if use_weighted_sampler and not sampler_enabled_fold:
            print(
                "[balance][info] weighted sampler disabled for this fold because it is ineffective under current n_way/classes; "
                "switching to loss-side compensation.",
                flush=True,
            )
        
        
        dl_num_workers = max(1, int(num_workers))
        dl_pin_memory = (device_type == 'cuda')
        dl_kwargs = {
            "batch_sampler": train_sampler,
            "num_workers": dl_num_workers,
            "pin_memory": dl_pin_memory,
            "collate_fn": collate_fn
        }
        if dl_num_workers > 0:
            dl_kwargs["persistent_workers"] = True
            dl_kwargs["prefetch_factor"] = prefetch_factor
        loader_train = DataLoader(ds_train, **dl_kwargs)
        
        # Validation also needs episodic sampling for meta-validation
        # For vanilla mode, use standard random sampling
        val_q_query = max(1, int(task_cfg.get('q_query', train_q_query)))
        val_episodes_meta = max(1, int(task_cfg.get('val_episodes_meta', 50)))
        use_raw_val_eval = False
        if is_vanilla_mode:
            from torch.utils.data import BatchSampler, RandomSampler
            val_n_per_batch = n_way * (k_shot + val_q_query)
            val_sampler = BatchSampler(
                RandomSampler(ds_val, replacement=False),
                batch_size=val_n_per_batch,
                drop_last=True,
            )
        else:
            val_class_counts = torch.bincount(torch.tensor(labels_val, dtype=torch.int64)).tolist()
            val_min_count = min(val_class_counts)
            if val_min_count < 2:
                use_raw_val_eval = True
                print(
                    f"[CV {fold_idx+1}] validation class count too small for episodic sampling (min={val_min_count}); "
                    "falling back to raw validation evaluation.",
                    flush=True,
                )
            else:
                val_k_shot = min(k_shot, max(1, val_min_count - 1))
                val_q_query = max(1, min(val_min_count - val_k_shot, val_q_query))
                if val_k_shot + val_q_query > val_min_count:
                    use_raw_val_eval = True
                    print(
                        f"[CV {fold_idx+1}] episodic validation cannot fit val_min_count={val_min_count} with "
                        f"val_k_shot={val_k_shot} val_q_query={val_q_query}; falling back to raw validation evaluation.",
                        flush=True,
                    )
                else:
                    if val_k_shot != k_shot:
                        print(
                            f"[CV {fold_idx+1}] adjusted val_k_shot from {k_shot} to {val_k_shot} to fit holdout val split.",
                            flush=True,
                        )
                    val_sampler = EpisodicBatchSampler(labels_val, n_way, val_k_shot, val_q_query, n_episodes=val_episodes_meta)

        if not use_raw_val_eval:
            dl_kwargs["batch_sampler"] = val_sampler # 修复：为验证集替换正确的采样器
            if dl_num_workers > 0:
                dl_kwargs["persistent_workers"] = True
                dl_kwargs["prefetch_factor"] = prefetch_factor
            loader_val = DataLoader(ds_val, **dl_kwargs)
        else:
            loader_val = None
        raw_val_batch_size = max(1, int(cfg.get('classifier', {}).get('batch_size', 4)))
        raw_val_kwargs = {
            "batch_size": raw_val_batch_size,
            "shuffle": False,
            "num_workers": dl_num_workers,
            "pin_memory": dl_pin_memory,
            "collate_fn": collate_fn
        }
        if dl_num_workers > 0:
            raw_val_kwargs["persistent_workers"] = True
            raw_val_kwargs["prefetch_factor"] = prefetch_factor
        raw_val_loader = DataLoader(ds_val, **raw_val_kwargs)
        raw_test_loader = DataLoader(ds_test, **raw_val_kwargs) if ds_test is not None else None

        # Build Model
        backbone_choice = str(cfg.get('classifier', {}).get('backbone', 'medmamba3d')).lower()
        backbone_n_dirs = int(cfg.get('classifier', {}).get('backbone_n_dirs_train', 8))
        backbone_use_dual_branch = bool(cfg.get('classifier', {}).get('backbone_use_dual_branch', True))
        if backbone_choice == "medmamba3d":
            model = MedMamba3D(
                in_chans=inferred_in_chans,
                embed_dim=cfg['model']['embed_dim'],
                depth=cfg['model']['depth'],
                patch_size=tuple(cfg['model']['patch_size']),
                num_classes=num_classes
            ).to(device)
        elif backbone_choice in ("medmamba_ss3m", "ss3m", "medmambass3m"):
            ss3m_cfg = cfg.get('ss3m', {}) if isinstance(cfg.get('ss3m', {}), dict) else {}
            model_cfg = cfg.get('model', {})
            pretrained_path = cfg.get('classifier', {}).get('pretrained_path')
            inferred_ss3m = _infer_ss3m_hparams_from_checkpoint(pretrained_path) if not ss3m_cfg else {}

            ss3m_embed_dim = int(ss3m_cfg.get('embed_dim', inferred_ss3m.get('embed_dim', model_cfg.get('embed_dim', 96))))
            ss3m_depth = int(ss3m_cfg.get('depth', inferred_ss3m.get('depth', model_cfg.get('depth', 4))))
            ss3m_patch_size = tuple(ss3m_cfg.get('patch_size', inferred_ss3m.get('patch_size', model_cfg.get('patch_size', (2, 2, 2)))) )
            ss3m_use_checkpoint = bool(ss3m_cfg.get('use_checkpoint', False))
            ss3m_dropout = float(ss3m_cfg.get('dropout', 0.0))
            ss3m_use_pos_emb = bool(ss3m_cfg.get('use_pos_emb', True))
            ss3m_merge_type = str(ss3m_cfg.get('merge_type', 'softmax'))

            raw_branch_types = ss3m_cfg.get('branch_types', ('conv', 'mamba'))
            if isinstance(raw_branch_types, (list, tuple)) and len(raw_branch_types) >= 2:
                ss3m_branch_types = (str(raw_branch_types[0]), str(raw_branch_types[1]))
            elif isinstance(raw_branch_types, str):
                # Handle shell-quoting issues: "[conv,conv]" or "['conv','conv']" etc.
                s = raw_branch_types.strip().strip("'").strip('"')
                s = s.strip("[]")
                parts = [p.strip().strip("'").strip('"') for p in s.split(",") if p.strip()]
                if len(parts) >= 2:
                    ss3m_branch_types = (str(parts[0]), str(parts[1]))
                    print(f"[Config] parsed ss3m.branch_types from string: {ss3m_branch_types}", flush=True)
                else:
                    ss3m_branch_types = ('conv', 'mamba')
                    print(f"[Config][warn] could not parse ss3m.branch_types='{raw_branch_types}', using default", flush=True)
            else:
                ss3m_branch_types = ('conv', 'mamba')

            if not ss3m_cfg:
                print("[model][warn] classifier.backbone=medmamba_ss3m but config.ss3m missing; fallback to config.model.*", flush=True)
                if inferred_ss3m:
                    print(
                        f"[model][diag] inferred_from_ckpt container={inferred_ss3m.get('container', 'na')} "
                        f"patch_key={inferred_ss3m.get('patch_key', 'na')} embed_dim={inferred_ss3m.get('embed_dim', 'na')} "
                        f"depth={inferred_ss3m.get('depth', 'na')} patch_size={inferred_ss3m.get('patch_size', 'na')}",
                        flush=True,
                    )
            print(
                f"[model][diag] backbone={backbone_choice} in_channels={inferred_in_chans} "
                f"embed_dim={ss3m_embed_dim} depth={ss3m_depth} patch_size={ss3m_patch_size} "
                f"branch_types={ss3m_branch_types} n_dirs_train={backbone_n_dirs} "
                f"use_dual_branch={backbone_use_dual_branch} use_checkpoint={ss3m_use_checkpoint} "
                f"dropout={ss3m_dropout} use_pos_emb={ss3m_use_pos_emb} merge_type={ss3m_merge_type}",
                flush=True,
            )
            print(
                "[model][diag] n_dirs_train=8 is an intentional meta strategy; "
                "it may differ from pretraining defaults.",
                flush=True,
            )

            model = MedMambaSS3M(
                in_channels=inferred_in_chans,
                embed_dim=ss3m_embed_dim,
                depth=ss3m_depth,
                patch_size=ss3m_patch_size,
                num_classes=num_classes,
                n_dirs_train=backbone_n_dirs,
                use_dual_branch=backbone_use_dual_branch,
                branch_types=ss3m_branch_types,
                use_checkpoint=ss3m_use_checkpoint,
                dropout=ss3m_dropout,
                use_pos_emb=ss3m_use_pos_emb,
                merge_type=ss3m_merge_type,
            ).to(device)
        elif backbone_choice in ("medmamba_ss3m_2dscan", "ss3m_2dscan", "ss3m2d", "medmambass3m2dscan"):
            model = MedMambaSS3M2DScan(
                in_channels=inferred_in_chans,
                embed_dim=cfg['model']['embed_dim'],
                depth=cfg['model']['depth'],
                patch_size=tuple(cfg['model']['patch_size']),
                num_classes=num_classes,
                n_dirs_train=min(backbone_n_dirs, 4),
                use_dual_branch=backbone_use_dual_branch,
            ).to(device)
        else:
            raise ValueError(f"Unsupported classifier.backbone: {backbone_choice}")

        resolved_pretrained_path, resolved_pretrained_state = _resolve_pretrained_checkpoint(model, cfg)
        if resolved_pretrained_path and (resolved_pretrained_state is not None):
            state = resolved_pretrained_state
            if isinstance(state, dict):
                top_keys = list(state.keys())
                print(
                    f"[pretrained][diag] path={resolved_pretrained_path} top_keys={top_keys[:8]}",
                    flush=True,
                )
            plog = _best_effort_load_pretrained(model, state)
            print(
                f"[pretrained][fold {fold_idx+1}] matched_load={plog.get('loaded', 0)}/{plog.get('total_model', 0)} "
                f"missing={plog.get('missing', -1)} unexpected={plog.get('unexpected', -1)} note={plog.get('note', 'na')}",
                flush=True
            )
            loaded = int(plog.get('loaded', 0))
            total_model = max(1, int(plog.get('total_model', 0)))
            matched_ratio = float(loaded) / float(total_model)
            min_ratio = float(cfg.get('classifier', {}).get('pretrained_failfast_min_ratio', 0.05))
            warn_ratio = float(cfg.get('classifier', {}).get('pretrained_warn_ratio', 0.20))
            print(
                f"[pretrained][fold {fold_idx+1}] matched_ratio={matched_ratio:.4f} "
                f"thresholds=(fail<{min_ratio:.2f}, warn<{warn_ratio:.2f})",
                flush=True,
            )
            if matched_ratio < min_ratio:
                raise RuntimeError(
                    "[pretrained][fail-fast] matched_ratio is too low. "
                    "Check classifier.backbone, pretrained_path, ss3m embed/depth/patch_size, and ss3m.branch_types."
                )
            if matched_ratio < warn_ratio:
                print(
                    "[pretrained][warn] low matched_ratio; training may under-utilize pretrained weights.",
                    flush=True,
                )

        model = apply_backbone_and_head_optimizations(model, cfg, num_classes).to(device)

        # torch.compile: SM80+ (A800) can get 20-40% speedup on Mamba-style models
        use_compile = bool(cfg.get('classifier', {}).get('use_compile', False))
        if use_compile and device_type == 'cuda':
            try:
                if hasattr(model, 'forward_encoder'):
                    model.forward_encoder = torch.compile(
                        model.forward_encoder,
                        mode="reduce-overhead",
                        fullgraph=False,
                    )
                    print("[Compile] torch.compile enabled for model.forward_encoder", flush=True)
            except Exception as e:
                print(f"[Compile] torch.compile failed, falling back to eager: {e}", flush=True)

        #=====initialize meta Strategy
        meta_cfg = cfg.get('meta_learning', {}).get('three_loyal_strategies', {})
        strategy = None
        is_protonet = False
        opt_params = None
        clf_strategy = str(cfg.get('classifier', {}).get('strategy', '')).lower()
        no_meta_requested = clf_strategy in (
            "fine_tuning",
            "finetune",
            "no_meta",
            "no-meta",
            "vanilla",
            "baseline",
            "none",
        )

        if no_meta_requested and (
            meta_cfg.get('hybrid_strategy', {}).get('enable')
            or meta_cfg.get('metric_strategy', {}).get('enable')
        ):
            print("[Meta][warn] classifier.strategy requests no_meta; ignoring metric/hybrid flags.", flush=True)
        
        # Priority: Hybrid > Optimizer > Metric (Based on enable flag)
        if meta_cfg.get('hybrid_strategy', {}).get('enable') and not no_meta_requested:
            from strategies.hybrid import HybridStrategy
            meta_cfg['optimizer_strategy'].setdefault('params', {})['label_smoothing'] = label_smoothing
            meta_cfg['optimizer_strategy'].setdefault('params', {})['use_class_weights'] = bool(cfg.get('classifier', {}).get('use_class_weights', False))
            meta_cfg['metric_strategy'].setdefault('params', {})['label_smoothing'] = label_smoothing
            hybrid_cfg = dict(meta_cfg.get('hybrid_strategy', {}))
            hybrid_cfg["optimizer_strategy"] = meta_cfg.get("optimizer_strategy", {})
            hybrid_cfg["metric_strategy"] = meta_cfg.get("metric_strategy", {})
            strategy = HybridStrategy(model, hybrid_cfg)
            is_protonet = True
            print(f"[Meta] Using Hybrid Strategy: {hybrid_cfg}")
            
        elif meta_cfg.get('optimizer_strategy', {}).get('enable') or no_meta_requested:
            opt_cfg = meta_cfg.get('optimizer_strategy', {})
            if not isinstance(opt_cfg, dict):
                opt_cfg = {}
            if no_meta_requested:
                opt_cfg.setdefault('name', 'none')
            strat_name = opt_cfg.get('name', 'maml').lower()
            if no_meta_requested and strat_name not in (
                "none",
                "no_meta",
                "no-meta",
                "vanilla",
                "fine_tuning",
                "finetune",
                "baseline",
            ):
                strat_name = 'none'
                opt_cfg['name'] = 'none'
            opt_params = opt_cfg.setdefault('params', {})
            opt_params['label_smoothing'] = label_smoothing
            opt_params['use_compile'] = bool(cfg.get('classifier', {}).get('use_compile', False))
            configured_use_class_weights = bool(cfg.get('classifier', {}).get('use_class_weights', False))
            configured_focal_weight = float(opt_params.get('focal_weight', 0.0))
            if imbalance_mode == 'sampler':
                opt_params['use_class_weights'] = False
                opt_params['inner_use_class_weights'] = False
                opt_params['focal_weight'] = 0.0
                print("[balance][meta] mode=sampler; loss-side compensation disabled.", flush=True)
            elif imbalance_mode == 'class_weight':
                opt_params['use_class_weights'] = configured_use_class_weights
                opt_params['inner_use_class_weights'] = configured_use_class_weights
                opt_params['focal_weight'] = 0.0
                print("[balance][meta] mode=class_weight; focal disabled.", flush=True)
            elif imbalance_mode == 'focal':
                opt_params['use_class_weights'] = False
                opt_params['inner_use_class_weights'] = False
                if configured_focal_weight <= 0.0:
                    opt_params['focal_weight'] = 0.5
                print(f"[balance][meta] mode=focal; focal_weight={opt_params.get('focal_weight', 0.0):.3f}.", flush=True)
            elif imbalance_mode == 'episode_balanced':
                opt_params['use_class_weights'] = False
                opt_params['inner_use_class_weights'] = False
                opt_params['focal_weight'] = min(max(configured_focal_weight, 0.0), max(0.0, balanced_episode_focal_weight))
                print(
                    f"[balance][meta] mode=episode_balanced; disable class weights and cap focal_weight to "
                    f"{opt_params.get('focal_weight', 0.0):.3f} because each episode is already class-balanced.",
                    flush=True,
                )
            else:
                # class_weight_focal: strongest default for extreme imbalance in 2-way setting.
                opt_params['use_class_weights'] = configured_use_class_weights
                opt_params['inner_use_class_weights'] = configured_use_class_weights
                if configured_focal_weight <= 0.0:
                    opt_params['focal_weight'] = 0.5
                print(
                    f"[balance][meta] mode=class_weight_focal; use_class_weights={opt_params['use_class_weights']} "
                    f"focal_weight={opt_params.get('focal_weight', 0.0):.3f}.",
                    flush=True,
                )
            
            if strat_name in (
                "none",
                "no_meta",
                "no-meta",
                "vanilla",
                "fine_tuning",
                "finetune",
                "baseline",
            ):
                from strategies.vanilla import VanillaStrategy
                strategy = VanillaStrategy(model, opt_cfg)
                strategy.to(device)
                print(f"[Meta] Using Vanilla Strategy: {opt_cfg}")
            elif strat_name == 'anil':
                from strategies.anil import ANILStrategy
                strategy = ANILStrategy(model, opt_cfg)
                strategy.to(device)
                print(f"[Meta] Using ANIL Strategy: {opt_cfg}")
            else:
                from strategies.maml import MAMLStrategy
                strategy = MAMLStrategy(model, opt_cfg)
                strategy.to(device)
                print(f"[Meta] Using MAML Strategy: {opt_cfg}")
            is_protonet = False
            
        elif meta_cfg.get('metric_strategy', {}).get('enable'):
            from strategies.protonet import ProtoNetStrategy
            meta_cfg['metric_strategy'].setdefault('params', {})['label_smoothing'] = label_smoothing
            strategy = ProtoNetStrategy(model, meta_cfg['metric_strategy'])
            is_protonet = True 
            print(f"[Meta] Using ProtoNet Strategy: {meta_cfg['metric_strategy']}")
        else:
            from strategies.vanilla import VanillaStrategy
            opt_cfg = meta_cfg.get('optimizer_strategy', {}) if isinstance(meta_cfg.get('optimizer_strategy', {}), dict) else {}
            opt_cfg.setdefault('name', 'none')
            opt_cfg.setdefault('params', {})
            strategy = VanillaStrategy(model, opt_cfg)
            strategy.to(device)
            opt_params = opt_cfg.get('params', {})
            print("[Meta] All strategies disabled, automatically falling back to VanillaStrategy.")
        strategy_params = {}
        if opt_params is not None:
            strategy_params = opt_params
        elif meta_cfg.get('optimizer_strategy', {}).get('enable'):
            strategy_params = meta_cfg.get('optimizer_strategy', {}).get('params', {})
        if hasattr(strategy, "set_class_weights") and bool(strategy_params.get("use_class_weights", False)):
            strategy.set_class_weights(ce_weight_fold)
        elif hasattr(strategy, "set_class_weights"):
            strategy.set_class_weights(None)
        # ========================================================================
        # ==================== 核心重构：差分学习率 (Differential LR) ====================
        head_params = []
        backbone_params = []
        
        for name, param in model.named_parameters():
            param.requires_grad = True # 全面解冻，打通整个 Mamba 的空间特征流
            
            # 精准剥离投影头/分类头参数
            if name.startswith('proj.') or 'head' in name or 'classifier' in name:
                head_params.append(param)
            else:
                backbone_params.append(param)
                
        base_lr = float(cfg['classifier']['lr'])
        backbone_ratio = float(cfg['classifier'].get('backbone_lr_ratio', 0.1)) # 默认骨干 LR 只有头部的 1/10
        
        print(f"[Meta] 启用差分学习率: Backbone LR={base_lr * backbone_ratio:.2e}, Head LR={base_lr:.2e}", flush=True)
        
        opt_groups = [
            {'params': backbone_params, 'lr': base_lr * backbone_ratio},
            {'params': head_params, 'lr': base_lr}
        ]

        # 动态检查策略中是否有注册的元超参数
        if hasattr(strategy, 'inner_lr') and isinstance(strategy.inner_lr, torch.nn.Parameter):
            opt_groups.append({'params': [strategy.inner_lr], 'lr': base_lr})

        opt = torch.optim.AdamW(opt_groups, weight_decay=cfg['classifier']['weight_decay'])
        # ==================================================================================
        

        # 【修改】使用过滤后的 trainable_params，而不是 model.parameters()
        
        scaler = torch.amp.GradScaler('cuda') if use_amp_scaler else None
        cosine = CosineAnnealingLR(opt, T_max=max(1, epochs - warmup_epochs)) if scheduler_type == "cosine" else None
        plateau = ReduceLROnPlateau(opt, mode="max" if early_mode == "max" else "min", factor=lr_factor, patience=lr_patience) if scheduler_type == "plateau" else None
        warmup = LinearLR(opt, start_factor=0.1, total_iters=warmup_epochs) if warmup_epochs > 0 else None
        
        
        hist = {"train_loss": [], "val_loss": [], "train_acc": [], "train_auc": [], "val_acc": [], "val_auc": []}
        best_val_bal_acc, best_epoch = -1.0, -1
        best_cm, best_score, best_metrics, best_raw_metrics = None, None, None, None
        early_stop = EarlyStopping(patience=early_patience, min_delta=early_min_delta, mode=early_mode)
        best_metrics = None
        best_raw_metrics = None
        best_ckpt_path = None
        test_metrics = None

        for epoch in range(epochs):
            model.train()
            if freeze_bn:
                set_bn_eval(model)
            epoch_start = time.time()
            running_loss = 0.0
            running_ce_loss = 0.0
            running_supcon_loss = 0.0
            seen, step_count = 0, 0
            train_metrics_accum = {m: 0.0 for m in metrics if m != "confusion_matrix"}
            train_metrics_counts = {m: 0 for m in metrics if m != "confusion_matrix"}
            train_cm_accum = None
            
            # 【新增】：用于追踪本 Epoch 是否发生过有效参数更新
            optimizer_stepped = False

            for step, batch in enumerate(loader_train):
                x, y = batch
                if y.numel() == 0: continue
                
                x = x.to(device, dtype=torch.float32, non_blocking=True)
                y = y.to(device, non_blocking=True)

                if do_augment:
                    x = augment_3d_batch(x, augment_params)
                    
                if is_vanilla_mode:
                    # Vanilla: 整 batch 直接传入（无 support/query 拆分）
                    # 需构造 dummy 拆分供 split_task_data 兼容，实际训练用 full batch
                    n_per_batch = n_way * (k_shot + train_q_query)
                    if x.size(0) != n_per_batch:
                        x = x[: (x.size(0) // n_per_batch) * n_per_batch]
                        y = y[: (y.size(0) // n_per_batch) * n_per_batch]
                        if x.size(0) == 0:
                            continue
                    try:
                        sx, sy, qx, qy = split_task_data(x, y, n_way, k_shot, train_q_query)
                    except ValueError:
                        continue
                else:
                    try:
                        sx, sy, qx, qy = split_task_data(x, y, n_way, k_shot, train_q_query)
                    except ValueError as e:
                        print(f"Skipping batch {step}: {e}")
                        continue

                amp_ctx = torch.amp.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else contextlib.nullcontext()
                with amp_ctx:
                    step_results = strategy.core_step(sx, sy, qx, qy, optimizer=opt, is_training=True)
                    loss = step_results['loss']
                    ce_loss = step_results.get('ce_loss', loss)
                    supcon_loss = step_results.get('supcon_loss', 0.0)

                if not torch.isfinite(loss).all():
                    print(f"[Train][warn] non-finite loss at step {step}, skip update", flush=True)
                    opt.zero_grad(set_to_none=True)
                    continue
                
                loss = loss / grad_accum
                
                # 梯度反向传播
                # 【修正】计算是否为更新步：满足梯度累积步数，或者是当前 Epoch 的最后一个 Batch
                is_update_step = ((step + 1) % grad_accum == 0) or (step == len(loader_train) - 1)
                
                # Backward with or without GradScaler
                if use_amp_scaler:
                    # FP16 path: use GradScaler to prevent underflow
                    scaler.scale(loss).backward()
                    if is_update_step:
                        if epoch == 0 and (step < 3 or (step < 10 and step % 5 == 0)):
                            grad_none = 0
                            grad_finite = 0
                            grad_inf = 0
                            grad_nan = 0
                            for p in model.parameters():
                                if p.grad is None:
                                    grad_none += 1
                                elif not torch.isfinite(p.grad).all():
                                    if torch.isinf(p.grad).any():
                                        grad_inf += 1
                                    if torch.isnan(p.grad).any():
                                        grad_nan += 1
                                else:
                                    grad_finite += 1
                            print(
                                f"[GradDiag] step={step} loss_req_grad={loss.requires_grad} "
                                f"loss_val={loss.item():.4f} scale={scaler.get_scale():.1f} "
                                f"params: none={grad_none} finite={grad_finite} inf={grad_inf} nan={grad_nan}",
                                flush=True,
                            )
                        scaler.unscale_(opt)
                        if max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        has_grad = any(p.grad is not None for p in model.parameters())
                        if has_grad:
                            scaler.step(opt)
                            scaler.update()
                            optimizer_stepped = True
                        opt.zero_grad(set_to_none=True)
                else:
                    # BF16 / FP32 path: no scaler needed
                    loss.backward()
                    if is_update_step:
                        if max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        opt.step()
                        optimizer_stepped = True
                        opt.zero_grad()
                        
                step_count += 1
                running_loss += loss.item() * grad_accum
                running_ce_loss += float(ce_loss)
                running_supcon_loss += float(supcon_loss)
                seen += 1
                
                # 收集评估数据
                if "logits" in step_results and "y_true" in step_results and ((step + 1) % collect_metrics_every == 0):
                    batch_logits = step_results["logits"].detach().float().cpu().numpy()
                    batch_y = step_results["y_true"].detach().cpu().numpy()
                    batch_metrics = compute_metrics_from_logits(batch_y, batch_logits, num_classes, metrics, is_distance=False)
                    
                    for k, v in batch_metrics.items():
                        if k == "confusion_matrix":
                            if train_cm_accum is None:
                                train_cm_accum = v
                            else:
                                train_cm_accum += v
                        elif k in train_metrics_accum:
                            if not math.isnan(v):
                                train_metrics_accum[k] += v
                                train_metrics_counts[k] += 1

                # (已移除拖慢 GPU 速度的 synchronize() 和 nvidia-smi 监控)

            # 学习率调度
            # if warmup and epoch < warmup_epochs: warmup.step()
            # elif cosine: cosine.step()
            if optimizer_stepped:
                if warmup and epoch < warmup_epochs: 
                    warmup.step()
                elif cosine: 
                    cosine.step()
            else:
                # 打印提示，帮助你监控是否出现了严重的梯度溢出导致整个 Epoch 无效
                print(f"[Warning] Epoch {epoch+1}: 优化器步骤已跳过（可能是由于 AMP 梯度溢出）。调度器步骤也已跳过。")

            train_loss_epoch = running_loss / max(seen, 1)
            train_ce_loss_epoch = running_ce_loss / max(seen, 1)
            train_supcon_loss_epoch = running_supcon_loss / max(seen, 1)

            # --- Validation Phase ---
            val_metrics_accum = {m: 0.0 for m in metrics if m != "confusion_matrix"}
            val_metrics_counts = {m: 0 for m in metrics if m != "confusion_matrix"}
            val_cm_accum = None
            val_losses = []
            all_val_logits = []
            all_val_y = []
            train_metrics = {k: (v / max(1, train_metrics_counts[k])) for k, v in train_metrics_accum.items()}
            if train_cm_accum is not None:
                train_metrics["confusion_matrix"] = train_cm_accum
            best_thr_acc = float("nan")
            best_thr = 0.5

            if use_raw_val_eval:
                val_loss = evaluate_loader_raw_loss(model, raw_val_loader, device, strategy='meta')
                raw_val_data = evaluate_loader_raw(model, raw_val_loader, device, num_classes, strategy='meta')
                if raw_val_data["y_true"].size > 0:
                    raw_val_metrics = compute_metrics_from_raw(
                        raw_val_data["y_true"], raw_val_data["y_prob"],
                        num_classes, threshold=None,
                    )
                    val_metrics = dict(raw_val_metrics)
                    if num_classes == 2:
                        full_thr, full_bal_acc_tuned, _ = find_best_threshold_bal_acc(
                            raw_val_data["y_true"], raw_val_data["y_prob"][:, 1],
                        )
                        raw_val_metrics["best_threshold_full"] = full_thr
                        raw_val_metrics["bal_acc_tuned_full"] = full_bal_acc_tuned

                        calib_fraction = float(threshold_cfg.get("calib_fraction", 0.4))
                        calib_fraction = min(max(calib_fraction, 0.2), 0.8)
                        split_seed = int(threshold_cfg.get("split_seed", cv_random_state + fold_idx))
                        calib_idx, eval_idx = split_calib_eval_indices(raw_val_data["y_true"], calib_fraction, split_seed)
                        if calib_idx is not None and eval_idx is not None:
                            thr_cv, _, _ = find_best_threshold_bal_acc(
                                raw_val_data["y_true"][calib_idx],
                                raw_val_data["y_prob"][calib_idx, 1],
                            )
                            y_eval = raw_val_data["y_true"][eval_idx]
                            pred_eval = (raw_val_data["y_prob"][eval_idx, 1] >= thr_cv).astype(np.int64)
                            _, _, bal_acc_cv = compute_confusion_and_metrics(list(y_eval), list(pred_eval), num_classes=2)
                            raw_val_metrics["best_threshold"] = float(thr_cv)
                            raw_val_metrics["bal_acc_tuned"] = float(bal_acc_cv)
                            raw_val_metrics["threshold_source"] = "calib_eval"
                            val_metrics["threshold_source"] = "calib_eval"
                            val_metrics["best_threshold"] = float(thr_cv)
                            val_metrics["bal_acc_tuned"] = float(bal_acc_cv)
                        else:
                            raw_val_metrics["best_threshold"] = float(full_thr)
                            raw_val_metrics["bal_acc_tuned"] = float(full_bal_acc_tuned)
                            raw_val_metrics["threshold_source"] = "full_fallback"
                            val_metrics["threshold_source"] = "full_fallback"
                            val_metrics["best_threshold"] = float(full_thr)
                            val_metrics["bal_acc_tuned"] = float(full_bal_acc_tuned)
                    else:
                        val_metrics = dict(raw_val_metrics)
                else:
                    raw_val_metrics = {"cm": torch.zeros((max(2, num_classes), max(2, num_classes)), dtype=torch.int64),
                                       "acc": 0.0, "bal_acc": 0.0, "auc": float("nan"),
                                       "sens": float("nan"), "spec": float("nan"), "best_threshold": 0.5}
                    val_metrics = dict(raw_val_metrics)
            else:
                # Episodic validation path
                model.eval()
                with torch.no_grad():
                    for batch in loader_val:
                        x, y = batch
                        x = x.to(device, dtype=torch.float32, non_blocking=True)
                        y = y.to(device, non_blocking=True)
                        try:
                            sx, sy, qx, qy = split_task_data(x, y, n_way, k_shot, val_q_query)
                            amp_ctx = torch.amp.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else contextlib.nullcontext()
                            with amp_ctx:
                                step_results = strategy.core_step(sx, sy, qx, qy, is_training=False)

                            val_losses.append(step_results['loss'].item())
                            if "logits" in step_results and "y_true" in step_results:
                                batch_logits = step_results["logits"].detach().float().cpu().numpy()
                                batch_y = step_results["y_true"].detach().cpu().numpy()

                                all_val_logits.append(batch_logits)
                                all_val_y.append(batch_y)

                                batch_metrics = compute_metrics_from_logits(batch_y, batch_logits, num_classes, metrics, is_distance=False)
                                for k, v in batch_metrics.items():
                                    if k == "confusion_matrix":
                                        if val_cm_accum is None:
                                            val_cm_accum = v
                                        else:
                                            val_cm_accum += v
                                    elif k in val_metrics_accum:
                                        if not math.isnan(v):
                                            val_metrics_accum[k] += v
                                            val_metrics_counts[k] += 1
                        except ValueError:
                            continue

                val_loss = float(np.mean(val_losses)) if val_losses else 0.0

                # 【优化】Threshold Tuning (仅针对二分类)
                best_thr_acc = float("nan")
                best_thr = 0.5
                if num_classes == 2 and all_val_logits and all_val_y:
                    val_logits_concat = np.concatenate(all_val_logits, axis=0)
                    val_y_concat = np.concatenate(all_val_y, axis=0)
                    val_probs = torch.softmax(torch.tensor(val_logits_concat), dim=1).numpy()
                    val_pos_probs = val_probs[:, 1]

                    best_thr, best_thr_bal_acc, _ = find_best_threshold_bal_acc(val_y_concat, val_pos_probs)
                    best_thr_acc = best_thr_bal_acc

                train_metrics = {k: (v / max(1, train_metrics_counts[k])) for k, v in train_metrics_accum.items()}
                if train_cm_accum is not None:
                    train_metrics["confusion_matrix"] = train_cm_accum

                val_metrics = {k: (v / max(1, val_metrics_counts[k])) for k, v in val_metrics_accum.items()}
                if val_cm_accum is not None:
                    val_metrics["confusion_matrix"] = val_cm_accum

                if not math.isnan(best_thr_acc):
                    val_metrics["bal_acc_tuned"] = best_thr_acc
                    val_metrics["best_threshold"] = best_thr

                raw_val_data = evaluate_loader_raw(model, raw_val_loader, device, num_classes, strategy='meta')
                if raw_val_data["y_true"].size > 0:
                    raw_val_metrics = compute_metrics_from_raw(
                        raw_val_data["y_true"], raw_val_data["y_prob"],
                        num_classes, threshold=None,
                    )
                    if num_classes == 2:
                        full_thr, full_bal_acc_tuned, _ = find_best_threshold_bal_acc(
                            raw_val_data["y_true"], raw_val_data["y_prob"][:, 1],
                        )
                        raw_val_metrics["best_threshold_full"] = full_thr
                        raw_val_metrics["bal_acc_tuned_full"] = full_bal_acc_tuned

                        calib_fraction = float(threshold_cfg.get("calib_fraction", 0.4))
                        calib_fraction = min(max(calib_fraction, 0.2), 0.8)
                        split_seed = int(threshold_cfg.get("split_seed", cv_random_state + fold_idx))
                        calib_idx, eval_idx = split_calib_eval_indices(raw_val_data["y_true"], calib_fraction, split_seed)
                        if calib_idx is not None and eval_idx is not None:
                            thr_cv, _, _ = find_best_threshold_bal_acc(
                                raw_val_data["y_true"][calib_idx],
                                raw_val_data["y_prob"][calib_idx, 1],
                            )
                            y_eval = raw_val_data["y_true"][eval_idx]
                            pred_eval = (raw_val_data["y_prob"][eval_idx, 1] >= thr_cv).astype(np.int64)
                            _, _, bal_acc_cv = compute_confusion_and_metrics(list(y_eval), list(pred_eval), num_classes=2)
                            raw_val_metrics["best_threshold"] = float(thr_cv)
                            raw_val_metrics["bal_acc_tuned"] = float(bal_acc_cv)
                            raw_val_metrics["threshold_source"] = "calib_eval"
                        else:
                            raw_val_metrics["best_threshold"] = float(full_thr)
                            raw_val_metrics["bal_acc_tuned"] = float(full_bal_acc_tuned)
                            raw_val_metrics["threshold_source"] = "full_fallback"
                else:
                    raw_val_metrics = {"cm": torch.zeros((max(2, num_classes), max(2, num_classes)), dtype=torch.int64),
                                       "acc": 0.0, "bal_acc": 0.0, "auc": float("nan"),
                                       "sens": float("nan"), "spec": float("nan"), "best_threshold": 0.5}

                train_metrics = {k: (v / max(1, train_metrics_counts[k])) for k, v in train_metrics_accum.items()}
                if train_cm_accum is not None:
                    train_metrics["confusion_matrix"] = train_cm_accum

                val_metrics = {k: (v / max(1, val_metrics_counts[k])) for k, v in val_metrics_accum.items()}
                if val_cm_accum is not None:
                    val_metrics["confusion_matrix"] = val_cm_accum

                if not math.isnan(best_thr_acc):
                    val_metrics["bal_acc_tuned"] = best_thr_acc
                    val_metrics["best_threshold"] = best_thr

            hist["train_loss"].append(train_loss_epoch)
            hist["val_loss"].append(val_loss)
            hist["train_acc"].append(train_metrics.get("acc", float("nan")))
            hist["val_acc"].append(val_metrics.get("acc", float("nan")))
            hist["train_auc"].append(train_metrics.get("auc", float("nan")))
            hist["val_auc"].append(val_metrics.get("auc", float("nan")))

            val_auc = val_metrics.get("auc", float("nan"))
            if early_monitor == "val_loss":
                monitor_value = val_loss
            elif early_monitor == "val_auc":
                monitor_value = val_auc
            elif early_monitor == "val_acc":
                monitor_value = val_metrics.get("acc", float("nan"))
            elif early_monitor in ("val_bal_acc_tuned", "bal_acc_tuned"):
                bal_acc_tuned = val_metrics.get("bal_acc_tuned", float("nan"))
                auc_tiebreak = val_metrics.get("auc", float("nan"))
                monitor_value = bal_acc_tuned if math.isnan(auc_tiebreak) else bal_acc_tuned + 1e-3 * auc_tiebreak
            elif early_monitor == "val_raw_auc":
                monitor_value = raw_val_metrics.get("auc", float("nan"))
            elif early_monitor == "val_raw_acc":
                monitor_value = raw_val_metrics.get("acc", float("nan"))
            elif early_monitor in ("val_raw_bal_acc_tuned", "raw_bal_acc_tuned"):
                raw_bal_acc_tuned = raw_val_metrics.get("bal_acc_tuned", float("nan"))
                raw_auc_tiebreak = raw_val_metrics.get("auc", float("nan"))
                monitor_value = raw_bal_acc_tuned if math.isnan(raw_auc_tiebreak) else raw_bal_acc_tuned + 1e-3 * raw_auc_tiebreak
            else:
                monitor_value = val_metrics.get("acc", float("nan"))
            
            improved = False
            if best_score is None:
                improved = True
            else:
                if early_mode == "max":
                    improved = monitor_value > best_score + early_min_delta
                else:
                    improved = monitor_value < best_score - early_min_delta
            if improved:
                sens_candidate = raw_val_metrics.get("sens", float("nan"))
                spec_candidate = raw_val_metrics.get("spec", float("nan"))
                gate_block = False
                if min_lbd_sens >= 0.0 and (math.isnan(sens_candidate) or sens_candidate < min_lbd_sens):
                    gate_block = True
                    tag = "sens"
                    gate_val = sens_candidate
                    gate_min = min_lbd_sens
                if min_lbd_spec >= 0.0 and (math.isnan(spec_candidate) or spec_candidate < min_lbd_spec):
                    gate_block = True
                    tag = "spec"
                    gate_val = spec_candidate
                    gate_min = min_lbd_spec
                if gate_block:
                    if early_gate_blocking:
                        print(
                            f"[EarlyGate][fold {fold_idx+1}] skip best-update: raw_val_{tag}={gate_val:.4f} < min_lbd_{tag}={gate_min:.4f}",
                            flush=True,
                        )
                        improved = False
                    else:
                        print(
                            f"[EarlyGate][fold {fold_idx+1}] warn only: raw_val_{tag}={gate_val:.4f} < min_lbd_{tag}={gate_min:.4f}",
                            flush=True,
                        )

            if improved:
                best_score = monitor_value
                best_epoch = epoch
                if "confusion_matrix" in val_metrics:
                    best_cm = val_metrics["confusion_matrix"]
                best_metrics = val_metrics
                best_raw_metrics = raw_val_metrics
                ckpt_path = os.path.join(cv_log_dir, f"best_checkpoint_fold{fold_idx+1}.pth")
                best_ckpt_path = ckpt_path
                tmp_ckpt_path = ckpt_path + ".tmp"
                try:
                    torch.save({
                        "model": model.state_dict(),
                        "epoch": epoch + 1,
                        "monitor": monitor_value,
                        "monitor_name": early_monitor,
                        "threshold": raw_val_metrics.get("best_threshold", val_metrics.get("best_threshold", 0.5)),
                        "threshold_name": raw_val_metrics.get("threshold_source", val_metrics.get("threshold_source", "val_checkpoint")),
                    }, tmp_ckpt_path)
                    os.replace(tmp_ckpt_path, ckpt_path)
                except Exception as e:
                    print(f"[Warning] Failed to save checkpoint at epoch {epoch+1}: {e}")
                    if os.path.exists(tmp_ckpt_path):
                        try:
                            os.remove(tmp_ckpt_path)
                        except:
                            pass

            if plateau:
                plateau.step(monitor_value)

            if writer:
                try:
                    writer.add_scalar(f"fold{fold_idx+1}/train_loss", train_loss_epoch, epoch + 1)
                    writer.add_scalar(f"fold{fold_idx+1}/val_loss", val_loss, epoch + 1)
                    if hasattr(strategy, 'inner_lr'):
                        if hasattr(strategy, "get_inner_lr"):
                            current_inner_lr = float(strategy.get_inner_lr().detach().item())
                        else:
                            current_inner_lr = float(strategy.inner_lr.detach().item())
                        writer.add_scalar(f"fold{fold_idx+1}/meta_inner_lr", current_inner_lr, epoch + 1)

                    for k, v in train_metrics.items():
                        if k != "confusion_matrix":
                            writer.add_scalar(f"fold{fold_idx+1}/train_{k}", v, epoch + 1)
                    for k, v in val_metrics.items():
                        if k != "confusion_matrix":
                            writer.add_scalar(f"fold{fold_idx+1}/val_{k}", v, epoch + 1)
                    for k, v in raw_val_metrics.items():
                        if k != "cm":
                            writer.add_scalar(f"fold{fold_idx+1}/raw_val_{k}", v, epoch + 1)
                except Exception as e:
                    print(f"[tensorboard][warn] write failed: {e}", flush=True)

            try:
                row = [
                    epoch + 1,
                    train_loss_epoch,
                    val_loss,
                    raw_val_metrics.get("acc", float("nan")),
                    raw_val_metrics.get("auc", float("nan")),
                    raw_val_metrics.get("bal_acc", float("nan")),
                    raw_val_metrics.get("bal_acc_tuned", float("nan")),
                    raw_val_metrics.get("sens", float("nan")),
                    raw_val_metrics.get("spec", float("nan")),
                    raw_val_metrics.get("best_threshold", float("nan"))
                ]
                for m in metrics:
                    if m == "confusion_matrix":
                        continue
                    row.append(train_metrics.get(m, float("nan")))
                    row.append(val_metrics.get(m, float("nan")))
                with open(log_csv_path, "a", newline="", encoding="utf-8") as f:
                    writer_csv = csv.writer(f)
                    writer_csv.writerow(row)
                    f.flush()
            except Exception as e:
                print(f"[log][fold {fold_idx+1}] csv write failed: {e}", flush=True)

            ep_time = (time.time() - epoch_start) / 60.0
            raw_val_auc_str = f"{raw_val_metrics.get('auc', float('nan')):.3f}" if raw_val_metrics else "nan"
            raw_val_bal_acc_tuned_str = f"{raw_val_metrics.get('bal_acc_tuned', float('nan')):.3f}" if raw_val_metrics and not math.isnan(raw_val_metrics.get('bal_acc_tuned', float('nan'))) else "nan"
            raw_val_sens_str = f"{raw_val_metrics.get('sens', float('nan')):.3f}" if raw_val_metrics and not math.isnan(raw_val_metrics.get('sens', float('nan'))) else "nan"
            raw_val_spec_str = f"{raw_val_metrics.get('spec', float('nan')):.3f}" if raw_val_metrics and not math.isnan(raw_val_metrics.get('spec', float('nan'))) else "nan"
            raw_val_thr_src = raw_val_metrics.get('threshold_source', 'na') if raw_val_metrics else 'na'
            print(
                f"[CV {fold_idx+1}] Epoch {epoch+1}/{epochs} "
                f"train_loss={train_loss_epoch:.4f} val_loss={val_loss:.4f} "
                f"raw_val_auc={raw_val_auc_str} raw_val_bal_acc_tuned={raw_val_bal_acc_tuned_str} "
                f"raw_val_sens={raw_val_sens_str} raw_val_spec={raw_val_spec_str} "
                f"thr_src={raw_val_thr_src} "
                f"time={ep_time:.1f} min",
                flush=True
            )

            if early_stop.step(monitor_value) and epoch >= early_min_epochs:
                print(f"[CV {fold_idx+1}] Early stopping at epoch {epoch+1}", flush=True)
                break

        plot_curves(cv_log_dir, fold_idx, hist)
        if best_cm is not None:
            cm_tensor = torch.tensor(best_cm) if not isinstance(best_cm, torch.Tensor) else best_cm
            save_cm_png(cm_tensor, class_names, os.path.join(cv_log_dir, f"cm_fold_{fold_idx+1}.png"))
        if best_metrics is not None:
            best_items = " ".join([f"{k}={best_metrics.get(k):.4f}" for k in best_metrics.keys() if k != "confusion_matrix"])
            raw_best_items = ""
            if best_raw_metrics is not None:
                raw_parts = []
                for k, v in best_raw_metrics.items():
                    if k == "cm":
                        continue
                    if isinstance(v, (int, float, np.integer, np.floating)):
                        raw_parts.append(f"raw_{k}={float(v):.4f}")
                    else:
                        raw_parts.append(f"raw_{k}={v}")
                raw_best_items = " " + " ".join(raw_parts)
            print(f"[CV {fold_idx+1}] best_epoch={best_epoch+1} {best_items}{raw_best_items}", flush=True)

            test_metrics = None
            if raw_test_loader is not None and best_ckpt_path and os.path.exists(best_ckpt_path):
                try:
                    best_state = torch.load(best_ckpt_path, map_location=device)
                    model.load_state_dict(best_state.get("model", best_state), strict=False)
                    raw_test_data = evaluate_loader_raw(model, raw_test_loader, device, num_classes, strategy='meta')
                    if raw_test_data["y_true"].size > 0:
                        test_threshold = best_state.get("threshold", best_raw_metrics.get("best_threshold", 0.5) if best_raw_metrics else 0.5)
                        test_metrics = compute_metrics_from_raw(
                            raw_test_data["y_true"],
                            raw_test_data["y_prob"],
                            num_classes,
                            threshold=test_threshold if num_classes == 2 else None,
                        )
                        test_metrics["best_threshold"] = float(test_threshold)
                        test_metrics["threshold_source"] = best_state.get("threshold_name", "val_checkpoint")
                        test_items = " ".join(
                            [
                                f"test_{k}={float(v):.4f}" if isinstance(v, (int, float, np.integer, np.floating)) else f"test_{k}={v}"
                                for k, v in test_metrics.items()
                                if k != "cm"
                            ]
                        )
                        print(f"[TEST {fold_idx+1}] threshold_source=val_checkpoint {test_items}", flush=True)
                except Exception as e:
                    print(f"[TEST][warn][fold {fold_idx+1}] evaluation failed: {e}", flush=True)

            fold_results.append({
                "fold": fold_idx + 1,
                "best_epoch": int(best_epoch + 1),
                "metrics": {k: float(best_metrics.get(k)) for k in best_metrics.keys() if k != "confusion_matrix"},
                "raw_metrics": ({
                    k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                    for k, v in best_raw_metrics.items()
                    if k != "cm"
                } if best_raw_metrics is not None else {}),
                "test_metrics": ({
                    k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                    for k, v in test_metrics.items()
                    if k != "cm"
                } if test_metrics is not None else {})
            })

    # ===== Phase 3: nested_cv — 全量dev重训 + 锁定测试集评估 =====
    final_test_metrics = None
    if split_mode == "nested_cv" and locked_test_idx is not None:
        print("\n" + "=" * 60, flush=True)
        print("[Phase 3] 全量dev重训 + 锁定测试集评估", flush=True)
        print("=" * 60, flush=True)

        dev_labels_full = [labels_all[i] for i in dev_idx]
        final_eval_dir = os.path.join(cv_log_dir, "final_evaluation")
        os.makedirs(final_eval_dir, exist_ok=True)

        inner_splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=cv_random_state)
        train_local_idx, val_local_idx = next(inner_splitter.split(np.zeros(len(dev_labels_full)), dev_labels_full))
        ds_train_final = torch.utils.data.Subset(ds, [dev_idx[i] for i in train_local_idx])
        ds_val_final = torch.utils.data.Subset(ds, [dev_idx[i] for i in val_local_idx])
        ds_test_final = torch.utils.data.Subset(ds, locked_test_idx)

        labels_train_final = [labels_all[dev_idx[i]] for i in train_local_idx]
        labels_val_final = [labels_all[dev_idx[i]] for i in val_local_idx]
        print(f"[Phase 3] train={len(ds_train_final)} val={len(ds_val_final)} test={len(ds_test_final)}", flush=True)

        labels_t_final = torch.tensor(labels_train_final, dtype=torch.int64)
        class_counts_final = torch.bincount(labels_t_final, minlength=num_classes).tolist()
        class_weights_final = compute_class_weights_from_counts(class_counts_final, mode=class_weight_mode, beta=class_weight_beta)
        ce_weight_final = torch.tensor(class_weights_final, dtype=torch.float32, device=device) if loss_type in ('ce', 'focal') else None
        class_sampling_weights_final = {int(ci): float(class_weights_final[ci]) for ci in range(len(class_weights_final))}

        train_class_counts_final = torch.bincount(torch.tensor(labels_train_final, dtype=torch.int64)).tolist()
        minority_class_count_final = min(train_class_counts_final)
        minority_samples_per_episode_final = max(1, k_shot + train_q_query)
        safe_episodes_limit_final = max(1, int((minority_class_count_final * max_minority_reuse) // minority_samples_per_episode_final))
        train_episodes_final = min(safe_episodes_limit_final, max_train_episodes) if max_train_episodes > 0 else safe_episodes_limit_final
        if min_epoch_coverage_ratio > 0.0:
            episodes_for_target_final = int(math.ceil(
                (min_epoch_coverage_ratio * max(1, len(labels_train_final))) / max(1, n_way * (k_shot + train_q_query))
            ))
            train_episodes_final = min(max(1, episodes_for_target_final), max(1, train_episodes_final))
            if max_train_episodes > 0:
                train_episodes_final = min(train_episodes_final, max_train_episodes)
        print(f"[Phase 3][Meta-Sampler] train_episodes={train_episodes_final}", flush=True)

        if is_vanilla_mode:
            from torch.utils.data import BatchSampler, RandomSampler
            n_per_batch_final = n_way * (k_shot + train_q_query)
            train_sampler_final = BatchSampler(
                RandomSampler(ds_train_final, replacement=True, num_samples=int(train_episodes_final * n_per_batch_final)),
                batch_size=n_per_batch_final, drop_last=False,
            )
        else:
            train_sampler_final = EpisodicBatchSampler(
                labels_train_final, n_way, k_shot, train_q_query,
                n_episodes=train_episodes_final,
                class_sampling_weights=(class_sampling_weights_final if sampler_enabled_fold else None),
            )

        dl_kwargs_final = {
            "batch_sampler": train_sampler_final,
            "num_workers": dl_num_workers, "pin_memory": dl_pin_memory,
            "collate_fn": collate_fn,
        }
        if dl_num_workers > 0:
            dl_kwargs_final["persistent_workers"] = True
            dl_kwargs_final["prefetch_factor"] = prefetch_factor
        loader_train_final = DataLoader(ds_train_final, **dl_kwargs_final)

        val_class_counts_final = torch.bincount(torch.tensor(labels_val_final, dtype=torch.int64)).tolist()
        val_min_count_final = min(val_class_counts_final)
        val_q_query_final = max(1, min(val_min_count_final - k_shot, val_q_query))
        val_sampler_final = EpisodicBatchSampler(labels_val_final, n_way, k_shot, val_q_query_final, n_episodes=val_episodes_meta)
        dl_kwargs_final["batch_sampler"] = val_sampler_final
        loader_val_final = DataLoader(ds_val_final, **dl_kwargs_final)
        raw_val_loader_final = DataLoader(ds_val_final, batch_size=raw_val_batch_size, shuffle=False,
                                          num_workers=dl_num_workers, pin_memory=dl_pin_memory, collate_fn=collate_fn)
        raw_test_loader_final = DataLoader(ds_test_final, batch_size=raw_val_batch_size, shuffle=False,
                                           num_workers=dl_num_workers, pin_memory=dl_pin_memory, collate_fn=collate_fn)

        # Build fresh model
        if backbone_choice == "medmamba3d":
            model_final = MedMamba3D(in_chans=inferred_in_chans, embed_dim=cfg['model']['embed_dim'],
                                     depth=cfg['model']['depth'], patch_size=tuple(cfg['model']['patch_size']),
                                     num_classes=num_classes).to(device)
        elif backbone_choice in ("medmamba_ss3m", "ss3m", "medmambass3m"):
            model_final = MedMambaSS3M(
                in_channels=inferred_in_chans, embed_dim=ss3m_embed_dim, depth=ss3m_depth,
                patch_size=ss3m_patch_size, num_classes=num_classes, n_dirs_train=backbone_n_dirs,
                use_dual_branch=backbone_use_dual_branch, branch_types=ss3m_branch_types,
                use_checkpoint=ss3m_use_checkpoint, dropout=ss3m_dropout,
                use_pos_emb=ss3m_use_pos_emb, merge_type=ss3m_merge_type,
            ).to(device)
        elif backbone_choice in ("medmamba_ss3m_2dscan", "ss3m_2dscan", "ss3m2d", "medmambass3m2dscan"):
            model_final = MedMambaSS3M2DScan(
                in_channels=inferred_in_chans, embed_dim=cfg['model']['embed_dim'],
                depth=cfg['model']['depth'], patch_size=tuple(cfg['model']['patch_size']),
                num_classes=num_classes, n_dirs_train=min(backbone_n_dirs, 4),
                use_dual_branch=backbone_use_dual_branch,
            ).to(device)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_choice}")

        resolved_ckpt_final, resolved_state_final = _resolve_pretrained_checkpoint(model_final, cfg)
        if resolved_ckpt_final and resolved_state_final is not None:
            _best_effort_load_pretrained(model_final, resolved_state_final)
        model_final = apply_backbone_and_head_optimizations(model_final, cfg, num_classes).to(device)

        # Build strategy
        if meta_cfg.get('hybrid_strategy', {}).get('enable') and not no_meta_requested:
            from strategies.hybrid import HybridStrategy
            strategy_final = HybridStrategy(model_final, hybrid_cfg)
        elif meta_cfg.get('optimizer_strategy', {}).get('enable') or no_meta_requested:
            opt_cfg_final = dict(meta_cfg.get('optimizer_strategy', {}))
            if no_meta_requested:
                opt_cfg_final.setdefault('name', 'none')
            opt_cfg_final.setdefault('params', {})['label_smoothing'] = label_smoothing
            opt_params_final = opt_cfg_final.get('params', {})
            if imbalance_mode == 'sampler':
                opt_params_final['use_class_weights'] = False
                opt_params_final['inner_use_class_weights'] = False
                opt_params_final['focal_weight'] = 0.0
            elif imbalance_mode in ('class_weight', 'class_weight_focal'):
                opt_params_final['use_class_weights'] = configured_use_class_weights
                opt_params_final['inner_use_class_weights'] = configured_use_class_weights
            strat_name_final = opt_cfg_final.get('name', 'maml').lower()
            if strat_name_final in ("none", "no_meta", "no-meta", "vanilla", "fine_tuning", "finetune", "baseline"):
                from strategies.vanilla import VanillaStrategy
                strategy_final = VanillaStrategy(model_final, opt_cfg_final)
            elif strat_name_final == 'anil':
                from strategies.anil import ANILStrategy
                strategy_final = ANILStrategy(model_final, opt_cfg_final)
            else:
                from strategies.maml import MAMLStrategy
                strategy_final = MAMLStrategy(model_final, opt_cfg_final)
        elif meta_cfg.get('metric_strategy', {}).get('enable'):
            from strategies.protonet import ProtoNetStrategy
            strategy_final = ProtoNetStrategy(model_final, meta_cfg['metric_strategy'])
        else:
            from strategies.vanilla import VanillaStrategy
            strategy_final = VanillaStrategy(model_final, meta_cfg.get('optimizer_strategy', {}))
        strategy_final.to(device)
        if hasattr(strategy_final, "set_class_weights") and ce_weight_final is not None:
            strategy_final.set_class_weights(ce_weight_final)
        elif hasattr(strategy_final, "set_class_weights"):
            strategy_final.set_class_weights(None)

        # Build optimizer
        head_params_final, backbone_params_final = [], []
        for name, param in model_final.named_parameters():
            param.requires_grad = True
            if name.startswith('proj.') or 'head' in name or 'classifier' in name:
                head_params_final.append(param)
            else:
                backbone_params_final.append(param)
        opt_groups_final = [
            {'params': backbone_params_final, 'lr': base_lr * backbone_ratio},
            {'params': head_params_final, 'lr': base_lr},
        ]
        if hasattr(strategy_final, 'inner_lr') and isinstance(strategy_final.inner_lr, torch.nn.Parameter):
            opt_groups_final.append({'params': [strategy_final.inner_lr], 'lr': base_lr})
        opt_final = torch.optim.AdamW(opt_groups_final, weight_decay=cfg['classifier']['weight_decay'])
        scaler_final = torch.amp.GradScaler('cuda') if use_amp_scaler else None
        cosine_final = CosineAnnealingLR(opt_final, T_max=max(1, epochs - warmup_epochs)) if scheduler_type == "cosine" else None
        plateau_final = ReduceLROnPlateau(opt_final, mode="max" if early_mode == "max" else "min",
                                          factor=lr_factor, patience=lr_patience) if scheduler_type == "plateau" else None
        warmup_final = LinearLR(opt_final, start_factor=0.1, total_iters=warmup_epochs) if warmup_epochs > 0 else None

        # Training loop (Phase 3)
        hist_final = {"train_loss": [], "val_loss": [], "train_acc": [], "train_auc": [], "val_acc": [], "val_auc": []}
        best_score_final, best_epoch_final, best_metrics_final, best_raw_metrics_final = None, -1, None, None
        best_ckpt_path_final, test_metrics_final = None, None
        early_stop_final = EarlyStopping(patience=early_patience, min_delta=early_min_delta, mode=early_mode)

        for epoch in range(epochs):
            model_final.train()
            if freeze_bn:
                set_bn_eval(model_final)
            epoch_start_f = time.time()
            running_loss_f, seen_f, step_count_f = 0.0, 0, 0
            optimizer_stepped_f = False

            for step, batch in enumerate(loader_train_final):
                x, y = batch
                if y.numel() == 0:
                    continue
                x = x.to(device, dtype=torch.float32, non_blocking=True)
                y = y.to(device, non_blocking=True)
                if do_augment:
                    x = augment_3d_batch(x, augment_params)
                if is_vanilla_mode:
                    n_per_batch_f = n_way * (k_shot + train_q_query)
                    if x.size(0) != n_per_batch_f:
                        x = x[:(x.size(0) // n_per_batch_f) * n_per_batch_f]
                        y = y[:(y.size(0) // n_per_batch_f) * n_per_batch_f]
                        if x.size(0) == 0:
                            continue
                try:
                    sx, sy, qx, qy = split_task_data(x, y, n_way, k_shot, train_q_query)
                except ValueError:
                    continue

                amp_ctx = torch.amp.autocast(device_type='cuda', dtype=amp_dtype) if use_amp else contextlib.nullcontext()
                with amp_ctx:
                    step_results = strategy_final.core_step(sx, sy, qx, qy, optimizer=opt_final, is_training=True)
                    loss = step_results['loss']
                if not torch.isfinite(loss).all():
                    opt_final.zero_grad(set_to_none=True)
                    continue
                loss = loss / grad_accum
                is_update_step = ((step + 1) % grad_accum == 0) or (step == len(loader_train_final) - 1)
                if use_amp_scaler:
                    scaler_final.scale(loss).backward()
                    if is_update_step:
                        scaler_final.unscale_(opt_final)
                        if max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model_final.parameters(), max_grad_norm)
                        if any(p.grad is not None for p in model_final.parameters()):
                            scaler_final.step(opt_final)
                            scaler_final.update()
                            optimizer_stepped_f = True
                        opt_final.zero_grad(set_to_none=True)
                else:
                    loss.backward()
                    if is_update_step:
                        if max_grad_norm > 0:
                            torch.nn.utils.clip_grad_norm_(model_final.parameters(), max_grad_norm)
                        opt_final.step()
                        optimizer_stepped_f = True
                        opt_final.zero_grad()
                step_count_f += 1
                running_loss_f += loss.item() * grad_accum
                seen_f += 1

            if optimizer_stepped_f:
                if warmup_final and epoch < warmup_epochs:
                    warmup_final.step()
                elif cosine_final:
                    cosine_final.step()

            train_loss_f = running_loss_f / max(seen_f, 1)

            # Validation
            raw_val_data_f = evaluate_loader_raw(model_final, raw_val_loader_final, device, num_classes, strategy='meta')
            raw_val_metrics_f = {"cm": torch.zeros((max(2, num_classes), max(2, num_classes)), dtype=torch.int64),
                                 "acc": 0.0, "bal_acc": 0.0, "auc": float("nan"),
                                 "sens": float("nan"), "spec": float("nan"), "best_threshold": 0.5}
            if raw_val_data_f["y_true"].size > 0:
                raw_val_metrics_f = compute_metrics_from_raw(raw_val_data_f["y_true"], raw_val_data_f["y_prob"], num_classes, threshold=None)
                if num_classes == 2:
                    full_thr_f, full_bal_acc_f, _ = find_best_threshold_bal_acc(raw_val_data_f["y_true"], raw_val_data_f["y_prob"][:, 1])
                    raw_val_metrics_f["best_threshold"] = float(full_thr_f)
                    raw_val_metrics_f["bal_acc_tuned"] = float(full_bal_acc_f)

            hist_final["train_loss"].append(train_loss_f)
            hist_final["val_loss"].append(0.0)
            hist_final["train_acc"].append(float("nan"))
            hist_final["train_auc"].append(float("nan"))
            hist_final["val_acc"].append(raw_val_metrics_f.get("acc", float("nan")))
            hist_final["val_auc"].append(raw_val_metrics_f.get("auc", float("nan")))

            monitor_value_f = raw_val_metrics_f.get("auc", float("nan"))
            if early_monitor == "val_raw_auc":
                monitor_value_f = raw_val_metrics_f.get("auc", float("nan"))
            elif early_monitor == "val_raw_acc":
                monitor_value_f = raw_val_metrics_f.get("acc", float("nan"))
            elif early_monitor in ("val_raw_bal_acc_tuned", "raw_bal_acc_tuned"):
                monitor_value_f = raw_val_metrics_f.get("bal_acc_tuned", float("nan"))

            improved_f = False
            if best_score_final is None:
                improved_f = True
            else:
                if early_mode == "max":
                    improved_f = monitor_value_f > best_score_final + early_min_delta
                else:
                    improved_f = monitor_value_f < best_score_final - early_min_delta
            if improved_f:
                best_score_final = monitor_value_f
                best_epoch_final = epoch
                best_metrics_final = raw_val_metrics_f
                best_raw_metrics_final = raw_val_metrics_f
                ckpt_path_f = os.path.join(final_eval_dir, "best_checkpoint_final.pth")
                best_ckpt_path_final = ckpt_path_f
                try:
                    torch.save({"model": model_final.state_dict(), "epoch": epoch + 1,
                                "monitor": monitor_value_f, "monitor_name": early_monitor,
                                "threshold": raw_val_metrics_f.get("best_threshold", 0.5)}, ckpt_path_f)
                except Exception as e:
                    print(f"[Phase 3][warn] checkpoint save failed: {e}", flush=True)
            if plateau_final:
                plateau_final.step(monitor_value_f)
            ep_time_f = (time.time() - epoch_start_f) / 60.0
            print(
                f"[Phase 3] Epoch {epoch+1}/{epochs} "
                f"train_loss={train_loss_f:.4f} "
                f"raw_val_auc={raw_val_metrics_f.get('auc', float('nan')):.3f} "
                f"raw_val_bal_acc={raw_val_metrics_f.get('bal_acc_tuned', float('nan')):.3f} "
                f"time={ep_time_f:.1f}min",
                flush=True,
            )
            if early_stop_final.step(monitor_value_f) and epoch >= early_min_epochs:
                print(f"[Phase 3] Early stopping at epoch {epoch+1}", flush=True)
                break

        # Evaluate on locked test set
        if best_ckpt_path_final and os.path.exists(best_ckpt_path_final):
            best_state_f = torch.load(best_ckpt_path_final, map_location=device)
            model_final.load_state_dict(best_state_f.get("model", best_state_f), strict=False)
            raw_test_data_f = evaluate_loader_raw(model_final, raw_test_loader_final, device, num_classes, strategy='meta')
            if raw_test_data_f["y_true"].size > 0:
                test_thr_f = best_state_f.get("threshold", 0.5)
                test_metrics_final = compute_metrics_from_raw(raw_test_data_f["y_true"], raw_test_data_f["y_prob"], num_classes, threshold=test_thr_f if num_classes == 2 else None)
                test_metrics_final["best_threshold"] = float(test_thr_f)
                test_items_f = " ".join([
                    f"test_{k}={float(v):.4f}" if isinstance(v, (int, float, np.integer, np.floating)) else f"test_{k}={v}"
                    for k, v in test_metrics_final.items() if k != "cm"
                ])
                print(f"[Phase 3][TEST] {test_items_f}", flush=True)
                if best_cm is not None:
                    cm_tensor_f = torch.tensor(test_metrics_final.get("cm", best_cm)) if "cm" in test_metrics_final else None
                    if cm_tensor_f is not None:
                        save_cm_png(cm_tensor_f, class_names, os.path.join(final_eval_dir, "cm_final.png"))
            final_test_metrics = test_metrics_final
            with open(os.path.join(final_eval_dir, "final_test_report.json"), "w", encoding="utf-8") as f:
                json.dump({k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                           for k, v in (test_metrics_final or {}).items() if k != "cm"}, f, ensure_ascii=False, indent=2)

        # Cleanup
        del model_final, strategy_final, opt_final
        if device_type == 'cuda':
            torch.cuda.empty_cache()

    if writer:
        writer.close()
    if fold_results:
        summary = {}
        for m in ["acc", "f1_macro", "auc", "f1_weighted", "precision_macro", "recall_macro", "bal_acc_tuned"]:
            vals = [fr["metrics"].get(m, float("nan")) for fr in fold_results]
            if all(math.isnan(v) for v in vals):
                continue
            summary[m] = {
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals))
            }
        for m in ["acc", "auc", "bal_acc", "bal_acc_tuned", "sens", "spec"]:
            vals = [fr.get("raw_metrics", {}).get(m, float("nan")) for fr in fold_results]
            if all(math.isnan(v) for v in vals):
                continue
            summary[f"raw_{m}"] = {
                "mean": float(np.nanmean(vals)),
                "std": float(np.nanstd(vals))
            }
        results = {
            "phase2_inner_cv": {"folds": fold_results, "summary": summary},
            "phase3_final_test": {k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v)
                                  for k, v in (final_test_metrics or {}).items() if k != "cm"},
            "split_info": {"n_dev": len(dev_idx), "n_test": len(locked_test_idx)},
        }
        with open(os.path.join(cv_log_dir, "nested_cv_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)

        plot_metrics = [m for m in ["acc", "f1_macro", "auc"] if m in summary]
        if plot_metrics:
            data = [[fr["metrics"].get(m, float("nan")) for fr in fold_results] for m in plot_metrics]
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.boxplot(data, labels=plot_metrics, showmeans=True)
            ax.set_title("CV Metrics Boxplot")
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(cv_log_dir, "cv_metrics_boxplot.png"), dpi=300, bbox_inches="tight")
            plt.close(fig)

        report_parts = []
        if "raw_acc" in summary:
            report_parts.append(f"RawAcc={summary['raw_acc']['mean']:.4f}±{summary['raw_acc']['std']:.4f}")
        if "raw_auc" in summary:
            report_parts.append(f"RawAUC={summary['raw_auc']['mean']:.4f}±{summary['raw_auc']['std']:.4f}")
        if "raw_bal_acc_tuned" in summary:
            report_parts.append(f"RawBalAcc={summary['raw_bal_acc_tuned']['mean']:.4f}±{summary['raw_bal_acc_tuned']['std']:.4f}")
        if "acc" in summary:
            report_parts.append(f"EpisodeAcc={summary['acc']['mean']:.4f}±{summary['acc']['std']:.4f}")
        if "f1_macro" in summary:
            report_parts.append(f"F1={summary['f1_macro']['mean']:.4f}±{summary['f1_macro']['std']:.4f}")
        if "auc" in summary:
            report_parts.append(f"EpisodeAUC={summary['auc']['mean']:.4f}±{summary['auc']['std']:.4f}")
        if "raw_sens" in summary:
            report_parts.append(f"RawSens={summary['raw_sens']['mean']:.4f}±{summary['raw_sens']['std']:.4f}")
        if "raw_spec" in summary:
            report_parts.append(f"RawSpec={summary['raw_spec']['mean']:.4f}±{summary['raw_spec']['std']:.4f}")
        if report_parts:
            print(f"{len(fold_results)}-Fold CV: " + ", ".join(report_parts), flush=True)
        if split_mode == "nested_cv" and final_test_metrics:
            final_parts = []
            for m in ["acc", "auc", "bal_acc_tuned", "sens", "spec"]:
                v = final_test_metrics.get(m, float("nan"))
                if not math.isnan(v):
                    final_parts.append(f"{m}={v:.4f}")
            if final_parts:
                print(f"[Phase 3][Final Test] " + ", ".join(final_parts), flush=True)
    print("Meta-Training Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Meta-learning trainer")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--cv_n_splits", type=int, default=None, help="Override number of CV folds")
    parser.add_argument("--run-folds", "--run_folds", dest="run_folds", type=int, default=None, help="Override number of folds to run")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER, help="Override nested config parameters: cv.cv_random_state=123")
    args = parser.parse_args()
    train_meta(config_path=args.config, cv_n_splits_override=args.cv_n_splits, run_folds_override=args.run_folds)
