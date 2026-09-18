# -*- coding: utf-8 -*-
"""One-shot: profile Params / FLOPs / Throughput / GPU Memory for all backbones.

Usage (on your HPC server):
    python meta/profile_all.py --repeat 100 --table

Outputs a paper-ready Markdown table with all backbones side by side.
"""

import argparse
import os
import sys
import time
from typing import Optional

import torch

# ── add both code roots ──────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
for p in [ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

# ── your SS3M ────────────────────────────────────────────────────────
from medmamba_ss3m import MedMambaSS3M

_COMPARE_ROOT = None


def _init_compare(compare_root: str):
    """Lazy-import 3D CNN / Swin after user provides the compare root."""
    global _COMPARE_ROOT
    if _COMPARE_ROOT is not None:
        return
    _COMPARE_ROOT = compare_root

    cnn_pkg = os.path.join(compare_root, "3D-CNN-PyTorch")
    swin_pkg = os.path.join(compare_root, "3D_Swin_transformer_classification")

    for p in [cnn_pkg, swin_pkg]:
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Directory not found: {p}")
        if p not in sys.path:
            sys.path.insert(0, p)
        # Ensure __init__.py exists in models/ and layers/
        for sub in ["models", "layers"]:
            sub_path = os.path.join(p, sub)
            init_file = os.path.join(sub_path, "__init__.py")
            if os.path.isdir(sub_path) and not os.path.isfile(init_file):
                open(init_file, "w").close()
                print(f"[init] created {init_file}", flush=True)

    print(f"[init] compare_root={compare_root}", flush=True)


_RN = None
_DN = None
_SWIN = None


def _resnet(depth: int) -> torch.nn.Module:
    global _RN
    if _RN is None:
        from models import resnet as rn
        _RN = rn
    return _RN.generate_model(
        depth, n_input_channels=IN_CHANS, n_classes=NUM_CLASSES,
    )


def _densenet() -> torch.nn.Module:
    global _DN
    if _DN is None:
        from models import DenseNet as dn
        _DN = dn
    return _DN.DenseNet(
        n_input_channels=IN_CHANS, num_classes=NUM_CLASSES,
    )


def _swin() -> torch.nn.Module:
    global _SWIN
    if _SWIN is None:
        from layers.swin3d_layer import SwinTransformerForClassification as Sw
        _SWIN = Sw
    return _SWIN(
        img_size=INPUT_SHAPE, in_channels=IN_CHANS, num_classes=NUM_CLASSES,
        out_channels=2,
    )


# ══════════════════════════════════════════════════════════════════════
# flops helpers
# ══════════════════════════════════════════════════════════════════════

try:
    from fvcore.nn import FlopCountAnalysis
    HAS_FVCORE = True
except Exception:
    HAS_FVCORE = False

try:
    from thop import profile as thop_profile
    HAS_THOP = True
except Exception:
    HAS_THOP = False


def compute_flops(model: torch.nn.Module, sample: torch.Tensor):
    if HAS_FVCORE:
        try:
            return float(FlopCountAnalysis(model, sample).total()), "fvcore"
        except Exception:
            pass
    if HAS_THOP:
        try:
            flops, _ = thop_profile(model, inputs=(sample,))
            return float(flops), "thop"
        except Exception:
            pass
    return 0.0, "unavailable"


# ══════════════════════════════════════════════════════════════════════
# throughput + GPU memory
# ══════════════════════════════════════════════════════════════════════

def measure(model: torch.nn.Module, sample: torch.Tensor, warmup: int, repeat: int):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(sample)
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(repeat):
            _ = model(sample)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        mem = torch.cuda.max_memory_allocated() / (1024 ** 3)
    return elapsed, mem, repeat * sample.shape[0]


# ══════════════════════════════════════════════════════════════════════
# build each backbone with a uniform forward
# ══════════════════════════════════════════════════════════════════════

INPUT_SHAPE = (112, 112, 112)
IN_CHANS    = 1
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ss3m(embed_dim: int) -> torch.nn.Module:
    return MedMambaSS3M(
        in_channels=IN_CHANS, embed_dim=embed_dim, depth=6,
        patch_size=(16, 16, 16), num_classes=NUM_CLASSES,
    )


BACKBONE_SPECS = [
    # (label,       build_fn,       embed_dim | None)
    ("SS3M-768",     lambda: _ss3m(768),   768),
    ("SS3M-384",     lambda: _ss3m(384),   384),
    ("SS3M-192",     lambda: _ss3m(192),   192),
    ("SS3M-96",      lambda: _ss3m(96),     96),
    ("3D-ResNet50",  lambda: _resnet(50),   None),
    ("3D-ResNet101", lambda: _resnet(101),  None),
    ("3D-DenseNet",  _densenet,             None),
    ("3D-SwinT",     _swin,                 None),
]


def profile_one(label: str, build, embed_dim, warmup: int, repeat: int):
    model = build().to(DEVICE)
    sample = torch.randn(1, IN_CHANS, *INPUT_SHAPE, device=DEVICE)

    params = sum(p.numel() for p in model.parameters()) / 1e6
    flops_raw, flops_note = compute_flops(model, sample)
    flops = flops_raw / 1e9
    elapsed, mem, total_samples = measure(model, sample, warmup, repeat)
    throughput = total_samples / elapsed
    latency = elapsed * 1000 / repeat

    return {
        "label": label, "params_m": params, "flops_g": flops,
        "flops_note": flops_note, "latency_ms": latency,
        "throughput": throughput, "gpu_gb": mem, "embed_dim": embed_dim,
    }


# ══════════════════════════════════════════════════════════════════════
# main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-root", default="~/compare",
                        help="Path to the 'compare' directory on this machine")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--table", action="store_true")
    parser.add_argument("--skip", nargs="*", default=[])
    args = parser.parse_args()

    _init_compare(args.compare_root)

    results = []
    for label, build, ed in BACKBONE_SPECS:
        if label in args.skip:
            continue
        try:
            print(f"[profile] {label} ...", flush=True)
            r = profile_one(label, build, ed, args.warmup, args.repeat)
            results.append(r)
            print(f"[profile] {label} OK", flush=True)
        except Exception as exc:
            print(f"[SKIP] {label}: {type(exc).__name__}: {exc}", flush=True)

    if args.table:
        header = (
            f"| {'Backbone':<16} | {'Params(M)':>9} | {'FLOPs(G)':>9} "
            f"| {'Lat.(ms)':>8} | {'Thr.(/s)':>9} | {'GPU(GB)':>7} |"
        )
        sep = "|" + "|".join(["-" * (len(c) + 2) for c in header.split("|")[1:-1]]) + "|"
        print("\n" + header)
        print(sep)
        for r in results:
            print(
                f"| {r['label']:<16} | {r['params_m']:>9.2f} | {r['flops_g']:>9.2f} "
                f"| {r['latency_ms']:>8.2f} | {r['throughput']:>9.1f} | {r['gpu_gb']:>7.2f} |"
            )
    else:
        for r in results:
            print(
                f"{r['label']:<16} params={r['params_m']:.2f}M flops={r['flops_g']:.2f}G "
                f"lat={r['latency_ms']:.2f}ms thr={r['throughput']:.1f}/s mem={r['gpu_gb']:.2f}GB"
            )


if __name__ == "__main__":
    main()
