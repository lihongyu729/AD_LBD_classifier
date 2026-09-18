# -*- coding: utf-8 -*-
"""Profile parameter count and FLOPs for MedMambaSS3M.

Default target:
- embed_dim: 768, 384, 192, 96
- input shape (C,D,H,W): 1,112,112,112
- depth: 6
- patch_size: 2,2,2
"""

import argparse
import os
import sys
from typing import List, Optional, Tuple

import torch

# Make sure repo root is on sys.path so we can import medmamba_ss3m.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

from medmamba_ss3m import MedMambaSS3M


def _parse_int_list(vals: List[str]) -> List[int]:
    out = []
    for v in vals:
        try:
            out.append(int(v))
        except Exception:
            raise ValueError(f"Invalid int value: {v}")
    return out


def _parse_tuple3(vals: List[str]) -> Tuple[int, int, int]:
    if len(vals) != 3:
        raise ValueError("Expected 3 integers, e.g. 2 2 2")
    nums = _parse_int_list(vals)
    return (nums[0], nums[1], nums[2])


def _count_params(model: torch.nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _compute_flops(model: torch.nn.Module, x: torch.Tensor) -> Optional[float]:
    """Return total FLOPs, or None if no supported tool is available."""
    model.eval()
    with torch.no_grad():
        # Prefer fvcore if available.
        try:
            from fvcore.nn import FlopCountAnalysis

            flops = FlopCountAnalysis(model, x).total()
            return float(flops)
        except Exception:
            pass

        # Fallback to thop if available.
        try:
            from thop import profile as thop_profile

            flops, _params = thop_profile(model, inputs=(x,), verbose=False)
            return float(flops)
        except Exception:
            pass

        # Last resort: torch.profiler (may undercount).
        try:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if x.device.type == "cuda":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(
                activities=activities,
                with_flops=True,
                record_shapes=False,
            ) as prof:
                model(x)
            total = 0.0
            for evt in prof.key_averages():
                if evt.flops is not None:
                    total += float(evt.flops)
            if total > 0:
                return total
        except Exception:
            pass

    return None


def _format_float(val: Optional[float]) -> str:
    if val is None:
        return "NA"
    return f"{val:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embed-dims", nargs="+", default=["768", "384", "192", "96"],
                        help="Embed dims to profile, e.g. 768 384 192 96")
    parser.add_argument("--input-shape", nargs=4, default=["1", "112", "112", "112"],
                        help="Input shape (C D H W)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--patch-size", nargs=3, default=["2", "2", "2"],
                        help="Patch size (D H W)")
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--branch-types", nargs=2, default=["conv", "conv"])
    parser.add_argument("--n-dirs-train", type=int, default=8)
    parser.add_argument("--use-dual-branch", action="store_true", default=True)
    parser.add_argument("--no-dual-branch", dest="use_dual_branch", action="store_false")
    parser.add_argument("--use-pos-emb", action="store_true", default=True)
    parser.add_argument("--no-pos-emb", dest="use_pos_emb", action="store_false")
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use-checkpoint", action="store_true", default=False)
    parser.add_argument("--skip-flops", action="store_true", default=False)
    args = parser.parse_args()

    embed_dims = _parse_int_list(args.embed_dims)
    in_shape = _parse_int_list(args.input_shape)
    if len(in_shape) != 4:
        raise ValueError("input-shape must be 4 integers: C D H W")

    patch_size = _parse_tuple3(args.patch_size)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[Warn] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    c, d, h, w = in_shape
    x = torch.randn((args.batch_size, c, d, h, w), device=device, dtype=torch.float32)

    rows = []
    for ed in embed_dims:
        model = MedMambaSS3M(
            in_channels=c,
            num_classes=args.num_classes,
            embed_dim=ed,
            depth=args.depth,
            patch_size=patch_size,
            dropout=args.dropout,
            use_pos_emb=args.use_pos_emb,
            merge_type="softmax",
            use_checkpoint=args.use_checkpoint,
            branch_types=(args.branch_types[0], args.branch_types[1]),
            n_dirs_train=args.n_dirs_train,
            use_dual_branch=args.use_dual_branch,
        ).to(device)

        total_params, trainable_params = _count_params(model)
        flops = None
        if not args.skip_flops:
            flops = _compute_flops(model, x)

        rows.append({
            "embed_dim": ed,
            "params_total": total_params,
            "params_trainable": trainable_params,
            "params_m": total_params / 1e6,
            "flops_total": flops,
            "flops_g": (flops / 1e9) if flops is not None else None,
        })

    # Standardized output (CSV + table)
    print("# Params/FLOPs report")
    print(f"# input_shape=(B,C,D,H,W) = ({args.batch_size},{c},{d},{h},{w})")
    print(f"# depth={args.depth} patch_size={patch_size} n_dirs_train={args.n_dirs_train}")
    print("embed_dim,params_total,params_trainable,params_m,flops_total,flops_g")
    for r in rows:
        print(
            f"{r['embed_dim']},"
            f"{r['params_total']},"
            f"{r['params_trainable']},"
            f"{r['params_m']:.6f},"
            f"{_format_float(r['flops_total'])},"
            f"{_format_float(r['flops_g'])}"
        )

    print("")
    print("| embed_dim | params (M) | FLOPs (G) |")
    print("|---:|---:|---:|")
    for r in rows:
        params_m = f"{r['params_m']:.3f}"
        flops_g = "NA" if r["flops_g"] is None else f"{r['flops_g']:.3f}"
        print(f"| {r['embed_dim']} | {params_m} | {flops_g} |")


if __name__ == "__main__":
    main()
