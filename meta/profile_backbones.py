import argparse
import json
import os
import sys
import time

from typing import Optional

import torch
import yaml

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from medmamba3d import MedMamba3D
from medmamba_ss3m import MedMambaSS3M
from medmamba_ss3m_2dscan import MedMambaSS3M2DScan

try:
    from fvcore.nn import FlopCountAnalysis
    _HAS_FVCORE_FLOPS = True
except Exception:
    _HAS_FVCORE_FLOPS = False

try:
    from thop import profile as thop_profile
    _HAS_THOP = True
except Exception:
    _HAS_THOP = False


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(backbone: str, cfg: dict, in_channels: int, num_classes: int, embed_dim_override: Optional[int]):
    model_cfg = cfg.get("model", {})
    embed_dim = int(embed_dim_override if embed_dim_override is not None else model_cfg.get("embed_dim", 768))
    depth = int(model_cfg.get("depth", 6))
    patch_size = tuple(model_cfg.get("patch_size", [16, 16, 16]))
    backbone = str(backbone).lower()
    if backbone == "medmamba3d":
        return MedMamba3D(
            in_chans=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            patch_size=patch_size,
            num_classes=num_classes,
        )
    if backbone in ("medmamba_ss3m", "ss3m", "medmambass3m"):
        return MedMambaSS3M(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            patch_size=patch_size,
            num_classes=num_classes,
        )
    if backbone in ("medmamba_ss3m_2dscan", "ss3m_2dscan", "ss3m2d", "medmambass3m2dscan"):
        return MedMambaSS3M2DScan(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            patch_size=patch_size,
            num_classes=num_classes,
        )
    raise ValueError(f"Unsupported backbone: {backbone}")


def resolve_device(device_arg: str):
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def count_parameters(model: torch.nn.Module):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def format_count(value: float):
    if value >= 1e9:
        return f"{value / 1e9:.3f}B"
    if value >= 1e6:
        return f"{value / 1e6:.3f}M"
    if value >= 1e3:
        return f"{value / 1e3:.3f}K"
    return f"{value:.0f}"


def compute_flops_fvcore(model: torch.nn.Module, sample: torch.Tensor):
    if _HAS_FVCORE_FLOPS:
        try:
            analysis = FlopCountAnalysis(model, sample)
            total_flops = float(analysis.total())
            unsupported = dict(analysis.unsupported_ops())
            note = ""
            if unsupported:
                note = "unsupported_ops=" + json.dumps({str(k): int(v) for k, v in unsupported.items()}, ensure_ascii=False)
            return total_flops, note
        except Exception as exc:
            pass  # Fall through to thop
    
    if _HAS_THOP:
        try:
            flops, _ = thop_profile(model, inputs=(sample,))
            return float(flops), "thop_macs"
        except Exception as exc:
            return None, f"thop_failed: {type(exc).__name__}: {exc}"
    
    return None, "no_flops_library"


class ForwardWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, forward_fn):
        super().__init__()
        self.model = model
        self._forward_fn = forward_fn

    def forward(self, x: torch.Tensor):
        return self._forward_fn(x)


def resolve_forward_model(model: torch.nn.Module):
    if model.__class__.forward is not torch.nn.Module.forward:
        return model, "forward"
    if hasattr(model, "forward_classifier"):
        return ForwardWrapper(model, model.forward_classifier), "forward_classifier"
    if hasattr(model, "forward_projection"):
        return ForwardWrapper(model, model.forward_projection), "forward_projection"
    if hasattr(model, "forward_encoder"):
        return ForwardWrapper(model, model.forward_encoder), "forward_encoder"
    raise NotImplementedError(f"No usable forward found for {type(model).__name__}")


def measure_latency(model: torch.nn.Module, sample: torch.Tensor, device: torch.device, warmup: int, repeat: int):
    model.eval()
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repeat):
            _ = model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        else:
            peak_mem = 0.0
        elapsed = time.perf_counter() - start
    latency_ms = elapsed * 1000.0 / max(repeat, 1)
    throughput = (repeat * sample.shape[0]) / elapsed
    return latency_ms, throughput, peak_mem


def profile_backbone(backbone: str, cfg: dict, device: torch.device, batch_size: int, in_channels: int, input_shape, num_classes: int, warmup: int, repeat: int, embed_dim_override: Optional[int]):
    model = build_model(backbone, cfg, in_channels=in_channels, num_classes=num_classes, embed_dim_override=embed_dim_override).to(device)
    forward_model, forward_mode = resolve_forward_model(model)
    sample = torch.randn(batch_size, in_channels, *input_shape, device=device)
    total_params, trainable_params = count_parameters(model)
    flops, flops_note = compute_flops_fvcore(forward_model, sample)
    latency_ms, throughput, gpu_mem = measure_latency(forward_model, sample, device, warmup=warmup, repeat=repeat)
    result = {
        "backbone": backbone,
        "device": str(device),
        "input_shape": [batch_size, in_channels, *input_shape],
        "embed_dim": int(model.embed_dim) if hasattr(model, "embed_dim") else None,
        "forward_mode": forward_mode,
        "total_params": total_params,
        "trainable_params": trainable_params,
        "total_params_human": format_count(total_params),
        "trainable_params_human": format_count(trainable_params),
        "latency_ms": latency_ms,
        "throughput": throughput,
        "gpu_mem_gb": gpu_mem,
    }
    if flops is not None:
        if flops_note == "thop_macs":
            result["macs"] = flops
            result["macs_human"] = format_count(flops)
        else:
            result["flops"] = flops
            result["flops_human"] = format_count(flops)
    if flops_note:
        result["flops_note"] = flops_note
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    parser.add_argument("--backbones", nargs="+", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--in-channels", type=int, default=1)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--shape-dhw", nargs=3, type=int, default=None)
    parser.add_argument("--embed-dim", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--table", action="store_true", help="Output as paper-ready table rows")
    args = parser.parse_args()

    cfg = load_config(args.config)
    input_shape = tuple(args.shape_dhw) if args.shape_dhw is not None else tuple(cfg.get("input", {}).get("shape_dhw", [112, 112, 112]))
    default_backbone = cfg.get("classifier", {}).get("backbone", "medmamba_ss3m")
    backbones = args.backbones or [default_backbone, "medmamba_ss3m_2dscan", "medmamba3d"]
    device = resolve_device(args.device)

    results = []
    for backbone in backbones:
        results.append(
            profile_backbone(
                backbone=backbone,
                cfg=cfg,
                device=device,
                batch_size=args.batch_size,
                in_channels=args.in_channels,
                input_shape=input_shape,
                num_classes=args.num_classes,
                warmup=args.warmup,
                repeat=args.repeat,
                embed_dim_override=args.embed_dim,
            )
        )

    if args.json:
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return

    if args.table:
        header = f"{'Backbone':<22} {'Params(M)':>10} {'FLOPs(G)':>10} {'Latency(ms)':>12} {'Throughput':>12} {'GPU(GB)':>8}"
        print(header)
        print("-" * len(header))
        for item in results:
            flops_val = item.get('flops', item.get('macs', 0))
            if flops_val:
                flops_g = flops_val / 1e9
            else:
                flops_g = 0.0
            params_m = item['total_params'] / 1e6
            print(
                f"{item['backbone']:<22} "
                f"{params_m:>10.2f} "
                f"{flops_g:>10.2f} "
                f"{item['latency_ms']:>12.2f} "
                f"{item['throughput']:>11.1f}/s "
                f"{item['gpu_mem_gb']:>7.2f}"
            )
        return

    print(f"[Profile] device={device} input_shape={[args.batch_size, args.in_channels, *input_shape]}")
    for item in results:
        line = (
            f"{item['backbone']}: "
            f"embed_dim={item.get('embed_dim')} "
            f"params={item['total_params_human']} "
            f"trainable={item['trainable_params_human']} "
            f"latency={item['latency_ms']:.2f} ms "
            f"throughput={item['throughput']:.1f}/s "
            f"gpu_mem={item['gpu_mem_gb']:.2f} GB"
        )
        if "flops_human" in item:
            line += f" flops={item['flops_human']}"
        if "macs_human" in item:
            line += f" macs={item['macs_human']}"
        if item.get("flops_note"):
            line += f" note={item['flops_note']}"
        print(line)


if __name__ == "__main__":
    main()
