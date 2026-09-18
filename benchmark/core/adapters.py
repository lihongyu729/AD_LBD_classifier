"""
Input adapter module — transparently transform 112³ 3D MRI volumes
to whatever input shape each method expects.

Adapter strategies:
    - identity:   Pass-through (method already accepts 112³)
    - resize:     Trilinear interpolation to target shape
    - adaptive_pool: AdaptiveAvgPool3d to fixed spatial size
    - slice_2d:   Extract center slice for 2D-only models

Usage:
    from core.adapters import InputAdapter, create_adapter
    adapter = create_adapter(source_shape=(112,112,112), target_shape=(96,96,96), strategy="resize")
    x_adapted = adapter(x)  # [B,1,112,112,112] → [B,1,96,96,96]
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, Any


class InputAdapter(nn.Module):
    """
    Unified input adapter that wraps multiple transformation strategies.

    Each method wrapper should declare its expected input via:
        expected_input_shape: Tuple[int,int,int] or None (=any size accepted)
        expected_channels: int (=1 for grayscale MRI)
    """

    def __init__(self, transform: nn.Module, description: str = ""):
        super().__init__()
        self.transform = transform
        self.description = description

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, C, D, H, W] → adapted tensor"""
        return self.transform(x)

    def __repr__(self):
        return f"InputAdapter({self.description})"


class IdentityAdapter(nn.Module):
    """Pass-through — for methods that already accept 112³."""
    def forward(self, x):
        return x

    def __repr__(self):
        return "IdentityAdapter"


class Resize3DAdapter(nn.Module):
    """
    Trilinear interpolation to target shape.
    Used when method expects a different cubic size (e.g., 96³ or 160³).

    Args:
        target_shape: (D, H, W) target spatial dimensions.
        mode: interpolation mode ('trilinear' recommended for 3D MRI).
    """

    def __init__(self, target_shape: Tuple[int, int, int], mode: str = "trilinear"):
        super().__init__()
        self.target_shape = target_shape
        self.mode = mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W]
        if tuple(x.shape[2:]) == self.target_shape:
            return x
        return F.interpolate(
            x,
            size=self.target_shape,
            mode=self.mode,
            align_corners=False,
        )

    def __repr__(self):
        return f"Resize3DAdapter(target={self.target_shape}, mode={self.mode})"


class AdaptivePool3DAdapter(nn.Module):
    """
    Adaptive average pooling to fixed spatial size.
    Used when a method's FC layer expects a specific flattened feature size
    but the input volume can be any size (e.g., CNN with hard-coded FC dims).
    """
    def __init__(self, output_size: Tuple[int, int, int]):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(x)

    def __repr__(self):
        return f"AdaptivePool3DAdapter(output_size={self.pool.output_size})"


class Slice2DAdapter(nn.Module):
    """
    Extract center slice(s) from 3D volume for 2D-only models.
    Used for legacy 2D architectures (e.g., original CrossFormer, ResNet2D).

    Args:
        axis: Axis to slice along (0=D, 1=H, 2=W). Default=0 (axial/sagittal).
        num_slices: Number of slices to extract as "channels".
        normalize_3d: Whether to replicate 2D features back to 3D.
    """
    def __init__(self, axis: int = 0, num_slices: int = 3):
        super().__init__()
        self.axis = axis + 2  # Offset for [B, C, D, H, W] indexing
        self.num_slices = num_slices

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, D, H, W] → extract center slices along axis
        D = x.shape[self.axis]
        center = D // 2
        half = self.num_slices // 2
        start = max(0, center - half)
        indices = [slice(None), slice(None), slice(None), slice(None), slice(None)]

        slices_out = []
        for i in range(self.num_slices):
            idx = min(start + i, D - 1)
            indices[self.axis] = slice(idx, idx + 1)
            sl = x[indices]  # [B, C, 1 or 1, H or 1, W or 1]
            # Squeeze the sliced dim and treat as channel
            sl = sl.squeeze(self.axis)  # [B, C, H, W] or [B, C, D, W] etc
            slices_out.append(sl)

        # Stack sliced dim into channels: [B, C*num_slices, H, W]
        return torch.cat(slices_out, dim=1)

    def __repr__(self):
        return f"Slice2DAdapter(axis={self.axis - 2}, num_slices={self.num_slices})"


class ChannelAdapter(nn.Module):
    """
    Adapt channel count (e.g., 1→3 for RGB-pretrained models).
    Simply repeats the single channel.
    """
    def __init__(self, target_channels: int = 3):
        super().__init__()
        self.target_channels = target_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.size(1) == self.target_channels:
            return x
        return x.repeat(1, self.target_channels, 1, 1, 1)

    def __repr__(self):
        return f"ChannelAdapter(target={self.target_channels})"


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_adapter(
    source_shape: Tuple[int, int, int],
    target_shape: Optional[Tuple[int, int, int]],
    strategy: str = "identity",
    in_channels: int = 1,
    target_channels: Optional[int] = None,
) -> InputAdapter:
    """
    Create an input adapter based on source and target shapes.

    Args:
        source_shape: (D, H, W) of the data (typically 112,112,112).
        target_shape: (D, H, W) expected by the model, or None if flexible.
        strategy: 'identity' | 'resize' | 'adaptive_pool' | 'slice_2d'
        in_channels: Input channels (usually 1 for MRI).
        target_channels: If not None, adapt channel count.

    Returns:
        InputAdapter wrapping the appropriate transform.
    """
    transforms = []
    description_parts = []

    # Spatial adaptation
    if target_shape is None or tuple(source_shape) == tuple(target_shape):
        transforms.append(IdentityAdapter())
        description_parts.append("identity")
    elif strategy == "resize":
        transforms.append(Resize3DAdapter(target_shape))
        description_parts.append(f"resize {source_shape}→{target_shape}")
    elif strategy == "adaptive_pool":
        transforms.append(AdaptivePool3DAdapter(target_shape))
        description_parts.append(f"adaptive_pool→{target_shape}")
    elif strategy == "slice_2d":
        transforms.append(Slice2DAdapter(axis=0, num_slices=3))
        description_parts.append("slice_2d(3 slices as channels)")
    else:
        # Default: try resize
        transforms.append(Resize3DAdapter(target_shape))
        description_parts.append(f"resize(auto) {source_shape}→{target_shape}")

    # Channel adaptation
    if target_channels is not None and target_channels != in_channels:
        transforms.append(ChannelAdapter(target_channels))
        description_parts.append(f"channels {in_channels}→{target_channels}")

    # Compose
    if len(transforms) == 1:
        return InputAdapter(transforms[0], description_parts[0])
    else:
        return InputAdapter(nn.Sequential(*transforms), " + ".join(description_parts))


# ---------------------------------------------------------------------------
# Predefined adapters for common methods
# ---------------------------------------------------------------------------

ADAPTER_REGISTRY: Dict[str, Dict[str, Any]] = {
    # Methods that natively support 112³
    "medmamba_ss3m":   {"target_shape": None,        "strategy": "identity"},
    "medmamba3d":      {"target_shape": None,        "strategy": "identity"},
    "vanilla":          {"target_shape": None,        "strategy": "identity"},
    "anil":             {"target_shape": None,        "strategy": "identity"},
    "protonet":         {"target_shape": None,        "strategy": "identity"},
    "maml":             {"target_shape": None,        "strategy": "identity"},
    "hybrid":           {"target_shape": None,        "strategy": "identity"},

    # Methods that are flexible (use AdaptiveAvgPool3d internally)
    "resnet3d":        {"target_shape": None,        "strategy": "identity"},
    "cnn3d_baseline":  {"target_shape": (5, 5, 5),  "strategy": "adaptive_pool"},
    "crossformer3d":   {"target_shape": None,        "strategy": "identity"},

    # Methods needing explicit resize
    "mm3dmcf":         {"target_shape": (96, 96, 96),  "strategy": "resize"},
    "diamond":         {"target_shape": (112, 112, 112), "strategy": "resize"},  # needs block_size=16/28/56
    "densenet3d":      {"target_shape": None,        "strategy": "identity"},
}


def get_method_adapter(method_name: str, source_shape: Tuple[int,int,int]=(112,112,112)) -> InputAdapter:
    """
    Get the recommended adapter for a given method name.

    Args:
        method_name: Registered method name (e.g., 'resnet3d', 'cnn3d_baseline').
        source_shape: Source data shape (D, H, W).

    Returns:
        InputAdapter instance.
    """
    info = ADAPTER_REGISTRY.get(method_name)
    if info is None:
        # Unknown method — assume identity (pass-through)
        print(f"[Adapter] No adapter info for '{method_name}', using identity.", flush=True)
        return InputAdapter(IdentityAdapter(), "identity(fallback)")

    return create_adapter(
        source_shape=source_shape,
        target_shape=info["target_shape"],
        strategy=info["strategy"],
    )


def print_adapter_info():
    """Print adapter compatibility table."""
    print("\n" + "=" * 70)
    print("Input Adapter Compatibility (source: 112³)")
    print("=" * 70)
    print(f"{'Method':<20s} {'Target Shape':<16s} {'Strategy':<16s} {'Status':<10s}")
    print("-" * 70)
    for method, info in ADAPTER_REGISTRY.items():
        target = str(info.get("target_shape", "flexible"))
        strategy = info.get("strategy", "identity")
        status = "✓" if strategy in ("identity", "adaptive_pool") else "~resize"
        print(f"{method:<20s} {target:<16s} {strategy:<16s} {status:<10s}")
    print("=" * 70 + "\n")
