"""
Utility functions: seed setting, GPU detection, logging helpers.
"""
import os
import sys
import random
import logging
import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False):
    """Set random seed for reproducibility across torch, numpy, random, and cuda."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False


def detect_gpu() -> tuple:
    """Detect available GPUs. Returns (device, device_type, gpu_count)."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        device = torch.device("cuda")
        device_type = "cuda"
    else:
        gpu_count = 0
        device = torch.device("cpu")
        device_type = "cpu"
    return device, device_type, gpu_count


def setup_logging(name: str = "benchmark", level: int = logging.INFO) -> logging.Logger:
    """Create a logger with console handler."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def get_git_hash() -> str:
    """Get current git commit hash for experiment tracking."""
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def format_time(seconds: float) -> str:
    """Format elapsed time in seconds to a human-readable string."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h > 0:
        return f"{h}h{m:02d}m{s:02d}s"
    elif m > 0:
        return f"{m}m{s:02d}s"
    else:
        return f"{s}s"


# ---------------------------------------------------------------------------
# Python 3.8 / PyTorch < 2.1 compatibility: AMP context manager
# ---------------------------------------------------------------------------
# torch.amp.autocast(device_type=...) was added in PyTorch 2.1.
# On Python 3.8 servers with older PyTorch, we fall back to
# torch.cuda.amp.autocast (which only works on CUDA — but that's all we need).
_AMP_HAS_DEVICE_TYPE = hasattr(torch, "amp") and hasattr(torch.amp, "autocast")


class _AmpContext:
    """
    Context-manager wrapper that auto-selects the correct AMP API.

    Usage:
        with autocast_ctx(enabled=True, dtype=torch.bfloat16):
            y = model(x)

    On PyTorch >= 2.1:   uses torch.amp.autocast("cuda", ...)
    On PyTorch <  2.1:   uses torch.cuda.amp.autocast(...)
    """
    __slots__ = ("_enabled", "_dtype", "_ctx")

    def __init__(self, enabled: bool = True, dtype=None):
        self._enabled = enabled
        self._dtype = dtype
        self._ctx = None

    def __enter__(self):
        if _AMP_HAS_DEVICE_TYPE:
            self._ctx = torch.amp.autocast(
                device_type="cuda", enabled=self._enabled, dtype=self._dtype,
            )
        else:
            self._ctx = torch.cuda.amp.autocast(
                enabled=self._enabled, dtype=self._dtype,
            )
        return self._ctx.__enter__()

    def __exit__(self, *args):
        if self._ctx is not None:
            return self._ctx.__exit__(*args)


autocast_ctx = _AmpContext
