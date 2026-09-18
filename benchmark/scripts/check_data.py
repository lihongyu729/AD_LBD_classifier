#!/usr/bin/env python
"""
Check dataset integrity before running experiments.

Usage:
    python scripts/check_data.py --config configs/base_config.yaml
    python scripts/check_data.py --config configs/base_config.yaml --method anil
"""
import os
import sys
import argparse

# Add benchmark root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config_loader import load_config, resolve_data_paths


def main():
    parser = argparse.ArgumentParser(description="Check dataset integrity")
    parser.add_argument("--config", type=str, default="configs/base_config.yaml",
                        help="Path to base config")
    parser.add_argument("--method", type=str, default=None,
                        help="Also load method-specific config")
    args = parser.parse_args()

    # Load configs
    configs = [args.config]
    if args.method:
        method_cfg = f"configs/methods/{args.method}.yaml"
        if os.path.isfile(method_cfg):
            configs.append(method_cfg)

    cfg = load_config(*configs)
    cfg = resolve_data_paths(cfg)

    # Check data paths
    data_cfg = cfg.get("data", {})
    label_roots = data_cfg.get("label_roots", {})

    if not label_roots:
        print("[ERROR] No data.label_roots configured!")
        print("Please update configs/base_config.yaml with your actual data paths.")
        return 1

    import nibabel as nib
    import numpy as np

    file_exts = tuple(data_cfg.get("folder_file_exts", [".nii", ".nii.gz"]))
    target_shape = tuple(data_cfg.get("target_shape", [112, 112, 112]))

    total_files = 0
    for label, root in label_roots.items():
        roots = root if isinstance(root, (list, tuple)) else [root]
        for r in roots:
            if not os.path.isdir(r):
                print(f"[WARN] Directory not found: {r} (label={label})")
                continue
            count = 0
            for dirpath, _, filenames in os.walk(r):
                for fn in filenames:
                    if fn.lower().endswith(file_exts):
                        count += 1
                        total_files += 1
            print(f"  label={label}: {count} files found in {r}")

    if total_files == 0:
        print("[ERROR] No NIfTI files found!")
        return 1

    print(f"\n[OK] Total files found: {total_files}")
    print(f"[OK] Target shape: {target_shape}")
    print(f"[OK] Config looks valid. Ready to run experiments.")

    # Check one sample
    for label, root in label_roots.items():
        roots = root if isinstance(root, (list, tuple)) else [root]
        for r in roots:
            if not os.path.isdir(r):
                continue
            for dirpath, _, filenames in os.walk(r):
                for fn in filenames:
                    if fn.lower().endswith(file_exts):
                        p = os.path.join(dirpath, fn)
                        try:
                            if p.lower().endswith(".npy"):
                                a = np.load(p)
                                print(f"\n[Sample] {fn}: shape={a.shape}, dtype={a.dtype}, "
                                      f"mean={a.mean():.4f}, frac_nonzero={(a != 0).mean():.3f}")
                            else:
                                img = nib.load(p)
                                print(f"\n[Sample] {fn}: shape={img.shape}, dtype={img.get_data_dtype()}")
                        except Exception as e:
                            print(f"\n[WARN] Cannot load {fn}: {e}")
                        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
