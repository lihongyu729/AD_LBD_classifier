#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Offline .npy conversion: 128^3 (or N^3) preprocessed volumes -> target-size .npy.

Input  : E:\\nifti\\proprecessed\\preprocessed\\images\\{AD,LBD}\\*.npy
         (already brain-mask Z-scored, background=0, float32, shape (D,H,W))
Output : {output_root}/{AD,LBD}/{same_name}.npy  resized to target_size
         + manifest.csv (file,label,patient_id)

Resize strategy:
  - image : trilinear (order=1) with anti-aliasing + preserve_range
  - mask  : nearest (order=0), threshold 0.5, then re-zero background
            (keeps the "background=0" invariant exact after downsampling)

Usage:
  python scripts/prepare_npy.py
  python scripts/prepare_npy.py --source-root "E:/nifti/proprecessed/preprocessed/images" \
      --output-root ./Dataset_112_npy --target-size 112 --overwrite
"""
import os
import sys
import csv
import argparse

import numpy as np

try:
    from skimage.transform import resize as _sk_resize
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def _stem(path: str) -> str:
    """Filename without extension (handles .nii.gz / .npy)."""
    name = os.path.basename(path)
    for ext in (".npy", ".nii.gz", ".nii", ".npz"):
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return name


def resize_volume(vol: np.ndarray, target: tuple, order: int, aa: bool) -> np.ndarray:
    """Resize a 3D volume to `target` using skimage (fallback: scipy.zoom)."""
    if tuple(vol.shape) == tuple(target):
        return vol.astype(np.float32)
    if HAS_SKIMAGE:
        return _sk_resize(
            vol, target, order=order, preserve_range=True,
            anti_aliasing=aa, mode="constant", cval=0.0,
        ).astype(np.float32)
    from scipy.ndimage import zoom
    factors = [t / s for t, s in zip(target, vol.shape)]
    return zoom(vol, factors, order=order, mode="constant", cval=0.0).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description="Convert preprocessed .npy volumes to a target-size .npy dataset.")
    parser.add_argument("--source-root", type=str,
                        default=r"E:/nifti/proprecessed/preprocessed/images",
                        help="Root containing one subfolder per label with .npy files")
    parser.add_argument("--output-root", type=str, default="./Dataset_112_npy",
                        help="Output dataset root (label subfolders created here)")
    parser.add_argument("--labels", type=str, default="AD,LBD", help="Comma-separated labels (class subfolders)")
    parser.add_argument("--source-size", type=int, default=128, help="Expected source volume size (cubic)")
    parser.add_argument("--target-size", type=int, default=112, help="Target volume size (cubic)")
    parser.add_argument("--mask-root", type=str, default=None,
                        help="Root containing label subfolders with *_mask.npy (default: source-root/../masks)")
    parser.add_argument("--no-resize-masks", action="store_true",
                        help="Skip applying masks (background rim not re-zeroed)")
    parser.add_argument("--overwrite", action="store_true", help="Re-process existing output files")
    args = parser.parse_args()

    labels = [x.strip() for x in args.labels.split(",") if x.strip()]
    target = (args.target_size, args.target_size, args.target_size)
    mask_root = args.mask_root
    if mask_root is None:
        mask_root = os.path.normpath(os.path.join(args.source_root, "..", "masks"))

    os.makedirs(args.output_root, exist_ok=True)
    manifest_path = os.path.join(args.output_root, "manifest.csv")
    manifest_written_header = os.path.isfile(manifest_path) and not args.overwrite

    rows = []
    totals = {}
    for label in labels:
        src_dir = os.path.join(args.source_root, label)
        out_dir = os.path.join(args.output_root, label)
        if not os.path.isdir(src_dir):
            print(f"[prepare_npy][warn] source label dir not found: {src_dir}", flush=True)
            continue
        os.makedirs(out_dir, exist_ok=True)
        mask_dir = os.path.join(mask_root, label) if os.path.isdir(os.path.join(mask_root, label)) else None

        files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(".npy"))
        n_ok, n_skip, n_err = 0, 0, 0
        for fn in files:
            src_path = os.path.join(src_dir, fn)
            out_path = os.path.join(out_dir, fn)
            if os.path.isfile(out_path) and not args.overwrite:
                n_skip += 1
                rows.append([fn, label, _stem(fn)])
                continue
            try:
                vol = np.load(src_path)
                if vol.ndim == 4:  # (1,D,H,W) or (C,D,H,W) -> take first channel
                    vol = vol[0]
                if vol.ndim != 3:
                    print(f"[prepare_npy][warn] skip {fn}: ndim={vol.ndim}", flush=True)
                    n_err += 1
                    continue
                vol = vol.astype(np.float32)
                if tuple(vol.shape) != (args.source_size,) * 3:
                    # tolerate non-exact source size: still resize to target
                    pass
                out = resize_volume(vol, target, order=1, aa=True)

                # Re-zero background using resized binary mask (exact background=0 invariant)
                if mask_dir is not None and not args.no_resize_masks:
                    mask_path = os.path.join(mask_dir, fn.replace(".npy", "_mask.npy"))
                    if os.path.isfile(mask_path):
                        mask = np.load(mask_path).astype(np.float32)
                        if mask.ndim == 4:
                            mask = mask[0]
                        m = resize_volume(mask, target, order=0, aa=False)
                        out[m < 0.5] = 0.0

                np.save(out_path, np.ascontiguousarray(out))
                rows.append([fn, label, _stem(fn)])
                n_ok += 1
            except Exception as e:
                print(f"[prepare_npy][error] {fn}: {e}", flush=True)
                n_err += 1

        totals[label] = {"files": len(files), "ok": n_ok, "skip": n_skip, "err": n_err}
        print(f"[prepare_npy] {label}: total={len(files)} ok={n_ok} skip={n_skip} err={n_err}", flush=True)

    # manifest
    with open(manifest_path, "a" if manifest_written_header else "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not manifest_written_header:
            writer.writerow(["file", "label", "patient_id"])
        writer.writerows(rows)

    # verify output
    print("\n[prepare_npy] verify output:", flush=True)
    all_ids = set()
    for label in labels:
        out_dir = os.path.join(args.output_root, label)
        if not os.path.isdir(out_dir):
            continue
        fns = sorted(f for f in os.listdir(out_dir) if f.endswith(".npy"))
        shapes = {}
        for fn in fns[:5]:
            a = np.load(os.path.join(out_dir, fn))
            shapes[tuple(a.shape)] = shapes.get(tuple(a.shape), 0) + 1
        for fn in fns:
            all_ids.add(_stem(fn))
        print(f"  {label}: {len(fns)} files, sample shapes={shapes}", flush=True)
        if fns:
            a = np.load(os.path.join(out_dir, fns[0]))
            nz = a != 0
            print(f"    e.g. {fns[0]}: min={a.min():.3f} max={a.max():.3f} "
                  f"mean={a.mean():.3f} frac_nonzero={nz.mean():.3f}", flush=True)
    print(f"[prepare_npy] total unique patient ids: {len(all_ids)}", flush=True)
    print(f"[prepare_npy] manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
