import os
import sys
import argparse
import yaml
import torch
from tqdm import tqdm

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(__file__))

from train_classifier import load_config, MRIVolumeFolderDataset, _cache_path, _cache_key_for_path

def preprocess_all(config_path):
    print(f"Loading config from {config_path}")
    cfg = load_config(config_path)
    
    # Force cache enablement for this script
    cache_cfg = cfg.get("preprocess_cache", {})
    if not cache_cfg.get("enabled", False):
        print("Warning: Cache is disabled in config. Enabling it temporarily for preprocessing.")
        cache_cfg["enabled"] = True
        
    cache_dir = cache_cfg.get("cache_dir")
    if not cache_dir:
        print("Error: No 'cache_dir' specified in 'preprocess_cache' config.")
        return
        
    os.makedirs(cache_dir, exist_ok=True)
    print(f"Cache directory: {cache_dir}")
    
    # Setup Dataset (Reuse existing logic to find files)
    # We use the Folder dataset logic as it's the primary one used
    label_roots = cfg['paths'].get('label_roots')
    if not label_roots:
        # Check for split roots
        train_roots = cfg['paths'].get('train_label_roots')
        val_roots = cfg['paths'].get('val_label_roots')
        test_roots = cfg['paths'].get('test_label_roots')
        
        roots_list = []
        if train_roots: roots_list.append(train_roots)
        if val_roots: roots_list.append(val_roots)
        if test_roots: roots_list.append(test_roots)
        
        if not roots_list:
            print("Error: No label_roots or train/val/test_label_roots found.")
            return
    else:
        roots_list = [label_roots]

    # Common params
    target_shape = tuple(cfg['input']['shape_dhw'])
    allowed_labels = cfg.get('dataset', {}).get('allowed_labels')
    folder_label_map = cfg.get('dataset', {}).get('folder_label_map', {})
    
    label_map_eff = ({name: idx for idx, name in enumerate(allowed_labels)} if allowed_labels else folder_label_map)
    
    # We will instantiate Datasets just to get the file list and processing logic
    # But we want to iterate and save, not just load
    
    # Since __getitem__ already does caching if cache_dir is set, 
    # we can just iterate over the dataset!
    
    total_processed = 0
    total_skipped = 0
    
    for roots in roots_list:
        print(f"Processing roots: {roots}")
        try:
            ds = MRIVolumeFolderDataset(
                label_roots=roots,
                label_map=label_map_eff,
                target_shape=target_shape,
                validate_nifti=False, # Speed up init
                file_exts=tuple(cfg['dataset'].get('folder_file_exts', [".nii", ".nii.gz"])),
                require_name_substring=cfg['dataset'].get('folder_require_substring'),
                cache_dir=cache_dir,
                cache_enabled=True 
            )
        except RuntimeError as e:
            print(f"Skipping roots {roots}: {e}")
            continue
            
        print(f"Found {len(ds)} samples.")
        
        # Iterate with TQDM
        for i in tqdm(range(len(ds)), desc="Preprocessing"):
            # Accessing the item triggers __getitem__, which triggers caching logic
            # We need to ensure we don't just load from cache if we want to RE-process,
            # but usually 'preprocess' means 'ensure cache exists'.
            # If we want to force re-process, we should delete files.
            # Here we assume 'ensure cache exists'.
            try:
                _ = ds[i]
                total_processed += 1
            except Exception as e:
                print(f"Failed to process sample {i}: {e}")
                total_skipped += 1

    print(f"Done. Processed/Checked: {total_processed}, Skipped/Failed: {total_skipped}")
    print(f"Cache location: {cache_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yaml")
    args = parser.parse_args()
    
    cfg_path = os.path.abspath(args.config)
    if not os.path.isfile(cfg_path):
        # Try finding it in current dir
        cfg_path = os.path.join(os.path.dirname(__file__), args.config)
        
    preprocess_all(cfg_path)
