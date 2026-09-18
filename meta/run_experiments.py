import os
import sys
import yaml
import json
import subprocess
import torch
from thop import profile
import re
import time

# Add the parent directory to sys.path so we can import MedMamba models
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from medmamba_ss3m import MedMambaSS3M
from medmamba_ss3m_2dscan import MedMambaSS3M2DScan


def log(msg):
    print(msg, flush=True)


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


def inspect_label_roots(config_path):
    """
    训练前检查 label_roots：目录是否存在、每个标签可扫描到多少个 NIfTI 文件。
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    dataset_cfg = cfg.get('dataset', {})
    paths_cfg = cfg.get('paths', {})
    allowed_labels = dataset_cfg.get('allowed_labels')
    site_mode = _normalize_site_mode(paths_cfg.get('site_mode', 'both'))
    label_roots = _filter_label_roots_by_site_mode(paths_cfg.get('label_roots', {}), site_mode)
    file_exts = tuple(dataset_cfg.get('folder_file_exts', [".nii", ".nii.gz"]))
    file_exts = tuple(x.lower() for x in file_exts)

    if not label_roots:
        raise RuntimeError("config paths.label_roots is empty.")

    allowed = set(allowed_labels) if isinstance(allowed_labels, (list, tuple)) and allowed_labels else None
    summary = []
    total_files = 0

    for label_name, root in label_roots.items():
        if allowed is not None and label_name not in allowed:
            continue
        roots = root if isinstance(root, (list, tuple)) else [root]
        for r in roots:
            exists = isinstance(r, str) and os.path.isdir(r)
            count = 0
            if exists:
                for dirpath, _, filenames in os.walk(r):
                    for fn in filenames:
                        if fn.lower().endswith(file_exts):
                            count += 1
            summary.append((label_name, r, exists, count))
            total_files += count

    log("[DataCheck] label_roots scan summary:")
    log(f"[DataCheck] site_mode={site_mode}")
    for label_name, root_path, exists, count in summary:
        log(f"  - label={label_name} exists={exists} files={count} root={root_path}")

    if total_files == 0:
        raise RuntimeError("No NIfTI files found from config paths.label_roots (after allowed_labels filter).")

    return summary, total_files

def get_model_stats(backbone, embed_dim, in_channels=1, depth=8, patch_size=(16, 16, 16), num_classes=2):
    """
    计算给定主干网络和嵌入维度的参数量 (Parameters) 和计算量 (FLOPs)。
    使用 thop 库进行性能评估。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if backbone in ("medmamba_ss3m", "ss3m", "medmambass3m"):
        model = MedMambaSS3M(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            patch_size=patch_size,
            num_classes=num_classes
        ).to(device)
    elif backbone in ("medmamba_ss3m_2dscan", "ss3m_2dscan", "ss3m2d", "medmambass3m2dscan"):
        model = MedMambaSS3M2DScan(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            patch_size=patch_size,
            num_classes=num_classes
        ).to(device)
    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    model.eval()
    x = torch.randn(1, in_channels, 112, 112, 112).to(device)
    macs, params = profile(model, inputs=(x,), verbose=False)
    flops = macs * 2  # FLOPS is roughly 2 * MACs
    
    # Clean up memory
    del model
    del x
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return params, flops

def update_config(config_path, embed_dim, backbone):
    """
    修改 config.yaml 配置文件中的变量以匹配当前实验参数。
    将设置网络主干、特征维度，并固定为 nested_cv 5-fold。
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    config['model']['embed_dim'] = embed_dim
    config['classifier']['backbone'] = backbone
    config.setdefault('cv', {})['split_mode'] = 'nested_cv'
    config['cv']['run_folds'] = 5
    config['cv']['cv_n_splits'] = 5
    config['cv'].setdefault('nested_cv', {})['outer_folds'] = 5
    config['cv']['nested_cv'].setdefault('locked_test_ratio', 0.2)
    config['cv']['nested_cv'].setdefault('patient_id_mode', 'parent_dir')
    
    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
        
    # Return useful info for model stats
    in_channels = 1
    depth = config['model'].get('depth', 8)
    patch_size = tuple(config['model'].get('patch_size', [16, 16, 16]))
    return in_channels, depth, patch_size

def run_experiment_and_parse(train_script, config_path):
    """
    通过子进程运行 train_classifier.py 脚本并解析其控制台输出以获取 BAC 和 AUC。
    """
    cmd = [sys.executable, "-u", train_script, "--config", config_path]
    log(f"Running command: {' '.join(cmd)}")
    
    child_env = os.environ.copy()
    child_env["PYTHONUNBUFFERED"] = "1"

    # 在 train_script 所在目录运行，避免相对路径配置读取错误
    process = subprocess.Popen(
        cmd,
        cwd=os.path.dirname(train_script),
        env=child_env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8',
        bufsize=1
    )
    
    bac = None
    auc = None
    test_bac = None
    test_auc = None
    
    for line in process.stdout:
        print(line, end='', flush=True)
        # 训练脚本 best 行通常形如：
        # [CV 1] best_epoch=4 ... raw_acc=... raw_auc=... raw_bal_acc=... raw_bal_acc_tuned=...
        # 也兼容按 epoch 打印的字段名：raw_val_auc/raw_val_bal_acc_tuned
        if re.search(r"\[CV\s+\d+\]\s+best_epoch=", line):
            match_bac = re.search(r'raw_bal_acc_tuned=([\d\.]+|nan)', line)
            if not match_bac:
                match_bac = re.search(r'raw_bal_acc=([\d\.]+|nan)', line)
            if not match_bac:
                match_bac = re.search(r'raw_val_bal_acc_tuned=([\d\.]+|nan)', line)
            if not match_bac:
                match_bac = re.search(r'raw_val_bal_acc=([\d\.]+|nan)', line)
            if match_bac:
                val = match_bac.group(1)
                bac = float('nan') if val == 'nan' else float(val)
                
            match_auc = re.search(r'raw_auc=([\d\.]+|nan)', line)
            if not match_auc:
                match_auc = re.search(r'raw_val_auc=([\d\.]+|nan)', line)
            if match_auc:
                val = match_auc.group(1)
                auc = float('nan') if val == 'nan' else float(val)

        if re.search(r"^\[TEST\s+\d+\]", line):
            match_test_bac = re.search(r'test_bal_acc=([\d\.]+|nan)', line)
            if not match_test_bac:
                match_test_bac = re.search(r'test_bal_acc_tuned=([\d\.]+|nan)', line)
            if match_test_bac:
                val = match_test_bac.group(1)
                test_bac = float('nan') if val == 'nan' else float(val)

            match_test_auc = re.search(r'test_auc=([\d\.]+|nan)', line)
            if match_test_auc:
                val = match_test_auc.group(1)
                test_auc = float('nan') if val == 'nan' else float(val)
    
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"train_classifier.py exited with code {process.returncode}")

    if test_bac is not None or test_auc is not None:
        return test_bac, test_auc
    return bac, auc

def main():
    """
    主控制流：遍历各个网络主干和不同嵌入维度的组合配置。
    更新配置文件、计算模型参数量与计算量，然后运行分类训练脚本收集 BAC 和 AUC，并写入 experiment_results.json。
    """
    embed_dims = [768, 384, 192]
    backbones = ["medmamba_ss3m", "medmamba_ss3m_2dscan"]
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(base_dir, "config.yaml")
    train_script = os.path.join(base_dir, "train_classifier.py")

    # Preflight data path check to fail fast with clear diagnostics.
    inspect_label_roots(config_path)
    
    results = []
    
    for backbone in backbones:
        for embed_dim in embed_dims:
            log(f"\\n{'='*50}")
            log(f"Testing configuration: backbone={backbone}, embed_dim={embed_dim}")
            log(f"{'='*50}")
            
            # Update config and get model settings
            in_channels, depth, patch_size = update_config(config_path, embed_dim, backbone)
            
            # Calculate Params and FLOPs
            t0 = time.time()
            log("[Stage] Calculating parameters and FLOPs (this can be slow on 3D input)...")
            params, flops = get_model_stats(backbone, embed_dim, in_channels, depth, patch_size, num_classes=2)
            log(f"[Stage] Model stats done in {time.time() - t0:.1f}s")
            log(f"Parameters: {params:,}, FLOPs: {flops:,}")
            
            # Run train_classifier.py
            t1 = time.time()
            log("[Stage] Launching train_classifier.py...")
            bac, auc = run_experiment_and_parse(train_script, config_path)
            log(f"[Stage] train_classifier.py finished in {time.time() - t1:.1f}s")
            
            log(f"Results for backbone={backbone}, embed_dim={embed_dim}:")
            log(f"BAC: {bac}, AUC: {auc}")
            
            result_entry = {
                "backbone": backbone,
                "embed_dim": embed_dim,
                "parameter": params,
                "flops": flops,
                "BAC": bac,
                "AUC": auc
            }
            results.append(result_entry)
            
            # 保存中间结果以防崩溃
            with open(os.path.join(base_dir, "experiment_results.json"), "w", encoding='utf-8') as f:
                json.dump(results, f, indent=4)
                
    log("\\nAll experiments completed.")
    log("Final Results saved to experiment_results.json")

if __name__ == "__main__":
    main()
