import os
import yaml
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from dataset_mri3d import MRIVolumeDataset, ensure_3d_volume, zscore_normalize, center_pad_or_crop
from medmamba3d import MedMamba3D
from medmamba_ss3m import MedMambaSS3MContrast
import argparse
import numpy as np
import json
import random


def load_config(cfg_path: str):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def _load_pretrained_partial(model, ckpt_path):
    state = torch.load(ckpt_path, map_location='cpu')
    if isinstance(state, dict):
        if 'state_dict' in state and isinstance(state['state_dict'], dict):
            state = state['state_dict']
        elif 'model_state' in state and isinstance(state['model_state'], dict):
            state = state['model_state']
        elif 'model' in state and isinstance(state['model'], dict):
            state = state['model']
    tgt = model.state_dict()
    new_state = {}
    for k, v in state.items():
        candidates = [k]
        if k.startswith("module."):
            candidates.append(k[7:])
        if k.startswith("encoder."):
            candidates.append(k[len("encoder."):])
        if k.startswith("backbone."):
            candidates.append(k[len("backbone."):])
        if k.startswith("model."):
            candidates.append(k[len("model."):])
        k2 = None
        for cand in candidates:
            if cand in tgt:
                k2 = cand
                break
        if k2 is None:
            continue
        if tgt[k2].shape == v.shape:
            new_state[k2] = v
            continue
    for k in tgt.keys():
        if ".ssm_b." in k and k not in new_state:
            k_a = k.replace(".ssm_b.", ".ssm_a.")
            if k_a in new_state and tgt[k].shape == new_state[k_a].shape:
                new_state[k] = new_state[k_a].clone()
    missing, unexpected = model.load_state_dict(new_state, strict=False)
    print(f"[Contrast] MAE weights loaded keys={len(new_state)} missing={len(missing)} unexpected={len(unexpected)}")
    return missing, unexpected


def _build_proj_head(in_dim, proj_dim, head_type):
    if head_type == "mlp3":
        return nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )
    return nn.Sequential(
        nn.Linear(in_dim, proj_dim),
        nn.GELU(),
        nn.Linear(proj_dim, proj_dim)
    )


def augment(x: torch.Tensor, strength: float = 1.0, noise_scale: float = 0.02, crop_ratio: float = 0.9):
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=[-1])
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=[-2])
    if torch.rand(1) < 0.5:
        x = torch.flip(x, dims=[-3])

    B, C, D, H, W = x.shape
    ratio = max(0.7, min(0.98, crop_ratio * strength))
    d = max(int(D * ratio), D - 8)
    h = max(int(H * ratio), H - 8)
    w = max(int(W * ratio), W - 8)
    sd = torch.randint(0, D - d + 1, (1,)).item()
    sh = torch.randint(0, H - h + 1, (1,)).item()
    sw = torch.randint(0, W - w + 1, (1,)).item()
    x = x[:, :, sd:sd + d, sh:sh + h, sw:sw + w]
    pad_d = (D - d) // 2
    pad_h = (H - h) // 2
    pad_w = (W - w) // 2
    x = F.pad(x, (pad_w, W - w - pad_w, pad_h, H - h - pad_h, pad_d, D - d - pad_d))

    noise = torch.randn_like(x) * (noise_scale * strength)
    x = x + noise
    return x


def nt_xent(z1, z2, temperature=0.2):
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    batch_size = z1.shape[0]
    logits = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(batch_size, device=z1.device)
    return F.cross_entropy(logits, labels)


def supcon_loss(features, labels, temperature=0.2, class_weights=None):
    device = features.device
    bsz = features.shape[0]
    n_views = features.shape[1]
    labels = labels.contiguous().view(-1, 1)
    mask = torch.eq(labels, labels.T).float().to(device)

    contrast = torch.cat(torch.unbind(features, dim=1), dim=0)
    anchor = contrast

    logits = torch.div(torch.matmul(anchor, contrast.T), temperature)
    logits_max, _ = torch.max(logits, dim=1, keepdim=True)
    logits = logits - logits_max.detach()

    mask = mask.repeat(n_views, n_views)
    logits_mask = torch.ones_like(mask, device=device)
    logits_mask.scatter_(1, torch.arange(bsz * n_views, device=device).view(-1, 1), 0)
    mask = mask * logits_mask

    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)
    loss = -mean_log_prob_pos.view(n_views, bsz).mean(0)

    if class_weights is not None:
        # Ensure class_weights is on the same device as labels for indexing
        if class_weights.device != labels.device:
            class_weights = class_weights.to(labels.device)
        w = class_weights[labels.view(-1)]
        loss = loss * w
    return loss.mean()


class LabeledVolumeFolderDataset(torch.utils.data.Dataset):
    def __init__(self, label_roots: dict, label_map: dict, target_shape=(112, 112, 112), file_exts=(".nii", ".nii.gz")):
        import nibabel as nib
        self.nib = nib
        self.items = []
        self.target_shape = target_shape
        for lab_name, root in label_roots.items():
            y = label_map.get(lab_name, None)
            if y is None or not os.path.isdir(root):
                continue
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    if not fn.lower().endswith(file_exts):
                        continue
                    p = os.path.join(dirpath, fn)
                    if os.path.isfile(p):
                        self.items.append((p, y))
        if not self.items:
            raise RuntimeError("No labeled items found from label_roots.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        p, y = self.items[idx]
        img = self.nib.load(p)
        v = img.get_fdata(dtype=np.float32)
        v = ensure_3d_volume(v, reduce_strategy='first')
        v = zscore_normalize(v)
        v = center_pad_or_crop(v, self.target_shape)
        v = np.expand_dims(v, 0)
        ten = torch.from_numpy(v)
        return ten, torch.tensor(y, dtype=torch.long)


def _class_weights_from_labels(labels, num_classes):
    counts = np.zeros((num_classes,), dtype=np.float64)
    for y in labels:
        if 0 <= y < num_classes:
            counts[y] += 1.0
    counts = np.clip(counts, 1.0, None)
    w = counts.sum() / counts
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def _f1_from_cm(cm):
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    f1_macro = float(np.mean(f1))
    tp_sum = tp.sum()
    fp_sum = fp.sum()
    fn_sum = fn.sum()
    precision_micro = tp_sum / (tp_sum + fp_sum + 1e-12)
    recall_micro = tp_sum / (tp_sum + fn_sum + 1e-12)
    f1_micro = float(2 * precision_micro * recall_micro / (precision_micro + recall_micro + 1e-12))
    return f1_micro, f1_macro


def _recall_at_k(emb, labels, k):
    sims = emb @ emb.T
    np.fill_diagonal(sims, -1.0)
    idx = np.argpartition(-sims, kth=k-1, axis=1)[:, :k]
    hit = 0
    for i in range(labels.shape[0]):
        if np.any(labels[idx[i]] == labels[i]):
            hit += 1
    return float(hit / max(labels.shape[0], 1))


def _auc_binary(y_true, y_score):
    y_true = np.asarray(y_true, dtype=np.int32)
    y_score = np.asarray(y_score, dtype=np.float64)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    x = np.concatenate([neg, pos])
    order = np.argsort(x)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(order.size) + 1
    auc = (ranks[neg.size:].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size)
    return float(auc)


def evaluate_embeddings(model, loader, device, k_list, knn_k):
    model.eval()
    feats = []
    labels = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            z = model.forward_projection(x)
            z = F.normalize(z, dim=-1)
            feats.append(z.detach().cpu().numpy())
            labels.append(y.numpy())
    emb = np.concatenate(feats, axis=0)
    lab = np.concatenate(labels, axis=0)
    sims = emb @ emb.T
    np.fill_diagonal(sims, -1.0)
    nn_idx = np.argmax(sims, axis=1)
    preds = lab[nn_idx]
    num_classes = int(lab.max()) + 1 if lab.size > 0 else 0
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(lab, preds):
        cm[int(t), int(p)] += 1
    f1_micro, f1_macro = _f1_from_cm(cm) if num_classes > 0 else (float("nan"), float("nan"))
    recalls = {f"recall@{k}": _recall_at_k(emb, lab, k) for k in k_list}
    gmean = float(np.sqrt(np.prod(np.diag(cm) / (cm.sum(axis=1) + 1e-12)))) if num_classes > 1 else float("nan")
    auc = float("nan")
    if num_classes == 2:
        score = sims[:, lab == 1].mean(axis=1)
        auc = _auc_binary(lab, score)
    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "gmean": gmean,
        "auc": auc,
        "recall": recalls,
        "cm": cm.tolist(),
        "embeddings": emb,
        "labels": lab
    }


def main():
    """
    对比预训练入口
    改动原因：
        - 从 config 读取 contrastive.backbone，实现 MedMamba3D/SS3M 的选择。
        - 统一 AMP 新接口到 torch.amp.autocast / torch.amp.GradScaler，确保 PyTorch 2.4.0 下无弃用警告。
    """
    cfg = load_config(os.path.join(os.path.dirname(__file__), 'config.yaml'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

    use_labeled = bool(cfg.get('contrast', {}).get('use_labeled', False))
    target_shape = tuple(cfg['input']['shape_dhw'])
    label_map = cfg.get('dataset', {}).get('folder_label_map', {})
    if use_labeled:
        label_roots = cfg.get('paths', {}).get('train_label_roots', {})
        allowed = cfg.get('dataset', {}).get('allowed_labels', None)
        if allowed is not None:
            label_roots = {k: v for k, v in label_roots.items() if k in allowed}
            label_map = {name: idx for idx, name in enumerate(allowed)}
        ds = LabeledVolumeFolderDataset(label_roots, label_map, target_shape=target_shape)
    else:
        ds = MRIVolumeDataset(cfg['paths']['pretrain_manifest'], cfg['dataset']['pretrain_column'], target_shape=target_shape)

    sampler = None
    class_weights = None
    if use_labeled and bool(cfg.get('contrast', {}).get('balanced_sampler', True)):
        labels = [y for _, y in ds.items]
        num_classes = int(max(labels)) + 1 if labels else 2
        class_weights = _class_weights_from_labels(labels, num_classes)
        sample_weights = [float(class_weights[y]) for y in labels]
        sampler = torch.utils.data.WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    loader = DataLoader(ds, batch_size=cfg['contrast']['batch_size'], shuffle=(sampler is None), sampler=sampler, num_workers=cfg['dataset']['num_workers'], pin_memory=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, choices=["medmamba3d", "ss3m"], default="medmamba3d",
                        help="选择预训练骨干：medmamba3d（旧）或 ss3m（新）")
    args = parser.parse_args()
    backbone_choice = str(cfg.get('contrastive', {}).get('backbone', 'medmamba3d')).lower()
    in_channels = int(cfg.get('input', {}).get('in_channels', 1))
    if backbone_choice == "ss3m":
        embed_dim = int(cfg.get('contrastive', {}).get('embed_dim', cfg.get('ss3m', {}).get('embed_dim', 96)))
        depth = int(cfg.get('contrastive', {}).get('depth', cfg.get('ss3m', {}).get('depth', 4)))
        patch_size = tuple(cfg.get('contrastive', {}).get('patch_size', cfg.get('ss3m', {}).get('patch_size', (16, 16, 16))))
    else:
        embed_dim = int(cfg['model']['embed_dim'])
        depth = int(cfg['model']['depth'])
        patch_size = tuple(cfg['model']['patch_size'])
    proj_dim = int(cfg.get('contrast', {}).get('proj_dim', 128))
    proj_head = str(cfg.get('contrast', {}).get('proj_head', 'mlp2')).lower()

    if backbone_choice == "medmamba3d":
        model = MedMamba3D(in_chans=in_channels, embed_dim=embed_dim, depth=depth, patch_size=patch_size)
        model.projection = _build_proj_head(embed_dim, proj_dim, proj_head)
    else:
        model = MedMambaSS3MContrast(in_channels=in_channels, embed_dim=embed_dim, depth=depth,
                                     patch_size=patch_size, proj_dim=proj_dim)
        model.proj = _build_proj_head(embed_dim, proj_dim, proj_head)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['contrast']['lr'], weight_decay=cfg['contrast']['weight_decay'])
    # 改动：统一使用 torch.amp.GradScaler（CUDA 启用，CPU禁用）
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    os.makedirs(cfg['paths']['out_dir'], exist_ok=True)

    eval_every = int(cfg.get('contrast', {}).get('eval_every', 0))
    eval_knn_k = int(cfg.get('contrast', {}).get('eval_knn_k', 1))
    recall_k = cfg.get('contrast', {}).get('eval_recall_k', [1, 5, 10])
    minority_labels = set(cfg.get('contrast', {}).get('minority_labels', []))
    minority_strength = float(cfg.get('contrast', {}).get('minority_augment', 1.2))
    base_strength = float(cfg.get('contrast', {}).get('augment_strength', 1.0))
    noise_scale = float(cfg.get('contrast', {}).get('noise_scale', 0.02))
    crop_ratio = float(cfg.get('contrast', {}).get('crop_ratio', 0.9))
    mixup_prob = float(cfg.get('contrast', {}).get('mixup_prob', 0.0))
    mixup_alpha = float(cfg.get('contrast', {}).get('mixup_alpha', 0.2))
    loss_type = str(cfg.get('contrast', {}).get('loss_type', 'nt_xent'))
    loss_weight = float(cfg.get('contrast', {}).get('loss_weight', 1.0))
    best_metric = str(cfg.get('contrast', {}).get('best_metric', 'f1_macro'))
    metrics_path = os.path.join(cfg['paths']['out_dir'], "contrast_metrics.jsonl")

    val_loader = None
    best_score = None
    best_path = os.path.join(cfg['paths']['out_dir'], "contrast_best.pth")
    if use_labeled and eval_every > 0:
        val_roots = cfg.get('paths', {}).get('val_label_roots', {})
        allowed = cfg.get('dataset', {}).get('allowed_labels', None)
        if allowed is not None:
            val_roots = {k: v for k, v in val_roots.items() if k in allowed}
            label_map = {name: idx for idx, name in enumerate(allowed)}
        ds_val = LabeledVolumeFolderDataset(val_roots, label_map, target_shape=target_shape)
        val_loader = DataLoader(ds_val, batch_size=cfg['contrast']['batch_size'], shuffle=False, num_workers=cfg['dataset']['num_workers'], pin_memory=True)

    use_mae_init = bool(cfg.get('contrast', {}).get('use_mae_init', False))
    mae_pretrained_path = cfg.get('contrast', {}).get('mae_pretrained_path')
    if use_mae_init and mae_pretrained_path and os.path.isfile(mae_pretrained_path):
        _load_pretrained_partial(model, mae_pretrained_path)
    elif use_mae_init:
        print(f"[Contrast] MAE weights not found: {mae_pretrained_path}")

    for epoch in range(cfg['contrast']['epochs']):
        model.train()
        total = 0.0
        for batch in loader:
            if use_labeled:
                x, y = batch
            else:
                x, _ = batch
            x = x.to(device, non_blocking=True)
            if use_labeled:
                y = y.to(device)
                strength = torch.ones((x.shape[0],), device=device) * base_strength
                if minority_labels:
                    for name, idx in (label_map or {}).items():
                        if name in minority_labels:
                            strength = torch.where(y == idx, torch.tensor(minority_strength, device=device), strength)
                x1 = torch.stack([augment(x[i:i+1], strength=float(strength[i]), noise_scale=noise_scale, crop_ratio=crop_ratio).squeeze(0) for i in range(x.shape[0])], dim=0)
                x2 = torch.stack([augment(x[i:i+1], strength=float(strength[i]), noise_scale=noise_scale, crop_ratio=crop_ratio).squeeze(0) for i in range(x.shape[0])], dim=0)
            else:
                x1 = augment(x.clone(), strength=base_strength, noise_scale=noise_scale, crop_ratio=crop_ratio)
                x2 = augment(x.clone(), strength=base_strength, noise_scale=noise_scale, crop_ratio=crop_ratio)

            if mixup_prob > 0 and torch.rand(1).item() < mixup_prob:
                lam = np.random.beta(mixup_alpha, mixup_alpha)
                idx = torch.randperm(x1.size(0))
                x1 = lam * x1 + (1 - lam) * x1[idx]
                x2 = lam * x2 + (1 - lam) * x2[idx]

            # 改动：统一使用 torch.amp.autocast('cuda', enabled=...) 新接口
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                z1 = model.forward_projection(x1)
                z2 = model.forward_projection(x2)
                if use_labeled and loss_type == "supcon":
                    feats = torch.stack([F.normalize(z1, dim=-1), F.normalize(z2, dim=-1)], dim=1)
                    loss = supcon_loss(feats, y, temperature=cfg['contrast']['temperature'], class_weights=class_weights)
                else:
                    loss = nt_xent(z1, z2, temperature=cfg['contrast']['temperature'])
                loss = loss * loss_weight

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item()

        avg = total / max(len(loader), 1)
        print(f"[Contrast] Epoch {epoch+1}/{cfg['contrast']['epochs']} | loss={avg:.4f}")

        if val_loader is not None and (epoch + 1) % eval_every == 0:
            metrics = evaluate_embeddings(model, val_loader, device, recall_k, eval_knn_k)
            emb = metrics.pop("embeddings")
            lab = metrics.pop("labels")
            metrics.update({"epoch": int(epoch + 1), "loss": float(avg)})
            with open(metrics_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(metrics, ensure_ascii=False) + "\n")
            np.savez(os.path.join(cfg['paths']['out_dir'], f"contrast_emb_epoch_{epoch+1}.npz"), emb=emb, labels=lab)
            if best_metric.startswith("recall@"):
                score = metrics.get("recall", {}).get(best_metric)
            else:
                score = metrics.get(best_metric)
            if score is not None:
                if best_score is None or float(score) > best_score:
                    best_score = float(score)
                    torch.save(model.state_dict(), best_path)
                    print(f"[Contrast] best model saved -> {best_path} ({best_metric}={best_score:.4f})")
            if bool(cfg.get('contrast', {}).get('save_tsne', False)):
                try:
                    from sklearn.manifold import TSNE
                    max_n = int(cfg.get('contrast', {}).get('tsne_max_samples', 1000))
                    sel = np.random.choice(emb.shape[0], size=min(max_n, emb.shape[0]), replace=False)
                    tsne = TSNE(n_components=2, perplexity=float(cfg.get('contrast', {}).get('tsne_perplexity', 30.0)))
                    pts = tsne.fit_transform(emb[sel])
                    out = {"points": pts.tolist(), "labels": lab[sel].tolist()}
                    with open(os.path.join(cfg['paths']['out_dir'], f"contrast_tsne_epoch_{epoch+1}.json"), "w", encoding="utf-8") as f:
                        json.dump(out, f, ensure_ascii=False)
                except Exception:
                    pass

        if (epoch + 1) % cfg['contrast']['ckpt_every'] == 0:
            torch.save(model.state_dict(), os.path.join(cfg['paths']['out_dir'], f'contrast_epoch_{epoch+1}.pth'))

    print("Contrastive training done.")


if __name__ == "__main__":
    main()
