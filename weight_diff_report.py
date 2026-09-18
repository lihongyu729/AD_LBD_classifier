import argparse
import json
import os
import yaml
import torch

from medmamba_ss3m import MedMambaSS3M
from medmamba3d import MedMamba3D


def _load_state(ckpt_path):
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict):
        if isinstance(state.get("state_dict"), dict):
            state = state["state_dict"]
        elif isinstance(state.get("model_state"), dict):
            state = state["model_state"]
        elif isinstance(state.get("model"), dict):
            state = state["model"]
    return state


def _build_model(cfg, backbone_choice, in_chans, num_classes):
    if backbone_choice == "medmamba3d":
        return MedMamba3D(
            in_chans=in_chans,
            embed_dim=int(cfg.get("model", {}).get("embed_dim", 128)),
            depth=int(cfg.get("model", {}).get("depth", 12)),
            patch_size=tuple(cfg.get("model", {}).get("patch_size", (16, 16, 16))),
            num_classes=num_classes
        )
    return MedMambaSS3M(
        in_channels=in_chans,
        embed_dim=int(cfg.get("ss3m", {}).get("embed_dim", 96)),
        depth=int(cfg.get("ss3m", {}).get("depth", 4)),
        patch_size=tuple(cfg.get("ss3m", {}).get("patch_size", (16, 16, 16))),
        num_classes=num_classes,
        dropout=float(cfg.get("ss3m", {}).get("dropout", 0.0))
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--in-chans", type=int, default=1)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ckpt = cfg.get("classifier", {}).get("pretrained_path")
    backbone = str(cfg.get("classifier", {}).get("backbone", "medmamba3d")).lower()
    report = {"checkpoint": ckpt, "backbone": backbone}
    if not ckpt or not os.path.isfile(ckpt):
        report["error"] = "checkpoint_not_found"
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    model = _build_model(cfg, backbone, args.in_chans, args.num_classes)
    state = _load_state(ckpt)

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

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    miss = list(missing)
    unexp = list(unexpected)
    by_prefix = {}
    for k in miss:
        pref = k.split(".")[0]
        by_prefix[pref] = by_prefix.get(pref, 0) + 1

    report.update({
        "loaded_keys": len(new_state),
        "missing_count": len(miss),
        "unexpected_count": len(unexp),
        "missing_by_prefix": by_prefix,
        "missing_keys": miss,
        "unexpected_keys": unexp
    })
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
