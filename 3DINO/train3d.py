def _resolve_dataset_path(cfg):
    
    """
    函数作用：
    - 统一解析数据清单路径：优先使用 `cfg.DATASET.dataset_path`，如未设置则回退到 `cfg.TRAIN.dataset_path`（或 `cfg.train.dataset_path`）。
    - 在解析后立即进行存在性校验，确保返回的路径真实存在，避免在下游 `make_dataset_3d` 处抛出 FileNotFoundError。

    设计原因：
    - 你当前的错误显示代码仍在读取历史默认路径（/home/txu/.../dinov2_subset_datalist.json），说明路径解析存在“多源”且优先级不明确。
    - 通过入口统一解析与校验，强制使用正确的 JSON（比如 /webdav/MyData/MRI/data/3dino_cls.json），彻底消除歧义。

    返回：
    - 字符串，指向存在的 JSON 文件路径。

    异常：
    - 若两个位置都为空或文件不存在，抛出 FileNotFoundError，并明确提示两个字段当前值以及修复建议。
    """
    import os

    # 收集候选路径（优先 DATASET，再回退 TRAIN/train）
    candidates = []
    try:
        if hasattr(cfg, "DATASET") and getattr(cfg.DATASET, "dataset_path", None):
            candidates.append(cfg.DATASET.dataset_path)
    except Exception:
        pass

    # 有的配置对象字段为 TRAIN（大写），有的为 train（小写），都尝试
    try:
        if hasattr(cfg, "TRAIN") and getattr(cfg.TRAIN, "dataset_path", None):
            candidates.append(cfg.TRAIN.dataset_path)
    except Exception:
        pass
    try:
        if hasattr(cfg, "train") and getattr(cfg.train, "dataset_path", None):
            candidates.append(cfg.train.dataset_path)
    except Exception:
        pass

    # 选择第一个非空候选，并校验存在性
    for p in candidates:
        if p and os.path.isfile(p):
            return p

    # 构造详细错误信息，提示两个字段当前值
    msg_lines = [
        "未找到有效的数据清单文件，请在配置 YAML 中统一设置存在的 JSON 路径（建议两个位置都指向同一文件）。",
        f"DATASET.dataset_path: {getattr(getattr(cfg, 'DATASET', object()), 'dataset_path', None)}",
        f"TRAIN.dataset_path: {getattr(getattr(cfg, 'TRAIN', object()), 'dataset_path', None)}",
        f"train.dataset_path: {getattr(getattr(cfg, 'train', object()), 'dataset_path', None)}",
        "修复建议：将以上字段统一为 /webdav/MyData/MRI/data/3dino_cls.json（或你实际的 JSON）。"
    ]
    raise FileNotFoundError("\n".join(msg_lines))

# ... existing code ...

def do_train(cfg, model, resume=True):
    # ... existing code ...
    # 显式解析并校验数据清单路径（原因：入口统一控制，避免读取旧值）
    dataset_path = _resolve_dataset_path(cfg)

    # 将解析出的路径传给数据集构建函数
    dataset = make_dataset_3d(
        dataset_path=dataset_path,
        # ... existing code ...
    )
    # ... existing code ...