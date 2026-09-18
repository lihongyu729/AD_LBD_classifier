import os
import shutil
import random
import json
from typing import List, Tuple, Dict

try:
    import nibabel as nib
except ImportError:
    nib = None


# 函数级注释：
# 该函数返回类别与标签的映射关系。默认采用 AD=0，LBD=1，MCI=2，
# 其中 MCI=2 与用户示例保持一致。如果需要调整标签数字，可在此处修改。
def get_class_to_label_map() -> Dict[str, int]:
    return {
        "AD": 0,
        "LBD": 1,
        "MCI": 2,
    }


# 函数级注释：
# 在给定的根目录下（如 /webdav/MyData/MRI/data/MRI_Preprocessed_1mm）递归收集 NIfTI 文件，
# 并依据类名（子目录名 AD/MCI/LBD）赋予标签。返回列表 [(绝对路径, 标签, 类名), ...]
def collect_files(root_dir: str, exts: Tuple[str, ...] = (".nii", ".nii.gz")) -> List[Tuple[str, int, str]]:
    class_to_label = get_class_to_label_map()
    samples: List[Tuple[str, int, str]] = []

    for class_name, label in class_to_label.items():
        class_dir = os.path.join(root_dir, class_name)
        if not os.path.isdir(class_dir):
            print(f"[WARN] 类目录不存在：{class_dir}，将跳过该类")
            continue
        for r, _, files in os.walk(class_dir):
            for fn in files:
                if fn.lower().endswith(exts):
                    full_path = os.path.join(r, fn)
                    samples.append((full_path, label, class_name))
    return samples


# 函数级注释：
# 将样本按类别分组，使用固定随机种子进行打乱，并按 8:1:1 比例分割为 train/val/test。
# 返回三个列表，每个列表包含 (path, label, class_name) 元组。
def stratified_split(
    samples: List[Tuple[str, int, str]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[Tuple[str, int, str]], List[Tuple[str, int, str]], List[Tuple[str, int, str]]]:
    random.seed(seed)
    # 按类别分组
    by_class: Dict[str, List[Tuple[str, int, str]]] = {}
    for p, l, c in samples:
        by_class.setdefault(c, []).append((p, l, c))

    train, val, test = [], [], []
    for c, group in by_class.items():
        random.shuffle(group)
        n = len(group)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val
        train.extend(group[:n_train])
        val.extend(group[n_train:n_train + n_val])
        test.extend(group[n_train + n_val:])
        print(f"[SPLIT] 类 {c}: 总数={n}, train={n_train}, val={n_val}, test={n_test}")
    return train, val, test


# 函数级注释：
# 将样本复制到目标拆分目录（/webdav/MyData/MRI/data/1mm_split）下，保持类内相对路径结构。
# 例如源为 /root/MCI/3T/Subject/t1_orig.nii，目标为 /dest/train/MCI/3T/Subject/t1_orig.nii。
# 返回新位置的样本列表（路径更新到目标位置）。
def copy_split(
    split_samples: List[Tuple[str, int, str]],
    src_root: str,
    dst_root: str,
    split_name: str,
) -> List[Tuple[str, int, str]]:
    moved_samples: List[Tuple[str, int, str]] = []
    for src_path, label, class_name in split_samples:
        # 计算相对于类别目录的相对路径，用于保持原有层级
        class_dir = os.path.join(src_root, class_name)
        try:
            rel_path = os.path.relpath(src_path, class_dir)
        except ValueError:
            # 不同盘符或路径异常时，退化为仅使用文件名
            rel_path = os.path.basename(src_path)

        dst_path = os.path.join(dst_root, split_name, class_name, rel_path)
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)
        moved_samples.append((dst_path, label, class_name))
    print(f"[COPY] {split_name}: 完成复制 {len(moved_samples)} 个样本")
    return moved_samples


# 函数级注释：
# 从 NIfTI 文件中读取 spacing（体素间距），返回三个维度的列表 [sx, sy, sz]。
# 若无法读取（nibabel 不可用或文件损坏），则回退为 [1.0, 1.0, 1.0]。
def read_spacing(nifti_path: str) -> List[float]:
    default = [1.0, 1.0, 1.0]
    if nib is None:
        return default
    try:
        img = nib.load(nifti_path)
        zooms = img.header.get_zooms()
        # 取前三个空间维度的间距
        spacing = list(map(float, zooms[:3]))
        # 如果某个值异常或为 0，则回退为默认值
        if len(spacing) != 3 or any([s <= 0 or not (s == s) for s in spacing]):  # s==s 过滤 NaN
            return default
        return spacing
    except Exception:
        return default


# 函数级注释：
# 根据样本列表生成 JSON 文件，格式为：
# { "image": "<绝对路径>", "label": <数字标签>, "spacing": [sx, sy, sz] }
# 输出到目标根目录下的 split_name.json，如 /dest/train.json。
def write_split_json(
    split_samples: List[Tuple[str, int, str]],
    dst_root: str,
    split_name: str,
    json_name: str = None,
) -> str:
    records = []
    for p, l, _ in split_samples:
        spacing = read_spacing(p)
        records.append({
            "image": p.replace("\\", "/"),  # 统一为 POSIX 风格，避免反斜杠影响解析
            "label": int(l),
            "spacing": spacing,
        })
    json_fname = json_name if json_name else f"{split_name}.json"
    json_path = os.path.join(dst_root, json_fname)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"[JSON] 写出 {split_name} -> {json_path}，样本数={len(records)}")
    return json_path


# 函数级注释：
# 主入口：执行收集、分层划分、复制与 JSON 生成。
# 输入：
# - src_root: 源数据根目录（包含 AD/MCI/LBD 子目录）
# - dst_root: 目标拆分根目录（将创建 train/val/test 及 JSON）
# 输出：
# - 返回生成的 JSON 文件路径字典，以便你后续检查或使用。
def main(src_root: str, dst_root: str) -> Dict[str, str]:
    print(f"[BEGIN] 源数据路径: {src_root}")
    print(f"[BEGIN] 目标拆分路径: {dst_root}")

    samples = collect_files(src_root)
    print(f"[COLLECT] 共收集样本数: {len(samples)}")
    if len(samples) == 0:
        raise RuntimeError("未在源目录下找到任何 .nii 或 .nii.gz 文件，请检查路径是否正确或类目录是否存在。")

    train_s, val_s, test_s = stratified_split(samples, train_ratio=0.8, val_ratio=0.1, seed=42)

    # 复制到目标目录并生成 JSON
    train_moved = copy_split(train_s, src_root, dst_root, "train")
    val_moved = copy_split(val_s, src_root, dst_root, "val")
    test_moved = copy_split(test_s, src_root, dst_root, "test")

    json_paths = {
        "train": write_split_json(train_moved, dst_root, "train", json_name="train.json"),
        "val": write_split_json(val_moved, dst_root, "val", json_name="val.json"),
        "test": write_split_json(test_moved, dst_root, "test", json_name="test.json"),
    }
    print(f"[DONE] 生成的 JSON 文件：{json_paths}")
    return json_paths


if __name__ == "__main__":
    # 源与目标路径按你的要求设置
    SRC_ROOT = "/webdav/MyData/MRI/data/MRI_Preprocessed_1mm"
    DST_ROOT = "/webdav/MyData/MRI/data/1mm_split"
    os.makedirs(DST_ROOT, exist_ok=True)
    main(SRC_ROOT, DST_ROOT)