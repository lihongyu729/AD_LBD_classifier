import os
import json
import math
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple
import nibabel as nib

def create_dataset_json_for_3dino(root_dir, output_json_path, class_mapping, train_split=0.8, val_split=0.1):
    """
    为 3DINO 微调准备数据集 JSON 文件。

    参数:
    - root_dir (str): 数据集的根目录，其下应包含每个类别的子目录。
    - output_json_path (str): 输出 JSON 文件的路径。
    - class_mapping (dict): 类别名称到整数标签的映射，例如 {'health': 0, 'patient': 1}。
    - train_split (float): 训练集所占比例。
    - val_split (float): 验证集所占比例。测试集比例将自动计算。
    """
    dataset_files = []
    print(f"正在扫描根目录: {root_dir}")

    for class_name, label in class_mapping.items():
        class_dir = Path(root_dir) / class_name
        if not class_dir.is_dir():
            print(f"警告：未找到类别目录 {class_dir}，将跳过。")
            continue

        print(f"正在处理类别 '{class_name}' (标签: {label})，路径: {class_dir}")
        count = 0
        for file_path in class_dir.rglob('*.nii.gz'):
            dataset_files.append({
                "image": str(file_path.resolve()),  # 使用绝对路径
                "label": label
            })
            count += 1
        print(f" -> 找到 {count} 个 .nii.gz 文件。")

    if not dataset_files:
        raise ValueError("在指定目录下未找到任何 .nii.gz 文件。请检查 root_dir 和目录结构。")

    print(f"\n总共找到 {len(dataset_files)} 个样本。")
    random.shuffle(dataset_files)

    # 计算分割点
    total_size = len(dataset_files)
    train_end = int(total_size * train_split)
    val_end = train_end + int(total_size * val_split)

    # 分割数据集
    train_files = dataset_files[:train_end]
    val_files = dataset_files[train_end:val_end]
    test_files = dataset_files[val_end:]

    final_structure = {
        "training": train_files,
        "validation": val_files,
        "test": test_files
    }

    # 保存为 JSON 文件
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(final_structure, f, indent=4)

    print(f"\n数据集 JSON 文件已成功生成！")
    print(f"  - 路径: {output_json_path}")
    print(f"  - 训练集样本数: {len(train_files)}")
    print(f"  - 验证集样本数: {len(val_files)}")
    print(f"  - 测试集样本数: {len(test_files)}")


def _list_nii_recursively(root_dir: Path, exts: Tuple[str, str] = (".nii", ".nii.gz")) -> List[Path]:
    """
    函数作用：
        递归遍历 root_dir 下的所有子目录，收集扩展名为 .nii/.nii.gz 的医学影像文件。

    设计原因：
        - 您的数据并不是直接存放在类别目录下，而是存在多级子目录。
        - 使用递归搜索确保不会漏掉深层次的文件。

    参数：
        root_dir: 起始目录（类别目录）
        exts: 允许的扩展名集合

    返回：
        影像文件的绝对路径列表（Path）
    """
    files: List[Path] = []
    for f in root_dir.rglob("*"):
        if f.is_file() and f.suffix.lower() in exts:
            files.append(f.resolve())
    return files


def _compute_split_counts(n: int, ratios: Tuple[float, float, float]) -> Tuple[int, int, int]:
    """
    函数作用：
        根据总样本数 n 和 (train, val, test) 比例，计算每个 split 的样本数，保证和为 n。

    设计原因：
        - 按比例划分时使用向下取整，可能导致总数少于 n；剩余部分补给 train，确保总数一致。
        - 对样本量很小的类别，可能会出现 val/test 为 0 的情况，这是合理的；我们会提示统计信息。

    参数：
        n: 总样本数
        ratios: 三元组，表示 train/val/test 的比例，例如 (0.7, 0.15, 0.15)

    返回：
        (train_count, val_count, test_count)
    """
    t_ratio, v_ratio, s_ratio = ratios
    t = math.floor(n * t_ratio)
    v = math.floor(n * v_ratio)
    s = math.floor(n * s_ratio)
    # 补足剩余到 train
    remainder = n - (t + v + s)
    t += remainder
    return t, v, s


def _materialize_items(
    items: List[Path],
    dst_dir: Path,
    link_mode: str = "symlink",
) -> None:
    """
    函数作用：
        将给定的文件列表“落地”到目标目录下（按照原文件名），支持三种模式：
        - symlink: 创建软链接（推荐，节省空间）
        - hardlink: 创建硬链接（同一文件系统内更稳固，但不跨分区）
        - copy: 直接复制文件（最兼容，但占用空间）

    设计原因：
        - 医学影像体积较大，软/硬链接能显著节省空间与时间。
        - 不同系统/挂载盘可能对链接有限制，因此提供多模式选择。

    参数：
        items: 源文件路径列表
        dst_dir: 目标目录（不存在会创建）
        link_mode: 选择 "symlink" | "hardlink" | "copy"
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in items:
        dst = dst_dir / src.name
        if dst.exists():
            continue
        try:
            if link_mode == "symlink":
                os.symlink(src, dst)
            elif link_mode == "hardlink":
                os.link(src, dst)
            elif link_mode == "copy":
                shutil.copy2(src, dst)
            else:
                raise ValueError(f"不支持的 link_mode: {link_mode}")
        except Exception as e:
            # 链接失败时回退为复制，提高鲁棒性
            shutil.copy2(src, dst)


def split_and_organize_dataset(
    src_root: str,
    out_root: str,
    ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    link_mode: str = "symlink",
    exts: Tuple[str, str] = (".nii", ".nii.gz"),
) -> Dict[str, Dict[str, int]]:
    """
    函数作用：
        从 src_root（例如 /webdav/MyData/MRI/data/classifier）下自动识别类别目录，
        递归收集各类别的影像文件，按比例随机划分为 train/val/test，
        并将样本“落地”为标准目录结构：out_root/{train,val,test}/{class_name}/file.nii[.gz]

    设计原因：
        - 统一输出目录结构，便于 3DINO 等训练管线直接使用。
        - 提供随机种子，保证划分的可复现性（论文复现实验要求）。

    参数：
        src_root: 源数据根目录（包含 MCI/AD/LBD 等类别子目录）
        out_root: 输出根目录（将创建 train/val/test 结构）
        ratios: 划分比例（train/val/test）
        seed: 随机种子
        link_mode: 落地模式（"symlink"|"hardlink"|"copy"）
        exts: 允许的扩展名

    返回：
        每个 split 下每个类的样本计数统计，用于打印与审查：
        {
          "train": {"MCI": 123, "AD": 100, "LBD": 98},
          "validation": {...},
          "test": {...}
        }
    """
    random.seed(seed)
    src = Path(src_root).resolve()
    out = Path(out_root).resolve()
    out.mkdir(parents=True, exist_ok=True)

    # 自动识别类别（一级子目录名即类别）
    class_dirs = [d for d in src.iterdir() if d.is_dir()]
    if not class_dirs:
        raise RuntimeError(f"在 {src_root} 下未发现任何类别目录。")

    # 准备统计
    stats: Dict[str, Dict[str, int]] = {"training": {}, "validation": {}, "test": {}}

    for class_dir in class_dirs:
        class_name = class_dir.name
        files = _list_nii_recursively(class_dir, exts=exts)
        if not files:
            print(f"[警告] 类别 {class_name} 未找到任何 .nii/.nii.gz 文件，跳过。")
            stats["training"][class_name] = 0
            stats["validation"][class_name] = 0
            stats["test"][class_name] = 0
            continue

        # 随机打乱（可复现）
        random.shuffle(files)

        # 按比例划分
        t_count, v_count, s_count = _compute_split_counts(len(files), ratios)
        train_items = files[:t_count]
        val_items = files[t_count:t_count + v_count]
        test_items = files[t_count + v_count:t_count + v_count + s_count]

        # 落地到 out_root
        _materialize_items(train_items, out / "train" / class_name, link_mode=link_mode)
        _materialize_items(val_items, out / "val" / class_name, link_mode=link_mode)
        _materialize_items(test_items, out / "test" / class_name, link_mode=link_mode)

        # 统计
        stats["training"][class_name] = len(train_items)
        stats["validation"][class_name] = len(val_items)
        stats["test"][class_name] = len(test_items)

    # 打印汇总信息
    print("划分完成，样本统计如下：")
    for split_key, human_key in [("training", "train"), ("validation", "val"), ("test", "test")]:
        print(f"[{human_key}]")
        for cls, cnt in stats[split_key].items():
            print(f"  {cls}: {cnt} 样本")

    return stats


def build_3dino_classification_json(
    split_root: str,
    output_json: str,
    exts: Tuple[str, str] = (".nii", ".nii.gz"),
) -> None:
    """
    函数作用：
        将已“落地”的标准目录结构（split_root/train|val|test/{class}/file）
        转换为 3DINO 微调需要的 JSON 格式：
        {
          "training": [{"image": "/abs/path.nii.gz", "label": 0}, ...],
          "validation": [...],
          "test": [...]
        }

    设计原因：
        - 3DINO 分类微调需要整数标签；我们按字典序构建类名到ID的映射，保证可重复性。
        - 读取已落地结构可避免重复逻辑，保证样本路径与 JSON 一致。

    参数：
        split_root: 标准目录结构的根目录
        output_json: 输出 JSON 的路径
        exts: 允许的扩展名
    """
    root = Path(split_root).resolve()
    split_map = {"train": "training", "val": "validation", "test": "test"}

    # 收集所有类别名
    class_names = set()
    for s in ["train", "val", "test"]:
        s_dir = root / s
        if not s_dir.exists():
            continue
        for class_dir in s_dir.iterdir():
            if class_dir.is_dir():
                class_names.add(class_dir.name)
    if not class_names:
        raise RuntimeError(f"未在 {split_root} 下发现任何类别目录。")

    # 按字典序确定标签ID
    class_to_id = {name: idx for idx, name in enumerate(sorted(class_names))}
    print(f"类别ID映射（字典序）：{class_to_id}")

    def collect_split(split: str) -> List[Dict]:
        items: List[Dict] = []
        s_dir = root / split
        if not s_dir.exists():
            return items
        for class_dir in s_dir.iterdir():
            if not class_dir.is_dir():
                continue
            label_id = class_to_id[class_dir.name]
            for f in class_dir.iterdir():
                if f.is_file() and f.suffix.lower() in exts:
                    items.append({"image": str(f.resolve()), "label": int(label_id)})
        return items

    out: Dict[str, List[Dict]] = {
        "training": collect_split("train"),
        "validation": collect_split("val"),
        "test": collect_split("test"),
    }

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    # 打印数量，帮助检查
    for k, v in out.items():
        print(f"{k}: {len(v)} 样本")


def get_nifti_shape_spacing(image_path: str):
    """
    函数作用：
        读取单个 NIfTI 影像，返回其空间维度的形状（shape）与体素间距（spacing）。

    为什么这样设计：
        - 医学影像可能是 3D 或 4D（多时间点/多通道），我们仅提取前 3 个空间维度，确保与 3D 变换兼容。
        - 体素间距从头信息 zooms 中获取，保证与数据集的真实空间分辨率一致。

    返回：
        (shape_list, spacing_list)，例如：([128, 128, 64], [0.5, 0.5, 1.0])
        若读取失败，返回 (None, None)，由调用方决定是否跳过写入。
    """
    try:
        img = nib.load(image_path)
        shape = list(img.shape[:3])
        zooms = list(img.header.get_zooms()[:3])
        # 防御：有些数据的 zooms 可能缺失或异常
        if not shape or len(shape) != 3:
            return None, None
        if not zooms or len(zooms) != 3:
            return shape, None
        return shape, zooms
    except Exception:
        return None, None


if __name__ == '__main__':
    """
    主入口作用：
        1) 按比例划分 /webdav/MyData/MRI/data/classifier 下的 MCI/AD/LBD 多级目录数据
        2) 生成标准目录结构到 /webdav/MyData/MRI/data/classifier_splits
        3) 输出 3DINO 分类微调所需 JSON 到 /webdav/MyData/MRI/data/3dino_cls.json

    为什么默认使用软链接：
        - 速度快、节省存储；如遇文件系统或权限限制，会自动回退为复制，提高鲁棒性。

    您可以按需调整比例、随机种子与 link_mode。
    """
    # 您的实际路径（Linux）
    SRC_ROOT = "/webdav/MyData/MRI/data/classifier"           # 原始类别根目录（嵌套子目录）
    OUT_ROOT = "/webdav/MyData/MRI/data/classifier_splits"    # 标准目录结构输出位置
    OUTPUT_JSON = "/webdav/MyData/MRI/data/3dino_cls.json"    # 3DINO 分类 JSON 输出位置

    # 1) 划分与落地
    split_and_organize_dataset(
        src_root=SRC_ROOT,
        out_root=OUT_ROOT,
        ratios=(0.7, 0.15, 0.15),
        seed=42,
        link_mode="symlink",  # 可改为 "hardlink" 或 "copy"
        exts=(".nii", ".nii.gz"),
    )

    # 2) 生成 3DINO 分类 JSON
    build_3dino_classification_json(
        split_root=OUT_ROOT,
        output_json=OUTPUT_JSON,
        exts=(".nii", ".nii.gz"),
    )