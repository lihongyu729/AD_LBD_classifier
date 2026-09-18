import os
import json
import shutil
import random
from typing import List, Dict, Tuple
import SimpleITK as sitk

# 固定随机种子以保证可复现
random.seed(42)

# 路径配置（按你的要求固定）
SRC_ROOT = "/webdav/MyData/MRI/data/classifier"       # 原始数据根目录
SPLIT_ROOT = "/webdav/MyData/MRI/data/split"          # 划分后复制的目录
RESAMPLED_ROOT = "/webdav/MyData/MRI/data/split_1mm"  # 重采样后的目录
FINAL_JSON_PATH = "/webdav/MyData/MRI/data/split_1mm.json"  # 最终总 JSON 文件

# 类别与标签映射（与示例一致，MCI -> 2）
CLASS_LABELS = {"AD": 0, "LBD": 1, "MCI": 2}

# 仅处理 .nii（不兼容 .nii.gz）
VALID_EXT = ".nii"

# 目标重采样 spacing
TARGET_SPACING = (1.0, 1.0, 1.0)


def is_nii_file(file_path: str) -> bool:
    """
    函数作用：
        判断给定路径是否为 .nii 文件（大小写不敏感）。
    这样设计的原因：
        满足“强兼容性”查找要求，但按你的补充排除 .nii.gz，仅保留 .nii。

    参数：
        file_path: 文件路径字符串

    返回：
        True 表示是 .nii 文件；False 表示不是。
    """
    return file_path.lower().endswith(VALID_EXT)


def list_class_files(class_dir: str) -> List[Tuple[str, str]]:
    """
    函数作用：
        递归列出某个类别目录下所有 .nii 文件，返回 (绝对路径, 相对路径) 的列表。
    这样设计的原因：
        - 保留原相对层级，方便在 split 和 split_1mm 中构造对应层级；
        - 相对路径基于类别根目录，确保映射清晰。

    参数：
        class_dir: 某个类别的根目录（例如 /webdav/.../classifier/AD）

    返回：
        列表，元素为 (abs_path, rel_path_from_class_dir)
    """
    results = []
    for root, _, files in os.walk(class_dir):
        for fname in files:
            fpath = os.path.join(root, fname)
            if is_nii_file(fpath):
                rel = os.path.relpath(fpath, start=class_dir)
                results.append((fpath, rel))
    return results


def split_by_ratio(items: List[Tuple[str, str]], ratios=(0.8, 0.1, 0.1)) -> Dict[str, List[Tuple[str, str]]]:
    """
    函数作用：
        按给定比例将样本列表划分为 train/test/val 三个集合。
    这样设计的原因：
        - 分层（每类单独划分）+ 固定随机种子，保证类别比例一致和可复现；
        - 对小样本作兜底处理，避免集合过小或为空影响训练。

    参数：
        items: 列表，元素为 (abs_path, rel_path)
        ratios: 三元组，代表 train/test/val 的比例（默认 0.8/0.1/0.1）

    返回：
        字典，键为 "train"/"test"/"val"，值为对应的 (abs_path, rel_path) 列表。
    """
    total = len(items)
    # 小样本兜底策略
    if total == 0:
        return {"train": [], "test": [], "val": []}
    if total < 3:
        # 约定：全部放入 train；若你希望 n=2 时 train=1, test=1, val=0，可在此调整
        return {"train": items, "test": [], "val": []}

    # 打乱确保随机
    shuffled = items[:]
    random.shuffle(shuffled)

    test_count = int(total * ratios[1])
    val_count = int(total * ratios[2])
    train_count = total - test_count - val_count

    train_items = shuffled[:train_count]
    test_items = shuffled[train_count:train_count + test_count]
    val_items = shuffled[train_count + test_count:train_count + test_count + val_count]

    return {"train": train_items, "test": test_items, "val": val_items}


def safe_copy(src_path: str, dst_path: str) -> bool:
    """
    函数作用：
        安全复制文件到目标路径，自动创建中间目录。
    这样设计的原因：
        - 保证不破坏原始数据；
        - 保留原文件名与层级结构。

    参数：
        src_path: 源文件绝对路径
        dst_path: 目标文件绝对路径

    返回：
        True 表示复制成功；False 表示失败（并打印错误信息）。
    """
    try:
        # 如果目标已存在，直接跳过（满足你的恢复需求）
        if os.path.exists(dst_path):
            return True

        parent = os.path.dirname(dst_path)
        # 创建父目录
        os.makedirs(parent, exist_ok=True)
        # 再次校验父目录是否存在且为目录
        if not os.path.isdir(parent):
            print(f"[COPY ERROR] Parent is not a directory: {parent}")
            return False

        shutil.copy2(src_path, dst_path)
        return True
    except Exception as e:
        print(f"[COPY ERROR] {src_path} -> {dst_path} : {e}")
        return False


def copy_split(class_name: str, items_by_set: Dict[str, List[Tuple[str, str]]]) -> Dict[str, int]:
    """
    函数作用：
        将划分好的 (abs_path, rel_path) 文件复制到 SPLIT_ROOT 下的 train/test/val 目录中，
        结构为 SPLIT_ROOT/{subset}/{class_name}/{rel_path}。
    这样设计的原因：
        - 保留原始相对路径和文件名；
        - 与后续重采样输出结构一一对应，便于对齐。

    参数：
        class_name: 类别名称（AD/LBD/MCI）
        items_by_set: 划分结果字典，键为 "train"/"test"/"val"，值为列表 (abs_path, rel_path)

    返回：
        复制成功的计数字典，键为 "train"/"test"/"val"
    """
    counts = {"train": 0, "test": 0, "val": 0}
    for subset, items in items_by_set.items():
        for abs_path, rel_path in items:
            dst_path = os.path.join(SPLIT_ROOT, subset, class_name, rel_path)
            if safe_copy(abs_path, dst_path):
                counts[subset] += 1
    return counts


def resample_to_1mm(src_path: str, dst_path: str) -> bool:
    """
    函数作用：
        将 src_path 指向的 .nii 图像重采样至 1.0mm 等距，并保存到 dst_path。
    这样设计的原因：
        - 使用 SimpleITK 保留空间信息（方向/原点/仿射）；
        - 线性插值适用于 MRI 强度图像；
        - 保持原数据类型，避免不必要的类型变化。

    参数：
        src_path: 源 .nii 文件路径（来自 SPLIT_ROOT）
        dst_path: 目标 .nii 文件路径（写入 RESAMPLED_ROOT）

    返回：
        True 表示重采样并保存成功；False 表示失败（并打印错误信息）。
    """
    try:
        img = sitk.ReadImage(src_path)
        if img.GetDimension() != 3:
            print(f"[RESAMPLE SKIP] Non-3D image: {src_path}")
            return False

        original_spacing = img.GetSpacing()
        original_size = img.GetSize()
        original_direction = img.GetDirection()
        original_origin = img.GetOrigin()
        pixel_type = img.GetPixelID()

        # 计算新尺寸：round(size * spacing / new_spacing)
        new_size = [
            int(round(original_size[i] * (original_spacing[i] / TARGET_SPACING[i])))
            for i in range(3)
        ]

        resampled = sitk.Resample(
            img,
            new_size,
            sitk.Transform(),             # 恒等变换
            sitk.sitkLinear,              # 线性插值
            original_origin,
            TARGET_SPACING,
            original_direction,
            0.0,                          # 默认填充值
            pixel_type
        )

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        sitk.WriteImage(resampled, dst_path)
        return True
    except Exception as e:
        print(f"[RESAMPLE ERROR] {src_path} -> {dst_path} : {e}")
        return False
def resample_to_1mm(src_path: str, dst_path: str) -> bool:
    """
    函数作用：
        将 src_path 指向的 .nii 图像重采样至 1.0mm 等距，并保存到 dst_path；若目标已存在则跳过。
    这样设计的原因：
        - 满足“已处理文件跳过”的可恢复要求；
        - 使用 SimpleITK 保留空间信息（方向/原点/仿射）；
        - 线性插值适用于 MRI 强度图像；
        - 保持原数据类型，避免不必要的类型变化。

    参数：
        src_path: 源 .nii 文件路径（来自 SPLIT_ROOT）
        dst_path: 目标 .nii 文件路径（写入 RESAMPLED_ROOT）

    返回：
        True 表示重采样/跳过成功；False 表示失败（并打印错误信息）。
    """
    try:
        # 若目标已存在，直接跳过重采样
        if os.path.exists(dst_path):
            return True

        img = sitk.ReadImage(src_path)
        if img.GetDimension() != 3:
            print(f"[RESAMPLE SKIP] Non-3D image: {src_path}")
            return False

        original_spacing = img.GetSpacing()
        original_size = img.GetSize()
        original_direction = img.GetDirection()
        original_origin = img.GetOrigin()
        pixel_type = img.GetPixelID()

        # 计算新尺寸：round(size * spacing / new_spacing)
        new_size = [
            int(round(original_size[i] * (original_spacing[i] / TARGET_SPACING[i])))
            for i in range(3)
        ]

        resampled = sitk.Resample(
            img,
            new_size,
            sitk.Transform(),             # 恒等变换
            sitk.sitkLinear,              # 线性插值
            original_origin,
            TARGET_SPACING,
            original_direction,
            0.0,                          # 默认填充值
            pixel_type
        )

        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        sitk.WriteImage(resampled, dst_path)
        return True
    except Exception as e:
        print(f"[RESAMPLE ERROR] {src_path} -> {dst_path} : {e}")
        return False

def resample_split() -> Dict[str, int]:
    """
    函数作用：
        对 SPLIT_ROOT 下的 train/test/val 三个集合、三个类别中的所有 .nii 文件进行重采样，
        并写入 RESAMPLED_ROOT 对应结构。
    这样设计的原因：
        - 与划分目录结构保持一致，便于后续统一管理与生成 JSON。

    参数：
        无

    返回：
        成功重采样的计数字典，总计按 "train"/"test"/"val" 统计。
    """
    counts = {"train": 0, "test": 0, "val": 0}
    for subset in ["train", "test", "val"]:
        for class_name in CLASS_LABELS.keys():
            base_dir = os.path.join(SPLIT_ROOT, subset, class_name)
            if not os.path.isdir(base_dir):
                continue
            for root, _, files in os.walk(base_dir):
                for fname in files:
                    src_path = os.path.join(root, fname)
                    # 以完整路径判断扩展名，提升鲁棒性
                    if not is_nii_file(src_path):
                        continue
                    rel_path = os.path.relpath(src_path, start=base_dir)
                    dst_path = os.path.join(RESAMPLED_ROOT, subset, class_name, rel_path)
                    if resample_to_1mm(src_path, dst_path):
                        counts[subset] += 1
    return counts


def generate_final_json(json_path: str) -> int:
    """
    函数作用：
        遍历 RESAMPLED_ROOT 下的所有 .nii 文件（train/test/val 三个集合与三个类别），
        生成一个总的 JSON 文件，条目格式严格为：
            {
                "image": "<绝对路径>",
                "label": <整数标签>,
                "spacing": [1.0, 1.0, 1.0]
            }
    这样设计的原因：
        - 满足你“产出一个总文件”的要求；
        - 不包含 subset 字段，与你给出的示例一致。

    参数：
        json_path: 输出 JSON 的绝对路径（例如 /webdav/.../split_1mm.json）

    返回：
        写入的条目数量。
    """
    entries = []
    for subset in ["train", "test", "val"]:
        for class_name, label in CLASS_LABELS.items():
            base_dir = os.path.join(RESAMPLED_ROOT, subset, class_name)
            if not os.path.isdir(base_dir):
                continue
            for root, _, files in os.walk(base_dir):
                for fname in files:
                    if not is_nii_file(fname):
                        continue
                    abs_path = os.path.join(root, fname)
                    entries.append({
                        "image": abs_path.replace("\\", "/"),  # 统一为正斜杠风格
                        "label": label,
                        "spacing": [1.0, 1.0, 1.0]
                    })
    # 写入 JSON（数组形式）
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False, indent=2)
    print(f"[JSON] Written {len(entries)} entries to {json_path}")
    return len(entries)


def summarize_original_counts() -> Dict[str, int]:
    """
    函数作用：
        统计原始数据根目录下三个类别的 .nii 文件数量以及总数。
    这样设计的原因：
        - 为数据划分前后提供对照与审计信息。

    参数：
        无

    返回：
        字典：{"AD": n_ad, "LBD": n_lbd, "MCI": n_mci, "total": total}
    """
    stats = {}
    total = 0
    for class_name in CLASS_LABELS.keys():
        class_dir = os.path.join(SRC_ROOT, class_name)
        files = list_class_files(class_dir)
        stats[class_name] = len(files)
        total += len(files)
    stats["total"] = total
    return stats


def main():
    """
    函数作用：
        主流程：
        1) 统计原始 .nii 数量；
        2) 三个类别分别按 8:1:1 划分并复制到 split；
        3) 对 split 进行 1mm 重采样并写入 split_1mm；
        4) 生成总 JSON 文件 split_1mm.json；
        5) 打印并保存简单汇总信息。
    这样设计的原因：
        - 将流程串联，易于一次性执行；
        - 提供清晰的阶段性统计输出，便于核对。
    """
    # 1) 原始统计
    orig_stats = summarize_original_counts()
    print("[ORIG] Counts:", orig_stats)

    # 2) 划分并复制
    split_counts_total = {"train": 0, "test": 0, "val": 0}
    for class_name in CLASS_LABELS.keys():
        class_dir = os.path.join(SRC_ROOT, class_name)
        files = list_class_files(class_dir)  # (abs_path, rel_path)
        items_by_set = split_by_ratio(files, ratios=(0.8, 0.1, 0.1))
        copied_counts = copy_split(class_name, items_by_set)
        print(f"[SPLIT COPY] {class_name} ->", copied_counts)
        for k in split_counts_total.keys():
            split_counts_total[k] += copied_counts[k]
    print("[SPLIT COPY TOTAL] ->", split_counts_total)

    # 3) 重采样
    resampled_counts = resample_split()
    print("[RESAMPLED TOTAL] ->", resampled_counts)

    # 4) 生成总 JSON
    json_entries = generate_final_json(FINAL_JSON_PATH)

    # 5) 输出汇总到文本文件（可选）
    summary_path = "/webdav/MyData/MRI/data/split_summary.txt"
    try:
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("Original counts:\n")
            f.write(json.dumps(orig_stats, ensure_ascii=False, indent=2) + "\n\n")
            f.write("Split copy counts:\n")
            f.write(json.dumps(split_counts_total, ensure_ascii=False, indent=2) + "\n\n")
            f.write("Resampled counts:\n")
            f.write(json.dumps(resampled_counts, ensure_ascii=False, indent=2) + "\n\n")
            f.write(f"Final JSON entries: {json_entries}\n")
        print(f"[SUMMARY] Written to {summary_path}")
    except Exception as e:
        print(f"[SUMMARY ERROR] {e}")


if __name__ == "__main__":
    main()