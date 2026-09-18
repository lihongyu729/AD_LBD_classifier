import os
import sys
from typing import Dict, List, Tuple

import nibabel as nib


def scan_nii_files(root: str, file_exts: Tuple[str, ...] = (".nii", ".nii.gz")) -> List[str]:
    """
    函数用途：
        - 递归扫描 root 下所有子目录，收集后缀为 .nii/.nii.gz 的文件路径。
    设计原因与作用：
        - 适配深层目录结构，确保不遗漏任意层级中的 NIfTI 文件。
    参数说明：
        - root: 递归扫描的根目录
        - file_exts: 允许的文件后缀集合
    返回：
        - 满足条件的 NIfTI 文件绝对路径列表
    """
    paths: List[str] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            low = fn.lower()
            if low.endswith(file_exts):
                p = os.path.join(dirpath, fn)
                if os.path.isfile(p):
                    paths.append(p)
    return paths


def get_nii_shape(path: str) -> Tuple[int, Tuple[int, ...]]:
    """
    函数用途：
        - 加载 NIfTI 文件并返回维度数和形状（不加载全体数据）。
    设计原因与作用：
        - 通过 nib.load 获取 header 与 shape，即可判断维度，避免 get_fdata 的高内存读取。
    参数说明：
        - path: NIfTI 文件绝对路径
    返回：
        - (ndim, shape) 二元组；若读取失败，返回 (0, ()) 表示异常
    """
    try:
        img = nib.load(path)
        shape = tuple(img.shape)
        ndim = len(shape)
        return ndim, shape
    except Exception:
        return 0, ()


def summarize_dimensions(paths: List[str]) -> Dict[str, int]:
    """
    函数用途：
        - 对给定 NIfTI 路径列表按维度进行统计，输出 3D/4D/5D/其他/损坏 的计数。
    设计原因与作用：
        - 快速了解目录中各维度数据的分布情况，为数据预处理策略（如降维）提供依据。
    参数说明：
        - paths: NIfTI 文件路径列表
    返回：
        - 维度分类计数字典，键包括：'3d', '4d', '5d', 'other', 'broken', 'total'
    """
    counts = {"3d": 0, "4d": 0, "5d": 0, "other": 0, "broken": 0, "total": len(paths)}
    for p in paths:
        ndim, shape = get_nii_shape(p)
        if ndim == 0:
            counts["broken"] += 1
        elif ndim == 3:
            counts["3d"] += 1
        elif ndim == 4:
            counts["4d"] += 1
        elif ndim == 5:
            counts["5d"] += 1
        else:
            counts["other"] += 1
    return counts


def main():
    """
    函数用途：
        - 作为脚本入口：解析目录路径，执行扫描与统计，并打印结果。
    设计原因与作用：
        - 提供默认目录 E:\\AD-MRI\\preSelect，也支持命令行传入其他目录，便于复用。
    """
    default_root = r"E:\AD-MRI\preSelect"
    root = sys.argv[1] if len(sys.argv) >= 2 else default_root

    print(f"[Scan] 目标目录: {root}")
    if not os.path.isdir(root):
        print("[Error] 目录不存在或不可访问，请检查路径。")
        return

    paths = scan_nii_files(root)
    print(f"[Scan] 发现 NIfTI 文件数: {len(paths)}")
    counts = summarize_dimensions(paths)

    print("[Result] 维度统计：")
    print(f"  3D:     {counts['3d']}")
    print(f"  4D:     {counts['4d']}")
    print(f"  5D:     {counts['5d']}")
    print(f"  其他:    {counts['other']}")
    print(f"  损坏/不可读: {counts['broken']}")
    print(f"  总计:    {counts['total']}")


if __name__ == "__main__":
    main()