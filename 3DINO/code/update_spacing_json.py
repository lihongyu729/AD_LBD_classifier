#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
from typing import Any, Dict, List, Tuple, Union

def compute_spacing(image_path: str) -> Tuple[float, float, float]:
    """
    函数作用：
    - 读取 NIfTI 文件的头信息，尽可能稳健地计算 3D 体素间距 (spacing)。

    设计原因：
    - 下游数据增强依赖 'spacing'；源数据可能来自不同工具链，头信息的有效字段不完全一致，
      因此采用分层回退策略提升健壮性。

    实现细节：
    1) 优先使用 nibabel 的 header.get_zooms()（标准可靠）。
    2) 回退使用 NIfTI 头部的 pixdim[1:4]。
    3) 再次回退用仿射矩阵的对角绝对值（近似）。
    4) 若 nibabel 全部失败，尝试 SimpleITK 的 GetSpacing()。
    5) 仍失败时返回 (1.0, 1.0, 1.0) 作为默认值。
    """
    # 优先 nibabel
    try:
        import nibabel as nib
        img = nib.load(image_path)
        hdr = img.header
        # 1) get_zooms
        zo = hdr.get_zooms()
        if isinstance(zo, (list, tuple)) and len(zo) >= 3:
            return float(zo[0]), float(zo[1]), float(zo[2])
        # 2) pixdim 1:4
        pixdim = hdr.get('pixdim', None)
        if pixdim is not None and len(pixdim) >= 4:
            return float(pixdim[1]), float(pixdim[2]), float(pixdim[3])
        # 3) 仿射矩阵对角
        import numpy as np
        A = img.affine
        if A is not None:
            return float(abs(A[0, 0])), float(abs(A[1, 1])), float(abs(A[2, 2]))
    except Exception:
        pass

    # 尝试 SimpleITK 作为兜底
    try:
        import SimpleITK as sitk
        itk_img = sitk.ReadImage(image_path)
        sp = itk_img.GetSpacing()
        if isinstance(sp, (list, tuple)) and len(sp) >= 3:
            return float(sp[0]), float(sp[1]), float(sp[2])
    except Exception:
        pass

    # 最后的默认
    return (1.0, 1.0, 1.0)


def find_image_path(entry: Dict[str, Any], base_dir: str) -> Union[str, None]:
    """
    函数作用：
    - 在样本条目里尽可能找到影像文件路径（支持常见键：'image'、'path'、'img'、'file'），并解析为可访问的绝对/相对路径。

    设计原因：
    - 不同标注方案可能使用不同键名；此外路径可能是相对路径，需要拼接 base_dir。

    返回值：
    - 找到则返回解析后的路径字符串，否则返回 None。
    """
    candidate_keys = ["image", "path", "img", "file"]
    for k in candidate_keys:
        if k in entry and isinstance(entry[k], str) and entry[k]:
            p = entry[k]
            # 如果文件存在，直接返回
            if os.path.exists(p):
                return p
            # 如果是相对路径，拼接 base_dir
            jp = os.path.join(base_dir, p)
            if os.path.exists(jp):
                return jp
            # 如果是类似 Unix 的绝对路径，在 Windows 下也尝试原样返回
            if p.startswith("/") or p.startswith("\\"):
                # 无法验证存在时，仍然返回原路径，让 compute_spacing 自己尝试读取（如网络挂载）
                return p
    return None


def update_entry_spacing(entry: Dict[str, Any], base_dir: str) -> bool:
    """
    函数作用：
    - 为单个样本条目计算并写入 'spacing'，返回是否成功更新。

    设计原因：
    - 将路径解析和间距计算分离，便于复用与出错定位。

    返回值：
    - True 表示已写入 spacing；False 表示无法解析路径或计算失败。
    """
    image_path = find_image_path(entry, base_dir)
    if not image_path:
        return False
    spacing = compute_spacing(image_path)
    entry["spacing"] = [float(spacing[0]), float(spacing[1]), float(spacing[2])]
    return True


def update_json_structure(data: Union[List[Any], Dict[str, Any]], base_dir: str) -> Tuple[int, int]:
    """
    函数作用：
    - 遍历 JSON 数据结构（支持列表或字典），为每个样本条目添加/更新 'spacing'。

    设计原因：
    - 许多数据标注使用 {train: [...], val: [...]} 或单列表结构；统一处理更方便。

    返回值：
    - (total, updated) 元组，表示总条目数与成功更新的条目数。
    """
    total = 0
    updated = 0

    def process_list(lst: List[Any]):
        nonlocal total, updated
        for item in lst:
            # 只处理字典条目
            if isinstance(item, dict):
                total += 1
                if update_entry_spacing(item, base_dir):
                    updated += 1

    if isinstance(data, list):
        process_list(data)
    elif isinstance(data, dict):
        # 字典值为列表的情况
        for key, val in data.items():
            if isinstance(val, list):
                process_list(val)
            elif isinstance(val, dict):
                # 深层结构：再找里面的列表
                for sub_k, sub_v in val.items():
                    if isinstance(sub_v, list):
                        process_list(sub_v)

    return total, updated


def main():
    """
    函数作用：
    - 命令行入口：读取输入 JSON，批量生成/更新 spacing，并写出到指定输出路径。

    使用说明：
    - --input 指定原始 JSON 文件（例如 /webdav/MyData/MRI/data/3dino_cls_128.json）
    - --output 指定输出 JSON 文件（例如 /webdav/MyData/MRI/data/3dino_cls_128.fixed.json）
    - --base-dir 指定数据根目录（用于解析相对路径与补全），默认 /webdav/MyData/MRI/data
    """
    import argparse

    parser = argparse.ArgumentParser(description="为 3D MRI 数据集 JSON 批量注入体素间距 spacing")
    parser.add_argument("--input", required=True, help="输入 JSON 文件路径")
    parser.add_argument("--output", required=True, help="输出 JSON 文件路径（建议不要覆盖原文件）")
    parser.add_argument("--base-dir", default="/webdav/MyData/MRI/data", help="数据根目录，用于解析相对路径")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    total, updated = update_json_structure(data, args.base_dir)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"总条目：{total}，已写入 spacing：{updated}，输出文件：{args.output}")


if __name__ == "__main__":
    main()