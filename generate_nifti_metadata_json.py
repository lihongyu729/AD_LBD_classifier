import argparse
import json
import logging
import os
from typing import Dict, List, Tuple, Any


def setup_logging() -> None:
    """
    函数用途：
        - 初始化日志输出格式与级别，便于批处理过程记录进度与异常。
    设计原因与作用：
        - 批量扫描与读取 NIfTI 文件时易遇到损坏文件或路径错误，
          统一的日志格式便于定位问题与统计处理结果。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )


def validate_paths(input_dir: str, output_json: str) -> None:
    """
    函数用途：
        - 校验输入目录与输出 JSON 路径的合法性。
    设计原因与作用：
        - 保障批处理前置条件满足，避免因路径错误导致空输出或运行中断。
    """
    if not input_dir or not isinstance(input_dir, str):
        raise ValueError("输入目录为空或不是字符串。")
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"输入目录不存在或不可访问：{input_dir}")
    if not output_json or not isinstance(output_json, str):
        raise ValueError("输出 JSON 路径为空或不是字符串。")
    if not output_json.lower().endswith(".json"):
        raise ValueError(f"输出路径必须以 .json 结尾：{output_json}")
    out_dir = os.path.dirname(output_json)
    if out_dir and not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok=True)


def collect_nifti_files(input_dir: str) -> List[str]:
    """
    函数用途：
        - 递归扫描输入目录，收集所有 .nii.gz 文件路径。
    设计原因与作用：
        - 适配深层目录结构，确保批处理覆盖全部数据。
    """
    files: List[str] = []
    for root, _, filenames in os.walk(input_dir):
        for name in filenames:
            if name.lower().endswith(".nii.gz"):
                full_path = os.path.join(root, name)
                if os.path.isfile(full_path):
                    files.append(full_path)
    files.sort()
    return files


def get_label_from_path(nifti_path: str, label_map: Dict[str, int]) -> int:
    """
    函数用途：
        - 从文件路径中识别类别目录并映射为标签。
    设计原因与作用：
        - 数据集目录通常包含 AD/LBD/MCI 等类别子目录，
          通过路径解析即可生成训练所需 label。
    """
    parts = [p.lower() for p in nifti_path.replace("\\", "/").split("/")]
    for name, label in label_map.items():
        if name.lower() in parts:
            return int(label)
    raise ValueError(f"无法从路径中识别类别目录: {nifti_path}")


def minimal_read_check(img: Any) -> None:
    """
    函数用途：
        - 对 NIfTI 数据做最小读写触发，提前发现损坏文件。
    设计原因与作用：
        - nibabel 仅加载头部不一定触发数据校验；最小切片读取可更稳健地发现损坏文件。
    """
    import numpy as np

    shape = getattr(img, "shape", None)
    if not shape:
        raise ValueError("无法获取 NIfTI shape。")
    slices = tuple(slice(0, 1) for _ in shape)
    _ = np.asanyarray(img.dataobj[slices])


def extract_metadata(nifti_path: str, label_map: Dict[str, int]) -> Dict[str, Any]:
    """
    函数用途：
        - 读取单个 NIfTI 文件的元信息并结构化返回。
    设计原因与作用：
        - 输出与现有数据清单一致的字段（image/label/spacing），便于直接复用。
    """
    import nibabel as nib

    img = nib.load(nifti_path)
    minimal_read_check(img)
    header = img.header
    shape = tuple(int(x) for x in img.shape)
    zooms = header.get_zooms()
    spacing = [float(v) for v in zooms[: len(shape)]]
    label = get_label_from_path(nifti_path, label_map)
    return {
        "image": os.path.abspath(nifti_path).replace("\\", "/"),
        "label": label,
        "spacing": spacing,
    }


def validate_metadata_list(records: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
    """
    函数用途：
        - 验证元数据列表结构是否满足 JSON 输出约束。
    设计原因与作用：
        - 在写盘前进行结构校验，保证输出 JSON 可被下游稳定解析。
    """
    errors: List[str] = []
    if not isinstance(records, list):
        return False, ["顶层数据不是列表。"]
    required_keys = {"image", "label", "spacing"}
    for idx, item in enumerate(records):
        if not isinstance(item, dict):
            errors.append(f"第 {idx} 项不是对象。")
            continue
        missing = required_keys - set(item.keys())
        if missing:
            errors.append(f"第 {idx} 项缺少字段: {sorted(missing)}")
    return len(errors) == 0, errors


def write_json(output_json: str, records: List[Dict[str, Any]]) -> None:
    """
    函数用途：
        - 将元数据列表写入 JSON 文件。
    设计原因与作用：
        - 统一输出格式，便于复现与后续程序使用。
    """
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def verify_output(output_json: str, expected_count: int) -> None:
    """
    函数用途：
        - 读取已生成 JSON 文件并验证结构与条目数。
    设计原因与作用：
        - 提供写盘后的二次校验，确保输出可解析且包含全部处理结果。
    """
    with open(output_json, "r", encoding="utf-8") as f:
        data = json.load(f)
    ok, errors = validate_metadata_list(data)
    if not ok:
        raise ValueError("输出 JSON 结构校验失败: " + "; ".join(errors))
    if len(data) != expected_count:
        raise ValueError(f"输出条目数不一致：期望 {expected_count}，实际 {len(data)}")


def main() -> None:
    """
    函数用途：
        - 解析命令行参数，批量读取 NIfTI 文件并生成 JSON 元数据。
    设计原因与作用：
        - 统一入口便于复用与自动化运行，支持自定义输入目录与输出路径。
    """
    parser = argparse.ArgumentParser(description="批量生成 NIfTI 元数据 JSON")
    parser.add_argument(
        "--input-dir",
        default="~/split_1mm_112",
        help="待扫描的 .nii.gz 根目录",
    )
    parser.add_argument(
        "--output-json",
        default="~/MRI/code/split_1mm_122.json",
        help="输出 JSON 文件路径",
    )
    args = parser.parse_args()

    setup_logging()
    validate_paths(args.input_dir, args.output_json)

    try:
        import nibabel
    except Exception as exc:
        raise RuntimeError("未检测到 nibabel，请先安装：pip install nibabel") from exc

    files = collect_nifti_files(args.input_dir)
    if not files:
        raise RuntimeError(f"在目录中未找到 .nii.gz 文件：{args.input_dir}")

    logging.info(f"发现 .nii.gz 文件数: {len(files)}")
    label_map = {"LBD": 0, "AD": 1, "MCI": 2}
    records: List[Dict[str, Any]] = []
    failed: List[str] = []
    for idx, path in enumerate(files, start=1):
        try:
            metadata = extract_metadata(path, label_map)
            records.append(metadata)
            logging.info(f"[{idx}/{len(files)}] 成功: {path}")
        except Exception as exc:
            failed.append(path)
            logging.error(f"[{idx}/{len(files)}] 失败: {path}，原因: {exc}")

    if not records:
        raise RuntimeError("所有文件均读取失败，未生成任何元数据。")

    ok, errors = validate_metadata_list(records)
    if not ok:
        raise ValueError("元数据结构校验失败: " + "; ".join(errors))

    write_json(args.output_json, records)
    verify_output(args.output_json, len(records))

    logging.info(f"写出 JSON: {args.output_json}")
    logging.info(f"成功条目数: {len(records)}")
    if failed:
        logging.warning(f"失败文件数: {len(failed)}")


if __name__ == "__main__":
    main()
