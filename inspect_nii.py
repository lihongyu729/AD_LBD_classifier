import os
import sys
from typing import Tuple

import nibabel as nib
import numpy as np


def bytes_to_human(nbytes: int) -> str:
    """
    函数用途：
        - 将字节数转换为更友好的单位显示（KB/MB/GB）。
    设计原因与作用：
        - 打印文件与数据估算大小时更直观，便于判断是否安全读取全体数据。
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if nbytes < 1024.0:
            return f"{nbytes:.2f} {unit}"
        nbytes /= 1024.0
    return f"{nbytes:.2f} PB"


def summarize_affine(aff: np.ndarray) -> str:
    """
    函数用途：
        - 将 4x4 仿射矩阵格式化为字符串，便于人读。
    设计原因与作用：
        - 直观展示体素到世界坐标的映射，快速发现异常（如非正交或奇异矩阵）。
    """
    with np.printoptions(precision=4, suppress=True):
        return str(aff)


def print_nii_info(path: str, compute_stats_threshold: int = 256 * 256 * 256) -> None:
    """
    函数用途：
        - 加载 NIfTI 文件并详细打印维度、头信息、体素间距、数据类型、仿射矩阵等；
        - 若体素总数较小（低于安全阈值），计算全局强度统计（min/max/mean）。
    设计原因与作用：
        - 在不加载整幅体的前提下获取关键元数据；
        - 为定位 3D/4D/5D 维度问题提供依据，指导 ensure_3d_volume 的降维策略。
    参数说明：
        - path: NIfTI 文件绝对路径
        - compute_stats_threshold: 当体素总数低于该阈值时，执行全体数据统计（避免内存峰值）
    """
    print(f"[Info] NIfTI path: {path}")
    if not os.path.exists(path):
        print("[Error] 文件不存在，请检查路径。")
        return
    size_bytes = os.path.getsize(path)
    print(f"[Info] 文件大小: {bytes_to_human(size_bytes)}")

    img = nib.load(path)
    print(f"[Info] 对象类型: {type(img).__name__}")
    shape = img.shape
    ndim = len(shape)
    print(f"[Info] 体素形状 (shape): {shape}  |  维度数 (ndim): {ndim}")

    hdr = img.header
    zooms = hdr.get_zooms()[:min(3, len(hdr.get_zooms()))]
    dtype = hdr.get_data_dtype()
    print(f"[Info] 体素间距 (zooms): {zooms}")
    print(f"[Info] 存储数据类型 (header dtype): {dtype}")
    print(f"[Info] 仿射矩阵 (affine):\n{summarize_affine(img.affine)}")

    # 尝试读取一个体素进行轻量验证
    try:
        if ndim >= 3:
            sample_idx = (0, 0, 0) if ndim == 3 else (0, 0, 0, 0)
            sample_val = np.asanyarray(img.dataobj[sample_idx])
            print(f"[Info] 读取一个体素以验证可读性: idx={sample_idx}, value={sample_val}")
        else:
            print("[Warn] 维度数 < 3，可能不是期望的体数据。")
    except Exception as e:
        print(f"[Error] 轻量体素读取失败: {e}")

    # 估算体素总数与内存占用
    try:
        total_voxels = int(np.prod(shape))
        bytes_per_voxel = np.dtype(dtype).itemsize
        est_nbytes = total_voxels * bytes_per_voxel
        print(f"[Info] 体素总数估算: {total_voxels}  |  原始字节估算: {bytes_to_human(est_nbytes)}")
    except Exception as e:
        print(f"[Warn] 无法估算体素总数/字节数: {e}")

    # 若体较小，计算全局统计（min/max/mean）；否则跳过以节省内存
    if ndim >= 3 and (np.prod(shape) <= compute_stats_threshold):
        try:
            vol = img.get_fdata(dtype=np.float32)
            vmin, vmax, vmean = float(vol.min()), float(vol.max()), float(vol.mean())
            print(f"[Stats] 全局强度: min={vmin:.4f}, max={vmax:.4f}, mean={vmean:.4f}")
        except Exception as e:
            print(f"[Warn] 读取全体数据进行统计失败: {e}")
    else:
        print("[Info] 跳过全局强度统计（体素过多或维度异常），如需强制统计可调高 compute_stats_threshold。")

    # 针对维度的额外提示
    if ndim == 3:
        print("[Hint] 这是标准 3D 体 (D,H,W)；可以直接送入 center_pad_or_crop。")
    elif ndim == 4:
        print("[Hint] 这是 4D 体 (D,H,W,T 或 C)；为避免报错，应在数据集阶段降维到 3D（例如选择第一时刻/通道或做均值）。")
    elif ndim > 4:
        print("[Hint] 这是 >=5D 的体，通常包含多重单例或扩展维；建议先 squeeze 再沿最后维逐步降到 3D。")


def main():
    """
    函数用途：
        - 作为脚本入口，从命令行解析 NIfTI 路径（缺省使用你提供的示例路径），调用检查函数。
    设计原因与作用：
        - 方便你快速对特定文件做维度与头信息审计，辅助后续数据集/训练脚本修复。
    """
    default_path = r"d:\py_project\MRI\1.3.12.2.1107.5.2.19.45255.2018122013211713953460238.0.0.0.nii"
    path = sys.argv[1] if len(sys.argv) >= 2 else default_path
    print_nii_info(path)


if __name__ == "__main__":
    main()