import csv
import os
import random
from typing import List, Optional, Tuple, Union

import nibabel as nib
import numpy as np
import torch
import scipy.ndimage
from torch.utils.data import Dataset


def apply_elastic_transform(vol: np.ndarray, alpha: float = 15.0, sigma: float = 3.0) -> np.ndarray:
    """
    Apply 3D elastic deformation to a volume.
    
    Args:
        vol: 3D numpy array (D, H, W) or (C, D, H, W).
        alpha: Scaling factor for deformation field intensity.
        sigma: Smoothing factor for Gaussian filter (controls elasticity).
    
    Returns:
        Deformed volume with same shape.
    """
    # Ensure working with 3D array for coordinates
    is_channel_first = (vol.ndim == 4)
    if is_channel_first:
        c, d, h, w = vol.shape
        shape = (d, h, w)
    else:
        shape = vol.shape
    
    # Generate random displacement fields
    dx = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dz = scipy.ndimage.gaussian_filter((np.random.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    # Create meshgrid of coordinates
    x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]), indexing='ij')
    indices = np.reshape(x+dx, (-1, 1)), np.reshape(y+dy, (-1, 1)), np.reshape(z+dz, (-1, 1))

    if is_channel_first:
        out = np.zeros_like(vol)
        for i in range(c):
            # Map coordinates for each channel
            out[i] = scipy.ndimage.map_coordinates(vol[i], indices, order=1, mode='reflect').reshape(shape)
        return out
    else:
        return scipy.ndimage.map_coordinates(vol, indices, order=1, mode='reflect').reshape(shape)


def zscore_normalize(vol: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    # 鲁棒截断 + Z-score
    v = vol.astype(np.float32)
    p1, p99 = np.percentile(v, 1.0), np.percentile(v, 99.0)
    v = np.clip(v, p1, p99)
    m, s = v.mean(), v.std() + eps
    return (v - m) / s


def center_pad_or_crop(vol: np.ndarray, target_shape: Tuple[int, int, int]) -> np.ndarray:
    # 仅做中心pad/crop到目标尺寸 (D,H,W)=(192,256,256)
    D, H, W = vol.shape
    tD, tH, tW = target_shape
    out = vol

    # pad
    pad_d = max(tD - D, 0)
    pad_h = max(tH - H, 0)
    pad_w = max(tW - W, 0)
    if pad_d or pad_h or pad_w:
        out = np.pad(out,
                     ((pad_d // 2, pad_d - pad_d // 2),
                      (pad_h // 2, pad_h - pad_h // 2),
                      (pad_w // 2, pad_w - pad_w // 2)),
                     mode='constant', constant_values=0)

    # crop
    D, H, W = out.shape
    sD = (D - tD) // 2 if D > tD else 0
    sH = (H - tH) // 2 if H > tH else 0
    sW = (W - tW) // 2 if W > tW else 0
    out = out[sD:sD + tD, sH:sH + tH, sW:sW + tW]
    return out


def ensure_3d_volume(vol: np.ndarray, reduce_strategy: str = 'first') -> np.ndarray:
    """
    函数用途：
        - 将任意维度（≥3D）的 NIfTI 数组规范化为 3D (D, H, W)，以便后续中心 pad/crop。
    设计原因与作用：
        - 避免 center_pad_or_crop 在 4D/5D 数据上解包失败；
        - 默认 'first' 选择额外维度的第一个切片以避免额外计算；如需更鲁棒可用 'mean'。
    参数：
        - vol: 从 nibabel 读取的数组，可能是 3D/4D/带单例维度的 5D
        - reduce_strategy: 'first' 或 'mean'，决定在存在额外维度时如何降维
    返回：
        - 3D 的体素数组，形状 (D, H, W)
    """
    v = np.squeeze(vol)  # 去除所有单例维度
    if v.ndim == 3:
        return v.astype(np.float32, copy=False)
    if v.ndim > 3:
        # 约定额外维度位于最后（常见布局 [D, H, W, T]）；否则逐步降维
        if reduce_strategy == 'mean':
            v = v.mean(axis=-1)
        else:
            v = v[..., 0]
        v = np.squeeze(v)
        while v.ndim > 3:
            v = np.take(v, indices=0, axis=-1)
            v = np.squeeze(v)
    return v.astype(np.float32, copy=False)


# 顶部新增：递归扫描 NIfTI 文件的辅助函数
def scan_nii_files(root: str, file_exts: tuple = (".nii", ".nii.gz"), require_name_substring: Optional[str] = None) -> List[str]:
    """
    函数用途：
    - 递归遍历 root 下的所有子目录，收集后缀为 .nii/.nii.gz 的文件路径；
      若设置 require_name_substring，则仅保留文件名包含该子串的路径。

    设计原因与作用：
    - 适配“深层目录结构+通用文件命名”的数据组织；
    - 提供可选的子串过滤以提升扫描效率或限定文件类型；
    - 作为通用工具，避免在数据集初始化中冗长的遍历逻辑。

    参数说明：
    - root: 递归扫描的根目录
    - file_exts: 允许的文件后缀集合
    - require_name_substring: 可选的文件名子串过滤（None 表示不过滤）

    返回：
    - 满足条件的 NIfTI 文件绝对路径列表
    """
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            low = fn.lower()
            if not low.endswith(file_exts):
                continue
            if require_name_substring and (require_name_substring not in low):
                continue
            p = os.path.join(dirpath, fn)
            if os.path.isfile(p):
                paths.append(p)
    return paths


def apply_augmentations(vol: np.ndarray) -> np.ndarray:
    """
    随机数据增强组合：
    1. 随机翻转（轴0, 1, 2）
    2. 随机旋转90度（轴1-2）
    3. 随机高斯噪声
    4. 随机Gamma变换
    """
    # 随机翻转
    if random.random() < 0.5:
        vol = np.flip(vol, axis=0)
    if random.random() < 0.5:
        vol = np.flip(vol, axis=1)
    if random.random() < 0.5:
        vol = np.flip(vol, axis=2)
    
    # 随机旋转 (XY平面旋转)
    if random.random() < 0.5:
        k = random.randint(1, 3)
        vol = np.rot90(vol, k=k, axes=(1, 2))
    
    # 随机高斯噪声
    if random.random() < 0.3:
        noise = np.random.normal(0, 0.05, vol.shape)
        vol = vol + noise
        
    # 随机Gamma变换 (亮度/对比度)
    if random.random() < 0.3:
        gamma = random.uniform(0.7, 1.3)
        vol_min, vol_max = vol.min(), vol.max()
        vol_norm = (vol - vol_min) / (vol_max - vol_min + 1e-6)
        vol = np.power(vol_norm, gamma) * (vol_max - vol_min) + vol_min
        
    return vol.copy()


class MRIVolumeDataset(Dataset):
    """
    函数用途：
        - 无标签体积数据集，支持两种输入：
          1) 传入 CSV 路径（按列名获取 NIfTI 路径）；
          2) 传入目录路径（递归扫描 .nii/.nii.gz）。
    设计原因与作用：
        - 目录模式扩展为“匹配所有 NIfTI 后缀”，适配深层目录与通用命名；
        - validate_nifti 控制初始化时是否读取头进行轻量验证，以权衡速度与鲁棒性；
        - require_name_substring 可用于只采集包含特定关键字的文件名（需要时开启）。
        - 新增对非 3D 文件的跳过与统计：挤压单例维度后仅保留 3D，跳过真实 4D/5D 并统计数量，以避免 3D 预处理出错并满足你的“先跳过 4D”需求。
        - 新增 augment 参数：支持训练时数据增强。
        注意：
        - 你当前完全不使用 CSV，本类会自动走“目录模式”（os.path.isdir(manifest_csv) 为 True）。
    """
    def __init__(self, manifest_csv: str, path_column: str, target_shape: Tuple[int, int, int] = (112, 112, 112), check_spacing: bool = True, validate_nifti: bool = True, file_exts: tuple = (".nii", ".nii.gz"), require_name_substring: Optional[str] = None, augment: bool = False, normalized: bool = False):
        """
        函数说明：
        - 支持目录扫描或CSV两种模式加载数据（NIfTI 或 .npy）。
        - 修复：对“.nii.gz”不再进行“头信息字节数 vs 实际文件大小”的错误一致性校验，仅做轻量体素读取验证，避免误判压缩文件为损坏。
        - normalized=True：输入已是（脑区）Z-score 归一化（如 .npy 预卷），加载时跳过 zscore_normalize。
        """
        super().__init__()
        self.target_shape = target_shape
        self.check_spacing = check_spacing
        self.augment = augment
        self.normalized = normalized
        self.samples: List[str] = []
        # 新增：记录“已打印过的损坏/不可读文件路径”，确保只打印一次
        self.reported_bad_paths = set()
        # 统计非 3D 的跳过数量（初始化阶段）
        self.skipped_non3d_count: int = 0
        # 新增：统计读入阶段（__getitem__）因损坏/IO错误而跳过的数量
        self.skipped_bad_io_count: int = 0

        if os.path.isdir(manifest_csv):
            # 目录模式：递归扫描所有 .nii/.nii.gz；可选子串过滤；可选验证
            candidate_paths = scan_nii_files(manifest_csv, file_exts=file_exts, require_name_substring=require_name_substring)
            bad_paths = []
            for p in candidate_paths:
                try:
                    size_ok = os.path.getsize(p) > 0
                    if not size_ok:
                        bad_paths.append(p)
                        continue
                    if p.lower().endswith(".npy"):
                        # .npy 预卷：直接 np.load 校验形状（3D 或 (1,D,H,W)）
                        arr = np.load(p)
                        eff_shape = tuple(d for d in arr.shape if d != 1)
                        if len(eff_shape) != 3:
                            self.skipped_non3d_count += 1
                            continue
                        self.samples.append(p)
                        continue
                    # 读取头部信息（不触发完整数据载入）
                    img = nib.load(p)
                    shape = img.header.get_data_shape()
                    eff_shape = tuple(d for d in shape if d != 1)
                    eff_ndim = len(eff_shape)
                    if eff_ndim != 3:
                        self.skipped_non3d_count += 1
                        continue

                    if validate_nifti:
                        # 修复点：.nii.gz 为压缩文件，实际文件大小会远小于未压缩体素字节数，不能用文件大小做一致性校验
                        is_gz = p.lower().endswith(".nii.gz")
                        if not is_gz:
                            try:
                                dtype = img.get_data_dtype()
                                offset = int(float(img.header.get("vox_offset", 0)))
                                expected = offset + int(np.prod(shape)) * np.dtype(dtype).itemsize
                                actual = os.path.getsize(p)
                                if actual < expected:
                                    bad_paths.append(p)
                                    continue
                            except Exception:
                                bad_paths.append(p)
                                continue

                        # 轻量读一个体素，进一步确认可读性（适用于 .nii 和 .nii.gz）
                        try:
                            idx = tuple(0 for _ in shape)
                            _ = img.dataobj[idx]
                        except Exception:
                            bad_paths.append(p)
                            continue

                    self.samples.append(p)
                except Exception:
                    bad_paths.append(p)
            if bad_paths:
                print(f"[MRIVolumeDataset] Skipped {len(bad_paths)} invalid files (empty or unreadable).")
            if self.skipped_non3d_count:
                print(f"[MRIVolumeDataset] Skipped {self.skipped_non3d_count} non-3D files (e.g., true 4D/5D).")
        else:
            # CSV 模式（保留以兼容，但你当前不会使用到）
            with open(manifest_csv, 'r', newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    p = row.get(path_column, '')
                    if p and os.path.isfile(p):
                        try:
                            size_ok = os.path.getsize(p) > 0
                            if not size_ok:
                                continue
                            if validate_nifti:
                                try:
                                    img = nib.load(p)
                                    _ = np.asanyarray(img.dataobj[0, 0, 0])  # 读一个体素验证
                                except Exception:
                                    continue
                            self.samples.append(p)
                        except Exception:
                            continue

        if not self.samples:
            raise RuntimeError(
                f"No valid NIfTI paths found. "
                f"{'Scanned directory: ' + manifest_csv if os.path.isdir(manifest_csv) else f'Column {path_column} in {manifest_csv}'}"
            )


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path = self.samples[idx]
        try:
            if path.lower().endswith(".npy"):
                # .npy 预卷（已归一化，无 header/zooms）
                vol = np.load(path).astype(np.float32)
            else:
                img = nib.load(path)
                vol = img.get_fdata(dtype=np.float32)
                if self.check_spacing:
                    zooms = img.header.get_zooms()[:3]
                    if not (abs(zooms[0] - 1.0) < 0.05 and abs(zooms[1] - 1.0) < 0.05 and abs(zooms[2] - 1.0) < 0.05):
                        pass
            vol = ensure_3d_volume(vol, reduce_strategy='first')
            if not self.normalized:
                vol = zscore_normalize(vol)
            vol = center_pad_or_crop(vol, self.target_shape)
            
            # Apply augmentation if enabled
            if self.augment:
                if random.random() < 0.3:
                    vol = apply_elastic_transform(vol)
                vol = apply_augmentations(vol)
                
            vol = np.expand_dims(vol, 0)
            ten = torch.from_numpy(vol)
            return ten, path
        except (OSError, ValueError, RuntimeError) as e:
            self.skipped_bad_io_count += 1
            if path not in self.reported_bad_paths:
                print(f"[MRIVolumeDataset] 跳过损坏或不可读文件: {path} ({type(e).__name__}: {e})")
                self.reported_bad_paths.add(path)
            return None  # 返回 None 而不是重试
        except Exception as e:
            self.skipped_bad_io_count += 1
            if path not in self.reported_bad_paths:
                print(f"[MRIVolumeDataset] 跳过异常文件: {path} ({type(e).__name__}: {e})")
                self.reported_bad_paths.add(path)
            return None  # 返回 None 而不是重试

def collate_fn_skip_none(batch):
    """
    一个 collate_fn，它会过滤掉批次中值为 None 的样本。
    这对于处理在 __getitem__ 中可能返回 None 的数据集（例如，当文件损坏时）很有用。
    """
    batch = [item for item in batch if item is not None]
    if not batch:
        return None, None  # 如果整个批次都是无效的，返回 None
    return torch.utils.data.dataloader.default_collate(batch)