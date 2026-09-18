# Author: Tony Xu
#
# This code is adapted from the original DINOv2 repository: https://github.com/facebookresearch/dinov2
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

import logging
import random

from torchvision import transforms
from monai.transforms import (
    Crop,
    Randomizable,
    RandFlip,
    Compose,
    RandRotate90,
    OneOf,
    RandAdjustContrast,
    RandGaussianSharpen,
    RandGaussianSmooth,
    RandGaussianNoise,
    RandHistogramShift,
    RandGibbsNoise,
    CropForeground,
    ToTensor,
)
from monai.transforms.compose import get_seed
import monai.transforms.compose as monai_compose
from monai.transforms.intensity.array import GibbsNoise
from monai.data.utils import get_random_patch, get_valid_patch_size
from torch.nn.functional import interpolate
import math
import numpy as np


logger = logging.getLogger("dinov2")


# --- Monkey Patch for MONAI Randomizable.set_random_state ---
# 问题描述：MONAI 的 set_random_state 内部使用 seed % MAX_SEED，若 seed 为 numpy.uint32 且 MAX_SEED 为 2**32 (int)，
# numpy 会尝试将 int 转换为 uint32 进行运算，导致 OverflowError。
# 解决方案：拦截 set_random_state 调用，强制将 seed 转换为 Python 原生 int 类型，避免 numpy 类型参与运算。
_original_set_random_state = Randomizable.set_random_state

def _patched_set_random_state(self, seed=None, state=None):
    if seed is not None:
        try:
            # 强制转换为 Python int，脱离 numpy 类型系统
            seed = int(seed)
        except (TypeError, ValueError):
            pass
    return _original_set_random_state(self, seed=seed, state=state)

Randomizable.set_random_state = _patched_set_random_state
logger.info("已应用 Randomizable.set_random_state 补丁以修复 uint32 溢出问题。")
# --- End Monkey Patch ---

# --- Monkey Patch for Torch/Torchvision NumPy Compatibility (NumPy 2.x) ---
# 问题描述：Torchvision 0.15+ (及旧版 PyTorch) 无法兼容 NumPy 2.x，导致 `RuntimeError: Numpy is not available`。
# 解决方案：拦截 torch.tensor / torch.as_tensor 调用时对 numpy 的检测，或者直接在代码入口处回退 numpy 版本（不现实）。
# 这里采用更激进的补丁：在导入 augmentations 时，尝试修复 numpy 的部分兼容性，或者捕获该错误并提供友好提示。
# 实际上，最彻底的解决是让用户降级 numpy。但在无法控制环境时，我们尝试 patch monai.utils.type_conversion.dtype_torch_to_numpy
# 使得它不再依赖 torch.empty().numpy().dtype 这种可能触发 "Numpy is not available" 的调用。

# 针对 "RuntimeError: Numpy is not available" 的具体 patch：
# 错误发生在 monai.utils.type_conversion.dtype_torch_to_numpy -> torch.empty([], dtype=dtype).numpy().dtype
# 如果 torch 认为 numpy 不可用（因为 numpy 2.0 兼容性问题），.numpy() 方法会报错。
# 我们拦截 dtype_torch_to_numpy，手动映射常见 torch dtype 到 numpy dtype。
import monai.utils.type_conversion as monai_type_conversion
import torch
_original_dtype_torch_to_numpy = monai_type_conversion.dtype_torch_to_numpy

def _patched_dtype_torch_to_numpy(dtype):
    try:
        return _original_dtype_torch_to_numpy(dtype)
    except RuntimeError as e:
        if "Numpy is not available" in str(e):
            # 手动映射常见类型，规避 torch.numpy() 调用
            mapping = {
                torch.float32: np.float32,
                torch.float64: np.float64,
                torch.float16: np.float16,
                torch.int32: np.int32,
                torch.int64: np.int64,
                torch.int16: np.int16,
                torch.int8: np.int8,
                torch.uint8: np.uint8,
                torch.bool: np.bool_,
            }
            if dtype in mapping:
                return mapping[dtype]
            # 如果映射表中没有，尝试用字符串匹配或回退
            return np.dtype(str(dtype).replace("torch.", ""))
        raise e

monai_type_conversion.dtype_torch_to_numpy = _patched_dtype_torch_to_numpy
logger.info("已应用 monai.utils.type_conversion.dtype_torch_to_numpy 补丁以修复 NumPy 2.0 兼容性问题。")

# --- Monkey Patch for MONAI type_conversion.convert_to_tensor ---
# 问题描述：在某些 PyTorch 版本中，torch.as_tensor 无法直接推断 numpy.bool_ 标量的 dtype，导致 RuntimeError。
# RandGibbsNoise -> GibbsNoise._apply_mask -> convert_to_tensor(mask) 会触发此问题。
# 解决方案：拦截 convert_to_tensor，若输入数据为 bool 类型，强制转换为 uint8，确保 torch 能正确处理。
import monai.utils.type_conversion as monai_type_conversion
import torch
_original_convert_to_tensor = monai_type_conversion.convert_to_tensor

def _patched_convert_to_tensor(data, dtype=None, device=None, wrap_sequence=False, **kwargs):
    # 针对 numpy.bool_ 标量或 bool 标量，以及 bool 类型的 numpy 数组进行预处理
    if isinstance(data, (np.bool_, bool)):
        data = int(data) # 转换为 0 或 1
        if dtype is None:
            dtype = torch.uint8 # 默认转为 uint8 tensor
    elif isinstance(data, np.ndarray) and data.dtype == np.bool_:
        data = data.astype(np.uint8) # 转换为 uint8 数组
        if dtype is None:
            dtype = torch.uint8

    return _original_convert_to_tensor(data, dtype=dtype, device=device, wrap_sequence=wrap_sequence, **kwargs)

monai_type_conversion.convert_to_tensor = _patched_convert_to_tensor
logger.info("已应用 monai.utils.type_conversion.convert_to_tensor 补丁以修复 numpy.bool 类型推断问题。")
# --- End Monkey Patch ---


def _normalize_seed(seed):
    """
    函数作用：
    - 将输入的随机种子归一化到 uint32 的合法范围 [0, 2**32 - 1]。

    设计原因：
    - MONAI 的 Randomizable 仅接受 uint32；当上游生成 2**32 会触发溢出异常。
    - 通过显式的取模与兜底随机数生成，避免 set_random_state 调用时越界。

    参数：
    - seed: 任意类型的种子输入（None、int、float、numpy 整数等）。

    返回：
    - int：归一化后的 uint32 种子。
    """
    if seed is None:
        return random.getrandbits(32)
    try:
        seed_int = int(seed)
    except (TypeError, ValueError, OverflowError):
        return random.getrandbits(32)
    return seed_int % (2 ** 32)


def _mask_seed_uint32(seed_int):
    """
    函数作用：
    - 使用位掩码将整数种子限制在 uint32 范围内。

    设计原因：
    - get_seed() 可能返回 2**32，导致 MONAI 内部 uint32 取模触发 OverflowError。
    - 位掩码确保上界回落到合法范围。

    参数：
    - seed_int: 已转换为 int 的种子值。

    返回：
    - int：mask 后的 uint32 种子。
    """
    return int(seed_int) & 0xFFFFFFFF


def _coerce_seed_uint32(raw_seed):
    """
    函数作用：
    - 将任意输入种子转换为合法 uint32，并对越界值做降级与告警。

    设计原因：
    - Compose 初始化会直接调用 get_seed()，若返回 2**32 会触发溢出。
    - 统一入口确保所有路径都落在合法范围内。

    参数：
    - raw_seed: 任意类型的种子输入。

    返回：
    - int：合法 uint32 范围内的种子。
    """
    try:
        raw_seed_int = int(raw_seed)
    except (TypeError, ValueError, OverflowError):
        raw_seed_int = random.getrandbits(32)
    if raw_seed_int >= 2 ** 32:
        logger.warning(
            f"seed 越界({raw_seed_int})，已降级到 uint32 合法范围内以避免 OverflowError。"
        )
        raw_seed_int = raw_seed_int % (2 ** 32)
    return _mask_seed_uint32(raw_seed_int)


class RandomResizedCrop3d(Crop, Randomizable):
    def __init__(
        self,
        size,
        in_slice_scale,
        cross_slice_scale,
        interpolation='trilinear',
        aspect_ratio=(0.9, 1/0.9),
    ):
        """
        Adapting torch RandomResizedCrop to 3D data by separating in-slice/in-plane and cross-slice dimensions.

        Args:
            size: Size of output image.
            in_slice_scale: Range of the random size of the cropped in-slice/in-plane dimensions.
            cross_slice_scale: Range of the random size of the cropped cross-slice dimensions.
            interpolation: 3D interpolation method, defaults to 'trilinear'.
            aspect_ratio: Range of aspect ratios of the cropped in-slice/in-plane dimensions.
        """
        super().__init__()
        self.size = size
        self.in_slice_scale = in_slice_scale
        self.cross_slice_scale = cross_slice_scale
        self.interpolation = interpolation
        self.aspect_ratio = aspect_ratio
        self._slices: tuple[slice, ...] = ()

    def get_in_slice_crop(self, height, width):
        """
        Adapted from torchvision RandomResizedCrop, applied to the in-slice/in-plane dimensions
        """
        # Guard against invalid shapes propagated from upstream transforms.
        # Returning a minimal valid crop avoids ZeroDivisionError and keeps workers alive.
        if height <= 0 or width <= 0:
            return 1, 1

        area = height * width

        log_ratio = math.log(self.aspect_ratio[0]), math.log(self.aspect_ratio[1])
        for _ in range(10):
            target_area = area * self.R.uniform(*self.in_slice_scale)
            aspect_ratio = math.exp(self.R.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                return h, w

        # Fallback to central crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(self.aspect_ratio):
            w = width
            h = int(round(w / min(self.aspect_ratio)))
        elif in_ratio > max(self.aspect_ratio):
            h = height
            w = int(round(h * max(self.aspect_ratio)))
        else:  # whole image
            w = width
            h = height
        return h, w

    def randomize(self, img_size):
        # first two dimensions are dicom slice dims/in-plane dims, third is number of slices
        height, width, depth = img_size

        # Ensure random crop parameters are always valid.
        height = max(1, int(height))
        width = max(1, int(width))
        depth = max(1, int(depth))

        # get in-slice crop size
        crop_h, crop_w = self.get_in_slice_crop(height, width)

        # get cross-slice crop size
        crop_d = max(1, int(round(depth * self.R.uniform(*self.cross_slice_scale))))

        crop_size = (crop_h, crop_w, crop_d)
        valid_size = get_valid_patch_size(img_size, crop_size)
        self._slices = get_random_patch(img_size, valid_size, self.R)

    def __call__(self, img, lazy=False):
        """
        函数作用：
        - 对三维张量进行随机裁剪与重采样，模拟 3D 版本的 RandomResizedCrop。
        - 兼容输入为字典的情况：若为 dict，提取 'image' 键对应的张量后再执行后续逻辑。
        - 增强：手动裁剪 + 强制 5D 插值 + 详细形状日志，确保鲁棒性。
        """
        try:
            # 1. 兼容 dict 输入
            if isinstance(img, dict):
                img = img.get("image", img)
            if isinstance(img, dict):
                raise TypeError(f"RandomResizedCrop3d 期望输入为张量，但收到字典且无法提取 'image': {type(img)}")

            # 2. 维度检查与标准化 (Ensure C, H, W, D)
            input_shape = img.shape
            if img.ndim == 3:
                # 假设输入是 (H, W, D)，补充 Channel 维度 -> (1, H, W, D)
                img = img.unsqueeze(0)
            elif img.ndim != 4:
                raise ValueError(f"输入图像维度不正确: {input_shape}，期望 3D (H,W,D) 或 4D (C,H,W,D)")

            # 3.1 空轴兜底：某些样本在前景裁剪后可能出现 0 尺寸轴，直接构造最小合法张量避免 worker 崩溃
            spatial_shape = img.shape[1:]
            if any(int(s) <= 0 for s in spatial_shape):
                logger.warning(
                    f"RandomResizedCrop3d 检测到非法空间维度 {tuple(spatial_shape)}，"
                    "将回退为最小合法零张量 (C,1,1,1) 以跳过坏样本。"
                )
                c = max(1, int(img.shape[0]))
                img = torch.zeros((c, 1, 1, 1), dtype=img.dtype, device=img.device)

            # 3. 生成随机切片参数 (基于空间维度 H, W, D)
            # img.shape[1:] 对应 (H, W, D)
            self.randomize(img.shape[1:])
            
            # 4. 手动执行裁剪 (Manual Cropping)
            # self._slices 是 (slice_h, slice_w, slice_d)
            if len(self._slices) != 3:
                 raise RuntimeError(f"生成的切片数量不正确: {len(self._slices)}，期望 3 个 (H, W, D)")
            
            # 使用切片裁剪，保留 Channel 维度
            cropped = img[:, self._slices[0], self._slices[1], self._slices[2]]

            # 5. 执行插值重采样 (Interpolation)
            # interpolate 需要 5D 输入: (Batch, Channel, D, H, W)
            # cropped 当前形状: (C, D', H', W') -> unsqueeze(0) -> (1, C, D', H', W')
            cropped_5d = cropped.unsqueeze(0)
            
            # 确保插值参数正确
            mode = self.interpolation
            align_corners = False if mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None

            resized_5d = interpolate(
                cropped_5d, 
                size=self.size, 
                mode=mode, 
                align_corners=align_corners
            )
            
            # 恢复形状: (1, C, size, size, size) -> (C, size, size, size)
            resized = resized_5d.squeeze(0)
            
            return resized

        except Exception as e:
            import traceback
            logger.error(f"RandomResizedCrop3d 增强错误 | Input: {input_shape if 'input_shape' in locals() else 'Unknown'} | Error: {e}")
            logger.error(traceback.format_exc())
            raise e


class CropForegroundSwapSliceDims(CropForeground):
    """
    Same functionality as CropForeground, but permutes in-plane dimensions to first two spatial dims for
    RandomResizedCrop3d.
    """
    @staticmethod
    def get_permutation(shape_or_spacing):
        # get permutation for how to swap slice axes, add small tolerance
        if abs(shape_or_spacing[0] - shape_or_spacing[1]) < 1e-2:
            permutation = (0, 1, 2, 3)
        elif abs(shape_or_spacing[0] - shape_or_spacing[2]) < 1e-2:
            permutation = (0, 1, 3, 2)
        elif abs(shape_or_spacing[1] - shape_or_spacing[2]) < 1e-2:
            permutation = (0, 2, 3, 1)
        else:
            permutation = None
        return permutation

    def __call__(self, img_dict, mode=None, lazy=None, **pad_kwargs):
        """
        函数作用：
        - 对输入字典中的图像进行前景裁剪前，依据 spacing 或空间维度推断需要的维度排列，
          将切片相关的两个平面轴置于前两位，再调用 MONAI 的 CropForeground 完成裁剪。

        设计原因：
        - 数据集 JSON 或 MONAI 加载的 meta 有时不提供 'spacing'，原实现强索引导致 KeyError。
        - 改为使用 img_dict.get('spacing', None)，在缺失 spacing 时优雅降级为根据空间维度判断，
          保持功能完整并避免崩溃。

        参数：
        - img_dict: 包含 'image' 必填键和可选 'spacing' 键的字典
        - mode, lazy, **pad_kwargs: 透传到父类 CropForeground 的参数

        返回：
        - 前景裁剪后的图像张量（维度已按推断的排列进行重排）
        """
        # get image spacing and spatial dims（改为 .get 避免 KeyError）
        img_spacing = img_dict.get('spacing', None)
        img = img_dict['image']
        spatial_dims = img.shape[1:]

        # try getting from pixel spacing first, NOTE: verified that at least two dims have similar spacing in datasets
        if img_spacing is not None:
            perm = self.get_permutation(img_spacing)
        else:
            perm = self.get_permutation(spatial_dims)

        if perm is None:
            raise RuntimeError('Could not determine slice dimension permutation')

        # swap slice dims
        img = img.permute(*perm)

        # crop foreground
        return super().__call__(img, mode, lazy, **pad_kwargs)


class DataAugmentationDINO3d(object):

    def __init__(
        self,
        global_crops_in_slice_scale,
        global_crops_cross_slice_scale,
        local_crops_in_slice_scale,
        local_crops_cross_slice_scale,
        local_crops_number,
        global_crops_size=96,
        local_crops_size=48,
        seed=None,
    ):
        self.global_crops_in_slice_scale = global_crops_in_slice_scale
        self.global_crops_cross_slice_scale = global_crops_cross_slice_scale
        self.local_crops_in_slice_scale = local_crops_in_slice_scale
        self.local_crops_cross_slice_scale = local_crops_cross_slice_scale
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size
        raw_seed = get_seed() if seed is None else seed
        masked_seed = _coerce_seed_uint32(raw_seed)
        self._base_seed = _normalize_seed(masked_seed)

        logger.info("###################################")
        logger.info("Using 3d data augmentation parameters:")
        logger.info(f"global_crops_in_slice_scale: {global_crops_in_slice_scale}")
        logger.info(f"global_crops_cross_slice_scale: {global_crops_cross_slice_scale}")
        logger.info(f"local_crops_in_slice_scale: {local_crops_in_slice_scale}")
        logger.info(f"local_crops_cross_slice_scale: {local_crops_cross_slice_scale}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        original_get_seed = monai_compose.get_seed
        def _patched_get_seed():
            return _coerce_seed_uint32(original_get_seed())
        monai_compose.get_seed = _patched_get_seed
        try:
            # random resized crop, flip and rot
            self.geometric_augmentation_global = Compose(
                [
                    RandomResizedCrop3d(
                        global_crops_size,
                        in_slice_scale=global_crops_in_slice_scale,
                        cross_slice_scale=global_crops_cross_slice_scale
                    ),
                    RandFlip(prob=0.3, spatial_axis=[0]),
                    RandFlip(prob=0.3, spatial_axis=[1]),
                    RandFlip(prob=0.3, spatial_axis=[2]),
                    RandRotate90(prob=0.3, spatial_axes=(0, 1)),
                    RandRotate90(prob=0.3, spatial_axes=(1, 2)),
                    RandRotate90(prob=0.3, spatial_axes=(0, 2))
                ]
            )

            self.geometric_augmentation_local = Compose(
                [
                    RandomResizedCrop3d(
                        local_crops_size,
                        in_slice_scale=local_crops_in_slice_scale,
                        cross_slice_scale=local_crops_cross_slice_scale
                    ),
                    RandFlip(prob=0.3, spatial_axis=[0]),
                    RandFlip(prob=0.3, spatial_axis=[1]),
                    RandFlip(prob=0.3, spatial_axis=[2]),
                    RandRotate90(prob=0.3, spatial_axes=(0, 1)),
                    RandRotate90(prob=0.3, spatial_axes=(1, 2)),
                    RandRotate90(prob=0.3, spatial_axes=(0, 2))
                ]
            )

            # noise, contrast, blurring
            gaussian_transforms = OneOf(
                [
                    RandAdjustContrast(prob=0.8, gamma=(0.5, 2)),
                    RandGaussianNoise(prob=0.8, std=0.002),
                    RandHistogramShift(num_control_points=10, prob=0.8),
                ]
            )

            global_transfo1_extra = OneOf(
                [
                    RandGaussianSmooth(prob=1.0),
                    RandGaussianSharpen(prob=1.0),
                ]
            )

            global_transfo2_extra = transforms.Compose(
                [
                    OneOf(
                        [
                            RandGaussianSmooth(prob=0.1),
                            RandGaussianSharpen(prob=0.1),
                        ]
                    ),
                    RandGibbsNoise(prob=0.2)
                ]
            )

            local_transfo_extra = RandGaussianSmooth(prob=0.5)

            self.global_transfo1 = Compose([gaussian_transforms, global_transfo1_extra, ToTensor()])
            self.global_transfo2 = Compose([gaussian_transforms, global_transfo2_extra, ToTensor()])
            self.local_transfo = Compose([gaussian_transforms, local_transfo_extra, ToTensor()])
        finally:
            monai_compose.get_seed = original_get_seed
        self._set_safe_random_state(self._base_seed)

    def _set_safe_random_state(self, seed):
        """
        函数作用：
        - 为内部多个 MONAI Compose 统一设置安全随机种子，避免 uint32 溢出。

        设计原因：
        - Compose.set_random_state 在 seed 为空时会调用 get_seed，可能返回 2**32。
        - 通过显式传入归一化 seed，彻底规避超出 uint32 的边界值。

        参数：
        - seed: 基础随机种子（任意类型，内部会归一化）。
        """
        base_seed = _normalize_seed(seed)
        transforms_list = [
            self.geometric_augmentation_global,
            self.geometric_augmentation_local,
            self.global_transfo1,
            self.global_transfo2,
            self.local_transfo,
        ]
        for index, transform in enumerate(transforms_list):
            if hasattr(transform, "set_random_state"):
                transform.set_random_state(seed=_normalize_seed(base_seed + index + 1))

    def __call__(self, image):
        """
        函数作用：
        - 对输入 3D 图像（张量）应用几何与强度增强，生成 global/local crops 以及 teacher 用的 global crops。
        - 兼容字典输入：若为 dict，优先提取 'image' 键对应的张量再进行增强。

        设计原因：
        - 在 MONAI 的 Compose 管线中，上游变换可能产出字典或张量。
        - 原实现假定输入为张量，导致在字典输入时下游增强（RandomResizedCrop3d）访问 .shape 报错。
        - 做输入兼容以提升整体管线稳定性。

        参数：
        - image: 输入张量（形状 [C, H, W, D]）或包含 'image' 键的字典

        返回：
        - (output, None)：output 为包含 global_crops、global_crops_teacher、local_crops、offsets 的字典，标签为 None
        """
        # 兼容 dict 输入：若是字典则提取 'image' 键的张量
        if isinstance(image, dict):
            img = image.get("image", image)
        else:
            img = image
        # 若仍为字典则类型不合法
        if isinstance(img, dict):
            raise TypeError("DataAugmentationDINO3d 期望输入为张量或包含 'image' 键的字典")

        output = {}

        # global crops:
        im1_base = self.geometric_augmentation_global(img)
        global_crop_1 = self.global_transfo1(im1_base)

        im2_base = self.geometric_augmentation_global(img)
        global_crop_2 = self.global_transfo2(im2_base)

        output["global_crops"] = [global_crop_1, global_crop_2]

        # global crops for teacher:
        output["global_crops_teacher"] = [global_crop_1, global_crop_2]

        # local crops:
        local_crops = [
            self.local_transfo(self.geometric_augmentation_local(img)) for _ in range(self.local_crops_number)
        ]
        output["local_crops"] = local_crops
        output["offsets"] = ()

        # "label" expected, but return nothing
        return output, None
