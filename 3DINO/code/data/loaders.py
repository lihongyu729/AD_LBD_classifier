# Author: Tony Xu
#
# This code is adapted from the original DINOv2 repository: https://github.com/facebookresearch/dinov2
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

import logging
from enum import Enum
from typing import Any, Callable, List, Optional, TypeVar
import os
from copy import deepcopy
import random

import torch
from torch.utils.data import Sampler
from monai.data import CacheNTransDataset, PersistentDataset
import json

from .samplers import EpochSampler, InfiniteSampler, ShardedInfiniteSampler


logger = logging.getLogger("dinov2")


class SamplerType(Enum):
    DISTRIBUTED = 0
    EPOCH = 1
    INFINITE = 2
    SHARDED_INFINITE = 3
    SHARDED_INFINITE_NEW = 4


def _make_bool_str(b: bool) -> str:
    return "yes" if b else "no"


def _make_sample_transform(image_transform: Optional[Callable] = None, target_transform: Optional[Callable] = None):
    def transform(sample):
        image, target = sample
        if image_transform is not None:
            image = image_transform(image)
        if target_transform is not None:
            target = target_transform(target)
        return image, target

    return transform


def make_dataset_3d(
    *,
    dataset_path: str,
    cache_path: str,
    data_min_axis_size: int,
    transform: Optional[Callable] = None,
):
    """
    函数作用：
    - 统一并归一化 3D 数据集 JSON，兼容 list 与 dict（含 training/validation/test）两种格式。
    - 保留原始条目中的关键字段，包括 'image'、可选的 'label' 与 'spacing'。
    - 在需要最小轴过滤时（data_min_axis_size > 0），若缺少 shape 尝试从 NIfTI 头获取形状。
    - 使用 PersistentDataset 做磁盘级缓存，显著降低多进程场景下的内存占用，避免 OOM。

    设计原因：
    - 之前使用 CacheNTransDataset（内存缓存）在 3D 医学影像 + 多 worker 场景下容易触发内存爆炸，导致 worker 被系统 Killed。
    - PersistentDataset 以磁盘缓存替代内存缓存，稳定性更好。

    返回：
    - MONAI PersistentDataset 数据集
    """
    logger.info(f'creating 3d dataset from datalist: {dataset_path}')

    # 加载 JSON（兼容 UTF-8）
    with open(dataset_path, 'r', encoding='utf-8') as json_f:
        datalist = json.load(json_f)

    # 1) 若顶层是 dict（含 training/validation/test），先合并为一个列表
    if isinstance(datalist, dict):
        merged = []
        for k in ("training", "validation", "test"):
            v = datalist.get(k, [])
            if isinstance(v, list):
                merged.extend(v)
        datalist = merged

    # 2) 统一元素结构，抽取路径，并在需要时补充 shape
    def _extract_path(item):
        if isinstance(item, dict):
            return item.get("image") or item.get("path") or item.get("file") or item.get("img")
        elif isinstance(item, str):
            return item  # 兼容纯路径字符串
        return None

    normalized = []
    for item in datalist:
        p = _extract_path(item)
        if not p:
            # 无法解析路径，跳过
            continue

        entry = {"image": p}
        if isinstance(item, dict) and "label" in item:
            entry["label"] = item["label"]

        # 新增：保留 JSON 中的 spacing 字段，若存在且合法（长度为 3）
        if isinstance(item, dict) and "spacing" in item:
            s = item["spacing"]
            if isinstance(s, (list, tuple)) and len(s) == 3:
                entry["spacing"] = [float(s[0]), float(s[1]), float(s[2])]

        # 优先使用已有 shape
        shape = None
        if isinstance(item, dict) and "shape" in item:
            s = item["shape"]
            if isinstance(s, (list, tuple)) and len(s) >= 3:
                shape = tuple(int(x) for x in s[:3])

        # 若需要过滤且无 shape，则尝试读 NIfTI 头部形状（仅头部，不读数据）
        if shape is None and data_min_axis_size and data_min_axis_size > 0:
            try:
                import nibabel as nib
                img = nib.load(p)
                header_shape = img.header.get_data_shape()
                eff_shape = tuple(d for d in header_shape if d != 1)[:3]  # 去除长度为 1 的轴
                shape = eff_shape if len(eff_shape) == 3 else None
            except Exception:
                shape = None  # 失败不强制报错，仅在后续过滤时跳过

        entry["shape"] = shape
        normalized.append(entry)

    # 3) 最小轴过滤：仅当阈值>0且成功获取 shape 才过滤
    if data_min_axis_size and data_min_axis_size > 0:
        datalist = [x for x in normalized if x["shape"] and min(x["shape"]) > data_min_axis_size]
    else:
        datalist = normalized

    # 改为磁盘缓存，降低内存占用
    dataset = PersistentDataset(datalist, transform=transform, cache_dir=cache_path)

    # Aggregated datasets do not expose (yet) these attributes, so add them.
    if not hasattr(dataset, "transform"):
        setattr(dataset, "transform", transform)

    return dataset


def make_segmentation_dataset_3d(
    dataset_name: str,
    dataset_percent: int,
    base_directory: str,
    train_transforms: Callable,
    val_transforms: Callable,
    cache_path: str,
    batch_size: int,
):
    """
    Creates a 3d segmentation dataset with the specified parameters.

    Args:
        dataset_name: Name of the segmentation dataset (BTCV, BraTS, LA-SEG, TDSC-ABUS).
        dataset_percent: Percentage of the dataset to use for training.
        base_directory: Base directory where dataset json files are stored.
        train_transforms: Training transforms to apply to images.
        val_transforms: Validation transforms to apply to images.
        cache_path: A path to a directory to cache the dataset, used in PersistentDataset.
        batch_size: Batch size for the dataset.
    Returns:
        Created train, val, and test datasets, number of input channels, and number of classes for the dataset.
    """

    if dataset_name == 'BTCV':
        datalist_path = os.path.join(base_directory, 'BTCV_100_datalist.json')
        class_num = 14
        input_channels = 1
    elif dataset_name == 'BraTS':
        datalist_path = os.path.join(base_directory, 'BraTS_100_datalist.json')
        class_num = 3
        input_channels = 4
    elif dataset_name == 'LA-SEG':
        datalist_path = os.path.join(base_directory, 'LA-SEG_100_datalist.json')
        class_num = 2
        input_channels = 1
    elif dataset_name == 'TDSC-ABUS':
        datalist_path = os.path.join(base_directory, 'TDSC-ABUS_100_datalist.json')
        class_num = 2
        input_channels = 1
    else:
        raise ValueError(f'Unsupported dataset "{dataset_name}"')

    with open(datalist_path, 'r') as json_f:
        datalist = json.load(json_f)

    train_data_ind = int(round(len(datalist['training']) * (dataset_percent / 100)))

    train_datalist = datalist['training'][:train_data_ind]
    val_datalist = datalist['validation']
    test_datalist = datalist['test']
    logger.info(f"# of train samples: {len(train_datalist):,d}")
    logger.info(f"# of val samples: {len(val_datalist):,d}")
    logger.info(f"# of test samples: {len(test_datalist):,d}")

    if len(train_datalist) < batch_size:
        logger.info(f"copying train samples to match batch size: {batch_size:,d}")
        copied_datalist = []
        for i in range(batch_size // len(train_datalist)):
            copied_datalist.extend(deepcopy(train_datalist))
        assert len(copied_datalist) == batch_size
        train_datalist = copied_datalist

    train_dataset = PersistentDataset(train_datalist, transform=train_transforms, cache_dir=cache_path)
    val_dataset = PersistentDataset(val_datalist, transform=val_transforms, cache_dir=cache_path)
    test_dataset = PersistentDataset(test_datalist, transform=val_transforms, cache_dir=cache_path)

    return train_dataset, val_dataset, test_dataset, input_channels, class_num


def make_classification_dataset_3d(
    dataset_name: str,
    dataset_percent: int,
    base_directory: str,
    train_transforms: Callable,
    val_transforms: Callable,
    cache_path: str,
    dataset_seed: int,
):
    """
    Creates a 3d classification dataset with the specified parameters.

    Args:
        dataset_name: Name of the classification dataset (ICBM, COVID-CT-MD).
        dataset_percent: Percentage of the dataset to use for training.
        base_directory: Base directory where dataset json files are stored.
        train_transforms: Training transforms to apply to images.
        val_transforms: Validation transforms to apply to images.
        cache_path: A path to a directory to cache the dataset, used in PersistentDataset.
        dataset_seed: Seed for random shuffling of the dataset.
    Returns:
        Created train, val, and test datasets, and number of classes for the dataset.
    """

    if dataset_name == 'ICBM':
        datalist_path = os.path.join(base_directory, 'ICBM_cls_datalist.json')
        class_num = 4
    elif dataset_name == 'COVID-CT-MD':
        datalist_path = os.path.join(base_directory, 'COVID-CT-MD_cls_datalist.json')
        class_num = 3
    else:
        raise ValueError(f'Unsupported dataset "{dataset_name}"')

    with open(datalist_path, 'r') as json_f:
        datalist = json.load(json_f)

    # filter ages for icbm
    if dataset_name == 'ICBM':

        for k in datalist:
            for item in datalist[k]:
                item['image'] = item['image'].replace('.nii.gz', '_mask.nii.gz')

        datalist['training'] = [x for x in datalist['training'] if 20 <= x['label'] <= 60]
        datalist['validation'] = [x for x in datalist['validation'] if 20 <= x['label'] <= 60]
        datalist['test'] = [x for x in datalist['test'] if 20 <= x['label'] <= 60]

    # ensure reproducible shuffling
    random.Random(dataset_seed).shuffle(datalist['training'])
    print(f'Shuffled with seed: {dataset_seed}')

    train_data_ind = int(round(len(datalist['training']) * (dataset_percent / 100)))
    train_datalist = datalist['training'][:train_data_ind]
    val_datalist = datalist['validation']
    test_datalist = datalist['test']

    logger.info(f"# of train samples: {len(train_datalist):,d}")
    logger.info(f"# of val samples: {len(val_datalist):,d}")
    logger.info(f"# of test samples: {len(test_datalist):,d}")

    train_dataset = PersistentDataset(train_datalist, transform=train_transforms, cache_dir=cache_path)
    val_dataset = PersistentDataset(val_datalist, transform=val_transforms, cache_dir=cache_path)
    test_dataset = PersistentDataset(test_datalist, transform=val_transforms, cache_dir=cache_path)

    return train_dataset, val_dataset, test_dataset, class_num


def _make_sampler(
    *,
    dataset,
    type: Optional[SamplerType] = None,
    shuffle: bool = False,
    seed: int = 0,
    size: int = -1,
    advance: int = 0,
) -> Optional[Sampler]:
    sample_count = len(dataset)

    if type == SamplerType.INFINITE:
        logger.info("sampler: infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        return InfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
        )
    elif type in (SamplerType.SHARDED_INFINITE, SamplerType.SHARDED_INFINITE_NEW):
        logger.info("sampler: sharded infinite")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        # TODO: Remove support for old shuffling
        use_new_shuffle_tensor_slice = type == SamplerType.SHARDED_INFINITE_NEW
        return ShardedInfiniteSampler(
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
            advance=advance,
            use_new_shuffle_tensor_slice=use_new_shuffle_tensor_slice,
        )
    elif type == SamplerType.EPOCH:
        logger.info("sampler: epoch")
        if advance > 0:
            raise NotImplementedError("sampler advance > 0 is not supported")
        size = size if size > 0 else sample_count
        logger.info(f"# of samples / epoch: {size:,d}")
        return EpochSampler(
            size=size,
            sample_count=sample_count,
            shuffle=shuffle,
            seed=seed,
        )
    elif type == SamplerType.DISTRIBUTED:
        logger.info("sampler: distributed")
        if size > 0:
            raise ValueError("sampler size > 0 is invalid")
        if advance > 0:
            raise ValueError("sampler advance > 0 is invalid")
        return torch.utils.data.DistributedSampler(
            dataset=dataset,
            shuffle=shuffle,
            seed=seed,
            drop_last=False,
        )

    logger.info("sampler: none")
    return None


T = TypeVar("T")


def make_data_loader(
    *,
    dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    seed: int = 0,
    sampler_type: Optional[SamplerType] = SamplerType.INFINITE,
    sampler_size: int = -1,
    sampler_advance: int = 0,
    drop_last: bool = True,
    persistent_workers: bool = False,
    collate_fn: Optional[Callable[[List[T]], Any]] = None,
    pin_memory: bool = True,
    prefetch_factor: int = 2,
):
    """
    函数作用：
    - 构建并返回 PyTorch 的 DataLoader，支持无限/分片采样等不同采样器类型。
    - 新增两个关键参数：
      - pin_memory：是否启用 CPU 锁页内存（用于更快地拷贝到 GPU），在内存压力大时建议设为 False。
      - prefetch_factor：每个 worker 的预取批次数，默认 2；3D 大样本场景建议为 1 以降低内存峰值。

    设计原因：
    - 你遇到的 TypeError 来自调用处传入了 pin_memory，但函数签名没有该参数；这里补充签名并正确传递。
    - 在多 worker + 大体积 3D NIfTI 的场景，合理降低预取与禁用 pin_memory 有助于防止 worker 被系统 kill。

    参数：
    - dataset: 数据集实例
    - batch_size: 批大小
    - num_workers: worker 数量
    - shuffle/seed/sampler_type/...: 采样与 DataLoader 配置
    - persistent_workers: 是否维持 worker 常驻
    - collate_fn: 批内样本的拼接函数
    - pin_memory: 是否启用锁页内存（默认 True）
    - prefetch_factor: 预取批次数（默认 2），仅在 num_workers > 0 时生效

    返回：
    - 构建完成的 DataLoader
    """
    sampler = _make_sampler(
        dataset=dataset,
        type=sampler_type,
        shuffle=shuffle,
        seed=seed,
        size=sampler_size,
        advance=sampler_advance,
    )

    logger.info("using PyTorch data loader")
    # 使用 kwargs 以便在 num_workers=0 时不传 prefetch_factor，避免不兼容
    loader_kwargs = {
        "dataset": dataset,
        "sampler": sampler,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "persistent_workers": persistent_workers,
        "collate_fn": collate_fn,
    }
    if num_workers and num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor

    data_loader = torch.utils.data.DataLoader(**loader_kwargs)

    try:
        logger.info(f"# of batches: {len(data_loader):,d}")
    except TypeError:  # data loader has no length
        logger.info("infinite data loader")
    return data_loader
