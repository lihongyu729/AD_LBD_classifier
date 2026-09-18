# Author: Tony Xu
#
# This code is adapted from the original DINOv2 repository: https://github.com/facebookresearch/dinov2
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

import argparse
import logging
import math
import os

# --- Force disable xformers to avoid CUDA version mismatch (libcublasLt.so.12 error) ---
# The environment has PyTorch 2.0 (CUDA 11.7) but xformers might be trying to load CUDA 12 libs.
os.environ["XFORMERS_DISABLED"] = "0"
# ---------------------------------------------------------------------------------------

from functools import partial
from monai.transforms import Compose, LoadImaged, ScaleIntensityRangePercentilesd, Lambdad
import random

from fvcore.common.checkpoint import PeriodicCheckpointer
import torch

from dinov2.data import SamplerType, make_data_loader, make_dataset_3d
from dinov2.data import collate_data_and_cast, DataAugmentationDINO3d, MaskingGenerator3d, CropForegroundSwapSliceDims
import dinov2.distributed as distributed
from dinov2.fsdp import FSDPCheckpointer

from dinov2.dino_logging import MetricLogger
from dinov2.utils.config import setup_3d
from dinov2.utils.utils import CosineScheduler

from dinov2.train.ssl_meta_arch import SSLMetaArch
from monai.transforms import MapTransform
import numpy as np


class AddSpacingFromMeta(MapTransform):
    """
    从 MONAI 加载的元数据字典中提取 spacing 信息，
    并将其作为 'spacing' 键添加到数据字典的顶层。
    """
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            if f"{key}_meta_dict" in d:
                meta_dict = d[f"{key}_meta_dict"]
                # 从 affine 矩阵的对角线获取 spacing
                if "affine" in meta_dict:
                    spacing = np.abs(meta_dict["affine"].diagonal()[:3])
                    d["spacing"] = spacing.tolist()
        return d


torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")


def _normalize_seed(seed):
    """
    函数作用：
    - 将输入的随机种子归一化到 uint32 的合法范围 [0, 2**32 - 1]。

    设计原因：
    - MONAI 的 Randomizable 仅接受 uint32；当上游生成 2**32 会触发溢出异常。
    - 通过显式的取模与兜底随机数生成，避免 Compose.set_random_state 触发越界。

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


def _smoke_test_max_seed(cfg):
    """
    函数作用：
    - 以 uint32 最大合法种子实例化 DataAugmentationDINO3d，并执行一次前向增强以验证初始化链。

    设计原因：
    - 避免 MONAI Compose 在初始化时因种子越界触发 OverflowError。
    - 在训练入口提前暴露潜在问题，确保后续前向传播正常进行。

    参数：
    - cfg: 训练配置对象，提供裁剪尺寸与比例参数。
    """
    max_seed = (2 ** 32) - 1
    aug = DataAugmentationDINO3d(
        cfg.crops.global_crops_in_slice_scale,
        cfg.crops.global_crops_cross_slice_scale,
        cfg.crops.local_crops_in_slice_scale,
        cfg.crops.local_crops_cross_slice_scale,
        cfg.crops.local_crops_number,
        global_crops_size=cfg.crops.global_crops_size,
        local_crops_size=cfg.crops.local_crops_size,
        seed=max_seed,
    )
    dummy = torch.zeros(
        1,
        cfg.crops.global_crops_size,
        cfg.crops.global_crops_size,
        cfg.crops.global_crops_size,
    )
    aug(dummy)

def _resolve_dataset_path(cfg):
    """
    函数作用：
    - 统一解析数据清单路径：优先使用 `cfg.DATASET.dataset_path`，如未设置则回退到 `cfg.TRAIN.dataset_path`（或 `cfg.train.dataset_path`）。
    - 在解析后立即进行存在性校验，确保返回的路径真实存在，避免在下游 `make_dataset_3d` 处抛出 FileNotFoundError。

    设计原因：
    - 你当前的错误显示代码仍在读取历史默认路径（/home/txu/.../dinov2_subset_datalist.json），说明路径解析存在“多源”且优先级不明确。
    - 通过入口统一解析与校验，强制使用正确的 JSON（比如 /webdav/MyData/MRI/data/3dino_cls.json），彻底消除歧义。

    返回：
    - 字符串，指向存在的 JSON 文件路径。

    异常：
    - 若两个位置都为空或文件不存在，抛出 FileNotFoundError，并明确提示两个字段当前值以及修复建议。
    """
    import os

    # 收集候选路径（优先 DATASET，再回退 TRAIN/train）
    candidates = []
    try:
        if hasattr(cfg, "DATASET") and getattr(cfg.DATASET, "dataset_path", None):
            candidates.append(cfg.DATASET.dataset_path)
    except Exception:
        pass

    # 有的配置对象字段为 TRAIN（大写），有的为 train（小写），都尝试
    try:
        if hasattr(cfg, "TRAIN") and getattr(cfg.TRAIN, "dataset_path", None):
            candidates.append(cfg.TRAIN.dataset_path)
    except Exception:
        pass
    try:
        if hasattr(cfg, "train") and getattr(cfg.train, "dataset_path", None):
            candidates.append(cfg.train.dataset_path)
    except Exception:
        pass

    # 选择第一个非空候选，并校验存在性
    for p in candidates:
        if p and os.path.isfile(p):
            return p

    # 构造详细错误信息，提示两个字段当前值
    msg_lines = [
        "未找到有效的数据清单文件，请在配置 YAML 中统一设置存在的 JSON 路径（建议两个位置都指向同一文件）。",
        f"DATASET.dataset_path: {getattr(getattr(cfg, 'DATASET', object()), 'dataset_path', None)}",
        f"TRAIN.dataset_path: {getattr(getattr(cfg, 'TRAIN', object()), 'dataset_path', None)}",
        f"train.dataset_path: {getattr(getattr(cfg, 'train', object()), 'dataset_path', None)}",
        "修复建议：将以上字段统一为 /webdav/MyData/MRI/data/3dino_cls.json（或你实际的 JSON）。"
    ]
    raise FileNotFoundError("\n".join(msg_lines))

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("3DINO training", add_help=add_help)
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )
    parser.add_argument("--local-rank", default=0, type=int, help="Variable for distributed computing.")
    parser.add_argument(
        "--cache-dir",
        default=None,
        type=str,
        help="path to cache directory for monai persistent dataset"
    )

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def do_test(cfg, model, iteration):
    new_state_dict = model.teacher.state_dict()

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)


def do_train(cfg, model, resume=False):
    """
    函数作用：
    - 执行 3DINO 训练主流程，包括优化器/调度器构建、数据加载、训练循环与周期性评估/保存。
    - 新增：对 DataLoader 的 worker 数与内存相关配置进行保守化设置，避免 worker 被系统 kill。
    - 新增：启用 AMP autocast（fp16/bf16）、动态夹紧批大小、设置 CUDA 分配碎片上限，并在前向阶段增加 OOM 捕获与跳过。
    
    设计原因：
    - 3D 医学影像体素数据体积大、增强复杂，多 worker 容易导致 OOM 或远程 IO 拥堵。
    - 动态夹紧 num_workers，并禁用 pin_memory、降低 prefetch_factor，有助于提升稳定性。
    - 尽管 FSDP MixedPrecision 已将参数/归约/buffer 设为半精度，但算子计算仍可能以 fp32 进行；显式 autocast 降低算子显存占用。
    - 根据报错提示设置 PYTORCH_CUDA_ALLOC_CONF 可缓解内存碎片问题。
    - OOM 捕获与跳过批次避免训练进程直接崩溃。
    """
    # 设置 CUDA 分配碎片上限（若未设置），缓解内存碎片导致的 OOM
    import os as _os
    if "PYTORCH_CUDA_ALLOC_CONF" not in _os.environ:
        _os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training

    # 在训练函数内引入 autocast dtype 解析（复用 eval 中的实现）
    from dinov2.eval.setup import get_autocast_dtype
    autocast_dtype = get_autocast_dtype(cfg)

    # setup optimizer
    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    # checkpointer
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)
    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=3750,
        max_iter=max_iter,
        max_to_keep=3,
    )

    # setup data preprocessing
    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 3
    mask_generator = MaskingGenerator3d(
        input_size=(img_size // patch_size, img_size // patch_size, img_size // patch_size)
    )
    _smoke_test_max_seed(cfg)

    def random_select_time(x):
        # if time axis exists, select random time slice
        if x.shape[0] > 1:
            t = random.randint(0, x.shape[0] - 1)
            x = x[t:t + 1]
        return x

    # Compose the loading and intensity scaling here to cache transforms in monai persistent dataset
    base_seed = _normalize_seed(getattr(cfg.train, "seed", 0))
    data_transform = Compose(
            [
                LoadImaged(keys=["image"], ensure_channel_first=True),
                AddSpacingFromMeta(keys=["image"]),
                Lambdad(keys=["image"], func=random_select_time),
                Lambdad(
                    keys=["image"], func=lambda x: torch.nan_to_num(x, torch.nanmean(x).item())
                ),  # replace NaNs with mean
                ScaleIntensityRangePercentilesd(keys=["image"], lower=0.05, upper=99.95, b_min=-1, b_max=1, clip=True),
                # Use inclusive threshold to avoid empty foreground when volume values collapse to -1 after scaling.
                CropForegroundSwapSliceDims(select_fn=lambda x: x >= -1),
                DataAugmentationDINO3d(
                    cfg.crops.global_crops_in_slice_scale,
                    cfg.crops.global_crops_cross_slice_scale,
                    cfg.crops.local_crops_in_slice_scale,
                    cfg.crops.local_crops_cross_slice_scale,
                    cfg.crops.local_crops_number,
                    global_crops_size=cfg.crops.global_crops_size,
                    local_crops_size=cfg.crops.local_crops_size,
                    seed=base_seed,
                )
            ]
        )
    data_transform.set_random_state(seed=base_seed)

    # data collate
    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )
    dataset_path = _resolve_dataset_path(cfg)
    print(f"[train3d] Using dataset_path: {dataset_path}")  # 新增：打印最终使用的数据清单
    # setup data loader
    dataset = make_dataset_3d(
        dataset_path=dataset_path,
        cache_path=cfg.train.cache_dir,
        data_min_axis_size=cfg.train.data_min_axis_size,
        transform=data_transform
    )
    sampler_type = SamplerType.SHARDED_INFINITE

    # 动态夹紧 num_workers，避免过多进程导致 OOM
    cpu_cnt = os.cpu_count() or 4
    safe_num_workers = max(1, min(cfg.train.num_workers, cpu_cnt // 2))

    # 动态夹紧 batch_size_per_gpu，直接控制显存峰值（3D + 多裁剪对显存极其敏感）
    safe_bs = max(1, min(cfg.train.batch_size_per_gpu, 4))

    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=safe_bs,
        num_workers=safe_num_workers,
        shuffle=True,
        seed=start_iter,
        sampler_type=sampler_type,
        sampler_advance=0,
        drop_last=True,
        collate_fn=collate_fn,
        # 更保守的设置：禁用 pin_memory，降低预取批次数
        pin_memory=False,
        prefetch_factor=1,
        persistent_workers=False,
    )

    # training loop
    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    import time
    st = time.time()

    for data in metric_logger.log_every(
        data_loader,
        5,
        header,
        max_iter,
        start_iter,
    ):
        print(f'batch time: {time.time() - st}')
        current_batch_size = data["collated_global_crops"].shape[0] / 2
        if iteration > max_iter:
            return

        # apply schedules
        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses
        optimizer.zero_grad(set_to_none=True)

        # 在 autocast 下进行前向与损失计算，降低显存开销；同时捕获 OOM 并跳过该批次
        try:
            with torch.amp.autocast('cuda', enabled=True, dtype=autocast_dtype):
                loss_dict = model.forward_backward(data, teacher_temp=teacher_temp)
        except torch.cuda.OutOfMemoryError as oom_err:
            logger.warning(f"CUDA OOM 捕获，跳过该批次；建议进一步调小 batch_size_per_gpu 或 local_crops_number。错误详情：{oom_err}")
            torch.cuda.empty_cache()
            st = time.time()
            continue

        # 在参数更新前检查损失是否为有限值，避免把坏梯度写入模型
        if any((not torch.isfinite(v).all().item()) for v in loss_dict.values()):
            logger.warning("检测到非有限损失（NaN/Inf），跳过该批次并清理梯度。")
            optimizer.zero_grad(set_to_none=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            st = time.time()
            continue

        # clip gradients
        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()

        # perform teacher EMA update
        model.update_teacher(mom)

        # logging
        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        if not math.isfinite(losses_reduced):
            logger.warning("分布式归约后损失非有限（NaN/Inf），跳过该次日志与保存步骤。")
            iteration = iteration + 1
            st = time.time()
            continue

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        # checkpointing and testing
        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        iteration = iteration + 1

        st = time.time()
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup_3d(args)

    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()

    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")

    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    main(args)
