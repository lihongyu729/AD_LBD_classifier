import os
import time
import numbers
import math
import warnings
import yaml
import torch
import torch.nn.functional as F
from torch import nn
import contextlib
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader

from dataset_mri3d import MRIVolumeDataset, collate_fn_skip_none
from medmamba3d import MedMamba3D
from medmamba_ss3m import MedMambaSS3MEncoder
import argparse


def load_config(cfg_path: str):
    with open(cfg_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

class SS3MMAE(nn.Module):
    def __init__(self, encoder, in_channels=1, patch_size=(2, 2, 2)):
        super().__init__()
        self.encoder = encoder
        self.in_channels = in_channels
        self.patch_size = patch_size

        # 定义一个简单的解码器，将编码器的特征图还原回原始分辨率
        # 编码器的输出维度通过 encoder.out_channels 获取
        embed_dim = encoder.out_channels
        
        self.decoder = nn.Sequential(
            nn.Conv3d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.GELU(),
            # 使用转置卷积进行上采样，恢复到原始图像尺寸
            nn.ConvTranspose3d(
                embed_dim, 
                in_channels, 
                kernel_size=patch_size, 
                stride=patch_size
            )
        )
    def forward_mae(self, x: torch.Tensor, mask_ratio: float = 0.0):
        """
        前向：
            - 输入 [B, C, D, H, W]
            - 输出 recon 与占位 mask（None）
        说明：
            - 此处未实现真正的 token 掩码 MAE，仅提供重建能力以便与 MedMamba3D 的 MAE 对齐流程。

        改动原因：
            - 在前向开始处，确保编码器与解码器都位于与输入 x 相同的设备；
            - 修复当 self.pos_emb 等内部缓冲仍在 CPU 而 x 在 CUDA 时的设备不一致错误。
        """
        # 新增：在前向时统一模块设备到输入 x 的设备（CUDA 或 CPU）
        # self.encoder = self.encoder.to(x.device)
        # self.decoder = self.decoder.to(x.device)

        tokens_ref = self.encoder.forward_tokens(x)
        _, _, d_tok, h_tok, w_tok = tokens_ref.shape

        if mask_ratio <= 0.0:
            mask = torch.zeros((x.size(0), 1, x.size(2), x.size(3), x.size(4)), device=x.device, dtype=x.dtype)
            x_visible = x
        else:
            mask_ratio = float(max(0.0, min(1.0, mask_ratio)))
            mask_token = (torch.rand((x.size(0), 1, d_tok, h_tok, w_tok), device=x.device) < mask_ratio).to(x.dtype)
            pd, ph, pw = self.patch_size
            mask = mask_token.repeat_interleave(pd, dim=2).repeat_interleave(ph, dim=3).repeat_interleave(pw, dim=4)
            mask = mask[:, :, :x.size(2), :x.size(3), :x.size(4)]
            x_visible = x * (1.0 - mask)

        tokens = self.encoder.forward_tokens(x_visible)  # [B, C', D', H', W']
        recon = self.decoder(tokens)                     # [B, C, D, H, W]
        return recon, mask


def mae_loss(recon: torch.Tensor, target: torch.Tensor, mask: torch.Tensor = None):
    # 仅在 masked 区域计算重建损失；若无 mask 则回退全图 L1。
    if mask is None:
        return F.l1_loss(recon, target)
    if mask.dtype != recon.dtype:
        mask = mask.to(dtype=recon.dtype)
    if mask.shape[1] == 1 and recon.shape[1] > 1:
        mask = mask.expand(-1, recon.shape[1], -1, -1, -1)
    denom = mask.sum().clamp_min(1.0)
    loss = (recon - target).abs() * mask
    return loss.sum() / denom


def linear_probe_eval(model: MedMamba3D, loader: DataLoader, device: torch.device, label_map: dict = None, max_samples: int = 512):
    """
    简化版线性探针（兼容 MedMamba3D 与 SS3M 包装）：
    - 目标：冻结编码器，对每个样本做全局平均得到特征，用一个临时线性层在二分类上评估。
    - 输入：loader.dataset 返回 (tensor, label_int)，label 取值 {0,1}。
    - 兼容性设计：
        * MedMamba3D：model.encoder(x) 返回 (tokens, grid)，取 tokens.mean(dim=1)
        * SS3MMAE/SS3MEncoder：优先使用 encoder.forward_tokens(x) 得到 [B,C,D,H,W]，在空间维做均值；
          若仅有 encoder(x) 返回 [B, embed_dim] 全局特征，则直接使用。
    设计原因：
        - 原函数仅适配 MedMamba3D，导致 SS3M 预训练阶段无法探针评估；此改造确保两种后端统一流程。
    """
    # 收集特征与标签
    feats, labels = [], []
    model.eval()

    label_map = label_map or {}
    label_tokens = sorted([(str(k).lower(), int(v)) for k, v in label_map.items()], key=lambda x: len(x[0]), reverse=True)

    def _to_label_tensor(batch_y):
        if torch.is_tensor(batch_y):
            return batch_y.long().cpu()
        if isinstance(batch_y, (list, tuple)):
            ys = []
            for item in batch_y:
                if isinstance(item, numbers.Integral):
                    ys.append(int(item))
                    continue
                if isinstance(item, str):
                    low = item.lower().replace('\\\\', '/')
                    hit = None
                    for tok, cls in label_tokens:
                        if f"/{tok}/" in low or low.endswith(f"/{tok}"):
                            hit = cls
                            break
                    ys.append(hit)
                    continue
                ys.append(None)
            valid_idx = [i for i, v in enumerate(ys) if v is not None]
            if not valid_idx:
                return None, None
            y_t = torch.tensor([ys[i] for i in valid_idx], dtype=torch.long)
            return y_t, valid_idx
        return None, None

    collected = 0
    for x, y in loader:
        if x is None:
            continue
        x = x.to(device)
        y_t, valid_idx = _to_label_tensor(y)
        if y_t is None:
            continue
        if valid_idx is not None:
            x = x[valid_idx]
        # 冻结编码器取特征，避免构建大计算图；仅线性头参与梯度更新。
        with torch.no_grad():
            if hasattr(model, "encoder"):
                enc = model.encoder
                if callable(getattr(enc, "forward_tokens", None)):
                    # SS3M：返回空间特征图 [B, C', D', H', W']，做 3D GAP
                    tokens = enc.forward_tokens(x)
                    feat = tokens.mean(dim=[2, 3, 4])  # (B, C')
                else:
                    # MedMamba3D：encoder(x) -> (tokens, grid)；或 SS3M：encoder(x) -> [B, C]
                    out = enc(x)
                    if isinstance(out, tuple) and len(out) == 2:
                        tokens, _ = out
                        feat = tokens.mean(dim=1)  # (B, C)
                    else:
                        feat = out  # 已是 [B, C]
            else:
                raise RuntimeError("model 缺少 encoder 属性，无法进行线性探针评估。")
        feats.append(feat.detach().cpu())
        labels.append(y_t)
        collected += int(y_t.numel())
        if max_samples > 0 and collected >= max_samples:
            break

    if not feats:
        return float("nan")
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0).long()

    # Remap arbitrary labels (e.g., {0,2}) to contiguous ids {0,1,...} for CE.
    uniq = torch.unique(labels)
    if uniq.numel() < 2:
        return float("nan")
    label_remap = {int(v.item()): i for i, v in enumerate(uniq)}
    labels = torch.tensor([label_remap[int(v.item())] for v in labels], dtype=torch.long)
    num_probe_classes = int(uniq.numel())

    # 训练线性分类器（按类频率加权，避免不平衡下塌缩到多数类）
    lin = nn.Linear(feats.shape[1], num_probe_classes)
    opt = torch.optim.SGD(lin.parameters(), lr=0.1, momentum=0.9)
    cls_counts = torch.bincount(labels, minlength=num_probe_classes).float()
    cls_weights = cls_counts.sum() / (cls_counts.clamp_min(1.0) * float(num_probe_classes))
    for _ in range(200):  # 小步数快速探针
        opt.zero_grad()
        logits = lin(feats)
        loss = F.cross_entropy(logits, labels, weight=cls_weights)
        loss.backward()
        opt.step()

    with torch.no_grad():
        logits = lin(feats)
        pred = logits.argmax(dim=1)
        acc = (pred == labels).float().mean().item()

        # Class-wise recall and balanced accuracy are more robust under severe imbalance.
        recalls = []
        supports = []
        for cls in range(num_probe_classes):
            cls_mask = (labels == cls)
            support = int(cls_mask.sum().item())
            supports.append(support)
            if support > 0:
                recall = ((pred[cls_mask] == cls).float().mean().item())
            else:
                recall = float("nan")
            recalls.append(recall)
        valid_recalls = [r for r in recalls if not math.isnan(r)]
        bal_acc = float(sum(valid_recalls) / len(valid_recalls)) if valid_recalls else float("nan")

    return {
        "acc": float(acc),
        "bal_acc": float(bal_acc),
        "class_recalls": recalls,
        "class_supports": supports,
        "num_classes": int(num_probe_classes),
    }


def init_training_logger(out_dir: str, cfg: dict):
    """
    函数用途：
    - 创建输出目录并初始化训练日志文件 training.log。
    - 在日志头部写入本次作业开始时间与关键性能参数配置，便于复现实验与对齐训练环境。

    设计原因与作用：
    - 将日志初始化与参数打印集中在一个函数，减少主流程的重复代码；
    - 每次运行都写入完整配置信息，方便后续绘制损失曲线与比较不同超参的影响。

    参数：
    - out_dir: 输出目录（将保存 checkpoint 与 training.log）
    - cfg: 已解析的 YAML 配置字典

    返回：
    - log_path: 日志文件的完整路径
    - log_fh: 已打开的文件句柄（文本写入模式）
    """
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "training.log")
    log_fh = open(log_path, "a", encoding="utf-8")

    # 作业开始时间
    start_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_fh.write(f"=== Training Start: {start_time_str} ===\n")

    # 关键性能参数（从 cfg 提取并记录）
    mae_cfg = cfg.get('mae', {})
    ds_cfg = cfg.get('dataset', {})
    model_cfg = cfg.get('model', {})
    ss3m_cfg = cfg.get('ss3m', {})
    paths_cfg = cfg.get('paths', {})

    log_fh.write("Config summary:\n")
    log_fh.write(f"  out_dir={paths_cfg.get('out_dir', './out')}\n")
    log_fh.write(f"  mae.batch_size={mae_cfg.get('batch_size', 1)}\n")
    log_fh.write(f"  mae.lr={mae_cfg.get('lr', 1e-4)}\n")
    log_fh.write(f"  mae.weight_decay={mae_cfg.get('weight_decay', 0.05)}\n")
    log_fh.write(f"  mae.epochs={mae_cfg.get('epochs', 50)}\n")
    log_fh.write(f"  mae.warmup_epochs={mae_cfg.get('warmup_epochs', 0)}\n")
    log_fh.write(f"  mae.grad_accum_steps={mae_cfg.get('grad_accum_steps', 1)}\n")
    log_fh.write(f"  mae.max_grad_norm={mae_cfg.get('max_grad_norm', 0.0)}\n")
    log_fh.write(f"  mae.mask_ratio_start={mae_cfg.get('mask_ratio_start', mae_cfg.get('mask_ratio', 0.75))}\n")
    log_fh.write(f"  mae.mask_ratio_end={mae_cfg.get('mask_ratio_end', mae_cfg.get('mask_ratio', 0.75))}\n")
    log_fh.write(f"  dataset.num_workers={ds_cfg.get('num_workers', 1)}\n")
    log_fh.write(f"  model.embed_dim={model_cfg.get('embed_dim', 128)}\n")
    log_fh.write(f"  model.depth={model_cfg.get('depth', 12)}\n")
    log_fh.write(f"  model.patch_size={tuple(model_cfg.get('patch_size', (16,16,16)))}\n")
    log_fh.write(f"  ss3m.embed_dim={ss3m_cfg.get('embed_dim', 96)}\n")
    log_fh.write(f"  ss3m.depth={ss3m_cfg.get('depth', 4)}\n")
    log_fh.write(f"  ss3m.patch_size={tuple(ss3m_cfg.get('patch_size', (2,2,2)))}\n")
    log_fh.write("=== End of Config ===\n")
    log_fh.flush()
    return log_path, log_fh


def _ensure_mamba_available():
    """
    函数用途：
    - 在使用包含 'mamba' 的序列分支前，检查 mamba_ssm 依赖是否已安装；若缺失则抛出明确错误。

    改动原因：
    - 修复 NameError: _ensure_mamba_available 未定义 的报错；
    - 在用户已安装 mamba-ssm 的环境中，import mamba_ssm 正常通过，不影响训练流程；
    - 若依赖缺失或安装在不同环境中，给出清晰的安装/环境提示。
    """
    try:
        import mamba_ssm  # noqa: F401
    except Exception as e:
        raise ImportError(
            "未检测到 mamba_ssm，请先安装：pip install mamba-ssm；"
            "如已安装仍报错，请确认当前 Python 环境与安装环境一致（例如 conda 环境、虚拟环境），并检查安装位置。\n"
            f"原始异常: {e}"
        )

def should_log_step(step: int, total_steps: int, log_every_steps: int) -> bool:
    """
    函数用途：
    - 控制训练过程中“步级日志”的输出频率，仅在满足条件时返回 True。

    设计原因与作用：
    - 频繁的 I/O（磁盘写入与控制台打印）会拖慢训练，改为按固定步间隔输出可显著降低开销；
    - 同时在本 epoch 的最后一个 step 强制输出，确保收尾信息完整。

    参数：
    - step: 当前训练步（从 1 开始）
    - total_steps: 本 epoch 的总步数（len(loader)）
    - log_every_steps: 步级日志输出的间隔（例如 500）

    返回：
    - 是否在当前 step 输出日志（True/False）
    """
    if log_every_steps <= 0:
        return False
    return (step % log_every_steps == 0) or (step == total_steps)

def should_log_heartbeat(step: int, total_steps: int, heartbeat_sec: int, last_hb_time: float, now_time: float) -> bool:
    """
    函数用途:
    - 判断是否需要进行“心跳”打印（按时间间隔），用于长时间没有步级日志时仍能持续输出训练进展。

    设计原因:
    - 当单步耗时很长或总步数非常多时，按步频打印可能间隔过久；
      按固定时间间隔打印可以保证日志的可见性与可监控性。

    参数:
    - step: 当前已处理的步数（从 1 开始）
    - total_steps: 本 epoch 的总步数（len(loader)）
    - heartbeat_sec: 心跳打印的时间间隔（秒）
    - last_hb_time: 上次心跳打印的时间戳（time.time()）
    - now_time: 当前时间戳（time.time()）

    返回:
    - bool: 是否触发心跳打印
    """
    if heartbeat_sec <= 0:
        return False
    return (now_time - last_hb_time) >= heartbeat_sec

def init_distributed():
    """
    函数用途:
    - 初始化分布式训练环境(DDP), 若未设置相关环境变量则回退为单进程训练。
    返回:
    - (ddp_enabled, local_rank, world_size)
      ddp_enabled: 是否启用分布式
      local_rank: 当前进程的本地 GPU 索引
      world_size: 总进程数
    设计原因:
    - 统一处理 torch.distributed 初始化, 避免 NameError 并提升健壮性。
    """
    import os
    import torch

    ddp_enabled = False
    local_rank = 0
    world_size = 1

    try:
        # 读取由 torchrun 或集群环境提供的环境变量
        rank = int(os.environ.get("RANK", "-1"))
        world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
        local_rank_env = int(os.environ.get("LOCAL_RANK", "-1"))

        if rank >= 0 and world_size_env > 1 and local_rank_env >= 0:
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            torch.distributed.init_process_group(backend=backend, rank=rank, world_size=world_size_env)
            ddp_enabled = True
            local_rank = local_rank_env
            world_size = world_size_env

            # 设置当前设备(仅在 CUDA 下需要)
            if torch.cuda.is_available():
                torch.cuda.set_device(local_rank)
    except Exception:
        # 任意初始化异常则回退单进程
        ddp_enabled = False
        local_rank = 0
        world_size = 1

    return ddp_enabled, local_rank, world_size

def main():
    """
    MAE / 重建式预训练入口（CPU/CUDA 兼容，健壮配置读取）
    改动原因：
        - 初始化日志句柄 log_fh：在首次写日志前调用 init_training_logger，修复 NameError。
        - 定义 out_dir：供日志与 checkpoint 保存使用，避免后续路径变量未定义。
        - 构造 DataLoader(loader)：补齐训练循环数据源，避免 NameError。
        - 其余已有的优化器、调度器、AMP 等逻辑保持不变。
    """
    cfg = load_config(os.path.join(os.path.dirname(__file__), 'config.yaml'))
    ddp_enabled, local_rank, world_size = init_distributed()
    device = torch.device(f'cuda:{local_rank}' if (ddp_enabled and torch.cuda.is_available()) else ('cuda' if torch.cuda.is_available() else 'cpu'))
    device_type = 'cuda' if device.type == 'cuda' else 'cpu'
    is_main = (not ddp_enabled) or (torch.distributed.get_rank() == 0 if ddp_enabled else True)

    if device_type == 'cuda':
        # Favor throughput in pretraining unless user explicitly needs strict determinism.
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    # 新增：资源监控辅助函数（内嵌定义，确保在调用前可用）
    def get_epoch_io_snapshot():
        """
        函数用途: 在每个 epoch 开始时获取系统与本进程的磁盘 IO 读写基线, 用于计算该 epoch 的 IO 增量。
        返回: (io_sys_start, io_proc_start)
          - io_sys_start: (read_bytes, write_bytes) 或 (None, None) 当不可用时
          - io_proc_start: (read_bytes, write_bytes) 或 (None, None) 当不可用时
        设计原因: 训练期间 IO 通常由 DataLoader 读盘造成, 用基线差值反映该 epoch 的累计读写量。
        """
        try:
            import psutil
            sys_io = psutil.disk_io_counters()
            proc_io = psutil.Process().io_counters()
            return (sys_io.read_bytes, sys_io.write_bytes), (proc_io.read_bytes, proc_io.write_bytes)
        except Exception:
            return (None, None), (None, None)

    def _query_gpu_stats():
        """
        函数用途: 查询当前 CUDA 设备的显存占用与利用率(若可用), 并返回设备名称。
        返回: dict, 包含:
          - index: 设备索引, 或 None
          - name: 设备名称, 或 None
          - mem_used: 已占用显存(字节), 或 None
          - mem_total: 总显存(字节), 或 None
          - util: GPU 利用率(%), 或 None
        设计原因: 训练瓶颈定位需要 GPU 显存与利用率; 在未安装 pynvml 时回退到 torch 的显存查询。
        """
        stats = {"index": None, "name": None, "mem_used": None, "mem_total": None, "util": None}
        try:
            import torch
            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                stats["index"] = idx
                stats["name"] = torch.cuda.get_device_name(idx)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    stats["mem_used"] = int(mem.used)
                    stats["mem_total"] = int(mem.total)
                    util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    stats["util"] = int(util_rates.gpu)
                    pynvml.nvmlShutdown()
                except Exception:
                    stats["mem_used"] = int(torch.cuda.memory_allocated(idx))
                    try:
                        stats["mem_total"] = int(torch.cuda.get_device_properties(idx).total_memory)
                    except Exception:
                        stats["mem_total"] = None
                    stats["util"] = None
        except Exception:
            pass
        return stats

    def _query_torch_cuda_mem():
        """
        函数用途:
        - 返回当前进程的 GPU 张量显存占用（allocated）与已保留的显存（reserved），单位为字节。
          当 CUDA 不可用时，返回 (None, None)。
    
        设计原因:
        - NVML 不可用时无法获得设备利用率与设备总显存，但我们仍然可以用 torch 的接口
          观察到“本进程张量”占用了多少显存，这更贴近训练的真实占用。
    
        返回:
        - (alloc_bytes, reserved_bytes): 两者均为 int 或 None
        """
        try:
            if torch.cuda.is_available():
                alloc = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                return int(alloc), int(reserved)
        except Exception:
            pass
        return None, None

    def report_resource_usage(io_sys_start, io_proc_start, is_main_flag, file_handle):
        """
        函数用途: 在每个 epoch 结束时输出资源占用汇总信息到控制台与日志文件(仅主进程):
          - GPU: 显存占用/总显存, 利用率(若可用), 设备名
          - CPU: 总体利用率, 当前进程常驻内存(RSS)
          - IO: 本 epoch 的系统与进程的读/写增量 (MB)
        参数:
          - io_sys_start: (read_bytes, write_bytes) 或 (None, None), 来自 epoch 开始的基线
          - io_proc_start: (read_bytes, write_bytes) 或 (None, None), 来自 epoch 开始的基线
          - is_main_flag: 是否为主进程, 仅主进程输出
          - file_handle: 日志文件句柄, 可为 None
        设计原因: 在不影响训练的前提下, 提供关键资源指标以定位瓶颈。
        """
        if not is_main_flag:
            return

        cpu_percent = None
        proc_rss = None
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.0)
            proc_rss = int(psutil.Process().memory_info().rss)
        except Exception:
            pass

        sys_delta = (None, None)
        proc_delta = (None, None)
        try:
            import psutil
            sys_now = psutil.disk_io_counters()
            proc_now = psutil.Process().io_counters()
            if io_sys_start[0] is not None:
                sys_delta = (
                    max(0, sys_now.read_bytes - io_sys_start[0]),
                    max(0, sys_now.write_bytes - io_sys_start[1]),
                )
            if io_proc_start[0] is not None:
                proc_delta = (
                    max(0, proc_now.read_bytes - io_proc_start[0]),
                    max(0, proc_now.write_bytes - io_proc_start[1]),
                )
        except Exception:
            pass

        gpu = _query_gpu_stats()

        def _fmt_bytes_mb(v):
            return f"{(v / (1024 ** 2)):.1f}MB" if v is not None else "n/a"

        msg_parts = []
        if gpu["mem_used"] is not None and gpu["mem_total"] is not None:
            used_gb = gpu["mem_used"] / (1024 ** 3)
            total_gb = gpu["mem_total"] / (1024 ** 3)
            util_s = f"{gpu['util']}%" if gpu["util"] is not None else "n/a"
            msg_parts.append(f"GPU[{gpu.get('index','?')} {gpu.get('name','?')}] mem={used_gb:.2f}GB/{total_gb:.2f}GB util={util_s}")
        else:
            msg_parts.append("GPU=unavailable")

        cpu_s = f"CPU={cpu_percent:.1f}%" if cpu_percent is not None else "CPU=n/a"
        rss_s = _fmt_bytes_mb(proc_rss) if proc_rss is not None else "proc_rss=n/a"
        msg_parts.append(f"{cpu_s} proc_rss={rss_s}")

        if sys_delta[0] is not None:
            msg_parts.append(f"IO(sys) read={_fmt_bytes_mb(sys_delta[0])} write={_fmt_bytes_mb(sys_delta[1])}")
        else:
            msg_parts.append("IO(sys)=n/a")
        if proc_delta[0] is not None:
            msg_parts.append(f"IO(proc) read={_fmt_bytes_mb(proc_delta[0])} write={_fmt_bytes_mb(proc_delta[1])}")
        else:
            msg_parts.append("IO(proc)=n/a")

        msg = "[Resource] " + "; ".join(msg_parts)
        print(msg)
        if file_handle is not None:
            try:
                file_handle.write(msg + "\n")
                file_handle.flush()
            except Exception:
                pass

    # 初始化日志（仅主进程）
    out_dir = cfg.get('paths', {}).get('out_dir', './out')
    log_path, log_fh = (None, None)
    if is_main:
        log_path, log_fh = init_training_logger(out_dir, cfg)

    def _build_ckpt_suffix(cfg_local: dict) -> str:
        ss3m_local = cfg_local.get('ss3m', {}) if isinstance(cfg_local.get('ss3m', {}), dict) else {}
        n_dirs = int(ss3m_local.get('n_dirs_train', 8))
        emb = int(ss3m_local.get('embed_dim', 96))
        return f"{n_dirs}dir_{emb}dim"

    ckpt_suffix = _build_ckpt_suffix(cfg)
    if is_main:
        print(f"[Ckpt] naming_suffix={ckpt_suffix}", flush=True)

    # 删除 CSV 清单读取与报错；仅使用 label_roots
    target_shape = tuple(cfg.get('input', {}).get('shape_dhw', (112, 112, 112)))
    
    sources = resolve_pretrain_sources(cfg)
    if not sources:
        raise KeyError("未找到可用的预训练数据来源：请在 config.yaml 的 paths.label_roots 中提供存在的目录。")

    from torch.utils.data import ConcatDataset
    # input.format: "nii"(默认, NIfTI) | "npy"(预卷 .npy，已归一化)
    # 若 config 未显式指定 format，自动按数据目录内容检测（含 .npy -> npy 模式）
    data_format = str(cfg.get('input', {}).get('format', '')).lower()
    if data_format not in ("npy", "nii"):
        _has_npy = any(
            any(fn.lower().endswith(".npy") for fn in os.listdir(p))
            for _kind, p in sources if _kind == "dir" and os.path.isdir(p)
        )
        data_format = "npy" if _has_npy else "nii"
        print(f"[Data] input.format 未指定，自动检测为 {data_format} ({_has_npy=})", flush=True)
    normalized_in = bool(cfg.get('input', {}).get('normalized', data_format == "npy"))
    file_exts = (".npy",) if data_format == "npy" else (".nii", ".nii.gz")
    dir_datasets = [
        # 目录模式：NIfTI 开启初始化强校验；.npy 预卷已归一化，跳过 spacing/zscore 校验
        MRIVolumeDataset(manifest_csv=p, path_column='ignored', target_shape=target_shape,
                         check_spacing=(data_format != "npy"),
                         validate_nifti=(data_format != "npy"),
                         file_exts=file_exts,
                         normalized=normalized_in)
        for kind, p in sources if kind == "dir"
    ]
    if not dir_datasets:
        raise RuntimeError("label_roots 未解析到任何有效目录或目录为空。")
    ds = ConcatDataset(dir_datasets)
    total_len = sum(len(d) for d in dir_datasets)
    print(f"[Data] 使用 paths.label_roots 加载 {len(dir_datasets)} 个目录，共 {total_len} 个样本。")

    # 新增：构造训练 DataLoader（修复后续 loader 未定义）——调优参数
    batch_size = cfg.get('mae', {}).get('batch_size', 1)
    ds_cfg = cfg.get('dataset', {})
    cfg_workers = ds_cfg.get('num_workers', None)
    auto_num_workers = bool(ds_cfg.get('auto_num_workers', False))
    reserve_cores = int(ds_cfg.get('reserve_cores', 2))
    enforce_worker_cap = bool(ds_cfg.get('enforce_worker_cap', True))

    try:
        cpu_visible = len(os.sched_getaffinity(0))
    except Exception:
        cpu_visible = int(os.cpu_count() or 4)
    suggested_max_workers = max(1, cpu_visible - max(0, reserve_cores))

    if auto_num_workers or (not isinstance(cfg_workers, int)):
        num_workers = suggested_max_workers
    else:
        num_workers = int(cfg_workers)
        if enforce_worker_cap:
            num_workers = min(num_workers, suggested_max_workers)

    if is_main:
        print(
            f"[DataLoader] workers={num_workers} (cfg={cfg_workers}, auto={auto_num_workers}, "
            f"cpu_visible={cpu_visible}, suggested_max={suggested_max_workers})",
            flush=True,
        )
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(not ddp_enabled),
        num_workers=num_workers,
        pin_memory=(device_type == 'cuda'),
        drop_last=True,
        collate_fn=collate_fn_skip_none,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4
    )
    # DDP 使用分布式采样器
    if ddp_enabled:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(ds, shuffle=True)
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=(device_type == 'cuda'),
            drop_last=True,
            collate_fn=collate_fn_skip_none,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4
        )

    # 解析数据源并构造数据集与加载器（略）
    in_chans = cfg.get('model', {}).get('in_chans', 1)
    embed_dim = cfg.get('model', {}).get('embed_dim', 128)
    depth = cfg.get('model', {}).get('depth', 12)
    patch_size = tuple(cfg.get('model', {}).get('patch_size', (16, 16, 16)))
    decoder_dim = cfg.get('mae', {}).get('decoder_dim', 256)
    proj_dim = cfg.get('contrast', {}).get('proj_dim', 256)
    num_classes = cfg.get('dataset', {}).get('num_classes', 2)
    # model = MedMamba3D(
    #     in_chans=in_chans, embed_dim=embed_dim, depth=depth, patch_size=patch_size,
    #     decoder_dim=decoder_dim, proj_dim=proj_dim, num_classes=num_classes
    # )
    
        # SS3M 编码器 + 轻量 MAE 解码包装（并行双支路：conv + mamba）
    in_channels = cfg.get('model', {}).get('in_chans', 1)
    embed_dim = cfg.get('ss3m', {}).get('embed_dim', 96)
    depth = cfg.get('ss3m', {}).get('depth', 4)
    patch_size = tuple(cfg.get('ss3m', {}).get('patch_size', (2, 2, 2)))
    dropout = cfg.get('ss3m', {}).get('dropout', 0.0)
    use_pos_emb = cfg.get('ss3m', {}).get('use_pos_emb', True)
    merge_type = cfg.get('ss3m', {}).get('merge_type', 'softmax')
    use_checkpoint = cfg.get('ss3m', {}).get('use_checkpoint', False)

    # 新增：集中解析 SS3M 超参，避免 NameError 并提供健壮默认
    def _parse_ss3m_hyperparams(cfg: dict):
        """
        函数用途:
        - 从配置中解析 SS3M 的序列分块长度 chunk_len 与训练期方向采样数 n_dirs_train，
          并进行类型转换与边界检查，返回可直接使用的整数值。

        设计原因:
        - 当前代码在构造 MedMambaSS3MEncoder 时直接使用 chunk_len/n_dirs_train，
          若未在作用域内定义将触发 NameError；
          将解析逻辑集中为函数，提升可读性与健壮性。

        返回:
        - (chunk_len, n_dirs_train)
          chunk_len: int，<0 回退为 0（禁用分块）
                    n_dirs_train: int，限制在 [1, 8]，默认 8（启用完整 8 向扫描）
        """
        # chunk_len 可在 cfg['ss3m'] 或顶层 cfg 中配置；默认 0 表示不分块
        raw_chunk = cfg.get('ss3m', {}).get('chunk_len', cfg.get('chunk_len', 0))
        try:
            chunk = int(raw_chunk)
        except Exception:
            chunk = 0
        if chunk < 0:
            chunk = 0

        # 训练期方向子集采样，默认 8；有效范围 1..8（SS3M 方向上限）
        raw_dirs = cfg.get('ss3m', {}).get('n_dirs_train', 8)
        try:
            n_dirs = int(raw_dirs)
        except Exception:
            n_dirs = 8
        if n_dirs <= 0:
            n_dirs = 1
        if n_dirs > 8:
            n_dirs = 8

        return chunk, n_dirs

    # 调用解析函数，确保后续使用的变量已定义
    chunk_len, n_dirs_train = _parse_ss3m_hyperparams(cfg)

    _ensure_mamba_available()  # 若未安装 mamba_ssm，这里会抛出明确错误
    branch_types = ('conv', 'mamba')
    if is_main:
        print(
            f"[SS3M-Pretrain] embed_dim={embed_dim} depth={depth} patch_size={patch_size} "
            f"branch_types={branch_types} n_dirs_train={n_dirs_train} use_checkpoint={use_checkpoint} "
            f"chunk_len={chunk_len}",
            flush=True,
        )

    encoder = MedMambaSS3MEncoder(
            in_channels=in_channels, embed_dim=embed_dim, depth=depth, patch_size=patch_size,
            dropout=dropout, use_pos_emb=use_pos_emb,
            merge_type=merge_type, use_checkpoint=use_checkpoint,
            branch_types=branch_types,
            chunk_len=chunk_len,
            n_dirs_train=n_dirs_train
        )
    model = SS3MMAE(encoder=encoder, in_channels=in_channels, patch_size=patch_size)

    model = model.to(device)

    # 作用说明：
    # - 若启用分布式训练，则用 DistributedDataParallel 包装模型；
    # - 无论是否启用 DDP，都统一提供 model_without_ddp 引用，保存权重时使用它的 state_dict()。
    if ddp_enabled:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == 'cuda' else None,
            output_device=local_rank if device.type == 'cuda' else None,
            find_unused_parameters=False
        )
    model_without_ddp = model.module if hasattr(model, "module") else model

    # 优化器与调度器（warmup + cosine）
    lr = cfg.get('mae', {}).get('lr', 1e-4)
    weight_decay = cfg.get('mae', {}).get('weight_decay', 0.05)
    epochs = cfg.get('mae', {}).get('epochs', 50)
    warmup_epochs = cfg.get('mae', {}).get('warmup_epochs', 0)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # AMP（CUDA）与混合精度
    amp_dtype_name = str(cfg.get('mae', {}).get('amp_dtype', 'bf16')).lower()
    amp_dtype = torch.bfloat16 if amp_dtype_name in ('bf16', 'bfloat16') else torch.float16
    scaler = torch.amp.GradScaler('cuda') if (device_type == 'cuda' and amp_dtype == torch.float16) else None
    amp_ctx = torch.amp.autocast('cuda', dtype=amp_dtype) if device_type == 'cuda' else contextlib.nullcontext()

    # 目的：初始化“最佳损失”记录变量，确保在后续比较/更新时同一作用域中已赋值，避免 UnboundLocalError。
    # 原因：若未在 main 作用域内先行赋值，在 if epoch_avg_loss < best_loss: 时会因未定义而抛错。
    best_loss = float('inf')

    # 新增：从配置读取周期性保存间隔（ckpt_every），默认每5个epoch保存一次
    # 原因：之前在保存逻辑中使用 ckpt_every，但未先定义，导致 NameError
    ckpt_every = int(cfg.get('mae', {}).get('ckpt_every', 5))

    # 渐进掩码与训练监控参数
    mr_start = cfg.get('mae', {}).get('mask_ratio_start', cfg.get('mae', {}).get('mask_ratio', 0.75))
    mr_end = cfg.get('mae', {}).get('mask_ratio_end', cfg.get('mae', {}).get('mask_ratio', 0.75))
    grad_accum_steps = int(cfg.get('mae', {}).get('grad_accum_steps', 1))
    max_grad_norm = float(cfg.get('mae', {}).get('max_grad_norm', 0.0))

    # Step-wise warmup + cosine via LambdaLR (avoids SequentialLR deprecation warnings).
    updates_per_epoch = max(1, math.ceil(len(loader) / max(1, grad_accum_steps)))
    total_updates = max(1, updates_per_epoch * max(1, int(epochs)))
    warmup_updates = max(0, int(warmup_epochs) * updates_per_epoch)

    def _lr_lambda(step_idx: int) -> float:
        if warmup_updates > 0 and step_idx < warmup_updates:
            return 0.1 + 0.9 * (float(step_idx) / float(max(1, warmup_updates)))
        progress = (step_idx - warmup_updates) / float(max(1, total_updates - warmup_updates))
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_lambda)

    # 记录数据规模与设备信息到日志（仅主进程）
    if is_main:
        total_len = len(ds)
        log_fh.write(f"[Data] datasets={len(dir_datasets)} total_samples={total_len} device={device_type} world_size={world_size} workers={num_workers}\n")
        log_fh.flush()

    best_epoch_loss = float('inf')
    best_probe_acc = 0.0
    best_probe_bal_acc = 0.0
    mae_cfg = cfg.get('mae', {})
    probe_every = int(mae_cfg.get('probe_every', 10))
    probe_max_samples = int(mae_cfg.get('probe_max_samples', 512))
    probe_metric = str(mae_cfg.get('probe_metric', 'bal_acc')).lower()
    if probe_metric not in ('acc', 'bal_acc'):
        probe_metric = 'bal_acc'
    label_map_probe = cfg.get('dataset', {}).get('folder_label_map', {})
    log_every_steps = int(mae_cfg.get('log_every_steps', 0))
    heartbeat_sec = int(mae_cfg.get('heartbeat_sec', 0))
    enable_resource_report = bool(mae_cfg.get('enable_resource_report', False))
    step_log_to_console = bool(mae_cfg.get('step_log_to_console', False))
    step_log_to_file = bool(mae_cfg.get('step_log_to_file', False))
    invalid_batch_log_every = int(mae_cfg.get('invalid_batch_log_every', 20))

    try:
        for epoch in range(epochs):
            if ddp_enabled:
                # 分布式采样器需在每个 epoch 设置不同的种子
                if 'sampler' in locals() and hasattr(sampler, 'set_epoch'):
                    sampler.set_epoch(epoch)

            # 新增：在 epoch 开始时记录 IO 基线（系统与本进程）
            io_sys_start, io_proc_start = get_epoch_io_snapshot()

            model.train()
            opt.zero_grad(set_to_none=True)
            cur_mask_ratio = compute_mask_ratio(epoch, epochs, mr_start, mr_end)

            # 本轮统计（用于保存最佳模型）
            epoch_loss_sum = 0.0
            epoch_step_count = 0

            # 日志步频、心跳参数与时间基线
            total_steps = len(loader)
            epoch_start_time = time.time()
            last_hb_time = epoch_start_time
            invalid_batch_count = 0
            nonfinite_loss_count = 0
            valid_step_count = 0

            for step, (x, _) in enumerate(loader, start=1):
                if x is None:  # 如果整个批次都无效，则跳过
                    invalid_batch_count += 1
                    if invalid_batch_log_every > 0 and (invalid_batch_count % invalid_batch_log_every == 0):
                        print(f"[Data] skipped invalid batch count={invalid_batch_count} at step={step}", flush=True)
                    continue

                x = x.to(device, non_blocking=True)
                x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
                with amp_ctx:
                    # 对齐 MAE 前向：MedMamba3D/SS3MMAE
                    recon, mask = model.forward_mae(x, mask_ratio=cur_mask_ratio)
                    loss = mae_loss(recon, x, mask)

                if not torch.isfinite(loss):
                    nonfinite_loss_count += 1
                    if invalid_batch_log_every > 0 and (nonfinite_loss_count % invalid_batch_log_every == 0):
                        print(
                            f"[Train] skip non-finite loss at step={step}, nonfinite_skips={nonfinite_loss_count}, "
                            f"invalid_data_skips={invalid_batch_count}",
                            flush=True,
                        )
                    opt.zero_grad(set_to_none=True)
                    continue

                # 累计损失用于 epoch 平均
                epoch_loss_sum += float(loss.item())
                epoch_step_count += 1
                valid_step_count += 1
                # 修复：去除冗余的步计数自增，避免错位影响日志频率
                # （enumerate 已提供正确的 step，无需再 += 1）

                loss_for_step = loss / max(1, grad_accum_steps)
                if scaler is not None:
                    scaler.scale(loss_for_step).backward()
                else:
                    loss_for_step.backward()

                if step % max(1, grad_accum_steps) == 0:
                    if scaler is not None:
                        scaler.unscale_(opt)
                        if max_grad_norm and max_grad_norm > 0.0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        scaler.step(opt)
                        scaler.update()
                    else:
                        if max_grad_norm and max_grad_norm > 0.0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                        opt.step()
                    opt.zero_grad(set_to_none=True)
                    sched.step()

                # 步级日志（仅主进程，降低频率 + flush）
                if is_main and should_log_step(step, total_steps, log_every_steps):
                    current_lr = opt.param_groups[0]['lr']
                    avg_step_time = (time.time() - epoch_start_time) / max(step, 1)
                    eta_min = (avg_step_time * max(total_steps - step, 0)) / 60.0
                    if step_log_to_file and log_fh is not None:
                        log_fh.write(f"[Step] epoch={epoch+1}/{epochs} step={step}/{total_steps} loss={loss.item():.6f} lr={current_lr:.6e} mask_ratio={cur_mask_ratio:.3f} avg_step_time={avg_step_time:.2f}s ETA={eta_min:.1f}m\n")
                    if step_log_to_console:
                        print(f"[Step] epoch={epoch+1}/{epochs} step={step}/{total_steps} loss={loss.item():.6f} lr={current_lr:.6e} mask_ratio={cur_mask_ratio:.3f} avg_step_time={avg_step_time:.2f}s ETA={eta_min:.1f}m", flush=True)

                # 新增：心跳打印（仅主进程，按秒触发），保证长时间也能看到进展
                now_t = time.time()
                if is_main and should_log_heartbeat(step, total_steps, heartbeat_sec, last_hb_time, now_t):
                    current_lr = opt.param_groups[0]['lr']
                    avg_step_time = (now_t - epoch_start_time) / max(step, 1)
                    eta_min = (avg_step_time * max(total_steps - step, 0)) / 60.0
                    hb_msg = (f"[Heartbeat] epoch={epoch+1}/{epochs} progress={step}/{total_steps} "
                              f"avg_step_time={avg_step_time:.2f}s ETA={eta_min:.1f}m lr={current_lr:.6e}")
                    if step_log_to_file and log_fh is not None:
                        log_fh.write(hb_msg + "\n")
                    if step_log_to_console:
                        print(hb_msg, flush=True)
                    last_hb_time = now_t

            # 计算并记录 epoch 平均损失
            def _log_epoch_summary(epoch_idx, epochs, avg_loss, is_main_flag, file_handle):
                """
                函数功能: 安全地记录每个 epoch 的平均损失到日志文件与控制台, 并避免在多进程或无文件句柄时产生错误。
                参数:
                - epoch_idx: 当前 epoch 的索引(从 0 开始)
                - epochs: 总 epoch 数
                - avg_loss: 该 epoch 的平均损失
                - is_main_flag: 是否为主进程(仅主进程打印与写文件)
                - file_handle: 文件日志句柄, 可为 None
                返回: 无
                """
                if is_main_flag:
                    if file_handle is not None:
                        file_handle.write(f"[Epoch] {epoch_idx+1}/{epochs} avg_loss={avg_loss:.6f}\n")
                    print(f"[Epoch] {epoch_idx+1}/{epochs} avg_loss={avg_loss:.6f}", flush=True)
            if epoch_step_count > 0:
                epoch_avg_loss = epoch_loss_sum / epoch_step_count
            else:
                epoch_avg_loss = float('nan')
            _log_epoch_summary(epoch, epochs, epoch_avg_loss, is_main, log_fh)

            if is_main:
                print(
                    f"[EpochStats] epoch={epoch+1} valid_steps={valid_step_count}/{total_steps} "
                    f"invalid_data_skips={invalid_batch_count} nonfinite_skips={nonfinite_loss_count}",
                    flush=True,
                )

            # 现有探针评估与常规 checkpoint 保持
            # 定位：主训练循环内，epoch 结束后、现有周期性保存之前
            if enable_resource_report:
                report_resource_usage(io_sys_start, io_proc_start, is_main, log_fh)

            # ---------- 3. 新增：覆盖保存最优模型 ----------
            # 只在主进程操作，避免 DDP 重复写盘
            if is_main:
                # 首次或发现更优损失时更新
                if (not math.isnan(epoch_avg_loss)) and (epoch_avg_loss < best_loss):
                    best_loss = epoch_avg_loss
                    best_state = {
                        'epoch': epoch + 1,
                        'model_state': model_without_ddp.state_dict(),
                        'optimizer_state': opt.state_dict(),
                        'scaler': scaler.state_dict() if scaler is not None else None,
                        'config': cfg,
                        'best_loss': best_loss,
                    }
                    best_pth = os.path.join(out_dir, f'best_mae_{ckpt_suffix}.pth')
                    torch.save(best_state, best_pth)
                    print(f'[Best]  epoch={epoch+1}  new best loss={best_loss:.6f}  -> {best_pth}')
                    if log_fh:
                        log_fh.write(f'[Best]  epoch={epoch+1}  new best loss={best_loss:.6f}\n')
                        log_fh.flush()

            # ---------- 4. 原有：每 ckpt_every 个 epoch 保存 ----------
            if is_main and probe_every > 0 and (((epoch + 1) % probe_every) == 0 or (epoch + 1) == epochs):
                probe_result = linear_probe_eval(
                    model_without_ddp,
                    loader,
                    device,
                    label_map=label_map_probe,
                    max_samples=probe_max_samples,
                )
                probe_acc = float(probe_result.get("acc", float("nan")))
                probe_bal_acc = float(probe_result.get("bal_acc", float("nan")))
                class_recalls = probe_result.get("class_recalls", [])
                class_supports = probe_result.get("class_supports", [])

                probe_msg = (
                    f"[Probe] epoch={epoch+1} metric={probe_metric} acc={probe_acc:.4f} "
                    f"bal_acc={probe_bal_acc:.4f} recalls={class_recalls} supports={class_supports} "
                    f"samples<={probe_max_samples}"
                )
                print(probe_msg, flush=True)
                if log_fh is not None:
                    log_fh.write(probe_msg + "\n")
                    log_fh.flush()

                selected_probe_metric = probe_bal_acc if probe_metric == 'bal_acc' else probe_acc
                if (not math.isnan(selected_probe_metric)) and (
                    (selected_probe_metric > best_probe_bal_acc and probe_metric == 'bal_acc')
                    or (selected_probe_metric > best_probe_acc and probe_metric == 'acc')
                ):
                    if probe_metric == 'bal_acc':
                        best_probe_bal_acc = selected_probe_metric
                    else:
                        best_probe_acc = selected_probe_metric
                    probe_state = {
                        'epoch': epoch + 1,
                        'model_state': model_without_ddp.state_dict(),
                        'optimizer_state': opt.state_dict(),
                        'scaler': scaler.state_dict() if scaler is not None else None,
                        'config': cfg,
                        'best_probe_acc': probe_acc,
                        'best_probe_bal_acc': probe_bal_acc,
                        'probe_metric': probe_metric,
                    }
                    probe_pth = os.path.join(out_dir, f'best_mae_probe_{ckpt_suffix}.pth')
                    torch.save(probe_state, probe_pth)
                    print(
                        f"[BestProbe] epoch={epoch+1} acc={probe_acc:.4f} bal_acc={probe_bal_acc:.4f} "
                        f"metric={probe_metric} -> {probe_pth}",
                        flush=True,
                    )
                    if log_fh is not None:
                        log_fh.write(
                            f"[BestProbe] epoch={epoch+1} acc={probe_acc:.4f} bal_acc={probe_bal_acc:.4f} "
                            f"metric={probe_metric}\n"
                        )
                        log_fh.flush()

            if (epoch + 1) % ckpt_every == 0 or (epoch + 1) == epochs:
                if is_main:
                    periodic_state = {
                        'epoch': epoch + 1,
                        'model_state': model_without_ddp.state_dict(),
                        'optimizer_state': opt.state_dict(),
                        'scaler': scaler.state_dict() if scaler is not None else None,
                        'config': cfg,
                        'loss': epoch_avg_loss,
                    }
                    periodic_pth = os.path.join(out_dir, f'mae_epoch_{epoch+1}_{ckpt_suffix}.pth')
                    torch.save(periodic_state, periodic_pth)
                    print(f'[Ckpt]  epoch={epoch+1}  saved  -> {periodic_pth}')
                    if log_fh:
                        log_fh.write(f'[Ckpt]  epoch={epoch+1}  saved  -> {periodic_pth}\n')
            if is_main and log_fh is not None:
                log_fh.flush()
    finally:
        # 分布式收尾（安全）
        def _destroy_ddp_if_initialized():
            """
            函数功能: 在分布式后端可用且已初始化时安全销毁进程组, 避免 ddp_enabled 未定义或未初始化导致的异常。
            返回: 无
            """
            try:
                import torch.distributed as dist
                if dist.is_available() and dist.is_initialized():
                    dist.destroy_process_group()
            except Exception:
                pass

        _destroy_ddp_if_initialized()


# 文件顶部新增：仅使用 label_roots 的数据来源解析
def resolve_pretrain_sources(cfg: dict) -> list:
    """
    函数用途：
    - 仅使用 config.yaml 中 paths.label_roots 指定的目录作为无标签预训练数据来源；
      每个目录递归扫描 .nii/.nii.gz 文件，合并为一个预训练集合。

    设计原因与作用：
    - 满足“彻底移除 CSV 清单逻辑，仅通过目录进行数据加载”的需求；
    - 集中路径解析逻辑，避免在 main() 中散落判断，提高可维护性与健壮性。

    返回说明：
    - 返回形如 [("dir", abs_path), ...] 的列表；
    - 若未找到任何有效目录，返回空列表（交由上层报错处理）。
    """
    sources = []
    lr = cfg.get('paths', {}).get('label_roots', {})
    if isinstance(lr, dict):
        for name, p in lr.items():
            if isinstance(p, str) and os.path.isdir(p):
                sources.append(("dir", os.path.abspath(p)))
    return sources


def compute_mask_ratio(cur_epoch: int, total_epochs: int, start: float, end: float) -> float:
    """
    函数用途：
    - 按训练进度在 [start, end] 之间线性插值当前 MAE 掩码比例。
    设计原因与作用：
    - MAE 在高掩码下（如 0.75）初期较难学习，采用“低→高”渐进掩码能加速前期收敛，同时保持最终的强掩码训练以提升表征能力。
    参数：
    - cur_epoch: 当前 epoch（从 0 开始）
    - total_epochs: 总训练轮数
    - start: 初始掩码比例（如 0.50）
    - end: 最终掩码比例（如 0.75）
    返回：
    - 当前 epoch 的掩码比例（介于 start 与 end）
    """
    if total_epochs <= 1:
        return end
    t = cur_epoch / float(total_epochs - 1)
    return start + (end - start) * t

# 新增：资源监控辅助函数
def get_epoch_io_snapshot():
    """
    函数用途: 在每个 epoch 开始时获取系统与本进程的磁盘 IO 读写基线, 用于计算该 epoch 的 IO 增量。
    返回: (io_sys_start, io_proc_start)
      - io_sys_start: (read_bytes, write_bytes) 或 (None, None) 当不可用时
      - io_proc_start: (read_bytes, write_bytes) 或 (None, None) 当不可用时
    设计原因: 训练期间 IO 通常由 DataLoader 读盘造成, 用基线差值反映该 epoch 的累计读写量。
    """
    try:
        import psutil
        sys_io = psutil.disk_io_counters()
        proc_io = psutil.Process().io_counters()
        return (sys_io.read_bytes, sys_io.write_bytes), (proc_io.read_bytes, proc_io.write_bytes)
    except Exception:
        return (None, None), (None, None)

def _query_gpu_stats():
    """
    函数用途: 查询当前 CUDA 设备的显存占用与利用率(若可用), 并返回设备名称。
    返回: dict, 包含:
      - index: 设备索引, 或 None
      - name: 设备名称, 或 None
      - mem_used: 已占用显存(字节), 或 None
      - mem_total: 总显存(字节), 或 None
      - util: GPU 利用率(%), 或 None
    设计原因: 训练瓶颈定位需要 GPU 显存与利用率; 在未安装 pynvml 时回退到 torch 的显存查询。
    """
    stats = {"index": None, "name": None, "mem_used": None, "mem_total": None, "util": None}
    try:
        import torch
        if torch.cuda.is_available():
            idx = torch.cuda.current_device()
            stats["index"] = idx
            stats["name"] = torch.cuda.get_device_name(idx)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                stats["mem_used"] = int(mem.used)
                stats["mem_total"] = int(mem.total)
                util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                stats["util"] = int(util_rates.gpu)
                pynvml.nvmlShutdown()
            except Exception:
                stats["mem_used"] = int(torch.cuda.memory_allocated(idx))
                try:
                    stats["mem_total"] = int(torch.cuda.get_device_properties(idx).total_memory)
                except Exception:
                    stats["mem_total"] = None
                stats["util"] = None
    except Exception:
        pass
    return stats

def _query_torch_cuda_mem():
    """
    函数用途:
    - 返回当前进程的 GPU 张量显存占用（allocated）与已保留的显存（reserved），单位为字节。
      当 CUDA 不可用时，返回 (None, None)。

    设计原因:
    - NVML 不可用时无法获得设备利用率与设备总显存，但我们仍然可以用 torch 的接口
      观察到“本进程张量”占用了多少显存，这更贴近训练的真实占用。

    返回:
    - (alloc_bytes, reserved_bytes): 两者均为 int 或 None
    """
    try:
        if torch.cuda.is_available():
            alloc = torch.cuda.memory_allocated()
            reserved = torch.cuda.memory_reserved()
            return int(alloc), int(reserved)
    except Exception:
        pass
    return None, None

def report_resource_usage(io_sys_start, io_proc_start, is_main_flag, file_handle):
    """
    函数用途: 在每个 epoch 结束时输出资源占用汇总信息到控制台与日志文件(仅主进程):
      - GPU: 设备显存与利用率（若 NVML 可用），以及本进程张量显存占用（allocated/reserved）
      - CPU: 总体利用率与本进程 RSS
      - IO: 本 epoch 的系统与进程的读/写增量 (MB)，系统级为 system-wide，仅供参考

    设计原因:
    - 增强日志的解释性：在 NVML 不可用时仍能得到张量显存占用；
      对系统级 IO 加上 system-wide 说明，避免被误解为训练进程写盘。
    """
    if not is_main_flag:
        return

    def _fmt_bytes_mb(b):
        try:
            return f"{b/1024/1024:.1f}MB"
        except Exception:
            return "n/a"

    msg_parts = []

    # GPU 设备级（NVML）与进程级张量显存
    gpu_stats = _query_gpu_stats()
    if gpu_stats is not None:
        name = gpu_stats.get("name", "GPU")
        total = gpu_stats.get("mem_total_bytes")
        used = gpu_stats.get("mem_used_bytes")
        util = gpu_stats.get("utilization")
        if total is not None and used is not None:
            gpu_part = f"GPU[0 {name}] mem={_fmt_bytes_mb(used)}/{_fmt_bytes_mb(total)}"
        else:
            gpu_part = f"GPU[0 {name}] mem=n/a"
        gpu_part += f" util={util if util is not None else 'n/a'}"
        msg_parts.append(gpu_part)
    else:
        msg_parts.append("GPU=n/a")

    # 新增：Torch 原生张量显存占用
    alloc, reserved = _query_torch_cuda_mem()
    if alloc is not None and reserved is not None:
        msg_parts.append(f"GPU(tensor_mem) alloc={_fmt_bytes_mb(alloc)} reserved={_fmt_bytes_mb(reserved)}")

    # CPU 及本进程 RSS
    try:
        import psutil
        cpu_pct = psutil.cpu_percent(interval=0.1)
        rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
        msg_parts.append(f"CPU={cpu_pct:.1f}% proc_rss={rss_mb:.1f}MB")
    except Exception:
        msg_parts.append("CPU=n/a proc_rss=n/a")

    # IO（系统级与进程级）
    try:
        import psutil
        io_sys_end = psutil.disk_io_counters()
        if io_sys_start and io_sys_end:
            sys_read_delta = max(0, io_sys_end.read_bytes - io_sys_start[0])
            sys_write_delta = max(0, io_sys_end.write_bytes - io_sys_start[1])
            msg_parts.append(f"IO(sys, system-wide) read={_fmt_bytes_mb(sys_read_delta)} write={_fmt_bytes_mb(sys_write_delta)}")
        else:
            msg_parts.append("IO(sys)=n/a")
    except Exception:
        msg_parts.append("IO(sys)=n/a")

    # 进程级 IO：不可用时显示为 n/a，避免误解
    try:
        import psutil
        proc = psutil.Process(os.getpid())
        io_proc_end = proc.io_counters()
        if io_proc_start and io_proc_end:
            proc_read_delta = max(0, io_proc_end.read_bytes - io_proc_start[0])
            proc_write_delta = max(0, io_proc_end.write_bytes - io_proc_start[1])
            # 某些环境下这两项可能一直为 0（计数器缺失），此时仍按数值打印，但通常你会看到 n/a 更合理
            msg_parts.append(f"IO(proc) read={_fmt_bytes_mb(proc_read_delta)} write={_fmt_bytes_mb(proc_write_delta)}")
        else:
            msg_parts.append("IO(proc)=n/a")
    except Exception:
        msg_parts.append("IO(proc)=n/a")

    msg = "[Resource] " + "; ".join(msg_parts)
    print(msg, flush=True)
    if file_handle is not None:
        try:
            file_handle.write(msg + "\n")
            file_handle.flush()
        except Exception:
            pass
    # ... existing code ...

    def _query_gpu_stats():
        """
        函数用途: 查询当前 CUDA 设备的显存占用与利用率(若可用), 并返回设备名称。
        返回: dict, 包含:
          - index: 设备索引, 或 None
          - name: 设备名称, 或 None
          - mem_used: 已占用显存(字节), 或 None
          - mem_total: 总显存(字节), 或 None
          - util: GPU 利用率(%), 或 None
        设计原因: 训练瓶颈定位需要 GPU 显存与利用率; 在未安装 pynvml 时回退到 torch 的显存查询。
        """
        stats = {"index": None, "name": None, "mem_used": None, "mem_total": None, "util": None}
        try:
            import torch
            if torch.cuda.is_available():
                idx = torch.cuda.current_device()
                stats["index"] = idx
                stats["name"] = torch.cuda.get_device_name(idx)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    stats["mem_used"] = int(mem.used)
                    stats["mem_total"] = int(mem.total)
                    util_rates = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    stats["util"] = int(util_rates.gpu)
                    pynvml.nvmlShutdown()
                except Exception:
                    stats["mem_used"] = int(torch.cuda.memory_allocated(idx))
                    try:
                        stats["mem_total"] = int(torch.cuda.get_device_properties(idx).total_memory)
                    except Exception:
                        stats["mem_total"] = None
                    stats["util"] = None
        except Exception:
            pass
        return stats

    def report_resource_usage(io_sys_start, io_proc_start, is_main_flag, file_handle):
        """
        函数用途: 在每个 epoch 结束时输出资源占用汇总信息到控制台与日志文件(仅主进程):
          - GPU: 显存占用/总显存, 利用率(若可用), 设备名
          - CPU: 总体利用率, 当前进程常驻内存(RSS)
          - IO: 本 epoch 的系统与进程的读/写增量 (MB)
        参数:
          - io_sys_start: (read_bytes, write_bytes) 或 (None, None), 来自 epoch 开始的基线
          - io_proc_start: (read_bytes, write_bytes) 或 (None, None), 来自 epoch 开始的基线
          - is_main_flag: 是否为主进程, 仅主进程输出
          - file_handle: 日志文件句柄, 可为 None
        设计原因: 在不影响训练的前提下, 提供关键资源指标以定位瓶颈。
        """
        if not is_main_flag:
            return

        cpu_percent = None
        proc_rss = None
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=0.0)
            proc_rss = int(psutil.Process().memory_info().rss)
        except Exception:
            pass

        sys_delta = (None, None)
        proc_delta = (None, None)
        try:
            import psutil
            sys_now = psutil.disk_io_counters()
            proc_now = psutil.Process().io_counters()
            if io_sys_start[0] is not None:
                sys_delta = (
                    max(0, sys_now.read_bytes - io_sys_start[0]),
                    max(0, sys_now.write_bytes - io_sys_start[1]),
                )
            if io_proc_start[0] is not None:
                proc_delta = (
                    max(0, proc_now.read_bytes - io_proc_start[0]),
                    max(0, proc_now.write_bytes - io_proc_start[1]),
                )
        except Exception:
            pass

        gpu = _query_gpu_stats()

        def _fmt_bytes_mb(v):
            return f"{(v / (1024 ** 2)):.1f}MB" if v is not None else "n/a"

        msg_parts = []
        if gpu["mem_used"] is not None and gpu["mem_total"] is not None:
            used_gb = gpu["mem_used"] / (1024 ** 3)
            total_gb = gpu["mem_total"] / (1024 ** 3)
            util_s = f"{gpu['util']}%" if gpu["util"] is not None else "n/a"
            msg_parts.append(f"GPU[{gpu.get('index','?')} {gpu.get('name','?')}] mem={used_gb:.2f}GB/{total_gb:.2f}GB util={util_s}")
        else:
            msg_parts.append("GPU=unavailable")

        cpu_s = f"CPU={cpu_percent:.1f}%" if cpu_percent is not None else "CPU=n/a"
        rss_s = _fmt_bytes_mb(proc_rss) if proc_rss is not None else "proc_rss=n/a"
        msg_parts.append(f"{cpu_s} proc_rss={rss_s}")

        if sys_delta[0] is not None:
            msg_parts.append(f"IO(sys) read={_fmt_bytes_mb(sys_delta[0])} write={_fmt_bytes_mb(sys_delta[1])}")
        else:
            msg_parts.append("IO(sys)=n/a")
        if proc_delta[0] is not None:
            msg_parts.append(f"IO(proc) read={_fmt_bytes_mb(proc_delta[0])} write={_fmt_bytes_mb(proc_delta[1])}")
        else:
            msg_parts.append("IO(proc)=n/a")

        msg = "[Resource] " + "; ".join(msg_parts)
        print(msg)
        if file_handle is not None:
            try:
                file_handle.write(msg + "\n")
                file_handle.flush()
            except Exception:
                pass
if __name__ == "__main__":
    main()