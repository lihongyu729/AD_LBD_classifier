import math
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed3D(nn.Module):
    """
    PatchEmbed3D
    作用：
        - 使用 3D 卷积将输入体素映射到指定的嵌入维度，并按 patch_size 做下采样/分块。
    参数：
        in_channels: 输入通道数（如 MRI 中可能为 1）
        embed_dim:   输出嵌入维度
        patch_size:  3D patch 的尺寸（D, H, W），用于 Conv3d 的 kernel/stride
        norm:        是否在输出后增加归一化层
    输入：
        x: 张量，形状 [B, C, D, H, W]
    输出：
        x_embed: 张量，形状 [B, embed_dim, D', H', W']，其中 D' = D//patch_size[0] 等
    设计原因：
        - 标准的 3D patch embedding，可将高维体素映射到更紧凑的表征空间，减少后续序列长度与计算。
    """
    def __init__(self, in_channels: int, embed_dim: int,
                 patch_size: Tuple[int, int, int] = (2, 2, 2),
                 norm: bool = True):
        super().__init__()
        self.proj = nn.Conv3d(in_channels, embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size,
                              padding=0, bias=False)
        self.norm = nn.LayerNorm(embed_dim) if norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, embed_dim, D', H', W']
        if self.norm is not None:
            # LayerNorm 期望最后一维为通道，这里将通道移到最后再移回
            B, C, D, H, W = x.shape
            x = x.permute(0, 2, 3, 4, 1).contiguous()  # [B, D, H, W, C]
            x = self.norm(x)
            x = x.permute(0, 4, 1, 2, 3).contiguous()  # [B, C, D, H, W]
        return x


class SeqSSM(nn.Module):
    """
    SeqSSM（序列 SSM 抽象层）
    作用：
        - 对输入的一维序列（从 3D 体素展开得到）进行长程依赖建模。
        - 默认使用轻量的“卷积 + 门控 + 残差 + FFN”替代（不依赖外部库），确保代码可直接运行。
        - 若你安装了 mamba_ssm 等库，可替换为真实的 Mamba/S6 实现。
    参数：
        dim:       序列的通道维（即嵌入维度）
        ssm_type:  'conv'（默认，可运行）或 'mamba'（需要外部库支持）
        depth:     堆叠的层数（影响表达能力与计算量）
        dropout:   随机失活概率
    输入：
        x_seq: 张量，形状 [B, L, C]，其中 L 为序列长度（D'*H'*W'）
    输出：
        y_seq: 张量，形状 [B, L, C]，与输入同形
    设计原因：
        - 抽象出序列建模层，便于在不改动 SS3M 框架的情况下替换不同的 SSM 内核（如 Mamba/S6）。
    """
    def __init__(self, dim: int, ssm_type: str = 'conv', depth: int = 2, dropout: float = 0.0, mamba_cfg: Optional[dict] = None):
        super().__init__()
        self.dim = dim
        self.ssm_type = ssm_type
        self.depth = depth
        self.dropout = nn.Dropout(dropout)
        self.mamba_cfg = mamba_cfg or {}

        if ssm_type == 'conv':
            blocks = []
            for _ in range(depth):
                # 深度可分离卷积 1D（按通道独立），模拟顺序相关的局部滤波
                blocks.append(nn.Conv1d(in_channels=dim, out_channels=dim * 2,
                                        kernel_size=1, padding=0, bias=False))
                
                # 门控（GLU）以提高稳定性与表达
                blocks.append(nn.GLU(dim=1))
            self.conv_seq = nn.Sequential(*blocks)
            # FFN
            self.ffn = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
            )
            self.norm = nn.LayerNorm(dim)
        elif ssm_type == 'mamba':
            # 接入 mamba_ssm 的真实 Mamba 实现（支持轻量配置）
            try:
                from mamba_ssm import Mamba
            except ImportError:
                raise ImportError("未检测到 mamba_ssm，请先安装：pip install mamba_ssm")
            
            d_state = int(self.mamba_cfg.get('d_state', 16))
            d_conv = int(self.mamba_cfg.get('d_conv', 4))
            expand = int(self.mamba_cfg.get('expand', 2))

            self.conv_seq = Mamba(
                d_model=dim,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.norm = nn.LayerNorm(dim)
        else:
            raise ValueError(f"Unsupported ssm_type: {self.ssm_type}")

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        # x_seq: [B, L, C]
        if self.ssm_type == 'conv':
            y = self.norm(x_seq)
            y = y.transpose(1, 2)            # [B, C, L]
            y = self.conv_seq(y)              # [B, C, L]
            y = y.transpose(1, 2)            # [B, L, C]
            y = self.dropout(y)
            x_seq = x_seq + y
            y2 = self.ffn(self.norm(x_seq))
            y2 = self.dropout(y2)
            x_seq = x_seq + y2
            return x_seq
        elif self.ssm_type == 'mamba':
            # Mamba 分支：Pre-Norm + 残差
            res = x_seq
            y = self.norm(x_seq)              # [B, L, C]
            y = self.conv_seq(y)              # Mamba 前向，输出 [B, L, C]
            y = self.dropout(y)
            return res + y
        else:
            raise ValueError(f"Unsupported ssm_type: {self.ssm_type}")


class SS3MBlock3DDual(nn.Module):
    """
    SS3MBlock3DDual（并行双支路 3D 块）
    作用：
        - 同时运行两个序列建模支路（如 'conv' 与 'mamba'，或两个 'conv'），各自完成 8 路翻转方向增强与序列处理；
        - 在体素空间对两条支路输出进行可学习的 softmax 加权融合；
        - 残差连接：返回 input + fused。

    设计原因：
        - 对齐“两个支路都走然后特征融合”的结构设想，替代原先的二选一 ssm_type 分支；
        - 保留原 SS3M 的 8 方向增强，同时引入支路层面的融合权重以自适应不同支路贡献。

    参数：
        dim:           通道维（嵌入维度）
        dropout:       随机失活概率
        use_pos_emb:   是否使用位置编码
        merge_type:    8 方向融合方式：'softmax' 或 'avg'
        branch_types:  两个分支的类型元组，如 ('conv','mamba') 或 ('conv','conv')
        use_checkpoint:是否在序列前向上使用梯度检查点
    """
    def __init__(self, dim: int, dropout: float = 0.0,
                 use_pos_emb: bool = True, merge_type: str = 'softmax',
                 branch_types: Tuple[str, str] = ('conv', 'conv'),
                 use_checkpoint: bool = False,
                 mamba_cfg: Optional[dict] = None, chunk_len: int = -1,
                 n_dirs_train: int = 8, use_dual_branch: bool = True):
        super().__init__()
        self.dim = dim
        self.use_pos_emb = use_pos_emb
        self.merge_type = merge_type
        self.use_checkpoint = use_checkpoint
        self.mamba_cfg = mamba_cfg or {}
        self._mamba_chunk_len = int(chunk_len) if chunk_len > 0 else -1
        self.n_dirs_train = max(1, min(int(n_dirs_train), 8))
        self.use_dual_branch = bool(use_dual_branch)

        # Pre-Norm（对每个体素通道维做 LayerNorm）
        self.norm = nn.LayerNorm(dim)

        # 两个序列分支（若未安装 mamba_ssm 而选择了 'mamba'，自动回退为 'conv'）
        self.ssm_a = SeqSSM(dim=dim, ssm_type=branch_types[0], depth=2, dropout=dropout, mamba_cfg=self.mamba_cfg)
        if self.use_dual_branch:
            try:
                self.ssm_b = SeqSSM(dim=dim, ssm_type=branch_types[1], depth=2, dropout=dropout, mamba_cfg=self.mamba_cfg)
            except ImportError:
                self.ssm_b = SeqSSM(dim=dim, ssm_type='conv', depth=2, dropout=dropout, mamba_cfg=self.mamba_cfg)
        else:
            self.ssm_b = None

        # 8 方向融合权重（与单支路块一致）
        if merge_type == 'softmax':
            self.dir_logits = nn.Parameter(torch.zeros(self.n_dirs_train))
        else:
            self.register_parameter('dir_logits', None)

        # 支路融合权重（2 分支 softmax）
        if self.use_dual_branch:
            self.branch_logits = nn.Parameter(torch.zeros(2))
        else:
            self.register_parameter('branch_logits', None)

        # 简易可学习位置编码
        self.pos_emb = None

    @staticmethod
    def _flip_dims() -> List[Tuple[bool, bool, bool]]:
        return [
            (False, False, False),
            (True, False, False),
            (False, True, False),
            (False, False, True),
            (True, True, False),
            (True, False, True),
            (False, True, True),
            (True, True, True),
        ]

    @staticmethod
    def _generate_8_flips(x: torch.Tensor, dims: List[Tuple[bool, bool, bool]]) -> List[torch.Tensor]:
        """
        生成 8 个方向的翻转张量。
        翻转模式（与三维坐标 D/H/W 的二元选择对应）：
            000, 100, 010, 001, 110, 101, 011, 111
        """
        flips = []
        B, C, D, H, W = x.shape
        flips = []
        for fd, fh, fw in dims:
            fx = x
            if fd:
                fx = torch.flip(fx, dims=[2])
            if fh:
                fx = torch.flip(fx, dims=[3])
            if fw:
                fx = torch.flip(fx, dims=[4])
            flips.append(fx)
        return flips

    @staticmethod
    def _reverse_8_flips(xs: List[torch.Tensor], dims: List[Tuple[bool, bool, bool]]) -> List[torch.Tensor]:
        """
        对 8 路输出做“反向翻转”，将其还原到与原始坐标一致的方向。
        注意：反向翻转与正向翻转是同一操作（flip 的自反性），因此直接按对应维度 flip 即可。
        """
        outs = []
        for (fd, fh, fw), x in zip(dims, xs):
            fx = x
            if fd:
                fx = torch.flip(fx, dims=[2])
            if fh:
                fx = torch.flip(fx, dims=[3])
            if fw:
                fx = torch.flip(fx, dims=[4])
            outs.append(fx)
        return outs

    @staticmethod
    def _vol_to_seq(x: torch.Tensor) -> torch.Tensor:
        """
        将 [B, C, D, H, W] 序列化为 [B, L, C]，其中 L = D*H*W。
        使用 D-H-W 的行优先（row-major）顺序。
        """
        B, C, D, H, W = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous()  # [B, D, H, W, C]
        x = x.view(B, D * H * W, C)                # [B, L, C]
        return x

    @staticmethod
    def _seq_to_vol(x_seq: torch.Tensor, shape_3d: Tuple[int, int, int]) -> torch.Tensor:
        """
        将 [B, L, C] 还原为 [B, C, D, H, W]。
        """
        B, L, C = x_seq.shape
        D, H, W = shape_3d
        x = x_seq.view(B, D, H, W, C).contiguous()  # [B, D, H, W, C]
        x = x.permute(0, 4, 1, 2, 3).contiguous()   # [B, C, D, H, W]
        return x

    def _maybe_pos_emb(self, x: torch.Tensor) -> torch.Tensor:
        """
        函数用途：
        - 若启用位置编码，将模块内的 self.pos_emb 移动到与输入 x 相同的设备后再相加，避免设备不一致错误。
    
        改动原因：
        - 你当前的报错源自 self.pos_emb 位于 CPU 而 x 位于 CUDA，导致 RuntimeError；
        - 在这里进行按需搬移（to(x.device)），即便 pos_emb 初始化在 CPU，也可在 CUDA 运行时安全使用。
        """
    def _maybe_pos_emb(self, x: torch.Tensor) -> torch.Tensor:
        """
        修复设备不一致报错，并确保参数在正确的设备上初始化。
        """
        if not self.use_pos_emb:
            return x
            
        B, C, D, H, W = x.shape
        
        # 1. 如果 pos_emb 还没初始化，或者输入尺寸变了，则重新创建
        if self.pos_emb is None or self.pos_emb.shape != (1, C, D, H, W):
            # 直接在输入 x 所在的设备上创建张量，避免后续搬运
            temp_emb = torch.zeros(1, C, D, H, W, device=x.device)
            nn.init.trunc_normal_(temp_emb, std=0.02)
            # 注册为 Parameter
            self.pos_emb = nn.Parameter(temp_emb)
            
        # 2. 核心修复：如果 x 在 GPU 而 pos_emb 还在 CPU（或反之），强制同步
        if self.pos_emb.device != x.device:
            self.pos_emb.data = self.pos_emb.data.to(x.device)
            
        return x + self.pos_emb

    def _direction_merge(self, vols_8: List[torch.Tensor]) -> torch.Tensor:
        """
        8 方向融合：softmax 可学习权重或简单平均
        """
        stacked = torch.stack(vols_8, dim=1)
        if self.merge_type == 'softmax' and self.dir_logits is not None:
            w = torch.softmax(self.dir_logits, dim=0).view(1, len(vols_8), 1, 1, 1, 1)
            return (stacked * w).sum(dim=1)
        else:
            return stacked.mean(dim=1)

    def _mamba_forward_in_chunks(self, ssm_branch, x_seq: torch.Tensor) -> torch.Tensor:
        """
        对序列进行分块前向，降低 Mamba 的显存占用。
        x_seq: (B, L, D)
        """
        B, L, D = x_seq.shape
        # This function is only called when self._mamba_chunk_len > 0
        if L <= self._mamba_chunk_len:
            # Sequence is shorter than a chunk, process directly
            if self.use_checkpoint and x_seq.requires_grad:
                return torch.utils.checkpoint.checkpoint(ssm_branch, x_seq, use_reentrant=False)
            else:
                return ssm_branch(x_seq)
        
        # Split into chunks
        chunks = torch.split(x_seq, self._mamba_chunk_len, dim=1)
        output_chunks = []
        for chunk in chunks:
            if self.use_checkpoint and chunk.requires_grad:
                output_chunk = torch.utils.checkpoint.checkpoint(ssm_branch, chunk, use_reentrant=False)
            else:
                output_chunk = ssm_branch(chunk)
            output_chunks.append(output_chunk)
        
        return torch.cat(output_chunks, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, D, H, W = x.shape
        input_res = x

        # Pre-Norm + 位置编码
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self._maybe_pos_emb(x)

        # --- 优化点：并行处理，用空间换时间 ---
        # 1) 生成方向翻转并转换为序列（可通过 n_dirs_train 降低自由度）
        dims_subset = self._flip_dims()[:self.n_dirs_train]
        xs_8 = self._generate_8_flips(x, dims_subset)
        feat_seqs = [self._vol_to_seq(xi) for xi in xs_8] # list of [B, L, C]
        
        # 将方向分支在 Batch 维度拼接
        feat_seqs_batch = torch.cat(feat_seqs, dim=0)

        # 处理分支 A
        if self.ssm_a.ssm_type == 'mamba' and self._mamba_chunk_len > 0:
            out_seq_a_batch = self._mamba_forward_in_chunks(self.ssm_a, feat_seqs_batch)
        else:
            if self.use_checkpoint and feat_seqs_batch.requires_grad:
                out_seq_a_batch = torch.utils.checkpoint.checkpoint(self.ssm_a, feat_seqs_batch, use_reentrant=False)
            else:
                out_seq_a_batch = self.ssm_a(feat_seqs_batch)
        
        # 拆分回方向并转回体素
        out_seqs_a = torch.chunk(out_seq_a_batch, len(xs_8), dim=0)
        vols_8_a = [self._seq_to_vol(seq, (D, H, W)) for seq in out_seqs_a]
        vols_8_a = self._reverse_8_flips(vols_8_a, dims_subset)
        y_a = self._direction_merge(vols_8_a)

        if not self.use_dual_branch or self.ssm_b is None:
            fused = y_a
        else:
            if self.ssm_b.ssm_type == 'mamba' and self._mamba_chunk_len > 0:
                out_seq_b_batch = self._mamba_forward_in_chunks(self.ssm_b, feat_seqs_batch)
            else:
                if self.use_checkpoint and feat_seqs_batch.requires_grad:
                    out_seq_b_batch = torch.utils.checkpoint.checkpoint(self.ssm_b, feat_seqs_batch, use_reentrant=False)
                else:
                    out_seq_b_batch = self.ssm_b(feat_seqs_batch)

            out_seqs_b = torch.chunk(out_seq_b_batch, len(xs_8), dim=0)
            vols_8_b = [self._seq_to_vol(seq, (D, H, W)) for seq in out_seqs_b]
            vols_8_b = self._reverse_8_flips(vols_8_b, dims_subset)
            y_b = self._direction_merge(vols_8_b)

            bw = torch.softmax(self.branch_logits, dim=0).view(2, 1, 1, 1, 1)
            fused = bw[0] * y_a + bw[1] * y_b

        return input_res + fused


class MedMambaSS3M(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 2,
                 embed_dim: int = 96, depth: int = 4,
                 patch_size: Tuple[int, int, int] = (2, 2, 2),
                 ssm_type: str = 'conv', dropout: float = 0.0,
                 use_pos_emb: bool = True, merge_type: str = 'softmax',
                 use_checkpoint: bool = False, branch_types: Tuple[str, str] = ('conv', 'conv'),
                 n_dirs_train: int = 8, use_dual_branch: bool = True):
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, patch_size=patch_size, norm=True)
        self.blocks = nn.ModuleList([
            SS3MBlock3DDual(dim=embed_dim,
                            dropout=dropout,
                            use_pos_emb=use_pos_emb,
                            merge_type=merge_type,
                            branch_types=branch_types,
                            use_checkpoint=use_checkpoint,
                            n_dirs_train=n_dirs_train,
                            use_dual_branch=use_dual_branch)
            for _ in range(depth)
        ])
        self.norm = nn.BatchNorm3d(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.classifier = self.head

    def forward_encoder(self, x: torch.Tensor):
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        b, c, d, h, w = x.shape
        tokens = x.permute(0, 2, 3, 4, 1).contiguous().view(b, d * h * w, c)
        return tokens, (d, h, w)

    def forward_classifier(self, x: torch.Tensor):
        tokens, _ = self.forward_encoder(x)
        feat = tokens.mean(dim=1)
        return self.head(feat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向：
            - 输入形状 [B, C, D, H, W]
            - 输出 [B, num_classes]
        """
        return self.forward_classifier(x)


class MedMambaSS3MEncoder(nn.Module):
    """
    MedMambaSS3MEncoder（SS3M 编码器版）
    作用：
        - 仅输出全局特征向量，便于接入 MAE 解码器或线性探针。
    参数：
        in_channels, embed_dim, depth, patch_size, ssm_type, dropout, use_pos_emb, merge_type, use_checkpoint：
            与 MedMambaSS3M 一致，但不含分类头。
        decoder_dim:
            为 MAE 等下游任务保留的接口（这里不直接使用，只暴露 out_channels）。
    输出：
        feats: [B, embed_dim]
    """
    def __init__(self, in_channels: int = 1, embed_dim: int = 96, depth: int = 4,
                 patch_size: Tuple[int, int, int] = (2, 2, 2),
                 ssm_type: str = 'conv', dropout: float = 0.0,
                 use_pos_emb: bool = True, merge_type: str = 'softmax',
                 use_checkpoint: bool = False, decoder_dim: int = 256,
                 branch_types: Tuple[str, str] = ('conv', 'conv'),
                 chunk_len: int = -1,
                 n_dirs_train: int = 8,
                 use_dual_branch: bool = True):
        """
        函数用途:
        - SS3M 编码器初始化，新增对 chunk_len 与 n_dirs_train 的兼容。
          - chunk_len: 启用 Mamba 序列分块前向（>0 时生效），降低显存、提升吞吐。
          - n_dirs_train: 训练期方向子集采样的兼容参数（当前不在本类使用，保留以兼容上层调用）。

        设计原因:
        - 训练脚本已向 MedMambaSS3MEncoder 传入 chunk_len/n_dirs_train，原签名未包含导致 TypeError；
          增加可选参数并向下透传 chunk_len，可立即消除此类报错并启用性能优化。
        """
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, patch_size=patch_size, norm=True)
        # 记录兼容参数（当前仅 chunk_len 会往下传给模块块；n_dirs_train 先保留以兼容上层）
        self._mamba_chunk_len = int(chunk_len) if chunk_len and chunk_len > 0 else -1
        self._n_dirs_train = int(n_dirs_train) if n_dirs_train and n_dirs_train > 0 else 8

        # 使用并行双支路块替代原先的二选一块（传入 chunk_len 以启用分块前向）
        self.blocks = nn.ModuleList([
            SS3MBlock3DDual(dim=embed_dim,
                            dropout=dropout,
                            use_pos_emb=use_pos_emb,
                            merge_type=merge_type,
                            branch_types=branch_types,
                            use_checkpoint=use_checkpoint,
                            chunk_len=self._mamba_chunk_len,
                            n_dirs_train=n_dirs_train,
                            use_dual_branch=use_dual_branch)
            for _ in range(depth)
        ])
        self.norm = nn.BatchNorm3d(embed_dim)
        self.out_channels = embed_dim  # 暴露编码维度

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向：
            - 输入 [B, C, D, H, W]
            - 输出 [B, embed_dim] 作为全局特征
        """
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        feats = x.mean(dim=[2, 3, 4])  # GAP -> [B, embed_dim]
        return feats

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        功能：返回空间 token 特征图，形状 [B, C', D', H', W']。
        改动原因：为 MAE/重建式预训练提供解码器输入。
        """
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


class MedMambaSS3MContrast(nn.Module):
    """
    MedMambaSS3MContrast（SS3M 对比预训练版）
    作用：
        - 在编码器输出的全局特征上附加投影头 ProjectionHead，用于 SimCLR/MoCo 等对比学习。
    参数：
        与编码器相同，新增：
        proj_dim: 投影维度，用于对比损失空间
    输出：
        z: [B, proj_dim]，用于对比损失
    """
    def __init__(self, in_channels: int = 1, embed_dim: int = 96, depth: int = 4,
                 patch_size: Tuple[int, int, int] = (2, 2, 2),
                 ssm_type: str = 'conv', dropout: float = 0.0,
                 use_pos_emb: bool = True, merge_type: str = 'softmax',
                 use_checkpoint: bool = False, proj_dim: int = 128):
        super().__init__()
        self.encoder = MedMambaSS3MEncoder(in_channels=in_channels, embed_dim=embed_dim, depth=depth,
                                           patch_size=patch_size, ssm_type=ssm_type, dropout=dropout,
                                           use_pos_emb=use_pos_emb, merge_type=merge_type,
                                           use_checkpoint=use_checkpoint)
        # 简单投影头：MLP + L2 归一化（与 SimCLR/MoCo 常见设置一致）
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向：
            - 输入 [B, C, D, H, W]
            - 输出 [B, proj_dim]，用于对比学习损失
        """
        feats = self.encoder(x)           # [B, embed_dim]
        z = self.proj(feats)              # [B, proj_dim]
        z = F.normalize(z, dim=1)         # L2 归一化，提升对比稳定性
        return z

    def forward_projection(self, x: torch.Tensor) -> torch.Tensor:
        """
        对比学习脚本调用的别名方法，与 forward 保持一致。
        """
        return self.forward(x)


if __name__ == "__main__":
    # 简单自测：随机 3D 体（例如 MRI 体）
    B, C, D, H, W = 2, 1, 32, 64, 64
    num_classes = 3
    x = torch.randn(B, C, D, H, W)

    model = MedMambaSS3M(in_channels=C, num_classes=num_classes,
                         embed_dim=64, depth=3, patch_size=(2, 2, 2),
                         ssm_type='conv', dropout=0.1,
                         use_pos_emb=True, merge_type='softmax')
    with torch.no_grad():
        y = model(x)
    print("output shape:", y.shape)  # 期望 [B, num_classes]
