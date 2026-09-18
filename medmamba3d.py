import math
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbed3D(nn.Module):
    def __init__(self, in_chans: int = 1, embed_dim: int = 128, patch_size: Tuple[int, int, int] = (16, 16, 16)):
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x: torch.Tensor):
        # x: (B, C=1, D=112, H=112, W=112)
        x = self.proj(x)  # (B, C=embed, D', H', W')
        B, C, Dp, Hp, Wp = x.shape
        x = x.view(B, C, Dp * Hp * Wp).transpose(1, 2)  # (B, N, C), N = Dp*Hp*Wp
        return x, (Dp, Hp, Wp)


class DepthwiseConv3D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3, padding: int = 1, bias: bool = True):
        super().__init__()
        self.dw = nn.Conv3d(channels, channels, kernel_size=kernel_size, padding=padding, groups=channels, bias=bias)

    def forward(self, x_3d: torch.Tensor):
        return self.dw(x_3d)


class SSMBlock3D(nn.Module):
    """
    轻量化的MedMamba风格块：
    - LayerNorm（token维）
    - 线性扩展 + 3D depthwise conv混合（在体素空间）
    - GLU式门控（SiLU）
    - 残差
    """
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.SiLU()
        self.dwconv = DepthwiseConv3D(hidden, kernel_size=3, padding=1, bias=True)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x_tokens: torch.Tensor, grid_shape: Tuple[int, int, int]):
        # x_tokens: (B, N, C), grid_shape=(D',H',W')
        residual = x_tokens
        B, N, C = x_tokens.shape
        Dp, Hp, Wp = grid_shape

        x = self.norm(x_tokens)
        x = self.fc1(x)  # (B, N, hidden)
        x = self.act(x)

        # reshape到3D做空间混合
        x3d = x.transpose(1, 2).contiguous().view(B, -1, Dp, Hp, Wp)  # (B, hidden, D', H', W')
        x3d = self.dwconv(x3d)
        x = x3d.view(B, -1, N).transpose(1, 2).contiguous()  # 回到(B, N, hidden)

        x = self.fc2(x)
        x = self.drop(x)
        x = x + residual
        return x


class MedMamba3DEncoder(nn.Module):
    def __init__(self, in_chans: int = 1, embed_dim: int = 128, depth: int = 12, patch_size: Tuple[int, int, int] = (16, 16, 16), mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.patch = PatchEmbed3D(in_chans=in_chans, embed_dim=embed_dim, patch_size=patch_size)
        self.blocks = nn.ModuleList([SSMBlock3D(embed_dim, mlp_ratio=mlp_ratio, drop=drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        tokens, grid_shape = self.patch(x)  # (B, N, C), grid
        for blk in self.blocks:
            tokens = blk(tokens, grid_shape)
        tokens = self.norm(tokens)
        return tokens, grid_shape  # tokens: (B, N, C)


class MAEDecoder3D(nn.Module):
    """
    MAE解码器：将token重建为patch体素，再拼接为完整体积。
    """
    def __init__(self, embed_dim: int = 128, decoder_dim: int = 256, patch_size: Tuple[int, int, int] = (16, 16, 16)):
        super().__init__()
        self.patch_size = patch_size
        pv = patch_size[0] * patch_size[1] * patch_size[2]
        self.proj_in = nn.Linear(embed_dim, decoder_dim)
        self.act = nn.GELU()
        self.block = nn.Sequential(
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU(),
            nn.Linear(decoder_dim, decoder_dim),
            nn.GELU()
        )
        self.proj_out = nn.Linear(decoder_dim, pv)

    def forward(self, tokens: torch.Tensor, grid_shape: Tuple[int, int, int]):
        # tokens: (B, N, C)
        B, N, C = tokens.shape
        Dp, Hp, Wp = grid_shape
        psD, psH, psW = self.patch_size

        x = self.proj_in(tokens)
        x = self.act(x)
        x = self.block(x)
        x = self.proj_out(x)  # (B, N, pv)
        x = x.view(B, N, psD, psH, psW)

        # 把patch网格拼成完整体积
        x = x.view(B, Dp, Hp, Wp, psD, psH, psW).permute(0, 4, 5, 6, 1, 2, 3)
        # (B, psD, psH, psW, Dp, Hp, Wp) -> 拼接
        x = x.reshape(B, 1, psD * Dp, psH * Hp, psW * Wp)  # (B, 1, D, H, W)
        return x


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int = 128, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim)
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int = 128, num_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor):
        return self.fc(x)


class MedMamba3D(nn.Module):
    """
    统一封装：编码器 + MAE解码 + 对比投影 + 分类头
    """
    def __init__(self, in_chans: int = 1, embed_dim: int = 128, depth: int = 12, patch_size: Tuple[int, int, int] = (16, 16, 16), decoder_dim: int = 256, proj_dim: int = 256, num_classes: int = 2):
        super().__init__()
        self.encoder = MedMamba3DEncoder(in_chans=in_chans, embed_dim=embed_dim, depth=depth, patch_size=patch_size)
        self.mae_decoder = MAEDecoder3D(embed_dim=embed_dim, decoder_dim=decoder_dim, patch_size=patch_size)
        self.projection = ProjectionHead(in_dim=embed_dim, proj_dim=proj_dim)
        self.classifier = ClassifierHead(in_dim=embed_dim, num_classes=num_classes)

        # 掩蔽比率默认，可在训练脚本设置
        self.mask_ratio = 0.75

    def forward_encoder(self, x: torch.Tensor):
        tokens, grid = self.encoder(x)
        return tokens, grid

    def forward_mae(self, x: torch.Tensor, mask_ratio: Optional[float] = None):
        # 简化：随机mask部分token（置零），仅用于重建
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        tokens, grid = self.encoder(x)  # (B, N, C)
        B, N, C = tokens.shape
        num_mask = int(N * mask_ratio)
        # 每个样本随机mask
        mask = torch.zeros(B, N, dtype=torch.bool, device=x.device)
        for i in range(B):
            idx = torch.randperm(N, device=x.device)[:num_mask]
            mask[i, idx] = True

        tokens_masked = tokens.clone()
        tokens_masked[mask] = 0.0

        recon = self.mae_decoder(tokens_masked, grid)  # (B, 1, D, H, W)
        return recon, mask

    def forward_projection(self, x: torch.Tensor):
        tokens, _ = self.encoder(x)
        # 简单池化为全局特征
        feat = tokens.mean(dim=1)  # (B, C)
        z = self.projection(feat)
        return F.normalize(z, dim=-1)

    def forward_classifier(self, x: torch.Tensor):
        tokens, _ = self.encoder(x)
        feat = tokens.mean(dim=1)  # (B, C)
        logits = self.classifier(feat)
        return logits