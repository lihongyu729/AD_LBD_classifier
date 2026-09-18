"""
3D CrossFormer with Dual-Range Attention (LDA/SDA) — proper implementation.

Based on the original MML-3DCrossFormer architecture:
  - LDA (Long-range Dynamic Attention): dilated group sampling with interval I
  - SDA (Short-range Dynamic Attention): contiguous window grouping
  - DynamicPositionBias: MLP-predicted 3D relative position biases
  - Multi-scale patch embedding
  - 4-stage hierarchical design with alternating SDA/LDA blocks

Key difference from the previous simplified version: real LDA/SDA alternating
attention with proper 3D group partitioning.
"""
import sys
import os
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from core.base_method import BaseMethod


# ===========================================================================
# Helpers
# ===========================================================================

def _to_3tuple(x):
    if isinstance(x, (list, tuple)):
        return tuple(x)
    return (x, x, x)


# ===========================================================================
# Dynamic Position Bias (3D)
# ===========================================================================

class DynamicPosBias3D(nn.Module):
    """
    MLP-based relative position bias predictor for 3D volumes.
    Takes relative coordinate differences (dD, dH, dW) and outputs
    a bias value per attention head.
    """
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(3, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim))
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim))
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim), nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, num_heads))
        nn.init.zeros_(self.pos3[-1].weight)
        nn.init.zeros_(self.pos3[-1].bias)

    def forward(self, biases: torch.Tensor) -> torch.Tensor:
        # biases: [N, 3] relative coords (dD, dH, dW)
        pos = self.pos_proj(biases)
        pos = pos + self.pos1(pos)
        pos = pos + self.pos2(pos)
        pos = self.pos3(pos)
        return pos  # [N, num_heads]


# ===========================================================================
# 3D Attention with Group Partitioning
# ===========================================================================

class Attention3D_Groups(nn.Module):
    """
    3D multi-head attention supporting window/group partitioning and
    dynamic position bias. Works with both SDA (contiguous) and
    LDA (dilated) partitioning performed in the calling SwiFTBlock3D.
    """
    def __init__(self, dim: int, group_size: Tuple[int,int,int],
                 num_heads: int, qkv_bias: bool = True, attn_drop: float = 0.,
                 proj_drop: float = 0.):
        super().__init__()
        self.group_size = group_size  # (Dg, Hg, Wg)
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Dynamic position bias (3D)
        Dg, Hg, Wg = group_size
        position_bias_d = torch.arange(1 - Dg, Dg)
        position_bias_h = torch.arange(1 - Hg, Hg)
        position_bias_w = torch.arange(1 - Wg, Wg)
        biases = torch.stack(torch.meshgrid(
            position_bias_d, position_bias_h, position_bias_w, indexing='ij'))
        biases = biases.flatten(1).transpose(0, 1).float()  # [(2Dg-1)*(2Hg-1)*(2Wg-1), 3]
        self.register_buffer("biases", biases)

        # Relative position index table
        self.register_buffer("relative_position_index",
                             self._build_rel_pos_idx(group_size))

        self.pos = DynamicPosBias3D(dim // 4, num_heads)

    def _build_rel_pos_idx(self, group_size):
        Dg, Hg, Wg = group_size
        coords_d = torch.arange(Dg)
        coords_h = torch.arange(Hg)
        coords_w = torch.arange(Wg)
        coords = torch.stack(torch.meshgrid(
            coords_d, coords_h, coords_w, indexing='ij')).flatten(1)  # [3, Dg*Hg*Wg]
        relative_coords = coords[:, :, None] - coords[:, None, :]  # [3, N, N]
        relative_coords = relative_coords.permute(1, 2, 0)  # [N, N, 3]

        # Shift to positive indices
        relative_coords[:, :, 0] += Dg - 1
        relative_coords[:, :, 1] += Hg - 1
        relative_coords[:, :, 2] += Wg - 1
        rel_idx = (relative_coords[:, :, 0] * (2 * Hg - 1) * (2 * Wg - 1) +
                    relative_coords[:, :, 1] * (2 * Wg - 1) +
                    relative_coords[:, :, 2])
        return rel_idx  # [N, N]

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: [num_groups * B, N, C] where N = Dg*Hg*Wg
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B_, num_heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # [B_, num_heads, N, N]

        # Dynamic position bias
        pos_bias = self.pos(self.biases)  # [(2Dg-1)*(2Hg-1)*(2Wg-1), num_heads]
        # Index to shape [N, N, num_heads]
        rel_idx = self.relative_position_index  # [N, N]
        pb = pos_bias[rel_idx.view(-1)].view(N, N, self.num_heads)
        pb = pb.permute(2, 0, 1).unsqueeze(0)  # [1, num_heads, N, N]
        attn = attn + pb

        # Apply mask (padded positions get -inf)
        if mask is not None:
            # mask: [num_groups, N, N]
            nG = mask.shape[0]
            attn_rs = attn.view(-1, nG, self.num_heads, N, N)
            mask_rs = mask.unsqueeze(2)  # [nG, 1, N, N]
            attn_rs = attn_rs + mask_rs
            attn = attn_rs.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ===========================================================================
# SwiFTBlock3D — Alternating SDA / LDA
# ===========================================================================

class SwiFTBlock3D(nn.Module):
    """
    3D CrossFormer block with alternating SDA / LDA attention.

    Args:
        lsda_flag: 0 = SDA (short-range contiguous window)
                   1 = LDA (long-range dilated sampling)
    """
    def __init__(self, dim: int, input_resolution: Tuple[int,int,int],
                 num_heads: int, group_size: Tuple[int,int,int] = (4,4,4),
                 interval: Tuple[int,int,int] = (2,2,2), lsda_flag: int = 0,
                 mlp_ratio: float = 4., drop: float = 0., attn_drop: float = 0.,
                 drop_path: float = 0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution  # (D, H, W)
        self.num_heads = num_heads
        self.group_size = group_size  # (Dg, Hg, Wg)
        self.interval = interval      # (Id, Ih, Iw)
        self.lsda_flag = lsda_flag

        # Enforce: if resolution < group_size, fallback to SDA
        D, H, W = input_resolution
        Dg, Hg, Wg = group_size
        if min(D, H, W) <= max(Dg, Hg, Wg):
            self.lsda_flag = 0
            self.group_size = (min(D, Dg), min(H, Hg), min(W, Wg))

        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention3D_Groups(dim, self.group_size, num_heads,
                                       attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
        )
        self.drop_path = nn.Identity()  # Could be DropPath in training

        self._make_attn_mask()

    def _make_attn_mask(self):
        """Pre-compute attention mask for padded regions."""
        D, H, W = self.input_resolution
        Dg, Hg, Wg = self.group_size
        Id, Ih, Iw = self.interval

        if self.lsda_flag == 0:  # SDA
            size_div = (Dg, Hg, Wg)
            pad_d = (Dg - D % Dg) % Dg
            pad_h = (Hg - H % Hg) % Hg
            pad_w = (Wg - W % Wg) % Wg
            Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w

            # Partition into contiguous Dg x Hg x Wg groups
            mask = torch.zeros((1, Dp, Hp, Wp, 1))
            if pad_d > 0:
                mask[:, -pad_d:, :, :] = -1
            if pad_h > 0:
                mask[:, :, -pad_h:, :] = -1
            if pad_w > 0:
                mask[:, :, :, -pad_w:] = -1

            # Reshape to groups
            nD, nH, nW = Dp // Dg, Hp // Hg, Wp // Wg
            mask = mask.view(1, nD, Dg, nH, Hg, nW, Wg, 1)
            mask = mask.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
            nG = nD * nH * nW
            mask = mask.view(nG, Dg * Hg * Wg, 1)
            # attn_mask: [nG, N, N] where N = Dg*Hg*Wg
            self.attn_mask = (mask @ mask.transpose(1, 2)).float() * -10000.0
            self.pad = (0, pad_w, 0, pad_h, 0, pad_d)
            self.nG = nG
            self.Dg, self.Hg, self.Wg = Dg, Hg, Wg
            self.Dp, self.Hp, self.Wp = Dp, Hp, Wp
        else:  # LDA
            size_div = (Dg * Id, Hg * Ih, Wg * Iw)
            pad_d = (size_div[0] - D % size_div[0]) % size_div[0]
            pad_h = (size_div[1] - H % size_div[1]) % size_div[1]
            pad_w = (size_div[2] - W % size_div[2]) % size_div[2]
            Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w

            mask = torch.zeros((1, Dp, Hp, Wp, 1))
            if pad_d > 0:
                mask[:, -pad_d:, :, :] = -1
            if pad_h > 0:
                mask[:, :, -pad_h:, :] = -1
            if pad_w > 0:
                mask[:, :, :, -pad_w:] = -1

            Rh, Rw, Rd = Dp // (Dg * Id), Hp // (Hg * Ih), Wp // (Wg * Iw)
            # LDA: I^3 groups of GxGxG each, with stride I
            mask = mask.view(1, Rd, Dg, Id, Rh, Hg, Ih, Rw, Wg, Iw, 1)
            mask = mask.permute(0, 4, 7, 3, 6, 9, 2, 5, 8, 10, 1).contiguous()
            nG = Id * Ih * Iw * Rd * Rh * Rw
            mask = mask.view(nG, Dg * Hg * Wg, 1)
            self.attn_mask = (mask @ mask.transpose(1, 2)).float() * -10000.0
            self.pad = (0, pad_w, 0, pad_h, 0, pad_d)
            self.nG = nG
            self.Dg, self.Hg, self.Wg = Dg, Hg, Wg
            self.Dp, self.Hp, self.Wp = Dp, Hp, Wp

    def forward(self, x: torch.Tensor):
        # x: [B, L, C] where L = D * H * W
        B, L, C = x.shape
        D, H, W = self.input_resolution
        Dg, Hg, Wg = self.Dg, self.Hg, self.Wg
        Id, Ih, Iw = self.interval

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        # Pad
        pad_dr, pad_wr, pad_hr, pad_dl, pad_hl, pad_wl = self.pad
        x = F.pad(x, (0, 0, pad_wl, pad_wr, pad_hl, pad_hr, pad_dl, pad_dr))
        Dp, Hp, Wp = self.Dp, self.Hp, self.Wp

        # Group embeddings
        if self.lsda_flag == 0:  # SDA — contiguous Dg x Hg x Wg groups
            x = x.view(B, Dp // Dg, Dg, Hp // Hg, Hg, Wp // Wg, Wg, C)
            x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
            x = x.view(B * self.nG, Dg * Hg * Wg, C)
        else:  # LDA — dilated groups with stride I
            Rd, Rh, Rw = Dp // (Dg * Id), Hp // (Hg * Ih), Wp // (Wg * Iw)
            x = x.view(B, Rd, Dg, Id, Rh, Hg, Ih, Rw, Wg, Iw, C)
            x = x.permute(0, 4, 7, 3, 6, 9, 2, 5, 8, 10, 1).contiguous()
            x = x.view(B * self.nG, Dg * Hg * Wg, C)

        # Attention with mask
        x = self.attn(x, mask=self.attn_mask.to(x.device) if self.attn_mask is not None else None)

        # Ungroup
        if self.lsda_flag == 0:
            x = x.view(B, Dp // Dg, Hp // Hg, Wp // Wg, Dg, Hg, Wg, C)
            x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
            x = x.view(B, Dp, Hp, Wp, C)
        else:
            Rd, Rh, Rw = Dp // (Dg * Id), Hp // (Hg * Ih), Wp // (Wg * Iw)
            x = x.view(B, Rh, Rw, Id, Ih, Iw, Rd, Dg, Hg, Wg, C)
            x = x.permute(0, 9, 6, 3, 10, 7, 4, 11, 8, 5, 1, 2).contiguous()
            x = x.view(B, Dp, Hp, Wp, C)

        # Remove padding
        if pad_dr > 0:
            x = x[:, :D]
        if pad_hr > 0:
            x = x[:, :, :H]
        if pad_wr > 0:
            x = x[:, :, :, :W]

        x = x.reshape(B, D * H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ===========================================================================
# PatchEmbed3D (multi-scale)
# ===========================================================================

class PatchEmbed3D_MS(nn.Module):
    """Multi-scale 3D patch embedding with multiple kernel sizes."""
    def __init__(self, img_size=(112,112,112), patch_sizes=[(4,4,4),(8,8,8),(16,16,16)],
                 in_chans=1, embed_dim=96, norm_layer=nn.LayerNorm):
        super().__init__()
        self.img_size = img_size
        n_branches = len(patch_sizes)
        self.projs = nn.ModuleList()
        self.norms = nn.ModuleList() if norm_layer is not None else None

        for i, ps in enumerate(patch_sizes):
            out_dim = embed_dim // (2 ** i) if i < n_branches - 1 else embed_dim // (2 ** (i - 1))
            self.projs.append(nn.Conv3d(in_chans, out_dim, kernel_size=ps, stride=patch_sizes[0]))
            if norm_layer is not None:
                self.norms.append(norm_layer(out_dim))

    def forward(self, x):
        # x: [B, 1, D, H, W]
        outs = []
        for i, proj in enumerate(self.projs):
            o = proj(x)  # [B, out_dim, D', H', W']
            o = o.flatten(2).transpose(1, 2)  # [B, N, out_dim]
            if self.norms is not None:
                o = self.norms[i](o)
            outs.append(o)
        x = torch.cat(outs, dim=-1)  # [B, N, embed_dim]
        return x


class PatchMerging3D_MS(nn.Module):
    """Multi-scale 3D patch merging."""
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Conv3d(dim, dim * 2, kernel_size=2, stride=2)
        self.norm = norm_layer(dim * 2) if norm_layer is not None else nn.Identity()

    def forward(self, x):
        D, H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, D, H, W, C).permute(0, 4, 1, 2, 3)
        x = self.reduction(x)
        _, _, D2, H2, W2 = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (D2, H2, W2)


# ===========================================================================
# CrossFormer3D (full backbone)
# ===========================================================================

class CrossFormer3DBackbone(nn.Module):
    """
    3D CrossFormer with proper LDA/SDA dual-range attention.

    Architecture:
      1. Multi-scale PatchEmbed3D → [B, N, embed_dim]
      2. 4 hierarchical stages with alternating SDA/LDA blocks
      3. PatchMerging between stages (2x downsampling)
      4. Global average pooling → classifier

    Config variants (aligned with paper):
      Tiny:   embed_dim=64,  depths=[1,1,8,6],  heads=[2,4,8,16]
      Small:  embed_dim=96,  depths=[2,2,6,2],  heads=[3,6,12,24]
      Base:   embed_dim=96,  depths=[2,2,18,2], heads=[3,6,12,24]
      Large:  embed_dim=128, depths=[2,2,18,2], heads=[4,8,16,32]
    """
    def __init__(self, in_chans=1, num_classes=2, embed_dim=96,
                 depths=(2,2,6,2), num_heads=(3,6,12,24),
                 group_size=(4,4,4), interval=(2,2,2),
                 patch_sizes=((4,4,4),(8,8,8),(16,16,16)),
                 mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1):
        super().__init__()
        input_shape = (112, 112, 112)

        # Patch embedding
        patch_stride = patch_sizes[0]
        self.patch_embed = PatchEmbed3D_MS(
            img_size=input_shape, patch_sizes=patch_sizes,
            in_chans=in_chans, embed_dim=embed_dim)
        D0 = input_shape[0] // patch_stride[0]
        H0 = input_shape[1] // patch_stride[1]
        W0 = input_shape[2] // patch_stride[2]
        self.num_patches = D0 * H0 * W0

        self.pos_drop = nn.Dropout(drop_rate)

        # Build stages
        self.stages = nn.ModuleList()
        self.mergings = nn.ModuleList()
        resolution = (D0, H0, W0)
        dim = embed_dim

        for i in range(len(depths)):
            stage = nn.ModuleList()
            for j in range(depths[i]):
                lsda_flag = 0 if (j % 2 == 0) else 1  # SDA → LDA → SDA → ...
                stage.append(SwiFTBlock3D(
                    dim=dim, input_resolution=resolution,
                    num_heads=num_heads[i], group_size=group_size,
                    interval=interval, lsda_flag=lsda_flag,
                    mlp_ratio=mlp_ratio, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=drop_path_rate))
            self.stages.append(stage)

            if i < len(depths) - 1:
                merge = PatchMerging3D_MS(resolution, dim)
                self.mergings.append(merge)
                x_dummy = torch.randn(1, resolution[0]*resolution[1]*resolution[2], dim)
                _, resolution = merge(x_dummy)
                dim = dim * 2

        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')

    def forward_encoder(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, D, H, W] → [B, dim]"""
        x = self.patch_embed(x)
        x = self.pos_drop(x)

        for i, stage in enumerate(self.stages):
            for blk in stage:
                x = blk(x)
            if i < len(self.mergings):
                x, _ = self.mergings[i](x)

        x = self.norm(x)
        return x.mean(dim=1)

    def forward_classifier(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, D, H, W] → [B, num_classes]"""
        z = self.forward_encoder(x)
        return self.classifier(z)

    def forward(self, x):
        return self.forward_classifier(x)


# ===========================================================================
# Benchmark Method wrapper
# ===========================================================================

class CrossFormer3DMethod(BaseMethod):
    VARIANTS = {
        "Tiny":   {"embed_dim": 64,  "depths": (1,1,8,6),  "heads": (2,4,8,16)},
        "Small":  {"embed_dim": 96,  "depths": (2,2,6,2),  "heads": (3,6,12,24)},
        "Base":   {"embed_dim": 96,  "depths": (2,2,18,2), "heads": (3,6,12,24)},
        "Large":  {"embed_dim": 128, "depths": (2,2,18,2), "heads": (4,8,16,32)},
    }

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

    def build_model(self) -> nn.Module:
        m_cfg = self.config.get("model", {})
        variant = str(m_cfg.get("variant", "Tiny"))
        v = self.VARIANTS.get(variant, self.VARIANTS["Tiny"])

        group_size = tuple(m_cfg.get("group_size", (4, 4, 4)))
        interval = tuple(m_cfg.get("interval", (2, 2, 2)))

        self.model = CrossFormer3DBackbone(
            in_chans=int(m_cfg.get("in_channels", 1)),
            num_classes=int(m_cfg.get("num_classes", 2)),
            embed_dim=int(m_cfg.get("embed_dim", v["embed_dim"])),
            depths=tuple(m_cfg.get("depths", v["depths"])),
            num_heads=tuple(m_cfg.get("num_heads", v["heads"])),
            group_size=group_size, interval=interval,
            drop_rate=float(m_cfg.get("dropout", 0.0)),
            drop_path_rate=float(m_cfg.get("drop_path_rate", 0.1)),
        )
        n = sum(p.numel() for p in self.model.parameters())
        print(f"[CrossFormer3D] variant={variant}, embed_dim={v['embed_dim']}, "
              f"depths={v['depths']}, params={n/1e6:.2f}M", flush=True)
        return self.model

    def forward_encoder(self, x):
        return self.model.forward_encoder(x)
    def forward_classifier(self, x):
        return self.model.forward_classifier(x)

    def get_optimizer_param_groups(self):
        r = float(self.config.get("training", {}).get("backbone_lr_ratio", 0.1))
        head_ids = {id(p) for p in self.model.classifier.parameters()}
        head, backbone = [], []
        for _, p in self.model.named_parameters():
            if not p.requires_grad: continue
            (head if id(p) in head_ids else backbone).append(p)
        g = []
        if backbone: g.append({"params": backbone, "lr_ratio": r})
        if head:     g.append({"params": head, "lr_ratio": 1.0})
        return g

from methods.method_registry import register_method
register_method("crossformer3d", CrossFormer3DMethod)
