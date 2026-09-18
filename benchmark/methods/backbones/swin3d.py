"""
3D Swin Transformer — Window-based shifted-window attention for 3D volumes.

Adapted from Swin Transformer (Liu et al. 2021) for 3D medical imaging.
Key features:
  - W-MSA: Window Multi-head Self-Attention within 3D windows
  - SW-MSA: Shifted windows in alternating blocks
  - Patch merging: hierarchical feature pyramid (4 stages)
  - Relative position bias (3D)

Config: Tiny (embed_dim=96, depths=[2,2,6,2], heads=[3,6,12,24])
"""
import sys, os
from typing import Any, Dict, List, Optional, Tuple
import torch, torch.nn as nn, torch.nn.functional as F, math

from core.base_method import BaseMethod


# ===========================================================================
# Window utils
# ===========================================================================

def window_partition3d(x: torch.Tensor, window_size: Tuple[int,int,int]):
    """x: [B, C, D, H, W] -> [B*num_windows, C, wD*wH*wW]"""
    B, C, D, H, W = x.shape
    wD, wH, wW = window_size
    x = x.view(B, C, D // wD, wD, H // wH, wH, W // wW, wW)
    x = x.permute(0, 2, 4, 6, 3, 5, 7, 1).contiguous()
    x = x.view(-1, wD * wH * wW, C)
    return x

def window_reverse3d(x: torch.Tensor, window_size: Tuple[int,int,int], D: int, H: int, W: int):
    """[B*num_windows, wD*wH*wW, C] -> [B, C, D, H, W]"""
    wD, wH, wW = window_size
    B_ = x.shape[0]
    nD, nH, nW = D // wD, H // wH, W // wW
    B = B_ // (nD * nH * nW)
    x = x.view(B, nD, nH, nW, wD, wH, wW, -1)
    x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
    x = x.view(B, -1, D, H, W)
    return x


# ===========================================================================
# 3D Window Attention
# ===========================================================================

class WindowAttention3D(nn.Module):
    """W-MSA / SW-MSA with 3D relative position bias."""

    def __init__(self, dim: int, window_size: Tuple[int,int,int], num_heads: int,
                 qkv_bias: bool = True, attn_drop: float = 0., proj_drop: float = 0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Relative position bias table
        wD, wH, wW = window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * wD - 1) * (2 * wH - 1) * (2 * wW - 1), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Relative position index
        coords_d = torch.arange(wD)
        coords_h = torch.arange(wH)
        coords_w = torch.arange(wW)
        coords = torch.stack(torch.meshgrid(coords_d, coords_h, coords_w, indexing='ij'))
        coords_flatten = coords.reshape(3, -1)  # [3, wD*wH*wW]
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0)  # [N, N, 3]
        relative_coords[:, :, 0] += wD - 1
        relative_coords[:, :, 1] += wH - 1
        relative_coords[:, :, 2] += wW - 1
        relative_coords[:, :, 0] *= (2 * wH - 1) * (2 * wW - 1)
        relative_coords[:, :, 1] *= (2 * wW - 1)
        self.register_buffer("relative_position_index",
                             relative_coords.sum(-1))  # [N, N]

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        idx = self.relative_position_index[:N, :N]
        rpb = self.relative_position_bias_table[idx.view(-1)].view(N, N, self.num_heads)
        rpb = rpb.permute(2, 0, 1).unsqueeze(0)
        attn = attn + rpb

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


# ===========================================================================
# Swin Block
# ===========================================================================

class SwinBlock3D(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size,
                 shift_size=0, mlp_ratio=4., drop=0., attn_drop=0.,
                 drop_path=0.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size

        if min(input_resolution) <= window_size[0]:
            self.shift_size = 0
            self.window_size = (min(input_resolution),) * 3

        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, self.window_size, num_heads,
                                       attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(drop))
        self.drop_path = nn.Identity()

        # Attention mask for cyclic shift
        if self.shift_size > 0:
            D, H, W = input_resolution
            img_mask = torch.zeros((1, D, H, W, 1))
            d_slices = (slice(0, -self.window_size[0]),
                       slice(-self.window_size[0], -self.shift_size),
                       slice(-self.shift_size, None))
            h_slices = (slice(0, -self.window_size[1]),
                       slice(-self.window_size[1], -self.shift_size),
                       slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size[2]),
                       slice(-self.window_size[2], -self.shift_size),
                       slice(-self.shift_size, None))
            cnt = 0
            for d in d_slices:
                for h in h_slices:
                    for w in w_slices:
                        img_mask[:, d, h, w, :] = cnt
                        cnt += 1
            mask_windows = window_partition3d(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1] * self.window_size[2])
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        D, H, W = self.input_resolution
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, D, H, W, C)

        # Cyclic shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size, -self.shift_size), dims=(1, 2, 3))

        # Window partition
        x = x.permute(0, 4, 1, 2, 3).contiguous()  # [B,C,D,H,W]
        x = window_partition3d(x, self.window_size)
        x = self.attn(x, mask=self.attn_mask)
        x = window_reverse3d(x, self.window_size, D, H, W)

        # Reverse shift
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size, self.shift_size), dims=(2, 3, 4))

        x = x.permute(0, 2, 3, 4, 1).contiguous().view(B, D * H * W, C)
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ===========================================================================
# Patch Merging & Patch Embed
# ===========================================================================

class PatchMergingSwin3D(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(8 * dim)

    def forward(self, x):
        D, H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, D, H, W, C)
        D2, H2, W2 = D // 2, H // 2, W // 2
        # Merge 2x2x2 patches
        x0 = x[:, 0::2, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, 0::2, :]
        x3 = x[:, 1::2, 1::2, 0::2, :]
        x4 = x[:, 0::2, 0::2, 1::2, :]
        x5 = x[:, 1::2, 0::2, 1::2, :]
        x6 = x[:, 0::2, 1::2, 1::2, :]
        x7 = x[:, 1::2, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3, x4, x5, x6, x7], -1)
        x = x.view(B, D2 * H2 * W2, 8 * C)
        x = self.norm(x)
        x = self.reduction(x)
        return x, (D2, H2, W2)


class PatchEmbedSwin3D(nn.Module):
    def __init__(self, patch_size=(4,4,4), in_chans=1, embed_dim=96):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


# ===========================================================================
# Swin3D Backbone
# ===========================================================================

class Swin3DBackbone(nn.Module):
    def __init__(self, in_chans=1, num_classes=2, embed_dim=96,
                 depths=(2,2,6,2), num_heads=(3,6,12,24),
                 window_size=(7,7,7), patch_size=(4,4,4),
                 mlp_ratio=4., drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1):
        super().__init__()
        input_shape = (112, 112, 112)
        self.patch_embed = PatchEmbedSwin3D(patch_size, in_chans, embed_dim)
        D0 = input_shape[0] // patch_size[0]
        H0 = input_shape[1] // patch_size[1]
        W0 = input_shape[2] // patch_size[2]
        resolution = (D0, H0, W0)

        self.pos_drop = nn.Dropout(drop_rate)
        self.layers = nn.ModuleList()
        dim = embed_dim

        for i in range(len(depths)):
            stage = nn.ModuleList()
            for j in range(depths[i]):
                shift_size = 0 if (j % 2 == 0) else (window_size[0] // 2)
                stage.append(SwinBlock3D(
                    dim=dim, input_resolution=resolution, num_heads=num_heads[i],
                    window_size=window_size, shift_size=shift_size,
                    mlp_ratio=mlp_ratio, drop=drop_rate, attn_drop=attn_drop_rate))
            self.layers.append(stage)
            if i < len(depths) - 1:
                merge = PatchMergingSwin3D(resolution, dim)
                self.layers.append(merge)
                _, resolution = merge(x=torch.randn(1, resolution[0]*resolution[1]*resolution[2], dim))
                dim = dim * 2

        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Linear(dim, num_classes)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1.0); nn.init.constant_(m.bias, 0)

    def forward_encoder(self, x):
        x = self.patch_embed(x)
        x = self.pos_drop(x)
        for module in self.layers:
            if isinstance(module, PatchMergingSwin3D):
                x, _ = module(x)
            else:
                module = module  # nn.ModuleList of blocks
                if hasattr(module, '__iter__') and not isinstance(module, PatchMergingSwin3D):
                    for blk in module:
                        x = blk(x)
                else:
                    x = module(x)
        x = self.norm(x)
        return x.mean(dim=1)

    def forward_classifier(self, x):
        return self.classifier(self.forward_encoder(x))
    def forward(self, x):
        return self.forward_classifier(x)


class Swin3DMethod(BaseMethod):
    def __init__(self, config): super().__init__(config)

    def build_model(self):
        c = self.config.get("model", {})
        variant = str(c.get("variant", "Tiny"))
        v = {"Tiny": {"embed_dim":96,"depths":(2,2,6,2),"heads":(3,6,12,24)}}
        vc = v.get(variant, v["Tiny"])
        self.model = Swin3DBackbone(
            in_chans=int(c.get("in_channels", 1)),
            num_classes=int(c.get("num_classes", 2)),
            embed_dim=int(c.get("embed_dim", vc["embed_dim"])),
            depths=vc["depths"], num_heads=vc["heads"],
            window_size=tuple(c.get("window_size", (7, 7, 7))),
            drop_rate=float(c.get("dropout", 0.)), drop_path_rate=float(c.get("drop_path_rate", 0.1)))
        n = sum(p.numel() for p in self.model.parameters())
        print(f"[Swin3D] variant={variant}, params={n/1e6:.2f}M", flush=True)
        return self.model

    def forward_encoder(self, x): return self.model.forward_encoder(x)
    def forward_classifier(self, x): return self.model.forward_classifier(x)

    def get_optimizer_param_groups(self):
        r = float(self.config.get("training", {}).get("backbone_lr_ratio", 0.1))
        hid = {id(p) for p in self.model.classifier.parameters()}
        h, bk = [], []
        for _, p in self.model.named_parameters():
            if not p.requires_grad: continue
            (h if id(p) in hid else bk).append(p)
        g = []
        if bk: g.append({"params": bk, "lr_ratio": r})
        if h:  g.append({"params": h, "lr_ratio": 1.0})
        return g

from methods.method_registry import register_method
register_method("swin3d", Swin3DMethod)
