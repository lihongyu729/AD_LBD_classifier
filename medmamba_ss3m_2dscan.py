import torch
import torch.nn as nn
import torch.nn.functional as F

from medmamba_ss3m import PatchEmbed3D, SeqSSM


class SS2DScanBlock3DDual(nn.Module):
    def __init__(
        self,
        dim: int,
        dropout: float = 0.0,
        use_pos_emb: bool = True,
        merge_type: str = "softmax",
        branch_types=("conv", "conv"),
        use_checkpoint: bool = False,
        n_dirs_train: int = 4,
        use_dual_branch: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.use_pos_emb = use_pos_emb
        self.merge_type = merge_type
        self.use_checkpoint = use_checkpoint
        self.n_dirs_train = max(1, min(int(n_dirs_train), 4))
        self.use_dual_branch = bool(use_dual_branch)

        self.norm = nn.LayerNorm(dim)
        self.ssm_a = SeqSSM(dim=dim, ssm_type=branch_types[0], depth=2, dropout=dropout)
        if self.use_dual_branch:
            try:
                self.ssm_b = SeqSSM(dim=dim, ssm_type=branch_types[1], depth=2, dropout=dropout)
            except ImportError:
                self.ssm_b = SeqSSM(dim=dim, ssm_type="conv", depth=2, dropout=dropout)
        else:
            self.ssm_b = None

        if merge_type == "softmax":
            self.dir_logits = nn.Parameter(torch.zeros(self.n_dirs_train))
        else:
            self.register_parameter("dir_logits", None)
        if self.use_dual_branch:
            self.branch_logits = nn.Parameter(torch.zeros(2))
        else:
            self.register_parameter("branch_logits", None)
        self.pos_emb = None

    @staticmethod
    def _generate_4_hw_flips(x: torch.Tensor):
        outs = []
        for fh, fw in ((False, False), (True, False), (False, True), (True, True)):
            fx = x
            if fh:
                fx = torch.flip(fx, dims=[3])
            if fw:
                fx = torch.flip(fx, dims=[4])
            outs.append(fx)
        return outs

    @staticmethod
    def _reverse_4_hw_flips(xs):
        outs = []
        for (fh, fw), x in zip(((False, False), (True, False), (False, True), (True, True)), xs):
            fx = x
            if fh:
                fx = torch.flip(fx, dims=[3])
            if fw:
                fx = torch.flip(fx, dims=[4])
            outs.append(fx)
        return outs

    @staticmethod
    def _vol_to_seq_2d(x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        x = x.permute(0, 2, 3, 4, 1).contiguous().view(b * d, h * w, c)
        return x

    @staticmethod
    def _seq_to_vol_2d(x_seq: torch.Tensor, shape_3d):
        bd, hw, c = x_seq.shape
        d, h, w, b = shape_3d
        x = x_seq.view(b, d, h, w, c).permute(0, 4, 1, 2, 3).contiguous()
        return x

    def _maybe_pos_emb(self, x: torch.Tensor) -> torch.Tensor:
        if not self.use_pos_emb:
            return x
        b, c, d, h, w = x.shape
        if self.pos_emb is None or self.pos_emb.shape != (1, c, d, h, w):
            temp_emb = torch.zeros(1, c, d, h, w, device=x.device)
            nn.init.trunc_normal_(temp_emb, std=0.02)
            self.pos_emb = nn.Parameter(temp_emb)
        if self.pos_emb.device != x.device:
            self.pos_emb.data = self.pos_emb.data.to(x.device)
        return x + self.pos_emb

    def _direction_merge(self, vols):
        stacked = torch.stack(vols, dim=1)
        if self.merge_type == "softmax" and self.dir_logits is not None:
            w = torch.softmax(self.dir_logits, dim=0).view(1, len(vols), 1, 1, 1, 1)
            return (stacked * w).sum(dim=1)
        return stacked.mean(dim=1)

    def _run_branch(self, ssm_branch, xs_4, shape_info):
        vols_4 = []
        for x in xs_4:
            feat_seq = self._vol_to_seq_2d(x)
            if self.use_checkpoint and feat_seq.requires_grad:
                out_seq = torch.utils.checkpoint.checkpoint(ssm_branch, feat_seq)
            else:
                out_seq = ssm_branch(feat_seq)
            vols_4.append(self._seq_to_vol_2d(out_seq, shape_info))
        vols_4 = self._reverse_4_hw_flips(vols_4)
        return self._direction_merge(vols_4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, d, h, w = x.shape
        res = x
        x = x.permute(0, 2, 3, 4, 1).contiguous()
        x = self.norm(x)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        x = self._maybe_pos_emb(x)

        xs_4 = self._generate_4_hw_flips(x)[:self.n_dirs_train]
        shape_info = (d, h, w, b)
        y_a = self._run_branch(self.ssm_a, xs_4, shape_info)
        if not self.use_dual_branch or self.ssm_b is None:
            fused = y_a
        else:
            y_b = self._run_branch(self.ssm_b, xs_4, shape_info)
            bw = torch.softmax(self.branch_logits, dim=0).view(2, 1, 1, 1, 1)
            fused = bw[0] * y_a + bw[1] * y_b
        return res + fused


class MedMambaSS3M2DScan(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        embed_dim: int = 96,
        depth: int = 4,
        patch_size=(2, 2, 2),
        dropout: float = 0.0,
        use_pos_emb: bool = True,
        merge_type: str = "softmax",
        use_checkpoint: bool = False,
        branch_types=("conv", "conv"),
        n_dirs_train: int = 4,
        use_dual_branch: bool = True,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, patch_size=patch_size, norm=True)
        self.blocks = nn.ModuleList([
            SS2DScanBlock3DDual(
                dim=embed_dim,
                dropout=dropout,
                use_pos_emb=use_pos_emb,
                merge_type=merge_type,
                branch_types=branch_types,
                use_checkpoint=use_checkpoint,
                n_dirs_train=n_dirs_train,
                use_dual_branch=use_dual_branch,
            )
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

    def forward(self, x: torch.Tensor):
        return self.forward_classifier(x)


class MedMambaSS3M2DScanEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 96,
        depth: int = 4,
        patch_size=(2, 2, 2),
        dropout: float = 0.0,
        use_pos_emb: bool = True,
        merge_type: str = "softmax",
        use_checkpoint: bool = False,
        decoder_dim: int = 256,
        branch_types=("conv", "conv"),
        chunk_len: int = -1,
        n_dirs_train: int = 8,
        use_dual_branch: bool = True,
    ):
        super().__init__()
        self.patch_embed = PatchEmbed3D(in_channels, embed_dim, patch_size=patch_size, norm=True)
        self.blocks = nn.ModuleList([
            SS2DScanBlock3DDual(
                dim=embed_dim,
                dropout=dropout,
                use_pos_emb=use_pos_emb,
                merge_type=merge_type,
                branch_types=branch_types,
                use_checkpoint=use_checkpoint,
                n_dirs_train=n_dirs_train,
                use_dual_branch=use_dual_branch,
            )
            for _ in range(depth)
        ])
        self.norm = nn.BatchNorm3d(embed_dim)
        self.out_channels = embed_dim

    def forward(self, x: torch.Tensor):
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x.mean(dim=[2, 3, 4])

    def forward_tokens(self, x: torch.Tensor):
        x = self.patch_embed(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x


class MedMambaSS3M2DScanContrast(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        embed_dim: int = 96,
        depth: int = 4,
        patch_size=(2, 2, 2),
        dropout: float = 0.0,
        use_pos_emb: bool = True,
        merge_type: str = "softmax",
        use_checkpoint: bool = False,
        proj_dim: int = 128,
    ):
        super().__init__()
        self.encoder = MedMambaSS3M2DScanEncoder(
            in_channels=in_channels,
            embed_dim=embed_dim,
            depth=depth,
            patch_size=patch_size,
            dropout=dropout,
            use_pos_emb=use_pos_emb,
            merge_type=merge_type,
            use_checkpoint=use_checkpoint,
        )
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor):
        feats = self.encoder(x)
        z = self.proj(feats)
        return F.normalize(z, dim=-1)
