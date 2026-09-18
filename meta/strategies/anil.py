import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import higher
from .base import BaseMetaStrategy


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, target):
        logpt = F.log_softmax(logits, dim=1)
        pt = logpt.exp()
        logpt_t = logpt.gather(1, target.view(-1, 1)).squeeze(1)
        pt_t = pt.gather(1, target.view(-1, 1)).squeeze(1)

        eps = 1e-7
        p_diff = (1.0 - pt_t).clamp(min=eps)

        loss = -(p_diff ** self.gamma) * logpt_t
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            alpha_t = alpha.gather(0, target)
            loss = alpha_t * loss
        return loss.mean()


class ANILStrategy(BaseMetaStrategy):
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        self.params = config.get("params", {})
        self.inner_steps = int(self.params.get("inner_steps", 5))
        self.first_order = bool(self.params.get("first_order", False))
        self.label_smoothing = float(self.params.get("label_smoothing", 0.0))
        self.inner_label_smoothing = float(self.params.get("inner_label_smoothing", 0.0))
        self.dropout_rate = float(self.params.get("dropout", 0.1))
        self.focal_gamma = float(self.params.get("focal_gamma", 2.0))
        self.focal_weight = float(self.params.get("focal_weight", 0.0))
        self.max_inner_lr = float(self.params.get("max_inner_lr", 0.2))
        self.use_class_weights = bool(self.params.get("use_class_weights", False))
        self.inner_use_class_weights = bool(self.params.get("inner_use_class_weights", self.use_class_weights))
        self.inner_weight_decay = float(self.params.get("inner_weight_decay", 1e-4))
        self._debug_bb = False
        self._debug_fhead = False
        self._debug_logged = False

        self.class_weights = None
        init_class_weights = self.params.get("class_weights")
        if init_class_weights is not None:
            self.class_weights = torch.tensor(init_class_weights, dtype=torch.float32)

        init_lr = max(float(self.params.get("inner_lr", 0.05)), 1e-6)
        self.inner_lr = nn.Parameter(torch.log(torch.tensor(init_lr, dtype=torch.float32)))

        # Performance: pre-allocate inner optimizer (parameter groups updated per-episode)
        # Fix: add weight_decay to prevent the inner loop from memorizing support-set noise.
        # Without weight_decay, 2-3 steps of unregularized SGD on 4-5 shot support cause
        # the head to overfit individual samples instead of learning class-prototypical weights.
        head_module = self._get_head_module()
        self._inner_opt = torch.optim.SGD(
            head_module.parameters(),
            lr=init_lr,
            weight_decay=self.inner_weight_decay,
        )

        # torch.compile support
        self._use_compile = bool(self.params.get("use_compile", False))
        self._compiled_backbone = None

    def _get_compiled_backbone(self):
        if self._compiled_backbone is None and self._use_compile:
            try:
                if hasattr(self.model, 'forward_encoder'):
                    self._compiled_backbone = torch.compile(
                        self.model.forward_encoder,
                        mode="reduce-overhead",
                        fullgraph=False,
                    )
                    print("[ANIL] torch.compile enabled for backbone forward_encoder", flush=True)
            except Exception as e:
                print(f"[ANIL] torch.compile failed, falling back to eager: {e}", flush=True)
                self._use_compile = False
        return self._compiled_backbone

    def set_class_weights(self, class_weights: Optional[torch.Tensor]) -> None:
        if class_weights is None:
            self.class_weights = None
            return
        self.class_weights = class_weights.detach().float().cpu()

    def get_inner_lr(self) -> torch.Tensor:
        return self.inner_lr.exp().clamp(min=1e-6, max=self.max_inner_lr)

    def _get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        # -----------------------------------------------
        # Backbone forward: now runs in BF16/FP16 via the
        # outer autocast context (no longer force-disabled).
        # BF16 has the same exponent range as FP32, so Mamba
        # exp/log no longer overflow → safe to use AMP.
        # -----------------------------------------------
        compiled_fn = self._get_compiled_backbone()
        if compiled_fn is not None and self._use_compile:
            z, _ = compiled_fn(x)
        elif hasattr(self.model, "forward_encoder"):
            z, _ = self.model.forward_encoder(x)
        else:
            z = self.model(x)
        if z.dim() == 3:
            z = z.mean(dim=1)
        return z

    def _get_head_module(self) -> nn.Module:
        head_module = getattr(self.model, "head", getattr(self.model, "classifier", None))
        if head_module is None:
            raise RuntimeError("ANILStrategy requires model.head or model.classifier.")
        return head_module

    def _compute_outer_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        class_weights = self.class_weights if self.use_class_weights else None
        if class_weights is not None:
            class_weights = class_weights.to(logits.device)
        ce_loss = F.cross_entropy(
            logits,
            target,
            weight=class_weights,
            label_smoothing=self.label_smoothing
        )
        if self.focal_weight <= 0.0:
            return ce_loss
        focal_loss = FocalLoss(alpha=class_weights, gamma=self.focal_gamma)(logits, target)
        return (1.0 - self.focal_weight) * ce_loss + self.focal_weight * focal_loss

    def core_step(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        optimizer: Optional[torch.optim.Optimizer] = None,
        inner_optimizer: Optional[torch.optim.Optimizer] = None,
        is_training: bool = True
    ) -> Dict[str, Any]:
        support_y = support_y.long()
        query_y = query_y.long()

        # Backbone runs inside the outer autocast (BF16/FP16).
        # No longer force FP32 here — BF16 handles Mamba precision safely.
        s_z = self._get_embedding(support_x)
        q_z = self._get_embedding(query_x)

        s_z = s_z.float()
        q_z = q_z.float()

        if is_training and self.dropout_rate > 0.0:
            s_z = F.dropout(s_z, p=self.dropout_rate, training=True)
            q_z = F.dropout(q_z, p=self.dropout_rate, training=True)

        inner_class_weights = None
        if self.inner_use_class_weights and self.class_weights is not None:
            inner_class_weights = self.class_weights.to(support_x.device)

        head_module = self._get_head_module()
        actual_inner_lr = self.get_inner_lr().float().to(support_x.device)

        # Reuse pre-allocated optimizer: reset state + update lr
        self._inner_opt.param_groups[0]["lr"] = float(actual_inner_lr.detach().item())
        for group in self._inner_opt.param_groups:
            for p in group["params"]:
                state = self._inner_opt.state.get(p)
                if state:
                    state.clear()

        inner_losses_history = []
        # Inner loop: keep in FP32 for numerical stability.
        # higher's multi-step SGD adaptation amplifies fp16 rounding errors
        # into inf/nan, especially with learned inner_lr. BF16 still has
        # reduced mantissa (7 bits vs 10 for fp16 vs 23 for fp32), so the
        # inner loop stays in fp32.
        with torch.amp.autocast(device_type='cuda', enabled=False):
            with torch.enable_grad():
                with higher.innerloop_ctx(
                    head_module,
                    self._inner_opt,
                    copy_initial_weights=True,
                    track_higher_grads=(not self.first_order and is_training)
                ) as (fhead, diffopt):
                    for _ in range(self.inner_steps):
                        s_logits = fhead(s_z)
                        if s_logits.dim() > 2:
                            s_logits = s_logits.view(s_logits.size(0), -1)
                        inner_loss = F.cross_entropy(
                            s_logits,
                            support_y,
                            weight=inner_class_weights,
                            label_smoothing=self.inner_label_smoothing,
                        )
                        inner_losses_history.append(float(inner_loss.detach().item()))
                        diffopt.step(inner_loss, override={"lr": [actual_inner_lr]})

                    q_logits = fhead(q_z)
                    if q_logits.dim() > 2:
                        q_logits = q_logits.view(q_logits.size(0), -1)
                    outer_loss = self._compute_outer_loss(q_logits, query_y)

        return {
            "loss": outer_loss,
            "logits": q_logits.detach(),
            "y_true": query_y,
            "inner_losses": inner_losses_history,
            "ce_loss": outer_loss.detach(),
            "supcon_loss": 0.0
        }
