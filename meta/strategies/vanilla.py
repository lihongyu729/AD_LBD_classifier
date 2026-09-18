import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
from .base import BaseMetaStrategy


class FocalLoss(nn.Module):
    def __init__(self, alpha: Optional[torch.Tensor] = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logpt = F.log_softmax(logits, dim=1)
        pt = logpt.exp()
        logpt_t = logpt.gather(1, target.view(-1, 1)).squeeze(1)
        pt_t = pt.gather(1, target.view(-1, 1)).squeeze(1)
        eps = 1e-7
        p_diff = (1.0 - pt_t).clamp(min=eps)
        loss = -(p_diff ** self.gamma) * logpt_t
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            loss = alpha.gather(0, target) * loss
        return loss.mean()


class VanillaStrategy(BaseMetaStrategy):
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        params = config.get("params", {})
        self.label_smoothing = float(params.get("label_smoothing", 0.0))
        self.focal_gamma = float(params.get("focal_gamma", 2.0))
        self.focal_weight = float(params.get("focal_weight", 0.0))
        self.use_class_weights = bool(params.get("use_class_weights", False))
        self.use_support_in_loss = bool(params.get("use_support_in_loss", True))
        self.class_weights: Optional[torch.Tensor] = None

    def set_class_weights(self, class_weights: Optional[torch.Tensor]) -> None:
        if class_weights is None:
            self.class_weights = None
            return
        self.class_weights = class_weights.detach().float().cpu()

    def _forward_logits(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, "forward_classifier"):
            logits = self.model.forward_classifier(x)
        elif hasattr(self.model, "forward_encoder"):
            tokens, _ = self.model.forward_encoder(x)
            feats = tokens.mean(dim=1) if tokens.dim() == 3 else tokens
            if hasattr(self.model, "classifier"):
                logits = self.model.classifier(feats)
            elif hasattr(self.model, "head"):
                logits = self.model.head(feats)
            else:
                logits = feats
        else:
            logits = self.model(x)
        if logits.dim() > 2:
            logits = logits.view(logits.size(0), -1)
        return logits

    def _compute_loss(self, logits: torch.Tensor, target: torch.Tensor):
        class_weights = self.class_weights if self.use_class_weights else None
        if class_weights is not None:
            class_weights = class_weights.to(logits.device)
        ce_loss = F.cross_entropy(
            logits,
            target,
            weight=class_weights,
            label_smoothing=self.label_smoothing,
        )
        if self.focal_weight <= 0.0:
            return ce_loss, ce_loss
        focal_loss = FocalLoss(alpha=class_weights, gamma=self.focal_gamma)(logits, target)
        loss = (1.0 - self.focal_weight) * ce_loss + self.focal_weight * focal_loss
        return loss, ce_loss

    def core_step(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        optimizer: Optional[torch.optim.Optimizer] = None,
        inner_optimizer: Optional[torch.optim.Optimizer] = None,
        is_training: bool = True,
    ) -> Dict[str, torch.Tensor]:
        support_y = support_y.long()
        query_y = query_y.long()

        if self.use_support_in_loss:
            x = torch.cat([support_x, query_x], dim=0)
            y = torch.cat([support_y, query_y], dim=0)
            query_size = query_x.size(0)
        else:
            x = query_x
            y = query_y
            query_size = query_x.size(0)

        logits = self._forward_logits(x)
        loss, ce_loss = self._compute_loss(logits, y)
        logits_query = logits[-query_size:] if query_size > 0 else logits

        return {
            "loss": loss,
            "ce_loss": ce_loss.detach(),
            "supcon_loss": 0.0,
            "logits": logits_query.detach(),
            "y_true": query_y,
        }
