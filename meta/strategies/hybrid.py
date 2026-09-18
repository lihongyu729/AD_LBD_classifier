import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from .base import BaseMetaStrategy
from .maml import MAMLStrategy
from .protonet import ProtoNetStrategy


class HybridStrategy(BaseMetaStrategy):
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        params = config.get("params", {})
        opt_weight = float(params.get("opt_weight", 0.1))
        metric_weight = float(params.get("metric_weight", 0.5))
        total = opt_weight + metric_weight
        if total <= 0:
            opt_weight, metric_weight, total = 0.5, 0.5, 1.0
        self.opt_weight = opt_weight / total
        self.metric_weight = metric_weight / total
        self._shape_warned = False
        opt_cfg = config.get("optimizer_strategy", {})
        metric_cfg = config.get("metric_strategy", {})
        self.opt_strategy = MAMLStrategy(model, opt_cfg)
        self.metric_strategy = ProtoNetStrategy(model, metric_cfg)

    def _normalize_logits(self, logits: Optional[torch.Tensor], target_classes: Optional[int] = None) -> Optional[torch.Tensor]:
        if logits is None:
            return None
        if logits.dim() <= 2:
            return logits
        class_dim = None
        if target_classes is not None and target_classes > 0:
            candidates = [d for d in range(1, logits.dim()) if logits.size(d) == target_classes]
            if len(candidates) > 0:
                class_dim = candidates[0]
        if class_dim is None:
            class_dim = min(range(1, logits.dim()), key=lambda d: logits.size(d))
        if class_dim != 1:
            logits = logits.movedim(class_dim, 1)
        return logits.flatten(start_dim=2).mean(dim=-1)

    def core_step(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
        query_y: torch.Tensor,
        optimizer: Optional[torch.optim.Optimizer] = None,
        inner_optimizer: Optional[torch.optim.Optimizer] = None,
        is_training: bool = True
    ) -> Dict[str, torch.Tensor]:
        opt_out = self.opt_strategy.core_step(
            support_x, support_y, query_x, query_y,
            optimizer=optimizer, inner_optimizer=inner_optimizer, is_training=is_training
        )
        metric_out = self.metric_strategy.core_step(
            support_x, support_y, query_x, query_y,
            optimizer=optimizer, inner_optimizer=inner_optimizer, is_training=is_training
        )
        loss = self.opt_weight * opt_out["loss"] + self.metric_weight * metric_out["loss"]
        sup_loss = self.opt_weight * opt_out.get("sup_loss", loss.new_tensor(0.0)) + self.metric_weight * metric_out.get("sup_loss", loss.new_tensor(0.0))

        y_true = opt_out.get("y_true", None)
        if y_true is None:
            y_true = metric_out.get("y_true", None)

        inferred_classes = None
        if y_true is not None and y_true.numel() > 0:
            inferred_classes = int(y_true.max().item()) + 1

        opt_logits = self._normalize_logits(opt_out.get("logits"), target_classes=inferred_classes)
        metric_logits = self._normalize_logits(metric_out.get("logits"), target_classes=inferred_classes)

        combined_logits = None
        if opt_logits is not None and metric_logits is not None:
            if opt_logits.shape == metric_logits.shape:
                combined_logits = self.opt_weight * opt_logits + self.metric_weight * metric_logits
            else:
                # Keep training alive when branch logits are structurally different.
                combined_logits = metric_logits if metric_logits.dim() == 2 else opt_logits
                if not self._shape_warned:
                    print(
                        f"[Hybrid][warn] logits shape mismatch: opt={tuple(opt_logits.shape)}, "
                        f"metric={tuple(metric_logits.shape)}. Using single-branch logits for metrics.",
                        flush=True,
                    )
                    self._shape_warned = True
        elif opt_logits is not None:
            combined_logits = opt_logits
        elif metric_logits is not None:
            combined_logits = metric_logits

        if combined_logits is not None and y_true is not None:
            with torch.no_grad():
                preds = combined_logits.argmax(dim=1)
                acc = (preds == y_true).float().mean()
        else:
            acc = self.opt_weight * opt_out.get("acc", loss.new_tensor(0.0)) + self.metric_weight * metric_out.get("acc", loss.new_tensor(0.0))
        return {
            "loss": loss,
            "acc": acc,
            "sup_loss": sup_loss,
            "logits": combined_logits,
            "y_true": y_true
        }
