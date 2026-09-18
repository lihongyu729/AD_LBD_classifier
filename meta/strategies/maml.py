
import torch
import torch.nn as nn
import torch.nn.functional as F
import higher
from typing import Dict, Any, Optional
from .base import BaseMetaStrategy

class MAMLStrategy(BaseMetaStrategy):
    """
    Model-Agnostic Meta-Learning (MAML) 及其变体 ANIL/FOMAML 实现。
    """
    def __init__(self, model: nn.Module, config: Dict[str, Any]):
        super().__init__(model, config)
        self.params = config.get("params", {})
        self.inner_lr = self.params.get("inner_lr", 0.01)
        self.inner_steps = self.params.get("inner_steps", 5)
        self.first_order = self.params.get("first_order", False)
        self.is_anil = config.get("name", "maml").lower() == "anil"
        self.label_smoothing = float(self.params.get("label_smoothing", 0.0))

        mode_str = "FOMAML (First-Order)" if self.first_order else "MAML (Second-Order)"
        print(f"[MAMLStrategy] Initialized in {mode_str} mode. Inner LR: {self.inner_lr}, Steps: {self.inner_steps}")
        
    def _get_anil_params(self):
        if hasattr(self.model, "classifier"):
            return self.model.classifier.parameters()
        if hasattr(self.model, "head"):
            return self.model.head.parameters()
        return self.model.parameters()

    def _prepare_ce_inputs(self, logits: torch.Tensor, target: torch.Tensor):
        """Normalize classification logits/targets for cross_entropy in meta episodes."""
        if target.dim() > 1:
            target = target.reshape(target.size(0), -1)
            target = target[:, 0]
        if target.dtype != torch.long:
            target = target.long()

        # Meta-classification expects [N, C]. If extra dims exist, detect class dim first.
        if logits.dim() > 2:
            class_count = int(target.max().item()) + 1 if target.numel() > 0 else None
            class_dim = None
            if class_count is not None and class_count > 0:
                candidates = [d for d in range(1, logits.dim()) if logits.size(d) == class_count]
                if len(candidates) > 0:
                    class_dim = candidates[0]
            if class_dim is None:
                # Fallback: class dim is usually the smallest non-batch dimension.
                class_dim = min(range(1, logits.dim()), key=lambda d: logits.size(d))
            if class_dim != 1:
                logits = logits.movedim(class_dim, 1)
            logits = logits.flatten(start_dim=2).mean(dim=-1)

        if logits.size(0) != target.size(0):
            if target.numel() == logits.size(0):
                target = target.reshape(logits.size(0))
            else:
                raise RuntimeError(f"Shape mismatch: logits {logits.shape}, target {target.shape}")
        return logits, target

    def core_step(self, 
                  support_x: torch.Tensor, 
                  support_y: torch.Tensor, 
                  query_x: torch.Tensor, 
                  query_y: torch.Tensor,
                  optimizer: Optional[torch.optim.Optimizer] = None,
                  inner_optimizer: Optional[torch.optim.Optimizer] = None,
                  is_training: bool = True) -> Dict[str, torch.Tensor]:
        
        if inner_optimizer is None:
            inner_params = self._get_anil_params() if self.is_anil else self.model.parameters()
            inner_optimizer = torch.optim.SGD(inner_params, lr=self.inner_lr)

        train_mode = self.model.training
        self.model.train()
        
        # 关键修复：MAML 需要在内循环计算梯度，即使是在验证/测试阶段（外层可能包裹了 no_grad）
        # 因此必须显式开启梯度计算，否则 higher 无法进行内循环更新
        with torch.enable_grad():
            if self.is_anil:
                # ANIL: extract features once (backbone not updated via inner loop),
                # then adapt only the head. Backbone now runs in BF16/FP16 via
                # the outer autocast context (no longer force-disabled). BF16 has
                # the same exponent range as FP32, so Mamba exp/log are safe.
                if hasattr(self.model, 'forward_encoder'):
                    sup_features, _ = self.model.forward_encoder(support_x)
                    que_features, _ = self.model.forward_encoder(query_x)
                else:
                    sup_features = support_x
                    que_features = query_x

                sup_features = sup_features.float()
                que_features = que_features.float()

                if hasattr(self.model, 'classifier'):
                    head_module = self.model.classifier
                elif hasattr(self.model, 'head'):
                    head_module = self.model.head
                else:
                    head_module = self.model

                # Inner loop: stay in FP32 for numerical stability with higher
                with torch.amp.autocast(device_type='cuda', enabled=False):
                    with higher.innerloop_ctx(head_module, inner_optimizer,
                                              copy_initial_weights=True,
                                              track_higher_grads=(not self.first_order and is_training)) as (fhead, diffopt):

                        for _ in range(self.inner_steps):
                            sup_logits = fhead(sup_features)
                            sup_logits, curr_target = self._prepare_ce_inputs(sup_logits, support_y)

                            sup_loss = F.cross_entropy(sup_logits, curr_target, label_smoothing=0.0)
                            diffopt.step(sup_loss)

                        que_logits = fhead(que_features)
                        que_logits, curr_query = self._prepare_ce_inputs(que_logits, query_y)
                        que_loss = F.cross_entropy(que_logits, curr_query, label_smoothing=self.label_smoothing)

                        with torch.no_grad():
                            preds = que_logits.argmax(dim=1)
                            acc = (preds == curr_query).float().mean()

                        sup_loss = sup_loss.detach()
                        
            else:
                # 标准 MAML 实现
                with higher.innerloop_ctx(self.model, inner_optimizer, 
                                          copy_initial_weights=True, 
                                          track_higher_grads=(not self.first_order and is_training)) as (fmodel, diffopt):
                    
                    for _ in range(self.inner_steps):
                        if hasattr(fmodel, 'forward_classifier'):
                            sup_logits = fmodel.forward_classifier(support_x)
                        else:
                            sup_logits = fmodel(support_x)

                        sup_logits, curr_support = self._prepare_ce_inputs(sup_logits, support_y)
                        sup_loss = F.cross_entropy(sup_logits, curr_support, label_smoothing=0.0) # Disable smoothing in inner loop
                        diffopt.step(sup_loss)
                    
                    if hasattr(fmodel, 'forward_classifier'):
                        que_logits = fmodel.forward_classifier(query_x)
                    else:
                        que_logits = fmodel(query_x)

                    que_logits, curr_query = self._prepare_ce_inputs(que_logits, query_y)
                    que_loss = F.cross_entropy(que_logits, curr_query, label_smoothing=self.label_smoothing)
                    
                    with torch.no_grad():
                        preds = que_logits.argmax(dim=1)
                        acc = (preds == curr_query).float().mean()
        
        if not train_mode:
            self.model.eval()
            
        return {
            "loss": que_loss,
            "acc": acc,
            "sup_loss": sup_loss.detach(),
            "logits": que_logits.detach(),
            "y_true": curr_query.detach()
        }
