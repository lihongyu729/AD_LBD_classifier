"""
Abstract base class for all benchmark methods.

Every method must implement: build_model, forward_encoder, forward_classifier,
and get_optimizer_param_groups.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import os
import torch.nn as nn


class BaseMethod(ABC):
    """
    Abstract base for all classification methods in the benchmark.

    Lifecycle:
        1. __init__(config)  — store config, model=None
        2. build_model()     — construct nn.Module
        3. prepare(gpu_id)   — move to device, load pretrained weights
        4. Training loop calls train_step() or meta_step()
        5. Evaluation calls forward_classifier()

    Subclasses for meta-learning methods should additionally override
    get_strategy_config() and use meta_step().
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.model: Optional[nn.Module] = None
        self.device = None

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Construct and return the PyTorch model. Called once per fold."""
        pass

    @abstractmethod
    def forward_encoder(self, x):
        """
        Extract features/embeddings from input tensor.
        Args:
            x: [B, C, D, H, W] input tensor
        Returns:
            features: [B, feat_dim] tensor
        """
        pass

    @abstractmethod
    def forward_classifier(self, x):
        """
        Return classification logits from input tensor.
        Args:
            x: [B, C, D, H, W] input tensor
        Returns:
            logits: [B, num_classes] tensor
        """
        pass

    @abstractmethod
    def get_optimizer_param_groups(self) -> List[Dict]:
        """
        Return parameter groups for the optimizer.

        Should implement the backbone_lr_ratio pattern:
            - backbone params: lr = base_lr * backbone_lr_ratio
            - head/classifier params: lr = base_lr

        Returns:
            List of dicts, each with 'params' and optional 'lr' key.
        """
        pass

    def prepare_for_training(self, device, pretrained_path: Optional[str] = None):
        """Move model to device, load pretrained weights if provided."""
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained(pretrained_path)
        return self

    def _load_pretrained(self, path: str):
        """Load pretrained weights with best-effort matching.

        Includes ssm_b→ssm_a remapping for single-branch ablation:
        dual-branch pretrained weights (ssm_a=conv, ssm_b=mamba) are
        semantically valid for mamba-only single-branch ssm_a.
        """
        import torch
        checkpoint = torch.load(path, map_location="cpu")
        # Handle various checkpoint formats
        if isinstance(checkpoint, dict):
            state_dict = None
            for key in ["state_dict", "model_state", "model", "student", "teacher", "ema", "backbone"]:
                if key in checkpoint:
                    state_dict = checkpoint[key]
                    break
            if state_dict is None:
                state_dict = checkpoint
        else:
            state_dict = checkpoint

        # Strip common prefixes
        cleaned = {}
        for k, v in state_dict.items():
            for prefix in ["module.", "encoder.", "backbone.", "model."]:
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            # Alias patch.* to patch_embed.*
            if k.startswith("patch.") and not k.startswith("patch_embed."):
                k = "patch_embed." + k[len("patch."):]
            cleaned[k] = v

        # ------------------------------------------------------------------
        # ssm_b → ssm_a remapping for single-branch ablation
        #
        # In dual-branch MAE pretraining, ssm_b (mamba) processes the same
        # feat_seqs_batch as ssm_a (conv) and receives real reconstruction
        # gradients.  When the target model is a single-branch mamba-only
        # variant, those pretrained mamba weights are semantically valid for
        # ssm_a (same SeqSSM architecture, same input, same data domain).
        # ------------------------------------------------------------------
        remapped = {}
        for k, v in cleaned.items():
            if ".ssm_b." in k:
                remapped[k.replace(".ssm_b.", ".ssm_a.")] = v
            else:
                remapped[k] = v

        model_dict = self.model.state_dict()
        # Merge clean matches, then fill remaining gaps with remapped matches
        matched = {}
        for k, v in cleaned.items():
            if k in model_dict and v.shape == model_dict[k].shape:
                matched[k] = v
        n_before = len(matched)
        for k, v in remapped.items():
            if k not in matched and k in model_dict and v.shape == model_dict[k].shape:
                matched[k] = v
        n_after = len(matched)
        tag = " (ssm_b→ssm_a remap)" if n_after > n_before else ""
        print(f"[Pretrained] Loaded {n_after}/{len(model_dict)} params from {os.path.basename(path)}{tag}", flush=True)
        model_dict.update(matched)
        self.model.load_state_dict(model_dict, strict=False)

    def get_strategy_config(self) -> Optional[Dict]:
        """Return meta-learning strategy config. None = standard training."""
        return None

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Method", "").lower()

    def train(self):
        if self.model is not None:
            self.model.train()

    def eval(self):
        if self.model is not None:
            self.model.eval()

    def parameters(self):
        return self.model.parameters() if self.model is not None else []
