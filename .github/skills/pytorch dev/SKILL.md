---
name: pytorch-dev
description: "Use when: PyTorch training/debugging, tensor shapes, device issues, dataloader, optimization."
---

# PyTorch Development Standards
- **Tensor Shapes**: Always add inline comments showing the expected tensor shapes (e.g., `[B, C, H, W]`) after any dimension-altering operations (`view`, `reshape`, `transpose`, `unsqueeze`).
- **Device Agnostic**: Never hardcode `.cuda()` or `.cpu()`. Always use `.to(device)` where `device` is defined dynamically.
- **Reproducibility**: When initializing training scripts, always include code to set random seeds for `torch`, `numpy`, and `random`.
- **Memory Management**: When writing validation or inference loops, always wrap the code in `with torch.no_grad():` or `with torch.inference_mode():`.
- **Training Loops**: Ensure `optimizer.zero_grad()` is called before `loss.backward()`.