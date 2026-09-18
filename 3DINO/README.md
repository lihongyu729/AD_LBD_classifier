---
license: cc-by-nc-nd-4.0
extra_gated_prompt: "3DINO code and 3DINO-ViT weights are released under the CC BY-NC-ND 4.0 license. This means that you may use this framework for academic, research, and educational purposes and share or redistribute the original, unmodified version of this framework with proper attribution as detailed below. This means that you may not use this framework for commercial purposes, modify, adapt, or create derivative works based on this framework, or distribute a modified version of this framework. For full license details, refer to the official CC BY-NC-ND 4.0 License"
extra_gated_fields:
  Full Name: text
  Affiliation: text
  Country: country
  Affiliation Email: text
  Affiliation Type:
    type: select
    options: 
      - Academia
      - Industry
      - label: Other
        value: other
  Briefly explain your intended research use: text
  I agree to the terms of this form: checkbox
  I agree to use this model for non-commercial, academic use ONLY: checkbox
---

# Model Card for 3DINO-ViT

## A generalizable 3D framework and model for self-supervised learning in medical imaging

---
***npj Digital Medicine (2025)***

Tony Xu, 
Sepehr Hosseini, 
Chris Anderson, 
Anthony Rinaldi, 
Rahul G. Krishnan, 
Anne Martel,
Maged Goubran

[Codebase](https://github.com/AICONSlab/3DINO) | [Paper](https://doi.org/10.1038/s41746-025-02035-w) | [Preprint](https://arxiv.org/abs/2501.11755)

This repository contains the pretrained 3DINO-ViT model trained using the 3DINO framework, for training networks on unlabelled 3D medical images. 

**Abstract:** Current self-supervised learning methods for 3D medical imaging rely on simple pretext formulations and organ- or modality-specific datasets, limiting their generalizability and scalability. We present 3DINO, a cutting-edge SSL method adapted to 3D datasets, and use it to pretrain 3DINO-ViT: a general-purpose medical imaging model, on an exceptionally large, multimodal, and multi-organ dataset of ~100,000 3D medical imaging scans from over 10 organs. We validate 3DINO-ViT using extensive experiments on numerous medical imaging segmentation and classification tasks. Our results demonstrate that 3DINO-ViT generalizes across modalities and organs, including out-of-distribution tasks and datasets, outperforming state-of-the-art methods on the majority of evaluation metrics and labeled dataset sizes. Our 3DINO framework and 3DINO-ViT will be made available to enable research on 3D foundation models or further finetuning for a wide range of medical imaging applications.  

### Model Details

3DINO-ViT was trained using the [3DINO](https://doi.org/10.1038/s41746-025-02035-w) framework on around 100,000 3D medical images. It uses a 3D ViT architecture, and generates salient representations for diverse tasks in 3D medical imaging.  

**More details on model training and example usage can be found in the original [codebase](https://github.com/AICONSlab/3DINO).**   

### License
3DINO code and 3DINO-ViT weights are released under the CC BY-NC-ND 4.0 license.

✅ **You MAY:**

- Use this framework for **academic, research, and educational purposes**.
- Share or redistribute the original, **unmodified** version of this framework with proper attribution as detailed below.

❌ **You MAY NOT:**

- Use this framework for **commercial purposes** (as defined below).
- Modify, adapt, or create derivative works based on this framework.
- Distribute a modified version of this framework.

For full license details, refer to the official **[CC BY-NC-ND 4.0 License](https://creativecommons.org/licenses/by-nc-nd/4.0/)**

By Commercial Purposes, we mean that this framework **may not** be used:

- By **for-profit entities** for internal research, product development, or services.
- In **industry-funded** or **corporate-sponsored** research.
- As part of **commercially funded academic projects** without prior approval.
- In any project where the results will be used for **monetary gain** (e.g., patent filings, proprietary software development, licensing to industry).

If you are unsure whether your use qualifies as non-commercial, contact **maged.goubran@utoronto.ca**.

### Contact

For inquiries regarding permissions, exceptions, or licensing, contact **maged.goubran@utoronto.ca**.

### Acknowledgements

3DINO builds upon the excellent work from the original [DINOv2](https://github.com/facebookresearch/dinov2) and [ViT-Adapter](https://github.com/czczup/ViT-Adapter) for 2D natural images.

### Citing 3DINO-ViT

If you find 3DINO and 3DINO-ViT useful or use them in your research, please cite the following paper:

```
@article{xu3dino2025,
  title={A generalizable 3D framework and model for self-supervised learning in medical imaging},
  author={Xu, Tony and Hosseini, Sepehr and Anderson, Chris and Rinaldi, Anthony and Krishnan, Rahul G. and Martel, Anne L. and Goubran, Maged},
  journal={npj Digital Medicine},
  year={2025},
  doi={10.1038/s41746-025-02035-w},
}
```