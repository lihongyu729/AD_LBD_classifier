
import sys
import os
import torch
import numpy as np
import monai.transforms.compose
import monai.transforms.transform
import traceback

sys.path.insert(0, os.path.join(os.getcwd(), 'code'))
from data.augmentations import DataAugmentationDINO3d

def reproduce():
    print('Starting reproduction script...')
    print(f"PyTorch Version: {torch.__version__}")
    print(f"MONAI Version: {monai.__version__}")
    print(f"NumPy Version: {np.__version__}")
    
    aug = DataAugmentationDINO3d(
        global_crops_in_slice_scale=(0.5, 1.0),
        global_crops_cross_slice_scale=(0.5, 1.0),
        local_crops_in_slice_scale=(0.4, 0.8),
        local_crops_cross_slice_scale=(0.4, 0.8),
        local_crops_number=1,
        global_crops_size=16,
        local_crops_size=8,
        seed=(2**32)-1
    )
    
    img = torch.zeros(1, 16, 16, 16)
    print(f'Input image shape: {img.shape}')
    
    print("Running 20 iterations to trigger probabilistic transforms...")
    try:
        for i in range(20):
            aug(img)
            print(f'Iteration {i+1}/20 Success')
    except Exception:
        print("\n!!! EXCEPTION CAUGHT !!!")
        traceback.print_exc()
        print("!!! END OF TRACEBACK !!!")

if __name__ == '__main__':
    reproduce()
