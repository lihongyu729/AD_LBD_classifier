import os
import sys
import unittest

import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CODE_ROOT = os.path.join(REPO_ROOT, "code")
if CODE_ROOT not in sys.path:
    sys.path.insert(0, CODE_ROOT)

from data.augmentations import DataAugmentationDINO3d


class TestSeedBounds(unittest.TestCase):
    """
    函数作用：
    - 验证种子边界值不会触发 OverflowError，并确保随机状态可复现。
    """

    def _build_aug(self, seed):
        """
        函数作用：
        - 构造 DataAugmentationDINO3d 实例以用于边界种子测试。
        """
        return DataAugmentationDINO3d(
            global_crops_in_slice_scale=(0.5, 1.0),
            global_crops_cross_slice_scale=(0.5, 1.0),
            local_crops_in_slice_scale=(0.4, 0.8),
            local_crops_cross_slice_scale=(0.4, 0.8),
            local_crops_number=1,
            global_crops_size=16,
            local_crops_size=8,
            seed=seed,
        )

    def _run_once(self, aug):
        """
        函数作用：
        - 对固定输入执行一次增强，返回输出字典用于复现性比较。
        """
        img = torch.arange(1 * 16 * 16 * 16, dtype=torch.float32).reshape(1, 16, 16, 16)
        output, _ = aug(img)
        return output

    def _assert_reproducible(self, seed):
        """
        函数作用：
        - 对同一 seed 重置随机状态后验证输出一致性。
        """
        aug = self._build_aug(seed)
        out1 = self._run_once(aug)
        aug._set_safe_random_state(seed)
        out2 = self._run_once(aug)
        for t1, t2 in zip(out1["global_crops"], out2["global_crops"]):
            self.assertTrue(torch.allclose(t1, t2))
        for t1, t2 in zip(out1["local_crops"], out2["local_crops"]):
            self.assertTrue(torch.allclose(t1, t2))

    def test_seed_bounds(self):
        """
        函数作用：
        - 传入边界种子并验证不抛出 OverflowError 且结果可复现。
        """
        seeds = [0, 4294967295, 4294967296, (2 ** 32) + 1000]
        for seed in seeds:
            self._assert_reproducible(seed)


if __name__ == "__main__":
    unittest.main()
