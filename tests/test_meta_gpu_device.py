import os
import sys
import unittest
import torch

META_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "meta"))
if META_DIR not in sys.path:
    sys.path.insert(0, META_DIR)

from train_classifier import resolve_device


class TestMetaGPUDevice(unittest.TestCase):
    def test_device_selection(self):
        cfg = {"device": {"cuda_device": 0}}
        device, device_type = resolve_device(cfg)
        if torch.cuda.is_available():
            self.assertEqual(device.type, "cuda")
            self.assertEqual(device_type, "cuda")
            self.assertEqual(device.index, torch.cuda.current_device())
        else:
            self.assertEqual(device.type, "cpu")
            self.assertEqual(device_type, "cpu")


if __name__ == "__main__":
    unittest.main()
