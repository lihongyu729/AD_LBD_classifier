
import unittest
print("Starting test_train_refactor.py...")
import os
import sys
import shutil
import tempfile
import torch
import json
import numpy as np

# Add parent directory to path to import train_classifier
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train_classifier import _generate_grid_candidates, _run_hpo, _train_one_trial_cv, MRIVolumeLabeledDataset

class DummyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.items = [("a", 0), ("b", 1), ("c", 0), ("d", 1)]
        self.data = [torch.randn(1, 16, 16, 16) for _ in self.items] # 16^3 small size
        self.in_channels = 1

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        # return tensor, label
        return self.data[idx], torch.tensor(self.items[idx][1], dtype=torch.long)

class TestTrainRefactor(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.report_file = os.path.join(self.test_dir, "report.txt")
        self.cfg = {
            "paths": {
                "out_dir": self.test_dir,
                "report_file": self.report_file
            },
            "dataset": {
                "num_classes": 2
            },
            "classifier": {
                "epochs": 1,
                "batch_size": 2,
                "loss": "ce"
            },
            "hpo": {
                "enabled": True,
                "algorithm": "grid",
                "trials": 5,
                "folds": 2,
                "search_space": {
                    "lr": {"type": "choice", "values": [0.01, 0.001]},
                    "dropout": {"type": "choice", "values": [0.1, 0.2]}
                }
            },
            "ss3m": {
                "embed_dim": 8,
                "depth": 1,
                "patch_size": (4, 4, 4)
            }
        }
    
    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_grid_generation(self):
        space = {
            "a": {"type": "choice", "values": [1, 2]},
            "b": {"type": "choice", "values": [True, False]}
        }
        candidates = _generate_grid_candidates(space)
        # Should be 2 * 2 = 4 candidates
        self.assertEqual(len(candidates), 4)
        expected = [
            {"a": 1, "b": True},
            {"a": 1, "b": False},
            {"a": 2, "b": True},
            {"a": 2, "b": False}
        ]
        # Sort by 'a' then 'b' to compare (bool sorts as int)
        candidates.sort(key=lambda x: (x['a'], x['b']))
        expected.sort(key=lambda x: (x['a'], x['b']))
        self.assertEqual(candidates, expected)

    def test_run_hpo_grid_report(self):
        # This integration test runs the HPO loop with grid search
        # It uses DummyDataset and checks if report.txt is created
        ds = DummyDataset()
        device = torch.device("cpu")
        
        # Override config for speed
        self.cfg["hpo"]["search_space"] = {
            "lr": {"type": "choice", "values": [0.01]}, # single value to run fast
            "dropout": {"type": "choice", "values": [0.0]}
        }
        self.cfg["classifier"]["epochs"] = 1
        
        _run_hpo(self.cfg, "ss3m", ds, 1, (16, 16, 16), device, "cpu")
        
        self.assertTrue(os.path.isfile(self.report_file), "report.txt should be created")
        
        with open(self.report_file, "r") as f:
            content = f.read()
            self.assertIn("====== Training Report", content)
            self.assertIn("Best Hyperparameters:", content)
            self.assertIn("lr: 0.01", content)
        
        with open(os.path.join(os.path.dirname(__file__), "..", "test_success.txt"), "w") as f:
            f.write("SUCCESS")

if __name__ == "__main__":
    unittest.main()
