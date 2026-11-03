"""Small offline contract tests; they do not pretend to evaluate ECG accuracy."""

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "nn"))
sys.path.insert(0, str(ROOT / "nn" / "scripts"))

from models import ECGResNet1D
from prepare_ptbxl import LABELS, map_labels
from preprocess import canonicalize_lead_names, filter_record, normalize


class TrainingPipelineTests(unittest.TestCase):
    def test_label_mapping_is_multilabel_and_ignores_nondiagnostic_codes(self):
        statements = pd.DataFrame(
            {
                "diagnostic": [1, 1, 0],
                "diagnostic_class": ["MI", "STTC", "MI"],
            },
            index=["A", "B", "C"],
        )
        actual = map_labels({"A": 100, "B": 100, "C": 100, "UNKNOWN": 1}, statements)
        expected = np.array([0, 1, 1, 0, 0], dtype=np.uint8)
        self.assertEqual(LABELS, ("NORM", "MI", "STTC", "CD", "HYP"))
        np.testing.assert_array_equal(actual, expected)

    def test_filter_keeps_fixed_shape_finite_physical_signal(self):
        t = np.arange(5000, dtype=np.float64) / 500.0
        x = np.stack([0.7 * np.sin(2 * np.pi * 1.2 * t + phase)
                      for phase in np.linspace(0, 1, 12)], axis=1)
        y = filter_record(x)
        self.assertEqual(y.shape, (5000, 12))
        self.assertEqual(y.dtype, np.float32)
        self.assertTrue(np.isfinite(y).all())
        self.assertGreater(float(np.std(y[:, 0])), 0.05)

    def test_wfdb_augmented_lead_names_are_canonicalized_without_reordering(self):
        wfdb_names = ("I", "II", "III", "AVR", "AVL", "AVF",
                      "V1", "V2", "V3", "V4", "V5", "V6")
        self.assertEqual(canonicalize_lead_names(wfdb_names), (
            "I", "II", "III", "aVR", "aVL", "aVF",
            "V1", "V2", "V3", "V4", "V5", "V6",
        ))

    def test_training_normalization_and_model_interface(self):
        x = np.zeros((12, 5000), dtype=np.float32)
        normalized = normalize(x, np.zeros(12), np.ones(12))
        self.assertEqual(normalized.shape, (12, 5000))
        model = ECGResNet1D().eval()
        with torch.inference_mode():
            logits = model(torch.from_numpy(normalized[None]))
        self.assertEqual(tuple(logits.shape), (1, 5))
        self.assertTrue(torch.isfinite(logits).all())


if __name__ == "__main__":
    unittest.main()
