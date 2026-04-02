"""Minimal regression tests for the classification module."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover
    torch = None


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if torch is not None:
    from classification_utils import (  # noqa: E402
        ACFTransform,
        ConfigurableResNet18,
        ConfigurableSimpleCNN,
        NormalizeTensor,
        SpeckleRealDataset,
        compute_acf,
    )


@unittest.skipIf(torch is None, "PyTorch is not installed in the current environment.")
class ClassificationModuleTest(unittest.TestCase):
    """Fast interface checks for the migrated classification code."""

    def test_simplecnn_forward_shape(self) -> None:
        model = ConfigurableSimpleCNN(num_classes=15, in_channels=1, padding_mode="zeros")
        output = model(torch.randn(2, 1, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 15))

    def test_resnet_forward_shape(self) -> None:
        model = ConfigurableResNet18(num_classes=15, in_channels=1, padding_mode="zeros")
        output = model(torch.randn(2, 1, 64, 64))
        self.assertEqual(tuple(output.shape), (2, 15))

    def test_acf_center_is_normalized(self) -> None:
        image = np.random.default_rng(0).normal(size=(16, 16))
        acf = compute_acf(image)
        self.assertTrue(np.isclose(acf[8, 8], 1.0, atol=1e-5))

    def test_real_dataset_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            array = np.ones((32, 32), dtype=np.float32)
            np.save(root / "sample.npy", array)
            metadata = pd.DataFrame([["sample.npy", 1]], columns=["filepath", "class_label"])
            transform = lambda x: NormalizeTensor(1.0, 1.0)(ACFTransform()(x))
            dataset = SpeckleRealDataset(
                metadata,
                root,
                transform=transform,
                canvas_size=(32, 32),
                final_size=(16, 16),
                shift=0,
                shift_mode="fixed",
                is_train=False,
            )
            image, label = dataset[0]
            self.assertEqual(tuple(image.shape), (1, 16, 16))
            self.assertEqual(int(label.item()), 0)


if __name__ == "__main__":
    unittest.main()
