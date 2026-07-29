from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import torch

    from src.demand_field_model import MLPConfig
    from src.demand_field_torch_model import TorchCudaMLPRegressor, require_cuda
except ModuleNotFoundError:
    torch = None


@unittest.skipIf(torch is None or not torch.cuda.is_available(), "CUDA PyTorch unavailable")
class TorchCudaMLPTest(unittest.TestCase):
    def test_cuda_model_learns_and_round_trips_weights(self) -> None:
        rng = np.random.default_rng(42)
        features = rng.normal(size=(128, 8)).astype(np.float32)
        labels = (
            3.0 * features[:, 0]
            - 2.0 * features[:, 1]
            + 0.5 * features[:, 2]
        ).astype(np.float32)
        config = MLPConfig(
            hidden_dims=(16, 8),
            learning_rate=0.01,
            rank_weight=0.1,
            batch_size=32,
            max_epochs=150,
            patience=30,
        )
        device = require_cuda()
        model = TorchCudaMLPRegressor(8, config, 42, device)
        model.fit(features[:96], labels[:96], features[96:], labels[96:])
        expected = model.predict(features)

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            weights = directory / "model.pt"
            manifest = directory / "model.json"
            saved = model.save(weights, manifest, {"test": True})
            loaded = TorchCudaMLPRegressor.load(weights, manifest, device)
            actual = loaded.predict(features)

        self.assertEqual(saved["execution"]["device_type"], "cuda")
        self.assertLess(float(np.mean(np.abs(expected - labels))), 0.7)
        np.testing.assert_array_equal(actual, expected)


if __name__ == "__main__":
    unittest.main()
