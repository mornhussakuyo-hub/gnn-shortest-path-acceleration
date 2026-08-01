from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.export_g3_full_predictions import _validate_partial_predictions


class G3FullPredictionsTest(unittest.TestCase):
    def test_partial_replay_requires_train_validation_and_small_delta(self) -> None:
        region_ids = np.arange(1200)
        prediction = np.linspace(0.0, 1.0, 1200)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(("region_id", "split", "prediction"))
                for index in range(1020):
                    writer.writerow(
                        (
                            index,
                            "train" if index < 840 else "validation",
                            f"{prediction[index]:.9f}",
                        )
                    )
            delta = _validate_partial_predictions(path, region_ids, prediction)
            self.assertLess(delta, 1.0e-8)

    def test_partial_replay_rejects_changed_score(self) -> None:
        region_ids = np.arange(1200)
        prediction = np.zeros(1200)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(("region_id", "split", "prediction"))
                for index in range(1020):
                    writer.writerow(
                        (index, "train" if index < 840 else "validation", 1.0)
                    )
            with self.assertRaises(ValueError):
                _validate_partial_predictions(path, region_ids, prediction)


if __name__ == "__main__":
    unittest.main()
