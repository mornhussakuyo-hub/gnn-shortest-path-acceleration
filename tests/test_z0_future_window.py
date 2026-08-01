from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_z0_future_window import (
    FORMAL_QUERY_COUNT,
    _all_scope_metrics,
    _top_k_overlap,
    _validate_and_align,
)
from src.demand_field_data import SPLIT_NAMES
from src.region_labels import LABEL_SCHEMA, LABEL_WORK_DEFINITION


class _Dataset:
    def __init__(self) -> None:
        self.region_ids = np.arange(1, 25, dtype=np.int32)
        self.region_nodes = np.arange(24 * 4, dtype=np.int32).reshape(24, 4)
        self.labels = np.linspace(2.0, 25.0, 24, dtype=np.float32)
        self.split = np.repeat(np.arange(3, dtype=np.int8), 8)
        self.history_query_ids = np.asarray([1, 2, 3], dtype=np.int64)
        self.manifest = {
            "candidate_sha256": "candidate-digest",
            "dataset_sha256": "dataset-digest",
            "history_window": {"start_fraction": 0.0, "end_fraction": 0.35},
        }

    def split_mask(self, name: str) -> np.ndarray:
        return self.split == SPLIT_NAMES.index(name)


class Z0FutureWindowTest(unittest.TestCase):
    def setUp(self) -> None:
        self.dataset = _Dataset()
        self.current_manifest = {
            "label_start_fraction": 0.35,
            "label_end_fraction": 0.70,
            "query_ids": [10, 11],
            "query_sample_seed": 42,
        }
        region_ids = list(map(int, self.dataset.region_ids))
        future_query_ids = list(range(10_000, 10_000 + FORMAL_QUERY_COUNT))
        self.future_manifest = {
            "schema": LABEL_SCHEMA,
            "status": "complete",
            "candidate_sha256": "candidate-digest",
            "label_start_fraction": 0.70,
            "label_end_fraction": 1.00,
            "query_ids": future_query_ids,
            "query_sample_seed": 42,
            "work_definition": LABEL_WORK_DEFINITION,
            "endpoint_cache_capacity": 0,
            "target_region_count": len(region_ids),
            "completed_region_count": len(region_ids),
            "target_region_ids": region_ids,
            "completed_region_ids": region_ids,
        }
        self.future_rows = {
            region_id: {
                "avg_workload_gain": str(30.0 - index),
                "label_query_count": str(FORMAL_QUERY_COUNT),
                "correctness_rate": "1.0",
            }
            for index, region_id in enumerate(region_ids)
        }
        self.z0_rows = {
            region_id: {
                "label": str(float(self.dataset.labels[index])),
                "z0_score": str(float(index)),
            }
            for index, region_id in enumerate(region_ids)
        }

    def test_validation_aligns_frozen_scores_and_future_labels(self) -> None:
        future, z0 = _validate_and_align(
            dataset=self.dataset,
            current_manifest=self.current_manifest,
            future_manifest=self.future_manifest,
            future_rows=self.future_rows,
            z0_rows=self.z0_rows,
        )

        self.assertEqual(future.shape, (24,))
        self.assertEqual(z0.shape, (24,))
        self.assertEqual(float(future[0]), 30.0)
        self.assertEqual(float(z0[-1]), 23.0)

    def test_validation_rejects_temporal_overlap(self) -> None:
        self.future_manifest["query_ids"][0] = 2

        with self.assertRaisesRegex(ValueError, "history and future"):
            _validate_and_align(
                dataset=self.dataset,
                current_manifest=self.current_manifest,
                future_manifest=self.future_manifest,
                future_rows=self.future_rows,
                z0_rows=self.z0_rows,
            )

    def test_metrics_report_all_and_spatial_scopes(self) -> None:
        prediction = np.arange(24, dtype=np.float64)
        target = np.arange(24, dtype=np.float64)

        metrics = _all_scope_metrics(self.dataset, prediction, target)

        self.assertEqual(
            set(metrics), {"all_candidates", "train", "validation", "holdout"}
        )
        self.assertAlmostEqual(metrics["all_candidates"]["spearman"], 1.0)
        self.assertEqual(metrics["holdout"]["ranking_at_k"]["18"]["k"], 8)

    def test_top_k_overlap_uses_fixed_k_sets(self) -> None:
        values = np.arange(24, dtype=np.float64)
        overlap = _top_k_overlap(values, values, values)

        self.assertEqual(overlap["5"]["z0_future_oracle_intersection"], 5)
        self.assertAlmostEqual(overlap["18"]["current_future_oracle_jaccard"], 1.0)


if __name__ == "__main__":
    unittest.main()
