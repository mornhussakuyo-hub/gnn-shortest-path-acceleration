from __future__ import annotations

import unittest

import numpy as np

from scripts.evaluate_multi_region_online import (
    K_VALUES,
    METHOD_NAMES,
    _build_selections,
    _history_hotspot_scores,
)


class _Dataset:
    region_ids = np.asarray([10, 11, 12, 13])
    region_nodes = np.asarray([[0, 1], [0, 2], [3, 4], [5, 6]])
    region_feature_names = (
        "mean_history_origin_count",
        "mean_history_destination_count",
        "max_history_origin_count",
        "max_history_destination_count",
    )
    region_features = np.asarray(
        [
            [1.0, 2.0, 3.0, 4.0],
            [0.0, 1.0, 0.0, 1.0],
            [2.0, 2.0, 2.0, 2.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )


class _SelectionDataset:
    region_ids = np.arange(20)
    region_nodes = np.arange(40).reshape(20, 2)


class MultiRegionOnlineTest(unittest.TestCase):
    def test_history_hotspot_uses_only_four_frozen_demand_features(self) -> None:
        np.testing.assert_array_equal(
            _history_hotspot_scores(_Dataset()), [10.0, 2.0, 8.0, 4.0]
        )

    def test_every_method_builds_full_hard_disjoint_sets(self) -> None:
        dataset = _SelectionDataset()
        score_sources = {
            method: np.arange(20, dtype=np.float64)
            for method in METHOD_NAMES
        }
        rows = {
            int(region_id): {"shortcut_count": "5"}
            for region_id in dataset.region_ids
        }
        selections = _build_selections(
            dataset=dataset,
            score_sources=score_sources,
            current_labels=np.ones(20),
            future_labels=np.ones(20),
            current_rows=rows,
        )
        for method in METHOD_NAMES:
            for k in K_VALUES:
                selection = selections[method][str(k)]
                self.assertEqual(len(selection["selected_region_ids"]), k)
                self.assertTrue(
                    selection["overlap"]["deployable_without_region_overlap"]
                )


if __name__ == "__main__":
    unittest.main()
