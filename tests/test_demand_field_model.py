"""验证第二版统一数据接口。"""

from __future__ import annotations

import unittest

import numpy as np

from src.demand_field_data import (
    EdgeRecord,
    build_demand_prototypes,
    build_node_features,
    overlap_grouped_candidate_split,
    stratified_candidate_split,
)
from src.demand_field_model import ranking_metrics_at_k
from src.graph_types import Query


class DemandFieldDataTest(unittest.TestCase):
    def test_node_features_use_history_counts_and_directed_static_edges(self) -> None:
        node_ids = np.asarray([10, 20, 30], dtype=np.int64)
        node_to_index = {10: 0, 20: 1, 30: 2}
        edges = [
            EdgeRecord(10, 20, 5.0, "primary"),
            EdgeRecord(20, 30, 10.0, "secondary"),
            EdgeRecord(30, 20, 7.0, "primary"),
        ]
        queries = [
            Query(0, 10, 30, count=2, timestamp=1),
            Query(1, 10, 20, count=1, timestamp=2),
        ]

        features, names, _ = build_node_features(
            node_ids=node_ids,
            node_to_index=node_to_index,
            edge_records=edges,
            history_queries=queries,
            road_types=("primary", "secondary"),
        )

        self.assertEqual(features.shape, (3, 10))
        self.assertEqual(names[0], "history_origin_count")
        self.assertEqual(names[1], "history_destination_count")
        self.assertEqual(features[0, 0], 1.0)
        self.assertEqual(features[2, 1], 1.0)
        self.assertGreater(features[1, names.index("in_road_type_primary")], 0.0)
        self.assertEqual(features[0, names.index("in_degree")], 0.0)

    def test_candidate_split_is_deterministic_and_stratified(self) -> None:
        methods = np.asarray(["a"] * 20 + ["b"] * 10, dtype="U1")
        first = stratified_candidate_split(
            methods,
            seed=42,
            train_fraction=0.70,
            validation_fraction=0.15,
        )
        second = stratified_candidate_split(
            methods,
            seed=42,
            train_fraction=0.70,
            validation_fraction=0.15,
        )

        np.testing.assert_array_equal(first, second)
        for method in ("a", "b"):
            group = first[methods == method]
            self.assertEqual(set(group.tolist()), {0, 1, 2})

    def test_overlap_grouped_split_keeps_near_duplicates_together(self) -> None:
        regions = np.asarray(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 4],
                [5, 6, 7, 8],
                [5, 6, 7, 9],
                [10, 11, 12, 13],
                [14, 15, 16, 17],
            ],
            dtype=np.int32,
        )
        methods = np.asarray(["random"] * len(regions))
        first, metadata = overlap_grouped_candidate_split(
            regions,
            methods,
            seed=42,
            train_fraction=0.50,
            validation_fraction=0.25,
            overlap_threshold=0.50,
        )
        second, _ = overlap_grouped_candidate_split(
            regions,
            methods,
            seed=42,
            train_fraction=0.50,
            validation_fraction=0.25,
            overlap_threshold=0.50,
        )

        np.testing.assert_array_equal(first, second)
        self.assertEqual(first[0], first[1])
        self.assertEqual(first[2], first[3])
        self.assertEqual(metadata["group_count"], 4)
        self.assertEqual(metadata["largest_group"], 2)

    def test_demand_prototypes_keep_weighted_origin_and_destination_sets(self) -> None:
        queries = [
            Query(0, 10, 30, count=2, timestamp=1),
            Query(1, 10, 20, count=1, timestamp=2),
            Query(2, 30, 10, count=1, timestamp=3),
            Query(3, 20, 10, count=2, timestamp=4),
        ]
        prototypes = build_demand_prototypes(
            history_queries=queries,
            coordinates={10: (0.0, 0.0), 20: (1.0, 0.0), 30: (1.0, 1.0)},
            node_to_index={10: 0, 20: 1, 30: 2},
            prototype_count=2,
            seed=42,
        )

        self.assertEqual(prototypes["prototype_weight"].shape, (2,))
        self.assertAlmostEqual(float(prototypes["prototype_weight"].sum()), 1.0)
        self.assertEqual(prototypes["history_query_prototype"].shape, (4,))
        for prototype_id in range(2):
            origin_start = prototypes["prototype_origin_offsets"][prototype_id]
            origin_end = prototypes["prototype_origin_offsets"][prototype_id + 1]
            destination_start = prototypes["prototype_destination_offsets"][prototype_id]
            destination_end = prototypes["prototype_destination_offsets"][prototype_id + 1]
            self.assertAlmostEqual(
                float(
                    prototypes["prototype_origin_weights"][origin_start:origin_end].sum()
                ),
                1.0,
            )
            self.assertAlmostEqual(
                float(
                    prototypes["prototype_destination_weights"][
                        destination_start:destination_end
                    ].sum()
                ),
                1.0,
            )

    def test_multi_k_metrics_report_candidate_redundancy(self) -> None:
        metrics = ranking_metrics_at_k(
            np.asarray([3.0, 2.0, 1.0]),
            np.asarray([30.0, 20.0, 10.0]),
            (1, 2),
            region_nodes=np.asarray([[0, 1], [1, 2], [3, 4]]),
        )
        self.assertEqual(metrics["1"]["mean_gain"], 30.0)
        self.assertEqual(metrics["2"]["unique_node_count"], 3)
        self.assertAlmostEqual(metrics["2"]["membership_redundancy"], 4 / 3)


if __name__ == "__main__":
    unittest.main()
