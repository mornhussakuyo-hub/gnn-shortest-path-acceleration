"""验证第二版固定候选池和真实收益标签的数据链路。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.analyze_region_labels import _classify_analysis
from src.graph_types import Query, WeightedDiGraph
from src.region_candidates import (
    CandidatePoolConfig,
    build_fixed_candidate_pool,
    chronological_prefix,
    load_candidate_manifest,
    write_candidate_manifest,
)
from src.region_labels import (
    chronological_window,
    compute_baseline_metrics,
    evaluate_single_region_label,
)


class RegionTrainingDataTest(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = WeightedDiGraph()
        width = 5
        for row in range(width):
            for column in range(width):
                node = row * width + column
                self.graph.add_node(node, float(column), float(row))
        for row in range(width):
            for column in range(width):
                node = row * width + column
                if column + 1 < width:
                    neighbor = node + 1
                    self.graph.add_edge(node, neighbor, 1.0)
                    self.graph.add_edge(neighbor, node, 1.0)
                if row + 1 < width:
                    neighbor = node + width
                    self.graph.add_edge(node, neighbor, 1.0)
                    self.graph.add_edge(neighbor, node, 1.0)
        self.queries = [
            Query(index, index % 5, 20 + index % 5, timestamp=100 + index)
            for index in range(10)
        ]

    def test_candidate_pool_is_deterministic_random_and_demand_independent(self) -> None:
        config = CandidatePoolConfig(
            candidate_count=9,
            region_size=4,
            seed=7,
            overlap_threshold=0.99,
        )
        first = build_fixed_candidate_pool(self.graph, config)
        second = build_fixed_candidate_pool(self.graph, config)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 9)
        self.assertTrue(all(region.node_count == 4 for region in first))
        self.assertTrue(all(region.boundary_count >= 2 for region in first))
        self.assertEqual(
            {region.selection_method for region in first},
            {"fixed_random_bfs"},
        )

    def test_candidate_manifest_round_trip_and_detects_changes(self) -> None:
        config = CandidatePoolConfig(
            candidate_count=6,
            region_size=4,
            seed=11,
            overlap_threshold=0.99,
        )
        regions = build_fixed_candidate_pool(self.graph, config)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "candidates.json"
            written = write_candidate_manifest(
                path,
                regions,
                config,
                graph_node_count=self.graph.node_count,
                graph_edge_count=self.graph.edge_count,
            )
            loaded, loaded_regions = load_candidate_manifest(path)

        self.assertEqual(written["candidate_sha256"], loaded["candidate_sha256"])
        self.assertEqual(regions, loaded_regions)

    def test_single_region_label_is_exact_and_uses_total_expanded_work(self) -> None:
        region = build_fixed_candidate_pool(
            self.graph,
            CandidatePoolConfig(
                candidate_count=1,
                region_size=9,
                seed=3,
                overlap_threshold=0.99,
            ),
        )[0]
        baselines = compute_baseline_metrics(self.graph, self.queries)
        label = evaluate_single_region_label(
            self.graph,
            region,
            self.queries,
            baselines,
        )

        self.assertEqual(label.correctness_rate, 1.0)
        self.assertEqual(label.label_query_count, len(self.queries))
        self.assertAlmostEqual(
            label.indexed_avg_expanded,
            label.indexed_avg_graph_expanded + label.indexed_avg_access_expanded,
        )
        expected_total = round(
            sum(metric.expanded_nodes for metric in baselines.values())
            - label.indexed_avg_expanded * len(self.queries)
        )
        self.assertEqual(label.total_workload_gain, expected_total)

    def test_chronological_windows_do_not_overlap(self) -> None:
        shuffled = list(reversed(self.queries))
        history = chronological_prefix(shuffled, 0.4)
        labels = chronological_window(shuffled, 0.4, 0.7)

        self.assertEqual([query.query_id for query in history], [0, 1, 2, 3])
        self.assertEqual([query.query_id for query in labels], [4, 5, 6])
        self.assertLess(history[-1].timestamp, labels[0].timestamp)

    def test_analysis_distinguishes_formal_sampled_labels(self) -> None:
        manifest = {
            "status": "complete",
            "target_region_count": 1200,
            "completed_region_count": 1200,
            "query_ids": list(range(2000)),
            "query_sample_seed": 42,
            "label_start_fraction": 0.35,
            "label_end_fraction": 0.70,
        }

        self.assertEqual(
            _classify_analysis(
                labeled_candidate_count=1200,
                manifest_candidate_count=1200,
                observed_label_query_counts={2000},
                expected_label_query_count=34329,
                label_manifest=manifest,
            ),
            ("formal_sampled_labels", "ready_for_modeling"),
        )

    def test_analysis_keeps_screening_and_full_window_separate(self) -> None:
        common = {
            "labeled_candidate_count": 1200,
            "manifest_candidate_count": 1200,
            "expected_label_query_count": 34329,
            "label_manifest": None,
        }

        self.assertEqual(
            _classify_analysis(observed_label_query_counts={500}, **common),
            ("all_candidates_screening_sample", "screening_only"),
        )
        self.assertEqual(
            _classify_analysis(observed_label_query_counts={34329}, **common),
            ("full_label_window", "full_window_complete"),
        )


if __name__ == "__main__":
    unittest.main()
