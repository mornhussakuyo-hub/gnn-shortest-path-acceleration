"""验证 GNN 节点分数选区的去重与分散行为。"""

from __future__ import annotations

import unittest

from src.graph_types import Query, WeightedDiGraph
from src.regions import build_risk_aware_scored_regions, build_scored_regions


class GnnRegionSelectionTest(unittest.TestCase):
    def test_high_scores_do_not_create_overlapping_regions(self) -> None:
        graph = WeightedDiGraph()
        for node in range(1, 21):
            graph.add_node(node)
        for node in range(1, 20):
            graph.add_edge(node, node + 1, 1.0)
            graph.add_edge(node + 1, node, 1.0)

        scores = {node: 1.0 - abs(node - 5) * 0.01 for node in range(1, 21)}
        scores[16] = 0.8
        regions = build_scored_regions(
            graph,
            scores,
            region_count=2,
            region_size=4,
            seed_exclusion_hops=1,
        )

        self.assertEqual(len(regions), 2)
        self.assertFalse(regions[0].nodes & regions[1].nodes)
        self.assertEqual(regions[0].seed_node, 5)

    def test_region_level_endpoint_risk_changes_selection(self) -> None:
        graph = WeightedDiGraph()
        for node in range(1, 13):
            graph.add_node(node)
        for node in range(1, 12):
            graph.add_edge(node, node + 1, 1.0)
            graph.add_edge(node + 1, node, 1.0)

        scores = {node: 0.5 for node in range(1, 13)}
        scores[3] = 1.0
        scores[9] = 0.9
        queries = [Query(index, 3, 4) for index in range(100)]
        regions = build_risk_aware_scored_regions(
            graph,
            scores,
            queries,
            region_count=1,
            region_size=3,
            seed_exclusion_hops=0,
            candidate_limit=12,
            endpoint_risk_penalty=100.0,
        )

        self.assertEqual(len(regions), 1)
        self.assertEqual(regions[0].seed_node, 9)


if __name__ == "__main__":
    unittest.main()
