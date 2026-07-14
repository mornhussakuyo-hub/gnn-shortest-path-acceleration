"""验证物化压缩索引的节点状态、正确性与回退行为。"""

from __future__ import annotations

import unittest

from src.compression_index import (
    NODE_BOUNDARY,
    NODE_INTERNAL,
    NODE_OUTSIDE,
    build_compression_index,
)
from src.graph_types import WeightedDiGraph
from src.indexed_query import EndpointAccessCache, indexed_bidirectional_dijkstra_distance
from src.regions import Region
from src.shortest_path import bidirectional_dijkstra_distance


class MaterializedCompressionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = WeightedDiGraph()
        for node in range(1, 7):
            self.graph.add_node(node)
        for source, target, weight in [
            (1, 2, 1.0),
            (2, 1, 1.0),
            (2, 3, 1.0),
            (3, 2, 1.0),
            (3, 4, 1.0),
            (4, 3, 1.0),
            (4, 5, 1.0),
            (5, 4, 1.0),
            (2, 4, 10.0),
            (5, 6, 1.0),
            (6, 5, 1.0),
        ]:
            self.graph.add_edge(source, target, weight)

        region = Region(
            region_id=0,
            nodes=frozenset({2, 3, 4}),
            boundary_nodes=frozenset({2, 4}),
            seed_node=3,
            selection_method="test",
        )
        self.index = build_compression_index(self.graph, [region])

    def test_materializes_node_states_and_shortcut(self) -> None:
        self.assertEqual(self.index.node_states[1], NODE_OUTSIDE)
        self.assertEqual(self.index.node_states[2], NODE_BOUNDARY)
        self.assertEqual(self.index.node_states[3], NODE_INTERNAL)
        self.assertFalse(self.index.compressed_graph.has_node(3))
        self.assertEqual(self.index.compressed_graph.node_count, 5)
        self.assertIn((4, 2.0), self.index.compressed_graph.out_neighbors(2))
        self.assertNotIn((4, 10.0), self.index.compressed_graph.out_neighbors(2))

    def test_compressed_query_matches_original_graph(self) -> None:
        expected = bidirectional_dijkstra_distance(self.graph, 1, 6)
        actual = indexed_bidirectional_dijkstra_distance(self.graph, self.index, 1, 6)
        self.assertEqual(actual.distance, expected.distance)
        self.assertFalse(self.index.requires_original_graph(1, 6))

    def test_internal_endpoint_uses_exact_local_access(self) -> None:
        expected = bidirectional_dijkstra_distance(self.graph, 3, 6)
        actual = indexed_bidirectional_dijkstra_distance(self.graph, self.index, 3, 6)
        self.assertEqual(actual.distance, expected.distance)
        self.assertFalse(self.index.requires_original_graph(3, 6))
        self.assertTrue(self.index.requires_endpoint_access(3, 6))

    def test_endpoint_access_cache_reuses_boundary_frontier(self) -> None:
        cache = EndpointAccessCache(capacity=4)

        first = indexed_bidirectional_dijkstra_distance(
            self.graph,
            self.index,
            3,
            6,
            endpoint_cache=cache,
        )
        second = indexed_bidirectional_dijkstra_distance(
            self.graph,
            self.index,
            3,
            6,
            endpoint_cache=cache,
        )

        self.assertEqual(first.distance, second.distance)
        self.assertEqual(first.endpoint_cache_misses, 1)
        self.assertEqual(first.endpoint_cache_hits, 0)
        self.assertGreater(first.endpoint_access_expanded_nodes, 0)
        self.assertEqual(second.endpoint_cache_hits, 1)
        self.assertEqual(second.endpoint_cache_misses, 0)
        self.assertEqual(second.endpoint_access_expanded_nodes, 0)

    def test_endpoint_access_cache_uses_lru_eviction(self) -> None:
        cache = EndpointAccessCache(capacity=2)
        cache.put(1, False, 0, {2: 1.0})
        cache.put(2, False, 0, {3: 1.0})
        self.assertIsNotNone(cache.get(1, False, 0))

        cache.put(3, False, 0, {4: 1.0})

        self.assertEqual(len(cache), 2)
        self.assertIsNone(cache.get(2, False, 0))
        self.assertIsNotNone(cache.get(1, False, 0))
        self.assertIsNotNone(cache.get(3, False, 0))

    def test_cache_enabled_and_disabled_match_for_all_pairs(self) -> None:
        cache = EndpointAccessCache(capacity=4)
        for source in range(1, 7):
            for target in range(1, 7):
                with self.subTest(source=source, target=target):
                    without_cache = indexed_bidirectional_dijkstra_distance(
                        self.graph,
                        self.index,
                        source,
                        target,
                    )
                    with_cache = indexed_bidirectional_dijkstra_distance(
                        self.graph,
                        self.index,
                        source,
                        target,
                        endpoint_cache=cache,
                    )
                    self.assertEqual(with_cache.distance, without_cache.distance)

    def test_all_node_pairs_match_original_graph(self) -> None:
        for source in range(1, 7):
            for target in range(1, 7):
                with self.subTest(source=source, target=target):
                    expected = bidirectional_dijkstra_distance(self.graph, source, target)
                    actual = indexed_bidirectional_dijkstra_distance(
                        self.graph,
                        self.index,
                        source,
                        target,
                    )
                    self.assertEqual(actual.distance, expected.distance)

    def test_internal_endpoints_in_same_region_use_direct_distance(self) -> None:
        region = Region(
            region_id=0,
            nodes=frozenset({2, 3, 4, 5}),
            boundary_nodes=frozenset({2, 5}),
            seed_node=3,
            selection_method="test",
        )
        index = build_compression_index(self.graph, [region])

        expected = bidirectional_dijkstra_distance(self.graph, 3, 4)
        actual = indexed_bidirectional_dijkstra_distance(self.graph, index, 3, 4)

        self.assertEqual(actual.distance, expected.distance)
        self.assertEqual(actual.distance, 1.0)

    def test_directed_target_access_uses_reverse_region_edges(self) -> None:
        graph = WeightedDiGraph()
        for node in range(1, 6):
            graph.add_node(node)
        for source, target, weight in [
            (1, 2, 1.0),
            (2, 3, 2.0),
            (3, 4, 3.0),
            (4, 5, 4.0),
            (4, 3, 5.0),
            (3, 2, 7.0),
        ]:
            graph.add_edge(source, target, weight)
        region = Region(
            region_id=0,
            nodes=frozenset({2, 3, 4}),
            boundary_nodes=frozenset({2, 4}),
            seed_node=3,
            selection_method="test",
        )
        index = build_compression_index(graph, [region])

        for source, target in [(1, 3), (3, 5), (5, 3), (3, 1)]:
            with self.subTest(source=source, target=target):
                expected = bidirectional_dijkstra_distance(graph, source, target)
                actual = indexed_bidirectional_dijkstra_distance(
                    graph,
                    index,
                    source,
                    target,
                )
                self.assertEqual(actual.distance, expected.distance)

    def test_overlapping_regions_are_rejected(self) -> None:
        first = Region(
            region_id=0,
            nodes=frozenset({1, 2, 3}),
            boundary_nodes=frozenset({1, 3}),
            seed_node=2,
            selection_method="test",
        )
        second = Region(
            region_id=1,
            nodes=frozenset({3, 4, 5}),
            boundary_nodes=frozenset({3, 5}),
            seed_node=4,
            selection_method="test",
        )

        with self.assertRaises(ValueError):
            build_compression_index(self.graph, [first, second])

    def test_internal_endpoints_across_two_regions_match_original(self) -> None:
        graph = WeightedDiGraph()
        for node in range(1, 11):
            graph.add_node(node)
        for node in range(1, 10):
            graph.add_edge(node, node + 1, float(node))
            graph.add_edge(node + 1, node, float(node + 1))
        regions = [
            Region(
                region_id=0,
                nodes=frozenset({2, 3, 4}),
                boundary_nodes=frozenset({2, 4}),
                seed_node=3,
                selection_method="test",
            ),
            Region(
                region_id=1,
                nodes=frozenset({6, 7, 8}),
                boundary_nodes=frozenset({6, 8}),
                seed_node=7,
                selection_method="test",
            ),
        ]
        index = build_compression_index(graph, regions)

        for source, target in [(3, 7), (7, 3)]:
            with self.subTest(source=source, target=target):
                expected = bidirectional_dijkstra_distance(graph, source, target)
                actual = indexed_bidirectional_dijkstra_distance(
                    graph,
                    index,
                    source,
                    target,
                )
                self.assertEqual(actual.distance, expected.distance)


if __name__ == "__main__":
    unittest.main()
