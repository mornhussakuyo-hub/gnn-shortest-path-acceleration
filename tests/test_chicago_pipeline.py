from __future__ import annotations

import unittest

from scripts.finalize_chicago_queries import (
    _equal_index_sample,
    _row_order_key,
    _strong_components,
)
from src.graph_types import WeightedDiGraph


class ChicagoPipelineTest(unittest.TestCase):
    def test_equal_index_sample_is_deterministic_and_covers_endpoints(self) -> None:
        observed = _equal_index_sample(11, 5)

        self.assertEqual(observed, [0, 2, 5, 7, 10])
        self.assertEqual(len(observed), len(set(observed)))

    def test_strong_components_respect_edge_direction(self) -> None:
        graph = WeightedDiGraph()
        graph.add_edge(1, 2, 1.0)
        graph.add_edge(2, 1, 1.0)
        graph.add_edge(2, 3, 1.0)
        graph.add_edge(3, 4, 1.0)
        graph.add_edge(4, 3, 1.0)
        graph.add_node(5)

        component, sizes = _strong_components(graph)

        self.assertEqual(component[1], component[2])
        self.assertEqual(component[3], component[4])
        self.assertNotEqual(component[2], component[3])
        self.assertNotEqual(component[4], component[5])
        self.assertEqual(sorted(sizes.values()), [1, 2, 2])

    def test_query_order_accepts_opaque_ride_ids(self) -> None:
        rows = [
            {"timestamp": "2", "trip_id": "ride-z"},
            {"timestamp": "1", "trip_id": "opaque-without-colon"},
            {"timestamp": "2", "trip_id": "ride-a"},
        ]

        self.assertEqual(
            [row["trip_id"] for row in sorted(rows, key=_row_order_key)],
            ["opaque-without-colon", "ride-a", "ride-z"],
        )


if __name__ == "__main__":
    unittest.main()
