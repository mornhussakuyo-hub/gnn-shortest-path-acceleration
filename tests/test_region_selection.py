from __future__ import annotations

import unittest

import numpy as np

from src.region_selection import select_region_indices, selection_overlap_statistics


class RegionSelectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.scores = np.asarray([4.0, 3.0, 2.0, 1.0])
        self.region_nodes = np.asarray(
            [
                [0, 1, 2, 3],
                [0, 1, 2, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12],
            ]
        )

    def test_direct_topk_preserves_score_order_and_overlap(self) -> None:
        selected = select_region_indices(
            self.scores, self.region_nodes, 2, "direct_topk"
        )
        np.testing.assert_array_equal(selected, [0, 1])
        statistics = selection_overlap_statistics(self.region_nodes[selected])
        self.assertFalse(statistics["deployable_without_region_overlap"])
        self.assertEqual(statistics["overlapping_pair_count"], 1)

    def test_diversity_relaxations_prefer_novel_region(self) -> None:
        for method in ("jaccard_penalty", "marginal_coverage"):
            with self.subTest(method=method):
                selected = select_region_indices(
                    self.scores, self.region_nodes, 2, method
                )
                np.testing.assert_array_equal(selected, [0, 2])

    def test_hard_disjoint_is_exact_and_deterministic(self) -> None:
        first = select_region_indices(
            self.scores, self.region_nodes, 3, "hard_disjoint"
        )
        second = select_region_indices(
            self.scores, self.region_nodes, 3, "hard_disjoint"
        )
        np.testing.assert_array_equal(first, [0, 2, 3])
        np.testing.assert_array_equal(first, second)
        statistics = selection_overlap_statistics(self.region_nodes[first])
        self.assertTrue(statistics["deployable_without_region_overlap"])
        self.assertEqual(statistics["membership_redundancy"], 1.0)

    def test_hard_disjoint_can_stop_before_requested_k(self) -> None:
        nodes = np.asarray([[0, 1], [0, 2], [0, 3]])
        selected = select_region_indices(
            np.asarray([3.0, 2.0, 1.0]), nodes, 3, "hard_disjoint"
        )
        np.testing.assert_array_equal(selected, [0])


if __name__ == "__main__":
    unittest.main()
