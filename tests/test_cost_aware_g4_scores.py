from __future__ import annotations

import unittest

import numpy as np

from scripts.build_cost_aware_g4_scores import (
    _parse_named_paths,
    _rank_quality,
    _weight_token,
)


class CostAwareG4ScoresTest(unittest.TestCase):
    def test_rank_quality_is_stable_and_normalized(self) -> None:
        values = np.asarray([5.0, 1.0, 5.0, 3.0])
        np.testing.assert_allclose(_rank_quality(values), [2 / 3, 0.0, 1.0, 1 / 3])

    def test_named_paths_and_weight_tokens_are_deterministic(self) -> None:
        parsed = _parse_named_paths(["g4_seed42=a.csv", "g4_seed43=b.csv"])
        self.assertEqual(list(parsed), ["g4_seed42", "g4_seed43"])
        self.assertEqual(str(parsed["g4_seed42"]), "a.csv")
        self.assertEqual(_weight_token(0.1), "0p100")
        with self.assertRaises(ValueError):
            _parse_named_paths(["bad-name=a.csv"])


if __name__ == "__main__":
    unittest.main()
