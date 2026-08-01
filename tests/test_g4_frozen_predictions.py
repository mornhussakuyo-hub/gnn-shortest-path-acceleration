from __future__ import annotations

import unittest

from scripts.export_g4_frozen_predictions import (
    POLICIES,
    _parse_ints,
    _validate_selection,
    _validate_validation_replay,
)


class _Dataset:
    manifest = {"dataset_sha256": "dataset", "candidate_sha256": "candidate"}


class G4FrozenPredictionsTest(unittest.TestCase):
    def test_seed_parser_is_strict(self) -> None:
        self.assertEqual(_parse_ints("42,43,44", "--seeds"), (42, 43, 44))
        with self.assertRaises(SystemExit):
            _parse_ints("42,42", "--seeds")

    def test_selection_rejects_holdout_use(self) -> None:
        selection = {
            "schema": "aic.gnn_v2.g4_validation_peak_selection.v1",
            "dataset_sha256": "dataset",
            "candidate_sha256": "candidate",
            "selection_split": "validation",
            "holdout_used": False,
            "selections": {policy: {} for policy in POLICIES},
        }
        _validate_selection(selection, _Dataset())
        selection["holdout_used"] = True
        with self.assertRaises(ValueError):
            _validate_selection(selection, _Dataset())

    def test_validation_replay_checks_all_ranking_metrics(self) -> None:
        ranking = {
            str(k): {"ndcg": 0.9 + k / 1000, "mean_gain": 20.0 + k}
            for k in (5, 10, 18)
        }
        selected = {
            "spearman": 0.95,
            "ndcg_at_k": {str(k): ranking[str(k)]["ndcg"] for k in (5, 10, 18)},
            "top_gain_at_k": {
                str(k): ranking[str(k)]["mean_gain"] for k in (5, 10, 18)
            },
        }
        result = _validate_validation_replay(
            {"spearman": 0.95, "ranking_at_k": ranking}, selected
        )
        self.assertTrue(result["passed"])
        selected["ndcg_at_k"]["10"] += 0.01
        with self.assertRaises(ValueError):
            _validate_validation_replay(
                {"spearman": 0.95, "ranking_at_k": ranking}, selected
            )


if __name__ == "__main__":
    unittest.main()
