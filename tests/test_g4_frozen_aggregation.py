from __future__ import annotations

import copy
import unittest

from scripts.aggregate_g4_frozen_evaluations import POLICIES, aggregate


def _metrics(value: float) -> dict:
    return {
        "spearman": value,
        "ranking_at_k": {
            str(k): {"ndcg": value, "mean_gain": value * 100} for k in (5, 10, 18)
        },
    }


class G4FrozenAggregationTest(unittest.TestCase):
    def test_requires_complete_policy_seed_matrix(self) -> None:
        runs = {}
        for policy in POLICIES:
            for seed in (42, 43, 44):
                runs[f"{policy}.seed_{seed}"] = {
                    "seed": seed,
                    "validation_replay": {"passed": True},
                    "residual_gate": {
                        "alpha": 0.25,
                        "validation_metrics": {"epoch": 20},
                    },
                    "current_metrics": {
                        "validation": _metrics(0.95),
                        "holdout": _metrics(0.94),
                    },
                    "future_metrics": {"all_candidates": _metrics(0.90)},
                }
        shard = {
            "schema": "aic.gnn_v2.g4_frozen_evaluation_shard.v1",
            "dataset_sha256": "dataset",
            "candidate_sha256": "candidate",
            "protocol": {
                "training_performed": False,
                "selection_split": "validation",
                "holdout_or_future_used_for_selection": False,
            },
            "runs": runs,
        }
        z0 = {
            "dataset_sha256": "dataset",
            "candidate_sha256": "candidate",
            "z0_current_window_metrics": {
                "validation": _metrics(0.90),
                "holdout": _metrics(0.89),
            },
            "z0_future_window_metrics": {"all_candidates": _metrics(0.88)},
        }
        summary = aggregate([shard], z0, (42, 43, 44), "test")
        self.assertAlmostEqual(
            summary["aggregate"]["global_spearman"]["holdout"]
            ["delta_spearman_vs_z0"]["mean"],
            0.05,
        )
        broken = copy.deepcopy(shard)
        del broken["runs"]["global_spearman.seed_44"]
        with self.assertRaises(ValueError):
            aggregate([broken], z0, (42, 43, 44), "test")


if __name__ == "__main__":
    unittest.main()
