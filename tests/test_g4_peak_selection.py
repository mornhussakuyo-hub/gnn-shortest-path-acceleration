from __future__ import annotations

import copy
import unittest

from scripts.select_g4_validation_peaks import _select_validation_peaks


def _gate(alpha: float, spearman: float, gain18: float) -> dict[str, object]:
    return {
        "alpha": alpha,
        "spearman": spearman,
        "ndcg_at_k": {"5": 0.8, "10": 0.8, "18": 0.8},
        "top_gain_at_k": {"5": gain18, "10": gain18, "18": gain18},
    }


class G4PeakSelectionTest(unittest.TestCase):
    def test_global_and_budget_peaks_are_both_preserved(self) -> None:
        catalog = {
            "schema": "aic.gnn_v2.validation_snapshot_catalog.v1",
            "holdout_evaluated": False,
            "entries": [
                {
                    "epoch": 0,
                    "snapshot": "validation_snapshots/epoch_000.pt",
                    "optimizer_step_effective": False,
                    "gates": [_gate(0.0, 0.90, 50.0), _gate(1.0, 0.90, 50.0)],
                },
                {
                    "epoch": 17,
                    "snapshot": "validation_snapshots/epoch_017.pt",
                    "optimizer_step_effective": True,
                    "gates": [_gate(1.0, 0.932, 52.0)],
                },
                {
                    "epoch": 35,
                    "snapshot": "validation_snapshots/epoch_035.pt",
                    "optimizer_step_effective": True,
                    "gates": [_gate(0.5, 0.933, 49.0)],
                },
            ],
        }
        selected = _select_validation_peaks(catalog)
        self.assertEqual(selected["global_spearman"]["epoch"], 35)
        self.assertEqual(selected["budget_safe_spearman"]["epoch"], 17)
        self.assertEqual(selected["topgain18_safe"]["epoch"], 17)

    def test_budget_selection_always_has_alpha_zero_fallback(self) -> None:
        catalog = {
            "schema": "aic.gnn_v2.validation_snapshot_catalog.v1",
            "holdout_evaluated": False,
            "entries": [
                {
                    "epoch": 0,
                    "snapshot": "validation_snapshots/epoch_000.pt",
                    "optimizer_step_effective": False,
                    "gates": [_gate(0.0, 0.90, 50.0)],
                },
                {
                    "epoch": 1,
                    "snapshot": "validation_snapshots/epoch_001.pt",
                    "optimizer_step_effective": True,
                    "gates": [_gate(1.0, 0.95, 40.0)],
                },
            ],
        }
        selected = _select_validation_peaks(catalog)
        self.assertEqual(selected["global_spearman"]["epoch"], 1)
        self.assertEqual(selected["budget_safe_spearman"]["alpha"], 0.0)

    def test_holdout_contaminated_catalog_is_rejected(self) -> None:
        catalog = {
            "schema": "aic.gnn_v2.validation_snapshot_catalog.v1",
            "holdout_evaluated": False,
            "entries": [
                {
                    "epoch": 0,
                    "snapshot": "validation_snapshots/epoch_000.pt",
                    "optimizer_step_effective": False,
                    "gates": [_gate(0.0, 0.90, 50.0)],
                }
            ],
        }
        contaminated = copy.deepcopy(catalog)
        contaminated["holdout_evaluated"] = True
        with self.assertRaises(ValueError):
            _select_validation_peaks(contaminated)


if __name__ == "__main__":
    unittest.main()
