from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.run_gradient_structure_screening import (
    TRAINING_SCHEMA,
    _completed_output_is_valid,
    _performance_gate,
    _training_command,
)


class GradientStructureScreeningTest(unittest.TestCase):
    def test_command_freezes_non_pretraining_g_line_protocol(self) -> None:
        args = Namespace(
            python=Path(".venv-gnn/bin/python"),
            seed=42,
            prototype_batch_size=4,
            learning_rate=3.0e-4,
            lr_scheduler="reduce_on_plateau",
            lr_scheduler_factor=0.3,
            lr_scheduler_patience=3,
            lr_scheduler_threshold=1.0e-4,
            lr_scheduler_min_lr=5.0e-5,
            max_epochs=40,
            patience=40,
            structure="g2",
            head_warmup_steps=8,
        )
        command = _training_command(args, Path("results/test-g2"))
        self.assertEqual(command[command.index("--fixed-prior") + 1], "z0")
        self.assertEqual(command[command.index("--propagation-structure") + 1], "g2")
        self.assertEqual(command[command.index("--head-warmup-steps") + 1], "8")
        self.assertEqual(
            command[command.index("--lr-scheduler") + 1],
            "reduce_on_plateau",
        )
        self.assertIn("--withhold-holdout", command)
        self.assertNotIn("pretrain", " ".join(command))

    def test_gate_accepts_either_preregistered_performance_channel(self) -> None:
        summary = self._summary(spearman=0.9450, ndcg5=0.9420)
        z0 = self._z0(spearman=0.9410, ndcg5=0.9440)
        gate = _performance_gate(summary, z0)
        self.assertTrue(gate["passed"])
        self.assertTrue(gate["channels"]["spearman_channel"])

        summary = self._summary(spearman=0.9400, ndcg5=0.9560)
        gate = _performance_gate(summary, z0)
        self.assertTrue(gate["passed"])
        self.assertTrue(gate["channels"]["ndcg5_channel"])

        summary["runs"][0]["diagnostics"]["skipped_optimizer_steps"] = 1
        gate = _performance_gate(summary, z0)
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["numerical_passed"])

    def test_completed_output_requires_holdout_to_be_absent(self) -> None:
        identity = {
            "dataset_sha256": "dataset",
            "structure": "g3",
            "head_warmup_steps": 8,
            "lr_scheduler": "reduce_on_plateau",
            "lr_scheduler_factor": 0.3,
            "lr_scheduler_patience": 3,
            "lr_scheduler_threshold": 1.0e-4,
            "lr_scheduler_min_lr": 5.0e-4,
        }
        payload = {
            "schema": TRAINING_SCHEMA,
            "dataset_sha256": "dataset",
            "training_objective": "rank_first",
            "fixed_prior": {"name": "z0"},
            "config": {
                "propagation_structure": "g3",
                "propagation_layers": 32,
            },
            "training_protocol": {
                "head_warmup_steps": 8,
                "holdout_withheld": True,
                "lr_scheduler": {
                    "name": "reduce_on_plateau",
                    "factor": 0.3,
                    "patience": 3,
                    "threshold": 1.0e-4,
                    "min_lr": 5.0e-4,
                },
            },
            "aggregate": {"train": {}, "validation": {}},
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
            self.assertTrue(_completed_output_is_valid(output, identity))
            payload["aggregate"]["holdout"] = {}
            (output / "summary.json").write_text(json.dumps(payload), encoding="utf-8")
            self.assertFalse(_completed_output_is_valid(output, identity))

    @staticmethod
    def _summary(*, spearman: float, ndcg5: float) -> dict:
        return {
            "runs": [
                {
                    "best_epoch": 20,
                    "epochs_ran": 40,
                    "metrics": {
                        "validation": {
                            "spearman": spearman,
                            "ranking_at_k": {"5": {"ndcg": ndcg5}},
                        }
                    },
                    "diagnostics": {
                        "effective_optimizer_steps": 40,
                        "skipped_optimizer_steps": 0,
                        "nonfinite_gradient_norm_steps": 0,
                        "maximum_consecutive_zero_gradient_steps_after_clip": 0,
                        "unrecovered_validation_loss_doublings": 0,
                        "best_checkpoint_backbone_effective_step_count": 12,
                    },
                }
            ]
        }

    @staticmethod
    def _z0(*, spearman: float, ndcg5: float) -> dict:
        return {
            "z0_metrics": {
                "validation": {
                    "spearman": spearman,
                    "ranking_at_k": {"5": {"ndcg": ndcg5}},
                }
            }
        }


if __name__ == "__main__":
    unittest.main()
