from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.run_gradient_stabilization import (
    DIAGNOSTIC_SCHEMA,
    _command_for,
    _completed_output_is_valid,
    _hook_amplification,
    _load_or_create_manifest,
    _optimizer_steps_gate,
    _parse_depths,
    _parse_structures,
    _snapshot_gate,
)


class GradientStabilizationRunnerTest(unittest.TestCase):
    def test_structure_and_depth_parsers_are_strict(self) -> None:
        self.assertEqual(_parse_structures("g0,g2"), ["g0", "g2"])
        self.assertEqual(_parse_depths("32,8,16"), [8, 16, 32])
        with self.assertRaises(ValueError):
            _parse_structures("g0,g0")
        with self.assertRaises(ValueError):
            _parse_structures("g4")
        with self.assertRaises(ValueError):
            _parse_depths("8,0,32")

    def test_command_keeps_structure_orthogonal_to_variant(self) -> None:
        args = Namespace(
            python=Path(".venv-gnn/bin/python"),
            variant="propagation_doubling",
            residual_scale=0.01,
            seed=42,
            hidden_dim=32,
            prototype_batch_size=4,
            learning_rate=0.001,
            max_grad_norm=1.0,
            optimizer_steps=3,
        )
        command, output, _log = _command_for(
            args, "g3", 32, "optimizer_steps", Path("results/test-s1")
        )
        self.assertEqual(command[command.index("--variant") + 1], "propagation_doubling")
        self.assertEqual(
            command[command.index("--propagation-structure") + 1], "g3"
        )
        self.assertIn("g3/depth_32/optimizer_steps", output.as_posix())

    def test_snapshot_gate_checks_all_preregistered_limits(self) -> None:
        hooks = {
            "origin.depth_01.gradient": self._hook(100.0),
            "origin.depth_32.gradient": self._hook(1.0),
            "destination.depth_01.gradient": self._hook(50.0),
            "destination.depth_32.gradient": self._hook(1.0),
        }
        summary = {
            "result": {
                "snapshot": {
                    "parameter_gradients_unscaled": {
                        "summary": {
                            "all_gradient_elements_finite": True,
                            "raw_fp32_norm_is_finite": True,
                            "raw_l2_norm_fp32": 10.0,
                            "finite_l2_norm_fp64": 10.0,
                        }
                    },
                    "tensor_hooks": hooks,
                    "clipping": {"coefficient": 0.1},
                }
            }
        }
        gate = _snapshot_gate(
            summary,
            depth=32,
            fp64_norm_limit=1.0e4,
            clip_coefficient_min=1.0e-4,
            hook_amplification_limit=1.0e6,
        )
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["metrics"]["hook_amplification"], 100.0)
        summary["result"]["snapshot"]["clipping"]["coefficient"] = 1.0e-5
        self.assertFalse(
            _snapshot_gate(
                summary,
                depth=32,
                fp64_norm_limit=1.0e4,
                clip_coefficient_min=1.0e-4,
                hook_amplification_limit=1.0e6,
            )["passed"]
        )

    def test_optimizer_gate_requires_three_real_noncollapsed_updates(self) -> None:
        rows = [{"step": 0, "validation_spearman": 0.0}]
        for step, norm in enumerate((10.0, 20.0, 30.0), start=1):
            rows.append(
                {
                    "step": step,
                    "optimizer_step": "applied",
                    "parameter_delta_norm": 0.01,
                    "parameter_gradients_unscaled": {
                        "summary": {
                            "finite_l2_norm_fp64": norm,
                            "all_gradient_elements_finite": True,
                            "raw_fp32_norm_is_finite": True,
                        }
                    },
                    "parameter_gradients_clipped": {
                        "summary": {"finite_l2_norm_fp64": 1.0}
                    },
                    "score_prediction": {"all_finite": True, "std": 0.2},
                    "clipping": {"coefficient": 0.1},
                }
            )
        summary = {"result": {"applied_steps": 3, "history": rows}}
        self.assertTrue(_optimizer_steps_gate(summary, requested_steps=3)["passed"])
        rows[-1]["score_prediction"]["std"] = 0.0
        self.assertFalse(_optimizer_steps_gate(summary, requested_steps=3)["passed"])

    def test_manifest_identity_and_completed_output_are_checked(self) -> None:
        identity = {"schema": "test", "experiment_id": "abc"}
        args = Namespace(
            python=Path(".venv-gnn/bin/python"),
            variant="propagation_doubling",
            residual_scale=0.01,
            seed=42,
            hidden_dim=32,
            prototype_batch_size=4,
            learning_rate=0.001,
            max_grad_norm=1.0,
            optimizer_steps=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "manifest.json"
            manifest = _load_or_create_manifest(path, identity, force=False)
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(
                _load_or_create_manifest(path, identity, force=False)["identity"],
                identity,
            )
            with self.assertRaises(SystemExit):
                _load_or_create_manifest(
                    path, {"schema": "other"}, force=False
                )

            command, output, _log = _command_for(
                args, "g2", 32, "snapshot", root
            )
            output.mkdir(parents=True)
            (output / "summary.json").write_text(
                json.dumps(
                    {
                        "schema": DIAGNOSTIC_SCHEMA,
                        "mode": "snapshot",
                        "config": {
                            "propagation_structure": "g2",
                            "propagation_layers": 32,
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.assertTrue(_completed_output_is_valid(output, command))

    @staticmethod
    def _hook(maximum: float) -> dict[str, object]:
        return {
            "all_finite": True,
            "maximum_absolute_finite_value": maximum,
        }


if __name__ == "__main__":
    unittest.main()
