from __future__ import annotations

import json
import os
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from scripts.run_nbfnet_ablation import (
    MANIFEST_SCHEMA,
    _acquire_lock,
    _completed_output_is_valid,
    _prototype_batch_size,
    _load_or_create_manifest,
    _parse_variants,
    _training_command,
)


class NBFNetAblationRunnerTest(unittest.TestCase):
    def test_variant_parser_rejects_unknown_and_duplicate_names(self) -> None:
        self.assertEqual(
            _parse_variants("origin_only,undirected"),
            ["origin_only", "undirected"],
        )
        self.assertEqual(
            _parse_variants("propagation_deep,propagation_residual_doubling"),
            ["propagation_deep", "propagation_residual_doubling"],
        )
        with self.assertRaises(ValueError):
            _parse_variants("origin_only,origin_only")
        with self.assertRaises(ValueError):
            _parse_variants("unknown")

    def test_completed_output_requires_matching_identity_and_config(self) -> None:
        identity = {
            "dataset_sha256": "dataset",
            "candidate_sha256": "candidate",
            "training": {
                "hidden_dim": 32,
                "layers": 6,
                "prototype_batch_size": 8,
                "expanded_graph_batch_size": 4,
                "max_epochs": 100,
                "patience": 20,
                "randomization_seed": 20260730,
            },
        }
        summary = {
            "variant": "undirected",
            "seeds": [44],
            "dataset_sha256": "dataset",
            "candidate_sha256": "candidate",
            "config": {
                "hidden_dim": 32,
                "propagation_layers": 6,
                "prototype_batch_size": 4,
                "max_epochs": 100,
                "patience": 20,
                "randomization_seed": 20260730,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "summary.json").write_text(
                json.dumps(summary),
                encoding="utf-8",
            )
            self.assertTrue(
                _completed_output_is_valid(output, "undirected", 44, identity)
            )
            self.assertFalse(
                _completed_output_is_valid(output, "origin_only", 44, identity)
            )

    def test_expanded_graph_variants_use_safe_batch_size(self) -> None:
        args = Namespace(
            prototype_batch_size=8,
            expanded_graph_batch_size=4,
        )
        self.assertEqual(_prototype_batch_size(args, "undirected"), 4)
        self.assertEqual(_prototype_batch_size(args, "graphsage"), 4)
        self.assertEqual(_prototype_batch_size(args, "origin_only"), 8)

    def test_training_command_preserves_virtualenv_python_path(self) -> None:
        args = Namespace(
            python=Path(".venv-gnn") / "bin" / "python",
            hidden_dim=32,
            layers=6,
            prototype_batch_size=8,
            expanded_graph_batch_size=4,
            max_epochs=100,
            patience=20,
            randomization_seed=20260730,
        )
        command = _training_command(
            args,
            "origin_only",
            44,
            Path("results") / "test-output",
        )
        self.assertEqual(
            command[0],
            os.path.abspath(args.python),
        )

    def test_manifest_identity_mismatch_requires_force(self) -> None:
        first = {
            "schema": MANIFEST_SCHEMA,
            "experiment_id": "first",
        }
        second = {
            "schema": MANIFEST_SCHEMA,
            "experiment_id": "second",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.json"
            manifest = _load_or_create_manifest(path, first, force=False)
            path.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(SystemExit):
                _load_or_create_manifest(path, second, force=False)
            replaced = _load_or_create_manifest(path, second, force=True)
            self.assertEqual(replaced["identity"], second)

    def test_lock_rejects_live_runner_and_replaces_stale_pid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".runner.lock"
            path.write_text(json.dumps({"pid": os.getpid()}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                _acquire_lock(path)

            path.write_text(json.dumps({"pid": 2_000_000_000}), encoding="utf-8")
            _acquire_lock(path)
            payload = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(payload["pid"], os.getpid())


if __name__ == "__main__":
    unittest.main()
