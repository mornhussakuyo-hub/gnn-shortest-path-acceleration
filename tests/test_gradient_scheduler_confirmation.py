from __future__ import annotations

import unittest
from pathlib import Path

from scripts.run_gradient_scheduler_confirmation import _parse_seeds, _seed_command


class GradientSchedulerConfirmationTest(unittest.TestCase):
    def test_command_freezes_scheduler_and_holdout_protocol(self) -> None:
        command = _seed_command(
            Path(".venv-gnn/bin/python"),
            Path("results/scheduler"),
            43,
        )
        self.assertEqual(command[command.index("--structure") + 1], "g3")
        self.assertEqual(command[command.index("--learning-rate") + 1], "0.005")
        self.assertEqual(
            command[command.index("--lr-scheduler") + 1],
            "reduce_on_plateau",
        )
        self.assertEqual(command[command.index("--lr-scheduler-factor") + 1], "0.3")
        self.assertEqual(command[command.index("--lr-scheduler-min-lr") + 1], "0.0005")

    def test_seed_parser_is_strict(self) -> None:
        self.assertEqual(_parse_seeds("42,44"), [42, 44])
        with self.assertRaises(SystemExit):
            _parse_seeds("42,42")
        with self.assertRaises(SystemExit):
            _parse_seeds("")


if __name__ == "__main__":
    unittest.main()
