import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.export_cpp_online_benchmark import load_frozen_selections


ROOT_DIR = Path(__file__).resolve().parents[1]


class CppOnlineBenchmarkTest(unittest.TestCase):
    def test_cpp_self_test_is_exact(self) -> None:
        if shutil.which("g++") is None:
            self.skipTest("g++ is not installed")
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "benchmark"
            subprocess.run(
                [
                    "g++",
                    "-std=c++20",
                    "-O2",
                    str(ROOT_DIR / "cpp/aic_cpp_online_benchmark.cpp"),
                    "-o",
                    str(binary),
                ],
                check=True,
                cwd=ROOT_DIR,
                capture_output=True,
                text=True,
            )
            completed = subprocess.run(
                [str(binary), "--self-test"],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("all 36 OD pairs exact", completed.stdout)

    def test_selection_loader_requires_complete_rank_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "selections.csv"
            path.write_text(
                "method,k,rank,region_id\n"
                "z0,2,2,12\n"
                "z0,2,1,11\n",
                encoding="utf-8",
            )
            self.assertEqual(
                load_frozen_selections((path,), methods=("z0",), k=2),
                {"z0": [11, 12]},
            )
            with self.assertRaises(ValueError):
                load_frozen_selections((path,), methods=("z0",), k=3)


if __name__ == "__main__":
    unittest.main()
