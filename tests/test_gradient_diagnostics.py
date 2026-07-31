from __future__ import annotations

import math
import unittest

try:
    import torch

    from src.gradient_diagnostics import (
        DeviceTensorAccumulator,
        finite_tensor_statistics,
        parameter_gradient_report,
    )
except ImportError:  # pragma: no cover - covered by the CUDA environment.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class GradientDiagnosticsTest(unittest.TestCase):
    def test_finite_statistics_count_each_nonfinite_kind(self) -> None:
        values = torch.tensor([3.0, 4.0, float("nan"), float("inf"), float("-inf")])
        statistics = finite_tensor_statistics(values)
        self.assertEqual(statistics["finite_count"], 2)
        self.assertEqual(statistics["nonfinite_count"], 3)
        self.assertEqual(statistics["nan_count"], 1)
        self.assertEqual(statistics["positive_inf_count"], 1)
        self.assertEqual(statistics["negative_inf_count"], 1)
        self.assertEqual(statistics["maximum_absolute_finite_value"], 4.0)
        self.assertEqual(statistics["finite_l2_norm_fp64"], 5.0)

    def test_parameter_report_separates_fp32_norm_overflow(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        model.weight.grad = torch.full_like(model.weight, 1.0e20)
        report = parameter_gradient_report(model)["summary"]
        self.assertTrue(report["all_gradient_elements_finite"])
        self.assertFalse(report["raw_fp32_norm_is_finite"])
        self.assertTrue(math.isfinite(report["finite_l2_norm_fp64"]))
        self.assertAlmostEqual(
            report["finite_l2_norm_fp64"] / 1.0e20,
            2.0**0.5,
            places=5,
        )

    def test_device_accumulator_merges_calls_without_storing_tensors(self) -> None:
        accumulator = DeviceTensorAccumulator()
        accumulator.add("layer.gradient", torch.tensor([1.0, 2.0]))
        accumulator.add("layer.gradient", torch.tensor([float("inf"), -3.0]))
        result = accumulator.finalize()["layer.gradient"]
        self.assertEqual(result["call_count"], 2)
        self.assertEqual(result["element_count"], 4)
        self.assertEqual(result["nonfinite_count"], 1)
        self.assertEqual(result["positive_inf_count"], 1)
        self.assertEqual(result["maximum_absolute_finite_value"], 3.0)


if __name__ == "__main__":
    unittest.main()
