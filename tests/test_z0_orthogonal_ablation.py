from __future__ import annotations

import unittest

try:
    from scripts.evaluate_z0_orthogonal_ablations import _build_ablation_specs
    from src.train_free_demand_field import DEFAULT_DIFFUSION_DEPTHS
except ImportError:  # pragma: no cover - covered in the CUDA environment.
    _build_ablation_specs = None
    DEFAULT_DIFFUSION_DEPTHS = ()


@unittest.skipIf(_build_ablation_specs is None, "PyTorch is not installed")
class Z0OrthogonalAblationTest(unittest.TestCase):
    def test_frozen_catalog_is_unique_and_contains_every_axis(self) -> None:
        specs = _build_ablation_specs()
        by_name = {spec.name: spec for spec in specs}
        self.assertEqual(len(specs), len(by_name))
        self.assertEqual(
            set(by_name),
            {
                "z0_base",
                "origin_only",
                "destination_only",
                "undirected",
                "degree_rewired",
                "shuffled_od",
                "pooling_mean",
                "pooling_max",
                *(f"depth_{depth:02d}" for depth in DEFAULT_DIFFUSION_DEPTHS),
            },
        )
        self.assertEqual(by_name["z0_base"].depths, DEFAULT_DIFFUSION_DEPTHS)
        self.assertEqual(by_name["origin_only"].destination_weight, 0.0)
        self.assertEqual(by_name["destination_only"].origin_weight, 0.0)
        self.assertEqual(by_name["undirected"].tensor_variant, "undirected")
        self.assertEqual(
            by_name["degree_rewired"].tensor_variant, "degree_rewired"
        )
        self.assertEqual(by_name["shuffled_od"].tensor_variant, "shuffled_od")
        self.assertEqual(by_name["pooling_mean"].region_pooling, "mean")
        self.assertEqual(by_name["pooling_max"].region_pooling, "max")


if __name__ == "__main__":
    unittest.main()
