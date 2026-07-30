from __future__ import annotations

import unittest

import numpy as np

try:
    import torch

    from scripts.train_demand_field_nbfnet import (
        _marginal_preserving_od_shuffle,
        _transform_edge_arrays,
    )
    from src.demand_field_nbfnet import (
        NBFNET_VARIANTS,
        BidirectionalNBFNet,
        NBFNetConfig,
        build_edge_features,
        build_receiver_normalizers,
    )
except ImportError:  # pragma: no cover - covered in the CUDA environment.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class NBFNetAblationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.edge_source = torch.tensor([0, 1, 2, 3, 1], dtype=torch.long)
        self.edge_target = torch.tensor([1, 2, 3, 4, 0], dtype=torch.long)
        self.edge_features = build_edge_features(
            torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6]),
            torch.tensor([0, 1, 0, 1, 0]),
            edge_type_count=2,
        )
        self.common = {
            "node_features": torch.randn(5, 3),
            "edge_source": self.edge_source,
            "edge_target": self.edge_target,
            "edge_features": self.edge_features,
            "region_nodes": torch.tensor([[0, 1, 2], [2, 3, 4]]),
            "region_features": torch.randn(2, 2),
            "receiver_normalizer_forward": build_receiver_normalizers(
                self.edge_target, 5
            ),
            "receiver_normalizer_reverse": build_receiver_normalizers(
                self.edge_source, 5
            ),
        }
        self.origin = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]])
        self.destination = torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]])

    def _model(self, variant: str) -> BidirectionalNBFNet:
        return BidirectionalNBFNet(
            3,
            2,
            2,
            NBFNetConfig(
                hidden_dim=4,
                propagation_layers=2,
                max_epochs=1,
                variant=variant,
            ),
        )

    def test_every_variant_runs_forward_and_backward(self) -> None:
        for variant in NBFNET_VARIANTS:
            with self.subTest(variant=variant):
                model = self._model(variant)
                output = model(
                    origin_fields=self.origin,
                    destination_fields=self.destination,
                    **self.common,
                )
                self.assertEqual(tuple(output.shape), (1, 2))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertTrue(
                    any(
                        parameter.grad is not None
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    )
                )

    def test_single_field_variants_ignore_removed_field(self) -> None:
        origin_only = self._model("origin_only").eval()
        first = origin_only(
            origin_fields=self.origin,
            destination_fields=self.destination,
            **self.common,
        )
        second = origin_only(
            origin_fields=self.origin,
            destination_fields=torch.roll(self.destination, shifts=2, dims=1),
            **self.common,
        )
        self.assertTrue(torch.equal(first, second))

        destination_only = self._model("destination_only").eval()
        first = destination_only(
            origin_fields=self.origin,
            destination_fields=self.destination,
            **self.common,
        )
        second = destination_only(
            origin_fields=torch.roll(self.origin, shifts=2, dims=1),
            destination_fields=self.destination,
            **self.common,
        )
        self.assertTrue(torch.equal(first, second))

    def test_shared_parameter_variant_reuses_modules(self) -> None:
        model = self._model("shared_parameters")
        self.assertIs(model.origin_encoder, model.destination_encoder)
        self.assertIs(model.origin_token, model.destination_token)
        self.assertIs(model.origin_layers, model.destination_layers)

    def test_graph_transforms_are_deterministic_and_degree_preserving(self) -> None:
        source = np.asarray([0, 0, 1, 2, 3, 4], dtype=np.int64)
        target = np.asarray([1, 2, 2, 3, 4, 0], dtype=np.int64)
        length = np.arange(6, dtype=np.float32)
        edge_type = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)

        undirected = _transform_edge_arrays(
            source, target, length, edge_type, "undirected", 7
        )
        self.assertEqual(undirected[0].size, source.size * 2)
        np.testing.assert_array_equal(undirected[0][source.size :], target)
        np.testing.assert_array_equal(undirected[1][source.size :], source)

        rewired_a = _transform_edge_arrays(
            source, target, length, edge_type, "degree_rewired", 7
        )
        rewired_b = _transform_edge_arrays(
            source, target, length, edge_type, "degree_rewired", 7
        )
        np.testing.assert_array_equal(rewired_a[0], source)
        np.testing.assert_array_equal(np.sort(rewired_a[1]), np.sort(target))
        np.testing.assert_array_equal(rewired_a[1], rewired_b[1])
        np.testing.assert_array_equal(rewired_a[2], length)
        np.testing.assert_array_equal(rewired_a[3], edge_type)

    def test_unknown_variant_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            NBFNetConfig(variant="not-a-variant").validate()

    def test_shuffled_od_preserves_both_weighted_marginals(self) -> None:
        origin = torch.eye(4)
        destination = torch.roll(torch.eye(4), shifts=1, dims=1)
        weights = torch.tensor([0.1, 0.2, 0.3, 0.4])
        shuffled_origin, shuffled_destination, shuffled_weight, metadata = (
            _marginal_preserving_od_shuffle(origin, destination, weights)
        )
        expected_origin = (origin * weights[:, None]).sum(0)
        expected_destination = (destination * weights[:, None]).sum(0)
        actual_origin = (shuffled_origin * shuffled_weight[:, None]).sum(0)
        actual_destination = (
            shuffled_destination * shuffled_weight[:, None]
        ).sum(0)
        self.assertTrue(torch.allclose(actual_origin, expected_origin, atol=1e-6))
        self.assertTrue(
            torch.allclose(actual_destination, expected_destination, atol=1e-6)
        )
        self.assertAlmostEqual(float(shuffled_weight.sum()), 1.0)
        self.assertTrue(metadata["preserves_origin_marginal"])
        self.assertTrue(metadata["preserves_destination_marginal"])

    def test_propagation_variants_exclude_layer_zero_and_region_bypass(self) -> None:
        expected_depths = {
            "propagation_deep": (8,),
            "propagation_residual": tuple(range(1, 9)),
            "propagation_doubling": (1, 2, 4, 8),
            "propagation_residual_doubling": (1, 2, 4, 8),
        }
        for variant, depths in expected_depths.items():
            with self.subTest(variant=variant):
                model = BidirectionalNBFNet(
                    3,
                    2,
                    2,
                    NBFNetConfig(
                        hidden_dim=4,
                        propagation_layers=8,
                        max_epochs=1,
                        variant=variant,
                    ),
                ).eval()
                self.assertEqual(model.readout_depths, depths)
                self.assertNotIn(0, model.readout_depths)
                first = model(
                    origin_fields=self.origin,
                    destination_fields=self.destination,
                    **self.common,
                )
                changed = dict(self.common)
                changed["region_features"] = self.common["region_features"] + 1000.0
                second = model(
                    origin_fields=self.origin,
                    destination_fields=self.destination,
                    **changed,
                )
                self.assertTrue(torch.equal(first, second))


if __name__ == "__main__":
    unittest.main()
