from __future__ import annotations

import unittest

try:
    import torch

    from src.demand_field_nbfnet import (
        BidirectionalNBFNet,
        NBFNetConfig,
        build_edge_features,
        build_receiver_normalizers,
    )
except ImportError:  # pragma: no cover - covered by the CUDA environment instead.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class BidirectionalNBFNetTest(unittest.TestCase):
    def test_forward_shape_and_gradients(self) -> None:
        config = NBFNetConfig(hidden_dim=4, propagation_layers=2, max_epochs=1)
        model = BidirectionalNBFNet(
            node_feature_dim=3,
            region_feature_dim=2,
            edge_type_count=2,
            config=config,
        )
        edge_source = torch.tensor([0, 1, 2, 3, 1], dtype=torch.long)
        edge_target = torch.tensor([1, 2, 3, 4, 0], dtype=torch.long)
        edge_features = build_edge_features(
            torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6]),
            torch.tensor([0, 1, 0, 1, 0]),
            edge_type_count=2,
        )
        output = model(
            node_features=torch.randn(5, 3),
            edge_source=edge_source,
            edge_target=edge_target,
            edge_features=edge_features,
            origin_fields=torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.0, 0.0]]
            ),
            destination_fields=torch.tensor(
                [[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 0.0]]
            ),
            region_nodes=torch.tensor([[0, 1, 2], [2, 3, 4]], dtype=torch.long),
            region_features=torch.randn(2, 2),
            receiver_normalizer_forward=build_receiver_normalizers(edge_target, 5),
            receiver_normalizer_reverse=build_receiver_normalizers(edge_source, 5),
        )
        self.assertEqual(tuple(output.shape), (2, 2))
        output.square().mean().backward()
        self.assertIsNotNone(model.origin_encoder[0].weight.grad)
        self.assertTrue(torch.isfinite(output).all())

    def test_config_rejects_invalid_batch_size(self) -> None:
        with self.assertRaises(ValueError):
            NBFNetConfig(prototype_batch_size=0).validate()
