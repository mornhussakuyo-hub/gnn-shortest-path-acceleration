from __future__ import annotations

import unittest

try:
    import torch

    from src.demand_field_nbfnet import build_receiver_normalizers
    from src.train_free_demand_field import (
        deterministic_diffusion_batch_scores,
        orient_from_training_split,
    )
except ImportError:  # pragma: no cover - covered by the CUDA environment.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class TrainFreeDemandFieldTest(unittest.TestCase):
    def test_deterministic_diffusion_is_repeatable_and_label_free(self) -> None:
        edge_source = torch.tensor([0, 1, 2, 3])
        edge_target = torch.tensor([1, 2, 3, 4])
        arguments = {
            "origin_fields": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]),
            "destination_fields": torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]]),
            "edge_source": edge_source,
            "edge_target": edge_target,
            "region_nodes": torch.tensor([[0, 1], [2, 3]]),
            "receiver_normalizer_forward": build_receiver_normalizers(edge_target, 5),
            "receiver_normalizer_reverse": build_receiver_normalizers(edge_source, 5),
            "depths": (1, 2),
        }
        first = deterministic_diffusion_batch_scores(**arguments)
        second = deterministic_diffusion_batch_scores(**arguments)
        self.assertEqual(tuple(first.shape), (1, 2))
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.isfinite(first).all())

    def test_train_orientation_only_changes_global_sign(self) -> None:
        prediction = torch.tensor([3.0, 2.0, 1.0])
        target = torch.tensor([1.0, 2.0, 3.0])
        oriented, sign = orient_from_training_split(prediction, target)
        self.assertEqual(sign, -1)
        self.assertTrue(torch.equal(oriented, -prediction))

    def test_direction_and_pooling_ablations_are_orthogonal(self) -> None:
        edge_source = torch.tensor([0, 1, 2, 3])
        edge_target = torch.tensor([1, 2, 3, 4])
        arguments = {
            "origin_fields": torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]),
            "destination_fields": torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]]),
            "edge_source": edge_source,
            "edge_target": edge_target,
            "region_nodes": torch.tensor([[0, 1], [2, 3]]),
            "receiver_normalizer_forward": build_receiver_normalizers(edge_target, 5),
            "receiver_normalizer_reverse": build_receiver_normalizers(edge_source, 5),
            "depths": (1, 2),
        }
        combined = deterministic_diffusion_batch_scores(**arguments)
        origin_only = deterministic_diffusion_batch_scores(
            **arguments, destination_weight=0.0
        )
        destination_only = deterministic_diffusion_batch_scores(
            **arguments, origin_weight=0.0
        )
        mean_only = deterministic_diffusion_batch_scores(
            **arguments, region_pooling="mean"
        )
        max_only = deterministic_diffusion_batch_scores(
            **arguments, region_pooling="max"
        )
        self.assertTrue(torch.allclose(combined, mean_only + max_only))
        self.assertFalse(torch.equal(combined, origin_only))
        self.assertFalse(torch.equal(combined, destination_only))

    def test_invalid_deterministic_ablation_is_rejected(self) -> None:
        edge_source = torch.tensor([0])
        edge_target = torch.tensor([1])
        arguments = {
            "origin_fields": torch.tensor([[1.0, 0.0]]),
            "destination_fields": torch.tensor([[0.0, 1.0]]),
            "edge_source": edge_source,
            "edge_target": edge_target,
            "region_nodes": torch.tensor([[0, 1]]),
            "receiver_normalizer_forward": build_receiver_normalizers(edge_target, 2),
            "receiver_normalizer_reverse": build_receiver_normalizers(edge_source, 2),
        }
        with self.assertRaises(ValueError):
            deterministic_diffusion_batch_scores(
                **arguments, origin_weight=0.0, destination_weight=0.0
            )
        with self.assertRaises(ValueError):
            deterministic_diffusion_batch_scores(
                **arguments, region_pooling="median"
            )


if __name__ == "__main__":
    unittest.main()
