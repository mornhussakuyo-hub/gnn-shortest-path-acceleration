from __future__ import annotations

import copy
import unittest

try:
    import torch

    from src.demand_field_nbfnet import (
        BidirectionalNBFNet,
        NBFNetConfig,
        _DirectionalLayer,
        build_edge_features,
        build_receiver_normalizers,
    )
    from scripts.train_demand_field_nbfnet import (
        PrecisionPolicy,
        _full_pairwise_accuracy,
        _optimizer_step_was_skipped,
        _optimizer_state_step,
        _parameter_delta_norm,
        _parameter_snapshot,
        _resolve_precision_policy,
        _evaluation_loss,
    )
except ImportError:  # pragma: no cover - covered by the CUDA environment instead.
    torch = None


@unittest.skipIf(torch is None, "PyTorch is not installed")
class BidirectionalNBFNetTest(unittest.TestCase):
    def test_precision_policy_keeps_diagnostic_comparisons_single_variable(self) -> None:
        self.assertEqual(_resolve_precision_policy(None, False, 65536.0).mode, "fp16")
        self.assertEqual(_resolve_precision_policy("bf16", False, 65536.0).mode, "bf16")
        self.assertEqual(_resolve_precision_policy(None, True, 65536.0).mode, "fp32")
        with self.assertRaises(ValueError):
            _resolve_precision_policy("bf16", True, 65536.0)

    def test_optimizer_skip_uses_actual_adam_step_counter(self) -> None:
        model = torch.nn.Linear(2, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        self.assertEqual(_optimizer_state_step(optimizer), 0)
        self.assertTrue(_optimizer_step_was_skipped(0, 0))
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()
        self.assertEqual(_optimizer_state_step(optimizer), 1)
        self.assertFalse(_optimizer_step_was_skipped(0, 1))

    def test_pairwise_accuracy_and_parameter_delta_are_directly_observed(self) -> None:
        prediction = torch.tensor([0.1, 0.4, 0.2])
        target = torch.tensor([1.0, 3.0, 2.0])
        self.assertEqual(_full_pairwise_accuracy(prediction, target), 1.0)

        model = torch.nn.Linear(2, 1, bias=False)
        snapshot = _parameter_snapshot(model)
        with torch.no_grad():
            model.weight.add_(1.0)
        self.assertAlmostEqual(_parameter_delta_norm(snapshot, model), 2.0**0.5)

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

    def test_zero_initialized_residual_head_is_exactly_zero(self) -> None:
        model = BidirectionalNBFNet(
            3,
            2,
            2,
            NBFNetConfig(
                hidden_dim=4,
                propagation_layers=2,
                max_epochs=1,
                zero_initialize_prediction_head=True,
            ),
        )
        output_layer = model.prediction_head[-1]
        self.assertTrue(
            torch.equal(output_layer.weight, torch.zeros_like(output_layer.weight))
        )
        self.assertTrue(torch.equal(output_layer.bias, torch.zeros_like(output_layer.bias)))

    def test_rank_first_evaluation_excludes_huber_from_total(self) -> None:
        prediction = torch.tensor([0.0, 2.0, 1.0])
        target = torch.tensor([1.0, 3.0, 2.0])
        config = NBFNetConfig(hidden_dim=4, propagation_layers=1, max_epochs=1)
        total, huber, rank = _evaluation_loss(
            prediction,
            target,
            config,
            "rank_first",
        )
        self.assertGreater(huber, 0.0)
        self.assertAlmostEqual(total, rank)

    def test_zero_state_does_not_create_messages(self) -> None:
        layer = _DirectionalLayer(hidden_dim=4, edge_dim=3)
        state = torch.zeros(2, 5, 4)
        edge_source = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        edge_target = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        result = layer(
            state,
            edge_source,
            edge_target,
            torch.randn(4, 3),
            build_receiver_normalizers(edge_target, 5),
        )
        self.assertTrue(torch.equal(result, torch.zeros_like(result)))

    def test_prototype_condition_changes_prediction(self) -> None:
        torch.manual_seed(42)
        config = NBFNetConfig(hidden_dim=4, propagation_layers=2, max_epochs=1)
        model = BidirectionalNBFNet(3, 2, 2, config)
        edge_source = torch.tensor([0, 1, 2, 3, 1], dtype=torch.long)
        edge_target = torch.tensor([1, 2, 3, 4, 0], dtype=torch.long)
        common = {
            "node_features": torch.randn(5, 3),
            "edge_source": edge_source,
            "edge_target": edge_target,
            "edge_features": build_edge_features(
                torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6]),
                torch.tensor([0, 1, 0, 1, 0]),
                2,
            ),
            "region_nodes": torch.tensor([[0, 1, 2], [2, 3, 4]]),
            "region_features": torch.zeros(2, 2),
            "receiver_normalizer_forward": build_receiver_normalizers(
                edge_target, 5
            ),
            "receiver_normalizer_reverse": build_receiver_normalizers(
                edge_source, 5
            ),
        }
        first = model(
            origin_fields=torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0]]),
            destination_fields=torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0]]),
            **common,
        )
        second = model(
            origin_fields=torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]]),
            destination_fields=torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0]]),
            **common,
        )
        self.assertFalse(torch.allclose(first, second))

    def test_chunked_two_pass_gradient_matches_full_mixture(self) -> None:
        torch.manual_seed(7)
        config = NBFNetConfig(
            hidden_dim=4,
            propagation_layers=1,
            rank_weight=0.0,
            max_epochs=1,
        )
        direct_model = BidirectionalNBFNet(3, 2, 2, config)
        chunked_model = copy.deepcopy(direct_model)
        edge_source = torch.tensor([0, 1, 2, 3, 1], dtype=torch.long)
        edge_target = torch.tensor([1, 2, 3, 4, 0], dtype=torch.long)
        origin_fields = torch.tensor(
            [[1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.5, 0.0, 0.0]]
        )
        destination_fields = torch.tensor(
            [[0.0, 0.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 0.0]]
        )
        prototype_weight = torch.tensor([0.3, 0.7])
        target = torch.tensor([0.4, -0.2])
        common = {
            "node_features": torch.randn(5, 3),
            "edge_source": edge_source,
            "edge_target": edge_target,
            "edge_features": build_edge_features(
                torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6]),
                torch.tensor([0, 1, 0, 1, 0]),
                2,
            ),
            "region_nodes": torch.tensor([[0, 1, 2], [2, 3, 4]]),
            "region_features": torch.randn(2, 2),
            "receiver_normalizer_forward": build_receiver_normalizers(
                edge_target, 5
            ),
            "receiver_normalizer_reverse": build_receiver_normalizers(
                edge_source, 5
            ),
        }
        direct_prediction = direct_model(
            origin_fields=origin_fields,
            destination_fields=destination_fields,
            **common,
        )
        direct_mixture = (direct_prediction * prototype_weight[:, None]).sum(0)
        torch.nn.functional.huber_loss(direct_mixture, target).backward()

        with torch.no_grad():
            mixture_value = (
                chunked_model(
                    origin_fields=origin_fields,
                    destination_fields=destination_fields,
                    **common,
                )
                * prototype_weight[:, None]
            ).sum(0)
        mixture_leaf = mixture_value.detach().requires_grad_(True)
        leaf_loss = torch.nn.functional.huber_loss(mixture_leaf, target)
        mixture_gradient = torch.autograd.grad(leaf_loss, mixture_leaf)[0]
        for prototype_id in range(2):
            chunk_prediction = chunked_model(
                origin_fields=origin_fields[prototype_id : prototype_id + 1],
                destination_fields=destination_fields[prototype_id : prototype_id + 1],
                **common,
            )[0]
            surrogate = (
                chunk_prediction * prototype_weight[prototype_id] * mixture_gradient
            ).sum()
            surrogate.backward()

        for direct_parameter, chunked_parameter in zip(
            direct_model.parameters(), chunked_model.parameters()
        ):
            self.assertIsNotNone(direct_parameter.grad)
            self.assertIsNotNone(chunked_parameter.grad)
            self.assertTrue(
                torch.allclose(
                    direct_parameter.grad,
                    chunked_parameter.grad,
                    atol=1e-6,
                    rtol=1e-5,
                )
            )
