from __future__ import annotations

import copy
import unittest
from dataclasses import asdict

import numpy as np

try:
    import torch

    from src.demand_field_nbfnet import (
        BidirectionalNBFNet,
        NBFNetConfig,
        _DirectionalLayer,
        _PropagationOnlyLayer,
        build_edge_features,
        build_receiver_normalizers,
    )
    from scripts.train_demand_field_nbfnet import (
        PrecisionPolicy,
        _build_learning_rate_scheduler,
        _full_pairwise_accuracy,
        _optimizer_step_was_skipped,
        _optimizer_state_step,
        _parameter_delta_norm,
        _parameter_snapshot,
        _parse_float_grid,
        _resolve_precision_policy,
        _residual_gate_metrics,
        _select_residual_gate,
        _set_training_scope,
        _soft_spearman_loss_tensor,
        _trainable_parameter_count,
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

    def test_plateau_scheduler_reduces_only_after_frozen_patience(self) -> None:
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5.0e-3)
        scheduler = _build_learning_rate_scheduler(
            optimizer,
            "reduce_on_plateau",
            factor=0.3,
            patience=3,
            threshold=1.0e-4,
            min_lr=5.0e-4,
        )
        self.assertIsNotNone(scheduler)
        for value in (0.94, 0.94, 0.94, 0.94):
            scheduler.step(value)
        self.assertEqual(optimizer.param_groups[0]["lr"], 5.0e-3)
        scheduler.step(0.94)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.5e-3)

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

    def test_propagation_structure_is_explicit_and_serializable(self) -> None:
        legacy = NBFNetConfig(variant="propagation_residual_doubling")
        self.assertEqual(legacy.resolved_propagation_structure(), "g1")
        explicit = NBFNetConfig(
            variant="propagation_residual_doubling",
            propagation_structure="g3",
            propagation_residual_scale=0.01,
        )
        explicit.validate()
        self.assertEqual(explicit.resolved_propagation_structure(), "g3")
        self.assertEqual(asdict(explicit)["propagation_structure"], "g3")
        with self.assertRaises(ValueError):
            NBFNetConfig(
                variant="propagation_doubling",
                propagation_structure="unknown",
            ).validate()
        with self.assertRaises(ValueError):
            NBFNetConfig(variant="base", propagation_structure="g2").validate()

    def test_s0_structures_preserve_shape_zero_input_and_gradient_scope(self) -> None:
        edge_source = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        edge_target = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        edge_features = torch.randn(4, 3)
        normalizer = build_receiver_normalizers(edge_target, 5)
        for structure in ("g0", "g1", "g2", "g3"):
            with self.subTest(structure=structure):
                layer = _PropagationOnlyLayer(4, 3, structure)
                zero = torch.zeros(2, 5, 4)
                zero_output = layer(
                    zero,
                    edge_source,
                    edge_target,
                    edge_features,
                    normalizer,
                )
                self.assertEqual(tuple(zero_output.shape), (2, 5, 4))
                self.assertTrue(torch.equal(zero_output, torch.zeros_like(zero_output)))

                state = torch.randn(2, 5, 4, requires_grad=True)
                output = layer(
                    state,
                    edge_source,
                    edge_target,
                    edge_features,
                    normalizer,
                )
                output.square().mean().backward()
                self.assertIsNotNone(state.grad)
                self.assertIsNotNone(layer.message_state.weight.grad)
                self.assertTrue(torch.isfinite(output).all())

    def test_g2_and_g3_have_an_exact_identity_when_branch_is_zero(self) -> None:
        edge_source = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        edge_target = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        edge_features = torch.randn(4, 3)
        normalizer = build_receiver_normalizers(edge_target, 5)
        state = torch.randn(2, 5, 4)
        for structure in ("g2", "g3"):
            with self.subTest(structure=structure):
                layer = _PropagationOnlyLayer(4, 3, structure, residual_scale=0.01)
                for parameter in layer.aggregate_projection.parameters():
                    torch.nn.init.zeros_(parameter)
                output = layer(
                    state,
                    edge_source,
                    edge_target,
                    edge_features,
                    normalizer,
                )
                self.assertTrue(torch.equal(output, state))

    def test_g3_normalizes_before_message_projection(self) -> None:
        layer = _PropagationOnlyLayer(4, 3, "g3")
        state = torch.randn(1, 5, 4)
        edge_source = torch.tensor([0, 1, 2, 3], dtype=torch.long)
        edge_target = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        captured: list[torch.Tensor] = []
        handle = layer.message_state.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach().clone())
        )
        try:
            layer(
                state,
                edge_source,
                edge_target,
                torch.randn(4, 3),
                build_receiver_normalizers(edge_target, 5),
            )
        finally:
            handle.remove()
        expected = layer.normalization(state)[:, edge_source, :]
        self.assertTrue(torch.allclose(captured[0], expected))

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

    def test_head_warmup_scope_only_unfreezes_final_residual_layer(self) -> None:
        model = BidirectionalNBFNet(
            3,
            2,
            2,
            NBFNetConfig(
                hidden_dim=4,
                propagation_layers=2,
                max_epochs=1,
                variant="propagation_doubling",
                propagation_structure="g2",
                zero_initialize_prediction_head=True,
            ),
        )
        _set_training_scope(model, "output_head")
        output_layer = model.prediction_head[-1]
        self.assertTrue(all(parameter.requires_grad for parameter in output_layer.parameters()))
        self.assertEqual(
            _trainable_parameter_count(model),
            sum(parameter.numel() for parameter in output_layer.parameters()),
        )
        self.assertFalse(model.origin_encoder[0].weight.requires_grad)
        _set_training_scope(model, "all")
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))

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

    def test_soft_spearman_is_scale_invariant_and_rank_aligned(self) -> None:
        target = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        aligned = torch.tensor([-2.0, -0.5, 0.0, 1.0, 3.0], requires_grad=True)
        scaled = aligned.detach() * 17.0 + 23.0
        reversed_prediction = -aligned.detach()
        aligned_loss = _soft_spearman_loss_tensor(aligned, target, 0.1)
        scaled_loss = _soft_spearman_loss_tensor(scaled, target, 0.1)
        reversed_loss = _soft_spearman_loss_tensor(reversed_prediction, target, 0.1)
        self.assertAlmostEqual(aligned_loss.item(), scaled_loss.item(), places=5)
        self.assertLess(aligned_loss.item(), reversed_loss.item())
        aligned_loss.backward()
        self.assertTrue(torch.isfinite(aligned.grad).all())

    def test_residual_gate_can_fall_back_to_fixed_prior(self) -> None:
        target = np.asarray([0.0, 1.0, 2.0, 3.0])
        prior = torch.tensor([0.0, 1.0, 2.0, 3.0])
        prediction = torch.tensor([3.0, 2.0, 1.0, 0.0])
        alpha, selected, raw_spearman = _select_residual_gate(
            prediction,
            prior,
            target,
            (0.0, 0.25, 0.5, 1.0),
        )
        self.assertEqual(alpha, 0.0)
        self.assertTrue(torch.equal(selected, prior))
        self.assertLess(raw_spearman, 0.0)

    def test_residual_gate_catalog_records_every_frozen_budget_metric(self) -> None:
        target = np.asarray([5.0, 20.0, 10.0, 30.0, 15.0, 25.0])
        prior = torch.tensor([0.0, 3.0, 1.0, 5.0, 2.0, 4.0])
        prediction = torch.tensor([5.0, 0.0, 4.0, 1.0, 3.0, 2.0])
        rows = _residual_gate_metrics(
            prediction,
            prior,
            target,
            (0.0, 0.5, 1.0),
        )
        self.assertEqual([row["alpha"] for row in rows], [0.0, 0.5, 1.0])
        for row in rows:
            self.assertEqual(set(row["ndcg_at_k"]), {"5", "10", "18"})
            self.assertEqual(set(row["top_gain_at_k"]), {"5", "10", "18"})
        self.assertEqual(rows[0]["spearman"], 1.0)

    def test_residual_gate_grid_is_strict_and_sorted(self) -> None:
        self.assertEqual(
            _parse_float_grid("1,0,0.5", "--gate"),
            (0.0, 0.5, 1.0),
        )
        with self.assertRaises(ValueError):
            _parse_float_grid("0,1.5", "--gate")

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
