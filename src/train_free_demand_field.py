"""Training-free demand-field scoring utilities."""

from __future__ import annotations

import torch

from .demand_field_nbfnet import _fixed_mean_diffusion


DEFAULT_DIFFUSION_DEPTHS = (1, 2, 4, 8, 16, 32)


def deterministic_diffusion_batch_scores(
    *,
    origin_fields: torch.Tensor,
    destination_fields: torch.Tensor,
    edge_source: torch.Tensor,
    edge_target: torch.Tensor,
    region_nodes: torch.Tensor,
    receiver_normalizer_forward: torch.Tensor,
    receiver_normalizer_reverse: torch.Tensor,
    depths: tuple[int, ...] = DEFAULT_DIFFUSION_DEPTHS,
    demand_scale: float = 1000.0,
) -> torch.Tensor:
    """Score regions using fixed bidirectional scalar diffusion only."""

    if origin_fields.shape != destination_fields.shape or origin_fields.ndim != 2:
        raise ValueError("origin and destination fields must be equal matrices")
    if region_nodes.ndim != 2:
        raise ValueError("region_nodes must be a matrix")
    if not depths or any(depth <= 0 for depth in depths):
        raise ValueError("depths must contain positive integers")
    if tuple(sorted(set(depths))) != depths:
        raise ValueError("depths must be strictly increasing")
    if demand_scale <= 0.0:
        raise ValueError("demand_scale must be positive")

    origin_state = torch.log1p(origin_fields * demand_scale).unsqueeze(-1)
    destination_state = torch.log1p(destination_fields * demand_scale).unsqueeze(-1)
    scores = origin_state.new_zeros((origin_fields.shape[0], region_nodes.shape[0]))
    requested_depths = set(depths)
    for depth in range(1, depths[-1] + 1):
        origin_state = _fixed_mean_diffusion(
            origin_state,
            edge_source,
            edge_target,
            receiver_normalizer_forward,
        )
        destination_state = _fixed_mean_diffusion(
            destination_state,
            edge_target,
            edge_source,
            receiver_normalizer_reverse,
        )
        if depth not in requested_depths:
            continue
        exposure = (origin_state + destination_state).squeeze(-1)
        region_values = exposure[:, region_nodes]
        scores += region_values.mean(dim=2) + region_values.amax(dim=2)
    return scores / len(depths)


def orient_from_training_split(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Choose only a global sign from training data using rank covariance."""

    if prediction.shape != target.shape or prediction.ndim != 1 or not len(target):
        raise ValueError("prediction and target must be equal non-empty vectors")
    prediction_order = torch.argsort(prediction, stable=True)
    target_order = torch.argsort(target, stable=True)
    prediction_rank = torch.empty_like(prediction, dtype=torch.float64)
    target_rank = torch.empty_like(target, dtype=torch.float64)
    rank = torch.arange(len(prediction), device=prediction.device, dtype=torch.float64)
    prediction_rank[prediction_order] = rank
    target_rank[target_order] = rank
    covariance = torch.sum(
        (prediction_rank - prediction_rank.mean())
        * (target_rank - target_rank.mean())
    )
    sign = 1 if float(covariance) >= 0.0 else -1
    return prediction * sign, sign
