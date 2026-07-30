"""第二版无传播区域 MLP 基线及统一回归指标。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class MLPConfig:
    hidden_dims: tuple[int, ...] = (64, 32)
    learning_rate: float = 0.003
    weight_decay: float = 0.0001
    rank_weight: float = 0.20
    huber_delta: float = 1.0
    batch_size: int = 64
    max_epochs: int = 500
    patience: int = 60
    min_improvement: float = 0.0001

    def validate(self) -> None:
        if not self.hidden_dims or any(width <= 0 for width in self.hidden_dims):
            raise ValueError("hidden_dims must contain positive widths")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0 or self.rank_weight < 0.0:
            raise ValueError("loss weights must be non-negative")
        if self.huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive")
        if self.batch_size <= 1 or self.max_epochs <= 0 or self.patience <= 0:
            raise ValueError("batch_size, max_epochs and patience must be positive")


def regression_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    top_fraction: float = 0.10,
) -> dict[str, float]:
    if prediction.shape != target.shape or prediction.ndim != 1 or not len(target):
        raise ValueError("prediction and target must be equal non-empty vectors")
    top_k = max(1, math.ceil(len(target) * top_fraction))
    predicted_order = np.argsort(-prediction, kind="stable")[:top_k]
    ideal_order = np.argsort(-target, kind="stable")[:top_k]
    ideal_dcg = _dcg(target[ideal_order])
    return {
        "count": int(len(target)),
        "mae": float(np.mean(np.abs(prediction - target))),
        "huber": _huber_loss(prediction, target, 10.0),
        "spearman": _spearman(prediction, target),
        "ndcg_at_k": _dcg(target[predicted_order]) / ideal_dcg if ideal_dcg else 0.0,
        "top_k": int(top_k),
        "top_k_mean_gain": float(np.mean(target[predicted_order])),
        "oracle_top_k_mean_gain": float(np.mean(target[ideal_order])),
        "all_mean_gain": float(np.mean(target)),
    }


def ranking_metrics_at_k(
    prediction: np.ndarray,
    target: np.ndarray,
    k_values: tuple[int, ...],
    *,
    region_nodes: np.ndarray | None = None,
) -> dict[str, dict[str, float | int]]:
    if prediction.shape != target.shape or prediction.ndim != 1 or not len(target):
        raise ValueError("prediction and target must be equal non-empty vectors")
    if not k_values or any(k <= 0 for k in k_values):
        raise ValueError("k_values must contain positive integers")
    if region_nodes is not None and (
        region_nodes.ndim != 2 or len(region_nodes) != len(target)
    ):
        raise ValueError("region_nodes must align with prediction and target")
    predicted_order = np.argsort(-prediction, kind="stable")
    ideal_order = np.argsort(-target, kind="stable")
    result: dict[str, dict[str, float | int]] = {}
    for requested_k in k_values:
        k = min(requested_k, len(target))
        selected = predicted_order[:k]
        ideal = ideal_order[:k]
        ideal_dcg = _dcg(target[ideal])
        values: dict[str, float | int] = {
            "k": int(k),
            "ndcg": _dcg(target[selected]) / ideal_dcg if ideal_dcg else 0.0,
            "mean_gain": float(np.mean(target[selected])),
            "oracle_mean_gain": float(np.mean(target[ideal])),
        }
        if region_nodes is not None:
            membership_count = int(region_nodes[selected].size)
            unique_node_count = len(set(region_nodes[selected].reshape(-1).tolist()))
            values.update(
                {
                    "membership_count": membership_count,
                    "unique_node_count": unique_node_count,
                    "membership_redundancy": (
                        membership_count / unique_node_count
                        if unique_node_count
                        else 0.0
                    ),
                }
            )
        result[str(requested_k)] = values
    return result


def _huber_loss(prediction: np.ndarray, target: np.ndarray, delta: float) -> float:
    difference = prediction - target
    absolute = np.abs(difference)
    values = np.where(
        absolute <= delta,
        0.5 * difference * difference,
        delta * (absolute - 0.5 * delta),
    )
    return float(np.mean(values))


def _spearman(left: np.ndarray, right: np.ndarray) -> float:
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    left_centered = left_rank - left_rank.mean()
    right_centered = right_rank - right_rank.mean()
    denominator = math.sqrt(
        float(np.sum(left_centered**2) * np.sum(right_centered**2))
    )
    if denominator == 0.0:
        return 0.0
    return float(np.sum(left_centered * right_centered) / denominator)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _dcg(gains: np.ndarray) -> float:
    discounts = np.log2(np.arange(2, len(gains) + 2, dtype=np.float64))
    return float(np.sum(np.maximum(gains, 0.0) / discounts))
