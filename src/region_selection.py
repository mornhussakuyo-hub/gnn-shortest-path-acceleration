"""Deterministic diversity-aware selection over scored candidate regions."""

from __future__ import annotations

import numpy as np


SELECTION_METHODS = (
    "direct_topk",
    "jaccard_penalty",
    "marginal_coverage",
    "hard_disjoint",
)


def select_region_indices(
    scores: np.ndarray,
    region_nodes: np.ndarray,
    k: int,
    method: str,
) -> np.ndarray:
    """Select candidate row indices without reading any outcome labels."""

    if scores.ndim != 1 or region_nodes.ndim != 2 or len(scores) != len(region_nodes):
        raise ValueError("scores and region_nodes must align")
    if not len(scores) or not np.isfinite(scores).all():
        raise ValueError("scores must be finite and non-empty")
    if k <= 0:
        raise ValueError("k must be positive")
    if method not in SELECTION_METHODS:
        raise ValueError(f"method must be one of {', '.join(SELECTION_METHODS)}")
    k = min(k, len(scores))
    order = np.argsort(-scores, kind="stable")
    if method == "direct_topk":
        return order[:k]

    node_sets = [set(map(int, nodes)) for nodes in region_nodes]
    quality = _rank_quality(scores)
    selected: list[int] = []
    selected_set: set[int] = set()
    covered_nodes: set[int] = set()
    while len(selected) < k:
        best_index: int | None = None
        best_key: tuple[float, float, int] | None = None
        for index, nodes in enumerate(node_sets):
            if index in selected_set:
                continue
            intersection = len(nodes & covered_nodes)
            if method == "hard_disjoint" and intersection:
                continue
            if method == "jaccard_penalty":
                maximum_jaccard = max(
                    (_jaccard(nodes, node_sets[chosen]) for chosen in selected),
                    default=0.0,
                )
                objective = 0.5 * quality[index] + 0.5 * (1.0 - maximum_jaccard)
            elif method == "marginal_coverage":
                novel_fraction = len(nodes - covered_nodes) / max(1, len(nodes))
                objective = 0.5 * quality[index] + 0.5 * novel_fraction
            else:
                objective = quality[index]
            key = (float(objective), float(quality[index]), -index)
            if best_key is None or key > best_key:
                best_index = index
                best_key = key
        if best_index is None:
            break
        selected.append(best_index)
        selected_set.add(best_index)
        covered_nodes.update(node_sets[best_index])
    return np.asarray(selected, dtype=np.int64)


def selection_overlap_statistics(region_nodes: np.ndarray) -> dict[str, float | int | bool]:
    """Report exact node overlap and pairwise Jaccard for a selected set."""

    if region_nodes.ndim != 2:
        raise ValueError("region_nodes must be a matrix")
    node_sets = [set(map(int, nodes)) for nodes in region_nodes]
    membership_count = sum(len(nodes) for nodes in node_sets)
    union = set().union(*node_sets) if node_sets else set()
    overlapping_pairs = 0
    maximum_jaccard = 0.0
    for left in range(len(node_sets)):
        for right in range(left + 1, len(node_sets)):
            intersection = node_sets[left] & node_sets[right]
            if intersection:
                overlapping_pairs += 1
            maximum_jaccard = max(
                maximum_jaccard, _jaccard(node_sets[left], node_sets[right])
            )
    unique_node_count = len(union)
    return {
        "membership_count": membership_count,
        "unique_node_count": unique_node_count,
        "duplicate_membership_count": membership_count - unique_node_count,
        "membership_redundancy": (
            membership_count / unique_node_count if unique_node_count else 0.0
        ),
        "overlapping_pair_count": overlapping_pairs,
        "maximum_pairwise_jaccard": maximum_jaccard,
        "deployable_without_region_overlap": overlapping_pairs == 0,
    }


def _rank_quality(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(scores, kind="stable")
    quality = np.empty(len(scores), dtype=np.float64)
    quality[order] = np.linspace(0.0, 1.0, len(scores))
    return quality


def _jaccard(left: set[int], right: set[int]) -> float:
    union_size = len(left | right)
    return len(left & right) / union_size if union_size else 0.0
