"""构造无路径监督 GNN 的节点特征、图边和代理训练目标。"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .graph_types import NodeId, Query, WeightedDiGraph


FEATURE_NAMES = (
    "longitude",
    "latitude",
    "log_out_degree",
    "log_in_degree",
    "mean_out_edge_length",
    "origin_frequency",
    "destination_frequency",
    "diffused_origin_demand",
    "diffused_destination_demand",
)


@dataclass(frozen=True, slots=True)
class GnnData:
    nodes: tuple[NodeId, ...]
    node_to_index: dict[NodeId, int]
    features: np.ndarray
    train_target: np.ndarray
    validation_target: np.ndarray
    test_target: np.ndarray
    endpoint_risk: np.ndarray
    edge_source: np.ndarray
    edge_target: np.ndarray
    target_degree: np.ndarray


def build_gnn_data(
    graph: WeightedDiGraph,
    train_queries: list[Query],
    validation_queries: list[Query],
    test_queries: list[Query],
    diffusion_steps: int = 3,
    diffusion_restart: float = 0.4,
    endpoint_penalty: float = 2.0,
    target_mode: str = "midpoint",
) -> GnnData:
    if target_mode not in {"midpoint", "demand_overlap"}:
        raise ValueError(f"不支持的代理目标模式: {target_mode}")
    if not 0.0 <= diffusion_restart <= 1.0:
        raise ValueError("扩散重启系数必须位于 0 到 1 之间。")
    nodes = tuple(graph.adjacency)
    node_to_index = {node: index for index, node in enumerate(nodes)}
    edge_source, edge_target = _build_undirected_edges(graph, node_to_index)
    target_degree = np.bincount(edge_target, minlength=len(nodes)).astype(np.float32)

    origin_counts, destination_counts = _endpoint_counts(
        train_queries,
        node_to_index,
        len(nodes),
    )
    diffused_origin = _diffuse(
        _density_signal(origin_counts, len(train_queries)),
        edge_source,
        edge_target,
        target_degree,
        diffusion_steps,
        diffusion_restart,
    )
    diffused_destination = _diffuse(
        _density_signal(destination_counts, len(train_queries)),
        edge_source,
        edge_target,
        target_degree,
        diffusion_steps,
        diffusion_restart,
    )
    endpoint_risk = _unit_scale(
        np.log1p(diffused_origin) + np.log1p(diffused_destination)
    )

    features = _node_features(
        graph,
        nodes,
        origin_counts,
        destination_counts,
        diffused_origin,
        diffused_destination,
    )
    train_target = _proxy_target(
        graph,
        nodes,
        node_to_index,
        train_queries,
        endpoint_risk,
        edge_source,
        edge_target,
        target_degree,
        diffusion_steps,
        diffusion_restart,
        endpoint_penalty,
        target_mode,
    )
    validation_target = _proxy_target(
        graph,
        nodes,
        node_to_index,
        validation_queries,
        endpoint_risk,
        edge_source,
        edge_target,
        target_degree,
        diffusion_steps,
        diffusion_restart,
        endpoint_penalty,
        target_mode,
    )
    test_target = _proxy_target(
        graph,
        nodes,
        node_to_index,
        test_queries,
        endpoint_risk,
        edge_source,
        edge_target,
        target_degree,
        diffusion_steps,
        diffusion_restart,
        endpoint_penalty,
        target_mode,
    )
    return GnnData(
        nodes=nodes,
        node_to_index=node_to_index,
        features=features,
        train_target=train_target,
        validation_target=validation_target,
        test_target=test_target,
        endpoint_risk=endpoint_risk.astype(np.float32),
        edge_source=edge_source,
        edge_target=edge_target,
        target_degree=target_degree,
    )


def _build_undirected_edges(
    graph: WeightedDiGraph,
    node_to_index: dict[NodeId, int],
) -> tuple[np.ndarray, np.ndarray]:
    sources: list[int] = []
    targets: list[int] = []
    for source, neighbors in graph.adjacency.items():
        source_index = node_to_index[source]
        for target, _ in neighbors:
            target_index = node_to_index[target]
            sources.extend((source_index, target_index))
            targets.extend((target_index, source_index))
    return np.asarray(sources, dtype=np.int64), np.asarray(targets, dtype=np.int64)


def _endpoint_counts(
    queries: list[Query],
    node_to_index: dict[NodeId, int],
    node_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    origins = np.zeros(node_count, dtype=np.float32)
    destinations = np.zeros(node_count, dtype=np.float32)
    for query in queries:
        origin_index = node_to_index.get(query.origin)
        destination_index = node_to_index.get(query.destination)
        if origin_index is not None:
            origins[origin_index] += query.count
        if destination_index is not None:
            destinations[destination_index] += query.count
    return origins, destinations


def _density_signal(counts: np.ndarray, query_count: int) -> np.ndarray:
    scale = 10_000.0 / max(1, query_count)
    return counts.astype(np.float32) * scale


def _diffuse(
    values: np.ndarray,
    edge_source: np.ndarray,
    edge_target: np.ndarray,
    target_degree: np.ndarray,
    steps: int,
    restart: float = 0.4,
) -> np.ndarray:
    base = values.astype(np.float32, copy=True)
    state = base.copy()
    safe_degree = np.maximum(target_degree, 1.0)
    for _ in range(max(0, steps)):
        aggregated = np.zeros_like(state)
        np.add.at(aggregated, edge_target, state[edge_source])
        state = restart * base + (1.0 - restart) * aggregated / safe_degree
    return state


def _node_features(
    graph: WeightedDiGraph,
    nodes: tuple[NodeId, ...],
    origin_counts: np.ndarray,
    destination_counts: np.ndarray,
    diffused_origin: np.ndarray,
    diffused_destination: np.ndarray,
) -> np.ndarray:
    longitude = np.asarray([graph.coordinates[node][0] for node in nodes], dtype=np.float32)
    latitude = np.asarray([graph.coordinates[node][1] for node in nodes], dtype=np.float32)
    out_degree = np.asarray([len(graph.out_neighbors(node)) for node in nodes], dtype=np.float32)
    in_degree = np.asarray([len(graph.in_neighbors(node)) for node in nodes], dtype=np.float32)
    mean_out_length = np.asarray(
        [
            sum(weight for _, weight in graph.out_neighbors(node)) / max(1, len(graph.out_neighbors(node)))
            for node in nodes
        ],
        dtype=np.float32,
    )
    raw_features = np.column_stack(
        (
            longitude,
            latitude,
            np.log1p(out_degree),
            np.log1p(in_degree),
            np.log1p(mean_out_length),
            np.log1p(origin_counts),
            np.log1p(destination_counts),
            np.log1p(diffused_origin),
            np.log1p(diffused_destination),
        )
    ).astype(np.float32)
    means = raw_features.mean(axis=0, keepdims=True)
    standard_deviations = raw_features.std(axis=0, keepdims=True)
    return ((raw_features - means) / np.maximum(standard_deviations, 1e-6)).astype(np.float32)


def _proxy_target(
    graph: WeightedDiGraph,
    nodes: tuple[NodeId, ...],
    node_to_index: dict[NodeId, int],
    queries: list[Query],
    endpoint_risk: np.ndarray,
    edge_source: np.ndarray,
    edge_target: np.ndarray,
    target_degree: np.ndarray,
    diffusion_steps: int,
    diffusion_restart: float,
    endpoint_penalty: float,
    target_mode: str,
) -> np.ndarray:
    if target_mode == "demand_overlap":
        origin_counts, destination_counts = _endpoint_counts(
            queries,
            node_to_index,
            len(nodes),
        )
        diffused_origin = _diffuse(
            _density_signal(origin_counts, len(queries)),
            edge_source,
            edge_target,
            target_degree,
            diffusion_steps,
            diffusion_restart,
        )
        diffused_destination = _diffuse(
            _density_signal(destination_counts, len(queries)),
            edge_source,
            edge_target,
            target_degree,
            diffusion_steps,
            diffusion_restart,
        )
        overlap_value = _unit_scale(
            np.log1p(np.sqrt(diffused_origin * diffused_destination))
        )
        target = overlap_value / (1.0 + endpoint_penalty * endpoint_risk)
        return _unit_scale(target).astype(np.float32)

    midpoint_counts = np.zeros(len(nodes), dtype=np.float32)
    grid = _coordinate_grid(graph, nodes)
    for query in queries:
        origin_coordinate = graph.coordinates.get(query.origin)
        destination_coordinate = graph.coordinates.get(query.destination)
        if origin_coordinate is None or destination_coordinate is None:
            continue
        midpoint = (
            (origin_coordinate[0] + destination_coordinate[0]) / 2.0,
            (origin_coordinate[1] + destination_coordinate[1]) / 2.0,
        )
        midpoint_index = _nearest_grid_node(midpoint, grid, graph, nodes, node_to_index)
        midpoint_counts[midpoint_index] += query.count

    midpoint_density = _diffuse(
        _density_signal(midpoint_counts, len(queries)),
        edge_source,
        edge_target,
        target_degree,
        diffusion_steps,
        diffusion_restart,
    )
    midpoint_value = _unit_scale(np.log1p(midpoint_density))
    target = midpoint_value / (1.0 + endpoint_penalty * endpoint_risk)
    return _unit_scale(target).astype(np.float32)


def _coordinate_grid(
    graph: WeightedDiGraph,
    nodes: tuple[NodeId, ...],
    cell_size: float = 0.002,
) -> tuple[dict[tuple[int, int], list[int]], float]:
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, node in enumerate(nodes):
        longitude, latitude = graph.coordinates[node]
        grid[(math.floor(longitude / cell_size), math.floor(latitude / cell_size))].append(index)
    return dict(grid), cell_size


def _nearest_grid_node(
    coordinate: tuple[float, float],
    grid_data: tuple[dict[tuple[int, int], list[int]], float],
    graph: WeightedDiGraph,
    nodes: tuple[NodeId, ...],
    node_to_index: dict[NodeId, int],
) -> int:
    grid, cell_size = grid_data
    longitude, latitude = coordinate
    base_x = math.floor(longitude / cell_size)
    base_y = math.floor(latitude / cell_size)
    candidates: list[int] = []
    for radius in range(8):
        for grid_x in range(base_x - radius, base_x + radius + 1):
            for grid_y in range(base_y - radius, base_y + radius + 1):
                if radius > 0 and abs(grid_x - base_x) < radius and abs(grid_y - base_y) < radius:
                    continue
                candidates.extend(grid.get((grid_x, grid_y), ()))
        if candidates:
            break
    if not candidates:
        nearest_node = min(
            nodes,
            key=lambda node: (
                graph.coordinates[node][0] - longitude
            ) ** 2
            + (graph.coordinates[node][1] - latitude) ** 2,
        )
        return node_to_index[nearest_node]
    return min(
        candidates,
        key=lambda index: (
            graph.coordinates[nodes[index]][0] - longitude
        ) ** 2
        + (graph.coordinates[nodes[index]][1] - latitude) ** 2,
    )


def _unit_scale(values: np.ndarray) -> np.ndarray:
    minimum = float(values.min(initial=0.0))
    maximum = float(values.max(initial=0.0))
    if maximum - minimum <= 1e-12:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - minimum) / (maximum - minimum)).astype(np.float32)
