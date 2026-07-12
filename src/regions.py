"""为 shortcut 索引实验生成连通区域。"""

from __future__ import annotations

import random
from collections import Counter, deque
from dataclasses import dataclass
from typing import Iterable

from .graph_types import NodeId, Query, WeightedDiGraph


@dataclass(frozen=True, slots=True)
class Region:
    region_id: int
    nodes: frozenset[NodeId]
    boundary_nodes: frozenset[NodeId]
    seed_node: NodeId
    selection_method: str

    @property
    def node_count(self) -> int:
        return len(self.nodes)

    @property
    def boundary_count(self) -> int:
        return len(self.boundary_nodes)

    @property
    def storage_cost_estimate(self) -> int:
        return self.boundary_count * self.boundary_count


def build_random_regions(
    graph: WeightedDiGraph,
    region_count: int,
    region_size: int,
    seed: int = 42,
    allow_overlap: bool = False,
) -> list[Region]:
    rng = random.Random(seed)
    candidates = list(graph.adjacency)
    rng.shuffle(candidates)
    return _build_regions_from_seeds(
        graph,
        candidates,
        region_count,
        region_size,
        "random_bfs",
        allow_overlap=allow_overlap,
    )


def build_hotspot_regions(
    graph: WeightedDiGraph,
    queries: list[Query],
    region_count: int,
    region_size: int,
    allow_overlap: bool = False,
) -> list[Region]:
    counts: Counter[NodeId] = Counter()
    for query in queries:
        counts[query.origin] += 1
        counts[query.destination] += 1
    seeds = [node for node, _ in counts.most_common()]
    return _build_regions_from_seeds(
        graph,
        seeds,
        region_count,
        region_size,
        "od_hotspot_bfs",
        allow_overlap=allow_overlap,
    )


def build_scored_regions(
    graph: WeightedDiGraph,
    node_scores: dict[NodeId, float],
    region_count: int,
    region_size: int,
    seed_exclusion_hops: int = 2,
) -> list[Region]:
    candidates = sorted(
        graph.adjacency,
        key=lambda node: (-node_scores.get(node, float("-inf")), node),
    )
    regions: list[Region] = []
    used_nodes: set[NodeId] = set()
    forbidden_seeds: set[NodeId] = set()

    for seed_node in candidates:
        if seed_node in forbidden_seeds:
            continue
        nodes = grow_bfs_region(graph, seed_node, region_size)
        if len(nodes) < 2 or nodes & used_nodes:
            continue
        boundary_nodes = find_boundary_nodes(graph, nodes)
        if len(boundary_nodes) < 2:
            continue
        regions.append(
            Region(
                region_id=len(regions),
                nodes=nodes,
                boundary_nodes=boundary_nodes,
                seed_node=seed_node,
                selection_method="gnn_seed_score",
            )
        )
        used_nodes.update(nodes)
        forbidden_seeds.update(_expand_node_set(graph, nodes, seed_exclusion_hops))
        if len(regions) >= region_count:
            break

    return regions


def build_risk_aware_scored_regions(
    graph: WeightedDiGraph,
    node_scores: dict[NodeId, float],
    queries: list[Query],
    region_count: int,
    region_size: int,
    seed_exclusion_hops: int = 2,
    candidate_limit: int = 20_000,
    endpoint_risk_penalty: float = 100.0,
) -> list[Region]:
    endpoint_counts: Counter[NodeId] = Counter()
    for query in queries:
        endpoint_counts[query.origin] += query.count
        endpoint_counts[query.destination] += query.count
    endpoint_total = max(1, sum(endpoint_counts.values()))
    seeds = sorted(
        graph.adjacency,
        key=lambda node: (-node_scores.get(node, float("-inf")), node),
    )[: max(region_count, candidate_limit)]

    candidates: list[tuple[float, float, NodeId, frozenset[NodeId], frozenset[NodeId]]] = []
    for seed_node in seeds:
        nodes = grow_bfs_region(graph, seed_node, region_size)
        if len(nodes) < 2:
            continue
        boundary_nodes = find_boundary_nodes(graph, nodes)
        if len(boundary_nodes) < 2:
            continue
        internal_nodes = nodes - boundary_nodes
        endpoint_fraction = (
            sum(endpoint_counts[node] for node in internal_nodes) / endpoint_total
        )
        raw_score = node_scores.get(seed_node, 0.0)
        adjusted_score = raw_score / (
            1.0 + endpoint_risk_penalty * endpoint_fraction
        )
        candidates.append(
            (adjusted_score, raw_score, seed_node, nodes, boundary_nodes)
        )

    candidates.sort(key=lambda item: (-item[0], -item[1], item[2]))
    regions: list[Region] = []
    used_nodes: set[NodeId] = set()
    forbidden_seeds: set[NodeId] = set()
    for _, _, seed_node, nodes, boundary_nodes in candidates:
        if seed_node in forbidden_seeds or nodes & used_nodes:
            continue
        regions.append(
            Region(
                region_id=len(regions),
                nodes=nodes,
                boundary_nodes=boundary_nodes,
                seed_node=seed_node,
                selection_method="gnn_seed_score_risk_aware",
            )
        )
        used_nodes.update(nodes)
        forbidden_seeds.update(_expand_node_set(graph, nodes, seed_exclusion_hops))
        if len(regions) >= region_count:
            break
    return regions


def grow_bfs_region(graph: WeightedDiGraph, seed_node: NodeId, max_nodes: int) -> frozenset[NodeId]:
    if max_nodes <= 0 or not graph.has_node(seed_node):
        return frozenset()

    region: set[NodeId] = {seed_node}
    queue: deque[NodeId] = deque([seed_node])

    while queue and len(region) < max_nodes:
        node = queue.popleft()
        neighbors = sorted(_undirected_neighbors(graph, node))
        for neighbor in neighbors:
            if neighbor in region:
                continue
            region.add(neighbor)
            queue.append(neighbor)
            if len(region) >= max_nodes:
                break

    return frozenset(region)


def find_boundary_nodes(graph: WeightedDiGraph, nodes: Iterable[NodeId]) -> frozenset[NodeId]:
    region = set(nodes)
    boundary: set[NodeId] = set()
    for node in region:
        if any(neighbor not in region for neighbor, _ in graph.out_neighbors(node)):
            boundary.add(node)
        if any(neighbor not in region for neighbor, _ in graph.in_neighbors(node)):
            boundary.add(node)
    return frozenset(boundary)


def _build_regions_from_seeds(
    graph: WeightedDiGraph,
    seeds: Iterable[NodeId],
    region_count: int,
    region_size: int,
    selection_method: str,
    allow_overlap: bool,
) -> list[Region]:
    regions: list[Region] = []
    used_nodes: set[NodeId] = set()

    for seed_node in seeds:
        if not allow_overlap and seed_node in used_nodes:
            continue
        nodes = grow_bfs_region(graph, seed_node, region_size)
        if len(nodes) < 2:
            continue
        if not allow_overlap and nodes & used_nodes:
            continue
        boundary_nodes = find_boundary_nodes(graph, nodes)
        if len(boundary_nodes) < 2:
            continue
        region = Region(
            region_id=len(regions),
            nodes=nodes,
            boundary_nodes=boundary_nodes,
            seed_node=seed_node,
            selection_method=selection_method,
        )
        regions.append(region)
        used_nodes.update(nodes)
        if len(regions) >= region_count:
            break

    return regions


def _undirected_neighbors(graph: WeightedDiGraph, node: NodeId) -> set[NodeId]:
    neighbors = {neighbor for neighbor, _ in graph.out_neighbors(node)}
    neighbors.update(neighbor for neighbor, _ in graph.in_neighbors(node))
    return neighbors


def _expand_node_set(
    graph: WeightedDiGraph,
    nodes: Iterable[NodeId],
    hops: int,
) -> set[NodeId]:
    expanded = set(nodes)
    frontier = set(expanded)
    for _ in range(max(0, hops)):
        next_frontier: set[NodeId] = set()
        for node in frontier:
            next_frontier.update(_undirected_neighbors(graph, node))
        next_frontier.difference_update(expanded)
        if not next_frontier:
            break
        expanded.update(next_frontier)
        frontier = next_frontier
    return expanded
