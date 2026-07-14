"""为连通区域构建 shortcut 压缩索引。"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass

from .graph_types import NodeId, WeightedDiGraph
from .regions import Region


NODE_OUTSIDE = 0
NODE_BOUNDARY = 1
NODE_INTERNAL = 2


@dataclass(frozen=True, slots=True)
class ShortcutEdge:
    source: NodeId
    target: NodeId
    weight: float
    region_id: int


@dataclass(frozen=True, slots=True)
class CompressionIndex:
    regions: tuple[Region, ...]
    shortcuts: tuple[ShortcutEdge, ...]
    compressed_graph: WeightedDiGraph
    node_states: dict[NodeId, int]
    node_regions: dict[NodeId, int]
    regions_by_id: dict[int, Region]
    build_seconds: float

    @property
    def region_count(self) -> int:
        return len(self.regions)

    @property
    def shortcut_count(self) -> int:
        return len(self.shortcuts)

    @property
    def internal_node_count(self) -> int:
        return sum(state == NODE_INTERNAL for state in self.node_states.values())

    def requires_original_graph(self, source: NodeId, target: NodeId) -> bool:
        del source, target
        return False

    def requires_endpoint_access(self, source: NodeId, target: NodeId) -> bool:
        return (
            self.node_states.get(source) == NODE_INTERNAL
            or self.node_states.get(target) == NODE_INTERNAL
        )

    def region_for_node(self, node: NodeId) -> Region | None:
        region_id = self.node_regions.get(node)
        return self.regions_by_id.get(region_id) if region_id is not None else None


def build_compression_index(graph: WeightedDiGraph, regions: list[Region]) -> CompressionIndex:
    start = time.perf_counter()
    shortcuts: list[ShortcutEdge] = []
    node_states = {node: NODE_OUTSIDE for node in graph.adjacency}
    node_regions: dict[NodeId, int] = {}
    regions_by_id: dict[int, Region] = {}

    for region in regions:
        region_nodes = set(region.nodes)
        if region.region_id in regions_by_id:
            raise ValueError(f"duplicate region id: {region.region_id}")
        overlapping_nodes = region_nodes & node_regions.keys()
        if overlapping_nodes:
            overlapping_node = min(overlapping_nodes)
            raise ValueError(f"compression regions overlap at node {overlapping_node}")
        regions_by_id[region.region_id] = region
        node_regions.update({node: region.region_id for node in region_nodes})
        boundary_nodes = sorted(region.boundary_nodes)
        boundary_set = set(boundary_nodes)
        for node in boundary_nodes:
            node_states[node] = NODE_BOUNDARY
        for node in region_nodes - boundary_set:
            node_states[node] = NODE_INTERNAL
        for source in boundary_nodes:
            distances = _restricted_dijkstra(graph, source, region_nodes)
            for target in boundary_nodes:
                if target == source:
                    continue
                distance = distances.get(target, math.inf)
                if math.isfinite(distance):
                    shortcuts.append(ShortcutEdge(source, target, distance, region.region_id))

    compressed_graph = _materialize_compressed_graph(graph, node_states, shortcuts)

    return CompressionIndex(
        regions=tuple(regions),
        shortcuts=tuple(shortcuts),
        compressed_graph=compressed_graph,
        node_states=node_states,
        node_regions=node_regions,
        regions_by_id=regions_by_id,
        build_seconds=time.perf_counter() - start,
    )


def _materialize_compressed_graph(
    graph: WeightedDiGraph,
    node_states: dict[NodeId, int],
    shortcuts: list[ShortcutEdge],
) -> WeightedDiGraph:
    compressed = WeightedDiGraph()
    retained_nodes = {
        node for node, state in node_states.items() if state != NODE_INTERNAL
    }

    for node in retained_nodes:
        coordinate = graph.coordinates.get(node)
        if coordinate is None:
            compressed.add_node(node)
        else:
            compressed.add_node(node, coordinate[0], coordinate[1])

    edge_weights: dict[NodeId, dict[NodeId, float]] = {}
    for source in retained_nodes:
        for target, weight in graph.out_neighbors(source):
            if target in retained_nodes:
                _keep_minimum_edge(edge_weights, source, target, weight)
    for shortcut in shortcuts:
        _keep_minimum_edge(
            edge_weights,
            shortcut.source,
            shortcut.target,
            shortcut.weight,
        )

    for source, targets in edge_weights.items():
        for target, weight in targets.items():
            compressed.add_edge(source, target, weight)
    return compressed


def _keep_minimum_edge(
    edge_weights: dict[NodeId, dict[NodeId, float]],
    source: NodeId,
    target: NodeId,
    weight: float,
) -> None:
    targets = edge_weights.setdefault(source, {})
    if weight < targets.get(target, math.inf):
        targets[target] = weight


def _restricted_dijkstra(
    graph: WeightedDiGraph,
    source: NodeId,
    allowed_nodes: set[NodeId],
) -> dict[NodeId, float]:
    distances: dict[NodeId, float] = {source: 0.0}
    queue: list[tuple[float, NodeId]] = [(0.0, source)]
    settled: set[NodeId] = set()

    while queue:
        distance, node = heapq.heappop(queue)
        if node in settled:
            continue
        settled.add(node)
        for neighbor, weight in graph.out_neighbors(node):
            if neighbor not in allowed_nodes or neighbor in settled:
                continue
            new_distance = distance + weight
            if new_distance < distances.get(neighbor, math.inf):
                distances[neighbor] = new_distance
                heapq.heappush(queue, (new_distance, neighbor))

    return distances
