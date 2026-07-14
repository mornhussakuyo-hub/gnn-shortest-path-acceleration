"""查询完全物化的压缩图，并在必要时精确回退到原图。"""

from __future__ import annotations

import heapq
import math
import time
from collections import OrderedDict
from dataclasses import dataclass, field

from .compression_index import CompressionIndex, NODE_INTERNAL
from .graph_types import NodeId, WeightedDiGraph
from .regions import Region
from .shortest_path import (
    ShortestPathResult,
    bidirectional_dijkstra_distance,
    bidirectional_dijkstra_from_frontiers,
)


@dataclass(slots=True)
class EndpointAccessCache:
    capacity: int
    _entries: OrderedDict[tuple[NodeId, bool, int], dict[NodeId, float]] = field(
        default_factory=OrderedDict,
        init=False,
    )

    def __post_init__(self) -> None:
        if self.capacity < 0:
            raise ValueError("endpoint access cache capacity must be non-negative")

    def get(
        self,
        node: NodeId,
        reverse: bool,
        region_id: int,
    ) -> dict[NodeId, float] | None:
        key = (node, reverse, region_id)
        value = self._entries.get(key)
        if value is not None:
            self._entries.move_to_end(key)
        return value

    def put(
        self,
        node: NodeId,
        reverse: bool,
        region_id: int,
        frontier: dict[NodeId, float],
    ) -> None:
        if self.capacity == 0:
            return
        key = (node, reverse, region_id)
        self._entries[key] = frontier
        self._entries.move_to_end(key)
        while len(self._entries) > self.capacity:
            self._entries.popitem(last=False)

    def __len__(self) -> int:
        return len(self._entries)


def indexed_bidirectional_dijkstra_distance(
    graph: WeightedDiGraph,
    index: CompressionIndex,
    source: int,
    target: int,
    endpoint_cache: EndpointAccessCache | None = None,
) -> ShortestPathResult:
    start = time.perf_counter()
    if not graph.has_node(source) or not graph.has_node(target):
        return ShortestPathResult(
            math.inf,
            0,
            (time.perf_counter() - start) * 1000.0,
            False,
        )
    if source == target:
        return ShortestPathResult(
            0.0,
            1,
            (time.perf_counter() - start) * 1000.0,
            True,
        )
    if not index.requires_endpoint_access(source, target):
        return bidirectional_dijkstra_distance(
            index.compressed_graph,
            source,
            target,
            _start_time=start,
        )

    forward_frontier = {source: 0.0}
    backward_frontier = {target: 0.0}
    local_expanded = 0
    cache_hits = 0
    cache_misses = 0
    direct_distance = math.inf

    if index.node_states.get(source) == NODE_INTERNAL:
        source_region = index.region_for_node(source)
        if source_region is None:
            raise RuntimeError(f"internal source {source} has no compression region")
        (
            forward_frontier,
            expanded,
            source_cache_hit,
            source_cache_miss,
            source_distances,
        ) = _endpoint_frontier(
            graph,
            source,
            source_region,
            reverse=False,
            cache=endpoint_cache,
        )
        local_expanded += expanded
        cache_hits += source_cache_hit
        cache_misses += source_cache_miss
        if target in source_region.nodes:
            if source_distances is not None:
                direct_distance = source_distances.get(target, math.inf)
            else:
                direct_distances, expanded = _restricted_distances(
                    graph,
                    source,
                    source_region.nodes,
                    reverse=False,
                    target=target,
                )
                local_expanded += expanded
                direct_distance = direct_distances.get(target, math.inf)

    if index.node_states.get(target) == NODE_INTERNAL:
        target_region = index.region_for_node(target)
        if target_region is None:
            raise RuntimeError(f"internal target {target} has no compression region")
        (
            backward_frontier,
            expanded,
            target_cache_hit,
            target_cache_miss,
            _,
        ) = _endpoint_frontier(
            graph,
            target,
            target_region,
            reverse=True,
            cache=endpoint_cache,
        )
        local_expanded += expanded
        cache_hits += target_cache_hit
        cache_misses += target_cache_miss

    compressed_result = bidirectional_dijkstra_from_frontiers(
        index.compressed_graph,
        forward_frontier,
        backward_frontier,
        _start_time=start,
    )
    distance = min(direct_distance, compressed_result.distance)
    return ShortestPathResult(
        distance=distance,
        expanded_nodes=local_expanded + compressed_result.expanded_nodes,
        elapsed_ms=(time.perf_counter() - start) * 1000.0,
        reachable=math.isfinite(distance),
        endpoint_access_expanded_nodes=local_expanded,
        graph_search_expanded_nodes=compressed_result.expanded_nodes,
        endpoint_cache_hits=cache_hits,
        endpoint_cache_misses=cache_misses,
    )


def _endpoint_frontier(
    graph: WeightedDiGraph,
    source: NodeId,
    region: Region,
    *,
    reverse: bool,
    cache: EndpointAccessCache | None,
) -> tuple[dict[NodeId, float], int, int, int, dict[NodeId, float] | None]:
    if cache is not None and cache.capacity > 0:
        cached = cache.get(source, reverse, region.region_id)
        if cached is not None:
            return cached, 0, 1, 0, None

    distances, expanded = _restricted_distances(
        graph,
        source,
        region.nodes,
        reverse=reverse,
    )
    frontier = {
        boundary: distances[boundary]
        for boundary in region.boundary_nodes
        if boundary in distances
    }
    cache_miss = int(cache is not None and cache.capacity > 0)
    if cache is not None:
        cache.put(source, reverse, region.region_id, frontier)
    return frontier, expanded, 0, cache_miss, distances


def _restricted_distances(
    graph: WeightedDiGraph,
    source: NodeId,
    allowed_nodes: frozenset[NodeId],
    *,
    reverse: bool,
    target: NodeId | None = None,
) -> tuple[dict[NodeId, float], int]:
    distances: dict[NodeId, float] = {source: 0.0}
    queue: list[tuple[float, NodeId]] = [(0.0, source)]
    settled: set[NodeId] = set()

    while queue:
        distance, node = heapq.heappop(queue)
        if node in settled:
            continue
        settled.add(node)
        if node == target:
            break
        neighbors = graph.in_neighbors(node) if reverse else graph.out_neighbors(node)
        for neighbor, weight in neighbors:
            if neighbor not in allowed_nodes or neighbor in settled:
                continue
            new_distance = distance + weight
            if new_distance < distances.get(neighbor, math.inf):
                distances[neighbor] = new_distance
                heapq.heappush(queue, (new_distance, neighbor))

    return distances, len(settled)
