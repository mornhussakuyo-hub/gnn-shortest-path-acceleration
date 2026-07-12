"""最短路基线使用的轻量图数据结构。"""

from __future__ import annotations

from dataclasses import dataclass, field


NodeId = int
Adjacency = dict[NodeId, list[tuple[NodeId, float]]]


@dataclass(slots=True)
class WeightedDiGraph:
    adjacency: Adjacency = field(default_factory=dict)
    reverse_adjacency: Adjacency = field(default_factory=dict)
    coordinates: dict[NodeId, tuple[float, float]] = field(default_factory=dict)
    edge_count: int = 0

    def add_node(self, node: NodeId, lon: float | None = None, lat: float | None = None) -> None:
        self.adjacency.setdefault(node, [])
        self.reverse_adjacency.setdefault(node, [])
        if lon is not None and lat is not None:
            self.coordinates[node] = (lon, lat)

    def add_edge(self, source: NodeId, target: NodeId, weight: float) -> None:
        if weight <= 0:
            raise ValueError(f"edge weight must be positive: {source}->{target} has {weight}")
        self.add_node(source)
        self.add_node(target)
        self.adjacency[source].append((target, weight))
        self.reverse_adjacency[target].append((source, weight))
        self.edge_count += 1

    @property
    def node_count(self) -> int:
        return len(self.adjacency)

    def has_node(self, node: NodeId) -> bool:
        return node in self.adjacency

    def out_neighbors(self, node: NodeId) -> list[tuple[NodeId, float]]:
        return self.adjacency.get(node, [])

    def in_neighbors(self, node: NodeId) -> list[tuple[NodeId, float]]:
        return self.reverse_adjacency.get(node, [])


@dataclass(frozen=True, slots=True)
class Query:
    query_id: int
    origin: NodeId
    destination: NodeId
    query_type: str = "porto_od"
    count: int = 1
    timestamp: int | None = None


@dataclass(frozen=True, slots=True)
class GraphSummary:
    node_count: int
    edge_count: int
    isolated_nodes: int
    max_out_degree: int
    average_out_degree: float
    min_weight: float
    max_weight: float
