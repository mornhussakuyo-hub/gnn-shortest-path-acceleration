from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Edge:
    from_point: int
    to_point: int
    value: float
    directed: bool = True


@dataclass
class Point:
    x_loc: float
    y_loc: float
    id: int = 0
    outEdges: List[Edge] = field(default_factory=list)
    inEdges: List[Edge] = field(default_factory=list)


class Graph:
    points: List[Point]
    edges: List[Edge]
    num_points: int
    num_directed_edges: int
    num_undirected_edges: int
    num_total_edges: int
    info: str

    def __init__(self) -> None:
        self.points = []
        self.edges = []
        self._directed_edges: Dict[tuple[int, int, float], Edge] = {}
        self._undirected_edges: Dict[tuple[int, int, float], Edge] = {}
        self._directed_pair_edges: Dict[tuple[int, int], List[Edge]] = {}
        self._undirected_pair_edges: Dict[tuple[int, int], List[Edge]] = {}
        self._edge_indexes: Dict[tuple[bool, int, int, float], int] = {}
        self.num_points = 0
        self.num_directed_edges = 0
        self.num_undirected_edges = 0
        self.num_total_edges = 0
        self.info = ""

    def set_info(self, in_info: str) -> bool:
        self.info = in_info
        return True

    def add_edge(self, edge: Edge) -> bool:
        if edge.from_point == edge.to_point:
            return False

        if not edge.directed:
            if self._find_undirected_edge(edge.from_point, edge.to_point, edge.value) is not None:
                return False
            if self._find_directed_edge(edge.from_point, edge.to_point, edge.value) is not None:
                return False
            if self._find_directed_edge(edge.to_point, edge.from_point, edge.value) is not None:
                return False
            self._append_edge(edge)
            return True

        if self._find_undirected_edge(edge.from_point, edge.to_point, edge.value) is not None:
            return False

        reverse_edge = self._find_directed_edge(
            from_point=edge.to_point,
            to_point=edge.from_point,
            value=edge.value,
        )

        if reverse_edge is not None:
            self._remove_edge(reverse_edge)
            undirected_edge = Edge(
                from_point=min(edge.from_point, edge.to_point),
                to_point=max(edge.from_point, edge.to_point),
                value=edge.value,
                directed=False,
            )
            self._append_edge(undirected_edge)
            return True

        if self._find_directed_edge(edge.from_point, edge.to_point, edge.value) is not None:
            return False

        self._append_edge(edge)
        return True

    def add_points(self, point: Point) -> bool:
        self.points.append(point)
        self.num_points = len(self.points)
        return True

    def get_point(self, point_id: int) -> Point | None:
        return self._get_point(point_id)

    def has_point(self, point_id: int) -> bool:
        return self._get_point(point_id) is not None

    def get_edge(
        self,
        from_point: int,
        to_point: int,
        value: float | None = None,
        directed: bool | None = None,
    ) -> Edge | None:
        if directed is True:
            return self._find_directed_edge(from_point, to_point, value)

        if directed is False:
            return self._find_undirected_edge(from_point, to_point, value)

        directed_edge = self._find_directed_edge(from_point, to_point, value)
        if directed_edge is not None:
            return directed_edge

        return self._find_undirected_edge(from_point, to_point, value)

    def has_edge(
        self,
        from_point: int,
        to_point: int,
        value: float | None = None,
        directed: bool | None = None,
    ) -> bool:
        return self.get_edge(from_point, to_point, value, directed) is not None

    def get_edges_between(
        self,
        point_a: int,
        point_b: int,
        directed: bool | None = None,
    ) -> List[Edge]:
        result: List[Edge] = []

        if directed is not False:
            result.extend(self._directed_pair_edges.get((point_a, point_b), []))

        if directed is not True:
            pair_key = self._normalize_pair(point_a, point_b)
            result.extend(self._undirected_pair_edges.get(pair_key, []))

        return list(result)

    def get_out_edges(self, point_id: int) -> List[Edge]:
        point = self._get_point(point_id)
        if point is None:
            return []
        return list(point.outEdges)

    def get_in_edges(self, point_id: int) -> List[Edge]:
        point = self._get_point(point_id)
        if point is None:
            return []
        return list(point.inEdges)

    def get_neighbors(self, point_id: int) -> List[int]:
        neighbors: List[int] = []

        for edge in self.get_out_edges(point_id):
            if edge.from_point == point_id:
                neighbors.append(edge.to_point)
            else:
                neighbors.append(edge.from_point)

        return neighbors

    def get_degree(self, point_id: int) -> int:
        return len(self.get_neighbors(point_id))

    def get_in_degree(self, point_id: int) -> int:
        return len(self.get_in_edges(point_id))

    def get_out_degree(self, point_id: int) -> int:
        return len(self.get_out_edges(point_id))

    def get_directed_edges(self) -> List[Edge]:
        return list(self._directed_edges.values())

    def get_undirected_edges(self) -> List[Edge]:
        return list(self._undirected_edges.values())

    def get_summary(self) -> dict[str, int | str]:
        return {
            "info": self.info,
            "num_points": self.num_points,
            "num_directed_edges": self.num_directed_edges,
            "num_undirected_edges": self.num_undirected_edges,
            "num_total_edges": self.num_total_edges,
        }

    def _append_edge(self, edge: Edge) -> None:
        edge_key = self._edge_key(edge)
        self._edge_indexes[edge_key] = len(self.edges)
        self.edges.append(edge)

        if edge.directed:
            self._directed_edges[self._directed_key(edge.from_point, edge.to_point, edge.value)] = edge
            self._directed_pair_edges.setdefault(
                self._directed_pair_key(edge.from_point, edge.to_point), []
            ).append(edge)
            self.num_directed_edges += 1
        else:
            self._undirected_edges[self._undirected_key(edge.from_point, edge.to_point, edge.value)] = edge
            self._undirected_pair_edges.setdefault(
                self._undirected_pair_key(edge.from_point, edge.to_point), []
            ).append(edge)
            self.num_undirected_edges += 1
        self.num_total_edges += 1
        self._link_edge_to_points(edge)

    def _remove_edge(self, edge: Edge) -> None:
        edge_key = self._edge_key(edge)
        edge_index = self._edge_indexes.pop(edge_key)
        last_edge = self.edges.pop()

        if edge_index < len(self.edges):
            self.edges[edge_index] = last_edge
            self._edge_indexes[self._edge_key(last_edge)] = edge_index

        if edge.directed:
            self._directed_edges.pop(
                self._directed_key(edge.from_point, edge.to_point, edge.value), None
            )
            self._remove_from_pair_index(
                self._directed_pair_edges,
                self._directed_pair_key(edge.from_point, edge.to_point),
                edge,
            )
            self.num_directed_edges -= 1
        else:
            self._undirected_edges.pop(
                self._undirected_key(edge.from_point, edge.to_point, edge.value), None
            )
            self._remove_from_pair_index(
                self._undirected_pair_edges,
                self._undirected_pair_key(edge.from_point, edge.to_point),
                edge,
            )
            self.num_undirected_edges -= 1
        self.num_total_edges -= 1
        self._unlink_edge_from_points(edge)

    def _find_directed_edge(
        self, from_point: int, to_point: int, value: float | None = None
    ) -> Edge | None:
        if value is not None:
            return self._directed_edges.get(self._directed_key(from_point, to_point, value))

        edges = self._directed_pair_edges.get(self._directed_pair_key(from_point, to_point), [])
        return edges[0] if edges else None

    def _find_undirected_edge(
        self, point_a: int, point_b: int, value: float | None = None
    ) -> Edge | None:
        if value is not None:
            return self._undirected_edges.get(self._undirected_key(point_a, point_b, value))

        edges = self._undirected_pair_edges.get(self._undirected_pair_key(point_a, point_b), [])
        return edges[0] if edges else None

    def _link_edge_to_points(self, edge: Edge) -> None:
        from_point = self._get_point(edge.from_point)
        to_point = self._get_point(edge.to_point)

        if from_point is None or to_point is None:
            return

        from_point.outEdges.append(edge)
        to_point.inEdges.append(edge)

        if not edge.directed:
            to_point.outEdges.append(edge)
            from_point.inEdges.append(edge)

    def _unlink_edge_from_points(self, edge: Edge) -> None:
        from_point = self._get_point(edge.from_point)
        to_point = self._get_point(edge.to_point)

        if from_point is not None:
            self._remove_from_list(from_point.outEdges, edge)
            if not edge.directed:
                self._remove_from_list(from_point.inEdges, edge)

        if to_point is not None:
            self._remove_from_list(to_point.inEdges, edge)
            if not edge.directed:
                self._remove_from_list(to_point.outEdges, edge)

    def _get_point(self, point_id: int) -> Point | None:
        index = point_id - 1
        if index < 0 or index >= len(self.points):
            return None
        return self.points[index]

    @staticmethod
    def _directed_key(from_point: int, to_point: int, value: float) -> tuple[int, int, float]:
        return (from_point, to_point, value)

    @staticmethod
    def _undirected_key(point_a: int, point_b: int, value: float) -> tuple[int, int, float]:
        left, right = Graph._normalize_pair(point_a, point_b)
        return (left, right, value)

    @staticmethod
    def _directed_pair_key(from_point: int, to_point: int) -> tuple[int, int]:
        return (from_point, to_point)

    @staticmethod
    def _undirected_pair_key(point_a: int, point_b: int) -> tuple[int, int]:
        return Graph._normalize_pair(point_a, point_b)

    @staticmethod
    def _edge_key(edge: Edge) -> tuple[bool, int, int, float]:
        if edge.directed:
            return (True, edge.from_point, edge.to_point, edge.value)

        left, right = Graph._normalize_pair(edge.from_point, edge.to_point)
        return (False, left, right, edge.value)

    @staticmethod
    def _normalize_pair(point_a: int, point_b: int) -> tuple[int, int]:
        return (min(point_a, point_b), max(point_a, point_b))

    @staticmethod
    def _remove_from_list(edges: List[Edge], target: Edge) -> None:
        try:
            edges.remove(target)
        except ValueError:
            pass

    @staticmethod
    def _remove_from_pair_index(
        pair_index: Dict[tuple[int, int], List[Edge]],
        pair_key: tuple[int, int],
        edge: Edge,
    ) -> None:
        edges = pair_index.get(pair_key)
        if edges is None:
            return

        Graph._remove_from_list(edges, edge)
        if not edges:
            pair_index.pop(pair_key, None)

