"""加载道路图 CSV 文件。"""

from __future__ import annotations

import csv
from pathlib import Path

from .graph_types import GraphSummary, WeightedDiGraph


def load_road_graph(node_csv: Path, edge_csv: Path) -> WeightedDiGraph:
    graph = WeightedDiGraph()

    with node_csv.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            graph.add_node(int(row["node_id"]), float(row["lon"]), float(row["lat"]))

    with edge_csv.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            graph.add_edge(int(row["source"]), int(row["target"]), float(row["length_m"]))

    return graph


def load_porto_graph(node_csv: Path, edge_csv: Path) -> WeightedDiGraph:
    """Backward-compatible alias for older experiment scripts."""

    return load_road_graph(node_csv, edge_csv)


def summarize_graph(graph: WeightedDiGraph) -> GraphSummary:
    out_degrees = [len(neighbors) for neighbors in graph.adjacency.values()]
    weights = [
        weight
        for neighbors in graph.adjacency.values()
        for _, weight in neighbors
    ]
    return GraphSummary(
        node_count=graph.node_count,
        edge_count=graph.edge_count,
        isolated_nodes=sum(1 for degree in out_degrees if degree == 0),
        max_out_degree=max(out_degrees, default=0),
        average_out_degree=(sum(out_degrees) / len(out_degrees)) if out_degrees else 0.0,
        min_weight=min(weights, default=0.0),
        max_weight=max(weights, default=0.0),
    )
