"""生成第二版 GNN 的单区域真实查询收益标签。"""

from __future__ import annotations

import csv
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .compression_index import build_compression_index
from .graph_types import Query, WeightedDiGraph
from .indexed_query import indexed_bidirectional_dijkstra_distance
from .regions import Region
from .shortest_path import ShortestPathResult, bidirectional_dijkstra_distance


LABEL_SCHEMA = "aic.gnn_v2.single_region_labels.v1"
LABEL_WORK_DEFINITION = (
    "baseline.expanded_nodes - "
    "(indexed.graph_search_expanded_nodes + indexed.endpoint_access_expanded_nodes)"
)


@dataclass(frozen=True, slots=True)
class BaselineMetric:
    distance: float
    expanded_nodes: int
    elapsed_ms: float


@dataclass(frozen=True, slots=True)
class RegionLabel:
    region_id: int
    selection_method: str
    seed_node: int
    node_count: int
    boundary_count: int
    internal_node_count: int
    shortcut_count: int
    label_query_count: int
    reachable_query_count: int
    endpoint_access_query_count: int
    endpoint_access_rate_pct: float
    baseline_avg_expanded: float
    indexed_avg_expanded: float
    indexed_avg_graph_expanded: float
    indexed_avg_access_expanded: float
    avg_workload_gain: float
    total_workload_gain: int
    positive_gain_query_rate_pct: float
    baseline_avg_ms_unpaired: float
    indexed_avg_ms_unpaired: float
    indexed_p95_ms_unpaired: float
    preprocessing_seconds: float
    evaluation_seconds: float
    correctness_rate: float


def load_baseline_metrics(
    path: Path,
    *,
    method: str = "bidirectional_dijkstra",
) -> dict[int, BaselineMetric]:
    metrics: dict[int, BaselineMetric] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row["method"] != method:
                continue
            distance = (
                math.inf if row["distance_m"] == "inf" else float(row["distance_m"])
            )
            metrics[int(row["query_id"])] = BaselineMetric(
                distance=distance,
                expanded_nodes=int(row["expanded_nodes"]),
                elapsed_ms=float(row["elapsed_ms"]),
            )
    if not metrics:
        raise ValueError(f"no {method} rows found in {path}")
    return metrics


def compute_baseline_metrics(
    graph: WeightedDiGraph,
    queries: list[Query],
) -> dict[int, BaselineMetric]:
    metrics: dict[int, BaselineMetric] = {}
    for query in queries:
        result = bidirectional_dijkstra_distance(
            graph,
            query.origin,
            query.destination,
        )
        metrics[query.query_id] = BaselineMetric(
            distance=result.distance,
            expanded_nodes=result.expanded_nodes,
            elapsed_ms=result.elapsed_ms,
        )
    return metrics


def evaluate_single_region_label(
    graph: WeightedDiGraph,
    region: Region,
    queries: list[Query],
    baseline_metrics: Mapping[int, BaselineMetric],
    *,
    tolerance: float = 1e-6,
) -> RegionLabel:
    """测量单区域边际收益；端点缓存固定关闭以保证标签与顺序无关。"""

    missing = [query.query_id for query in queries if query.query_id not in baseline_metrics]
    if missing:
        raise ValueError(f"missing baseline metrics for query {missing[0]}")

    preprocessing_start = time.perf_counter()
    index = build_compression_index(graph, [region])
    preprocessing_seconds = time.perf_counter() - preprocessing_start

    evaluation_start = time.perf_counter()
    indexed_expanded: list[int] = []
    indexed_graph_expanded: list[int] = []
    indexed_access_expanded: list[int] = []
    indexed_elapsed: list[float] = []
    baseline_expanded: list[int] = []
    baseline_elapsed: list[float] = []
    gains: list[int] = []
    reachable_count = 0
    correct_count = 0
    endpoint_access_count = 0

    for query in queries:
        baseline = baseline_metrics[query.query_id]
        indexed = indexed_bidirectional_dijkstra_distance(
            graph,
            index,
            query.origin,
            query.destination,
            endpoint_cache=None,
        )
        correct_count += int(_same_distance(baseline.distance, indexed.distance, tolerance))
        reachable_count += int(math.isfinite(baseline.distance))
        endpoint_access_count += int(
            index.requires_endpoint_access(query.origin, query.destination)
        )
        indexed_expanded.append(indexed.expanded_nodes)
        indexed_graph_expanded.append(indexed.graph_search_expanded_nodes)
        indexed_access_expanded.append(indexed.endpoint_access_expanded_nodes)
        indexed_elapsed.append(indexed.elapsed_ms)
        baseline_expanded.append(baseline.expanded_nodes)
        baseline_elapsed.append(baseline.elapsed_ms)
        gains.append(baseline.expanded_nodes - indexed.expanded_nodes)

    query_count = len(queries)
    return RegionLabel(
        region_id=region.region_id,
        selection_method=region.selection_method,
        seed_node=region.seed_node,
        node_count=region.node_count,
        boundary_count=region.boundary_count,
        internal_node_count=index.internal_node_count,
        shortcut_count=index.shortcut_count,
        label_query_count=query_count,
        reachable_query_count=reachable_count,
        endpoint_access_query_count=endpoint_access_count,
        endpoint_access_rate_pct=_rate(endpoint_access_count, query_count),
        baseline_avg_expanded=_mean(baseline_expanded),
        indexed_avg_expanded=_mean(indexed_expanded),
        indexed_avg_graph_expanded=_mean(indexed_graph_expanded),
        indexed_avg_access_expanded=_mean(indexed_access_expanded),
        avg_workload_gain=_mean(gains),
        total_workload_gain=sum(gains),
        positive_gain_query_rate_pct=_rate(sum(gain > 0 for gain in gains), query_count),
        baseline_avg_ms_unpaired=_mean(baseline_elapsed),
        indexed_avg_ms_unpaired=_mean(indexed_elapsed),
        indexed_p95_ms_unpaired=_percentile(indexed_elapsed, 95),
        preprocessing_seconds=preprocessing_seconds,
        evaluation_seconds=time.perf_counter() - evaluation_start,
        correctness_rate=_rate(correct_count, query_count) / 100.0,
    )


def chronological_window(
    queries: list[Query],
    start_fraction: float,
    end_fraction: float,
) -> list[Query]:
    if not 0.0 <= start_fraction < end_fraction <= 1.0:
        raise ValueError("window fractions must satisfy 0 <= start < end <= 1")
    ordered = sorted(
        queries,
        key=lambda query: (
            query.timestamp if query.timestamp is not None else query.query_id,
            query.query_id,
        ),
    )
    start = int(len(ordered) * start_fraction)
    end = int(len(ordered) * end_fraction)
    return ordered[start:end]


def _same_distance(left: float, right: float, tolerance: float) -> bool:
    if math.isinf(left) or math.isinf(right):
        return math.isinf(left) and math.isinf(right)
    return abs(left - right) <= tolerance


def _mean(values: list[int] | list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator * 100.0 if denominator else 0.0
