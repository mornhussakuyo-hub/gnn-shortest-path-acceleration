"""在同一工作进程内配对比较端点接入缓存开关。"""

from __future__ import annotations

import argparse
import csv
import math
import multiprocessing
import os
import statistics
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.compression_index import CompressionIndex, build_compression_index
from src.graph_io import load_porto_graph
from src.indexed_query import EndpointAccessCache, indexed_bidirectional_dijkstra_distance
from src.regions import build_hotspot_regions, build_random_regions
from src.workloads import load_porto_queries


DEFAULT_NODE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路节点.csv"
DEFAULT_EDGE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路边.csv"
DEFAULT_QUERY_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图可用起终点节点查询_200米.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "endpoint_access" / "cache_comparison"

_WORKER_GRAPH = None
_WORKER_INDEX: CompressionIndex | None = None
_WORKER_CACHE: EndpointAccessCache | None = None


@dataclass(frozen=True, slots=True)
class CacheComparisonRow:
    method: str
    query_count: int
    region_count: int
    region_size: int
    shortcut_count: int
    endpoint_access_query_count: int
    endpoint_access_rate_pct: float
    cache_capacity_per_worker: int
    cache_hits: int
    cache_misses: int
    cache_hit_rate_pct: float
    uncached_avg_ms: float
    cached_avg_ms: float
    elapsed_change_pct: float
    uncached_p95_ms: float
    cached_p95_ms: float
    p95_change_pct: float
    uncached_avg_expanded: float
    cached_avg_expanded: float
    expanded_change_pct: float
    uncached_avg_access_expanded: float
    cached_avg_access_expanded: float
    access_expanded_change_pct: float
    cached_faster_query_rate_pct: float
    median_delta_ms: float
    correctness_rate: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare endpoint access with and without a finite LRU cache."
    )
    parser.add_argument("--node-csv", type=Path, default=DEFAULT_NODE_CSV)
    parser.add_argument("--edge-csv", type=Path, default=DEFAULT_EDGE_CSV)
    parser.add_argument("--query-csv", type=Path, default=DEFAULT_QUERY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--region-count", type=int, default=100)
    parser.add_argument("--region-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--cache-capacity", type=int, default=4096)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=("random", "hotspot"),
        default=["random", "hotspot"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cache_capacity <= 0:
        raise SystemExit("cache capacity must be positive")
    graph = load_porto_graph(args.node_csv, args.edge_csv)
    queries = load_porto_queries(args.query_csv, limit=args.limit)
    strategies = []
    if "random" in args.strategies:
        strategies.append(
            (
                "random_bfs",
                lambda: build_random_regions(
                    graph,
                    args.region_count,
                    args.region_size,
                    args.seed,
                ),
            )
        )
    if "hotspot" in args.strategies:
        strategies.append(
            (
                "od_hotspot_bfs",
                lambda: build_hotspot_regions(
                    graph,
                    queries,
                    args.region_count,
                    args.region_size,
                ),
            )
        )

    rows: list[CacheComparisonRow] = []
    for method, build_regions in strategies:
        print(f"preprocessing {method}", flush=True)
        index = build_compression_index(graph, build_regions())
        endpoint_access_query_count = sum(
            index.requires_endpoint_access(query.origin, query.destination)
            for query in queries
        )
        print(
            f"cache comparison {method}: workers={args.workers}, "
            f"capacity_per_worker={args.cache_capacity}, "
            f"endpoint_access={endpoint_access_query_count:,}/{len(queries):,}",
            flush=True,
        )
        metrics = evaluate_cache_paired(
            graph,
            index,
            queries,
            method,
            workers=args.workers,
            chunk_size=args.chunk_size,
            cache_capacity=args.cache_capacity,
        )
        uncached_average = _mean(metrics["uncached_elapsed"])
        cached_average = _mean(metrics["cached_elapsed"])
        uncached_p95 = _percentile(metrics["uncached_elapsed"], 95)
        cached_p95 = _percentile(metrics["cached_elapsed"], 95)
        uncached_expanded = _mean(metrics["uncached_expanded"])
        cached_expanded = _mean(metrics["cached_expanded"])
        uncached_access_expanded = _mean(metrics["uncached_access_expanded"])
        cached_access_expanded = _mean(metrics["cached_access_expanded"])
        cache_hits = sum(metrics["cache_hits"])
        cache_misses = sum(metrics["cache_misses"])
        row = CacheComparisonRow(
            method=method,
            query_count=len(queries),
            region_count=index.region_count,
            region_size=args.region_size,
            shortcut_count=index.shortcut_count,
            endpoint_access_query_count=endpoint_access_query_count,
            endpoint_access_rate_pct=endpoint_access_query_count / len(queries) * 100.0,
            cache_capacity_per_worker=args.cache_capacity,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cache_hit_rate_pct=(
                cache_hits / (cache_hits + cache_misses) * 100.0
                if cache_hits + cache_misses
                else 0.0
            ),
            uncached_avg_ms=uncached_average,
            cached_avg_ms=cached_average,
            elapsed_change_pct=_change_pct(cached_average, uncached_average),
            uncached_p95_ms=uncached_p95,
            cached_p95_ms=cached_p95,
            p95_change_pct=_change_pct(cached_p95, uncached_p95),
            uncached_avg_expanded=uncached_expanded,
            cached_avg_expanded=cached_expanded,
            expanded_change_pct=_change_pct(cached_expanded, uncached_expanded),
            uncached_avg_access_expanded=uncached_access_expanded,
            cached_avg_access_expanded=cached_access_expanded,
            access_expanded_change_pct=_change_pct(
                cached_access_expanded,
                uncached_access_expanded,
            ),
            cached_faster_query_rate_pct=(
                sum(delta < 0 for delta in metrics["elapsed_deltas"])
                / len(queries)
                * 100.0
            ),
            median_delta_ms=statistics.median(metrics["elapsed_deltas"]),
            correctness_rate=_mean(metrics["correct_values"]),
        )
        rows.append(row)
        print(
            f"{method}: cache_hit={row.cache_hit_rate_pct:.2f}% "
            f"time_change={row.elapsed_change_pct:.2f}% "
            f"access_expanded_change={row.access_expanded_change_pct:.2f}% "
            f"correctness={row.correctness_rate:.6f}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"porto_{len(queries)}queries_r{args.region_count}_s{args.region_size}"
    summary_path = args.output_dir / f"{suffix}_cache_comparison.csv"
    report_path = args.output_dir / f"{suffix}_cache_comparison.md"
    _write_summary(summary_path, rows)
    _write_report(report_path, rows)
    print(f"summary={_display_path(summary_path)}")
    print(f"report={_display_path(report_path)}")


def evaluate_cache_paired(
    graph,
    index: CompressionIndex,
    queries,
    method: str,
    *,
    workers: int,
    chunk_size: int,
    cache_capacity: int,
) -> dict[str, list]:
    chunks = list(_chunked(queries, max(1, chunk_size)))
    if workers <= 1:
        _init_worker(graph, index, cache_capacity)
        results = [_evaluate_chunk(chunk) for chunk in chunks]
    else:
        results = []
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_init_worker,
            initargs=(graph, index, cache_capacity),
            mp_context=_process_context(),
        ) as pool:
            futures = [pool.submit(_evaluate_chunk, chunk) for chunk in chunks]
            for completed, future in enumerate(as_completed(futures), start=1):
                results.append(future.result())
                if completed == len(futures) or completed % 10 == 0:
                    print(
                        f"{method}: completed {completed}/{len(futures)} cache chunks",
                        flush=True,
                    )

    metrics = {
        "uncached_elapsed": [],
        "cached_elapsed": [],
        "elapsed_deltas": [],
        "uncached_expanded": [],
        "cached_expanded": [],
        "uncached_access_expanded": [],
        "cached_access_expanded": [],
        "cache_hits": [],
        "cache_misses": [],
        "correct_values": [],
    }
    for partial in results:
        for key in metrics:
            metrics[key].extend(partial[key])
    return metrics


def _init_worker(
    graph,
    index: CompressionIndex,
    cache_capacity: int,
) -> None:
    global _WORKER_GRAPH, _WORKER_INDEX, _WORKER_CACHE
    _WORKER_GRAPH = graph
    _WORKER_INDEX = index
    _WORKER_CACHE = EndpointAccessCache(cache_capacity)


def _evaluate_chunk(queries) -> dict[str, list]:
    if _WORKER_GRAPH is None or _WORKER_INDEX is None or _WORKER_CACHE is None:
        raise RuntimeError("worker was not initialized")
    output = {
        "uncached_elapsed": [],
        "cached_elapsed": [],
        "elapsed_deltas": [],
        "uncached_expanded": [],
        "cached_expanded": [],
        "uncached_access_expanded": [],
        "cached_access_expanded": [],
        "cache_hits": [],
        "cache_misses": [],
        "correct_values": [],
    }
    for query in queries:
        if query.query_id % 2 == 0:
            cached = indexed_bidirectional_dijkstra_distance(
                _WORKER_GRAPH,
                _WORKER_INDEX,
                query.origin,
                query.destination,
                endpoint_cache=_WORKER_CACHE,
            )
            uncached = indexed_bidirectional_dijkstra_distance(
                _WORKER_GRAPH,
                _WORKER_INDEX,
                query.origin,
                query.destination,
            )
        else:
            uncached = indexed_bidirectional_dijkstra_distance(
                _WORKER_GRAPH,
                _WORKER_INDEX,
                query.origin,
                query.destination,
            )
            cached = indexed_bidirectional_dijkstra_distance(
                _WORKER_GRAPH,
                _WORKER_INDEX,
                query.origin,
                query.destination,
                endpoint_cache=_WORKER_CACHE,
            )
        output["uncached_elapsed"].append(uncached.elapsed_ms)
        output["cached_elapsed"].append(cached.elapsed_ms)
        output["elapsed_deltas"].append(cached.elapsed_ms - uncached.elapsed_ms)
        output["uncached_expanded"].append(uncached.expanded_nodes)
        output["cached_expanded"].append(cached.expanded_nodes)
        output["uncached_access_expanded"].append(
            uncached.endpoint_access_expanded_nodes
        )
        output["cached_access_expanded"].append(cached.endpoint_access_expanded_nodes)
        output["cache_hits"].append(cached.endpoint_cache_hits)
        output["cache_misses"].append(cached.endpoint_cache_misses)
        output["correct_values"].append(
            int(_same_distance(uncached.distance, cached.distance))
        )
    return output


def _write_summary(path: Path, rows: list[CacheComparisonRow]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(CacheComparisonRow.__dataclass_fields__),
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: f"{value:.6f}" if isinstance(value, float) else value
                    for key, value in asdict(row).items()
                }
            )


def _write_report(path: Path, rows: list[CacheComparisonRow]) -> None:
    lines = [
        "# 端点局部接入缓存全量配对报告",
        "",
        "| 方法 | 缓存容量/进程 | 命中率 | 无缓存平均耗时 | 缓存平均耗时 | 缓存耗时变化 | P95 变化 | 接入展开变化 | 缓存查询更快比例 | 正确率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row.method} | {row.cache_capacity_per_worker} | "
            f"{row.cache_hit_rate_pct:.2f}% | {row.uncached_avg_ms:.3f} ms | "
            f"{row.cached_avg_ms:.3f} ms | {row.elapsed_change_pct:.2f}% | "
            f"{row.p95_change_pct:.2f}% | {row.access_expanded_change_pct:.2f}% | "
            f"{row.cached_faster_query_rate_pct:.2f}% | {row.correctness_rate:.6f} |"
        )
    lines.extend(
        [
            "",
            "每条 OD 在同一工作进程内连续运行无缓存和缓存查询，并按查询编号奇偶交替执行顺序。缓存为每个工作进程独立的有限容量 LRU，只保存端点到区域边界的精确距离向量。",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _same_distance(left: float, right: float, tolerance: float = 1e-6) -> bool:
    if math.isinf(left) or math.isinf(right):
        return math.isinf(left) and math.isinf(right)
    return abs(left - right) <= tolerance


def _chunked(values, chunk_size: int):
    for start in range(0, len(values), chunk_size):
        yield values[start : start + chunk_size]


def _process_context():
    if os.name == "posix":
        return multiprocessing.get_context("fork")
    return multiprocessing.get_context()


def _mean(values: list[float] | list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _change_pct(new_value: float, old_value: float) -> float:
    return (new_value - old_value) / old_value * 100.0 if old_value else 0.0


def _display_path(path: Path) -> Path:
    try:
        return path.relative_to(ROOT_DIR)
    except ValueError:
        return path


if __name__ == "__main__":
    main()
