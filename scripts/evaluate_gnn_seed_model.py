"""在测试期 OD 上评测第一版 GNN 选区与传统策略。"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.verify_materialized_queries import evaluate_paired
from src.compression_index import build_compression_index
from src.graph_io import load_porto_graph
from src.regions import (
    build_hotspot_regions,
    build_random_regions,
    build_risk_aware_scored_regions,
)
from src.workloads import load_porto_queries, split_queries_chronologically


DEFAULT_NODE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路节点.csv"
DEFAULT_EDGE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路边.csv"
DEFAULT_QUERY_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图可用起终点节点查询_200米.csv"
DEFAULT_SCORE_CSV = ROOT_DIR / "results" / "gnn_v1" / "node_scores.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "gnn_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate GNN-selected compression regions.")
    parser.add_argument("--node-csv", type=Path, default=DEFAULT_NODE_CSV)
    parser.add_argument("--edge-csv", type=Path, default=DEFAULT_EDGE_CSV)
    parser.add_argument("--query-csv", type=Path, default=DEFAULT_QUERY_CSV)
    parser.add_argument("--score-csv", type=Path, default=DEFAULT_SCORE_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--region-count", type=int, default=100)
    parser.add_argument("--gnn-region-count", type=int, default=85)
    parser.add_argument("--region-size", type=int, default=512)
    parser.add_argument("--seed-exclusion-hops", type=int, default=0)
    parser.add_argument("--candidate-limit", type=int, default=80_000)
    parser.add_argument("--region-endpoint-risk-penalty", type=float, default=200.0)
    parser.add_argument("--random-seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--workers", type=int, default=min(10, os.cpu_count() or 1))
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument(
        "--evaluation-split",
        choices=("validation", "test"),
        default="test",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("gnn", "random", "hotspot"),
        default=["gnn", "random", "hotspot"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = load_porto_graph(args.node_csv, args.edge_csv)
    queries = load_porto_queries(args.query_csv)
    train_queries, validation_queries, test_queries = split_queries_chronologically(queries)
    evaluation_queries = (
        validation_queries if args.evaluation_split == "validation" else test_queries
    )
    node_scores = _load_node_scores(args.score_csv) if "gnn" in args.methods else {}

    strategies = []
    if "gnn" in args.methods:
        strategies.append(
            (
                "gnn_v1_budget_matched",
                lambda: build_risk_aware_scored_regions(
                    graph,
                    node_scores,
                    train_queries,
                    args.gnn_region_count,
                    args.region_size,
                    args.seed_exclusion_hops,
                    args.candidate_limit,
                    args.region_endpoint_risk_penalty,
                ),
            )
        )
    if "random" in args.methods:
        strategies.extend(
            (
                f"random_bfs_seed{seed}",
                lambda seed=seed: build_random_regions(
                    graph,
                    args.region_count,
                    args.region_size,
                    seed,
                ),
            )
            for seed in args.random_seeds
        )
    if "hotspot" in args.methods:
        strategies.append(
            (
                "od_hotspot_bfs",
                lambda: build_hotspot_regions(
                    graph,
                    train_queries,
                    args.region_count,
                    args.region_size,
                ),
            )
        )

    rows: list[dict[str, object]] = []
    selected_region_rows: list[dict[str, object]] = []
    for method, build_regions in strategies:
        print(f"preprocessing {method}", flush=True)
        preprocessing_start = time.perf_counter()
        regions = build_regions()
        index = build_compression_index(graph, regions)
        preprocessing_seconds = time.perf_counter() - preprocessing_start
        fallback_count = sum(
            index.requires_original_graph(query.origin, query.destination)
            for query in evaluation_queries
        )
        evaluation_start = time.perf_counter()
        metrics, _ = evaluate_paired(
            graph,
            index,
            evaluation_queries,
            method,
            args.workers,
            args.chunk_size,
            collect_details=False,
        )
        evaluation_seconds = time.perf_counter() - evaluation_start
        baseline_average = statistics.mean(metrics["baseline_elapsed"])
        indexed_average = statistics.mean(metrics["indexed_elapsed"])
        baseline_p95 = _percentile(metrics["baseline_elapsed"], 95)
        indexed_p95 = _percentile(metrics["indexed_elapsed"], 95)
        baseline_expanded = statistics.mean(metrics["baseline_expanded"])
        indexed_expanded = statistics.mean(metrics["indexed_expanded"])
        row = {
            "method": method,
            "train_query_count": len(train_queries),
            "evaluation_split": args.evaluation_split,
            "evaluation_query_count": len(evaluation_queries),
            "region_count": index.region_count,
            "region_size": args.region_size,
            "shortcut_count": index.shortcut_count,
            "internal_node_count": index.internal_node_count,
            "compressed_node_count": index.compressed_graph.node_count,
            "compressed_edge_count": index.compressed_graph.edge_count,
            "fallback_query_count": fallback_count,
            "fallback_rate_pct": fallback_count / len(evaluation_queries) * 100.0,
            "preprocessing_seconds": preprocessing_seconds,
            "evaluation_wall_seconds": evaluation_seconds,
            "baseline_avg_ms": baseline_average,
            "indexed_avg_ms": indexed_average,
            "elapsed_change_pct": _change_pct(indexed_average, baseline_average),
            "baseline_p95_ms": baseline_p95,
            "indexed_p95_ms": indexed_p95,
            "p95_change_pct": _change_pct(indexed_p95, baseline_p95),
            "baseline_avg_expanded": baseline_expanded,
            "indexed_avg_expanded": indexed_expanded,
            "expanded_change_pct": _change_pct(indexed_expanded, baseline_expanded),
            "faster_query_rate_pct": (
                sum(delta < 0 for delta in metrics["elapsed_deltas"])
                / len(evaluation_queries)
                * 100.0
            ),
            "correctness_rate": statistics.mean(metrics["correct_values"]),
        }
        rows.append(row)
        for region in regions:
            selected_region_rows.append(
                {
                    "method": method,
                    "region_id": region.region_id,
                    "seed_node": region.seed_node,
                    "seed_score": node_scores.get(region.seed_node, ""),
                    "node_count": region.node_count,
                    "boundary_count": region.boundary_count,
                }
            )
        print(
            f"{method}: regions={index.region_count} shortcuts={index.shortcut_count} "
            f"fallback={row['fallback_rate_pct']:.2f}% "
            f"time_change={row['elapsed_change_pct']:.2f}% "
            f"expanded_change={row['expanded_change_pct']:.2f}% "
            f"correctness={row['correctness_rate']:.6f}",
            flush=True,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(args.output_dir / "evaluation_summary.csv", rows)
    _write_rows(args.output_dir / "evaluation_regions.csv", selected_region_rows)
    _write_report(args.output_dir / "evaluation_report.md", rows)


def _load_node_scores(path: Path) -> dict[int, float]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return {
            int(row["node_id"]): float(row["seed_score"])
            for row in csv.DictReader(file)
        }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# 第一版 GNN 种子模型精确配对评测",
        "",
        "| 方法 | 区域数 | Shortcut | 回退率 | 平均耗时变化 | P95 变化 | 展开节点变化 | 正确率 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['region_count']} | {row['shortcut_count']} | "
            f"{row['fallback_rate_pct']:.2f}% | {row['elapsed_change_pct']:.2f}% | "
            f"{row['p95_change_pct']:.2f}% | {row['expanded_change_pct']:.2f}% | "
            f"{row['correctness_rate']:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile / 100.0
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _change_pct(new_value: float, old_value: float) -> float:
    return (new_value / old_value - 1.0) * 100.0 if old_value else 0.0


if __name__ == "__main__":
    main()
