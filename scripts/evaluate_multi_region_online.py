"""Run resumable W4 paired online evaluation for strict-disjoint region sets."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import statistics
import sys
import time
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.verify_materialized_queries import evaluate_paired
from src.compression_index import build_compression_index
from src.demand_field_data import load_demand_field_dataset
from src.graph_io import load_porto_graph
from src.region_candidates import load_candidate_manifest
from src.region_selection import select_region_indices, selection_overlap_statistics
from src.workloads import load_porto_queries


DEFAULT_NODE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路节点.csv"
DEFAULT_EDGE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路边.csv"
DEFAULT_QUERY_CSV = (
    ROOT_DIR / "data" / "processed" / "porto" / "波尔图可用起终点节点查询_200米.csv"
)
DEFAULT_CANDIDATES = ROOT_DIR / "results" / "gnn_v2" / "candidate_manifest.json"
DEFAULT_DATASET = ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.npz"
DEFAULT_DATASET_MANIFEST = ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.json"
DEFAULT_CURRENT_LABELS = ROOT_DIR / "results" / "gnn_v2" / "region_training_labels.csv"
DEFAULT_CURRENT_MANIFEST = ROOT_DIR / "results" / "gnn_v2" / "label_manifest.json"
DEFAULT_FUTURE_LABELS = (
    ROOT_DIR / "results" / "gnn_v2" / "future_window_z0" / "region_future_labels.csv"
)
DEFAULT_FUTURE_MANIFEST = (
    ROOT_DIR / "results" / "gnn_v2" / "future_window_z0" / "future_label_manifest.json"
)
DEFAULT_Z0_PREDICTIONS = (
    ROOT_DIR
    / "results"
    / "gnn_v2"
    / "nbfnet_propagation"
    / "train_free_baselines"
    / "predictions.csv"
)
DEFAULT_PROXY_PREDICTIONS = (
    ROOT_DIR / "results" / "gnn_v2" / "proxy_overlap_group_split" / "predictions.csv"
)
DEFAULT_G3_PREDICTIONS = {
    f"g3_seed{seed}": ROOT_DIR
    / "results"
    / "gnn_v2"
    / "nbfnet_propagation"
    / "g3_full_predictions"
    / f"predictions_seed_{seed}.csv"
    for seed in (42, 43, 44)
}
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "gnn_v2" / "multi_region_online"
K_VALUES = (5, 10, 18)
METHOD_NAMES = (
    "random_seed42",
    "history_hotspot",
    "midpoint_proxy",
    "z0",
    "g3_seed42",
    "g3_seed43",
    "g3_seed44",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate strict-disjoint multi-region indexes on frozen Y and F queries."
    )
    parser.add_argument("--node-csv", type=Path, default=DEFAULT_NODE_CSV)
    parser.add_argument("--edge-csv", type=Path, default=DEFAULT_EDGE_CSV)
    parser.add_argument("--query-csv", type=Path, default=DEFAULT_QUERY_CSV)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST
    )
    parser.add_argument("--current-labels", type=Path, default=DEFAULT_CURRENT_LABELS)
    parser.add_argument(
        "--current-manifest", type=Path, default=DEFAULT_CURRENT_MANIFEST
    )
    parser.add_argument("--future-labels", type=Path, default=DEFAULT_FUTURE_LABELS)
    parser.add_argument(
        "--future-manifest", type=Path, default=DEFAULT_FUTURE_MANIFEST
    )
    parser.add_argument("--z0-predictions", type=Path, default=DEFAULT_Z0_PREDICTIONS)
    parser.add_argument(
        "--proxy-predictions", type=Path, default=DEFAULT_PROXY_PREDICTIONS
    )
    parser.add_argument(
        "--g3-predictions-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing predictions_seed_42/43/44.csv. "
            "Defaults to the frozen Porto export paths."
        ),
    )
    parser.add_argument(
        "--score",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Append a frozen prediction CSV with a prediction column.",
    )
    parser.add_argument(
        "--extra-only",
        action="store_true",
        help="Evaluate only methods supplied through --score.",
    )
    parser.add_argument(
        "--without-g3",
        action="store_true",
        help="Keep the four non-G3 baselines and omit frozen G3 score files.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=min(40, os.cpu_count() or 1))
    parser.add_argument("--chunk-size", type=int, default=100)
    parser.add_argument(
        "--k",
        action="append",
        type=int,
        default=None,
        help="Evaluate only this region budget; repeat for multiple K values.",
    )
    parser.add_argument("--no-details", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0 or args.chunk_size <= 0:
        raise SystemExit("--workers and --chunk-size must be positive")
    k_values = _resolve_k_values(args.k)
    dataset = load_demand_field_dataset(args.dataset, args.dataset_manifest)
    graph = load_porto_graph(args.node_csv, args.edge_csv)
    candidate_manifest, regions = load_candidate_manifest(args.candidates)
    region_by_id = {region.region_id: region for region in regions}
    ordered_regions = [region_by_id[int(region_id)] for region_id in dataset.region_ids]
    _validate_candidate_alignment(dataset, candidate_manifest, ordered_regions)

    all_queries = load_porto_queries(args.query_csv)
    query_by_id = {query.query_id: query for query in all_queries}
    current_manifest = json.loads(args.current_manifest.read_text(encoding="utf-8"))
    future_manifest = json.loads(args.future_manifest.read_text(encoding="utf-8"))
    windows = {
        "current_y": _query_subset(query_by_id, current_manifest, "current"),
        "future_f": _query_subset(query_by_id, future_manifest, "future"),
    }
    _validate_temporal_windows(dataset, current_manifest, future_manifest)

    current_rows = _load_rows(args.current_labels, dataset.region_ids)
    future_rows = _load_rows(args.future_labels, dataset.region_ids)
    _validate_label_rows(current_rows, future_rows, dataset.region_ids)
    current_labels = np.asarray(
        [float(current_rows[int(region_id)]["avg_workload_gain"]) for region_id in dataset.region_ids]
    )
    future_labels = np.asarray(
        [float(future_rows[int(region_id)]["avg_workload_gain"]) for region_id in dataset.region_ids]
    )
    score_sources, score_paths = _load_score_sources(args, dataset)
    selections = _build_selections(
        dataset=dataset,
        score_sources=score_sources,
        current_labels=current_labels,
        future_labels=future_labels,
        current_rows=current_rows,
        k_values=k_values,
    )
    identity = _identity(args, dataset, score_paths, tuple(score_sources), k_values)
    if args.validate_only:
        print(
            f"validated methods={len(score_sources)} budgets={len(k_values)} "
            f"current_queries={len(windows['current_y'])} "
            f"future_queries={len(windows['future_f'])}",
            flush=True,
        )
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    details_dir = args.output_dir / "details"
    details_dir.mkdir(exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary = _load_or_initialize_summary(summary_path, identity, selections, args)
    _write_selections(args.output_dir / "selections.csv", selections, k_values)

    method_names = tuple(score_sources)
    for method in method_names:
        for k in k_values:
            key = f"{method}.k{k}"
            missing_windows = [
                name for name in windows if name not in summary["runs"].get(key, {})
            ]
            if not missing_windows:
                print(f"skip complete {key}", flush=True)
                continue
            selected_region_ids = selections[method][str(k)]["selected_region_ids"]
            selected_regions = [region_by_id[region_id] for region_id in selected_region_ids]
            preprocessing_start = time.perf_counter()
            index = build_compression_index(graph, selected_regions)
            preprocessing_seconds = time.perf_counter() - preprocessing_start
            _validate_index_budget(index, selections[method][str(k)])
            print(
                f"built {key}: regions={index.region_count} internal={index.internal_node_count} "
                f"shortcuts={index.shortcut_count} seconds={preprocessing_seconds:.3f}",
                flush=True,
            )
            for window_name in missing_windows:
                queries = windows[window_name]
                run_name = f"{key}.{window_name}"
                metrics, details = evaluate_paired(
                    graph,
                    index,
                    queries,
                    run_name,
                    args.workers,
                    args.chunk_size,
                    collect_details=not args.no_details,
                    endpoint_cache_capacity=0,
                )
                run = _summarize_run(
                    method=method,
                    k=k,
                    window=window_name,
                    index=index,
                    preprocessing_seconds=preprocessing_seconds,
                    queries=queries,
                    metrics=metrics,
                    selection=selections[method][str(k)],
                )
                summary["runs"].setdefault(key, {})[window_name] = run
                _write_summary(summary_path, summary)
                _write_report(args.output_dir / "report.md", summary)
                if not args.no_details:
                    _write_details_gzip(details_dir / f"{run_name}.csv.gz", details)
                print(
                    f"complete {run_name}: indexed={run['indexed_avg_ms']:.3f}ms "
                    f"expanded={run['indexed_avg_expanded']:.1f} "
                    f"correctness={run['correctness_rate']:.6f}",
                    flush=True,
                )
    _write_summary(summary_path, summary)
    _write_report(args.output_dir / "report.md", summary)
    print(f"summary={summary_path}", flush=True)


def _load_score_sources(args: argparse.Namespace, dataset) -> tuple[dict[str, np.ndarray], dict[str, Path]]:
    extra_paths = _parse_score_paths(args.score)
    if args.extra_only:
        if args.without_g3:
            raise ValueError("--extra-only and --without-g3 cannot be combined")
        if not extra_paths:
            raise ValueError("--extra-only requires at least one --score")
        scores = {
            method: _load_column(path, dataset.region_ids, "prediction")
            for method, path in extra_paths.items()
        }
        return scores, extra_paths
    g3_paths = {} if args.without_g3 else (
        {
            f"g3_seed{seed}": args.g3_predictions_dir
            / f"predictions_seed_{seed}.csv"
            for seed in (42, 43, 44)
        }
        if args.g3_predictions_dir is not None
        else DEFAULT_G3_PREDICTIONS
    )
    score_paths = {
        "midpoint_proxy": args.proxy_predictions,
        "z0": args.z0_predictions,
        **g3_paths,
    }
    scores = {
        "random_seed42": np.random.default_rng(42).standard_normal(len(dataset.region_ids)),
        "history_hotspot": _history_hotspot_scores(dataset),
        "midpoint_proxy": _load_column(
            args.proxy_predictions, dataset.region_ids, "proxy_score"
        ),
        "z0": _load_column(args.z0_predictions, dataset.region_ids, "z0_score"),
    }
    for method, path in g3_paths.items():
        scores[method] = _load_column(path, dataset.region_ids, "prediction")
    expected_base = METHOD_NAMES[:4] if args.without_g3 else METHOD_NAMES
    if tuple(scores) != expected_base:
        raise ValueError("score method order does not match frozen protocol")
    for method, path in extra_paths.items():
        if method in scores:
            raise ValueError(f"duplicate score method: {method}")
        scores[method] = _load_column(path, dataset.region_ids, "prediction")
        score_paths[method] = path
    return scores, score_paths


def _parse_score_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path_text = value.partition("=")
        name = name.strip()
        path_text = path_text.strip()
        if not separator or not re.fullmatch(r"[a-z0-9][a-z0-9_]*", name) or not path_text:
            raise ValueError(f"invalid --score value: {value!r}; expected name=path")
        if name in result:
            raise ValueError(f"duplicate --score method: {name}")
        result[name] = Path(path_text)
    return result


def _resolve_k_values(values: list[int] | None) -> tuple[int, ...]:
    if values is None:
        return K_VALUES
    if any(value <= 0 for value in values):
        raise ValueError("K values must be positive")
    return tuple(dict.fromkeys(values))


def _history_hotspot_scores(dataset) -> np.ndarray:
    names = list(dataset.region_feature_names)
    required = (
        "mean_history_origin_count",
        "mean_history_destination_count",
        "max_history_origin_count",
        "max_history_destination_count",
    )
    try:
        indices = [names.index(name) for name in required]
    except ValueError as error:
        raise ValueError(f"missing history hotspot feature: {error.args[0]}") from error
    return dataset.region_features[:, indices].sum(axis=1).astype(np.float64)


def _build_selections(
    *,
    dataset,
    score_sources: dict[str, np.ndarray],
    current_labels: np.ndarray,
    future_labels: np.ndarray,
    current_rows: dict[int, dict[str, str]],
    k_values: tuple[int, ...] = K_VALUES,
) -> dict[str, dict[str, dict[str, object]]]:
    selections: dict[str, dict[str, dict[str, object]]] = {}
    for method, scores in score_sources.items():
        selections[method] = {}
        for k in k_values:
            selected = select_region_indices(
                scores, dataset.region_nodes, k, "hard_disjoint"
            )
            overlap = selection_overlap_statistics(dataset.region_nodes[selected])
            if len(selected) != k or not overlap["deployable_without_region_overlap"]:
                raise ValueError(f"{method} k={k} did not produce a full disjoint set")
            region_ids = [int(dataset.region_ids[index]) for index in selected]
            selections[method][str(k)] = {
                "selected_region_ids": region_ids,
                "overlap": overlap,
                "summed_single_region_shortcut_count": sum(
                    int(current_rows[region_id]["shortcut_count"])
                    for region_id in region_ids
                ),
                "current_mean_single_region_gain": float(np.mean(current_labels[selected])),
                "future_mean_single_region_gain": float(np.mean(future_labels[selected])),
            }
    return selections


def _summarize_run(
    *,
    method: str,
    k: int,
    window: str,
    index,
    preprocessing_seconds: float,
    queries,
    metrics: dict[str, list],
    selection: dict[str, object],
) -> dict[str, object]:
    baseline_avg = _mean(metrics["baseline_elapsed"])
    indexed_avg = _mean(metrics["indexed_elapsed"])
    baseline_p95 = _percentile(metrics["baseline_elapsed"], 95)
    indexed_p95 = _percentile(metrics["indexed_elapsed"], 95)
    baseline_expanded = _mean(metrics["baseline_expanded"])
    indexed_expanded = _mean(metrics["indexed_expanded"])
    endpoint_access_count = sum(
        index.requires_endpoint_access(query.origin, query.destination)
        for query in queries
    )
    return {
        "method": method,
        "k": k,
        "window": window,
        "query_count": len(queries),
        "selected_region_ids": selection["selected_region_ids"],
        "unique_region_node_budget": selection["overlap"]["unique_node_count"],
        "region_count": index.region_count,
        "internal_node_count": index.internal_node_count,
        "boundary_node_count": (
            selection["overlap"]["unique_node_count"] - index.internal_node_count
        ),
        "shortcut_count": index.shortcut_count,
        "compressed_node_count": index.compressed_graph.node_count,
        "compressed_edge_count": index.compressed_graph.edge_count,
        "preprocessing_seconds": preprocessing_seconds,
        "endpoint_access_query_count": endpoint_access_count,
        "endpoint_access_rate_pct": endpoint_access_count / len(queries) * 100.0,
        "baseline_avg_ms": baseline_avg,
        "baseline_p50_ms": _percentile(metrics["baseline_elapsed"], 50),
        "baseline_p95_ms": baseline_p95,
        "indexed_avg_ms": indexed_avg,
        "indexed_p50_ms": _percentile(metrics["indexed_elapsed"], 50),
        "indexed_p95_ms": indexed_p95,
        "elapsed_change_pct": _change_pct(indexed_avg, baseline_avg),
        "p95_change_pct": _change_pct(indexed_p95, baseline_p95),
        "baseline_avg_expanded": baseline_expanded,
        "indexed_avg_expanded": indexed_expanded,
        "indexed_avg_access_expanded": _mean(metrics["indexed_access_expanded"]),
        "indexed_avg_graph_expanded": _mean(metrics["indexed_graph_expanded"]),
        "expanded_change_pct": _change_pct(indexed_expanded, baseline_expanded),
        "faster_query_rate_pct": (
            sum(delta < 0 for delta in metrics["elapsed_deltas"]) / len(queries) * 100.0
        ),
        "median_delta_ms": statistics.median(metrics["elapsed_deltas"]),
        "correctness_rate": sum(metrics["correct_values"]) / len(queries),
        "endpoint_cache_capacity": 0,
    }


def _validate_index_budget(index, selection: dict[str, object]) -> None:
    expected_count = len(selection["selected_region_ids"])
    if index.region_count != expected_count:
        raise ValueError("materialized region count mismatch")
    if index.shortcut_count != selection["summed_single_region_shortcut_count"]:
        raise ValueError("combined shortcut count differs from single-region sum")


def _identity(
    args: argparse.Namespace,
    dataset,
    score_paths: dict[str, Path],
    method_names: tuple[str, ...],
    k_values: tuple[int, ...],
) -> dict[str, object]:
    paths = {
        "node_csv": args.node_csv,
        "edge_csv": args.edge_csv,
        "query_csv": args.query_csv,
        "candidates": args.candidates,
        "dataset": args.dataset,
        "dataset_manifest": args.dataset_manifest,
        "current_labels": args.current_labels,
        "current_manifest": args.current_manifest,
        "future_labels": args.future_labels,
        "future_manifest": args.future_manifest,
        **score_paths,
    }
    return {
        "dataset_sha256": dataset.manifest["dataset_sha256"],
        "candidate_sha256": dataset.manifest["candidate_sha256"],
        "source_sha256": {name: _sha256(path) for name, path in paths.items()},
        "methods": list(method_names),
        "k_values": list(k_values),
        "selection": "hard_disjoint",
        "query_windows": ["current_y", "future_f"],
        "endpoint_cache_capacity": 0,
        "workers": args.workers,
        "chunk_size": args.chunk_size,
    }


def _load_or_initialize_summary(
    path: Path,
    identity: dict[str, object],
    selections: dict,
    args: argparse.Namespace,
) -> dict:
    if path.exists():
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary.get("identity") != identity:
            raise ValueError("existing W4 summary identity does not match frozen protocol")
        return summary
    summary = {
        "schema": "aic.gnn_v2.multi_region_online.v1",
        "identity": identity,
        "protocol": {
            "primary_budget": (
                "strict-disjoint candidate count K with 512 nodes per candidate; "
                "unique region node budget is exactly K*512"
            ),
            "secondary_budget": (
                "shortcut count, boundary/internal counts, compressed graph size, "
                "and preprocessing time are reported rather than assumed equal"
            ),
            "paired_timing": (
                "baseline and indexed query run consecutively per OD; execution order "
                "alternates by query-id parity"
            ),
            "distance_policy": "every indexed distance must equal original-graph distance",
            "endpoint_cache": "disabled",
            "workers": args.workers,
            "chunk_size": args.chunk_size,
            "future_labels_or_queries_used_for_selection": False,
        },
        "selections": selections,
        "runs": {},
    }
    _write_summary(path, summary)
    return summary


def _validate_candidate_alignment(dataset, manifest: dict, regions: list) -> None:
    if manifest.get("candidate_sha256") != dataset.manifest["candidate_sha256"]:
        raise ValueError("candidate manifest identity mismatch")
    if len(regions) != len(dataset.region_ids):
        raise ValueError("candidate count mismatch")
    for region, region_id, node_indices in zip(
        regions, dataset.region_ids, dataset.region_nodes
    ):
        expected_nodes = {int(dataset.node_ids[index]) for index in node_indices}
        if region.region_id != int(region_id) or set(region.nodes) != expected_nodes:
            raise ValueError(f"candidate alignment mismatch at region {region_id}")


def _query_subset(query_by_id: dict, manifest: dict, name: str) -> list:
    query_ids = list(map(int, manifest.get("query_ids", [])))
    if len(query_ids) != 2000 or len(set(query_ids)) != len(query_ids):
        raise ValueError(f"{name} manifest must contain 2,000 unique query ids")
    try:
        return [query_by_id[query_id] for query_id in query_ids]
    except KeyError as error:
        raise ValueError(f"{name} query missing from source data: {error.args[0]}") from error


def _validate_temporal_windows(dataset, current: dict, future: dict) -> None:
    history_ids = set(map(int, dataset.history_query_ids))
    current_ids = set(map(int, current.get("query_ids", [])))
    future_ids = set(map(int, future.get("query_ids", [])))
    if history_ids & current_ids or history_ids & future_ids or current_ids & future_ids:
        raise ValueError("history, current, and future query windows must be disjoint")
    if current.get("label_end_fraction") != future.get("label_start_fraction"):
        raise ValueError("current and future windows must be contiguous")
    if current.get("status") != "complete" or future.get("status") != "complete":
        raise ValueError("label manifests must be complete")


def _load_rows(path: Path, region_ids: np.ndarray) -> dict[int, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = {int(row["region_id"]): row for row in csv.DictReader(file)}
    if set(rows) != set(map(int, region_ids)):
        raise ValueError(f"region rows do not align: {path}")
    return rows


def _load_column(path: Path, region_ids: np.ndarray, column: str) -> np.ndarray:
    rows = _load_rows(path, region_ids)
    values = np.asarray([float(rows[int(region_id)][column]) for region_id in region_ids])
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite {column} in {path}")
    return values


def _validate_label_rows(
    current_rows: dict[int, dict[str, str]],
    future_rows: dict[int, dict[str, str]],
    region_ids: np.ndarray,
) -> None:
    errors: list[str] = []
    for region_id in map(int, region_ids):
        current = current_rows[region_id]
        future = future_rows[region_id]
        for column in ("node_count", "internal_node_count", "shortcut_count"):
            if int(current[column]) != int(future[column]):
                errors.append(f"region {region_id} changed {column}")
        if int(current["label_query_count"]) != 2000:
            errors.append(f"region {region_id} current query count mismatch")
        if int(future["label_query_count"]) != 2000:
            errors.append(f"region {region_id} future query count mismatch")
        if float(current["correctness_rate"]) != 1.0:
            errors.append(f"region {region_id} current label is inexact")
        if float(future["correctness_rate"]) != 1.0:
            errors.append(f"region {region_id} future label is inexact")
    if errors:
        raise ValueError("invalid label rows: " + "; ".join(errors[:10]))


def _write_selections(
    path: Path, selections: dict, k_values: tuple[int, ...] = K_VALUES
) -> None:
    rows = []
    for method in selections:
        for k in k_values:
            for rank, region_id in enumerate(
                selections[method][str(k)]["selected_region_ids"], start=1
            ):
                rows.append(
                    {
                        "method": method,
                        "k": k,
                        "rank": rank,
                        "region_id": region_id,
                    }
                )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_details_gzip(path: Path, rows: list[dict[str, str]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_summary(path: Path, summary: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _write_report(path: Path, summary: dict) -> None:
    lines = [
        "# 严格非重叠多区域在线配对评测",
        "",
        "主预算为严格零重叠的 K×512 个候选节点；缓存关闭。shortcut、内部节点、预处理成本和"
        "压缩图规模按实际值报告。",
        "",
        "| 方法 | K | 窗口 | shortcuts | 内部节点 | 端点接入率 | 基线均值 | 索引均值 | 耗时变化 | "
        "P95 变化 | 展开变化 | 正确率 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in summary["identity"]["methods"]:
        for k in summary["identity"]["k_values"]:
            key = f"{method}.k{k}"
            for window in ("current_y", "future_f"):
                run = summary["runs"].get(key, {}).get(window)
                if run is None:
                    continue
                lines.append(
                    f"| {method} | {k} | {window} | {run['shortcut_count']} | "
                    f"{run['internal_node_count']} | {run['endpoint_access_rate_pct']:.2f}% | "
                    f"{run['baseline_avg_ms']:.3f} ms | {run['indexed_avg_ms']:.3f} ms | "
                    f"{run['elapsed_change_pct']:.2f}% | {run['p95_change_pct']:.2f}% | "
                    f"{run['expanded_change_pct']:.2f}% | {run['correctness_rate']:.6f} |"
                )
    lines.extend(
        [
            "",
            "计时受共享服务器噪声影响，主要系统结论同时参考展开节点与逐查询配对耗时。未来查询"
            "及未来标签均未参与候选打分或集合选择。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _mean(values: list[float] | list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _change_pct(new_value: float, old_value: float) -> float:
    return (new_value - old_value) / old_value * 100.0 if old_value else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
