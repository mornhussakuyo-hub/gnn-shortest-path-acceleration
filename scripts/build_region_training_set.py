"""为固定候选池生成后续时间窗口的单区域真实收益标签。"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, fields
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.graph_io import load_porto_graph
from src.graph_types import Query, WeightedDiGraph
from src.region_candidates import load_candidate_manifest
from src.region_labels import (
    LABEL_SCHEMA,
    LABEL_WORK_DEFINITION,
    BaselineMetric,
    RegionLabel,
    chronological_window,
    compute_baseline_metrics,
    evaluate_single_region_label,
    load_baseline_metrics,
)
from src.regions import Region
from src.workloads import load_porto_queries


DEFAULT_NODE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路节点.csv"
DEFAULT_EDGE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路边.csv"
DEFAULT_QUERY_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图可用起终点节点查询_200米.csv"
DEFAULT_CANDIDATES = ROOT_DIR / "results" / "gnn_v2" / "candidate_manifest.json"
DEFAULT_BASELINE_DETAILS = ROOT_DIR / "results" / "baselines" / "porto_allqueries_details.csv"
DEFAULT_LABELS = ROOT_DIR / "results" / "gnn_v2" / "region_training_labels.csv"
DEFAULT_LABEL_MANIFEST = ROOT_DIR / "results" / "gnn_v2" / "label_manifest.json"

_WORKER_GRAPH: WeightedDiGraph | None = None
_WORKER_QUERIES: list[Query] = []
_WORKER_BASELINES: dict[int, BaselineMetric] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build exact single-region workload-gain labels for GNN v2."
    )
    parser.add_argument("--node-csv", type=Path, default=DEFAULT_NODE_CSV)
    parser.add_argument("--edge-csv", type=Path, default=DEFAULT_EDGE_CSV)
    parser.add_argument("--query-csv", type=Path, default=DEFAULT_QUERY_CSV)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--baseline-details", type=Path, default=DEFAULT_BASELINE_DETAILS)
    parser.add_argument("--output", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_LABEL_MANIFEST)
    parser.add_argument("--label-start-fraction", type=float, default=0.35)
    parser.add_argument("--label-end-fraction", type=float, default=0.70)
    parser.add_argument("--candidate-limit", type=int, default=None)
    parser.add_argument("--query-limit", type=int, default=None)
    parser.add_argument(
        "--candidate-sample",
        type=int,
        default=None,
        help="Fixed random sample from the full candidate pool.",
    )
    parser.add_argument(
        "--query-sample",
        type=int,
        default=None,
        help="Fixed random sample from the full label window.",
    )
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument(
        "--candidate-sample-seed",
        type=int,
        default=None,
        help="Override --sample-seed for candidate sampling only.",
    )
    parser.add_argument(
        "--query-sample-seed",
        type=int,
        default=None,
        help="Override --sample-seed for query sampling only.",
    )
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1) // 2))
    parser.add_argument(
        "--recompute-baseline",
        action="store_true",
        help="Ignore the saved all-query baseline details.",
    )
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Replace existing label progress after validating arguments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.workers <= 0:
        raise SystemExit("--workers must be positive")
    candidate_manifest, regions = load_candidate_manifest(args.candidates)
    candidate_sample_seed = (
        args.sample_seed
        if args.candidate_sample_seed is None
        else args.candidate_sample_seed
    )
    query_sample_seed = (
        args.sample_seed if args.query_sample_seed is None else args.query_sample_seed
    )
    if args.candidate_limit is not None and args.candidate_sample is not None:
        raise SystemExit("--candidate-limit and --candidate-sample are mutually exclusive")
    if args.candidate_limit is not None:
        if args.candidate_limit <= 0:
            raise SystemExit("--candidate-limit must be positive")
        regions = regions[: args.candidate_limit]
    elif args.candidate_sample is not None:
        if not 0 < args.candidate_sample <= len(regions):
            raise SystemExit("--candidate-sample must be between 1 and candidate count")
        regions = sorted(
            random.Random(candidate_sample_seed).sample(regions, args.candidate_sample),
            key=lambda region: region.region_id,
        )

    graph = load_porto_graph(args.node_csv, args.edge_csv)
    all_queries = load_porto_queries(args.query_csv)
    label_queries = chronological_window(
        all_queries,
        args.label_start_fraction,
        args.label_end_fraction,
    )
    if args.query_limit is not None and args.query_sample is not None:
        raise SystemExit("--query-limit and --query-sample are mutually exclusive")
    if args.query_limit is not None:
        if args.query_limit <= 0:
            raise SystemExit("--query-limit must be positive")
        label_queries = label_queries[: args.query_limit]
    elif args.query_sample is not None:
        if not 0 < args.query_sample <= len(label_queries):
            raise SystemExit("--query-sample must be between 1 and label-window query count")
        label_queries = sorted(
            random.Random(query_sample_seed).sample(label_queries, args.query_sample),
            key=lambda query: (
                query.timestamp if query.timestamp is not None else query.query_id,
                query.query_id,
            ),
        )
    if not label_queries:
        raise SystemExit("label query window is empty")

    if args.recompute_baseline or not args.baseline_details.exists():
        print("computing baseline metrics for the selected label window", flush=True)
        baseline_metrics = compute_baseline_metrics(graph, label_queries)
        baseline_source = "recomputed"
    else:
        all_baselines = load_baseline_metrics(args.baseline_details)
        missing = [q.query_id for q in label_queries if q.query_id not in all_baselines]
        if missing:
            raise SystemExit(
                f"saved baseline does not contain label query {missing[0]}; "
                "use --recompute-baseline"
            )
        baseline_metrics = {q.query_id: all_baselines[q.query_id] for q in label_queries}
        baseline_source = str(args.baseline_details.resolve())

    run_identity = {
        "schema": LABEL_SCHEMA,
        "candidate_sha256": candidate_manifest["candidate_sha256"],
        "target_region_ids": [region.region_id for region in regions],
        "label_start_fraction": args.label_start_fraction,
        "label_end_fraction": args.label_end_fraction,
        "query_ids": [query.query_id for query in label_queries],
        "work_definition": LABEL_WORK_DEFINITION,
        "endpoint_cache_capacity": 0,
        "candidate_sample_seed": candidate_sample_seed,
        "query_sample_seed": query_sample_seed,
    }
    completed = _prepare_outputs(args, run_identity)
    pending = [region for region in regions if region.region_id not in completed]
    print(
        f"label generation: candidates={len(regions):,}, pending={len(pending):,}, "
        f"queries={len(label_queries):,}, workers={args.workers}, "
        f"window={args.label_start_fraction:.2f}-{args.label_end_fraction:.2f}",
        flush=True,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [field.name for field in fields(RegionLabel)]
    write_header = not args.output.exists() or args.output.stat().st_size == 0
    with args.output.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
            file.flush()

        if args.workers == 1:
            _init_worker(graph, label_queries, baseline_metrics)
            for done, region in enumerate(pending, start=1):
                row = _evaluate_region(region)
                _write_row(writer, file, row)
                completed.add(region.region_id)
                _update_manifest(
                    args.manifest_output,
                    run_identity,
                    baseline_source,
                    completed,
                    len(regions),
                )
                _print_progress(done, len(pending), row)
        else:
            with ProcessPoolExecutor(
                max_workers=args.workers,
                initializer=_init_worker,
                initargs=(graph, label_queries, baseline_metrics),
                mp_context=_process_context(),
            ) as pool:
                futures = {pool.submit(_evaluate_region, region): region for region in pending}
                for done, future in enumerate(as_completed(futures), start=1):
                    row = future.result()
                    _write_row(writer, file, row)
                    completed.add(row.region_id)
                    _update_manifest(
                        args.manifest_output,
                        run_identity,
                        baseline_source,
                        completed,
                        len(regions),
                    )
                    _print_progress(done, len(pending), row)

    _sort_label_rows(args.output)
    _update_manifest(
        args.manifest_output,
        run_identity,
        baseline_source,
        completed,
        len(regions),
    )
    print(f"labels={_display_path(args.output)}")
    print(f"manifest={_display_path(args.manifest_output)}")


def _init_worker(
    graph: WeightedDiGraph,
    queries: list[Query],
    baselines: dict[int, BaselineMetric],
) -> None:
    global _WORKER_GRAPH, _WORKER_QUERIES, _WORKER_BASELINES
    _WORKER_GRAPH = graph
    _WORKER_QUERIES = queries
    _WORKER_BASELINES = baselines


def _evaluate_region(region: Region) -> RegionLabel:
    if _WORKER_GRAPH is None:
        raise RuntimeError("label worker is not initialized")
    return evaluate_single_region_label(
        _WORKER_GRAPH,
        region,
        _WORKER_QUERIES,
        _WORKER_BASELINES,
    )


def _prepare_outputs(args: argparse.Namespace, identity: dict) -> set[int]:
    if args.restart:
        args.output.unlink(missing_ok=True)
        args.manifest_output.unlink(missing_ok=True)
        return set()
    if args.manifest_output.exists():
        previous = json.loads(args.manifest_output.read_text(encoding="utf-8"))
        for key, value in identity.items():
            if previous.get(key) != value:
                raise SystemExit(
                    f"existing label manifest differs at {key}; "
                    "use different output paths or --restart"
                )
    if not args.output.exists():
        return set()
    completed: set[int] = set()
    with args.output.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            completed.add(int(row["region_id"]))
    return completed


def _write_row(writer: csv.DictWriter, file, row: RegionLabel) -> None:
    writer.writerow(
        {
            key: f"{value:.9f}" if isinstance(value, float) else value
            for key, value in asdict(row).items()
        }
    )
    file.flush()


def _update_manifest(
    path: Path,
    identity: dict,
    baseline_source: str,
    completed: set[int],
    target_count: int,
) -> None:
    payload = {
        **identity,
        "baseline_source": baseline_source,
        "completed_region_count": len(completed),
        "target_region_count": target_count,
        "completed_region_ids": sorted(completed),
        "status": "complete" if len(completed) >= target_count else "in_progress",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sort_label_rows(path: Path) -> None:
    with path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames:
        return
    rows.sort(key=lambda row: int(row["region_id"]))
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_progress(done: int, total: int, row: RegionLabel) -> None:
    print(
        f"completed {done:,}/{total:,}: region={row.region_id}, "
        f"gain={row.avg_workload_gain:.3f}, "
        f"correctness={row.correctness_rate:.6f}",
        flush=True,
    )


def _process_context():
    if os.name == "nt":
        return multiprocessing.get_context("spawn")
    try:
        return multiprocessing.get_context("fork")
    except ValueError:
        return multiprocessing.get_context()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
