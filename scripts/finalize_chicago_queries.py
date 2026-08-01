"""Apply the frozen snap/SCC gates and finalize at most 100k Chicago queries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]

import sys

sys.path.insert(0, str(ROOT_DIR))

from src.graph_io import load_road_graph
from src.graph_types import WeightedDiGraph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--node-csv",
        type=Path,
        default=ROOT_DIR / "data/processed/chicago/chicago_road_nodes.csv",
    )
    parser.add_argument(
        "--edge-csv",
        type=Path,
        default=ROOT_DIR / "data/processed/chicago/chicago_road_edges.csv",
    )
    parser.add_argument(
        "--snapped-query-csv",
        type=Path,
        default=ROOT_DIR / "data/processed/chicago/chicago_queries_snapped_120k.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT_DIR / "data/processed/chicago/chicago_queries_100k.csv",
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=ROOT_DIR / "data/processed/chicago/chicago_query_manifest.json",
    )
    parser.add_argument("--target-count", type=int, default=100_000)
    parser.add_argument("--minimum-count", type=int, default=90_000)
    parser.add_argument("--minimum-unique-endpoints", type=int, default=500)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = load_road_graph(args.node_csv, args.edge_csv)
    component_by_node, component_sizes = _strong_components(graph)
    rows, counts = _read_and_filter(args.snapped_query_csv, component_by_node)
    rows.sort(key=_row_order_key)
    if len(rows) < args.minimum_count:
        raise SystemExit(
            f"Chicago quality gate failed: reachable usable queries={len(rows):,} "
            f"< minimum={args.minimum_count:,}"
        )
    if len(rows) > args.target_count:
        indices = _equal_index_sample(len(rows), args.target_count)
        rows = [rows[index] for index in indices]

    unique_origins = {int(row["origin_node"]) for row in rows}
    unique_destinations = {int(row["dest_node"]) for row in rows}
    if min(len(unique_origins), len(unique_destinations)) < args.minimum_unique_endpoints:
        raise SystemExit(
            "Chicago endpoint diversity gate failed: "
            f"origins={len(unique_origins):,}, destinations={len(unique_destinations):,}"
        )

    _write_rows(args.output, rows)
    windows = _window_summary(rows)
    manifest = {
        "schema": "aic.chicago_queries.v1",
        "protocol": {
            "snap_threshold_m": 200,
            "different_endpoint_nodes_required": True,
            "same_strong_component_required": True,
            "target_count": args.target_count,
            "minimum_count": args.minimum_count,
            "minimum_unique_origins_and_destinations": args.minimum_unique_endpoints,
            "selection": "chronological equal-index sample; stable trip-id tie order",
            "history_fraction": [0.0, 0.35],
            "current_label_fraction": [0.35, 0.70],
            "future_fraction": [0.70, 1.0],
        },
        "graph": {
            "node_count": graph.node_count,
            "edge_count": graph.edge_count,
            "strong_component_count": len(component_sizes),
            "largest_strong_component_nodes": max(component_sizes.values(), default=0),
        },
        "filter_counts": dict(sorted(counts.items())),
        "final_query_count": len(rows),
        "unique_origin_nodes": len(unique_origins),
        "unique_destination_nodes": len(unique_destinations),
        "time_range": [int(rows[0]["timestamp"]), int(rows[-1]["timestamp"])],
        "windows": windows,
        "source_sha256": {
            "nodes": _sha256(args.node_csv),
            "edges": _sha256(args.edge_csv),
            "snapped_queries": _sha256(args.snapped_query_csv),
        },
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        f"final_queries={len(rows):,} unique_origins={len(unique_origins):,} "
        f"unique_destinations={len(unique_destinations):,}",
        flush=True,
    )
    print(f"windows={windows}", flush=True)
    print(f"output={args.output}", flush=True)
    print(f"manifest={args.manifest_output}", flush=True)


def _strong_components(
    graph: WeightedDiGraph,
) -> tuple[dict[int, int], Counter[int]]:
    visited: set[int] = set()
    finish_order: list[int] = []
    for root in sorted(graph.adjacency):
        if root in visited:
            continue
        visited.add(root)
        stack: list[tuple[int, bool]] = [(root, False)]
        while stack:
            node, finishing = stack.pop()
            if finishing:
                finish_order.append(node)
                continue
            stack.append((node, True))
            for neighbor, _ in graph.out_neighbors(node):
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append((neighbor, False))

    component_by_node: dict[int, int] = {}
    component_sizes: Counter[int] = Counter()
    for root in reversed(finish_order):
        if root in component_by_node:
            continue
        component_id = len(component_sizes)
        component_by_node[root] = component_id
        stack = [root]
        while stack:
            node = stack.pop()
            component_sizes[component_id] += 1
            for neighbor, _ in graph.in_neighbors(node):
                if neighbor not in component_by_node:
                    component_by_node[neighbor] = component_id
                    stack.append(neighbor)
    return component_by_node, component_sizes


def _read_and_filter(
    path: Path, component_by_node: dict[int, int]
) -> tuple[list[dict[str, str]], Counter[str]]:
    kept: list[dict[str, str]] = []
    counts: Counter[str] = Counter()
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            counts["snapped_input"] += 1
            if row.get("snap_usable", "False").lower() != "true":
                counts["snap_rejected"] += 1
                continue
            origin = int(row["origin_node"])
            destination = int(row["dest_node"])
            if origin == destination:
                counts["same_node_rejected"] += 1
                continue
            if component_by_node.get(origin) != component_by_node.get(destination):
                counts["different_scc_rejected"] += 1
                continue
            kept.append(row)
    counts["quality_kept_before_final_sample"] = len(kept)
    return kept, counts


def _equal_index_sample(size: int, target: int) -> list[int]:
    if not 0 < target <= size:
        raise ValueError("target must be between one and size")
    if target == size:
        return list(range(size))
    return [(index * (size - 1)) // (target - 1) for index in range(target)]


def _row_order_key(row: dict[str, str]) -> tuple[int, str]:
    return int(row["timestamp"]), row["trip_id"]


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for query_id, row in enumerate(rows):
            output = dict(row)
            output["query_id"] = str(query_id)
            writer.writerow(output)


def _window_summary(rows: list[dict[str, str]]) -> dict[str, dict[str, object]]:
    boundaries = {
        "history": (0.0, 0.35),
        "current": (0.35, 0.70),
        "future": (0.70, 1.0),
    }
    summary: dict[str, dict[str, object]] = {}
    for name, (start_fraction, end_fraction) in boundaries.items():
        start = int(len(rows) * start_fraction)
        end = int(len(rows) * end_fraction)
        window = rows[start:end]
        summary[name] = {
            "start_fraction": start_fraction,
            "end_fraction": end_fraction,
            "count": len(window),
            "time_range": [int(window[0]["timestamp"]), int(window[-1]["timestamp"])],
        }
    return summary


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
