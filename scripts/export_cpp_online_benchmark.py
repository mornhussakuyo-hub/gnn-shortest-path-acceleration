"""Export frozen graph, region selections, and query windows for the C++ benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import struct
from pathlib import Path
from typing import BinaryIO, Iterable


MAGIC = b"AICCPP1\0"
FORMAT_VERSION = 1
DEFAULT_METHODS = (
    "random_seed42",
    "history_hotspot",
    "midpoint_proxy",
    "z0",
    "g4_global_seed42",
    "g4_global_seed43",
    "g4_global_seed44",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a frozen two-window online benchmark to a compact binary."
    )
    parser.add_argument("--node-csv", type=Path, required=True)
    parser.add_argument("--edge-csv", type=Path, required=True)
    parser.add_argument("--query-csv", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--current-manifest", type=Path, required=True)
    parser.add_argument("--future-manifest", type=Path, required=True)
    parser.add_argument("--selections", type=Path, action="append", required=True)
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--k", type=int, default=18)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_frozen_selections(
    paths: Iterable[Path],
    *,
    methods: tuple[str, ...],
    k: int,
) -> dict[str, list[int]]:
    selected: dict[str, list[tuple[int, int]]] = {}
    for path in paths:
        with path.open(encoding="utf-8", newline="") as file:
            for row in csv.DictReader(file):
                method = row["method"]
                if method not in methods or int(row["k"]) != k:
                    continue
                selected.setdefault(method, []).append(
                    (int(row["rank"]), int(row["region_id"]))
                )

    output: dict[str, list[int]] = {}
    for method in methods:
        ranked = sorted(selected.get(method, []))
        if len(ranked) != k or [rank for rank, _ in ranked] != list(range(1, k + 1)):
            raise ValueError(f"{method} must contain exactly ranks 1..{k}")
        region_ids = [region_id for _, region_id in ranked]
        if len(set(region_ids)) != k:
            raise ValueError(f"{method} contains duplicate selected regions")
        output[method] = region_ids
    return output


def export_benchmark_input(
    *,
    node_csv: Path,
    edge_csv: Path,
    query_csv: Path,
    candidate_path: Path,
    current_manifest_path: Path,
    future_manifest_path: Path,
    selection_paths: tuple[Path, ...],
    methods: tuple[str, ...],
    k: int,
    output_path: Path,
) -> dict[str, object]:
    if k <= 0:
        raise ValueError("k must be positive")
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("methods must be non-empty and unique")

    node_ids = _load_node_ids(node_csv)
    node_index = {node_id: index for index, node_id in enumerate(node_ids)}
    edges = _load_edges(edge_csv, node_index)
    selections = load_frozen_selections(selection_paths, methods=methods, k=k)
    regions = _load_selected_regions(candidate_path, selections, node_index)
    queries = _load_queries(query_csv, node_index)
    windows = {
        "current_y": _load_query_window(current_manifest_path, queries),
        "future_f": _load_query_window(future_manifest_path, queries),
    }
    if set(query.query_id for query in windows["current_y"]) & set(
        query.query_id for query in windows["future_f"]
    ):
        raise ValueError("current and future query windows overlap")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("wb") as file:
        file.write(MAGIC)
        _write(file, "I", FORMAT_VERSION)
        _write(file, "I", len(node_ids))
        _write(file, "Q", len(edges))
        for node_id in node_ids:
            _write(file, "q", node_id)
        for source, target, weight in edges:
            _write(file, "IId", source, target, weight)

        _write(file, "I", len(regions))
        for region_id in sorted(regions):
            region = regions[region_id]
            _write(file, "III", region_id, len(region["nodes"]), len(region["boundary_nodes"]))
            for node in region["nodes"]:
                _write(file, "I", node)
            for node in region["boundary_nodes"]:
                _write(file, "I", node)

        _write(file, "I", len(methods))
        for method in methods:
            _write_string(file, method)
            _write(file, "I", len(selections[method]))
            for region_id in selections[method]:
                _write(file, "I", region_id)

        _write(file, "I", len(windows))
        for window_name, window_queries in windows.items():
            _write_string(file, window_name)
            _write(file, "I", len(window_queries))
            for query in window_queries:
                _write(file, "qII", query.query_id, query.origin, query.destination)
    temporary.replace(output_path)

    metadata: dict[str, object] = {
        "schema": "aic.cpp_online_benchmark_input.v1",
        "format_version": FORMAT_VERSION,
        "node_count": len(node_ids),
        "edge_count": len(edges),
        "selected_region_pool_count": len(regions),
        "methods": list(methods),
        "k": k,
        "query_windows": {name: len(values) for name, values in windows.items()},
        "input_sha256": _sha256(output_path),
        "source_sha256": {
            "nodes": _sha256(node_csv),
            "edges": _sha256(edge_csv),
            "queries": _sha256(query_csv),
            "candidates": _sha256(candidate_path),
            "current_manifest": _sha256(current_manifest_path),
            "future_manifest": _sha256(future_manifest_path),
            **{
                f"selections_{index}": _sha256(path)
                for index, path in enumerate(selection_paths)
            },
        },
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return metadata


class _Query:
    __slots__ = ("query_id", "origin", "destination")

    def __init__(self, query_id: int, origin: int, destination: int) -> None:
        self.query_id = query_id
        self.origin = origin
        self.destination = destination


def _load_node_ids(path: Path) -> list[int]:
    with path.open(encoding="utf-8", newline="") as file:
        node_ids = [int(row["node_id"]) for row in csv.DictReader(file)]
    if not node_ids or len(set(node_ids)) != len(node_ids):
        raise ValueError("node CSV must contain unique nodes")
    return node_ids


def _load_edges(path: Path, node_index: dict[int, int]) -> list[tuple[int, int, float]]:
    edges = []
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            try:
                source = node_index[int(row["source"])]
                target = node_index[int(row["target"])]
            except KeyError as error:
                raise ValueError(f"edge references unknown node {error.args[0]}") from error
            weight = float(row["length_m"])
            if weight <= 0.0:
                raise ValueError("edge weights must be positive")
            edges.append((source, target, weight))
    if not edges:
        raise ValueError("edge CSV is empty")
    return edges


def _load_selected_regions(
    path: Path,
    selections: dict[str, list[int]],
    node_index: dict[int, int],
) -> dict[int, dict[str, list[int]]]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    candidates = {int(item["region_id"]): item for item in manifest["candidates"]}
    requested = {region_id for values in selections.values() for region_id in values}
    regions: dict[int, dict[str, list[int]]] = {}
    for region_id in requested:
        if region_id not in candidates:
            raise ValueError(f"selection references unknown region {region_id}")
        item = candidates[region_id]
        nodes = sorted(node_index[int(node)] for node in item["nodes"])
        boundaries = sorted(node_index[int(node)] for node in item["boundary_nodes"])
        if len(nodes) != int(item["node_count"]) or not set(boundaries) <= set(nodes):
            raise ValueError(f"invalid candidate region {region_id}")
        regions[region_id] = {"nodes": nodes, "boundary_nodes": boundaries}
    return regions


def _load_queries(path: Path, node_index: dict[int, int]) -> dict[int, _Query]:
    output: dict[int, _Query] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("snap_usable", "True") != "True":
                continue
            query_id = int(row["query_id"])
            if query_id in output:
                raise ValueError(f"duplicate query id {query_id}")
            try:
                origin = node_index[int(row["origin_node"])]
                destination = node_index[int(row["dest_node"])]
            except KeyError as error:
                raise ValueError(f"query references unknown node {error.args[0]}") from error
            output[query_id] = _Query(query_id, origin, destination)
    return output


def _load_query_window(path: Path, queries: dict[int, _Query]) -> list[_Query]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete":
        raise ValueError(f"query manifest is incomplete: {path}")
    query_ids = [int(value) for value in manifest.get("query_ids", [])]
    if len(query_ids) != 2_000 or len(set(query_ids)) != len(query_ids):
        raise ValueError(f"query manifest must contain 2,000 unique ids: {path}")
    try:
        return [queries[query_id] for query_id in query_ids]
    except KeyError as error:
        raise ValueError(f"manifest references unknown query {error.args[0]}") from error


def _write(file: BinaryIO, format_string: str, *values: object) -> None:
    file.write(struct.pack("<" + format_string, *values))


def _write_string(file: BinaryIO, value: str) -> None:
    encoded = value.encode("utf-8")
    if len(encoded) > 65_535:
        raise ValueError("binary string is too long")
    _write(file, "H", len(encoded))
    file.write(encoded)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    methods = tuple(args.method) if args.method else DEFAULT_METHODS
    metadata = export_benchmark_input(
        node_csv=args.node_csv,
        edge_csv=args.edge_csv,
        query_csv=args.query_csv,
        candidate_path=args.candidates,
        current_manifest_path=args.current_manifest,
        future_manifest_path=args.future_manifest,
        selection_paths=tuple(args.selections),
        methods=methods,
        k=args.k,
        output_path=args.output,
    )
    print(
        f"exported nodes={metadata['node_count']} edges={metadata['edge_count']} "
        f"methods={len(methods)} output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
