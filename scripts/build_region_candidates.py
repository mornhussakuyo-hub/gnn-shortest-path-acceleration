"""构建第二版 GNN 使用的固定 512 节点候选区域池。"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.graph_io import load_porto_graph
from src.region_candidates import (
    CandidatePoolConfig,
    build_fixed_candidate_pool,
    write_candidate_manifest,
)


DEFAULT_NODE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路节点.csv"
DEFAULT_EDGE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路边.csv"
DEFAULT_OUTPUT = ROOT_DIR / "results" / "gnn_v2" / "candidate_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the frozen random full-graph GNN-v2 candidate pool."
    )
    parser.add_argument("--node-csv", type=Path, default=DEFAULT_NODE_CSV)
    parser.add_argument("--edge-csv", type=Path, default=DEFAULT_EDGE_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--candidate-count", type=int, default=1_200)
    parser.add_argument("--region-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overlap-threshold", type=float, default=0.80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    graph = load_porto_graph(args.node_csv, args.edge_csv)
    config = CandidatePoolConfig(
        candidate_count=args.candidate_count,
        region_size=args.region_size,
        seed=args.seed,
        overlap_threshold=args.overlap_threshold,
    )
    print(
        f"building candidates: graph={graph.node_count:,} nodes/{graph.edge_count:,} edges, "
        f"candidates={config.candidate_count:,}, size={config.region_size}",
        flush=True,
    )
    regions = build_fixed_candidate_pool(graph, config)
    manifest = write_candidate_manifest(
        args.output,
        regions,
        config,
        graph_node_count=graph.node_count,
        graph_edge_count=graph.edge_count,
        source_files={
            "nodes": _file_sha256(args.node_csv),
            "edges": _file_sha256(args.edge_csv),
        },
    )
    print(f"candidate_sha256={manifest['candidate_sha256']}")
    print(f"selection_method_counts={manifest['selection_method_counts']}")
    print(f"output={_display_path(args.output)}")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
