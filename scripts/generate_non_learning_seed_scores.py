"""为非学习消融基线生成节点种子分数。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.gnn_data import build_gnn_data
from src.graph_io import load_porto_graph
from src.workloads import load_porto_queries, split_queries_chronologically


DEFAULT_NODE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路节点.csv"
DEFAULT_EDGE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路边.csv"
DEFAULT_QUERY_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图可用起终点节点查询_200米.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 risk-only 或 proxy-only 节点分数。")
    parser.add_argument("--node-csv", type=Path, default=DEFAULT_NODE_CSV)
    parser.add_argument("--edge-csv", type=Path, default=DEFAULT_EDGE_CSV)
    parser.add_argument("--query-csv", type=Path, default=DEFAULT_QUERY_CSV)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-source", choices=("risk_only", "proxy_only"), required=True)
    parser.add_argument("--diffusion-steps", type=int, default=3)
    parser.add_argument("--diffusion-restart", type=float, default=0.4)
    parser.add_argument("--endpoint-penalty", type=float, default=2.0)
    parser.add_argument(
        "--target-mode",
        choices=("midpoint", "demand_overlap"),
        default="midpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    graph = load_porto_graph(args.node_csv, args.edge_csv)
    queries = load_porto_queries(args.query_csv)
    train_queries, validation_queries, test_queries = split_queries_chronologically(queries)
    data = build_gnn_data(
        graph,
        train_queries,
        validation_queries,
        test_queries,
        diffusion_steps=args.diffusion_steps,
        diffusion_restart=args.diffusion_restart,
        endpoint_penalty=args.endpoint_penalty,
        target_mode=args.target_mode,
    )
    if args.score_source == "risk_only":
        scores = 1.0 - data.endpoint_risk
    else:
        scores = data.train_target.copy()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_scores(args.output_dir / "node_scores.csv", data, scores)
    summary = {
        "score_source": args.score_source,
        "graph_node_count": graph.node_count,
        "graph_edge_count": graph.edge_count,
        "train_query_count": len(train_queries),
        "validation_query_count": len(validation_queries),
        "test_query_count": len(test_queries),
        "test_correlation": _correlation(scores, data.test_target),
        "diffusion_steps": args.diffusion_steps,
        "diffusion_restart": args.diffusion_restart,
        "endpoint_penalty": args.endpoint_penalty,
        "target_mode": args.target_mode,
        "scoring_seconds": time.perf_counter() - start,
    }
    (args.output_dir / "scoring_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def _write_scores(path: Path, data, scores: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            [
                "node_id",
                "seed_score",
                "endpoint_risk",
                "train_proxy_target",
                "validation_proxy_target",
                "test_proxy_target",
                "selected_region_id",
            ]
        )
        for index, node in enumerate(data.nodes):
            writer.writerow(
                [
                    node,
                    f"{scores[index]:.8f}",
                    f"{data.endpoint_risk[index]:.8f}",
                    f"{data.train_target[index]:.8f}",
                    f"{data.validation_target[index]:.8f}",
                    f"{data.test_target[index]:.8f}",
                    "",
                ]
            )


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    if float(left.std()) <= 1e-12 or float(right.std()) <= 1e-12:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


if __name__ == "__main__":
    main()
