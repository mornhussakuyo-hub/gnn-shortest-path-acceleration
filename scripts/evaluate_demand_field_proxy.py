"""在统一候选 split 上评估第一版 midpoint Proxy。"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.analyze_region_labels import _v1_proxy_scores
from src.demand_field_data import SPLIT_NAMES, load_demand_field_dataset
from src.demand_field_model import ranking_metrics_at_k, regression_metrics
from src.graph_io import load_porto_graph
from src.region_candidates import chronological_prefix, load_candidate_manifest
from src.workloads import load_porto_queries


DEFAULT_NODE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路节点.csv"
DEFAULT_EDGE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路边.csv"
DEFAULT_QUERY_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图可用起终点节点查询_200米.csv"
DEFAULT_CANDIDATES = ROOT_DIR / "results" / "gnn_v2" / "candidate_manifest.json"
DEFAULT_DATASET = ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.npz"
DEFAULT_MANIFEST = ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "gnn_v2" / "proxy_overlap_group_split"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the deterministic midpoint Proxy on fixed dataset splits."
    )
    parser.add_argument("--node-csv", type=Path, default=DEFAULT_NODE_CSV)
    parser.add_argument("--edge-csv", type=Path, default=DEFAULT_EDGE_CSV)
    parser.add_argument("--query-csv", type=Path, default=DEFAULT_QUERY_CSV)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--history-fraction", type=float, default=0.35)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.history_fraction < 1.0:
        raise SystemExit("--history-fraction must be in (0, 1)")

    dataset = load_demand_field_dataset(args.dataset, args.dataset_manifest)
    _, regions = load_candidate_manifest(args.candidates)
    region_by_id = {region.region_id: region for region in regions}
    try:
        ordered_regions = [region_by_id[int(region_id)] for region_id in dataset.region_ids]
    except KeyError as error:
        raise SystemExit(f"dataset region missing from candidate manifest: {error.args[0]}")

    queries = load_porto_queries(args.query_csv)
    history_queries = chronological_prefix(queries, args.history_fraction)
    graph = load_porto_graph(args.node_csv, args.edge_csv)
    score_by_region = _v1_proxy_scores(graph, ordered_regions, history_queries)
    prediction = np.asarray(
        [score_by_region[int(region_id)] for region_id in dataset.region_ids],
        dtype=np.float64,
    )

    metrics = {}
    for split_name in SPLIT_NAMES:
        mask = dataset.split_mask(split_name)
        split_metrics = regression_metrics(prediction[mask], dataset.labels[mask])
        metrics[split_name] = {
            "count": split_metrics["count"],
            "spearman": split_metrics["spearman"],
            "ndcg_at_k": split_metrics["ndcg_at_k"],
            "top_k": split_metrics["top_k"],
            "top_k_mean_gain": split_metrics["top_k_mean_gain"],
            "oracle_top_k_mean_gain": split_metrics["oracle_top_k_mean_gain"],
            "all_mean_gain": split_metrics["all_mean_gain"],
            "ranking_at_k": ranking_metrics_at_k(
                prediction[mask],
                dataset.labels[mask],
                (5, 10, 18),
                region_nodes=dataset.region_nodes[mask],
            ),
        }

    summary = {
        "schema": "aic.gnn_v2.midpoint_proxy_evaluation.v1",
        "model": "v1_midpoint_proxy_mean",
        "execution": {"device_type": "cpu", "deterministic": True},
        "dataset_sha256": dataset.manifest["dataset_sha256"],
        "candidate_sha256": dataset.manifest["candidate_sha256"],
        "history_fraction": args.history_fraction,
        "history_query_count": len(history_queries),
        "split": dataset.manifest["split"],
        "metrics": metrics,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_predictions(args.output_dir / "predictions.csv", dataset, prediction)
    (args.output_dir / "report.md").write_text(
        _render_report(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def _write_predictions(path: Path, dataset, prediction: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("region_id", "split", "label", "proxy_score"))
        for region_id, split_id, label, score in zip(
            dataset.region_ids,
            dataset.split,
            dataset.labels,
            prediction,
        ):
            writer.writerow(
                (
                    int(region_id),
                    SPLIT_NAMES[int(split_id)],
                    f"{float(label):.9f}",
                    f"{float(score):.9f}",
                )
            )


def _render_report(summary: dict) -> str:
    holdout = summary["metrics"]["holdout"]
    ranking = holdout["ranking_at_k"]
    return "\n".join(
        [
            "# 空间隔离 split 第一版 midpoint Proxy 基线",
            "",
            f"- 数据集摘要：`{summary['dataset_sha256']}`",
            f"- Holdout Spearman：`{holdout['spearman']:.4f}`",
            "",
            "| K | NDCG | 平均真实收益 | Oracle 收益 | 成员冗余 |",
            "| ---: | ---: | ---: | ---: | ---: |",
            *[
                f"| {k_value} | {ranking[k_value]['ndcg']:.4f} | "
                f"{ranking[k_value]['mean_gain']:.3f} | "
                f"{ranking[k_value]['oracle_mean_gain']:.3f} | "
                f"{ranking[k_value]['membership_redundancy']:.3f} |"
                for k_value in ("5", "10", "18")
            ],
            "",
            "该结果只是在当前 H→Y 内的候选重叠组隔离 holdout 上评估确定性 Proxy，",
            "不代表冻结未来时间或跨城市泛化。",
            "",
        ]
    )


if __name__ == "__main__":
    main()
