"""构建第二版 MLP 与 NBFNet 共用的数据集。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.demand_field_data import build_demand_field_dataset, save_demand_field_dataset


DEFAULT_NODE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路节点.csv"
DEFAULT_EDGE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路边.csv"
DEFAULT_QUERY_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图可用起终点节点查询_200米.csv"
DEFAULT_CANDIDATES = ROOT_DIR / "results" / "gnn_v2" / "candidate_manifest.json"
DEFAULT_LABELS = ROOT_DIR / "results" / "gnn_v2" / "region_training_labels.csv"
DEFAULT_LABEL_MANIFEST = ROOT_DIR / "results" / "gnn_v2" / "label_manifest.json"
DEFAULT_OUTPUT = ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.npz"
DEFAULT_MANIFEST = ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the shared path-free GNN-v2 demand-field dataset."
    )
    parser.add_argument("--node-csv", type=Path, default=DEFAULT_NODE_CSV)
    parser.add_argument("--edge-csv", type=Path, default=DEFAULT_EDGE_CSV)
    parser.add_argument("--query-csv", type=Path, default=DEFAULT_QUERY_CSV)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument("--label-manifest", type=Path, default=DEFAULT_LABEL_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest-output", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--prototype-count", type=int, default=128)
    parser.add_argument("--prototype-seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = build_demand_field_dataset(
        node_csv=args.node_csv,
        edge_csv=args.edge_csv,
        query_csv=args.query_csv,
        candidate_manifest_path=args.candidates,
        label_csv=args.labels,
        label_manifest_path=args.label_manifest,
        split_seed=args.split_seed,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        prototype_count=args.prototype_count,
        prototype_seed=args.prototype_seed,
    )
    manifest = save_demand_field_dataset(
        dataset,
        args.output,
        args.manifest_output,
    )
    print(
        json.dumps(
            {
                "schema": manifest["schema"],
                "dataset_sha256": manifest["dataset_sha256"],
                "graph": manifest["graph"],
                "candidate_count": manifest["candidate_count"],
                "node_feature_count": len(manifest["node_feature_names"]),
                "region_feature_count": len(manifest["region_feature_names"]),
                "split": manifest["split"]["counts"],
                "history_query_count": manifest["history_window"]["query_count"],
                "formal_label_query_count": manifest["formal_label_query_count"],
                "demand_prototypes": manifest["demand_prototypes"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    print(f"dataset={_display_path(args.output)}")
    print(f"manifest={_display_path(args.manifest_output)}")


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
