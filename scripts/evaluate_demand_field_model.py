"""复核已保存的第二版无传播区域 MLP。"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.demand_field_data import SPLIT_NAMES, load_demand_field_dataset
from src.demand_field_model import regression_metrics
from src.demand_field_torch_model import TorchCudaMLPRegressor, require_cuda


DEFAULT_DATASET = ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.npz"
DEFAULT_DATASET_MANIFEST = ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.json"
DEFAULT_EXPERIMENT_DIR = ROOT_DIR / "results" / "gnn_v2" / "mlp_baseline"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved GNN-v2 region MLP.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument(
        "--model-weights",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--model-manifest",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--experiment-summary",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIR / "summary.json",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.model_weights is None) != (args.model_manifest is None):
        raise SystemExit("--model-weights and --model-manifest must be provided together")
    if args.model_weights is None:
        summary = json.loads(args.experiment_summary.read_text(encoding="utf-8"))
        selected_dir = args.experiment_summary.parent / f"seed_{summary['selected_seed']}"
        args.model_weights = selected_dir / "model.pt"
        args.model_manifest = selected_dir / "model.json"
    dataset = load_demand_field_dataset(args.dataset, args.dataset_manifest)
    model_manifest = json.loads(args.model_manifest.read_text(encoding="utf-8"))
    if model_manifest.get("dataset_sha256") != dataset.manifest["dataset_sha256"]:
        raise SystemExit("model and dataset digests do not match")
    if model_manifest.get("region_feature_names") != list(
        dataset.region_feature_names
    ):
        raise SystemExit("model and dataset feature schemas do not match")
    device = require_cuda(args.device)
    model = TorchCudaMLPRegressor.load(
        args.model_weights,
        args.model_manifest,
        device,
    )
    prediction = model.predict(dataset.region_features)
    metrics = {
        name: regression_metrics(
            prediction[dataset.split_mask(name)],
            dataset.labels[dataset.split_mask(name)],
        )
        for name in SPLIT_NAMES
    }
    print(
        json.dumps(
            {
                "model_seed": model.seed,
                "best_epoch": model.best_epoch,
                "dataset_sha256": dataset.manifest["dataset_sha256"],
                "metrics": metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
