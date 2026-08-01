"""Build auditable shortcut-cost-aware rerankings from frozen G4 predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.demand_field_data import SPLIT_NAMES, load_demand_field_dataset


DEFAULT_WEIGHTS = (0.1, 0.3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build rank-normalized G4 scores with an explicit shortcut penalty."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--cost-csv", type=Path, required=True)
    parser.add_argument(
        "--score", action="append", required=True, metavar="NAME=PATH"
    )
    parser.add_argument(
        "--cost-weight", action="append", type=float, default=None
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weights = tuple(args.cost_weight or DEFAULT_WEIGHTS)
    if not weights or any(not np.isfinite(value) or value < 0.0 for value in weights):
        raise SystemExit("cost weights must be finite and non-negative")
    score_paths = _parse_named_paths(args.score)
    dataset = load_demand_field_dataset(args.dataset, args.dataset_manifest)
    costs = _load_column(args.cost_csv, dataset.region_ids, "shortcut_count")
    cost_rank = _rank_quality(costs)
    split_names = np.asarray([SPLIT_NAMES[int(value)] for value in dataset.split])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[dict[str, object]] = []
    for source_name, source_path in score_paths.items():
        raw_score = _load_column(source_path, dataset.region_ids, "prediction")
        score_rank = _rank_quality(raw_score)
        for weight in weights:
            method = f"{source_name}_cost_{_weight_token(weight)}"
            adjusted = score_rank - float(weight) * cost_rank
            output_path = args.output_dir / f"{method}.csv"
            _write_predictions(
                output_path,
                dataset.region_ids,
                split_names,
                adjusted,
                raw_score,
                score_rank,
                costs,
                cost_rank,
                weight,
            )
            outputs.append(
                {
                    "method": method,
                    "source": str(source_path),
                    "cost_weight": float(weight),
                    "path": str(output_path),
                    "sha256": _sha256(output_path),
                }
            )

    manifest = {
        "schema": "aic.gnn_v2.cost_aware_g4_scores.v1",
        "dataset_sha256": dataset.manifest["dataset_sha256"],
        "candidate_sha256": dataset.manifest["candidate_sha256"],
        "formula": "rank_percentile(prediction) - lambda * rank_percentile(shortcut_count)",
        "selection_labels_used": False,
        "cost_source_column": "shortcut_count",
        "cost_weights": list(map(float, weights)),
        "inputs": {
            "dataset": _sha256(args.dataset),
            "dataset_manifest": _sha256(args.dataset_manifest),
            "cost_csv": _sha256(args.cost_csv),
            **{f"score_{name}": _sha256(path) for name, path in score_paths.items()},
        },
        "outputs": outputs,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"outputs={len(outputs)} manifest={manifest_path}", flush=True)


def _parse_named_paths(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, path_text = value.partition("=")
        name = name.strip()
        path_text = path_text.strip()
        if not separator or not re.fullmatch(r"[a-z0-9][a-z0-9_]*", name):
            raise ValueError(f"invalid score specification: {value!r}")
        if name in result or not path_text:
            raise ValueError(f"duplicate or empty score specification: {value!r}")
        result[name] = Path(path_text)
    return result


def _load_column(path: Path, region_ids: np.ndarray, column: str) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = {int(row["region_id"]): row for row in csv.DictReader(file)}
    expected = set(map(int, region_ids))
    if set(rows) != expected:
        raise ValueError(f"region ids do not align in {path}")
    values = np.asarray([float(rows[int(index)][column]) for index in region_ids])
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite {column} in {path}")
    return values


def _rank_quality(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("rank input must be a non-empty finite vector")
    order = np.argsort(values, kind="stable")
    quality = np.empty(len(values), dtype=np.float64)
    quality[order] = np.linspace(0.0, 1.0, len(values))
    return quality


def _weight_token(value: float) -> str:
    return f"{value:.3f}".replace(".", "p")


def _write_predictions(
    path: Path,
    region_ids: np.ndarray,
    split_names: np.ndarray,
    adjusted: np.ndarray,
    raw_score: np.ndarray,
    score_rank: np.ndarray,
    costs: np.ndarray,
    cost_rank: np.ndarray,
    weight: float,
) -> None:
    fields = [
        "region_id",
        "split",
        "prediction",
        "raw_prediction",
        "prediction_rank",
        "shortcut_count",
        "shortcut_rank",
        "cost_weight",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for index, region_id in enumerate(region_ids):
            writer.writerow(
                {
                    "region_id": int(region_id),
                    "split": str(split_names[index]),
                    "prediction": f"{adjusted[index]:.12f}",
                    "raw_prediction": f"{raw_score[index]:.12f}",
                    "prediction_rank": f"{score_rank[index]:.12f}",
                    "shortcut_count": int(costs[index]),
                    "shortcut_rank": f"{cost_rank[index]:.12f}",
                    "cost_weight": f"{weight:.12f}",
                }
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
