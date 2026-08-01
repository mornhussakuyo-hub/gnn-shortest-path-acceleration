"""Export all-candidate predictions from frozen G3 checkpoints without training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.train_demand_field_nbfnet import (
    PrecisionPolicy,
    _all_split_metrics,
    _attach_fixed_prior,
    _predict_weighted,
    _prepare_tensors,
    _unscale_prediction,
)
from src.demand_field_data import SPLIT_NAMES, load_demand_field_dataset
from src.demand_field_model import regression_metrics
from src.demand_field_nbfnet import BidirectionalNBFNet, NBFNetConfig
from src.demand_field_torch_model import cuda_environment, require_cuda


DEFAULT_DATASET = ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.npz"
DEFAULT_DATASET_MANIFEST = ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.json"
DEFAULT_RUNS = {
    42: ROOT_DIR
    / "results"
    / "gnn_v2"
    / "nbfnet_propagation"
    / "gradient_structure_screening"
    / "runs"
    / "g3"
    / "seed_42"
    / "seed_42",
    43: ROOT_DIR
    / "results"
    / "gnn_v2"
    / "nbfnet_propagation"
    / "gradient_structure_confirmation"
    / "seed_43"
    / "runs"
    / "g3"
    / "seed_43"
    / "seed_43",
    44: ROOT_DIR
    / "results"
    / "gnn_v2"
    / "nbfnet_propagation"
    / "gradient_structure_confirmation"
    / "seed_44"
    / "runs"
    / "g3"
    / "seed_44"
    / "seed_44",
}
DEFAULT_OUTPUT_DIR = (
    ROOT_DIR / "results" / "gnn_v2" / "nbfnet_propagation" / "g3_full_predictions"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference only for frozen G3 seeds 42/43/44 on all candidates."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = require_cuda(args.device)
    dataset = load_demand_field_dataset(args.dataset, args.dataset_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict[str, object]] = {}
    for seed, run_dir in DEFAULT_RUNS.items():
        checkpoint_path = run_dir / "model.pt"
        partial_path = run_dir / "predictions.csv"
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        _validate_checkpoint(checkpoint, dataset, seed)
        config = NBFNetConfig(**checkpoint["config"])
        config.validate()
        tensors, fresh_scalers, _ = _prepare_tensors(
            dataset,
            device,
            config.variant,
            config.randomization_seed,
        )
        _validate_scalers(fresh_scalers, checkpoint["scalers"])
        prior = _attach_fixed_prior(
            tensors,
            "z0",
            config.prototype_batch_size,
        )
        model = BidirectionalNBFNet(
            node_feature_dim=dataset.node_features.shape[1],
            region_feature_dim=dataset.region_features.shape[1],
            edge_type_count=len(dataset.manifest["road_types"]),
            config=config,
        ).to(device)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        model.eval()
        precision = PrecisionPolicy(mode=checkpoint["numerics"]["mode"])
        all_indices = torch.arange(
            len(dataset.region_ids), device=device, dtype=torch.long
        )
        with torch.inference_mode():
            standardized = _predict_weighted(
                model,
                tensors,
                all_indices,
                config.prototype_batch_size,
                precision,
            )
        prediction = _unscale_prediction(standardized, checkpoint["scalers"])
        replay = _validate_partial_predictions(
            partial_path, dataset.region_ids, prediction
        )
        output_path = args.output_dir / f"predictions_seed_{seed}.csv"
        _write_predictions(output_path, dataset, prediction)
        metrics = _all_split_metrics(dataset, prediction)
        runs[str(seed)] = {
            "seed": seed,
            "checkpoint": str(checkpoint_path.relative_to(ROOT_DIR)),
            "checkpoint_sha256": _sha256(checkpoint_path),
            "partial_predictions_sha256": _sha256(partial_path),
            "full_predictions_sha256": _sha256(output_path),
            "config": asdict(config),
            "numerics": checkpoint["numerics"],
            "fixed_prior": prior,
            "partial_prediction_count": 1020,
            "full_prediction_count": len(prediction),
            "partial_replay": replay,
            "metrics": metrics,
        }
        holdout = metrics["holdout"]
        print(
            f"seed={seed} replay_delta={replay['maximum_absolute_delta']:.9g} "
            f"replay_spearman={replay['spearman']:.12f} "
            f"holdout_spearman={holdout['spearman']:.6f} "
            f"holdout_ndcg5={holdout['ranking_at_k']['5']['ndcg']:.6f}",
            flush=True,
        )
        del model, tensors, standardized
        torch.cuda.empty_cache()

    summary = {
        "schema": "aic.gnn_v2.g3_frozen_full_inference.v1",
        "execution": cuda_environment(device),
        "dataset_sha256": dataset.manifest["dataset_sha256"],
        "candidate_sha256": dataset.manifest["candidate_sha256"],
        "protocol": {
            "training_performed": False,
            "checkpoint_selection_changed": False,
            "holdout_used_for_model_or_hyperparameter_selection": False,
            "purpose": (
                "one-time post-freeze all-candidate inference for final W4 deployment evaluation"
            ),
        },
        "runs": runs,
        "aggregate": _aggregate(runs),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        _render_report(summary), encoding="utf-8"
    )
    print(f"summary={args.output_dir / 'summary.json'}", flush=True)


def _validate_checkpoint(checkpoint: dict, dataset, seed: int) -> None:
    if checkpoint.get("schema") != "aic.gnn_v2.od_conditioned_bidirectional_nbfnet.v4":
        raise ValueError(f"seed {seed} checkpoint schema mismatch")
    if checkpoint.get("dataset_sha256") != dataset.manifest["dataset_sha256"]:
        raise ValueError(f"seed {seed} dataset identity mismatch")
    if checkpoint.get("candidate_sha256") != dataset.manifest["candidate_sha256"]:
        raise ValueError(f"seed {seed} candidate identity mismatch")
    config = checkpoint.get("config", {})
    if config.get("propagation_structure") != "g3":
        raise ValueError(f"seed {seed} is not a G3 checkpoint")
    if config.get("learning_rate") != 0.005:
        raise ValueError(f"seed {seed} is not the frozen constant-lr protocol")
    if checkpoint.get("numerics", {}).get("mode") != "fp32":
        raise ValueError(f"seed {seed} is not the frozen FP32 protocol")


def _validate_scalers(fresh: dict, saved: dict) -> None:
    for key in ("region_feature_mean", "region_feature_scale"):
        if not np.array_equal(np.asarray(fresh[key]), np.asarray(saved[key])):
            raise ValueError(f"checkpoint scaler mismatch: {key}")
    for key in ("label_mean", "label_scale"):
        if float(fresh[key]) != float(saved[key]):
            raise ValueError(f"checkpoint scaler mismatch: {key}")


def _validate_partial_predictions(
    path: Path,
    region_ids: np.ndarray,
    full_prediction: np.ndarray,
    tolerance: float = 1.0e-4,
    relative_tolerance: float = 2.0 * np.finfo(np.float32).eps,
) -> dict[str, float | bool]:
    index_by_region = {
        int(region_id): index for index, region_id in enumerate(region_ids)
    }
    saved_values: list[float] = []
    replayed_values: list[float] = []
    split_values: dict[str, tuple[list[float], list[float]]] = {
        "train": ([], []),
        "validation": ([], []),
    }
    with path.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 1020 or {row["split"] for row in rows} != {"train", "validation"}:
        raise ValueError(f"partial prediction scope mismatch: {path}")
    for row in rows:
        index = index_by_region[int(row["region_id"])]
        saved = float(row["prediction"])
        replayed = float(full_prediction[index])
        saved_values.append(saved)
        replayed_values.append(replayed)
        split_saved, split_replayed = split_values[row["split"]]
        split_saved.append(saved)
        split_replayed.append(replayed)
    saved_array = np.asarray(saved_values)
    replayed_array = np.asarray(replayed_values)
    deltas = np.abs(saved_array - replayed_array)
    maximum = float(deltas.max(initial=0.0))
    scales = np.maximum(np.abs(saved_array), np.abs(replayed_array))
    allowed_deltas = tolerance + relative_tolerance * scales
    within_numeric_tolerance = deltas <= allowed_deltas
    spearman = regression_metrics(replayed_array, saved_array)["spearman"]
    full_order_equal = bool(
        np.array_equal(
            np.argsort(-saved_array, kind="stable"),
            np.argsort(-replayed_array, kind="stable"),
        )
    )
    top18_equal = True
    for split_saved, split_replayed in split_values.values():
        saved_top = set(np.argsort(-np.asarray(split_saved), kind="stable")[:18].tolist())
        replayed_top = set(
            np.argsort(-np.asarray(split_replayed), kind="stable")[:18].tolist()
        )
        top18_equal = top18_equal and saved_top == replayed_top
    if not bool(np.all(within_numeric_tolerance)):
        worst_index = int(np.argmax(deltas / np.maximum(allowed_deltas, 1.0e-30)))
        raise ValueError(
            "frozen checkpoint replay mismatch: "
            f"delta {float(deltas[worst_index])} > allowed "
            f"{float(allowed_deltas[worst_index])} at replay row {worst_index}"
        )
    if spearman < 0.999999 or not top18_equal:
        raise ValueError(
            "frozen checkpoint replay changed ranking: "
            f"spearman={spearman}, top18_equal={top18_equal}"
        )
    return {
        "maximum_absolute_delta": maximum,
        "mean_absolute_delta": float(deltas.mean()),
        "p99_absolute_delta": float(np.percentile(deltas, 99)),
        "spearman": spearman,
        "full_score_order_equal": full_order_equal,
        "train_validation_top18_sets_equal": top18_equal,
        "absolute_tolerance": float(tolerance),
        "relative_tolerance": float(relative_tolerance),
    }


def _write_predictions(path: Path, dataset, prediction: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("region_id", "split", "label_avg_workload_gain", "prediction"))
        for region_id, split_id, label, score in zip(
            dataset.region_ids, dataset.split, dataset.labels, prediction
        ):
            writer.writerow(
                (
                    int(region_id),
                    SPLIT_NAMES[int(split_id)],
                    f"{float(label):.9f}",
                    f"{float(score):.9f}",
                )
            )


def _aggregate(runs: dict[str, dict[str, object]]) -> dict[str, object]:
    holdout_spearman = [run["metrics"]["holdout"]["spearman"] for run in runs.values()]
    result: dict[str, object] = {
        "holdout_spearman_mean": statistics.fmean(holdout_spearman),
        "holdout_spearman_std": statistics.pstdev(holdout_spearman),
    }
    for k in (5, 10, 18):
        values = [
            run["metrics"]["holdout"]["ranking_at_k"][str(k)]["ndcg"]
            for run in runs.values()
        ]
        result[f"holdout_ndcg_at_{k}_mean"] = statistics.fmean(values)
        result[f"holdout_ndcg_at_{k}_std"] = statistics.pstdev(values)
    return result


def _render_report(summary: dict) -> str:
    lines = [
        "# G3 冻结 checkpoint 全候选推理",
        "",
        "本次只运行冻结 checkpoint 推理，不训练、不重新选 epoch、不使用 holdout 选择模型或超参数。",
        "",
        "| Seed | 重放最大误差 | Holdout Spearman | NDCG@5 | @10 | @18 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for seed, run in summary["runs"].items():
        holdout = run["metrics"]["holdout"]
        ranking = holdout["ranking_at_k"]
        lines.append(
            f"| {seed} | {run['partial_replay']['maximum_absolute_delta']:.3e} | "
            f"{holdout['spearman']:.6f} | {ranking['5']['ndcg']:.6f} | "
            f"{ranking['10']['ndcg']:.6f} | {ranking['18']['ndcg']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"三种子 holdout Spearman：`{summary['aggregate']['holdout_spearman_mean']:.6f} ± "
            f"{summary['aggregate']['holdout_spearman_std']:.6f}`。",
            "",
        ]
    )
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
