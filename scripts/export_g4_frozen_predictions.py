"""Evaluate validation-frozen G4 checkpoints on holdout and future windows."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.evaluate_z0_orthogonal_ablations import (
    _all_scope_metrics,
    _load_and_validate_future_labels,
)
from scripts.export_g3_full_predictions import (
    _parse_run_dirs,
    _validate_scalers,
)
from scripts.select_g4_validation_peaks import SELECTION_FILES
from scripts.train_demand_field_nbfnet import (
    PrecisionPolicy,
    _all_split_metrics,
    _apply_residual_gate,
    _attach_fixed_prior,
    _predict_weighted,
    _prepare_tensors,
    _unscale_prediction,
)
from src.demand_field_data import SPLIT_NAMES, load_demand_field_dataset
from src.demand_field_nbfnet import BidirectionalNBFNet, NBFNetConfig
from src.demand_field_torch_model import cuda_environment, require_cuda


POLICIES = tuple(SELECTION_FILES)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--dataset-manifest", type=Path, required=True)
    parser.add_argument("--future-labels", type=Path, required=True)
    parser.add_argument("--future-manifest", type=Path, required=True)
    parser.add_argument("--current-manifest", type=Path, required=True)
    parser.add_argument(
        "--run-dir", action="append", required=True, metavar="SEED=PATH"
    )
    parser.add_argument("--seeds", required=True)
    parser.add_argument(
        "--policies",
        default=",".join(POLICIES),
        help="Comma-separated validation-only checkpoint policies.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--summary-name", default="summary.json")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = require_cuda(args.device)
    seeds = _parse_ints(args.seeds, "--seeds")
    policies = tuple(value.strip() for value in args.policies.split(",") if value.strip())
    if not policies or len(set(policies)) != len(policies):
        raise SystemExit("--policies must contain unique values")
    unknown = sorted(set(policies) - set(POLICIES))
    if unknown:
        raise SystemExit(f"unknown policies: {unknown}")
    run_dirs = _parse_run_dirs(args.run_dir)
    if set(run_dirs) != set(seeds):
        raise SystemExit(
            f"--run-dir seeds must match --seeds: {sorted(run_dirs)} != {sorted(seeds)}"
        )

    dataset = load_demand_field_dataset(args.dataset, args.dataset_manifest)
    future_labels, temporal_validation = _load_and_validate_future_labels(
        dataset=dataset,
        future_labels_path=args.future_labels,
        future_manifest_path=args.future_manifest,
        current_manifest_path=args.current_manifest,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict[str, object]] = {}
    for seed in seeds:
        seed_dir = run_dirs[seed]
        selection_path = seed_dir / "validation_peak_selection.json"
        selection = json.loads(selection_path.read_text(encoding="utf-8"))
        _validate_selection(selection, dataset)
        for policy in policies:
            selected = selection["selections"][policy]
            checkpoint_path = seed_dir / SELECTION_FILES[policy]
            if _sha256(checkpoint_path) != selected["checkpoint_sha256"]:
                raise ValueError(f"seed {seed} {policy} checkpoint SHA-256 mismatch")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            _validate_checkpoint(checkpoint, dataset, seed, policy, selected)
            config = NBFNetConfig(**checkpoint["config"])
            config.validate()
            tensors, fresh_scalers, _ = _prepare_tensors(
                dataset, device, config.variant, config.randomization_seed
            )
            _validate_scalers(fresh_scalers, checkpoint["scalers"])
            prior = _attach_fixed_prior(tensors, "z0", config.prototype_batch_size)
            model = BidirectionalNBFNet(
                node_feature_dim=dataset.node_features.shape[1],
                region_feature_dim=dataset.region_features.shape[1],
                edge_type_count=len(dataset.manifest["road_types"]),
                config=config,
            ).to(device)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.eval()
            all_indices = torch.arange(
                len(dataset.region_ids), device=device, dtype=torch.long
            )
            with torch.inference_mode():
                standardized = _predict_weighted(
                    model,
                    tensors,
                    all_indices,
                    config.prototype_batch_size,
                    PrecisionPolicy(mode=checkpoint["numerics"]["mode"]),
                )
                standardized = _apply_residual_gate(
                    standardized,
                    tensors["fixed_prior"][all_indices],
                    float(checkpoint["residual_gate"]["alpha"]),
                )
            prediction = _unscale_prediction(standardized, checkpoint["scalers"])
            current_metrics = _all_split_metrics(dataset, prediction)
            validation_replay = _validate_validation_replay(
                current_metrics["validation"], selected
            )
            future_metrics = _all_scope_metrics(dataset, prediction, future_labels)
            output_path = (
                args.output_dir / f"predictions_{policy}_seed_{seed}.csv"
            )
            _write_predictions(output_path, dataset, future_labels, prediction)
            key = f"{policy}.seed_{seed}"
            runs[key] = {
                "seed": seed,
                "policy": policy,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": selected["checkpoint_sha256"],
                "selection_summary_sha256": _sha256(selection_path),
                "predictions": str(output_path),
                "predictions_sha256": _sha256(output_path),
                "config": asdict(config),
                "numerics": checkpoint["numerics"],
                "fixed_prior": prior,
                "training_objective": checkpoint["training_objective"],
                "soft_rank_temperature": checkpoint["soft_rank_temperature"],
                "residual_gate": checkpoint["residual_gate"],
                "validation_replay": validation_replay,
                "current_metrics": current_metrics,
                "future_metrics": future_metrics,
            }
            holdout = current_metrics["holdout"]
            future = future_metrics["all_candidates"]
            print(
                f"seed={seed} policy={policy} val={validation_replay['spearman']:.6f} "
                f"holdout={holdout['spearman']:.6f} future={future['spearman']:.6f}",
                flush=True,
            )
            del model, tensors, standardized
            torch.cuda.empty_cache()

    summary = {
        "schema": "aic.gnn_v2.g4_frozen_evaluation_shard.v1",
        "execution": cuda_environment(device),
        "dataset_sha256": dataset.manifest["dataset_sha256"],
        "candidate_sha256": dataset.manifest["candidate_sha256"],
        "protocol": {
            "training_performed": False,
            "selection_split": "validation",
            "holdout_or_future_used_for_selection": False,
            "policies": list(policies),
            "purpose": "one-time evaluation of validation-frozen G4 checkpoints",
        },
        "temporal_validation": temporal_validation,
        "source_sha256": {
            "dataset": _sha256(args.dataset),
            "dataset_manifest": _sha256(args.dataset_manifest),
            "future_labels": _sha256(args.future_labels),
            "future_manifest": _sha256(args.future_manifest),
            "current_manifest": _sha256(args.current_manifest),
        },
        "runs": runs,
    }
    summary_path = args.output_dir / args.summary_name
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"summary={summary_path}", flush=True)


def _parse_ints(value: str, option: str) -> tuple[int, ...]:
    try:
        values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise SystemExit(f"{option} must contain comma-separated integers") from error
    if not values or len(set(values)) != len(values):
        raise SystemExit(f"{option} must contain unique integers")
    return values


def _validate_selection(selection: dict, dataset) -> None:
    if selection.get("schema") != "aic.gnn_v2.g4_validation_peak_selection.v1":
        raise ValueError("G4 selection schema mismatch")
    if selection.get("dataset_sha256") != dataset.manifest["dataset_sha256"]:
        raise ValueError("G4 selection dataset identity mismatch")
    if selection.get("candidate_sha256") != dataset.manifest["candidate_sha256"]:
        raise ValueError("G4 selection candidate identity mismatch")
    if selection.get("selection_split") != "validation":
        raise ValueError("G4 selection was not made on validation")
    if selection.get("holdout_used") is not False:
        raise ValueError("G4 selection contains holdout use")
    if set(selection.get("selections", {})) != set(POLICIES):
        raise ValueError("G4 selection policy set mismatch")


def _validate_checkpoint(
    checkpoint: dict, dataset, seed: int, policy: str, selected: dict
) -> None:
    if checkpoint.get("schema") != "aic.gnn_v2.od_conditioned_bidirectional_nbfnet.v4":
        raise ValueError(f"seed {seed} checkpoint schema mismatch")
    if checkpoint.get("dataset_sha256") != dataset.manifest["dataset_sha256"]:
        raise ValueError(f"seed {seed} dataset identity mismatch")
    if checkpoint.get("candidate_sha256") != dataset.manifest["candidate_sha256"]:
        raise ValueError(f"seed {seed} candidate identity mismatch")
    config = checkpoint.get("config", {})
    expected = {
        "propagation_structure": "g3",
        "variant": "propagation_doubling",
        "learning_rate": 0.005,
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"seed {seed} frozen config mismatch: {name}")
    if checkpoint.get("numerics", {}).get("mode") != "fp32":
        raise ValueError(f"seed {seed} is not FP32")
    if checkpoint.get("training_objective") != "soft_spearman":
        raise ValueError(f"seed {seed} objective mismatch")
    if checkpoint.get("soft_rank_temperature") != 0.1:
        raise ValueError(f"seed {seed} soft-rank temperature mismatch")
    if checkpoint.get("fixed_prior") != "z0":
        raise ValueError(f"seed {seed} fixed prior mismatch")
    peak = checkpoint.get("validation_peak_selection", {})
    gate = checkpoint.get("residual_gate", {})
    if peak.get("policy") != policy or peak.get("holdout_used") is not False:
        raise ValueError(f"seed {seed} policy metadata mismatch")
    if gate.get("selection_split") != "validation" or gate.get("selection_policy") != policy:
        raise ValueError(f"seed {seed} residual gate metadata mismatch")
    if int(gate.get("validation_metrics", {}).get("epoch", -1)) != int(selected["epoch"]):
        raise ValueError(f"seed {seed} selected epoch mismatch")
    if float(gate.get("alpha", -1.0)) != float(selected["alpha"]):
        raise ValueError(f"seed {seed} selected alpha mismatch")


def _validate_validation_replay(metrics: dict, selected: dict) -> dict[str, object]:
    checks = {
        "spearman": (float(metrics["spearman"]), float(selected["spearman"]), 1.0e-5),
    }
    for k in (5, 10, 18):
        ranking = metrics["ranking_at_k"][str(k)]
        checks[f"ndcg_at_{k}"] = (
            float(ranking["ndcg"]),
            float(selected["ndcg_at_k"][str(k)]),
            1.0e-5,
        )
        checks[f"top_gain_at_{k}"] = (
            float(ranking["mean_gain"]),
            float(selected["top_gain_at_k"][str(k)]),
            1.0e-4,
        )
    deltas = {name: abs(current - saved) for name, (current, saved, _) in checks.items()}
    failed = [
        name for name, (current, saved, tolerance) in checks.items()
        if abs(current - saved) > tolerance
    ]
    if failed:
        raise ValueError(
            "validation-frozen checkpoint replay mismatch: "
            + ", ".join(f"{name} delta={deltas[name]:.3e}" for name in failed)
        )
    return {
        "passed": True,
        "spearman": float(metrics["spearman"]),
        "maximum_metric_delta": max(deltas.values(), default=0.0),
        "metric_deltas": deltas,
    }


def _write_predictions(path: Path, dataset, future: np.ndarray, prediction: np.ndarray) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ("region_id", "split", "current_label", "future_label", "prediction")
        )
        for region_id, split_id, current, future_value, score in zip(
            dataset.region_ids, dataset.split, dataset.labels, future, prediction
        ):
            writer.writerow(
                (
                    int(region_id),
                    SPLIT_NAMES[int(split_id)],
                    f"{float(current):.9f}",
                    f"{float(future_value):.9f}",
                    f"{float(score):.9f}",
                )
            )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
