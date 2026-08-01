"""Train the base OD-conditioned bidirectional NBFNet on CUDA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as functional


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.demand_field_data import SPLIT_NAMES, DemandFieldDataset, load_demand_field_dataset
from src.demand_field_model import ranking_metrics_at_k, regression_metrics
from src.demand_field_nbfnet import (
    DOUBLING_PROPAGATION_VARIANTS,
    NBFNET_VARIANTS,
    PROPAGATION_ONLY_VARIANTS,
    PROPAGATION_STRUCTURES,
    BidirectionalNBFNet,
    NBFNetConfig,
    build_edge_features,
    build_receiver_normalizers,
    iter_slices,
)
from src.demand_field_torch_model import cuda_environment, require_cuda
from src.region_selection import select_region_indices
from src.train_free_demand_field import (
    deterministic_diffusion_batch_scores,
)


DEFAULT_DATASET = ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.npz"
DEFAULT_DATASET_MANIFEST = ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "gnn_v2" / "nbfnet_base"
DEFAULT_COST_LABELS = ROOT_DIR / "results" / "gnn_v2" / "region_training_labels.csv"
EXPERIMENT_SCHEMA = "aic.gnn_v2.od_conditioned_bidirectional_nbfnet.v4"
PRECISION_MODES = ("fp16", "bf16", "fp32")
TRAINING_OBJECTIVES = (
    "regression_rank",
    "rank_first",
    "soft_spearman",
    "deployment_aware",
)
RANK_ONLY_OBJECTIVES = frozenset(
    {"rank_first", "soft_spearman", "deployment_aware"}
)
FIXED_PRIORS = ("none", "z0")
LR_SCHEDULERS = ("none", "reduce_on_plateau")


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    mode: str = "fp16"
    grad_scaler_init_scale: float = 65536.0

    def validate(self) -> None:
        if self.mode not in PRECISION_MODES:
            raise ValueError(f"precision must be one of {', '.join(PRECISION_MODES)}")
        if self.grad_scaler_init_scale <= 0.0:
            raise ValueError("grad scaler initial scale must be positive")

    @property
    def autocast_enabled(self) -> bool:
        return self.mode in {"fp16", "bf16"}

    @property
    def autocast_dtype(self) -> torch.dtype:
        return torch.float16 if self.mode == "fp16" else torch.bfloat16

    @property
    def grad_scaler_enabled(self) -> bool:
        return self.mode == "fp16"

    def metadata(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "autocast_enabled": self.autocast_enabled,
            "autocast_dtype": (
                str(self.autocast_dtype).removeprefix("torch.")
                if self.autocast_enabled
                else None
            ),
            "grad_scaler_enabled": self.grad_scaler_enabled,
            "grad_scaler_init_scale": self.grad_scaler_init_scale,
        }


@dataclass(frozen=True, slots=True)
class DeploymentObjective:
    shortcut_cost: torch.Tensor
    train_conflict: torch.Tensor
    validation_conflict: torch.Tensor
    top_k: int
    train_top_k: int
    validation_top_k: int
    topk_temperature: float
    topk_gain_weight: float
    budget_weight: float
    conflict_weight: float
    shortcut_budget_ratio: float
    spearman_tolerance: float
    train_soft_budget: float
    validation_soft_budget: float
    train_hard_budget: float
    validation_hard_budget: float
    train_z0_gain: float
    validation_z0_gain: float
    train_z0_spearman: float
    validation_z0_spearman: float
    cost_source: str
    cost_source_sha256: str

    def metadata(self) -> dict[str, object]:
        return {
            "top_k": self.top_k,
            "train_top_k": self.train_top_k,
            "validation_top_k": self.validation_top_k,
            "split_top_k_policy": "round(full_k * split_count / total_count)",
            "topk_temperature": self.topk_temperature,
            "topk_gain_weight": self.topk_gain_weight,
            "budget_weight": self.budget_weight,
            "conflict_weight": self.conflict_weight,
            "shortcut_budget_ratio": self.shortcut_budget_ratio,
            "spearman_tolerance": self.spearman_tolerance,
            "train_soft_budget": self.train_soft_budget,
            "validation_soft_budget": self.validation_soft_budget,
            "train_hard_budget": self.train_hard_budget,
            "validation_hard_budget": self.validation_hard_budget,
            "train_z0_gain": self.train_z0_gain,
            "validation_z0_gain": self.validation_z0_gain,
            "train_z0_spearman": self.train_z0_spearman,
            "validation_z0_spearman": self.validation_z0_spearman,
            "cost_source": self.cost_source,
            "cost_source_sha256": self.cost_source_sha256,
            "selection_data": "train and validation only",
            "future_or_holdout_used_for_selection": False,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the base OD-conditioned bidirectional NBFNet on CUDA."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--demand-scale", type=float, default=1000.0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--rank-weight", type=float, default=0.20)
    parser.add_argument(
        "--training-objective",
        choices=TRAINING_OBJECTIVES,
        default="regression_rank",
        help="regression_rank preserves the legacy Huber-plus-sampled-rank loss.",
    )
    parser.add_argument(
        "--fixed-prior",
        choices=FIXED_PRIORS,
        default="none",
        help="Add a frozen score to the learned residual before every ranking metric.",
    )
    parser.add_argument(
        "--soft-rank-temperature",
        type=float,
        default=0.1,
        help="Temperature after score standardization for soft_spearman ranks.",
    )
    parser.add_argument("--cost-labels", type=Path, default=DEFAULT_COST_LABELS)
    parser.add_argument("--deployment-k", type=int, default=18)
    parser.add_argument("--deployment-topk-temperature", type=float, default=0.15)
    parser.add_argument("--deployment-topk-gain-weight", type=float, default=0.25)
    parser.add_argument("--deployment-budget-weight", type=float, default=1.0)
    parser.add_argument("--deployment-conflict-weight", type=float, default=0.10)
    parser.add_argument("--deployment-shortcut-budget-ratio", type=float, default=1.0)
    parser.add_argument("--deployment-spearman-tolerance", type=float, default=0.005)
    parser.add_argument(
        "--residual-gate-grid",
        default="1",
        help=(
            "Comma-separated validation-only residual multipliers. Include 0 to "
            "permit a conservative fallback to the frozen prior."
        ),
    )
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--prototype-batch-size", type=int, default=1)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument(
        "--head-warmup-steps",
        type=int,
        default=0,
        help="Train only the final residual output layer for the first N steps.",
    )
    parser.add_argument(
        "--withhold-holdout",
        action="store_true",
        help="Do not evaluate or write holdout predictions before protocol freeze.",
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=LR_SCHEDULERS,
        default="none",
        help="Optional validation-Spearman scheduler after head warmup.",
    )
    parser.add_argument("--lr-scheduler-factor", type=float, default=0.3)
    parser.add_argument("--lr-scheduler-patience", type=int, default=3)
    parser.add_argument("--lr-scheduler-threshold", type=float, default=1.0e-4)
    parser.add_argument("--lr-scheduler-min-lr", type=float, default=5.0e-4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--variant", choices=NBFNET_VARIANTS, default="base")
    parser.add_argument(
        "--propagation-structure",
        choices=PROPAGATION_STRUCTURES,
        help=(
            "Orthogonal S0 propagation update. If omitted, legacy propagation "
            "variants retain their historical G0/G1 behavior."
        ),
    )
    parser.add_argument("--propagation-residual-scale", type=float, default=0.01)
    parser.add_argument("--randomization-seed", type=int, default=20260730)
    parser.add_argument(
        "--precision",
        choices=PRECISION_MODES,
        help="CUDA numerical path. Defaults to fp16 for backward compatibility.",
    )
    parser.add_argument("--grad-scaler-init-scale", type=float, default=65536.0)
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="Legacy alias for --precision fp32.",
    )
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    parser.add_argument(
        "--save-validation-snapshots",
        action="store_true",
        help=(
            "Persist every epoch-zero/post-update model state and all validation "
            "gate metrics so later validation-only peak selection cannot lose peaks."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_int_list(args.seeds, "--seeds")
    residual_gate_grid = _parse_float_grid(
        args.residual_gate_grid,
        "--residual-gate-grid",
    )
    precision_policy = _resolve_precision_policy(
        args.precision,
        args.no_mixed_precision,
        args.grad_scaler_init_scale,
    )
    rank_weight = 0.0 if args.variant == "no_ranking" else args.rank_weight
    if args.training_objective in RANK_ONLY_OBJECTIVES and args.variant == "no_ranking":
        raise SystemExit("a ranking objective cannot be combined with variant=no_ranking")
    if args.fixed_prior != "none" and args.training_objective not in RANK_ONLY_OBJECTIVES:
        raise SystemExit(
            "a fixed prior is only supported by a rank-only training objective"
        )
    if args.fixed_prior == "z0" and args.variant not in PROPAGATION_ONLY_VARIANTS:
        raise SystemExit("the Z0 residual protocol requires a propagation-only variant")
    if args.head_warmup_steps < 0:
        raise SystemExit("head-warmup-steps must be non-negative")
    if args.head_warmup_steps and (
        args.training_objective not in RANK_ONLY_OBJECTIVES
        or args.fixed_prior == "none"
    ):
        raise SystemExit(
            "head-warmup-steps requires a rank-only objective with a fixed prior"
        )
    if args.soft_rank_temperature <= 0.0:
        raise SystemExit("--soft-rank-temperature must be positive")
    if args.training_objective == "deployment_aware":
        if args.fixed_prior != "z0":
            raise SystemExit("deployment_aware requires --fixed-prior z0")
        if residual_gate_grid != (1.0,):
            raise SystemExit("deployment_aware currently requires --residual-gate-grid 1")
        if args.deployment_k <= 0:
            raise SystemExit("--deployment-k must be positive")
        if args.deployment_topk_temperature <= 0.0:
            raise SystemExit("--deployment-topk-temperature must be positive")
        if any(
            value < 0.0
            for value in (
                args.deployment_topk_gain_weight,
                args.deployment_budget_weight,
                args.deployment_conflict_weight,
                args.deployment_spearman_tolerance,
            )
        ):
            raise SystemExit("deployment loss weights and tolerance must be non-negative")
        if not 0.0 < args.deployment_shortcut_budget_ratio <= 1.0:
            raise SystemExit("--deployment-shortcut-budget-ratio must be in (0, 1]")
    if residual_gate_grid != (1.0,) and args.fixed_prior == "none":
        raise SystemExit("a residual gate grid requires a fixed prior")
    if args.lr_scheduler != "none":
        if not 0.0 < args.lr_scheduler_factor < 1.0:
            raise SystemExit("lr-scheduler-factor must be in (0, 1)")
        if args.lr_scheduler_patience < 0:
            raise SystemExit("lr-scheduler-patience must be non-negative")
        if args.lr_scheduler_threshold < 0.0:
            raise SystemExit("lr-scheduler-threshold must be non-negative")
        if not 0.0 < args.lr_scheduler_min_lr <= args.learning_rate:
            raise SystemExit("lr-scheduler-min-lr must be in (0, learning-rate]")
    config = NBFNetConfig(
        hidden_dim=args.hidden_dim,
        propagation_layers=args.layers,
        demand_scale=args.demand_scale,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        rank_weight=rank_weight,
        huber_delta=args.huber_delta,
        prototype_batch_size=args.prototype_batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        mixed_precision=precision_policy.autocast_enabled,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        zero_initialize_prediction_head=args.fixed_prior != "none",
        propagation_structure=args.propagation_structure,
        propagation_residual_scale=args.propagation_residual_scale,
        variant=args.variant,
        randomization_seed=args.randomization_seed,
    )
    config.validate()
    device = require_cuda(args.device)
    dataset = load_demand_field_dataset(args.dataset, args.dataset_manifest)
    tensors, scalers, ablation_metadata = _prepare_tensors(
        dataset,
        device,
        config.variant,
        config.randomization_seed,
    )
    prior_metadata = _attach_fixed_prior(
        tensors,
        args.fixed_prior,
        config.prototype_batch_size,
    )
    deployment_objective = (
        _prepare_deployment_objective(
            dataset=dataset,
            tensors=tensors,
            cost_labels=args.cost_labels,
            top_k=args.deployment_k,
            topk_temperature=args.deployment_topk_temperature,
            topk_gain_weight=args.deployment_topk_gain_weight,
            budget_weight=args.deployment_budget_weight,
            conflict_weight=args.deployment_conflict_weight,
            shortcut_budget_ratio=args.deployment_shortcut_budget_ratio,
            spearman_tolerance=args.deployment_spearman_tolerance,
        )
        if args.training_objective == "deployment_aware"
        else None
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []

    for seed in seeds:
        print(
            f"training NBFNet variant={config.variant} seed={seed}",
            flush=True,
        )
        seed_dir = args.output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        run = _train_one_seed(
            dataset,
            tensors,
            scalers,
            config,
            seed,
            device,
            precision_policy,
            args.training_objective,
            args.fixed_prior,
            args.soft_rank_temperature,
            residual_gate_grid,
            seed_dir / "validation_snapshots"
            if args.save_validation_snapshots
            else None,
            args.head_warmup_steps,
            ("train", "validation") if args.withhold_holdout else SPLIT_NAMES,
            args.lr_scheduler,
            args.lr_scheduler_factor,
            args.lr_scheduler_patience,
            args.lr_scheduler_threshold,
            args.lr_scheduler_min_lr,
            deployment_objective,
        )
        _write_history(seed_dir / "training_history.csv", run.pop("history"))
        snapshot_catalog = run.pop("validation_snapshot_catalog")
        if snapshot_catalog is not None:
            (seed_dir / "validation_snapshot_catalog.json").write_text(
                json.dumps(snapshot_catalog, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        _write_predictions(
            seed_dir / "predictions.csv",
            dataset,
            run["prediction"],
        )
        if run.get("model_state") is not None:
            _save_checkpoint(seed_dir / "model.pt", run, config, scalers, dataset)
        run.pop("prediction")
        runs.append(run)
        selected_split = "holdout" if "holdout" in run["metrics"] else "validation"
        selected_metrics = run["metrics"][selected_split]
        print(
            f"seed={seed} best_epoch={run['best_epoch']} "
            f"{selected_split}_spearman={selected_metrics['spearman']:.4f} "
            f"{selected_split}_top_gain={selected_metrics['top_k_mean_gain']:.3f}",
            flush=True,
        )

    selected_seed = max(
        runs,
        key=(
            lambda run: (
                run.get("deployment_validation", {}).get(
                    "mean_gain", float("-inf")
                ),
                -run.get("deployment_validation", {}).get(
                    "shortcut_count", float("inf")
                ),
                run["metrics"]["validation"]["spearman"],
            )
            if args.training_objective == "deployment_aware"
            else (run["metrics"]["validation"]["spearman"], 0.0, 0.0)
        ),
    )["seed"]
    summary = {
        "schema": EXPERIMENT_SCHEMA,
        "model": "od_conditioned_bidirectional_nbfnet",
        "variant": config.variant,
        "execution": cuda_environment(device),
        "dataset_sha256": dataset.manifest["dataset_sha256"],
        "candidate_sha256": dataset.manifest["candidate_sha256"],
        "config": asdict(config),
        "numerics": precision_policy.metadata(),
        "training_objective": args.training_objective,
        "deployment_objective": (
            deployment_objective.metadata()
            if deployment_objective is not None
            else None
        ),
        "training_protocol": {
            "head_warmup_steps": args.head_warmup_steps,
            "soft_rank_temperature": args.soft_rank_temperature,
            "residual_gate_grid": list(residual_gate_grid),
            "save_validation_snapshots": args.save_validation_snapshots,
            "evaluation_splits": (
                ["train", "validation"]
                if args.withhold_holdout
                else list(SPLIT_NAMES)
            ),
            "holdout_withheld": args.withhold_holdout,
            "lr_scheduler": {
                "name": args.lr_scheduler,
                "monitor": (
                    "validation_spearman"
                    if args.lr_scheduler == "reduce_on_plateau"
                    else None
                ),
                "starts_after_head_warmup": True,
                "factor": args.lr_scheduler_factor,
                "patience": args.lr_scheduler_patience,
                "threshold": args.lr_scheduler_threshold,
                "min_lr": args.lr_scheduler_min_lr,
            },
        },
        "fixed_prior": prior_metadata,
        "seeds": seeds,
        "selected_seed": selected_seed,
        "selection_rule": (
            (
                "highest validation hard-disjoint Top-K mean gain among post-update "
                "checkpoints whose shortcut cost is within the Z0 validation budget "
                "and whose global Spearman is within the frozen tolerance; holdout and "
                "future windows are never used"
            )
            if args.training_objective == "deployment_aware"
            else (
                "highest validation Spearman among post-update checkpoints; holdout is "
                "never used"
            )
            if args.training_objective in RANK_ONLY_OBJECTIVES
            else "highest validation Spearman; holdout is never used"
        ),
        "architecture": _architecture_metadata(config),
        "split": dataset.manifest["split"],
        "input_policy": dataset.manifest["model_input_policy"],
        "ablation": ablation_metadata,
        "prototype_batching": {
            "prototype_count": int(tensors["prototype_weight"].size(0)),
            "prototype_batch_size": config.prototype_batch_size,
            "training_objective": (
                "Two-pass exact full-mixture gradient: the first pass evaluates the "
                "weighted score of all frozen prototypes; the second recomputes one "
                "memory chunk at a time and accumulates its exact gradient contribution."
            ),
            "ranking_loss": (
                (
                    "soft Spearman plus differentiable Top-K gain, Z0-relative shortcut "
                    "budget, and overlap-conflict penalties"
                )
                if args.training_objective == "deployment_aware"
                else "scale-invariant differentiable soft Spearman"
                if args.training_objective == "soft_spearman"
                else (
                    "all unique training candidate pairs"
                    if args.training_objective == "rank_first"
                    else "one sampled pair per training candidate"
                )
            ),
            "batch_size_affects_objective": False,
        },
        "runs": runs,
        "aggregate": _aggregate_runs(runs),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        _render_report(summary), encoding="utf-8"
    )
    print(f"summary={_display_path(args.output_dir / 'summary.json')}")


def _prepare_tensors(
    dataset: DemandFieldDataset,
    device: torch.device,
    variant: str = "base",
    randomization_seed: int = 20260730,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, np.ndarray | float],
    dict[str, object],
]:
    train_mask = dataset.split_mask("train")
    feature_mean = dataset.region_features[train_mask].mean(axis=0).astype(np.float32)
    feature_scale = dataset.region_features[train_mask].std(axis=0).astype(np.float32)
    feature_scale[feature_scale < 1e-6] = 1.0
    label_mean = float(dataset.labels[train_mask].mean())
    label_scale = float(dataset.labels[train_mask].std())
    if label_scale < 1e-6:
        label_scale = 1.0
    edge_type_count = len(dataset.manifest["road_types"])
    edge_source_values = np.asarray(dataset.edge_source, dtype=np.int64)
    edge_target_values = np.asarray(dataset.edge_target, dtype=np.int64)
    edge_length_values = np.asarray(dataset.edge_length, dtype=np.float32)
    edge_type_values = np.asarray(dataset.edge_type, dtype=np.int64)
    (
        edge_source_values,
        edge_target_values,
        edge_length_values,
        edge_type_values,
        transform,
    ) = _transform_edge_arrays(
        edge_source_values,
        edge_target_values,
        edge_length_values,
        edge_type_values,
        variant,
        randomization_seed,
    )

    edge_source = torch.as_tensor(
        edge_source_values, device=device, dtype=torch.long
    )
    edge_target = torch.as_tensor(
        edge_target_values, device=device, dtype=torch.long
    )
    edge_length = torch.as_tensor(
        edge_length_values, device=device, dtype=torch.float32
    )
    edge_type = torch.as_tensor(edge_type_values, device=device, dtype=torch.long)
    prototype_origins, prototype_destinations = _build_prototype_fields(dataset, device)
    prototype_weight = torch.as_tensor(dataset.prototype_weight, device=device)
    destination_permutation: list[int] | None = None
    shuffled_od_metadata: dict[str, object] | None = None
    if variant == "shuffled_od":
        (
            prototype_origins,
            prototype_destinations,
            prototype_weight,
            shuffled_od_metadata,
        ) = _marginal_preserving_od_shuffle(
            prototype_origins,
            prototype_destinations,
            prototype_weight,
        )
    edge_features = build_edge_features(edge_length, edge_type, edge_type_count)
    if variant == "no_edge_features":
        edge_features = torch.zeros_like(edge_features)
    tensors = {
        "node_features": torch.as_tensor(dataset.node_features, device=device),
        "edge_source": edge_source,
        "edge_target": edge_target,
        "edge_features": edge_features,
        "origin_fields": prototype_origins,
        "destination_fields": prototype_destinations,
        "prototype_weight": prototype_weight,
        "region_nodes": torch.as_tensor(dataset.region_nodes, device=device, dtype=torch.long),
        "region_features": torch.as_tensor(
            (dataset.region_features - feature_mean) / feature_scale,
            device=device,
        ),
        "labels": torch.as_tensor(
            (dataset.labels - label_mean) / label_scale,
            device=device,
        ),
        "split_train_mask": torch.as_tensor(
            train_mask, device=device, dtype=torch.bool
        ),
        "forward_degree": build_receiver_normalizers(
            edge_target, dataset.node_ids.size
        ),
        "reverse_degree": build_receiver_normalizers(
            edge_source, dataset.node_ids.size
        ),
    }
    return (
        tensors,
        {
            "region_feature_mean": feature_mean,
            "region_feature_scale": feature_scale,
            "label_mean": label_mean,
            "label_scale": label_scale,
        },
        {
            "variant": variant,
            "randomization_seed": randomization_seed,
            "graph_transform": transform,
            "original_edge_count": int(dataset.edge_source.size),
            "effective_edge_count": int(edge_source_values.size),
            "edge_features_zeroed": variant == "no_edge_features",
            "destination_prototypes_permuted": variant == "shuffled_od",
            "destination_permutation": destination_permutation,
            "shuffled_od_coupling": shuffled_od_metadata,
            "note": (
                "degree_rewired preserves exact directed in/out degree sequences "
                "by permuting edge targets; parallel edges and self-loops are retained "
                "as part of the randomized control."
                if variant == "degree_rewired"
                else None
            ),
        },
    )


def _attach_fixed_prior(
    tensors: dict[str, torch.Tensor],
    fixed_prior: str,
    prototype_batch_size: int,
) -> dict[str, object]:
    if fixed_prior == "none":
        return {"name": "none"}
    if fixed_prior != "z0":
        raise ValueError(f"unsupported fixed prior: {fixed_prior}")

    raw_score = torch.zeros(
        tensors["region_nodes"].shape[0],
        device=tensors["region_nodes"].device,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        for prototype_slice in iter_slices(
            tensors["prototype_weight"].size(0), prototype_batch_size
        ):
            batch_score = deterministic_diffusion_batch_scores(
                origin_fields=tensors["origin_fields"][prototype_slice],
                destination_fields=tensors["destination_fields"][prototype_slice],
                edge_source=tensors["edge_source"],
                edge_target=tensors["edge_target"],
                region_nodes=tensors["region_nodes"],
                receiver_normalizer_forward=tensors["forward_degree"],
                receiver_normalizer_reverse=tensors["reverse_degree"],
            )
            raw_score += (
                batch_score.float()
                * tensors["prototype_weight"][prototype_slice, None].float()
            ).sum(dim=0)
    train_mask = tensors["split_train_mask"]
    train_score = raw_score[train_mask]
    train_mean = train_score.mean()
    train_std = train_score.std(unbiased=False).clamp_min(1e-6)
    tensors["fixed_prior"] = (raw_score - train_mean) / train_std
    return {
        "name": "z0",
        "description": (
            "deterministic scalar bidirectional fixed-mean diffusion over depths "
            "1,2,4,8,16,32; centered and standardized on train candidates only"
        ),
        "readout_depths": [1, 2, 4, 8, 16, 32],
        "train_mean": float(train_mean.item()),
        "train_std": float(train_std.item()),
    }


def _prepare_deployment_objective(
    *,
    dataset: DemandFieldDataset,
    tensors: dict[str, torch.Tensor],
    cost_labels: Path,
    top_k: int,
    topk_temperature: float,
    topk_gain_weight: float,
    budget_weight: float,
    conflict_weight: float,
    shortcut_budget_ratio: float,
    spearman_tolerance: float,
) -> DeploymentObjective:
    if "fixed_prior" not in tensors:
        raise ValueError("deployment objective requires a fixed prior")
    shortcut_cost_values = _load_shortcut_costs(cost_labels, dataset.region_ids)
    shortcut_cost = torch.as_tensor(
        shortcut_cost_values,
        device=tensors["labels"].device,
        dtype=torch.float32,
    )
    split_context: dict[str, dict[str, object]] = {}
    conflicts: dict[str, torch.Tensor] = {}
    total_candidate_count = len(dataset.region_ids)
    for split_name in ("train", "validation"):
        mask = dataset.split_mask(split_name)
        indices = np.flatnonzero(mask)
        split_top_k = _scaled_split_top_k(
            top_k,
            len(indices),
            total_candidate_count,
        )
        split_scores = tensors["fixed_prior"][torch.as_tensor(
            indices, device=shortcut_cost.device, dtype=torch.long
        )]
        split_cost = shortcut_cost[torch.as_tensor(
            indices, device=shortcut_cost.device, dtype=torch.long
        )]
        split_weights = _soft_topk_weights(
            split_scores,
            split_top_k,
            topk_temperature,
        )
        soft_budget = float((split_weights * split_cost).sum().item())
        hard_metrics = _hard_disjoint_deployment_metrics(
            prediction=split_scores.detach().float().cpu().numpy(),
            target=dataset.labels[mask],
            region_nodes=dataset.region_nodes[mask],
            shortcut_cost=shortcut_cost_values[mask],
            top_k=split_top_k,
        )
        split_context[split_name] = {
            "top_k": split_top_k,
            "soft_budget": soft_budget * shortcut_budget_ratio,
            "hard_budget": float(hard_metrics["shortcut_count"])
            * shortcut_budget_ratio,
            "z0_gain": float(hard_metrics["mean_gain"]),
            "z0_spearman": regression_metrics(
                split_scores.detach().float().cpu().numpy(),
                dataset.labels[mask],
            )["spearman"],
        }
        conflicts[split_name] = torch.as_tensor(
            _region_conflict_matrix(dataset.region_nodes[mask]),
            device=shortcut_cost.device,
            dtype=torch.float32,
        )
    return DeploymentObjective(
        shortcut_cost=shortcut_cost,
        train_conflict=conflicts["train"],
        validation_conflict=conflicts["validation"],
        top_k=top_k,
        train_top_k=int(split_context["train"]["top_k"]),
        validation_top_k=int(split_context["validation"]["top_k"]),
        topk_temperature=topk_temperature,
        topk_gain_weight=topk_gain_weight,
        budget_weight=budget_weight,
        conflict_weight=conflict_weight,
        shortcut_budget_ratio=shortcut_budget_ratio,
        spearman_tolerance=spearman_tolerance,
        train_soft_budget=float(split_context["train"]["soft_budget"]),
        validation_soft_budget=float(split_context["validation"]["soft_budget"]),
        train_hard_budget=float(split_context["train"]["hard_budget"]),
        validation_hard_budget=float(split_context["validation"]["hard_budget"]),
        train_z0_gain=float(split_context["train"]["z0_gain"]),
        validation_z0_gain=float(split_context["validation"]["z0_gain"]),
        train_z0_spearman=float(split_context["train"]["z0_spearman"]),
        validation_z0_spearman=float(
            split_context["validation"]["z0_spearman"]
        ),
        cost_source=_display_path(cost_labels),
        cost_source_sha256=_sha256(cost_labels),
    )


def _scaled_split_top_k(
    full_top_k: int,
    split_candidate_count: int,
    total_candidate_count: int,
) -> int:
    if full_top_k <= 0 or full_top_k > total_candidate_count:
        raise ValueError("full deployment K must be in [1, total_candidate_count]")
    if split_candidate_count <= 0 or split_candidate_count > total_candidate_count:
        raise ValueError(
            "split candidate count must be in [1, total_candidate_count]"
        )
    scaled = int(
        np.floor(
            full_top_k * split_candidate_count / total_candidate_count + 0.5
        )
    )
    return max(1, min(split_candidate_count, scaled))


def _load_shortcut_costs(path: Path, region_ids: np.ndarray) -> np.ndarray:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames or not {"region_id", "shortcut_count"}.issubset(
            reader.fieldnames
        ):
            raise ValueError("cost labels must contain region_id and shortcut_count")
        rows: dict[int, float] = {}
        for row in reader:
            region_id = int(row["region_id"])
            if region_id in rows:
                raise ValueError(f"duplicate cost row for region {region_id}")
            rows[region_id] = float(row["shortcut_count"])
    expected = set(map(int, region_ids))
    if set(rows) != expected:
        raise ValueError("cost label region ids do not exactly match the dataset")
    values = np.asarray([rows[int(region_id)] for region_id in region_ids], dtype=np.float32)
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("shortcut costs must be finite and non-negative")
    return values


def _region_conflict_matrix(region_nodes: np.ndarray) -> np.ndarray:
    if region_nodes.ndim != 2:
        raise ValueError("region nodes must be a two-dimensional array")
    matrix = np.zeros((len(region_nodes), len(region_nodes)), dtype=np.float32)
    containing_regions: dict[int, list[int]] = {}
    for region_index, nodes in enumerate(region_nodes):
        prior_conflicts: set[int] = set()
        for node in map(int, nodes):
            prior_conflicts.update(containing_regions.get(node, ()))
        for other_index in prior_conflicts:
            matrix[region_index, other_index] = 1.0
            matrix[other_index, region_index] = 1.0
        for node in map(int, nodes):
            containing_regions.setdefault(node, []).append(region_index)
    return matrix


def _marginal_preserving_od_shuffle(
    origin_fields: torch.Tensor,
    destination_fields: torch.Tensor,
    prototype_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    """Break OD pairing with a half-cycle coupling while preserving both marginals."""

    weights = prototype_weight.detach().double().cpu().numpy()
    if (
        origin_fields.shape != destination_fields.shape
        or len(origin_fields) != len(weights)
        or np.any(weights <= 0.0)
    ):
        raise ValueError("prototype fields and positive weights must align")
    weights = weights / weights.sum()
    cumulative = np.cumsum(weights)
    shift = 0.5
    shifted_boundaries = np.mod(cumulative - shift, 1.0)
    boundaries = np.unique(
        np.concatenate(([0.0, 1.0], cumulative[:-1], shifted_boundaries))
    )
    origin_indices: list[int] = []
    destination_indices: list[int] = []
    coupling_weights: list[float] = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end - start <= 1e-12:
            continue
        midpoint = (start + end) / 2.0
        origin_index = int(np.searchsorted(cumulative, midpoint, side="right"))
        shifted_midpoint = (midpoint + shift) % 1.0
        destination_index = int(
            np.searchsorted(cumulative, shifted_midpoint, side="right")
        )
        origin_indices.append(origin_index)
        destination_indices.append(destination_index)
        coupling_weights.append(end - start)
    origin_index_tensor = torch.as_tensor(
        origin_indices, device=origin_fields.device, dtype=torch.long
    )
    destination_index_tensor = torch.as_tensor(
        destination_indices, device=destination_fields.device, dtype=torch.long
    )
    coupling = torch.as_tensor(
        coupling_weights,
        device=prototype_weight.device,
        dtype=prototype_weight.dtype,
    )
    coupling = coupling / coupling.sum()
    same_pair_mass = sum(
        weight
        for origin_index, destination_index, weight in zip(
            origin_indices, destination_indices, coupling_weights
        )
        if origin_index == destination_index
    )
    return (
        origin_fields[origin_index_tensor],
        destination_fields[destination_index_tensor],
        coupling,
        {
            "method": "half_cycle_measure_preserving_coupling",
            "original_prototype_count": len(weights),
            "coupled_prototype_count": len(coupling_weights),
            "shift": shift,
            "same_pair_mass": same_pair_mass,
            "preserves_origin_marginal": True,
            "preserves_destination_marginal": True,
        },
    )


def _transform_edge_arrays(
    edge_source: np.ndarray,
    edge_target: np.ndarray,
    edge_length: np.ndarray,
    edge_type: np.ndarray,
    variant: str,
    randomization_seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    if not (
        edge_source.shape
        == edge_target.shape
        == edge_length.shape
        == edge_type.shape
    ):
        raise ValueError("edge arrays must have identical shapes")
    if variant in {"undirected", "graphsage"}:
        return (
            np.concatenate((edge_source, edge_target)),
            np.concatenate((edge_target, edge_source)),
            np.concatenate((edge_length, edge_length)),
            np.concatenate((edge_type, edge_type)),
            "bidirectional_edge_expansion",
        )
    if variant == "degree_rewired":
        permutation = np.random.default_rng(randomization_seed).permutation(
            edge_target.size
        )
        return (
            edge_source.copy(),
            edge_target[permutation],
            edge_length.copy(),
            edge_type.copy(),
            "degree_preserving_target_permutation",
        )
    return (
        edge_source.copy(),
        edge_target.copy(),
        edge_length.copy(),
        edge_type.copy(),
        "directed_original",
    )


def _build_prototype_fields(
    dataset: DemandFieldDataset,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    prototype_count = dataset.prototype_weight.size
    node_count = dataset.node_ids.size
    origins = torch.zeros((prototype_count, node_count), device=device)
    destinations = torch.zeros((prototype_count, node_count), device=device)
    for prototype_id in range(prototype_count):
        origin_start = dataset.prototype_origin_offsets[prototype_id]
        origin_end = dataset.prototype_origin_offsets[prototype_id + 1]
        destination_start = dataset.prototype_destination_offsets[prototype_id]
        destination_end = dataset.prototype_destination_offsets[prototype_id + 1]
        origins[prototype_id].index_add_(
            0,
            torch.as_tensor(
                dataset.prototype_origin_nodes[origin_start:origin_end],
                device=device,
                dtype=torch.long,
            ),
            torch.as_tensor(
                dataset.prototype_origin_weights[origin_start:origin_end],
                device=device,
            ),
        )
        destinations[prototype_id].index_add_(
            0,
            torch.as_tensor(
                dataset.prototype_destination_nodes[destination_start:destination_end],
                device=device,
                dtype=torch.long,
            ),
            torch.as_tensor(
                dataset.prototype_destination_weights[destination_start:destination_end],
                device=device,
            ),
        )
    return origins, destinations


def _train_one_seed(
    dataset: DemandFieldDataset,
    tensors: dict[str, torch.Tensor],
    scalers: dict[str, np.ndarray | float],
    config: NBFNetConfig,
    seed: int,
    device: torch.device,
    precision_policy: PrecisionPolicy,
    training_objective: str,
    fixed_prior: str,
    soft_rank_temperature: float = 0.1,
    residual_gate_grid: tuple[float, ...] = (1.0,),
    validation_snapshot_dir: Path | None = None,
    head_warmup_steps: int = 0,
    evaluation_splits: tuple[str, ...] = SPLIT_NAMES,
    lr_scheduler: str = "none",
    lr_scheduler_factor: float = 0.3,
    lr_scheduler_patience: int = 3,
    lr_scheduler_threshold: float = 1.0e-4,
    lr_scheduler_min_lr: float = 5.0e-4,
    deployment_objective: DeploymentObjective | None = None,
) -> dict:
    precision_policy.validate()
    if (training_objective == "deployment_aware") != (
        deployment_objective is not None
    ):
        raise ValueError("deployment objective configuration mismatch")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    model = BidirectionalNBFNet(
        node_feature_dim=dataset.node_features.shape[1],
        region_feature_dim=dataset.region_features.shape[1],
        edge_type_count=len(dataset.manifest["road_types"]),
        config=config,
    ).to(device)
    _set_training_scope(model, "output_head" if head_warmup_steps else "all")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    learning_rate_scheduler = _build_learning_rate_scheduler(
        optimizer,
        lr_scheduler,
        factor=lr_scheduler_factor,
        patience=lr_scheduler_patience,
        threshold=lr_scheduler_threshold,
        min_lr=lr_scheduler_min_lr,
    )
    gradient_scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=precision_policy.grad_scaler_init_scale,
        enabled=precision_policy.grad_scaler_enabled,
    )
    train_indices = torch.as_tensor(
        np.flatnonzero(dataset.split_mask("train")), device=device, dtype=torch.long
    )
    validation_indices = torch.as_tensor(
        np.flatnonzero(dataset.split_mask("validation")), device=device, dtype=torch.long
    )
    best_validation_spearman = float("-inf")
    best_validation_deployment: dict[str, float | int | bool] | None = None
    best_validation_key: tuple[float, float, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    best_residual_gate = 1.0
    epochs_without_improvement = 0
    history: list[dict[str, float | int | str]] = []
    validation_snapshot_entries: list[dict[str, object]] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        initial_train_prediction = _predict_weighted(
            model,
            tensors,
            train_indices,
            config.prototype_batch_size,
            precision_policy,
        )
        initial_validation_prediction = _predict_weighted(
            model,
            tensors,
            validation_indices,
            config.prototype_batch_size,
            precision_policy,
        )
    train_target = tensors["labels"][train_indices]
    validation_target = tensors["labels"][validation_indices]
    initial_train_loss, initial_train_huber, initial_train_rank = _evaluation_loss(
        initial_train_prediction,
        train_target,
        config,
        training_objective,
        soft_rank_temperature,
    )
    (
        initial_validation_loss,
        initial_validation_huber,
        initial_validation_rank,
    ) = _evaluation_loss(
        initial_validation_prediction,
        validation_target,
        config,
        training_objective,
        soft_rank_temperature,
    )
    initial_train_deployment_components = None
    initial_validation_deployment_components = None
    if deployment_objective is not None:
        initial_train_total, initial_train_deployment_components = (
            _deployment_aware_loss_tensor(
                prediction=initial_train_prediction,
                target=train_target,
                shortcut_cost=deployment_objective.shortcut_cost[train_indices],
                conflict_matrix=deployment_objective.train_conflict,
                top_k=deployment_objective.train_top_k,
                temperature=deployment_objective.topk_temperature,
                soft_budget=deployment_objective.train_soft_budget,
                topk_gain_weight=deployment_objective.topk_gain_weight,
                budget_weight=deployment_objective.budget_weight,
                conflict_weight=deployment_objective.conflict_weight,
                soft_rank_temperature=soft_rank_temperature,
            )
        )
        initial_validation_total, initial_validation_deployment_components = (
            _deployment_aware_loss_tensor(
                prediction=initial_validation_prediction,
                target=validation_target,
                shortcut_cost=deployment_objective.shortcut_cost[validation_indices],
                conflict_matrix=deployment_objective.validation_conflict,
                top_k=deployment_objective.validation_top_k,
                temperature=deployment_objective.topk_temperature,
                soft_budget=deployment_objective.validation_soft_budget,
                topk_gain_weight=deployment_objective.topk_gain_weight,
                budget_weight=deployment_objective.budget_weight,
                conflict_weight=deployment_objective.conflict_weight,
                soft_rank_temperature=soft_rank_temperature,
            )
        )
        initial_train_loss = float(initial_train_total.item())
        initial_train_rank = float(
            initial_train_deployment_components["global_rank_loss"].item()
        )
        initial_validation_loss = float(initial_validation_total.item())
        initial_validation_rank = float(
            initial_validation_deployment_components["global_rank_loss"].item()
        )
    validation_prior = (
        tensors["fixed_prior"][validation_indices]
        if "fixed_prior" in tensors
        else None
    )
    (
        initial_residual_gate,
        initial_gated_validation_prediction,
        initial_raw_validation_spearman,
    ) = _select_residual_gate(
        initial_validation_prediction,
        validation_prior,
        dataset.labels[dataset.split_mask("validation")],
        residual_gate_grid,
    )
    initial_validation_unscaled = _unscale_prediction(
        initial_gated_validation_prediction,
        scalers,
    )
    initial_validation_metrics = regression_metrics(
        initial_validation_unscaled,
        dataset.labels[dataset.split_mask("validation")],
    )
    initial_validation_deployment = (
        _validation_deployment_metrics(
            prediction=initial_gated_validation_prediction,
            dataset=dataset,
            shortcut_cost=deployment_objective.shortcut_cost,
            objective=deployment_objective,
        )
        if deployment_objective is not None
        else None
    )
    initial_train_pairwise_accuracy = _full_pairwise_accuracy(
        initial_train_prediction,
        train_target,
    )
    initial_validation_pairwise_accuracy = _full_pairwise_accuracy(
        initial_validation_prediction,
        validation_target,
    )
    initial_history_row = _diagnostic_history_row(
        epoch=0,
        train_loss=initial_train_loss,
        train_huber=initial_train_huber,
        train_rank=float("nan"),
        train_full_pairwise_loss=initial_train_rank,
        train_pairwise_accuracy=initial_train_pairwise_accuracy,
        train_prediction=initial_train_prediction,
        validation_loss=initial_validation_loss,
        validation_huber=initial_validation_huber,
        validation_rank=initial_validation_rank,
        validation_pairwise_accuracy=initial_validation_pairwise_accuracy,
        validation_spearman=initial_validation_metrics["spearman"],
        validation_prediction=initial_validation_prediction,
        scaler_scale_before=gradient_scaler.get_scale(),
        scaler_scale_after=gradient_scaler.get_scale(),
        optimizer_state_step_before=0,
        optimizer_state_step_after=0,
        optimizer_step_skipped=False,
        optimizer_step_effective=False,
        gradient_norm_before_clip=float("nan"),
        gradient_norm_after_clip=float("nan"),
        parameter_delta_norm=0.0,
        learning_rate=optimizer.param_groups[0]["lr"],
        next_learning_rate=optimizer.param_groups[0]["lr"],
    )
    initial_history_row["trainable_scope"] = (
        "output_head" if head_warmup_steps else "all"
    )
    initial_history_row["trainable_parameter_count"] = _trainable_parameter_count(model)
    initial_history_row["validation_raw_spearman"] = initial_raw_validation_spearman
    initial_history_row["residual_gate"] = initial_residual_gate
    if deployment_objective is not None:
        _attach_deployment_history(
            initial_history_row,
            initial_train_deployment_components,
            initial_validation_deployment_components,
            initial_validation_deployment,
        )
    history.append(initial_history_row)
    _record_validation_snapshot(
        snapshot_dir=validation_snapshot_dir,
        entries=validation_snapshot_entries,
        epoch=0,
        model=model,
        dataset=dataset,
        config=config,
        scalers=scalers,
        precision_policy=precision_policy,
        training_objective=training_objective,
        soft_rank_temperature=soft_rank_temperature,
        fixed_prior_name=fixed_prior,
        prediction=initial_validation_prediction,
        fixed_prior=validation_prior,
        target=dataset.labels[dataset.split_mask("validation")],
        residual_gate_grid=residual_gate_grid,
        validation_loss=initial_validation_loss,
        optimizer_step_effective=False,
        deployment_objective=(
            deployment_objective.metadata()
            if deployment_objective is not None
            else None
        ),
        deployment_validation=initial_validation_deployment,
    )
    print(
        f"seed={seed} epoch=000/{config.max_epochs} "
        f"validation_loss={initial_validation_loss:.6f} "
            f"validation_spearman={initial_validation_metrics['spearman']:.4f} "
            f"raw_spearman={initial_raw_validation_spearman:.4f} "
            f"residual_gate={initial_residual_gate:g} "
        f"precision={precision_policy.mode} optimizer_step=none"
        + (
            f" validation_deploy_gain={initial_validation_deployment['mean_gain']:.6f}"
            f" validation_shortcuts={initial_validation_deployment['shortcut_count']}"
            if initial_validation_deployment is not None
            else ""
        ),
        flush=True,
    )
    effective_optimizer_steps = 0
    skipped_optimizer_steps = 0
    first_effective_optimizer_epoch: int | None = None
    first_positive_validation_epoch = (
        0 if initial_validation_metrics["spearman"] > 0.0 else None
    )
    best_checkpoint_effective_step_count = 0
    backbone_effective_optimizer_steps = 0
    first_backbone_effective_optimizer_epoch: int | None = None
    best_checkpoint_backbone_effective_step_count = 0
    learning_rate_reduction_epochs: list[int] = []
    stopped_reason = "max_epochs"

    for epoch in range(1, config.max_epochs + 1):
        epoch_started = time.perf_counter()
        learning_rate_used = float(optimizer.param_groups[0]["lr"])
        trainable_scope = "output_head" if epoch <= head_warmup_steps else "all"
        _set_training_scope(model, trainable_scope)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            full_train_prediction = _predict_weighted(
                model,
                tensors,
                train_indices,
                config.prototype_batch_size,
                precision_policy,
            )
        train_full_pairwise_loss = _full_pairwise_loss(
            full_train_prediction,
            train_target,
        )
        train_pairwise_accuracy = _full_pairwise_accuracy(
            full_train_prediction,
            train_target,
        )
        prediction_leaf = full_train_prediction.detach().requires_grad_(True)
        train_deployment_components = None
        if training_objective in RANK_ONLY_OBJECTIVES:
            train_huber = prediction_leaf.sum() * 0.0
            if deployment_objective is not None:
                train_loss, train_deployment_components = (
                    _deployment_aware_loss_tensor(
                        prediction=prediction_leaf,
                        target=train_target,
                        shortcut_cost=deployment_objective.shortcut_cost[train_indices],
                        conflict_matrix=deployment_objective.train_conflict,
                        top_k=deployment_objective.train_top_k,
                        temperature=deployment_objective.topk_temperature,
                        soft_budget=deployment_objective.train_soft_budget,
                        topk_gain_weight=deployment_objective.topk_gain_weight,
                        budget_weight=deployment_objective.budget_weight,
                        conflict_weight=deployment_objective.conflict_weight,
                        soft_rank_temperature=soft_rank_temperature,
                    )
                )
                train_rank = train_deployment_components["global_rank_loss"]
            else:
                train_rank = (
                    _soft_spearman_loss_tensor(
                        prediction_leaf,
                        train_target,
                        soft_rank_temperature,
                    )
                    if training_objective == "soft_spearman"
                    else _full_pairwise_loss_tensor(prediction_leaf, train_target)
                )
                train_loss = train_rank
        else:
            train_huber = functional.huber_loss(
                prediction_leaf,
                train_target,
                reduction="mean",
                delta=config.huber_delta,
            )
            train_rank = _sampled_pairwise_loss(
                prediction_leaf, train_target, generator
            )
            train_loss = train_huber + config.rank_weight * train_rank
        prediction_gradient = torch.autograd.grad(train_loss, prediction_leaf)[0].detach()

        prototype_count = tensors["prototype_weight"].size(0)
        for prototype_slice in iter_slices(
            prototype_count, config.prototype_batch_size
        ):
            prototype_ids = torch.arange(
                prototype_slice.start,
                prototype_slice.stop,
                device=device,
            )
            with _autocast_context(device, precision_policy):
                prediction = _forward_batch(model, tensors, prototype_ids, train_indices)
                weighted_contribution = (
                    prediction
                    * tensors["prototype_weight"][prototype_ids, None]
                ).sum(dim=0)
                gradient_surrogate = (
                    weighted_contribution * prediction_gradient
                ).sum()
            gradient_scaler.scale(gradient_surrogate).backward()
        scaler_scale_before = gradient_scaler.get_scale()
        gradient_scaler.unscale_(optimizer)
        gradient_norm_before_clip = _global_gradient_norm(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        gradient_norm_after_clip = _global_gradient_norm(model)
        parameters_before_step = _parameter_snapshot(model)
        optimizer_state_step_before = _optimizer_state_step(optimizer)
        gradient_scaler.step(optimizer)
        gradient_scaler.update()
        scaler_scale_after = gradient_scaler.get_scale()
        optimizer_state_step_after = _optimizer_state_step(optimizer)
        parameter_delta_norm = _parameter_delta_norm(parameters_before_step, model)
        optimizer_step_skipped = _optimizer_step_was_skipped(
            optimizer_state_step_before,
            optimizer_state_step_after,
        )
        optimizer_step_effective = (
            not optimizer_step_skipped and parameter_delta_norm > 0.0
        )
        if optimizer_step_skipped:
            skipped_optimizer_steps += 1
        if optimizer_step_effective:
            effective_optimizer_steps += 1
            if first_effective_optimizer_epoch is None:
                first_effective_optimizer_epoch = epoch
            if trainable_scope == "all":
                backbone_effective_optimizer_steps += 1
                if first_backbone_effective_optimizer_epoch is None:
                    first_backbone_effective_optimizer_epoch = epoch

        model.eval()
        with torch.no_grad():
            validation_prediction = _predict_weighted(
                model,
                tensors,
                validation_indices,
                config.prototype_batch_size,
                precision_policy,
            )
            validation_loss, validation_huber, validation_rank = _evaluation_loss(
                validation_prediction,
                validation_target,
                config,
                training_objective,
                soft_rank_temperature,
            )
            validation_deployment_components = None
            if deployment_objective is not None:
                validation_total, validation_deployment_components = (
                    _deployment_aware_loss_tensor(
                        prediction=validation_prediction,
                        target=validation_target,
                        shortcut_cost=deployment_objective.shortcut_cost[
                            validation_indices
                        ],
                        conflict_matrix=deployment_objective.validation_conflict,
                        top_k=deployment_objective.validation_top_k,
                        temperature=deployment_objective.topk_temperature,
                        soft_budget=deployment_objective.validation_soft_budget,
                        topk_gain_weight=deployment_objective.topk_gain_weight,
                        budget_weight=deployment_objective.budget_weight,
                        conflict_weight=deployment_objective.conflict_weight,
                        soft_rank_temperature=soft_rank_temperature,
                    )
                )
                validation_loss = float(validation_total.item())
                validation_rank = float(
                    validation_deployment_components["global_rank_loss"].item()
                )
            (
                residual_gate,
                gated_validation_prediction,
                raw_validation_spearman,
            ) = _select_residual_gate(
                validation_prediction,
                validation_prior,
                dataset.labels[dataset.split_mask("validation")],
                residual_gate_grid,
            )
            validation_unscaled = _unscale_prediction(
                gated_validation_prediction,
                scalers,
            )
            validation_metrics = regression_metrics(
                validation_unscaled,
                dataset.labels[dataset.split_mask("validation")],
            )
            validation_deployment = (
                _validation_deployment_metrics(
                    prediction=gated_validation_prediction,
                    dataset=dataset,
                    shortcut_cost=deployment_objective.shortcut_cost,
                    objective=deployment_objective,
                )
                if deployment_objective is not None
                else None
            )
        validation_pairwise_accuracy = _full_pairwise_accuracy(
            validation_prediction,
            validation_target,
        )
        if learning_rate_scheduler is not None and trainable_scope == "all":
            previous_learning_rate = float(optimizer.param_groups[0]["lr"])
            learning_rate_scheduler.step(validation_metrics["spearman"])
            if float(optimizer.param_groups[0]["lr"]) < previous_learning_rate:
                learning_rate_reduction_epochs.append(epoch)
        next_learning_rate = float(optimizer.param_groups[0]["lr"])
        history_row = _diagnostic_history_row(
            epoch=epoch,
            train_loss=float(train_loss.detach()),
            train_huber=float(train_huber.detach()),
            train_rank=float(train_rank.detach()),
            train_full_pairwise_loss=train_full_pairwise_loss,
            train_pairwise_accuracy=train_pairwise_accuracy,
            train_prediction=full_train_prediction,
            validation_loss=validation_loss,
            validation_huber=validation_huber,
            validation_rank=validation_rank,
            validation_pairwise_accuracy=validation_pairwise_accuracy,
            validation_spearman=validation_metrics["spearman"],
            validation_prediction=validation_prediction,
            scaler_scale_before=scaler_scale_before,
            scaler_scale_after=scaler_scale_after,
            optimizer_state_step_before=optimizer_state_step_before,
            optimizer_state_step_after=optimizer_state_step_after,
            optimizer_step_skipped=optimizer_step_skipped,
            optimizer_step_effective=optimizer_step_effective,
            gradient_norm_before_clip=gradient_norm_before_clip,
            gradient_norm_after_clip=gradient_norm_after_clip,
            parameter_delta_norm=parameter_delta_norm,
            learning_rate=learning_rate_used,
            next_learning_rate=next_learning_rate,
        )
        history_row["trainable_scope"] = trainable_scope
        history_row["trainable_parameter_count"] = _trainable_parameter_count(model)
        history_row["validation_raw_spearman"] = raw_validation_spearman
        history_row["residual_gate"] = residual_gate
        if deployment_objective is not None:
            _attach_deployment_history(
                history_row,
                train_deployment_components,
                validation_deployment_components,
                validation_deployment,
            )
        history.append(history_row)
        _record_validation_snapshot(
            snapshot_dir=validation_snapshot_dir,
            entries=validation_snapshot_entries,
            epoch=epoch,
            model=model,
            dataset=dataset,
            config=config,
            scalers=scalers,
            precision_policy=precision_policy,
            training_objective=training_objective,
            soft_rank_temperature=soft_rank_temperature,
            fixed_prior_name=fixed_prior,
            prediction=validation_prediction,
            fixed_prior=validation_prior,
            target=dataset.labels[dataset.split_mask("validation")],
            residual_gate_grid=residual_gate_grid,
            validation_loss=validation_loss,
            optimizer_step_effective=optimizer_step_effective,
            deployment_objective=(
                deployment_objective.metadata()
                if deployment_objective is not None
                else None
            ),
            deployment_validation=validation_deployment,
        )
        if (
            first_positive_validation_epoch is None
            and validation_metrics["spearman"] > 0.0
        ):
            first_positive_validation_epoch = epoch
        checkpoint_is_eligible = (
            training_objective not in RANK_ONLY_OBJECTIVES
            or (
                optimizer_step_effective
                and trainable_scope == "all"
                and backbone_effective_optimizer_steps > 0
            )
        )
        deployment_checkpoint_eligible = (
            deployment_objective is not None
            and validation_deployment is not None
            and bool(validation_deployment["budget_feasible"])
            and bool(validation_deployment["spearman_safe"])
            and float(validation_deployment["mean_gain"])
            >= deployment_objective.validation_z0_gain
        )
        validation_key = (
            (
                float(validation_deployment["mean_gain"]),
                -float(validation_deployment["shortcut_count"]),
                float(validation_metrics["spearman"]),
            )
            if validation_deployment is not None
            else None
        )
        checkpoint_improved = (
            validation_key is not None
            and deployment_checkpoint_eligible
            and (best_validation_key is None or validation_key > best_validation_key)
            if deployment_objective is not None
            else validation_metrics["spearman"]
            > best_validation_spearman + config.min_improvement
        )
        if checkpoint_is_eligible and checkpoint_improved:
            best_validation_spearman = validation_metrics["spearman"]
            best_validation_deployment = validation_deployment
            best_validation_key = validation_key
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_epoch = epoch
            best_residual_gate = residual_gate
            best_checkpoint_effective_step_count = effective_optimizer_steps
            best_checkpoint_backbone_effective_step_count = (
                backbone_effective_optimizer_steps
            )
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        elapsed_seconds = time.perf_counter() - started
        epoch_seconds = time.perf_counter() - epoch_started
        estimated_remaining_seconds = (
            elapsed_seconds / epoch * (config.max_epochs - epoch)
        )
        print(
            f"seed={seed} epoch={epoch:03d}/{config.max_epochs} "
            f"train_loss={float(train_loss.detach()):.6f} "
            f"validation_loss={validation_loss:.6f} "
            f"validation_spearman={validation_metrics['spearman']:.4f} "
            f"raw_spearman={raw_validation_spearman:.4f} "
            f"residual_gate={residual_gate:g} "
            f"scale={scaler_scale_before:g}->{scaler_scale_after:g} "
            f"optimizer_step={optimizer_state_step_before}->{optimizer_state_step_after} "
            f"step_skipped={int(optimizer_step_skipped)} "
            f"grad_norm={gradient_norm_before_clip:.3e}->{gradient_norm_after_clip:.3e} "
            f"param_delta={parameter_delta_norm:.3e} "
            f"lr={learning_rate_used:.3e}->{next_learning_rate:.3e} "
            + (
                f"deploy_gain={validation_deployment['mean_gain']:.6f} "
                f"shortcuts={validation_deployment['shortcut_count']} "
                f"budget_ok={int(validation_deployment['budget_feasible'])} "
                f"spearman_ok={int(validation_deployment['spearman_safe'])} "
                if validation_deployment is not None
                else ""
            )
            + f"best_epoch={best_epoch:03d} "
            f"patience={epochs_without_improvement:02d}/{config.patience} "
            f"epoch_time={_format_duration(epoch_seconds)} "
            f"elapsed={_format_duration(elapsed_seconds)} "
            f"eta_to_max={_format_duration(estimated_remaining_seconds)}",
            flush=True,
        )
        if epochs_without_improvement >= config.patience:
            stopped_reason = (
                "validation_deployment_patience"
                if deployment_objective is not None
                else "validation_spearman_patience"
            )
            print(
                f"seed={seed} early_stop epoch={epoch} "
                f"best_epoch={best_epoch} "
                f"best_validation_spearman={best_validation_spearman:.6f}",
                flush=True,
            )
            break

    evaluation_mask = np.zeros(dataset.region_ids.size, dtype=bool)
    for split_name in evaluation_splits:
        evaluation_mask |= dataset.split_mask(split_name)
    evaluation_indices = torch.as_tensor(
        np.flatnonzero(evaluation_mask), device=device, dtype=torch.long
    )
    if best_state is None and (
        training_objective not in RANK_ONLY_OBJECTIVES or fixed_prior == "none"
    ):
        raise RuntimeError("NBFNet training did not produce a finite validation checkpoint")
    if best_state is None:
        print(
            f"seed={seed} no_effective_checkpoint; retaining fixed prior only",
            flush=True,
        )
        model_state = None
        selected_prediction = tensors["fixed_prior"][evaluation_indices].detach().clone()
        best_epoch = 0
        best_residual_gate = 0.0
        best_checkpoint_effective_step_count = 0
        best_checkpoint_backbone_effective_step_count = 0
        stopped_reason = (
            "no_validation_deployment_safe_checkpoint"
            if deployment_objective is not None
            else "no_effective_optimizer_checkpoint"
        )
        selected_validation_deployment = initial_validation_deployment
    else:
        print(
            f"seed={seed} evaluating_best_checkpoint best_epoch={best_epoch}",
            flush=True,
        )
        model.load_state_dict(best_state)
        model.to(device)
        model.eval()
        with torch.no_grad():
            selected_prediction = _predict_weighted(
                model,
                tensors,
                evaluation_indices,
                config.prototype_batch_size,
                precision_policy,
            )
            if "fixed_prior" in tensors:
                selected_prior = tensors["fixed_prior"][evaluation_indices]
                selected_prediction = _apply_residual_gate(
                    selected_prediction,
                    selected_prior,
                    best_residual_gate,
                )
        model_state = best_state
        selected_validation_deployment = best_validation_deployment
    prediction = torch.full(
        (dataset.region_ids.size,),
        float("nan"),
        device=device,
        dtype=selected_prediction.dtype,
    )
    prediction[evaluation_indices] = selected_prediction
    torch.cuda.synchronize(device)
    unscaled_prediction = _unscale_prediction(prediction, scalers)
    optimizer_history = history[1:]
    finite_gradient_norms = [
        float(row["gradient_norm_before_clip"])
        for row in optimizer_history
        if np.isfinite(float(row["gradient_norm_before_clip"]))
    ]
    nonfinite_gradient_norm_steps = (
        len(optimizer_history) - len(finite_gradient_norms)
    )
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": len(history) - 1,
        "training_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "metrics": _all_split_metrics(
            dataset,
            unscaled_prediction,
            split_names=evaluation_splits,
        ),
        "diagnostics": {
            "numerics": precision_policy.metadata(),
            "epoch_zero_validation_spearman": initial_validation_metrics["spearman"],
            "first_effective_optimizer_epoch": first_effective_optimizer_epoch,
            "first_positive_validation_epoch": first_positive_validation_epoch,
            "effective_optimizer_steps": effective_optimizer_steps,
            "skipped_optimizer_steps": skipped_optimizer_steps,
            "best_checkpoint_effective_step_count": (
                best_checkpoint_effective_step_count
            ),
            "head_warmup_steps": head_warmup_steps,
            "first_backbone_effective_optimizer_epoch": (
                first_backbone_effective_optimizer_epoch
            ),
            "backbone_effective_optimizer_steps": backbone_effective_optimizer_steps,
            "best_checkpoint_backbone_effective_step_count": (
                best_checkpoint_backbone_effective_step_count
            ),
            "nonfinite_gradient_norm_steps": nonfinite_gradient_norm_steps,
            "maximum_finite_gradient_norm_before_clip": (
                max(finite_gradient_norms) if finite_gradient_norms else None
            ),
            "maximum_consecutive_zero_gradient_steps_after_clip": (
                _maximum_consecutive_matches(
                    optimizer_history,
                    lambda row: float(row["gradient_norm_after_clip"]) == 0.0,
                )
            ),
            "unrecovered_validation_loss_doublings": (
                _unrecovered_validation_loss_doublings(history)
            ),
            "lr_scheduler": lr_scheduler,
            "learning_rate_reduction_epochs": learning_rate_reduction_epochs,
            "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
            "initialization_dominant": (
                best_checkpoint_effective_step_count == 0
                or best_validation_spearman
                <= initial_validation_metrics["spearman"] + config.min_improvement
            ),
            "selection_requires_effective_optimizer_step": training_objective
            in RANK_ONLY_OBJECTIVES,
            "selection_requires_deployment_gate": deployment_objective is not None,
            "fixed_prior": fixed_prior,
            "stopped_reason": stopped_reason,
        },
        "prediction": unscaled_prediction,
        "training_objective": training_objective,
        "deployment_objective": (
            deployment_objective.metadata()
            if deployment_objective is not None
            else None
        ),
        "deployment_validation": selected_validation_deployment,
        "soft_rank_temperature": soft_rank_temperature,
        "residual_gate": {
            "alpha": best_residual_gate,
            "grid": list(residual_gate_grid),
            "selection_split": "validation",
            "fallback_to_fixed_prior_available": 0.0 in residual_gate_grid,
        },
        "history": history,
        "model_state": model_state,
        "validation_snapshot_catalog": (
            _validation_snapshot_catalog(
                dataset,
                training_objective,
                soft_rank_temperature,
                residual_gate_grid,
                validation_snapshot_entries,
                (
                    deployment_objective.metadata()
                    if deployment_objective is not None
                    else None
                ),
            )
            if validation_snapshot_dir is not None
            else None
        ),
    }


def _forward_batch(
    model: BidirectionalNBFNet,
    tensors: dict[str, torch.Tensor],
    prototype_ids: torch.Tensor,
    region_indices: torch.Tensor,
) -> torch.Tensor:
    return model(
        tensors["node_features"],
        tensors["edge_source"],
        tensors["edge_target"],
        tensors["edge_features"],
        tensors["origin_fields"][prototype_ids],
        tensors["destination_fields"][prototype_ids],
        tensors["region_nodes"][region_indices],
        tensors["region_features"][region_indices],
        tensors["forward_degree"],
        tensors["reverse_degree"],
    )


def _predict_weighted(
    model: BidirectionalNBFNet,
    tensors: dict[str, torch.Tensor],
    region_indices: torch.Tensor,
    prototype_batch_size: int,
    precision_policy: PrecisionPolicy,
) -> torch.Tensor:
    aggregate = torch.zeros(len(region_indices), device=region_indices.device)
    prototype_count = tensors["prototype_weight"].size(0)
    for prototype_slice in iter_slices(prototype_count, prototype_batch_size):
        prototype_ids = torch.arange(
            prototype_slice.start,
            prototype_slice.stop,
            device=region_indices.device,
        )
        with _autocast_context(region_indices.device, precision_policy):
            prediction = _forward_batch(model, tensors, prototype_ids, region_indices)
        aggregate += (prediction.float() * tensors["prototype_weight"][prototype_ids, None]).sum(dim=0)
    fixed_prior = tensors.get("fixed_prior")
    if fixed_prior is not None:
        aggregate = aggregate + fixed_prior[region_indices]
    return aggregate


def _autocast_context(
    device: torch.device,
    precision_policy: PrecisionPolicy,
):
    return torch.autocast(
        device_type=device.type,
        dtype=precision_policy.autocast_dtype,
        enabled=precision_policy.autocast_enabled,
    )


def _sampled_pairwise_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    generator: torch.Generator,
) -> torch.Tensor:
    pair_count = max(1, len(prediction))
    left = torch.randint(
        len(prediction),
        (pair_count,),
        generator=generator,
        device=prediction.device,
    )
    right = torch.randint(
        len(prediction),
        (pair_count,),
        generator=generator,
        device=prediction.device,
    )
    valid = target[left] != target[right]
    left = left[valid]
    right = right[valid]
    if not len(left):
        return prediction.sum() * 0.0
    sign = torch.sign(target[left] - target[right])
    margin = sign * (prediction[left] - prediction[right])
    return functional.softplus(-margin).mean()


def _evaluation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    config: NBFNetConfig,
    training_objective: str,
    soft_rank_temperature: float = 0.1,
) -> tuple[float, float, float]:
    huber = functional.huber_loss(
        prediction,
        target,
        reduction="mean",
        delta=config.huber_delta,
    )
    rank = (
        _soft_spearman_loss_tensor(prediction, target, soft_rank_temperature)
        if training_objective in {"soft_spearman", "deployment_aware"}
        else _full_pairwise_loss_tensor(prediction, target)
    )
    total = (
        rank
        if training_objective in RANK_ONLY_OBJECTIVES
        else huber + config.rank_weight * rank
    )
    return float(total.item()), float(huber.item()), float(rank.item())


def _full_pairwise_loss_tensor(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    left, right = _ordered_pair_indices(target)
    if not len(left):
        return prediction.sum() * 0.0
    sign = torch.sign(target[left] - target[right])
    margin = sign * (prediction[left] - prediction[right])
    return functional.softplus(-margin).mean()


def _soft_spearman_loss_tensor(
    prediction: torch.Tensor,
    target: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """Scale-invariant differentiable approximation of one minus Spearman."""

    if temperature <= 0.0:
        raise ValueError("soft-rank temperature must be positive")
    if prediction.ndim != 1 or target.shape != prediction.shape:
        raise ValueError("prediction and target must be aligned one-dimensional tensors")
    if len(prediction) < 2:
        return prediction.sum() * 0.0
    centered_prediction = prediction - prediction.mean()
    prediction_scale = centered_prediction.square().mean().sqrt().clamp_min(1.0e-6)
    standardized_prediction = centered_prediction / prediction_scale
    soft_ranks = torch.sigmoid(
        (standardized_prediction[:, None] - standardized_prediction[None, :])
        / temperature
    ).sum(dim=1)
    target_ranks = _average_ranks(target).to(dtype=prediction.dtype)
    centered_soft_ranks = soft_ranks - soft_ranks.mean()
    centered_target_ranks = target_ranks - target_ranks.mean()
    denominator = (
        centered_soft_ranks.square().mean().sqrt()
        * centered_target_ranks.square().mean().sqrt()
    ).clamp_min(1.0e-6)
    correlation = (
        centered_soft_ranks * centered_target_ranks
    ).mean() / denominator
    return 1.0 - correlation.clamp(-1.0, 1.0)


def _soft_topk_weights(
    prediction: torch.Tensor,
    top_k: int,
    temperature: float,
) -> torch.Tensor:
    if prediction.ndim != 1 or not len(prediction):
        raise ValueError("soft Top-K prediction must be a non-empty vector")
    if top_k <= 0 or top_k > len(prediction):
        raise ValueError("soft Top-K must be in [1, candidate_count]")
    if temperature <= 0.0:
        raise ValueError("soft Top-K temperature must be positive")
    if top_k == len(prediction):
        return torch.ones_like(prediction)
    centered = prediction - prediction.mean()
    standardized = centered / centered.square().mean().sqrt().clamp_min(1.0e-6)
    threshold = torch.topk(standardized.detach(), top_k).values[-1]
    raw_weights = torch.sigmoid((standardized - threshold) / temperature)
    scale = float(top_k) / raw_weights.sum().detach().clamp_min(1.0e-6)
    return raw_weights * scale


def _deployment_aware_loss_tensor(
    *,
    prediction: torch.Tensor,
    target: torch.Tensor,
    shortcut_cost: torch.Tensor,
    conflict_matrix: torch.Tensor,
    top_k: int,
    temperature: float,
    soft_budget: float,
    topk_gain_weight: float,
    budget_weight: float,
    conflict_weight: float,
    soft_rank_temperature: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if prediction.ndim != 1 or target.shape != prediction.shape:
        raise ValueError("deployment prediction and target must be aligned vectors")
    if shortcut_cost.shape != prediction.shape:
        raise ValueError("shortcut costs must align with deployment prediction")
    if conflict_matrix.shape != (len(prediction), len(prediction)):
        raise ValueError("conflict matrix must align with deployment prediction")
    if soft_budget <= 0.0:
        raise ValueError("soft shortcut budget must be positive")
    if any(weight < 0.0 for weight in (topk_gain_weight, budget_weight, conflict_weight)):
        raise ValueError("deployment loss weights must be non-negative")
    weights = _soft_topk_weights(prediction, top_k, temperature)
    weight_sum = weights.sum().clamp_min(1.0e-6)
    global_rank_loss = _soft_spearman_loss_tensor(
        prediction,
        target,
        soft_rank_temperature,
    )
    soft_topk_gain = (weights * target).sum() / weight_sum
    soft_shortcut_cost = (weights * shortcut_cost).sum()
    shortcut_budget_ratio = soft_shortcut_cost / soft_budget
    budget_penalty = 0.05 * functional.softplus(
        (shortcut_budget_ratio - 1.0) / 0.05
    )
    conflicting_pair_mass = 0.5 * torch.dot(
        weights,
        torch.mv(conflict_matrix, weights),
    )
    all_pair_mass = (0.5 * weight_sum * (weight_sum - 1.0)).clamp_min(1.0)
    conflict_penalty = conflicting_pair_mass / all_pair_mass
    total = (
        global_rank_loss
        - topk_gain_weight * soft_topk_gain
        + budget_weight * budget_penalty
        + conflict_weight * conflict_penalty
    )
    return total, {
        "global_rank_loss": global_rank_loss,
        "soft_topk_gain": soft_topk_gain,
        "soft_shortcut_cost": soft_shortcut_cost,
        "shortcut_budget_ratio": shortcut_budget_ratio,
        "budget_penalty": budget_penalty,
        "conflict_penalty": conflict_penalty,
    }


def _hard_disjoint_deployment_metrics(
    *,
    prediction: np.ndarray,
    target: np.ndarray,
    region_nodes: np.ndarray,
    shortcut_cost: np.ndarray,
    top_k: int,
) -> dict[str, float | int]:
    selected = select_region_indices(
        prediction,
        region_nodes,
        top_k,
        "hard_disjoint",
    )
    if len(selected) != top_k:
        raise ValueError("hard-disjoint deployment could not fill the requested budget")
    return {
        "selected_count": int(len(selected)),
        "mean_gain": float(np.mean(target[selected])),
        "shortcut_count": int(round(float(np.sum(shortcut_cost[selected])))),
    }


def _validation_deployment_metrics(
    *,
    prediction: torch.Tensor,
    dataset: DemandFieldDataset,
    shortcut_cost: torch.Tensor,
    objective: DeploymentObjective,
) -> dict[str, float | int | bool]:
    mask = dataset.split_mask("validation")
    prediction_values = prediction.detach().float().cpu().numpy()
    metrics = _hard_disjoint_deployment_metrics(
        prediction=prediction_values,
        target=dataset.labels[mask],
        region_nodes=dataset.region_nodes[mask],
        shortcut_cost=shortcut_cost.detach().float().cpu().numpy()[mask],
        top_k=objective.validation_top_k,
    )
    spearman = regression_metrics(prediction_values, dataset.labels[mask])["spearman"]
    metrics.update(
        {
            "gain_delta_vs_z0": float(metrics["mean_gain"])
            - objective.validation_z0_gain,
            "shortcut_delta_vs_z0": float(metrics["shortcut_count"])
            - objective.validation_hard_budget,
            "budget_feasible": float(metrics["shortcut_count"])
            <= objective.validation_hard_budget + 1.0e-6,
            "spearman": spearman,
            "spearman_delta_vs_z0": spearman - objective.validation_z0_spearman,
            "spearman_safe": spearman
            >= objective.validation_z0_spearman - objective.spearman_tolerance,
        }
    )
    return metrics


def _attach_deployment_history(
    row: dict[str, float | int | str],
    train_components: dict[str, torch.Tensor] | None,
    validation_components: dict[str, torch.Tensor] | None,
    validation_deployment: dict[str, float | int | bool] | None,
) -> None:
    if (
        train_components is None
        or validation_components is None
        or validation_deployment is None
    ):
        raise ValueError("deployment history requires complete components")
    for prefix, components in (
        ("train", train_components),
        ("validation", validation_components),
    ):
        for name, value in components.items():
            row[f"{prefix}_{name}"] = float(value.detach().item())
    for name, value in validation_deployment.items():
        row[f"validation_hard_{name}"] = (
            int(value) if isinstance(value, bool) else value
        )


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    less = (values[:, None] > values[None, :]).sum(dim=1)
    equal = (values[:, None] == values[None, :]).sum(dim=1)
    return less.to(torch.float32) + (equal.to(torch.float32) - 1.0) * 0.5


def _apply_residual_gate(
    prediction: torch.Tensor,
    fixed_prior: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    if prediction.shape != fixed_prior.shape:
        raise ValueError("prediction and fixed prior shapes do not match")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("residual gate alpha must be in [0, 1]")
    return fixed_prior + alpha * (prediction - fixed_prior)


def _select_residual_gate(
    prediction: torch.Tensor,
    fixed_prior: torch.Tensor | None,
    target: np.ndarray,
    grid: tuple[float, ...],
) -> tuple[float, torch.Tensor, float]:
    metrics = _residual_gate_metrics(prediction, fixed_prior, target, grid)
    raw_spearman = regression_metrics(
        prediction.detach().float().cpu().numpy(),
        target,
    )["spearman"]
    best = max(metrics, key=lambda row: (row["spearman"], -row["alpha"]))
    best_alpha = float(best["alpha"])
    best_prediction = (
        prediction
        if fixed_prior is None
        else _apply_residual_gate(prediction, fixed_prior, best_alpha)
    )
    return best_alpha, best_prediction, raw_spearman


def _residual_gate_metrics(
    prediction: torch.Tensor,
    fixed_prior: torch.Tensor | None,
    target: np.ndarray,
    grid: tuple[float, ...],
    k_values: tuple[int, ...] = (5, 10, 18),
) -> list[dict[str, object]]:
    if fixed_prior is None and grid != (1.0,):
        raise ValueError("multiple residual gates require a fixed prior")
    rows: list[dict[str, object]] = []
    for alpha in grid:
        candidate = (
            prediction
            if fixed_prior is None
            else _apply_residual_gate(prediction, fixed_prior, alpha)
        )
        values = candidate.detach().float().cpu().numpy()
        regression = regression_metrics(values, target)
        ranking = ranking_metrics_at_k(values, target, k_values)
        rows.append(
            {
                "alpha": float(alpha),
                "spearman": regression["spearman"],
                "ndcg_at_k": {
                    key: float(metrics["ndcg"]) for key, metrics in ranking.items()
                },
                "top_gain_at_k": {
                    key: float(metrics["mean_gain"])
                    for key, metrics in ranking.items()
                },
            }
        )
    return rows


def _full_pairwise_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    return float(_full_pairwise_loss_tensor(prediction, target).item())


def _full_pairwise_accuracy(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> float:
    left, right = _ordered_pair_indices(target)
    if not len(left):
        return float("nan")
    target_sign = torch.sign(target[left] - target[right])
    prediction_sign = torch.sign(prediction[left] - prediction[right])
    return float((target_sign == prediction_sign).float().mean().item())


def _ordered_pair_indices(target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    left, right = torch.triu_indices(
        len(target),
        len(target),
        offset=1,
        device=target.device,
    )
    valid = target[left] != target[right]
    return left[valid], right[valid]


def _prediction_statistics(prediction: torch.Tensor) -> dict[str, float]:
    values = prediction.detach().float()
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def _diagnostic_history_row(
    *,
    epoch: int,
    train_loss: float,
    train_huber: float,
    train_rank: float,
    train_full_pairwise_loss: float,
    train_pairwise_accuracy: float,
    train_prediction: torch.Tensor,
    validation_loss: float,
    validation_huber: float,
    validation_rank: float,
    validation_pairwise_accuracy: float,
    validation_spearman: float,
    validation_prediction: torch.Tensor,
    scaler_scale_before: float,
    scaler_scale_after: float,
    optimizer_state_step_before: int,
    optimizer_state_step_after: int,
    optimizer_step_skipped: bool,
    optimizer_step_effective: bool,
    gradient_norm_before_clip: float,
    gradient_norm_after_clip: float,
    parameter_delta_norm: float,
    learning_rate: float,
    next_learning_rate: float,
) -> dict[str, float | int]:
    train_stats = _prediction_statistics(train_prediction)
    validation_stats = _prediction_statistics(validation_prediction)
    return {
        "epoch": epoch,
        "train_loss": train_loss,
        "train_huber": train_huber,
        "train_rank": train_rank,
        "train_full_pairwise_loss": train_full_pairwise_loss,
        "train_pairwise_accuracy": train_pairwise_accuracy,
        "train_pre_step_prediction_mean": train_stats["mean"],
        "train_pre_step_prediction_std": train_stats["std"],
        "train_pre_step_prediction_min": train_stats["min"],
        "train_pre_step_prediction_max": train_stats["max"],
        "validation_loss": validation_loss,
        "validation_huber": validation_huber,
        "validation_rank": validation_rank,
        "validation_pairwise_accuracy": validation_pairwise_accuracy,
        "validation_spearman": validation_spearman,
        "validation_post_step_prediction_mean": validation_stats["mean"],
        "validation_post_step_prediction_std": validation_stats["std"],
        "validation_post_step_prediction_min": validation_stats["min"],
        "validation_post_step_prediction_max": validation_stats["max"],
        "scaler_scale_before": scaler_scale_before,
        "scaler_scale_after": scaler_scale_after,
        "optimizer_state_step_before": optimizer_state_step_before,
        "optimizer_state_step_after": optimizer_state_step_after,
        "optimizer_step_skipped": int(optimizer_step_skipped),
        "optimizer_step_effective": int(optimizer_step_effective),
        "gradient_norm_before_clip": gradient_norm_before_clip,
        "gradient_norm_after_clip": gradient_norm_after_clip,
        "parameter_delta_norm": parameter_delta_norm,
        "learning_rate": learning_rate,
        "next_learning_rate": next_learning_rate,
    }


def _set_training_scope(model: BidirectionalNBFNet, scope: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(scope == "all")
    if scope == "output_head":
        output_layer = model.prediction_head[-1]
        for parameter in output_layer.parameters():
            parameter.requires_grad_(True)
    elif scope != "all":
        raise ValueError(f"unsupported training scope: {scope}")


def _build_learning_rate_scheduler(
    optimizer: torch.optim.Optimizer,
    name: str,
    *,
    factor: float,
    patience: int,
    threshold: float,
    min_lr: float,
):
    if name == "none":
        return None
    if name != "reduce_on_plateau":
        raise ValueError(f"unsupported learning-rate scheduler: {name}")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=factor,
        patience=patience,
        threshold=threshold,
        threshold_mode="abs",
        min_lr=min_lr,
    )


def _trainable_parameter_count(model: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _maximum_consecutive_matches(
    rows: list[dict[str, float | int | str]],
    predicate,
) -> int:
    maximum = 0
    current = 0
    for row in rows:
        current = current + 1 if predicate(row) else 0
        maximum = max(maximum, current)
    return maximum


def _unrecovered_validation_loss_doublings(
    history: list[dict[str, float | int | str]],
) -> int:
    losses = [float(row["validation_loss"]) for row in history]
    unrecovered = 0
    for index in range(1, len(losses)):
        previous = losses[index - 1]
        if previous <= 0.0 or losses[index] <= 0.0:
            continue
        if losses[index] >= 2.0 * previous and not any(
            later <= previous for later in losses[index + 1 :]
        ):
            unrecovered += 1
    return unrecovered


def _global_gradient_norm(model: torch.nn.Module) -> float:
    total = torch.zeros((), device=next(model.parameters()).device)
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += parameter.grad.detach().float().square().sum()
    return float(torch.sqrt(total).item())


def _parameter_snapshot(model: torch.nn.Module) -> tuple[torch.Tensor, ...]:
    return tuple(
        parameter.detach().clone()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def _parameter_delta_norm(
    before: tuple[torch.Tensor, ...],
    model: torch.nn.Module,
) -> float:
    parameters = tuple(
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    if len(before) != len(parameters):
        raise ValueError("parameter snapshot does not match model")
    total = torch.zeros((), device=next(model.parameters()).device)
    for old_value, parameter in zip(before, parameters):
        difference = parameter.detach().float() - old_value.float()
        total += difference.square().sum()
    return float(torch.sqrt(total).item())


def _optimizer_state_step(optimizer: torch.optim.Optimizer) -> int:
    steps: list[int] = []
    for state in optimizer.state.values():
        value = state.get("step")
        if value is None:
            continue
        steps.append(int(value.item()) if torch.is_tensor(value) else int(value))
    return max(steps, default=0)


def _optimizer_step_was_skipped(step_before: int, step_after: int) -> bool:
    if step_after < step_before or step_after > step_before + 1:
        raise ValueError("optimizer step counter changed unexpectedly")
    return step_after == step_before


def _unscale_prediction(
    prediction: torch.Tensor, scalers: dict[str, np.ndarray | float]
) -> np.ndarray:
    return (
        prediction.detach().float().cpu().numpy() * float(scalers["label_scale"])
        + float(scalers["label_mean"])
    )


def _all_split_metrics(
    dataset: DemandFieldDataset,
    prediction: np.ndarray,
    *,
    split_names: tuple[str, ...] = SPLIT_NAMES,
) -> dict[str, dict]:
    metrics: dict[str, dict] = {}
    for name in split_names:
        mask = dataset.split_mask(name)
        split_prediction = prediction[mask]
        split_target = dataset.labels[mask]
        values = regression_metrics(split_prediction, split_target)
        values["ranking_at_k"] = ranking_metrics_at_k(
            split_prediction,
            split_target,
            (5, 10, 18),
            region_nodes=dataset.region_nodes[mask],
        )
        metrics[name] = values
    return metrics


def _architecture_metadata(config: NBFNetConfig) -> dict[str, object]:
    propagation_only = config.variant in PROPAGATION_ONLY_VARIANTS
    doubling = config.variant in DOUBLING_PROPAGATION_VARIANTS
    structure = config.resolved_propagation_structure()
    readout_depths = BidirectionalNBFNet._readout_depths(config)
    return {
        "propagation_only": propagation_only,
        "uses_direct_region_features": not propagation_only,
        "uses_layer_zero_readout": 0 in readout_depths,
        "propagation_structure": structure,
        "propagation_residual_scale": config.propagation_residual_scale,
        "residual_propagation": structure in {"g1", "g2", "g3"},
        "strict_identity_path": structure in {"g2", "g3"},
        "doubling_scale_readout": doubling,
        "readout_depths": list(readout_depths),
        "maximum_hop_depth": config.propagation_layers,
        "materializes_multi_hop_edges": False,
    }


def _write_history(
    path: Path,
    history: list[dict[str, float | int | str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _write_predictions(path: Path, dataset: DemandFieldDataset, prediction: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("region_id", "split", "label_avg_workload_gain", "prediction"))
        for index, region_id in enumerate(dataset.region_ids):
            if not np.isfinite(prediction[index]):
                continue
            writer.writerow(
                (
                    int(region_id),
                    SPLIT_NAMES[int(dataset.split[index])],
                    f"{dataset.labels[index]:.9f}",
                    f"{prediction[index]:.9f}",
                )
            )


def _save_checkpoint(
    path: Path,
    run: dict,
    config: NBFNetConfig,
    scalers: dict[str, np.ndarray | float],
    dataset: DemandFieldDataset,
) -> None:
    torch.save(
        {
            "schema": EXPERIMENT_SCHEMA,
            "model_state": run.pop("model_state"),
            "config": asdict(config),
            "numerics": run["diagnostics"]["numerics"],
            "dataset_sha256": dataset.manifest["dataset_sha256"],
            "candidate_sha256": dataset.manifest["candidate_sha256"],
            "scalers": scalers,
            "training_objective": run["training_objective"],
            "soft_rank_temperature": run["soft_rank_temperature"],
            "residual_gate": run["residual_gate"],
            "deployment_objective": run.get("deployment_objective"),
            "deployment_validation": run.get("deployment_validation"),
        },
        path,
    )


def _record_validation_snapshot(
    *,
    snapshot_dir: Path | None,
    entries: list[dict[str, object]],
    epoch: int,
    model: BidirectionalNBFNet,
    dataset: DemandFieldDataset,
    config: NBFNetConfig,
    scalers: dict[str, np.ndarray | float],
    precision_policy: PrecisionPolicy,
    training_objective: str,
    soft_rank_temperature: float,
    fixed_prior_name: str,
    prediction: torch.Tensor,
    fixed_prior: torch.Tensor | None,
    target: np.ndarray,
    residual_gate_grid: tuple[float, ...],
    validation_loss: float,
    optimizer_step_effective: bool,
    deployment_objective: dict[str, object] | None = None,
    deployment_validation: dict[str, float | int | bool] | None = None,
) -> None:
    if snapshot_dir is None:
        return
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / f"epoch_{epoch:03d}.pt"
    torch.save(
        {
            "schema": EXPERIMENT_SCHEMA,
            "model_state": {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            },
            "config": asdict(config),
            "numerics": precision_policy.metadata(),
            "dataset_sha256": dataset.manifest["dataset_sha256"],
            "candidate_sha256": dataset.manifest["candidate_sha256"],
            "scalers": scalers,
            "training_objective": training_objective,
            "soft_rank_temperature": soft_rank_temperature,
            "fixed_prior": fixed_prior_name,
            "deployment_objective": deployment_objective,
            "validation_snapshot": {
                "epoch": epoch,
                "optimizer_step_effective": optimizer_step_effective,
                "residual_gate_grid": list(residual_gate_grid),
            },
        },
        snapshot_path,
    )
    entries.append(
        {
            "epoch": epoch,
            "snapshot": str(Path("validation_snapshots") / snapshot_path.name),
            "validation_loss": validation_loss,
            "optimizer_step_effective": optimizer_step_effective,
            "deployment_validation": deployment_validation,
            "gates": _residual_gate_metrics(
                prediction,
                fixed_prior,
                target,
                residual_gate_grid,
            ),
        }
    )
    catalog = _validation_snapshot_catalog(
        dataset,
        training_objective,
        soft_rank_temperature,
        residual_gate_grid,
        entries,
        deployment_objective,
    )
    (snapshot_dir.parent / "validation_snapshot_catalog.json").write_text(
        json.dumps(catalog, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _validation_snapshot_catalog(
    dataset: DemandFieldDataset,
    training_objective: str,
    soft_rank_temperature: float,
    residual_gate_grid: tuple[float, ...],
    entries: list[dict[str, object]],
    deployment_objective: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "schema": "aic.gnn_v2.validation_snapshot_catalog.v1",
        "dataset_sha256": dataset.manifest["dataset_sha256"],
        "candidate_sha256": dataset.manifest["candidate_sha256"],
        "selection_data": "validation_only",
        "holdout_evaluated": False,
        "training_objective": training_objective,
        "soft_rank_temperature": soft_rank_temperature,
        "deployment_objective": deployment_objective,
        "residual_gate_grid": list(residual_gate_grid),
        "ranking_k_values": [5, 10, 18],
        "entries": entries,
    }


def _aggregate_runs(runs: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for split_name in runs[0]["metrics"]:
        result[split_name] = {}
        for metric_name in ("mae", "huber", "spearman", "ndcg_at_k", "top_k_mean_gain"):
            values = [run["metrics"][split_name][metric_name] for run in runs]
            result[split_name][metric_name] = {
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values),
                "min": min(values),
                "max": max(values),
            }
    return result


def _render_report(summary: dict) -> str:
    reported_split = (
        "holdout" if "holdout" in summary["aggregate"] else "validation"
    )
    reported = summary["aggregate"][reported_split]
    split_label = "Holdout" if reported_split == "holdout" else "Validation"
    selection_text = (
        "validation 严格非重叠 Top-K 收益、Z0 shortcut 预算与全局排序安全门"
        if summary["training_objective"] == "deployment_aware"
        else "验证集 Spearman"
    )
    return "\n".join(
        (
            "# OD 条件化双向 NBFNet 训练结果",
            "",
            f"- 实验变体：`{summary['variant']}`",
            f"- 训练目标：`{summary['training_objective']}`",
            f"- 固定先验：`{summary['fixed_prior']['name']}`",
            f"- 数据摘要：`{summary['dataset_sha256']}`",
            f"- 候选摘要：`{summary['candidate_sha256']}`",
            f"- 选定种子：`{summary['selected_seed']}`（仅按{selection_text}选择）",
            f"- {split_label} Spearman：`{reported['spearman']['mean']:.4f} ± {reported['spearman']['std']:.4f}`",
            f"- {split_label} NDCG@K：`{reported['ndcg_at_k']['mean']:.4f} ± {reported['ndcg_at_k']['std']:.4f}`",
            f"- {split_label} Top-K 收益：`{reported['top_k_mean_gain']['mean']:.3f} ± {reported['top_k_mean_gain']['std']:.3f}`",
            "",
            (
                "Holdout 已按协议锁定，本报告只包含 train/validation。"
                if reported_split == "validation"
                else "这是同一 H→Y 内的候选泛化结果，不代替冻结未来时间窗口测试。"
            ),
            "",
        )
    )


def _parse_int_list(value: str, option: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise ValueError(f"{option} must be a comma-separated integer list") from error
    if not values or len(set(values)) != len(values):
        raise ValueError(f"{option} must contain unique integers")
    return values


def _parse_float_grid(value: str, option: str) -> tuple[float, ...]:
    try:
        values = tuple(
            sorted(float(item.strip()) for item in value.split(",") if item.strip())
        )
    except ValueError as error:
        raise ValueError(f"{option} must be a comma-separated float list") from error
    if (
        not values
        or len(set(values)) != len(values)
        or any(not 0.0 <= item <= 1.0 for item in values)
    ):
        raise ValueError(f"{option} must contain unique values in [0, 1]")
    return values


def _resolve_precision_policy(
    precision: str | None,
    no_mixed_precision: bool,
    grad_scaler_init_scale: float,
) -> PrecisionPolicy:
    if no_mixed_precision and precision not in {None, "fp32"}:
        raise ValueError(
            "--no-mixed-precision cannot be combined with a non-fp32 --precision"
        )
    policy = PrecisionPolicy(
        mode="fp32" if no_mixed_precision else precision or "fp16",
        grad_scaler_init_scale=grad_scaler_init_scale,
    )
    policy.validate()
    return policy


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes:d}m{seconds:02d}s"
    return f"{seconds:d}s"


if __name__ == "__main__":
    main()
