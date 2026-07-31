"""Run short CUDA gradient anatomy experiments for demand propagation models."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.train_demand_field_nbfnet import (
    DEFAULT_DATASET,
    DEFAULT_DATASET_MANIFEST,
    PrecisionPolicy,
    _attach_fixed_prior,
    _forward_batch,
    _full_pairwise_accuracy,
    _full_pairwise_loss_tensor,
    _predict_weighted,
    _prepare_tensors,
    _unscale_prediction,
)
from src.demand_field_data import DemandFieldDataset, load_demand_field_dataset
from src.demand_field_model import regression_metrics
from src.demand_field_nbfnet import (
    PROPAGATION_ONLY_VARIANTS,
    BidirectionalNBFNet,
    NBFNetConfig,
    iter_slices,
)
from src.demand_field_torch_model import cuda_environment, require_cuda
from src.gradient_diagnostics import (
    DeviceTensorAccumulator,
    finite_tensor_statistics,
    parameter_gradient_report,
)


DEFAULT_OUTPUT_DIR = (
    ROOT_DIR / "results" / "gnn_v2" / "nbfnet_propagation" / "gradient_anatomy"
)
DIAGNOSTIC_SCHEMA = "aic.gnn_v2.gradient_anatomy.v1"
DIAGNOSTIC_MODES = ("snapshot", "head_only")
HOOK_DEPTH_MODES = ("none", "doubling", "all")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run bounded CUDA gradient diagnostics without long training."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mode", choices=DIAGNOSTIC_MODES, default="snapshot")
    parser.add_argument(
        "--variant",
        choices=sorted(PROPAGATION_ONLY_VARIANTS),
        default="propagation_doubling",
    )
    parser.add_argument("--fixed-prior", choices=("none", "z0"), default="none")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--prototype-batch-size", type=int, default=4)
    parser.add_argument("--loss-scale", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--head-only-steps", type=int, default=8)
    parser.add_argument("--warmup-output-head-steps", type=int, default=0)
    parser.add_argument("--hook-depths", choices=HOOK_DEPTH_MODES, default="doubling")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-gradient-checkpointing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _validate_args(args)
    device = require_cuda(args.device)
    precision_policy = PrecisionPolicy(mode="fp32", grad_scaler_init_scale=1.0)
    dataset = load_demand_field_dataset(args.dataset, args.dataset_manifest)
    config = NBFNetConfig(
        hidden_dim=args.hidden_dim,
        propagation_layers=args.layers,
        learning_rate=args.learning_rate,
        weight_decay=0.0,
        prototype_batch_size=args.prototype_batch_size,
        max_epochs=1,
        patience=1,
        mixed_precision=False,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        zero_initialize_prediction_head=args.fixed_prior == "z0",
        variant=args.variant,
    )
    config.validate()
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
    model = _build_model(dataset, config, args.seed, device)
    started = time.perf_counter()

    if args.mode == "head_only":
        result = _run_output_head_training(
            model=model,
            dataset=dataset,
            tensors=tensors,
            scalers=scalers,
            config=config,
            precision_policy=precision_policy,
            steps=args.head_only_steps,
            loss_scale=args.loss_scale,
        )
    else:
        warmup_history: list[dict[str, object]] = []
        if args.warmup_output_head_steps:
            warmup_history = _run_output_head_training(
                model=model,
                dataset=dataset,
                tensors=tensors,
                scalers=scalers,
                config=config,
                precision_policy=precision_policy,
                steps=args.warmup_output_head_steps,
                loss_scale=args.loss_scale,
            )["history"]
        _set_trainable_scope(model, "all")
        result = {
            "warmup_history": warmup_history,
            "snapshot": _run_gradient_snapshot(
                model=model,
                dataset=dataset,
                tensors=tensors,
                scalers=scalers,
                config=config,
                precision_policy=precision_policy,
                loss_scale=args.loss_scale,
                hook_depth_mode=args.hook_depths,
            ),
        }

    torch.cuda.synchronize(device)
    summary = {
        "schema": DIAGNOSTIC_SCHEMA,
        "mode": args.mode,
        "seed": args.seed,
        "loss_scale": args.loss_scale,
        "hook_depth_mode": args.hook_depths,
        "warmup_output_head_steps": args.warmup_output_head_steps,
        "head_only_steps": args.head_only_steps if args.mode == "head_only" else 0,
        "config": asdict(config),
        "fixed_prior": prior_metadata,
        "ablation": ablation_metadata,
        "dataset": {
            "digest": dataset.manifest["dataset_sha256"],
            "train_candidate_count": int(dataset.split_mask("train").sum()),
            "prototype_count": int(tensors["prototype_weight"].size(0)),
        },
        "environment": cuda_environment(device),
        "elapsed_seconds": time.perf_counter() - started,
        "result": result,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "summary.json"
    output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _print_result(summary, output_path)


def _validate_args(args: argparse.Namespace) -> None:
    if args.loss_scale <= 0.0:
        raise SystemExit("loss-scale must be positive")
    if args.layers <= 0 or args.hidden_dim <= 0 or args.prototype_batch_size <= 0:
        raise SystemExit("layers, hidden-dim and prototype-batch-size must be positive")
    if args.head_only_steps <= 0:
        raise SystemExit("head-only-steps must be positive")
    if args.warmup_output_head_steps < 0:
        raise SystemExit("warmup-output-head-steps must be non-negative")
    if args.fixed_prior == "z0" and args.layers != 32:
        raise SystemExit("the current Z0 prior is fixed to depths 1,2,4,8,16,32")


def _build_model(
    dataset: DemandFieldDataset,
    config: NBFNetConfig,
    seed: int,
    device: torch.device,
) -> BidirectionalNBFNet:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    return BidirectionalNBFNet(
        node_feature_dim=dataset.node_features.shape[1],
        region_feature_dim=dataset.region_features.shape[1],
        edge_type_count=len(dataset.manifest["road_types"]),
        config=config,
    ).to(device)


def _run_gradient_snapshot(
    *,
    model: BidirectionalNBFNet,
    dataset: DemandFieldDataset,
    tensors: dict[str, torch.Tensor],
    scalers: dict[str, np.ndarray | float],
    config: NBFNetConfig,
    precision_policy: PrecisionPolicy,
    loss_scale: float,
    hook_depth_mode: str,
) -> dict[str, object]:
    recorder = DeviceTensorAccumulator()
    hook_handles = _register_tensor_hooks(model, recorder, hook_depth_mode)
    try:
        backward_result = _backward_rank_first(
            model=model,
            dataset=dataset,
            tensors=tensors,
            config=config,
            precision_policy=precision_policy,
            loss_scale=loss_scale,
            recorder=recorder,
        )
    finally:
        for handle in hook_handles:
            handle.remove()
    validation_spearman = _validation_spearman(
        model,
        dataset,
        tensors,
        scalers,
        config,
        precision_policy,
    )
    backward_result["validation_spearman"] = validation_spearman
    backward_result["tensor_hooks"] = recorder.finalize()
    return backward_result


def _run_output_head_training(
    *,
    model: BidirectionalNBFNet,
    dataset: DemandFieldDataset,
    tensors: dict[str, torch.Tensor],
    scalers: dict[str, np.ndarray | float],
    config: NBFNetConfig,
    precision_policy: PrecisionPolicy,
    steps: int,
    loss_scale: float,
) -> dict[str, object]:
    _set_trainable_scope(model, "output_head")
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    history: list[dict[str, object]] = [
        {
            "step": 0,
            "validation_spearman": _validation_spearman(
                model,
                dataset,
                tensors,
                scalers,
                config,
                precision_policy,
            ),
        }
    ]
    for step in range(1, steps + 1):
        backward_result = _backward_rank_first(
            model=model,
            dataset=dataset,
            tensors=tensors,
            config=config,
            precision_policy=precision_policy,
            loss_scale=loss_scale,
            recorder=None,
        )
        gradient_summary = backward_result["parameter_gradients_unscaled"]["summary"]
        if not gradient_summary["all_gradient_elements_finite"]:
            history.append(
                {
                    "step": step,
                    "train_pairwise_loss": backward_result["train_pairwise_loss"],
                    "train_pairwise_accuracy": backward_result[
                        "train_pairwise_accuracy"
                    ],
                    "gradient_summary": gradient_summary,
                    "optimizer_step": "skipped_nonfinite_gradient",
                }
            )
            break
        torch.nn.utils.clip_grad_norm_(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            1.0,
        )
        optimizer.step()
        history.append(
            {
                "step": step,
                "train_pairwise_loss": backward_result["train_pairwise_loss"],
                "train_pairwise_accuracy": backward_result["train_pairwise_accuracy"],
                "score_gradient": backward_result["score_gradient"],
                "gradient_summary": gradient_summary,
                "optimizer_step": "applied",
                "validation_spearman": _validation_spearman(
                    model,
                    dataset,
                    tensors,
                    scalers,
                    config,
                    precision_policy,
                ),
            }
        )
    return {
        "trainable_scope": "final_linear_output_head_only",
        "history": history,
    }


def _backward_rank_first(
    *,
    model: BidirectionalNBFNet,
    dataset: DemandFieldDataset,
    tensors: dict[str, torch.Tensor],
    config: NBFNetConfig,
    precision_policy: PrecisionPolicy,
    loss_scale: float,
    recorder: DeviceTensorAccumulator | None,
) -> dict[str, object]:
    device = tensors["labels"].device
    train_indices = torch.as_tensor(
        np.flatnonzero(dataset.split_mask("train")),
        device=device,
        dtype=torch.long,
    )
    train_target = tensors["labels"][train_indices]
    model.train()
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        full_train_prediction = _predict_weighted(
            model,
            tensors,
            train_indices,
            config.prototype_batch_size,
            precision_policy,
        )
    prediction_leaf = full_train_prediction.detach().requires_grad_(True)
    train_loss = _full_pairwise_loss_tensor(prediction_leaf, train_target)
    score_gradient = torch.autograd.grad(train_loss, prediction_leaf)[0].detach()
    prototype_count = tensors["prototype_weight"].size(0)
    for prototype_slice in iter_slices(prototype_count, config.prototype_batch_size):
        prototype_ids = torch.arange(
            prototype_slice.start,
            prototype_slice.stop,
            device=device,
        )
        prediction = _forward_batch(model, tensors, prototype_ids, train_indices)
        weighted_contribution = (
            prediction * tensors["prototype_weight"][prototype_ids, None]
        ).sum(dim=0)
        if recorder is not None:
            _watch_tensor(recorder, "weighted_score", weighted_contribution)
        gradient_surrogate = (weighted_contribution * score_gradient).sum()
        (gradient_surrogate * loss_scale).backward()
    scaled_report = parameter_gradient_report(model)
    if loss_scale != 1.0:
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.div_(loss_scale)
        unscaled_report = parameter_gradient_report(model)
    else:
        unscaled_report = scaled_report
    return {
        "train_pairwise_loss": float(train_loss.item()),
        "train_pairwise_accuracy": _full_pairwise_accuracy(
            full_train_prediction,
            train_target,
        ),
        "score_prediction": finite_tensor_statistics(full_train_prediction),
        "score_gradient": finite_tensor_statistics(score_gradient),
        "parameter_gradients_scaled": scaled_report,
        "parameter_gradients_unscaled": unscaled_report,
    }


def _validation_spearman(
    model: BidirectionalNBFNet,
    dataset: DemandFieldDataset,
    tensors: dict[str, torch.Tensor],
    scalers: dict[str, np.ndarray | float],
    config: NBFNetConfig,
    precision_policy: PrecisionPolicy,
) -> float:
    validation_mask = dataset.split_mask("validation")
    validation_indices = torch.as_tensor(
        np.flatnonzero(validation_mask),
        device=tensors["labels"].device,
        dtype=torch.long,
    )
    model.eval()
    with torch.no_grad():
        prediction = _predict_weighted(
            model,
            tensors,
            validation_indices,
            config.prototype_batch_size,
            precision_policy,
        )
    unscaled_prediction = _unscale_prediction(prediction, scalers)
    return float(
        regression_metrics(
            unscaled_prediction,
            dataset.labels[validation_mask],
        )["spearman"]
    )


def _set_trainable_scope(model: BidirectionalNBFNet, scope: str) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(scope == "all")
    if scope == "output_head":
        output_layer = model.prediction_head[-1]
        for parameter in output_layer.parameters():
            parameter.requires_grad_(True)
    elif scope != "all":
        raise ValueError(f"unsupported trainable scope: {scope}")


def _register_tensor_hooks(
    model: BidirectionalNBFNet,
    recorder: DeviceTensorAccumulator,
    hook_depth_mode: str,
) -> list[torch.utils.hooks.RemovableHandle]:
    if hook_depth_mode == "none":
        selected_depths: set[int] = set()
    elif hook_depth_mode == "all":
        selected_depths = set(range(1, model.config.propagation_layers + 1))
    else:
        selected_depths = set(model.readout_depths)
    handles: list[torch.utils.hooks.RemovableHandle] = []
    for direction, layers in (
        ("origin", model.origin_propagation_layers),
        ("destination", model.destination_propagation_layers),
    ):
        for depth, layer in enumerate(layers, start=1):
            if depth not in selected_depths:
                continue
            handles.append(
                layer.register_forward_hook(
                    _make_forward_hook(recorder, f"{direction}.depth_{depth:02d}")
                )
            )
    handles.append(
        model.prediction_head.register_forward_pre_hook(
            _make_forward_pre_hook(recorder, "prediction_head.input")
        )
    )
    handles.append(
        model.prediction_head.register_forward_hook(
            _make_forward_hook(recorder, "prediction_head.output")
        )
    )
    return handles


def _make_forward_hook(
    recorder: DeviceTensorAccumulator,
    name: str,
) -> Callable:
    def hook(
        _module: torch.nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        _watch_tensor(recorder, name, output)

    return hook


def _make_forward_pre_hook(
    recorder: DeviceTensorAccumulator,
    name: str,
) -> Callable:
    def hook(
        _module: torch.nn.Module,
        inputs: tuple[torch.Tensor, ...],
    ) -> None:
        _watch_tensor(recorder, name, inputs[0])

    return hook


def _watch_tensor(
    recorder: DeviceTensorAccumulator,
    name: str,
    tensor: torch.Tensor,
) -> None:
    recorder.add(f"{name}.activation", tensor)
    if tensor.requires_grad:
        tensor.register_hook(
            lambda gradient: recorder.add(f"{name}.gradient", gradient)
        )


def _print_result(summary: dict[str, object], output_path: Path) -> None:
    result = summary["result"]
    if summary["mode"] == "snapshot":
        snapshot = result["snapshot"]
        gradient_summary = snapshot["parameter_gradients_unscaled"]["summary"]
        print(
            f"snapshot layers={summary['config']['propagation_layers']} "
            f"scale={summary['loss_scale']:.8g} "
            f"validation_spearman={snapshot['validation_spearman']:.4f} "
            f"all_finite={gradient_summary['all_gradient_elements_finite']} "
            f"nonfinite={gradient_summary['nonfinite_gradient_element_count']} "
            f"fp32_norm={gradient_summary['raw_l2_norm_fp32']} "
            f"fp64_norm={gradient_summary['finite_l2_norm_fp64']}",
            flush=True,
        )
    else:
        history = result["history"]
        print(
            f"head_only steps={len(history) - 1} "
            f"validation_spearman={history[0]['validation_spearman']:.4f}"
            f"->{history[-1].get('validation_spearman', float('nan')):.4f}",
            flush=True,
        )
    print(f"summary={output_path}", flush=True)


if __name__ == "__main__":
    main()
