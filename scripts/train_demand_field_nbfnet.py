"""Train the base OD-conditioned bidirectional NBFNet on CUDA."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as functional


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.demand_field_data import SPLIT_NAMES, DemandFieldDataset, load_demand_field_dataset
from src.demand_field_model import regression_metrics
from src.demand_field_nbfnet import (
    BidirectionalNBFNet,
    NBFNetConfig,
    build_edge_features,
    build_receiver_normalizers,
    iter_slices,
)
from src.demand_field_torch_model import cuda_environment, require_cuda


DEFAULT_DATASET = ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.npz"
DEFAULT_DATASET_MANIFEST = ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "gnn_v2" / "nbfnet_base"
EXPERIMENT_SCHEMA = "aic.gnn_v2.od_conditioned_bidirectional_nbfnet.v1"


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
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--prototype-batch-size", type=int, default=4)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-mixed-precision", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_int_list(args.seeds, "--seeds")
    config = NBFNetConfig(
        hidden_dim=args.hidden_dim,
        propagation_layers=args.layers,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        huber_delta=args.huber_delta,
        prototype_batch_size=args.prototype_batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        mixed_precision=not args.no_mixed_precision,
    )
    config.validate()
    device = require_cuda(args.device)
    dataset = load_demand_field_dataset(args.dataset, args.dataset_manifest)
    tensors, scalers = _prepare_tensors(dataset, device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []

    for seed in seeds:
        print(f"training NBFNet seed={seed}", flush=True)
        run = _train_one_seed(dataset, tensors, scalers, config, seed, device)
        seed_dir = args.output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        _write_history(seed_dir / "training_history.csv", run.pop("history"))
        _write_predictions(
            seed_dir / "predictions.csv",
            dataset,
            run["prediction"],
        )
        _save_checkpoint(seed_dir / "model.pt", run, config, scalers, dataset)
        run.pop("prediction")
        runs.append(run)
        print(
            f"seed={seed} best_epoch={run['best_epoch']} "
            f"holdout_spearman={run['metrics']['holdout']['spearman']:.4f} "
            f"holdout_top_gain={run['metrics']['holdout']['top_k_mean_gain']:.3f}",
            flush=True,
        )

    selected_seed = max(runs, key=lambda run: run["metrics"]["validation"]["spearman"])[
        "seed"
    ]
    summary = {
        "schema": EXPERIMENT_SCHEMA,
        "model": "base_od_conditioned_bidirectional_nbfnet",
        "execution": cuda_environment(device),
        "dataset_sha256": dataset.manifest["dataset_sha256"],
        "candidate_sha256": dataset.manifest["candidate_sha256"],
        "config": asdict(config),
        "seeds": seeds,
        "selected_seed": selected_seed,
        "selection_rule": "highest validation Spearman; holdout is not used",
        "split": dataset.manifest["split"],
        "input_policy": dataset.manifest["model_input_policy"],
        "prototype_batching": {
            "prototype_count": int(dataset.prototype_weight.size),
            "prototype_batch_size": config.prototype_batch_size,
            "training_objective": (
                "Each prototype batch first averages scores using its frozen weights, "
                "then compares the batch-level region score with the region-value label. "
                "Evaluation averages all prototype scores using frozen prototype weights."
            ),
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
) -> tuple[dict[str, torch.Tensor], dict[str, np.ndarray | float]]:
    train_mask = dataset.split_mask("train")
    feature_mean = dataset.region_features[train_mask].mean(axis=0).astype(np.float32)
    feature_scale = dataset.region_features[train_mask].std(axis=0).astype(np.float32)
    feature_scale[feature_scale < 1e-6] = 1.0
    label_mean = float(dataset.labels[train_mask].mean())
    label_scale = float(dataset.labels[train_mask].std())
    if label_scale < 1e-6:
        label_scale = 1.0
    edge_type_count = len(dataset.manifest["road_types"])
    edge_source = torch.as_tensor(dataset.edge_source, device=device, dtype=torch.long)
    edge_target = torch.as_tensor(dataset.edge_target, device=device, dtype=torch.long)
    edge_length = torch.as_tensor(dataset.edge_length, device=device, dtype=torch.float32)
    edge_type = torch.as_tensor(dataset.edge_type, device=device, dtype=torch.long)
    prototype_origins, prototype_destinations = _build_prototype_fields(dataset, device)
    tensors = {
        "node_features": torch.as_tensor(dataset.node_features, device=device),
        "edge_source": edge_source,
        "edge_target": edge_target,
        "edge_features": build_edge_features(edge_length, edge_type, edge_type_count),
        "origin_fields": prototype_origins,
        "destination_fields": prototype_destinations,
        "prototype_weight": torch.as_tensor(dataset.prototype_weight, device=device),
        "region_nodes": torch.as_tensor(dataset.region_nodes, device=device, dtype=torch.long),
        "region_features": torch.as_tensor(
            (dataset.region_features - feature_mean) / feature_scale,
            device=device,
        ),
        "labels": torch.as_tensor(
            (dataset.labels - label_mean) / label_scale,
            device=device,
        ),
        "forward_degree": build_receiver_normalizers(
            edge_target, dataset.node_ids.size
        ),
        "reverse_degree": build_receiver_normalizers(
            edge_source, dataset.node_ids.size
        ),
    }
    return tensors, {
        "region_feature_mean": feature_mean,
        "region_feature_scale": feature_scale,
        "label_mean": label_mean,
        "label_scale": label_scale,
    }


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
) -> dict:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    model = BidirectionalNBFNet(
        node_feature_dim=dataset.node_features.shape[1],
        region_feature_dim=dataset.region_features.shape[1],
        edge_type_count=len(dataset.manifest["road_types"]),
        config=config,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    amp_enabled = config.mixed_precision and device.type == "cuda"
    gradient_scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    train_indices = torch.as_tensor(
        np.flatnonzero(dataset.split_mask("train")), device=device, dtype=torch.long
    )
    validation_indices = torch.as_tensor(
        np.flatnonzero(dataset.split_mask("validation")), device=device, dtype=torch.long
    )
    best_validation_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        prototype_order = torch.randperm(
            dataset.prototype_weight.size, generator=generator, device=device
        )
        train_loss_total = 0.0
        train_weight_total = 0.0
        for prototype_slice in iter_slices(len(prototype_order), config.prototype_batch_size):
            prototype_ids = prototype_order[prototype_slice]
            prototype_weight = tensors["prototype_weight"][prototype_ids]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                prediction = _forward_batch(model, tensors, prototype_ids, train_indices)
                target = tensors["labels"][train_indices].unsqueeze(0).expand_as(prediction)
                weighted_prediction = (prediction * prototype_weight[:, None]).sum(
                    dim=0
                ) / prototype_weight.sum()
                loss = functional.huber_loss(
                    weighted_prediction,
                    target[0],
                    reduction="mean",
                    delta=config.huber_delta,
                )
            gradient_scaler.scale(loss).backward()
            gradient_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            gradient_scaler.step(optimizer)
            gradient_scaler.update()
            train_loss_total += float(loss.detach() * prototype_weight.sum())
            train_weight_total += float(prototype_weight.sum())

        model.eval()
        with torch.no_grad():
            validation_prediction = _predict_weighted(
                model, tensors, validation_indices, config.prototype_batch_size, amp_enabled
            )
            validation_target = tensors["labels"][validation_indices]
            validation_loss = float(
                functional.huber_loss(
                    validation_prediction,
                    validation_target,
                    reduction="mean",
                    delta=config.huber_delta,
                ).item()
            )
            validation_unscaled = _unscale_prediction(validation_prediction, scalers)
            validation_metrics = regression_metrics(
                validation_unscaled,
                dataset.labels[dataset.split_mask("validation")],
            )
        history.append(
            {
                "epoch": float(epoch),
                "train_prototype_huber": train_loss_total / train_weight_total,
                "validation_huber": validation_loss,
                "validation_spearman": validation_metrics["spearman"],
            }
        )
        if validation_loss < best_validation_loss - config.min_improvement:
            best_validation_loss = validation_loss
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("NBFNet training did not produce a finite validation checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    with torch.no_grad():
        prediction = _predict_weighted(
            model,
            tensors,
            torch.arange(dataset.region_ids.size, device=device),
            config.prototype_batch_size,
            amp_enabled,
        )
    torch.cuda.synchronize(device)
    unscaled_prediction = _unscale_prediction(prediction, scalers)
    return {
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "training_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "metrics": {
            name: regression_metrics(
                unscaled_prediction[dataset.split_mask(name)],
                dataset.labels[dataset.split_mask(name)],
            )
            for name in SPLIT_NAMES
        },
        "prediction": unscaled_prediction,
        "history": history,
        "model_state": best_state,
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
    amp_enabled: bool,
) -> torch.Tensor:
    aggregate = torch.zeros(len(region_indices), device=region_indices.device)
    prototype_count = tensors["prototype_weight"].size(0)
    for prototype_slice in iter_slices(prototype_count, prototype_batch_size):
        prototype_ids = torch.arange(
            prototype_slice.start,
            prototype_slice.stop,
            device=region_indices.device,
        )
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
            prediction = _forward_batch(model, tensors, prototype_ids, region_indices)
        aggregate += (prediction.float() * tensors["prototype_weight"][prototype_ids, None]).sum(dim=0)
    return aggregate


def _unscale_prediction(
    prediction: torch.Tensor, scalers: dict[str, np.ndarray | float]
) -> np.ndarray:
    return (
        prediction.detach().float().cpu().numpy() * float(scalers["label_scale"])
        + float(scalers["label_mean"])
    )


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)


def _write_predictions(path: Path, dataset: DemandFieldDataset, prediction: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("region_id", "split", "label_avg_workload_gain", "prediction"))
        for index, region_id in enumerate(dataset.region_ids):
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
            "dataset_sha256": dataset.manifest["dataset_sha256"],
            "candidate_sha256": dataset.manifest["candidate_sha256"],
            "scalers": scalers,
        },
        path,
    )


def _aggregate_runs(runs: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    for split_name in SPLIT_NAMES:
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
    holdout = summary["aggregate"]["holdout"]
    return "\n".join(
        (
            "# OD 条件化双向 NBFNet 训练结果",
            "",
            f"- 数据摘要：`{summary['dataset_sha256']}`",
            f"- 候选摘要：`{summary['candidate_sha256']}`",
            f"- 选定种子：`{summary['selected_seed']}`（仅按验证集 Spearman 选择）",
            f"- Holdout Spearman：`{holdout['spearman']['mean']:.4f} ± {holdout['spearman']['std']:.4f}`",
            f"- Holdout NDCG@K：`{holdout['ndcg_at_k']['mean']:.4f} ± {holdout['ndcg_at_k']['std']:.4f}`",
            f"- Holdout Top-K 收益：`{holdout['top_k_mean_gain']['mean']:.3f} ± {holdout['top_k_mean_gain']['std']:.3f}`",
            "",
            "这是同一 H→Y 内的候选泛化结果，不代替冻结未来时间窗口测试。",
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


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT_DIR))
    except ValueError:
        return str(path)


if __name__ == "__main__":
    main()
