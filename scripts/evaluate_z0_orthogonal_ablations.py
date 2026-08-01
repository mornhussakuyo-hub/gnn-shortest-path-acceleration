"""Run the frozen W2 orthogonal ablations for deterministic Z0 on CUDA."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.train_demand_field_nbfnet import _prepare_tensors
from src.demand_field_data import SPLIT_NAMES, load_demand_field_dataset
from src.demand_field_model import regression_metrics, ranking_metrics_at_k
from src.demand_field_nbfnet import iter_slices
from src.demand_field_torch_model import cuda_environment, require_cuda
from src.train_free_demand_field import (
    DEFAULT_DIFFUSION_DEPTHS,
    deterministic_diffusion_batch_scores,
)


DEFAULT_DATASET = ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.npz"
DEFAULT_DATASET_MANIFEST = (
    ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.json"
)
DEFAULT_CURRENT_LABEL_MANIFEST = (
    ROOT_DIR / "results" / "gnn_v2" / "label_manifest.json"
)
DEFAULT_FUTURE_DIR = ROOT_DIR / "results" / "gnn_v2" / "future_window_z0"
DEFAULT_FUTURE_LABELS = DEFAULT_FUTURE_DIR / "region_future_labels.csv"
DEFAULT_FUTURE_LABEL_MANIFEST = DEFAULT_FUTURE_DIR / "future_label_manifest.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "gnn_v2" / "z0_orthogonal_ablation"
K_VALUES = (5, 10, 18)
FORMAL_QUERY_COUNT = 2000


@dataclass(frozen=True, slots=True)
class AblationSpec:
    name: str
    category: str
    tensor_variant: str = "propagation_doubling"
    depths: tuple[int, ...] = DEFAULT_DIFFUSION_DEPTHS
    origin_weight: float = 1.0
    destination_weight: float = 1.0
    region_pooling: str = "mean_max"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate preregistered, label-free Z0 orthogonal ablations."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST
    )
    parser.add_argument(
        "--current-label-manifest", type=Path, default=DEFAULT_CURRENT_LABEL_MANIFEST
    )
    parser.add_argument("--future-labels", type=Path, default=DEFAULT_FUTURE_LABELS)
    parser.add_argument(
        "--future-label-manifest", type=Path, default=DEFAULT_FUTURE_LABEL_MANIFEST
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prototype-batch-size", type=int, default=8)
    parser.add_argument("--randomization-seed", type=int, default=20260730)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.prototype_batch_size <= 0:
        raise SystemExit("--prototype-batch-size must be positive")
    device = require_cuda(args.device)
    dataset = load_demand_field_dataset(args.dataset, args.dataset_manifest)
    future_labels, temporal_validation = _load_and_validate_future_labels(
        dataset=dataset,
        future_labels_path=args.future_labels,
        future_manifest_path=args.future_label_manifest,
        current_manifest_path=args.current_label_manifest,
    )
    specs = _build_ablation_specs()
    grouped_specs: dict[str, list[AblationSpec]] = {}
    for spec in specs:
        grouped_specs.setdefault(spec.tensor_variant, []).append(spec)

    variants: dict[str, dict[str, object]] = {}
    predictions: dict[str, np.ndarray] = {}
    for tensor_variant, tensor_specs in grouped_specs.items():
        tensors, _, tensor_metadata = _prepare_tensors(
            dataset,
            device,
            tensor_variant,
            args.randomization_seed,
        )
        for spec in tensor_specs:
            prediction = _predict(
                tensors=tensors,
                prototype_batch_size=args.prototype_batch_size,
                spec=spec,
            )
            predictions[spec.name] = prediction
            variants[spec.name] = {
                "spec": _serializable_spec(spec),
                "tensor_metadata": tensor_metadata,
                "current_metrics": _all_scope_metrics(
                    dataset, prediction, dataset.labels
                ),
                "future_metrics": _all_scope_metrics(
                    dataset, prediction, future_labels
                ),
            }
            current_holdout = variants[spec.name]["current_metrics"]["holdout"]
            future_all = variants[spec.name]["future_metrics"]["all_candidates"]
            print(
                f"variant={spec.name} "
                f"current_holdout={current_holdout['spearman']:.6f} "
                f"future_all={future_all['spearman']:.6f} "
                f"future_ndcg5={future_all['ranking_at_k']['5']['ndcg']:.6f}",
                flush=True,
            )
        del tensors
        torch.cuda.empty_cache()

    baseline = variants["z0_base"]
    for name, result in variants.items():
        result["delta_from_z0_base"] = _metric_deltas(result, baseline)

    summary = {
        "schema": "aic.gnn_v2.z0_orthogonal_ablation.v1",
        "execution": cuda_environment(device),
        "dataset_sha256": dataset.manifest["dataset_sha256"],
        "candidate_sha256": dataset.manifest["candidate_sha256"],
        "source_sha256": {
            "dataset": _sha256(args.dataset),
            "dataset_manifest": _sha256(args.dataset_manifest),
            "current_label_manifest": _sha256(args.current_label_manifest),
            "future_labels": _sha256(args.future_labels),
            "future_label_manifest": _sha256(args.future_label_manifest),
        },
        "protocol": {
            "method": "deterministic_label_free_z0_orthogonal_ablation",
            "base_depths": list(DEFAULT_DIFFUSION_DEPTHS),
            "base_direction": "origin_forward_plus_destination_reverse",
            "base_pooling": "region_mean_plus_region_max",
            "prototype_batch_size": args.prototype_batch_size,
            "randomization_seed": args.randomization_seed,
            "selection_policy": (
                "all variants are explanatory controls; no variant replaces Z0 "
                "using validation, holdout, or future labels"
            ),
        },
        "temporal_validation": temporal_validation,
        "variants": variants,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_predictions(
        args.output_dir / "predictions.csv",
        dataset,
        future_labels,
        specs,
        predictions,
    )
    (args.output_dir / "report.md").write_text(
        _render_report(summary, specs), encoding="utf-8"
    )
    print(f"summary={args.output_dir / 'summary.json'}", flush=True)


def _build_ablation_specs() -> tuple[AblationSpec, ...]:
    specs = [
        AblationSpec(name="z0_base", category="base"),
        AblationSpec(
            name="origin_only",
            category="direction",
            destination_weight=0.0,
        ),
        AblationSpec(
            name="destination_only",
            category="direction",
            origin_weight=0.0,
        ),
        AblationSpec(
            name="undirected",
            category="topology",
            tensor_variant="undirected",
        ),
        AblationSpec(
            name="degree_rewired",
            category="topology",
            tensor_variant="degree_rewired",
        ),
        AblationSpec(
            name="shuffled_od",
            category="od_coupling",
            tensor_variant="shuffled_od",
        ),
    ]
    specs.extend(
        AblationSpec(
            name=f"depth_{depth:02d}",
            category="depth",
            depths=(depth,),
        )
        for depth in DEFAULT_DIFFUSION_DEPTHS
    )
    specs.extend(
        (
            AblationSpec(
                name="pooling_mean",
                category="region_pooling",
                region_pooling="mean",
            ),
            AblationSpec(
                name="pooling_max",
                category="region_pooling",
                region_pooling="max",
            ),
        )
    )
    names = [spec.name for spec in specs]
    if len(names) != len(set(names)):
        raise ValueError("ablation names must be unique")
    return tuple(specs)


def _serializable_spec(spec: AblationSpec) -> dict[str, object]:
    values = asdict(spec)
    values["depths"] = list(spec.depths)
    return values


def _predict(
    *,
    tensors: dict[str, torch.Tensor],
    prototype_batch_size: int,
    spec: AblationSpec,
) -> np.ndarray:
    prediction = torch.zeros(
        tensors["region_nodes"].shape[0],
        device=tensors["region_nodes"].device,
        dtype=torch.float32,
    )
    with torch.inference_mode():
        for prototype_slice in iter_slices(
            tensors["prototype_weight"].size(0), prototype_batch_size
        ):
            scores = deterministic_diffusion_batch_scores(
                origin_fields=tensors["origin_fields"][prototype_slice],
                destination_fields=tensors["destination_fields"][prototype_slice],
                edge_source=tensors["edge_source"],
                edge_target=tensors["edge_target"],
                region_nodes=tensors["region_nodes"],
                receiver_normalizer_forward=tensors["forward_degree"],
                receiver_normalizer_reverse=tensors["reverse_degree"],
                depths=spec.depths,
                origin_weight=spec.origin_weight,
                destination_weight=spec.destination_weight,
                region_pooling=spec.region_pooling,
            )
            prediction += (
                scores * tensors["prototype_weight"][prototype_slice, None]
            ).sum(dim=0)
    result = prediction.float().cpu().numpy()
    if not np.isfinite(result).all():
        raise ValueError(f"{spec.name} produced non-finite predictions")
    return result


def _load_and_validate_future_labels(
    *,
    dataset,
    future_labels_path: Path,
    future_manifest_path: Path,
    current_manifest_path: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    future_manifest = json.loads(future_manifest_path.read_text(encoding="utf-8"))
    current_manifest = json.loads(current_manifest_path.read_text(encoding="utf-8"))
    with future_labels_path.open(encoding="utf-8", newline="") as file:
        rows = {int(row["region_id"]): row for row in csv.DictReader(file)}
    region_ids = set(map(int, dataset.region_ids))
    errors: list[str] = []
    if future_manifest.get("status") != "complete":
        errors.append("future manifest is not complete")
    if future_manifest.get("target_region_count") != len(dataset.region_ids):
        errors.append("future target count does not match dataset")
    if future_manifest.get("completed_region_count") != len(dataset.region_ids):
        errors.append("future completed count does not match dataset")
    if set(rows) != region_ids:
        errors.append("future label rows do not match candidate ids")
    future_query_ids = set(map(int, future_manifest.get("query_ids", [])))
    current_query_ids = set(map(int, current_manifest.get("query_ids", [])))
    history_query_ids = set(map(int, dataset.history_query_ids))
    if len(future_query_ids) != FORMAL_QUERY_COUNT:
        errors.append("future query sample count mismatch")
    if history_query_ids & current_query_ids:
        errors.append("history and current queries overlap")
    if history_query_ids & future_query_ids:
        errors.append("history and future queries overlap")
    if current_query_ids & future_query_ids:
        errors.append("current and future queries overlap")
    if rows:
        query_counts = {int(row["label_query_count"]) for row in rows.values()}
        correctness = {float(row["correctness_rate"]) for row in rows.values()}
        if query_counts != {FORMAL_QUERY_COUNT}:
            errors.append("future label query counts mismatch")
        if correctness != {1.0}:
            errors.append("future labels contain inexact query results")
    if errors:
        raise ValueError("invalid future-window inputs: " + "; ".join(errors))
    labels = np.asarray(
        [float(rows[int(region_id)]["avg_workload_gain"]) for region_id in dataset.region_ids],
        dtype=np.float64,
    )
    if not np.isfinite(labels).all():
        raise ValueError("future labels contain non-finite values")
    return labels, {
        "future_manifest_status": future_manifest["status"],
        "future_candidate_count": len(rows),
        "future_query_count": len(future_query_ids),
        "history_current_query_overlap": len(history_query_ids & current_query_ids),
        "history_future_query_overlap": len(history_query_ids & future_query_ids),
        "current_future_query_overlap": len(current_query_ids & future_query_ids),
        "all_future_distances_exact": True,
    }


def _all_scope_metrics(dataset, prediction: np.ndarray, target: np.ndarray) -> dict:
    masks = {"all_candidates": np.ones(len(target), dtype=bool)}
    masks.update({name: dataset.split_mask(name) for name in SPLIT_NAMES})
    result: dict[str, dict] = {}
    for name, mask in masks.items():
        values = regression_metrics(prediction[mask], target[mask])
        values["ranking_at_k"] = ranking_metrics_at_k(
            prediction[mask],
            target[mask],
            K_VALUES,
            region_nodes=dataset.region_nodes[mask],
        )
        result[name] = values
    return result


def _metric_deltas(result: dict, baseline: dict) -> dict[str, float]:
    return {
        "current_validation_spearman": (
            result["current_metrics"]["validation"]["spearman"]
            - baseline["current_metrics"]["validation"]["spearman"]
        ),
        "current_holdout_spearman": (
            result["current_metrics"]["holdout"]["spearman"]
            - baseline["current_metrics"]["holdout"]["spearman"]
        ),
        "future_all_spearman": (
            result["future_metrics"]["all_candidates"]["spearman"]
            - baseline["future_metrics"]["all_candidates"]["spearman"]
        ),
        **{
            f"future_all_ndcg_at_{k}": (
                result["future_metrics"]["all_candidates"]["ranking_at_k"][str(k)][
                    "ndcg"
                ]
                - baseline["future_metrics"]["all_candidates"]["ranking_at_k"][str(k)][
                    "ndcg"
                ]
            )
            for k in K_VALUES
        },
    }


def _write_predictions(
    path: Path,
    dataset,
    future_labels: np.ndarray,
    specs: tuple[AblationSpec, ...],
    predictions: dict[str, np.ndarray],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["region_id", "split", "current_label", "future_label"]
            + [spec.name for spec in specs]
        )
        for index, (region_id, split_id) in enumerate(
            zip(dataset.region_ids, dataset.split)
        ):
            writer.writerow(
                [
                    int(region_id),
                    SPLIT_NAMES[int(split_id)],
                    f"{float(dataset.labels[index]):.9f}",
                    f"{float(future_labels[index]):.9f}",
                ]
                + [f"{float(predictions[spec.name][index]):.9f}" for spec in specs]
            )


def _render_report(summary: dict, specs: tuple[AblationSpec, ...]) -> str:
    lines = [
        "# Z0 正交消融",
        "",
        "全部方法无参数、无标签，固定候选与数据切分；消融只解释 Z0 的有效来源，不使用 "
        "validation、holdout 或未来标签替换主方法。",
        "",
        "| 变体 | 类别 | 当前 Val Spearman | 当前 Holdout Spearman | 未来全候选 Spearman | "
        "未来 NDCG@5 | @10 | @18 |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for spec in specs:
        result = summary["variants"][spec.name]
        current = result["current_metrics"]
        future = result["future_metrics"]["all_candidates"]
        ranking = future["ranking_at_k"]
        lines.append(
            f"| {spec.name} | {spec.category} | "
            f"{current['validation']['spearman']:.6f} | "
            f"{current['holdout']['spearman']:.6f} | "
            f"{future['spearman']:.6f} | "
            f"{ranking['5']['ndcg']:.6f} | "
            f"{ranking['10']['ndcg']:.6f} | "
            f"{ranking['18']['ndcg']:.6f} |"
        )
    lines.extend(
        [
            "",
            "正式解释必须同时参考当前 validation/holdout 与冻结未来窗口；单个消融高于 Z0 不会"
            "触发事后替换主方法。",
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
