"""Evaluate the frozen H=[0,.35) Z0 ranking on the F=[.70,1) window."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.demand_field_data import SPLIT_NAMES, load_demand_field_dataset
from src.demand_field_model import regression_metrics, ranking_metrics_at_k
from src.region_labels import LABEL_SCHEMA, LABEL_WORK_DEFINITION


DEFAULT_DATASET = ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.npz"
DEFAULT_DATASET_MANIFEST = (
    ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.json"
)
DEFAULT_CURRENT_LABEL_MANIFEST = (
    ROOT_DIR / "results" / "gnn_v2" / "label_manifest.json"
)
DEFAULT_Z0_PREDICTIONS = (
    ROOT_DIR
    / "results"
    / "gnn_v2"
    / "nbfnet_propagation"
    / "train_free_baselines"
    / "predictions.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT_DIR / "results" / "gnn_v2" / "future_window_z0"
)
DEFAULT_FUTURE_LABELS = DEFAULT_OUTPUT_DIR / "region_future_labels.csv"
DEFAULT_FUTURE_LABEL_MANIFEST = DEFAULT_OUTPUT_DIR / "future_label_manifest.json"

FUTURE_START_FRACTION = 0.70
FUTURE_END_FRACTION = 1.00
FORMAL_QUERY_COUNT = 2_000
FORMAL_QUERY_SAMPLE_SEED = 42
K_VALUES = (5, 10, 18)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST
    )
    parser.add_argument(
        "--current-label-manifest",
        type=Path,
        default=DEFAULT_CURRENT_LABEL_MANIFEST,
    )
    parser.add_argument(
        "--z0-predictions", type=Path, default=DEFAULT_Z0_PREDICTIONS
    )
    parser.add_argument("--future-labels", type=Path, default=DEFAULT_FUTURE_LABELS)
    parser.add_argument(
        "--future-label-manifest",
        type=Path,
        default=DEFAULT_FUTURE_LABEL_MANIFEST,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_demand_field_dataset(args.dataset, args.dataset_manifest)
    current_manifest = json.loads(
        args.current_label_manifest.read_text(encoding="utf-8")
    )
    future_manifest = json.loads(
        args.future_label_manifest.read_text(encoding="utf-8")
    )
    future_rows = _load_future_labels(args.future_labels)
    z0_rows = _load_z0_predictions(args.z0_predictions)
    future_labels, z0_prediction = _validate_and_align(
        dataset=dataset,
        current_manifest=current_manifest,
        future_manifest=future_manifest,
        future_rows=future_rows,
        z0_rows=z0_rows,
    )

    current_metrics = _all_scope_metrics(dataset, z0_prediction, dataset.labels)
    future_metrics = _all_scope_metrics(dataset, z0_prediction, future_labels)
    persistence_metrics = _all_scope_metrics(dataset, dataset.labels, future_labels)
    summary = {
        "schema": "aic.gnn_v2.z0_future_window.v1",
        "dataset_sha256": dataset.manifest["dataset_sha256"],
        "candidate_sha256": dataset.manifest["candidate_sha256"],
        "protocol": {
            "method": "z0_frozen_zero_shot",
            "z0_history_window": dataset.manifest["history_window"],
            "current_label_window": {
                "start_fraction": current_manifest["label_start_fraction"],
                "end_fraction": current_manifest["label_end_fraction"],
                "query_sample_count": len(current_manifest["query_ids"]),
                "query_sample_seed": current_manifest.get("query_sample_seed"),
            },
            "future_label_window": {
                "start_fraction": future_manifest["label_start_fraction"],
                "end_fraction": future_manifest["label_end_fraction"],
                "query_sample_count": len(future_manifest["query_ids"]),
                "query_sample_seed": future_manifest.get("query_sample_seed"),
            },
            "future_labels_used_to_fit_or_orient_z0": False,
            "candidate_pool_frozen": True,
            "endpoint_cache_capacity": 0,
            "work_definition": LABEL_WORK_DEFINITION,
            "primary_scope": "all_candidates",
            "secondary_scopes": list(SPLIT_NAMES),
            "ranking_k": list(K_VALUES),
            "holdout_used_for_selection": False,
        },
        "source_sha256": {
            "dataset": _sha256(args.dataset),
            "dataset_manifest": _sha256(args.dataset_manifest),
            "current_label_manifest": _sha256(args.current_label_manifest),
            "z0_predictions": _sha256(args.z0_predictions),
            "future_labels": _sha256(args.future_labels),
            "future_label_manifest": _sha256(args.future_label_manifest),
        },
        "temporal_isolation": {
            "history_current_query_overlap": len(
                set(map(int, dataset.history_query_ids))
                & set(map(int, current_manifest["query_ids"]))
            ),
            "history_future_query_overlap": len(
                set(map(int, dataset.history_query_ids))
                & set(map(int, future_manifest["query_ids"]))
            ),
            "current_future_query_overlap": len(
                set(map(int, current_manifest["query_ids"]))
                & set(map(int, future_manifest["query_ids"]))
            ),
        },
        "z0_current_window_metrics": current_metrics,
        "z0_future_window_metrics": future_metrics,
        "current_label_as_future_predictor_metrics": persistence_metrics,
        "top_k_overlap": _top_k_overlap(z0_prediction, dataset.labels, future_labels),
        "future_label_drift": _label_drift(dataset.labels, future_labels),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_predictions(
        args.output_dir / "predictions.csv",
        dataset.region_ids,
        dataset.split,
        dataset.labels,
        future_labels,
        z0_prediction,
    )
    (args.output_dir / "report.md").write_text(
        _render_report(summary),
        encoding="utf-8",
    )
    overall = future_metrics["all_candidates"]
    print(
        "future Z0 "
        f"spearman={overall['spearman']:.6f} "
        f"ndcg5={overall['ranking_at_k']['5']['ndcg']:.6f} "
        f"ndcg10={overall['ranking_at_k']['10']['ndcg']:.6f} "
        f"ndcg18={overall['ranking_at_k']['18']['ndcg']:.6f}",
        flush=True,
    )
    print(f"summary={_display_path(args.output_dir / 'summary.json')}", flush=True)


def _load_future_labels(path: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            region_id = int(row["region_id"])
            if region_id in rows:
                raise ValueError(f"duplicate future region label: {region_id}")
            rows[region_id] = row
    if not rows:
        raise ValueError("future label CSV is empty")
    return rows


def _load_z0_predictions(path: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            region_id = int(row["region_id"])
            if region_id in rows:
                raise ValueError(f"duplicate Z0 prediction: {region_id}")
            rows[region_id] = row
    if not rows:
        raise ValueError("Z0 prediction CSV is empty")
    return rows


def _validate_and_align(
    *,
    dataset,
    current_manifest: dict,
    future_manifest: dict,
    future_rows: dict[int, dict[str, str]],
    z0_rows: dict[int, dict[str, str]],
) -> tuple[np.ndarray, np.ndarray]:
    errors: list[str] = []
    region_ids = [int(value) for value in dataset.region_ids]
    region_id_set = set(region_ids)
    if future_manifest.get("schema") != LABEL_SCHEMA:
        errors.append("future label schema mismatch")
    if future_manifest.get("status") != "complete":
        errors.append("future label manifest is not complete")
    if future_manifest.get("candidate_sha256") != dataset.manifest.get(
        "candidate_sha256"
    ):
        errors.append("future candidate digest mismatch")
    if future_manifest.get("label_start_fraction") != FUTURE_START_FRACTION:
        errors.append("future label start fraction mismatch")
    if future_manifest.get("label_end_fraction") != FUTURE_END_FRACTION:
        errors.append("future label end fraction mismatch")
    future_query_ids = [int(value) for value in future_manifest.get("query_ids", [])]
    if len(future_query_ids) != FORMAL_QUERY_COUNT:
        errors.append("future query sample count mismatch")
    if len(future_query_ids) != len(set(future_query_ids)):
        errors.append("future query ids are duplicated")
    if future_manifest.get("query_sample_seed") != FORMAL_QUERY_SAMPLE_SEED:
        errors.append("future query sample seed mismatch")
    if future_manifest.get("work_definition") != LABEL_WORK_DEFINITION:
        errors.append("future work definition mismatch")
    if future_manifest.get("endpoint_cache_capacity") != 0:
        errors.append("future endpoint cache must be disabled")
    if future_manifest.get("target_region_count") != len(region_ids):
        errors.append("future target candidate count mismatch")
    if future_manifest.get("completed_region_count") != len(region_ids):
        errors.append("future completed candidate count mismatch")
    if set(map(int, future_manifest.get("target_region_ids", []))) != region_id_set:
        errors.append("future target candidate ids mismatch")
    if set(map(int, future_manifest.get("completed_region_ids", []))) != region_id_set:
        errors.append("future completed candidate ids mismatch")
    if set(future_rows) != region_id_set:
        errors.append("future label rows do not match candidate ids")
    if set(z0_rows) != region_id_set:
        errors.append("Z0 prediction rows do not match candidate ids")
    if current_manifest.get("label_end_fraction") != FUTURE_START_FRACTION:
        errors.append("current and future label windows are not contiguous")
    history_ids = set(map(int, dataset.history_query_ids))
    current_query_ids = set(map(int, current_manifest.get("query_ids", [])))
    future_query_id_set = set(future_query_ids)
    if history_ids & current_query_ids:
        errors.append("history and current label queries overlap")
    if history_ids & future_query_id_set:
        errors.append("history and future label queries overlap")
    if current_query_ids & future_query_id_set:
        errors.append("current and future label queries overlap")

    future_labels = np.asarray(
        [float(future_rows[region_id]["avg_workload_gain"]) for region_id in region_ids],
        dtype=np.float64,
    )
    z0_prediction = np.asarray(
        [float(z0_rows[region_id]["z0_score"]) for region_id in region_ids],
        dtype=np.float64,
    )
    future_counts = {
        int(future_rows[region_id]["label_query_count"]) for region_id in region_ids
    }
    if future_counts != {FORMAL_QUERY_COUNT}:
        errors.append("future label query counts mismatch")
    if min(
        float(future_rows[region_id]["correctness_rate"]) for region_id in region_ids
    ) != 1.0:
        errors.append("future labels contain an inexact query result")
    saved_current = np.asarray(
        [float(z0_rows[region_id]["label"]) for region_id in region_ids],
        dtype=np.float64,
    )
    if not np.allclose(saved_current, dataset.labels, rtol=0.0, atol=1e-5):
        errors.append("Z0 prediction file does not match current dataset labels")
    if not np.isfinite(future_labels).all() or not np.isfinite(z0_prediction).all():
        errors.append("future labels or Z0 scores contain non-finite values")
    if errors:
        raise ValueError("invalid future-window inputs: " + "; ".join(errors))
    return future_labels, z0_prediction


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


def _top_k_overlap(
    z0_prediction: np.ndarray,
    current_labels: np.ndarray,
    future_labels: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    z0_order = np.argsort(-z0_prediction, kind="stable")
    current_order = np.argsort(-current_labels, kind="stable")
    future_order = np.argsort(-future_labels, kind="stable")
    result: dict[str, dict[str, float | int]] = {}
    for k in K_VALUES:
        z0 = set(map(int, z0_order[:k]))
        current = set(map(int, current_order[:k]))
        future = set(map(int, future_order[:k]))
        result[str(k)] = {
            "k": k,
            "z0_future_oracle_intersection": len(z0 & future),
            "z0_future_oracle_jaccard": _jaccard(z0, future),
            "current_future_oracle_intersection": len(current & future),
            "current_future_oracle_jaccard": _jaccard(current, future),
        }
    return result


def _label_drift(current: np.ndarray, future: np.ndarray) -> dict[str, float]:
    difference = future - current
    return {
        "current_mean": float(np.mean(current)),
        "future_mean": float(np.mean(future)),
        "mean_difference": float(np.mean(difference)),
        "mean_absolute_difference": float(np.mean(np.abs(difference))),
        "p95_absolute_difference": float(np.percentile(np.abs(difference), 95)),
        "spearman": regression_metrics(current, future)["spearman"],
    }


def _write_predictions(
    path: Path,
    region_ids: np.ndarray,
    split: np.ndarray,
    current_labels: np.ndarray,
    future_labels: np.ndarray,
    z0_prediction: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "region_id",
                "split",
                "current_label",
                "future_label",
                "future_minus_current",
                "z0_score",
            ]
        )
        for region_id, split_id, current, future, score in zip(
            region_ids, split, current_labels, future_labels, z0_prediction
        ):
            writer.writerow(
                [
                    int(region_id),
                    SPLIT_NAMES[int(split_id)],
                    f"{float(current):.9f}",
                    f"{float(future):.9f}",
                    f"{float(future - current):.9f}",
                    f"{float(score):.9f}",
                ]
            )


def _render_report(summary: dict) -> str:
    current = summary["z0_current_window_metrics"]["all_candidates"]
    future = summary["z0_future_window_metrics"]["all_candidates"]
    persistence = summary["current_label_as_future_predictor_metrics"][
        "all_candidates"
    ]
    lines = [
        "# Z0 冻结未来时间窗口验证",
        "",
        "Z0 仅由最早历史窗口 `H=[0,0.35)` 构造。本次直接在未来标签窗口 "
        "`F=[0.70,1.00)` 上零样本评估，不使用未来标签重新定向、拟合或选择参数。",
        "",
        "## 全候选结果",
        "",
        "| 指标 | 当前 Y=[0.35,0.70) | 未来 F=[0.70,1.00) |",
        "| --- | ---: | ---: |",
        f"| Spearman | {current['spearman']:.6f} | {future['spearman']:.6f} |",
    ]
    for k in K_VALUES:
        key = str(k)
        lines.append(
            f"| NDCG@{k} | {current['ranking_at_k'][key]['ndcg']:.6f} | "
            f"{future['ranking_at_k'][key]['ndcg']:.6f} |"
        )
        lines.append(
            f"| Top-{k} 平均收益 | "
            f"{current['ranking_at_k'][key]['mean_gain']:.3f} | "
            f"{future['ranking_at_k'][key]['mean_gain']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## 时间漂移参照",
            "",
            f"当前标签直接预测未来标签的 Spearman 为 `{persistence['spearman']:.6f}`。",
            "该值描述候选真实收益排序自身的时间稳定性，不使用任何模型。",
            "",
            "候选空间 train/validation/holdout 的同口径结果见 `summary.json`；"
            "holdout 未用于任何结构或参数选择。",
            "",
        ]
    )
    return "\n".join(lines)


def _jaccard(left: set[int], right: set[int]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
