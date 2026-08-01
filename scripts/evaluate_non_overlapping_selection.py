"""Evaluate W3 diversity and hard-disjoint selection under frozen Z0 scores."""

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
from src.region_selection import (
    SELECTION_METHODS,
    select_region_indices,
    selection_overlap_statistics,
)


DEFAULT_DATASET = ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.npz"
DEFAULT_DATASET_MANIFEST = (
    ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.json"
)
DEFAULT_Z0_PREDICTIONS = (
    ROOT_DIR
    / "results"
    / "gnn_v2"
    / "nbfnet_propagation"
    / "train_free_baselines"
    / "predictions.csv"
)
DEFAULT_CURRENT_LABELS = ROOT_DIR / "results" / "gnn_v2" / "region_training_labels.csv"
DEFAULT_FUTURE_LABELS = (
    ROOT_DIR / "results" / "gnn_v2" / "future_window_z0" / "region_future_labels.csv"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "gnn_v2" / "non_overlapping_selection"
K_VALUES = (5, 10, 18)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare direct, diverse, and hard-disjoint Z0 region sets."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--dataset-manifest", type=Path, default=DEFAULT_DATASET_MANIFEST
    )
    parser.add_argument("--z0-predictions", type=Path, default=DEFAULT_Z0_PREDICTIONS)
    parser.add_argument("--current-labels", type=Path, default=DEFAULT_CURRENT_LABELS)
    parser.add_argument("--future-labels", type=Path, default=DEFAULT_FUTURE_LABELS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_demand_field_dataset(args.dataset, args.dataset_manifest)
    z0_scores = _load_column(
        args.z0_predictions, dataset.region_ids, "z0_score"
    )
    current_rows = _load_rows(args.current_labels, dataset.region_ids)
    future_rows = _load_rows(args.future_labels, dataset.region_ids)
    current_labels = np.asarray(
        [float(current_rows[int(region_id)]["avg_workload_gain"]) for region_id in dataset.region_ids]
    )
    future_labels = np.asarray(
        [float(future_rows[int(region_id)]["avg_workload_gain"]) for region_id in dataset.region_ids]
    )
    _validate_geometry(current_rows, future_rows, dataset.region_ids)

    scopes = {"all_candidates": np.arange(len(dataset.region_ids), dtype=np.int64)}
    scopes.update(
        {
            name: np.flatnonzero(dataset.split_mask(name)).astype(np.int64)
            for name in SPLIT_NAMES
        }
    )
    results: dict[str, dict[str, dict]] = {}
    selection_rows: list[dict[str, object]] = []
    for scope_name, scope_indices in scopes.items():
        results[scope_name] = {}
        for method in SELECTION_METHODS:
            results[scope_name][method] = {}
            for k in K_VALUES:
                local_selected = select_region_indices(
                    z0_scores[scope_indices],
                    dataset.region_nodes[scope_indices],
                    k,
                    method,
                )
                selected = scope_indices[local_selected]
                metrics = _selection_metrics(
                    selected=selected,
                    dataset=dataset,
                    scores=z0_scores,
                    current_labels=current_labels,
                    future_labels=future_labels,
                    current_rows=current_rows,
                )
                results[scope_name][method][str(k)] = metrics
                for rank, index in enumerate(selected, start=1):
                    selection_rows.append(
                        {
                            "scope": scope_name,
                            "method": method,
                            "requested_k": k,
                            "rank": rank,
                            "region_id": int(dataset.region_ids[index]),
                            "z0_score": f"{float(z0_scores[index]):.9f}",
                            "current_gain": f"{float(current_labels[index]):.9f}",
                            "future_gain": f"{float(future_labels[index]):.9f}",
                        }
                    )
                print(
                    f"scope={scope_name} method={method} k={k} "
                    f"selected={metrics['selected_count']} "
                    f"deployable={metrics['overlap']['deployable_without_region_overlap']} "
                    f"current_gain={metrics['current_mean_individual_region_gain']:.3f} "
                    f"future_gain={metrics['future_mean_individual_region_gain']:.3f}",
                    flush=True,
                )

    summary = {
        "schema": "aic.gnn_v2.non_overlapping_selection.v1",
        "dataset_sha256": dataset.manifest["dataset_sha256"],
        "candidate_sha256": dataset.manifest["candidate_sha256"],
        "source_sha256": {
            "dataset": _sha256(args.dataset),
            "dataset_manifest": _sha256(args.dataset_manifest),
            "z0_predictions": _sha256(args.z0_predictions),
            "current_labels": _sha256(args.current_labels),
            "future_labels": _sha256(args.future_labels),
        },
        "protocol": {
            "score": "frozen label-free Z0",
            "k_values": list(K_VALUES),
            "methods": list(SELECTION_METHODS),
            "jaccard_penalty": (
                "0.5 * global score-rank quality + 0.5 * (1 - maximum pairwise Jaccard)"
            ),
            "marginal_coverage": (
                "0.5 * global score-rank quality + 0.5 * newly covered node fraction"
            ),
            "deployment_rule": (
                "hard_disjoint is fixed before label inspection because the compression "
                "index rejects any shared region node; relaxed methods are diagnostics only"
            ),
            "individual_label_warning": (
                "mean region gains are single-region labels and are not additive combined-index gains"
            ),
        },
        "results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_selections(args.output_dir / "selections.csv", selection_rows)
    (args.output_dir / "report.md").write_text(
        _render_report(summary), encoding="utf-8"
    )
    print(f"summary={args.output_dir / 'summary.json'}", flush=True)


def _load_rows(path: Path, region_ids: np.ndarray) -> dict[int, dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as file:
        rows = {int(row["region_id"]): row for row in csv.DictReader(file)}
    if set(rows) != set(map(int, region_ids)):
        raise ValueError(f"region rows do not align: {path}")
    return rows


def _load_column(path: Path, region_ids: np.ndarray, column: str) -> np.ndarray:
    rows = _load_rows(path, region_ids)
    values = np.asarray([float(rows[int(region_id)][column]) for region_id in region_ids])
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite {column} in {path}")
    return values


def _validate_geometry(
    current_rows: dict[int, dict[str, str]],
    future_rows: dict[int, dict[str, str]],
    region_ids: np.ndarray,
) -> None:
    errors: list[str] = []
    for region_id in map(int, region_ids):
        current = current_rows[region_id]
        future = future_rows[region_id]
        for column in ("node_count", "internal_node_count", "shortcut_count"):
            if int(current[column]) != int(future[column]):
                errors.append(f"region {region_id} changed {column}")
        if float(current["correctness_rate"]) != 1.0:
            errors.append(f"region {region_id} current result is inexact")
        if float(future["correctness_rate"]) != 1.0:
            errors.append(f"region {region_id} future result is inexact")
    if errors:
        raise ValueError("invalid selection inputs: " + "; ".join(errors[:10]))


def _selection_metrics(
    *,
    selected: np.ndarray,
    dataset,
    scores: np.ndarray,
    current_labels: np.ndarray,
    future_labels: np.ndarray,
    current_rows: dict[int, dict[str, str]],
) -> dict[str, object]:
    selected_region_ids = [int(dataset.region_ids[index]) for index in selected]
    overlap = selection_overlap_statistics(dataset.region_nodes[selected])
    shortcut_count = sum(
        int(current_rows[region_id]["shortcut_count"]) for region_id in selected_region_ids
    )
    internal_node_count = sum(
        int(current_rows[region_id]["internal_node_count"])
        for region_id in selected_region_ids
    )
    return {
        "selected_count": len(selected_region_ids),
        "selected_region_ids": selected_region_ids,
        "mean_z0_score": float(np.mean(scores[selected])) if len(selected) else 0.0,
        "current_mean_individual_region_gain": (
            float(np.mean(current_labels[selected])) if len(selected) else 0.0
        ),
        "future_mean_individual_region_gain": (
            float(np.mean(future_labels[selected])) if len(selected) else 0.0
        ),
        "current_sum_individual_region_gain": float(np.sum(current_labels[selected])),
        "future_sum_individual_region_gain": float(np.sum(future_labels[selected])),
        "summed_internal_node_count": internal_node_count,
        "summed_single_region_shortcut_count": shortcut_count,
        "overlap": overlap,
    }


def _write_selections(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _render_report(summary: dict) -> str:
    lines = [
        "# Z0 非重叠集合选择",
        "",
        "`hard_disjoint` 因压缩索引的零共享节点约束而预先固定为可部署规则；其他方法只用于"
        "量化排序质量与覆盖多样性的权衡。单区域收益标签不可相加，组合索引真实收益留给 W4。",
        "",
        "## 全候选集合",
        "",
        "| 方法 | K | 可部署 | 唯一节点 | 冗余 | 重叠对 | 当前单区域平均收益 | 未来单区域平均收益 | shortcuts 合计 |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in SELECTION_METHODS:
        for k in K_VALUES:
            result = summary["results"]["all_candidates"][method][str(k)]
            overlap = result["overlap"]
            lines.append(
                f"| {method} | {k} | "
                f"{'是' if overlap['deployable_without_region_overlap'] else '否'} | "
                f"{overlap['unique_node_count']} | "
                f"{overlap['membership_redundancy']:.3f} | "
                f"{overlap['overlapping_pair_count']} | "
                f"{result['current_mean_individual_region_gain']:.3f} | "
                f"{result['future_mean_individual_region_gain']:.3f} | "
                f"{result['summed_single_region_shortcut_count']} |"
            )
    lines.extend(
        [
            "",
            "## Validation 与 Holdout（K=18）",
            "",
            "| Scope | 方法 | 可部署 | 唯一节点 | 当前平均收益 | 未来平均收益 |",
            "| --- | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for scope in ("validation", "holdout"):
        for method in SELECTION_METHODS:
            result = summary["results"][scope][method]["18"]
            overlap = result["overlap"]
            lines.append(
                f"| {scope} | {method} | "
                f"{'是' if overlap['deployable_without_region_overlap'] else '否'} | "
                f"{overlap['unique_node_count']} | "
                f"{result['current_mean_individual_region_gain']:.3f} | "
                f"{result['future_mean_individual_region_gain']:.3f} |"
            )
    lines.append("")
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
