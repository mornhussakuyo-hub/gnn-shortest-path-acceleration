"""Evaluate deterministic diffusion and untrained random NBFNet baselines."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from scripts.train_demand_field_nbfnet import (
    PrecisionPolicy,
    _all_split_metrics,
    _parse_int_list,
    _predict_weighted,
    _prepare_tensors,
)
from src.demand_field_data import SPLIT_NAMES, load_demand_field_dataset
from src.demand_field_model import regression_metrics
from src.demand_field_nbfnet import BidirectionalNBFNet, NBFNetConfig, iter_slices
from src.demand_field_torch_model import cuda_environment, require_cuda
from src.train_free_demand_field import (
    DEFAULT_DIFFUSION_DEPTHS,
    deterministic_diffusion_batch_scores,
    orient_from_training_split,
)


DEFAULT_DATASET = ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.npz"
DEFAULT_MANIFEST = ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.json"
DEFAULT_NODE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路节点.csv"
DEFAULT_PROXY_PREDICTIONS = (
    ROOT_DIR / "results" / "gnn_v2" / "proxy_overlap_group_split" / "predictions.csv"
)
DEFAULT_OUTPUT_DIR = (
    ROOT_DIR / "results" / "gnn_v2" / "nbfnet_propagation" / "train_free_baselines"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Z0 deterministic diffusion and Z1 untrained NBFNet."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dataset-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--node-csv", type=Path, default=DEFAULT_NODE_CSV)
    parser.add_argument("--proxy-predictions", type=Path, default=DEFAULT_PROXY_PREDICTIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--layers", type=int, default=32)
    parser.add_argument("--prototype-batch-size", type=int, default=1)
    parser.add_argument("--z0-prototype-batch-size", type=int, default=8)
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--visualization-edge-stride", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_int_list(args.seeds, "--seeds")
    if args.layers != DEFAULT_DIFFUSION_DEPTHS[-1]:
        raise SystemExit(f"--layers must be {DEFAULT_DIFFUSION_DEPTHS[-1]} for this baseline")
    if min(args.prototype_batch_size, args.z0_prototype_batch_size) <= 0:
        raise SystemExit("prototype batch sizes must be positive")
    if args.visualization_edge_stride <= 0:
        raise SystemExit("--visualization-edge-stride must be positive")

    device = require_cuda(args.device)
    dataset = load_demand_field_dataset(args.dataset, args.dataset_manifest)
    tensors, _, _ = _prepare_tensors(dataset, device, "propagation_doubling")
    all_indices = torch.arange(len(dataset.labels), device=device, dtype=torch.long)
    train_mask = dataset.split_mask("train")
    train_indices = torch.as_tensor(np.flatnonzero(train_mask), device=device)

    z0_prediction = _deterministic_prediction(
        tensors,
        args.z0_prototype_batch_size,
    )
    proxy_prediction = _load_proxy_prediction(args.proxy_predictions, dataset.region_ids)
    z1_predictions: dict[int, np.ndarray] = {}
    z2_predictions: dict[int, np.ndarray] = {}
    runs: list[dict[str, object]] = []
    precision_policy = PrecisionPolicy(mode=args.precision)

    for seed in seeds:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        config = NBFNetConfig(
            hidden_dim=args.hidden_dim,
            propagation_layers=args.layers,
            prototype_batch_size=args.prototype_batch_size,
            variant="propagation_doubling",
        )
        model = BidirectionalNBFNet(
            node_feature_dim=dataset.node_features.shape[1],
            region_feature_dim=dataset.region_features.shape[1],
            edge_type_count=len(dataset.manifest["road_types"]),
            config=config,
        ).to(device).eval()
        with torch.inference_mode():
            raw_prediction = _predict_weighted(
                model,
                tensors,
                all_indices,
                args.prototype_batch_size,
                precision_policy,
            )
            oriented_train_prediction, sign = orient_from_training_split(
                raw_prediction[train_indices],
                tensors["labels"][train_indices],
            )
        raw_numpy = raw_prediction.float().cpu().numpy()
        oriented_numpy = raw_numpy * sign
        z1_predictions[seed] = raw_numpy
        z2_predictions[seed] = oriented_numpy
        raw_metrics = _all_split_metrics(dataset, raw_numpy)
        oriented_metrics = _all_split_metrics(dataset, oriented_numpy)
        runs.append(
            {
                "seed": seed,
                "train_selected_sign": sign,
                "z1_raw_metrics": raw_metrics,
                "z2_train_sign_metrics": oriented_metrics,
                "correlation": {
                    "z0": _split_correlations(dataset, raw_numpy, z0_prediction),
                    "midpoint_proxy": _split_correlations(
                        dataset, raw_numpy, proxy_prediction
                    ),
                },
            }
        )
        print(
            f"seed={seed} z1_validation={raw_metrics['validation']['spearman']:.4f} "
            f"z1_holdout={raw_metrics['holdout']['spearman']:.4f} sign={sign:+d} "
            f"z2_holdout={oriented_metrics['holdout']['spearman']:.4f}",
            flush=True,
        )
        del model, raw_prediction, oriented_train_prediction
        torch.cuda.empty_cache()

    summary = {
        "schema": "aic.gnn_v2.train_free_demand_field_baselines.v1",
        "execution": cuda_environment(device),
        "dataset_sha256": dataset.manifest["dataset_sha256"],
        "candidate_sha256": dataset.manifest["candidate_sha256"],
        "protocol": {
            "z0": (
                "deterministic scalar bidirectional fixed-mean diffusion; equal "
                "readout over depths 1,2,4,8,16,32; no learned parameters or labels"
            ),
            "z1": (
                "randomly initialized propagation_doubling forward only; signed score "
                "reported without seed selection or sign correction"
            ),
            "z2": (
                "same frozen Z1 prediction with one global sign selected using train "
                "labels only; validation and holdout are not used for orientation"
            ),
            "seeds": seeds,
            "precision": args.precision,
            "prototype_batch_size": args.prototype_batch_size,
        },
        "z0_metrics": _all_split_metrics(dataset, z0_prediction),
        "z0_proxy_correlation": _split_correlations(
            dataset, z0_prediction, proxy_prediction
        ),
        "runs": runs,
        "aggregate": _aggregate_runs(runs),
        "z1_cross_seed_spearman": _cross_seed_correlations(
            dataset, z1_predictions
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_predictions(
        args.output_dir / "predictions.csv",
        dataset,
        z0_prediction,
        proxy_prediction,
        z1_predictions,
        z2_predictions,
    )
    _write_score_maps(
        args.output_dir / "visualizations",
        dataset,
        args.node_csv,
        z0_prediction,
        z1_predictions,
        z2_predictions,
        summary,
        args.visualization_edge_stride,
    )
    (args.output_dir / "report.md").write_text(
        _render_report(summary),
        encoding="utf-8",
    )
    print(f"summary={args.output_dir / 'summary.json'}", flush=True)


def _deterministic_prediction(
    tensors: dict[str, torch.Tensor],
    prototype_batch_size: int,
) -> np.ndarray:
    prediction = torch.zeros(
        tensors["region_nodes"].shape[0],
        device=tensors["region_nodes"].device,
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
            )
            prediction += (
                scores * tensors["prototype_weight"][prototype_slice, None]
            ).sum(dim=0)
    return prediction.float().cpu().numpy()


def _load_proxy_prediction(path: Path, region_ids: np.ndarray) -> np.ndarray:
    by_region: dict[int, float] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            by_region[int(row["region_id"])] = float(row["proxy_score"])
    try:
        return np.asarray([by_region[int(region_id)] for region_id in region_ids])
    except KeyError as error:
        raise ValueError(f"proxy prediction missing region {error.args[0]}") from error


def _split_correlations(dataset, left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    return {
        name: regression_metrics(left[dataset.split_mask(name)], right[dataset.split_mask(name)])[
            "spearman"
        ]
        for name in SPLIT_NAMES
    }


def _cross_seed_correlations(dataset, predictions: dict[int, np.ndarray]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    seeds = sorted(predictions)
    for split_name in SPLIT_NAMES:
        mask = dataset.split_mask(split_name)
        result[split_name] = {
            str(left_seed): {
                str(right_seed): regression_metrics(
                    predictions[left_seed][mask], predictions[right_seed][mask]
                )["spearman"]
                for right_seed in seeds
            }
            for left_seed in seeds
        }
    return result


def _aggregate_runs(runs: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for protocol_key, metric_key in (
        ("z1_raw", "z1_raw_metrics"),
        ("z2_train_sign", "z2_train_sign_metrics"),
    ):
        for split_name in SPLIT_NAMES:
            values = [
                float(run[metric_key][split_name]["spearman"])
                for run in runs
            ]
            absolute = [abs(value) for value in values]
            result[f"{protocol_key}_{split_name}"] = {
                "mean_spearman": statistics.fmean(values),
                "std_spearman": statistics.pstdev(values),
                "mean_absolute_spearman": statistics.fmean(absolute),
                "min_spearman": min(values),
                "max_spearman": max(values),
            }
    return result


def _write_predictions(
    path: Path,
    dataset,
    z0_prediction: np.ndarray,
    proxy_prediction: np.ndarray,
    z1_predictions: dict[int, np.ndarray],
    z2_predictions: dict[int, np.ndarray],
) -> None:
    seeds = sorted(z1_predictions)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["region_id", "split", "label", "z0_score", "midpoint_proxy"]
            + [f"z1_seed_{seed}" for seed in seeds]
            + [f"z2_seed_{seed}" for seed in seeds]
        )
        for index, (region_id, split_id, label) in enumerate(
            zip(dataset.region_ids, dataset.split, dataset.labels)
        ):
            writer.writerow(
                [
                    int(region_id),
                    SPLIT_NAMES[int(split_id)],
                    f"{float(label):.9f}",
                    f"{float(z0_prediction[index]):.9f}",
                    f"{float(proxy_prediction[index]):.9f}",
                ]
                + [f"{float(z1_predictions[seed][index]):.9f}" for seed in seeds]
                + [f"{float(z2_predictions[seed][index]):.9f}" for seed in seeds]
            )


def _write_score_maps(
    output_dir: Path,
    dataset,
    node_csv: Path,
    z0_prediction: np.ndarray,
    z1_predictions: dict[int, np.ndarray],
    z2_predictions: dict[int, np.ndarray],
    summary: dict,
    edge_stride: int,
) -> None:
    coordinates = _load_coordinates(node_csv, dataset.node_ids)
    region_centroids = coordinates[dataset.region_nodes].mean(axis=1)
    panel_width = 440
    panel_height = 500
    title_height = 54
    margin = 18
    plot_width = panel_width - margin * 2
    plot_height = panel_height - title_height - margin
    minimum = coordinates.min(axis=0)
    maximum = coordinates.max(axis=0)
    span = np.maximum(maximum - minimum, 1e-9)

    def project(values: np.ndarray) -> np.ndarray:
        normalized = (values - minimum) / span
        projected = np.empty_like(normalized)
        projected[:, 0] = margin + normalized[:, 0] * plot_width
        projected[:, 1] = title_height + (1.0 - normalized[:, 1]) * plot_height
        return projected

    projected_nodes = project(coordinates)
    projected_regions = project(region_centroids)
    sampled_edges = np.arange(0, len(dataset.edge_source), edge_stride)
    road_segments = "".join(
        f"M{projected_nodes[source, 0]:.1f},{projected_nodes[source, 1]:.1f}"
        f"L{projected_nodes[target, 0]:.1f},{projected_nodes[target, 1]:.1f}"
        for source, target in zip(
            dataset.edge_source[sampled_edges],
            dataset.edge_target[sampled_edges],
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for run in summary["runs"]:
        seed = int(run["seed"])
        panels = (
            (
                "Z0 确定性扩散",
                z0_prediction,
                summary["z0_metrics"]["holdout"]["spearman"],
            ),
            (
                f"Z1 随机前向 seed {seed}",
                z1_predictions[seed],
                run["z1_raw_metrics"]["holdout"]["spearman"],
            ),
            (
                f"Z2 train 定向 ({run['train_selected_sign']:+d})",
                z2_predictions[seed],
                run["z2_train_sign_metrics"]["holdout"]["spearman"],
            ),
        )
        svg = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{panel_width * 3}" '
            f'height="{panel_height + 48}" viewBox="0 0 {panel_width * 3} {panel_height + 48}">',
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            '<defs>',
            f'<path id="roads" d="{road_segments}" fill="none" stroke="#c9ced6" '
            'stroke-width="0.42" stroke-opacity="0.48"/>',
            '</defs>',
        ]
        for panel_index, (title, prediction, holdout_spearman) in enumerate(panels):
            offset = panel_index * panel_width
            svg.extend(
                [
                    f'<g transform="translate({offset},0)">',
                    f'<text x="{panel_width / 2:.1f}" y="23" text-anchor="middle" '
                    'font-family="sans-serif" font-size="16" font-weight="600">'
                    f'{title}</text>',
                    f'<text x="{panel_width / 2:.1f}" y="43" text-anchor="middle" '
                    'font-family="sans-serif" font-size="12" fill="#475569">'
                    f'Holdout Spearman {holdout_spearman:.4f}</text>',
                    '<use href="#roads"/>',
                    *_region_circle_elements(projected_regions, prediction),
                    '</g>',
                ]
            )
        svg.extend(
            [
                '<text x="18" y="532" font-family="sans-serif" font-size="12" fill="#475569">'
                '蓝色=低分　白色=中位　红色=高分　黑圈=Top-18；坐标仅用于可视化，不是模型输入。</text>',
                '</svg>',
            ]
        )
        (output_dir / f"score_maps_seed_{seed}.svg").write_text(
            "\n".join(svg) + "\n",
            encoding="utf-8",
        )


def _load_coordinates(node_csv: Path, node_ids: np.ndarray) -> np.ndarray:
    by_node: dict[int, tuple[float, float]] = {}
    with node_csv.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            by_node[int(row["node_id"])] = (float(row["x_m"]), float(row["y_m"]))
    try:
        return np.asarray([by_node[int(node_id)] for node_id in node_ids], dtype=np.float64)
    except KeyError as error:
        raise ValueError(f"coordinate missing node {error.args[0]}") from error


def _region_circle_elements(
    projected_regions: np.ndarray,
    prediction: np.ndarray,
) -> list[str]:
    order = np.argsort(prediction, kind="stable")
    quantile = np.empty(len(prediction), dtype=np.float64)
    quantile[order] = np.linspace(0.0, 1.0, len(prediction))
    top_indices = set(order[-min(18, len(order)) :].tolist())
    circles: list[str] = []
    for index, ((x_value, y_value), rank_quantile) in enumerate(
        zip(projected_regions, quantile)
    ):
        color = _diverging_color(float(rank_quantile))
        if index in top_indices:
            circles.append(
                f'<circle cx="{x_value:.1f}" cy="{y_value:.1f}" r="3.7" '
                f'fill="{color}" fill-opacity="0.92" stroke="#111827" stroke-width="1.0"/>'
            )
        else:
            circles.append(
                f'<circle cx="{x_value:.1f}" cy="{y_value:.1f}" r="2.0" '
                f'fill="{color}" fill-opacity="0.76"/>'
            )
    return circles


def _diverging_color(quantile: float) -> str:
    low = np.asarray((49, 130, 189), dtype=np.float64)
    middle = np.asarray((247, 247, 247), dtype=np.float64)
    high = np.asarray((215, 48, 39), dtype=np.float64)
    if quantile <= 0.5:
        color = low + (middle - low) * (quantile * 2.0)
    else:
        color = middle + (high - middle) * ((quantile - 0.5) * 2.0)
    red, green, blue = np.rint(color).astype(int)
    return f"#{red:02x}{green:02x}{blue:02x}"


def _render_report(summary: dict) -> str:
    z0_ranking = summary["z0_metrics"]["holdout"]["ranking_at_k"]
    z2_aggregate = summary["aggregate"]["z2_train_sign_holdout"]
    lines = [
        "# 零训练需求传播基线",
        "",
        f"- Z0 holdout Spearman：`{summary['z0_metrics']['holdout']['spearman']:.4f}`",
        "- Z0 holdout NDCG@5/10/18："
        f"`{z0_ranking['5']['ndcg']:.4f} / {z0_ranking['10']['ndcg']:.4f} / "
        f"{z0_ranking['18']['ndcg']:.4f}`",
        f"- Z0 与 midpoint Proxy 的 holdout Spearman：`{summary['z0_proxy_correlation']['holdout']:.4f}`",
        "- Z2 五种子 holdout Spearman："
        f"`{z2_aggregate['mean_spearman']:.4f} ± {z2_aggregate['std_spearman']:.4f}`",
        "",
        "| Seed | Z1 Validation | Z1 Holdout | Train 定向 | Z2 Holdout |",
        "| ---: | ---: | ---: | ---: | ---: |",
    ]
    for run in summary["runs"]:
        lines.append(
            f"| {run['seed']} | "
            f"{run['z1_raw_metrics']['validation']['spearman']:.4f} | "
            f"{run['z1_raw_metrics']['holdout']['spearman']:.4f} | "
            f"{run['train_selected_sign']:+d} | "
            f"{run['z2_train_sign_metrics']['holdout']['spearman']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Z0 完全不使用可学习参数或标签；Z1 保留随机初始化网络但不训练；",
            "Z2 只使用 train split 决定一个全局正负号，不使用 validation 或 holdout。",
            "",
        ]
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
