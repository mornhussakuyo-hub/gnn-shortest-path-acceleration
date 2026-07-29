"""训练第二版无传播区域 MLP 基线。"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.demand_field_data import SPLIT_NAMES, load_demand_field_dataset
from src.demand_field_model import MLPConfig, regression_metrics
from src.demand_field_torch_model import (
    TorchCudaMLPRegressor,
    cuda_environment,
    require_cuda,
)


DEFAULT_DATASET = ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.npz"
DEFAULT_DATASET_MANIFEST = ROOT_DIR / "results" / "gnn_v2" / "demand_field_dataset.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "gnn_v2" / "mlp_baseline"
EXPERIMENT_SCHEMA = "aic.gnn_v2.region_mlp_experiment.torch_cuda.v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the no-message-passing region MLP baseline."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument(
        "--dataset-manifest",
        type=Path,
        default=DEFAULT_DATASET_MANIFEST,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seeds", default="42,43,44,45,46")
    parser.add_argument("--hidden-dims", default="64,32")
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--rank-weight", type=float, default=0.20)
    parser.add_argument("--huber-delta", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=60)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = _parse_int_list(args.seeds, "--seeds")
    hidden_dims = tuple(_parse_int_list(args.hidden_dims, "--hidden-dims"))
    config = MLPConfig(
        hidden_dims=hidden_dims,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        rank_weight=args.rank_weight,
        huber_delta=args.huber_delta,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
    )
    config.validate()
    device = require_cuda(args.device)
    dataset = load_demand_field_dataset(args.dataset, args.dataset_manifest)
    train_mask = dataset.split_mask("train")
    validation_mask = dataset.split_mask("validation")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[dict] = []

    raw_frequency_score = _raw_frequency_score(dataset)
    baseline_metrics = {
        name: _ranking_only_metrics(
            raw_frequency_score[dataset.split_mask(name)],
            dataset.labels[dataset.split_mask(name)],
        )
        for name in SPLIT_NAMES
    }

    for seed in seeds:
        print(f"training seed={seed}", flush=True)
        model = TorchCudaMLPRegressor(
            dataset.region_features.shape[1],
            config,
            seed,
            device,
        )
        history = model.fit(
            dataset.region_features[train_mask],
            dataset.labels[train_mask],
            dataset.region_features[validation_mask],
            dataset.labels[validation_mask],
        )
        prediction = model.predict(dataset.region_features)
        metrics = {
            name: regression_metrics(
                prediction[dataset.split_mask(name)],
                dataset.labels[dataset.split_mask(name)],
            )
            for name in SPLIT_NAMES
        }
        seed_dir = args.output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        _write_history(seed_dir / "training_history.csv", history)
        _write_predictions(
            seed_dir / "predictions.csv",
            dataset.region_ids,
            dataset.selection_method,
            dataset.split,
            dataset.labels,
            prediction,
        )
        manifest = model.save(
            seed_dir / "model.pt",
            seed_dir / "model.json",
            {
                "dataset_sha256": dataset.manifest["dataset_sha256"],
                "candidate_sha256": dataset.manifest["candidate_sha256"],
                "region_feature_names": list(dataset.region_feature_names),
                "metrics": metrics,
            },
        )
        run = {
            "seed": seed,
            "best_epoch": model.best_epoch,
            "epochs_ran": len(history),
            "execution": manifest["execution"],
            "metrics": metrics,
        }
        runs.append(run)
        print(
            f"seed={seed} best_epoch={model.best_epoch} "
            f"holdout_spearman={metrics['holdout']['spearman']:.4f} "
            f"holdout_top_gain={metrics['holdout']['top_k_mean_gain']:.3f}",
            flush=True,
        )

    aggregate = _aggregate_runs(runs)
    stage_gate = _stage_gate(runs)
    selected_seed = max(
        runs,
        key=lambda run: run["metrics"]["validation"]["spearman"],
    )["seed"]
    summary = {
        "schema": EXPERIMENT_SCHEMA,
        "model": "no_message_passing_region_mlp",
        "execution": cuda_environment(device),
        "dataset_sha256": dataset.manifest["dataset_sha256"],
        "candidate_sha256": dataset.manifest["candidate_sha256"],
        "config": {
            "hidden_dims": list(config.hidden_dims),
            "learning_rate": config.learning_rate,
            "weight_decay": config.weight_decay,
            "rank_weight": config.rank_weight,
            "huber_delta": config.huber_delta,
            "batch_size": config.batch_size,
            "max_epochs": config.max_epochs,
            "patience": config.patience,
        },
        "seeds": seeds,
        "selected_seed": selected_seed,
        "selection_rule": "highest validation Spearman; holdout is not used",
        "split": dataset.manifest["split"],
        "input_policy": dataset.manifest["model_input_policy"],
        "raw_frequency_baseline": baseline_metrics,
        "runs": runs,
        "aggregate": aggregate,
        "stage_gate": stage_gate,
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "report.md").write_text(
        _render_report(summary),
        encoding="utf-8",
    )
    print(json.dumps({"aggregate": aggregate, "stage_gate": stage_gate}, indent=2))
    print(f"summary={_display_path(args.output_dir / 'summary.json')}")
    print(f"report={_display_path(args.output_dir / 'report.md')}")


def _raw_frequency_score(dataset) -> np.ndarray:
    names = list(dataset.region_feature_names)
    origin = names.index("mean_history_origin_count")
    destination = names.index("mean_history_destination_count")
    return dataset.region_features[:, origin] + dataset.region_features[:, destination]


def _ranking_only_metrics(score: np.ndarray, target: np.ndarray) -> dict[str, float]:
    metrics = regression_metrics(score.astype(np.float64), target.astype(np.float64))
    return {
        key: metrics[key]
        for key in (
            "count",
            "spearman",
            "ndcg_at_k",
            "top_k",
            "top_k_mean_gain",
            "oracle_top_k_mean_gain",
            "all_mean_gain",
        )
    }


def _aggregate_runs(runs: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    aggregate: dict[str, dict[str, dict[str, float]]] = {}
    metric_names = (
        "mae",
        "huber",
        "spearman",
        "ndcg_at_k",
        "top_k_mean_gain",
    )
    for split_name in SPLIT_NAMES:
        aggregate[split_name] = {}
        for metric_name in metric_names:
            values = [run["metrics"][split_name][metric_name] for run in runs]
            aggregate[split_name][metric_name] = {
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values),
                "min": min(values),
                "max": max(values),
            }
    return aggregate


def _stage_gate(runs: list[dict]) -> dict:
    validation_spearman = [run["metrics"]["validation"]["spearman"] for run in runs]
    holdout_spearman = [run["metrics"]["holdout"]["spearman"] for run in runs]
    holdout_top_better_than_mean = [
        run["metrics"]["holdout"]["top_k_mean_gain"]
        > run["metrics"]["holdout"]["all_mean_gain"]
        for run in runs
    ]
    passed = (
        min(validation_spearman) > 0.0
        and min(holdout_spearman) > 0.0
        and all(holdout_top_better_than_mean)
    )
    return {
        "all_validation_spearman_positive": min(validation_spearman) > 0.0,
        "all_holdout_spearman_positive": min(holdout_spearman) > 0.0,
        "all_holdout_top_k_better_than_mean": all(holdout_top_better_than_mean),
        "status": "ready_for_nbfnet" if passed else "mlp_gate_failed",
        "interpretation": (
            "Candidate holdout proves learnability within the current H→Y pair; "
            "it does not replace the frozen future temporal test."
        ),
    }


def _write_history(path: Path, history: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        for row in history:
            writer.writerow({**row, "epoch": int(row["epoch"])})


def _write_predictions(
    path: Path,
    region_ids: np.ndarray,
    selection_method: np.ndarray,
    split: np.ndarray,
    labels: np.ndarray,
    prediction: np.ndarray,
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            ["region_id", "selection_method", "split", "label", "prediction"]
        )
        for region_id, method, split_id, label, predicted in zip(
            region_ids,
            selection_method,
            split,
            labels,
            prediction,
        ):
            writer.writerow(
                [
                    int(region_id),
                    str(method),
                    SPLIT_NAMES[int(split_id)],
                    f"{float(label):.9f}",
                    f"{float(predicted):.9f}",
                ]
            )


def _render_report(summary: dict) -> str:
    holdout = summary["aggregate"]["holdout"]
    raw = summary["raw_frequency_baseline"]["holdout"]
    lines = [
        "# GNN 第二版无传播区域 MLP 基线",
        "",
        f"- 数据集摘要：`{summary['dataset_sha256']}`",
        f"- 随机种子：{', '.join(str(seed) for seed in summary['seeds'])}",
        f"- 隐藏层：{summary['config']['hidden_dims']}",
        f"- 训练框架：PyTorch `{summary['execution']['torch_version']}`",
        f"- 训练设备：`{summary['execution']['device_name']}`（CUDA `{summary['execution']['cuda_version']}`）",
        f"- 阶段门：`{summary['stage_gate']['status']}`",
        "- 评测口径：同一 H→Y 时间对内按候选划分 train/validation/holdout；holdout 不是最终未来时间测试。",
        "",
        "## Candidate holdout 五种子结果",
        "",
        "| 指标 | MLP 均值 | 标准差 | 原始频率 |",
        "| --- | ---: | ---: | ---: |",
        f"| Spearman | {holdout['spearman']['mean']:.4f} | {holdout['spearman']['std']:.4f} | {raw['spearman']:.4f} |",
        f"| NDCG@K | {holdout['ndcg_at_k']['mean']:.4f} | {holdout['ndcg_at_k']['std']:.4f} | {raw['ndcg_at_k']:.4f} |",
        f"| Top-K 平均真实收益 | {holdout['top_k_mean_gain']['mean']:.3f} | {holdout['top_k_mean_gain']['std']:.3f} | {raw['top_k_mean_gain']:.3f} |",
        f"| MAE | {holdout['mae']['mean']:.3f} | {holdout['mae']['std']:.3f} | — |",
        "",
        "## 结论边界",
        "",
        "该 MLP 只读取历史起终点计数和静态道路特征的区域 mean/max 池化，不读取路径、标签窗口输入、",
        "候选来源、shortcut、查询耗时或端点接入工作量。它是后续 NBFNet 必须公平超过的无传播对照。",
        "",
    ]
    return "\n".join(lines)


def _parse_int_list(value: str, option: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as error:
        raise SystemExit(f"{option} must be a comma-separated integer list") from error
    if not values or any(item <= 0 for item in values):
        raise SystemExit(f"{option} must contain positive integers")
    return values


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
