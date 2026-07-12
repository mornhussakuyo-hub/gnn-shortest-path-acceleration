"""使用 GPU 训练第一版无路径监督 GNN 节点种子价值模型。"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.gnn_data import FEATURE_NAMES, build_gnn_data
from src.gnn_model import SeedValueGraphSage, SeedValueMlp
from src.graph_io import load_porto_graph
from src.regions import build_risk_aware_scored_regions
from src.workloads import load_porto_queries, split_queries_chronologically


DEFAULT_NODE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路节点.csv"
DEFAULT_EDGE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路边.csv"
DEFAULT_QUERY_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图可用起终点节点查询_200米.csv"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "results" / "gnn_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the first GPU GraphSAGE seed-value model.")
    parser.add_argument("--node-csv", type=Path, default=DEFAULT_NODE_CSV)
    parser.add_argument("--edge-csv", type=Path, default=DEFAULT_EDGE_CSV)
    parser.add_argument("--query-csv", type=Path, default=DEFAULT_QUERY_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--positive-weight", type=float, default=20.0)
    parser.add_argument("--model-type", choices=("graph_sage", "mlp"), default="graph_sage")
    parser.add_argument(
        "--target-mode",
        choices=("midpoint", "demand_overlap"),
        default="midpoint",
    )
    parser.add_argument(
        "--exclude-features",
        nargs="*",
        choices=FEATURE_NAMES,
        default=[],
    )
    parser.add_argument("--diffusion-steps", type=int, default=3)
    parser.add_argument("--endpoint-penalty", type=float, default=2.0)
    parser.add_argument("--region-count", type=int, default=100)
    parser.add_argument("--region-size", type=int, default=512)
    parser.add_argument("--seed-exclusion-hops", type=int, default=0)
    parser.add_argument("--candidate-limit", type=int, default=80_000)
    parser.add_argument("--region-endpoint-risk-penalty", type=float, default=200.0)
    parser.add_argument("--skip-region-selection", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU 不可用；第一版 GNN 训练按要求拒绝回退到 CPU。")
    _set_seed(args.seed)
    total_start = time.perf_counter()
    device = torch.device("cuda")
    print(
        f"device={device} gpu={torch.cuda.get_device_name(0)} "
        f"cuda_runtime={torch.version.cuda}",
        flush=True,
    )

    graph = load_porto_graph(args.node_csv, args.edge_csv)
    queries = load_porto_queries(args.query_csv)
    train_queries, validation_queries, test_queries = split_queries_chronologically(queries)
    print(
        f"graph_nodes={graph.node_count:,} graph_edges={graph.edge_count:,} "
        f"train={len(train_queries):,} validation={len(validation_queries):,} "
        f"test={len(test_queries):,}",
        flush=True,
    )
    feature_start = time.perf_counter()
    data = build_gnn_data(
        graph,
        train_queries,
        validation_queries,
        test_queries,
        diffusion_steps=args.diffusion_steps,
        endpoint_penalty=args.endpoint_penalty,
        target_mode=args.target_mode,
    )
    feature_seconds = time.perf_counter() - feature_start

    excluded_features = set(args.exclude_features)
    selected_feature_indices = [
        index for index, name in enumerate(FEATURE_NAMES) if name not in excluded_features
    ]
    if not selected_feature_indices:
        raise SystemExit("不能排除全部节点特征。")
    selected_feature_names = tuple(FEATURE_NAMES[index] for index in selected_feature_indices)
    features = torch.from_numpy(data.features[:, selected_feature_indices]).to(device)
    train_target = torch.from_numpy(data.train_target).to(device)
    validation_target = torch.from_numpy(data.validation_target).to(device)
    test_target = torch.from_numpy(data.test_target).to(device)
    edge_source = torch.from_numpy(data.edge_source).to(device)
    edge_target = torch.from_numpy(data.edge_target).to(device)
    target_degree = torch.from_numpy(data.target_degree).to(device)

    model_class = SeedValueGraphSage if args.model_type == "graph_sage" else SeedValueMlp
    model = model_class(
        input_dim=features.shape[1],
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_validation_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []

    training_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features, edge_source, edge_target, target_degree)
        train_weights = 1.0 + args.positive_weight * train_target
        train_loss = torch.mean(train_weights * (prediction - train_target) ** 2)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_prediction = model(features, edge_source, edge_target, target_degree)
            validation_weights = 1.0 + args.positive_weight * validation_target
            validation_loss = torch.mean(
                validation_weights * (validation_prediction - validation_target) ** 2
            )
            validation_correlation = _pearson_correlation(
                validation_prediction,
                validation_target,
            )
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss.item()),
                "validation_loss": float(validation_loss.item()),
                "validation_correlation": validation_correlation,
            }
        )
        if validation_loss.item() < best_validation_loss - 1e-8:
            best_validation_loss = float(validation_loss.item())
            best_epoch = epoch
            best_state = {
                name: parameter.detach().cpu().clone()
                for name, parameter in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} train_loss={train_loss.item():.6f} "
                f"validation_loss={validation_loss.item():.6f} "
                f"validation_corr={validation_correlation:.4f} "
                f"gpu_memory_mb={torch.cuda.max_memory_allocated() / 1024**2:.1f}",
                flush=True,
            )
        if epochs_without_improvement >= args.patience:
            print(f"early_stop epoch={epoch} best_epoch={best_epoch}", flush=True)
            break

    if best_state is None:
        raise RuntimeError("training did not produce a model state")
    training_seconds = time.perf_counter() - training_start
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scores_tensor = model(features, edge_source, edge_target, target_degree)
        test_weights = 1.0 + args.positive_weight * test_target
        test_loss = torch.mean(test_weights * (scores_tensor - test_target) ** 2)
        test_correlation = _pearson_correlation(scores_tensor, test_target)
    scores = scores_tensor.detach().cpu().numpy().astype(np.float32)
    node_scores = dict(zip(data.nodes, map(float, scores), strict=True))
    if args.skip_region_selection:
        regions = []
        selection_seconds = 0.0
    else:
        selection_start = time.perf_counter()
        regions = build_risk_aware_scored_regions(
            graph,
            node_scores,
            train_queries,
            region_count=args.region_count,
            region_size=args.region_size,
            seed_exclusion_hops=args.seed_exclusion_hops,
            candidate_limit=args.candidate_limit,
            endpoint_risk_penalty=args.region_endpoint_risk_penalty,
        )
        selection_seconds = time.perf_counter() - selection_start

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "input_dim": features.shape[1],
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "feature_names": selected_feature_names,
            "model_type": args.model_type,
            "target_mode": args.target_mode,
            "best_epoch": best_epoch,
            "seed": args.seed,
        },
        args.output_dir / "model.pt",
    )
    _write_history(args.output_dir / "training_history.csv", history)
    selected_seeds = {region.seed_node: region.region_id for region in regions}
    _write_scores(
        args.output_dir / "node_scores.csv",
        data.nodes,
        scores,
        data.endpoint_risk,
        data.train_target,
        data.validation_target,
        data.test_target,
        selected_seeds,
    )
    _write_regions(args.output_dir / "selected_regions.csv", regions, node_scores)
    summary = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "graph_node_count": graph.node_count,
        "graph_edge_count": graph.edge_count,
        "query_count": len(queries),
        "train_query_count": len(train_queries),
        "validation_query_count": len(validation_queries),
        "test_query_count": len(test_queries),
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "test_loss": float(test_loss.item()),
        "test_correlation": test_correlation,
        "requested_region_count": args.region_count,
        "actual_region_count": len(regions),
        "region_size": args.region_size,
        "seed_exclusion_hops": args.seed_exclusion_hops,
        "candidate_limit": args.candidate_limit,
        "region_endpoint_risk_penalty": args.region_endpoint_risk_penalty,
        "endpoint_penalty": args.endpoint_penalty,
        "diffusion_steps": args.diffusion_steps,
        "model_type": args.model_type,
        "target_mode": args.target_mode,
        "feature_names": list(selected_feature_names),
        "excluded_features": list(args.exclude_features),
        "random_seed": args.seed,
        "feature_construction_seconds": feature_seconds,
        "gpu_training_seconds": training_seconds,
        "region_selection_seconds": selection_seconds,
        "total_seconds": time.perf_counter() - total_start,
        "peak_gpu_memory_mb": torch.cuda.max_memory_allocated() / 1024**2,
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _pearson_correlation(left: torch.Tensor, right: torch.Tensor) -> float:
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = torch.sqrt(
        torch.sum(left_centered**2) * torch.sum(right_centered**2)
    )
    if denominator.item() <= 1e-12:
        return 0.0
    return float((torch.sum(left_centered * right_centered) / denominator).item())


def _write_history(path: Path, rows: list[dict[str, float | int]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_scores(
    path: Path,
    nodes: tuple[int, ...],
    scores: np.ndarray,
    endpoint_risk: np.ndarray,
    train_target: np.ndarray,
    validation_target: np.ndarray,
    test_target: np.ndarray,
    selected_seeds: dict[int, int],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            [
                "node_id",
                "seed_score",
                "endpoint_risk",
                "train_proxy_target",
                "validation_proxy_target",
                "test_proxy_target",
                "selected_region_id",
            ]
        )
        for index, node in enumerate(nodes):
            writer.writerow(
                [
                    node,
                    f"{scores[index]:.8f}",
                    f"{endpoint_risk[index]:.8f}",
                    f"{train_target[index]:.8f}",
                    f"{validation_target[index]:.8f}",
                    f"{test_target[index]:.8f}",
                    selected_seeds.get(node, ""),
                ]
            )


def _write_regions(path: Path, regions, node_scores: dict[int, float]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(
            ["region_id", "seed_node", "seed_score", "node_count", "boundary_count"]
        )
        for region in regions:
            writer.writerow(
                [
                    region.region_id,
                    region.seed_node,
                    f"{node_scores[region.seed_node]:.8f}",
                    region.node_count,
                    region.boundary_count,
                ]
            )


if __name__ == "__main__":
    main()
