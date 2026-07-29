"""分析单区域收益标签，并执行阶段一 Oracle/Proxy 排序门检查。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.graph_io import load_porto_graph
from src.region_candidates import chronological_prefix, load_candidate_manifest
from src.workloads import load_porto_queries


DEFAULT_NODE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路节点.csv"
DEFAULT_EDGE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路边.csv"
DEFAULT_QUERY_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图可用起终点节点查询_200米.csv"
DEFAULT_CANDIDATES = ROOT_DIR / "results" / "gnn_v2" / "candidate_manifest.json"
DEFAULT_LABELS = ROOT_DIR / "results" / "gnn_v2" / "region_training_labels.csv"
DEFAULT_LABEL_MANIFEST = ROOT_DIR / "results" / "gnn_v2" / "label_manifest.json"
DEFAULT_JSON = ROOT_DIR / "results" / "gnn_v2" / "label_analysis.json"
DEFAULT_REPORT = ROOT_DIR / "results" / "gnn_v2" / "label_analysis.md"
FORMAL_QUERY_SAMPLE_SIZE = 2_000
FORMAL_QUERY_SAMPLE_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze exact region labels and compare ranking baselines."
    )
    parser.add_argument("--node-csv", type=Path, default=DEFAULT_NODE_CSV)
    parser.add_argument("--edge-csv", type=Path, default=DEFAULT_EDGE_CSV)
    parser.add_argument("--query-csv", type=Path, default=DEFAULT_QUERY_CSV)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    parser.add_argument(
        "--label-manifest",
        type=Path,
        default=None,
        help="Label run manifest. The default formal labels infer label_manifest.json.",
    )
    parser.add_argument(
        "--comparison-labels",
        type=Path,
        default=None,
        help="Independent query-sample labels for stability analysis.",
    )
    parser.add_argument("--json-output", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report-output", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--history-fraction", type=float, default=0.35)
    parser.add_argument("--top-fraction", type=float, default=0.10)
    parser.add_argument("--skip-v1-proxy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 0.0 < args.top_fraction <= 1.0:
        raise SystemExit("--top-fraction must be in (0, 1]")
    candidate_manifest, regions = load_candidate_manifest(args.candidates)
    labels = _load_labels(args.labels)
    selected_regions = [region for region in regions if region.region_id in labels]
    if not selected_regions:
        raise SystemExit("no labeled candidates matched the candidate manifest")

    all_queries = load_porto_queries(args.query_csv)
    history_queries = chronological_prefix(all_queries, args.history_fraction)
    expected_label_query_count = (
        int(len(all_queries) * 0.70) - int(len(all_queries) * 0.35)
    )
    raw_frequency = _raw_frequency_scores(selected_regions, history_queries)
    proxy_scores: dict[int, float] | None = None
    if not args.skip_v1_proxy:
        graph = load_porto_graph(args.node_csv, args.edge_csv)
        proxy_scores = _v1_proxy_scores(graph, selected_regions, history_queries)

    gains = {region_id: row["avg_workload_gain"] for region_id, row in labels.items()}
    k = max(1, math.ceil(len(selected_regions) * args.top_fraction))
    observed_label_query_counts = {
        int(row["label_query_count"]) for row in labels.values()
    }
    label_manifest_path = _resolve_label_manifest(args.labels, args.label_manifest)
    label_manifest = (
        _load_and_validate_label_manifest(
            label_manifest_path,
            candidate_manifest,
            labels,
            expected_label_query_count,
        )
        if label_manifest_path is not None
        else None
    )
    analysis_type, stage_status = _classify_analysis(
        labeled_candidate_count=len(selected_regions),
        manifest_candidate_count=len(regions),
        observed_label_query_counts=observed_label_query_counts,
        expected_label_query_count=expected_label_query_count,
        label_manifest=label_manifest,
    )
    analysis = {
        "analysis_type": analysis_type,
        "labeled_candidate_count": len(selected_regions),
        "manifest_candidate_count": len(regions),
        "history_query_count": len(history_queries),
        "label_query_count": next(iter(observed_label_query_counts), None),
        "top_k": k,
        "correctness_min": min(row["correctness_rate"] for row in labels.values()),
        "label_distribution": _distribution(list(gains.values())),
        "selection_method_distribution": _method_distribution(selected_regions, gains),
        "rankings": {
            "raw_endpoint_frequency": _ranking_metrics(raw_frequency, gains, k),
        },
    }
    if proxy_scores is not None:
        analysis["rankings"]["v1_midpoint_proxy_mean"] = _ranking_metrics(
            proxy_scores,
            gains,
            k,
        )
    oracle_top_mean = _top_mean(gains, gains, k)
    analysis["rankings"]["oracle"] = {
        "top_k_mean_gain": oracle_top_mean,
        "ndcg_at_k": 1.0,
        "spearman": 1.0,
    }
    proxy_top_mean = (
        analysis["rankings"].get("v1_midpoint_proxy_mean", {})
        .get("top_k_mean_gain")
    )
    analysis["stage_gate"] = {
        "correctness_100_percent": analysis["correctness_min"] == 1.0,
        "non_constant_labels": analysis["label_distribution"]["std"] > 0.0,
        "oracle_minus_proxy_top_k_mean": (
            oracle_top_mean - proxy_top_mean if proxy_top_mean is not None else None
        ),
        "status": stage_status,
    }
    if args.comparison_labels is not None:
        comparison_rows = _load_labels(args.comparison_labels)
        comparison_gains = {
            region_id: row["avg_workload_gain"]
            for region_id, row in comparison_rows.items()
        }
        analysis["label_stability"] = _label_stability(gains, comparison_gains, k)
        analysis["cross_sample_selection"] = {
            "v1_midpoint_proxy_mean": _cross_sample_selection(
                gains,
                comparison_gains,
                proxy_scores,
                k,
            )
            if proxy_scores is not None
            else None,
            "raw_endpoint_frequency": _cross_sample_selection(
                gains,
                comparison_gains,
                raw_frequency,
                k,
            ),
        }

    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.report_output.parent.mkdir(parents=True, exist_ok=True)
    args.report_output.write_text(_render_report(analysis), encoding="utf-8")
    print(json.dumps(analysis, ensure_ascii=False, indent=2), flush=True)
    print(f"json={_display_path(args.json_output)}")
    print(f"report={_display_path(args.report_output)}")


def _load_labels(path: Path) -> dict[int, dict[str, float]]:
    labels: dict[int, dict[str, float]] = {}
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            labels[int(row["region_id"])] = {
                "avg_workload_gain": float(row["avg_workload_gain"]),
                "correctness_rate": float(row["correctness_rate"]),
                "label_query_count": float(row["label_query_count"]),
            }
    return labels


def _resolve_label_manifest(labels_path: Path, explicit_path: Path | None) -> Path | None:
    if explicit_path is not None:
        return explicit_path
    try:
        is_default_labels = labels_path.resolve() == DEFAULT_LABELS.resolve()
    except OSError:
        is_default_labels = False
    if is_default_labels and DEFAULT_LABEL_MANIFEST.exists():
        return DEFAULT_LABEL_MANIFEST
    return None


def _load_and_validate_label_manifest(
    path: Path,
    candidate_manifest: dict,
    labels: dict[int, dict[str, float]],
    expected_label_query_count: int,
) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    target_ids = {int(region_id) for region_id in manifest.get("target_region_ids", [])}
    completed_ids = {
        int(region_id) for region_id in manifest.get("completed_region_ids", [])
    }
    query_ids = manifest.get("query_ids", [])
    observed_query_counts = {
        int(row["label_query_count"]) for row in labels.values()
    }
    errors: list[str] = []
    if manifest.get("candidate_sha256") != candidate_manifest.get("candidate_sha256"):
        errors.append("candidate_sha256 does not match the candidate manifest")
    if target_ids != set(labels):
        errors.append("target_region_ids do not match the label CSV")
    if manifest.get("target_region_count") != len(target_ids):
        errors.append("target_region_count does not match target_region_ids")
    if manifest.get("status") == "complete":
        if completed_ids != target_ids:
            errors.append("complete manifest does not list every target as completed")
        if manifest.get("completed_region_count") != len(completed_ids):
            errors.append("completed_region_count does not match completed_region_ids")
    if len(observed_query_counts) != 1:
        errors.append("label CSV contains inconsistent label_query_count values")
    elif len(query_ids) != next(iter(observed_query_counts)):
        errors.append("query_ids count does not match label_query_count")
    start = manifest.get("label_start_fraction")
    end = manifest.get("label_end_fraction")
    if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
        errors.append("label window fractions are missing")
    elif int(expected_label_query_count) < len(query_ids):
        errors.append("manifest query count exceeds the label window")
    if errors:
        raise SystemExit(f"invalid label manifest {path}: " + "; ".join(errors))
    return manifest


def _classify_analysis(
    *,
    labeled_candidate_count: int,
    manifest_candidate_count: int,
    observed_label_query_counts: set[int],
    expected_label_query_count: int,
    label_manifest: dict | None,
) -> tuple[str, str]:
    if labeled_candidate_count < manifest_candidate_count:
        return "candidate_subset_pilot", "pilot_only"
    if observed_label_query_counts == {expected_label_query_count}:
        return "full_label_window", "full_window_complete"
    if _is_formal_sampled_label_run(label_manifest, observed_label_query_counts):
        return "formal_sampled_labels", "ready_for_modeling"
    return "all_candidates_screening_sample", "screening_only"


def _is_formal_sampled_label_run(
    label_manifest: dict | None,
    observed_label_query_counts: set[int],
) -> bool:
    if label_manifest is None:
        return False
    return (
        label_manifest.get("status") == "complete"
        and label_manifest.get("target_region_count")
        == label_manifest.get("completed_region_count")
        and len(label_manifest.get("query_ids", [])) == FORMAL_QUERY_SAMPLE_SIZE
        and observed_label_query_counts == {FORMAL_QUERY_SAMPLE_SIZE}
        and label_manifest.get("query_sample_seed") == FORMAL_QUERY_SAMPLE_SEED
        and label_manifest.get("label_start_fraction") == 0.35
        and label_manifest.get("label_end_fraction") == 0.70
    )


def _raw_frequency_scores(regions, queries) -> dict[int, float]:
    counts: Counter[int] = Counter()
    for query in queries:
        counts[query.origin] += query.count
        counts[query.destination] += query.count
    return {
        region.region_id: sum(counts[node] for node in region.nodes) / region.node_count
        for region in regions
    }


def _v1_proxy_scores(graph, regions, history_queries) -> dict[int, float]:
    """用纯 Python 复现第一版 midpoint Proxy，避免分析脚本依赖训练环境。"""

    nodes = tuple(graph.adjacency)
    node_to_index = {node: index for index, node in enumerate(nodes)}
    node_count = len(nodes)
    origin_counts = [0.0] * node_count
    destination_counts = [0.0] * node_count
    for query in history_queries:
        origin_counts[node_to_index[query.origin]] += query.count
        destination_counts[node_to_index[query.destination]] += query.count

    target_degree = [0.0] * node_count
    edges: list[tuple[int, int]] = []
    for source, neighbors in graph.adjacency.items():
        source_index = node_to_index[source]
        for target, _ in neighbors:
            target_index = node_to_index[target]
            edges.append((source_index, target_index))
            edges.append((target_index, source_index))
            target_degree[target_index] += 1.0
            target_degree[source_index] += 1.0

    scale = 10_000.0 / max(1, len(history_queries))
    diffused_origin = _diffuse(
        [value * scale for value in origin_counts],
        edges,
        target_degree,
    )
    diffused_destination = _diffuse(
        [value * scale for value in destination_counts],
        edges,
        target_degree,
    )
    endpoint_risk = _unit_scale(
        [
            math.log1p(origin) + math.log1p(destination)
            for origin, destination in zip(diffused_origin, diffused_destination)
        ]
    )

    grid, cell_size = _coordinate_grid(graph, nodes)
    midpoint_counts = [0.0] * node_count
    for query in history_queries:
        origin = graph.coordinates.get(query.origin)
        destination = graph.coordinates.get(query.destination)
        if origin is None or destination is None:
            continue
        midpoint = (
            (origin[0] + destination[0]) / 2.0,
            (origin[1] + destination[1]) / 2.0,
        )
        midpoint_index = _nearest_grid_node(
            midpoint,
            grid,
            cell_size,
            graph,
            nodes,
        )
        midpoint_counts[midpoint_index] += query.count
    midpoint_density = _diffuse(
        [value * scale for value in midpoint_counts],
        edges,
        target_degree,
    )
    midpoint_value = _unit_scale([math.log1p(value) for value in midpoint_density])
    target = _unit_scale(
        [
            value / (1.0 + 2.0 * risk)
            for value, risk in zip(midpoint_value, endpoint_risk)
        ]
    )
    scores = {node: target[index] for index, node in enumerate(nodes)}
    return {
        region.region_id: statistics.fmean(scores[node] for node in region.nodes)
        for region in regions
    }


def _diffuse(
    values: list[float],
    edges: list[tuple[int, int]],
    target_degree: list[float],
    *,
    steps: int = 3,
    restart: float = 0.4,
) -> list[float]:
    base = list(values)
    state = list(values)
    for _ in range(steps):
        aggregated = [0.0] * len(state)
        for source, target in edges:
            aggregated[target] += state[source]
        state = [
            restart * base[index]
            + (1.0 - restart) * aggregated[index] / max(target_degree[index], 1.0)
            for index in range(len(state))
        ]
    return state


def _coordinate_grid(graph, nodes, cell_size: float = 0.002):
    grid: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, node in enumerate(nodes):
        longitude, latitude = graph.coordinates[node]
        grid[
            (math.floor(longitude / cell_size), math.floor(latitude / cell_size))
        ].append(index)
    return dict(grid), cell_size


def _nearest_grid_node(coordinate, grid, cell_size, graph, nodes) -> int:
    longitude, latitude = coordinate
    base_x = math.floor(longitude / cell_size)
    base_y = math.floor(latitude / cell_size)
    candidates: list[int] = []
    for radius in range(8):
        for grid_x in range(base_x - radius, base_x + radius + 1):
            for grid_y in range(base_y - radius, base_y + radius + 1):
                if (
                    radius > 0
                    and abs(grid_x - base_x) < radius
                    and abs(grid_y - base_y) < radius
                ):
                    continue
                candidates.extend(grid.get((grid_x, grid_y), ()))
        if candidates:
            break
    if not candidates:
        return min(
            range(len(nodes)),
            key=lambda index: (
                graph.coordinates[nodes[index]][0] - longitude
            )
            ** 2
            + (graph.coordinates[nodes[index]][1] - latitude) ** 2,
        )
    return min(
        candidates,
        key=lambda index: (
            graph.coordinates[nodes[index]][0] - longitude
        )
        ** 2
        + (graph.coordinates[nodes[index]][1] - latitude) ** 2,
    )


def _unit_scale(values: list[float]) -> list[float]:
    minimum = min(0.0, min(values, default=0.0))
    maximum = max(0.0, max(values, default=0.0))
    if maximum - minimum <= 1e-12:
        return [0.0] * len(values)
    return [(value - minimum) / (maximum - minimum) for value in values]


def _ranking_metrics(
    scores: dict[int, float],
    gains: dict[int, float],
    k: int,
) -> dict[str, float]:
    common = sorted(scores.keys() & gains.keys())
    score_values = [scores[item] for item in common]
    gain_values = [gains[item] for item in common]
    return {
        "spearman": _spearman(score_values, gain_values),
        "ndcg_at_k": _ndcg_at_k(scores, gains, k),
        "top_k_mean_gain": _top_mean(scores, gains, k),
        "all_mean_gain": statistics.fmean(gain_values),
    }


def _label_stability(
    first: dict[int, float],
    second: dict[int, float],
    k: int,
) -> dict[str, float]:
    common = sorted(first.keys() & second.keys())
    if not common:
        raise SystemExit("comparison labels have no region ids in common")
    first_values = [first[item] for item in common]
    second_values = [second[item] for item in common]
    first_top = set(sorted(common, key=lambda item: (-first[item], item))[:k])
    second_top = set(sorted(common, key=lambda item: (-second[item], item))[:k])
    return {
        "common_candidate_count": len(common),
        "spearman": _spearman(first_values, second_values),
        "mae": statistics.fmean(
            abs(left - right) for left, right in zip(first_values, second_values)
        ),
        "top_k_overlap_count": len(first_top & second_top),
        "top_k_overlap_rate": len(first_top & second_top) / max(1, k),
    }


def _cross_sample_selection(
    first_labels: dict[int, float],
    second_labels: dict[int, float],
    fixed_scores: dict[int, float],
    k: int,
) -> dict[str, float]:
    common = sorted(first_labels.keys() & second_labels.keys() & fixed_scores.keys())
    first_oracle = sorted(
        common,
        key=lambda item: (-first_labels[item], item),
    )[:k]
    second_oracle = sorted(
        common,
        key=lambda item: (-second_labels[item], item),
    )[:k]
    fixed = sorted(common, key=lambda item: (-fixed_scores[item], item))[:k]
    first_oracle_on_second = statistics.fmean(
        second_labels[item] for item in first_oracle
    )
    second_oracle_on_first = statistics.fmean(
        first_labels[item] for item in second_oracle
    )
    fixed_on_second = statistics.fmean(second_labels[item] for item in fixed)
    fixed_on_first = statistics.fmean(first_labels[item] for item in fixed)
    return {
        "first_oracle_on_second_mean_gain": first_oracle_on_second,
        "fixed_on_second_mean_gain": fixed_on_second,
        "first_to_second_gain_gap": first_oracle_on_second - fixed_on_second,
        "second_oracle_on_first_mean_gain": second_oracle_on_first,
        "fixed_on_first_mean_gain": fixed_on_first,
        "second_to_first_gain_gap": second_oracle_on_first - fixed_on_first,
    }


def _top_mean(scores: dict[int, float], gains: dict[int, float], k: int) -> float:
    ordered = sorted(scores, key=lambda item: (-scores[item], item))[:k]
    return statistics.fmean(gains[item] for item in ordered)


def _ndcg_at_k(scores: dict[int, float], gains: dict[int, float], k: int) -> float:
    predicted = sorted(scores, key=lambda item: (-scores[item], item))[:k]
    ideal = sorted(gains, key=lambda item: (-gains[item], item))[:k]

    def dcg(items: list[int]) -> float:
        return sum(
            max(0.0, gains[item]) / math.log2(rank + 2)
            for rank, item in enumerate(items)
        )

    ideal_dcg = dcg(ideal)
    return dcg(predicted) / ideal_dcg if ideal_dcg > 0.0 else 0.0


def _spearman(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or not left:
        return 0.0
    left_ranks = _ranks(left)
    right_ranks = _ranks(right)
    left_mean = statistics.fmean(left_ranks)
    right_mean = statistics.fmean(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left_ranks))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right_ranks))
    return numerator / (left_scale * right_scale) if left_scale and right_scale else 0.0


def _ranks(values: list[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[ordered[position]] = average_rank
        start = end
    return ranks


def _distribution(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "min": ordered[0],
        "p10": _percentile(ordered, 10),
        "median": statistics.median(ordered),
        "p90": _percentile(ordered, 90),
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
        "std": statistics.pstdev(ordered),
        "positive_rate_pct": sum(value > 0.0 for value in ordered) / len(ordered) * 100.0,
    }


def _method_distribution(regions, gains) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for region in regions:
        grouped[region.selection_method].append(gains[region.region_id])
    return {
        method: {
            "count": len(values),
            "mean_gain": statistics.fmean(values),
            "median_gain": statistics.median(values),
        }
        for method, values in sorted(grouped.items())
    }


def _percentile(ordered: list[float], percentile: float) -> float:
    index = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _render_report(analysis: dict) -> str:
    distribution = analysis["label_distribution"]
    lines = [
        "# GNN 第二版区域收益标签分析",
        "",
        f"- 分析类型：`{analysis['analysis_type']}`",
        f"- 已标注候选：{analysis['labeled_candidate_count']} / {analysis['manifest_candidate_count']}",
        f"- 历史窗口查询：{analysis['history_query_count']}",
        f"- 标签正确率下界：{analysis['correctness_min']:.6f}",
        f"- 收益均值 / 标准差：{distribution['mean']:.3f} / {distribution['std']:.3f}",
        f"- 收益 P10 / 中位数 / P90：{distribution['p10']:.3f} / "
        f"{distribution['median']:.3f} / {distribution['p90']:.3f}",
        "",
        "## 排序诊断",
        "",
        "| 方法 | Spearman | NDCG@K | Top-K 平均收益 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method, metrics in analysis["rankings"].items():
        lines.append(
            f"| {method} | {metrics['spearman']:.4f} | "
            f"{metrics['ndcg_at_k']:.4f} | {metrics['top_k_mean_gain']:.3f} |"
        )
    stage_messages = {
        "pilot_only": "候选子集只用于链路试跑，不能据此进入正式模型训练。",
        "screening_only": "全候选筛查只用于检查标签稳定性，不能替代正式标签阶段门。",
        "ready_for_modeling": (
            "正式抽样标签阶段门已通过，可以进入无传播 MLP；完整 Y 窗口保留为后续复核。"
        ),
        "full_window_complete": "完整 Y 标签窗口已完成，可以进入或复核模型训练。",
    }
    lines.extend(
        [
            "",
            "## 阶段门",
            "",
            f"- 状态：`{analysis['stage_gate']['status']}`",
            f"- 100% 正确性：{analysis['stage_gate']['correctness_100_percent']}",
            f"- 标签非常数：{analysis['stage_gate']['non_constant_labels']}",
            f"- 结论：{stage_messages[analysis['stage_gate']['status']]}",
            "",
        ]
    )
    if "label_stability" in analysis:
        stability = analysis["label_stability"]
        lines.extend(
            [
                "## 独立查询样本稳定性",
                "",
                f"- 共同候选：{stability['common_candidate_count']}",
                f"- 标签 Spearman：{stability['spearman']:.4f}",
                f"- 标签 MAE：{stability['mae']:.3f}",
                f"- Top-K 重合率：{stability['top_k_overlap_rate']:.2%}",
                "",
            ]
        )
        proxy_cross = analysis["cross_sample_selection"].get(
            "v1_midpoint_proxy_mean"
        )
        if proxy_cross is not None:
            lines.extend(
                [
                    "## 跨样本 Oracle 选择",
                    "",
                    "- 第一批 Oracle 候选在第二批上的平均收益："
                    f"{proxy_cross['first_oracle_on_second_mean_gain']:.3f}",
                    "- Proxy 候选在第二批上的平均收益："
                    f"{proxy_cross['fixed_on_second_mean_gain']:.3f}",
                    "- 第二批 Oracle 候选在第一批上的平均收益："
                    f"{proxy_cross['second_oracle_on_first_mean_gain']:.3f}",
                    "- Proxy 候选在第一批上的平均收益："
                    f"{proxy_cross['fixed_on_first_mean_gain']:.3f}",
                    "",
                ]
            )
    return "\n".join(lines)


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
