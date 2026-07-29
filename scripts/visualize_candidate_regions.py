"""绘制 Porto 路网、固定候选区域与候选覆盖密度。"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import Line2D


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_NODE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路节点.csv"
DEFAULT_EDGE_CSV = ROOT_DIR / "data" / "processed" / "porto" / "波尔图道路边.csv"
DEFAULT_CANDIDATES = ROOT_DIR / "results" / "gnn_v2" / "candidate_manifest.json"
DEFAULT_OUTPUT = ROOT_DIR / "results" / "gnn_v2" / "candidate_region_map.png"
DEFAULT_SUMMARY = ROOT_DIR / "results" / "gnn_v2" / "candidate_region_map.json"
DEFAULT_REPORT = ROOT_DIR / "results" / "gnn_v2" / "candidate_region_map.md"

METHOD_COLORS = {
    "fixed_random_bfs": "#7B2CBF",
}
METHOD_LABELS = {
    "fixed_random_bfs": "固定随机 BFS",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=Path, default=DEFAULT_NODE_CSV)
    parser.add_argument("--edges", type=Path, default=DEFAULT_EDGE_CSV)
    parser.add_argument("--candidates", type=Path, default=DEFAULT_CANDIDATES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    node_ids, coordinates_m, node_to_index = _load_nodes(args.nodes)
    sources, targets = _load_edges(args.edges, node_to_index)
    manifest = json.loads(args.candidates.read_text(encoding="utf-8"))
    candidates = manifest["candidates"]
    candidate_indices = [
        np.asarray([node_to_index[int(node)] for node in item["nodes"]], dtype=np.int64)
        for item in candidates
    ]
    boundary_indices = [
        np.asarray(
            [node_to_index[int(node)] for node in item["boundary_nodes"]],
            dtype=np.int64,
        )
        for item in candidates
    ]
    coverage = np.zeros(len(node_ids), dtype=np.int32)
    for indices in candidate_indices:
        coverage[indices] += 1

    coordinates_km = coordinates_m / 1000.0
    segments_km = np.stack(
        [coordinates_km[sources], coordinates_km[targets]], axis=1
    )
    hulls = [_convex_hull(coordinates_km[indices]) for indices in candidate_indices]
    centers = np.asarray(
        [coordinates_km[indices].mean(axis=0) for indices in candidate_indices]
    )
    methods = [str(item["selection_method"]) for item in candidates]
    summary = _build_summary(
        manifest,
        coordinates_km,
        candidate_indices,
        hulls,
        coverage,
        methods,
    )
    representatives = _representative_candidates(candidates, centers, hulls)
    figure = _draw_figure(
        coordinates_km,
        segments_km,
        sources,
        targets,
        candidates,
        candidate_indices,
        boundary_indices,
        hulls,
        centers,
        coverage,
        representatives,
        summary,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight", facecolor="#FAFAF8")
    plt.close(figure)
    args.summary.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    args.report.write_text(_render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"image={_display_path(args.output)}")
    print(f"summary={_display_path(args.summary)}")
    print(f"report={_display_path(args.report)}")


def _load_nodes(path: Path) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    node_ids: list[int] = []
    coordinates: list[tuple[float, float]] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            node_ids.append(int(row["node_id"]))
            coordinates.append((float(row["x_m"]), float(row["y_m"])))
    ids = np.asarray(node_ids, dtype=np.int64)
    return ids, np.asarray(coordinates, dtype=np.float64), {
        node_id: index for index, node_id in enumerate(node_ids)
    }


def _load_edges(path: Path, node_to_index: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    sources: list[int] = []
    targets: list[int] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            source = node_to_index.get(int(row["source"]))
            target = node_to_index.get(int(row["target"]))
            if source is not None and target is not None:
                sources.append(source)
                targets.append(target)
    return np.asarray(sources, dtype=np.int64), np.asarray(targets, dtype=np.int64)


def _convex_hull(points: np.ndarray) -> np.ndarray:
    unique = sorted({(float(point[0]), float(point[1])) for point in points})
    if len(unique) <= 2:
        return np.asarray(unique, dtype=np.float64)

    def cross(origin, left, right) -> float:
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (
            left[1] - origin[1]
        ) * (right[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


def _polygon_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    return 0.5 * abs(
        float(
            np.dot(points[:, 0], np.roll(points[:, 1], 1))
            - np.dot(points[:, 1], np.roll(points[:, 0], 1))
        )
    )


def _build_summary(
    manifest: dict,
    coordinates_km: np.ndarray,
    candidate_indices: list[np.ndarray],
    hulls: list[np.ndarray],
    coverage: np.ndarray,
    methods: list[str],
) -> dict:
    spans = np.asarray(
        [
            np.ptp(coordinates_km[indices], axis=0)
            for indices in candidate_indices
        ]
    )
    diagonals = np.linalg.norm(spans, axis=1)
    areas = np.asarray([_polygon_area(hull) for hull in hulls])
    equivalent_diameters = 2.0 * np.sqrt(areas / math.pi)
    covered = coverage > 0
    graph_span = np.ptp(coordinates_km, axis=0)
    memberships = int(sum(len(indices) for indices in candidate_indices))
    return {
        "schema": "aic.gnn_v2.candidate_region_visualization.v1",
        "candidate_sha256": manifest["candidate_sha256"],
        "graph": {
            "node_count": int(manifest["graph"]["node_count"]),
            "edge_count": int(manifest["graph"]["edge_count"]),
            "width_km": float(graph_span[0]),
            "height_km": float(graph_span[1]),
        },
        "candidates": {
            "count": len(candidate_indices),
            "nodes_per_candidate": int(manifest["config"]["region_size"]),
            "selection_method_counts": dict(sorted(Counter(methods).items())),
            "total_node_memberships": memberships,
            "bbox_diagonal_km": _quantiles(diagonals),
            "convex_hull_area_km2": _quantiles(areas),
            "equivalent_diameter_km": _quantiles(equivalent_diameters),
        },
        "coverage": {
            "unique_covered_nodes": int(np.count_nonzero(covered)),
            "covered_node_fraction": float(np.mean(covered)),
            "mean_memberships_all_nodes": float(np.mean(coverage)),
            "mean_memberships_covered_nodes": float(np.mean(coverage[covered])),
            "p90_memberships_covered_nodes": float(np.quantile(coverage[covered], 0.90)),
            "p99_memberships_covered_nodes": float(np.quantile(coverage[covered], 0.99)),
            "max_memberships": int(np.max(coverage)),
        },
    }


def _quantiles(values: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(values)),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(np.max(values)),
    }


def _representative_candidates(
    candidates: list[dict],
    centers: np.ndarray,
    hulls: list[np.ndarray],
) -> list[int]:
    present_methods = sorted(
        {str(candidate["selection_method"]) for candidate in candidates}
    )
    if len(present_methods) == 1:
        ordered = sorted(
            range(len(candidates)),
            key=lambda index: _polygon_area(hulls[index]),
        )
        return [
            ordered[round((len(ordered) - 1) * quantile)]
            for quantile in (0.10, 0.50, 0.90)
        ]
    graph_center = np.median(centers, axis=0)
    representatives: list[int] = []
    for method in present_methods:
        indices = [
            index
            for index, candidate in enumerate(candidates)
            if candidate["selection_method"] == method
        ]
        areas = np.asarray([_polygon_area(hulls[index]) for index in indices])
        median_area = float(np.median(areas))
        scale = max(float(np.std(areas)), 1e-9)
        score = [
            abs(_polygon_area(hulls[index]) - median_area) / scale
            + 0.03 * float(np.linalg.norm(centers[index] - graph_center))
            for index in indices
        ]
        representatives.append(indices[int(np.argmin(score))])
    return representatives


def _draw_figure(
    coordinates: np.ndarray,
    segments: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    candidates: list[dict],
    candidate_indices: list[np.ndarray],
    boundary_indices: list[np.ndarray],
    hulls: list[np.ndarray],
    centers: np.ndarray,
    coverage: np.ndarray,
    representatives: list[int],
    summary: dict,
) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Noto Sans CJK SC", "DejaVu Sans"],
            "axes.unicode_minus": False,
        }
    )
    figure = plt.figure(figsize=(18, 12), facecolor="#FAFAF8")
    grid = figure.add_gridspec(2, 3, height_ratios=(1.55, 1.0), hspace=0.16, wspace=0.08)
    overview = figure.add_subplot(grid[0, :2])
    density = figure.add_subplot(grid[0, 2])
    detail_axes = [figure.add_subplot(grid[1, column]) for column in range(3)]

    _add_roads(overview, segments, color="#4A4A48", linewidth=0.13, alpha=0.20)
    polygons_by_method: dict[str, list[np.ndarray]] = {
        method: [] for method in METHOD_COLORS
    }
    for candidate, hull in zip(candidates, hulls):
        polygons_by_method[candidate["selection_method"]].append(hull)
    for method, polygons in polygons_by_method.items():
        overview.add_collection(
            PolyCollection(
                polygons,
                facecolors=METHOD_COLORS[method],
                edgecolors=METHOD_COLORS[method],
                linewidths=0.18,
                alpha=0.035,
                rasterized=True,
            )
        )
        method_mask = np.asarray(
            [candidate["selection_method"] == method for candidate in candidates]
        )
        overview.scatter(
            centers[method_mask, 0],
            centers[method_mask, 1],
            s=2.5,
            color=METHOD_COLORS[method],
            alpha=0.65,
            linewidths=0,
            rasterized=True,
        )
    _finish_map_axis(overview)
    overview.set_title("A  Porto 路网与全部 1,200 个候选区域", loc="left", fontsize=15, weight="bold")
    overview.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="none",
                color=METHOD_COLORS[method],
                label=f"{METHOD_LABELS[method]}（{count}）",
                markersize=6,
            )
            for method, count in summary["candidates"]["selection_method_counts"].items()
        ],
        loc="lower left",
        frameon=True,
        framealpha=0.92,
        fontsize=9,
    )

    _add_roads(density, segments, color="#A7A7A1", linewidth=0.12, alpha=0.15)
    covered = coverage > 0
    points = density.scatter(
        coordinates[covered, 0],
        coordinates[covered, 1],
        c=np.log1p(coverage[covered]),
        s=1.2,
        cmap="magma",
        alpha=0.9,
        linewidths=0,
        rasterized=True,
    )
    colorbar = figure.colorbar(points, ax=density, fraction=0.045, pad=0.02)
    ticks = np.asarray([1, 2, 5, 10, 20, 50, summary["coverage"]["max_memberships"]])
    ticks = np.unique(ticks[ticks <= summary["coverage"]["max_memberships"]])
    colorbar.set_ticks(np.log1p(ticks))
    colorbar.set_ticklabels([str(int(value)) for value in ticks])
    colorbar.set_label("包含该节点的候选数量", fontsize=9)
    _finish_map_axis(density)
    density.set_title("B  候选覆盖密度", loc="left", fontsize=15, weight="bold")

    for axis, candidate_index in zip(detail_axes, representatives):
        _draw_candidate_detail(
            axis,
            candidate_index,
            coordinates,
            segments,
            sources,
            targets,
            candidates,
            candidate_indices,
            boundary_indices,
        )

    coverage_summary = summary["coverage"]
    size_summary = summary["candidates"]["equivalent_diameter_km"]
    figure.suptitle(
        "Porto 第二版固定候选区域：空间覆盖与实际尺度",
        x=0.055,
        y=0.985,
        ha="left",
        fontsize=20,
        weight="bold",
    )
    figure.text(
        0.055,
        0.952,
        (
            f"候选覆盖 {coverage_summary['covered_node_fraction']:.1%} 的路网节点；"
            f"典型等效直径 {size_summary['median']:.2f} km（P10–P90："
            f"{size_summary['p10']:.2f}–{size_summary['p90']:.2f} km）"
        ),
        ha="left",
        fontsize=11,
        color="#444440",
    )
    return figure


def _draw_candidate_detail(
    axis: plt.Axes,
    candidate_index: int,
    coordinates: np.ndarray,
    segments: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    candidates: list[dict],
    candidate_indices: list[np.ndarray],
    boundary_indices: list[np.ndarray],
) -> None:
    candidate = candidates[candidate_index]
    method = candidate["selection_method"]
    indices = candidate_indices[candidate_index]
    candidate_points = coordinates[indices]
    minimum = candidate_points.min(axis=0)
    maximum = candidate_points.max(axis=0)
    span = np.maximum(maximum - minimum, 0.2)
    margin = max(float(np.max(span)) * 0.25, 0.25)
    lower = minimum - margin
    upper = maximum + margin
    local_edges = (
        (segments[:, :, 0].max(axis=1) >= lower[0])
        & (segments[:, :, 0].min(axis=1) <= upper[0])
        & (segments[:, :, 1].max(axis=1) >= lower[1])
        & (segments[:, :, 1].min(axis=1) <= upper[1])
    )
    _add_roads(axis, segments[local_edges], color="#B2B2AC", linewidth=0.35, alpha=0.48)
    membership = np.zeros(len(coordinates), dtype=bool)
    membership[indices] = True
    internal_edges = membership[sources] & membership[targets]
    _add_roads(
        axis,
        segments[internal_edges],
        color=METHOD_COLORS[method],
        linewidth=0.8,
        alpha=0.80,
    )
    axis.scatter(
        candidate_points[:, 0],
        candidate_points[:, 1],
        s=2.0,
        color=METHOD_COLORS[method],
        alpha=0.55,
        linewidths=0,
        rasterized=True,
    )
    boundary = coordinates[boundary_indices[candidate_index]]
    axis.scatter(
        boundary[:, 0],
        boundary[:, 1],
        s=8,
        facecolors="#FFFDF5",
        edgecolors="#151513",
        linewidths=0.45,
        zorder=5,
    )
    axis.set_xlim(lower[0], upper[0])
    axis.set_ylim(lower[1], upper[1])
    _finish_map_axis(axis)
    diagonal = float(np.linalg.norm(maximum - minimum))
    axis.set_title(
        f"{METHOD_LABELS[method]} · 区域 {candidate['region_id']}\n"
        f"512 节点，边界 {candidate['boundary_count']}，包围盒对角线 {diagonal:.2f} km",
        loc="left",
        fontsize=11,
        weight="bold",
    )
    _add_scale_bar(axis)


def _add_roads(
    axis: plt.Axes,
    segments: np.ndarray,
    *,
    color: str,
    linewidth: float,
    alpha: float,
) -> None:
    axis.add_collection(
        LineCollection(
            segments,
            colors=color,
            linewidths=linewidth,
            alpha=alpha,
            rasterized=True,
        )
    )


def _finish_map_axis(axis: plt.Axes) -> None:
    axis.autoscale_view()
    axis.set_aspect("equal", adjustable="box")
    axis.set_axis_off()


def _add_scale_bar(axis: plt.Axes) -> None:
    width = axis.get_xlim()[1] - axis.get_xlim()[0]
    length = 1.0 if width >= 3.0 else 0.5
    x_start = axis.get_xlim()[0] + width * 0.07
    y = axis.get_ylim()[0] + (axis.get_ylim()[1] - axis.get_ylim()[0]) * 0.08
    axis.plot([x_start, x_start + length], [y, y], color="#151513", linewidth=2.0)
    axis.text(
        x_start + length / 2,
        y,
        f"{length:g} km",
        ha="center",
        va="bottom",
        fontsize=8,
        color="#151513",
    )


def _render_report(summary: dict) -> str:
    candidate = summary["candidates"]
    coverage = summary["coverage"]
    diagonal = candidate["bbox_diagonal_km"]
    diameter = candidate["equivalent_diameter_km"]
    if len(candidate["selection_method_counts"]) == 1:
        detail_description = (
            "底部三个面板分别展示凸包面积约位于 P10、P50 和 P90 的固定随机候选。"
        )
    else:
        detail_description = "底部面板分别展示各候选生成方法的典型区域。"
    return "\n".join(
        [
            "# Porto 第二版候选区域空间可视化",
            "",
            f"- 候选数量：{candidate['count']:,}；每个候选 {candidate['nodes_per_candidate']} 个节点。",
            f"- 路网节点覆盖：{coverage['unique_covered_nodes']:,}，占全部节点 {coverage['covered_node_fraction']:.2%}。",
            f"- 被覆盖节点平均出现在 {coverage['mean_memberships_covered_nodes']:.2f} 个候选中，最大为 {coverage['max_memberships']} 个。",
            f"- 候选包围盒对角线中位数：{diagonal['median']:.3f} km；P10–P90 为 {diagonal['p10']:.3f}–{diagonal['p90']:.3f} km。",
            f"- 候选凸包等效直径中位数：{diameter['median']:.3f} km；P10–P90 为 {diameter['p10']:.3f}–{diameter['p90']:.3f} km。",
            "",
            "图中 A 面板叠加全部候选凸包和中心，B 面板显示每个节点被多少候选覆盖，"
            + detail_description,
            "",
        ]
    )


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT_DIR))
    except ValueError:
        return str(path.resolve())


if __name__ == "__main__":
    main()
