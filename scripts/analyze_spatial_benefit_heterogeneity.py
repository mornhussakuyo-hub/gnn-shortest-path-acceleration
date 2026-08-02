"""Map query-level deployment gains for Z0, BRIDGE, and BRIDGE-B."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WINDOWS = ("current_y", "future_f")
METHODS = ("BRIDGE", "BRIDGE-B")
SEEDS = (42, 43, 44)
GRID_SIZE = 16
MIN_CELL_QUERIES = 5
HISTORY_FRACTION = 0.35
HOT_CELL_FRACTION = 0.20


@dataclass(frozen=True)
class CityConfig:
    name: str
    nodes: Path
    queries: Path
    z0_dir: Path
    bridge_dir: Path
    bridge_b_seed42_dir: Path
    bridge_b_seed43_44_dir: Path


CITY_CONFIGS = {
    "porto": CityConfig(
        name="porto",
        nodes=ROOT / "data/processed/porto/波尔图道路节点.csv",
        queries=ROOT / "data/processed/porto/波尔图可用起终点节点查询_200米.csv",
        z0_dir=ROOT / "results/gnn_v2/multi_region_online/details",
        bridge_dir=ROOT / "results/gnn_v2/multi_region_online_g4/details",
        bridge_b_seed42_dir=(
            ROOT
            / "results/gnn_v2/g5_cost_aware_exploration/s2_gain1_short_seed42_online_k18/details"
        ),
        bridge_b_seed43_44_dir=(
            ROOT
            / "results/gnn_v2/g5_cost_aware_exploration/s3_gain1_seeds43_44_short_online_k18/details"
        ),
    ),
    "chicago": CityConfig(
        name="chicago",
        nodes=ROOT / "data/processed/chicago/chicago_road_nodes.csv",
        queries=ROOT / "data/processed/chicago/chicago_queries_100k.csv",
        z0_dir=(
            ROOT / "results/chicago/gnn_v2/multi_region_online_g4_clean_rerun/details"
        ),
        bridge_dir=(
            ROOT / "results/chicago/gnn_v2/multi_region_online_g4_clean_rerun/details"
        ),
        bridge_b_seed42_dir=(
            ROOT
            / "results/chicago/gnn_v2/g5_cost_aware_exploration/s2_gain1_short_seed42_online_k18/details"
        ),
        bridge_b_seed43_44_dir=(
            ROOT
            / "results/chicago/gnn_v2/g5_cost_aware_exploration/s3_gain1_seeds43_44_short_online_k18/details"
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--city", choices=("all", *CITY_CONFIGS), default="all")
    parser.add_argument("--grid-size", type=int, default=GRID_SIZE)
    parser.add_argument("--min-cell-queries", type=int, default=MIN_CELL_QUERIES)
    return parser.parse_args()


def _detail_path(config: CityConfig, method: str, seed: int, window: str) -> Path:
    if method == "Z0":
        return config.z0_dir / f"z0.k18.{window}.csv.gz"
    if method == "BRIDGE":
        return config.bridge_dir / f"g4_global_seed{seed}.k18.{window}.csv.gz"
    if method == "BRIDGE-B" and seed == 42:
        return config.bridge_b_seed42_dir / f"g5_s2_seed42.k18.{window}.csv.gz"
    if method == "BRIDGE-B":
        return config.bridge_b_seed43_44_dir / f"g5_s3_seed{seed}.k18.{window}.csv.gz"
    raise ValueError(f"unknown method {method}")


def _load_nodes(path: Path) -> tuple[dict[int, tuple[float, float]], tuple[float, ...]]:
    coordinates: dict[int, tuple[float, float]] = {}
    xs: list[float] = []
    ys: list[float] = []
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            node_id = int(row["node_id"])
            x = float(row["x_m"])
            y = float(row["y_m"])
            coordinates[node_id] = (x, y)
            xs.append(x)
            ys.append(y)
    if not coordinates:
        raise ValueError(f"empty node table: {path}")
    return coordinates, (min(xs), max(xs), min(ys), max(ys))


def _load_queries(path: Path) -> list[dict[str, int]]:
    queries: list[dict[str, int]] = []
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row.get("snap_usable", "True") != "True":
                continue
            queries.append(
                {
                    "query_id": int(row["query_id"]),
                    "timestamp": int(row["timestamp"]),
                    "origin": int(row["origin_node"]),
                    "destination": int(row["dest_node"]),
                }
            )
    queries.sort(key=lambda row: (row["timestamp"], row["query_id"]))
    return queries


def _cell_index(
    x: float,
    y: float,
    bounds: tuple[float, ...],
    grid_size: int,
) -> tuple[int, int]:
    min_x, max_x, min_y, max_y = bounds
    x_index = min(grid_size - 1, max(0, int((x - min_x) / (max_x - min_x) * grid_size)))
    y_index = min(grid_size - 1, max(0, int((y - min_y) / (max_y - min_y) * grid_size)))
    return x_index, y_index


def _history_hot_cells(
    queries: list[dict[str, int]],
    coordinates: dict[int, tuple[float, float]],
    bounds: tuple[float, ...],
    grid_size: int,
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int], int]:
    history_count = math.floor(len(queries) * HISTORY_FRACTION)
    counts: dict[tuple[int, int], int] = defaultdict(int)
    for query in queries[:history_count]:
        for endpoint in (query["origin"], query["destination"]):
            counts[_cell_index(*coordinates[endpoint], bounds, grid_size)] += 1
    ranked = sorted(counts, key=lambda cell: (-counts[cell], cell))
    hot_count = max(1, math.ceil(len(ranked) * HOT_CELL_FRACTION))
    return set(ranked[:hot_count]), dict(counts), history_count


def _load_detail(path: Path) -> dict[int, dict[str, int]]:
    output: dict[int, dict[str, int]] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            if row["correct"] != "True":
                raise ValueError(f"incorrect query in {path}: {row['query_id']}")
            query_id = int(row["query_id"])
            if query_id in output:
                raise ValueError(f"duplicate query {query_id} in {path}")
            output[query_id] = {
                "origin": int(row["origin"]),
                "destination": int(row["destination"]),
                "baseline_expanded": int(row["baseline_expanded"]),
                "indexed_expanded": int(row["indexed_expanded"]),
            }
    if len(output) != 2_000:
        raise ValueError(f"expected 2,000 queries in {path}, found {len(output)}")
    return output


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def analyze_city(config: CityConfig, *, grid_size: int, min_cell_queries: int) -> None:
    coordinates, bounds = _load_nodes(config.nodes)
    queries = _load_queries(config.queries)
    query_lookup = {row["query_id"]: row for row in queries}
    hot_cells, history_counts, history_count = _history_hot_cells(
        queries, coordinates, bounds, grid_size
    )
    output_dir = ROOT / "results/spatial_benefit_heterogeneity" / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = [config.nodes, config.queries]
    detailed_rows: list[dict[str, object]] = []
    for window in WINDOWS:
        z0_path = _detail_path(config, "Z0", 42, window)
        source_paths.append(z0_path)
        z0 = _load_detail(z0_path)
        for method in METHODS:
            for seed in SEEDS:
                path = _detail_path(config, method, seed, window)
                source_paths.append(path)
                current = _load_detail(path)
                if set(current) != set(z0):
                    raise ValueError(f"query set mismatch: {path}")
                for query_id in sorted(z0):
                    reference = z0[query_id]
                    result = current[query_id]
                    if (
                        reference["origin"] != result["origin"]
                        or reference["destination"] != result["destination"]
                        or reference["baseline_expanded"] != result["baseline_expanded"]
                    ):
                        raise ValueError(f"query identity mismatch in {path}: {query_id}")
                    query = query_lookup[query_id]
                    if query["origin"] != result["origin"] or query["destination"] != result["destination"]:
                        raise ValueError(f"query CSV mismatch in {path}: {query_id}")
                    origin_x, origin_y = coordinates[result["origin"]]
                    destination_x, destination_y = coordinates[result["destination"]]
                    midpoint_x = 0.5 * (origin_x + destination_x)
                    midpoint_y = 0.5 * (origin_y + destination_y)
                    midpoint_cell = _cell_index(midpoint_x, midpoint_y, bounds, grid_size)
                    origin_cell = _cell_index(origin_x, origin_y, bounds, grid_size)
                    destination_cell = _cell_index(destination_x, destination_y, bounds, grid_size)
                    detailed_rows.append(
                        {
                            "city": config.name,
                            "window": window,
                            "method": method,
                            "seed": seed,
                            "query_id": query_id,
                            "origin": result["origin"],
                            "destination": result["destination"],
                            "origin_x": origin_x,
                            "origin_y": origin_y,
                            "destination_x": destination_x,
                            "destination_y": destination_y,
                            "midpoint_x": midpoint_x,
                            "midpoint_y": midpoint_y,
                            "cell_x": midpoint_cell[0],
                            "cell_y": midpoint_cell[1],
                            "historical_stratum": (
                                "head"
                                if origin_cell in hot_cells or destination_cell in hot_cells
                                else "non_head"
                            ),
                            "baseline_expanded": result["baseline_expanded"],
                            "z0_indexed_expanded": reference["indexed_expanded"],
                            "method_indexed_expanded": result["indexed_expanded"],
                            "absolute_gain": result["baseline_expanded"] - result["indexed_expanded"],
                            "delta_vs_z0": reference["indexed_expanded"] - result["indexed_expanded"],
                        }
                    )

    detail_path = output_dir / "query_deltas.csv.gz"
    _write_csv_gz(detail_path, detailed_rows)
    seed_mean_rows = _seed_mean_rows(detailed_rows)
    grid_rows = _grid_rows(seed_mean_rows, bounds, grid_size, min_cell_queries)
    _write_csv(output_dir / "grid_summary.csv", grid_rows)
    stratum_rows = _stratum_rows(detailed_rows)
    _write_csv(output_dir / "stratum_summary.csv", stratum_rows)
    length_rows = _length_stratification_rows(seed_mean_rows)
    _write_csv(output_dir / "length_stratification.csv", length_rows)
    gates = _spatial_gates(stratum_rows, grid_rows)

    figure_paths = _plot_city(
        config.name,
        coordinates,
        bounds,
        grid_rows,
        seed_mean_rows,
        output_dir,
        grid_size=grid_size,
        min_cell_queries=min_cell_queries,
    )
    summary = {
        "schema": "aic.spatial_benefit_heterogeneity.v1",
        "city": config.name,
        "methods": list(METHODS),
        "seeds": list(SEEDS),
        "windows": list(WINDOWS),
        "grid_size": grid_size,
        "min_cell_queries": min_cell_queries,
        "history_fraction": HISTORY_FRACTION,
        "history_query_count": history_count,
        "hot_cell_fraction": HOT_CELL_FRACTION,
        "nonempty_history_cell_count": len(history_counts),
        "hot_cell_count": len(hot_cells),
        "query_count_per_window": 2_000,
        "query_seed_rows": len(detailed_rows),
        "interpretation_gates": gates,
        "source_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in source_paths},
        "artifacts": {
            "query_deltas": str(detail_path.relative_to(ROOT)),
            "grid_summary": str((output_dir / "grid_summary.csv").relative_to(ROOT)),
            "stratum_summary": str((output_dir / "stratum_summary.csv").relative_to(ROOT)),
            "length_stratification": str(
                (output_dir / "length_stratification.csv").relative_to(ROOT)
            ),
            "figures": [str(path.relative_to(ROOT)) for path in figure_paths],
        },
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_report(output_dir / "report.md", summary, stratum_rows, grid_rows)
    print(f"completed {config.name}: {output_dir / 'report.md'}", flush=True)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_csv_gz(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    with gzip.open(path, "wt", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _seed_mean_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["window"]), int(row["query_id"]))].append(row)
    output: list[dict[str, object]] = []
    for (method, window, query_id), values in sorted(grouped.items()):
        if sorted(int(row["seed"]) for row in values) != list(SEEDS):
            raise ValueError(f"incomplete seeds for {(method, window, query_id)}")
        first = values[0]
        deltas = [float(row["delta_vs_z0"]) for row in values]
        output.append(
            {
                "method": method,
                "window": window,
                "query_id": query_id,
                "cell_x": int(first["cell_x"]),
                "cell_y": int(first["cell_y"]),
                "origin_x": float(first["origin_x"]),
                "origin_y": float(first["origin_y"]),
                "destination_x": float(first["destination_x"]),
                "destination_y": float(first["destination_y"]),
                "midpoint_x": float(first["midpoint_x"]),
                "midpoint_y": float(first["midpoint_y"]),
                "historical_stratum": first["historical_stratum"],
                "baseline_expanded": int(first["baseline_expanded"]),
                "absolute_gain_mean": float(
                    np.mean([float(row["absolute_gain"]) for row in values])
                ),
                "delta_mean": float(np.mean(deltas)),
                "delta_median": float(np.median(deltas)),
                "positive_seed_count": sum(value > 0 for value in deltas),
                "negative_seed_count": sum(value < 0 for value in deltas),
            }
        )
    return output


def _length_stratification_rows(
    rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for method in METHODS:
        for window in WINDOWS:
            selected = [
                row for row in rows if row["method"] == method and row["window"] == window
            ]
            if len(selected) != 2_000:
                raise ValueError(f"expected 2,000 seed-mean rows for {(method, window)}")
            values = []
            for row in selected:
                od_span_m = math.hypot(
                    float(row["destination_x"]) - float(row["origin_x"]),
                    float(row["destination_y"]) - float(row["origin_y"]),
                )
                values.append(
                    {
                        "query_id": int(row["query_id"]),
                        "baseline_expanded": float(row["baseline_expanded"]),
                        "od_span_m": od_span_m,
                        "absolute_gain_mean": float(row["absolute_gain_mean"]),
                        "delta_mean": float(row["delta_mean"]),
                    }
                )
            for basis, field in (
                ("baseline_expanded", "baseline_expanded"),
                ("od_span_m", "od_span_m"),
            ):
                ordered = sorted(values, key=lambda row: (float(row[field]), int(row["query_id"])))
                for quartile, indices in enumerate(np.array_split(np.arange(len(ordered)), 4), start=1):
                    group = [ordered[int(index)] for index in indices]
                    baseline_total = sum(float(row["baseline_expanded"]) for row in group)
                    gain_total = sum(float(row["absolute_gain_mean"]) for row in group)
                    output.append(
                        {
                            "method": method,
                            "window": window,
                            "stratification": basis,
                            "quartile": quartile,
                            "query_count": len(group),
                            "scale_mean": float(np.mean([float(row[field]) for row in group])),
                            "baseline_expanded_mean": float(
                                np.mean([float(row["baseline_expanded"]) for row in group])
                            ),
                            "absolute_gain_mean": float(
                                np.mean([float(row["absolute_gain_mean"]) for row in group])
                            ),
                            "expanded_reduction_pct": 100.0 * gain_total / baseline_total,
                            "delta_vs_z0_mean": float(
                                np.mean([float(row["delta_mean"]) for row in group])
                            ),
                            "improved_vs_z0_query_fraction": float(
                                np.mean([float(row["delta_mean"]) > 0 for row in group])
                            ),
                        }
                    )
    return output


def _grid_rows(
    rows: list[dict[str, object]],
    bounds: tuple[float, ...],
    grid_size: int,
    min_cell_queries: int,
) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, int], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["window"]), int(row["cell_x"]), int(row["cell_y"]))].append(row)
    min_x, max_x, min_y, max_y = bounds
    width = (max_x - min_x) / grid_size
    height = (max_y - min_y) / grid_size
    output: list[dict[str, object]] = []
    for (method, window, cell_x, cell_y), values in sorted(grouped.items()):
        deltas = np.asarray([float(row["delta_mean"]) for row in values], dtype=np.float64)
        output.append(
            {
                "method": method,
                "window": window,
                "cell_x": cell_x,
                "cell_y": cell_y,
                "center_x": min_x + (cell_x + 0.5) * width,
                "center_y": min_y + (cell_y + 0.5) * height,
                "query_count": len(values),
                "valid_for_color": len(values) >= min_cell_queries,
                "delta_mean": float(np.mean(deltas)),
                "delta_median": float(np.median(deltas)),
                "delta_q25": float(np.quantile(deltas, 0.25)),
                "delta_q75": float(np.quantile(deltas, 0.75)),
                "improved_query_fraction": float(np.mean(deltas > 0)),
                "tied_query_fraction": float(np.mean(deltas == 0)),
                "degraded_query_fraction": float(np.mean(deltas < 0)),
                "all_seed_positive_query_fraction": float(
                    np.mean([int(row["positive_seed_count"]) == len(SEEDS) for row in values])
                ),
            }
        )
    return output


def _stratum_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, object, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["method"]), str(row["window"]), row["seed"], str(row["historical_stratum"]))].append(
            float(row["delta_vs_z0"])
        )
    output: list[dict[str, object]] = []
    for (method, window, seed, stratum), values in sorted(grouped.items()):
        array = np.asarray(values, dtype=np.float64)
        output.append(
            {
                "method": method,
                "window": window,
                "seed": seed,
                "stratum": stratum,
                "query_count": len(array),
                "delta_mean": float(np.mean(array)),
                "delta_median": float(np.median(array)),
                "improved_query_fraction": float(np.mean(array > 0)),
                "tied_query_fraction": float(np.mean(array == 0)),
                "degraded_query_fraction": float(np.mean(array < 0)),
            }
        )
    for method in METHODS:
        for window in WINDOWS:
            for stratum in ("head", "non_head"):
                selected = [
                    row
                    for row in rows
                    if row["method"] == method
                    and row["window"] == window
                    and row["historical_stratum"] == stratum
                ]
                by_query: dict[int, list[float]] = defaultdict(list)
                for row in selected:
                    by_query[int(row["query_id"])].append(float(row["delta_vs_z0"]))
                query_means = []
                for query_id, values in by_query.items():
                    if len(values) != len(SEEDS):
                        raise ValueError(f"incomplete seeds for stratum query {query_id}")
                    query_means.append(float(np.mean(values)))
                array = np.asarray(query_means, dtype=np.float64)
                output.append(
                    {
                        "method": method,
                        "window": window,
                        "seed": "seed_mean",
                        "stratum": stratum,
                        "query_count": len(array),
                        "delta_mean": float(np.mean(array)),
                        "delta_median": float(np.median(array)),
                        "improved_query_fraction": float(np.mean(array > 0)),
                        "tied_query_fraction": float(np.mean(array == 0)),
                        "degraded_query_fraction": float(np.mean(array < 0)),
                    }
                )
    return output


def _spatial_gates(
    stratum_rows: list[dict[str, object]],
    grid_rows: list[dict[str, object]],
) -> dict[str, object]:
    output: dict[str, object] = {}
    for method in METHODS:
        windows: dict[str, object] = {}
        for window in WINDOWS:
            non_head_seed_rows = [
                row
                for row in stratum_rows
                if row["method"] == method
                and row["window"] == window
                and row["stratum"] == "non_head"
                and row["seed"] != "seed_mean"
            ]
            mean_rows = {
                str(row["stratum"]): row
                for row in stratum_rows
                if row["method"] == method
                and row["window"] == window
                and row["seed"] == "seed_mean"
            }
            valid_cells = [
                row
                for row in grid_rows
                if row["method"] == method
                and row["window"] == window
                and row["valid_for_color"]
            ]
            improved_cell_fraction = float(
                np.mean([float(row["delta_mean"]) > 0 for row in valid_cells])
            )
            windows[window] = {
                "positive_non_head_seed_count": sum(
                    float(row["delta_mean"]) > 0 for row in non_head_seed_rows
                ),
                "non_head_delta_mean": float(mean_rows["non_head"]["delta_mean"]),
                "head_delta_mean": float(mean_rows["head"]["delta_mean"]),
                "improved_valid_cell_fraction": improved_cell_fraction,
                "valid_cell_count": len(valid_cells),
            }
        non_head_gate = all(
            int(windows[window]["positive_non_head_seed_count"]) >= 2
            and float(windows[window]["non_head_delta_mean"]) > 0
            and float(windows[window]["improved_valid_cell_fraction"]) > 0.5
            for window in WINDOWS
        )
        head_values = [float(windows[window]["head_delta_mean"]) for window in WINDOWS]
        if non_head_gate and all(value < 0 for value in head_values):
            interpretation = "head_to_non_head_tradeoff"
        elif non_head_gate and all(value > 0 for value in head_values):
            interpretation = "benefit_expansion"
        elif non_head_gate:
            interpretation = "mixed_head_with_non_head_broadening"
        else:
            interpretation = "no_stable_spatial_broadening"
        output[method] = {
            "windows": windows,
            "non_head_broadening_gate": non_head_gate,
            "interpretation": interpretation,
        }
    return output


def _plot_city(
    city: str,
    coordinates: dict[int, tuple[float, float]],
    bounds: tuple[float, ...],
    grid_rows: list[dict[str, object]],
    seed_mean_rows: list[dict[str, object]],
    output_dir: Path,
    *,
    grid_size: int,
    min_cell_queries: int,
) -> list[Path]:
    valid_values = [
        abs(float(row["delta_mean"])) for row in grid_rows if row["valid_for_color"]
    ]
    color_limit = max(valid_values) if valid_values else 1.0
    min_x, max_x, min_y, max_y = bounds
    road = np.asarray(list(coordinates.values()), dtype=np.float64)
    stride = max(1, len(road) // 12_000)
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 8.0), constrained_layout=True)
    image = None
    for row_index, method in enumerate(METHODS):
        for column_index, window in enumerate(WINDOWS):
            ax = axes[row_index, column_index]
            ax.scatter(road[::stride, 0], road[::stride, 1], s=0.18, color="#CBD2D5", alpha=0.45, rasterized=True)
            array = np.full((grid_size, grid_size), np.nan)
            selected = [
                row
                for row in grid_rows
                if row["method"] == method
                and row["window"] == window
                and row["valid_for_color"]
            ]
            for row in selected:
                array[int(row["cell_x"]), int(row["cell_y"])] = float(row["delta_mean"])
            image = ax.imshow(
                array.T,
                origin="lower",
                extent=(min_x, max_x, min_y, max_y),
                cmap="RdBu",
                vmin=-color_limit,
                vmax=color_limit,
                interpolation="nearest",
                alpha=0.82,
            )
            ax.set_title(f"{method} · {'Y' if window == 'current_y' else 'F'}", fontsize=10, weight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            ax.text(
                0.02,
                0.02,
                f"valid cells={len(selected)}",
                transform=ax.transAxes,
                fontsize=7,
                color="#263238",
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 2},
            )
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, location="bottom", shrink=0.72, pad=0.035)
        colorbar.set_label("Mean expanded-node delta vs. Z0 per query  (positive = better)")
    fig.suptitle(f"{city.title()}: spatial deployment benefit relative to Z0", fontsize=13, weight="bold")
    paper_base = ROOT / "paper/figures" / f"spatial_benefit_{city}"
    output_base = output_dir / "spatial_benefit"
    paths = []
    for base in (paper_base, output_base):
        base.parent.mkdir(parents=True, exist_ok=True)
        for extension, dpi in (("pdf", 300), ("png", 220)):
            path = base.with_suffix(f".{extension}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
            paths.append(path)
    plt.close(fig)
    paths.extend(
        _plot_endpoint_appendix(
            city,
            coordinates,
            bounds,
            seed_mean_rows,
            output_dir,
            grid_size=grid_size,
            min_cell_queries=min_cell_queries,
        )
    )
    return paths


def _plot_endpoint_appendix(
    city: str,
    coordinates: dict[int, tuple[float, float]],
    bounds: tuple[float, ...],
    seed_mean_rows: list[dict[str, object]],
    output_dir: Path,
    *,
    grid_size: int,
    min_cell_queries: int,
) -> list[Path]:
    endpoint_grids: dict[tuple[str, str, str], dict[tuple[int, int], list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for row in seed_mean_rows:
        for endpoint in ("origin", "destination"):
            cell = _cell_index(
                float(row[f"{endpoint}_x"]),
                float(row[f"{endpoint}_y"]),
                bounds,
                grid_size,
            )
            endpoint_grids[(str(row["method"]), str(row["window"]), endpoint)][cell].append(
                float(row["delta_mean"])
            )
    valid_values = [
        abs(float(np.mean(values)))
        for cells in endpoint_grids.values()
        for values in cells.values()
        if len(values) >= min_cell_queries
    ]
    color_limit = max(valid_values) if valid_values else 1.0
    min_x, max_x, min_y, max_y = bounds
    road = np.asarray(list(coordinates.values()), dtype=np.float64)
    stride = max(1, len(road) // 12_000)
    columns = tuple((window, endpoint) for window in WINDOWS for endpoint in ("origin", "destination"))
    fig, axes = plt.subplots(2, 4, figsize=(12.6, 6.2), constrained_layout=True)
    image = None
    for row_index, method in enumerate(METHODS):
        for column_index, (window, endpoint) in enumerate(columns):
            ax = axes[row_index, column_index]
            ax.scatter(
                road[::stride, 0],
                road[::stride, 1],
                s=0.14,
                color="#CBD2D5",
                alpha=0.42,
                rasterized=True,
            )
            array = np.full((grid_size, grid_size), np.nan)
            cells = endpoint_grids[(method, window, endpoint)]
            valid_count = 0
            for (cell_x, cell_y), values in cells.items():
                if len(values) < min_cell_queries:
                    continue
                array[cell_x, cell_y] = float(np.mean(values))
                valid_count += 1
            image = ax.imshow(
                array.T,
                origin="lower",
                extent=(min_x, max_x, min_y, max_y),
                cmap="RdBu",
                vmin=-color_limit,
                vmax=color_limit,
                interpolation="nearest",
                alpha=0.82,
            )
            endpoint_label = "O" if endpoint == "origin" else "D"
            window_label = "Y" if window == "current_y" else "F"
            ax.set_title(f"{method} · {window_label}-{endpoint_label}", fontsize=8.5, weight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            ax.text(
                0.02,
                0.02,
                f"n={valid_count}",
                transform=ax.transAxes,
                fontsize=6.5,
                color="#263238",
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.5},
            )
    if image is not None:
        colorbar = fig.colorbar(image, ax=axes, location="bottom", shrink=0.62, pad=0.035)
        colorbar.set_label("Mean expanded-node delta vs. Z0 per query  (positive = better)")
    fig.suptitle(
        f"{city.title()}: origin/destination spatial benefit appendix",
        fontsize=12,
        weight="bold",
    )
    paper_base = ROOT / "paper/figures" / f"spatial_benefit_{city}_endpoints"
    output_base = output_dir / "spatial_benefit_endpoints"
    paths = []
    for base in (paper_base, output_base):
        base.parent.mkdir(parents=True, exist_ok=True)
        for extension, dpi in (("pdf", 300), ("png", 220)):
            path = base.with_suffix(f".{extension}")
            fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
            paths.append(path)
    plt.close(fig)
    return paths


def _write_report(
    path: Path,
    summary: dict[str, object],
    stratum_rows: list[dict[str, object]],
    grid_rows: list[dict[str, object]],
) -> None:
    lines = [
        f"# {str(summary['city']).title()} 空间收益异质性",
        "",
        "主量为神经方法相对 Z0 的逐查询展开节点差；正值表示神经方法更少展开。"
        "热点仅由历史 H 的端点需求定义，Y/F 不参与空间分层。",
        "",
        "| 方法 | 窗口 | 分层 | 平均差 | 改善查询 | 退化查询 |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for row in stratum_rows:
        if row["seed"] != "seed_mean":
            continue
        lines.append(
            f"| {row['method']} | {row['window']} | {row['stratum']} | "
            f"{row['delta_mean']:+.3f} | {100 * float(row['improved_query_fraction']):.2f}% | "
            f"{100 * float(row['degraded_query_fraction']):.2f}% |"
        )
    lines.extend(["", "## 预注册解释门", ""])
    for method, result in summary["interpretation_gates"].items():
        lines.append(
            f"- **{method}**：`{result['interpretation']}`；非头部扩展门 "
            f"`{'通过' if result['non_head_broadening_gate'] else '未通过'}`。"
        )
    valid_count = sum(bool(row["valid_for_color"]) for row in grid_rows)
    lines.extend(
        [
            "",
            f"固定 {summary['grid_size']}×{summary['grid_size']} 网格、最少 "
            f"{summary['min_cell_queries']} 条查询着色，共 {valid_count} 个方法—窗口有效网格记录。",
            "改善、持平和退化记录均保留；本分析是已解锁 Y/F 上的机制诊断，不是新的时间外确认。",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.grid_size <= 1 or args.min_cell_queries <= 0:
        raise SystemExit("grid size must exceed 1 and min cell queries must be positive")
    cities = CITY_CONFIGS.values() if args.city == "all" else (CITY_CONFIGS[args.city],)
    for city in cities:
        analyze_city(city, grid_size=args.grid_size, min_cell_queries=args.min_cell_queries)


if __name__ == "__main__":
    main()
