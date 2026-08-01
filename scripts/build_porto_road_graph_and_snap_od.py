"""从 OSM PBF 中抽取城市路网，并将 OD 坐标吸附到道路节点。"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import osmium
from pyproj import Transformer


DEFAULT_OD_INPUT = Path("data/processed/porto/波尔图起终点样本_10万.csv")
DEFAULT_OSM_INPUT = Path("data/compressed/porto/portugal-latest.osm.pbf")
DEFAULT_NODE_OUT = Path("data/processed/porto/波尔图道路节点.csv")
DEFAULT_EDGE_OUT = Path("data/processed/porto/波尔图道路边.csv")
DEFAULT_QUERY_OUT = Path("data/processed/porto/波尔图起终点节点查询_10万.csv")
DEFAULT_USABLE_QUERY_OUT = Path("data/processed/porto/波尔图可用起终点节点查询_200米.csv")
DEFAULT_REPORT_OUT = Path("data/processed/porto/波尔图起终点吸附质量报告.md")

ROAD_TYPES = {
    "motorway",
    "trunk",
    "primary",
    "secondary",
    "tertiary",
    "unclassified",
    "residential",
    "living_street",
    "motorway_link",
    "trunk_link",
    "primary_link",
    "secondary_link",
    "tertiary_link",
}

FORWARD_ONEWAY = {"yes", "true", "1"}
REVERSE_ONEWAY = {"-1", "reverse"}


@dataclass(frozen=True)
class ODRow:
    trip_id: str
    timestamp: int
    origin_lon: float
    origin_lat: float
    dest_lon: float
    dest_lat: float


@dataclass(frozen=True)
class RoadNode:
    node_id: int
    lon: float
    lat: float
    x: float
    y: float


@dataclass(frozen=True)
class RoadEdge:
    source: int
    target: int
    length_m: float
    highway: str
    osm_way_id: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--od-input", type=Path, default=DEFAULT_OD_INPUT)
    parser.add_argument("--osm-input", type=Path, default=DEFAULT_OSM_INPUT)
    parser.add_argument("--node-out", type=Path, default=DEFAULT_NODE_OUT)
    parser.add_argument("--edge-out", type=Path, default=DEFAULT_EDGE_OUT)
    parser.add_argument("--query-out", type=Path, default=DEFAULT_QUERY_OUT)
    parser.add_argument("--usable-query-out", type=Path, default=DEFAULT_USABLE_QUERY_OUT)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT_OUT)
    parser.add_argument("--snap-threshold-m", type=float, default=200.0)
    parser.add_argument("--grid-cell-m", type=float, default=300.0)
    parser.add_argument(
        "--projected-crs",
        default="EPSG:3763",
        help="Local projected CRS used for metre distances (Porto default: EPSG:3763).",
    )
    parser.add_argument("--city-name", default="波尔图")
    return parser.parse_args()


def read_od_rows(path: Path) -> list[ODRow]:
    rows: list[ODRow] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                ODRow(
                    trip_id=row["trip_id"],
                    timestamp=int(row["timestamp"]),
                    origin_lon=float(row["origin_lon"]),
                    origin_lat=float(row["origin_lat"]),
                    dest_lon=float(row["dest_lon"]),
                    dest_lat=float(row["dest_lat"]),
                )
            )
    return rows


def robust_bounds(values: np.ndarray, low: float = 0.5, high: float = 99.5) -> tuple[float, float]:
    lo, hi = np.percentile(values, [low, high])
    pad = (hi - lo) * 0.08
    return float(lo - pad), float(hi + pad)


def od_bounds(rows: list[ODRow]) -> tuple[tuple[float, float], tuple[float, float]]:
    lon = np.array([value for row in rows for value in (row.origin_lon, row.dest_lon)], dtype=float)
    lat = np.array([value for row in rows for value in (row.origin_lat, row.dest_lat)], dtype=float)
    return robust_bounds(lon), robust_bounds(lat)


def is_forward_oneway(tags: osmium.osm.TagList) -> bool:
    oneway = tags.get("oneway")
    junction = tags.get("junction")
    return oneway in FORWARD_ONEWAY or junction == "roundabout"


def is_reverse_oneway(tags: osmium.osm.TagList) -> bool:
    return tags.get("oneway") in REVERSE_ONEWAY


class RoadGraphCollector(osmium.SimpleHandler):
    def __init__(
        self,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
        projected_crs: str,
    ) -> None:
        super().__init__()
        self.xlim = xlim
        self.ylim = ylim
        self.transformer = Transformer.from_crs(
            "EPSG:4326", projected_crs, always_xy=True
        )
        self.nodes: dict[int, RoadNode] = {}
        self.edges: list[RoadEdge] = []

    def way(self, way: osmium.osm.Way) -> None:
        highway = way.tags.get("highway")
        if highway not in ROAD_TYPES:
            return

        way_nodes: list[RoadNode] = []
        for raw_node in way.nodes:
            try:
                lon = float(raw_node.lon)
                lat = float(raw_node.lat)
            except (osmium.InvalidLocationError, ValueError):
                return

            if not (self.xlim[0] <= lon <= self.xlim[1] and self.ylim[0] <= lat <= self.ylim[1]):
                way_nodes.append(None)  # type: ignore[arg-type]
                continue

            node_id = int(raw_node.ref)
            node = self.nodes.get(node_id)
            if node is None:
                x, y = self.transformer.transform(lon, lat)
                node = RoadNode(node_id=node_id, lon=lon, lat=lat, x=float(x), y=float(y))
                self.nodes[node_id] = node
            way_nodes.append(node)

        forward_only = is_forward_oneway(way.tags)
        reverse_only = is_reverse_oneway(way.tags)

        for a, b in zip(way_nodes, way_nodes[1:]):
            if a is None or b is None:
                continue
            length_m = math.hypot(a.x - b.x, a.y - b.y)
            if length_m <= 0:
                continue

            if reverse_only:
                self.edges.append(RoadEdge(b.node_id, a.node_id, length_m, highway, int(way.id)))
            elif forward_only:
                self.edges.append(RoadEdge(a.node_id, b.node_id, length_m, highway, int(way.id)))
            else:
                self.edges.append(RoadEdge(a.node_id, b.node_id, length_m, highway, int(way.id)))
                self.edges.append(RoadEdge(b.node_id, a.node_id, length_m, highway, int(way.id)))


def extract_road_graph(
    osm_input: Path,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    projected_crs: str = "EPSG:3763",
) -> tuple[list[RoadNode], list[RoadEdge]]:
    collector = RoadGraphCollector(xlim, ylim, projected_crs)
    collector.apply_file(str(osm_input), locations=True)
    nodes = sorted(collector.nodes.values(), key=lambda node: node.node_id)
    return nodes, collector.edges


def write_nodes(nodes: list[RoadNode], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["node_id", "lon", "lat", "x_m", "y_m"])
        for node in nodes:
            writer.writerow([node.node_id, node.lon, node.lat, round(node.x, 3), round(node.y, 3)])


def write_edges(edges: list[RoadEdge], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "length_m", "highway", "osm_way_id"])
        for edge in edges:
            writer.writerow([edge.source, edge.target, round(edge.length_m, 3), edge.highway, edge.osm_way_id])


class GridNearestIndex:
    def __init__(
        self,
        nodes: list[RoadNode],
        cell_size_m: float,
        projected_crs: str = "EPSG:3763",
    ) -> None:
        self.nodes = nodes
        self.cell_size_m = cell_size_m
        self.grid: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.transformer = Transformer.from_crs(
            "EPSG:4326", projected_crs, always_xy=True
        )

        for idx, node in enumerate(nodes):
            self.grid[self.cell_key(node.x, node.y)].append(idx)

    def cell_key(self, x: float, y: float) -> tuple[int, int]:
        return math.floor(x / self.cell_size_m), math.floor(y / self.cell_size_m)

    def nearest(self, lon: float, lat: float, max_radius_cells: int = 80) -> tuple[int, float]:
        x, y = self.transformer.transform(lon, lat)
        base_x, base_y = self.cell_key(float(x), float(y))

        best_node_id = -1
        best_dist2 = float("inf")

        for radius in range(max_radius_cells + 1):
            candidate_indices: list[int] = []
            for gx in range(base_x - radius, base_x + radius + 1):
                for gy in range(base_y - radius, base_y + radius + 1):
                    if radius > 0 and abs(gx - base_x) < radius and abs(gy - base_y) < radius:
                        continue
                    candidate_indices.extend(self.grid.get((gx, gy), []))

            if not candidate_indices:
                continue

            for idx in candidate_indices:
                node = self.nodes[idx]
                dist2 = (node.x - x) ** 2 + (node.y - y) ** 2
                if dist2 < best_dist2:
                    best_dist2 = dist2
                    best_node_id = node.node_id

            if best_node_id != -1:
                search_limit = ((radius + 1) * self.cell_size_m) ** 2
                if best_dist2 <= search_limit:
                    break

        if best_node_id == -1:
            return -1, float("inf")
        return best_node_id, math.sqrt(best_dist2)


def snap_queries(
    rows: list[ODRow],
    nodes: list[RoadNode],
    cell_size_m: float,
    projected_crs: str = "EPSG:3763",
) -> list[dict[str, object]]:
    index = GridNearestIndex(nodes, cell_size_m, projected_crs)
    snapped: list[dict[str, object]] = []

    for query_id, row in enumerate(rows):
        origin_node, origin_distance = index.nearest(row.origin_lon, row.origin_lat)
        dest_node, dest_distance = index.nearest(row.dest_lon, row.dest_lat)
        snapped.append(
            {
                "query_id": query_id,
                "trip_id": row.trip_id,
                "timestamp": row.timestamp,
                "origin_lon": row.origin_lon,
                "origin_lat": row.origin_lat,
                "dest_lon": row.dest_lon,
                "dest_lat": row.dest_lat,
                "origin_node": origin_node,
                "dest_node": dest_node,
                "origin_snap_distance_m": origin_distance,
                "dest_snap_distance_m": dest_distance,
                "origin_snap_success": origin_node != -1,
                "dest_snap_success": dest_node != -1,
            }
        )

    return snapped


def write_snapped_queries(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "query_id",
        "trip_id",
        "timestamp",
        "origin_lon",
        "origin_lat",
        "dest_lon",
        "dest_lat",
        "origin_node",
        "dest_node",
        "origin_snap_distance_m",
        "dest_snap_distance_m",
        "origin_snap_success",
        "dest_snap_success",
        "snap_usable",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            out["origin_snap_distance_m"] = round(float(out["origin_snap_distance_m"]), 3)
            out["dest_snap_distance_m"] = round(float(out["dest_snap_distance_m"]), 3)
            writer.writerow(out)


def mark_usable(rows: list[dict[str, object]], threshold_m: float) -> None:
    for row in rows:
        origin_ok = bool(row["origin_snap_success"]) and float(row["origin_snap_distance_m"]) <= threshold_m
        dest_ok = bool(row["dest_snap_success"]) and float(row["dest_snap_distance_m"]) <= threshold_m
        row["snap_usable"] = origin_ok and dest_ok


def write_usable_queries(rows: list[dict[str, object]], path: Path) -> int:
    usable_rows = [row for row in rows if bool(row["snap_usable"])]
    write_snapped_queries(usable_rows, path)
    return len(usable_rows)


def percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def write_report(
    path: Path,
    rows: list[dict[str, object]],
    nodes: list[RoadNode],
    edges: list[RoadEdge],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    threshold_m: float,
    city_name: str = "波尔图",
) -> None:
    origin_distances = np.array([float(row["origin_snap_distance_m"]) for row in rows], dtype=float)
    dest_distances = np.array([float(row["dest_snap_distance_m"]) for row in rows], dtype=float)
    max_distances = np.maximum(origin_distances, dest_distances)
    origin_finite = np.isfinite(origin_distances)
    dest_finite = np.isfinite(dest_distances)
    usable = np.array([bool(row["snap_usable"]) for row in rows], dtype=bool)
    origin_valid = origin_distances[origin_finite]
    dest_valid = dest_distances[dest_finite]

    def stats(values: np.ndarray) -> dict[str, float]:
        return {
            "mean": float(values.mean()),
            "p50": percentile(values, 50),
            "p90": percentile(values, 90),
            "p95": percentile(values, 95),
            "p99": percentile(values, 99),
            "max": float(values.max()),
        }

    origin_stats = stats(origin_valid)
    dest_stats = stats(dest_valid)

    path.parent.mkdir(parents=True, exist_ok=True)
    text = f"""# {city_name}起终点吸附质量报告

## 输入范围

- 经度范围：`{xlim[0]:.6f}` 到 `{xlim[1]:.6f}`
- 纬度范围：`{ylim[0]:.6f}` 到 `{ylim[1]:.6f}`

## 路网规模

- 路网节点数：{len(nodes):,}
- 有向边数：{len(edges):,}

## 查询吸附结果

- 查询总数：{len(rows):,}
- 可用阈值：起点和终点吸附距离都不超过 `{threshold_m:.0f}` 米
- 阈值内查询数：{int(usable.sum()):,}
- 阈值内占比：{usable.mean() * 100:.2f}%
- 起点吸附失败数：{int((~origin_finite).sum()):,}
- 终点吸附失败数：{int((~dest_finite).sum()):,}

## 起点吸附距离

- 统计口径：只统计吸附成功的起点
- 平均值：{origin_stats["mean"]:.2f} 米
- p50：{origin_stats["p50"]:.2f} 米
- p90：{origin_stats["p90"]:.2f} 米
- p95：{origin_stats["p95"]:.2f} 米
- p99：{origin_stats["p99"]:.2f} 米
- 最大值：{origin_stats["max"]:.2f} 米

## 终点吸附距离

- 统计口径：只统计吸附成功的终点
- 平均值：{dest_stats["mean"]:.2f} 米
- p50：{dest_stats["p50"]:.2f} 米
- p90：{dest_stats["p90"]:.2f} 米
- p95：{dest_stats["p95"]:.2f} 米
- p99：{dest_stats["p99"]:.2f} 米
- 最大值：{dest_stats["max"]:.2f} 米

## 说明

本报告使用最近道路节点吸附，不是完整地图匹配。它适合把 OD 点转成最短路查询节点；
如果后续需要还原真实轨迹路径，再单独做轨迹地图匹配。
"""
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    od_rows = read_od_rows(args.od_input)
    xlim, ylim = od_bounds(od_rows)

    nodes, edges = extract_road_graph(
        args.osm_input, xlim, ylim, args.projected_crs
    )
    if not nodes or not edges:
        raise SystemExit("没有抽取到路网节点或边，请检查 OSM 文件和 OD 范围。")

    write_nodes(nodes, args.node_out)
    write_edges(edges, args.edge_out)

    snapped_rows = snap_queries(
        od_rows, nodes, args.grid_cell_m, args.projected_crs
    )
    mark_usable(snapped_rows, args.snap_threshold_m)
    write_snapped_queries(snapped_rows, args.query_out)
    usable_count = write_usable_queries(snapped_rows, args.usable_query_out)
    write_report(
        args.report_out,
        snapped_rows,
        nodes,
        edges,
        xlim,
        ylim,
        args.snap_threshold_m,
        args.city_name,
    )

    print(f"nodes={len(nodes)}")
    print(f"edges={len(edges)}")
    print(f"queries={len(snapped_rows)}")
    print(f"usable_queries={usable_count}")
    print(f"node_out={args.node_out}")
    print(f"edge_out={args.edge_out}")
    print(f"query_out={args.query_out}")
    print(f"usable_query_out={args.usable_query_out}")
    print(f"report_out={args.report_out}")


if __name__ == "__main__":
    main()
