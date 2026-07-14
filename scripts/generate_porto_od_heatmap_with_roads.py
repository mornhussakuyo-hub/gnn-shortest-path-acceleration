"""在本地 OSM PBF 路网底图上绘制波尔图 OD 热力图。"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import osmium
from matplotlib.collections import LineCollection
from matplotlib import font_manager


DEFAULT_OD_INPUT = Path("data/processed/porto/波尔图起终点样本_10万.csv")
DEFAULT_OSM_INPUT = Path("data/compressed/porto/portugal-latest.osm.pbf")
DEFAULT_PLOT_OUT = Path("data/plots/波尔图道路底图起终点热力图_10万.png")

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


def setup_chinese_font() -> bool:
    windows_fonts = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    font_paths = [
        windows_fonts / "msyh.ttc",
        windows_fonts / "simhei.ttf",
        Path("/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/todesk/NotoSansCJK-Regular.ttc"),
    ]
    for font_path in font_paths:
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font_path)).get_name()
            plt.rcParams["axes.unicode_minus"] = False
            return True
    plt.rcParams["axes.unicode_minus"] = False
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--od-input", type=Path, default=DEFAULT_OD_INPUT)
    parser.add_argument("--osm-input", type=Path, default=DEFAULT_OSM_INPUT)
    parser.add_argument("--plot-out", type=Path, default=DEFAULT_PLOT_OUT)
    return parser.parse_args()


def robust_bounds(values: np.ndarray, low: float = 0.5, high: float = 99.5) -> tuple[float, float]:
    lo, hi = np.percentile(values, [low, high])
    pad = (hi - lo) * 0.08
    return float(lo - pad), float(hi + pad)


def read_od_csv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[tuple[float, float, float, float]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                (
                    float(row["origin_lon"]),
                    float(row["origin_lat"]),
                    float(row["dest_lon"]),
                    float(row["dest_lat"]),
                )
            )

    data = np.array(rows, dtype=float)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


class RoadCollector(osmium.SimpleHandler):
    def __init__(self, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
        super().__init__()
        self.xlim = xlim
        self.ylim = ylim
        self.lines: list[list[tuple[float, float]]] = []

    def way(self, way: osmium.osm.Way) -> None:
        highway = way.tags.get("highway")
        if highway not in ROAD_TYPES:
            return

        points: list[tuple[float, float]] = []
        for node in way.nodes:
            try:
                lon = float(node.lon)
                lat = float(node.lat)
            except (osmium.InvalidLocationError, ValueError):
                return

            if self.xlim[0] <= lon <= self.xlim[1] and self.ylim[0] <= lat <= self.ylim[1]:
                points.append((lon, lat))
            else:
                if len(points) >= 2:
                    self.lines.append(points)
                points = []

        if len(points) >= 2:
            self.lines.append(points)


def extract_roads(osm_input: Path, xlim: tuple[float, float], ylim: tuple[float, float]) -> list[list[tuple[float, float]]]:
    collector = RoadCollector(xlim, ylim)
    collector.apply_file(str(osm_input), locations=True)
    return collector.lines


def draw_plot(
    roads: list[list[tuple[float, float]]],
    o_lon: np.ndarray,
    o_lat: np.ndarray,
    d_lon: np.ndarray,
    d_lat: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    plot_out: Path,
    use_chinese_labels: bool,
) -> None:
    in_bounds = (
        (o_lon >= xlim[0])
        & (o_lon <= xlim[1])
        & (d_lon >= xlim[0])
        & (d_lon <= xlim[1])
        & (o_lat >= ylim[0])
        & (o_lat <= ylim[1])
        & (d_lat >= ylim[0])
        & (d_lat <= ylim[1])
    )
    o_lon, o_lat = o_lon[in_bounds], o_lat[in_bounds]
    d_lon, d_lat = d_lon[in_bounds], d_lat[in_bounds]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5), constrained_layout=True)
    if use_chinese_labels:
        panels = [
            (axes[0], o_lon, o_lat, "波尔图出租车起点热力图"),
            (axes[1], d_lon, d_lat, "波尔图出租车终点热力图"),
        ]
        x_label = "经度"
        y_label = "纬度"
        colorbar_label = "log10(行程数)"
        figure_title = f"基于 {len(o_lon):,} 条有效行程，底图来自本地 OSM PBF"
    else:
        panels = [
            (axes[0], o_lon, o_lat, "Porto taxi origins"),
            (axes[1], d_lon, d_lat, "Porto taxi destinations"),
        ]
        x_label = "longitude"
        y_label = "latitude"
        colorbar_label = "log10(trip count)"
        figure_title = f"OD heatmap from {len(o_lon):,} valid trips on local OSM roads"

    for ax, lon, lat, title in panels:
        if roads:
            road_collection = LineCollection(roads, colors="#8a8a8a", linewidths=0.25, alpha=0.38)
            ax.add_collection(road_collection)

        heatmap = ax.hexbin(lon, lat, gridsize=120, bins="log", mincnt=1, cmap="inferno", alpha=0.86)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#d8d8d8", linewidth=0.35, alpha=0.45)
        fig.colorbar(heatmap, ax=ax, label=colorbar_label)

    fig.suptitle(figure_title)
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_out, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    use_chinese_labels = setup_chinese_font()
    o_lon, o_lat, d_lon, d_lat = read_od_csv(args.od_input)

    all_lon = np.concatenate([o_lon, d_lon])
    all_lat = np.concatenate([o_lat, d_lat])
    xlim = robust_bounds(all_lon)
    ylim = robust_bounds(all_lat)

    roads = extract_roads(args.osm_input, xlim, ylim)
    draw_plot(
        roads,
        o_lon,
        o_lat,
        d_lon,
        d_lat,
        xlim,
        ylim,
        args.plot_out,
        use_chinese_labels,
    )

    print(f"road_segments={len(roads)}")
    print(f"heatmap={args.plot_out}")


if __name__ == "__main__":
    main()
