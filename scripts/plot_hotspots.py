from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def load_dimacs_coords(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes: list[int] = []
    lon_values: list[float] = []
    lat_values: list[float] = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.startswith("v "):
                continue
            _, node, lon, lat = line.split()
            nodes.append(int(node))
            lon_values.append(int(lon) / 1_000_000)
            lat_values.append(int(lat) / 1_000_000)

    return (
        np.array(nodes, dtype=np.int64),
        np.array(lon_values, dtype=np.float64),
        np.array(lat_values, dtype=np.float64),
    )


def load_hotspot_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def nearest_indexes(
    lon: np.ndarray,
    lat: np.ndarray,
    center_lon: float,
    center_lat: float,
    count: int,
) -> np.ndarray:
    distance_square = (lon - center_lon) ** 2 + (lat - center_lat) ** 2
    nearest_count = min(count, len(lon))
    return np.argpartition(distance_square, nearest_count - 1)[:nearest_count]


def plot_hotspots(
    coords_path: Path,
    hotspots_path: Path,
    output_path: Path,
    dpi: int,
) -> None:
    _nodes, lon, lat = load_dimacs_coords(coords_path)
    hotspot_rows = load_hotspot_rows(hotspots_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 10), dpi=dpi)
    ax.scatter(lon, lat, s=0.06, c="#c9c9c9", alpha=0.35, linewidths=0)

    colors = plt.cm.tab20(np.linspace(0, 1, max(len(hotspot_rows), 1)))
    for index, row in enumerate(hotspot_rows):
        center_lon = float(row["center_lon"])
        center_lat = float(row["center_lat"])
        num_nodes = int(row["num_nodes"])
        hotspot_id = int(row["hotspot_id"])
        region_indexes = nearest_indexes(lon, lat, center_lon, center_lat, num_nodes)

        color = colors[index % len(colors)]
        ax.scatter(
            lon[region_indexes],
            lat[region_indexes],
            s=2.4,
            color=color,
            alpha=0.72,
            linewidths=0,
        )
        ax.scatter(
            [center_lon],
            [center_lat],
            s=34,
            color=color,
            edgecolors="black",
            linewidths=0.45,
            zorder=5,
        )
        ax.text(
            center_lon,
            center_lat,
            str(hotspot_id),
            fontsize=5.5,
            ha="center",
            va="center",
            zorder=6,
        )

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Small graph hotspot regions ({len(hotspot_rows)} hotspots)")
    ax.grid(True, linewidth=0.25, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Loaded {len(lon):,} points from {coords_path}")
    print(f"Loaded {len(hotspot_rows):,} hotspots from {hotspots_path}")
    print(f"Saved plot to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--coords",
        type=Path,
        default=Path("data/raw/dimacs/small/USA-road-d.NY.co"),
    )
    parser.add_argument(
        "--hotspots",
        type=Path,
        default=Path("data/processed/query_loads/small/hotspots.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/plots/small_hotspots.png"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_hotspots(args.coords, args.hotspots, args.output, args.dpi)


if __name__ == "__main__":
    main()
