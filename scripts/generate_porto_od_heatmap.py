"""抽取波尔图出租车 OD 样本，并绘制经纬度热力图。"""

from __future__ import annotations

import argparse
import csv
import json
import os
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager


DEFAULT_INPUT = Path("data/compressed/porto/train.csv.zip")
DEFAULT_OD_OUT = Path("data/processed/porto/波尔图起终点样本_10万.csv")
DEFAULT_PLOT_OUT = Path("data/plots/波尔图起终点热力图_10万.png")


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
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--limit", type=int, default=100_000)
    parser.add_argument("--od-out", type=Path, default=DEFAULT_OD_OUT)
    parser.add_argument("--plot-out", type=Path, default=DEFAULT_PLOT_OUT)
    return parser.parse_args()


def read_od_points(zip_path: Path, limit: int) -> list[tuple[str, int, float, float, float, float]]:
    rows: list[tuple[str, int, float, float, float, float]] = []

    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("train.csv") as raw_file:
            text_file = (line.decode("utf-8") for line in raw_file)
            reader = csv.DictReader(text_file)

            for row in reader:
                if row.get("MISSING_DATA") == "True":
                    continue

                try:
                    polyline = json.loads(row["POLYLINE"])
                except (json.JSONDecodeError, KeyError):
                    continue

                if len(polyline) < 2:
                    continue

                try:
                    o_lon, o_lat = map(float, polyline[0])
                    d_lon, d_lat = map(float, polyline[-1])
                    timestamp = int(row["TIMESTAMP"])
                except (TypeError, ValueError):
                    continue

                rows.append((row["TRIP_ID"], timestamp, o_lon, o_lat, d_lon, d_lat))
                if len(rows) >= limit:
                    break

    return rows


def write_od_csv(rows: list[tuple[str, int, float, float, float, float]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["trip_id", "timestamp", "origin_lon", "origin_lat", "dest_lon", "dest_lat"])
        writer.writerows(rows)


def robust_bounds(values: np.ndarray, low: float = 0.5, high: float = 99.5) -> tuple[float, float]:
    lo, hi = np.percentile(values, [low, high])
    pad = (hi - lo) * 0.06
    return float(lo - pad), float(hi + pad)


def draw_heatmap(rows: list[tuple[str, int, float, float, float, float]], path: Path) -> None:
    data = np.array([(r[2], r[3], r[4], r[5]) for r in rows], dtype=float)
    o_lon, o_lat, d_lon, d_lat = data.T

    all_lon = np.concatenate([o_lon, d_lon])
    all_lat = np.concatenate([o_lat, d_lat])
    xlim = robust_bounds(all_lon)
    ylim = robust_bounds(all_lat)

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

    o_lon, o_lat, d_lon, d_lat = o_lon[in_bounds], o_lat[in_bounds], d_lon[in_bounds], d_lat[in_bounds]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    panels = [
        (axes[0], o_lon, o_lat, "Porto taxi origins"),
        (axes[1], d_lon, d_lat, "Porto taxi destinations"),
    ]

    for ax, lon, lat, title in panels:
        hb = ax.hexbin(lon, lat, gridsize=110, bins="log", mincnt=1, cmap="inferno")
        ax.set_title(title)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, color="#d0d0d0", linewidth=0.4, alpha=0.45)
        fig.colorbar(hb, ax=ax, label="log10(trip count)")

    fig.suptitle(f"OD heatmap from {len(rows):,} valid Porto taxi trips")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    setup_chinese_font()

    rows = read_od_points(args.input, args.limit)
    if not rows:
        raise SystemExit("No valid OD rows found.")

    write_od_csv(rows, args.od_out)
    draw_heatmap(rows, args.plot_out)

    print(f"valid_trips={len(rows)}")
    print(f"od_csv={args.od_out}")
    print(f"heatmap={args.plot_out}")


if __name__ == "__main__":
    main()
