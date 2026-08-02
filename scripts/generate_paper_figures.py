#!/usr/bin/env python3
"""Generate reproducible paper figures from frozen AIC experiment artifacts.

The script deliberately reads machine-generated summaries instead of copying values
from the manuscript.  Every plotted scalar is also exported to main_results.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean, pstdev

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from paper_figure_style import (
    BLUE,
    GREEN,
    GRID,
    INK,
    MUTED,
    ORANGE,
    PALE_BLUE,
    PALE_GREEN,
    PALE_ORANGE,
    PURPLE,
    RED,
    configure_style,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "paper" / "figures"

def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        output / f"{stem}.png",
        dpi=260,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)


def add_row(
    rows: list[dict[str, object]],
    *,
    figure: str,
    city: str,
    window: str,
    method: str,
    metric: str,
    value: float,
    std: float = 0.0,
    unit: str = "",
    seed_count: int = 1,
    status: str = "frozen",
    source: Path,
) -> None:
    rows.append(
        {
            "figure": figure,
            "city": city,
            "window": window,
            "method": method,
            "metric": metric,
            "value": f"{value:.12g}",
            "std": f"{std:.12g}",
            "unit": unit,
            "seed_count": seed_count,
            "status": status,
            "source": str(source.relative_to(ROOT)),
        }
    )


def draw_box(
    ax: plt.Axes,
    xy: tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    face: str,
    edge: str,
    fontsize: float = 9,
    weight: str = "normal",
) -> None:
    box = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=face,
        edgecolor=edge,
        linewidth=1.2,
    )
    ax.add_patch(box)
    ax.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        color=INK,
        fontsize=fontsize,
        weight=weight,
        linespacing=1.25,
    )


def arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            linewidth=1.15,
            color=MUTED,
            connectionstyle="arc3,rad=0",
        )
    )


def make_pipeline(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.25))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.02, 0.955, "Leakage-free temporal protocol", color=INK, weight="bold")
    starts = [0.02, 0.37, 0.72]
    widths = [0.33, 0.33, 0.26]
    labels = [
        ("H  History OD", "0–35%  ·  model input", PALE_BLUE, BLUE),
        ("Y  Current labels", "35–70%  ·  exact regional gain", PALE_ORANGE, ORANGE),
        ("F  Future test", "70–100%  ·  frozen evaluation", PALE_GREEN, GREEN),
    ]
    for x, width, (title, subtitle, face, edge) in zip(starts, widths, labels):
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.82), width, 0.10,
                boxstyle="round,pad=0.008,rounding_size=0.012",
                facecolor=face, edgecolor=edge, linewidth=1.05,
            )
        )
        ax.text(x + 0.015, 0.878, title, ha="left", va="center", weight="bold", color=INK)
        ax.text(x + 0.015, 0.842, subtitle, ha="left", va="center", fontsize=6.8, color=MUTED)

    draw_box(ax, (0.02, 0.46), 0.145, 0.20, "Road graph\n+ historical OD", face="#F4F6F7", edge="#8A989E", fontsize=6.9, weight="bold")
    draw_box(ax, (0.195, 0.46), 0.185, 0.20, "Z0\nparameter-free\nbidirectional\ndiffusion", face=PALE_BLUE, edge=BLUE, fontsize=6.1, weight="bold")
    draw_box(ax, (0.41, 0.46), 0.18, 0.20, "BRIDGE\nfrozen Z0\n+ neural residual", face=PALE_ORANGE, edge=ORANGE, fontsize=6.9, weight="bold")
    draw_box(ax, (0.62, 0.46), 0.17, 0.20, "BRIDGE-B\nbudget-aware\ndeployment head", face=PALE_GREEN, edge=GREEN, fontsize=6.4, weight="bold")
    draw_box(ax, (0.82, 0.46), 0.16, 0.20, "Disjoint\nregion selection", face="#F2EFFA", edge=PURPLE, fontsize=6.5, weight="bold")

    for start, end in [
        ((0.165, 0.56), (0.195, 0.56)),
        ((0.38, 0.56), (0.41, 0.56)),
        ((0.59, 0.56), (0.62, 0.56)),
        ((0.79, 0.56), (0.82, 0.56)),
    ]:
        arrow(ax, start, end)

    draw_box(ax, (0.215, 0.10), 0.205, 0.18, "Spatially isolated\ncandidate regions\n(Jaccard groups)", face="#F4F6F7", edge="#8A989E", fontsize=7.0)
    draw_box(ax, (0.49, 0.10), 0.205, 0.18, "Selective CRP-style\nboundary overlay\n(existing exact substrate)", face="#F4F6F7", edge="#8A989E", fontsize=7.0)
    draw_box(ax, (0.765, 0.10), 0.215, 0.18, "Exact bidirectional query\n+ local endpoint access\n(distance preserved)", face="#F4F6F7", edge="#8A989E", fontsize=7.0)
    arrow(ax, (0.317, 0.28), (0.287, 0.46))
    arrow(ax, (0.90, 0.46), (0.695, 0.20))
    arrow(ax, (0.695, 0.19), (0.765, 0.19))

    ax.text(
        0.02,
        0.02,
        "Learning allocates offline materialization resources; online shortest-path answers remain exact.",
        fontsize=7.2,
        color=MUTED,
        style="italic",
    )
    save_figure(fig, output, "method_pipeline")


def make_ranking(output: Path, rows: list[dict[str, object]]) -> None:
    sources = {
        "Porto": ROOT / "results/gnn_v2/nbfnet_propagation/g4_frozen_evaluation/summary.json",
        "Chicago": ROOT / "results/chicago/gnn_v2/nbfnet_propagation/g4_frozen_evaluation/summary.json",
    }
    metrics = [
        ("Spearman", "spearman"),
        ("NDCG@5", "5"),
        ("NDCG@18", "18"),
    ]
    categories = [(city, window) for city in sources for window in ("holdout", "future_all")]
    category_labels = ["P\nH/O", "P\nF", "C\nH/O", "C\nF"]
    data: dict[tuple[str, str, str, str], tuple[float, float]] = {}

    for city, source in sources.items():
        summary = load_json(source)
        aggregate = summary["aggregate"]["global_spearman"]
        for window in ("holdout", "future_all"):
            z0 = summary["z0"][window]
            g4 = aggregate[window]
            for label, key in metrics:
                if key == "spearman":
                    z_value = float(z0["spearman"])
                    g_value = float(g4["spearman"]["mean"])
                    g_std = float(g4["spearman"]["std"])
                else:
                    z_value = float(z0["ranking_at_k"][key]["ndcg"])
                    g_value = float(g4["ranking_at_k"][key]["ndcg"]["mean"])
                    g_std = float(g4["ranking_at_k"][key]["ndcg"]["std"])
                data[(city, window, label, "Z0")] = (z_value, 0.0)
                data[(city, window, label, "BRIDGE")] = (g_value, g_std)
                add_row(rows, figure="ranking_results", city=city, window=window,
                        method="Z0", metric=label, value=z_value, source=source)
                add_row(rows, figure="ranking_results", city=city, window=window,
                        method="BRIDGE", metric=label, value=g_value, std=g_std,
                        seed_count=3, source=source)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), sharex=True)
    x = np.arange(len(categories))
    width = 0.34
    for ax, (metric_label, _) in zip(axes, metrics):
        z_values = [data[(c, w, metric_label, "Z0")][0] for c, w in categories]
        b_values = [data[(c, w, metric_label, "BRIDGE")][0] for c, w in categories]
        b_stds = [data[(c, w, metric_label, "BRIDGE")][1] for c, w in categories]
        ax.bar(x - width / 2, z_values, width, color=BLUE, label="Z0", zorder=3)
        ax.bar(x + width / 2, b_values, width, color=ORANGE, label="BRIDGE", zorder=3)
        ax.errorbar(x + width / 2, b_values, yerr=b_stds, fmt="none", ecolor=INK,
                    elinewidth=0.8, capsize=2.2, capthick=0.8, zorder=4)
        ax.set_title(metric_label, weight="bold")
        ax.set_xticks(x, category_labels)
        lower = min(z_values + [v - s for v, s in zip(b_values, b_stds)])
        ax.set_ylim(max(0.55, lower - 0.045), 1.005)
        ax.grid(axis="y", color=GRID, linewidth=0.65, zorder=0)
        ax.axvline(1.5, color=GRID, linewidth=0.8)
    axes[0].set_ylabel("Ranking quality (higher is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.04))
    fig.subplots_adjust(wspace=0.22, top=0.79, bottom=0.19)
    save_figure(fig, output, "ranking_results")


def make_system(output: Path, rows: list[dict[str, object]]) -> None:
    metrics = [
        ("Expanded nodes", "expanded_change_pct"),
        ("Scanned edges", "scanned_edges_change_pct"),
        ("Mean latency", "elapsed_change_pct"),
        ("P95 latency", "p95_change_pct"),
    ]
    sources = {
        "Porto": ROOT / "results/cpp_online_benchmark/porto/summary.csv",
        "Chicago": ROOT / "results/cpp_online_benchmark/chicago/summary.csv",
    }
    windows = ["current_y", "future_f"]
    categories = [(city, window) for city in sources for window in windows]
    labels = ["Porto\nY", "Porto\nF", "Chicago\nY", "Chicago\nF"]
    values: dict[tuple[str, str, str, str], tuple[float, float]] = {}

    for city, source in sources.items():
        with source.open(newline="", encoding="utf-8") as handle:
            records = list(csv.DictReader(handle))
        for window in windows:
            z_record = next(r for r in records if r["method"] == "z0" and r["window"] == window)
            bridge_records = [
                r for r in records
                if r["method"].startswith("g4_global_seed") and r["window"] == window
            ]
            if len(bridge_records) != 3:
                raise RuntimeError(f"Expected three BRIDGE seeds in {source} ({window})")
            for label, metric in metrics:
                z_value = float(z_record[metric])
                b_raw = [float(r[metric]) for r in bridge_records]
                b_value, b_std = fmean(b_raw), pstdev(b_raw)
                values[(city, window, label, "Z0")] = (z_value, 0.0)
                values[(city, window, label, "BRIDGE")] = (b_value, b_std)
                add_row(rows, figure="system_results", city=city, window=window,
                        method="Z0", metric=label, value=z_value, unit="%",
                        source=source)
                add_row(rows, figure="system_results", city=city, window=window,
                        method="BRIDGE", metric=label, value=b_value, std=b_std,
                        unit="%", seed_count=3, source=source)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.0), sharex=True)
    x = np.arange(len(categories))
    width = 0.34
    for ax, (label, _) in zip(axes.flat, metrics):
        z_values = [values[(c, w, label, "Z0")][0] for c, w in categories]
        b_values = [values[(c, w, label, "BRIDGE")][0] for c, w in categories]
        b_stds = [values[(c, w, label, "BRIDGE")][1] for c, w in categories]
        ax.bar(x - width / 2, z_values, width, color=BLUE, label="Z0", zorder=3)
        ax.bar(x + width / 2, b_values, width, color=ORANGE, label="BRIDGE", zorder=3)
        ax.errorbar(x + width / 2, b_values, yerr=b_stds, fmt="none", ecolor=INK,
                    elinewidth=0.8, capsize=2.2, capthick=0.8, zorder=4)
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.axvline(1.5, color=GRID, linewidth=0.8)
        ax.grid(axis="y", color=GRID, linewidth=0.65, zorder=0)
        ax.set_title(label, loc="left", weight="bold")
        ax.set_ylabel("Change vs. original graph (%)")
        ax.set_xticks(x, labels)
        ax.text(0.99, 0.04, "lower is better", transform=ax.transAxes,
                ha="right", va="bottom", color=MUTED, fontsize=7.5)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, 1.015))
    fig.subplots_adjust(wspace=0.25, hspace=0.34, top=0.91, bottom=0.12)
    save_figure(fig, output, "system_results")


def select_online_run(summary: dict, method: str | None = None, k: int = 18) -> dict:
    for run in summary["runs"].values():
        current = run["current_y"]
        if int(current["k"]) != k:
            continue
        if method is None or current["method"] == method:
            return run
    raise KeyError(f"No online run for method={method!r}, k={k}")


def make_mechanism_ablation(output: Path, rows: list[dict[str, object]]) -> None:
    sources = {
        "Porto": ROOT / "results/gnn_v2/z0_orthogonal_ablation/summary.json",
        "Chicago": ROOT / "results/chicago/gnn_v2/z0_orthogonal_ablation/summary.json",
    }
    windows = [("holdout", "current_metrics", "holdout"),
               ("future", "future_metrics", "all_candidates")]
    summaries = {city: load_json(source) for city, source in sources.items()}

    def score(city: str, variant: str, window: str) -> float:
        _, outer, split = next(item for item in windows if item[0] == window)
        return float(summaries[city]["variants"][variant][outer][split]["spearman"])

    plotted = [
        "z0_base", "undirected", "degree_rewired", "origin_only", "destination_only",
        "shuffled_od", "pooling_mean", "pooling_max",
        "depth_01", "depth_02", "depth_04", "depth_08", "depth_16", "depth_32",
    ]
    for city, source in sources.items():
        for window, _, _ in windows:
            for variant in plotted:
                add_row(rows, figure="mechanism_ablation", city=city, window=window,
                        method=variant, metric="Spearman", value=score(city, variant, window),
                        source=source)

    fig = plt.figure(figsize=(7.2, 5.3))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.04], hspace=0.38, wspace=0.28)
    ax_topology = fig.add_subplot(grid[0, 0])
    ax_signal = fig.add_subplot(grid[0, 1])
    ax_depth = fig.add_subplot(grid[1, :])

    categories = [(city, window) for city in sources for window, _, _ in windows]
    category_labels = ["Porto\nH/O", "Porto\nFuture", "Chicago\nH/O", "Chicago\nFuture"]
    x = np.arange(len(categories))

    topology = [
        ("Directed Z0", "z0_base", BLUE),
        ("Undirected", "undirected", ORANGE),
        ("Degree-rewired", "degree_rewired", RED),
    ]
    width = 0.25
    for offset, (label, variant, color) in enumerate(topology):
        vals = [score(city, variant, window) for city, window in categories]
        ax_topology.bar(x + (offset - 1) * width, vals, width, label=label, color=color, zorder=3)
    ax_topology.set_title("(a) Road topology", loc="left", weight="bold", y=1.22)
    ax_topology.set_xticks(x, category_labels)
    ax_topology.set_ylim(0.40, 1.0)
    ax_topology.set_ylabel("Spearman")
    ax_topology.grid(axis="y", color=GRID, linewidth=0.65, zorder=0)
    ax_topology.legend(frameon=False, fontsize=6.4, ncol=3, loc="lower center",
                       bbox_to_anchor=(0.5, 1.01), columnspacing=0.8, handlelength=1.8)

    signal_variants = [
        ("Origin only", "origin_only", BLUE),
        ("Destination only", "destination_only", ORANGE),
        ("Shuffled OD", "shuffled_od", GREEN),
        ("Mean pool", "pooling_mean", RED),
        ("Max pool", "pooling_max", PURPLE),
    ]
    sx = np.arange(len(signal_variants))
    markers = {"Porto": "o", "Chicago": "s"}
    linestyles = {"holdout": "-", "future": "--"}
    for city in sources:
        for window, _, _ in windows:
            delta = [score(city, variant, window) - score(city, "z0_base", window)
                     for _, variant, _ in signal_variants]
            ax_signal.plot(sx, delta, marker=markers[city], linestyle=linestyles[window],
                           linewidth=1.2, markersize=4, label=f"{city} {window}")
    ax_signal.axhline(0, color=INK, linewidth=0.8)
    ax_signal.set_title("(b) Signal and pooling", loc="left", weight="bold", y=1.22)
    ax_signal.set_xticks(sx, [label.replace(" ", "\n") for label, _, _ in signal_variants])
    ax_signal.tick_params(axis="x", labelsize=6.6)
    ax_signal.set_ylabel("Spearman delta vs. Z0")
    ax_signal.grid(axis="y", color=GRID, linewidth=0.65, zorder=0)
    ax_signal.legend(frameon=False, fontsize=6.4, ncol=2, loc="lower center",
                     bbox_to_anchor=(0.5, 1.01), columnspacing=0.8)

    depths = [1, 2, 4, 8, 16, 32]
    for city, color in [("Porto", BLUE), ("Chicago", ORANGE)]:
        for window, _, _ in windows:
            vals = [score(city, f"depth_{depth:02d}", window) for depth in depths]
            ax_depth.plot(depths, vals, color=color, marker=markers[city],
                          linestyle=linestyles[window], linewidth=1.5, markersize=4,
                          label=f"{city} {window}")
    ax_depth.set_xscale("log", base=2)
    ax_depth.set_xticks(depths, [str(depth) for depth in depths])
    ax_depth.set_title("(c) Single-depth readout", loc="left", weight="bold")
    ax_depth.set_xlabel("Diffusion depth")
    ax_depth.set_ylabel("Spearman")
    ax_depth.grid(color=GRID, linewidth=0.65, zorder=0)
    ax_depth.legend(frameon=False, ncol=4, loc="upper center", fontsize=6.8,
                    bbox_to_anchor=(0.5, -0.24), columnspacing=1.0)
    fig.subplots_adjust(top=0.82, bottom=0.17)
    save_figure(fig, output, "mechanism_ablation")


def make_bridge_b_progression(output: Path, rows: list[dict[str, object]]) -> None:
    city_sources = {
        "Porto": {
            "base": ROOT / "results/gnn_v2/multi_region_online/summary.json",
            "dir": ROOT / "results/gnn_v2/g5_cost_aware_exploration",
        },
        "Chicago": {
            "base": ROOT / "results/chicago/gnn_v2/multi_region_online_g4_clean_rerun/summary.json",
            "dir": ROOT / "results/chicago/gnn_v2/g5_cost_aware_exploration",
        },
    }
    stage_names = ["Z0", "S0", "S1", "S2", "S3"]
    figure_data: dict[str, list[dict[str, float | str]]] = {}

    for city, paths in city_sources.items():
        base_summary = load_json(paths["base"])
        base_run = select_online_run(base_summary, method="z0", k=18)
        base_expanded = {
            window: float(base_run[window]["indexed_avg_expanded"])
            for window in ("current_y", "future_f")
        }
        base_shortcuts = float(base_run["current_y"]["shortcut_count"])
        seed42_source = paths["dir"] / "s2_gain1_short_seed42_online_k18" / "summary.json"
        seed42_run = select_online_run(load_json(seed42_source), k=18)
        s3_source = paths["dir"] / "s3_gain1_seeds43_44_short_online_k18" / "summary.json"
        s3_runs = list(load_json(s3_source)["runs"].values())
        stage_runs = [
            ("Z0", [("deterministic", base_run, paths["base"])], "frozen"),
            ("S0", [("seed42", select_online_run(load_json(
                paths["dir"] / "s0_short_seed42_online_k18" / "summary.json"), k=18),
                paths["dir"] / "s0_short_seed42_online_k18" / "summary.json")], "exploratory_seed42"),
            ("S1", [("seed42", select_online_run(load_json(
                paths["dir"] / "s1_scaled_topk_short_seed42_online_k18" / "summary.json"), k=18),
                paths["dir"] / "s1_scaled_topk_short_seed42_online_k18" / "summary.json")], "exploratory_seed42"),
            ("S2", [("seed42", seed42_run, seed42_source)], "exploratory_seed42"),
            ("S3", [
                ("seed42", seed42_run, seed42_source),
                ("seed43", s3_runs[0], s3_source),
                ("seed44", s3_runs[1], s3_source),
            ], "confirmation_3seeds"),
        ]
        city_data = []
        for stage, run_specs, stage_status in stage_runs:
            deltas = {
                window: [float(run[window]["indexed_avg_expanded"]) - base_expanded[window]
                         for _, run, _ in run_specs]
                for window in ("current_y", "future_f")
            }
            shortcut_deltas = [
                float(run["current_y"]["shortcut_count"]) - base_shortcuts
                for _, run, _ in run_specs
            ]
            entry: dict[str, float | str] = {
                "stage": stage,
                "current_y": fmean(deltas["current_y"]),
                "current_y_std": pstdev(deltas["current_y"]),
                "future_f": fmean(deltas["future_f"]),
                "future_f_std": pstdev(deltas["future_f"]),
                "shortcuts": fmean(shortcut_deltas),
                "shortcuts_std": pstdev(shortcut_deltas),
            }
            city_data.append(entry)
            for seed_label, run, source in run_specs:
                method = stage if len(run_specs) == 1 else f"{stage}-{seed_label}"
                for window in ("current_y", "future_f"):
                    add_row(rows, figure="bridge_b_progression", city=city, window=window,
                            method=method, metric="expanded_nodes_delta_vs_z0",
                            value=float(run[window]["indexed_avg_expanded"]) - base_expanded[window],
                            unit="nodes/query", status=stage_status, source=source)
                add_row(rows, figure="bridge_b_progression", city=city, window="Y/F",
                        method=method, metric="shortcut_delta_vs_z0",
                        value=float(run["current_y"]["shortcut_count"]) - base_shortcuts,
                        unit="shortcuts", status=stage_status, source=source)
        figure_data[city] = city_data

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    x = np.arange(len(stage_names))
    width = 0.30
    twin_axes = []
    for ax, city in zip(axes, city_sources):
        data = figure_data[city]
        y_delta = [float(d["current_y"]) for d in data]
        y_std = [float(d["current_y_std"]) for d in data]
        f_delta = [float(d["future_f"]) for d in data]
        f_std = [float(d["future_f_std"]) for d in data]
        shortcuts = [float(d["shortcuts"]) for d in data]
        shortcut_std = [float(d["shortcuts_std"]) for d in data]
        ax.bar(x - width / 2, y_delta, width, yerr=y_std, color=ORANGE,
               error_kw={"elinewidth": 0.8, "capsize": 2}, label="Current Y", zorder=3)
        ax.bar(x + width / 2, f_delta, width, yerr=f_std, color=GREEN,
               error_kw={"elinewidth": 0.8, "capsize": 2}, label="Future F", zorder=3)
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.grid(axis="y", color=GRID, linewidth=0.65, zorder=0)
        ax.set_xticks(x, stage_names)
        if city == "Porto":
            ax.set_ylabel("Expanded-node delta vs. Z0\n(nodes/query; lower is better)")
        ax.set_title(city, weight="bold")
        twin = ax.twinx()
        twin_axes.append(twin)
        twin.spines["right"].set_visible(True)
        twin.errorbar(x, shortcuts, yerr=shortcut_std, color=PURPLE, marker="o",
                      markersize=4.5, linewidth=1.4, capsize=2,
                      label="Shortcut delta")
        twin.axhline(0, color=PURPLE, linewidth=0.55, alpha=0.4, linestyle="--")
        if city == "Chicago":
            twin.set_ylabel("Shortcut delta vs. Z0", color=PURPLE)
        twin.tick_params(axis="y", colors=PURPLE)
        for i, value in enumerate(y_delta):
            if i > 0:
                ax.annotate(f"{value:+.1f}", (i - width / 2, value),
                            xytext=(0, 4), textcoords="offset points",
                            ha="center", fontsize=7.2, color=INK)
    bar_handles, bar_labels = axes[0].get_legend_handles_labels()
    line_handles, line_labels = twin_axes[0].get_legend_handles_labels()
    fig.legend(bar_handles + line_handles, bar_labels + line_labels,
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.text(0.5, 0.01,
             "S0–S2: seed 42 development; S3: mean ± SD over frozen seeds 42–44.",
             ha="center", color=MUTED, fontsize=8, style="italic")
    fig.subplots_adjust(wspace=0.48, top=0.83, bottom=0.19)
    save_figure(fig, output, "bridge_b_progression")


def write_rows(output: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "figure", "city", "window", "method", "metric", "value", "std",
        "unit", "seed_count", "status", "source",
    ]
    with (output / "main_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "schema": "aic.paper_figures.v1",
        "generator": str(Path(__file__).resolve().relative_to(ROOT)),
        "figure_files": [
            "method_pipeline.pdf",
            "ranking_results.pdf",
            "mechanism_ablation.pdf",
            "system_results.pdf",
            "bridge_b_progression.pdf",
            "spatial_benefit_porto.pdf",
            "spatial_benefit_chicago.pdf",
            "spatial_benefit_porto_endpoints.pdf",
            "spatial_benefit_chicago_endpoints.pdf",
        ],
        "data_table": "main_results.csv",
        "notes": {
            "BRIDGE": "Frozen G4 global-Spearman checkpoints, three seeds.",
            "BRIDGE-B": "S0-S2 are exploratory seed-42 results; S3 reports frozen seeds 42-44.",
            "error_bars": "Population standard deviation across seeds.",
            "spatial_benefit": (
                "Frozen query-level expanded-node delta relative to Z0; fixed 16x16 "
                "square metric cells, all signs retained."
            ),
        },
    }
    with (output / "figure_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configure_style()
    rows: list[dict[str, object]] = []
    make_pipeline(output)
    make_ranking(output, rows)
    make_mechanism_ablation(output, rows)
    make_system(output, rows)
    make_bridge_b_progression(output, rows)
    write_rows(output, rows)
    print(f"Generated 5 figures and {len(rows)} auditable data rows in {output}")


if __name__ == "__main__":
    main()
