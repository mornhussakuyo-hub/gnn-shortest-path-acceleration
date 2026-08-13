#!/usr/bin/env python3
"""Generate a dense, programmatic region-materialization schematic."""

from __future__ import annotations

from collections import deque
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Polygon

from paper_figure_style import (
    BLUE,
    FONT_SIZE_SMALL,
    GREEN,
    INK,
    MUTED,
    ORANGE,
    PALE_BLUE,
    PURPLE,
    RED,
    ROAD,
    configure_style,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "paper" / "figures"
Node = tuple[int, int]


def build_road_graph() -> tuple[dict[Node, tuple[float, float]], set[tuple[Node, Node]]]:
    rng = np.random.default_rng(20260814)
    columns, rows = 14, 10
    positions: dict[Node, tuple[float, float]] = {}
    for column in range(columns):
        for row in range(rows):
            base_x = column / (columns - 1)
            base_y = row / (rows - 1)
            x_value = base_x + 0.018 * np.sin(0.82 * row + 0.35 * column)
            y_value = base_y + 0.022 * np.sin(0.58 * column - 0.41 * row)
            x_value += float(rng.normal(0, 0.006))
            y_value += float(rng.normal(0, 0.007))
            positions[(column, row)] = (0.06 + 0.88 * x_value, 0.27 + 0.57 * y_value)

    edges: set[tuple[Node, Node]] = set()

    def add_edge(first: Node, second: Node) -> None:
        edges.add(tuple(sorted((first, second))))

    for column in range(columns):
        for row in range(rows):
            if column + 1 < columns:
                add_edge((column, row), (column + 1, row))
            if row + 1 < rows:
                add_edge((column, row), (column, row + 1))
            if column + 1 < columns and row + 1 < rows and rng.random() < 0.20:
                add_edge((column, row), (column + 1, row + 1))
            if column + 1 < columns and row > 0 and rng.random() < 0.16:
                add_edge((column, row), (column + 1, row - 1))
    return positions, edges


def adjacency(edges: set[tuple[Node, Node]]) -> dict[Node, set[Node]]:
    result: dict[Node, set[Node]] = {}
    for first, second in edges:
        result.setdefault(first, set()).add(second)
        result.setdefault(second, set()).add(first)
    return result


def select_region(positions: dict[Node, tuple[float, float]]) -> set[Node]:
    selected = set()
    for node, (x_value, y_value) in positions.items():
        dx = (x_value - 0.54) / 0.30
        dy = (y_value - 0.49) / 0.29
        angle = np.arctan2(dy, dx)
        radius = np.hypot(dx, dy)
        boundary_radius = 1.0 + 0.11 * np.sin(3 * angle) - 0.07 * np.cos(5 * angle)
        if radius <= boundary_radius:
            selected.add(node)
    return selected


def convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    ordered = sorted(set(points))
    if len(ordered) <= 1:
        return ordered

    def cross(origin, first, second) -> float:
        return ((first[0] - origin[0]) * (second[1] - origin[1])
                - (first[1] - origin[1]) * (second[0] - origin[0]))

    lower: list[tuple[float, float]] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def padded_hull(
    positions: dict[Node, tuple[float, float]],
    region_nodes: set[Node],
) -> list[tuple[float, float]]:
    hull = convex_hull([positions[node] for node in region_nodes])
    center = np.mean(np.asarray(hull), axis=0)
    return [tuple(center + 1.06 * (np.asarray(point) - center)) for point in hull]


def nearest_node(
    positions: dict[Node, tuple[float, float]],
    candidates: set[Node],
    target: tuple[float, float],
) -> Node:
    return min(
        candidates,
        key=lambda node: (positions[node][0] - target[0]) ** 2
        + (positions[node][1] - target[1]) ** 2,
    )


def shortest_path(
    graph: dict[Node, set[Node]],
    source: Node,
    target: Node,
    allowed: set[Node],
) -> tuple[Node, ...]:
    queue = deque([source])
    parent: dict[Node, Node | None] = {source: None}
    while queue:
        node = queue.popleft()
        if node == target:
            break
        for neighbor in sorted(graph[node]):
            if neighbor in allowed and neighbor not in parent:
                parent[neighbor] = node
                queue.append(neighbor)
    if target not in parent:
        raise RuntimeError(f"No path between {source} and {target}")
    path = []
    current: Node | None = target
    while current is not None:
        path.append(current)
        current = parent[current]
    return tuple(reversed(path))


POSITIONS, EDGES = build_road_graph()
GRAPH = adjacency(EDGES)
REGION_NODES = select_region(POSITIONS)
BOUNDARY_NODES = {
    node for node in REGION_NODES if any(neighbor not in REGION_NODES for neighbor in GRAPH[node])
}
INTERIOR_NODES = REGION_NODES - BOUNDARY_NODES
REGION_HULL = padded_hull(POSITIONS, REGION_NODES)
OUTSIDE_NODES = set(POSITIONS) - REGION_NODES

ENTRY = nearest_node(POSITIONS, BOUNDARY_NODES, (0.28, 0.72))
EXIT = nearest_node(POSITIONS, BOUNDARY_NODES, (0.80, 0.27))
ORIGIN_OUTSIDE = nearest_node(POSITIONS, OUTSIDE_NODES, (0.07, 0.72))
DESTINATION_OUTSIDE = nearest_node(POSITIONS, OUTSIDE_NODES, (0.94, 0.20))
ORIGIN_INSIDE = nearest_node(POSITIONS, INTERIOR_NODES, (0.48, 0.52))
DESTINATION_INSIDE = nearest_node(POSITIONS, INTERIOR_NODES, (0.65, 0.38))


def draw_base(ax: plt.Axes) -> None:
    ax.add_patch(
        Polygon(
            REGION_HULL,
            closed=True,
            facecolor=PALE_BLUE,
            edgecolor=BLUE,
            linewidth=0.8,
            linestyle=(0, (4, 2)),
            zorder=0,
        )
    )
    for first, second in EDGES:
        ax.plot(
            (POSITIONS[first][0], POSITIONS[second][0]),
            (POSITIONS[first][1], POSITIONS[second][1]),
            color=ROAD,
            linewidth=0.48,
            alpha=0.88,
            zorder=1,
        )
    coordinates = np.asarray(list(POSITIONS.values()))
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=3.8,
        facecolor="white",
        edgecolor=MUTED,
        linewidth=0.35,
        zorder=2,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0.02, 0.98)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")


def highlight_nodes(ax: plt.Axes, nodes: set[Node], color: str, size: float) -> None:
    coordinates = np.asarray([POSITIONS[node] for node in nodes])
    ax.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        s=size,
        facecolor=color,
        edgecolor="white",
        linewidth=0.45,
        zorder=5,
    )


def draw_path(ax: plt.Axes, path: tuple[Node, ...], color: str, width: float = 1.6) -> None:
    coordinates = np.asarray([POSITIONS[node] for node in path])
    ax.plot(
        coordinates[:, 0],
        coordinates[:, 1],
        color=color,
        linewidth=width,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=4,
    )


def draw_shortcut(ax: plt.Axes, source: Node, target: Node) -> None:
    ax.add_patch(
        FancyArrowPatch(
            POSITIONS[source],
            POSITIONS[target],
            arrowstyle="-|>",
            mutation_scale=8,
            connectionstyle="arc3,rad=-0.18",
            color=PURPLE,
            linewidth=1.6,
            linestyle=(0, (4, 2)),
            zorder=6,
        )
    )


def draw_endpoint(ax: plt.Axes, node: Node, label: str, color: str) -> None:
    x_value, y_value = POSITIONS[node]
    ax.scatter(
        x_value,
        y_value,
        s=38,
        facecolor=color,
        edgecolor="white",
        linewidth=0.8,
        zorder=7,
    )
    ax.text(
        x_value,
        y_value,
        label,
        ha="center",
        va="center",
        color="white",
        fontsize=6.5,
        weight="bold",
        zorder=8,
    )


def callout(
    ax: plt.Axes,
    text: str,
    target: tuple[float, float],
    label: tuple[float, float],
    color: str = INK,
    align: str = "left",
) -> None:
    ax.annotate(
        text,
        xy=target,
        xytext=label,
        textcoords="axes fraction",
        ha=align,
        va="center",
        color=color,
        fontsize=FONT_SIZE_SMALL,
        weight="bold",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": color,
              "linewidth": 0.65, "alpha": 0.96},
        arrowprops={"arrowstyle": "->", "color": color, "linewidth": 0.7,
                    "shrinkA": 2, "shrinkB": 2},
        zorder=10,
    )


def outside_path(source: Node, target: Node) -> tuple[Node, ...]:
    return shortest_path(GRAPH, source, target, OUTSIDE_NODES | BOUNDARY_NODES)


def make_figure() -> plt.Figure:
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.85))
    titles = (
        "(a) Region interior",
        "(b) Boundary nodes",
        "(c) Materialized shortcut",
        "(d) Both endpoints outside",
        "(e) One endpoint inside",
        "(f) Both endpoints in one region",
    )
    for ax, title in zip(axes.flat, titles):
        draw_base(ax)
        ax.set_title(title, loc="left", pad=2)

    highlight_nodes(axes[0, 0], INTERIOR_NODES, BLUE, 10)
    interior_target = POSITIONS[nearest_node(POSITIONS, INTERIOR_NODES, (0.55, 0.50))]
    callout(axes[0, 0], "Interior nodes removed", interior_target, (0.50, 0.07), BLUE, "center")

    highlight_nodes(axes[0, 1], BOUNDARY_NODES, ORANGE, 15)
    callout(
        axes[0, 1],
        "Boundary nodes retained",
        POSITIONS[ENTRY],
        (0.50, 0.07),
        ORANGE,
        "center",
    )

    in_region_path = shortest_path(GRAPH, ENTRY, EXIT, REGION_NODES)
    draw_path(axes[0, 2], in_region_path, GREEN, 1.05)
    draw_shortcut(axes[0, 2], ENTRY, EXIT)
    highlight_nodes(axes[0, 2], {ENTRY, EXIT}, ORANGE, 18)
    midpoint = tuple((np.asarray(POSITIONS[ENTRY]) + np.asarray(POSITIONS[EXIT])) / 2)
    callout(
        axes[0, 2],
        "Overlay shortcut = exact path distance",
        midpoint,
        (0.50, 0.07),
        PURPLE,
        "center",
    )

    draw_path(axes[1, 0], outside_path(ORIGIN_OUTSIDE, ENTRY), GREEN)
    draw_shortcut(axes[1, 0], ENTRY, EXIT)
    draw_path(axes[1, 0], outside_path(EXIT, DESTINATION_OUTSIDE), GREEN)
    draw_endpoint(axes[1, 0], ORIGIN_OUTSIDE, "s", GREEN)
    draw_endpoint(axes[1, 0], DESTINATION_OUTSIDE, "t", RED)

    local_access = shortest_path(GRAPH, ORIGIN_INSIDE, ENTRY, REGION_NODES)
    draw_path(axes[1, 1], local_access, GREEN)
    draw_shortcut(axes[1, 1], ENTRY, EXIT)
    draw_path(axes[1, 1], outside_path(EXIT, DESTINATION_OUTSIDE), GREEN)
    draw_endpoint(axes[1, 1], ORIGIN_INSIDE, "s", GREEN)
    draw_endpoint(axes[1, 1], DESTINATION_OUTSIDE, "t", RED)
    local_midpoint = POSITIONS[local_access[len(local_access) // 2]]
    callout(
        axes[1, 1],
        "Local endpoint access",
        local_midpoint,
        (0.50, 0.07),
        GREEN,
        "center",
    )

    local_path = shortest_path(GRAPH, ORIGIN_INSIDE, DESTINATION_INSIDE, REGION_NODES)
    draw_path(axes[1, 2], local_path, GREEN)
    draw_endpoint(axes[1, 2], ORIGIN_INSIDE, "s", GREEN)
    draw_endpoint(axes[1, 2], DESTINATION_INSIDE, "t", RED)
    local_target = POSITIONS[local_path[len(local_path) // 2]]
    callout(
        axes[1, 2],
        "Exact in-region path",
        local_target,
        (0.50, 0.07),
        GREEN,
        "center",
    )

    legend_handles = (
        Line2D([], [], color=ROAD, linewidth=1.0, label="Road edge"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=BLUE,
               markeredgecolor="white", markersize=5, label="Interior node"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=ORANGE,
               markeredgecolor="white", markersize=5, label="Boundary node"),
        Line2D([], [], color=PURPLE, linewidth=1.6, linestyle=(0, (4, 2)),
               label="Materialized shortcut (overlay)"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=GREEN,
               markeredgecolor="white", markersize=5, label="Origin s"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor=RED,
               markeredgecolor="white", markersize=5, label="Destination t"),
    )
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        bbox_to_anchor=(0.5, 0.004),
        handlelength=2.6,
        columnspacing=1.3,
        handletextpad=0.65,
    )
    fig.subplots_adjust(left=0.025, right=0.99, top=0.965, bottom=0.14, wspace=0.10, hspace=0.20)
    return fig


def main() -> None:
    configure_style()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    figure = make_figure()
    figure.savefig(OUTPUT / "materialization_schematic.pdf", bbox_inches="tight", pad_inches=0.04)
    figure.savefig(
        OUTPUT / "materialization_schematic.png",
        dpi=260,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(figure)
    print(
        f"Generated materialization schematic with {len(POSITIONS)} road nodes, "
        f"{len(REGION_NODES)} region nodes, and {len(BOUNDARY_NODES)} boundary nodes"
    )


if __name__ == "__main__":
    main()
