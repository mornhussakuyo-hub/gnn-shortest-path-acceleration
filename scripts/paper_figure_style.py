"""Shared visual style for all AIC manuscript figures."""

from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


BLUE = "#3B6FB6"
ORANGE = "#E6862A"
GREEN = "#3A9D70"
RED = "#C75252"
PURPLE = "#7562B5"
INK = "#263238"
MUTED = "#65747B"
GRID = "#D9E0E3"
ROAD = "#CBD2D5"
PALE_BLUE = "#EAF1FA"
PALE_ORANGE = "#FCEFE2"
PALE_GREEN = "#E8F4EE"
NEUTRAL = "#F7F7F5"

METHOD_COLORS = {
    "Z0": BLUE,
    "BRIDGE": ORANGE,
    "BRIDGE-B": GREEN,
}
WINDOW_COLORS = {
    "current_y": ORANGE,
    "future_f": GREEN,
}

# Negative deployment deltas are worse and positive deltas are better.
BENEFIT_CMAP = LinearSegmentedColormap.from_list(
    "aic_benefit",
    [RED, NEUTRAL, BLUE],
)


def configure_style() -> None:
    """Apply the manuscript-wide typography, stroke, and color conventions."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.labelsize": 9,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#8A989E",
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
