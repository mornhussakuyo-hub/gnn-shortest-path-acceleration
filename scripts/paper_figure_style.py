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
ROAD = "#9EAAAF"
PALE_BLUE = "#EAF1FA"
PALE_ORANGE = "#FCEFE2"
PALE_GREEN = "#E8F4EE"
NEUTRAL = "#F7F7F5"

FONT_FAMILY = "DejaVu Sans"
FONT_SIZE_BODY = 9
FONT_SIZE_SMALL = 8
FONT_SIZE_PANEL_TITLE = 10
FONT_SIZE_SUPTITLE = 11

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
# Transparent cells carry white RGB so PDF viewers cannot introduce dark halos
# when antialiasing the boundary between missing and colored raster cells.
BENEFIT_CMAP.set_bad(color=(1.0, 1.0, 1.0, 0.0))


def configure_style() -> None:
    """Apply the manuscript-wide typography, stroke, and color conventions."""
    plt.rcParams.update(
        {
            "font.family": FONT_FAMILY,
            "font.size": FONT_SIZE_BODY,
            "axes.titlesize": FONT_SIZE_PANEL_TITLE,
            "axes.titleweight": "bold",
            "axes.titlecolor": INK,
            "axes.labelsize": FONT_SIZE_BODY,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#8A989E",
            "axes.linewidth": 0.7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": FONT_SIZE_SMALL,
            "ytick.labelsize": FONT_SIZE_SMALL,
            "legend.fontsize": FONT_SIZE_SMALL,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
