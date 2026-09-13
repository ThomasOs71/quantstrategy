"""Shared institutional editorial style for flagship-series exhibits."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


OFF_WHITE = "#F6F3EA"
CHARCOAL = "#20252B"
COBALT = "#4D7CFE"
TEAL = "#20B8C6"
PALE_BLUE = "#A9D9E8"
MUTED_GREY = "#69727D"
LIGHT_GREY = "#D9D7D0"

# Static Substack images are commonly scaled to roughly 390 CSS pixels on a
# phone. Taller canvases preserve useful chart height after that scaling.
STANDARD_FIGURE_SIZE = (16, 12)
FRAMEWORK_FIGURE_SIZE = (16, 28)
ACF_FIGURE_SIZE = (16, 24)
OUTPUT_DPI = 100

_RC_PARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Arial", "DejaVu Sans"],
    "figure.facecolor": OFF_WHITE,
    "axes.facecolor": OFF_WHITE,
    "savefig.facecolor": OFF_WHITE,
    "text.color": CHARCOAL,
    "axes.labelcolor": CHARCOAL,
    "xtick.color": MUTED_GREY,
    "ytick.color": MUTED_GREY,
    "axes.edgecolor": LIGHT_GREY,
}


def create_figure(*, framework: bool = False, size: tuple[float, float] | None = None):
    """Create a fixed-size, opaque flagship figure and its global axis."""
    size = size or (FRAMEWORK_FIGURE_SIZE if framework else STANDARD_FIGURE_SIZE)
    with mpl.rc_context(_RC_PARAMS):
        figure = plt.figure(figsize=size, dpi=OUTPUT_DPI, facecolor=OFF_WHITE)
    return figure


def add_title(figure, title: str, subtitle: str) -> None:
    """Place a claim-led title and a compact explanatory subtitle."""
    figure.text(0.06, 0.945, title, fontsize=34, fontweight="semibold", color=CHARCOAL)
    figure.text(0.06, 0.905, subtitle, fontsize=20, color=MUTED_GREY)


def style_axis(axis, *, percent: bool = False, zero_line: bool = False) -> None:
    """Apply the shared lightweight editorial axis treatment."""
    axis.set_facecolor(OFF_WHITE)
    axis.spines[["top", "right", "left"]].set_visible(False)
    axis.spines["bottom"].set_color(LIGHT_GREY)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.tick_params(axis="both", labelsize=20, length=0, pad=8)
    axis.grid(axis="y", color=LIGHT_GREY, linewidth=0.9, alpha=0.8)
    axis.set_axisbelow(True)
    if zero_line:
        axis.axhline(0.0, color=CHARCOAL, linewidth=0.9, zorder=1)
    if percent:
        axis.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))


def add_empirical_footer(
    figure,
    *,
    sample: str = "Jan 2011–Dec 2025",
    note: str | None = None,
    frequency_label: str = "Monthly",
    return_description: str = "EUR monthly log returns where applicable",
) -> None:
    """Add the required compact source, sample, and interpretation disclosure."""
    if note:
        figure.text(0.06, 0.125, note, fontsize=15, color=MUTED_GREY)
    figure.text(
        0.06,
        0.082,
        f"Sample: {sample}  |  {frequency_label}  |  EUR investor perspective",
        fontsize=18,
        color=MUTED_GREY,
    )
    figure.text(
        0.06,
        0.049,
        return_description,
        fontsize=17,
        color=MUTED_GREY,
    )
    figure.text(
        0.06,
        0.020,
        "Source: QuantStrategy calculations  |  Author calculation",
        fontsize=17,
        color=MUTED_GREY,
    )


def add_direct_labels(axis, lines: Iterable[tuple[str, object, str]]) -> None:
    """Label plotted time series at their latest finite observation."""
    for label, series, color in lines:
        values = np.asarray(series, dtype=float)
        finite = np.flatnonzero(np.isfinite(values))
        if len(finite) == 0:
            continue
        position = finite[-1]
        x_values = series.index if hasattr(series, "index") else np.arange(len(values))
        axis.annotate(
            label,
            xy=(x_values[position], values[position]),
            xytext=(8, 0),
            textcoords="offset points",
            color=color,
            fontsize=20,
            fontweight="medium",
            va="center",
            clip_on=False,
        )


def save_figure(figure, path: Path) -> None:
    """Write an opaque, sRGB-compatible PNG at the requested pixel dimensions."""
    figure.savefig(path, dpi=OUTPUT_DPI, transparent=False, facecolor=OFF_WHITE)
    plt.close(figure)
