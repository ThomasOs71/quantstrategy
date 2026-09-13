"""Public-facing plotting helpers for Article 2 diagnostic exhibits."""

from __future__ import annotations

from itertools import cycle
from math import sqrt
import re
from typing import Mapping, Sequence

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import MaxNLocator

from series.from_assumptions_to_portfolios.flagship_plot_style import (
    CHARCOAL,
    ACF_FIGURE_SIZE,
    COBALT,
    LIGHT_GREY,
    MUTED_GREY,
    OFF_WHITE,
    PALE_BLUE,
    TEAL,
    add_direct_labels,
    add_empirical_footer,
    add_title,
    create_figure,
    style_axis,
)


DRIVER_LABELS = {
    "global_dm_ex_emu": "Global DM ex-EMU",
    "euro_govt_bond_7_10": "Euro Govt Bonds 7–10y",
    "euro_ig_credit": "Euro IG Credit",
    "euro_high_yield": "Euro High Yield",
    "commodities": "Commodities",
    "gold": "Gold",
    "fx_eurusd": "EUR/USD",
}


def _label(driver: str) -> str:
    return DRIVER_LABELS.get(driver, driver.replace("_", " ").title())


def _check_drivers(frame: pd.DataFrame, drivers: Sequence[str], column: str) -> None:
    if not drivers:
        raise ValueError("drivers must not be empty")
    missing = set(drivers) - set(frame[column])
    if missing:
        raise ValueError(f"drivers absent from input: {sorted(missing)}")


def plot_autocorrelation(
    acf_table: pd.DataFrame,
    drivers: Sequence[str],
    *,
    title: str,
    subtitle: str,
    sample_size: int,
    lag_unit: str = "months",
    footer: Mapping[str, str] | None = None,
):
    """Plot four ACF panels with common scale and a white-noise reference."""
    _check_drivers(acf_table, drivers, "driver")
    if len(drivers) != 4:
        raise ValueError("drivers must contain exactly four entries")
    if sample_size < 2:
        raise ValueError("sample_size must be at least two")

    reference = 1.96 / sqrt(sample_size)
    selected = acf_table.loc[acf_table["driver"].isin(drivers)]
    max_abs = max(float(selected["autocorrelation"].abs().max()), reference) + 0.04
    y_limit = min(1.0, max(0.20, np.ceil(max_abs * 10.0) / 10.0))

    figure = create_figure(size=ACF_FIGURE_SIZE)
    add_title(figure, title, subtitle)
    axes = figure.subplots(4, 1)
    for axis, driver in zip(axes.flat, drivers):
        subset = acf_table.loc[acf_table["driver"] == driver].sort_values("lag")
        axis.bar(
            subset["lag"],
            subset["autocorrelation"],
            color=COBALT,
            width=0.66,
            zorder=3,
        )
        axis.axhline(
            reference, color=MUTED_GREY, linewidth=1.8, linestyle="--", zorder=2
        )
        axis.axhline(
            -reference, color=MUTED_GREY, linewidth=1.8, linestyle="--", zorder=2
        )
        style_axis(axis, zero_line=True)
        axis.set_ylim(-y_limit, y_limit)
        axis.set_xlim(0.25, float(selected["lag"].max()) + 0.75)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        axis.set_title(
            _label(driver), loc="left", fontsize=25, fontweight="semibold", pad=14
        )
    figure.text(
        0.94,
        0.895,
        f"Dashed lines: ±{reference:.2f} approximate 95% white-noise reference",
        ha="right",
        fontsize=16,
        color=MUTED_GREY,
    )
    figure.text(
        0.025,
        0.50,
        "Autocorrelation",
        ha="center",
        va="center",
        rotation="vertical",
        fontsize=20,
        color=MUTED_GREY,
    )
    figure.text(
        0.5,
        0.155,
        f"Lag ({lag_unit})",
        ha="center",
        fontsize=20,
        color=MUTED_GREY,
    )
    figure.subplots_adjust(left=0.11, right=0.96, top=0.84, bottom=0.19, hspace=0.58)
    add_empirical_footer(figure, **(footer or {}))
    return figure


def plot_rolling_volatility(
    rolling_volatility: pd.DataFrame,
    drivers: Sequence[str],
    *,
    window_label: str = "12-month",
    footer: Mapping[str, str] | None = None,
):
    """Plot selected annualised rolling-volatility series with direct labels."""
    if not drivers:
        raise ValueError("drivers must not be empty")
    missing = set(drivers) - set(rolling_volatility.columns)
    if missing:
        raise ValueError(
            f"drivers absent from rolling volatility table: {sorted(missing)}"
        )

    colors = [COBALT, TEAL, CHARCOAL, PALE_BLUE]
    figure = create_figure()
    add_title(
        figure,
        "Volatility is not constant through time",
        f"{window_label} rolling annualized volatility",
    )
    axis = figure.add_axes([0.11, 0.23, 0.70, 0.57])
    plotted = []
    for driver, color in zip(drivers, cycle(colors)):
        series = rolling_volatility[driver]
        axis.plot(series.index, series, color=color, linewidth=2.6, zorder=3)
        plotted.append((_label(driver), series, color))
    style_axis(axis, percent=True)
    axis.set_ylabel("Annualized volatility", fontsize=20, color=MUTED_GREY)
    axis.xaxis.set_major_locator(mdates.YearLocator(2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.set_xlim(
        rolling_volatility.index.min(),
        rolling_volatility.index.max() + pd.DateOffset(months=10),
    )
    add_direct_labels(axis, plotted)
    add_empirical_footer(figure, **(footer or {}))
    return figure


def plot_rolling_correlation(
    rolling_correlation: pd.DataFrame,
    *,
    window_label: str = "24-month",
    footer: Mapping[str, str] | None = None,
):
    """Plot the selected rolling correlations with direct labels."""
    if rolling_correlation.empty:
        raise ValueError("rolling_correlation must not be empty")

    colors = [COBALT, TEAL, CHARCOAL, PALE_BLUE]
    figure = create_figure()
    add_title(
        figure,
        "Cross-asset dependence changes through time",
        f"{window_label} rolling correlations with Global DM ex-EMU",
    )
    axis = figure.add_axes([0.11, 0.23, 0.70, 0.57])
    plotted = []
    for column, color in zip(rolling_correlation.columns, cycle(colors)):
        series = rolling_correlation[column]
        other = column.split("__", maxsplit=1)[-1]
        axis.plot(series.index, series, color=color, linewidth=2.6, zorder=3)
        plotted.append((_label(other), series, color))
    style_axis(axis, zero_line=True)
    axis.set_ylim(-1.0, 1.0)
    axis.set_ylabel("Correlation", fontsize=20, color=MUTED_GREY)
    axis.xaxis.set_major_locator(mdates.YearLocator(2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    axis.set_xlim(
        rolling_correlation.index.min(),
        rolling_correlation.index.max() + pd.DateOffset(months=14),
    )
    add_direct_labels(axis, plotted)
    add_empirical_footer(figure, **(footer or {}))
    return figure


def plot_joint_tail_event_timeline(
    joint_tail_events: pd.DataFrame,
    *,
    period_label_plural: str = "Months",
    footer: Mapping[str, str] | None = None,
):
    """Plot simultaneous empirical lower-tail events as an ordered lollipop timeline."""
    figure = create_figure()
    add_title(
        figure,
        "Tail events arrive as joint market events",
        f"{period_label_plural} in which multiple return series fall beneath "
        "their own empirical lower 5% threshold",
    )
    axis = figure.add_axes([0.11, 0.23, 0.80, 0.57])
    style_axis(axis)
    axis.set_ylabel("Assets + FX factor in lower tail", fontsize=20, color=MUTED_GREY)
    axis.yaxis.set_major_locator(MaxNLocator(integer=True))

    events = joint_tail_events.copy()
    if not events.empty:
        events["date"] = pd.to_datetime(events["date"])
        events = events.sort_values("date")
        severe = events["tail_asset_count"] >= 5
        axis.vlines(
            events["date"],
            0,
            events["tail_asset_count"],
            color=LIGHT_GREY,
            linewidth=1.6,
            zorder=2,
        )
        axis.scatter(
            events.loc[~severe, "date"],
            events.loc[~severe, "tail_asset_count"],
            s=90,
            color=TEAL,
            zorder=3,
        )
        axis.scatter(
            events.loc[severe, "date"],
            events.loc[severe, "tail_asset_count"],
            s=120,
            color=COBALT,
            zorder=4,
        )
        annotations = {
            pd.Timestamp("2020-03-31"): "Mar 2020",
            pd.Timestamp("2022-04-30"): "Apr 2022",
            pd.Timestamp("2022-12-31"): "Dec 2022",
        }
        for date, label in annotations.items():
            match = events.loc[events["date"] == date]
            if not match.empty:
                value = match["tail_asset_count"].iloc[0]
                axis.annotate(
                    label,
                    (date, value),
                    xytext=(0, 16),
                    textcoords="offset points",
                    ha="center",
                    fontsize=17,
                    color=CHARCOAL,
                )
        axis.set_xlim(
            events["date"].min() - pd.DateOffset(months=4),
            events["date"].max() + pd.DateOffset(months=4),
        )
    else:
        axis.text(
            0.5,
            0.5,
            "No joint lower-tail events in the selected sample",
            transform=axis.transAxes,
            ha="center",
            color=MUTED_GREY,
            fontsize=20,
        )
    axis.xaxis.set_major_locator(mdates.YearLocator(2))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    add_empirical_footer(figure, **(footer or {}))
    return figure


def plot_state_dependence(
    state_summary: pd.DataFrame,
    drivers: Sequence[str],
    *,
    forward_horizon_label: str = "12-month",
    state_window_label: str = "12-month",
    footer: Mapping[str, str] | None = None,
):
    """Plot descriptive forward returns by observed volatility bucket."""
    subset = state_summary.loc[state_summary["driver"].isin(drivers)].copy()
    if subset.empty:
        raise ValueError("no selected drivers are present in state_summary")
    buckets = list(dict.fromkeys(subset["state_bucket"]))
    if len(buckets) != 3:
        raise ValueError("state_summary must contain exactly three state buckets")
    bucket_labels = [
        f"{label}\nn={_bucket_observations(subset, bucket)}"
        for label, bucket in zip(("LOW VOL", "MEDIUM VOL", "HIGH VOL"), buckets)
    ]
    colors = [COBALT, TEAL, CHARCOAL, PALE_BLUE]

    figure = create_figure()
    boundaries = " / ".join(_format_bucket_boundary(bucket) for bucket in buckets)
    add_title(
        figure,
        "Forward outcomes vary by volatility state",
        f"Average realised {forward_horizon_label} simple returns, by trailing "
        f"{state_window_label} Global DM ex-EMU volatility",
    )
    figure.text(
        0.06,
        0.865,
        f"Volatility buckets: {boundaries}",
        fontsize=16,
        color=MUTED_GREY,
    )
    axis = figure.add_axes([0.11, 0.25, 0.70, 0.54])
    plotted = []
    for driver, color in zip(drivers, cycle(colors)):
        ordered = (
            subset.loc[subset["driver"] == driver]
            .set_index("state_bucket")
            .reindex(buckets)
        )
        series = ordered["mean_forward_return"].copy()
        series.index = np.arange(len(series))
        axis.plot(
            series.index,
            series,
            color=color,
            linewidth=2.5,
            marker="o",
            markersize=8,
            zorder=3,
        )
        plotted.append((_label(driver), series, color))
    style_axis(axis, percent=True, zero_line=True)
    axis.set_xlim(-0.15, 3.35)
    axis.set_xticks(range(3), bucket_labels)
    axis.set_ylabel(
        f"Mean {forward_horizon_label} forward return",
        fontsize=20,
        color=MUTED_GREY,
    )
    add_direct_labels(axis, plotted)
    add_empirical_footer(
        figure,
        **(footer or {}),
        note=(
            f"Descriptive only; {forward_horizon_label} forward windows overlap "
            "and are not forecasts."
        ),
    )
    return figure


def _format_bucket_boundary(bucket: str) -> str:
    """Express pandas interval labels as readable annualized-volatility ranges."""
    bounds = re.findall(r"-?\d+(?:\.\d+)?", bucket)
    if len(bounds) != 2:
        return bucket
    left, right = (float(value) for value in bounds)
    return f"{left:.1%}–{right:.1%}"


def _bucket_observations(state_summary: pd.DataFrame, bucket: str) -> int:
    """Return the common number of forward windows assigned to one state bucket."""
    observations = state_summary.loc[
        state_summary["state_bucket"] == bucket, "n_observations"
    ]
    if observations.empty:
        raise ValueError(f"missing observations for state bucket '{bucket}'")
    if observations.nunique() != 1:
        raise ValueError(
            "state buckets must have a common observation count across drivers"
        )
    return int(observations.iloc[0])


def plot_scenario_architecture():
    """Draw the Article-2 conceptual bridge in a phone-readable flow."""
    figure = create_figure(framework=True)
    add_title(
        figure,
        "From historical return structure to scenario design",
        "Two complementary routes; neither is universally superior.",
    )
    axis = figure.add_axes([0.10, 0.075, 0.80, 0.79])
    axis.set_axis_off()

    def box(y, height, text, *, fill, edge=LIGHT_GREY, size=24, weight="medium"):
        patch = FancyBboxPatch(
            (0.04, y),
            0.92,
            height,
            boxstyle="round,pad=0.014,rounding_size=0.018",
            linewidth=1.4,
            edgecolor=edge,
            facecolor=fill,
            transform=axis.transAxes,
        )
        axis.add_patch(patch)
        axis.text(
            0.50,
            y + height / 2,
            text,
            transform=axis.transAxes,
            ha="center",
            va="center",
            fontsize=size,
            color=CHARCOAL,
            fontweight=weight,
            wrap=True,
        )

    def arrow(top, bottom):
        axis.add_patch(
            FancyArrowPatch(
                (0.50, top),
                (0.50, bottom),
                transform=axis.transAxes,
                arrowstyle="-|>",
                mutation_scale=20,
                linewidth=1.8,
                color=MUTED_GREY,
            )
        )

    # The y coordinates form a fixed vertical grid. Each arrow occupies only
    # the explicit gap between its source and target box.
    box(
        0.910,
        0.070,
        "HISTORICAL RETURN PANEL",
        fill=PALE_BLUE,
        edge=PALE_BLUE,
        size=29,
        weight="semibold",
    )
    axis.text(
        0.04,
        0.860,
        "EMPIRICAL-FIRST",
        transform=axis.transAxes,
        fontsize=25,
        color=COBALT,
        fontweight="semibold",
    )
    box(0.755, 0.075, "Observed market structure", fill=OFF_WHITE)
    arrow(0.745, 0.710)
    box(0.625, 0.075, "PRESERVE  /  CONDITION  /  FILTER", fill=OFF_WHITE)
    arrow(0.615, 0.580)
    box(
        0.440,
        0.130,
        "Scenario engines\n\nA  Synchronized Historical Block Bootstrap\nB  Conditional Similarity Resampling\nC  Filtered Historical Simulation",
        fill="#EEF2FF",
        edge=PALE_BLUE,
        size=20,
    )
    axis.text(
        0.04,
        0.400,
        "Preserves more observed market structure",
        transform=axis.transAxes,
        fontsize=18,
        color=MUTED_GREY,
    )
    axis.text(
        0.04,
        0.360,
        "INVARIANCE-FIRST",
        transform=axis.transAxes,
        fontsize=25,
        color=TEAL,
        fontweight="semibold",
    )
    box(0.280, 0.055, "Model conditional dynamics", fill=OFF_WHITE, size=21)
    arrow(0.270, 0.235)
    box(
        0.170,
        0.055,
        "Standardized, more invariant innovations",
        fill=OFF_WHITE,
        size=21,
    )
    arrow(0.160, 0.125)
    box(
        0.060,
        0.055,
        "Model dependence / reconstruct scenarios",
        fill="#EAF8F8",
        edge=PALE_BLUE,
        size=21,
    )
    figure.text(
        0.5,
        0.022,
        "Cleaner innovations require additional modelling assumptions.\n"
        "Less dependence in the residuals does not mean fewer assumptions in the model.",
        ha="center",
        fontsize=19,
        color=CHARCOAL,
        fontweight="medium",
    )
    return figure
