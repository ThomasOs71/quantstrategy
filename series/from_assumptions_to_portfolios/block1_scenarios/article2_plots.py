"""Publication-oriented plotting helpers for Article 2 diagnostics."""

from __future__ import annotations

from math import ceil
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd


def plot_autocorrelation(
    acf_table: pd.DataFrame,
    drivers: Sequence[str],
    title: str,
):
    """Plot per-driver ACF bars from an Article-2 autocorrelation table."""
    if not drivers:
        raise ValueError("drivers must not be empty")
    missing = set(drivers) - set(acf_table["driver"])
    if missing:
        raise ValueError(f"drivers absent from autocorrelation table: {sorted(missing)}")

    ncols = min(2, len(drivers))
    nrows = ceil(len(drivers) / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 3.5 * nrows), squeeze=False)
    for axis, driver in zip(axes.flat, drivers):
        subset = acf_table.loc[acf_table["driver"] == driver].sort_values("lag")
        axis.bar(subset["lag"], subset["autocorrelation"], color="#33658A")
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(driver.replace("_", " ").title())
        axis.set_xlabel("Lag (months)")
        axis.set_ylabel("Autocorrelation")
    for axis in axes.flat[len(drivers) :]:
        axis.set_visible(False)
    figure.suptitle(title, y=1.02)
    figure.tight_layout()
    return figure


def plot_rolling_volatility(rolling_volatility: pd.DataFrame, drivers: Sequence[str]):
    """Plot selected annualised rolling volatility series."""
    missing = set(drivers) - set(rolling_volatility.columns)
    if missing:
        raise ValueError(f"drivers absent from rolling volatility table: {sorted(missing)}")
    figure, axis = plt.subplots(figsize=(10, 5))
    for driver in drivers:
        axis.plot(
            rolling_volatility.index,
            rolling_volatility[driver],
            label=driver.replace("_", " ").title(),
        )
    axis.set_title("Rolling annualized volatility")
    axis.set_ylabel("Volatility")
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()
    return figure


def plot_rolling_correlation(rolling_correlation: pd.DataFrame):
    """Plot each named rolling correlation series in one chart."""
    if rolling_correlation.empty:
        raise ValueError("rolling_correlation must not be empty")
    figure, axis = plt.subplots(figsize=(10, 5))
    for column in rolling_correlation.columns:
        label = column.replace("__", " vs ").replace("_", " ").title()
        axis.plot(rolling_correlation.index, rolling_correlation[column], label=label)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_ylim(-1.0, 1.0)
    axis.set_title("Rolling cross-asset correlations")
    axis.set_ylabel("Correlation")
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure


def plot_joint_tail_event_timeline(joint_tail_events: pd.DataFrame):
    """Plot the breadth of historical simultaneous lower-tail events over time."""
    if joint_tail_events.empty:
        raise ValueError("joint_tail_events must not be empty")
    figure, axis = plt.subplots(figsize=(10, 4))
    axis.scatter(
        joint_tail_events["date"],
        joint_tail_events["tail_asset_count"],
        color="#B23A48",
        s=36,
    )
    axis.set_title("Historical simultaneous lower-tail events")
    axis.set_ylabel("Number of drivers in lower tail")
    figure.tight_layout()
    return figure


def plot_state_dependence(
    state_summary: pd.DataFrame,
    drivers: Sequence[str],
):
    """Plot mean forward returns by observed state bucket for selected drivers."""
    subset = state_summary.loc[state_summary["driver"].isin(drivers)]
    if subset.empty:
        raise ValueError("no selected drivers are present in state_summary")
    pivoted = subset.pivot(
        index="state_bucket", columns="driver", values="mean_forward_return"
    )
    pivoted = pivoted.rename(
        columns=lambda value: value.replace("_", " ").title()
    )
    figure, axis = plt.subplots(figsize=(10, 5))
    pivoted.plot(kind="bar", ax=axis, width=0.8)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_title("Mean forward returns by observed volatility state")
    axis.set_xlabel("Trailing-volatility bucket")
    axis.set_ylabel("Forward simple return")
    axis.legend(frameon=False)
    figure.tight_layout()
    return figure
