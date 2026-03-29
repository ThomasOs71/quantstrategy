from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .config import ExperimentConfig, to_serializable_dict

ARCHITECTURE_ORDER = ["A", "B", "C"]
ARCHITECTURE_COLORS = {"A": "#1f77b4", "B": "#d62728", "C": "#2ca02c"}
ASSET_COLORS = {"US": "#1f77b4", "Europe": "#ff7f0e", "EM": "#2ca02c", "Japan": "#7f7f7f"}
FEASIBLE_STATUSES = {"optimal", "optimal_inaccurate"}


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _percent_axis(axis: plt.Axes, x: bool = False, y: bool = False) -> None:
    formatter = mtick.PercentFormatter(xmax=1.0, decimals=1)
    if x:
        axis.xaxis.set_major_formatter(formatter)
    if y:
        axis.yaxis.set_major_formatter(formatter)


def make_summary_tables(
    results_df: pd.DataFrame,
    config: ExperimentConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create summary tables for architectures and selected sweep snapshots."""

    feasible = results_df[results_df["status"].isin(FEASIBLE_STATUSES)].copy()
    architecture_index = (
        results_df[["architecture", "architecture_label"]]
        .drop_duplicates()
        .assign(architecture=lambda df: pd.Categorical(df["architecture"], ARCHITECTURE_ORDER, ordered=True))
        .sort_values("architecture")
        .reset_index(drop=True)
    )

    summary = (
        feasible.groupby(["architecture", "architecture_label"], sort=False)
        .agg(
            average_expected_return=("expected_return", "mean"),
            average_portfolio_cvar=("portfolio_cvar", "mean"),
            average_portfolio_cvar_demeaned=("portfolio_cvar_demeaned", "mean"),
            average_portfolio_cvar_full=("portfolio_cvar_full", "mean"),
            average_te_cvar=("te_cvar", "mean"),
            average_te_cvar_demeaned=("te_cvar_demeaned", "mean"),
            average_te_cvar_full=("te_cvar_full", "mean"),
            average_l2_benchmark_distance=("bm_dist_l2", "mean"),
            average_turnover=("turnover", "mean"),
            average_herfindahl=("herfindahl", "mean"),
            average_instability_score=("instability_score", "mean"),
            mean_adjacent_l1_change=("adj_l1_change", "mean"),
            mean_adjacent_l2_change=("adj_l2_change", "mean"),
            max_jump=("adj_max_jump", "max"),
            average_abs_change_us=("adj_abs_change_US", "mean"),
            average_abs_change_europe=("adj_abs_change_Europe", "mean"),
            average_solve_time_seconds=("solve_time_seconds", "mean"),
            feasible_points=("expected_return", "count"),
        )
        .reset_index()
    )
    summary = architecture_index.merge(summary, how="left", on=["architecture", "architecture_label"])
    summary["feasible_points"] = summary["feasible_points"].fillna(0).astype(int)
    summary["total_points"] = len(config.mu_europe_grid)
    summary["infeasible_points"] = summary["total_points"] - summary["feasible_points"]

    selected_rows = []
    for target_mu in config.selected_sweep_points:
        subset = feasible[np.isclose(feasible["mu_europe"], target_mu)].copy()
        subset["selected_mu_europe"] = target_mu
        selected_rows.append(subset)

    snapshot = pd.concat(selected_rows, ignore_index=True) if selected_rows else feasible.iloc[0:0].copy()
    snapshot = snapshot[
        [
            "architecture",
            "architecture_label",
            "selected_mu_europe",
            "mu_europe",
            "expected_return",
            "stage1_expected_return",
            "stage2_expected_return",
            "objective_loss_pct",
            "portfolio_cvar_limit_mode",
            "requested_portfolio_cvar_limit",
            "benchmark_portfolio_cvar_demeaned",
            "effective_portfolio_cvar_limit",
            "portfolio_cvar",
            "portfolio_cvar_demeaned",
            "portfolio_cvar_full",
            "te_cvar",
            "te_cvar_demeaned",
            "te_cvar_full",
            "bm_dist_l2",
            "bm_dist_l1",
            "turnover",
            "max_weight",
            "herfindahl",
            "effective_n",
            "w_US",
            "w_Europe",
            "w_EM",
            "w_Japan",
            "aw_US",
            "aw_Europe",
            "aw_EM",
            "aw_Japan",
        ]
    ].sort_values(["selected_mu_europe", "architecture"], ascending=[False, True])

    return summary, snapshot


def make_plots(
    *,
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    config: ExperimentConfig,
) -> list[Path]:
    """Create the requested publication-style matplotlib figures."""

    output_paths: list[Path] = []
    feasible = results_df[results_df["status"].isin(FEASIBLE_STATUSES)].copy()

    output_paths.append(_plot_weight_sweep(feasible, config))
    output_paths.append(_plot_active_weight_sweep(feasible, config))
    output_paths.append(_plot_selected_point_bars(snapshot_df, config))
    output_paths.append(_plot_instability_comparison(summary_df, config))
    output_paths.append(_plot_governance_metrics(summary_df, config))
    return output_paths


def _plot_weight_sweep(results_df: pd.DataFrame, config: ExperimentConfig) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    plotted_assets = ("US", "Europe", "EM")

    for axis, architecture in zip(axes, ARCHITECTURE_ORDER):
        subset = results_df[results_df["architecture"] == architecture].sort_values("mu_europe", ascending=False)
        for asset in plotted_assets:
            axis.plot(
                subset["mu_europe"],
                subset[f"w_{asset}"],
                linewidth=2.2,
                label=asset,
                color=ASSET_COLORS[asset],
            )
        axis.set_title(subset["architecture_label"].iloc[0] if not subset.empty else architecture)
        axis.grid(alpha=0.25)
        axis.set_xlabel("Europe expected return")
        axis.set_xlim(config.mu_europe_grid.max(), config.mu_europe_grid.min())
        _percent_axis(axis, x=True, y=True)

    axes[0].set_ylabel("Portfolio weight")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle(
        "How Different Optimization Architectures Translate Return Signals into Portfolio Weights",
        fontsize=14,
    )
    output_path = config.output_dir / "plot_1_weight_sweep.png"
    _save_figure(fig, output_path)
    return output_path


def _plot_active_weight_sweep(results_df: pd.DataFrame, config: ExperimentConfig) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    plotted_assets = ("US", "Europe", "EM")

    for axis, architecture in zip(axes, ARCHITECTURE_ORDER):
        subset = results_df[results_df["architecture"] == architecture].sort_values("mu_europe", ascending=False)
        for asset in plotted_assets:
            axis.plot(
                subset["mu_europe"],
                subset[f"aw_{asset}"],
                linewidth=2.2,
                label=asset,
                color=ASSET_COLORS[asset],
            )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        axis.set_title(subset["architecture_label"].iloc[0] if not subset.empty else architecture)
        axis.grid(alpha=0.25)
        axis.set_xlabel("Europe expected return")
        axis.set_xlim(config.mu_europe_grid.max(), config.mu_europe_grid.min())
        _percent_axis(axis, x=True, y=True)

    axes[0].set_ylabel("Active weight")
    axes[-1].legend(frameon=False, loc="best")
    fig.suptitle("Active Weight Response Across Optimization Architectures", fontsize=14)
    output_path = config.output_dir / "plot_2_active_weight_sweep.png"
    _save_figure(fig, output_path)
    return output_path


def _plot_selected_point_bars(snapshot_df: pd.DataFrame, config: ExperimentConfig) -> Path:
    fig, axes = plt.subplots(1, len(config.selected_sweep_points), figsize=(18, 5), sharey=True)
    if len(config.selected_sweep_points) == 1:
        axes = [axes]

    assets = list(config.assets)
    width = 0.22
    x_positions = np.arange(len(assets))

    for axis, target_mu in zip(axes, config.selected_sweep_points):
        subset = snapshot_df[np.isclose(snapshot_df["selected_mu_europe"], target_mu)].copy()
        for offset, architecture in enumerate(ARCHITECTURE_ORDER):
            row = subset[subset["architecture"] == architecture]
            if row.empty:
                continue
            weights = row[[f"w_{asset}" for asset in assets]].iloc[0].to_numpy(dtype=float)
            axis.bar(
                x_positions + (offset - 1) * width,
                weights,
                width=width,
                label=architecture,
                color=ARCHITECTURE_COLORS[architecture],
                alpha=0.9,
            )

        axis.set_title(f"Europe ER = {target_mu:.1%}")
        axis.set_xticks(x_positions)
        axis.set_xticklabels(assets)
        axis.grid(axis="y", alpha=0.25)
        _percent_axis(axis, y=True)

    axes[0].set_ylabel("Final weight")
    axes[-1].legend(frameon=False, loc="best", title="Architecture")
    fig.suptitle("Selected Sweep Points: Final Weights Across Architectures", fontsize=14)
    output_path = config.output_dir / "plot_3_selected_point_bars.png"
    _save_figure(fig, output_path)
    return output_path


def _plot_instability_comparison(summary_df: pd.DataFrame, config: ExperimentConfig) -> Path:
    metrics = [
        ("mean_adjacent_l1_change", "Mean Adjacent L1"),
        ("mean_adjacent_l2_change", "Mean Adjacent L2"),
        ("max_jump", "Max Jump"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ordered = summary_df.sort_values("architecture")
    for axis, (column, title) in zip(axes, metrics):
        axis.bar(
            ordered["architecture"].astype(str),
            ordered[column],
            color=[ARCHITECTURE_COLORS[str(a)] for a in ordered["architecture"]],
            alpha=0.9,
        )
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        _percent_axis(axis, y=True)

    fig.suptitle("Instability Comparison by Architecture", fontsize=14)
    output_path = config.output_dir / "plot_4_instability_comparison.png"
    _save_figure(fig, output_path)
    return output_path


def _plot_governance_metrics(summary_df: pd.DataFrame, config: ExperimentConfig) -> Path:
    metrics = [
        ("average_l2_benchmark_distance", "Average L2 Benchmark Distance"),
        ("average_turnover", "Average L1 Deviation vs Benchmark"),
        ("average_herfindahl", "Average Herfindahl"),
        ("average_expected_return", "Average Expected Return"),
        ("average_portfolio_cvar", "Average Demeaned Portfolio CVaR"),
        ("average_te_cvar", "Average Demeaned TE-CVaR"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    ordered = summary_df.sort_values("architecture")

    for axis, (column, title) in zip(axes.flatten(), metrics):
        axis.bar(
            ordered["architecture"].astype(str),
            ordered[column],
            color=[ARCHITECTURE_COLORS[str(a)] for a in ordered["architecture"]],
            alpha=0.9,
        )
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
        if "Herfindahl" not in title:
            _percent_axis(axis, y=True)

    fig.suptitle("Governance and Practical Metrics by Architecture", fontsize=14)
    output_path = config.output_dir / "plot_5_governance_metrics.png"
    _save_figure(fig, output_path)
    return output_path


def plot_te_cvar_sensitivity_sweep(
    *,
    results_df: pd.DataFrame,
    config: ExperimentConfig,
    te_cvar_values: tuple[float, ...],
) -> Path:
    """Plot B/C weight sweeps for multiple TE-CVaR limits on one figure."""

    feasible = results_df[results_df["status"].isin(FEASIBLE_STATUSES)].copy()
    plotted_assets = ("US", "Europe", "EM")
    architectures = ("B", "C")
    style_specs = {
        te_value: spec
        for te_value, spec in zip(
            te_cvar_values,
            (
                {"linestyle": "-", "linewidth": 2.4, "marker": None, "zorder": 2},
                {"linestyle": (0, (7, 3)), "linewidth": 2.4, "marker": None, "zorder": 3},
                {
                    "linestyle": (0, (1.0, 2.0)),
                    "linewidth": 2.8,
                    "marker": "o",
                    "markersize": 3.2,
                    "markevery": 4,
                    "markerfacecolor": "white",
                    "markeredgewidth": 0.8,
                    "zorder": 4,
                },
            ),
            strict=False,
        )
    }

    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
    for axis, architecture in zip(axes, architectures):
        architecture_subset = feasible[feasible["architecture"] == architecture].copy()
        for asset in plotted_assets:
            for te_value in te_cvar_values:
                subset = architecture_subset[
                    np.isclose(architecture_subset["te_cvar_limit"], te_value)
                ].sort_values("mu_europe", ascending=False)
                if subset.empty:
                    continue
                axis.plot(
                    subset["mu_europe"],
                    subset[f"w_{asset}"],
                    color=ASSET_COLORS[asset],
                    **style_specs[te_value],
                )

        axis.set_title(
            architecture_subset["architecture_label"].iloc[0]
            if not architecture_subset.empty
            else f"Architecture {architecture}"
        )
        axis.grid(alpha=0.25)
        axis.set_xlabel("Europe expected return")
        axis.set_xlim(config.mu_europe_grid.max(), config.mu_europe_grid.min())
        _percent_axis(axis, x=True, y=True)

    axes[0].set_ylabel("Portfolio weight")

    asset_handles = [
        Line2D([0], [0], color=ASSET_COLORS[asset], linewidth=2.4, label=asset)
        for asset in plotted_assets
    ]
    te_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=style_specs[te_value]["linewidth"],
            linestyle=style_specs[te_value]["linestyle"],
            marker=style_specs[te_value].get("marker"),
            markersize=style_specs[te_value].get("markersize"),
            markerfacecolor=style_specs[te_value].get("markerfacecolor"),
            label=f"TE-CVaR {te_value:.0%}",
        )
        for te_value in te_cvar_values
    ]
    asset_legend = axes[-1].legend(
        handles=asset_handles,
        title="Asset",
        frameon=False,
        loc="upper left",
    )
    axes[-1].add_artist(asset_legend)
    axes[-1].legend(
        handles=te_handles,
        title="TE-CVaR",
        frameon=False,
        loc="lower left",
    )

    fig.suptitle(
        "TE-CVaR Sensitivity of Weight Sweeps for Architecture B and C",
        fontsize=14,
    )
    output_path = config.output_dir / "plot_6_te_cvar_sensitivity_bc.png"
    _save_figure(fig, output_path)
    return output_path


def export_outputs(
    *,
    raw_results: pd.DataFrame,
    summary_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    config: ExperimentConfig,
    scenario_metadata: dict[str, Any],
    plot_paths: list[Path],
) -> dict[str, Path]:
    """Export CSV, JSON, and plot artifacts to disk."""

    config.output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "raw_results": config.output_dir / "raw_sweep_results.csv",
        "summary": config.output_dir / "architecture_summary.csv",
        "snapshot": config.output_dir / "selected_snapshots.csv",
        "metadata": config.output_dir / "experiment_metadata.json",
    }

    raw_results.to_csv(output_paths["raw_results"], index=False)
    summary_df.to_csv(output_paths["summary"], index=False)
    snapshot_df.to_csv(output_paths["snapshot"], index=False)

    metadata_payload = {
        "assumptions": [
            "Expected-return maximization always uses the explicit objective mean vector mu rather than the scenario matrix itself.",
            "Primary portfolio CVaR and TE-CVaR constraints are evaluated on demeaned scenarios. If source scenarios already contain a mean, the current objective mean vector is subtracted before CVaR evaluation.",
            "Secondary reporting fields with the _full suffix show the corresponding full-return CVaR metrics after mean drift is included again.",
            "If an Excel scenario workbook is available, its rows are treated as scenarios and equal scenario weights are used unless a probability column is present.",
            "Synthetic scenarios are only used as a fallback when no external scenario workbook or matrix is supplied.",
            "By default, the portfolio CVaR limit at each sweep point equals the demeaned benchmark portfolio CVaR multiplied by the configured CVaR factor.",
            "Absolute portfolio CVaR mode remains available as a fallback. If its requested limit is tighter than the benchmark's own demeaned scenario CVaR, the effective portfolio CVaR limit can be lifted just enough to keep the benchmark feasible.",
            "When turnover constraints are enabled, every sweep point in every architecture is constrained against the benchmark portfolio rather than against the previous feasible portfolio.",
            "The reported turnover metric is the L1 deviation from the benchmark portfolio, so it behaves like a max deviation or active-share-style constraint relative to the benchmark.",
            "The composite instability score is defined as adjacent L1 change + adjacent L2 change + adjacent max jump.",
        ],
        "config": to_serializable_dict(config),
        "scenario_metadata": scenario_metadata,
        "plots": [str(path) for path in plot_paths],
    }
    with output_paths["metadata"].open("w", encoding="utf-8") as handle:
        json.dump(metadata_payload, handle, indent=2)

    return output_paths
