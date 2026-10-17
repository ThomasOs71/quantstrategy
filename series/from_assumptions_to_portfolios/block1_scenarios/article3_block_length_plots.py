"""Supplementary Article-3 exhibits from exported tables; no data downloads."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from series.from_assumptions_to_portfolios.block1_scenarios.article3_diagnostics import SELECTED_CORRELATION_PAIRS
from series.from_assumptions_to_portfolios.block1_scenarios.article3_plots import (
    LABELS, _footer, _save_pair, figure05_terminal_outcomes,
)
from series.from_assumptions_to_portfolios.flagship_plot_style import (
    CHARCOAL, COBALT, MUTED_GREY, TEAL, create_figure, style_axis,
)


def _header(figure, title: str, subtitle: str):
    figure.text(.06, .945, title, fontsize=30, fontweight="semibold", color=CHARCOAL)
    figure.text(.06, .908, subtitle, fontsize=18, color=MUTED_GREY)


def _length_axis(axis, lengths, *, ylabel: str):
    style_axis(axis)
    axis.set_xscale("log", base=2)
    axis.set_xticks(lengths, [str(int(value)) for value in lengths])
    axis.tick_params(labelsize=14)
    axis.set_xlabel("Block length L (weeks)", fontsize=15)
    axis.set_ylabel(ylabel, fontsize=15)
    axis.axvline(4, color=MUTED_GREY, linewidth=1, linestyle=":", zorder=0)


def _tradeoff(tables, parent, output):
    data = tables["analytic_preservation"].copy()
    wide = data.pivot(index="L", columns="metric", values="value").reset_index()
    structural = pd.read_csv(parent / "tables" / "block_length_tradeoff.csv")
    structural = structural.loc[structural.experiment.str.startswith("synchronized")]
    structural = structural.groupby("L", as_index=False)[["used_blocks_per_path", "candidate_block_starts", "within_fraction"]].first()
    edge = pd.read_csv(parent / "tables" / "edge_weighting_summary.csv")
    edge = edge.loc[(edge.metric == "total_variation_to_uniform") & edge.experiment.str.startswith("synchronized")]
    edge = edge.groupby("L", as_index=False).value.first().rename(columns={"value": "edge_total_variation"})
    wide = wide.merge(structural, on="L").merge(edge, on="L").sort_values("L")
    figure = create_figure(size=(17, 15))
    _header(figure, "Longer blocks preserve more — but change the sampler", "Exact population diagnostics of fixed, overlapping, non-circular MBB; L=4 is the reference, not a winner")
    axes = [figure.add_axes(position) for position in ([.09, .58, .37, .25], [.58, .58, .35, .25], [.09, .23, .37, .24], [.58, .23, .35, .24])]
    for transform, color, label in (("raw", COBALT, "Raw returns"), ("squared_demeaned", TEAL, "Squared demeaned returns")):
        axes[0].plot(wide.L, wide[f"{transform}_acf_rmse_non_cash11_lags1_13"], "o-", color=color, label=label, linewidth=2)
    _length_axis(axes[0], wide.L, ylabel="ACF RMSE, lags 1–13 (ex cash)")
    axes[0].legend(frameon=False, fontsize=12)
    axes[1].plot(wide.L, wide.weekly_correlation_mae_all12, "o-", color=COBALT, linewidth=2)
    _length_axis(axes[1], wide.L, ylabel="One-week correlation MAE (all 12)")
    axes[2].plot(wide.L, wide.used_blocks_per_path, "o-", color=CHARCOAL, linewidth=2)
    _length_axis(axes[2], wide.L, ylabel="Independent block draws per 260w path")
    axes[3].plot(wide.L, wide.edge_total_variation, "o-", color=TEAL, linewidth=2)
    _length_axis(axes[3], wide.L, ylabel="Source-weight distance to uniform (TV)")
    _footer(figure, note="Long-block costs shown here are recombination and edge weighting; this is not an empirical universal MSE-optimality curve.")
    return wide, _save_pair(figure, output, "supp01_block_length_tradeoff")


def _selectors(tables, output):
    data = tables["automatic_block_lengths"]
    data = data.loc[(data.family == "non_cash") & data["transform"].isin(["raw", "squared_demeaned"])].copy()
    keys = data.loc[data["transform"] == "raw", "driver_or_pair"].tolist()
    figure = create_figure(size=(18, 14))
    _header(figure, "The statistical target changes the suggested length", "Corrected Politis–White plug-in estimates for circular bootstrap; not an MBB or drawdown calibration")
    axis = figure.add_axes([.31, .30, .61, .54])
    positions = np.arange(len(keys))
    raw = data.loc[data["transform"] == "raw"].set_index("driver_or_pair").reindex(keys).circular_length.to_numpy()
    squared = data.loc[data["transform"] == "squared_demeaned"].set_index("driver_or_pair").reindex(keys).circular_length.to_numpy()
    axis.hlines(positions, raw, squared, color=MUTED_GREY, alpha=.45, linewidth=2)
    axis.scatter(raw, positions, color=COBALT, s=90, label="Mean of raw returns")
    axis.scatter(squared, positions, color=TEAL, marker="s", s=90, label="Mean of squared demeaned returns")
    axis.axvline(4, linestyle=":", color=CHARCOAL, label="Set-A reference L=4")
    axis.axvline(int(data.cube_root_floor.iloc[0]), linestyle="--", color=MUTED_GREY, label="floor(n^(1/3)) heuristic")
    style_axis(axis)
    axis.set_xscale("log", base=2)
    axis.set_xticks([1, 2, 4, 8, 16, 32, 64], ["1", "2", "4", "8", "16", "32", "64"])
    axis.set_yticks(positions, [LABELS[key] for key in keys])
    axis.invert_yaxis()
    axis.tick_params(labelsize=16)
    axis.set_xlabel("Continuous estimated block length (weeks)", fontsize=17)
    axis.legend(frameon=False, fontsize=12, loc="upper left", bbox_to_anchor=(0, -.16), ncol=2)
    _footer(figure, note="Cash is reported separately; stationary-bootstrap estimates and correlation-influence targets are also exported.")
    return data, _save_pair(figure, output, "supp02_target_specific_lengths")


def _rolling(tables, output):
    frame = tables["rolling_diagnostics"]
    data = frame.loc[frame.diagnostic == "rolling_correlation"].copy()
    figure = create_figure(size=(17, 17))
    _header(figure, "Block length changes local correlation regimes", "104-week within-path Pearson correlations; bands are window quantiles, not confidence intervals")
    for position, (left, right) in enumerate(SELECTED_CORRELATION_PAIRS):
        axis = figure.add_axes([.13, .69 - position * .235, .76, .16])
        subset = data.loc[data.asset_or_pair == f"{left}|{right}"]
        grid = subset.loc[subset.L > 0].groupby(["L", "metric"]).value.mean().unstack().sort_index()
        historical = subset.loc[subset.L == 0].set_index("metric").value
        axis.fill_between(grid.index, grid.q05, grid.q95, color=TEAL, alpha=.2, label="Scenario windows q05–q95")
        axis.plot(grid.index, grid.q50, "o-", color=TEAL, label="Scenario-window median")
        for metric, linestyle in (("q05", ":"), ("q50", "--"), ("q95", ":")):
            axis.axhline(historical[metric], color=CHARCOAL, linestyle=linestyle, linewidth=1.3,
                         label="Historical windows q05/50/95" if metric == "q50" else None)
        _length_axis(axis, grid.index, ylabel="Correlation")
        axis.set_title(f"{LABELS[left]} / {LABELS[right]}", fontsize=17, loc="left", pad=10)
        if position == 0:
            axis.legend(frameon=False, fontsize=11, loc="best")
    _footer(figure, note="Mean of replicate-specific quantiles; historical 104w windows each enter once. Overlapping windows are not independent.")
    return data, _save_pair(figure, output, "supp03_rolling_correlations")


def _terminal(tables, output):
    data = tables["terminal_correlations"]
    data = data.loc[(data.L > 0) & (data.return_type == "simple_compounded")].copy()
    figure = create_figure(size=(17, 17))
    _header(figure, "One-week and terminal correlations are different", "Across-scenario Pearson correlations of compounded simple returns, conditional on the frozen sample")
    colors = {52: COBALT, 156: TEAL, 260: CHARCOAL}
    for position, (left, right) in enumerate(SELECTED_CORRELATION_PAIRS):
        axis = figure.add_axes([.13, .69 - position * .235, .76, .16])
        subset = data.loc[(data.driver_i == left) & (data.driver_j == right)]
        # These selected pairs follow the canonical driver order.
        for horizon, color in colors.items():
            rows = subset.loc[subset.horizon_steps == horizon].groupby("L").correlation.mean().sort_index()
            axis.plot(rows.index, rows, "o-", color=color, label=f"{horizon} weeks", linewidth=2)
        _length_axis(axis, sorted(subset.L.unique()), ylabel="Terminal correlation")
        axis.set_title(f"{LABELS[left]} / {LABELS[right]}", fontsize=17, loc="left", pad=10)
        if position == 0:
            axis.legend(frameon=False, fontsize=12, ncol=3)
    _footer(figure, note="Five grid replicates in a full run. Log-sum terminal correlations and historical overlapping-window comparators are exported too.")
    return data, _save_pair(figure, output, "supp04_terminal_correlations")


def _resolution(tables, output):
    assets = ["global_dm_ex_emu", "euro_govt_bond_7_10", "euro_high_yield", "gold"]
    frame = tables["resolution_summary"]
    data = frame.loc[(frame.L == 4) & frame.asset.isin(assets) & frame.metric.isin(["weekly_mdd_q95", "coarse_mdd_q95"])].copy()
    figure = create_figure(size=(17, 13))
    _header(figure, "Coarse endpoints hide within-period drawdowns", "Same L=4 paths and exact same terminal wealth; mean replicate-specific 95th-percentile drawdown")
    axis = figure.add_axes([.13, .27, .76, .56])
    positions = np.arange(len(assets))
    labels = [("four_week_endpoints", "weekly_mdd_q95", "Weekly", CHARCOAL),
              ("four_week_endpoints", "coarse_mdd_q95", "Every 4 weeks (not calendar monthly)", COBALT),
              ("synthetic_last_friday_per_month", "coarse_mdd_q95", "Last Friday / synthetic month", TEAL)]
    for offset, (grid, metric, label, color) in zip([-.24, 0, .24], labels):
        rows = data.loc[(data.grid == grid) & (data.metric == metric)].groupby("asset").value.mean().reindex(assets)
        axis.bar(positions + offset, rows, width=.22, label=label, color=color)
    style_axis(axis, percent=True)
    axis.set_xticks(positions, ["Global DM ex-EMU", "Euro govt 7–10y", "Euro HY", "Gold"], rotation=10)
    axis.set_ylabel("95th-percentile maximum drawdown", fontsize=17)
    axis.legend(frameon=False, fontsize=13)
    _footer(figure, note="Controlled coarsening, not a rebuilt monthly bootstrap or daily calendar-month-end dataset. Weekly retains more path detail.")
    return data, _save_pair(figure, output, "supp05_time_resolution")


def _uncertainty(tables, output):
    frame = tables["source_uncertainty_intervals"]
    metric = "squared_demeaned_acf_rmse_non_cash11_lags1_13"
    data = frame.loc[(frame.metric == metric) & (frame.comparator_L > 0)].copy()
    figure = create_figure(size=(17, 12))
    _header(figure, "Source-sample uncertainty is not simulation noise", "Paired squared-ACF RMSE differences: L=4 minus comparator; exploratory 95% percentile intervals")
    axis = figure.add_axes([.13, .26, .76, .58])
    comparisons = sorted(data.comparator_L.unique())
    outer_lengths = sorted(data.outer_mean_block_length.unique())
    for shift, outer, color in zip(np.linspace(-.22, .22, len(outer_lengths)), outer_lengths, [COBALT, TEAL, CHARCOAL]):
        rows = data.loc[data.outer_mean_block_length == outer].set_index("comparator_L").reindex(comparisons)
        x = np.arange(len(comparisons)) + shift
        axis.vlines(x, rows.ci_lower, rows.ci_upper, color=color, linewidth=3)
        axis.scatter(x, rows.point_estimate, color=color, s=70, label=f"Outer mean block {outer}w")
    style_axis(axis, zero_line=True)
    axis.set_xticks(np.arange(len(comparisons)), [f"L=4 minus L={int(value)}" for value in comparisons])
    axis.set_ylabel("Difference in squared-return ACF RMSE", fontsize=17)
    axis.legend(frameon=False, fontsize=13, loc="upper left")
    _footer(figure, note="Negative favors L=4 on this preservation metric only. Outer bootstrap assumes suitable stationarity; intervals are not simultaneous.")
    return data, _save_pair(figure, output, "supp06_source_uncertainty")


def write_research_figures(tables: dict[str, pd.DataFrame], *, parent: Path, output_dir: Path) -> dict[str, str]:
    """Render six supplemental exhibits and the corrected original Figure 5."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_data = output_dir / "plot_data"
    plot_data.mkdir(exist_ok=True)
    captions = {}
    renderers = [(_tradeoff, "Trade-off: temporal preservation, weekly correlation edge distortion, independent block draws, and source weights."),
                 (_selectors, "Target-specific corrected selector estimates; do not transfer CBB recommendations as exact fixed-MBB optima."),
                 (_rolling, "Local correlations in 104-week windows; distribution quantiles, not confidence intervals."),
                 (_terminal, "Across-path terminal simple-return correlations across block lengths and three horizons."),
                 (_resolution, "Paired time-grid coarsening of the same L=4 paths; terminal wealth is unchanged, drawdowns can be hidden."),
                 (_uncertainty, "Historical sample uncertainty: paired preservation-error contrasts under three outer stationary-bootstrap lengths.")]
    for renderer, caption in renderers:
        data, paths = renderer(tables, parent, output_dir) if renderer is _tradeoff else renderer(tables, output_dir)
        data.to_csv(plot_data / f"{paths[0].stem}.csv", index=False, float_format="%.17g")
        captions[paths[0].stem] = caption
    terminal = pd.read_csv(parent / "tables" / "terminal_return_summary.csv")
    data, paths = figure05_terminal_outcomes(terminal, output_dir=output_dir)
    data.to_csv(plot_data / "fig05_terminal_outcomes.csv", index=False, float_format="%.17g")
    captions[paths[0].stem] = "CORRECTED: historical and 10,000-path MBB-L=4 reference 5/50/95% annualized terminal-return quantiles. New end values, not validated future uncertainty."
    return captions
