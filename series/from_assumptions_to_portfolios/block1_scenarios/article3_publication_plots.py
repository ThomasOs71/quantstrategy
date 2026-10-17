"""Article-numbered exhibits, rendered from the exported publication tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from series.from_assumptions_to_portfolios.block1_scenarios.article3_plots import LABELS
from series.from_assumptions_to_portfolios.block1_scenarios.article3_publication import ASSETS, EXHIBITS
from series.from_assumptions_to_portfolios.flagship_plot_style import (
    CHARCOAL, COBALT, LIGHT_GREY, MUTED_GREY, OFF_WHITE, TEAL, style_axis,
)


CAPTIONS = {
    "figure_a_workflow": "Figure A. Historical weekly returns form synchronized blocks, then asset paths, risk diagnostics and later portfolio analysis. Conceptual diagram; no portfolio optimizer is implemented here.",
    "fig01_tail_sequence": "Figure 1. Conditional probability of a further joint-tail week at lags 1 and 4. Joint tails mean at least two of ten noncash assets below fixed historical 5% thresholds. Simulation values average five runs of 3,000 paths; historical counts use valid lag pairs.",
    "fig02_block_lineage": "Figure 2. First twelve weeks of recorded reference scenario zero. The three illustrated assets share source weeks; all twelve drivers were sampled together. Dashed lines mark newly joined four-week blocks.",
    "fig03_time_resolution": "Figure 3. Weekly and four-week endpoint observations of the SAME L=4 paths. Bars average five replicate-specific 95th-percentile maximum drawdowns. Four-week endpoints are not calendar months; terminal wealth is unchanged.",
    "fig04_volatility_memory": "Figure 4. Exact expected-MBB squared-demeaned-return ACF RMSE at lags 1–13 over eleven noncash drivers, including FX. Lower means closer historical preservation, not better forecasts.",
    "fig05_drawdown_sensitivity": "Figure 5. Mean of five run-specific 95th-percentile maximum drawdowns, with 3,000 five-year paths per run. Longer blocks are not uniformly more conservative. This is NOT the older research Figure 5 showing terminal returns.",
    "fig06_wealth_paths": "Figure 6 REPLACEMENT. One hundred distinct reference paths with archived selection IDs; the default selection uses a new documented PCG64 seed. All start at one; the dark line is the pointwise median of ALL 10,000 reference paths. Original published path IDs were not archived, so this selection is not claimed to match the original image.",
    "table01_target_specific_lengths": "Table 1. Corrected Politis–White circular-bootstrap suggestions in weeks, rounded to one decimal. Scalar mean and squared-deviation targets; neither fixed-MBB parameters nor drawdown optima.",
    "table_a1_correlations": "Table A1. Global DM ex-EMU versus Euro government bonds 7–10y. Means of five simulation summaries: pooled weekly correlation, median 104-week rolling correlation and across-path simple compounded 260-week outcome correlation. Display rounded to three decimals.",
}


def _canvas(title: str, subtitle: str):
    # Offline exports must not depend on a working Tk/Qt installation or change
    # the backend of an existing interactive Python session.
    figure = Figure(figsize=(16, 10), dpi=100, facecolor=OFF_WHITE)
    FigureCanvasAgg(figure)
    figure.text(.07, .94, title, fontsize=28, color=CHARCOAL, fontweight="bold")
    figure.text(.07, .885, subtitle, fontsize=17, color=MUTED_GREY)
    return figure


def _footer(figure, note: str):
    figure.text(.07, .115, note, fontsize=12, color=MUTED_GREY)
    figure.text(.07, .075, "Source: QuantStrategy calculations | Jan 2011–Dec 2025 | Weekly EUR log returns",
                fontsize=13, color=MUTED_GREY)
    figure.text(.07, .045, "Historical preservation and model sensitivity; not validated future probabilities.",
                fontsize=12, color=MUTED_GREY)


def _axis(figure, *, percent=False):
    axis = figure.add_axes([.12, .24, .80, .56])
    style_axis(axis, percent=percent)
    axis.tick_params(labelsize=15)
    return axis


def _workflow(data):
    figure = _canvas("Figure A | From historical weeks to scenario paths",
                     "Block length governs temporal memory; portfolio allocation is a later step")
    axis = figure.add_axes([.04, .28, .92, .48])
    axis.set(xlim=(0, 5), ylim=(0, 1))
    axis.axis("off")
    for row in data.itertuples():
        x = row.order + .06
        box = FancyBboxPatch((x, .25), .86, .48, boxstyle="round,pad=0.02",
                             facecolor=OFF_WHITE, edgecolor=COBALT if row.order < 4 else MUTED_GREY,
                             linewidth=2, linestyle="-" if row.order < 4 else "--")
        axis.add_patch(box)
        axis.text(x + .43, .49, row.label, ha="center", va="center", fontsize=15, color=CHARCOAL)
        if row.order < 4:
            axis.annotate("", xy=(x + .98, .49), xytext=(x + .87, .49),
                          arrowprops={"arrowstyle": "->", "color": CHARCOAL, "lw": 1.8})
    _footer(figure, "The final portfolio step also needs allocation weights and a rebalancing rule.")
    return figure


def _tail(data):
    figure = _canvas("Figure 1 | Short blocks recover immediate stress clustering",
                     "Probability of another joint-tail week, conditional on a joint-tail week now")
    axis = _axis(figure, percent=True)
    for shift, series, color in ((-.25, "Historical", CHARCOAL), (0, "L=1", COBALT), (.25, "L=4", TEAL)):
        rows = data.loc[data.series == series].set_index("lag").reindex([1, 4])
        bars = axis.bar(np.arange(2) + shift, rows.probability, width=.23, color=color, label=series)
        axis.bar_label(bars, labels=[f"{p:.2%}" for p in rows.probability], padding=5, fontsize=14)
    axis.set(xticks=[0, 1], xticklabels=["One week later", "Four weeks later"], ylim=(0, .34),
             ylabel="Conditional probability")
    axis.legend(frameon=False, fontsize=15, ncol=3, loc="upper right")
    _footer(figure, "Ten noncash assets; fixed historical 5% thresholds. Scenario values average five separate runs.")
    return figure


def _lineage(data):
    figure = _canvas("Figure 2 | Historical blocks form one new path",
                     "First 12 weeks of saved scenario zero; all assets use the same source weeks")
    for position, asset in enumerate((ASSETS[0], ASSETS[1], "gold")):
        axis = figure.add_axes([.15, .65 - position * .18, .76, .135])
        rows = data.loc[data.asset == asset].sort_values("simulation_step")
        for block, color in zip(range(3), (COBALT, TEAL, CHARCOAL)):
            selected = rows.loc[rows.block_id == block]
            axis.plot(selected.simulation_step + 1, selected.weekly_log_return, "o-", color=color, lw=2)
        for boundary in (4.5, 8.5):
            axis.axvline(boundary, color=MUTED_GREY, linestyle="--", lw=1)
        style_axis(axis, percent=True, zero_line=True)
        axis.tick_params(labelsize=12)
        axis.set_title(LABELS[asset], loc="left", fontsize=13, pad=8)
        axis.set_xticks(range(1, 13))
        if position == 2:
            axis.set_xlabel("Weekly step in the newly joined path", fontsize=14)
    _footer(figure, "Within each block the sequence is historical; connections between blocks are newly constructed.")
    return figure


def _resolution(data):
    figure = _canvas("Figure 3 | Looking less often hides part of the drawdown",
                     "Same L=4 paths and terminal wealth; mean run-specific 95th-percentile maximum drawdown")
    axis = _axis(figure, percent=True)
    for offset, metric, label, color in ((-.18, "weekly_mdd_q95", "Weekly observations", CHARCOAL),
                                         (.18, "coarse_mdd_q95", "Four-week endpoints", TEAL)):
        rows = data.loc[data.metric == metric].set_index("asset").reindex(ASSETS)
        bars = axis.bar(np.arange(4) + offset, rows.value, width=.33, color=color, label=label)
        axis.bar_label(bars, labels=[f"{x:.2%}" for x in rows.value], padding=5, fontsize=12)
    axis.set_xticks(range(4), ["Global DM ex-EMU", "Euro govt 7–10y", "Euro HY", "Gold"])
    axis.set_ylabel("95th-percentile maximum drawdown", fontsize=15)
    axis.set_ylim(0, float(data.value.max()) * 1.25)
    axis.legend(frameon=False, fontsize=14)
    _footer(figure, "Four-week endpoints are not calendar-month data. Weekly observations also miss within-week movements.")
    return figure


def _memory(data):
    figure = _canvas("Figure 4 | Longer blocks retain more volatility memory",
                     "Exact expected squared-return ACF error over lags 1–13; eleven noncash drivers including FX")
    axis = _axis(figure)
    axis.plot(data.L, data.value, "o-", color=TEAL, lw=2.5)
    axis.axvline(4, color=MUTED_GREY, linestyle=":")
    axis.set_xscale("log", base=2)
    axis.set_xticks(data.L, [str(int(l)) for l in data.L])
    axis.set(xlabel="Block length (weeks)", ylabel="ACF RMSE (lower = closer to history)",
             ylim=(0, float(data.value.max()) * 1.20))
    for row in data.itertuples():
        axis.annotate(f"{row.value:.3f}", (row.L, row.value), xytext=(0, 10),
                      textcoords="offset points", ha="center", fontsize=12)
    _footer(figure, "L=4 is a short-block reference, not the winner of a universal bias–variance or forecasting test.")
    return figure


def _drawdown(data):
    figure = _canvas("Figure 5 | Longer blocks do not mean higher risk",
                     "Mean of five separate 95th-percentile maximum drawdowns; 3,000 five-year paths per run")
    axis = _axis(figure, percent=True)
    for asset, color in zip(ASSETS[:2], (COBALT, TEAL)):
        rows = data.loc[data.asset == asset].sort_values("L")
        axis.plot(rows.L, rows.value, "o-", color=color, lw=2.5, label=LABELS[asset])
    lengths = sorted(data.L.unique())
    axis.set_xscale("log", base=2)
    axis.set_xticks(lengths, [str(int(l)) for l in lengths])
    axis.set(xlabel="Block length (weeks)", ylabel="95th-percentile maximum drawdown")
    axis.legend(frameon=False, fontsize=14)
    _footer(figure, "Averaging five quantiles is not pooling all paths first. This is method sensitivity, not a future-risk validation.")
    return figure


def _wealth(data):
    figure = _canvas("Figure 6 | Complete five-year wealth paths",
                     "Reproducible replacement selection: 100 saved Global DM ex-EMU paths; initial wealth = 1")
    axis = _axis(figure)
    shown = data.loc[data.series == "selected_path"].pivot(index="step", columns="scenario_id", values="wealth")
    axis.plot(shown.index / 52, shown.to_numpy(), color=COBALT, alpha=.16, lw=.8)
    median = data.loc[data.series == "reference_median"].sort_values("step")
    line, = axis.plot(median.step / 52, median.wealth, color=CHARCOAL, lw=3)
    axis.legend(handles=[Line2D([0], [0], color=COBALT, lw=1.5), line],
                labels=["100 saved paths", "Pointwise median of all 10,000 paths"],
                frameon=False, fontsize=13, loc="upper left")
    axis.set(xlabel="Model years (52 weekly steps)", ylabel="Wealth from initial value 1", xlim=(0, 5))
    _footer(figure, "Original 100-path IDs were not archived. Selection IDs and method are exported; these are not forecast paths.")
    return figure


def _table(data, *, correlations=False):
    if correlations:
        title = "Table A1 | Correlation depends on what we measure"
        subtitle = "Global DM ex-EMU versus Euro government bonds 7–10y; mean of five simulation summaries"
        headers = ["Block length", "Same-week\ncorrelation", "Median 104-week\nrolling correlation", "Five-year\noutcome correlation"]
        cells = [[f"{int(row.L)} {'week' if row.L == 1 else 'weeks'}", f"{row.same_week_correlation:.3f}",
                  f"{row.median_104_week_rolling_correlation:.3f}",
                  f"{row.five_year_outcome_correlation:.3f}"] for row in data.itertuples()]
        widths = [.18, .23, .31, .28]
        note = "Final column: Pearson correlation of simple compounded 260-week returns across paths."
    else:
        title = "Table 1 | Different targets suggest different block lengths"
        subtitle = "Corrected Politis–White circular-bootstrap suggestions; continuous estimates displayed to one decimal"
        headers = ["Asset", "Preserving\naverage return", "Preserving\nvolatility clustering"]
        cells = [[LABELS[row.asset], f"{row.average_return_length_weeks:.1f} weeks",
                  f"{row.squared_deviation_length_weeks:.1f} weeks"] for row in data.itertuples()]
        widths = [.44, .28, .28]
        note = "Scalar mean and squared-deviation targets; not calibrated MBB settings or drawdown optima."
    figure = _canvas(title, subtitle)
    axis = figure.add_axes([.07, .25, .86, .54])
    axis.axis("off")
    table = axis.table(cellText=cells, colLabels=headers, colWidths=widths,
                       cellLoc="left", loc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(17 if correlations else 16)
    for (row, _), cell in table.get_celld().items():
        cell.set_linewidth(.5)
        cell.set_edgecolor(OFF_WHITE)
        cell.set_facecolor(CHARCOAL if row == 0 else (OFF_WHITE if row % 2 == 0 else LIGHT_GREY))
        cell.get_text().set_color("white" if row == 0 else CHARCOAL)
    _footer(figure, note)
    return figure


def render_publication(tables: dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Render all seven article figures and two table images, with fixed SVG IDs."""
    if set(tables) != set(EXHIBITS):
        raise ValueError("The publication exporter requires Figure A, Figures 1–6 and both tables")
    output_dir.mkdir(parents=True, exist_ok=True)
    renderers = [_workflow, _tail, _lineage, _resolution, _memory, _drawdown, _wealth,
                 _table, lambda data: _table(data, correlations=True)]
    with mpl.rc_context({"font.family": "DejaVu Sans", "svg.hashsalt": "article3-publication-v1"}):
        for stem, renderer in zip(EXHIBITS, renderers):
            figure = renderer(tables[stem])
            figure.savefig(output_dir / f"{stem}.png", dpi=110, facecolor=figure.get_facecolor())
            figure.savefig(output_dir / f"{stem}.svg", facecolor=figure.get_facecolor(), metadata={"Date": None})
            figure.clear()
