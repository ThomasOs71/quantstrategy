"""Article-3 figures rendered only from exported diagnostic tables."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from series.from_assumptions_to_portfolios.flagship_plot_style import (
    CHARCOAL,
    COBALT,
    LIGHT_GREY,
    MUTED_GREY,
    PALE_BLUE,
    TEAL,
    add_empirical_footer,
    add_title,
    create_figure,
    style_axis,
)


LABELS = {
    "euro_equities": "Euro equities",
    "global_dm_ex_emu": "Global DM ex-EMU",
    "em_equities": "EM equities",
    "euro_govt_bond_7_10": "Euro government bonds 7–10y",
    "euro_ig_credit": "Euro IG credit",
    "euro_high_yield": "Euro high yield",
    "global_govt_bond_eur_hedged": "Global government bonds (EUR hedged)",
    "em_hc_bond_eur_hedged": "EM hard-currency bonds (EUR hedged)",
    "gold": "Gold",
    "commodities": "Commodities",
    "cash": "Cash",
    "fx_eurusd": "EUR/USD",
}


def _save_pair(figure, output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    svg = output_dir / f"{stem}.svg"
    figure.savefig(png, dpi=100, facecolor=figure.get_facecolor(), transparent=False)
    figure.savefig(svg, facecolor=figure.get_facecolor(), transparent=False)
    plt.close(figure)
    return png, svg


def _footer(figure, *, note: str) -> None:
    add_empirical_footer(
        figure,
        sample="Jan 2011–Dec 2025",
        frequency_label="Weekly (W-FRI)",
        return_description="EUR weekly log returns; paths are historical-block resamples",
        note=note,
    )


def figure01_block_lineage(
    panel: pd.DataFrame,
    source_indices: np.ndarray,
    used_block_lengths: Sequence[int],
    *,
    output_dir: Path,
) -> tuple[pd.DataFrame, tuple[Path, Path]]:
    """Illustrate actual first-path block lineage for three representative assets."""
    selected = ("global_dm_ex_emu", "euro_govt_bond_7_10", "gold")
    first_steps = 12
    indices = np.asarray(source_indices)[0, :first_steps]
    block_ids = np.repeat(np.arange(len(used_block_lengths)), used_block_lengths)[:first_steps]
    records = []
    for step, source_idx in enumerate(indices):
        for asset in selected:
            records.append({"scenario_id": 0, "simulation_step": step, "block_id": int(block_ids[step]),
                            "source_idx": int(source_idx), "source_date": panel.index[source_idx].date().isoformat(),
                            "asset": asset, "weekly_log_return": panel.iloc[source_idx][asset]})
    data = pd.DataFrame.from_records(records)
    figure = create_figure(size=(16, 12))
    add_title(figure, "Historical blocks build one new path", "First 12 weeks of scenario 0; three assets shown from a synchronized 12-driver draw")
    axes = figure.subplots(3, 1)
    for axis, asset in zip(axes, selected):
        subset = data.loc[data["asset"] == asset]
        for block_id, block in subset.groupby("block_id", sort=True):
            axis.plot(block["simulation_step"], block["weekly_log_return"], marker="o", linewidth=2.5,
                      color=(COBALT, TEAL, PALE_BLUE)[int(block_id) % 3], label=f"Block {int(block_id) + 1}")
        for boundary in np.flatnonzero(np.diff(subset["block_id"].to_numpy())) + 1:
            axis.axvline(boundary - .5, color=MUTED_GREY, linestyle="--", linewidth=1.2)
        style_axis(axis, percent=True, zero_line=True)
        axis.set_ylabel(LABELS[asset], fontsize=15)
        axis.set_xlim(-.3, first_steps - .7)
    axes[0].legend(frameon=False, ncol=3, fontsize=13, loc="upper right")
    axes[-1].set_xlabel("Simulated weekly step (new joins marked by dashed lines)", fontsize=16)
    figure.subplots_adjust(left=.17, right=.96, top=.84, bottom=.20, hspace=.55)
    _footer(figure, note="Method illustration: each block retains observed within-block order; joins are newly constructed.")
    return data, _save_pair(figure, output_dir, "fig01_block_lineage")


def figure02_joint_stress(tail: pd.DataFrame, *, output_dir: Path) -> tuple[pd.DataFrame, tuple[Path, Path]]:
    """Plot the synchronization comparison for simultaneous 10-asset tails."""
    subset = tail.loc[(tail["group"] == "investable_ex_cash10") & (tail["metric"] == "p_k_ge") & (tail["k"].between(2, 10))].copy()
    selection = {
        "historical_uniform": "Historical",
        "synchronized_iid_L01": "Synchronized L=1",
        "synchronized_mbb_L04": "Synchronized L=4",
        "independent_iid_control": "Independent IID control",
    }
    subset = subset.loc[subset["experiment"].isin(selection)].copy()
    subset["series"] = subset["experiment"].map(selection)
    data = subset.groupby(["series", "k"], as_index=False)["value"].mean()
    figure = create_figure()
    add_title(figure, "Synchronization preserves joint stress", "Probability of at least k simultaneous 5% lower-tail events among 10 investable non-cash assets")
    axis = figure.add_axes([.11, .24, .78, .59])
    colors = {"Historical": CHARCOAL, "Synchronized L=1": COBALT, "Synchronized L=4": TEAL, "Independent IID control": PALE_BLUE}
    for label in ("Historical", "Synchronized L=1", "Synchronized L=4", "Independent IID control"):
        rows = data.loc[data["series"] == label]
        axis.plot(rows["k"], rows["value"], marker="o", linewidth=2.8, color=colors[label], label=label)
    style_axis(axis, percent=True)
    axis.set_xlabel("At least k simultaneous lower-tail events", fontsize=18)
    axis.set_ylabel("Probability", fontsize=18)
    axis.set_xticks(range(2, 11))
    axis.legend(frameon=False, fontsize=13, loc="upper right")
    _footer(figure, note="L=1 already samples complete observed 12-driver weekly vectors; this chart isolates synchronization, not block length.")
    return data, _save_pair(figure, output_dir, "fig02_joint_stress")


def figure03_preservation_tradeoff(
    tradeoff: pd.DataFrame,
    acf_errors: pd.DataFrame,
    *,
    output_dir: Path,
) -> tuple[pd.DataFrame, tuple[Path, Path]]:
    """Compare constructive within-block adjacency to squared-ACF error."""
    tradeoff = tradeoff.loc[tradeoff["experiment"].str.startswith("synchronized", na=False)]
    x = tradeoff.groupby("L", as_index=False)["within_fraction"].mean()
    errors = acf_errors.loc[(acf_errors["experiment"].str.startswith("synchronized", na=False)) & (acf_errors["metric"] == "panel_rmse") & (acf_errors["transform"] == "squared_demeaned") & (acf_errors["lag_set"] == "1..13")]
    y = errors.groupby("L", as_index=False).agg(acf_rmse=("value", "mean"), mc_std=("value", "std"))
    data = x.merge(y, on="L", how="inner").sort_values("L")
    figure = create_figure()
    add_title(figure, "The preservation–recombination trade-off", "Constructive within-block neighboring pairs versus squared-return ACF error, lags 1–13")
    axis = figure.add_axes([.12, .25, .76, .57])
    axis.errorbar(data["within_fraction"], data["acf_rmse"], yerr=data["mc_std"].fillna(0), fmt="o", color=COBALT, capsize=5, linewidth=1.5)
    for row in data.itertuples():
        axis.annotate(f"L={row.L}", (row.within_fraction, row.acf_rmse), xytext=(7, 7), textcoords="offset points", fontsize=14)
    style_axis(axis)
    axis.set_xlabel("Constructive within-block neighboring-pair fraction", fontsize=17)
    axis.set_ylabel("Panel squared-return ACF RMSE", fontsize=17)
    _footer(figure, note="Error bars show across-replicate Monte-Carlo standard deviation; they are not confidence intervals.")
    return data, _save_pair(figure, output_dir, "fig03_preservation_tradeoff")


def figure04_volatility_memory(acf: pd.DataFrame, *, output_dir: Path) -> tuple[pd.DataFrame, tuple[Path, Path]]:
    """Show the preregistered Euro-government-bond squared-ACF mechanism case."""
    asset = "euro_govt_bond_7_10"
    subset = acf.loc[(acf["driver"] == asset) & (acf["transform"] == "squared_demeaned")].copy()
    selection = {"historical_uniform": "Historical", "synchronized_iid_L01": "L=1", "synchronized_mbb_L04": "L=4", "synchronized_mbb_L26": "L=26"}
    subset = subset.loc[subset["experiment"].isin(selection)].copy()
    subset["series"] = subset["experiment"].map(selection)
    data = subset.groupby(["series", "lag"], as_index=False)["acf"].mean()
    figure = create_figure()
    add_title(figure, "How much volatility memory survives?", "Squared demeaned-return ACF: Euro government bonds 7–10y, weekly lags 1–52")
    axis = figure.add_axes([.11, .24, .78, .59])
    colors = {"Historical": CHARCOAL, "L=1": COBALT, "L=4": TEAL, "L=26": PALE_BLUE}
    for label in ("Historical", "L=1", "L=4", "L=26"):
        rows = data.loc[data["series"] == label]
        axis.plot(rows["lag"], rows["acf"], linewidth=2.3, color=colors[label], label=label)
    style_axis(axis, zero_line=True)
    axis.set_xlabel("Lag (weeks)", fontsize=18)
    axis.set_ylabel("Autocorrelation", fontsize=18)
    axis.legend(frameon=False, ncol=2, fontsize=14)
    _footer(figure, note="All lines use the same lag-pair-normalized estimator; Monte-Carlo variation is available in the exported tables.")
    return data, _save_pair(figure, output_dir, "fig04_volatility_memory")


def terminal_outcome_plot_data(terminal: pd.DataFrame) -> pd.DataFrame:
    """Select both Figure-5 series by canonical experiment, not display label.

    The reference run's detailed label is ``reference_L04_L04_r0`` while its
    experiment is ``reference_L04``. Exact filtering on the former silently
    dropped every MBB point. Reject incomplete or duplicate series instead of
    drawing an empty artist with a misleading legend.
    """
    assets = ("global_dm_ex_emu", "euro_govt_bond_7_10", "euro_high_yield", "gold")
    subset = terminal.loc[(terminal["asset"].isin(assets)) & (terminal["horizon_steps"] == 260) &
                          (terminal["return_kind"] == "annualized_geometric_return") &
                          (terminal["metric"].isin(("q05", "median", "q95"))) &
                          (terminal["experiment"].isin(("reference_L04", "historical_rolling")))].copy()
    expected = pd.MultiIndex.from_product(
        [assets, ("historical_rolling", "reference_L04"), ("q05", "median", "q95")],
        names=["asset", "experiment", "metric"],
    )
    actual = pd.MultiIndex.from_frame(subset[["asset", "experiment", "metric"]])
    if len(actual) != len(expected) or actual.has_duplicates or len(expected.difference(actual)):
        raise ValueError("Figure 5 requires one q05/median/q95 per asset for BOTH historical and reference_L04 experiments")
    if not np.isfinite(subset["value"].to_numpy(dtype=float)).all():
        raise ValueError("Figure 5 quantiles must be finite for both series")
    data = subset.pivot(index=["asset", "experiment"], columns="metric", values="value").reset_index()
    source = subset.groupby(["asset", "experiment"], as_index=False).agg(
        source_reference=("reference", "first"), n_paths=("n_paths", "first")
    )
    data = data.merge(source, on=["asset", "experiment"], validate="one_to_one").rename(columns={"experiment": "reference"})
    if ((data["q05"] > data["median"]) | (data["median"] > data["q95"])).any():
        raise ValueError("Figure 5 requires ordered q05 <= median <= q95")
    return data


def figure05_terminal_outcomes(terminal: pd.DataFrame, *, output_dir: Path) -> tuple[pd.DataFrame, tuple[Path, Path]]:
    """Show both historical and MBB five-year annualized-return quantiles."""
    assets = ("global_dm_ex_emu", "euro_govt_bond_7_10", "euro_high_yield", "gold")
    data = terminal_outcome_plot_data(terminal)
    figure = create_figure()
    add_title(figure, "Terminal outcomes from recombined history", "5/50/95% annualized returns over 260 weeks; descriptive historical windows")
    axis = figure.add_axes([.14, .24, .75, .60])
    positions = np.arange(len(assets))
    for offset, reference, color, marker in ((-.16, "historical_rolling", CHARCOAL, "s"), (.16, "reference_L04", TEAL, "o")):
        rows = data.loc[data["reference"] == reference].set_index("asset").reindex(assets)
        axis.vlines(positions + offset, rows["q05"], rows["q95"], colors=color, linewidth=4)
        count = int(rows["n_paths"].iloc[0])
        label = f"Historical rolling windows (n={count:,})" if reference == "historical_rolling" else f"MBB L=4 scenarios (n={count:,})"
        axis.scatter(positions + offset, rows["median"], color=color, marker=marker, s=75, label=label)
    style_axis(axis, percent=True, zero_line=True)
    axis.set_xticks(positions, ["Global DM ex-EMU", "Euro govt 7–10y", "Euro HY", "Gold"], rotation=12, ha="right")
    axis.set_ylabel("Annualized geometric return", fontsize=18)
    axis.legend(frameon=False, fontsize=13)
    _footer(figure, note="Scenario/path quantiles, not confidence intervals or a validation of future uncertainty.")
    return data, _save_pair(figure, output_dir, "fig05_terminal_outcomes")


def figure06_path_risk(path_risk: pd.DataFrame, *, output_dir: Path) -> tuple[pd.DataFrame, tuple[Path, Path]]:
    """Plot five-year 95th-percentile MDD sensitivity across block lengths."""
    assets = ("global_dm_ex_emu", "euro_govt_bond_7_10", "euro_high_yield", "gold")
    subset = path_risk.loc[(path_risk["experiment"].str.startswith("synchronized", na=False)) & (path_risk["asset"].isin(assets)) & (path_risk["horizon_steps"] == 260) &
                           (path_risk["risk_kind"] == "maximum_drawdown") & (path_risk["metric"] == "q95")].copy()
    data = subset.groupby(["L", "asset"], as_index=False)["value"].mean().sort_values("L")
    figure = create_figure()
    add_title(figure, "Block length changes the path, not just the endpoint", "95th-percentile non-negative maximum drawdown over 260 simulated weeks")
    axis = figure.add_axes([.11, .24, .78, .59])
    colors = (COBALT, TEAL, PALE_BLUE, CHARCOAL)
    ordered_lengths = sorted(int(length) for length in data["L"].unique())
    length_positions = {length: position for position, length in enumerate(ordered_lengths)}
    for asset, color in zip(assets, colors):
        rows = data.loc[data["asset"] == asset]
        positions = [length_positions[int(length)] for length in rows["L"]]
        axis.plot(positions, rows["value"], marker="o", linewidth=2.5, color=color, label=LABELS[asset])
    style_axis(axis, percent=True)
    axis.set_xlabel("Block length (weeks)", fontsize=18)
    axis.set_ylabel("95th-percentile maximum drawdown", fontsize=18)
    axis.set_xticks(range(len(ordered_lengths)), ordered_lengths)
    axis.legend(frameon=False, fontsize=12, loc="best")
    _footer(figure, note="This is method sensitivity on one historical sample, not out-of-sample superiority evidence.")
    return data, _save_pair(figure, output_dir, "fig06_path_risk")
