"""Pure publication-table selection; the frozen research estimands stay unchanged."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


LENGTHS = (1, 2, 4, 8, 13, 26, 52)
ASSETS = ("global_dm_ex_emu", "euro_govt_bond_7_10", "euro_high_yield", "gold")
PAIR = "global_dm_ex_emu|euro_govt_bond_7_10"
MEMORY_METRIC = "squared_demeaned_acf_rmse_non_cash11_lags1_13"
EXHIBITS = (
    "figure_a_workflow", "fig01_tail_sequence", "fig02_block_lineage",
    "fig03_time_resolution", "fig04_volatility_memory", "fig05_drawdown_sensitivity",
    "fig06_wealth_paths", "table01_target_specific_lengths", "table_a1_correlations",
)


def _finite(frame: pd.DataFrame, columns: Sequence[str]) -> None:
    if frame.empty or not np.isfinite(frame[list(columns)].to_numpy(dtype=float)).all():
        raise ValueError(f"Publication evidence is empty or nonfinite: {list(columns)}")


def _complete(frame: pd.DataFrame, columns: list[str], expected: pd.MultiIndex) -> None:
    actual = pd.MultiIndex.from_frame(frame[columns])
    if actual.has_duplicates or len(actual) != len(expected) or len(expected.difference(actual)):
        raise ValueError(f"Publication evidence requires exactly one row per {columns}")


def _replicate_mean(
    frame: pd.DataFrame, groups: list[str], combinations: Sequence[tuple], column: str
) -> pd.DataFrame:
    # Never pool paths or average duplicate/missing simulation summaries.
    expected = pd.MultiIndex.from_tuples(
        [(*combination, replicate) for combination in combinations for replicate in range(5)],
        names=groups + ["replicate_id"],
    )
    _complete(frame, groups + ["replicate_id"], expected)
    _finite(frame, [column])
    result = frame.groupby(groups, as_index=False, sort=True)[column].mean()
    result["n_replicates"] = 5
    return result


def tail_sequence_plot_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Conditional joint-tail probabilities, not simultaneous-tail frequencies."""
    selected = frame.loc[
        (frame.group == "investable_ex_cash10")
        & (frame.metric == "conditional_probability")
        & frame.lag.isin([1, 4])
    ].copy()
    history = selected.loc[selected.experiment == "historical_uniform"].copy()
    _complete(history, ["lag"], pd.MultiIndex.from_product([[1, 4]], names=["lag"]))
    _finite(history, ["value", "conditioning_events"])
    historical_rows = []
    for row in history.itertuples():
        count = row.conditioning_events * row.value
        if row.conditioning_events <= 0 or not np.isclose(count, round(count), atol=1e-10):
            raise ValueError("Historical tail probability must reconcile to integer event counts")
        historical_rows.append({"series": "Historical", "L": 0, "lag": row.lag,
                                "probability": row.value, "n_replicates": 1,
                                "conditioning_events": row.conditioning_events,
                                "joint_events": round(count)})
    simulated = selected.loc[selected.experiment.isin(
        ["synchronized_iid_L01", "synchronized_mbb_L04"]
    )]
    means = _replicate_mean(simulated, ["L", "lag"], [(l, k) for l in [1, 4] for k in [1, 4]], "value")
    means = means.rename(columns={"value": "probability"})
    means["series"] = means.L.map({1: "L=1", 4: "L=4"})
    result = pd.concat([pd.DataFrame(historical_rows), means], ignore_index=True)
    if not result.probability.between(0, 1).all():
        raise ValueError("Tail probabilities must lie in [0, 1]")
    return result.sort_values(["L", "lag"]).reset_index(drop=True)


def lineage_plot_data(frame: pd.DataFrame) -> pd.DataFrame:
    """The first twelve weeks of the existing recorded scenario zero."""
    assets = (ASSETS[0], ASSETS[1], "gold")
    selected = frame.copy()
    _complete(selected, ["asset", "simulation_step"], pd.MultiIndex.from_product(
        [assets, range(12)], names=["asset", "simulation_step"]
    ))
    _finite(selected, ["weekly_log_return", "source_idx", "block_id"])
    if not (selected.scenario_id == 0).all():
        raise ValueError("The lineage illustration must use recorded scenario zero")
    for step, group in selected.groupby("simulation_step"):
        if group.source_idx.nunique() != 1 or group.source_date.nunique() != 1:
            raise ValueError("All illustrated assets must use the same source week")
        if not (group.block_id == step // 4).all():
            raise ValueError("The lineage illustration must retain four-week blocks")
    ordered = selected.drop_duplicates("simulation_step").sort_values("simulation_step")
    for _, block in ordered.groupby("block_id"):
        if not (np.diff(block.source_idx) == 1).all():
            raise ValueError("Lineage source weeks must be consecutive within each block")
    return selected.sort_values(["asset", "simulation_step"]).reset_index(drop=True)


def resolution_plot_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean of five replicate-specific q95 MDDs on the identical L=4 paths."""
    metrics = ("weekly_mdd_q95", "coarse_mdd_q95")
    selected = frame.loc[(frame.L == 4) & frame.asset.isin(ASSETS)
                         & (frame.grid == "four_week_endpoints") & frame.metric.isin(metrics)]
    result = _replicate_mean(selected, ["asset", "metric"],
                             [(a, m) for a in ASSETS for m in metrics], "value")
    if not result.value.between(0, 1).all():
        raise ValueError("Maximum drawdowns must be decimal losses in [0, 1]")
    wide = result.pivot(index="asset", columns="metric", values="value")
    if (wide.coarse_mdd_q95 > wide.weekly_mdd_q95 + 1e-12).any():
        raise ValueError("Coarsening the same paths cannot increase drawdown")
    return result


def memory_plot_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Exact expected MBB squared-ACF error, eleven noncash drivers, lags 1–13."""
    selected = frame.loc[frame.metric == MEMORY_METRIC, ["L", "metric", "value"]].copy()
    _complete(selected, ["L"], pd.MultiIndex.from_product([LENGTHS], names=["L"]))
    _finite(selected, ["value"])
    if (selected.value < 0).any():
        raise ValueError("An RMSE cannot be negative")
    return selected.sort_values("L").reset_index(drop=True)


def drawdown_plot_data(frame: pd.DataFrame) -> pd.DataFrame:
    selected = frame.loc[
        frame.experiment.str.startswith("synchronized", na=False)
        & frame.asset.isin(ASSETS[:2]) & (frame.horizon_steps == 260)
        & (frame.risk_kind == "maximum_drawdown") & (frame.metric == "q95")
    ]
    result = _replicate_mean(selected, ["L", "asset"],
                             [(l, a) for l in LENGTHS for a in ASSETS[:2]], "value")
    if not result.value.between(0, 1).all():
        raise ValueError("Maximum drawdowns must be decimal losses in [0, 1]")
    return result


def selector_table_data(frame: pd.DataFrame) -> pd.DataFrame:
    transforms = ("raw", "squared_demeaned")
    selected = frame.loc[frame.driver_or_pair.isin(ASSETS) & frame["transform"].isin(transforms)]
    _complete(selected, ["driver_or_pair", "transform"], pd.MultiIndex.from_product(
        [ASSETS, transforms], names=["driver_or_pair", "transform"]
    ))
    _finite(selected, ["circular_length"])
    if (selected.circular_length <= 0).any():
        raise ValueError("Selector lengths must be positive")
    return selected.pivot(index="driver_or_pair", columns="transform", values="circular_length").reindex(
        ASSETS
    ).rename_axis("asset").rename(columns={
        "raw": "average_return_length_weeks", "squared_demeaned": "squared_deviation_length_weeks"
    }).reset_index()


def correlation_table_data(
    weekly: pd.DataFrame, rolling: pd.DataFrame, terminal: pd.DataFrame
) -> pd.DataFrame:
    lengths = [1, 4, 52]
    sources = [
        (weekly.loc[(weekly.driver_i == ASSETS[0]) & (weekly.driver_j == ASSETS[1])
                    & (weekly.group == "all12") & weekly.L.isin(lengths)],
         "correlation", "same_week_correlation"),
        (rolling.loc[(rolling.diagnostic == "rolling_correlation") & (rolling.asset_or_pair == PAIR)
                     & (rolling.window_steps == 104) & (rolling.metric == "q50") & rolling.L.isin(lengths)],
         "value", "median_104_week_rolling_correlation"),
        (terminal.loc[(terminal.driver_i == ASSETS[0]) & (terminal.driver_j == ASSETS[1])
                      & (terminal.horizon_steps == 260) & (terminal.return_type == "simple_compounded")
                      & terminal.L.isin(lengths)], "correlation", "five_year_outcome_correlation"),
    ]
    result = pd.DataFrame({"L": lengths})
    for selected, value_column, destination in sources:
        if not selected[value_column].between(-1, 1).all():
            raise ValueError("Pearson correlations must lie in [-1, 1]")
        means = _replicate_mean(selected, ["L"], [(l,) for l in lengths], value_column)
        result = result.merge(means[["L", value_column]].rename(columns={value_column: destination}),
                              on="L", validate="one_to_one")
    return result


def wealth_plot_data(
    log_returns: np.ndarray, *, seed: Sequence[int], n_paths: int = 100,
    path_ids: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Select an illustration, retaining the pointwise median of ALL reference paths.

    This does not infer the unrecorded original article selection. Initial wealth
    is one; no separate simulations, demeaning or probability changes occur.
    """
    values = np.asarray(log_returns, dtype=float)
    if values.ndim != 2 or min(values.shape) <= 0 or not np.isfinite(values).all():
        raise ValueError("Reference log returns must be a nonempty finite path-by-week matrix")
    if isinstance(n_paths, bool) or not isinstance(n_paths, (int, np.integer)) or not 0 < n_paths <= len(values):
        raise ValueError("n_paths must be a positive integer not exceeding the reference size")
    if not seed or any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) or x < 0 for x in seed):
        raise ValueError("selection seed must contain nonnegative integers")
    if path_ids is None:
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(list(seed))))
        selected = rng.choice(len(values), size=n_paths, replace=False)
        selection_source = "new_seeded_replacement"
    else:
        selected = np.asarray(path_ids)
        if selected.ndim != 1 or len(selected) != n_paths or not np.issubdtype(selected.dtype, np.integer):
            raise ValueError("Explicit path IDs must be an integer vector of length n_paths")
        selection_source = "explicit_ids_original_selection_not_verified"
    if len(np.unique(selected)) != n_paths or selected.min() < 0 or selected.max() >= len(values):
        raise ValueError("Path IDs must be distinct and inside the reference archive")
    with np.errstate(over="raise", invalid="raise"):
        wealth = np.exp(np.pad(np.cumsum(values, axis=1), ((0, 0), (1, 0))))
    steps = np.arange(wealth.shape[1])
    shown = pd.DataFrame({"series": "selected_path", "scenario_id": np.repeat(selected, len(steps)),
                          "step": np.tile(steps, n_paths), "wealth": wealth[selected].ravel()})
    # With an even reference count median(exp(x)) is not exactly exp(median(x)).
    median = pd.DataFrame({"series": "reference_median", "scenario_id": -1,
                           "step": steps, "wealth": np.median(wealth, axis=0)})
    ids = pd.DataFrame({"selection_order": np.arange(n_paths), "scenario_id": selected})
    metadata = {"selection_source": selection_source,
                "selection_seed": list(seed) if path_ids is None else None,
                "bit_generator": "PCG64" if path_ids is None else None, "n_selected_paths": int(n_paths),
                "n_reference_paths": len(values), "horizon_steps": values.shape[1],
                "median_population": "all_reference_paths_at_each_week",
                "matches_original_published_selection": False,
                "article_action": "Replace original Figure 6 to make its exact illustrated selection reproducible"}
    return pd.concat([shown, median], ignore_index=True), ids, metadata


def publication_claims(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Check the article's published rounding, not an invented numerical target.

    The comparison values below were transcribed from the linked published
    article and its two tables. Unrounded values always come from research.
    """
    records: list[dict] = []

    def add(stem: str, column: str, expected: str, decimals: int, scale: float = 1,
            *, unit: str = "decimal", **filters: object) -> None:
        selected = tables[stem]
        for key, value in filters.items():
            selected = selected.loc[selected[key] == value]
        if len(selected) != 1:
            raise ValueError(f"Published claim requires exactly one row: {stem}, {filters}")
        value = float(selected[column].iloc[0])
        if not np.isfinite(value):
            raise ValueError("Published claim evidence must be finite")
        display = f"{value * scale:.{decimals}f}"
        records.append({"claim_id": f"PUB{len(records) + 1:02d}",
                        "table_path": f"figures/plot_data/{stem}.csv",
                        "row_keys_and_columns": f"{filters}; column={column}",
                        "value": value, "unit": unit, "display_scale": scale,
                        "display_decimals": decimals, "published_display": expected,
                        "reproduced_display": display, "matches_published": display == expected})

    for length, lag, expected in ((0, 1, "24.44"), (0, 4, "25.56"),
                                  (1, 1, "11.51"), (1, 4, "11.42"),
                                  (4, 1, "21.26"), (4, 4, "11.54")):
        add("fig01_tail_sequence", "probability", expected, 2, 100,
            unit="probability", L=length, lag=lag)
    for column, expected in (("conditioning_events", "90"), ("joint_events", "22")):
        add("fig01_tail_sequence", column, expected, 0, unit="count", L=0, lag=1)
    for metric, expected in (("weekly_mdd_q95", "39.54"), ("coarse_mdd_q95", "38.00")):
        add("fig03_time_resolution", "value", expected, 2, 100,
            unit="decimal_loss", asset=ASSETS[0], metric=metric)
    memory = tables["fig04_volatility_memory"].set_index("L").value
    if memory.loc[1] <= 0:
        raise ValueError("The published relative improvement requires a positive L=1 error")
    improvement = 100 * (1 - memory.loc[4] / memory.loc[1])
    records.append({"claim_id": f"PUB{len(records) + 1:02d}",
                    "table_path": "figures/plot_data/fig04_volatility_memory.csv",
                    "row_keys_and_columns": "100 * (1 - value[L=4] / value[L=1])",
                    "value": improvement, "unit": "percent_reduction", "display_scale": 1,
                    "display_decimals": 0, "published_display": "31",
                    "reproduced_display": f"{improvement:.0f}",
                    "matches_published": f"{improvement:.0f}" == "31"})
    for asset, expected in ((ASSETS[0], ("36.02", "39.54", "29.14")),
                            (ASSETS[1], ("17.56", "19.10", "24.65"))):
        for length, number in zip((1, 4, 52), expected):
            add("fig05_drawdown_sensitivity", "value", number, 2, 100,
                unit="decimal_loss", asset=asset, L=length)
    for asset, numbers in zip(ASSETS, (("1.6", "18.3"), ("1.0", "50.0"),
                                       ("4.7", "11.5"), ("2.1", "7.6"))):
        for column, number in zip(("average_return_length_weeks", "squared_deviation_length_weeks"), numbers):
            add("table01_target_specific_lengths", column, number, 1, unit="weeks", asset=asset)
    for length, numbers in ((1, ("0.145", "0.131", "0.139")),
                            (4, ("0.144", "0.128", "0.192")),
                            (52, ("0.156", "0.151", "0.373"))):
        for column, number in zip(("same_week_correlation", "median_104_week_rolling_correlation",
                                   "five_year_outcome_correlation"), numbers):
            add("table_a1_correlations", column, number, 3, L=length)
    result = pd.DataFrame.from_records(records)
    if not result.matches_published.all():
        failures = result.loc[~result.matches_published, "claim_id"].tolist()
        raise ValueError(f"Published numerical claims do not reconcile: {failures}")
    return result
