"""Deterministic contracts for Article-3 weekly Set-A mechanics and diagnostics."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from data.asset_universe import get_asset_keys, get_driver_keys
from series.from_assumptions_to_portfolios.block1_scenarios.article3_diagnostics import (
    ensemble_acf,
    expected_source_weights,
    path_risk_summary_rows,
    poisson_binomial_pmf,
    tail_threshold_rows,
)
from series.from_assumptions_to_portfolios.block1_scenarios.article3_input import (
    WeeklyInputProvenance,
    validate_article3_panel,
)
from series.from_assumptions_to_portfolios.block1_scenarios.block_sampler import (
    block_join_metrics,
    build_lineage_table,
    expected_source_weights as sampler_expected_source_weights,
    lag_within_block_fraction,
    sample_moving_block_indices,
    used_block_lengths,
)
from series.from_assumptions_to_portfolios.block1_scenarios.scenario_schema import ScenarioSet
from series.from_assumptions_to_portfolios.block1_scenarios.set_a_block_bootstrap import (
    Article3Config,
    build_set_a,
    load_set_a_reference,
    save_set_a_reference,
)
from series.from_assumptions_to_portfolios.block1_scenarios.terminal_returns import (
    annualize_terminal_return,
    build_terminal_returns_steps,
    compound,
    compound_log_paths,
)


def _weekly_panel() -> pd.DataFrame:
    index = pd.date_range("2011-01-07", "2025-12-26", freq="W-FRI")
    values = np.arange(len(index) * 12, dtype=np.float64).reshape(len(index), 12) / 1_000_000
    panel = pd.DataFrame(values, index=index, columns=get_driver_keys())
    panel.attrs.update({
        "source_profile": "daily_proxy_2011", "frequency": "weekly_friday", "resample_rule": "W-FRI",
        "return_representation": "EUR log returns", "periods_per_year": 52,
    })
    return panel


def _provenance() -> WeeklyInputProvenance:
    manifest = {"source_profile": "daily_proxy_2011", "frequency_option": "weekly", "rows": 782,
                "columns": 12, "driver_keys": get_driver_keys()}
    return WeeklyInputProvenance(Path("input.csv"), Path("input.manifest.json"), "a" * 64, "b" * 64, "c" * 64, None, "not_checked", manifest)


def test_mbb_indices_are_synchronized_non_circular_and_reproducible() -> None:
    first = sample_moving_block_indices(n_obs=10, n_scenarios=3, horizon_steps=10, block_length_steps=4, rng=np.random.default_rng(17))
    second = sample_moving_block_indices(n_obs=10, n_scenarios=3, horizon_steps=10, block_length_steps=4, rng=np.random.default_rng(17))
    assert first.used_block_lengths == (4, 4, 2)
    assert np.array_equal(first.source_indices, second.source_indices)
    assert np.array_equal(first.block_starts, second.block_starts)
    assert first.source_indices.shape == (3, 10)
    assert first.block_starts.min() >= 0
    assert first.block_starts.max() <= 6
    for path in first.source_indices:
        assert np.all(np.diff(path[:4]) == 1)
        assert np.all(np.diff(path[4:8]) == 1)
        assert np.all(np.diff(path[8:]) == 1)


@pytest.mark.parametrize("horizon,length,expected", [(10, 4, (4, 4, 2)), (3, 4, (3,)), (4, 1, (1, 1, 1, 1)), (4, 4, (4,))])
def test_used_block_lengths_cover_horizon_without_wrap(horizon: int, length: int, expected: tuple[int, ...]) -> None:
    assert used_block_lengths(horizon_steps=horizon, block_length_steps=length) == expected


def test_expected_occupancy_matches_enumerated_block_prefixes() -> None:
    n_obs, horizon, length = 5, 4, 3
    expected = sampler_expected_source_weights(n_obs=n_obs, horizon_steps=horizon, block_length_steps=length)
    counts = np.zeros(n_obs)
    for first in range(n_obs - length + 1):
        for second in range(n_obs - length + 1):
            indices = [first, first + 1, first + 2, second]
            np.add.at(counts, indices, 1)
    enumerated = counts / counts.sum()
    assert np.allclose(expected, enumerated)
    assert expected.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.allclose(expected_source_weights(n_obs=5, horizon_steps=4, block_length_steps=1), np.full(5, .2))


def test_join_and_lag_identity_for_truncated_final_block() -> None:
    metrics = block_join_metrics(horizon_steps=10, block_length_steps=4)
    assert metrics["n_joins"] == 2
    assert metrics["n_within_pairs"] == 7
    assert metrics["within_fraction"] == pytest.approx(7 / 9)
    assert metrics["join_fraction"] == pytest.approx(2 / 9)
    assert lag_within_block_fraction(horizon_steps=10, block_length_steps=4, lag=1) == pytest.approx(7 / 9)


def test_lineage_uses_actual_final_block_endpoints() -> None:
    starts = np.array([[1, 3, 5]], dtype=np.int64)
    dates = pd.date_range("2020-01-03", periods=10, freq="W-FRI")
    lineage = build_lineage_table(block_starts=starts, source_dates=dates, used_block_lengths=(4, 4, 2))
    assert lineage.iloc[-1]["source_end_idx_inclusive"] == 6
    assert lineage.iloc[-1]["sim_end_step_exclusive"] == 10


def test_step_compounding_and_annualization_are_frequency_explicit() -> None:
    paths = np.full((2, 52, 1), np.log(1.01), dtype=np.float64)
    compounded = compound_log_paths(paths, horizon_steps=52)
    assert compounded[0, 0] == pytest.approx(1.01 ** 52 - 1.0)
    terminals = build_terminal_returns_steps(paths, horizon_steps_by_label={"12m": 52})
    assert np.array_equal(terminals["12m"], compounded)
    assert annualize_terminal_return(compounded, horizon_steps=52, steps_per_year=52)[0, 0] == pytest.approx(compounded[0, 0])
    assert compound(paths, 52)[0, 0] == pytest.approx(compounded[0, 0])


def test_weekly_schema_never_reports_weeks_as_months() -> None:
    paths = np.zeros((2, 260, 12), dtype=np.float64)
    assets = paths[:, :, :11]
    scenario = ScenarioSet("A", "weekly", paths, assets, {key: np.zeros((2, 11)) for key in ("12m", "36m", "60m")}, np.array([.5, .5]), metadata={"frequency": "weekly", "steps_per_year": 52})
    assert scenario.horizon_steps == 260
    with pytest.raises(ValueError, match="horizon_steps"):
        _ = scenario.horizon_months


def test_article3_panel_contract_rejects_monthly_metadata_and_missing_week() -> None:
    panel = _weekly_panel()
    validate_article3_panel(panel, provenance=_provenance().manifest)
    monthly = panel.copy()
    monthly.attrs["frequency"] = "monthly_month_end"
    with pytest.raises(ValueError, match="W-FRI"):
        validate_article3_panel(monthly, provenance=_provenance().manifest)
    missing = panel.drop(panel.index[10])
    with pytest.raises(ValueError, match="complete"):
        validate_article3_panel(missing, provenance=_provenance().manifest)


def test_set_a_reconstructs_paths_and_reference_round_trips(tmp_path: Path) -> None:
    config = replace(Article3Config(), production_n_scenarios=3)
    build = build_set_a(_weekly_panel(), config=config, provenance=_provenance())
    source = _weekly_panel().to_numpy()
    assert np.array_equal(build.scenario_set.driver_paths, source[build.sample.source_indices])
    assert np.array_equal(build.scenario_set.asset_paths, build.scenario_set.driver_paths[:, :, :11])
    assert build.scenario_set.p.sum() == pytest.approx(1.0, abs=1e-12)
    npz, lineage = save_set_a_reference(build, output_dir=tmp_path, block_length_steps=4)
    loaded = load_set_a_reference(npz)
    assert np.array_equal(loaded["source_indices"], build.sample.source_indices)
    assert list(loaded["asset_keys"].astype(str)) == get_asset_keys()
    assert lineage.is_file()


def test_acf_matches_direct_lag_pair_formula_and_squared_contract() -> None:
    paths = np.array([[[0.0], [1.0], [0.0], [2.0]], [[1.0], [0.0], [1.0], [0.0]]])
    acf, status = ensemble_acf(paths, max_lag=2, transform="raw")
    signal = paths[:, :, 0]
    mean = signal.mean()
    variance = np.mean((signal - mean) ** 2)
    direct = np.array([
        np.sum((signal[:, lag:] - mean) * (signal[:, :-lag] - mean)) / (signal.shape[0] * (signal.shape[1] - lag)) / variance
        for lag in (1, 2)
    ])
    assert status == ["ok"]
    assert np.allclose(acf[0], direct)
    squared, _ = ensemble_acf(paths, max_lag=2, transform="squared_demeaned", historical_means=np.array([.5]))
    assert squared.shape == (1, 2)


def test_strict_tail_threshold_and_poisson_binomial_control() -> None:
    history = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [2.0, 2.0]])
    table, threshold = tail_threshold_rows(history, ["a", "b"], probability=.5)
    assert threshold.tolist() == [0.5, 1.0]
    assert table.loc[0, "historical_tail_count"] == 2  # strict < 0.5 includes both zeros
    pmf = poisson_binomial_pmf([.2, .4])
    assert np.allclose(pmf, [.48, .44, .08])


def test_path_risk_includes_initial_wealth_and_compounds_windows() -> None:
    paths = np.array([[[np.log(.9)], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]])
    summary = path_risk_summary_rows(paths, ["asset"], horizons={"13w": 13}, reference="toy")
    mdd = summary.loc[(summary["risk_kind"] == "maximum_drawdown") & (summary["metric"] == "q95"), "value"].iloc[0]
    worst = summary.loc[(summary["risk_kind"] == "worst_13_week_return") & (summary["metric"] == "q05"), "value"].iloc[0]
    assert mdd == pytest.approx(.1)
    assert worst == pytest.approx(-.1)
