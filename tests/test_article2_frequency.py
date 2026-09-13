"""Frequency-contract tests for Article-2 research reruns."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from data.asset_universe import get_driver_keys
from data.panel_profiles import PanelFrequency
from series.from_assumptions_to_portfolios.block1_scenarios.article2_frequency import (
    MONTHLY_EQUIVALENT_WEEKLY_LAGS,
    WEEKLY_ACF_LAGS,
    get_article2_frequency_spec,
)
from series.from_assumptions_to_portfolios.block1_scenarios.diagnostics import (
    conditional_forward_period_return_summary,
    distribution_summary,
    lagged_schweizer_wolff_diagnostics,
    return_autocorrelation,
)
from series.from_assumptions_to_portfolios.block1_scenarios.returns_building_blocks import (
    prepare_article2_research_panel,
)
from series.from_assumptions_to_portfolios.block1_scenarios.rolling_stats import (
    trailing_state_features,
)
from series.from_assumptions_to_portfolios.block1_scenarios.run_article2_diagnostics import (
    generate_article2_outputs,
    run,
)
from series.from_assumptions_to_portfolios.flagship_plot_style import (
    add_empirical_footer,
    create_figure,
)


def _weekly_panel(periods: int = 160) -> pd.DataFrame:
    index = pd.date_range("2019-01-04", periods=periods, freq="W-FRI")
    generator = np.random.default_rng(127)
    values = generator.normal(0.0008, 0.018, size=(periods, len(get_driver_keys())))
    tail_rows = [row for row in (35, 88, 121) if row < periods]
    values[tail_rows, :] -= 0.08
    return pd.DataFrame(values, index=index, columns=get_driver_keys())


def test_article2_frequency_specs_preserve_calendar_horizons() -> None:
    monthly = get_article2_frequency_spec("monthly")
    weekly = get_article2_frequency_spec("weekly")

    assert monthly.periods_per_year == 12
    assert weekly.periods_per_year == 52
    assert weekly.ljung_box_lags == (26, 52)
    assert weekly.rolling_volatility_window == 52
    assert weekly.rolling_correlation_window == 104
    assert weekly.forward_horizon == 52
    assert weekly.acf_lags == WEEKLY_ACF_LAGS
    assert weekly.acf_lags == tuple(range(1, 53))
    assert weekly.schweizer_wolff_lags == MONTHLY_EQUIVALENT_WEEKLY_LAGS
    assert len(weekly.schweizer_wolff_lags) == len(monthly.schweizer_wolff_lags)


def test_legacy_article2_rejects_weekly_before_loading(tmp_path) -> None:
    with pytest.raises(ValueError, match="monthly_legacy.*monthly only"):
        run(
            tmp_path,
            source_profile="monthly_legacy",
            frequency="weekly",
            sw_permutations=1,
        )


def test_prepare_article2_research_panel_validates_complete_weekly_grid() -> None:
    panel = _weekly_panel(periods=20)
    prepared = prepare_article2_research_panel(
        panel,
        frequency="weekly",
        start=panel.index.min(),
        end=panel.index.max(),
    )
    assert prepared.equals(panel)

    with pytest.raises(ValueError, match="complete weekly"):
        prepare_article2_research_panel(
            panel.drop(panel.index[5]),
            frequency="weekly",
            start=panel.index.min(),
            end=panel.index.max(),
        )


def test_weekly_distribution_and_state_volatility_use_sqrt_52() -> None:
    panel = _weekly_panel(periods=60)[["global_dm_ex_emu"]]
    summary = distribution_summary(
        panel,
        periods_per_year=52,
        period_label="weekly",
    ).set_index("driver")
    expected = panel.std(ddof=1).iloc[0] * np.sqrt(52)
    assert summary.loc["global_dm_ex_emu", "annualized_volatility"] == pytest.approx(
        expected
    )
    assert "mean_weekly" in summary.columns
    assert "mean_monthly" not in summary.columns

    state = trailing_state_features(
        panel,
        "global_dm_ex_emu",
        window=52,
        periods_per_year=52,
    )
    expected_window = panel.iloc[:52, 0].std(ddof=1) * np.sqrt(52)
    assert state.iloc[51]["trailing_volatility"] == pytest.approx(expected_window)
    assert state.iloc[:51].isna().all().all()


def test_explicit_weekly_lags_and_forward_period_units_are_preserved() -> None:
    panel = _weekly_panel(periods=80)[["global_dm_ex_emu", "gold"]]
    selected_lags = (4, 13, 52)
    acf = return_autocorrelation(panel, lags=selected_lags)
    sw = lagged_schweizer_wolff_diagnostics(
        panel,
        lags=selected_lags,
        grid_size=20,
        n_permutations=9,
        seed=42,
    )
    assert tuple(acf["lag"].drop_duplicates()) == selected_lags
    assert tuple(sw["lag"].drop_duplicates()) == selected_lags

    state = pd.Series(np.arange(len(panel), dtype=float), index=panel.index)
    forward = conditional_forward_period_return_summary(
        panel,
        state,
        horizon_periods=4,
        period_unit="weeks",
        n_buckets=2,
    )
    assert set(forward["horizon_periods"]) == {4}
    assert set(forward["period_unit"]) == {"weeks"}


def test_weekly_article2_outputs_use_frequency_correct_contract(tmp_path) -> None:
    panel = _weekly_panel()
    generate_article2_outputs(
        panel,
        tmp_path,
        frequency=PanelFrequency.WEEKLY,
        sw_permutations=9,
        sw_grid_size=20,
    )

    distribution = pd.read_csv(tmp_path / "distribution_summary.csv")
    acf = pd.read_csv(tmp_path / "return_acf.csv")
    squared_acf = pd.read_csv(tmp_path / "squared_return_acf.csv")
    ljung_box = pd.read_csv(tmp_path / "article2_ljung_box_diagnostics.csv")
    sw = pd.read_csv(tmp_path / "schweizer_wolff_lagged.csv")
    state = pd.read_csv(tmp_path / "state_dependence.csv")
    metadata = json.loads((tmp_path / "run_metadata.json").read_text(encoding="utf-8"))

    assert "mean_weekly" in distribution.columns
    assert len(acf) == len(get_driver_keys()) * 52
    assert set(acf["lag"]) == set(WEEKLY_ACF_LAGS)
    assert len(squared_acf) == len(get_driver_keys()) * 52
    assert set(squared_acf["lag"]) == set(WEEKLY_ACF_LAGS)
    assert set(ljung_box["lag"]) == {26, 52}
    assert len(sw) == len(get_driver_keys()) * len(MONTHLY_EQUIVALENT_WEEKLY_LAGS)
    assert set(sw["lag"]) == set(MONTHLY_EQUIVALENT_WEEKLY_LAGS)
    assert set(state["horizon_periods"]) == {52}
    assert set(state["period_unit"]) == {"weeks"}
    assert metadata["analysis_frequency"] == "weekly"
    assert metadata["acf_lags"] == list(WEEKLY_ACF_LAGS)
    assert metadata["periods_per_year"] == 52
    assert metadata["rolling_correlation_window_periods"] == 104
    assert metadata["state_forward_horizon_periods"] == 52
    assert metadata["schweizer_wolff_hypotheses"] == 144


def test_empirical_footer_can_label_weekly_returns() -> None:
    figure = create_figure()
    add_empirical_footer(
        figure,
        sample="Jan 2011–Dec 2025",
        frequency_label="Weekly (W-FRI)",
        return_description="EUR weekly log returns where applicable",
    )
    labels = {text.get_text() for text in figure.texts}
    assert (
        "Sample: Jan 2011–Dec 2025  |  Weekly (W-FRI)  |  " "EUR investor perspective"
    ) in labels
    assert "EUR weekly log returns where applicable" in labels
