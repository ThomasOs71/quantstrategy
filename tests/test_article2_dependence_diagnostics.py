"""Deterministic tests for Article-2 linear and nonlinear dependence diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from series.from_assumptions_to_portfolios.block1_scenarios.diagnostics import (
    lagged_schweizer_wolff_diagnostics,
    ljung_box_summary,
    schweizer_wolff_dependence,
    schweizer_wolff_summary,
)


def _panel(values: dict[str, np.ndarray]) -> pd.DataFrame:
    length = len(next(iter(values.values())))
    return pd.DataFrame(
        values,
        index=pd.date_range("2000-01-31", periods=length, freq="ME"),
    )


def test_ljung_box_white_noise_and_persistent_ar_process_are_distinguished() -> None:
    generator = np.random.default_rng(7)
    white_noise = generator.normal(size=400)
    ar_values = np.empty(400)
    ar_values[0] = 0.0
    for index in range(1, len(ar_values)):
        ar_values[index] = 0.80 * ar_values[index - 1] + generator.normal()

    result = ljung_box_summary(_panel({"white_noise": white_noise, "ar_one": ar_values}))
    raw = result.loc[result["transform"] == "raw_return"].set_index("driver")
    assert raw.loc["white_noise", "lb_pvalue"] > 0.10
    assert raw.loc["ar_one", "lb_pvalue"] < 0.001
    assert set(result.columns) == {
        "driver", "transform", "lag", "n_observations", "lb_stat", "lb_pvalue",
        "lb_pvalue_fdr", "reject_raw_5pct", "reject_fdr_5pct",
    }


def test_ljung_box_detects_arch_like_dependence_in_squared_returns() -> None:
    generator = np.random.default_rng(31)
    values = np.empty(1_000)
    values[0] = generator.normal(scale=0.01)
    for index in range(1, len(values)):
        conditional_variance = 0.00002 + 0.65 * values[index - 1] ** 2
        values[index] = np.sqrt(conditional_variance) * generator.normal()

    result = ljung_box_summary(_panel({"arch_like": values})).set_index("transform")
    assert result.loc["raw_return", "lb_pvalue"] > 0.05
    assert result.loc["squared_demeaned_return", "lb_pvalue"] < 0.01


def test_ljung_box_reuses_existing_finite_panel_validation() -> None:
    panel = _panel({"a": np.array([0.0, np.nan] * 10)})
    with pytest.raises(ValueError, match="finite"):
        ljung_box_summary(panel)


def test_schweizer_wolff_distinguishes_independent_monotonic_and_nonlinear_cases() -> None:
    generator = np.random.default_rng(11)
    x_values = generator.normal(size=1_000)
    independent = generator.normal(size=1_000)
    monotonic = x_values + generator.normal(scale=0.01, size=1_000)
    nonlinear = x_values**2 + generator.normal(scale=0.03, size=1_000)

    independent_sw = schweizer_wolff_dependence(x_values, independent, grid_size=80)
    monotonic_sw = schweizer_wolff_dependence(x_values, monotonic, grid_size=80)
    nonlinear_sw = schweizer_wolff_dependence(x_values, nonlinear, grid_size=80)

    assert monotonic_sw > independent_sw * 4.0
    assert nonlinear_sw > independent_sw * 2.0
    assert abs(np.corrcoef(x_values, nonlinear)[0, 1]) < 0.10


def test_schweizer_wolff_is_invariant_to_strictly_monotonic_transforms() -> None:
    generator = np.random.default_rng(17)
    x_values = generator.normal(size=300)
    y_values = x_values**3 + generator.normal(scale=0.05, size=300)
    baseline = schweizer_wolff_dependence(x_values, y_values, grid_size=80)
    transformed = schweizer_wolff_dependence(
        np.exp(x_values),
        np.log(y_values - y_values.min() + 1.0),
        grid_size=80,
    )
    assert transformed == pytest.approx(baseline, abs=1e-12)


def test_schweizer_wolff_includes_observations_on_grid_boundaries() -> None:
    values = np.array([0.0, 1.0, 2.0])
    pseudo = np.array([0.25, 0.50, 0.75])
    grid = np.arange(1, 5, dtype=float) / 4
    empirical = np.array(
        [
            [np.mean((pseudo <= u_value) & (pseudo <= v_value)) for v_value in grid]
            for u_value in grid
        ]
    )
    expected = 12.0 * np.mean(np.abs(empirical - np.multiply.outer(grid, grid)))

    result = schweizer_wolff_dependence(values, values, grid_size=4)

    assert result == pytest.approx(expected, abs=1e-12)


def test_schweizer_wolff_lagged_output_is_deterministic_and_handles_ties() -> None:
    generator = np.random.default_rng(23)
    panel = _panel(
        {
            "a": generator.normal(size=90),
            "b": np.repeat([0.0, 0.01, -0.01], 30),
        }
    )
    first = lagged_schweizer_wolff_diagnostics(
        panel, max_lag=3, grid_size=40, n_permutations=49, seed=42
    )
    second = lagged_schweizer_wolff_diagnostics(
        panel, max_lag=3, grid_size=40, n_permutations=49, seed=42
    )
    pd.testing.assert_frame_equal(first, second)
    assert len(first) == 6
    assert set(first["lag"]) == {1, 2, 3}
    assert first["permutation_pvalue"].between(0.0, 1.0).all()
    summary = schweizer_wolff_summary(first)
    assert len(summary) == 2
    assert set(summary["driver"]) == {"a", "b"}


def test_schweizer_wolff_rejects_nonfinite_input() -> None:
    with pytest.raises(ValueError, match="finite"):
        schweizer_wolff_dependence(np.array([0.0, np.inf]), np.array([0.0, 1.0]))
