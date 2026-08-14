"""Tests for Article-2 empirical diagnostics using synthetic return panels."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from series.from_assumptions_to_portfolios.block1_scenarios.diagnostics import (
    conditional_forward_return_summary,
    joint_tail_event_summary,
    ljung_box_diagnostics,
    return_autocorrelation,
    squared_return_autocorrelation,
    tail_events,
)


def _panel(values: dict[str, list[float]]) -> pd.DataFrame:
    length = len(next(iter(values.values())))
    return pd.DataFrame(
        values,
        index=pd.date_range("2020-01-31", periods=length, freq="ME"),
    )


def test_return_autocorrelation_detects_persistent_series() -> None:
    panel = _panel({"persistent": [float(value) for value in range(20)]})
    result = return_autocorrelation(panel, lags=2)
    lag_one = result.loc[result["lag"] == 1, "autocorrelation"].iloc[0]
    assert lag_one > 0.99


def test_squared_return_autocorrelation_labels_its_transform() -> None:
    panel = _panel({"returns": [(-1.0) ** value * value for value in range(1, 20)]})
    result = squared_return_autocorrelation(panel, lags=1)
    assert result.loc[0, "transform"] == "squared_demeaned_return"
    assert np.isfinite(result.loc[0, "autocorrelation"])


def test_ljung_box_diagnostics_exports_requested_transforms_and_lags() -> None:
    panel = _panel(
        {
            "a": [0.01 * value for value in range(1, 25)],
            "b": [(-0.02) ** (value % 3 + 1) for value in range(1, 25)],
        }
    )
    result = ljung_box_diagnostics(panel, lags=(6, 12))
    assert len(result) == 2 * 2 * 2
    assert set(result["transform"]) == {"raw_return", "squared_demeaned_return"}
    assert set(result["lag"]) == {6, 12}
    assert result["p_value"].between(0.0, 1.0).all()


def test_tail_events_and_joint_tail_summary_find_shared_extreme_month() -> None:
    panel = _panel(
        {
            "a": [0.01, 0.02, -0.40, 0.01, 0.02, 0.03],
            "b": [0.03, 0.01, -0.35, 0.02, 0.04, 0.01],
            "c": [0.02, 0.01, 0.03, 0.01, 0.02, 0.04],
        }
    )
    events = tail_events(panel, quantile=0.20)
    joint = joint_tail_event_summary(panel, quantile=0.20, min_assets=2)
    crash_date = pd.Timestamp("2020-03-31")
    assert crash_date in set(events["date"])
    assert joint.loc[joint["date"] == crash_date, "tail_asset_count"].iloc[0] == 2


def test_conditional_forward_returns_use_only_following_months() -> None:
    panel = _panel(
        {
            "a": [0.00, 0.10, 0.20, 0.30, 0.40, 0.50],
            "b": [0.00, 0.05, 0.10, 0.15, 0.20, 0.25],
        }
    )
    state = pd.Series(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], index=panel.index, name="state"
    )
    result = conditional_forward_return_summary(
        panel,
        state,
        horizon_months=1,
        n_buckets=2,
        state_name="synthetic_state",
    )
    low_state_a = result.loc[
        (result["state_bucket"] == result["state_bucket"].iloc[0])
        & (result["driver"] == "a"),
        "mean_forward_return",
    ].iloc[0]
    assert low_state_a > 0.0
    assert set(result["horizon_months"]) == {1}


def test_conditional_forward_return_rejects_too_short_panel() -> None:
    panel = _panel({"a": [0.01, 0.02, 0.03]})
    state = pd.Series([1.0, 2.0, 3.0], index=panel.index)
    with pytest.raises(ValueError, match="horizon_months"):
        conditional_forward_return_summary(panel, state, horizon_months=3)
