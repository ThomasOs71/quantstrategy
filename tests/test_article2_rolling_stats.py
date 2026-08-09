"""Tests for Article-2 rolling statistical helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from series.from_assumptions_to_portfolios.block1_scenarios.rolling_stats import (
    rolling_annualized_volatility,
    rolling_pairwise_correlation,
    trailing_compound_return,
    trailing_state_features,
)


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "a": [0.00, 0.01, 0.02, 0.03, 0.04, 0.05],
            "b": [0.00, -0.01, -0.02, -0.03, -0.04, -0.05],
        },
        index=pd.date_range("2020-01-31", periods=6, freq="ME"),
    )


def test_rolling_annualized_volatility_uses_complete_window() -> None:
    result = rolling_annualized_volatility(_panel(), window=3)
    assert result.iloc[:2].isna().all().all()
    expected = np.std([0.00, 0.01, 0.02], ddof=1) * np.sqrt(12)
    assert np.isclose(result.iloc[2, 0], expected)


def test_rolling_pairwise_correlation_uses_named_pair() -> None:
    result = rolling_pairwise_correlation(_panel(), [("a", "b")], window=3)
    assert list(result.columns) == ["a__b"]
    assert np.isclose(result.iloc[-1, 0], -1.0)


def test_trailing_compound_return_composes_log_returns() -> None:
    panel = _panel()
    result = trailing_compound_return(panel[["a"]], window=2)
    assert np.isclose(result.iloc[1, 0], np.expm1(0.01))


def test_trailing_state_features_contains_return_and_volatility() -> None:
    result = trailing_state_features(_panel(), "a", window=3)
    assert list(result.columns) == ["trailing_return", "trailing_volatility"]
    assert result.iloc[:2].isna().all().all()


def test_rolling_pairwise_correlation_rejects_unknown_driver() -> None:
    with pytest.raises(ValueError, match="unknown drivers"):
        rolling_pairwise_correlation(_panel(), [("a", "missing")], window=3)
