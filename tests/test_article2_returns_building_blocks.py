"""Tests for the canonical Article-2 historical return panel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.asset_universe import get_driver_keys
import series.from_assumptions_to_portfolios.block1_scenarios.returns_building_blocks as article2_returns
from series.from_assumptions_to_portfolios.block1_scenarios.returns_building_blocks import (
    ARTICLE2_END,
    ARTICLE2_START,
    article2_panel_metadata,
    prepare_article2_panel,
    validate_article2_panel,
)


def _article2_panel() -> pd.DataFrame:
    index = pd.date_range(ARTICLE2_START, ARTICLE2_END, freq="ME")
    values = np.arange(len(index) * len(get_driver_keys()), dtype=float).reshape(
        len(index), len(get_driver_keys())
    )
    return pd.DataFrame(values / 10_000, index=index, columns=get_driver_keys())


def test_prepare_article2_panel_accepts_complete_expected_panel() -> None:
    panel = _article2_panel()
    panel.attrs["source_profile"] = "monthly_legacy"
    result = prepare_article2_panel(panel)
    assert result.equals(panel)
    assert result.shape == (180, 12)
    assert result.attrs["source_profile"] == "monthly_legacy"


def test_validate_article2_panel_rejects_wrong_column_order() -> None:
    panel = _article2_panel()
    panel = panel[list(reversed(panel.columns))]
    with pytest.raises(ValueError, match="driver order"):
        validate_article2_panel(panel)


def test_validate_article2_panel_rejects_monthly_gap() -> None:
    panel = _article2_panel().drop(pd.Timestamp("2015-06-30"))
    with pytest.raises(ValueError, match="complete monthly"):
        validate_article2_panel(panel)


def test_validate_article2_panel_rejects_nonfinite_return() -> None:
    panel = _article2_panel()
    panel.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        validate_article2_panel(panel)


def test_article2_panel_metadata_is_serializable_contract() -> None:
    panel = _article2_panel()
    panel.attrs.update(
        {
            "source_profile": "monthly_legacy",
            "known_currency_mismatch": True,
            "known_currency_mismatch_keys": ["em_equities", "commodities"],
        }
    )
    metadata = article2_panel_metadata(panel)
    assert metadata["n_observations"] == 180
    assert metadata["n_drivers"] == 12
    assert metadata["start"] == "2011-01-31"
    assert metadata["return_representation"] == "EUR monthly log returns"
    assert metadata["source_profile"] == "monthly_legacy"
    assert metadata["known_currency_mismatch"] is True


def test_article2_loader_explicitly_uses_monthly_legacy(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_build_return_panel(**kwargs):
        captured.update(kwargs)
        return _article2_panel()

    monkeypatch.setattr(article2_returns, "build_return_panel", fake_build_return_panel)

    result = article2_returns.load_article2_panel(fred_api_key="synthetic")

    assert result.equals(_article2_panel())
    assert captured["frequency"] == "monthly"
    assert captured["source_profile"] == "monthly_legacy"
