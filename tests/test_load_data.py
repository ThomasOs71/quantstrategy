"""Tests for Block-1 data loading helpers."""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data.load_data as load_data
from data.asset_universe import (
    USD_EXPOSED_KEYS,
    get_driver_keys,
    get_usd_exposed_keys,
    n_drivers,
    n_investable,
)


def test_convert_usd_to_eur_sign() -> None:
    r_usd = pd.Series([0.02], index=[pd.Timestamp("2020-01-31")])
    r_eurusd = pd.Series([0.01], index=[pd.Timestamp("2020-01-31")])

    result = load_data.convert_usd_to_eur(r_usd, r_eurusd)
    assert abs(result.iloc[0] - 0.01) < 1e-10


def test_hedge_formula_direction() -> None:
    r_usd = pd.Series([0.01], index=[pd.Timestamp("2020-01-31")])
    euribor = pd.Series([2.0], index=[pd.Timestamp("2020-01-31")])
    usd_3m = pd.Series([0.5], index=[pd.Timestamp("2020-01-31")])

    result = load_data.apply_eurusd_hedge_formula(r_usd, euribor, usd_3m)
    expected = 0.01 + (2.0 - 0.5) / 12 / 100
    assert abs(result.iloc[0] - expected) < 1e-10


def test_cash_rate_conversion() -> None:
    rate = pd.Series([2.4], index=[pd.Timestamp("2020-01-31")])
    result = np.log(1 + rate / 100 / 12)
    assert abs(result.iloc[0] - 0.001998) < 1e-5


def test_asset_universe_counts() -> None:
    assert n_investable() == 11
    assert n_drivers() == 12
    assert len(get_usd_exposed_keys()) == 4
    assert get_usd_exposed_keys() == USD_EXPOSED_KEYS


def test_msci_parser_skips_header_rows(tmp_path, monkeypatch) -> None:
    test_dir = tmp_path / "msci"
    test_dir.mkdir()
    file_path = test_dir / "msci_emu_ntr_usd.csv"
    file_path.write_text(
        "\n".join(
            [
                "MSCI Index Performance",
                "Index: MSCI EMU",
                "Currency: USD",
                "",
                "Date,Index Level",
                "Dec-86,100.00",
                "Jan-87,103.42",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(load_data, "MSCI_RAW_DIR", test_dir)
    result = load_data.load_msci_csv("msci_emu_ntr_usd.csv", start="1986-12")
    assert result.index.is_monotonic_increasing
    assert len(result) == 1
    assert np.isclose(result.iloc[0], np.log(103.42 / 100.00))


def _resolve_local_msci_file(base_name: str) -> Path:
    for ext in (".csv", ".xls", ".xlsx"):
        path = load_data.MSCI_RAW_DIR / f"{base_name}{ext}"
        if path.exists():
            return path
    raise FileNotFoundError(base_name)


def test_msci_download_present_and_parseable() -> None:
    """Check required local MSCI exports exist and can be parsed."""
    if os.environ.get("BLOCK1_CHECK_RAW_DATA") != "1":
        pytest.skip(
            "Set BLOCK1_CHECK_RAW_DATA=1 to verify local MSCI downloads before running this test."
        )

    for base_name in ("msci_emu_ntr_usd", "msci_world_ex_emu_ntr_usd"):
        path = _resolve_local_msci_file(base_name)
        assert path.suffix.lower() in {".csv", ".xls", ".xlsx"}
        assert path.stat().st_size > 0

        parsed = load_data.load_msci_csv(f"{base_name}.csv", start="2005-01")
        assert not parsed.empty
        assert parsed.index.is_monotonic_increasing
        assert parsed.isna().sum() == 0
        assert len(parsed) >= 60


def test_return_panel_quality_smoke() -> None:
    """Build the return panel and run basic economic sanity checks."""
    if os.environ.get("SKIP_DATA_QA") == "1":
        pytest.skip(
            "Set SKIP_DATA_QA=0 (or unset) to run the return-panel QA check."
        )

    fred_key = load_data._read_optional_fred_key()
    if not fred_key and not os.environ.get("FRED_API_KEY"):
        pytest.skip("Missing FRED API key for live data QA test.")

    panel = load_data.build_return_panel(
        start="2010-09-01",
        end="2026-06-01",
        fred_api_key=fred_key,
        warn_tbd=False,
    )

    assert panel is not None
    assert not panel.empty
    assert list(panel.columns) == get_driver_keys()
    assert panel.index.is_monotonic_increasing
    idx = panel.index
    assert idx.equals(idx.to_period("M").to_timestamp("M"))

    # No completely empty series.
    assert panel.notna().any().all()

    # Avoid absurd monthly jumps and keep correlations in a plausible band.
    assert panel.dropna().abs().le(0.5).all().all()

    # Each series may start and/or end with NaNs because of varying data coverage,
    # but should not have internal gaps in the active date window.
    for column in panel.columns:
        s = panel[column]
        valid = s.notna().to_numpy()
        if not valid.any():
            raise AssertionError(f"{column} has no valid monthly observations")

        first = np.flatnonzero(valid)[0]
        last = np.flatnonzero(valid)[-1]
        if not valid[first : last + 1].all():
            raise AssertionError(f"{column} has internal NaN gap in active window")

    corr = panel.corr()

    assert corr.loc["euro_equities", "global_dm_ex_emu"] > 0.5
    assert corr.loc["global_dm_ex_emu", "em_equities"] > 0.5
    assert corr.loc["euro_ig_credit", "euro_high_yield"] > 0.5
    assert corr.loc["em_hc_bond_eur_hedged", "global_govt_bond_eur_hedged"] > 0.5

    assert corr.loc["global_dm_ex_emu", "fx_eurusd"] < -0.3
    assert corr.loc["commodities", "fx_eurusd"] < -0.4

    assert abs(corr.loc["cash", "euro_equities"]) < 0.3
    assert abs(corr.loc["cash", "global_dm_ex_emu"]) < 0.3

    eigenvalues = np.linalg.eigvals(corr.fillna(0).values)
    assert eigenvalues.min().real > 0
