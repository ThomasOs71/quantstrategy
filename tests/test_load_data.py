"""Tests for Block-1 data loading helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data.load_data as load_data
from data.asset_universe import (
    USD_EXPOSED_KEYS,
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
