"""Tests for source-currency to EUR return conversion helpers."""

from __future__ import annotations

import numpy as np

from data.fx_mapping import apply_fx_mapping, convert_usd_to_eur_returns


def test_exact_fx_conversion_uses_gross_returns_including_principal() -> None:
    usd_log_returns = np.array([[[0.08], [-0.03]]])
    eurusd_log_returns = np.array([[0.02, -0.01]])

    result = convert_usd_to_eur_returns(
        usd_log_returns,
        eurusd_log_returns,
        method="exact",
    )

    expected = usd_log_returns - eurusd_log_returns[:, :, None]
    np.testing.assert_allclose(result, expected, atol=1e-15)


def test_already_eur_driver_paths_are_not_converted_twice() -> None:
    drivers = np.arange(2 * 3 * 12, dtype=float).reshape(2, 3, 12) / 1000
    names = [
        "global_dm_ex_emu",
        "em_equities",
        "gold",
        "commodities",
        "euro_equities",
        "euro_govt_bond_7_10",
        "euro_ig_credit",
        "euro_high_yield",
        "global_govt_bond_eur_hedged",
        "em_hc_bond_eur_hedged",
        "cash",
    ]
    asset_index_map = {name: index for index, name in enumerate(names)}

    result = apply_fx_mapping(
        drivers,
        asset_index_map=asset_index_map,
        fx_index=11,
        input_currency_state="eur_converted",
    )

    np.testing.assert_allclose(result, drivers[:, :, :11])
