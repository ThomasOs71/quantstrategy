"""FX mapping helpers for Block 1: conversion of USD-exposed assets to EUR."""

from __future__ import annotations

from typing import Dict

import numpy as np


USD_EXPOSED_ASSETS = (
    "global_dm_ex_emu",
    "em_equities",
    "gold",
    "commodities",
)
N_ASSETS = 11
N_DRIVERS = 12


def convert_usd_to_eur_returns(
    asset_returns_usd: np.ndarray,
    fx_returns_eurusd: np.ndarray,
    method: str = "log_additive",
) -> np.ndarray:
    """Convert USD log-return paths to EUR log-return paths.

    The function assumes log-return convention for inputs and outputs.
    For method="log_additive", EUR/USD is interpreted as "USD per EUR"
    and conversion uses log-return additivity.
    """
    r_usd = np.asarray(asset_returns_usd)
    r_fx = np.asarray(fx_returns_eurusd)

    if method not in {"log_additive", "exact"}:
        raise ValueError("method must be 'log_additive' or 'exact'")

    if r_fx.ndim != 2:
        raise ValueError("fx_returns_eurusd must be a 2D array (S, H)")

    if r_fx.shape[0] != r_usd.shape[0] or r_fx.shape[1] != r_usd.shape[1]:
        raise ValueError(
            f"fx_returns_eurusd shape {r_fx.shape} must match first two dims of asset_returns_usd {r_usd.shape[:2]}"
        )

    if not np.isfinite(r_usd).all():
        raise ValueError("asset_returns_usd must be finite")

    if not np.isfinite(r_fx).all():
        raise ValueError("fx_returns_eurusd must be finite")

    if method == "log_additive":
        # log return convention + explicit convention for EUR strengthening: subtract.
        return r_usd - r_fx

    # exact: convert to simple -> multiply with FX -> back to log.
    usd_simple = np.expm1(r_usd)
    fx_simple = np.expm1(r_fx)
    eur_simple = usd_simple / (1.0 + fx_simple)
    return np.log1p(eur_simple)


def apply_fx_mapping(
    driver_paths: np.ndarray,
    asset_index_map: Dict[str, int],
    fx_index: int,
    method: str = "log_additive",
) -> np.ndarray:
    """Apply USD conversion for the USD-exposed assets and drop the FX driver."""
    drivers = np.asarray(driver_paths)

    if drivers.ndim != 3:
        raise ValueError("driver_paths must be a 3D array (S, H, N_drivers)")

    if drivers.shape[2] != N_DRIVERS:
        raise ValueError(
            f"driver_paths has {drivers.shape[2]} drivers but expected {N_DRIVERS}"
        )

    if fx_index < 0 or fx_index >= drivers.shape[2]:
        raise ValueError("fx_index is out of bounds")

    if len(asset_index_map) != N_ASSETS:
        raise ValueError(
            f"asset_index_map must provide {N_ASSETS} assets, got {len(asset_index_map)}"
        )

    fx_returns = drivers[:, :, fx_index]
    mapped_asset_paths = []

    for asset_name, column_index in asset_index_map.items():
        if not isinstance(column_index, int):
            raise ValueError("asset_index_map values must be integer indices")

        if column_index < 0 or column_index >= drivers.shape[2]:
            raise ValueError(f"asset column index {column_index} out of bounds")

        path = drivers[:, :, column_index]
        if asset_name in USD_EXPOSED_ASSETS:
            path = convert_usd_to_eur_returns(
                asset_returns_usd=path,
                fx_returns_eurusd=fx_returns,
                method=method,
            )
        mapped_asset_paths.append(path)

    return np.stack(mapped_asset_paths, axis=2)
