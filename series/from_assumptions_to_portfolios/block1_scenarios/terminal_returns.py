"""Terminal return helpers for Block 1 scenario paths."""

from __future__ import annotations

import numpy as np


def compound(asset_paths: np.ndarray, horizon_months: int) -> np.ndarray:
    """Aggregate monthly log returns to one terminal simple return."""
    paths = np.asarray(asset_paths)
    horizon = int(horizon_months)

    if paths.ndim != 3:
        raise ValueError(
            f"asset_paths.ndim {paths.ndim} != expected 3"
        )

    if horizon <= 0:
        raise ValueError("horizon_months must be positive")

    if horizon > paths.shape[1]:
        raise ValueError(
            f"horizon_months {horizon} exceeds available months {paths.shape[1]}"
        )

    if not np.isfinite(paths).all():
        raise ValueError("asset_paths must be finite")

    cumulative = np.sum(paths[:, :horizon, :], axis=1)
    return np.expm1(cumulative)


def build_terminal_returns(
    asset_paths: np.ndarray,
    horizons: dict[str, int] = None,
) -> dict[str, np.ndarray]:
    """Build terminal simple returns for standard horizons."""
    if horizons is None:
        horizons = {"12m": 12, "36m": 36, "60m": 60}

    if not isinstance(horizons, dict):
        raise ValueError("horizons must be a dict")

    paths = np.asarray(asset_paths)
    max_h = max(int(months) for months in horizons.values())
    if max_h > paths.shape[1]:
        raise ValueError(
            f"max(horizons) {max_h} exceeds available months {paths.shape[1]}"
        )

    results: dict[str, np.ndarray] = {}
    for label, months in horizons.items():
        results[str(label)] = compound(paths, int(months))
    return results


def annualize(terminal_simple_return: np.ndarray, months: int) -> np.ndarray:
    """Annualize terminal simple returns from a horizon in months."""
    returns = np.asarray(terminal_simple_return)
    period = int(months)

    if period <= 0:
        raise ValueError("months must be positive")

    if not np.isfinite(returns).all():
        raise ValueError("terminal_simple_return must be finite")

    return np.power(1.0 + returns, 12.0 / period) - 1.0
