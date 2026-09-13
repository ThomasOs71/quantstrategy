"""Rolling statistics used to diagnose Article 2 return building blocks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _validate_panel(panel: pd.DataFrame) -> None:
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    if panel.empty:
        raise ValueError("panel must not be empty")
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise TypeError("panel index must be a DatetimeIndex")
    if not np.isfinite(panel.to_numpy(dtype=float)).all():
        raise ValueError("panel must contain only finite values")


def _validate_window(window: int, n_observations: int) -> None:
    if not isinstance(window, int) or window < 2:
        raise ValueError("window must be an integer of at least 2")
    if window > n_observations:
        raise ValueError("window must not exceed the number of observations")


def rolling_annualized_volatility(
    panel: pd.DataFrame,
    window: int = 12,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Compute rolling sample volatility from periodic log returns."""
    _validate_panel(panel)
    _validate_window(window, len(panel))
    if periods_per_year < 1:
        raise ValueError("periods_per_year must be positive")
    return panel.rolling(window=window, min_periods=window).std(ddof=1) * np.sqrt(
        periods_per_year
    )


def rolling_pairwise_correlation(
    panel: pd.DataFrame,
    pairs: Sequence[tuple[str, str]],
    window: int = 24,
) -> pd.DataFrame:
    """Compute rolling correlations for explicit, named driver pairs."""
    _validate_panel(panel)
    _validate_window(window, len(panel))
    if not pairs:
        raise ValueError("pairs must not be empty")

    result = pd.DataFrame(index=panel.index)
    for left, right in pairs:
        if left == right:
            raise ValueError("a correlation pair must contain two different drivers")
        missing = [driver for driver in (left, right) if driver not in panel.columns]
        if missing:
            raise ValueError(f"pair contains unknown drivers: {missing}")
        result[f"{left}__{right}"] = panel[left].rolling(window).corr(panel[right])
    return result


def trailing_compound_return(panel: pd.DataFrame, window: int = 12) -> pd.DataFrame:
    """Compound trailing periodic log returns into trailing simple returns."""
    _validate_panel(panel)
    _validate_window(window, len(panel))
    return np.expm1(panel.rolling(window=window, min_periods=window).sum())


def trailing_state_features(
    panel: pd.DataFrame,
    anchor_driver: str,
    window: int = 12,
    periods_per_year: int = 12,
) -> pd.DataFrame:
    """Create transparent trailing return and volatility features for diagnostics."""
    _validate_panel(panel)
    if anchor_driver not in panel.columns:
        raise ValueError(f"anchor_driver '{anchor_driver}' is not in panel")
    trailing_return = trailing_compound_return(panel[[anchor_driver]], window)[
        anchor_driver
    ]
    trailing_volatility = rolling_annualized_volatility(
        panel[[anchor_driver]],
        window,
        periods_per_year=periods_per_year,
    )[anchor_driver]
    return pd.DataFrame(
        {
            "trailing_return": trailing_return,
            "trailing_volatility": trailing_volatility,
        },
        index=panel.index,
    )
