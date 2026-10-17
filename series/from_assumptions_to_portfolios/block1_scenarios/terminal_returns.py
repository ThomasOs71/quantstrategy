"""Terminal return helpers for Block 1 scenario paths."""

from __future__ import annotations

import numpy as np


def _validate_paths_and_steps(paths: np.ndarray, horizon_steps: int) -> np.ndarray:
    """Validate a three-dimensional log-return path array and one horizon."""
    if isinstance(horizon_steps, (bool, np.bool_)) or not isinstance(
        horizon_steps, (int, np.integer)
    ):
        raise TypeError("horizon_steps must be an integer")
    horizon = int(horizon_steps)
    array = np.asarray(paths, dtype=np.float64)
    if array.ndim != 3:
        raise ValueError(f"paths.ndim {array.ndim} != expected 3")
    if horizon <= 0:
        raise ValueError("horizon_steps must be positive")
    if horizon > array.shape[1]:
        raise ValueError(
            f"horizon_steps {horizon} exceeds available steps {array.shape[1]}"
        )
    if not np.isfinite(array).all():
        raise ValueError("paths must be finite")
    return array


def compound_log_paths(paths: np.ndarray, *, horizon_steps: int) -> np.ndarray:
    """Compound log-return paths into simple returns over explicit steps.

    The caller supplies the period convention through metadata.  This helper is
    deliberately frequency-neutral so a weekly ScenarioSet cannot accidentally
    be interpreted as a monthly one.
    """
    array = _validate_paths_and_steps(paths, horizon_steps)
    return np.expm1(np.sum(array[:, :horizon_steps, :], axis=1))


def build_terminal_returns_steps(
    paths: np.ndarray, *, horizon_steps_by_label: dict[str, int]
) -> dict[str, np.ndarray]:
    """Build simple terminal returns from prefixes of the same log paths."""
    if not isinstance(horizon_steps_by_label, dict) or not horizon_steps_by_label:
        raise ValueError("horizon_steps_by_label must be a non-empty dict")
    results: dict[str, np.ndarray] = {}
    for label, horizon_steps in horizon_steps_by_label.items():
        if not isinstance(label, str) or not label:
            raise ValueError("terminal horizon labels must be non-empty strings")
        results[label] = compound_log_paths(paths, horizon_steps=horizon_steps)
    return results


def annualize_terminal_return(
    terminal_simple_return: np.ndarray,
    *,
    horizon_steps: int,
    steps_per_year: int,
) -> np.ndarray:
    """Annualize a simple terminal return using an explicit step convention."""
    if isinstance(horizon_steps, (bool, np.bool_)) or not isinstance(
        horizon_steps, (int, np.integer)
    ):
        raise TypeError("horizon_steps must be an integer")
    if isinstance(steps_per_year, (bool, np.bool_)) or not isinstance(
        steps_per_year, (int, np.integer)
    ):
        raise TypeError("steps_per_year must be an integer")
    if horizon_steps <= 0 or steps_per_year <= 0:
        raise ValueError("horizon_steps and steps_per_year must be positive")
    returns = np.asarray(terminal_simple_return, dtype=np.float64)
    if not np.isfinite(returns).all() or np.any(returns <= -1.0):
        raise ValueError("terminal_simple_return must be finite and greater than -1")
    return np.power(1.0 + returns, float(steps_per_year) / horizon_steps) - 1.0


def compound(asset_paths: np.ndarray, horizon_months: int) -> np.ndarray:
    """Legacy monthly wrapper for :func:`compound_log_paths`."""
    return compound_log_paths(asset_paths, horizon_steps=horizon_months)


def build_terminal_returns(
    asset_paths: np.ndarray,
    horizons: dict[str, int] = None,
) -> dict[str, np.ndarray]:
    """Build terminal simple returns for standard horizons."""
    if horizons is None:
        horizons = {"12m": 12, "36m": 36, "60m": 60}

    if not isinstance(horizons, dict):
        raise ValueError("horizons must be a dict")

    return build_terminal_returns_steps(
        asset_paths,
        horizon_steps_by_label={str(label): int(months) for label, months in horizons.items()},
    )


def annualize(terminal_simple_return: np.ndarray, months: int) -> np.ndarray:
    """Annualize terminal simple returns from a horizon in months."""
    return annualize_terminal_return(
        terminal_simple_return,
        horizon_steps=months,
        steps_per_year=12,
    )
