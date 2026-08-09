"""Empirical diagnostics for Article 2 historical return building blocks."""

from __future__ import annotations

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


def distribution_summary(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarise monthly log-return distributions by driver."""
    _validate_panel(panel)
    summary = pd.DataFrame(
        {
            "mean_monthly": panel.mean(),
            "std_monthly": panel.std(ddof=1),
            "annualized_volatility": panel.std(ddof=1) * np.sqrt(12.0),
            "skewness": panel.skew(),
            "excess_kurtosis": panel.kurt(),
            "minimum": panel.min(),
            "q01": panel.quantile(0.01),
            "q05": panel.quantile(0.05),
            "median": panel.median(),
            "q95": panel.quantile(0.95),
            "q99": panel.quantile(0.99),
            "maximum": panel.max(),
        }
    )
    summary.index.name = "driver"
    return summary.reset_index()


def return_autocorrelation(panel: pd.DataFrame, lags: int = 12) -> pd.DataFrame:
    """Return lagged autocorrelations of raw monthly log returns in tidy form."""
    return _autocorrelation_table(panel, lags=lags, transform="raw_return")


def squared_return_autocorrelation(panel: pd.DataFrame, lags: int = 12) -> pd.DataFrame:
    """Return autocorrelations of squared demeaned returns as a volatility proxy."""
    _validate_panel(panel)
    centered_squared = panel.subtract(panel.mean(), axis="columns").pow(2)
    return _autocorrelation_table(
        centered_squared,
        lags=lags,
        transform="squared_demeaned_return",
    )


def _autocorrelation_table(
    panel: pd.DataFrame,
    lags: int,
    transform: str,
) -> pd.DataFrame:
    _validate_panel(panel)
    if not isinstance(lags, int) or lags < 1:
        raise ValueError("lags must be a positive integer")
    if lags >= len(panel):
        raise ValueError("lags must be smaller than the number of observations")

    records: list[dict[str, object]] = []
    for driver in panel.columns:
        series = panel[driver]
        for lag in range(1, lags + 1):
            records.append(
                {
                    "driver": driver,
                    "transform": transform,
                    "lag": lag,
                    "autocorrelation": series.autocorr(lag=lag),
                    "n_observations": len(series) - lag,
                }
            )
    return pd.DataFrame.from_records(records)


def tail_thresholds(panel: pd.DataFrame, quantile: float = 0.05) -> pd.DataFrame:
    """Return per-driver lower-tail thresholds for a valid probability quantile."""
    _validate_panel(panel)
    _validate_quantile(quantile)
    thresholds = panel.quantile(quantile)
    result = thresholds.rename("threshold").reset_index()
    result.columns = ["driver", "threshold"]
    result.insert(1, "quantile", quantile)
    return result


def tail_events(panel: pd.DataFrame, quantile: float = 0.05) -> pd.DataFrame:
    """List all dates where a driver lies in its own empirical lower tail."""
    _validate_panel(panel)
    thresholds = tail_thresholds(panel, quantile).set_index("driver")["threshold"]
    records: list[dict[str, object]] = []
    for driver in panel.columns:
        series = panel[driver]
        for date, value in series[series <= thresholds[driver]].items():
            records.append(
                {
                    "date": date,
                    "driver": driver,
                    "return": value,
                    "quantile": quantile,
                    "threshold": thresholds[driver],
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=["date", "driver", "return", "quantile", "threshold"],
    ).sort_values(["date", "driver"], ignore_index=True)


def joint_tail_event_summary(
    panel: pd.DataFrame,
    quantile: float = 0.05,
    min_assets: int = 2,
) -> pd.DataFrame:
    """Summarise months with simultaneous lower-tail events across drivers."""
    _validate_panel(panel)
    _validate_quantile(quantile)
    if not isinstance(min_assets, int) or min_assets < 1:
        raise ValueError("min_assets must be a positive integer")

    thresholds = panel.quantile(quantile)
    flags = panel.le(thresholds, axis="columns")
    records: list[dict[str, object]] = []
    for date, row in flags.iterrows():
        drivers = list(row.index[row.to_numpy()])
        if len(drivers) >= min_assets:
            records.append(
                {
                    "date": date,
                    "tail_asset_count": len(drivers),
                    "tail_drivers": ", ".join(drivers),
                    "quantile": quantile,
                }
            )
    return pd.DataFrame.from_records(
        records,
        columns=["date", "tail_asset_count", "tail_drivers", "quantile"],
    )


def conditional_forward_return_summary(
    panel: pd.DataFrame,
    state: pd.Series,
    horizon_months: int = 12,
    n_buckets: int = 3,
    state_name: str = "state",
) -> pd.DataFrame:
    """Describe forward returns by a pre-observed state without fitting a model.

    ``state`` must be known at each panel date. Forward returns begin in the
    following month, preventing the state observation from using future returns.
    The resulting observations overlap and are descriptive, not an inference test.
    """
    _validate_panel(panel)
    if not isinstance(state, pd.Series):
        raise TypeError("state must be a pandas Series")
    if horizon_months < 1 or horizon_months >= len(panel):
        raise ValueError("horizon_months must be between 1 and len(panel) - 1")
    if n_buckets < 2:
        raise ValueError("n_buckets must be at least 2")

    state = pd.to_numeric(state.reindex(panel.index), errors="coerce")
    future_log = (
        panel.shift(-1)
        .rolling(window=horizon_months, min_periods=horizon_months)
        .sum()
        .shift(-(horizon_months - 1))
    )
    future_simple = np.expm1(future_log)
    available = state.notna() & future_simple.notna().all(axis=1)
    if available.sum() < n_buckets:
        raise ValueError("not enough state and forward-return observations for buckets")

    try:
        buckets = pd.qcut(state.loc[available], q=n_buckets, duplicates="drop")
    except ValueError as exc:
        raise ValueError("state values cannot be divided into distinct buckets") from exc
    if len(buckets.cat.categories) < 2:
        raise ValueError("state values must form at least two distinct buckets")

    records: list[dict[str, object]] = []
    for bucket in buckets.cat.categories:
        dates = buckets.index[buckets == bucket]
        values = future_simple.loc[dates]
        for driver in panel.columns:
            series = values[driver]
            records.append(
                {
                    "state_name": state_name,
                    "state_bucket": str(bucket),
                    "driver": driver,
                    "horizon_months": horizon_months,
                    "n_observations": int(len(series)),
                    "mean_forward_return": series.mean(),
                    "median_forward_return": series.median(),
                    "q05_forward_return": series.quantile(0.05),
                }
            )
    return pd.DataFrame.from_records(records)


def _validate_quantile(quantile: float) -> None:
    if not 0.0 < quantile < 1.0:
        raise ValueError("quantile must be strictly between 0 and 1")
