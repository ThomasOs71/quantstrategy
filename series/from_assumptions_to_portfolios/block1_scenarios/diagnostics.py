"""Empirical diagnostics for Article 2 historical return building blocks.

Ljung-Box tests joint zero *linear* autocorrelation through a selected lag;
it is stronger than visually inspecting individual ACF bars, but neither a
general independence test nor proof of i.i.d. on non-rejection.

Schweizer-Wolff measures bivariate lagged dependence through the distance of
the empirical copula from the independence copula. It can detect nonlinear
dependence missed by Pearson correlation, but it does not test stationarity,
identical distributions, or general process invariance.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from statsmodels.stats.diagnostic import acorr_ljungbox


def _validate_panel(panel: pd.DataFrame) -> None:
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    if panel.empty:
        raise ValueError("panel must not be empty")
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise TypeError("panel index must be a DatetimeIndex")
    if not np.isfinite(panel.to_numpy(dtype=float)).all():
        raise ValueError("panel must contain only finite values")


def distribution_summary(
    panel: pd.DataFrame,
    *,
    periods_per_year: int = 12,
    period_label: str = "monthly",
) -> pd.DataFrame:
    """Summarise periodic log-return distributions by driver."""
    _validate_panel(panel)
    if not isinstance(periods_per_year, int) or periods_per_year < 1:
        raise ValueError("periods_per_year must be a positive integer")
    if not isinstance(period_label, str) or not period_label.isidentifier():
        raise ValueError("period_label must be a valid identifier")
    summary = pd.DataFrame(
        {
            f"mean_{period_label}": panel.mean(),
            f"std_{period_label}": panel.std(ddof=1),
            "annualized_mean_log": panel.mean() * periods_per_year,
            "annualized_volatility": panel.std(ddof=1)
            * np.sqrt(float(periods_per_year)),
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


def return_autocorrelation(
    panel: pd.DataFrame,
    lags: int | Sequence[int] = 12,
) -> pd.DataFrame:
    """Return selected lagged autocorrelations of raw returns in tidy form."""
    return _autocorrelation_table(panel, lags=lags, transform="raw_return")


def squared_return_autocorrelation(
    panel: pd.DataFrame,
    lags: int | Sequence[int] = 12,
) -> pd.DataFrame:
    """Return autocorrelations of squared demeaned returns as a volatility proxy."""
    _validate_panel(panel)
    centered_squared = panel.subtract(panel.mean(), axis="columns").pow(2)
    return _autocorrelation_table(
        centered_squared,
        lags=lags,
        transform="squared_demeaned_return",
    )


def ljung_box_diagnostics(
    panel: pd.DataFrame,
    lags: tuple[int, ...] = (6, 12),
) -> pd.DataFrame:
    """Calculate raw and squared-return Ljung–Box diagnostics by driver.

    The result is a supplementary diagnostic table, not a multiple-testing
    inference panel. Squared returns are demeaned before squaring, matching the
    Article-2 volatility-persistence ACF diagnostic.
    """
    _validate_panel(panel)
    if not lags or any(not isinstance(lag, int) or lag < 1 for lag in lags):
        raise ValueError("lags must contain positive integers")
    if max(lags) >= len(panel):
        raise ValueError("lags must be smaller than the number of observations")

    transforms = {
        "raw_return": panel,
        "squared_demeaned_return": panel.subtract(panel.mean(), axis="columns").pow(2),
    }
    records: list[dict[str, object]] = []
    for transform, values in transforms.items():
        for driver in values.columns:
            result = acorr_ljungbox(values[driver], lags=list(lags), return_df=True)
            for lag, row in result.iterrows():
                records.append(
                    {
                        "driver": driver,
                        "transform": transform,
                        "lag": int(lag),
                        "ljung_box_q": float(row["lb_stat"]),
                        "p_value": float(row["lb_pvalue"]),
                        "n_observations": len(values[driver]),
                    }
                )
    return pd.DataFrame.from_records(records)


def ljung_box_summary(panel: pd.DataFrame, lag: int = 12) -> pd.DataFrame:
    """Return publication-facing Q(lag) Ljung-Box results with BH FDR control.

    The Benjamini-Hochberg adjustment is applied separately to raw returns and
    squared demeaned returns. Rejection is evidence of serial dependence of the
    selected type through ``lag``; non-rejection does not establish i.i.d.
    """
    _validate_panel(panel)
    if not isinstance(lag, int) or lag < 1 or lag >= len(panel):
        raise ValueError("lag must be a positive integer smaller than the panel length")

    transforms = {
        "raw_return": panel,
        "squared_demeaned_return": panel.subtract(panel.mean(), axis="columns").pow(2),
    }
    records: list[dict[str, object]] = []
    for transform, values in transforms.items():
        for driver in values.columns:
            result = acorr_ljungbox(
                values[driver],
                lags=[lag],
                model_df=0,
                return_df=True,
            ).iloc[0]
            records.append(
                {
                    "driver": driver,
                    "transform": transform,
                    "lag": lag,
                    "n_observations": len(values[driver]),
                    "lb_stat": float(result["lb_stat"]),
                    "lb_pvalue": float(result["lb_pvalue"]),
                }
            )

    output = pd.DataFrame.from_records(records)
    output["lb_pvalue_fdr"] = np.nan
    for transform, indices in output.groupby("transform", sort=False).groups.items():
        output.loc[indices, "lb_pvalue_fdr"] = benjamini_hochberg(
            output.loc[indices, "lb_pvalue"].to_numpy()
        )
    output["reject_raw_5pct"] = output["lb_pvalue"] <= 0.05
    output["reject_fdr_5pct"] = output["lb_pvalue_fdr"] <= 0.05
    return output[
        [
            "driver",
            "transform",
            "lag",
            "n_observations",
            "lb_stat",
            "lb_pvalue",
            "lb_pvalue_fdr",
            "reject_raw_5pct",
            "reject_fdr_5pct",
        ]
    ]


def benjamini_hochberg(pvalues: np.ndarray | list[float]) -> np.ndarray:
    """Apply deterministic Benjamini-Hochberg FDR adjustment to finite p-values."""
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or len(values) == 0:
        raise ValueError("pvalues must be a non-empty one-dimensional array")
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("pvalues must be finite values between zero and one")

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    adjusted_sorted = sorted_values * len(values) / np.arange(1, len(values) + 1)
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = np.clip(adjusted_sorted, 0.0, 1.0)
    return adjusted


def schweizer_wolff_dependence(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
    grid_size: int = 100,
) -> float:
    """Estimate p=1 Schweizer-Wolff dependence from a regular empirical-copula grid."""
    x_values, y_values = _validate_pair(x, y)
    _validate_grid_size(grid_size)
    return _schweizer_wolff_from_pseudo(
        _pseudo_observations(x_values),
        _pseudo_observations(y_values),
        grid_size,
    )


def lagged_schweizer_wolff_diagnostics(
    panel: pd.DataFrame,
    max_lag: int = 12,
    grid_size: int = 100,
    n_permutations: int = 999,
    seed: int = 42,
    *,
    lags: Sequence[int] | None = None,
) -> pd.DataFrame:
    """Estimate lagged Schweizer-Wolff dependence with permutation references.

    The permutation p-values are approximate independence references. They are
    not time-series invariance or i.i.d. tests. BH FDR adjustment spans all
    driver-lag comparisons in the explicitly selected lag family.
    """
    _validate_panel(panel)
    selected_lags = _normalize_lags(
        max_lag if lags is None else lags,
        n_observations=len(panel),
    )
    _validate_grid_size(grid_size)
    if not isinstance(n_permutations, int) or n_permutations < 1:
        raise ValueError("n_permutations must be a positive integer")
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer")

    generator = np.random.default_rng(seed)
    records: list[dict[str, object]] = []
    for driver in panel.columns:
        values = panel[driver].to_numpy(dtype=float)
        for lag in selected_lags:
            x_values = values[lag:]
            y_values = values[:-lag]
            observed, pvalue, q95 = _schweizer_wolff_permutation_reference(
                x_values,
                y_values,
                grid_size=grid_size,
                n_permutations=n_permutations,
                generator=generator,
            )
            records.append(
                {
                    "driver": driver,
                    "lag": lag,
                    "n_observations": len(x_values),
                    "sw_dependence": observed,
                    "permutation_pvalue": pvalue,
                    "permutation_q95": q95,
                    "n_permutations": n_permutations,
                    "seed": seed,
                    "grid_size": grid_size,
                }
            )

    output = pd.DataFrame.from_records(records)
    output["permutation_pvalue_fdr"] = benjamini_hochberg(
        output["permutation_pvalue"].to_numpy()
    )
    output["reject_raw_5pct"] = output["permutation_pvalue"] <= 0.05
    output["reject_fdr_5pct"] = output["permutation_pvalue_fdr"] <= 0.05
    return output[
        [
            "driver",
            "lag",
            "n_observations",
            "sw_dependence",
            "permutation_pvalue",
            "permutation_pvalue_fdr",
            "permutation_q95",
            "reject_raw_5pct",
            "reject_fdr_5pct",
            "n_permutations",
            "seed",
            "grid_size",
        ]
    ]


def schweizer_wolff_summary(lagged_results: pd.DataFrame) -> pd.DataFrame:
    """Summarise the strongest lagged nonlinear-dependence evidence per driver."""
    required = {
        "driver",
        "lag",
        "sw_dependence",
        "permutation_pvalue",
        "permutation_pvalue_fdr",
        "reject_raw_5pct",
        "reject_fdr_5pct",
    }
    missing = required - set(lagged_results.columns)
    if missing:
        raise ValueError(
            f"lagged_results is missing required columns: {sorted(missing)}"
        )

    records: list[dict[str, object]] = []
    for driver, group in lagged_results.groupby("driver", sort=False):
        maximum = group.loc[group["sw_dependence"].idxmax()]
        records.append(
            {
                "driver": driver,
                "max_sw": float(maximum["sw_dependence"]),
                "max_sw_lag": int(maximum["lag"]),
                "min_permutation_pvalue": float(group["permutation_pvalue"].min()),
                "min_fdr_pvalue": float(group["permutation_pvalue_fdr"].min()),
                "n_lags_raw_significant": int(group["reject_raw_5pct"].sum()),
                "n_lags_fdr_significant": int(group["reject_fdr_5pct"].sum()),
            }
        )
    return pd.DataFrame.from_records(records)


def _validate_pair(
    x: np.ndarray | pd.Series,
    y: np.ndarray | pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    if x_values.ndim != 1 or y_values.ndim != 1:
        raise ValueError("x and y must be one-dimensional")
    if len(x_values) != len(y_values) or len(x_values) < 2:
        raise ValueError("x and y must have the same length of at least two")
    if not np.isfinite(x_values).all() or not np.isfinite(y_values).all():
        raise ValueError("x and y must contain only finite values")
    return x_values, y_values


def _validate_grid_size(grid_size: int) -> None:
    if not isinstance(grid_size, int) or grid_size < 2:
        raise ValueError("grid_size must be an integer of at least two")


def _pseudo_observations(values: np.ndarray) -> np.ndarray:
    return rankdata(values, method="average") / (len(values) + 1.0)


def _schweizer_wolff_from_pseudo(
    u_values: np.ndarray,
    v_values: np.ndarray,
    grid_size: int,
) -> float:
    """Evaluate the empirical-copula grid without looping over grid cells."""
    grid = np.arange(1, grid_size + 1, dtype=float) / grid_size
    # Assign an observation to the first grid cutoff that contains it. Using
    # ``side="left"`` preserves the empirical-copula definition U <= u when a
    # pseudo-observation lies exactly on a grid boundary.
    u_bins = np.searchsorted(grid, u_values, side="left")
    v_bins = np.searchsorted(grid, v_values, side="left")
    counts = np.bincount(
        u_bins * grid_size + v_bins,
        minlength=grid_size * grid_size,
    ).reshape(grid_size, grid_size)
    empirical_copula = counts.cumsum(axis=0).cumsum(axis=1) / len(u_values)
    independence_copula = np.multiply.outer(grid, grid)
    return float(12.0 * np.mean(np.abs(empirical_copula - independence_copula)))


def _schweizer_wolff_permutation_reference(
    x: np.ndarray,
    y: np.ndarray,
    *,
    grid_size: int,
    n_permutations: int,
    generator: np.random.Generator,
) -> tuple[float, float, float]:
    x_values, y_values = _validate_pair(x, y)
    u_values = _pseudo_observations(x_values)
    v_values = _pseudo_observations(y_values)
    observed = _schweizer_wolff_from_pseudo(u_values, v_values, grid_size)
    permuted = np.empty(n_permutations, dtype=float)
    for iteration in range(n_permutations):
        permuted[iteration] = _schweizer_wolff_from_pseudo(
            u_values,
            generator.permutation(v_values),
            grid_size,
        )
    pvalue = (1.0 + float(np.count_nonzero(permuted >= observed))) / (
        n_permutations + 1.0
    )
    return observed, pvalue, float(np.quantile(permuted, 0.95))


def _autocorrelation_table(
    panel: pd.DataFrame,
    lags: int | Sequence[int],
    transform: str,
) -> pd.DataFrame:
    _validate_panel(panel)
    selected_lags = _normalize_lags(lags, n_observations=len(panel))

    records: list[dict[str, object]] = []
    for driver in panel.columns:
        series = panel[driver]
        for lag in selected_lags:
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


def _normalize_lags(
    lags: int | Sequence[int],
    *,
    n_observations: int,
) -> tuple[int, ...]:
    """Validate an inclusive maximum lag or an explicit ordered lag family."""
    if isinstance(lags, int):
        selected = tuple(range(1, lags + 1)) if lags >= 1 else ()
    elif isinstance(lags, Sequence) and not isinstance(lags, (str, bytes)):
        selected = tuple(lags)
    else:
        selected = ()
    if (
        not selected
        or any(not isinstance(lag, int) or lag < 1 for lag in selected)
        or tuple(sorted(set(selected))) != selected
    ):
        raise ValueError("lags must be positive, unique integers in ascending order")
    if selected[-1] >= n_observations:
        raise ValueError("lags must be smaller than the number of observations")
    return selected


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
    """Summarise periods with simultaneous lower-tail events across drivers."""
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
    if horizon_months < 1 or horizon_months >= len(panel):
        raise ValueError("horizon_months must be between 1 and len(panel) - 1")
    result = conditional_forward_period_return_summary(
        panel,
        state,
        horizon_periods=horizon_months,
        period_unit="months",
        n_buckets=n_buckets,
        state_name=state_name,
    )
    return result.rename(columns={"horizon_periods": "horizon_months"}).drop(
        columns="period_unit"
    )


def conditional_forward_period_return_summary(
    panel: pd.DataFrame,
    state: pd.Series,
    *,
    horizon_periods: int,
    period_unit: str,
    n_buckets: int = 3,
    state_name: str = "state",
) -> pd.DataFrame:
    """Describe following-period returns by a state observed at time ``t``.

    The forward sum contains exactly ``t+1`` through ``t+horizon_periods``.
    Full-sample state buckets make this a retrospective descriptive diagnostic,
    not an operational no-look-ahead signal.
    """
    _validate_panel(panel)
    if not isinstance(state, pd.Series):
        raise TypeError("state must be a pandas Series")
    if horizon_periods < 1 or horizon_periods >= len(panel):
        raise ValueError("horizon_periods must be between 1 and len(panel) - 1")
    if not isinstance(period_unit, str) or not period_unit:
        raise ValueError("period_unit must be a non-empty string")
    if n_buckets < 2:
        raise ValueError("n_buckets must be at least 2")

    state = pd.to_numeric(state.reindex(panel.index), errors="coerce")
    future_log = (
        panel.shift(-1)
        .rolling(window=horizon_periods, min_periods=horizon_periods)
        .sum()
        .shift(-(horizon_periods - 1))
    )
    future_simple = np.expm1(future_log)
    available = state.notna() & future_simple.notna().all(axis=1)
    if available.sum() < n_buckets:
        raise ValueError("not enough state and forward-return observations for buckets")

    try:
        buckets = pd.qcut(state.loc[available], q=n_buckets, duplicates="drop")
    except ValueError as exc:
        raise ValueError(
            "state values cannot be divided into distinct buckets"
        ) from exc
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
                    "horizon_periods": horizon_periods,
                    "period_unit": period_unit,
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
