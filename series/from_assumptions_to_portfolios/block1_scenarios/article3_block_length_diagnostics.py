"""Target-specific block-length diagnostics; no scenario-model calibration.

All inputs are finite decimal log returns. The outer stationary bootstrap is an
uncertainty diagnostic for the historical sample, not the non-circular MBB used
to construct Set A. No scenario observations are treated as independent data.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import lru_cache
from numbers import Integral
from typing import Any

import numpy as np
import pandas as pd

from series.from_assumptions_to_portfolios.block1_scenarios.article3_diagnostics import (
    SELECTED_CORRELATION_PAIRS,
    ensemble_acf,
    weighted_correlation,
)
from series.from_assumptions_to_portfolios.block1_scenarios.block_sampler import (
    expected_source_weights,
)


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not bool")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return int(value)


def _matrix(values: np.ndarray, keys: Sequence[str] | None = None) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or min(array.shape) == 0 or not np.isfinite(array).all():
        raise ValueError("input must be a non-empty finite observation-by-driver matrix")
    if keys is not None and (len(keys) != array.shape[1] or len(set(keys)) != len(keys)):
        raise ValueError("unique ordered keys must match the driver dimension")
    return array


def _paths(values: np.ndarray, keys: Sequence[str]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 3 or min(array.shape) == 0 or not np.isfinite(array).all():
        raise ValueError("paths must be a non-empty finite scenario-by-step-by-driver array")
    if len(keys) != array.shape[2] or len(set(keys)) != len(keys):
        raise ValueError("unique ordered keys must match the driver dimension")
    return array


def stationary_resample_indices(
    n_obs: int, *, mean_block_length: float, rng: np.random.Generator
) -> np.ndarray:
    """Synchronized outer-bootstrap indices with geometric lengths and wrap.

    Each position starts afresh with probability 1 / mean_block_length; otherwise
    its predecessor's source index is advanced circularly. This is deliberately
    a different sampler from Set A and is never returned as a ScenarioSet.
    """
    n = _positive_int("n_obs", n_obs)
    if not np.isfinite(mean_block_length) or mean_block_length < 1:
        raise ValueError("mean_block_length must be finite and at least one")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    restart = rng.random(n) < 1.0 / mean_block_length
    restart[0] = True
    starts = rng.integers(0, n, size=n)
    positions = np.arange(n)
    last_restart = np.maximum.accumulate(np.where(restart, positions, 0))
    return (starts[last_restart] + positions - last_restart) % n


@lru_cache(maxsize=128)
def _mbb_layout(n: int, horizon: int, length: int, max_lag: int):
    weights = expected_source_weights(
        n_obs=n, horizon_steps=horizon, block_length_steps=length
    )
    within_counts = np.zeros((max_lag, length), dtype=np.int64)
    join_counts = np.zeros_like(within_counts)
    for lag in range(1, max_lag + 1):
        steps = np.arange(horizon - lag)
        offsets = steps % length
        same = steps // length == (steps + lag) // length
        within_counts[lag - 1] = np.bincount(offsets[same], minlength=length)
        join_counts[lag - 1] = np.bincount(offsets[~same], minlength=length)
    return weights, within_counts, join_counts


def expected_mbb_acf(
    signal: np.ndarray, *, horizon_steps: int, block_length_steps: int, max_lag: int
) -> np.ndarray:
    """Exact infinite-ensemble lag-pair ACF for the actual Set-A sampler.

    With M = T-L+1 starts, the mean at block offset p is the mean of source
    rows p:p+M. Within-block pairs share a start; across-block pairs have
    independent starts, but nonzero offset means can still affect their
    covariance. Pooled centering, source-edge weighting, pair counts and a
    truncated final block are all included. This is not the approximation
    rho(k)*(1-k/L), nor a finite-path, individually centered ACF estimator.
    """
    values = _matrix(signal)
    horizon = _positive_int("horizon_steps", horizon_steps)
    length = _positive_int("block_length_steps", block_length_steps)
    lag_count = _positive_int("max_lag", max_lag)
    if length > len(values) or lag_count >= horizon:
        raise ValueError("L must not exceed sample length and max_lag must be below horizon")
    weights, within, joins = _mbb_layout(len(values), horizon, length, lag_count)
    centered = values - weights @ values
    variance = weights @ (centered * centered)
    # Summation round-off must not turn an exactly constant column into an
    # apparent nonzero variance. Do not threshold genuinely small cash returns.
    variance[np.ptp(values, axis=0) == 0] = 0.0
    library = len(values) - length + 1
    prefix = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(centered, axis=0)))
    offset = np.arange(length)
    offset_means = (prefix[offset + library] - prefix[offset]) / library
    output = np.full((values.shape[1], lag_count), np.nan)
    for lag in range(1, lag_count + 1):
        covariance_sum = np.sum(
            joins[lag - 1, :, None] * offset_means * offset_means[(offset + lag) % length],
            axis=0,
        )
        if lag < length:
            product = centered[:-lag] * centered[lag:]
            cumulative = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(product, axis=0)))
            p = np.arange(length - lag)
            mean_product = (cumulative[p + library] - cumulative[p]) / library
            covariance_sum += (within[lag - 1, :length - lag, None] * mean_product).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            output[:, lag - 1] = covariance_sum / (horizon - lag) / variance
    output[~np.isfinite(output)] = np.nan
    return output


def _offdiag_mae(left: np.ndarray, right: np.ndarray) -> float:
    upper = np.triu_indices_from(left, k=1)
    differences = np.abs(left[upper] - right[upper])
    return float(np.nanmean(differences)) if np.isfinite(differences).any() else np.nan


def analytic_preservation_scores(
    history: np.ndarray, keys: Sequence[str], *, lengths: Sequence[int],
    horizon_steps: int, max_lag: int = 13,
) -> pd.DataFrame:
    """Population preservation errors conditional on one historical panel.

    Errors compare a model implied by this panel with diagnostics of the same
    panel. They measure preservation, not validation against future outcomes.
    """
    values = _matrix(history, keys)
    if max_lag >= len(values):
        raise ValueError("max_lag must be below historical sample length")
    non_cash = [i for i, key in enumerate(keys) if key != "cash"]
    if len(non_cash) < 2:
        raise ValueError("at least two non-cash drivers are required")
    correlation = weighted_correlation(values)
    signals = {"raw": values, "squared_demeaned": (values - values.mean(axis=0)) ** 2}
    targets = {
        name: ensemble_acf(signal[None], max_lag=max_lag, transform="raw")[0]
        for name, signal in signals.items()
    }
    records: list[dict[str, Any]] = []
    for length in lengths:
        weights, _, _ = _mbb_layout(len(values), horizon_steps, length, max_lag)
        implied_corr = weighted_correlation(values, weights)
        for group, positions in (("all12", list(range(len(keys)))), ("non_cash11", non_cash)):
            subset = np.ix_(positions, positions)
            records.append({"metric": f"weekly_correlation_mae_{group}", "L": length,
                            "value": _offdiag_mae(implied_corr[subset], correlation[subset])})
        for transform, signal in signals.items():
            implied = expected_mbb_acf(
                signal, horizon_steps=horizon_steps, block_length_steps=length, max_lag=max_lag
            )
            error = (implied[non_cash] - targets[transform][non_cash]) ** 2
            records.append({"metric": f"{transform}_acf_rmse_non_cash11_lags1_{max_lag}",
                            "L": length, "value": float(np.sqrt(np.nanmean(error)))})
    return pd.DataFrame.from_records(records)


def source_uncertainty_diagnostics(
    history: np.ndarray, keys: Sequence[str], *, lengths: Sequence[int], horizon_steps: int,
    mean_block_lengths: Sequence[int], n_resamples: int, root_seed: int,
    max_lag: int = 13, contrasts: Sequence[tuple[int, int]] = ((4, 1), (4, 8), (4, 13)),
    confidence: float = 0.95, progress=None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Outer time-series bootstrap percentile intervals, separated from MC.

    All columns use the same resampled indices. The 13/26/52-week outer block
    choices are a sensitivity check, not a calibration to the L=4 reference.
    Contrast draws subtract scores in the SAME outer resample. These intervals
    are exploratory, not simultaneous or out-of-sample hypothesis tests.
    """
    values = _matrix(history, keys)
    _positive_int("n_resamples", n_resamples)
    _positive_int("root_seed", root_seed)
    if not 0 < confidence < 1 or not mean_block_lengths:
        raise ValueError("confidence must lie in (0,1) and outer block lengths must be non-empty")
    if any(left not in lengths or right not in lengths for left, right in contrasts):
        raise ValueError("all contrast block lengths must be in the grid")
    baseline = analytic_preservation_scores(
        values, keys, lengths=lengths, horizon_steps=horizon_steps, max_lag=max_lag
    )
    metric_names = baseline.metric.unique()
    pair_positions = [(keys.index(left), keys.index(right), f"{left}|{right}")
                      for left, right in SELECTED_CORRELATION_PAIRS
                      if left in keys and right in keys]

    def score_map(sample: np.ndarray) -> dict[tuple[str, int, int], float]:
        scores = analytic_preservation_scores(
            sample, keys, lengths=lengths, horizon_steps=horizon_steps, max_lag=max_lag
        )
        lookup = {(r.metric, int(r.L)): float(r.value) for r in scores.itertuples()}
        result = {(name, length, 0): lookup[name, length] for name in metric_names for length in lengths}
        for name in metric_names:
            for left, right in contrasts:
                result[name, left, right] = lookup[name, left] - lookup[name, right]
        corr = weighted_correlation(sample)
        for left, right, pair in pair_positions:
            result[f"historical_correlation_{pair}", 0, 0] = float(corr[left, right])
        return result

    point = score_map(values)
    draws: list[dict[str, Any]] = []
    for outer_length in mean_block_lengths:
        _positive_int("outer_mean_block_length", outer_length)
        for replicate in range(n_resamples):
            rng = np.random.default_rng(np.random.SeedSequence([root_seed, 301, outer_length, replicate]))
            indices = stationary_resample_indices(len(values), mean_block_length=outer_length, rng=rng)
            for (metric, length, comparator), value in score_map(values[indices]).items():
                draws.append({"outer_mean_block_length": outer_length, "resample_id": replicate,
                              "metric": metric, "L": length, "comparator_L": comparator, "value": value})
            if progress is not None and (replicate + 1) % 100 == 0:
                progress(f"Outer uncertainty: mean length {outer_length}, {replicate + 1}/{n_resamples}")
        if progress is not None:
            progress(f"Outer uncertainty: mean length {outer_length}, {n_resamples} resamples complete")
    frame = pd.DataFrame.from_records(draws)
    summaries = []
    alpha = (1 - confidence) / 2
    groups = ["outer_mean_block_length", "metric", "L", "comparator_L"]
    for identity, group in frame.groupby(groups, sort=False):
        finite = group.value.to_numpy(dtype=float)
        finite = finite[np.isfinite(finite)]
        metric, length, comparator = identity[1:]
        lower, upper = np.quantile(finite, [alpha, 1 - alpha], method="linear") if len(finite) else (np.nan, np.nan)
        summaries.append(dict(zip(groups, identity)) | {
            "point_estimate": point[metric, length, comparator],
            "bootstrap_mean": float(finite.mean()) if len(finite) else np.nan,
            "bootstrap_std": float(finite.std(ddof=1)) if len(finite) > 1 else np.nan,
            "ci_lower": lower, "ci_upper": upper, "confidence": confidence,
            "n_finite": len(finite), "n_resamples": n_resamples,
            "interval_method": "stationary_outer_bootstrap_percentile_exploratory",
            "status": "ok" if len(finite) == n_resamples else "undefined_resamples_reported",
        })
    return frame, pd.DataFrame.from_records(summaries)


def ljung_box_diagnostics(
    history: np.ndarray, keys: Sequence[str], *, lags: Sequence[int] = (4, 13, 52), alpha: float = 0.05
) -> pd.DataFrame:
    """Conventional Ljung-Box checks, with Holm correction within transform.

    Cash is a separate descriptive family. Chi-square calibration requires its
    usual time-series regularity assumptions and is not robust evidence of a
    stationary volatility regime. Squared-return tests concern dependence in
    squared demeaned returns, not raw-return predictability.
    """
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.stats.multitest import multipletests

    values = _matrix(history, keys)
    if not lags or any(_positive_int("lag", lag) >= len(values) for lag in lags):
        raise ValueError("lags must be positive and below sample length")
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    rows = []
    for transform, signal in (("raw", values), ("squared_demeaned", (values - values.mean(axis=0)) ** 2)):
        for i, key in enumerate(keys):
            family = "cash_separate" if key == "cash" else "non_cash11"
            constant = np.ptp(signal[:, i]) == 0
            result = None if constant else acorr_ljungbox(signal[:, i], lags=list(lags), model_df=0, return_df=True)
            for lag in lags:
                rows.append({"driver": key, "transform": transform, "lag": lag,
                             "family": family, "n_observations": len(values),
                             "q_statistic": np.nan if constant else float(result.loc[lag, "lb_stat"]),
                             "p_value": np.nan if constant else float(result.loc[lag, "lb_pvalue"]),
                             "status": "undefined_zero_variance" if constant else "ok"})
    frame = pd.DataFrame.from_records(rows)
    frame["p_holm"] = np.nan
    frame["reject_holm"] = False
    frame["family_size"] = 0
    frame["alpha"] = alpha
    for _, group in frame.groupby(["transform", "family"]):
        index = group.index[np.isfinite(group.p_value)]
        frame.loc[group.index, "family_size"] = len(index)
        if len(index):
            rejected, corrected, _, _ = multipletests(frame.loc[index, "p_value"], alpha=alpha, method="holm")
            frame.loc[index, "p_holm"] = corrected
            frame.loc[index, "reject_holm"] = rejected
    return frame


def automatic_block_length_diagnostics(history: np.ndarray, keys: Sequence[str]) -> pd.DataFrame:
    """Corrected Politis-White plug-in lengths via arch, not MBB/path optima.

    The selector targets long-run variance of a scalar mean. Raw and squared
    returns represent different targets. The correlation influence function
    z_i*z_j - rho*(z_i**2 + z_j**2)/2 includes the denominator's variation;
    centered cross-products alone would only represent covariance.
    """
    try:
        import arch
        from arch.bootstrap import optimal_block_length
    except ImportError as exc:
        raise RuntimeError("Block-length selector requires the optional research dependency arch>=7.2; no substitute is used") from exc
    values = _matrix(history, keys)
    centered = values - values.mean(axis=0)
    signals = [("raw", key, "long_run_variance_of_mean_log_return", values[:, i]) for i, key in enumerate(keys)]
    signals += [("squared_demeaned", key, "long_run_variance_of_mean_squared_demeaned_return", centered[:, i] ** 2)
                for i, key in enumerate(keys)]
    std = np.sqrt((centered * centered).mean(axis=0))
    for left, right in SELECTED_CORRELATION_PAIRS:
        if left not in keys or right not in keys:
            continue
        i, j = keys.index(left), keys.index(right)
        if std[i] == 0 or std[j] == 0:
            signal = np.zeros(len(values))
        else:
            zi, zj = centered[:, i] / std[i], centered[:, j] / std[j]
            rho = float(np.mean(zi * zj))
            signal = zi * zj - 0.5 * rho * (zi * zi + zj * zj)
        signals.append(("correlation_influence", f"{left}|{right}", "long_run_variance_of_Pearson_correlation_influence", signal))
    rows = []
    for transform, key, target, signal in signals:
        constant = np.ptp(signal) == 0
        result = None if constant else optimal_block_length(signal).iloc[0]
        rows.append({"transform": transform, "driver_or_pair": key, "target": target,
                     "family": "cash_separate" if key == "cash" else "non_cash",
                     "stationary_length": np.nan if constant else float(result.stationary),
                     "circular_length": np.nan if constant else float(result.circular),
                     "n_observations": len(values), "cube_root_n": float(np.cbrt(len(values))),
                     "cube_root_floor": int(np.floor(np.cbrt(len(values)))),
                     "method": "Politis_White_2004_Patton_Politis_White_2009_corrected",
                     "implementation": f"arch.bootstrap.optimal_block_length; arch={arch.__version__}",
                     "status": "undefined_zero_variance" if constant else "ok",
                     "interpretation": "continuous stationary/CBB recommendations; not a calibrated fixed-MBB or drawdown optimum"})
    return pd.DataFrame.from_records(rows)


def aggregate_log_paths(paths: np.ndarray, endpoints: Sequence[int]) -> np.ndarray:
    """Sum log returns to retained endpoints, preserving terminal wealth exactly."""
    array = np.asarray(paths, dtype=np.float64)
    if array.ndim != 3 or min(array.shape) == 0 or not np.isfinite(array).all():
        raise ValueError("paths must be a non-empty finite 3D array")
    ends = np.asarray(endpoints)
    if (ends.ndim != 1 or len(ends) == 0 or not np.issubdtype(ends.dtype, np.integer)
            or ends[0] <= 0 or np.any(np.diff(ends) <= 0) or ends[-1] != array.shape[1]):
        raise ValueError("endpoints must be increasing positive integer steps ending at the path horizon")
    cumulative = np.concatenate((np.zeros((len(array), 1, array.shape[2])), np.cumsum(array, axis=1)), axis=1)
    retained = cumulative[:, np.r_[0, ends], :]
    return np.diff(retained, axis=1)


def four_week_endpoints(horizon_steps: int) -> np.ndarray:
    horizon = _positive_int("horizon_steps", horizon_steps)
    return np.unique(np.r_[np.arange(4, horizon + 1, 4), horizon])


def last_friday_month_endpoints(dates: pd.DatetimeIndex) -> np.ndarray:
    """Retain the last observed Friday per calendar month, including final step.

    These are month-labelled endpoints on the original W-FRI grid, NOT true
    daily calendar-month-end returns. No daily data are inferred or rebuilt.
    """
    index = pd.DatetimeIndex(dates)
    if (len(index) == 0 or index.hasnans or not index.is_monotonic_increasing
            or index.has_duplicates or index.tz is not None
            or (index.dayofweek != 4).any() or (np.diff(index.asi8) != pd.Timedelta(days=7).value).any()):
        raise ValueError("dates must be a complete, timezone-naive, strictly increasing W-FRI index")
    periods = index.to_period("M")
    return np.flatnonzero(np.r_[periods[:-1] != periods[1:], True]) + 1


def maximum_drawdown(paths: np.ndarray) -> np.ndarray:
    """Non-negative wealth drawdown including the initial wealth of one."""
    array = np.asarray(paths, dtype=np.float64)
    if array.ndim != 3 or min(array.shape) == 0 or not np.isfinite(array).all():
        raise ValueError("paths must be a non-empty finite 3D array")
    wealth = np.concatenate((np.zeros((len(array), 1, array.shape[2])), np.cumsum(array, axis=1)), axis=1)
    peak = np.maximum.accumulate(wealth, axis=1)
    return (-np.expm1(wealth - peak)).max(axis=1)


def resolution_diagnostics(
    paths: np.ndarray, keys: Sequence[str], *, endpoints: Sequence[int], grid_name: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Paired drawdown and correlation effects of coarsening the SAME paths."""
    array = _paths(paths, keys)
    coarse = aggregate_log_paths(array, endpoints)
    fine_mdd, coarse_mdd = maximum_drawdown(array), maximum_drawdown(coarse)
    gaps = fine_mdd - coarse_mdd
    if np.min(gaps) < -1e-12:
        raise RuntimeError("retained-grid drawdown must not exceed full-grid drawdown")
    terminal_difference = np.max(np.abs(array.sum(axis=1) - coarse.sum(axis=1)), axis=0)
    records = []
    for i, key in enumerate(keys):
        for metric, value in (
            ("weekly_mdd_q95", np.quantile(fine_mdd[:, i], .95, method="linear")),
            ("coarse_mdd_q95", np.quantile(coarse_mdd[:, i], .95, method="linear")),
            ("paired_mdd_gap_mean", gaps[:, i].mean()),
            ("paired_mdd_gap_q95", np.quantile(gaps[:, i], .95, method="linear")),
            ("fraction_mdd_understated_gt_1pp", np.mean(gaps[:, i] > .01)),
            ("max_abs_terminal_log_difference", terminal_difference[i]),
        ):
            records.append({"grid": grid_name, "asset": key, "metric": metric, "value": float(value),
                            "n_paths": len(array), "n_retained_steps": coarse.shape[1],
                            "weekly_horizon_steps": array.shape[1], "unit": "decimal_or_probability"})
    fine_corr = weighted_correlation(array.reshape(-1, len(keys)))
    coarse_corr = weighted_correlation(coarse.reshape(-1, len(keys)))
    pairs = [{"grid": grid_name, "driver_i": keys[i], "driver_j": keys[j],
              "weekly_correlation": fine_corr[i, j], "coarse_correlation": coarse_corr[i, j],
              "coarse_minus_weekly": coarse_corr[i, j] - fine_corr[i, j],
              "status": "ok" if np.isfinite(fine_corr[i, j] + coarse_corr[i, j]) else "undefined_zero_variance",
              "n_retained_steps": coarse.shape[1]}
             for i in range(len(keys)) for j in range(i + 1, len(keys))]
    return pd.DataFrame.from_records(records), pd.DataFrame.from_records(pairs)


def terminal_correlation_rows(
    paths: np.ndarray, keys: Sequence[str], *, horizons: Sequence[int]
) -> pd.DataFrame:
    """Across-path terminal correlations; log sums and simple wealth returns."""
    array = _paths(paths, keys)
    records = []
    for horizon in horizons:
        horizon = _positive_int("terminal_horizon", horizon)
        if horizon > array.shape[1]:
            raise ValueError("terminal horizon must not exceed path length")
        total = array[:, :horizon].sum(axis=1)
        for kind, values in (("cumulative_log", total), ("simple_compounded", np.expm1(total))):
            corr = weighted_correlation(values)
            records += [{"horizon_steps": horizon, "return_type": kind,
                         "driver_i": keys[i], "driver_j": keys[j], "correlation": corr[i, j],
                         "n_paths": len(array),
                         "status": "ok" if np.isfinite(corr[i, j]) else "undefined_zero_variance"}
                        for i in range(len(keys)) for j in range(i + 1, len(keys))]
    return pd.DataFrame.from_records(records)
