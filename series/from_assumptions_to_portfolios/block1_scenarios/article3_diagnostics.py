"""Numerical, plotting-independent diagnostics for Article-3 Set A."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any

import numpy as np
import pandas as pd

from data.asset_universe import get_asset_keys
from series.from_assumptions_to_portfolios.block1_scenarios.block_sampler import (
    block_join_metrics,
    expected_source_weights,
    lag_within_block_fraction,
    realized_source_weights,
)
from series.from_assumptions_to_portfolios.block1_scenarios.terminal_returns import (
    annualize_terminal_return,
    build_terminal_returns_steps,
)


TAIL_GROUPS: Mapping[str, tuple[str, ...]] = {
    "article2_all12": (),
    "investable_ex_cash10": tuple(key for key in get_asset_keys() if key != "cash"),
}
SELECTED_ASSETS = (
    "global_dm_ex_emu",
    "euro_govt_bond_7_10",
    "euro_high_yield",
    "gold",
)
SELECTED_CORRELATION_PAIRS = (
    ("global_dm_ex_emu", "euro_govt_bond_7_10"),
    ("global_dm_ex_emu", "euro_high_yield"),
    ("global_dm_ex_emu", "commodities"),
)


def _as_float_matrix(values: np.ndarray, *, ndim: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != ndim:
        raise ValueError(f"expected {ndim} dimensions, received {array.ndim}")
    if not np.isfinite(array).all():
        raise ValueError("diagnostic input must contain only finite values")
    return array


def _normalized_weights(weights: np.ndarray | None, n: int) -> np.ndarray:
    if weights is None:
        return np.full(n, 1.0 / n, dtype=np.float64)
    parsed = np.asarray(weights, dtype=np.float64)
    if parsed.shape != (n,) or not np.isfinite(parsed).all() or np.any(parsed < 0):
        raise ValueError("weights must be finite, non-negative, and match observations")
    total = parsed.sum()
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-12):
        raise ValueError("weights must sum to one within 1e-12")
    return parsed


def _weighted_quantile_inverse_cdf(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    cumulative = np.cumsum(weights[order])
    return float(ordered_values[min(np.searchsorted(cumulative, q, side="left"), len(ordered_values) - 1)])


def _population_moment_rows(
    values: np.ndarray,
    weights: np.ndarray,
    *,
    quantile_mode: str,
) -> list[tuple[str, float | None, str]]:
    mean = float(np.dot(weights, values))
    centered = values - mean
    variance = float(np.dot(weights, centered * centered))
    std = float(np.sqrt(variance))
    rows: list[tuple[str, float | None, str]] = [
        ("mean_weekly_log_return", mean, "decimal_log_return"),
        ("std_weekly", std, "decimal_log_return"),
        ("annualized_log_rate_52x", 52.0 * mean, "decimal_log_return_per_year"),
        ("annualized_volatility_sqrt52", np.sqrt(52.0) * std, "decimal_per_year"),
        ("minimum", float(np.min(values)), "decimal_log_return"),
        ("maximum", float(np.max(values)), "decimal_log_return"),
    ]
    if std == 0.0:
        rows.extend([("skewness", None, "undefined_zero_variance"), ("excess_kurtosis", None, "undefined_zero_variance")])
    else:
        rows.extend([
            ("skewness", float(np.dot(weights, centered ** 3) / std ** 3), "dimensionless"),
            ("excess_kurtosis", float(np.dot(weights, centered ** 4) / std ** 4 - 3.0), "dimensionless"),
        ])
    for q, name in ((0.01, "q01"), (0.05, "q05"), (0.50, "q50"), (0.95, "q95"), (0.99, "q99")):
        value = (
            float(np.quantile(values, q, method="linear"))
            if quantile_mode == "linear"
            else _weighted_quantile_inverse_cdf(values, weights, q)
        )
        rows.append((name, value, "decimal_log_return"))
    return rows


def marginal_preservation_rows(
    values: np.ndarray,
    keys: Sequence[str],
    *,
    reference: str,
    weights: np.ndarray | None = None,
    quantile_mode: str = "linear",
) -> pd.DataFrame:
    """Describe every one-period driver marginal with explicit discrete weights."""
    matrix = _as_float_matrix(values, ndim=2)
    if matrix.shape[1] != len(keys):
        raise ValueError("keys must match the second matrix dimension")
    parsed_weights = _normalized_weights(weights, len(matrix))
    records: list[dict[str, Any]] = []
    for position, key in enumerate(keys):
        for metric, value, unit in _population_moment_rows(
            matrix[:, position], parsed_weights, quantile_mode=quantile_mode
        ):
            records.append({
                "reference": reference, "driver": key, "metric": metric,
                "value": value, "unit": unit, "quantile_mode": quantile_mode,
                "status": "ok" if value is not None else "undefined_zero_variance",
            })
    return pd.DataFrame.from_records(records)


def source_occupancy_rows(
    *,
    expected_weights: np.ndarray,
    realized_weights: np.ndarray | None,
    source_dates: pd.DatetimeIndex,
    experiment: str,
    block_length_steps: int,
    replicate_id: int | None,
) -> pd.DataFrame:
    """Return expected and realized source-row occupancy without reweighting data."""
    expected = np.asarray(expected_weights, dtype=np.float64)
    if len(expected) != len(source_dates):
        raise ValueError("source_dates must match expected_weights")
    realized = None if realized_weights is None else np.asarray(realized_weights, dtype=np.float64)
    if realized is not None and realized.shape != expected.shape:
        raise ValueError("realized_weights must match expected_weights")
    uniform = 1.0 / len(expected)
    return pd.DataFrame({
        "experiment": experiment,
        "L": block_length_steps,
        "replicate_id": replicate_id,
        "source_idx": np.arange(len(expected), dtype=np.int64),
        "source_date": source_dates.strftime("%Y-%m-%d"),
        "uniform_weight": uniform,
        "expected_mbb_weight": expected,
        "realized_weight": realized,
        "realized_minus_expected": None if realized is None else realized - expected,
    })


def edge_weighting_summary(
    expected_weights: np.ndarray,
    realized_weights: np.ndarray | None,
    *,
    experiment: str,
    block_length_steps: int,
    replicate_id: int | None,
) -> pd.DataFrame:
    """Summarize non-circular edge underweighting and Monte-Carlo deviation."""
    weights = np.asarray(expected_weights, dtype=np.float64)
    uniform = 1.0 / len(weights)
    records = [
        {"experiment": experiment, "L": block_length_steps, "replicate_id": replicate_id,
         "metric": "total_variation_to_uniform", "value": 0.5 * np.abs(weights - uniform).sum(), "unit": "probability"},
        {"experiment": experiment, "L": block_length_steps, "replicate_id": replicate_id,
         "metric": "edge_weight_first_row", "value": weights[0], "unit": "probability"},
        {"experiment": experiment, "L": block_length_steps, "replicate_id": replicate_id,
         "metric": "edge_weight_last_row", "value": weights[-1], "unit": "probability"},
    ]
    if realized_weights is not None:
        realized = np.asarray(realized_weights, dtype=np.float64)
        records.extend([
            {"experiment": experiment, "L": block_length_steps, "replicate_id": replicate_id,
             "metric": "realized_l1_to_expected", "value": np.abs(realized - weights).sum(), "unit": "probability"},
            {"experiment": experiment, "L": block_length_steps, "replicate_id": replicate_id,
             "metric": "realized_max_abs_to_expected", "value": np.abs(realized - weights).max(), "unit": "probability"},
        ])
    return pd.DataFrame.from_records(records)


def weighted_correlation(values: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    """Compute a population Pearson matrix, preserving undefined zero variance."""
    matrix = _as_float_matrix(values, ndim=2)
    weights = _normalized_weights(weights, len(matrix))
    mean = weights @ matrix
    centered = matrix - mean
    covariance = (centered * weights[:, None]).T @ centered
    std = np.sqrt(np.diag(covariance))
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = covariance / np.outer(std, std)
    correlation[~np.isfinite(correlation)] = np.nan
    np.fill_diagonal(correlation, np.where(std > 0, 1.0, np.nan))
    return correlation


def correlation_matrix_rows(
    correlation: np.ndarray,
    keys: Sequence[str],
    *,
    reference: str,
    group: str,
) -> pd.DataFrame:
    """Serialize a full named correlation matrix to long-form rows."""
    matrix = np.asarray(correlation, dtype=np.float64)
    if matrix.shape != (len(keys), len(keys)):
        raise ValueError("correlation dimensions do not match keys")
    return pd.DataFrame.from_records([
        {"reference": reference, "group": group, "driver_i": left, "driver_j": right,
         "correlation": matrix[i, j], "status": "ok" if np.isfinite(matrix[i, j]) else "undefined_zero_variance"}
        for i, left in enumerate(keys) for j, right in enumerate(keys)
    ])


def correlation_error_rows(
    observed: np.ndarray,
    reference: np.ndarray,
    keys: Sequence[str],
    *,
    experiment: str,
    block_length_steps: int | None,
    replicate_id: int | None,
    group: str,
) -> pd.DataFrame:
    """Return pair errors and aggregate off-diagonal correlation error metrics."""
    observed = np.asarray(observed, dtype=np.float64)
    target = np.asarray(reference, dtype=np.float64)
    if observed.shape != target.shape or observed.shape != (len(keys), len(keys)):
        raise ValueError("correlation shapes do not match keys")
    rows: list[dict[str, Any]] = []
    errors: list[float] = []
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            error = observed[i, j] - target[i, j]
            if np.isfinite(error):
                errors.append(float(error))
            rows.append({"experiment": experiment, "L": block_length_steps, "replicate_id": replicate_id,
                         "group": group, "metric": "pair_error", "driver_i": keys[i], "driver_j": keys[j],
                         "value": error, "unit": "correlation"})
    finite = np.asarray(errors, dtype=np.float64)
    aggregates = {
        "offdiag_mae": np.mean(np.abs(finite)) if finite.size else np.nan,
        "offdiag_max_abs_pair_error": np.max(np.abs(finite)) if finite.size else np.nan,
        "offdiag_frobenius": np.sqrt(2.0 * np.sum(finite ** 2)) if finite.size else np.nan,
    }
    rows.extend({"experiment": experiment, "L": block_length_steps, "replicate_id": replicate_id,
                 "group": group, "metric": metric, "driver_i": None, "driver_j": None,
                 "value": value, "unit": "correlation"} for metric, value in aggregates.items())
    return pd.DataFrame.from_records(rows)


def tail_threshold_rows(history: np.ndarray, keys: Sequence[str], *, probability: float) -> tuple[pd.DataFrame, np.ndarray]:
    """Fix linear lower-tail thresholds once from complete historical data."""
    matrix = _as_float_matrix(history, ndim=2)
    thresholds = np.quantile(matrix, probability, axis=0, method="linear")
    counts = (matrix < thresholds).sum(axis=0)
    frame = pd.DataFrame({
        "driver": keys, "tail_probability": probability, "threshold": thresholds,
        "comparison": "lt", "historical_tail_count": counts,
        "historical_tail_rate": counts / len(matrix), "quantile_method": "linear",
    })
    return frame, thresholds


def _group_positions(keys: Sequence[str], group: str) -> list[int]:
    if group == "article2_all12":
        return list(range(len(keys)))
    requested = TAIL_GROUPS[group]
    return [list(keys).index(key) for key in requested]


def _tail_count_distribution(
    values: np.ndarray,
    thresholds: np.ndarray,
    keys: Sequence[str],
    *,
    group: str,
    weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = _as_float_matrix(values, ndim=2)
    positions = _group_positions(keys, group)
    indicators = matrix[:, positions] < thresholds[positions]
    counts = indicators.sum(axis=1)
    parsed_weights = _normalized_weights(weights, len(matrix))
    pmf = np.bincount(counts, weights=parsed_weights, minlength=len(positions) + 1)
    return counts, pmf


def tail_cooccurrence_rows(
    values: np.ndarray,
    thresholds: np.ndarray,
    keys: Sequence[str],
    *,
    reference: str,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Summarize simultaneous lower-tail counts for both prescribed groups."""
    rows: list[dict[str, Any]] = []
    for group in TAIL_GROUPS:
        counts, pmf = _tail_count_distribution(values, thresholds, keys, group=group, weights=weights)
        for k, probability in enumerate(pmf):
            rows.append({"reference": reference, "group": group, "metric": "p_k_equals", "k": k,
                         "value": probability, "unit": "probability", "n_steps": len(counts)})
        for k in range(2, len(pmf)):
            rows.append({"reference": reference, "group": group, "metric": "p_k_ge", "k": k,
                         "value": float(pmf[k:].sum()), "unit": "probability", "n_steps": len(counts)})
    return pd.DataFrame.from_records(rows)


def tail_pairwise_rows(
    values: np.ndarray,
    thresholds: np.ndarray,
    keys: Sequence[str],
    *,
    reference: str,
    weights: np.ndarray | None = None,
) -> pd.DataFrame:
    """Return all pairwise lower-tail co-occurrence probabilities."""
    matrix = _as_float_matrix(values, ndim=2)
    parsed_weights = _normalized_weights(weights, len(matrix))
    flags = matrix < thresholds
    return pd.DataFrame.from_records([
        {"reference": reference, "driver_i": keys[i], "driver_j": keys[j],
         "value": float(np.dot(parsed_weights, flags[:, i] & flags[:, j])), "unit": "probability"}
        for i in range(len(keys)) for j in range(i + 1, len(keys))
    ])


def poisson_binomial_pmf(probabilities: Iterable[float]) -> np.ndarray:
    """Compute an independent-tail count PMF by convolving Bernoulli factors."""
    pmf = np.array([1.0], dtype=np.float64)
    for probability in probabilities:
        p = float(probability)
        if not 0.0 <= p <= 1.0:
            raise ValueError("Poisson-binomial probabilities must lie in [0, 1]")
        pmf = np.convolve(pmf, np.array([1.0 - p, p]))
    if not np.isclose(pmf.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("Poisson-binomial PMF does not sum to one")
    return pmf


def tail_independence_reference_rows(
    history: np.ndarray, thresholds: np.ndarray, keys: Sequence[str]
) -> pd.DataFrame:
    """Generate the analytic IID-control tail benchmark for both tail groups."""
    matrix = _as_float_matrix(history, ndim=2)
    rows: list[dict[str, Any]] = []
    rates = (matrix < thresholds).mean(axis=0)
    for group in TAIL_GROUPS:
        positions = _group_positions(keys, group)
        pmf = poisson_binomial_pmf(rates[positions])
        for k, probability in enumerate(pmf):
            rows.append({"group": group, "metric": "p_k_equals", "k": k, "value": probability,
                         "unit": "probability", "reference": "poisson_binomial_independent"})
        for k in range(2, len(pmf)):
            rows.append({"group": group, "metric": "p_k_ge", "k": k, "value": float(pmf[k:].sum()),
                         "unit": "probability", "reference": "poisson_binomial_independent"})
    return pd.DataFrame.from_records(rows)


def ensemble_acf(
    paths: np.ndarray,
    *,
    max_lag: int,
    transform: str,
    historical_means: np.ndarray | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Compute lag-pair-normalized ACFs without cross-scenario pairs."""
    array = _as_float_matrix(paths, ndim=3)
    scenarios, horizon, drivers = array.shape
    if not 1 <= max_lag < horizon:
        raise ValueError("max_lag must be between one and horizon_steps - 1")
    if transform == "raw":
        signal = array
    elif transform == "squared_demeaned":
        means = np.asarray(historical_means, dtype=np.float64)
        if means.shape != (drivers,):
            raise ValueError("historical_means must match the driver dimension")
        signal = (array - means[None, None, :]) ** 2
    else:
        raise ValueError("transform must be 'raw' or 'squared_demeaned'")
    ensemble_mean = signal.mean(axis=(0, 1))
    centered = signal - ensemble_mean[None, None, :]
    variance = np.mean(centered ** 2, axis=(0, 1))
    output = np.full((drivers, max_lag), np.nan, dtype=np.float64)
    statuses: list[str] = []
    fft_length = 1 << (2 * horizon - 1).bit_length()
    for driver in range(drivers):
        if variance[driver] == 0.0:
            statuses.append("undefined_zero_variance")
            continue
        transformed = np.fft.rfft(centered[:, :, driver], n=fft_length, axis=1)
        autocov_sums = np.fft.irfft(transformed * transformed.conj(), n=fft_length, axis=1)[:, : max_lag + 1].sum(axis=0)
        denominators = scenarios * (horizon - np.arange(max_lag + 1))
        output[driver, :] = (autocov_sums[1:] / denominators[1:]) / variance[driver]
        statuses.append("ok")
    return output, statuses


def acf_curve_rows(
    acf: np.ndarray,
    statuses: Sequence[str],
    keys: Sequence[str],
    *,
    reference: str,
    transform: str,
) -> pd.DataFrame:
    """Serialize all named ACF curves in the common Article-3 estimator form."""
    return pd.DataFrame.from_records([
        {"reference": reference, "driver": key, "transform": transform, "lag": lag,
         "acf": acf[position, lag - 1], "status": statuses[position],
         "estimator": "ensemble_lag_pair_normalized"}
        for position, key in enumerate(keys) for lag in range(1, acf.shape[1] + 1)
    ])


def acf_error_rows(
    observed: np.ndarray,
    historical: np.ndarray,
    keys: Sequence[str],
    *,
    transform: str,
    experiment: str,
    block_length_steps: int,
    replicate_id: int | None,
    lag_sets: Sequence[int],
) -> pd.DataFrame:
    """Compute exact driver and panel RMSEs against historical ACFs."""
    records: list[dict[str, Any]] = []
    non_cash = [i for i, key in enumerate(keys) if key != "cash"]
    for max_lag in lag_sets:
        if max_lag > observed.shape[1]:
            raise ValueError("requested ACF error lag exceeds available ACFs")
        squared_errors = (observed[:, :max_lag] - historical[:, :max_lag]) ** 2
        for position, key in enumerate(keys):
            value = np.sqrt(np.nanmean(squared_errors[position]))
            records.append({"experiment": experiment, "L": block_length_steps, "replicate_id": replicate_id,
                            "transform": transform, "group": "all12" if key == "cash" else "non_cash11",
                            "driver": key, "lag_set": f"1..{max_lag}", "metric": "driver_rmse",
                            "value": value, "unit": "acf"})
        panel_value = np.sqrt(np.nanmean(squared_errors[non_cash, :]))
        records.append({"experiment": experiment, "L": block_length_steps, "replicate_id": replicate_id,
                        "transform": transform, "group": "non_cash11", "driver": None,
                        "lag_set": f"1..{max_lag}", "metric": "panel_rmse",
                        "value": panel_value, "unit": "acf"})
    return pd.DataFrame.from_records(records)


def finite_path_acf_summary_rows(
    paths: np.ndarray,
    keys: Sequence[str],
    *,
    max_lag: int,
    transform: str,
    historical_means: np.ndarray,
    reference: str,
) -> pd.DataFrame:
    """Summarize pathwise finite-horizon ACFs without scenario-boundary joins."""
    array = _as_float_matrix(paths, ndim=3)
    scenarios, horizon, drivers = array.shape
    if max_lag >= horizon:
        raise ValueError("max_lag must be less than path horizon")
    if transform == "raw":
        signal = array
    elif transform == "squared_demeaned":
        means = np.asarray(historical_means, dtype=np.float64)
        if means.shape != (drivers,):
            raise ValueError("historical_means must match paths")
        signal = (array - means[None, None, :]) ** 2
    else:
        raise ValueError("unsupported ACF transform")
    fft_length = 1 << (2 * horizon - 1).bit_length()
    records: list[dict[str, Any]] = []
    for position, key in enumerate(keys):
        centered = signal[:, :, position] - signal[:, :, position].mean(axis=1, keepdims=True)
        variance = np.mean(centered ** 2, axis=1)
        transformed = np.fft.rfft(centered, n=fft_length, axis=1)
        autocov = np.fft.irfft(transformed * transformed.conj(), n=fft_length, axis=1)[:, 1:max_lag + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            acf = (autocov / (horizon - np.arange(1, max_lag + 1))[None, :]) / variance[:, None]
        acf[~np.isfinite(acf)] = np.nan
        for lag in range(1, max_lag + 1):
            finite = acf[:, lag - 1][np.isfinite(acf[:, lag - 1])]
            for metric, quantile in (("q05", .05), ("q50", .5), ("q95", .95)):
                records.append({"reference": reference, "driver": key, "transform": transform,
                                "lag": lag, "metric": metric,
                                "value": None if not len(finite) else float(np.quantile(finite, quantile, method="linear")),
                                "n_valid_paths": len(finite),
                                "status": "ok" if len(finite) else "undefined_zero_variance"})
    return pd.DataFrame.from_records(records)


def tail_sequence_dependence_rows(
    paths: np.ndarray,
    thresholds: np.ndarray,
    keys: Sequence[str],
    *,
    reference: str,
    lags: Sequence[int] = (1, 2, 4, 8, 13, 26, 52),
) -> pd.DataFrame:
    """Measure clustered joint-tail events only within a scenario path."""
    array = _as_float_matrix(paths, ndim=3)
    records: list[dict[str, Any]] = []
    for group in TAIL_GROUPS:
        positions = _group_positions(keys, group)
        event = ((array[:, :, positions] < thresholds[positions]).sum(axis=2) >= 2)
        for lag in lags:
            if lag >= array.shape[1]:
                continue
            left, right = event[:, :-lag], event[:, lag:]
            joint_events = int((left & right).sum())
            total_pairs = int(left.size)
            conditioning = int(left.sum())
            records.append({"reference": reference, "group": group, "lag": lag,
                            "metric": "joint_probability", "value": joint_events / total_pairs,
                            "conditioning_events": conditioning, "valid_pairs": total_pairs,
                            "status": "ok", "unit": "probability"})
            records.append({"reference": reference, "group": group, "lag": lag,
                            "metric": "conditional_probability", "value": None if conditioning == 0 else joint_events / conditioning,
                            "conditioning_events": conditioning, "valid_pairs": total_pairs,
                            "status": "undefined_no_conditioning_events" if conditioning == 0 else "ok",
                            "unit": "probability"})
    return pd.DataFrame.from_records(records)


def structural_rows(
    source_indices: np.ndarray,
    block_starts: np.ndarray,
    *,
    n_obs: int,
    horizon_steps: int,
    block_length_steps: int,
    experiment: str,
    replicate_id: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Describe MBB joins, lag preservation, and path diversity."""
    indices = np.asarray(source_indices, dtype=np.int64)
    starts = np.asarray(block_starts, dtype=np.int64)
    metrics = block_join_metrics(horizon_steps=horizon_steps, block_length_steps=block_length_steps)
    actual_neighbor = np.mean(indices[:, 1:] == indices[:, :-1] + 1) if horizon_steps > 1 else np.nan
    tradeoff = pd.DataFrame.from_records([
        {"experiment": experiment, "L": block_length_steps, "replicate_id": replicate_id,
         "n_obs": n_obs, "candidate_block_starts": n_obs - block_length_steps + 1,
         "used_blocks_per_path": metrics["n_blocks"], "n_joins": metrics["n_joins"],
         "n_within_pairs": metrics["n_within_pairs"], "within_fraction": metrics["within_fraction"],
         "join_fraction": metrics["join_fraction"], "actual_neighbor_fraction": actual_neighbor},
    ])
    lag_rows = pd.DataFrame.from_records([
        {"experiment": experiment, "L": block_length_steps, "replicate_id": replicate_id,
         "lag": lag, "constructive_within_block_fraction": lag_within_block_fraction(
             horizon_steps=horizon_steps, block_length_steps=block_length_steps, lag=lag
         )}
        for lag in range(1, min(52, horizon_steps - 1) + 1)
    ])
    unique_weeks = np.array([len(np.unique(path)) for path in indices], dtype=np.float64)
    unique_starts = np.array([len(np.unique(path)) for path in starts], dtype=np.float64)
    _, counts = np.unique(indices, axis=0, return_counts=True)
    diversity = pd.DataFrame.from_records([
        {"experiment": experiment, "L": block_length_steps, "replicate_id": replicate_id,
         "metric": "mean_unique_source_weeks_per_path", "value": float(unique_weeks.mean()), "unit": "weeks"},
        {"experiment": experiment, "L": block_length_steps, "replicate_id": replicate_id,
         "metric": "mean_unique_block_starts_per_path", "value": float(unique_starts.mean()), "unit": "starts"},
        {"experiment": experiment, "L": block_length_steps, "replicate_id": replicate_id,
         "metric": "identical_complete_index_paths", "value": int(np.sum(counts > 1)), "unit": "paths"},
    ])
    return tradeoff, lag_rows, diversity


def _summary_rows(values: np.ndarray, *, metrics: Mapping[str, float | np.ndarray | bool]) -> list[dict[str, Any]]:
    flattened = np.asarray(values, dtype=np.float64)
    return [
        {"metric": name, "value": float(np.quantile(flattened, value, method="linear")) if isinstance(value, float) else float(np.mean(flattened)) if value is True else float(np.mean(flattened < 0.0)) if value is False else None}
        for name, value in metrics.items()
    ]


def terminal_summary_rows(
    asset_paths: np.ndarray,
    asset_keys: Sequence[str],
    *,
    terminal_horizon_steps: Mapping[str, int],
    steps_per_year: int,
    reference: str,
) -> pd.DataFrame:
    """Summarize simple terminal and annualized geometric returns for all assets."""
    paths = _as_float_matrix(asset_paths, ndim=3)
    terminals = build_terminal_returns_steps(paths, horizon_steps_by_label=dict(terminal_horizon_steps))
    records: list[dict[str, Any]] = []
    for label, horizon in terminal_horizon_steps.items():
        terminal = terminals[label]
        annualized = annualize_terminal_return(terminal, horizon_steps=horizon, steps_per_year=steps_per_year)
        for kind, values, unit in (("simple_terminal_return", terminal, "decimal_simple_return"), ("annualized_geometric_return", annualized, "decimal_per_year")):
            for position, asset in enumerate(asset_keys):
                series = values[:, position]
                summary = {
                    "mean": float(np.mean(series)), "median": float(np.quantile(series, 0.5, method="linear")),
                    "q01": float(np.quantile(series, 0.01, method="linear")), "q05": float(np.quantile(series, 0.05, method="linear")),
                    "q25": float(np.quantile(series, 0.25, method="linear")), "q75": float(np.quantile(series, 0.75, method="linear")),
                    "q95": float(np.quantile(series, 0.95, method="linear")), "q99": float(np.quantile(series, 0.99, method="linear")),
                    "p_negative": float(np.mean(series < 0.0)),
                }
                records.extend({"reference": reference, "asset": asset, "horizon_label": label,
                                "horizon_steps": horizon, "return_kind": kind, "metric": metric,
                                "value": value, "unit": unit, "n_paths": len(series)}
                               for metric, value in summary.items())
    return pd.DataFrame.from_records(records)


def _path_risk_values(paths: np.ndarray, *, horizon_steps: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prefix = _as_float_matrix(paths, ndim=3)[:, :horizon_steps, :]
    # Every path is independent at this stage. Bound temporary 3-D allocations,
    # but retain ALL per-path observations before computing any quantile.
    # This changes neither scenario counts nor the mean-of-run-quantiles estimand.
    results = tuple(np.empty((len(prefix), prefix.shape[2]), dtype=np.float64) for _ in range(3))
    for start in range(0, len(prefix), 256):
        block = prefix[start:start + 256]
        log_wealth = np.cumsum(block, axis=1)
        cumulative = np.concatenate([np.zeros((len(block), 1, block.shape[2])), log_wealth], axis=1)
        running_max = np.maximum.accumulate(cumulative, axis=1)[:, 1:, :]
        results[0][start:start + len(block)] = np.max(-np.expm1(log_wealth - running_max), axis=1)
        for output, window in zip(results[1:], (13, 52)):
            output[start:start + len(block)] = (
                np.nan if window > horizon_steps else
                np.expm1(np.min(cumulative[:, window:, :] - cumulative[:, :-window, :], axis=1))
            )
    return results


def path_risk_summary_rows(
    asset_paths: np.ndarray,
    asset_keys: Sequence[str],
    *,
    horizons: Mapping[str, int],
    reference: str,
) -> pd.DataFrame:
    """Summarize non-negative MDD and compounded worst-window path risk."""
    records: list[dict[str, Any]] = []
    for label, horizon in horizons.items():
        mdd, worst13, worst52 = _path_risk_values(asset_paths, horizon_steps=horizon)
        for kind, values, quantiles, unit in (
            ("maximum_drawdown", mdd, {"q50": .5, "q90": .9, "q95": .95, "q99": .99}, "decimal_loss"),
            ("worst_13_week_return", worst13, {"q01": .01, "q05": .05, "q50": .5}, "decimal_simple_return"),
            ("worst_52_week_return", worst52, {"q01": .01, "q05": .05, "q50": .5}, "decimal_simple_return"),
        ):
            for position, asset in enumerate(asset_keys):
                series = values[:, position]
                for metric, quantile in quantiles.items():
                    value = None if not np.isfinite(series).any() else float(np.quantile(series[np.isfinite(series)], quantile, method="linear"))
                    records.append({"reference": reference, "asset": asset, "horizon_label": label,
                                    "horizon_steps": horizon, "risk_kind": kind, "metric": metric,
                                    "value": value, "unit": unit, "n_paths": len(series)})
    return pd.DataFrame.from_records(records)


def rolling_path_diagnostic_rows(
    asset_paths: np.ndarray,
    asset_keys: Sequence[str],
    *,
    reference: str,
    vol_window: int = 52,
    corr_window: int = 104,
) -> pd.DataFrame:
    """Summarize path-internal rolling volatility and selected correlations."""
    paths = _as_float_matrix(asset_paths, ndim=3)
    records: list[dict[str, Any]] = []
    def rolling_moments(left: np.ndarray, right: np.ndarray | None, window: int) -> tuple[np.ndarray, np.ndarray | None]:
        cumulative = np.concatenate([np.zeros((len(left), 1)), np.cumsum(left, axis=1)], axis=1)
        cumulative_sq = np.concatenate([np.zeros((len(left), 1)), np.cumsum(left * left, axis=1)], axis=1)
        sums = cumulative[:, window:] - cumulative[:, :-window]
        sums_sq = cumulative_sq[:, window:] - cumulative_sq[:, :-window]
        variance = np.maximum((sums_sq - sums * sums / window) / (window - 1), 0.0)
        if right is None:
            return np.sqrt(variance * 52.0), None
        cumulative_right = np.concatenate([np.zeros((len(right), 1)), np.cumsum(right, axis=1)], axis=1)
        cumulative_right_sq = np.concatenate([np.zeros((len(right), 1)), np.cumsum(right * right, axis=1)], axis=1)
        cumulative_product = np.concatenate([np.zeros((len(left), 1)), np.cumsum(left * right, axis=1)], axis=1)
        sr = cumulative_right[:, window:] - cumulative_right[:, :-window]
        sr2 = cumulative_right_sq[:, window:] - cumulative_right_sq[:, :-window]
        sp = cumulative_product[:, window:] - cumulative_product[:, :-window]
        covariance = (sp - sums * sr / window) / (window - 1)
        variance_right = np.maximum((sr2 - sr * sr / window) / (window - 1), 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = covariance / np.sqrt(variance * variance_right)
        return np.sqrt(variance * 52.0), corr
    for position, asset in enumerate(asset_keys):
        vol, _ = rolling_moments(paths[:, :, position], None, vol_window)
        for metric, q in (("q05", .05), ("q50", .5), ("q95", .95)):
            records.append({"reference": reference, "diagnostic": "rolling_volatility", "asset_or_pair": asset,
                            "window_steps": vol_window, "metric": metric, "value": float(np.quantile(vol, q)),
                            "unit": "decimal_per_year", "n_windows": vol.size})
    for left, right in SELECTED_CORRELATION_PAIRS:
        li, ri = asset_keys.index(left), asset_keys.index(right)
        _, corr = rolling_moments(paths[:, :, li], paths[:, :, ri], corr_window)
        finite = corr[np.isfinite(corr)] if corr is not None else np.array([])
        for metric, q in (("q05", .05), ("q50", .5), ("q95", .95)):
            records.append({"reference": reference, "diagnostic": "rolling_correlation", "asset_or_pair": f"{left}|{right}",
                            "window_steps": corr_window, "metric": metric,
                            "value": None if not finite.size else float(np.quantile(finite, q)),
                            "unit": "correlation", "n_windows": int(finite.size)})
    return pd.DataFrame.from_records(records)


@dataclass
class VariantDiagnostics:
    """All numerical outputs for one synchronized MBB simulation variant."""

    marginal: pd.DataFrame
    occupancy: pd.DataFrame
    edge: pd.DataFrame
    correlation_matrices: pd.DataFrame
    correlation_errors: pd.DataFrame
    tail_cooccurrence: pd.DataFrame
    tail_pairwise: pd.DataFrame
    acf_curves: pd.DataFrame
    acf_errors: pd.DataFrame
    tail_sequence: pd.DataFrame
    tradeoff: pd.DataFrame
    lag_pairs: pd.DataFrame
    diversity: pd.DataFrame
    terminal: pd.DataFrame
    path_risk: pd.DataFrame


def analyze_synchronized_variant(
    *,
    history: np.ndarray,
    history_correlation: np.ndarray,
    history_acfs: Mapping[str, np.ndarray],
    driver_paths: np.ndarray,
    source_indices: np.ndarray,
    block_starts: np.ndarray,
    keys: Sequence[str],
    thresholds: np.ndarray,
    asset_positions: Sequence[int],
    asset_keys: Sequence[str],
    terminal_horizon_steps: Mapping[str, int],
    steps_per_year: int,
    max_lag: int,
    acf_error_lag_sets: Sequence[int],
    experiment: str,
    block_length_steps: int,
    replicate_id: int | None,
) -> VariantDiagnostics:
    """Compute all Set-A diagnostics for one synchronized MBB draw."""
    paths = _as_float_matrix(driver_paths, ndim=3)
    flattened = paths.reshape(-1, paths.shape[2])
    expected = expected_source_weights(n_obs=len(history), horizon_steps=paths.shape[1], block_length_steps=block_length_steps)
    realized = realized_source_weights(source_indices, n_obs=len(history))
    reference = f"{experiment}_L{block_length_steps:02d}_r{replicate_id if replicate_id is not None else 'na'}"
    observed_corr = weighted_correlation(flattened)
    matrices = []
    errors = []
    for group, positions in (("all12", list(range(len(keys)))), ("investable_ex_cash10", [keys.index(k) for k in TAIL_GROUPS["investable_ex_cash10"]])):
        group_keys = [keys[i] for i in positions]
        matrices.append(correlation_matrix_rows(observed_corr[np.ix_(positions, positions)], group_keys, reference=reference, group=group))
        errors.append(correlation_error_rows(observed_corr[np.ix_(positions, positions)], history_correlation[np.ix_(positions, positions)], group_keys,
                                             experiment=experiment, block_length_steps=block_length_steps, replicate_id=replicate_id, group=group))
    acf_records = []
    error_records = []
    history_means = history.mean(axis=0)
    for transform in ("raw", "squared_demeaned"):
        acf, statuses = ensemble_acf(paths, max_lag=max_lag, transform=transform, historical_means=history_means)
        acf_records.append(acf_curve_rows(acf, statuses, keys, reference=reference, transform=transform))
        error_records.append(acf_error_rows(acf, history_acfs[transform], keys, transform=transform, experiment=experiment,
                                            block_length_steps=block_length_steps, replicate_id=replicate_id, lag_sets=acf_error_lag_sets))
    tradeoff, lag_pairs, diversity = structural_rows(source_indices, block_starts, n_obs=len(history), horizon_steps=paths.shape[1],
                                                      block_length_steps=block_length_steps, experiment=experiment, replicate_id=replicate_id)
    asset_paths = paths[:, :, asset_positions]
    return VariantDiagnostics(
        marginal=marginal_preservation_rows(flattened, keys, reference=reference),
        occupancy=source_occupancy_rows(expected_weights=expected, realized_weights=realized, source_dates=pd.date_range("2011-01-07", periods=len(history), freq="W-FRI"), experiment=experiment, block_length_steps=block_length_steps, replicate_id=replicate_id),
        edge=edge_weighting_summary(expected, realized, experiment=experiment, block_length_steps=block_length_steps, replicate_id=replicate_id),
        correlation_matrices=pd.concat(matrices, ignore_index=True), correlation_errors=pd.concat(errors, ignore_index=True),
        tail_cooccurrence=tail_cooccurrence_rows(flattened, thresholds, keys, reference=reference),
        tail_pairwise=tail_pairwise_rows(flattened, thresholds, keys, reference=reference),
        acf_curves=pd.concat(acf_records, ignore_index=True), acf_errors=pd.concat(error_records, ignore_index=True),
        tail_sequence=tail_sequence_dependence_rows(paths, thresholds, keys, reference=reference),
        tradeoff=tradeoff, lag_pairs=lag_pairs, diversity=diversity,
        terminal=terminal_summary_rows(asset_paths, asset_keys, terminal_horizon_steps=terminal_horizon_steps, steps_per_year=steps_per_year, reference=reference),
        path_risk=path_risk_summary_rows(asset_paths, asset_keys, horizons=terminal_horizon_steps, reference=reference),
    )


def historical_rolling_paths(history: np.ndarray, *, horizon_steps: int) -> np.ndarray:
    """Create overlapping descriptive historical windows with no resampling."""
    matrix = _as_float_matrix(history, ndim=2)
    if horizon_steps > len(matrix):
        raise ValueError("historical rolling horizon exceeds observations")
    return np.lib.stride_tricks.sliding_window_view(matrix, window_shape=horizon_steps, axis=0).transpose(0, 2, 1).copy()


def monte_carlo_stability_rows(headline_rows: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-replicate headline metrics; no inferential claims are made."""
    required = {"experiment", "L", "replicate_id", "metric", "value", "unit", "asset_or_group"}
    missing = required - set(headline_rows.columns)
    if missing:
        raise ValueError(f"headline rows miss required columns: {sorted(missing)}")
    grouped = headline_rows.groupby(["experiment", "L", "metric", "unit", "asset_or_group"], dropna=False)
    records = []
    for keys, group in grouped:
        values = group["value"].dropna().to_numpy(dtype=float)
        if not len(values):
            continue
        experiment, length, metric, unit, asset_or_group = keys
        records.append({"experiment": experiment, "L": length, "metric": metric, "unit": unit,
                        "asset_or_group": asset_or_group, "n_replicates": len(values),
                        "mean": values.mean(), "std": values.std(ddof=0), "minimum": values.min(), "maximum": values.max()})
    return pd.DataFrame.from_records(records)
