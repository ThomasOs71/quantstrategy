from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
from scipy.stats import kurtosis, skew


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_LOCAL_DATA_DIR = REPO_ROOT / "articles" / "cov_part2" / "data_raw"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"

DEFAULT_TICKERS = ("SPY", "EFA", "EEM", "AGG", "LQD", "HYG", "GLD", "VNQ", "DBC")
ASSET_LABELS = {
    "SPY": "US Equity",
    "EFA": "Dev ex-US Equity",
    "EEM": "EM Equity",
    "AGG": "US Aggregate Bonds",
    "LQD": "IG Credit",
    "HYG": "High Yield",
    "GLD": "Gold",
    "VNQ": "US REITs",
    "DBC": "Commodities",
}
METHOD_LABELS = {
    "mvpo": "MVPO",
    "repo": "REPO",
    "downside_repo": "Downside REPO",
}


@dataclass(frozen=True)
class FrontierPoint:
    method: str
    target_weekly_return: float
    weights: np.ndarray
    expected_weekly_return: float
    risk_value: float
    risk_label: str


def parse_tickers(raw: str) -> list[str]:
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def stooq_file_name(ticker: str, suffix: str = ".us") -> str:
    return f"{ticker.lower()}{suffix}.csv"


def load_local_stooq_prices(tickers: list[str], data_dir: Path, suffix: str = ".us") -> pd.DataFrame:
    series = []
    missing = []
    for ticker in tickers:
        path = data_dir / stooq_file_name(ticker, suffix=suffix)
        if not path.exists():
            missing.append(str(path))
            continue
        frame = pd.read_csv(path)
        if "Date" not in frame.columns or "Close" not in frame.columns:
            raise ValueError(f"{path} must contain Date and Close columns.")
        frame["Date"] = pd.to_datetime(frame["Date"])
        close = frame.sort_values("Date").set_index("Date")["Close"].astype(float).rename(ticker)
        series.append(close)
    if missing:
        raise FileNotFoundError("Missing local Stooq files:\n" + "\n".join(missing))
    prices = pd.concat(series, axis=1).sort_index()
    return prices.dropna(how="any")


def load_yfinance_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        interval="1d",
        group_by="column",
    )
    if raw.empty:
        raise ValueError("yfinance returned no data.")
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"].copy()
    else:
        close = raw[["Close"]].rename(columns={"Close": tickers[0]})
    close = close.loc[:, tickers]
    close.index = pd.to_datetime(close.index).tz_localize(None)
    return close.dropna(how="any")


def weekly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    weekly_prices = prices.resample("W-FRI").last().dropna(how="any")
    returns = weekly_prices.pct_change().dropna(how="any")
    returns.index.name = "date"
    return returns


def split_in_and_out_sample(
    returns: pd.DataFrame,
    in_start: str,
    in_end: str,
    oos_start: str,
    oos_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    returns = returns.sort_index()
    in_sample = returns.loc[pd.Timestamp(in_start) : pd.Timestamp(in_end)].copy()
    out_sample = returns.loc[pd.Timestamp(oos_start) : pd.Timestamp(oos_end)].copy()
    if in_sample.empty:
        raise ValueError("In-sample return window is empty.")
    if out_sample.empty:
        raise ValueError("Out-of-sample return window is empty.")
    return in_sample, out_sample


def annualized_return(returns: np.ndarray, periods_per_year: int = 52) -> float:
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return math.nan
    return float(np.prod(1.0 + returns) ** (periods_per_year / returns.size) - 1.0)


def annualized_vol(returns: np.ndarray, periods_per_year: int = 52) -> float:
    returns = np.asarray(returns, dtype=float)
    if returns.size < 2:
        return math.nan
    return float(np.std(returns, ddof=1) * math.sqrt(periods_per_year))


def max_drawdown(returns: np.ndarray) -> float:
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return math.nan
    wealth = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(wealth)
    drawdown = wealth / peak - 1.0
    return float(np.min(drawdown))


def var_cvar(returns: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    returns = np.asarray(returns, dtype=float)
    if returns.size == 0:
        return math.nan, math.nan
    var = float(np.quantile(returns, alpha))
    tail = returns[returns <= var]
    cvar = float(np.mean(tail)) if tail.size else var
    return var, cvar


def weight_entropy(weights: np.ndarray) -> float:
    p = np.asarray(weights, dtype=float)
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def diversification_stats(weights: np.ndarray, min_weight: float = 0.01) -> dict[str, float | int]:
    weights = np.asarray(weights, dtype=float)
    hhi = float(np.sum(weights**2))
    h_weights = weight_entropy(weights)
    return {
        "weight_entropy": h_weights,
        "effective_n_entropy": float(np.exp(h_weights)),
        "herfindahl_hhi": hhi,
        "effective_n_hhi": float(1.0 / hhi) if hhi > 0.0 else math.nan,
        "max_weight": float(np.max(weights)),
        "top3_weight": float(np.sum(np.sort(weights)[-3:])),
        "n_positions_gt_1pct": int(np.sum(weights > min_weight)),
    }


def build_bin_edges(returns: pd.DataFrame, n_bins: int) -> np.ndarray:
    values = returns.to_numpy(dtype=float).reshape(-1)
    lo = float(np.nanmin(values))
    hi = float(np.nanmax(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise ValueError("Cannot build bins from degenerate return data.")
    pad = max((hi - lo) * 1e-6, 1e-8)
    return np.linspace(lo - pad, hi + pad, n_bins + 1)


def entropy_from_probabilities(probabilities: np.ndarray) -> float:
    p = np.asarray(probabilities, dtype=float)
    p = p[p > 0.0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def portfolio_entropy(
    weights: np.ndarray,
    returns: pd.DataFrame,
    bin_edges: np.ndarray,
    method: str = "standard",
) -> float:
    rp = returns.to_numpy(dtype=float) @ np.asarray(weights, dtype=float)
    if method == "standard":
        counts, _ = np.histogram(rp, bins=bin_edges)
        probabilities = counts / len(rp)
        return entropy_from_probabilities(probabilities)

    if method == "downside":
        n_bins = len(bin_edges) - 1
        if n_bins < 2:
            raise ValueError("Downside entropy needs at least two bins.")
        lo = min(float(bin_edges[0]), float(np.min(rp)), -1e-12)
        negative_edges = np.linspace(lo, 0.0, n_bins)
        negative_returns = rp[rp <= 0.0]
        if negative_returns.size:
            negative_counts, _ = np.histogram(negative_returns, bins=negative_edges)
        else:
            negative_counts = np.zeros(n_bins - 1, dtype=int)
        p_negative = negative_counts / len(rp)
        p_positive = np.array([np.sum(rp > 0.0) / len(rp)])
        probabilities = np.concatenate([p_negative, p_positive])
        return entropy_from_probabilities(probabilities)

    raise ValueError(f"Unknown entropy method: {method}")


def candidate_standard_entropies(
    candidates: np.ndarray,
    returns: pd.DataFrame,
    bin_edges: np.ndarray,
) -> np.ndarray:
    portfolio_returns = returns.to_numpy(dtype=float) @ candidates.T
    n_obs, n_candidates = portfolio_returns.shape
    n_bins = len(bin_edges) - 1
    bin_index = np.searchsorted(bin_edges, portfolio_returns, side="right") - 1
    bin_index = np.clip(bin_index, 0, n_bins - 1)

    counts = np.zeros((n_candidates, n_bins), dtype=float)
    candidate_index = np.repeat(np.arange(n_candidates), n_obs)
    np.add.at(counts, (candidate_index, bin_index.T.reshape(-1)), 1.0)

    probabilities = counts / n_obs
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(probabilities > 0.0, probabilities * np.log(probabilities), 0.0)
    return -np.sum(terms, axis=1)


def generate_candidate_weights(
    n_assets: int,
    tickers: list[str],
    n_random: int,
    seed: int,
    max_weight: float,
) -> np.ndarray:
    if max_weight < 1.0 / n_assets:
        raise ValueError("max_weight is infeasible: it must be at least 1 / n_assets.")

    rng = np.random.default_rng(seed)
    weights = [np.full(n_assets, 1.0 / n_assets)]

    if max_weight < 1.0:
        for i in range(n_assets):
            capped = np.full(n_assets, (1.0 - max_weight) / (n_assets - 1))
            capped[i] = max_weight
            weights.append(capped)
    else:
        weights.extend(np.eye(n_assets)[i] for i in range(n_assets))

    if "SPY" in tickers and "AGG" in tickers:
        w_6040 = np.zeros(n_assets)
        w_6040[tickers.index("SPY")] = 0.60
        w_6040[tickers.index("AGG")] = 0.40
        weights.append(w_6040)

    accepted = []
    alpha_cycle = [np.ones(n_assets), np.full(n_assets, 0.45), np.full(n_assets, 2.0)]
    attempts = 0
    while sum(len(batch) for batch in accepted) < n_random and attempts < 100:
        alpha = alpha_cycle[attempts % len(alpha_cycle)]
        batch = rng.dirichlet(alpha, size=max(512, n_random // 2))
        batch = batch[np.max(batch, axis=1) <= max_weight + 1e-12]
        if len(batch):
            accepted.append(batch)
        attempts += 1
    if accepted:
        random_weights = np.vstack(accepted)[:n_random]
        weights.append(random_weights)

    matrix = np.vstack(weights)
    matrix = np.clip(matrix, 0.0, 1.0)
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    matrix = matrix[np.max(matrix, axis=1) <= max_weight + 1e-10]
    return np.unique(np.round(matrix, 12), axis=0)


def candidate_risk_arrays(
    candidates: np.ndarray,
    returns: pd.DataFrame,
    bin_edges: np.ndarray,
) -> dict[str, np.ndarray]:
    R = returns.to_numpy(dtype=float)
    portfolio_returns = R @ candidates.T
    mvpo_risk = np.var(portfolio_returns, axis=0, ddof=1)
    repo_risk = candidate_standard_entropies(candidates, returns, bin_edges)
    downside_risk = np.array(
        [portfolio_entropy(w, returns, bin_edges, method="downside") for w in candidates],
        dtype=float,
    )
    return {
        "mvpo": mvpo_risk,
        "repo": repo_risk,
        "downside_repo": downside_risk,
    }


def polish_mvpo(
    mu_weekly: np.ndarray,
    cov_weekly: np.ndarray,
    target_weekly_return: float,
    initial_weights: np.ndarray,
    max_weight: float,
) -> np.ndarray:
    n_assets = len(initial_weights)
    cov_reg = cov_weekly + np.eye(n_assets) * 1e-8

    def objective(w: np.ndarray) -> float:
        return float(w @ cov_reg @ w)

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        {"type": "ineq", "fun": lambda w: float(mu_weekly @ w - target_weekly_return)},
    ]
    result = opt.minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n_assets,
        constraints=constraints,
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )
    if result.success and np.all(np.isfinite(result.x)):
        weights = np.clip(result.x, 0.0, 1.0)
        return weights / weights.sum()
    return initial_weights


def build_frontier(
    method: str,
    returns: pd.DataFrame,
    candidates: np.ndarray,
    candidate_risks: np.ndarray,
    targets: np.ndarray,
    bin_edges: np.ndarray,
    max_weight: float,
) -> list[FrontierPoint]:
    mu_weekly = returns.mean().to_numpy(dtype=float)
    cov_weekly = returns.cov().to_numpy(dtype=float)
    expected = candidates @ mu_weekly
    points: list[FrontierPoint] = []

    for target in targets:
        feasible = np.where(expected >= target - 1e-12)[0]
        if feasible.size == 0:
            continue
        best_idx = feasible[np.argmin(candidate_risks[feasible])]
        weights = candidates[best_idx].copy()
        if method == "mvpo":
            weights = polish_mvpo(mu_weekly, cov_weekly, target, weights, max_weight=max_weight)
            rp = returns.to_numpy(dtype=float) @ weights
            risk_value = float(np.var(rp, ddof=1))
            risk_label = "weekly_variance"
        elif method == "repo":
            risk_value = portfolio_entropy(weights, returns, bin_edges, method="standard")
            risk_label = "entropy"
        elif method == "downside_repo":
            risk_value = portfolio_entropy(weights, returns, bin_edges, method="downside")
            risk_label = "downside_entropy"
        else:
            raise ValueError(f"Unknown method: {method}")
        points.append(
            FrontierPoint(
                method=method,
                target_weekly_return=float(target),
                weights=weights,
                expected_weekly_return=float(mu_weekly @ weights),
                risk_value=risk_value,
                risk_label=risk_label,
            )
        )
    return points


def select_representative_points(points: list[FrontierPoint]) -> list[tuple[str, FrontierPoint]]:
    if not points:
        return []
    indexes = [0, len(points) // 2, len(points) - 1]
    labels = ["min_risk", "target_return", "high_return"]
    selected = []
    seen = set()
    for label, idx in zip(labels, indexes):
        if idx in seen:
            continue
        seen.add(idx)
        selected.append((label, points[idx]))
    return selected


def portfolio_metric_row(
    label: str,
    point: FrontierPoint,
    in_sample: pd.DataFrame,
    out_sample: pd.DataFrame,
    bin_edges: np.ndarray,
    risk_free_rate: float,
    tickers: list[str],
) -> dict[str, float | str]:
    w = point.weights
    rp_in = in_sample.to_numpy(dtype=float) @ w
    rp_oos = out_sample.to_numpy(dtype=float) @ w
    ann_ret_oos = annualized_return(rp_oos)
    ann_vol_oos = annualized_vol(rp_oos)
    var5, cvar5 = var_cvar(rp_oos, alpha=0.05)

    row: dict[str, float | str] = {
        "selection": label,
        "method": point.method,
        "method_label": METHOD_LABELS[point.method],
        "expected_return_in_sample_ann": float((1.0 + point.expected_weekly_return) ** 52 - 1.0),
        "ann_vol_in_sample": annualized_vol(rp_in),
        "standard_entropy_in_sample": portfolio_entropy(w, in_sample, bin_edges, method="standard"),
        "downside_entropy_in_sample": portfolio_entropy(w, in_sample, bin_edges, method="downside"),
        "oos_total_return": float(np.prod(1.0 + rp_oos) - 1.0),
        "oos_ann_return": ann_ret_oos,
        "oos_ann_vol": ann_vol_oos,
        "oos_sharpe_rf3": (ann_ret_oos - risk_free_rate) / ann_vol_oos if ann_vol_oos > 0 else math.nan,
        "oos_max_drawdown": max_drawdown(rp_oos),
        "oos_var_5": var5,
        "oos_cvar_5": cvar5,
        "oos_skew": float(skew(rp_oos, bias=False)) if len(rp_oos) > 2 else math.nan,
        "oos_kurtosis": float(kurtosis(rp_oos, fisher=True, bias=False)) if len(rp_oos) > 3 else math.nan,
    }
    row.update(diversification_stats(w))
    for ticker, weight in zip(tickers, w):
        row[f"w_{ticker}"] = float(weight)
    return row


def make_frontier_frame(frontiers: dict[str, list[FrontierPoint]], tickers: list[str]) -> pd.DataFrame:
    rows = []
    for method, points in frontiers.items():
        for i, point in enumerate(points):
            row = {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "point": i,
                "target_weekly_return": point.target_weekly_return,
                "expected_weekly_return": point.expected_weekly_return,
                "expected_return_ann": (1.0 + point.expected_weekly_return) ** 52 - 1.0,
                "risk_value": point.risk_value,
                "risk_label": point.risk_label,
            }
            row.update(diversification_stats(point.weights))
            for ticker, weight in zip(tickers, point.weights):
                row[f"w_{ticker}"] = float(weight)
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_input_returns(returns: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in returns.columns:
        r = returns[ticker].dropna().to_numpy(dtype=float)
        rows.append(
            {
                "ticker": ticker,
                "asset_label": ASSET_LABELS.get(ticker, ticker),
                "weekly_mean": float(np.mean(r)),
                "annualized_mean_arithmetic": float(np.mean(r) * 52.0),
                "annualized_vol": annualized_vol(r),
                "skew": float(skew(r, bias=False)) if len(r) > 2 else math.nan,
                "kurtosis": float(kurtosis(r, fisher=True, bias=False)) if len(r) > 3 else math.nan,
            }
        )
    return pd.DataFrame(rows)


def rank_flip_count(weights_a: np.ndarray, weights_b: np.ndarray, eps: float = 1e-4) -> tuple[int, int]:
    flips = 0
    compared = 0
    n_assets = len(weights_a)
    for i in range(n_assets):
        for j in range(i + 1, n_assets):
            diff_a = weights_a[i] - weights_a[j]
            diff_b = weights_b[i] - weights_b[j]
            if abs(diff_a) <= eps or abs(diff_b) <= eps:
                continue
            compared += 1
            if np.sign(diff_a) != np.sign(diff_b):
                flips += 1
    return flips, compared


def run_bin_sensitivity(
    in_sample: pd.DataFrame,
    candidates: np.ndarray,
    tickers: list[str],
    bin_counts: list[int],
    target_weekly_return: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | int]]:
    equal_weight = np.full(len(tickers), 1.0 / len(tickers))
    rows = []
    optimized_weights = {}

    for n_bins in bin_counts:
        edges = build_bin_edges(in_sample, n_bins)
        entropy_equal = portfolio_entropy(equal_weight, in_sample, edges, method="standard")
        risks = np.array(
            [portfolio_entropy(w, in_sample, edges, method="standard") for w in candidates],
            dtype=float,
        )
        expected = candidates @ in_sample.mean().to_numpy(dtype=float)
        feasible = np.where(expected >= target_weekly_return - 1e-12)[0]
        if feasible.size:
            best_idx = feasible[np.argmin(risks[feasible])]
        else:
            best_idx = int(np.argmin(risks))
        best_weights = candidates[best_idx]
        optimized_weights[n_bins] = best_weights
        row = {
            "bin_count": n_bins,
            "equal_weight_entropy": entropy_equal,
            "equal_weight_entropy_normalized": entropy_equal / math.log(n_bins) if n_bins > 1 else math.nan,
            "optimized_entropy": float(risks[best_idx]),
            "optimized_expected_return_ann": float((1.0 + expected[best_idx]) ** 52 - 1.0),
            "optimized_weight_entropy": weight_entropy(best_weights),
        }
        for ticker, weight in zip(tickers, best_weights):
            row[f"w_{ticker}"] = float(weight)
        rows.append(row)

    distance_rows = []
    for a in bin_counts:
        row = {"bin_count": a}
        for b in bin_counts:
            row[str(b)] = float(np.linalg.norm(optimized_weights[a] - optimized_weights[b]))
        distance_rows.append(row)

    governance = {}
    if 10 in optimized_weights and 30 in optimized_weights:
        flips, compared = rank_flip_count(optimized_weights[10], optimized_weights[30])
        governance = {
            "rank_flips_m10_to_m30": int(flips),
            "rank_pairs_compared": int(compared),
            "rank_flip_share": float(flips / compared) if compared else math.nan,
            "l2_distance_m10_to_m30": float(np.linalg.norm(optimized_weights[10] - optimized_weights[30])),
            "l1_distance_m10_to_m30": float(np.sum(np.abs(optimized_weights[10] - optimized_weights[30]))),
        }
    return pd.DataFrame(rows), pd.DataFrame(distance_rows), governance


def minimize_variance_weights(returns: pd.DataFrame, max_weight: float) -> np.ndarray:
    n_assets = returns.shape[1]
    cov = returns.cov().to_numpy(dtype=float) + np.eye(n_assets) * 1e-8
    w0 = np.full(n_assets, 1.0 / n_assets)

    def objective(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    result = opt.minimize(
        objective,
        w0,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n_assets,
        constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )
    if not result.success or not np.all(np.isfinite(result.x)):
        return w0
    weights = np.clip(result.x, 0.0, max_weight)
    return weights / weights.sum()


def minimize_repo_weights(
    returns: pd.DataFrame,
    candidates: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    bin_edges = build_bin_edges(returns, n_bins)
    risks = candidate_standard_entropies(candidates, returns, bin_edges)
    return candidates[int(np.argmin(risks))].copy()


def performance_summary_from_returns(
    label: str,
    returns: pd.Series,
    risk_free_rate: float,
) -> dict[str, float | str]:
    values = returns.dropna().to_numpy(dtype=float)
    ann_ret = annualized_return(values)
    ann_vol = annualized_vol(values)
    var5, cvar5 = var_cvar(values, alpha=0.05)
    return {
        "strategy": label,
        "total_return": float(np.prod(1.0 + values) - 1.0) if values.size else math.nan,
        "ann_return": ann_ret,
        "ann_vol": ann_vol,
        "sharpe_rf3": (ann_ret - risk_free_rate) / ann_vol if ann_vol > 0.0 else math.nan,
        "max_drawdown": max_drawdown(values),
        "var_5": var5,
        "cvar_5": cvar5,
        "skew": float(skew(values, bias=False)) if len(values) > 2 else math.nan,
        "kurtosis": float(kurtosis(values, fisher=True, bias=False)) if len(values) > 3 else math.nan,
    }


def run_rolling_rebalance(
    all_returns: pd.DataFrame,
    out_sample: pd.DataFrame,
    candidates: np.ndarray,
    static_points: dict[str, list[tuple[str, FrontierPoint]]],
    tickers: list[str],
    n_bins: int,
    lookback_weeks: int,
    max_weight: float,
    risk_free_rate: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    monthly_periods = sorted(out_sample.index.to_period("M").unique())
    weights_by_method = {"rolling_mvpo": [], "rolling_repo": []}
    return_rows = []
    weight_rows = []

    previous_weights: dict[str, np.ndarray | None] = {"rolling_mvpo": None, "rolling_repo": None}
    method_labels = {"rolling_mvpo": "Rolling MVPO", "rolling_repo": "Rolling REPO"}

    for period in monthly_periods:
        eval_returns = out_sample[out_sample.index.to_period("M") == period]
        if eval_returns.empty:
            continue
        eval_start = eval_returns.index.min()
        estimation_pool = all_returns.loc[all_returns.index < eval_start].tail(lookback_weeks)
        if len(estimation_pool) < lookback_weeks:
            continue

        current_weights = {
            "rolling_mvpo": minimize_variance_weights(estimation_pool, max_weight=max_weight),
            "rolling_repo": minimize_repo_weights(estimation_pool, candidates, n_bins=n_bins),
        }

        for method, weights in current_weights.items():
            previous = previous_weights[method]
            l1_change = float(np.sum(np.abs(weights - previous))) if previous is not None else math.nan
            turnover = 0.5 * l1_change if previous is not None else math.nan
            stats = diversification_stats(weights)
            row = {
                "rebalance_month": str(period),
                "method": method,
                "method_label": method_labels[method],
                "eval_start": str(eval_returns.index.min().date()),
                "eval_end": str(eval_returns.index.max().date()),
                "lookback_start": str(estimation_pool.index.min().date()),
                "lookback_end": str(estimation_pool.index.max().date()),
                "l1_change": l1_change,
                "turnover": turnover,
                "max_abs_weight_change": float(np.max(np.abs(weights - previous))) if previous is not None else math.nan,
            }
            row.update(stats)
            for ticker, weight in zip(tickers, weights):
                row[f"w_{ticker}"] = float(weight)
            weight_rows.append(row)

            strategy_returns = eval_returns.to_numpy(dtype=float) @ weights
            for date, value in zip(eval_returns.index, strategy_returns):
                return_rows.append(
                    {
                        "date": date,
                        "strategy": method_labels[method],
                        "weekly_return": float(value),
                        "rebalance_month": str(period),
                    }
                )
            weights_by_method[method].append(weights)
            previous_weights[method] = weights

    min_risk_points = {
        method: point
        for method, points in static_points.items()
        for label, point in points
        if label == "min_risk" and method in {"mvpo", "repo"}
    }
    for method, point in min_risk_points.items():
        label = f"Static {METHOD_LABELS[method]}"
        strategy_returns = out_sample.to_numpy(dtype=float) @ point.weights
        for date, value in zip(out_sample.index, strategy_returns):
            return_rows.append(
                {
                    "date": date,
                    "strategy": label,
                    "weekly_return": float(value),
                    "rebalance_month": "static",
                }
            )

    rolling_returns = pd.DataFrame(return_rows)
    if not rolling_returns.empty:
        rolling_returns["date"] = pd.to_datetime(rolling_returns["date"])
        rolling_returns = rolling_returns.sort_values(["strategy", "date"])
        rolling_returns["wealth"] = rolling_returns.groupby("strategy")["weekly_return"].transform(
            lambda r: np.cumprod(1.0 + r.to_numpy(dtype=float))
        )

    rolling_weights = pd.DataFrame(weight_rows)
    summary_rows = []
    for strategy, group in rolling_returns.groupby("strategy", sort=False):
        summary_rows.append(
            performance_summary_from_returns(
                label=strategy,
                returns=group.sort_values("date")["weekly_return"],
                risk_free_rate=risk_free_rate,
            )
        )
    rolling_summary = pd.DataFrame(summary_rows)

    if not rolling_weights.empty and not rolling_summary.empty:
        stability_rows = []
        for method_label, group in rolling_weights.groupby("method_label"):
            stability_rows.append(
                {
                    "strategy": method_label,
                    "avg_turnover": float(group["turnover"].dropna().mean()),
                    "median_turnover": float(group["turnover"].dropna().median()),
                    "max_turnover": float(group["turnover"].dropna().max()),
                    "avg_l1_change": float(group["l1_change"].dropna().mean()),
                    "avg_max_abs_weight_change": float(group["max_abs_weight_change"].dropna().mean()),
                    "avg_weight_entropy": float(group["weight_entropy"].mean()),
                    "avg_effective_n_entropy": float(group["effective_n_entropy"].mean()),
                    "avg_top3_weight": float(group["top3_weight"].mean()),
                    "n_rebalances": int(len(group)),
                }
            )
        stability = pd.DataFrame(stability_rows)
        rolling_summary = rolling_summary.merge(stability, on="strategy", how="left")

    return rolling_returns, rolling_weights, rolling_summary


def plot_frontiers(frontier_df: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / "fig_1_frontiers.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    mvpo = frontier_df[frontier_df["method"] == "mvpo"].copy()
    if not mvpo.empty:
        axes[0].plot(
            np.sqrt(mvpo["risk_value"].to_numpy(dtype=float)) * math.sqrt(52.0),
            mvpo["expected_return_ann"],
            marker="o",
            linewidth=1.8,
            label="MVPO",
        )
    axes[0].set_title("MVPO frontier")
    axes[0].set_xlabel("Annualized volatility")
    axes[0].set_ylabel("Expected annual return")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    for method, color in [("repo", "#2f6fdd"), ("downside_repo", "#b6465f")]:
        subset = frontier_df[frontier_df["method"] == method].copy()
        if subset.empty:
            continue
        axes[1].plot(
            subset["risk_value"],
            subset["expected_return_ann"],
            marker="o",
            linewidth=1.8,
            label=METHOD_LABELS[method],
            color=color,
        )
    axes[1].set_title("Entropy frontiers")
    axes[1].set_xlabel("Entropy risk measure")
    axes[1].set_ylabel("Expected annual return")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_weight_comparison(summary_df: pd.DataFrame, tickers: list[str], output_dir: Path) -> Path:
    output_path = output_dir / "fig_2_min_risk_weights.png"
    subset = summary_df[summary_df["selection"] == "min_risk"].copy()
    weight_cols = [f"w_{ticker}" for ticker in tickers]
    x = np.arange(len(tickers))
    width = 0.25
    fig, ax = plt.subplots(figsize=(12, 5))
    for offset, (_, row) in zip([-width, 0.0, width], subset.iterrows()):
        ax.bar(x + offset, row[weight_cols].to_numpy(dtype=float), width=width, label=row["method_label"])
    ax.set_xticks(x)
    ax.set_xticklabels(tickers)
    ax.set_ylabel("Portfolio weight")
    ax.set_title("Minimum-risk allocations by risk definition")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_oos_paths(
    selected: dict[str, list[tuple[str, FrontierPoint]]],
    out_sample: pd.DataFrame,
    output_dir: Path,
) -> Path:
    output_path = output_dir / "fig_3_oos_cumulative_min_risk.png"
    fig, ax = plt.subplots(figsize=(11, 5))
    for method, points in selected.items():
        min_points = [point for label, point in points if label == "min_risk"]
        if not min_points:
            continue
        point = min_points[0]
        rp = out_sample.to_numpy(dtype=float) @ point.weights
        wealth = pd.Series(np.cumprod(1.0 + rp), index=out_sample.index)
        ann_ret = annualized_return(rp)
        ann_vol = annualized_vol(rp)
        sharpe = (ann_ret - 0.03) / ann_vol if ann_vol > 0 else math.nan
        ax.plot(wealth.index, wealth.values, linewidth=1.8, label=f"{METHOD_LABELS[method]} | Sharpe {sharpe:.2f}")
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("Out-of-sample wealth: minimum-risk portfolios")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_repo_outperformance(
    selected: dict[str, list[tuple[str, FrontierPoint]]],
    out_sample: pd.DataFrame,
    output_dir: Path,
) -> Path:
    output_path = output_dir / "fig_5_repo_vs_mvpo_outperformance.png"

    by_method = {
        method: {label: point for label, point in method_points}
        for method, method_points in selected.items()
    }
    common_labels = [
        label
        for label in ["min_risk", "target_return", "high_return"]
        if label in by_method.get("mvpo", {}) and label in by_method.get("repo", {})
    ]

    fig, ax = plt.subplots(figsize=(11, 5))
    for label in common_labels:
        mvpo_point = by_method["mvpo"][label]
        repo_point = by_method["repo"][label]
        mvpo_returns = out_sample.to_numpy(dtype=float) @ mvpo_point.weights
        repo_returns = out_sample.to_numpy(dtype=float) @ repo_point.weights
        mvpo_wealth = np.cumprod(1.0 + mvpo_returns)
        repo_wealth = np.cumprod(1.0 + repo_returns)
        active_wealth = repo_wealth / mvpo_wealth - 1.0
        ax.plot(
            out_sample.index,
            active_wealth,
            linewidth=1.8,
            label=label.replace("_", " ").title(),
        )

    ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.65)
    ax.set_title("REPO cumulative outperformance versus MVPO")
    ax.set_ylabel("REPO / MVPO - 1")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_rolling_vs_static_wealth(rolling_returns: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / "fig_6_rolling_vs_static_wealth.png"
    fig, ax = plt.subplots(figsize=(11, 5))
    style = {
        "Rolling MVPO": {"linewidth": 2.0, "linestyle": "-"},
        "Rolling REPO": {"linewidth": 2.0, "linestyle": "-"},
        "Static MVPO": {"linewidth": 1.6, "linestyle": "--"},
        "Static REPO": {"linewidth": 1.6, "linestyle": "--"},
    }
    for strategy, group in rolling_returns.groupby("strategy", sort=False):
        group = group.sort_values("date")
        ax.plot(
            group["date"],
            group["wealth"],
            label=strategy,
            **style.get(strategy, {"linewidth": 1.6, "linestyle": "-"}),
        )
    ax.axhline(1.0, color="black", linewidth=0.8, alpha=0.5)
    ax.set_title("Rolling monthly rebalance versus static buy-and-hold")
    ax.set_ylabel("Growth of $1")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_rolling_stability(rolling_weights: pd.DataFrame, output_dir: Path) -> Path:
    output_path = output_dir / "fig_7_rolling_weight_stability.png"
    frame = rolling_weights.copy()
    frame["date"] = pd.to_datetime(frame["eval_start"])

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    for method_label, group in frame.groupby("method_label"):
        group = group.sort_values("date")
        axes[0].plot(group["date"], group["turnover"], marker="o", linewidth=1.6, label=method_label)
        axes[1].plot(group["date"], group["effective_n_entropy"], marker="o", linewidth=1.6, label=method_label)

    axes[0].set_title("Monthly weight turnover")
    axes[0].set_ylabel("0.5 * L1 change")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].set_title("Effective number of positions")
    axes[1].set_ylabel("exp(H(weights))")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_bin_sensitivity(
    sensitivity_df: pd.DataFrame,
    distance_df: pd.DataFrame,
    output_dir: Path,
) -> Path:
    output_path = output_dir / "fig_4_bin_sensitivity.png"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    axes[0].plot(
        sensitivity_df["bin_count"],
        sensitivity_df["equal_weight_entropy"],
        marker="o",
        label="H(equal weight)",
    )
    axes[0].plot(
        sensitivity_df["bin_count"],
        sensitivity_df["equal_weight_entropy_normalized"],
        marker="s",
        label="H / log(m)",
    )
    axes[0].set_xlabel("Bin count")
    axes[0].set_title("Entropy depends on bin design")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    matrix = distance_df.drop(columns=["bin_count"]).to_numpy(dtype=float)
    image = axes[1].imshow(matrix, cmap="magma_r")
    axes[1].set_title("L2 distance between optimal weights")
    labels = distance_df["bin_count"].astype(str).tolist()
    axes[1].set_xticks(np.arange(len(labels)))
    axes[1].set_yticks(np.arange(len(labels)))
    axes[1].set_xticklabels(labels)
    axes[1].set_yticklabels(labels)
    axes[1].set_xlabel("Bin count")
    axes[1].set_ylabel("Bin count")
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def write_markdown_summary(
    output_dir: Path,
    summary_df: pd.DataFrame,
    governance: dict[str, float | int],
    data_source: str,
    tickers: list[str],
    rolling_summary: pd.DataFrame | None = None,
) -> Path:
    output_path = output_dir / "summary.md"
    min_rows = summary_df[summary_df["selection"] == "min_risk"].copy()
    lines = [
        "# REPO Entropy Study - Run Summary",
        "",
        f"- Data source: `{data_source}`",
        f"- Universe: {', '.join(tickers)}",
        "- In-sample: 2015-01-01 to 2021-12-31",
        "- Out-of-sample: 2022-01-01 to 2024-12-31",
        "",
        "## Minimum-risk portfolios",
        "",
        "| Method | OOS ann. return | OOS ann. vol | Sharpe rf=3% | Max drawdown | H(weights) | Eff. N |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in min_rows.iterrows():
        lines.append(
            "| {method} | {ret:.2%} | {vol:.2%} | {sharpe:.2f} | {dd:.2%} | {hw:.3f} | {neff:.2f} |".format(
                method=row["method_label"],
                ret=row["oos_ann_return"],
                vol=row["oos_ann_vol"],
                sharpe=row["oos_sharpe_rf3"],
                dd=row["oos_max_drawdown"],
                hw=row["weight_entropy"],
                neff=row["effective_n_entropy"],
            )
        )
    if governance:
        lines.extend(
            [
                "",
                "## Bin governance check",
                "",
                f"- Rank flips from m=10 to m=30: {governance['rank_flips_m10_to_m30']} of {governance['rank_pairs_compared']} comparable asset pairs.",
                f"- L1 distance m=10 to m=30: {governance['l1_distance_m10_to_m30']:.3f}",
                f"- L2 distance m=10 to m=30: {governance['l2_distance_m10_to_m30']:.3f}",
            ]
        )
    if rolling_summary is not None and not rolling_summary.empty:
        lines.extend(
            [
                "",
                "## Rolling monthly rebalance",
                "",
                "| Strategy | Total return | Ann. return | Ann. vol | Sharpe rf=3% | Max DD | Avg turnover | Avg eff. N |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for _, row in rolling_summary.iterrows():
            lines.append(
                "| {strategy} | {total:.2%} | {ret:.2%} | {vol:.2%} | {sharpe:.2f} | {dd:.2%} | {to} | {neff} |".format(
                    strategy=row["strategy"],
                    total=row["total_return"],
                    ret=row["ann_return"],
                    vol=row["ann_vol"],
                    sharpe=row["sharpe_rf3"],
                    dd=row["max_drawdown"],
                    to=f"{row['avg_turnover']:.2%}" if pd.notna(row.get("avg_turnover")) else "-",
                    neff=f"{row['avg_effective_n_entropy']:.2f}"
                    if pd.notna(row.get("avg_effective_n_entropy"))
                    else "-",
                )
            )
    lines.extend(
        [
            "",
            "Interpretation guardrail: REPO is non-parametric, but the histogram bins are a modeling choice. Treat bin sensitivity as a governance result, not a nuisance detail.",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Critical REPO vs MVPO implementation for a short Substack code companion."
    )
    parser.add_argument("--data-source", choices=["local", "yfinance"], default="local")
    parser.add_argument("--local-data-dir", type=Path, default=DEFAULT_LOCAL_DATA_DIR)
    parser.add_argument("--stooq-suffix", default=".us")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--in-start", default="2015-01-01")
    parser.add_argument("--in-end", default="2021-12-31")
    parser.add_argument("--oos-start", default="2022-01-01")
    parser.add_argument("--oos-end", default="2024-12-31")
    parser.add_argument("--bin-count", type=int, default=20)
    parser.add_argument("--frontier-points", type=int, default=21)
    parser.add_argument("--n-random", type=int, default=5000)
    parser.add_argument(
        "--rolling-lookback-weeks",
        type=int,
        default=156,
        help="Trailing weekly observations used for each monthly rolling rebalance.",
    )
    parser.add_argument(
        "--max-weight",
        type=float,
        default=0.60,
        help="Long-only per-asset cap. Use 1.0 for the paper-like unconstrained case.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--risk-free-rate", type=float, default=0.03)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    tickers = parse_tickers(args.tickers)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.data_source == "local":
        prices = load_local_stooq_prices(tickers, args.local_data_dir, suffix=args.stooq_suffix)
    else:
        prices = load_yfinance_prices(tickers, start=args.in_start, end=args.oos_end)

    returns = weekly_returns(prices)
    in_sample, out_sample = split_in_and_out_sample(
        returns,
        in_start=args.in_start,
        in_end=args.in_end,
        oos_start=args.oos_start,
        oos_end=args.oos_end,
    )
    bin_edges = build_bin_edges(in_sample, args.bin_count)
    candidates = generate_candidate_weights(
        n_assets=len(tickers),
        tickers=tickers,
        n_random=args.n_random,
        seed=args.seed,
        max_weight=args.max_weight,
    )
    risks = candidate_risk_arrays(candidates, in_sample, bin_edges)

    mu_weekly = in_sample.mean().to_numpy(dtype=float)
    candidate_expected = candidates @ mu_weekly
    min_target = float(np.min(candidate_expected))
    max_target = float(np.max(candidate_expected))
    targets = np.linspace(min_target, max_target, args.frontier_points)

    frontiers = {
        method: build_frontier(
            method=method,
            returns=in_sample,
            candidates=candidates,
            candidate_risks=risks[method],
            targets=targets,
            bin_edges=bin_edges,
            max_weight=args.max_weight,
        )
        for method in ["mvpo", "repo", "downside_repo"]
    }
    selected = {method: select_representative_points(points) for method, points in frontiers.items()}

    summary_rows = []
    for method_points in selected.values():
        for label, point in method_points:
            summary_rows.append(
                portfolio_metric_row(
                    label=label,
                    point=point,
                    in_sample=in_sample,
                    out_sample=out_sample,
                    bin_edges=bin_edges,
                    risk_free_rate=args.risk_free_rate,
                    tickers=tickers,
                )
            )
    summary_df = pd.DataFrame(summary_rows)
    frontier_df = make_frontier_frame(frontiers, tickers)
    input_stats = summarize_input_returns(in_sample)

    bin_counts = [5, 10, 15, 20, 30, 50, 100]
    target_for_bins = float(targets[len(targets) // 2])
    sensitivity_df, distance_df, governance = run_bin_sensitivity(
        in_sample=in_sample,
        candidates=candidates,
        tickers=tickers,
        bin_counts=bin_counts,
        target_weekly_return=target_for_bins,
    )
    rolling_returns, rolling_weights, rolling_summary = run_rolling_rebalance(
        all_returns=returns,
        out_sample=out_sample,
        candidates=candidates,
        static_points=selected,
        tickers=tickers,
        n_bins=args.bin_count,
        lookback_weeks=args.rolling_lookback_weeks,
        max_weight=args.max_weight,
        risk_free_rate=args.risk_free_rate,
    )

    summary_df.to_csv(args.output_dir / "summary_portfolios.csv", index=False)
    frontier_df.to_csv(args.output_dir / "frontiers.csv", index=False)
    input_stats.to_csv(args.output_dir / "input_return_stats.csv", index=False)
    sensitivity_df.to_csv(args.output_dir / "bin_sensitivity.csv", index=False)
    distance_df.to_csv(args.output_dir / "bin_weight_distances.csv", index=False)
    rolling_returns.to_csv(args.output_dir / "rolling_returns.csv", index=False)
    rolling_weights.to_csv(args.output_dir / "rolling_weights.csv", index=False)
    rolling_summary.to_csv(args.output_dir / "rolling_summary.csv", index=False)

    figure_paths = [
        plot_frontiers(frontier_df, args.output_dir),
        plot_weight_comparison(summary_df, tickers, args.output_dir),
        plot_oos_paths(selected, out_sample, args.output_dir),
        plot_bin_sensitivity(sensitivity_df, distance_df, args.output_dir),
        plot_repo_outperformance(selected, out_sample, args.output_dir),
        plot_rolling_vs_static_wealth(rolling_returns, args.output_dir),
        plot_rolling_stability(rolling_weights, args.output_dir),
    ]
    markdown_path = write_markdown_summary(
        output_dir=args.output_dir,
        summary_df=summary_df,
        governance=governance,
        data_source=args.data_source,
        tickers=tickers,
        rolling_summary=rolling_summary,
    )

    metadata = {
        "data_source": args.data_source,
        "local_data_dir": str(args.local_data_dir),
        "tickers": tickers,
        "in_sample": [args.in_start, args.in_end],
        "out_of_sample": [args.oos_start, args.oos_end],
        "bin_count": args.bin_count,
        "frontier_points": args.frontier_points,
        "n_random_requested": args.n_random,
        "n_candidate_weights": int(len(candidates)),
        "rolling_lookback_weeks": args.rolling_lookback_weeks,
        "max_weight": args.max_weight,
        "seed": args.seed,
        "governance": governance,
        "outputs": {
            "summary_portfolios": str(args.output_dir / "summary_portfolios.csv"),
            "frontiers": str(args.output_dir / "frontiers.csv"),
            "input_return_stats": str(args.output_dir / "input_return_stats.csv"),
            "bin_sensitivity": str(args.output_dir / "bin_sensitivity.csv"),
            "bin_weight_distances": str(args.output_dir / "bin_weight_distances.csv"),
            "rolling_returns": str(args.output_dir / "rolling_returns.csv"),
            "rolling_weights": str(args.output_dir / "rolling_weights.csv"),
            "rolling_summary": str(args.output_dir / "rolling_summary.csv"),
            "summary_markdown": str(markdown_path),
            "figures": [str(path) for path in figure_paths],
        },
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Done. Outputs written to: {args.output_dir.resolve()}")
    print(markdown_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
