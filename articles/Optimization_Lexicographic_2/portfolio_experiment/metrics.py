from __future__ import annotations

import numpy as np
import pandas as pd


def weighted_var_cvar(losses: np.ndarray, q: np.ndarray, alpha: float) -> tuple[float, float]:
    """Compute weighted VaR and CVaR using the Rockafellar-Uryasev tail formula."""

    losses = np.asarray(losses, dtype=float).reshape(-1)
    q = np.asarray(q, dtype=float).reshape(-1)
    if losses.shape[0] != q.shape[0]:
        raise ValueError("losses and q must have the same length.")
    if np.any(q < 0):
        raise ValueError("Scenario probabilities must be non-negative.")

    q = q / q.sum()
    order = np.argsort(losses)
    losses_sorted = losses[order]
    q_sorted = q[order]
    cumulative = np.cumsum(q_sorted)
    var_index = int(np.searchsorted(cumulative, alpha, side="left"))
    var_index = min(var_index, losses_sorted.shape[0] - 1)
    var_alpha = float(losses_sorted[var_index])
    tail_excess = np.maximum(losses - var_alpha, 0.0)
    cvar_alpha = float(var_alpha + np.sum(q * tail_excess) / max(1.0 - alpha, 1e-12))
    return var_alpha, cvar_alpha


def compute_portfolio_metrics(
    *,
    weights: np.ndarray | None,
    risk_R: np.ndarray,
    full_R: np.ndarray,
    q: np.ndarray,
    mu_obj: np.ndarray,
    w_bm: np.ndarray,
    alpha: float,
    w_prev: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute portfolio, active-risk, and governance metrics from final weights."""

    metrics = {
        "expected_return": np.nan,
        "portfolio_cvar": np.nan,
        "portfolio_cvar_demeaned": np.nan,
        "portfolio_cvar_full": np.nan,
        "te_cvar": np.nan,
        "te_cvar_demeaned": np.nan,
        "te_cvar_full": np.nan,
        "bm_dist_l2": np.nan,
        "bm_dist_l1": np.nan,
        "max_weight": np.nan,
        "herfindahl": np.nan,
        "effective_n": np.nan,
        "turnover": np.nan,
    }
    if weights is None:
        return metrics

    w = np.asarray(weights, dtype=float).reshape(-1)
    portfolio_returns_demeaned = risk_R @ w
    benchmark_returns_demeaned = risk_R @ w_bm
    portfolio_losses_demeaned = -portfolio_returns_demeaned
    active_losses_demeaned = -(portfolio_returns_demeaned - benchmark_returns_demeaned)

    portfolio_returns_full = full_R @ w
    benchmark_returns_full = full_R @ w_bm
    portfolio_losses_full = -portfolio_returns_full
    active_losses_full = -(portfolio_returns_full - benchmark_returns_full)

    _, portfolio_cvar_demeaned = weighted_var_cvar(portfolio_losses_demeaned, q, alpha)
    _, te_cvar_demeaned = weighted_var_cvar(active_losses_demeaned, q, alpha)
    _, portfolio_cvar_full = weighted_var_cvar(portfolio_losses_full, q, alpha)
    _, te_cvar_full = weighted_var_cvar(active_losses_full, q, alpha)
    herfindahl = float(np.sum(w**2))

    metrics.update(
        {
            "expected_return": float(mu_obj @ w),
            "portfolio_cvar": portfolio_cvar_demeaned,
            "portfolio_cvar_demeaned": portfolio_cvar_demeaned,
            "portfolio_cvar_full": portfolio_cvar_full,
            "te_cvar": te_cvar_demeaned,
            "te_cvar_demeaned": te_cvar_demeaned,
            "te_cvar_full": te_cvar_full,
            "bm_dist_l2": float(np.linalg.norm(w - w_bm, ord=2)),
            "bm_dist_l1": float(np.linalg.norm(w - w_bm, ord=1)),
            "max_weight": float(np.max(w)),
            "herfindahl": herfindahl,
            "effective_n": float(1.0 / herfindahl) if herfindahl > 0 else np.nan,
            "turnover": float(np.sum(np.abs(w - w_prev))) if w_prev is not None else np.nan,
        }
    )
    return metrics


def compute_instability_metrics(results_df: pd.DataFrame, assets: tuple[str, ...]) -> pd.DataFrame:
    """Augment the raw results with adjacent-point stability diagnostics."""

    enriched = results_df.copy()
    weight_columns = [f"w_{asset}" for asset in assets]

    for asset in assets:
        enriched[f"adj_abs_change_{asset}"] = np.nan
        enriched[f"adj_change_{asset}"] = np.nan

    enriched["adj_l1_change"] = np.nan
    enriched["adj_l2_change"] = np.nan
    enriched["adj_max_jump"] = np.nan
    enriched["instability_score"] = np.nan

    for architecture, group in enriched.groupby("architecture", sort=False):
        group_sorted = group.sort_values("mu_europe", ascending=False)
        previous_weights: np.ndarray | None = None

        for index, row in group_sorted.iterrows():
            current_weights = row[weight_columns].to_numpy(dtype=float)
            if np.isnan(current_weights).any() or previous_weights is None:
                previous_weights = None if np.isnan(current_weights).any() else current_weights
                continue

            diff = current_weights - previous_weights
            abs_diff = np.abs(diff)
            for asset, asset_abs_change, asset_change in zip(assets, abs_diff, diff):
                enriched.at[index, f"adj_abs_change_{asset}"] = float(asset_abs_change)
                enriched.at[index, f"adj_change_{asset}"] = float(asset_change)

            enriched.at[index, "adj_l1_change"] = float(np.sum(abs_diff))
            enriched.at[index, "adj_l2_change"] = float(np.linalg.norm(diff, ord=2))
            enriched.at[index, "adj_max_jump"] = float(np.max(abs_diff))
            enriched.at[index, "instability_score"] = float(
                np.sum(abs_diff) + np.linalg.norm(diff, ord=2) + np.max(abs_diff)
            )
            previous_weights = current_weights

    return enriched
