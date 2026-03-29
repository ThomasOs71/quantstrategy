from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig, build_expected_return_vector
from .metrics import compute_portfolio_metrics, weighted_var_cvar
from .optimization import (
    SolveResult,
    solve_architecture_A,
    solve_architecture_B,
    solve_architecture_C,
)

ARCHITECTURE_LABELS = {
    "A": "Architecture A: Standard Optimization",
    "B": "Architecture B: Stronger Single-Stage",
    "C": "Architecture C: Stronger Single-Stage + Lexicographic Stage 2",
}


def align_config_to_scenario_feasibility(
    *,
    config: ExperimentConfig,
    R: np.ndarray,
    q: np.ndarray,
    scenario_metadata: dict[str, Any],
) -> tuple[ExperimentConfig, dict[str, Any]]:
    """
    Summarize the benchmark's demeaned portfolio CVaR over the sweep and, in
    absolute-limit mode, optionally lift the absolute cap just enough to keep
    the benchmark feasible.
    """

    limit_mode = config.optimization.cvar_portfolio_limit_mode
    if limit_mode not in {"absolute", "benchmark_factor"}:
        raise ValueError(
            "cvar_portfolio_limit_mode must be either 'absolute' or 'benchmark_factor'."
        )

    requested_limit = float(config.optimization.cvar_portfolio_limit)
    benchmark_factor = float(config.optimization.cvar_portfolio_benchmark_factor)
    if benchmark_factor <= 0.0:
        raise ValueError("cvar_portfolio_benchmark_factor must be positive.")

    benchmark_cvars: list[tuple[float, float]] = []
    for mu_europe in config.mu_europe_grid:
        mu_obj = build_expected_return_vector(mu_europe=mu_europe, config=config)
        risk_R, _ = build_scenario_views(
            R=R,
            mu_obj=mu_obj,
            scenario_metadata=scenario_metadata,
        )
        benchmark_cvar = compute_benchmark_portfolio_cvar_demeaned(
            risk_R=risk_R,
            q=q,
            config=config,
        )
        benchmark_cvars.append((float(mu_europe), float(benchmark_cvar)))

    benchmark_reference_mu_min, benchmark_cvar_min = min(benchmark_cvars, key=lambda item: item[1])
    benchmark_reference_mu_max, benchmark_cvar_max = max(benchmark_cvars, key=lambda item: item[1])

    effective_limit = requested_limit
    adjusted = False
    if limit_mode == "absolute" and config.optimization.auto_adjust_portfolio_cvar_to_benchmark:
        minimum_benchmark_feasible_limit = (
            benchmark_cvar_max + config.optimization.portfolio_cvar_adjustment_buffer
        )
        if requested_limit < minimum_benchmark_feasible_limit:
            effective_limit = minimum_benchmark_feasible_limit
            adjusted = True

    if adjusted:
        config = replace(
            config,
            optimization=replace(config.optimization, cvar_portfolio_limit=effective_limit),
        )

    metadata = {
        "portfolio_cvar_limit_mode": limit_mode,
        "cvar_portfolio_benchmark_factor": benchmark_factor,
        "requested_portfolio_cvar_limit": requested_limit,
        "effective_portfolio_cvar_limit": effective_limit if limit_mode == "absolute" else None,
        "benchmark_portfolio_cvar": benchmark_cvar_max,
        "benchmark_portfolio_cvar_reference_mu_europe": benchmark_reference_mu_max,
        "benchmark_portfolio_cvar_demeaned_min": benchmark_cvar_min,
        "benchmark_portfolio_cvar_demeaned_min_reference_mu_europe": benchmark_reference_mu_min,
        "benchmark_portfolio_cvar_demeaned_max": benchmark_cvar_max,
        "benchmark_portfolio_cvar_demeaned_max_reference_mu_europe": benchmark_reference_mu_max,
        "benchmark_portfolio_cvar_demeaned_mean": float(
            np.mean([benchmark_cvar for _, benchmark_cvar in benchmark_cvars])
        ),
        "benchmark_factor_portfolio_cvar_limit_min": benchmark_cvar_min * benchmark_factor,
        "benchmark_factor_portfolio_cvar_limit_max": benchmark_cvar_max * benchmark_factor,
        "portfolio_cvar_limit_adjusted": adjusted,
    }
    return config, metadata


def build_scenario_views(
    *,
    R: np.ndarray,
    mu_obj: np.ndarray,
    scenario_metadata: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return both the demeaned risk matrix and the full-return reporting matrix.

    The objective continues to use the explicit expected-return vector `mu_obj`,
    while CVaR is evaluated on demeaned returns. When source scenarios are
    already demeaned, we use them directly for risk and reconstruct full returns
    only for secondary reporting. When source scenarios already include a mean,
    we subtract the current objective mean vector to obtain the demeaned risk
    basis.
    """

    mu_vector = np.asarray(mu_obj, dtype=float).reshape(1, -1)
    if scenario_metadata.get("demeaned", False):
        risk_R = np.asarray(R, dtype=float)
        full_R = risk_R + mu_vector
        return risk_R, full_R

    full_R = np.asarray(R, dtype=float)
    risk_R = full_R - mu_vector
    return risk_R, full_R


def compute_benchmark_portfolio_cvar_demeaned(
    *,
    risk_R: np.ndarray,
    q: np.ndarray,
    config: ExperimentConfig,
) -> float:
    _, benchmark_cvar = weighted_var_cvar(
        losses=-(risk_R @ config.benchmark_weights),
        q=q,
        alpha=config.optimization.alpha,
    )
    return float(benchmark_cvar)


def compute_effective_portfolio_cvar_limit(
    *,
    config: ExperimentConfig,
    benchmark_portfolio_cvar_demeaned: float,
) -> float:
    limit_mode = config.optimization.cvar_portfolio_limit_mode
    if limit_mode == "benchmark_factor":
        return float(
            benchmark_portfolio_cvar_demeaned * config.optimization.cvar_portfolio_benchmark_factor
        )
    if limit_mode == "absolute":
        return float(config.optimization.cvar_portfolio_limit)
    raise ValueError(
        "cvar_portfolio_limit_mode must be either 'absolute' or 'benchmark_factor'."
    )


def _empty_weight_dict(prefix: str, assets: tuple[str, ...]) -> dict[str, float]:
    return {f"{prefix}_{asset}": np.nan for asset in assets}


def _weights_to_dict(prefix: str, weights: np.ndarray | None, assets: tuple[str, ...]) -> dict[str, float]:
    if weights is None:
        return _empty_weight_dict(prefix, assets)
    return {
        f"{prefix}_{asset}": float(weight)
        for asset, weight in zip(assets, np.asarray(weights, dtype=float).reshape(-1))
    }


def _active_weights_to_dict(
    prefix: str,
    weights: np.ndarray | None,
    w_bm: np.ndarray,
    assets: tuple[str, ...],
) -> dict[str, float]:
    if weights is None:
        return _empty_weight_dict(prefix, assets)
    active_weights = np.asarray(weights, dtype=float).reshape(-1) - w_bm
    return {f"{prefix}_{asset}": float(weight) for asset, weight in zip(assets, active_weights)}


def _solution_to_row(
    *,
    result: SolveResult,
    mu_europe: float,
    mu_obj: np.ndarray,
    config: ExperimentConfig,
    risk_R: np.ndarray,
    full_R: np.ndarray,
    q: np.ndarray,
    scenario_metadata: dict[str, Any],
    w_prev: np.ndarray | None,
    benchmark_portfolio_cvar_demeaned: float,
    effective_portfolio_cvar_limit: float,
) -> dict[str, Any]:
    metrics = compute_portfolio_metrics(
        weights=result.weights,
        risk_R=risk_R,
        full_R=full_R,
        q=q,
        mu_obj=mu_obj,
        w_bm=config.benchmark_weights,
        alpha=config.optimization.alpha,
        w_prev=w_prev,
    )

    row: dict[str, Any] = {
        "architecture": result.architecture,
        "architecture_label": ARCHITECTURE_LABELS[result.architecture],
        "mu_europe": float(mu_europe),
        "status": result.status,
        "stage1_status": result.stage1_status,
        "stage2_status": result.stage2_status,
        "solver": result.solver_used,
        "stage1_solver": result.stage1_solver_used,
        "stage2_solver": result.stage2_solver_used,
        "solve_time_seconds": result.solve_time_seconds,
        "stage1_solve_time_seconds": result.stage1_solve_time_seconds,
        "stage2_solve_time_seconds": result.stage2_solve_time_seconds,
        "turnover_constraint_applied": result.turnover_constraint_applied,
        "turnover_reference_available": w_prev is not None,
        "n_scenarios": int(risk_R.shape[0]),
        "scenario_source": scenario_metadata["source"],
        "scenario_seed": scenario_metadata.get("seed"),
        "portfolio_cvar_limit_mode": config.optimization.cvar_portfolio_limit_mode,
        "cvar_portfolio_benchmark_factor": float(config.optimization.cvar_portfolio_benchmark_factor),
        "requested_portfolio_cvar_limit": float(config.optimization.cvar_portfolio_limit),
        "effective_portfolio_cvar_limit": float(effective_portfolio_cvar_limit),
        "benchmark_portfolio_cvar_demeaned": float(benchmark_portfolio_cvar_demeaned),
        "cvar_te_limit": float(config.optimization.cvar_te_limit),
        "expected_return": metrics["expected_return"],
        "stage1_expected_return": result.stage1_expected_return,
        "stage2_expected_return": metrics["expected_return"] if result.architecture == "C" else np.nan,
        "objective_loss_pct": result.objective_loss_pct,
        "stage2_bm_distance_objective": result.stage2_benchmark_objective,
        "portfolio_cvar": metrics["portfolio_cvar"],
        "portfolio_cvar_demeaned": metrics["portfolio_cvar_demeaned"],
        "portfolio_cvar_full": metrics["portfolio_cvar_full"],
        "te_cvar": metrics["te_cvar"],
        "te_cvar_demeaned": metrics["te_cvar_demeaned"],
        "te_cvar_full": metrics["te_cvar_full"],
        "bm_dist_l2": metrics["bm_dist_l2"],
        "bm_dist_l1": metrics["bm_dist_l1"],
        "max_weight": metrics["max_weight"],
        "herfindahl": metrics["herfindahl"],
        "effective_n": metrics["effective_n"],
        "turnover": metrics["turnover"],
        "error_message": result.error_message,
    }

    row.update(_weights_to_dict("w", result.weights, config.assets))
    row.update(_active_weights_to_dict("aw", result.weights, config.benchmark_weights, config.assets))
    row.update(_weights_to_dict("stage1_w", result.stage1_weights, config.assets))
    row.update(
        _active_weights_to_dict(
            "stage1_aw",
            result.stage1_weights,
            config.benchmark_weights,
            config.assets,
        )
    )
    return row


def run_sweep_experiment(
    *,
    config: ExperimentConfig,
    R: np.ndarray,
    q: np.ndarray,
    scenario_metadata: dict[str, Any],
    architectures: tuple[str, ...] = ("A", "B", "C"),
) -> pd.DataFrame:
    """Run the expected-return sweep across the three architectures."""

    requested_architectures = tuple(dict.fromkeys(architectures))
    invalid_architectures = set(requested_architectures) - set(ARCHITECTURE_LABELS)
    if invalid_architectures:
        raise ValueError(f"Unsupported architectures requested: {sorted(invalid_architectures)}")

    results: list[dict[str, Any]] = []
    turnover_reference = config.benchmark_weights.copy()

    for mu_europe in config.mu_europe_grid:
        mu_obj = build_expected_return_vector(mu_europe=mu_europe, config=config)
        risk_R, full_R = build_scenario_views(
            R=R,
            mu_obj=mu_obj,
            scenario_metadata=scenario_metadata,
        )
        benchmark_portfolio_cvar_demeaned = compute_benchmark_portfolio_cvar_demeaned(
            risk_R=risk_R,
            q=q,
            config=config,
        )
        effective_portfolio_cvar_limit = compute_effective_portfolio_cvar_limit(
            config=config,
            benchmark_portfolio_cvar_demeaned=benchmark_portfolio_cvar_demeaned,
        )
        optimization_config = replace(
            config.optimization,
            cvar_portfolio_limit=effective_portfolio_cvar_limit,
        )

        if "A" in requested_architectures:
            solution_a = solve_architecture_A(
                R=risk_R,
                q=q,
                mu_obj=mu_obj,
                w_bm=config.benchmark_weights,
                lb=config.lower_bounds,
                ub=config.upper_bounds,
                config=optimization_config,
                w_prev=turnover_reference,
            )
            results.append(
                _solution_to_row(
                    result=solution_a,
                    mu_europe=mu_europe,
                    mu_obj=mu_obj,
                    config=config,
                    risk_R=risk_R,
                    full_R=full_R,
                    q=q,
                    scenario_metadata=scenario_metadata,
                    w_prev=turnover_reference,
                    benchmark_portfolio_cvar_demeaned=benchmark_portfolio_cvar_demeaned,
                    effective_portfolio_cvar_limit=effective_portfolio_cvar_limit,
                )
            )

        solution_b: SolveResult | None = None
        if "B" in requested_architectures or "C" in requested_architectures:
            solution_b = solve_architecture_B(
                R=risk_R,
                q=q,
                mu_obj=mu_obj,
                w_bm=config.benchmark_weights,
                lb=config.lower_bounds,
                ub=config.upper_bounds,
                config=optimization_config,
                w_prev=turnover_reference,
            )
        if "B" in requested_architectures and solution_b is not None:
            results.append(
                _solution_to_row(
                    result=solution_b,
                    mu_europe=mu_europe,
                    mu_obj=mu_obj,
                    config=config,
                    risk_R=risk_R,
                    full_R=full_R,
                    q=q,
                    scenario_metadata=scenario_metadata,
                    w_prev=turnover_reference,
                    benchmark_portfolio_cvar_demeaned=benchmark_portfolio_cvar_demeaned,
                    effective_portfolio_cvar_limit=effective_portfolio_cvar_limit,
                )
            )

        if "C" in requested_architectures:
            solution_c = solve_architecture_C(
                R=risk_R,
                q=q,
                mu_obj=mu_obj,
                w_bm=config.benchmark_weights,
                lb=config.lower_bounds,
                ub=config.upper_bounds,
                config=optimization_config,
                w_prev=turnover_reference,
                stage1_result=solution_b,
            )
            results.append(
                _solution_to_row(
                    result=solution_c,
                    mu_europe=mu_europe,
                    mu_obj=mu_obj,
                    config=config,
                    risk_R=risk_R,
                    full_R=full_R,
                    q=q,
                    scenario_metadata=scenario_metadata,
                    w_prev=turnover_reference,
                    benchmark_portfolio_cvar_demeaned=benchmark_portfolio_cvar_demeaned,
                    effective_portfolio_cvar_limit=effective_portfolio_cvar_limit,
                )
            )

    return pd.DataFrame(results)
