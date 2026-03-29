from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cvxpy as cp
import numpy as np

from .config import OptimizationConfig

OPTIMAL_STATUSES = {"optimal", "optimal_inaccurate"}
TERMINAL_STATUSES = OPTIMAL_STATUSES | {
    "infeasible",
    "infeasible_inaccurate",
    "unbounded",
    "unbounded_inaccurate",
}


@dataclass
class SolveResult:
    architecture: str
    status: str
    weights: np.ndarray | None
    expected_return_objective: float | None
    solver_used: str | None
    solve_time_seconds: float | None
    stage1_weights: np.ndarray | None = None
    stage1_status: str | None = None
    stage2_status: str | None = None
    stage1_expected_return: float | None = None
    stage2_benchmark_objective: float | None = None
    objective_loss_pct: float | None = None
    stage1_solver_used: str | None = None
    stage2_solver_used: str | None = None
    stage1_solve_time_seconds: float | None = None
    stage2_solve_time_seconds: float | None = None
    turnover_constraint_applied: bool = False
    error_message: str | None = None


@dataclass
class ProblemArtifacts:
    problem: cp.Problem
    w: cp.Variable
    expected_return_expr: cp.Expression
    portfolio_cvar_expr: cp.Expression
    te_cvar_expr: cp.Expression | None
    turnover_expr: cp.Expression | None


def is_optimal_status(status: str | None) -> bool:
    return status in OPTIMAL_STATUSES


def _solver_options(solver_name: str, verbose: bool) -> dict[str, Any]:
    if solver_name == "CLARABEL":
        return {"verbose": verbose}
    if solver_name == "ECOS":
        return {"verbose": verbose, "abstol": 1e-8, "reltol": 1e-8, "feastol": 1e-8}
    if solver_name == "OSQP":
        return {"verbose": verbose, "eps_abs": 1e-7, "eps_rel": 1e-7, "max_iter": 50000}
    if solver_name == "SCS":
        return {"verbose": verbose, "eps": 1e-6, "max_iters": 50000}
    return {"verbose": verbose}


def _solve_problem(
    problem: cp.Problem,
    config: OptimizationConfig,
) -> tuple[str | None, str | None, float | None, str | None]:
    installed_solvers = set(cp.installed_solvers())
    last_error_message: str | None = None

    for solver_name in config.solver_preference:
        if solver_name not in installed_solvers:
            continue
        try:
            problem.solve(solver=solver_name, **_solver_options(solver_name, config.solver_verbose))
        except cp.SolverError as exc:
            last_error_message = f"{solver_name}: {exc}"
            continue

        solve_time = problem.solver_stats.solve_time
        status = problem.status
        if status in TERMINAL_STATUSES:
            return solver_name, status, solve_time, None

        last_error_message = f"{solver_name}: returned non-terminal status {status}"

    return None, None, None, last_error_message or "No configured solver was available."


def _build_weighted_cvar(
    losses: cp.Expression,
    q: np.ndarray,
    alpha: float,
    prefix: str,
) -> tuple[cp.Expression, list[cp.Constraint]]:
    eta = cp.Variable(name=f"{prefix}_eta")
    z = cp.Variable(losses.shape[0], nonneg=True, name=f"{prefix}_z")
    constraints = [z >= losses - eta]
    cvar_expr = eta + cp.sum(cp.multiply(q, z)) / (1.0 - alpha)
    return cvar_expr, constraints


def _build_turnover_expression(
    w: cp.Variable,
    w_prev: np.ndarray,
    prefix: str,
) -> tuple[cp.Expression, list[cp.Constraint]]:
    u = cp.Variable(w.shape[0], nonneg=True, name=f"{prefix}_turnover_slack")
    constraints = [u >= w - w_prev, u >= -(w - w_prev)]
    turnover_expr = cp.sum(u)
    return turnover_expr, constraints


def _build_problem(
    *,
    objective_kind: str,
    sense: str,
    R: np.ndarray,
    q: np.ndarray,
    mu_obj: np.ndarray,
    w_bm: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    config: OptimizationConfig,
    include_te_constraint: bool,
    w_prev: np.ndarray | None,
    include_turnover_constraint: bool,
    extra_constraints: list[cp.Constraint] | None = None,
    prefix: str,
) -> ProblemArtifacts:
    n_assets = R.shape[1]
    w = cp.Variable(n_assets, name=f"{prefix}_w")
    expected_return_expr = cp.sum(cp.multiply(mu_obj, w))

    constraints: list[cp.Constraint] = [
        cp.sum(w) == 1.0,
        w >= 0.0,
        w >= lb,
        w <= ub,
    ]

    portfolio_losses = -(R @ w)
    portfolio_cvar_expr, portfolio_cvar_constraints = _build_weighted_cvar(
        losses=portfolio_losses,
        q=q,
        alpha=config.alpha,
        prefix=f"{prefix}_portfolio_cvar",
    )
    constraints.extend(portfolio_cvar_constraints)
    constraints.append(portfolio_cvar_expr <= config.cvar_portfolio_limit)

    te_cvar_expr: cp.Expression | None = None
    benchmark_returns = R @ w_bm
    if include_te_constraint:
        active_losses = -((R @ w) - benchmark_returns)
        te_cvar_expr, te_constraints = _build_weighted_cvar(
            losses=active_losses,
            q=q,
            alpha=config.alpha,
            prefix=f"{prefix}_te_cvar",
        )
        constraints.extend(te_constraints)
        constraints.append(te_cvar_expr <= config.cvar_te_limit)

    turnover_expr: cp.Expression | None = None
    if include_turnover_constraint and w_prev is not None:
        turnover_expr, turnover_constraints = _build_turnover_expression(
            w=w,
            w_prev=w_prev,
            prefix=prefix,
        )
        constraints.extend(turnover_constraints)
        constraints.append(turnover_expr <= config.turnover_limit)

    if extra_constraints:
        constraints.extend(extra_constraints)

    if objective_kind == "expected_return":
        objective_expr = expected_return_expr
    elif objective_kind == "benchmark_distance":
        objective_expr = cp.sum_squares(w - w_bm)
    else:
        raise ValueError(f"Unsupported objective kind: {objective_kind}")

    if sense == "max":
        problem = cp.Problem(cp.Maximize(objective_expr), constraints)
    elif sense == "min":
        problem = cp.Problem(cp.Minimize(objective_expr), constraints)
    else:
        raise ValueError(f"Unsupported optimization sense: {sense}")

    return ProblemArtifacts(
        problem=problem,
        w=w,
        expected_return_expr=expected_return_expr,
        portfolio_cvar_expr=portfolio_cvar_expr,
        te_cvar_expr=te_cvar_expr,
        turnover_expr=turnover_expr,
    )


def solve_architecture_A(
    *,
    R: np.ndarray,
    q: np.ndarray,
    mu_obj: np.ndarray,
    w_bm: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    config: OptimizationConfig,
    w_prev: np.ndarray | None = None,
) -> SolveResult:
    turnover_constraint_applied = bool(config.enable_turnover_constraint and w_prev is not None)
    artifacts = _build_problem(
        objective_kind="expected_return",
        sense="max",
        R=R,
        q=q,
        mu_obj=mu_obj,
        w_bm=w_bm,
        lb=lb,
        ub=ub,
        config=config,
        include_te_constraint=False,
        w_prev=w_prev,
        include_turnover_constraint=turnover_constraint_applied,
        prefix="arch_a",
    )

    solver_used, status, solve_time, error_message = _solve_problem(artifacts.problem, config)
    if not is_optimal_status(status):
        return SolveResult(
            architecture="A",
            status=status or "solver_error",
            weights=None,
            expected_return_objective=None,
            solver_used=solver_used,
            solve_time_seconds=solve_time,
            turnover_constraint_applied=turnover_constraint_applied,
            error_message=error_message,
        )

    weights = np.asarray(artifacts.w.value, dtype=float).reshape(-1)
    return SolveResult(
        architecture="A",
        status=status,
        weights=weights,
        expected_return_objective=float(mu_obj @ weights),
        solver_used=solver_used,
        solve_time_seconds=solve_time,
        turnover_constraint_applied=turnover_constraint_applied,
    )


def solve_architecture_B(
    *,
    R: np.ndarray,
    q: np.ndarray,
    mu_obj: np.ndarray,
    w_bm: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    config: OptimizationConfig,
    w_prev: np.ndarray | None = None,
) -> SolveResult:
    turnover_constraint_applied = bool(config.enable_turnover_constraint and w_prev is not None)
    artifacts = _build_problem(
        objective_kind="expected_return",
        sense="max",
        R=R,
        q=q,
        mu_obj=mu_obj,
        w_bm=w_bm,
        lb=lb,
        ub=ub,
        config=config,
        include_te_constraint=True,
        w_prev=w_prev,
        include_turnover_constraint=turnover_constraint_applied,
        prefix="arch_b",
    )

    solver_used, status, solve_time, error_message = _solve_problem(artifacts.problem, config)
    if not is_optimal_status(status):
        return SolveResult(
            architecture="B",
            status=status or "solver_error",
            weights=None,
            expected_return_objective=None,
            solver_used=solver_used,
            solve_time_seconds=solve_time,
            turnover_constraint_applied=turnover_constraint_applied,
            error_message=error_message,
        )

    weights = np.asarray(artifacts.w.value, dtype=float).reshape(-1)
    return SolveResult(
        architecture="B",
        status=status,
        weights=weights,
        expected_return_objective=float(mu_obj @ weights),
        solver_used=solver_used,
        solve_time_seconds=solve_time,
        turnover_constraint_applied=turnover_constraint_applied,
    )


def solve_architecture_C(
    *,
    R: np.ndarray,
    q: np.ndarray,
    mu_obj: np.ndarray,
    w_bm: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    config: OptimizationConfig,
    w_prev: np.ndarray | None = None,
    stage1_result: SolveResult | None = None,
) -> SolveResult:
    if stage1_result is None:
        stage1_result = solve_architecture_B(
            R=R,
            q=q,
            mu_obj=mu_obj,
            w_bm=w_bm,
            lb=lb,
            ub=ub,
            config=config,
            w_prev=w_prev,
        )
    if not is_optimal_status(stage1_result.status):
        return SolveResult(
            architecture="C",
            status=stage1_result.status,
            weights=None,
            expected_return_objective=None,
            solver_used=stage1_result.solver_used,
            solve_time_seconds=stage1_result.solve_time_seconds,
            stage1_weights=None,
            stage1_status=stage1_result.status,
            stage2_status="not_run",
            stage1_expected_return=None,
            stage1_solver_used=stage1_result.solver_used,
            stage1_solve_time_seconds=stage1_result.solve_time_seconds,
            turnover_constraint_applied=stage1_result.turnover_constraint_applied,
            error_message=stage1_result.error_message,
        )

    obj_star = float(mu_obj @ stage1_result.weights)
    turnover_constraint_applied = bool(config.enable_turnover_constraint and w_prev is not None)
    stage2_artifacts = _build_problem(
        objective_kind="benchmark_distance",
        sense="min",
        R=R,
        q=q,
        mu_obj=mu_obj,
        w_bm=w_bm,
        lb=lb,
        ub=ub,
        config=config,
        include_te_constraint=True,
        w_prev=w_prev,
        include_turnover_constraint=turnover_constraint_applied,
        prefix="arch_c_stage2",
    )
    stage2_problem = cp.Problem(
        cp.Minimize(cp.sum_squares(stage2_artifacts.w - w_bm)),
        stage2_artifacts.problem.constraints
        + [stage2_artifacts.expected_return_expr >= (1.0 - config.epsilon) * obj_star],
    )
    stage2_artifacts.problem = stage2_problem

    solver_used, status, solve_time, error_message = _solve_problem(stage2_artifacts.problem, config)
    total_solve_time = (stage1_result.solve_time_seconds or 0.0) + (solve_time or 0.0)
    if not is_optimal_status(status):
        return SolveResult(
            architecture="C",
            status=status or "solver_error",
            weights=None,
            expected_return_objective=None,
            solver_used=solver_used,
            solve_time_seconds=total_solve_time,
            stage1_weights=stage1_result.weights,
            stage1_status=stage1_result.status,
            stage2_status=status or "solver_error",
            stage1_expected_return=obj_star,
            stage1_solver_used=stage1_result.solver_used,
            stage2_solver_used=solver_used,
            stage1_solve_time_seconds=stage1_result.solve_time_seconds,
            stage2_solve_time_seconds=solve_time,
            turnover_constraint_applied=turnover_constraint_applied,
            error_message=error_message,
        )

    final_weights = np.asarray(stage2_artifacts.w.value, dtype=float).reshape(-1)
    final_expected_return = float(mu_obj @ final_weights)
    objective_loss_pct = max(obj_star - final_expected_return, 0.0) / max(abs(obj_star), 1e-12)

    return SolveResult(
        architecture="C",
        status=status,
        weights=final_weights,
        expected_return_objective=final_expected_return,
        solver_used=solver_used,
        solve_time_seconds=total_solve_time,
        stage1_weights=stage1_result.weights,
        stage1_status=stage1_result.status,
        stage2_status=status,
        stage1_expected_return=obj_star,
        stage2_benchmark_objective=float(np.sum((final_weights - w_bm) ** 2)),
        objective_loss_pct=objective_loss_pct,
        stage1_solver_used=stage1_result.solver_used,
        stage2_solver_used=solver_used,
        stage1_solve_time_seconds=stage1_result.solve_time_seconds,
        stage2_solve_time_seconds=solve_time,
        turnover_constraint_applied=turnover_constraint_applied,
    )
