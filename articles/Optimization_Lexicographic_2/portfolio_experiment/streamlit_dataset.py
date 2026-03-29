from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import (
    ExperimentConfig,
    apply_overrides,
    build_default_config,
    build_inclusive_grid,
    to_serializable_dict,
)
from .experiment import run_sweep_experiment
from .metrics import compute_instability_metrics
from .scenarios import generate_or_load_scenarios

STREAMLIT_MU_EUROPE_GRID = build_inclusive_grid(0.070, 0.060, -0.0005)
STREAMLIT_PORTFOLIO_RISK_FACTORS = np.array([0.95, 1.00, 1.05, 1.10], dtype=float)
STREAMLIT_TE_CVAR_LIMITS = np.array([0.01, 0.02, 0.03, 0.04, 0.05], dtype=float)
STREAMLIT_TURNOVER_LIMITS = np.array([0.20, 0.30, 0.40], dtype=float)
STREAMLIT_BASE_OVERRIDES: dict[str, Any] = {
    "alpha": 0.90,
    "epsilon": 0.005,
    "enable_turnover_constraint": True,
    "auto_adjust_portfolio_cvar_to_benchmark": False,
    "cvar_portfolio_limit_mode": "benchmark_factor",
    "solver_preference": ("ECOS", "CLARABEL", "OSQP", "SCS"),
}
FEASIBLE_STATUSES = {"optimal", "optimal_inaccurate"}
ARCHITECTURE_ORDER = {"A": 0, "B": 1, "C": 2}


def get_streamlit_parameter_grid(smoke: bool = False) -> dict[str, np.ndarray]:
    """Return the discrete parameter grid used by the Streamlit dataset builder."""

    if smoke:
        return {
            "mu_europe": np.array([0.0700, 0.0650, 0.0600], dtype=float),
            "portfolio_risk_factor": np.array([0.95, 1.05], dtype=float),
            "te_cvar_limit": np.array([0.02, 0.04], dtype=float),
            "turnover_limit": np.array([0.25, 0.40], dtype=float),
        }

    return {
        "mu_europe": STREAMLIT_MU_EUROPE_GRID.copy(),
        "portfolio_risk_factor": STREAMLIT_PORTFOLIO_RISK_FACTORS.copy(),
        "te_cvar_limit": STREAMLIT_TE_CVAR_LIMITS.copy(),
        "turnover_limit": STREAMLIT_TURNOVER_LIMITS.copy(),
    }


def build_streamlit_base_config(output_dir: str | Path) -> ExperimentConfig:
    """Create the reusable base configuration for the precomputed app dataset."""

    config = build_default_config(output_dir=output_dir)
    return apply_overrides(config, STREAMLIT_BASE_OVERRIDES)


def _prepare_frame(
    frame: pd.DataFrame,
    *,
    portfolio_risk_factor: float,
    te_cvar_limit: float,
    turnover_limit: float,
) -> pd.DataFrame:
    prepared = frame.copy()
    prepared["portfolio_risk_factor"] = float(portfolio_risk_factor)
    prepared["te_cvar_limit"] = float(te_cvar_limit)
    prepared["turnover_limit"] = float(turnover_limit)
    prepared["is_feasible"] = prepared["status"].isin(FEASIBLE_STATUSES)
    prepared["cvar_te_limit"] = float(te_cvar_limit)
    return prepared


def _build_task_config(
    base_config: ExperimentConfig,
    *,
    portfolio_risk_factor: float,
    te_cvar_limit: float,
    turnover_limit: float,
) -> ExperimentConfig:
    return apply_overrides(
        base_config,
        {
            "cvar_portfolio_benchmark_factor": float(portfolio_risk_factor),
            "cvar_te_limit": float(te_cvar_limit),
            "turnover_limit": float(turnover_limit),
        },
    )


def _run_architecture_subset_task(
    *,
    base_config: ExperimentConfig,
    R: np.ndarray,
    q: np.ndarray,
    scenario_metadata: dict[str, Any],
    portfolio_risk_factor: float,
    te_cvar_limit: float,
    turnover_limit: float,
    architectures: tuple[str, ...],
) -> pd.DataFrame:
    task_config = _build_task_config(
        base_config,
        portfolio_risk_factor=portfolio_risk_factor,
        te_cvar_limit=te_cvar_limit,
        turnover_limit=turnover_limit,
    )
    sweep = run_sweep_experiment(
        config=task_config,
        R=R,
        q=q,
        scenario_metadata=scenario_metadata,
        architectures=architectures,
    )
    sweep = compute_instability_metrics(sweep, task_config.assets)
    return _prepare_frame(
        sweep,
        portfolio_risk_factor=portfolio_risk_factor,
        te_cvar_limit=te_cvar_limit,
        turnover_limit=turnover_limit,
    )


def _architecture_a_task_payload(
    base_config: ExperimentConfig,
    R: np.ndarray,
    q: np.ndarray,
    scenario_metadata: dict[str, Any],
    portfolio_risk_factor: float,
    turnover_limit: float,
) -> pd.DataFrame:
    default_te_limit = float(STREAMLIT_TE_CVAR_LIMITS[0])
    return _run_architecture_subset_task(
        base_config=base_config,
        R=R,
        q=q,
        scenario_metadata=scenario_metadata,
        portfolio_risk_factor=portfolio_risk_factor,
        te_cvar_limit=default_te_limit,
        turnover_limit=turnover_limit,
        architectures=("A",),
    )


def _architectures_bc_task_payload(
    base_config: ExperimentConfig,
    R: np.ndarray,
    q: np.ndarray,
    scenario_metadata: dict[str, Any],
    portfolio_risk_factor: float,
    te_cvar_limit: float,
    turnover_limit: float,
) -> pd.DataFrame:
    return _run_architecture_subset_task(
        base_config=base_config,
        R=R,
        q=q,
        scenario_metadata=scenario_metadata,
        portfolio_risk_factor=portfolio_risk_factor,
        te_cvar_limit=te_cvar_limit,
        turnover_limit=turnover_limit,
        architectures=("B", "C"),
    )


def _collect_frames(
    *,
    worker_fn,
    task_args: list[tuple[Any, ...]],
    workers: int,
) -> list[pd.DataFrame]:
    if workers <= 1:
        return [worker_fn(*args) for args in task_args]

    frames: list[pd.DataFrame] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_map = {executor.submit(worker_fn, *args): args for args in task_args}
        for future in as_completed(future_map):
            frames.append(future.result())
    return frames


def _replicate_architecture_a_across_te_limits(
    a_frame: pd.DataFrame,
    te_cvar_limits: np.ndarray,
) -> list[pd.DataFrame]:
    replicated: list[pd.DataFrame] = []
    for te_cvar_limit in te_cvar_limits:
        copied = a_frame.copy()
        copied["te_cvar_limit"] = float(te_cvar_limit)
        copied["cvar_te_limit"] = float(te_cvar_limit)
        replicated.append(copied)
    return replicated


def _sort_results_frame(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["_architecture_order"] = ordered["architecture"].map(ARCHITECTURE_ORDER)
    ordered = ordered.sort_values(
        [
            "portfolio_risk_factor",
            "te_cvar_limit",
            "turnover_limit",
            "mu_europe",
            "_architecture_order",
        ],
        ascending=[True, True, True, False, True],
    ).drop(columns="_architecture_order")
    return ordered.reset_index(drop=True)


def _weighted_asset_volatilities(
    R: np.ndarray,
    q: np.ndarray,
    assets: tuple[str, ...],
) -> list[dict[str, float]]:
    """Build static asset reference rows from the scenario distribution."""

    scenario_matrix = np.asarray(R, dtype=float)
    probabilities = np.asarray(q, dtype=float).reshape(-1)
    probabilities = probabilities / probabilities.sum()

    means = np.average(scenario_matrix, axis=0, weights=probabilities)
    variances = np.average((scenario_matrix - means) ** 2, axis=0, weights=probabilities)

    return [
        {
            "asset": asset,
            "volatility": float(np.sqrt(max(variance, 0.0))),
        }
        for asset, variance in zip(assets, variances)
    ]


def build_sweep_results(
    *,
    base_config: ExperimentConfig,
    R: np.ndarray,
    q: np.ndarray,
    scenario_metadata: dict[str, Any],
    smoke: bool = False,
    workers: int | None = None,
) -> pd.DataFrame:
    """Build the full precomputed sweep library for the Streamlit app."""

    grid = get_streamlit_parameter_grid(smoke=smoke)
    worker_count = 1 if smoke else max(1, workers or min(4, os.cpu_count() or 1))
    task_base_config = apply_overrides(base_config, {"mu_europe_grid": grid["mu_europe"]})

    a_tasks = [
        (
            task_base_config,
            R,
            q,
            scenario_metadata,
            float(portfolio_risk_factor),
            float(turnover_limit),
        )
        for portfolio_risk_factor in grid["portfolio_risk_factor"]
        for turnover_limit in grid["turnover_limit"]
    ]
    bc_tasks = [
        (
            task_base_config,
            R,
            q,
            scenario_metadata,
            float(portfolio_risk_factor),
            float(te_cvar_limit),
            float(turnover_limit),
        )
        for portfolio_risk_factor in grid["portfolio_risk_factor"]
        for te_cvar_limit in grid["te_cvar_limit"]
        for turnover_limit in grid["turnover_limit"]
    ]

    a_frames = _collect_frames(
        worker_fn=_architecture_a_task_payload,
        task_args=a_tasks,
        workers=worker_count,
    )
    bc_frames = _collect_frames(
        worker_fn=_architectures_bc_task_payload,
        task_args=bc_tasks,
        workers=worker_count,
    )

    replicated_a_frames: list[pd.DataFrame] = []
    for a_frame in a_frames:
        replicated_a_frames.extend(
            _replicate_architecture_a_across_te_limits(a_frame, grid["te_cvar_limit"])
        )

    combined = pd.concat([*replicated_a_frames, *bc_frames], ignore_index=True)
    return _sort_results_frame(combined)


def build_portfolio_snapshots(sweep_results: pd.DataFrame) -> pd.DataFrame:
    """Create selected-point portfolio snapshots keyed by all sidebar controls."""

    snapshot_columns = [
        "architecture",
        "architecture_label",
        "status",
        "is_feasible",
        "mu_europe",
        "portfolio_risk_factor",
        "te_cvar_limit",
        "turnover_limit",
        "effective_portfolio_cvar_limit",
        "benchmark_portfolio_cvar_demeaned",
        "expected_return",
        "portfolio_cvar",
        "portfolio_cvar_demeaned",
        "portfolio_cvar_full",
        "te_cvar",
        "te_cvar_demeaned",
        "te_cvar_full",
        "bm_dist_l2",
        "bm_dist_l1",
        "turnover",
        "max_weight",
        "herfindahl",
        "effective_n",
        "w_US",
        "w_Europe",
        "w_EM",
        "w_Japan",
        "aw_US",
        "aw_Europe",
        "aw_EM",
        "aw_Japan",
    ]
    return _sort_results_frame(sweep_results[snapshot_columns])


def build_summary_metrics(sweep_results: pd.DataFrame) -> pd.DataFrame:
    """Create the compact architecture comparison table used by the UI."""

    summary_columns = [
        "architecture",
        "architecture_label",
        "status",
        "is_feasible",
        "mu_europe",
        "portfolio_risk_factor",
        "te_cvar_limit",
        "turnover_limit",
        "expected_return",
        "portfolio_cvar",
        "te_cvar",
        "bm_dist_l2",
        "bm_dist_l1",
        "turnover",
        "herfindahl",
        "effective_n",
        "instability_score",
        "adj_max_jump",
        "objective_loss_pct",
        "max_weight",
    ]
    summary = _sort_results_frame(sweep_results[summary_columns])
    return summary.rename(columns={"adj_max_jump": "max_adjacent_jump"}).reset_index(drop=True)


def export_streamlit_dataset(
    *,
    sweep_results: pd.DataFrame,
    portfolio_snapshots: pd.DataFrame,
    summary_metrics: pd.DataFrame,
    output_dir: str | Path,
    manifest: dict[str, Any],
) -> dict[str, Path]:
    """Persist the Streamlit-ready datasets to parquet and CSV files."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    paths = {
        "sweep_results_parquet": output_path / "sweep_results.parquet",
        "sweep_results_csv": output_path / "sweep_results.csv",
        "portfolio_snapshots_parquet": output_path / "portfolio_snapshots.parquet",
        "portfolio_snapshots_csv": output_path / "portfolio_snapshots.csv",
        "summary_metrics_parquet": output_path / "summary_metrics.parquet",
        "summary_metrics_csv": output_path / "summary_metrics.csv",
        "manifest_json": output_path / "data_manifest.json",
    }

    sweep_results.to_parquet(paths["sweep_results_parquet"], index=False)
    sweep_results.to_csv(paths["sweep_results_csv"], index=False)
    portfolio_snapshots.to_parquet(paths["portfolio_snapshots_parquet"], index=False)
    portfolio_snapshots.to_csv(paths["portfolio_snapshots_csv"], index=False)
    summary_metrics.to_parquet(paths["summary_metrics_parquet"], index=False)
    summary_metrics.to_csv(paths["summary_metrics_csv"], index=False)

    with paths["manifest_json"].open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return paths


def build_streamlit_manifest(
    *,
    base_config: ExperimentConfig,
    R: np.ndarray,
    q: np.ndarray,
    scenario_metadata: dict[str, Any],
    smoke: bool,
    workers: int,
    sweep_results: pd.DataFrame,
) -> dict[str, Any]:
    """Capture a compact build manifest for debugging and deployment checks."""

    grid = get_streamlit_parameter_grid(smoke=smoke)
    feasible_counts = (
        sweep_results.groupby("architecture", sort=False)["is_feasible"].sum().to_dict()
        if not sweep_results.empty
        else {}
    )
    status_counts = (
        sweep_results.groupby(["architecture", "status"], sort=False)
        .size()
        .rename("count")
        .reset_index()
        .to_dict(orient="records")
    )
    asset_volatility_rows = _weighted_asset_volatilities(R=R, q=q, assets=base_config.assets)
    asset_reference = []
    for row in asset_volatility_rows:
        asset = row["asset"]
        asset_reference.append(
            {
                "asset": asset,
                "expected_return_base": (
                    None
                    if base_config.mu_base[asset] is None
                    else float(base_config.mu_base[asset])
                ),
                "volatility": row["volatility"],
            }
        )

    return {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "smoke_mode": smoke,
        "workers": workers,
        "row_count": int(len(sweep_results)),
        "grid": {key: values.tolist() for key, values in grid.items()},
        "feasible_counts_by_architecture": feasible_counts,
        "status_counts": status_counts,
        "asset_reference": asset_reference,
        "base_config": to_serializable_dict(base_config),
        "scenario_metadata": scenario_metadata,
    }


def build_and_export_streamlit_dataset(
    *,
    output_dir: str | Path,
    smoke: bool = False,
    workers: int | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, Path], dict[str, Any]]:
    """Build all Streamlit-ready data tables and write them to disk."""

    output_path = Path(output_dir)
    base_config = build_streamlit_base_config(output_dir=output_path)
    R, q, scenario_metadata = generate_or_load_scenarios(config=base_config)

    worker_count = 1 if smoke else max(1, workers or min(4, os.cpu_count() or 1))
    sweep_results = build_sweep_results(
        base_config=base_config,
        R=R,
        q=q,
        scenario_metadata=scenario_metadata,
        smoke=smoke,
        workers=worker_count,
    )
    portfolio_snapshots = build_portfolio_snapshots(sweep_results)
    summary_metrics = build_summary_metrics(sweep_results)
    manifest = build_streamlit_manifest(
        base_config=base_config,
        R=R,
        q=q,
        scenario_metadata=scenario_metadata,
        smoke=smoke,
        workers=worker_count,
        sweep_results=sweep_results,
    )
    paths = export_streamlit_dataset(
        sweep_results=sweep_results,
        portfolio_snapshots=portfolio_snapshots,
        summary_metrics=summary_metrics,
        output_dir=output_path,
        manifest=manifest,
    )
    return (
        {
            "sweep_results": sweep_results,
            "portfolio_snapshots": portfolio_snapshots,
            "summary_metrics": summary_metrics,
        },
        paths,
        manifest,
    )
