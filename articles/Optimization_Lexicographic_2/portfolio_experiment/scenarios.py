from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ExperimentConfig, ScenarioConfig

ASSET_COLUMN_ALIASES = {
    "US": ("us", "usa", "msciusa", "mscius", "unitedstates"),
    "Europe": ("europe", "mscieurope"),
    "EM": ("em", "emergingmarkets", "msciem", "msciemergingmarkets"),
    "Japan": ("japan", "mscijapan"),
}
PROBABILITY_COLUMN_ALIASES = ("q", "probability", "probabilities", "scenario_probability")


def _normalize_label(label: str) -> str:
    return "".join(character.lower() for character in str(label) if character.isalnum())


def _validate_scenario_inputs(
    R: np.ndarray,
    q: np.ndarray | None,
    n_assets: int,
) -> tuple[np.ndarray, np.ndarray]:
    R = np.asarray(R, dtype=float)
    if R.ndim != 2:
        raise ValueError("R must be a 2D matrix with shape (n_scenarios, n_assets).")
    if R.shape[1] != n_assets:
        raise ValueError(f"R must have exactly {n_assets} assets in its second dimension.")

    if q is None:
        q_array = np.full(R.shape[0], 1.0 / R.shape[0], dtype=float)
    else:
        q_array = np.asarray(q, dtype=float).reshape(-1)
        if q_array.shape[0] != R.shape[0]:
            raise ValueError("q must have one probability per scenario.")
        if np.any(q_array < 0):
            raise ValueError("Scenario probabilities must be non-negative.")
        if np.isclose(q_array.sum(), 0.0):
            raise ValueError("Scenario probabilities must sum to a positive value.")
        q_array = q_array / q_array.sum()

    return R, q_array


def generate_synthetic_scenarios(config: ScenarioConfig) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Create a synthetic scenario matrix with equity-like covariance structure."""

    covariance = config.covariance
    min_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
    if min_eigenvalue < -1e-10:
        raise ValueError("Synthetic covariance matrix is not positive semidefinite.")

    rng = np.random.default_rng(config.seed)
    if config.distribution != "multivariate_normal":
        raise ValueError(f"Unsupported synthetic distribution: {config.distribution}")

    R = rng.multivariate_normal(
        mean=config.synthetic_means,
        cov=covariance,
        size=config.n_scenarios,
        check_valid="raise",
    )
    q = np.full(config.n_scenarios, 1.0 / config.n_scenarios, dtype=float)
    metadata = {
        "source": "synthetic_multivariate_normal",
        "demeaned": False,
        "distribution": config.distribution,
        "n_scenarios": int(config.n_scenarios),
        "seed": int(config.seed),
        "synthetic_means": config.synthetic_means.tolist(),
        "volatilities": config.volatilities.tolist(),
        "correlation": config.correlation.tolist(),
    }
    return R, q, metadata


def _find_sheet_name(excel_path: Path, configured_sheet: str | None) -> str:
    workbook = pd.ExcelFile(excel_path)
    if configured_sheet is not None:
        if configured_sheet not in workbook.sheet_names:
            raise ValueError(
                f"Configured sheet '{configured_sheet}' was not found in workbook '{excel_path.name}'."
            )
        return configured_sheet
    if not workbook.sheet_names:
        raise ValueError(f"Workbook '{excel_path.name}' does not contain any sheets.")
    return workbook.sheet_names[0]


def _resolve_asset_columns(columns: list[str], assets: tuple[str, ...]) -> dict[str, str]:
    normalized_lookup = {_normalize_label(column): column for column in columns}
    resolved_columns: dict[str, str] = {}

    for asset in assets:
        aliases = ASSET_COLUMN_ALIASES.get(asset, (asset,))
        matched_column = None
        for alias in aliases:
            matched_column = normalized_lookup.get(alias)
            if matched_column is not None:
                break
        if matched_column is None:
            raise ValueError(
                f"Could not map asset '{asset}' to an Excel column. Available columns: {columns}"
            )
        resolved_columns[asset] = matched_column

    return resolved_columns


def _resolve_probability_column(columns: list[str], configured_column: str | None) -> str | None:
    normalized_lookup = {_normalize_label(column): column for column in columns}
    if configured_column is not None:
        resolved = normalized_lookup.get(_normalize_label(configured_column))
        if resolved is None:
            raise ValueError(
                f"Configured probability column '{configured_column}' was not found. Available columns: {columns}"
            )
        return resolved

    for alias in PROBABILITY_COLUMN_ALIASES:
        resolved = normalized_lookup.get(alias)
        if resolved is not None:
            return resolved
    return None


def load_scenarios_from_excel(
    config: ExperimentConfig,
    excel_path: Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load scenario returns from an Excel workbook and align them to the asset universe."""

    sheet_name = _find_sheet_name(excel_path, config.scenario.excel_sheet_name)
    scenario_df = pd.read_excel(excel_path, sheet_name=sheet_name)

    # Exported Excel files often include an unnamed index column; it is metadata, not a scenario return.
    scenario_df = scenario_df.loc[:, ~scenario_df.columns.astype(str).str.startswith("Unnamed:")].copy()
    asset_columns = _resolve_asset_columns(scenario_df.columns.tolist(), config.assets)
    probability_column = _resolve_probability_column(
        scenario_df.columns.tolist(),
        config.scenario.probability_column,
    )

    ordered_returns = scenario_df[[asset_columns[asset] for asset in config.assets]].astype(float)
    R = ordered_returns.to_numpy(dtype=float)

    if probability_column is None:
        q = np.full(R.shape[0], 1.0 / R.shape[0], dtype=float)
    else:
        q = scenario_df[probability_column].to_numpy(dtype=float)
        if np.any(q < 0):
            raise ValueError("Scenario probabilities loaded from Excel must be non-negative.")
        if np.isclose(q.sum(), 0.0):
            raise ValueError("Scenario probabilities loaded from Excel must sum to a positive value.")
        q = q / q.sum()

    metadata = {
        "source": "excel_workbook",
        "demeaned": bool(config.scenario.excel_scenarios_are_demeaned),
        "excel_path": str(excel_path),
        "sheet_name": sheet_name,
        "n_scenarios": int(R.shape[0]),
        "asset_columns": asset_columns,
        "probability_column": probability_column,
        "provided_probabilities": probability_column is not None,
    }
    return R, q, metadata


def generate_or_load_scenarios(
    config: ExperimentConfig,
    R: np.ndarray | None = None,
    q: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Return validated scenario inputs.

    Assumption:
    When no explicit scenario matrix is supplied, the experiment first looks for a
    configured Excel workbook and uses equal scenario weights unless a probability
    column is present. Synthetic scenarios are only used as a fallback.
    """

    if R is None:
        excel_path = config.scenario.excel_path
        if excel_path is not None and Path(excel_path).exists():
            return load_scenarios_from_excel(config=config, excel_path=Path(excel_path))
        return generate_synthetic_scenarios(config.scenario)

    validated_R, validated_q = _validate_scenario_inputs(R=R, q=q, n_assets=len(config.assets))
    metadata = {
        "source": "external_matrix",
        "demeaned": False,
        "n_scenarios": int(validated_R.shape[0]),
        "provided_probabilities": q is not None,
    }
    return validated_R, validated_q, metadata
