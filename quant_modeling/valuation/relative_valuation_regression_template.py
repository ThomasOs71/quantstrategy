from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RegressionSpec:
    name: str
    target: str
    predictors: list[str]
    lower_is_cheaper: bool = True


def _coerce_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    numeric = frame.copy()
    for column in columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric


def _zscore(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.mean()
    std = values.std(ddof=0)
    if pd.isna(std) or math.isclose(float(std), 0.0):
        return pd.Series(0.0, index=values.index)
    return (values - mean) / std


def fit_relative_valuation_regression(
    frame: pd.DataFrame,
    spec: RegressionSpec,
    id_column: str = "Industry Name",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    needed = [id_column, spec.target, *spec.predictors]
    working = _coerce_numeric(frame.loc[:, [column for column in needed if column in frame.columns]].copy(), needed[1:])

    clean = working.dropna(subset=[spec.target, *spec.predictors]).copy()
    if clean.empty:
        raise ValueError(f"No valid rows available for regression {spec.name}.")

    x = clean.loc[:, spec.predictors].to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(clean)), x])
    y = clean.loc[:, spec.target].to_numpy(dtype=float)

    coefficients = np.linalg.lstsq(x, y, rcond=None)[0]
    predicted = x @ coefficients
    residual = y - predicted

    y_mean = y.mean()
    ss_total = float(np.square(y - y_mean).sum())
    ss_resid = float(np.square(residual).sum())
    r_squared = math.nan if math.isclose(ss_total, 0.0) else 1.0 - ss_resid / ss_total

    out = clean.copy()
    out[f"{spec.name}_predicted"] = predicted
    out[f"{spec.name}_residual"] = residual
    attractive_residual = -residual if spec.lower_is_cheaper else residual
    out[f"{spec.name}_mispricing"] = attractive_residual
    out[f"{spec.name}_mispricing_z"] = _zscore(pd.Series(attractive_residual, index=out.index))

    summary: dict[str, Any] = {
        "model": spec.name,
        "target": spec.target,
        "predictors": list(spec.predictors),
        "observations": int(len(out)),
        "r_squared": float(r_squared) if not math.isnan(r_squared) else math.nan,
        "intercept": float(coefficients[0]),
    }

    for predictor, coefficient in zip(spec.predictors, coefficients[1:]):
        summary[f"coef::{predictor}"] = float(coefficient)

    return out, summary


def run_regression_suite(
    frame: pd.DataFrame,
    specs: list[RegressionSpec],
    id_column: str = "Industry Name",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    merged = frame.copy()
    summaries: list[dict[str, Any]] = []

    for spec in specs:
        result_frame, summary = fit_relative_valuation_regression(merged, spec, id_column=id_column)
        value_columns = [
            id_column,
            f"{spec.name}_predicted",
            f"{spec.name}_residual",
            f"{spec.name}_mispricing",
            f"{spec.name}_mispricing_z",
        ]
        merged = merged.merge(result_frame.loc[:, value_columns], on=id_column, how="left")
        summaries.append(summary)

    return merged, pd.DataFrame(summaries)


if __name__ == "__main__":
    print("Import this module and call run_regression_suite() with your own dataset.")
