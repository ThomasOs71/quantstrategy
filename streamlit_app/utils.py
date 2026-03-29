from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SUBSTACK_ARTICLE_URL = "https://quantstrategy.substack.com/"
GITHUB_REPO_URL = "https://github.com/ThomasOs71/quantstrategy"
ARCHITECTURE_ORDER = ["A", "B", "C"]
ARCHITECTURE_NAMES = {
    "A": "Standard Optimization",
    "B": "Risk-Constrained Optimization",
    "C": "Lexicographic Risk-Constrained Optimization",
}
ASSET_COLUMNS = ["US", "Europe", "EM", "Japan"]
SWEEP_CHART_ASSETS = ["US", "Europe", "EM"]
STATUS_LABELS = {
    "optimal": "Feasible",
    "optimal_inaccurate": "Feasible",
    "infeasible": "Infeasible",
    "infeasible_inaccurate": "Infeasible",
}


def format_pct(value: float | int | None, decimals: int = 2) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{decimals}%}"


def format_float(value: float | int | None, decimals: int = 3) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{decimals}f}"


def percent_display(value: float, decimals: int = 2) -> str:
    return f"{value:.{decimals}%}"


def format_status(status: str | None) -> str:
    if status is None or pd.isna(status):
        return "Unknown"
    status_text = str(status)
    return STATUS_LABELS.get(status_text, status_text.replace("_", " ").title())


def filter_parameter_slice(
    frame: pd.DataFrame,
    *,
    mu_europe: float | None = None,
    portfolio_risk_factor: float | None = None,
    te_cvar_limit: float | None = None,
    turnover_limit: float | None = None,
) -> pd.DataFrame:
    filtered = frame.copy()
    if mu_europe is not None:
        filtered = filtered[np.isclose(filtered["mu_europe"], mu_europe)]
    if portfolio_risk_factor is not None:
        filtered = filtered[np.isclose(filtered["portfolio_risk_factor"], portfolio_risk_factor)]
    if te_cvar_limit is not None:
        filtered = filtered[np.isclose(filtered["te_cvar_limit"], te_cvar_limit)]
    if turnover_limit is not None:
        filtered = filtered[np.isclose(filtered["turnover_limit"], turnover_limit)]
    return filtered.sort_values("architecture", key=lambda series: series.map({k: i for i, k in enumerate(ARCHITECTURE_ORDER)}))


def get_sidebar_options(sweep_results: pd.DataFrame) -> dict[str, list[float]]:
    return {
        "mu_europe": sorted(float(value) for value in sweep_results["mu_europe"].dropna().unique()),
        "portfolio_risk_factor": sorted(
            float(value) for value in sweep_results["portfolio_risk_factor"].dropna().unique()
        ),
        "te_cvar_limit": sorted(float(value) for value in sweep_results["te_cvar_limit"].dropna().unique()),
        "turnover_limit": sorted(float(value) for value in sweep_results["turnover_limit"].dropna().unique()),
    }


def make_comparison_table(summary_metrics: pd.DataFrame) -> pd.DataFrame:
    table = summary_metrics.copy()
    if table.empty:
        return table

    table = table[
        [
            "architecture",
            "status",
            "expected_return",
            "portfolio_cvar",
            "te_cvar",
            "bm_dist_l2",
            "turnover",
            "herfindahl",
            "effective_n",
            "instability_score",
            "max_adjacent_jump",
        ]
    ].copy()
    table["architecture"] = table["architecture"].map(
        lambda value: ARCHITECTURE_NAMES.get(str(value), str(value))
    )
    table["status"] = table["status"].map(format_status)
    table = table.rename(
        columns={
            "architecture": "Architecture",
            "status": "Status",
            "expected_return": "Exp. Return",
            "portfolio_cvar": "Portf. CVaR",
            "te_cvar": "Demeaned TE-CVaR",
            "bm_dist_l2": "L2 BM Dist.",
            "turnover": "L1 vs BM",
            "herfindahl": "Herfindahl",
            "effective_n": "Eff. N",
            "instability_score": "Instability Score",
            "max_adjacent_jump": "Max Jump",
        }
    )

    for column in [
        "Exp. Return",
        "Portf. CVaR",
        "Demeaned TE-CVaR",
        "L2 BM Dist.",
        "L1 vs BM",
        "Max Jump",
    ]:
        table[column] = table[column].map(format_pct)

    table["Herfindahl"] = table["Herfindahl"].map(lambda value: format_float(value, 3))
    table["Eff. N"] = table["Eff. N"].map(lambda value: format_float(value, 2))
    table["Instability Score"] = table["Instability Score"].map(lambda value: format_pct(value, 2))
    return table.reset_index(drop=True)
