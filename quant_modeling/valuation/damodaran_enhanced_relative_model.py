from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from relative_valuation_regression_template import RegressionSpec, run_regression_suite


OUTPUT_DIR = Path(__file__).resolve().parent / "research_outputs"
DATASET_BASE_URL = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/"

REGION_DATASETS: dict[str, dict[str, str]] = {
    "us": {
        "pe": "pedata.xls",
        "pbv": "pbvdata.xls",
        "ps": "psdata.xls",
        "vebitda": "vebitda.xls",
        "fundgr": "fundgr.xls",
        "beta": "betas.xls",
        "capex": "capex.xls",
        "divfund": "divfund.xls",
        "wacc": "wacc.xls",
    },
    "europe": {
        "pe": "peEurope.xls",
        "pbv": "pbvEurope.xls",
        "ps": "psEurope.xls",
        "vebitda": "vebitdaEurope.xls",
        "fundgr": "fundgrEurope.xls",
        "beta": "betaEurope.xls",
        "capex": "capexEurope.xls",
        "divfund": "divfundEurope.xls",
        "wacc": "waccEurope.xls",
    },
}

FIELD_SOURCES: dict[str, list[tuple[str, str]]] = {
    "number_of_firms": [("pe", "Number of firms"), ("fundgr", "Number of Firms")],
    "forward_pe": [("pe", "Forward PE")],
    "expected_growth": [("pe", "Expected growth - next 5 years")],
    "peg_ratio": [("pe", "PEG Ratio")],
    "pbv": [("pbv", "PBV")],
    "roe": [("pbv", "ROE"), ("fundgr", "ROE"), ("divfund", "ROE")],
    "roic": [("pbv", "ROIC")],
    "price_sales": [("ps", "Price/Sales")],
    "net_margin": [("ps", "Net Margin")],
    "ev_sales": [("ps", "EV/Sales")],
    "pre_tax_operating_margin": [("ps", "Pre-tax Operating Margin")],
    "ev_ebitda": [("vebitda", "EV/EBITDA")],
    "beta": [("beta", "Beta")],
    "de_ratio": [("beta", "D/E Ratio")],
    "effective_tax_rate": [("beta", "Effective Tax rate"), ("wacc", "Tax Rate")],
    "std_dev_equity": [("beta", "Standard deviation of equity"), ("divfund", "Std Dev in Stock Prices")],
    "retention_ratio": [("fundgr", "Retention Ratio")],
    "fundamental_growth": [("fundgr", "Fundamental Growth")],
    "net_capex_sales": [("capex", "Net Cap Ex/Sales")],
    "sales_invested_capital": [("capex", "Sales/ Invested Capital (LTM)")],
    "dividend_payout": [("divfund", "Dividend Payout")],
    "dividend_yield": [("divfund", "Dividend Yield")],
    "cost_of_equity": [("wacc", "Cost of Equity")],
    "after_tax_cost_of_debt": [("wacc", "After-tax Cost of Debt")],
    "cost_of_capital": [("wacc", "Cost of Capital")],
}

VARIABLE_TABLE_ROWS: list[dict[str, Any]] = [
    {
        "multiple": "Forward PE",
        "cheap_if": "lower",
        "add_variables": "Expected growth, Dividend Payout, Beta",
        "primary_driver": "Growth",
        "screen_formula": "-z(Forward PE) + 0.8*z(Expected growth) + 0.4*z(Dividend Payout) - 0.6*z(Beta)",
        "why_it_matters": "Low PE is only attractive when growth is intact and risk is not materially higher.",
    },
    {
        "multiple": "PBV",
        "cheap_if": "lower",
        "add_variables": "ROE, ROIC, Cost of Equity",
        "primary_driver": "ROE",
        "screen_formula": "-z(PBV) + 1.0*z(ROE) + 0.4*z(ROIC) - 0.4*z(Cost of Equity)",
        "why_it_matters": "Low PBV is often justified by weak profitability; ROE separates value from value traps.",
    },
    {
        "multiple": "Price/Sales",
        "cheap_if": "lower",
        "add_variables": "Net Margin, Fundamental Growth, Beta",
        "primary_driver": "Margin",
        "screen_formula": "-z(Price/Sales) + 1.0*z(Net Margin) + 0.7*z(Fundamental Growth) - 0.5*z(Beta)",
        "why_it_matters": "Sales-based multiples only work when margins and growth quality can sustain valuation.",
    },
    {
        "multiple": "EV/EBITDA",
        "cheap_if": "lower",
        "add_variables": "ROIC, Pre-tax Operating Margin, Net Cap Ex/Sales, Cost of Capital",
        "primary_driver": "Capital efficiency",
        "screen_formula": "-z(EV/EBITDA) + 0.8*z(ROIC) + 0.5*z(Pre-tax Operating Margin) - 0.6*z(Net Cap Ex/Sales) - 0.4*z(Cost of Capital)",
        "why_it_matters": "A low EV/EBITDA can be a capex trap; reinvestment needs and returns on capital matter.",
    },
]

REGRESSION_SPECS: list[RegressionSpec] = [
    RegressionSpec(
        name="pe_model",
        target="forward_pe",
        predictors=["expected_growth", "dividend_payout", "beta"],
    ),
    RegressionSpec(
        name="pbv_model",
        target="pbv",
        predictors=["roe", "roic", "cost_of_equity"],
    ),
    RegressionSpec(
        name="ps_model",
        target="price_sales",
        predictors=["net_margin", "fundamental_growth", "beta"],
    ),
    RegressionSpec(
        name="ev_ebitda_model",
        target="ev_ebitda",
        predictors=["roic", "pre_tax_operating_margin", "net_capex_sales", "cost_of_capital"],
    ),
]


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _dedupe_headers(values: list[Any]) -> list[str]:
    seen: dict[str, int] = {}
    headers: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text.lower() == "nan":
            headers.append("")
            continue
        count = seen.get(text, 0) + 1
        seen[text] = count
        headers.append(text if count == 1 else f"{text}__{count}")
    return headers


def _find_industry_sheet(sheet_names: list[str]) -> str:
    for sheet_name in sheet_names:
        normalized = sheet_name.lower()
        if "industry averages" in normalized or "indusry averages" in normalized:
            return sheet_name
    raise ValueError("Could not find an industry averages sheet.")


def _load_industry_average_table(file_name: str) -> pd.DataFrame:
    url = f"{DATASET_BASE_URL}{file_name}"
    workbook = pd.ExcelFile(url)
    sheet_name = _find_industry_sheet(workbook.sheet_names)
    raw = pd.read_excel(workbook, sheet_name=sheet_name, header=None)

    header_idx: int | None = None
    for idx in range(len(raw)):
        row = raw.iloc[idx].astype(str).str.strip()
        if any(value == "Industry Name" for value in row):
            header_idx = idx
            break
    if header_idx is None:
        raise ValueError(f"Could not locate header row in {file_name}.")

    headers = _dedupe_headers(raw.iloc[header_idx].tolist())
    table = raw.iloc[header_idx + 1 :].copy()
    table.columns = headers
    table = table.loc[:, [column for column in table.columns if column]]
    table = table.dropna(how="all")
    table["Industry Name"] = table["Industry Name"].astype(str).str.strip()
    table = table.loc[table["Industry Name"].ne("")]
    table = table.loc[~table["Industry Name"].str.lower().eq("nan")]
    table = table.loc[~table["Industry Name"].str.lower().str.startswith("total market")]
    table = table.drop_duplicates(subset=["Industry Name"], keep="first")
    return table.reset_index(drop=True)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _zscore(series: pd.Series) -> pd.Series:
    values = _coerce_numeric(series)
    mean = values.mean()
    std = values.std(ddof=0)
    if pd.isna(std) or math.isclose(float(std), 0.0):
        return pd.Series(0.0, index=values.index)
    return (values - mean) / std


def _row_mean(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    return frame.loc[:, columns].mean(axis=1, skipna=True)


def _classify_by_quantile(series: pd.Series) -> tuple[pd.Series, float, float]:
    numeric = _coerce_numeric(series)
    q25 = float(numeric.quantile(0.25))
    q75 = float(numeric.quantile(0.75))
    labels = np.where(numeric >= q75, "cheap", np.where(numeric <= q25, "expensive", "neutral"))
    return pd.Series(labels, index=series.index), q25, q75


def _action_from_regime(regime: pd.Series) -> pd.Series:
    mapping = {
        "cheap": "overweight",
        "neutral": "market_weight",
        "expensive": "underweight",
    }
    return regime.map(mapping).fillna("market_weight")


def load_region_frame(region: str) -> pd.DataFrame:
    region_files = REGION_DATASETS[region]
    tables: dict[str, pd.DataFrame] = {
        key: _load_industry_average_table(file_name) for key, file_name in region_files.items()
    }
    indexed = {
        key: table.assign(**{"Industry Name": table["Industry Name"].astype(str).str.strip()})
        .drop_duplicates(subset=["Industry Name"], keep="first")
        .set_index("Industry Name")
        for key, table in tables.items()
    }

    industries = sorted(
        {
            industry_name
            for table in indexed.values()
            for industry_name in table.index.tolist()
            if industry_name
        }
    )

    frame = pd.DataFrame({"Industry Name": industries})
    frame = frame.set_index("Industry Name")

    for field_name, sources in FIELD_SOURCES.items():
        values = pd.Series(np.nan, index=frame.index, dtype=float)
        for dataset_key, column_name in sources:
            table = indexed.get(dataset_key)
            if table is None or column_name not in table.columns:
                continue
            candidate = _coerce_numeric(table.loc[:, column_name]).reindex(frame.index)
            fill_mask = values.isna() & candidate.notna()
            values.loc[fill_mask] = candidate.loc[fill_mask]
        frame[field_name] = values

    frame = frame.reset_index()
    frame = frame.sort_values(["Industry Name"]).reset_index(drop=True)
    frame = frame.loc[frame["number_of_firms"].fillna(0) >= 5].reset_index(drop=True)
    return frame


def compute_enhanced_scores(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()

    scored["raw_pe_score"] = -_zscore(scored["forward_pe"])
    scored["raw_pbv_score"] = -_zscore(scored["pbv"])
    scored["raw_ps_score"] = -_zscore(scored["price_sales"])
    scored["raw_ev_ebitda_score"] = -_zscore(scored["ev_ebitda"])

    scored["adj_pe_score"] = (
        -_zscore(scored["forward_pe"])
        + 0.8 * _zscore(scored["expected_growth"])
        + 0.4 * _zscore(scored["dividend_payout"])
        - 0.6 * _zscore(scored["beta"])
    )
    scored["adj_pbv_score"] = (
        -_zscore(scored["pbv"])
        + 1.0 * _zscore(scored["roe"])
        + 0.4 * _zscore(scored["roic"])
        - 0.4 * _zscore(scored["cost_of_equity"])
    )
    scored["adj_ps_score"] = (
        -_zscore(scored["price_sales"])
        + 1.0 * _zscore(scored["net_margin"])
        + 0.7 * _zscore(scored["fundamental_growth"])
        - 0.5 * _zscore(scored["beta"])
    )
    scored["adj_ev_ebitda_score"] = (
        -_zscore(scored["ev_ebitda"])
        + 0.8 * _zscore(scored["roic"])
        + 0.5 * _zscore(scored["pre_tax_operating_margin"])
        - 0.6 * _zscore(scored["net_capex_sales"])
        - 0.4 * _zscore(scored["cost_of_capital"])
    )

    scored["raw_composite_score"] = _row_mean(
        scored,
        ["raw_pe_score", "raw_pbv_score", "raw_ps_score", "raw_ev_ebitda_score"],
    )
    scored["adjusted_composite_score"] = _row_mean(
        scored,
        ["adj_pe_score", "adj_pbv_score", "adj_ps_score", "adj_ev_ebitda_score"],
    )
    scored["quality_overlay_delta"] = scored["adjusted_composite_score"] - scored["raw_composite_score"]
    return scored


def attach_regression_outputs(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    modeled, summary = run_regression_suite(frame, REGRESSION_SPECS, id_column="Industry Name")
    modeled["regression_composite_score"] = _row_mean(
        modeled,
        [
            "pe_model_mispricing_z",
            "pbv_model_mispricing_z",
            "ps_model_mispricing_z",
            "ev_ebitda_model_mispricing_z",
        ],
    )
    modeled["final_value_score"] = 0.6 * modeled["adjusted_composite_score"] + 0.4 * modeled["regression_composite_score"]
    regime, q25, q75 = _classify_by_quantile(modeled["final_value_score"])
    modeled["valuation_regime"] = regime
    modeled["action"] = _action_from_regime(modeled["valuation_regime"])
    modeled["score_q25"] = q25
    modeled["score_q75"] = q75
    return modeled, summary


def build_variable_table() -> pd.DataFrame:
    return pd.DataFrame(VARIABLE_TABLE_ROWS)


def summarize_region(region: str, frame: pd.DataFrame, regression_summary: pd.DataFrame) -> dict[str, Any]:
    ranked = frame.sort_values("final_value_score", ascending=False).reset_index(drop=True)
    top = ranked.loc[:, ["Industry Name", "final_value_score", "valuation_regime", "action"]].head(5)
    bottom = (
        frame.sort_values("final_value_score", ascending=True)
        .reset_index(drop=True)
        .loc[:, ["Industry Name", "final_value_score", "valuation_regime", "action"]]
        .head(5)
    )
    actions = frame["action"].value_counts(dropna=False).to_dict()

    return {
        "region": region,
        "industry_count": int(len(frame)),
        "top_cheap_industries": top.to_dict(orient="records"),
        "top_expensive_industries": bottom.to_dict(orient="records"),
        "action_counts": {str(key): int(value) for key, value in actions.items()},
        "average_final_score": float(frame["final_value_score"].mean()),
        "average_quality_overlay_delta": float(frame["quality_overlay_delta"].mean()),
        "mean_r_squared": float(regression_summary["r_squared"].mean()),
    }


def write_markdown_summary(
    variable_table: pd.DataFrame,
    region_payloads: dict[str, dict[str, Any]],
) -> Path:
    lines = [
        "# Enhanced Relative Valuation Summary",
        "",
        "Data pulled online on 2026-02-28 from Damodaran's public datasets.",
        "",
        "## Variable Overlay Rules",
        "",
        "| Multiple | Add Variables | Primary Driver | Screen Formula |",
        "| --- | --- | --- | --- |",
    ]
    for row in variable_table.itertuples(index=False):
        lines.append(
            f"| {row.multiple} | {row.add_variables} | {row.primary_driver} | `{row.screen_formula}` |"
        )

    lines.extend(
        [
            "",
            "## Region Snapshots",
            "",
        ]
    )

    for region, payload in region_payloads.items():
        title = "US" if region == "us" else "Europe"
        lines.extend(
            [
                f"### {title}",
                "",
                f"- Coverage: {payload['industry_count']} industries with at least 5 firms.",
                f"- Mean regression R^2 across the four models: {payload['mean_r_squared']:.2f}",
                f"- Average quality overlay delta: {payload['average_quality_overlay_delta']:.2f}",
                f"- Actions: {json.dumps(payload['action_counts'], sort_keys=True)}",
                "",
                "Top cheap industries:",
            ]
        )
        for item in payload["top_cheap_industries"]:
            lines.append(
                f"- {item['Industry Name']}: score={item['final_value_score']:.2f}, regime={item['valuation_regime']}, action={item['action']}"
            )
        lines.append("")
        lines.append("Top expensive industries:")
        for item in payload["top_expensive_industries"]:
            lines.append(
                f"- {item['Industry Name']}: score={item['final_value_score']:.2f}, regime={item['valuation_regime']}, action={item['action']}"
            )
        lines.append("")

    lines.extend(
        [
            "## Online Sources",
            "",
            "- https://pages.stern.nyu.edu/~adamodar/pc/datasets/pedata.xls",
            "- https://pages.stern.nyu.edu/~adamodar/pc/datasets/pbvdata.xls",
            "- https://pages.stern.nyu.edu/~adamodar/pc/datasets/psdata.xls",
            "- https://pages.stern.nyu.edu/~adamodar/pc/datasets/vebitda.xls",
            "- https://pages.stern.nyu.edu/~adamodar/pc/datasets/fundgr.xls",
            "- https://pages.stern.nyu.edu/~adamodar/pc/datasets/betas.xls",
            "- https://pages.stern.nyu.edu/~adamodar/pc/datasets/capex.xls",
            "- https://pages.stern.nyu.edu/~adamodar/pc/datasets/divfund.xls",
            "- https://pages.stern.nyu.edu/~adamodar/pc/datasets/wacc.xls",
            "- https://pages.stern.nyu.edu/~adamodar/pc/datasets/peEurope.xls",
            "- https://pages.stern.nyu.edu/~adamodar/pc/datasets/pbvEurope.xls",
            "- https://pages.stern.nyu.edu/~adamodar/pc/datasets/psEurope.xls",
            "- https://pages.stern.nyu.edu/~adamodar/pc/datasets/vebitdaEurope.xls",
            "- https://pages.stern.nyu.edu/~adamodar/pc/datasets/fundgrEurope.xls",
            "- https://pages.stern.nyu.edu/~adamodar/pc/datasets/betaEurope.xls",
            "- https://pages.stern.nyu.edu/~adamodar/pc/datasets/capexEurope.xls",
            "- https://pages.stern.nyu.edu/~adamodar/pc/datasets/divfundEurope.xls",
            "- https://pages.stern.nyu.edu/~adamodar/pc/datasets/waccEurope.xls",
            "",
            "These overlays should usually improve relative valuation screens by reducing value traps, but they remain cross-sectional tools rather than market-timing signals.",
            "",
        ]
    )

    output_path = ensure_output_dir() / "damodaran_enhanced_summary.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def build_enhanced_model() -> dict[str, Any]:
    ensure_output_dir()
    variable_table = build_variable_table()
    variable_table_path = ensure_output_dir() / "damodaran_enhanced_variable_table.csv"
    variable_table.to_csv(variable_table_path, index=False)

    region_outputs: dict[str, dict[str, Any]] = {}
    artifact_paths: dict[str, str] = {
        "variable_table": str(variable_table_path),
    }

    for region in ("us", "europe"):
        base_frame = load_region_frame(region)
        scored_frame = compute_enhanced_scores(base_frame)
        modeled_frame, regression_summary = attach_regression_outputs(scored_frame)
        modeled_frame = modeled_frame.sort_values("final_value_score", ascending=False).reset_index(drop=True)

        sector_path = ensure_output_dir() / f"damodaran_{region}_enhanced_sector_scores.csv"
        regression_path = ensure_output_dir() / f"damodaran_{region}_regression_summary.csv"
        modeled_frame.to_csv(sector_path, index=False)
        regression_summary.to_csv(regression_path, index=False)

        artifact_paths[f"{region}_sector_scores"] = str(sector_path)
        artifact_paths[f"{region}_regression_summary"] = str(regression_path)
        region_outputs[region] = summarize_region(region, modeled_frame, regression_summary)

    summary_path = write_markdown_summary(variable_table, region_outputs)
    artifact_paths["summary_md"] = str(summary_path)

    payload = {
        "as_of": "2026-02-28",
        "artifacts": artifact_paths,
        "regions": region_outputs,
        "sources": {
            region: {
                key: f"{DATASET_BASE_URL}{file_name}"
                for key, file_name in REGION_DATASETS[region].items()
            }
            for region in REGION_DATASETS
        },
    }

    payload_path = ensure_output_dir() / "damodaran_enhanced_payload.json"
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    artifact_paths["payload_json"] = str(payload_path)

    return payload


def main() -> None:
    payload = build_enhanced_model()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
