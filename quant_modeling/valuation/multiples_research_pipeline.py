from __future__ import annotations

import json
import math
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rebuild_multiples_valuation import (
    PAPER_END_QUARTER,
    build_quarterly_frame,
    build_signal_frame,
    discover_data_url,
    load_monthly_data,
    safe_corr,
    summarize_forecasts,
)


OUTPUT_DIR = Path(__file__).with_name("research_outputs")
COUNTRYSTATS_URL = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/countrystats.xls"
COUNTRY_ERP_URL = "https://pages.stern.nyu.edu/~adamodar/pc/datasets/ctryprem.xlsx"
MIN_COUNTRY_COUNT = 20
MIN_COUNTRY_MARKET_CAP = 20_000.0
SPLIT_METHODS = ("quartile", "decile", "zscore")
SIGNAL_LABELS = {"dy": "DY", "ey": "EY", "cy": "CY"}
SAMPLE_LABELS = {"all": "All", "high_low": "H+L", "middle": "Middle"}


@dataclass(frozen=True)
class ExtremeSplit:
    labels: pd.Series
    high_cutoff: float
    low_cutoff: float


COUNTRY_ALIAS_MAP = {
    "south korea": "korea",
    "trinidad and tobago": "trinidad and tobago",
    "trinidad tobago": "trinidad and tobago",
    "guernsey": "guernsey states of",
    "jersey": "jersey states of",
    "macau": "macao",
    "turks and caicos islands": "turks and caicos islands",
    "british virgin islands": "british virgin islands",
    "curacao": "curacao",
}


def normalize_country_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", str(name))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = text.replace("&", " and ")
    cleaned = "".join(ch if ch.isalnum() or ch.isspace() else " " for ch in text)
    cleaned = " ".join(cleaned.split())
    return COUNTRY_ALIAS_MAP.get(cleaned, cleaned)


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def build_extreme_split(signal: pd.Series, method: str) -> ExtremeSplit:
    valid = signal.dropna()
    labels = pd.Series(index=signal.index, dtype="object")

    if valid.empty:
        return ExtremeSplit(labels=labels, high_cutoff=math.nan, low_cutoff=math.nan)

    if method in {"quartile", "decile"}:
        parts = 4 if method == "quartile" else 10
        ordered_index = valid.sort_values(kind="mergesort").index.to_numpy()
        buckets = np.array_split(ordered_index, parts)
        low_indices = buckets[0]
        high_indices = buckets[-1]
        middle_groups = buckets[1:-1]
        middle_indices = np.concatenate(middle_groups) if middle_groups else np.array([], dtype=int)

        labels.loc[low_indices] = "low"
        labels.loc[high_indices] = "high"
        if len(middle_indices) > 0:
            labels.loc[middle_indices] = "middle"

        low_cutoff = float(valid.loc[low_indices].max())
        high_cutoff = float(valid.loc[high_indices].min())
        return ExtremeSplit(labels=labels, high_cutoff=high_cutoff, low_cutoff=low_cutoff)

    if method == "zscore":
        mean = float(valid.mean())
        std = float(valid.std(ddof=0))
        low_cutoff = mean - std
        high_cutoff = mean + std

        low_mask = valid <= low_cutoff
        high_mask = valid >= high_cutoff
        middle_mask = ~(low_mask | high_mask)

        labels.loc[valid.index[low_mask]] = "low"
        labels.loc[valid.index[high_mask]] = "high"
        labels.loc[valid.index[middle_mask]] = "middle"

        return ExtremeSplit(labels=labels, high_cutoff=high_cutoff, low_cutoff=low_cutoff)

    raise ValueError(f"Unsupported split method: {method}")


def summarize_correlations(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "nominal": safe_corr(frame["signal"], frame["fwd_nominal"]),
        "real": safe_corr(frame["signal"], frame["fwd_real"]),
    }


def make_window(quarterly: pd.DataFrame, end_quarter: pd.Period) -> pd.DataFrame:
    window = quarterly[quarterly["quarter"] <= end_quarter].copy().reset_index(drop=True)
    window["fwd_nominal"] = (window["nominal_tr_index"].shift(-40) / window["nominal_tr_index"]) ** (1.0 / 10.0) - 1.0
    window["fwd_real"] = (window["real_tr_index"].shift(-40) / window["real_tr_index"]) ** (1.0 / 10.0) - 1.0
    return window


def analyze_signal_split(window: pd.DataFrame, signal_column: str, method: str) -> dict[str, Any]:
    frame = build_signal_frame(window, signal_column)
    split = build_extreme_split(frame["signal"], method)

    all_frame = frame
    hl_frame = frame[split.labels.isin(["high", "low"])].copy()
    middle_frame = frame[split.labels.eq("middle")].copy()

    current_signal = float(frame["signal"].iloc[-1])
    current_regime = str(split.labels.iloc[-1])

    forecasts = {
        "all": summarize_forecasts(all_frame),
        "high_low": summarize_forecasts(hl_frame),
        "middle": summarize_forecasts(middle_frame),
    }

    regime_key = "high_low" if current_regime in {"high", "low"} else "middle"
    current_regime_forecast = forecasts[regime_key]
    current_all_forecast = forecasts["all"]

    current_all_value = 100.0 * (
        current_all_forecast["alpha_raw"] + current_all_forecast["beta_raw"] * current_signal
    )
    current_regime_value = 100.0 * (
        current_regime_forecast["alpha_raw"] + current_regime_forecast["beta_raw"] * current_signal
    )

    return {
        "cutoffs": {
            "high_ge": split.high_cutoff,
            "low_le": split.low_cutoff,
        },
        "sample_sizes": {
            "all": int(len(all_frame)),
            "high_low": int(len(hl_frame)),
            "middle": int(len(middle_frame)),
        },
        "correlations": {
            "all": summarize_correlations(all_frame),
            "high_low": summarize_correlations(hl_frame),
            "middle": summarize_correlations(middle_frame),
        },
        "forecasts": forecasts,
        "current": {
            "value": current_signal,
            "regime": current_regime,
            "forecast_all_real_annual_pct": current_all_value,
            "forecast_regime_real_annual_pct": current_regime_value,
        },
    }


def build_us_payload() -> dict[str, Any]:
    data_url = discover_data_url()
    monthly = load_monthly_data(data_url)
    quarterly = build_quarterly_frame(monthly)
    latest_complete_quarter = quarterly["quarter"].max()

    windows = {
        "paper_aligned": make_window(quarterly, PAPER_END_QUARTER),
        "latest_online": make_window(quarterly, latest_complete_quarter),
    }

    analyses: dict[str, Any] = {}
    summary_rows: list[dict[str, Any]] = []
    live_rows: list[dict[str, Any]] = []

    for window_name, window in windows.items():
        analyses[window_name] = {
            "end_quarter": str(window["quarter"].iloc[-1]),
            "latest_month_in_window": str(window["period"].iloc[-1]),
            "methods": {},
        }

        for method in SPLIT_METHODS:
            method_results: dict[str, Any] = {}
            for signal_column, signal_label in SIGNAL_LABELS.items():
                result = analyze_signal_split(window, signal_column, method)
                method_results[signal_column] = result

                if window_name == "latest_online":
                    live_rows.append(
                        {
                            "quarter": str(window["quarter"].iloc[-1]),
                            "method": method,
                            "signal": signal_label,
                            "signal_value_pct": result["current"]["value"] * 100.0,
                            "regime": result["current"]["regime"],
                            "forecast_all_real_annual_pct": result["current"]["forecast_all_real_annual_pct"],
                            "forecast_regime_real_annual_pct": result["current"]["forecast_regime_real_annual_pct"],
                            "high_cutoff_pct": result["cutoffs"]["high_ge"] * 100.0,
                            "low_cutoff_pct": result["cutoffs"]["low_le"] * 100.0,
                        }
                    )

                for sample_key in ("all", "high_low", "middle"):
                    summary_rows.append(
                        {
                            "window": window_name,
                            "method": method,
                            "signal": signal_label,
                            "sample": SAMPLE_LABELS[sample_key],
                            "n_obs": result["sample_sizes"][sample_key],
                            "corr_nominal": result["correlations"][sample_key]["nominal"],
                            "corr_real": result["correlations"][sample_key]["real"],
                            "forecast_rho": result["forecasts"][sample_key]["rho"],
                            "forecast_mad_paper_pct": result["forecasts"][sample_key]["mad_pct_paper"],
                            "forecast_beta_paper": result["forecasts"][sample_key]["beta_paper"],
                            "forecast_count": result["forecasts"][sample_key]["forecast_count"],
                            "high_cutoff_pct": result["cutoffs"]["high_ge"] * 100.0,
                            "low_cutoff_pct": result["cutoffs"]["low_le"] * 100.0,
                        }
                    )

            analyses[window_name]["methods"][method] = method_results

    us_summary = pd.DataFrame(summary_rows)
    us_live = pd.DataFrame(live_rows)

    return {
        "data_url": data_url,
        "latest_month_online": str(monthly["period"].iloc[-1]),
        "latest_complete_quarter": str(latest_complete_quarter),
        "analyses": analyses,
        "summary_df": us_summary,
        "live_df": us_live,
    }


def load_damodaran_country_signal() -> dict[str, Any]:
    stats_meta = pd.read_excel(COUNTRYSTATS_URL, header=None, nrows=2)
    stats_update = pd.to_datetime(stats_meta.iat[0, 1], errors="coerce")

    erp_meta = pd.read_excel(COUNTRY_ERP_URL, sheet_name="ERPs by country", header=None, nrows=4)
    erp_update = pd.to_datetime(erp_meta.iat[1, 1], errors="coerce")
    mature_market_erp = float(erp_meta.iat[2, 4])
    us_equity_erp = float(erp_meta.iat[3, 4])

    stats = pd.read_excel(COUNTRYSTATS_URL, header=8)
    stats = stats.rename(columns={stats.columns[0]: "country"})
    stats = stats[stats["country"].notna()].copy()
    stats["normalized_country"] = stats["country"].map(normalize_country_name)
    stats["count"] = pd.to_numeric(stats["count"], errors="coerce")
    stats["sum(Market Cap (in US $))"] = pd.to_numeric(stats["sum(Market Cap (in US $))"], errors="coerce")
    stats["median(Forward PE)"] = pd.to_numeric(stats["median(Forward PE)"], errors="coerce")
    stats["Dividend Yield"] = pd.to_numeric(stats["Dividend Yield"], errors="coerce")

    erp = pd.read_excel(COUNTRY_ERP_URL, sheet_name="ERPs by country", header=7)
    erp = erp[erp["Country"].notna()].copy()
    erp = erp.rename(
        columns={
            "Country": "country",
            "Total Equity Risk Premium": "total_equity_risk_premium",
            "Country Risk Premium": "country_risk_premium",
        }
    )
    erp["normalized_country"] = erp["country"].map(normalize_country_name)
    erp["total_equity_risk_premium"] = pd.to_numeric(erp["total_equity_risk_premium"], errors="coerce")
    erp["country_risk_premium"] = pd.to_numeric(erp["country_risk_premium"], errors="coerce")

    merged = stats.merge(
        erp[["normalized_country", "country", "total_equity_risk_premium", "country_risk_premium"]],
        on="normalized_country",
        how="left",
        suffixes=("_stats", "_erp"),
    )

    merged["country_name"] = merged["country_stats"]
    merged["forward_pe"] = merged["median(Forward PE)"]
    merged["dividend_yield"] = merged["Dividend Yield"].fillna(0.0)
    merged["forward_earnings_yield"] = 1.0 / merged["forward_pe"]
    merged["valuation_spread"] = merged["forward_earnings_yield"] - merged["total_equity_risk_premium"]

    rankable = merged[
        merged["country_name"].notna()
        & merged["country_name"].ne("Global")
        & merged["count"].ge(MIN_COUNTRY_COUNT)
        & merged["sum(Market Cap (in US $))"].ge(MIN_COUNTRY_MARKET_CAP)
        & merged["forward_pe"].gt(0)
        & merged["total_equity_risk_premium"].notna()
    ].copy()

    q1 = float(rankable["valuation_spread"].quantile(0.25))
    q3 = float(rankable["valuation_spread"].quantile(0.75))
    rankable["allocation_signal"] = np.where(
        rankable["valuation_spread"] >= q3,
        "overweight",
        np.where(rankable["valuation_spread"] <= q1, "underweight", "neutral"),
    )

    rankable = rankable.sort_values("valuation_spread", ascending=False).reset_index(drop=True)

    top_bottom = pd.concat([rankable.head(10), rankable.tail(10)], ignore_index=True)
    us_row = rankable[rankable["country_name"].eq("United States")].head(1).copy()

    selected_columns = [
        "country_name",
        "count",
        "sum(Market Cap (in US $))",
        "forward_pe",
        "dividend_yield",
        "forward_earnings_yield",
        "total_equity_risk_premium",
        "country_risk_premium",
        "valuation_spread",
        "allocation_signal",
    ]

    return {
        "stats_update": None if pd.isna(stats_update) else stats_update.date().isoformat(),
        "erp_update": None if pd.isna(erp_update) else erp_update.date().isoformat(),
        "mature_market_erp": mature_market_erp,
        "us_equity_erp": us_equity_erp,
        "all_countries_df": rankable.loc[:, selected_columns].copy(),
        "top_bottom_df": top_bottom.loc[:, selected_columns].copy(),
        "us_row_df": us_row.loc[:, selected_columns].copy(),
        "coverage_rows": int(len(rankable)),
    }


def save_csv(df: pd.DataFrame, filename: str) -> str:
    path = ensure_output_dir() / filename
    df.to_csv(path, index=False)
    return str(path)


def plot_us_extremes_advantage(us_summary: pd.DataFrame) -> str:
    subset = us_summary[us_summary["window"].eq("paper_aligned")].copy()
    hl = subset[subset["sample"].eq("H+L")].copy()
    mid = subset[subset["sample"].eq("Middle")].copy()
    merged = hl.merge(mid, on=["window", "method", "signal"], suffixes=("_hl", "_mid"))
    merged["corr_advantage"] = merged["corr_real_hl"] - merged["corr_real_mid"]
    merged["rho_advantage"] = merged["forecast_rho_hl"] - merged["forecast_rho_mid"]

    signals = list(SIGNAL_LABELS.values())
    methods = list(SPLIT_METHODS)
    colors = {"quartile": "#0b6e4f", "decile": "#c97c00", "zscore": "#1f4e79"}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    x = np.arange(len(signals))
    width = 0.22

    for idx, method in enumerate(methods):
        current = merged[merged["method"].eq(method)].set_index("signal").reindex(signals)
        offset = (idx - 1) * width
        axes[0].bar(x + offset, current["corr_advantage"], width=width, label=method, color=colors[method])
        axes[1].bar(x + offset, current["rho_advantage"], width=width, label=method, color=colors[method])

    axes[0].set_title("Real Corr Advantage: H+L minus Middle")
    axes[1].set_title("Forecast Rho Advantage: H+L minus Middle")
    for axis in axes:
        axis.axhline(0.0, color="#444444", linewidth=0.8)
        axis.set_xticks(x)
        axis.set_xticklabels(signals)
        axis.grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, loc="upper right")

    path = ensure_output_dir() / "us_extremes_advantage.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_us_live_signal(us_live: pd.DataFrame) -> str:
    subset = us_live[us_live["method"].eq("quartile")].copy()
    subset = subset.sort_values("forecast_regime_real_annual_pct")
    colors = ["#b22222" if regime == "low" else "#1b7f3b" if regime == "high" else "#4c6a92" for regime in subset["regime"]]

    fig, ax = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    ax.barh(subset["signal"], subset["forecast_regime_real_annual_pct"], color=colors)
    ax.set_title("Latest US 10Y Real Return Forecast (Regime Model)")
    ax.set_xlabel("Annualized Real Return (%)")
    ax.grid(axis="x", alpha=0.25)

    path = ensure_output_dir() / "us_live_signal.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def plot_damodaran_country_spread(countries: pd.DataFrame) -> str:
    top = countries.head(8).copy()
    bottom = countries.tail(8).copy().sort_values("valuation_spread", ascending=True)
    chart = pd.concat([top, bottom], ignore_index=True)
    colors = ["#1b7f3b" if value >= 0 else "#b22222" for value in chart["valuation_spread"]]

    fig, ax = plt.subplots(figsize=(9, 6.5), constrained_layout=True)
    ax.barh(chart["country_name"], chart["valuation_spread"] * 100.0, color=colors)
    ax.axvline(0.0, color="#444444", linewidth=0.8)
    ax.set_title("Damodaran Country Valuation Spread")
    ax.set_xlabel("Forward Earnings Yield minus Total ERP (pct pts)")
    ax.grid(axis="x", alpha=0.25)

    path = ensure_output_dir() / "damodaran_country_spread.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def build_markdown_summary(us_payload: dict[str, Any], damodaran_payload: dict[str, Any], chart_paths: dict[str, str]) -> str:
    us_summary = us_payload["summary_df"]
    us_live = us_payload["live_df"]
    quartile_paper = us_summary[
        us_summary["window"].eq("paper_aligned") & us_summary["method"].eq("quartile")
    ].copy()
    hplus = quartile_paper[quartile_paper["sample"].eq("H+L")].set_index("signal")
    middle = quartile_paper[quartile_paper["sample"].eq("Middle")].set_index("signal")

    lines = [
        "# Multiples Research Extension",
        "",
        f"- Shiller data URL: {us_payload['data_url']}",
        f"- Latest Shiller month online: {us_payload['latest_month_online']}",
        f"- Latest Shiller complete quarter: {us_payload['latest_complete_quarter']}",
        f"- Damodaran country stats update: {damodaran_payload['stats_update']}",
        f"- Damodaran country ERP update: {damodaran_payload['erp_update']}",
        "",
        "## US robustness",
    ]

    for signal in SIGNAL_LABELS.values():
        lines.append(
            f"- {signal}: quartile real corr advantage = "
            f"{hplus.at[signal, 'corr_real'] - middle.at[signal, 'corr_real']:.2f}, "
            f"forecast rho advantage = {hplus.at[signal, 'forecast_rho'] - middle.at[signal, 'forecast_rho']:.2f}"
        )

    lines.extend(
        [
            "",
            "## US live signal (quartile split)",
        ]
    )

    latest_quartile = us_live[us_live["method"].eq("quartile")].copy()
    for _, row in latest_quartile.iterrows():
        lines.append(
            f"- {row['signal']}: regime={row['regime']}, signal={row['signal_value_pct']:.2f}%, "
            f"regime-model 10Y real forecast={row['forecast_regime_real_annual_pct']:.2f}%"
        )

    lines.extend(
        [
            "",
            "## Damodaran cross-market signal",
            (
                f"- Coverage: {damodaran_payload['coverage_rows']} countries with at least "
                f"{MIN_COUNTRY_COUNT} listed firms, at least {MIN_COUNTRY_MARKET_CAP:,.0f} in aggregate "
                "market cap, positive forward PE and ERP coverage"
            ),
            (
                f"- Mature market ERP={damodaran_payload['mature_market_erp']*100:.2f}%, "
                f"US ERP={damodaran_payload['us_equity_erp']*100:.2f}%"
            ),
        ]
    )

    top_bottom = damodaran_payload["top_bottom_df"]
    top_three = top_bottom.head(3)
    bottom_three = top_bottom.tail(3)
    lines.append(
        "- Top spread countries: "
        + ", ".join(
            f"{row.country_name} ({row.valuation_spread*100:.2f} pts)" for _, row in top_three.iterrows()
        )
    )
    lines.append(
        "- Bottom spread countries: "
        + ", ".join(
            f"{row.country_name} ({row.valuation_spread*100:.2f} pts)" for _, row in bottom_three.iterrows()
        )
    )

    lines.extend(
        [
            "",
            "## Charts",
            f"- US extremes advantage: {chart_paths['us_extremes_advantage']}",
            f"- US live signal: {chart_paths['us_live_signal']}",
            f"- Damodaran country spread: {chart_paths['damodaran_country_spread']}",
        ]
    )

    path = ensure_output_dir() / "summary.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def build_research_payload(save_outputs: bool = False) -> dict[str, Any]:
    us_payload = build_us_payload()
    damodaran_payload = load_damodaran_country_signal()

    payload: dict[str, Any] = {
        "us": {
            "data_url": us_payload["data_url"],
            "latest_month_online": us_payload["latest_month_online"],
            "latest_complete_quarter": us_payload["latest_complete_quarter"],
            "summary_df": us_payload["summary_df"],
            "live_df": us_payload["live_df"],
            "analyses": us_payload["analyses"],
        },
        "damodaran": {
            "stats_update": damodaran_payload["stats_update"],
            "erp_update": damodaran_payload["erp_update"],
            "mature_market_erp": damodaran_payload["mature_market_erp"],
            "us_equity_erp": damodaran_payload["us_equity_erp"],
            "coverage_rows": damodaran_payload["coverage_rows"],
            "all_countries_df": damodaran_payload["all_countries_df"],
            "top_bottom_df": damodaran_payload["top_bottom_df"],
            "us_row_df": damodaran_payload["us_row_df"],
        },
        "artifacts": {},
    }

    if save_outputs:
        chart_paths = {
            "us_extremes_advantage": plot_us_extremes_advantage(us_payload["summary_df"]),
            "us_live_signal": plot_us_live_signal(us_payload["live_df"]),
            "damodaran_country_spread": plot_damodaran_country_spread(damodaran_payload["all_countries_df"]),
        }

        artifact_paths = {
            "us_regime_summary_csv": save_csv(us_payload["summary_df"], "us_regime_summary.csv"),
            "us_live_signal_csv": save_csv(us_payload["live_df"], "us_live_signal.csv"),
            "damodaran_country_signal_csv": save_csv(damodaran_payload["all_countries_df"], "damodaran_country_signal.csv"),
            "damodaran_top_bottom_csv": save_csv(damodaran_payload["top_bottom_df"], "damodaran_top_bottom.csv"),
            **chart_paths,
        }

        artifact_paths["summary_md"] = build_markdown_summary(us_payload, damodaran_payload, chart_paths)

        json_payload = {
            "us": {
                "data_url": us_payload["data_url"],
                "latest_month_online": us_payload["latest_month_online"],
                "latest_complete_quarter": us_payload["latest_complete_quarter"],
            },
            "damodaran": {
                "stats_update": damodaran_payload["stats_update"],
                "erp_update": damodaran_payload["erp_update"],
                "mature_market_erp": damodaran_payload["mature_market_erp"],
                "us_equity_erp": damodaran_payload["us_equity_erp"],
                "coverage_rows": damodaran_payload["coverage_rows"],
            },
            "artifacts": artifact_paths,
        }
        json_path = ensure_output_dir() / "payload_summary.json"
        json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")
        artifact_paths["payload_summary_json"] = str(json_path)

        payload["artifacts"] = artifact_paths

    return payload


def main() -> None:
    payload = build_research_payload(save_outputs=True)
    us_live = payload["us"]["live_df"]
    latest_quartile = us_live[us_live["method"].eq("quartile")].copy()

    print("US robustness and Damodaran extension complete.")
    print(f"Latest Shiller quarter: {payload['us']['latest_complete_quarter']}")
    print(
        "Latest US quartile regimes: "
        + ", ".join(
            f"{row.signal}={row.regime} ({row.forecast_regime_real_annual_pct:.2f}% real)"
            for _, row in latest_quartile.iterrows()
        )
    )

    countries = payload["damodaran"]["all_countries_df"]
    print(
        "Top 5 Damodaran valuation spreads: "
        + ", ".join(
            f"{row.country_name} ({row.valuation_spread*100:.2f})"
            for _, row in countries.head(5).iterrows()
        )
    )
    print("")
    print(json.dumps(payload["artifacts"], indent=2))


if __name__ == "__main__":
    main()
