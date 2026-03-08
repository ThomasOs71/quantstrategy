from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from damodaran_enhanced_relative_model import build_enhanced_model
from market_signal_suite import (
    annualize_from_monthly,
    build_suite,
    download_monthly_etf_frame,
    max_drawdown,
)


OUTPUT_DIR = Path(__file__).resolve().parent / "research_outputs"
HISTORICAL_START = "2005-01-01"

SIGNAL_SCORE = {
    "underweight": -1.0,
    "relative_underweight": -0.5,
    "market_weight": 0.0,
    "neutral": 0.0,
    "relative_overweight": 0.5,
    "overweight": 1.0,
    "high_conviction_underweight": -1.0,
    "high_conviction_overweight": 1.0,
    "low": -1.0,
    "middle": 0.0,
    "high": 1.0,
}

SECTOR_ETF_MAP: dict[tuple[str, str], dict[str, str]] = {
    ("us", "Materials"): {"proxy_ticker": "XLB", "proxy_name": "Materials Select Sector SPDR", "precision": "direct"},
    ("us", "Communication Services"): {"proxy_ticker": "VOX", "proxy_name": "Vanguard Communication Services ETF", "precision": "direct"},
    ("us", "Energy"): {"proxy_ticker": "XLE", "proxy_name": "Energy Select Sector SPDR", "precision": "direct"},
    ("us", "Financials"): {"proxy_ticker": "XLF", "proxy_name": "Financial Select Sector SPDR", "precision": "direct"},
    ("us", "Industrials"): {"proxy_ticker": "XLI", "proxy_name": "Industrial Select Sector SPDR", "precision": "direct"},
    ("us", "Technology"): {"proxy_ticker": "XLK", "proxy_name": "Technology Select Sector SPDR", "precision": "direct"},
    ("us", "Consumer Staples"): {"proxy_ticker": "XLP", "proxy_name": "Consumer Staples Select Sector SPDR", "precision": "direct"},
    ("us", "Real Estate"): {"proxy_ticker": "VNQ", "proxy_name": "Vanguard Real Estate ETF", "precision": "direct"},
    ("us", "Utilities"): {"proxy_ticker": "XLU", "proxy_name": "Utilities Select Sector SPDR", "precision": "direct"},
    ("us", "Health Care"): {"proxy_ticker": "XLV", "proxy_name": "Health Care Select Sector SPDR", "precision": "direct"},
    ("us", "Consumer Discretionary"): {"proxy_ticker": "XLY", "proxy_name": "Consumer Discretionary Select Sector SPDR", "precision": "direct"},
    ("europe", "Financials"): {"proxy_ticker": "EUFN", "proxy_name": "iShares MSCI Europe Financials ETF", "precision": "direct"},
    ("europe", "Materials"): {"proxy_ticker": "VGK", "proxy_name": "Vanguard FTSE Europe ETF", "precision": "broad_region_proxy"},
    ("europe", "Communication Services"): {"proxy_ticker": "VGK", "proxy_name": "Vanguard FTSE Europe ETF", "precision": "broad_region_proxy"},
    ("europe", "Energy"): {"proxy_ticker": "VGK", "proxy_name": "Vanguard FTSE Europe ETF", "precision": "broad_region_proxy"},
    ("europe", "Industrials"): {"proxy_ticker": "VGK", "proxy_name": "Vanguard FTSE Europe ETF", "precision": "broad_region_proxy"},
    ("europe", "Technology"): {"proxy_ticker": "VGK", "proxy_name": "Vanguard FTSE Europe ETF", "precision": "broad_region_proxy"},
    ("europe", "Consumer Staples"): {"proxy_ticker": "VGK", "proxy_name": "Vanguard FTSE Europe ETF", "precision": "broad_region_proxy"},
    ("europe", "Real Estate"): {"proxy_ticker": "VGK", "proxy_name": "Vanguard FTSE Europe ETF", "precision": "broad_region_proxy"},
    ("europe", "Utilities"): {"proxy_ticker": "VGK", "proxy_name": "Vanguard FTSE Europe ETF", "precision": "broad_region_proxy"},
    ("europe", "Health Care"): {"proxy_ticker": "VGK", "proxy_name": "Vanguard FTSE Europe ETF", "precision": "broad_region_proxy"},
    ("europe", "Consumer Discretionary"): {"proxy_ticker": "VGK", "proxy_name": "Vanguard FTSE Europe ETF", "precision": "broad_region_proxy"},
}

US_BACKTEST_TICKERS: dict[str, str] = {
    "Materials": "XLB",
    "Communication Services": "VOX",
    "Energy": "XLE",
    "Financials": "XLF",
    "Industrials": "XLI",
    "Technology": "XLK",
    "Consumer Staples": "XLP",
    "Real Estate": "VNQ",
    "Utilities": "XLU",
    "Health Care": "XLV",
    "Consumer Discretionary": "XLY",
}


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
    valid = values.notna() & weights.notna()
    if not valid.any():
        return math.nan
    w = weights.loc[valid]
    if math.isclose(float(w.sum()), 0.0):
        return float(values.loc[valid].mean())
    return float(np.average(values.loc[valid], weights=w))


def infer_sector(industry_name: str) -> str:
    name = str(industry_name).lower()

    if "reit" in name or "real estate" in name:
        return "Real Estate"
    if any(token in name for token in ["bank", "insurance", "broker", "financial", "reit"]):
        return "Financials"
    if any(token in name for token in ["telecom", "cable", "broadcast", "publishing", "entertainment", "information services", "advertising"]):
        return "Communication Services"
    if any(token in name for token in ["software", "semiconductor", "computer", "electronics", "internet", "it services"]):
        return "Technology"
    if any(token in name for token in ["drug", "health", "hospital", "medical", "biotech", "pharma"]):
        return "Health Care"
    if any(token in name for token in ["utility", "water utility", "power"]):
        return "Utilities"
    if any(token in name for token in ["oil", "gas", "coal", "energy"]):
        return "Energy"
    if any(token in name for token in ["chemical", "metals", "mining", "paper", "forest", "packaging", "steel", "precious"]):
        return "Materials"
    if any(token in name for token in ["food", "beverage", "tobacco", "household", "grocery", "consumer products"]):
        return "Consumer Staples"
    if any(
        token in name
        for token in [
            "restaurant",
            "retail",
            "hotel",
            "gaming",
            "furn",
            "apparel",
            "auto",
            "shoe",
            "leisure",
            "homebuild",
            "building supply",
        ]
    ):
        return "Consumer Discretionary"
    if any(
        token in name
        for token in [
            "aerospace",
            "air transport",
            "transport",
            "railroad",
            "trucking",
            "machinery",
            "construction",
            "engineering",
            "equipment",
            "ship",
            "office equipment",
        ]
    ):
        return "Industrials"
    return "Industrials"


def classify_sector_regimes(frame: pd.DataFrame) -> pd.DataFrame:
    scored = frame.copy()
    q25 = float(scored["sector_score"].quantile(0.25))
    q75 = float(scored["sector_score"].quantile(0.75))
    scored["valuation_regime"] = np.where(
        scored["sector_score"] >= q75,
        "cheap",
        np.where(scored["sector_score"] <= q25, "expensive", "neutral"),
    )
    scored["sector_action"] = np.where(
        scored["valuation_regime"].eq("cheap"),
        "overweight",
        np.where(scored["valuation_regime"].eq("expensive"), "underweight", "market_weight"),
    )
    scored["score_q25"] = q25
    scored["score_q75"] = q75
    return scored


def aggregate_sector_scores(industry_frame: pd.DataFrame, region: str) -> pd.DataFrame:
    working = industry_frame.copy()
    working["region"] = region
    working["sector"] = working["Industry Name"].map(infer_sector)

    rows: list[dict[str, Any]] = []
    for sector, subset in working.groupby("sector", sort=True):
        weights = subset["number_of_firms"].fillna(1.0)
        rows.append(
            {
                "region": region,
                "sector": sector,
                "n_industries": int(len(subset)),
                "total_firms": float(weights.sum()),
                "sector_score": _weighted_average(subset["final_value_score"], weights),
                "raw_sector_score": _weighted_average(subset["raw_composite_score"], weights),
                "adjusted_sector_score": _weighted_average(subset["adjusted_composite_score"], weights),
                "regression_sector_score": _weighted_average(subset["regression_composite_score"], weights),
                "quality_overlay_delta": _weighted_average(subset["quality_overlay_delta"], weights),
            }
        )

    sector_frame = pd.DataFrame(rows).sort_values("sector").reset_index(drop=True)
    return classify_sector_regimes(sector_frame)


def combine_actions(region_action: str, sector_action: str) -> str:
    region_label = str(region_action)
    sector_label = str(sector_action)

    if region_label == "underweight" and sector_label == "overweight":
        return "relative_overweight"
    if region_label == "underweight" and sector_label == "market_weight":
        return "underweight"
    if region_label == "underweight" and sector_label == "underweight":
        return "high_conviction_underweight"

    if region_label == "neutral" and sector_label == "overweight":
        return "overweight"
    if region_label == "neutral" and sector_label == "market_weight":
        return "market_weight"
    if region_label == "neutral" and sector_label == "underweight":
        return "underweight"

    if region_label == "overweight" and sector_label == "overweight":
        return "high_conviction_overweight"
    if region_label == "overweight" and sector_label == "market_weight":
        return "overweight"
    if region_label == "overweight" and sector_label == "underweight":
        return "relative_underweight"

    return "market_weight"


def build_sector_etf_actions(sector_frame: pd.DataFrame, market_overview: pd.DataFrame) -> pd.DataFrame:
    region_action_map = {
        "us": str(market_overview.loc[market_overview["market"].eq("USA"), "combined_signal"].iloc[0]),
        "europe": str(market_overview.loc[market_overview["market"].eq("Europe"), "combined_signal"].iloc[0]),
    }

    mapped = sector_frame.copy()
    mapped["proxy_ticker"] = None
    mapped["proxy_name"] = None
    mapped["proxy_precision"] = None

    for idx, row in mapped.iterrows():
        info = SECTOR_ETF_MAP.get((str(row["region"]), str(row["sector"])))
        if info is None:
            info = {"proxy_ticker": "VGK" if row["region"] == "europe" else "SPY", "proxy_name": "Fallback broad ETF", "precision": "fallback"}
        mapped.at[idx, "proxy_ticker"] = info["proxy_ticker"]
        mapped.at[idx, "proxy_name"] = info["proxy_name"]
        mapped.at[idx, "proxy_precision"] = info["precision"]

    mapped["region_action"] = mapped["region"].map(region_action_map)
    mapped["portfolio_action"] = [
        combine_actions(region_action, sector_action)
        for region_action, sector_action in zip(mapped["region_action"], mapped["sector_action"])
    ]
    mapped["execution_note"] = np.where(
        mapped["proxy_precision"].eq("direct"),
        "Direct sector ETF proxy available.",
        "Use broad Europe beta proxy; sector purity is limited in the current ETF set.",
    )
    mapped = mapped.sort_values(["region", "sector_score"], ascending=[True, False]).reset_index(drop=True)
    return mapped


def _format_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:.2f}%"


def _signal_color(label: str) -> str:
    value = SIGNAL_SCORE.get(str(label), 0.0)
    if value > 0:
        return "#1b6e3a"
    if value < 0:
        return "#a83b32"
    return "#7a6c54"


def create_usa_europe_dashboard(market_overview: pd.DataFrame, sector_actions: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 8),
        gridspec_kw={"width_ratios": [1.0, 1.15, 1.15]},
    )
    ax_market, ax_us, ax_europe = axes

    market_scores = [SIGNAL_SCORE.get(str(value), 0.0) for value in market_overview["combined_signal"]]
    x = np.arange(len(market_overview))

    ax_market.axhspan(-1.2, -0.4, color="#f4d7d3", alpha=0.7)
    ax_market.axhspan(-0.4, 0.4, color="#efe8d7", alpha=0.8)
    ax_market.axhspan(0.4, 1.2, color="#d7ead8", alpha=0.7)
    ax_market.bar(x, market_scores, color=[_signal_color(value) for value in market_overview["combined_signal"]], width=0.5)
    ax_market.set_xticks(x)
    ax_market.set_xticklabels(market_overview["market"], fontsize=11)
    ax_market.set_ylim(-1.15, 1.15)
    ax_market.set_yticks([-1, 0, 1])
    ax_market.set_yticklabels(["Underweight", "Neutral", "Overweight"])
    ax_market.set_title("Market Regime", fontsize=14, fontweight="bold")
    ax_market.grid(axis="y", alpha=0.2)

    for idx, row in market_overview.reset_index(drop=True).iterrows():
        note = (
            f"{row['combined_signal']}\n"
            f"Fwd PE {row['damodaran_forward_pe']:.1f}x\n"
            f"Spread {_format_pct(row['damodaran_spread_pct'])}\n"
            f"ETF regime {row['etf_regime']}"
        )
        ax_market.text(idx, market_scores[idx] + (0.08 if market_scores[idx] >= 0 else -0.08), note, ha="center", va="bottom" if market_scores[idx] >= 0 else "top", fontsize=9)

    def draw_sector_panel(ax: plt.Axes, region_key: str, title: str) -> None:
        subset = sector_actions[sector_actions["region"].eq(region_key)].copy()
        top = subset.head(4)
        bottom = subset.tail(3).sort_values("sector_score", ascending=True)
        display = pd.concat([top, bottom], ignore_index=True)
        positions = np.arange(len(display))
        colors = [_signal_color(value) for value in display["portfolio_action"]]
        labels = [f"{sector} ({ticker})" for sector, ticker in zip(display["sector"], display["proxy_ticker"])]

        ax.barh(positions, display["sector_score"], color=colors, alpha=0.9)
        ax.axvline(0.0, color="#444444", linewidth=1.0)
        ax.set_yticks(positions)
        ax.set_yticklabels(labels, fontsize=10)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(axis="x", alpha=0.2)
        ax.set_xlabel("Sector score")

        for pos, row in zip(positions, display.itertuples(index=False)):
            text = f"{row.portfolio_action.replace('_', ' ')}"
            anchor = row.sector_score + (0.06 if row.sector_score >= 0 else -0.06)
            ax.text(anchor, pos, text, va="center", ha="left" if row.sector_score >= 0 else "right", fontsize=9)

    draw_sector_panel(ax_us, "us", "USA Sector Tilt")
    draw_sector_panel(ax_europe, "europe", "Europe Sector Tilt")

    fig.suptitle("USA vs Europe: Broad Regime and Sector Actions", fontsize=18, fontweight="bold")
    fig.text(0.5, 0.02, "Green = overweight bias, red = underweight bias. Europe uses VGK as fallback where direct sector ETFs are sparse.", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))

    path = ensure_output_dir() / "usa_europe_enhanced_allocation_dashboard.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _build_sector_history_panel() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for sector, ticker in US_BACKTEST_TICKERS.items():
        try:
            frame = download_monthly_etf_frame(ticker)
        except Exception:
            continue

        if frame.empty:
            continue

        frame = frame.copy().sort_index()
        frame["monthly_total_return"] = frame["adj_close"].pct_change()
        frame["div_growth_12m"] = frame["trailing_12m_div"] / frame["trailing_12m_div"].shift(12) - 1.0
        frame["price_momentum_12m"] = frame["adj_close"] / frame["adj_close"].shift(12) - 1.0
        frame["vol_12m"] = frame["monthly_total_return"].rolling(12).std(ddof=0) * math.sqrt(12.0)

        use = frame.loc[:, ["trailing_12m_dy", "div_growth_12m", "price_momentum_12m", "vol_12m", "next_1m_total_return"]].copy()
        use = use[use.index >= pd.Timestamp(HISTORICAL_START)]

        for date, point in use.iterrows():
            rows.append(
                {
                    "date": date,
                    "sector": sector,
                    "ticker": ticker,
                    "dy_12m": point["trailing_12m_dy"],
                    "div_growth_12m": point["div_growth_12m"],
                    "price_momentum_12m": point["price_momentum_12m"],
                    "vol_12m": point["vol_12m"],
                    "next_1m_total_return": point["next_1m_total_return"],
                }
            )

    panel = pd.DataFrame(rows)
    if panel.empty:
        raise ValueError("No sector ETF history could be downloaded.")
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def _cross_sectional_z(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    mean = values.mean()
    std = values.std(ddof=0)
    if pd.isna(std) or math.isclose(float(std), 0.0):
        return pd.Series(0.0, index=values.index)
    return (values - mean) / std


def run_proxy_backtest() -> tuple[pd.DataFrame, pd.DataFrame]:
    panel = _build_sector_history_panel()
    enriched = panel.copy()
    enriched["raw_score"] = np.nan
    enriched["enhanced_score"] = np.nan

    for date, idx in enriched.groupby("date").groups.items():
        slice_df = enriched.loc[idx].copy()
        z_dy = _cross_sectional_z(slice_df["dy_12m"])
        z_div_growth = _cross_sectional_z(slice_df["div_growth_12m"])
        z_momentum = _cross_sectional_z(slice_df["price_momentum_12m"])
        z_vol = _cross_sectional_z(slice_df["vol_12m"])

        raw_score = z_dy
        enhanced_score = z_dy + 0.35 * z_div_growth + 0.25 * z_momentum - 0.35 * z_vol

        enriched.loc[idx, "raw_score"] = raw_score.to_numpy(dtype=float)
        enriched.loc[idx, "enhanced_score"] = enhanced_score.to_numpy(dtype=float)

    rows: list[dict[str, Any]] = []
    for date, slice_df in enriched.groupby("date", sort=True):
        available = slice_df[slice_df["next_1m_total_return"].notna()].copy()
        available = available.dropna(subset=["raw_score", "enhanced_score"])
        if available.empty:
            continue

        selection_count = max(1, math.ceil(len(available) / 4))
        raw_selected = available.nlargest(selection_count, "raw_score")
        enhanced_selected = available.nlargest(selection_count, "enhanced_score")

        rows.append(
            {
                "date": date,
                "raw_return": float(raw_selected["next_1m_total_return"].mean()),
                "enhanced_return": float(enhanced_selected["next_1m_total_return"].mean()),
                "benchmark_return": float(available["next_1m_total_return"].mean()),
                "n_assets": int(len(available)),
                "n_selected": int(selection_count),
            }
        )

    backtest = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if backtest.empty:
        raise ValueError("Proxy backtest produced no observations.")

    backtest["raw_equity"] = (1.0 + backtest["raw_return"]).cumprod()
    backtest["enhanced_equity"] = (1.0 + backtest["enhanced_return"]).cumprod()
    backtest["benchmark_equity"] = (1.0 + backtest["benchmark_return"]).cumprod()

    summaries: list[dict[str, Any]] = []
    for label in ("raw", "enhanced", "benchmark"):
        return_col = f"{label}_return"
        equity_col = f"{label}_equity"
        series = backtest[return_col].dropna()
        annual_return = annualize_from_monthly(series)
        annual_vol = float(series.std(ddof=0) * math.sqrt(12.0))
        sharpe = annual_return / annual_vol if annual_vol and not math.isclose(annual_vol, 0.0) else math.nan
        summaries.append(
            {
                "portfolio": label,
                "annual_return": annual_return,
                "annual_vol": annual_vol,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown(backtest[equity_col]),
                "hit_rate": float((series > 0).mean()),
                "obs_months": int(len(series)),
                "avg_names": float(backtest["n_selected"].mean()) if label != "benchmark" else float(backtest["n_assets"].mean()),
            }
        )

    summary = pd.DataFrame(summaries)
    raw_return = float(summary.loc[summary["portfolio"].eq("raw"), "annual_return"].iloc[0])
    enhanced_return = float(summary.loc[summary["portfolio"].eq("enhanced"), "annual_return"].iloc[0])
    summary["vs_raw_annual_excess"] = np.where(
        summary["portfolio"].eq("enhanced"),
        enhanced_return - raw_return,
        np.where(summary["portfolio"].eq("raw"), 0.0, math.nan),
    )
    return backtest, summary


def create_backtest_chart(backtest: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(backtest["date"], backtest["raw_equity"], label="Raw value signal", color="#8c6a43", linewidth=2.0)
    ax.plot(backtest["date"], backtest["enhanced_equity"], label="Enhanced proxy signal", color="#1f6b5d", linewidth=2.4)
    ax.plot(backtest["date"], backtest["benchmark_equity"], label="Equal-weight benchmark", color="#555555", linewidth=1.8, linestyle="--")
    ax.set_title("US Sector ETF Proxy Backtest", fontsize=16, fontweight="bold")
    ax.set_ylabel("Growth of 1.0")
    ax.grid(alpha=0.2)
    ax.legend()
    fig.tight_layout()

    path = ensure_output_dir() / "us_sector_proxy_enhanced_backtest.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def write_notes(
    market_overview: pd.DataFrame,
    sector_actions: pd.DataFrame,
    backtest_summary: pd.DataFrame,
) -> Path:
    usa = market_overview.loc[market_overview["market"].eq("USA")].iloc[0]
    europe = market_overview.loc[market_overview["market"].eq("Europe")].iloc[0]

    us_top = sector_actions[sector_actions["region"].eq("us")].head(3)
    eu_top = sector_actions[sector_actions["region"].eq("europe")].head(3)

    enhanced_row = backtest_summary.loc[backtest_summary["portfolio"].eq("enhanced")].iloc[0]
    raw_row = backtest_summary.loc[backtest_summary["portfolio"].eq("raw")].iloc[0]

    lines = [
        "# Enhanced Allocation Extension",
        "",
        "## Broad Market",
        "",
        f"- USA: {usa['combined_signal']} (Forward PE {usa['damodaran_forward_pe']:.1f}x, spread {usa['damodaran_spread_pct']:.2f}%, ETF regime {usa['etf_regime']})",
        f"- Europe: {europe['combined_signal']} (Forward PE {europe['damodaran_forward_pe']:.1f}x, spread {europe['damodaran_spread_pct']:.2f}%, ETF regime {europe['etf_regime']})",
        "",
        "## Preferred Sector Tilts",
        "",
        "Top USA ideas:",
    ]

    for row in us_top.itertuples(index=False):
        lines.append(
            f"- {row.sector}: {row.portfolio_action} via {row.proxy_ticker} ({row.proxy_precision})"
        )

    lines.append("")
    lines.append("Top Europe ideas:")
    for row in eu_top.itertuples(index=False):
        lines.append(
            f"- {row.sector}: {row.portfolio_action} via {row.proxy_ticker} ({row.proxy_precision})"
        )

    lines.extend(
        [
            "",
            "## Proxy Backtest",
            "",
            f"- Raw signal annual return: {raw_row['annual_return']*100:.2f}%",
            f"- Enhanced signal annual return: {enhanced_row['annual_return']*100:.2f}%",
            f"- Improvement vs raw: {enhanced_row['vs_raw_annual_excess']*100:.2f}%",
            "",
            "Caveat: the backtest is a US sector ETF proxy. Damodaran does not publish a clean public archive of point-in-time industry snapshots suitable for a true historical replay, so the historical test uses live market proxies (dividend yield, dividend growth, momentum, volatility) rather than archived fundamental snapshots.",
            "",
        ]
    )

    path = ensure_output_dir() / "enhanced_allocation_extension_notes.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_extension() -> dict[str, Any]:
    ensure_output_dir()

    market_payload = build_suite(save_outputs=True)
    enhanced_payload = build_enhanced_model()

    market_overview = market_payload["main_market_overview"].copy()
    us_industries = pd.read_csv(enhanced_payload["artifacts"]["us_sector_scores"])
    europe_industries = pd.read_csv(enhanced_payload["artifacts"]["europe_sector_scores"])

    us_sector_scores = aggregate_sector_scores(us_industries, "us")
    europe_sector_scores = aggregate_sector_scores(europe_industries, "europe")
    combined_sector_scores = pd.concat([us_sector_scores, europe_sector_scores], ignore_index=True)
    sector_actions = build_sector_etf_actions(combined_sector_scores, market_overview)

    backtest_curve, backtest_summary = run_proxy_backtest()

    dashboard_path = create_usa_europe_dashboard(market_overview, sector_actions)
    backtest_chart_path = create_backtest_chart(backtest_curve)

    sector_scores_path = ensure_output_dir() / "usa_europe_sector_scores.csv"
    sector_actions_path = ensure_output_dir() / "usa_europe_sector_etf_actions.csv"
    backtest_curve_path = ensure_output_dir() / "us_sector_proxy_enhanced_backtest_curve.csv"
    backtest_summary_path = ensure_output_dir() / "us_sector_proxy_enhanced_backtest_summary.csv"

    combined_sector_scores.to_csv(sector_scores_path, index=False)
    sector_actions.to_csv(sector_actions_path, index=False)
    backtest_curve.to_csv(backtest_curve_path, index=False)
    backtest_summary.to_csv(backtest_summary_path, index=False)

    notes_path = write_notes(market_overview, sector_actions, backtest_summary)

    payload = {
        "as_of": "2026-02-28",
        "artifacts": {
            "sector_scores_csv": str(sector_scores_path),
            "sector_actions_csv": str(sector_actions_path),
            "dashboard_png": str(dashboard_path),
            "backtest_curve_csv": str(backtest_curve_path),
            "backtest_summary_csv": str(backtest_summary_path),
            "backtest_png": str(backtest_chart_path),
            "notes_md": str(notes_path),
        },
        "market_overview": market_overview.to_dict(orient="records"),
        "backtest_summary": backtest_summary.to_dict(orient="records"),
    }

    payload_path = ensure_output_dir() / "enhanced_allocation_extension.json"
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["artifacts"]["payload_json"] = str(payload_path)
    return payload


def main() -> None:
    payload = build_extension()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
