from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from multiples_research_pipeline import (
    COUNTRY_ERP_URL,
    COUNTRYSTATS_URL,
    MIN_COUNTRY_COUNT,
    MIN_COUNTRY_MARKET_CAP,
    build_research_payload,
    normalize_country_name,
)


OUTPUT_DIR = Path(__file__).with_name("research_outputs")
ETF_START_DATE = "2005-01-01"
BACKTEST_LOOKBACK_MONTHS = 36
FORWARD_HORIZON_MONTHS = 12


ETF_MAPPING = [
    {"market_key": "United States", "proxy_ticker": "SPY", "proxy_name": "SPDR S&P 500 ETF Trust", "backtest_included": True, "major_market": True},
    {"market_key": "Europe", "proxy_ticker": "VGK", "proxy_name": "Vanguard FTSE Europe ETF", "backtest_included": True, "major_market": True},
    {"market_key": "United Kingdom", "proxy_ticker": "EWU", "proxy_name": "iShares MSCI United Kingdom ETF", "backtest_included": True, "major_market": False},
    {"market_key": "Germany", "proxy_ticker": "EWG", "proxy_name": "iShares MSCI Germany ETF", "backtest_included": True, "major_market": False},
    {"market_key": "France", "proxy_ticker": "EWQ", "proxy_name": "iShares MSCI France ETF", "backtest_included": True, "major_market": False},
    {"market_key": "Spain", "proxy_ticker": "EWP", "proxy_name": "iShares MSCI Spain ETF", "backtest_included": True, "major_market": False},
    {"market_key": "Italy", "proxy_ticker": "EWI", "proxy_name": "iShares MSCI Italy ETF", "backtest_included": True, "major_market": False},
    {"market_key": "Switzerland", "proxy_ticker": "EWL", "proxy_name": "iShares MSCI Switzerland ETF", "backtest_included": True, "major_market": False},
    {"market_key": "Netherlands", "proxy_ticker": "EWN", "proxy_name": "iShares MSCI Netherlands ETF", "backtest_included": True, "major_market": False},
    {"market_key": "Sweden", "proxy_ticker": "EWD", "proxy_name": "iShares MSCI Sweden ETF", "backtest_included": True, "major_market": False},
    {"market_key": "Austria", "proxy_ticker": "EWO", "proxy_name": "iShares MSCI Austria ETF", "backtest_included": True, "major_market": False},
    {"market_key": "Japan", "proxy_ticker": "EWJ", "proxy_name": "iShares MSCI Japan ETF", "backtest_included": True, "major_market": False},
    {"market_key": "Australia", "proxy_ticker": "EWA", "proxy_name": "iShares MSCI Australia ETF", "backtest_included": True, "major_market": False},
    {"market_key": "Canada", "proxy_ticker": "EWC", "proxy_name": "iShares MSCI Canada ETF", "backtest_included": True, "major_market": False},
    {"market_key": "Hong Kong", "proxy_ticker": "EWH", "proxy_name": "iShares MSCI Hong Kong ETF", "backtest_included": True, "major_market": False},
    {"market_key": "South Korea", "proxy_ticker": "EWY", "proxy_name": "iShares MSCI South Korea ETF", "backtest_included": True, "major_market": False},
    {"market_key": "Taiwan", "proxy_ticker": "EWT", "proxy_name": "iShares MSCI Taiwan ETF", "backtest_included": True, "major_market": False},
    {"market_key": "India", "proxy_ticker": "INDA", "proxy_name": "iShares MSCI India ETF", "backtest_included": True, "major_market": False},
    {"market_key": "China", "proxy_ticker": "FXI", "proxy_name": "iShares China Large-Cap ETF", "backtest_included": True, "major_market": False},
    {"market_key": "Brazil", "proxy_ticker": "EWZ", "proxy_name": "iShares MSCI Brazil ETF", "backtest_included": True, "major_market": False},
    {"market_key": "Mexico", "proxy_ticker": "EWW", "proxy_name": "iShares MSCI Mexico ETF", "backtest_included": True, "major_market": False},
    {"market_key": "South Africa", "proxy_ticker": "EZA", "proxy_name": "iShares MSCI South Africa ETF", "backtest_included": True, "major_market": False},
    {"market_key": "Poland", "proxy_ticker": "EPOL", "proxy_name": "iShares MSCI Poland ETF", "backtest_included": True, "major_market": False},
    {"market_key": "Turkey", "proxy_ticker": "TUR", "proxy_name": "iShares MSCI Turkey ETF", "backtest_included": True, "major_market": False},
    {"market_key": "Chile", "proxy_ticker": "ECH", "proxy_name": "iShares MSCI Chile ETF", "backtest_included": True, "major_market": False},
    {"market_key": "Peru", "proxy_ticker": "EPU", "proxy_name": "iShares MSCI Peru and Global Exposure ETF", "backtest_included": True, "major_market": False},
    {"market_key": "Argentina", "proxy_ticker": "ARGT", "proxy_name": "Global X MSCI Argentina ETF", "backtest_included": True, "major_market": False},
    {"market_key": "Philippines", "proxy_ticker": "EPHE", "proxy_name": "iShares MSCI Philippines ETF", "backtest_included": True, "major_market": False},
    {"market_key": "Vietnam", "proxy_ticker": "VNM", "proxy_name": "VanEck Vietnam ETF", "backtest_included": True, "major_market": False},
    {"market_key": "Malaysia", "proxy_ticker": "EWM", "proxy_name": "iShares MSCI Malaysia ETF", "backtest_included": True, "major_market": False},
]


def ensure_output_dir() -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUTPUT_DIR


def save_csv(df: pd.DataFrame, filename: str) -> str:
    path = ensure_output_dir() / filename
    df.to_csv(path, index=False)
    return str(path)


def save_json(payload: dict[str, Any], filename: str) -> str:
    path = ensure_output_dir() / filename
    path.write_text(json.dumps(sanitize_for_json(payload), indent=2, allow_nan=False), encoding="utf-8")
    return str(path)


def mapping_dataframe() -> pd.DataFrame:
    return pd.DataFrame(ETF_MAPPING)


def sanitize_for_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_for_json(item) for item in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def load_damodaran_snapshot() -> dict[str, Any]:
    stats = pd.read_excel(COUNTRYSTATS_URL, header=8)
    stats = stats.rename(columns={stats.columns[0]: "country_name"})
    stats = stats[stats["country_name"].notna()].copy()
    stats["normalized_country"] = stats["country_name"].map(normalize_country_name)
    stats["count"] = pd.to_numeric(stats["count"], errors="coerce")
    stats["sum(Market Cap (in US $))"] = pd.to_numeric(stats["sum(Market Cap (in US $))"], errors="coerce")
    stats["median(Forward PE)"] = pd.to_numeric(stats["median(Forward PE)"], errors="coerce")
    stats["Dividend Yield"] = pd.to_numeric(stats["Dividend Yield"], errors="coerce")

    erp = pd.read_excel(COUNTRY_ERP_URL, sheet_name="ERPs by country", header=7)
    erp = erp[erp["Country"].notna()].copy()
    erp = erp.rename(
        columns={
            "Country": "country_name_erp",
            "Africa": "region_group",
            "Total Equity Risk Premium": "total_equity_risk_premium",
            "Country Risk Premium": "country_risk_premium",
        }
    )
    erp["normalized_country"] = erp["country_name_erp"].map(normalize_country_name)
    erp["total_equity_risk_premium"] = pd.to_numeric(erp["total_equity_risk_premium"], errors="coerce")
    erp["country_risk_premium"] = pd.to_numeric(erp["country_risk_premium"], errors="coerce")

    merged = stats.merge(
        erp[["normalized_country", "country_name_erp", "region_group", "total_equity_risk_premium", "country_risk_premium"]],
        on="normalized_country",
        how="left",
    )

    merged["forward_pe"] = merged["median(Forward PE)"]
    merged["forward_earnings_yield"] = 1.0 / merged["forward_pe"]
    merged["valuation_spread"] = merged["forward_earnings_yield"] - merged["total_equity_risk_premium"]

    investable = merged[
        merged["country_name"].notna()
        & merged["country_name"].ne("Global")
        & merged["count"].ge(MIN_COUNTRY_COUNT)
        & merged["sum(Market Cap (in US $))"].ge(MIN_COUNTRY_MARKET_CAP)
        & merged["forward_pe"].gt(0)
        & merged["total_equity_risk_premium"].notna()
    ].copy()

    q1 = float(investable["valuation_spread"].quantile(0.25))
    q3 = float(investable["valuation_spread"].quantile(0.75))
    investable["damodaran_signal"] = np.where(
        investable["valuation_spread"] >= q3,
        "overweight",
        np.where(investable["valuation_spread"] <= q1, "underweight", "neutral"),
    )
    investable["spread_percentile"] = investable["valuation_spread"].rank(pct=True, method="average")
    investable = investable.sort_values("valuation_spread", ascending=False).reset_index(drop=True)

    return {
        "all_df": investable,
        "spread_q1": q1,
        "spread_q3": q3,
    }


def merge_country_signals_with_etfs(country_df: pd.DataFrame, mapping_df: pd.DataFrame) -> pd.DataFrame:
    mapping = mapping_df.copy()
    mapping["normalized_country"] = mapping["market_key"].map(normalize_country_name)

    merged = mapping.merge(
        country_df,
        on="normalized_country",
        how="left",
        suffixes=("_map", ""),
    )

    merged["market_label"] = merged["market_key"]
    merged["country_name"] = merged["country_name"].fillna(merged["market_key"])

    keep = [
        "market_label",
        "proxy_ticker",
        "proxy_name",
        "major_market",
        "backtest_included",
        "country_name",
        "region_group",
        "count",
        "sum(Market Cap (in US $))",
        "forward_pe",
        "Dividend Yield",
        "forward_earnings_yield",
        "total_equity_risk_premium",
        "country_risk_premium",
        "valuation_spread",
        "spread_percentile",
        "damodaran_signal",
    ]
    return merged.loc[:, keep].copy()


def download_monthly_etf_frame(ticker: str) -> pd.DataFrame:
    hist_raw = yf.Ticker(ticker).history(start=ETF_START_DATE, auto_adjust=False)
    hist_adj = yf.Ticker(ticker).history(start=ETF_START_DATE, auto_adjust=True)
    if hist_raw.empty or hist_adj.empty:
        raise ValueError(f"No data returned for {ticker}")

    raw = hist_raw.copy()
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    adj = hist_adj.copy()
    adj.index = pd.to_datetime(adj.index).tz_localize(None)

    monthly_close = raw["Close"].resample("ME").last()
    monthly_div = raw["Dividends"].resample("ME").sum()
    monthly_adj = adj["Close"].resample("ME").last()

    frame = pd.DataFrame(
        {
            "price": monthly_close,
            "dividends": monthly_div,
            "adj_close": monthly_adj,
        }
    ).dropna(subset=["price", "adj_close"])

    frame["trailing_12m_div"] = frame["dividends"].rolling(FORWARD_HORIZON_MONTHS).sum()
    frame["trailing_12m_dy"] = frame["trailing_12m_div"] / frame["price"]
    frame["next_1m_total_return"] = frame["adj_close"].pct_change().shift(-1)
    frame["fwd_12m_total_return"] = frame["adj_close"].shift(-FORWARD_HORIZON_MONTHS) / frame["adj_close"] - 1.0
    frame["fwd_12m_total_return_ann"] = (1.0 + frame["fwd_12m_total_return"]) ** (1.0 / 1.0) - 1.0
    frame.index.name = "date"
    return frame


def classify_regime(current_value: float, history: pd.Series) -> tuple[str | None, float | None, float | None]:
    valid = history.dropna()
    if len(valid) < BACKTEST_LOOKBACK_MONTHS:
        return None, math.nan, math.nan

    q1 = float(valid.quantile(0.25))
    q3 = float(valid.quantile(0.75))
    if current_value >= q3:
        return "high", q1, q3
    if current_value <= q1:
        return "low", q1, q3
    return "middle", q1, q3


def build_etf_panel(mapping_df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    panel_rows: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}

    for row in mapping_df.itertuples(index=False):
        if not bool(row.backtest_included):
            continue
        ticker = str(row.proxy_ticker)
        try:
            frame = download_monthly_etf_frame(ticker)
        except Exception:
            continue

        frames[ticker] = frame
        for date, point in frame.iterrows():
            panel_rows.append(
                {
                    "date": date,
                    "market_label": row.market_key,
                    "ticker": ticker,
                    "proxy_name": row.proxy_name,
                    "price": point["price"],
                    "trailing_12m_dy": point["trailing_12m_dy"],
                    "next_1m_total_return": point["next_1m_total_return"],
                    "fwd_12m_total_return_ann": point["fwd_12m_total_return_ann"],
                }
            )

    panel = pd.DataFrame(panel_rows)
    if panel.empty:
        raise ValueError("No ETF data could be downloaded for the configured universe.")
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)
    return panel, frames


def add_expanding_regimes(panel: pd.DataFrame) -> pd.DataFrame:
    enriched = panel.copy()
    enriched["etf_regime"] = None
    enriched["q1"] = np.nan
    enriched["q3"] = np.nan

    for ticker, subset in enriched.groupby("ticker", sort=False):
        indices = subset.index.to_list()
        dy_values = subset["trailing_12m_dy"].tolist()
        for pos, idx in enumerate(indices):
            current_value = dy_values[pos]
            if pd.isna(current_value):
                continue
            history = pd.Series(dy_values[:pos])
            regime, q1, q3 = classify_regime(float(current_value), history)
            enriched.at[idx, "etf_regime"] = regime
            enriched.at[idx, "q1"] = q1
            enriched.at[idx, "q3"] = q3

    return enriched


def backtest_high_yield_strategy(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    dates = sorted(panel["date"].unique())

    for date in dates:
        slice_df = panel[panel["date"].eq(date)].copy()
        available = slice_df[
            slice_df["next_1m_total_return"].notna()
            & slice_df["etf_regime"].notna()
        ].copy()
        if available.empty:
            continue

        selected = available[available["etf_regime"].eq("high")].copy()
        if selected.empty:
            selected = available
            selected_flag = "fallback_all"
        else:
            selected_flag = "high_only"

        strategy_return = float(selected["next_1m_total_return"].mean())
        benchmark_return = float(available["next_1m_total_return"].mean())

        rows.append(
            {
                "date": date,
                "strategy_return": strategy_return,
                "benchmark_return": benchmark_return,
                "excess_return": strategy_return - benchmark_return,
                "n_selected": int(len(selected)),
                "n_available": int(len(available)),
                "selection_mode": selected_flag,
            }
        )

    backtest = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    if backtest.empty:
        raise ValueError("Backtest produced no observations.")

    backtest["strategy_equity"] = (1.0 + backtest["strategy_return"]).cumprod()
    backtest["benchmark_equity"] = (1.0 + backtest["benchmark_return"]).cumprod()
    backtest["excess_equity"] = backtest["strategy_equity"] / backtest["benchmark_equity"]

    summary_rows = [
        summarize_backtest_line(backtest["strategy_return"], "strategy", backtest["n_selected"]),
        summarize_backtest_line(backtest["benchmark_return"], "benchmark", backtest["n_available"]),
    ]
    summary = pd.DataFrame(summary_rows)
    summary["start"] = str(backtest["date"].iloc[0].date())
    summary["end"] = str(backtest["date"].iloc[-1].date())
    summary["obs_months"] = int(len(backtest))
    summary["annualized_excess_return"] = annualize_from_monthly(backtest["strategy_return"]) - annualize_from_monthly(backtest["benchmark_return"])
    summary["avg_excess_return_monthly"] = float(backtest["excess_return"].mean())
    return backtest, summary


def annualize_from_monthly(series: pd.Series) -> float:
    valid = pd.Series(series).dropna()
    if valid.empty:
        return math.nan
    growth = float((1.0 + valid).prod())
    years = len(valid) / 12.0
    if years <= 0:
        return math.nan
    return growth ** (1.0 / years) - 1.0


def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(dd.min())


def summarize_backtest_line(returns: pd.Series, label: str, counts: pd.Series) -> dict[str, Any]:
    valid = pd.Series(returns).dropna()
    annual_return = annualize_from_monthly(valid)
    annual_vol = float(valid.std(ddof=0) * math.sqrt(12.0))
    sharpe = annual_return / annual_vol if annual_vol and not math.isclose(annual_vol, 0.0) else math.nan
    equity = (1.0 + valid).cumprod()
    return {
        "portfolio": label,
        "annual_return": annual_return,
        "annual_vol": annual_vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(equity),
        "hit_rate": float((valid > 0).mean()),
        "avg_names": float(pd.Series(counts).mean()),
    }


def current_etf_signal(panel: pd.DataFrame, market_label: str, ticker: str) -> dict[str, Any]:
    subset = panel[panel["ticker"].eq(ticker)].copy().sort_values("date")
    latest = subset.iloc[-1]
    same_regime = subset[
        subset["etf_regime"].eq(latest["etf_regime"])
        & subset["fwd_12m_total_return_ann"].notna()
    ]
    expected = float(same_regime["fwd_12m_total_return_ann"].mean()) if not same_regime.empty else math.nan
    return {
        "market": market_label,
        "ticker": ticker,
        "date": str(pd.Timestamp(latest["date"]).date()),
        "trailing_12m_dy_pct": float(latest["trailing_12m_dy"] * 100.0),
        "q1_pct": float(latest["q1"] * 100.0) if pd.notna(latest["q1"]) else math.nan,
        "q3_pct": float(latest["q3"] * 100.0) if pd.notna(latest["q3"]) else math.nan,
        "etf_regime": latest["etf_regime"],
        "expected_12m_total_return_pct": expected * 100.0 if not math.isnan(expected) else math.nan,
    }


def compute_us_composite(payload: dict[str, Any]) -> dict[str, Any]:
    live = payload["us"]["live_df"].copy()
    quartile = live[live["method"].eq("quartile")].copy()
    regime_score_map = {"high": 1.0, "middle": 0.0, "low": -1.0}
    quartile["regime_score"] = quartile["regime"].map(regime_score_map)

    composite_score = float(quartile["regime_score"].mean())
    composite_forecast = float(quartile["forecast_regime_real_annual_pct"].mean())

    if composite_score <= -0.5:
        composite_signal = "underweight"
        valuation_state = "expensive"
    elif composite_score >= 0.5:
        composite_signal = "overweight"
        valuation_state = "cheap"
    else:
        composite_signal = "neutral"
        valuation_state = "neutral"

    return {
        "valuation_state": valuation_state,
        "composite_signal": composite_signal,
        "composite_score": composite_score,
        "composite_forecast_real_10y_pct": composite_forecast,
        "signal_details": quartile[["signal", "regime", "forecast_regime_real_annual_pct"]].to_dict(orient="records"),
    }


def compute_europe_aggregate(country_df: pd.DataFrame, spread_q1: float, spread_q3: float) -> dict[str, Any]:
    europe = country_df[country_df["region_group"].astype(str).str.contains("Europe", case=False, na=False)].copy()
    if europe.empty:
        raise ValueError("No European countries available in Damodaran snapshot.")

    weights = europe["sum(Market Cap (in US $))"] / europe["sum(Market Cap (in US $))"].sum()
    capw_ey = float((weights * europe["forward_earnings_yield"]).sum())
    capw_erp = float((weights * europe["total_equity_risk_premium"]).sum())
    spread = capw_ey - capw_erp
    capw_forward_pe = 1.0 / capw_ey if capw_ey > 0 else math.nan

    if spread <= spread_q1:
        signal = "underweight"
    elif spread >= spread_q3:
        signal = "overweight"
    else:
        signal = "neutral"

    return {
        "country_count": int(len(europe)),
        "cap_weighted_forward_pe": capw_forward_pe,
        "cap_weighted_forward_earnings_yield": capw_ey,
        "cap_weighted_total_erp": capw_erp,
        "valuation_spread": spread,
        "damodaran_signal": signal,
    }


def combine_market_signal(scores: list[float]) -> str:
    valid = [score for score in scores if not math.isnan(score)]
    if not valid:
        return "n/a"
    avg = float(np.mean(valid))
    if avg >= 0.5:
        return "overweight"
    if avg <= -0.5:
        return "underweight"
    return "neutral"


def build_market_overview(
    us_composite: dict[str, Any],
    usa_row: pd.Series,
    usa_etf: dict[str, Any],
    europe_agg: dict[str, Any],
    europe_etf: dict[str, Any],
) -> pd.DataFrame:
    regime_score_map = {"high": 1.0, "middle": 0.0, "low": -1.0}
    dam_signal_score = {"overweight": 1.0, "neutral": 0.0, "underweight": -1.0}

    usa_scores = [
        us_composite["composite_score"],
        dam_signal_score.get(str(usa_row["damodaran_signal"]), math.nan),
        regime_score_map.get(str(usa_etf["etf_regime"]), math.nan),
    ]
    europe_scores = [
        dam_signal_score.get(str(europe_agg["damodaran_signal"]), math.nan),
        regime_score_map.get(str(europe_etf["etf_regime"]), math.nan),
    ]

    rows = [
        {
            "market": "USA",
            "proxy_ticker": usa_etf["ticker"],
            "damodaran_forward_pe": float(usa_row["forward_pe"]),
            "damodaran_forward_earnings_yield_pct": float(usa_row["forward_earnings_yield"] * 100.0),
            "damodaran_total_erp_pct": float(usa_row["total_equity_risk_premium"] * 100.0),
            "damodaran_spread_pct": float(usa_row["valuation_spread"] * 100.0),
            "damodaran_signal": str(usa_row["damodaran_signal"]),
            "etf_trailing_12m_dy_pct": usa_etf["trailing_12m_dy_pct"],
            "etf_regime": usa_etf["etf_regime"],
            "etf_expected_12m_total_return_pct": usa_etf["expected_12m_total_return_pct"],
            "us_shiller_state": us_composite["valuation_state"],
            "us_shiller_composite_signal": us_composite["composite_signal"],
            "us_shiller_composite_10y_real_pct": us_composite["composite_forecast_real_10y_pct"],
            "combined_signal": combine_market_signal(usa_scores),
        },
        {
            "market": "Europe",
            "proxy_ticker": europe_etf["ticker"],
            "damodaran_forward_pe": float(europe_agg["cap_weighted_forward_pe"]),
            "damodaran_forward_earnings_yield_pct": float(europe_agg["cap_weighted_forward_earnings_yield"] * 100.0),
            "damodaran_total_erp_pct": float(europe_agg["cap_weighted_total_erp"] * 100.0),
            "damodaran_spread_pct": float(europe_agg["valuation_spread"] * 100.0),
            "damodaran_signal": str(europe_agg["damodaran_signal"]),
            "etf_trailing_12m_dy_pct": europe_etf["trailing_12m_dy_pct"],
            "etf_regime": europe_etf["etf_regime"],
            "etf_expected_12m_total_return_pct": europe_etf["expected_12m_total_return_pct"],
            "us_shiller_state": None,
            "us_shiller_composite_signal": None,
            "us_shiller_composite_10y_real_pct": None,
            "combined_signal": combine_market_signal(europe_scores),
        },
    ]

    return pd.DataFrame(rows)


def build_suite(save_outputs: bool = True) -> dict[str, Any]:
    ensure_output_dir()
    base_payload = build_research_payload(save_outputs=False)

    country_snapshot = load_damodaran_snapshot()
    mapping_df = mapping_dataframe()
    mapped_current = merge_country_signals_with_etfs(country_snapshot["all_df"], mapping_df)

    panel, _frames = build_etf_panel(mapping_df)
    panel = add_expanding_regimes(panel)
    backtest_curve, backtest_summary = backtest_high_yield_strategy(panel)

    us_composite = compute_us_composite(base_payload)
    usa_row = country_snapshot["all_df"][country_snapshot["all_df"]["country_name"].eq("United States")].iloc[0]
    europe_agg = compute_europe_aggregate(country_snapshot["all_df"], country_snapshot["spread_q1"], country_snapshot["spread_q3"])

    usa_etf = current_etf_signal(panel, "USA", "SPY")
    europe_etf = current_etf_signal(panel, "Europe", "VGK")
    main_market_overview = build_market_overview(us_composite, usa_row, usa_etf, europe_agg, europe_etf)

    payload = {
        "us_composite": us_composite,
        "backtest_summary": backtest_summary,
        "main_market_overview": main_market_overview,
        "mapped_current": mapped_current,
    }

    artifact_paths: dict[str, str] = {}
    if save_outputs:
        artifact_paths["country_etf_mapping_csv"] = save_csv(mapping_df, "country_etf_mapping.csv")
        artifact_paths["country_signal_with_etf_csv"] = save_csv(mapped_current, "country_signal_with_etf.csv")
        artifact_paths["etf_signal_panel_csv"] = save_csv(panel, "etf_signal_panel.csv")
        artifact_paths["etf_backtest_curve_csv"] = save_csv(backtest_curve, "etf_backtest_curve.csv")
        artifact_paths["etf_backtest_summary_csv"] = save_csv(backtest_summary, "etf_backtest_summary.csv")
        artifact_paths["main_market_overview_csv"] = save_csv(main_market_overview, "main_market_overview.csv")

        summary_payload = {
            "us_composite": us_composite,
            "backtest_summary": backtest_summary.to_dict(orient="records"),
            "main_market_overview": main_market_overview.to_dict(orient="records"),
        }
        artifact_paths["market_signal_suite_json"] = save_json(summary_payload, "market_signal_suite.json")

    payload["artifacts"] = artifact_paths
    return payload


def main() -> None:
    payload = build_suite(save_outputs=True)
    overview = payload["main_market_overview"]
    print("Market signal suite complete.")
    print(overview.to_string(index=False))
    print("")
    print(payload["backtest_summary"].to_string(index=False))
    print("")
    print(json.dumps(payload["artifacts"], indent=2))


if __name__ == "__main__":
    main()
