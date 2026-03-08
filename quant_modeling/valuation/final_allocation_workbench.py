from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from damodaran_enhanced_relative_model import build_enhanced_model
from enhanced_allocation_extension import aggregate_sector_scores, combine_actions
from market_signal_suite import annualize_from_monthly, build_suite, download_monthly_etf_frame, max_drawdown


OUTPUT_DIR = Path(__file__).resolve().parent / "research_outputs"
AS_OF_DATE = "2026-02-28"
HISTORICAL_START = "2005-01-01"
REGION_BENCHMARK_WEIGHTS = {"us": 0.60, "europe": 0.40}
REGION_MULTIPLIERS = {"underweight": 0.75, "neutral": 1.00, "overweight": 1.20}
ACTION_MULTIPLIERS = {
    "high_conviction_underweight": 0.55,
    "underweight": 0.75,
    "relative_underweight": 0.90,
    "market_weight": 1.00,
    "relative_overweight": 1.15,
    "overweight": 1.30,
    "high_conviction_overweight": 1.45,
}
REBALANCE_MONTHS = 3
MIN_SELECTED_NAMES = 4
MAX_TURNOVER_PER_REBALANCE = 0.35
TRANSACTION_COST_RATE = 0.0010
MAX_POSITION_WEIGHT = 0.25

SIGNAL_SCORE = {
    "high_conviction_underweight": -1.0,
    "underweight": -0.6,
    "relative_underweight": -0.3,
    "market_weight": 0.0,
    "neutral": 0.0,
    "relative_overweight": 0.3,
    "overweight": 0.6,
    "high_conviction_overweight": 1.0,
}

SECTOR_ETF_MAP: dict[tuple[str, str], dict[str, str]] = {
    ("us", "Materials"): {"proxy_ticker": "XLB", "proxy_name": "Materials Select Sector SPDR", "precision": "direct", "proxy_family": "US sector ETF"},
    ("us", "Communication Services"): {"proxy_ticker": "VOX", "proxy_name": "Vanguard Communication Services ETF", "precision": "direct", "proxy_family": "US sector ETF"},
    ("us", "Energy"): {"proxy_ticker": "XLE", "proxy_name": "Energy Select Sector SPDR", "precision": "direct", "proxy_family": "US sector ETF"},
    ("us", "Financials"): {"proxy_ticker": "XLF", "proxy_name": "Financial Select Sector SPDR", "precision": "direct", "proxy_family": "US sector ETF"},
    ("us", "Industrials"): {"proxy_ticker": "XLI", "proxy_name": "Industrial Select Sector SPDR", "precision": "direct", "proxy_family": "US sector ETF"},
    ("us", "Technology"): {"proxy_ticker": "XLK", "proxy_name": "Technology Select Sector SPDR", "precision": "direct", "proxy_family": "US sector ETF"},
    ("us", "Consumer Staples"): {"proxy_ticker": "XLP", "proxy_name": "Consumer Staples Select Sector SPDR", "precision": "direct", "proxy_family": "US sector ETF"},
    ("us", "Real Estate"): {"proxy_ticker": "VNQ", "proxy_name": "Vanguard Real Estate ETF", "precision": "direct", "proxy_family": "US sector ETF"},
    ("us", "Utilities"): {"proxy_ticker": "XLU", "proxy_name": "Utilities Select Sector SPDR", "precision": "direct", "proxy_family": "US sector ETF"},
    ("us", "Health Care"): {"proxy_ticker": "XLV", "proxy_name": "Health Care Select Sector SPDR", "precision": "direct", "proxy_family": "US sector ETF"},
    ("us", "Consumer Discretionary"): {"proxy_ticker": "XLY", "proxy_name": "Consumer Discretionary Select Sector SPDR", "precision": "direct", "proxy_family": "US sector ETF"},
    ("europe", "Financials"): {"proxy_ticker": "EUFN", "proxy_name": "iShares MSCI Europe Financials ETF", "precision": "direct", "proxy_family": "US-listed Europe sector ETF"},
    ("europe", "Materials"): {"proxy_ticker": "EXV6.DE", "proxy_name": "iShares STOXX Europe 600 Basic Resources UCITS ETF", "precision": "direct", "proxy_family": "Europe UCITS sector ETF"},
    ("europe", "Communication Services"): {"proxy_ticker": "EXV2.DE", "proxy_name": "iShares STOXX Europe 600 Telecommunications UCITS ETF", "precision": "direct", "proxy_family": "Europe UCITS sector ETF"},
    ("europe", "Energy"): {"proxy_ticker": "EXH1.DE", "proxy_name": "iShares STOXX Europe 600 Oil & Gas UCITS ETF", "precision": "direct", "proxy_family": "Europe UCITS sector ETF"},
    ("europe", "Industrials"): {"proxy_ticker": "EXH4.DE", "proxy_name": "iShares STOXX Europe 600 Industrial Goods & Services UCITS ETF", "precision": "direct", "proxy_family": "Europe UCITS sector ETF"},
    ("europe", "Technology"): {"proxy_ticker": "EXV3.DE", "proxy_name": "iShares STOXX Europe 600 Technology UCITS ETF", "precision": "direct", "proxy_family": "Europe UCITS sector ETF"},
    ("europe", "Consumer Staples"): {"proxy_ticker": "EXH3.DE", "proxy_name": "iShares STOXX Europe 600 Food & Beverage UCITS ETF", "precision": "direct_proxy", "proxy_family": "Europe UCITS sector ETF"},
    ("europe", "Real Estate"): {"proxy_ticker": "IPRP.AS", "proxy_name": "iShares European Property Yield UCITS ETF", "precision": "direct", "proxy_family": "Europe property ETF"},
    ("europe", "Utilities"): {"proxy_ticker": "EXH9.DE", "proxy_name": "iShares STOXX Europe 600 Utilities UCITS ETF", "precision": "direct", "proxy_family": "Europe UCITS sector ETF"},
    ("europe", "Health Care"): {"proxy_ticker": "EXV4.DE", "proxy_name": "iShares STOXX Europe 600 Health Care UCITS ETF", "precision": "direct", "proxy_family": "Europe UCITS sector ETF"},
    ("europe", "Consumer Discretionary"): {"proxy_ticker": "EXH7.DE", "proxy_name": "iShares STOXX Europe 600 Personal & Household Goods UCITS ETF", "precision": "direct_proxy", "proxy_family": "Europe UCITS sector ETF"},
}

BACKTEST_TICKERS = {
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


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _cross_sectional_z(series: pd.Series) -> pd.Series:
    values = _coerce_numeric(series)
    mean = values.mean()
    std = values.std(ddof=0)
    if pd.isna(std) or math.isclose(float(std), 0.0):
        return pd.Series(0.0, index=values.index)
    return (values - mean) / std


def _classify_current_regime(current_value: float, history: pd.Series) -> tuple[str | None, float, float]:
    valid = _coerce_numeric(history).dropna()
    if len(valid) < 24 or pd.isna(current_value):
        return None, math.nan, math.nan
    q1 = float(valid.quantile(0.25))
    q3 = float(valid.quantile(0.75))
    if current_value >= q3:
        return "high", q1, q3
    if current_value <= q1:
        return "low", q1, q3
    return "middle", q1, q3


def _infer_region_key(market_name: str) -> str:
    return "us" if str(market_name).upper() == "USA" else "europe"


def _proxy_info(region: str, sector: str) -> dict[str, str]:
    info = SECTOR_ETF_MAP.get((region, sector))
    if info is not None:
        return dict(info)
    fallback_ticker = "SPY" if region == "us" else "VGK"
    return {
        "proxy_ticker": fallback_ticker,
        "proxy_name": "Fallback broad ETF",
        "precision": "fallback",
        "proxy_family": "broad market ETF",
    }


def build_region_target_table(market_overview: pd.DataFrame) -> pd.DataFrame:
    working = market_overview.copy()
    working["region"] = working["market"].map(_infer_region_key)
    working["benchmark_weight"] = working["region"].map(REGION_BENCHMARK_WEIGHTS)
    working["region_multiplier"] = working["combined_signal"].map(REGION_MULTIPLIERS).fillna(1.0)
    working["raw_target_weight"] = working["benchmark_weight"] * working["region_multiplier"]

    total_raw = float(working["raw_target_weight"].sum())
    if total_raw > 1.0 and not math.isclose(total_raw, 0.0):
        working["target_weight"] = working["raw_target_weight"] / total_raw
        cash_weight = 0.0
    else:
        working["target_weight"] = working["raw_target_weight"]
        cash_weight = max(0.0, 1.0 - total_raw)

    working["active_weight"] = working["target_weight"] - working["benchmark_weight"]
    working["target_weight_pct"] = working["target_weight"] * 100.0
    working["active_weight_pct"] = working["active_weight"] * 100.0

    keep = [
        "region",
        "market",
        "combined_signal",
        "benchmark_weight",
        "region_multiplier",
        "target_weight",
        "active_weight",
        "target_weight_pct",
        "active_weight_pct",
        "damodaran_forward_pe",
        "damodaran_spread_pct",
        "etf_trailing_12m_dy_pct",
        "etf_regime",
        "etf_expected_12m_total_return_pct",
    ]
    region_table = working.loc[:, keep].sort_values("target_weight", ascending=False).reset_index(drop=True)

    cash_row = {
        "region": "cash",
        "market": "Cash",
        "combined_signal": "reserve",
        "benchmark_weight": 0.0,
        "region_multiplier": math.nan,
        "target_weight": cash_weight,
        "active_weight": cash_weight,
        "target_weight_pct": cash_weight * 100.0,
        "active_weight_pct": cash_weight * 100.0,
        "damodaran_forward_pe": math.nan,
        "damodaran_spread_pct": math.nan,
        "etf_trailing_12m_dy_pct": math.nan,
        "etf_regime": None,
        "etf_expected_12m_total_return_pct": math.nan,
    }
    region_table = pd.concat([region_table, pd.DataFrame([cash_row])], ignore_index=True)
    return region_table


def build_sector_action_table(sector_scores: pd.DataFrame, region_targets: pd.DataFrame) -> pd.DataFrame:
    live_regions = region_targets[region_targets["region"].isin(["us", "europe"])].copy()
    region_action_map = live_regions.set_index("region")["combined_signal"].to_dict()

    rows: list[dict[str, Any]] = []
    for row in sector_scores.itertuples(index=False):
        info = _proxy_info(str(row.region), str(row.sector))
        portfolio_action = combine_actions(region_action_map.get(str(row.region), "neutral"), str(row.sector_action))
        rows.append(
            {
                "region": row.region,
                "sector": row.sector,
                "n_industries": int(row.n_industries),
                "total_firms": float(row.total_firms),
                "sector_score": float(row.sector_score),
                "raw_sector_score": float(row.raw_sector_score),
                "adjusted_sector_score": float(row.adjusted_sector_score),
                "regression_sector_score": float(row.regression_sector_score),
                "quality_overlay_delta": float(row.quality_overlay_delta),
                "valuation_regime": row.valuation_regime,
                "sector_action": row.sector_action,
                "region_action": region_action_map.get(str(row.region), "neutral"),
                "portfolio_action": portfolio_action,
                "action_multiplier": ACTION_MULTIPLIERS.get(portfolio_action, 1.0),
                "proxy_ticker": info["proxy_ticker"],
                "proxy_name": info["proxy_name"],
                "proxy_precision": info["precision"],
                "proxy_family": info["proxy_family"],
            }
        )

    action_table = pd.DataFrame(rows)
    action_table = action_table.sort_values(["region", "sector_score"], ascending=[True, False]).reset_index(drop=True)
    return action_table


def build_current_etf_diagnostics(sector_actions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    unique_pairs = sector_actions.loc[:, ["region", "sector", "proxy_ticker", "proxy_name", "proxy_precision", "proxy_family"]].drop_duplicates()

    for row in unique_pairs.itertuples(index=False):
        try:
            frame = download_monthly_etf_frame(str(row.proxy_ticker))
        except Exception:
            rows.append(
                {
                    "region": row.region,
                    "sector": row.sector,
                    "proxy_ticker": row.proxy_ticker,
                    "proxy_name": row.proxy_name,
                    "proxy_precision": row.proxy_precision,
                    "proxy_family": row.proxy_family,
                    "history_months": 0,
                    "last_date": None,
                    "price": math.nan,
                    "trailing_12m_dy_pct": math.nan,
                    "dy_q1_pct": math.nan,
                    "dy_q3_pct": math.nan,
                    "dy_regime": None,
                    "expected_12m_total_return_pct": math.nan,
                    "data_status": "download_failed",
                    "proxy_reference_url": f"https://finance.yahoo.com/quote/{row.proxy_ticker}",
                }
            )
            continue

        latest = frame.iloc[-1]
        current_dy = float(latest["trailing_12m_dy"]) if pd.notna(latest["trailing_12m_dy"]) else math.nan
        regime, q1, q3 = _classify_current_regime(current_dy, frame["trailing_12m_dy"].iloc[:-1])
        history_months = int(frame["trailing_12m_dy"].notna().sum())
        same_regime = frame.iloc[:-1].copy()
        if regime is not None:
            labels: list[str | None] = []
            values = same_regime["trailing_12m_dy"].tolist()
            for pos, value in enumerate(values):
                past = pd.Series(values[:pos])
                label, _, _ = _classify_current_regime(float(value) if pd.notna(value) else math.nan, past)
                labels.append(label)
            same_regime["historical_regime"] = labels
            same_regime = same_regime[
                same_regime["historical_regime"].eq(regime)
                & same_regime["fwd_12m_total_return_ann"].notna()
            ]
        expected = float(same_regime["fwd_12m_total_return_ann"].mean()) if not same_regime.empty else math.nan

        rows.append(
            {
                "region": row.region,
                "sector": row.sector,
                "proxy_ticker": row.proxy_ticker,
                "proxy_name": row.proxy_name,
                "proxy_precision": row.proxy_precision,
                "proxy_family": row.proxy_family,
                "history_months": history_months,
                "last_date": str(pd.Timestamp(frame.index[-1]).date()),
                "price": float(latest["price"]),
                "trailing_12m_dy_pct": current_dy * 100.0 if not math.isnan(current_dy) else math.nan,
                "dy_q1_pct": q1 * 100.0 if not math.isnan(q1) else math.nan,
                "dy_q3_pct": q3 * 100.0 if not math.isnan(q3) else math.nan,
                "dy_regime": regime,
                "expected_12m_total_return_pct": expected * 100.0 if not math.isnan(expected) else math.nan,
                "data_status": "ok",
                "proxy_reference_url": f"https://finance.yahoo.com/quote/{row.proxy_ticker}",
            }
        )

    diagnostics = pd.DataFrame(rows)
    diagnostics = diagnostics.sort_values(["region", "sector"]).reset_index(drop=True)
    return diagnostics


def build_sector_target_weights(
    sector_actions: pd.DataFrame,
    region_targets: pd.DataFrame,
    etf_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    live_regions = region_targets[region_targets["region"].isin(["us", "europe"])].copy()
    region_weight_map = live_regions.set_index("region")["target_weight"].to_dict()
    region_benchmark_map = live_regions.set_index("region")["benchmark_weight"].to_dict()

    working = sector_actions.merge(
        etf_diagnostics,
        on=["region", "sector", "proxy_ticker", "proxy_name", "proxy_precision", "proxy_family"],
        how="left",
    )

    rows: list[dict[str, Any]] = []
    for region, subset in working.groupby("region", sort=False):
        region_target = float(region_weight_map.get(region, 0.0))
        region_benchmark = float(region_benchmark_map.get(region, 0.0))
        total_firms = float(subset["total_firms"].sum())
        if math.isclose(total_firms, 0.0):
            neutral_shares = pd.Series(1.0 / len(subset), index=subset.index)
        else:
            neutral_shares = subset["total_firms"] / total_firms

        benchmark_neutral = neutral_shares * region_benchmark
        region_neutral = neutral_shares * region_target
        raw_target = region_neutral * subset["action_multiplier"]
        raw_total = float(raw_target.sum())
        scaled_target = raw_target * (region_target / raw_total) if region_target > 0 and raw_total > 0 else raw_target

        for idx, item in subset.iterrows():
            final_weight = float(scaled_target.loc[idx])
            benchmark_weight = float(benchmark_neutral.loc[idx])
            neutral_weight = float(region_neutral.loc[idx])
            rows.append(
                {
                    **item.to_dict(),
                    "region_benchmark_weight": region_benchmark,
                    "region_target_weight": region_target,
                    "sector_neutral_share": float(neutral_shares.loc[idx]),
                    "benchmark_neutral_weight": benchmark_weight,
                    "region_neutral_weight": neutral_weight,
                    "target_weight": final_weight,
                    "target_weight_pct": final_weight * 100.0,
                    "active_weight_vs_benchmark": final_weight - benchmark_weight,
                    "active_weight_vs_benchmark_pct": (final_weight - benchmark_weight) * 100.0,
                    "active_weight_vs_region_neutral": final_weight - neutral_weight,
                    "active_weight_vs_region_neutral_pct": (final_weight - neutral_weight) * 100.0,
                    "max_position_breach": bool(final_weight > MAX_POSITION_WEIGHT),
                    "portfolio_signal_score": SIGNAL_SCORE.get(str(item["portfolio_action"]), 0.0),
                }
            )

    target_table = pd.DataFrame(rows)
    target_table["rank_within_region"] = (
        target_table.groupby("region")["target_weight"].rank(method="first", ascending=False).astype(int)
    )
    target_table = target_table.sort_values(["region", "target_weight"], ascending=[True, False]).reset_index(drop=True)
    return target_table


def build_final_recommendation_table(region_targets: pd.DataFrame, sector_targets: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    live_regions = region_targets[region_targets["region"].isin(["us", "europe"])].copy()
    for region_row in live_regions.itertuples(index=False):
        region_slice = sector_targets[sector_targets["region"].eq(region_row.region)].copy()
        top = region_slice.head(3)
        bottom = region_slice.tail(3).sort_values("target_weight", ascending=True)
        for item in pd.concat([top, bottom], ignore_index=True).itertuples(index=False):
            rows.append(
                {
                    "region": item.region,
                    "market_signal": region_row.combined_signal,
                    "sector": item.sector,
                    "portfolio_action": item.portfolio_action,
                    "proxy_ticker": item.proxy_ticker,
                    "proxy_name": item.proxy_name,
                    "target_weight_pct": item.target_weight_pct,
                    "active_weight_vs_benchmark_pct": item.active_weight_vs_benchmark_pct,
                    "sector_score": item.sector_score,
                    "quality_overlay_delta": item.quality_overlay_delta,
                    "dy_regime": item.dy_regime,
                    "expected_12m_total_return_pct": item.expected_12m_total_return_pct,
                }
            )

    final_table = pd.DataFrame(rows).drop_duplicates(subset=["region", "sector"], keep="first")
    final_table = final_table.sort_values(["region", "target_weight_pct"], ascending=[True, False]).reset_index(drop=True)
    return final_table


def _build_backtest_feature_panel() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sector, ticker in BACKTEST_TICKERS.items():
        try:
            frame = download_monthly_etf_frame(ticker)
        except Exception:
            continue

        if frame.empty:
            continue

        work = frame.copy().sort_index()
        work = work[work.index >= pd.Timestamp(HISTORICAL_START)]
        work["monthly_total_return"] = work["adj_close"].pct_change()
        work["div_growth_12m"] = work["trailing_12m_div"] / work["trailing_12m_div"].shift(12) - 1.0
        work["price_momentum_12m"] = work["adj_close"] / work["adj_close"].shift(12) - 1.0
        work["vol_12m"] = work["monthly_total_return"].rolling(12).std(ddof=0) * math.sqrt(12.0)

        use = work.loc[:, ["trailing_12m_dy", "div_growth_12m", "price_momentum_12m", "vol_12m", "next_1m_total_return"]].copy()
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
        raise ValueError("No data available for the constrained proxy backtest.")
    return panel.sort_values(["date", "ticker"]).reset_index(drop=True)


def build_backtest_signal_panel() -> pd.DataFrame:
    panel = _build_backtest_feature_panel()
    panel["raw_score"] = np.nan
    panel["enhanced_score"] = np.nan

    for _date, idx in panel.groupby("date").groups.items():
        slice_df = panel.loc[idx].copy()
        panel.loc[idx, "raw_score"] = _cross_sectional_z(slice_df["dy_12m"]).to_numpy(dtype=float)
        panel.loc[idx, "enhanced_score"] = (
            _cross_sectional_z(slice_df["dy_12m"])
            + 0.35 * _cross_sectional_z(slice_df["div_growth_12m"])
            + 0.25 * _cross_sectional_z(slice_df["price_momentum_12m"])
            - 0.35 * _cross_sectional_z(slice_df["vol_12m"])
        ).to_numpy(dtype=float)

    return panel


def _is_rebalance_index(position: int) -> bool:
    return position % REBALANCE_MONTHS == 0


def _top_equal_target(slice_df: pd.DataFrame, score_column: str) -> pd.Series:
    available = slice_df.dropna(subset=[score_column]).sort_values(score_column, ascending=False).copy()
    names = available["ticker"].tolist()
    if not names:
        return pd.Series(dtype=float)

    count = max(MIN_SELECTED_NAMES, math.ceil(len(names) / 4))
    count = min(count, len(names))
    selected = names[:count]
    weight = min(1.0 / count, MAX_POSITION_WEIGHT)

    target = pd.Series(0.0, index=names, dtype=float)
    target.loc[selected] = weight
    allocated = float(target.sum())
    if allocated < 1.0:
        remainder = 1.0 - allocated
        for ticker in names[count:]:
            if remainder <= 0:
                break
            add = min(MAX_POSITION_WEIGHT, remainder)
            target.loc[ticker] = add
            remainder -= add
        if remainder > 1e-12:
            target.loc[selected] = target.loc[selected] + remainder / len(selected)
    target = target / target.sum()
    return target


def _equal_weight_target(slice_df: pd.DataFrame) -> pd.Series:
    names = slice_df["ticker"].tolist()
    if not names:
        return pd.Series(dtype=float)
    return pd.Series(1.0 / len(names), index=names, dtype=float)


def _apply_turnover_cap(current: pd.Series, target: pd.Series) -> tuple[pd.Series, float]:
    all_names = sorted(set(current.index) | set(target.index))
    current_full = current.reindex(all_names).fillna(0.0)
    target_full = target.reindex(all_names).fillna(0.0)
    diff = target_full - current_full
    turnover = float(np.abs(diff).sum() / 2.0)
    if turnover <= MAX_TURNOVER_PER_REBALANCE or math.isclose(turnover, 0.0):
        return target_full, turnover
    scale = MAX_TURNOVER_PER_REBALANCE / turnover
    adjusted = current_full + scale * diff
    adjusted = adjusted.clip(lower=0.0)
    if float(adjusted.sum()) > 0:
        adjusted = adjusted / adjusted.sum()
    actual_turnover = float(np.abs(adjusted - current_full).sum() / 2.0)
    return adjusted, actual_turnover


def _simulate_strategy(panel: pd.DataFrame, score_column: str, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(panel["date"].unique())
    weights = pd.Series(dtype=float)
    monthly_rows: list[dict[str, Any]] = []
    rebalance_rows: list[dict[str, Any]] = []

    for pos, date in enumerate(dates):
        slice_df = panel[panel["date"].eq(date)].copy()
        slice_df = slice_df[slice_df["next_1m_total_return"].notna() & slice_df["ticker"].notna()].copy()
        if slice_df.empty:
            continue

        names = slice_df["ticker"].tolist()
        current = weights.reindex(names).fillna(0.0)
        current = current / current.sum() if float(current.sum()) > 0 else pd.Series(0.0, index=names, dtype=float)

        rebalance = _is_rebalance_index(pos) or weights.empty
        target = _equal_weight_target(slice_df) if score_column == "benchmark" else _top_equal_target(slice_df, score_column)

        if rebalance:
            trade_weights, turnover = _apply_turnover_cap(current, target)
            cost = turnover * TRANSACTION_COST_RATE
            selected = trade_weights[trade_weights > 1e-8].sort_values(ascending=False)
            rebalance_rows.append(
                {
                    "date": date,
                    "portfolio": label,
                    "turnover": turnover,
                    "transaction_cost": cost,
                    "n_positions": int((selected > 0).sum()),
                    "top_positions": " | ".join(f"{ticker}:{weight:.2%}" for ticker, weight in selected.head(5).items()),
                }
            )
        else:
            trade_weights = current
            turnover = 0.0
            cost = 0.0

        trade_weights = trade_weights.reindex(names).fillna(0.0)
        asset_returns = pd.Series(slice_df["next_1m_total_return"].to_numpy(dtype=float), index=names)
        gross_portfolio_return = float((trade_weights * asset_returns).sum())
        net_portfolio_return = gross_portfolio_return - cost

        growth = trade_weights * (1.0 + asset_returns)
        gross_end_value = float(growth.sum())
        weights = growth / gross_end_value if gross_end_value > 0 else pd.Series(0.0, index=names, dtype=float)

        selected_now = trade_weights[trade_weights > 1e-8].sort_values(ascending=False)
        monthly_rows.append(
            {
                "date": date,
                "portfolio": label,
                "gross_return": gross_portfolio_return,
                "net_return": net_portfolio_return,
                "turnover": turnover,
                "transaction_cost": cost,
                "rebalance_flag": rebalance,
                "n_positions": int((selected_now > 0).sum()),
                "largest_weight": float(selected_now.iloc[0]) if not selected_now.empty else 0.0,
                "selected_tickers": " | ".join(selected_now.index.tolist()),
            }
        )

    monthly = pd.DataFrame(monthly_rows).sort_values("date").reset_index(drop=True)
    monthly["equity_curve"] = (1.0 + monthly["net_return"]).cumprod()
    rebalances = pd.DataFrame(rebalance_rows).sort_values("date").reset_index(drop=True)
    return monthly, rebalances


def summarize_backtests(monthly_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    raw_ann = math.nan

    for label, frame in monthly_frames.items():
        returns = frame["net_return"].dropna()
        annual_return = annualize_from_monthly(returns)
        annual_vol = float(returns.std(ddof=0) * math.sqrt(12.0))
        sharpe = annual_return / annual_vol if annual_vol and not math.isclose(annual_vol, 0.0) else math.nan
        avg_turnover = float(frame.loc[frame["rebalance_flag"], "turnover"].mean()) if frame["rebalance_flag"].any() else 0.0
        avg_cost = float(frame.loc[frame["rebalance_flag"], "transaction_cost"].mean()) if frame["rebalance_flag"].any() else 0.0

        if label == "raw_constrained":
            raw_ann = annual_return

        rows.append(
            {
                "portfolio": label,
                "annual_return": annual_return,
                "annual_vol": annual_vol,
                "sharpe": sharpe,
                "max_drawdown": max_drawdown(frame["equity_curve"]),
                "hit_rate": float((returns > 0).mean()),
                "obs_months": int(len(returns)),
                "avg_positions": float(frame["n_positions"].mean()),
                "avg_rebalance_turnover": avg_turnover,
                "annualized_turnover": avg_turnover * (12.0 / REBALANCE_MONTHS),
                "avg_rebalance_cost": avg_cost,
                "annualized_cost_drag": avg_cost * (12.0 / REBALANCE_MONTHS),
                "max_position_observed": float(frame["largest_weight"].max()),
            }
        )

    summary = pd.DataFrame(rows)
    summary["vs_raw_annual_excess"] = summary["annual_return"] - raw_ann if not math.isnan(raw_ann) else math.nan
    return summary


def run_constrained_backtests() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    panel = build_backtest_signal_panel()

    raw_monthly, raw_rebalances = _simulate_strategy(panel, "raw_score", "raw_constrained")
    enhanced_monthly, enhanced_rebalances = _simulate_strategy(panel, "enhanced_score", "enhanced_constrained")
    benchmark_monthly, benchmark_rebalances = _simulate_strategy(panel, "benchmark", "benchmark_constrained")

    monthly = raw_monthly.merge(
        enhanced_monthly,
        on="date",
        how="outer",
        suffixes=("_raw", "_enhanced"),
    ).merge(
        benchmark_monthly,
        on="date",
        how="outer",
    )
    monthly = monthly.rename(
        columns={
            "portfolio": "portfolio_benchmark",
            "gross_return": "gross_return_benchmark",
            "net_return": "net_return_benchmark",
            "turnover": "turnover_benchmark",
            "transaction_cost": "transaction_cost_benchmark",
            "rebalance_flag": "rebalance_flag_benchmark",
            "n_positions": "n_positions_benchmark",
            "largest_weight": "largest_weight_benchmark",
            "selected_tickers": "selected_tickers_benchmark",
            "equity_curve": "equity_curve_benchmark",
        }
    )
    monthly = monthly.sort_values("date").reset_index(drop=True)

    rebalances = pd.concat([raw_rebalances, enhanced_rebalances, benchmark_rebalances], ignore_index=True)
    rebalances = rebalances.sort_values(["portfolio", "date"]).reset_index(drop=True)

    summary = summarize_backtests(
        {
            "raw_constrained": raw_monthly,
            "enhanced_constrained": enhanced_monthly,
            "benchmark_constrained": benchmark_monthly,
        }
    )
    return monthly, rebalances, summary


def write_final_notes(
    region_targets: pd.DataFrame,
    final_recommendations: pd.DataFrame,
    backtest_summary: pd.DataFrame,
) -> Path:
    live_regions = region_targets[region_targets["region"].isin(["us", "europe"])].copy()
    raw_row = backtest_summary[backtest_summary["portfolio"].eq("raw_constrained")].iloc[0]
    enhanced_row = backtest_summary[backtest_summary["portfolio"].eq("enhanced_constrained")].iloc[0]
    benchmark_row = backtest_summary[backtest_summary["portfolio"].eq("benchmark_constrained")].iloc[0]

    lines = [
        "# Final Allocation Workbench",
        "",
        "## Region Targets",
        "",
    ]
    for row in live_regions.itertuples(index=False):
        lines.append(
            f"- {row.market}: signal={row.combined_signal}, target={row.target_weight_pct:.1f}%, active={row.active_weight_pct:.1f}%, forward PE={row.damodaran_forward_pe:.1f}x, spread={row.damodaran_spread_pct:.2f}%"
        )

    cash = region_targets[region_targets["region"].eq("cash")].iloc[0]
    lines.append(f"- Cash reserve: {cash['target_weight_pct']:.1f}%")
    lines.extend(["", "## Highest-Conviction Sector Positions", ""])

    for row in final_recommendations.head(6).itertuples(index=False):
        lines.append(
            f"- {row.region}/{row.sector}: {row.portfolio_action} via {row.proxy_ticker}, target={row.target_weight_pct:.2f}%, expected 12m={row.expected_12m_total_return_pct:.2f}%"
        )

    lines.extend(
        [
            "",
            "## Constrained Backtest",
            "",
            f"- Raw constrained: {raw_row['annual_return']*100:.2f}% p.a., Sharpe {raw_row['sharpe']:.2f}",
            f"- Enhanced constrained: {enhanced_row['annual_return']*100:.2f}% p.a., Sharpe {enhanced_row['sharpe']:.2f}, excess vs raw {enhanced_row['vs_raw_annual_excess']*100:.2f}%",
            f"- Benchmark constrained: {benchmark_row['annual_return']*100:.2f}% p.a., Sharpe {benchmark_row['sharpe']:.2f}",
            "",
            "The constrained backtest is still a US sector ETF proxy because that is the only clean long history with consistent currency and total-return series in the current setup.",
            "",
        ]
    )

    path = ensure_output_dir() / "final_allocation_workbench_notes.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def build_workbench() -> dict[str, Any]:
    ensure_output_dir()

    market_payload = build_suite(save_outputs=True)
    enhanced_payload = build_enhanced_model()

    market_overview = market_payload["main_market_overview"].copy()
    us_industries = pd.read_csv(enhanced_payload["artifacts"]["us_sector_scores"])
    europe_industries = pd.read_csv(enhanced_payload["artifacts"]["europe_sector_scores"])

    us_sector_scores = aggregate_sector_scores(us_industries, "us")
    europe_sector_scores = aggregate_sector_scores(europe_industries, "europe")
    sector_scores = pd.concat([us_sector_scores, europe_sector_scores], ignore_index=True)

    region_targets = build_region_target_table(market_overview)
    sector_actions = build_sector_action_table(sector_scores, region_targets)
    etf_diagnostics = build_current_etf_diagnostics(sector_actions)
    sector_targets = build_sector_target_weights(sector_actions, region_targets, etf_diagnostics)
    final_recommendations = build_final_recommendation_table(region_targets, sector_targets)

    backtest_monthly, backtest_rebalances, backtest_summary = run_constrained_backtests()

    region_path = ensure_output_dir() / "final_region_target_table.csv"
    sector_scores_path = ensure_output_dir() / "final_sector_score_table.csv"
    diagnostics_path = ensure_output_dir() / "final_sector_etf_diagnostics.csv"
    sector_targets_path = ensure_output_dir() / "final_sector_target_table.csv"
    recommendations_path = ensure_output_dir() / "final_recommendation_table.csv"
    backtest_monthly_path = ensure_output_dir() / "final_constrained_backtest_monthly.csv"
    backtest_rebalances_path = ensure_output_dir() / "final_constrained_backtest_rebalances.csv"
    backtest_summary_path = ensure_output_dir() / "final_constrained_backtest_summary.csv"
    europe_map_path = ensure_output_dir() / "final_europe_sector_proxy_map.csv"

    region_targets.to_csv(region_path, index=False)
    sector_scores.to_csv(sector_scores_path, index=False)
    etf_diagnostics.to_csv(diagnostics_path, index=False)
    sector_targets.to_csv(sector_targets_path, index=False)
    final_recommendations.to_csv(recommendations_path, index=False)
    backtest_monthly.to_csv(backtest_monthly_path, index=False)
    backtest_rebalances.to_csv(backtest_rebalances_path, index=False)
    backtest_summary.to_csv(backtest_summary_path, index=False)

    europe_proxy_map = pd.DataFrame(
        [
            {
                "region": "europe",
                "sector": sector,
                "proxy_ticker": info["proxy_ticker"],
                "proxy_name": info["proxy_name"],
                "proxy_precision": info["precision"],
                "proxy_family": info["proxy_family"],
                "proxy_reference_url": f"https://finance.yahoo.com/quote/{info['proxy_ticker']}",
            }
            for (region, sector), info in SECTOR_ETF_MAP.items()
            if region == "europe"
        ]
    ).sort_values("sector")
    europe_proxy_map.to_csv(europe_map_path, index=False)

    notes_path = write_final_notes(region_targets, final_recommendations, backtest_summary)

    payload = {
        "as_of": AS_OF_DATE,
        "artifacts": {
            "region_targets_csv": str(region_path),
            "sector_scores_csv": str(sector_scores_path),
            "sector_etf_diagnostics_csv": str(diagnostics_path),
            "sector_targets_csv": str(sector_targets_path),
            "final_recommendations_csv": str(recommendations_path),
            "backtest_monthly_csv": str(backtest_monthly_path),
            "backtest_rebalances_csv": str(backtest_rebalances_path),
            "backtest_summary_csv": str(backtest_summary_path),
            "europe_proxy_map_csv": str(europe_map_path),
            "notes_md": str(notes_path),
        },
        "region_targets": region_targets.to_dict(orient="records"),
        "top_recommendations": final_recommendations.head(8).to_dict(orient="records"),
        "backtest_summary": backtest_summary.to_dict(orient="records"),
    }

    payload_path = ensure_output_dir() / "final_allocation_workbench.json"
    payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["artifacts"]["payload_json"] = str(payload_path)
    return payload


def main() -> None:
    payload = build_workbench()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
