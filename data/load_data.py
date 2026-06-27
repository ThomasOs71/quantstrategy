"""Data loaders and return panel assembly for Block-1.

All outputs are monthly log returns in EUR perspective unless explicitly noted.
"""

from __future__ import annotations

import csv
import logging
import os
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from data.asset_universe import (
    ASSET_UNIVERSE,
    COMMON_START,
    DataSource,
    get_driver_keys,
    get_tbd_sources,
)

logger = logging.getLogger(__name__)

MSCI_RAW_DIR = Path(__file__).parent / "raw" / "msci"
EURIBOR_3M_SERIES = "IR3TIB01EZM156N"
USD_3M_SERIES = "TB3MS"


def _pad_start_window(start: str | None) -> str:
    """Return a slightly earlier start date to allow first lagged return."""
    if not start:
        return COMMON_START
    start_ts = pd.Timestamp(start)
    return (start_ts - pd.DateOffset(months=2)).strftime("%Y-%m-%d")


def _coerce_series_to_datetime_index(series: pd.Series) -> pd.Series:
    values = pd.Series(pd.to_numeric(series, errors="coerce"), index=series.index)
    values.index = pd.to_datetime(values.index)
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    return values.sort_index()


def _to_month_end(series: pd.Series) -> pd.Series:
    monthly = _coerce_series_to_datetime_index(series).resample("ME").last().dropna()
    return monthly


def load_fred_series(
    series_id: str,
    start: str = COMMON_START,
    end: str | None = None,
    fred_api_key: str | None = None,
) -> pd.Series:
    """Load a FRED series and aggregate to month end."""
    try:
        from fredapi import Fred
    except ImportError as exc:
        raise ImportError("fredapi is required for FRED loading: pip install fredapi") from exc

    api_key = fred_api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        raise ValueError(
            "Missing FRED API key. Pass fred_api_key or set FRED_API_KEY env var."
        )

    fred = Fred(api_key=api_key)
    raw = fred.get_series(series_id, observation_start=start, observation_end=end)
    raw.name = series_id
    return _to_month_end(pd.Series(raw))


def _extract_close_series(raw: pd.DataFrame | pd.Series) -> pd.Series:
    """Normalize yfinance return shape to a 1-D price series."""
    if isinstance(raw, pd.Series):
        return raw
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("yfinance output expected as Series or DataFrame")
    if raw.empty:
        raise ValueError("No yfinance rows returned.")

    close_candidates = ["Close", "Adj Close", "Adj_Close", "close", "adj close"]
    for column in close_candidates:
        if column in raw.columns:
            series = raw[column]
            if isinstance(series, pd.DataFrame):
                if series.shape[1] > 1:
                    raise ValueError(
                        f"Expected a single close column for {column}, got {series.shape[1]}."
                    )
                return series.squeeze()
            return series

    if isinstance(raw.columns, pd.MultiIndex):
        for col in raw.columns:
            if col[1] in close_candidates or col[1].lower() in close_candidates:
                return raw[col].squeeze()

    # Fall back to first numeric column.
    return raw.select_dtypes("number").iloc[:, 0].squeeze()


def load_etf_returns(
    ticker: str,
    start: str = COMMON_START,
    end: str | None = None,
) -> pd.Series:
    """Load adjusted monthly log returns for a yfinance ticker."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required for ETF loading: pip install yfinance") from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1mo",
            auto_adjust=True,
            progress=False,
        )

    prices = _extract_close_series(raw)
    prices = _coerce_series_to_datetime_index(prices)
    prices.index = prices.index.to_period("M").to_timestamp("M")
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


def _parse_date_token(value: object) -> pd.Timestamp:
    if value is None:
        return pd.NaT
    token = str(value).strip().strip('"').strip("'")
    if not token:
        return pd.NaT

    # Common MSCI date formats used in exports.
    formats = (
        "%b-%y",
        "%b-%Y",
        "%Y-%m",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d.%m.%Y",
    )
    for fmt in formats:
        parsed = pd.to_datetime(token, format=fmt, errors="coerce")
        if not pd.isna(parsed):
            return parsed
    parsed = pd.to_datetime(token, dayfirst=False, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(token, dayfirst=True, errors="coerce")
    return parsed


def _parse_float_token(value: object) -> float:
    if value is None:
        return np.nan
    token = str(value).strip()
    if not token:
        return np.nan
    token = token.replace("\u00a0", "").replace(" ", "")
    token = token.replace("N/A", "").replace("NA", "")
    # Keep sign, digits and decimal separators.
    # Convert european thousands separators while preserving decimals:
    if "," in token and "." in token:
        token = token.replace(",", "")
    elif "," in token and "." not in token:
        token = token.replace(",", ".")
    token = re.sub(r"[^0-9.+-eE]", "", token)
    try:
        return float(token)
    except ValueError:
        return np.nan


def _parse_msci_rows(rows: Iterable[list[str]]) -> pd.DataFrame:
    parsed: list[tuple[pd.Timestamp, float]] = []
    for row in rows:
        if len(row) < 2:
            continue

        date = _parse_date_token(row[0])
        if pd.isna(date):
            continue
        value = _parse_float_token(row[1])
        if pd.isna(value):
            continue
        parsed.append((date, value))

    if not parsed:
        raise ValueError("MSCI CSV contains no valid date/value rows.")

    df = pd.DataFrame(parsed, columns=["date", "index_level"])
    return df.sort_values("date").drop_duplicates("date")


def load_msci_csv(
    filename: str,
    start: str = COMMON_START,
) -> pd.Series:
    """Load monthly MSCI index CSVs exported from app2.msci.com."""
    filepath = MSCI_RAW_DIR / filename
    if not filepath.exists():
        raise FileNotFoundError(
            f"Missing MSCI file: {filepath}\n"
            "Download this file manually and place it under data/raw/msci/. "
            "Raw files are ignored in git and never checked in."
        )

    with filepath.open("r", encoding="utf-8", errors="ignore") as handle:
        reader = csv.reader(handle)
        rows = [row for row in reader]

    frame = _parse_msci_rows(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    frame.index = frame["date"].dt.to_period("M").dt.to_timestamp("M")

    prices = pd.Series(frame["index_level"].to_numpy(dtype=float), index=frame.index)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns = log_returns[log_returns.index >= pd.Timestamp(start)]
    log_returns.name = filename
    return log_returns


def apply_eurusd_hedge_formula(
    usd_log_returns: pd.Series,
    euribor_3m_annual: pd.Series,
    usd_3m_annual: pd.Series,
) -> pd.Series:
    """Apply approximate FX-hedged return formula in log-return space.

    r_hedged_t = r_usd_t + (euribor_3m_t - usd_3m_t) / 12 / 100
    """
    common_idx = (
        usd_log_returns.index.intersection(euribor_3m_annual.index).intersection(
            usd_3m_annual.index
        )
    )
    carry = (euribor_3m_annual.loc[common_idx] - usd_3m_annual.loc[common_idx]) / 12 / 100
    hedged = usd_log_returns.loc[common_idx] + carry
    base_name = usd_log_returns.name or "usd_asset"
    hedged.name = base_name + "_eur_hedged"
    return hedged


def convert_usd_to_eur(
    usd_log_returns: pd.Series,
    eurusd_log_returns: pd.Series,
) -> pd.Series:
    """Convert USD log returns to EUR with the DEXUSEU convention.

    DEXUSEU is USD per 1 EUR, so the EUR-return is:
        r_eur_t = r_usd_t - r_eurusd_t
    """
    common_idx = usd_log_returns.index.intersection(eurusd_log_returns.index)
    r_eur = usd_log_returns.loc[common_idx] - eurusd_log_returns.loc[common_idx]
    base_name = usd_log_returns.name or "usd_asset"
    r_eur.name = base_name + "_eur"
    return r_eur


def build_return_panel(
    start: str = COMMON_START,
    end: str | None = None,
    fred_api_key: str | None = None,
    warn_tbd: bool = True,
    strict_no_nan: bool = False,
) -> pd.DataFrame:
    """Build the monthly log-return panel with all 12 driver series."""
    if warn_tbd:
        tbd = get_tbd_sources()
        if tbd:
            logger.warning(
                "Source IDs still marked tbd: %s. Verify before production.",
                ", ".join(tbd),
            )

    panel_start = pd.Timestamp(start)
    series_start = _pad_start_window(start)
    panel_end = pd.Timestamp(end) if end else None
    series: dict[str, pd.Series] = {}

    logger.info("Loading FRED helper series...")
    euribor_3m = load_fred_series(
        EURIBOR_3M_SERIES,
        start=series_start,
        end=end,
        fred_api_key=fred_api_key,
    )
    usd_3m = load_fred_series(
        USD_3M_SERIES,
        start=series_start,
        end=end,
        fred_api_key=fred_api_key,
    )
    eurusd_raw = load_fred_series(
        "DEXUSEU",
        start=series_start,
        end=end,
        fred_api_key=fred_api_key,
    )
    eurusd_lr = np.log(eurusd_raw / eurusd_raw.shift(1)).dropna()
    series["fx_eurusd"] = eurusd_lr

    logger.info("Loading MSCI index returns from CSV...")
    euro_equities_lr = load_msci_csv("msci_emu_ntr_usd.csv", start=series_start)
    global_dm_ex_emu_lr = load_msci_csv(
        "msci_world_ex_emu_ntr_usd.csv", start=series_start
    )
    series["euro_equities"] = convert_usd_to_eur(euro_equities_lr, eurusd_lr)
    series["global_dm_ex_emu"] = convert_usd_to_eur(global_dm_ex_emu_lr, eurusd_lr)

    logger.info("Loading USD-exposed assets...")
    for key in ("em_equities", "gold", "commodities"):
        asset = ASSET_UNIVERSE[key]
        if asset.source == DataSource.FRED:
            raw_prices = load_fred_series(
                asset.source_id,
                start=series_start,
                end=end,
                fred_api_key=fred_api_key,
            )
            usd_lr = np.log(raw_prices / raw_prices.shift(1)).dropna()
        else:
            if asset.source != DataSource.YFINANCE:
                raise ValueError(
                    f"Unsupported source for USD-exposed asset {key}: {asset.source}"
                )
            usd_lr = load_etf_returns(asset.source_id, start=series_start, end=end)
        series[key] = convert_usd_to_eur(usd_lr, eurusd_lr)

    logger.info("Loading EUR-native ETF assets...")
    for key in ("euro_govt_bond_7_10", "euro_ig_credit", "euro_high_yield"):
        asset = ASSET_UNIVERSE[key]
        if asset.source != DataSource.YFINANCE:
            raise ValueError(
                f"Unsupported source for EUR-native asset {key}: {asset.source}"
            )
        series[key] = load_etf_returns(
            asset.source_id, start=series_start, end=end
        )

    logger.info("Applying EUR carry approximation for hedged fixed-income proxies...")
    for key in ("global_govt_bond_eur_hedged", "em_hc_bond_eur_hedged"):
        asset = ASSET_UNIVERSE[key]
        if asset.source != DataSource.YFINANCE_HEDGE:
            raise ValueError(
                f"Unsupported source for hedged asset {key}: {asset.source}"
            )
        usd_lr = load_etf_returns(asset.source_id, start=series_start, end=end)
        series[key] = apply_eurusd_hedge_formula(usd_lr, euribor_3m, usd_3m)

    logger.info("Converting and adding cash return series...")
    cash_rate = euribor_3m.copy()
    series["cash"] = np.log(1 + cash_rate / 100 / 12)

    ordered_keys = get_driver_keys()
    missing = [key for key in ordered_keys if key not in series]
    if missing:
        raise ValueError(f"Missing series for keys: {missing}")

    panel = pd.DataFrame({key: series[key] for key in ordered_keys})
    panel = panel[panel.index >= panel_start]
    if panel_end is not None:
        panel = panel[panel.index <= panel_end]

    if panel.empty:
        raise ValueError("Built panel is empty. Check sources and date range.")

    nan_counts = panel.isna().sum()
    if nan_counts.any():
        nan_summary = nan_counts[nan_counts > 0]
        logger.warning("Panel contains NaN values after merge:\n%s", nan_summary)
        if strict_no_nan:
            raise ValueError(
                "Panel contains NaN values. Set strict_no_nan=False to allow warnings."
            )

    logger.info(
        "Build complete: shape=%s, from=%s to=%s",
        panel.shape,
        panel.index[0].date(),
        panel.index[-1].date(),
    )
    return panel
