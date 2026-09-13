"""Frequency-neutral ingestion for the no-purchase Block-1 research profile.

``daily_proxy_2011`` consumes daily levels and rate observations, selects
period-end levels, and only then calculates log returns.  The published monthly
implementation remains isolated in :mod:`data.load_data` as ``monthly_legacy``.
"""

from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata as importlib_metadata
import logging
import time
import warnings
from typing import Any

import numpy as np
import pandas as pd

from data.asset_universe import get_driver_keys
from data.panel_profiles import (
    DAILY_PROXY_2011_ISHARES_PERFORMANCE_SERIES,
    DAILY_PROXY_2011_YAHOO_LEVEL_SERIES,
    CurrencyOperation,
    ISharesPerformanceDefinition,
    PanelFrequency,
    RemoteLevelDefinition,
    SourceProfile,
    get_frequency_definition,
)

logger = logging.getLogger(__name__)

EURUSD_DAILY_SERIES = "DEXUSEU"
EUR_CASH_DAILY_SERIES = "ECBDFR"
ISHARES_PRODUCT_DATA_URL = (
    "https://www.ishares.com/varnish-api/uk-retail01-product-data/"
    "product-data/api/v2/get-product-data"
)
FRED_OBSERVATIONS_URL = "https://api.stlouisfed.org/fred/series/observations"
MAX_RATE_STALENESS_DAYS = 10
MAX_LEVEL_STALENESS_DAYS = 4
MAX_FX_STALENESS_DAYS = 4


def _package_version(package: str) -> str:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return "unknown"


def _validate_currency_operation(
    *,
    quote_currency: str,
    target_currency: str,
    operation: CurrencyOperation,
) -> None:
    quote = quote_currency.upper()
    target = target_currency.upper()
    if operation is CurrencyOperation.IDENTITY and quote != target:
        raise ValueError(
            f"identity conversion requires equal currencies, got {quote}->{target}"
        )
    if operation is CurrencyOperation.USD_TO_EUR_SPOT and (
        quote,
        target,
    ) != ("USD", "EUR"):
        raise ValueError(f"{operation.value} requires USD->EUR, got {quote}->{target}")


def _coerce_daily_series(series: pd.Series, *, name: str) -> pd.Series:
    values = pd.Series(pd.to_numeric(series, errors="coerce"), index=series.index)
    index = pd.to_datetime(values.index, errors="coerce")
    if isinstance(index, pd.DatetimeIndex) and index.tz is not None:
        # Preserve the provider's local trading date. Converting to UTC before
        # dropping the timezone can move European midnight stamps one day back.
        index = index.tz_localize(None)
    values.index = index
    values = values[~values.index.isna()].replace([np.inf, -np.inf], np.nan).dropna()
    values.index = values.index.normalize()
    if values.index.has_duplicates:
        raise ValueError(
            f"{name} contains duplicate dates after timezone normalization."
        )
    values = values.sort_index().astype(float)
    values.name = name
    return values


def _load_yfinance_daily_levels(
    definition: RemoteLevelDefinition,
    *,
    start: str,
    end: str | None,
) -> pd.Series:
    """Load adjusted daily levels and verify the exact listing currency."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for ETF loading: pip install yfinance"
        ) from exc

    ticker = yf.Ticker(definition.ticker)
    try:
        observed_currency = str(ticker.fast_info["currency"]).upper()
    except Exception as exc:
        raise ValueError(
            f"Could not verify listing currency for {definition.ticker}; "
            "the daily proxy profile will not infer it from the ticker or fund name."
        ) from exc
    if observed_currency != definition.quote_currency:
        raise ValueError(
            f"Listing currency mismatch for {definition.ticker}: observed "
            f"{observed_currency}, expected {definition.quote_currency}."
        )
    _validate_currency_operation(
        quote_currency=definition.quote_currency,
        target_currency=definition.target_currency,
        operation=definition.currency_operation,
    )

    download_end = None
    if end is not None:
        # yfinance treats ``end`` as exclusive.
        download_end = (pd.Timestamp(end) + pd.Timedelta(days=1)).date().isoformat()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            definition.ticker,
            start=start,
            end=download_end,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
    from data.load_data import _extract_close_series

    levels = _coerce_daily_series(_extract_close_series(raw), name=definition.key)
    if levels.empty or (levels <= 0).any():
        raise ValueError(
            f"No valid positive daily levels returned for {definition.ticker}."
        )
    levels.attrs.update(
        {
            "source_identifier": definition.ticker,
            "observed_currency": observed_currency,
            "quote_currency": definition.quote_currency,
            "target_currency": definition.target_currency,
            "currency_operation": definition.currency_operation.value,
            "value_kind": definition.value_kind,
            "income_treatment": definition.income_treatment,
            "provider": definition.provider,
            "license_class": definition.license_class,
            "first_level_date": definition.first_level_date,
            "known_limitation": definition.known_limitation,
            "requested_start": start,
            "requested_end": end,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "package_version": _package_version("yfinance"),
            "returned_first_observation": levels.index.min().date().isoformat(),
            "returned_last_observation": levels.index.max().date().isoformat(),
        }
    )
    return levels


def _load_ishares_performance_levels(
    definition: ISharesPerformanceDefinition,
    *,
    start: str,
    end: str | None,
) -> pd.Series:
    """Load the public iShares NAV growth series with gross income reinvested."""
    try:
        import requests
    except ImportError as exc:
        raise ImportError("requests is required for iShares loading") from exc

    params = {
        "appSubType": "ISHARES",
        "appType": "PRODUCT_PAGE",
        "component": "performance.chart",
        "locale": "en_GB",
        "portfolioId": str(definition.portfolio_id),
        "targetSite": "ishares-uk",
        "userType": "individual",
        "excludeContent": "true",
        "asOfDate": "",
        "includeConfig": "true",
    }
    response = requests.get(ISHARES_PRODUCT_DATA_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if int(payload.get("productId", -1)) != definition.portfolio_id:
        raise ValueError(
            "iShares response product ID does not match the source contract."
        )
    observed_currency = str(payload.get("currencyCode", "")).upper()
    if observed_currency != definition.quote_currency:
        raise ValueError(
            f"iShares performance currency mismatch: observed {observed_currency!r}, "
            f"expected {definition.quote_currency!r}."
        )
    _validate_currency_operation(
        quote_currency=definition.quote_currency,
        target_currency=definition.target_currency,
        operation=definition.currency_operation,
    )

    try:
        chart = payload["componentsByNameMap"]["performance"]["containersByNameMap"][
            "chart"
        ]["dataPointsByNameMap"]["performanceData"]
        raw_dates = chart["asOfDate"]
        raw_values = chart["value"]
    except (KeyError, TypeError) as exc:
        raise ValueError("iShares response is missing performance chart data.") from exc
    if not isinstance(raw_dates, list) or not isinstance(raw_values, list):
        raise ValueError("iShares performance dates and values must be lists.")
    if len(raw_dates) != len(raw_values) or len(raw_dates) < 2:
        raise ValueError("iShares performance dates and values have invalid lengths.")

    parsed_dates = pd.to_datetime(
        pd.Series(raw_dates, dtype="string"), format="%Y%m%d", errors="coerce"
    )
    levels = _coerce_daily_series(
        pd.Series(raw_values, index=pd.DatetimeIndex(parsed_dates)),
        name=definition.key,
    )
    if levels.empty or (levels <= 0).any():
        raise ValueError("iShares returned no valid positive NAV-performance levels.")
    if levels.index.min() > pd.Timestamp(definition.first_level_date):
        raise ValueError(
            "iShares performance history starts after the documented source floor."
        )

    full_first = levels.index.min().date().isoformat()
    full_last = levels.index.max().date().isoformat()
    levels = levels[levels.index >= pd.Timestamp(start)]
    if end is not None:
        levels = levels[levels.index <= pd.Timestamp(end)]
    if levels.empty:
        raise ValueError(
            "iShares returned no performance levels in the requested window."
        )
    levels.attrs.update(
        {
            "source_identifier": definition.source_identifier,
            "portfolio_id": definition.portfolio_id,
            "observed_currency": observed_currency,
            "quote_currency": definition.quote_currency,
            "target_currency": definition.target_currency,
            "currency_operation": definition.currency_operation.value,
            "value_kind": definition.value_kind,
            "distribution_treatment": definition.distribution_treatment,
            "provider": definition.provider,
            "license_class": definition.license_class,
            "first_level_date": definition.first_level_date,
            "requested_start": start,
            "requested_end": end,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "endpoint": ISHARES_PRODUCT_DATA_URL,
            "returned_first_observation": levels.index.min().date().isoformat(),
            "returned_last_observation": levels.index.max().date().isoformat(),
            "full_history_first_observation": full_first,
            "full_history_last_observation": full_last,
        }
    )
    return levels


def _read_fred_api_key(explicit_key: str | None) -> str:
    import os

    from data.load_data import _read_optional_fred_key

    key = explicit_key or os.environ.get("FRED_API_KEY") or _read_optional_fred_key()
    if not key:
        raise ValueError(
            "Missing FRED API key. Pass fred_api_key, set FRED_API_KEY, or use an "
            "ignored supported FredAPI.txt helper file."
        )
    return key


def _load_fred_daily_observations(
    series_id: str,
    *,
    start: str,
    end: str | None,
    fred_api_key: str | None,
) -> pd.Series:
    try:
        import requests
    except ImportError as exc:
        raise ImportError("requests is required for FRED loading") from exc

    params = {
        "series_id": series_id,
        "observation_start": start,
        "observation_end": end,
        "api_key": _read_fred_api_key(fred_api_key),
        "file_type": "json",
    }
    response = None
    for attempt in range(3):
        try:
            candidate = requests.get(FRED_OBSERVATIONS_URL, params=params, timeout=30)
        except requests.RequestException:
            candidate = None
        if candidate is not None and candidate.status_code == 200:
            response = candidate
            break
        status = candidate.status_code if candidate is not None else "network error"
        retriable = candidate is None or candidate.status_code in {
            429,
            500,
            502,
            503,
            504,
        }
        if not retriable:
            raise RuntimeError(
                f"FRED request failed for {series_id} (HTTP {status})."
            ) from None
        if attempt < 2:
            time.sleep(0.5 * (2**attempt))
    if response is None:
        raise RuntimeError(
            f"FRED request failed for {series_id} after 3 attempts."
        ) from None
    try:
        payload = response.json()
        observations = payload["observations"]
        if not isinstance(observations, list):
            raise TypeError("observations must be a list")
        raw = pd.Series(
            [item.get("value") for item in observations],
            index=[item.get("date") for item in observations],
            name=series_id,
        )
    except (AttributeError, ValueError, KeyError, TypeError):
        raise ValueError(f"FRED returned malformed JSON for {series_id}.") from None
    values = _coerce_daily_series(pd.Series(raw), name=series_id)
    if values.empty:
        raise ValueError(f"FRED returned no observations for {series_id}.")
    values.attrs.update(
        {
            "source_identifier": series_id,
            "requested_start": start,
            "requested_end": end,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "package_version": _package_version("requests"),
            "vintage_policy": "latest_available_at_retrieval",
            "returned_first_observation": values.index.min().date().isoformat(),
            "returned_last_observation": values.index.max().date().isoformat(),
        }
    )
    return values


def _resample_period_end_observations(
    levels: pd.Series,
    *,
    frequency: str | PanelFrequency,
    end: str | pd.Timestamp | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Return period-end levels and their actual selected observation dates."""
    definition = get_frequency_definition(frequency)
    values = _coerce_daily_series(levels, name=levels.name or "levels")
    frame = pd.DataFrame(
        {
            "value": values,
            "observation_date": pd.Series(values.index, index=values.index),
        }
    )
    sampled = frame.resample(definition.resample_rule).last()
    if end is not None:
        sampled = sampled[sampled.index <= pd.Timestamp(end)]

    observation_dates = pd.to_datetime(sampled["observation_date"])
    available = observation_dates.notna()
    staleness = pd.Series(
        sampled.index[available] - pd.DatetimeIndex(observation_dates[available]),
        index=sampled.index[available],
    ).dt.days
    invalid = staleness[(staleness < 0) | (staleness > MAX_LEVEL_STALENESS_DAYS)]
    if len(invalid):
        sample = ", ".join(
            f"{idx.date().isoformat()} ({int(days)}d)"
            for idx, days in invalid.iloc[:5].items()
        )
        raise ValueError(f"Period-end level observations are stale: {sample}")

    period_levels = sampled["value"].astype(float)
    period_levels.name = values.name
    observation_dates.name = "observation_date"
    return period_levels, observation_dates


def resample_period_end_levels(
    levels: pd.Series,
    *,
    frequency: str | PanelFrequency,
    end: str | pd.Timestamp | None = None,
) -> pd.Series:
    """Select the last fresh level in each completed period bucket."""
    period_levels, _ = _resample_period_end_observations(
        levels, frequency=frequency, end=end
    )
    return period_levels


def period_log_returns(
    levels: pd.Series,
    *,
    frequency: str | PanelFrequency,
    end: str | pd.Timestamp | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Aggregate total-return proxy levels first, then calculate log returns."""
    period_levels, observation_dates = _resample_period_end_observations(
        levels, frequency=frequency, end=end
    )
    returns = np.log(period_levels).diff()
    returns.name = levels.name
    return returns, observation_dates


def _asof_rates_at_anchors(
    annual_rate_percent: pd.Series,
    anchors: pd.DatetimeIndex,
    *,
    max_staleness_days: int = MAX_RATE_STALENESS_DAYS,
) -> pd.Series:
    rates = _coerce_daily_series(
        annual_rate_percent, name=annual_rate_percent.name or "annual_rate"
    )
    observation_dates = pd.Series(rates.index, index=rates.index)
    union = rates.index.union(anchors).sort_values()
    aligned_rates = rates.reindex(union).ffill().reindex(anchors)
    aligned_dates = observation_dates.reindex(union).ffill().reindex(anchors)
    missing = aligned_rates[aligned_rates.isna()].index
    if len(missing):
        sample = ", ".join(item.date().isoformat() for item in missing[:5])
        raise ValueError(f"No rate known on or before period anchors: {sample}")
    stale_days = pd.Series(
        anchors - pd.DatetimeIndex(aligned_dates), index=anchors
    ).dt.days
    stale = stale_days[stale_days > max_staleness_days]
    if len(stale):
        sample = ", ".join(
            f"{idx.date().isoformat()} ({int(days)}d)"
            for idx, days in stale.iloc[:5].items()
        )
        raise ValueError(f"Rate observations are stale at period anchors: {sample}")
    return aligned_rates.astype(float)


def convert_usd_levels_to_eur(
    usd_levels: pd.Series,
    eurusd_levels: pd.Series,
) -> pd.Series:
    """Convert daily USD levels using FX known on each asset observation date."""
    usd = _coerce_daily_series(usd_levels, name=usd_levels.name or "usd_asset")
    fx = _coerce_daily_series(eurusd_levels, name="DEXUSEU")
    if (usd <= 0).any() or (fx <= 0).any():
        raise ValueError("USD asset and EUR/USD levels must be strictly positive.")

    fx_dates = pd.Series(fx.index, index=fx.index)
    union = fx.index.union(usd.index).sort_values()
    aligned_fx = fx.reindex(union).ffill().reindex(usd.index)
    aligned_fx_dates = fx_dates.reindex(union).ffill().reindex(usd.index)
    if aligned_fx.isna().any():
        missing = aligned_fx[aligned_fx.isna()].index
        sample = ", ".join(item.date().isoformat() for item in missing[:5])
        raise ValueError(f"No EUR/USD level known on or before asset dates: {sample}")
    staleness = pd.Series(
        usd.index - pd.DatetimeIndex(aligned_fx_dates), index=usd.index
    ).dt.days
    stale = staleness[staleness > MAX_FX_STALENESS_DAYS]
    if len(stale):
        sample = ", ".join(
            f"{idx.date().isoformat()} ({int(days)}d)"
            for idx, days in stale.iloc[:5].items()
        )
        raise ValueError(f"EUR/USD observations are stale at asset dates: {sample}")

    converted = usd / aligned_fx
    converted.name = usd.name
    return converted


def annual_rate_log_accrual(
    annual_rate_percent: pd.Series,
    anchors: pd.DatetimeIndex,
    *,
    day_count_basis: int = 360,
) -> pd.Series:
    """Accrue the rate known at the start anchor over actual calendar days."""
    if day_count_basis <= 0:
        raise ValueError("day_count_basis must be positive")
    anchors = pd.DatetimeIndex(anchors)
    if len(anchors) < 2:
        return pd.Series(dtype=float, index=anchors[:0], name=annual_rate_percent.name)
    if not anchors.is_monotonic_increasing or anchors.has_duplicates:
        raise ValueError("anchors must be unique and sorted in ascending order")

    days = np.diff(anchors).astype("timedelta64[D]").astype(int)
    if np.any(days <= 0):
        raise ValueError("anchors must increase")
    start_rates = _asof_rates_at_anchors(annual_rate_percent, anchors[:-1])
    simple_accrual = start_rates.to_numpy() / 100.0 * days / day_count_basis
    if np.any(simple_accrual <= -1.0):
        raise ValueError("Annual rates imply an invalid simple accrual <= -100%.")
    return pd.Series(
        np.log1p(simple_accrual),
        index=anchors[1:],
        name=annual_rate_percent.name,
        dtype=float,
    )


def daily_rate_total_return_level(
    annual_rate_percent: pd.Series,
    *,
    end: str | pd.Timestamp,
    day_count_basis: int = 360,
) -> pd.Series:
    """Build a daily-compounded cash level without using future rate fixings.

    The fixing known at the start of each calendar day accrues over that day's
    one-day ACT interval.  Returning a level makes monthly and weekly sampling
    exact aggregations of the same underlying cash path.
    """
    if day_count_basis <= 0:
        raise ValueError("day_count_basis must be positive")
    rates = _coerce_daily_series(
        annual_rate_percent, name=annual_rate_percent.name or "annual_rate"
    )
    if rates.empty:
        raise ValueError("annual_rate_percent must contain at least one observation")
    end_date = pd.Timestamp(end).normalize()
    if rates.index.min() > end_date:
        raise ValueError("end must not precede the first rate observation")

    calendar = pd.date_range(rates.index.min(), end_date, freq="D")
    if len(calendar) == 1:
        return pd.Series([1.0], index=calendar, name="cash_total_return_level")
    start_rates = _asof_rates_at_anchors(rates, calendar[:-1])
    daily_simple = start_rates.to_numpy() / 100.0 / day_count_basis
    if np.any(daily_simple <= -1.0):
        raise ValueError("Annual rates imply an invalid daily accrual <= -100%.")
    log_levels = np.concatenate(([0.0], np.cumsum(np.log1p(daily_simple), dtype=float)))
    return pd.Series(
        np.exp(log_levels),
        index=calendar,
        name="cash_total_return_level",
        dtype=float,
    )


def _prepare_levels_for_currency_operation(
    levels: pd.Series,
    *,
    operation: CurrencyOperation,
    eurusd_levels: pd.Series,
) -> pd.Series:
    if operation is CurrencyOperation.USD_TO_EUR_SPOT:
        return convert_usd_levels_to_eur(levels, eurusd_levels)
    if operation is CurrencyOperation.IDENTITY:
        return levels
    raise ValueError(f"Unsupported currency operation: {operation}")


def _pad_start(start: str | pd.Timestamp, frequency: PanelFrequency) -> str:
    start_ts = pd.Timestamp(start)
    if frequency is PanelFrequency.MONTHLY:
        padded = start_ts - pd.DateOffset(months=2)
    else:
        padded = start_ts - pd.Timedelta(days=21)
    return padded.date().isoformat()


def _last_completed_period_end(
    frequency: PanelFrequency,
    *,
    as_of: pd.Timestamp | None = None,
) -> pd.Timestamp:
    """Return the last fully completed month-end or Friday anchor."""
    current_date = (as_of or pd.Timestamp.today()).normalize()
    if frequency is PanelFrequency.MONTHLY:
        return (current_date.to_period("M") - 1).end_time.normalize()
    return (current_date.to_period("W-FRI") - 1).end_time.normalize()


def _floor_to_period_end(
    value: pd.Timestamp,
    frequency: PanelFrequency,
) -> pd.Timestamp:
    """Return the latest grid anchor not after ``value``."""
    value = pd.Timestamp(value).normalize()
    period_rule = "M" if frequency is PanelFrequency.MONTHLY else "W-FRI"
    candidate = value.to_period(period_rule).end_time.normalize()
    if candidate <= value:
        return candidate
    return (value.to_period(period_rule) - 1).end_time.normalize()


def _ceil_to_period_end(
    value: pd.Timestamp,
    frequency: PanelFrequency,
) -> pd.Timestamp:
    """Return the first grid anchor not before ``value``."""
    definition = get_frequency_definition(frequency)
    anchors = pd.date_range(
        pd.Timestamp(value), periods=1, freq=definition.resample_rule
    )
    return pd.Timestamp(anchors[0]).normalize()


def _validate_complete_panel(
    panel: pd.DataFrame,
    *,
    start: pd.Timestamp,
    end: pd.Timestamp,
    frequency: PanelFrequency,
) -> None:
    definition = get_frequency_definition(frequency)
    expected = pd.date_range(start, end, freq=definition.resample_rule)
    missing_rows = expected.difference(panel.index)
    if len(missing_rows):
        sample = ", ".join(item.date().isoformat() for item in missing_rows[:5])
        raise ValueError(f"Panel is missing required period anchors: {sample}")
    missing_counts = panel.reindex(expected).isna().sum()
    missing_counts = missing_counts[missing_counts > 0]
    if len(missing_counts):
        summary = ", ".join(
            f"{key}={int(count)}" for key, count in missing_counts.items()
        )
        raise ValueError(f"Panel has incomplete requested coverage: {summary}")


def build_daily_proxy_2011_return_panel(
    *,
    start: str,
    end: str | None,
    frequency: PanelFrequency,
    fred_api_key: str | None,
    strict_no_nan: bool,
) -> pd.DataFrame:
    """Build EUR monthly or W-FRI returns from the daily no-purchase profile."""
    frequency_definition = get_frequency_definition(frequency)
    requested_start = pd.Timestamp(start).normalize()
    panel_start = _ceil_to_period_end(requested_start, frequency)
    requested_end = pd.Timestamp(end).normalize() if end is not None else None
    last_completed_end = _last_completed_period_end(frequency)
    requested_anchor = (
        _floor_to_period_end(requested_end, frequency)
        if requested_end is not None
        else last_completed_end
    )
    panel_end = min(requested_anchor, last_completed_end)
    if panel_start > panel_end:
        raise ValueError("start must not be after the last requested completed period")
    source_start = _pad_start(panel_start, frequency)

    eurusd_levels = _load_fred_daily_observations(
        EURUSD_DAILY_SERIES,
        start=source_start,
        end=panel_end.date().isoformat(),
        fred_api_key=fred_api_key,
    )
    if (eurusd_levels <= 0).any():
        raise ValueError("DEXUSEU levels must be strictly positive.")
    eurusd_returns, _ = period_log_returns(
        eurusd_levels, frequency=frequency, end=panel_end
    )
    eurusd_returns.name = "fx_eurusd"

    eur_cash_rate = _load_fred_daily_observations(
        EUR_CASH_DAILY_SERIES,
        start=source_start,
        end=panel_end.date().isoformat(),
        fred_api_key=fred_api_key,
    )

    result_series: dict[str, pd.Series] = {"fx_eurusd": eurusd_returns}
    provenance: dict[str, dict[str, Any]] = {
        "fx_eurusd": {
            **dict(eurusd_levels.attrs),
            "source_identifier": EURUSD_DAILY_SERIES,
            "provider": "Federal Reserve Board via FRED",
            "license_class": "public_domain_citation_requested",
            "quote_convention": "USD per EUR",
            "value_kind": "fx_level",
            "target_currency": "EUR/USD risk driver",
            "currency_operation": CurrencyOperation.IDENTITY.value,
        }
    }

    for key, source_definition in DAILY_PROXY_2011_YAHOO_LEVEL_SERIES.items():
        levels = _load_yfinance_daily_levels(
            source_definition,
            start=source_start,
            end=panel_end.date().isoformat(),
        )
        provenance[key] = dict(levels.attrs)
        prepared_levels = _prepare_levels_for_currency_operation(
            levels,
            operation=source_definition.currency_operation,
            eurusd_levels=eurusd_levels,
        )
        returns, _ = period_log_returns(
            prepared_levels, frequency=frequency, end=panel_end
        )
        returns.name = key
        result_series[key] = returns

    for key, source_definition in DAILY_PROXY_2011_ISHARES_PERFORMANCE_SERIES.items():
        levels = _load_ishares_performance_levels(
            source_definition,
            start=source_start,
            end=panel_end.date().isoformat(),
        )
        provenance[key] = dict(levels.attrs)
        prepared_levels = _prepare_levels_for_currency_operation(
            levels,
            operation=source_definition.currency_operation,
            eurusd_levels=eurusd_levels,
        )
        returns, _ = period_log_returns(
            prepared_levels, frequency=frequency, end=panel_end
        )
        returns.name = key
        result_series[key] = returns

    cash_levels = daily_rate_total_return_level(
        eur_cash_rate,
        end=panel_end,
        day_count_basis=360,
    )
    cash, _ = period_log_returns(
        cash_levels,
        frequency=frequency,
        end=panel_end,
    )
    cash.name = "cash"
    result_series["cash"] = cash
    provenance["cash"] = {
        **dict(eur_cash_rate.attrs),
        "source_identifier": EUR_CASH_DAILY_SERIES,
        "provider": "European Central Bank via FRED",
        "license_class": "ecb_reuse_with_attribution",
        "quote_currency": "EUR",
        "target_currency": "EUR",
        "currency_operation": CurrencyOperation.IDENTITY.value,
        "value_kind": "annual_policy_rate_percent",
        "return_construction": (
            "daily ACT/360 compounding with the last rate known at day start; "
            "period-end level sampling"
        ),
        "known_limitation": "Overnight policy-rate proxy; not 3-month Euribor.",
    }

    ordered_keys = get_driver_keys()
    missing = [key for key in ordered_keys if key not in result_series]
    extra = [key for key in result_series if key not in ordered_keys]
    if missing or extra:
        raise ValueError(f"Profile driver mismatch: missing={missing}, extra={extra}")

    panel = pd.DataFrame({key: result_series[key] for key in ordered_keys})
    panel = panel[(panel.index >= panel_start) & (panel.index <= panel_end)]
    if panel.empty:
        raise ValueError("Built daily_proxy_2011 panel is empty.")

    if strict_no_nan:
        _validate_complete_panel(
            panel,
            start=panel_start,
            end=panel_end,
            frequency=frequency,
        )
        if panel.index.min() != panel_start or panel.index.max() != panel_end:
            raise ValueError(
                "Panel boundaries do not match the effective completed-period window."
            )
    else:
        missing_counts = panel.isna().sum()
        if missing_counts.any():
            logger.warning(
                "daily_proxy_2011 panel contains missing values:\n%s",
                missing_counts[missing_counts > 0],
            )

    panel.attrs.update(
        {
            "source_profile": SourceProfile.DAILY_PROXY_2011.value,
            "frequency": frequency_definition.metadata_label,
            "periods_per_year": frequency_definition.periods_per_year,
            "return_representation": "EUR log returns",
            "resample_rule": frequency_definition.resample_rule,
            "cash_accrual": "ECBDFR daily-compounded ACT/360",
            "hedge_method": "native EUR-hedged ETF share classes",
            "requested_start": requested_start.date().isoformat(),
            "requested_end": (
                requested_end.date().isoformat() if requested_end is not None else None
            ),
            "effective_completed_start": panel_start.date().isoformat(),
            "effective_completed_end": panel_end.date().isoformat(),
            "source_provenance": provenance,
            "data_access_note": (
                "No paid index export is required. Yahoo/yfinance and the public "
                "iShares endpoint are free-access research sources, not Open Data; "
                "their terms do not guarantee redistribution or publication rights."
            ),
        }
    )
    return panel
