from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from data import load_data as legacy_load_data
from data import profiled_load_data as profiled
from data.asset_universe import get_driver_keys
from data.panel_profiles import (
    DAILY_PROXY_2011_ISHARES_PERFORMANCE_SERIES,
    DAILY_PROXY_2011_YAHOO_LEVEL_SERIES,
    CurrencyOperation,
    PanelFrequency,
    SourceProfile,
    get_profile_default_start,
    get_required_local_files,
)


def test_daily_proxy_registry_has_one_contract_per_non_rate_driver() -> None:
    registered = (
        set(DAILY_PROXY_2011_YAHOO_LEVEL_SERIES)
        | set(DAILY_PROXY_2011_ISHARES_PERFORMANCE_SERIES)
        | {"cash", "fx_eurusd"}
    )
    assert registered == set(get_driver_keys())
    assert len(DAILY_PROXY_2011_YAHOO_LEVEL_SERIES) == 9
    assert set(DAILY_PROXY_2011_ISHARES_PERFORMANCE_SERIES) == {"euro_high_yield"}

    for item in DAILY_PROXY_2011_YAHOO_LEVEL_SERIES.values():
        assert item.license_class == "free_access_not_open_data"
        assert item.income_treatment != "distributing"
    gold = DAILY_PROXY_2011_YAHOO_LEVEL_SERIES["gold"]
    assert gold.currency_operation is CurrencyOperation.USD_TO_EUR_SPOT
    assert gold.quote_currency == "USD"
    for key, item in DAILY_PROXY_2011_YAHOO_LEVEL_SERIES.items():
        if key != "gold":
            assert item.currency_operation is CurrencyOperation.IDENTITY
            assert item.quote_currency == "EUR"


def test_daily_proxy_defaults_and_local_file_contract() -> None:
    assert get_profile_default_start("daily_proxy_2011", "monthly") == "2011-01-31"
    assert get_profile_default_start("daily_proxy_2011", "weekly") == "2011-01-07"
    assert get_required_local_files(SourceProfile.DAILY_PROXY_2011) == ()
    with pytest.raises(ValueError, match="does not support"):
        get_profile_default_start("monthly_legacy", "weekly")


def test_w_fri_uses_last_available_observation_and_keeps_friday_label() -> None:
    levels = pd.Series(
        [100.0, 102.0, 103.0],
        index=pd.to_datetime(["2020-12-31", "2021-01-07", "2021-01-08"]),
        name="asset",
    )
    result = profiled.resample_period_end_levels(
        levels.iloc[:2], frequency="weekly", end="2021-01-08"
    )
    assert result.index[-1] == pd.Timestamp("2021-01-08")
    assert result.iloc[-1] == pytest.approx(102.0)


def test_last_completed_anchor_excludes_current_incomplete_period() -> None:
    assert profiled._last_completed_period_end(
        PanelFrequency.MONTHLY, as_of=pd.Timestamp("2021-01-15")
    ) == pd.Timestamp("2020-12-31")
    assert profiled._last_completed_period_end(
        PanelFrequency.WEEKLY, as_of=pd.Timestamp("2021-01-06")
    ) == pd.Timestamp("2021-01-01")


def test_period_end_resampling_rejects_materially_stale_level() -> None:
    levels = pd.Series(
        [100.0, 101.0],
        index=pd.to_datetime(["2021-01-01", "2021-01-10"]),
        name="asset",
    )
    with pytest.raises(ValueError, match="stale"):
        profiled.resample_period_end_levels(
            levels, frequency="weekly", end="2021-01-15"
        )


def test_daily_coercion_preserves_local_trading_date_when_dropping_timezone() -> None:
    index = pd.DatetimeIndex(["2021-01-04 00:00:00+01:00"])
    result = profiled._coerce_daily_series(
        pd.Series([100.0], index=index), name="asset"
    )
    assert result.index[0] == pd.Timestamp("2021-01-04")


def test_missing_week_does_not_become_a_two_week_return() -> None:
    levels = pd.Series(
        [100.0, 101.0, 102.0],
        index=pd.to_datetime(["2021-01-01", "2021-01-08", "2021-01-22"]),
        name="asset",
    )
    returns, _ = profiled.period_log_returns(
        levels, frequency="weekly", end="2021-01-22"
    )
    assert pd.isna(returns.loc["2021-01-15"])
    assert pd.isna(returns.loc["2021-01-22"])


def test_monthly_return_equals_sum_of_weekly_returns_when_anchors_coincide() -> None:
    levels = pd.Series(
        np.exp(np.linspace(0.0, 0.2, 32)),
        index=pd.date_range("2020-12-31", "2021-01-31", freq="D"),
        name="asset",
    )
    monthly, _ = profiled.period_log_returns(
        levels, frequency="monthly", end="2021-01-31"
    )
    weekly, _ = profiled.period_log_returns(
        levels, frequency="weekly", end="2021-01-29"
    )
    # Add the two boundary stubs; the identity is about common endpoints.
    initial_stub = np.log(levels.loc["2021-01-01"] / levels.loc["2020-12-31"])
    stub = np.log(levels.loc["2021-01-31"] / levels.loc["2021-01-29"])
    assert monthly.loc["2021-01-31"] == pytest.approx(
        initial_stub + weekly.loc["2021-01-01":"2021-01-29"].sum() + stub
    )


def test_rate_accrual_uses_rate_known_at_start_of_period() -> None:
    rates = pd.Series(
        [3.6, 36.0],
        index=pd.to_datetime(["2021-01-01", "2021-01-08"]),
        name="rate",
    )
    anchors = pd.to_datetime(["2021-01-01", "2021-01-08"])
    result = profiled.annual_rate_log_accrual(rates, anchors)
    assert result.loc["2021-01-08"] == pytest.approx(np.log1p(0.036 * 7 / 360))


def test_daily_cash_level_is_frequency_neutral_across_rate_change() -> None:
    rate_dates = pd.date_range("2020-12-31", "2021-01-31", freq="D")
    rates = pd.Series(
        np.where(rate_dates < pd.Timestamp("2021-01-15"), 0.0, 36.0),
        index=rate_dates,
        name="rate",
    )
    levels = profiled.daily_rate_total_return_level(rates, end="2021-01-31")
    monthly, _ = profiled.period_log_returns(
        levels, frequency="monthly", end="2021-01-31"
    )
    weekly, _ = profiled.period_log_returns(
        levels, frequency="weekly", end="2021-01-29"
    )
    end_stub = np.log(levels.loc["2021-01-31"] / levels.loc["2021-01-29"])

    assert monthly.loc["2021-01-31"] == pytest.approx(
        weekly.loc["2021-01-01":"2021-01-29"].sum() + end_stub
    )
    expected = 16 * np.log1p(0.36 / 360)
    assert monthly.loc["2021-01-31"] == pytest.approx(expected)


def test_rate_staleness_is_rejected() -> None:
    rates = pd.Series([2.0], index=pd.to_datetime(["2021-01-01"]))
    anchors = pd.to_datetime(["2021-01-15", "2021-01-22"])
    with pytest.raises(ValueError, match="Rate observations are stale"):
        profiled.annual_rate_log_accrual(rates, anchors)


def test_spot_conversion_uses_fx_on_asset_date_not_later_period_fx() -> None:
    usd_levels = pd.Series(
        [100.0, 110.0],
        index=pd.to_datetime(["2020-12-31", "2021-01-07"]),
        name="asset",
    )
    fx_levels = pd.Series(
        [1.0, 1.1, 1.2],
        index=pd.to_datetime(["2020-12-31", "2021-01-07", "2021-01-08"]),
    )
    converted = profiled.convert_usd_levels_to_eur(usd_levels, fx_levels)
    period_levels = profiled.resample_period_end_levels(
        converted, frequency="weekly", end="2021-01-08"
    )
    assert period_levels.loc["2021-01-08"] == pytest.approx(110.0 / 1.1)


def test_spot_conversion_rejects_missing_prior_fx_level() -> None:
    usd_levels = pd.Series([100.0], index=pd.to_datetime(["2021-01-01"]))
    fx_levels = pd.Series([1.2], index=pd.to_datetime(["2021-01-04"]))
    with pytest.raises(ValueError, match="No EUR/USD level known"):
        profiled.convert_usd_levels_to_eur(usd_levels, fx_levels)


def test_yfinance_listing_currency_mismatch_fails_before_download(monkeypatch) -> None:
    definition = DAILY_PROXY_2011_YAHOO_LEVEL_SERIES["em_equities"]
    fake_yfinance = SimpleNamespace(
        Ticker=lambda _ticker: SimpleNamespace(fast_info={"currency": "USD"}),
        download=lambda *_args, **_kwargs: pytest.fail("download must not be called"),
    )
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance)
    with pytest.raises(ValueError, match="Listing currency mismatch"):
        profiled._load_yfinance_daily_levels(
            definition, start="2020-01-01", end="2020-01-31"
        )


def test_yfinance_loader_requests_adjusted_daily_levels(monkeypatch) -> None:
    definition = DAILY_PROXY_2011_YAHOO_LEVEL_SERIES["euro_equities"]
    captured: dict[str, object] = {}

    def fake_download(*args, **kwargs):
        captured.update(kwargs)
        return pd.DataFrame(
            {"Close": [100.0, 101.0]},
            index=pd.to_datetime(["2020-01-02", "2020-01-03"]),
        )

    fake_yfinance = SimpleNamespace(
        Ticker=lambda _ticker: SimpleNamespace(fast_info={"currency": "EUR"}),
        download=fake_download,
    )
    monkeypatch.setitem(sys.modules, "yfinance", fake_yfinance)
    result = profiled._load_yfinance_daily_levels(
        definition, start="2020-01-01", end="2020-01-03"
    )
    assert list(result) == [100.0, 101.0]
    assert captured["interval"] == "1d"
    assert captured["auto_adjust"] is True


def test_ishares_loader_parses_nav_total_return_growth(monkeypatch) -> None:
    definition = DAILY_PROXY_2011_ISHARES_PERFORMANCE_SERIES["euro_high_yield"]
    captured: dict[str, object] = {}

    class FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "productId": 251843,
                "currencyCode": "EUR",
                "componentsByNameMap": {
                    "performance": {
                        "containersByNameMap": {
                            "chart": {
                                "dataPointsByNameMap": {
                                    "performanceData": {
                                        "asOfDate": [20100903, 20101231, 20110103],
                                        "value": [10000.0, 10100.0, 10110.0],
                                    }
                                }
                            }
                        }
                    }
                },
            }

    import requests

    def fake_get(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return FakeResponse()

    monkeypatch.setattr(requests, "get", fake_get)
    result = profiled._load_ishares_performance_levels(
        definition, start="2010-12-01", end="2011-01-03"
    )
    assert result.index[0] == pd.Timestamp("2010-12-31")
    assert result.iloc[-1] == pytest.approx(10110.0)
    assert result.attrs["distribution_treatment"] == "gross_income_reinvested"
    assert result.attrs["value_kind"] == "nav_total_return_growth"
    assert captured["params"]["portfolioId"] == "251843"


def test_fred_loader_retries_without_exposing_api_key(monkeypatch) -> None:
    import requests

    calls = 0

    def failing_get(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise requests.ConnectionError("request failed with api_key=TOP_SECRET")

    monkeypatch.setattr(requests, "get", failing_get)
    monkeypatch.setattr(profiled.time, "sleep", lambda _seconds: None)
    with pytest.raises(RuntimeError) as exc_info:
        profiled._load_fred_daily_observations(
            "DEXUSEU",
            start="2020-01-01",
            end="2020-01-31",
            fred_api_key="TOP_SECRET",
        )
    assert calls == 3
    assert "TOP_SECRET" not in str(exc_info.value)


def test_monthly_legacy_is_default_and_preserves_old_formulas(monkeypatch) -> None:
    index = pd.to_datetime(["2019-12-31", "2020-01-31", "2020-02-29"])

    def fake_fred(series_id, **_kwargs):
        if series_id == "DEXUSEU":
            return pd.Series([1.0, 1.1, 1.21], index=index, name=series_id)
        value = 1.0 if series_id == legacy_load_data.EURIBOR_3M_SERIES else 2.0
        return pd.Series(value, index=index, name=series_id)

    def fake_msci(filename, **_kwargs):
        return pd.Series([0.05, 0.05], index=index[1:], name=filename)

    def fake_etf(ticker, **_kwargs):
        return pd.Series([0.02, 0.02], index=index[1:], name=ticker)

    monkeypatch.setattr(legacy_load_data, "load_fred_series", fake_fred)
    monkeypatch.setattr(legacy_load_data, "load_msci_csv", fake_msci)
    monkeypatch.setattr(legacy_load_data, "load_etf_returns", fake_etf)

    default = legacy_load_data.build_return_panel(
        start="2020-01-31",
        end="2020-02-29",
        warn_legacy_currency=False,
        strict_no_nan=True,
    )
    explicit = legacy_load_data.build_return_panel(
        start="2020-01-31",
        end="2020-02-29",
        warn_legacy_currency=False,
        strict_no_nan=True,
        frequency="monthly",
        source_profile="monthly_legacy",
    )

    pd.testing.assert_frame_equal(default, explicit)
    fx_return = np.log(1.1)
    assert default.loc["2020-01-31", "euro_equities"] == pytest.approx(0.05 - fx_return)
    assert default.loc["2020-01-31", "em_equities"] == pytest.approx(0.02 - fx_return)
    assert default.loc["2020-01-31", "commodities"] == pytest.approx(0.02 - fx_return)
    expected_hedged = 0.02 + (1.0 - 2.0) / 12 / 100
    assert default.loc["2020-01-31", "global_govt_bond_eur_hedged"] == pytest.approx(
        expected_hedged
    )
    assert default.loc["2020-01-31", "cash"] == pytest.approx(np.log1p(1.0 / 100 / 12))
    assert default.attrs["known_currency_mismatch"] is True


def test_monthly_legacy_rejects_weekly_frequency() -> None:
    with pytest.raises(ValueError, match="supports only"):
        legacy_load_data.build_return_panel(
            frequency="weekly", source_profile="monthly_legacy"
        )


def test_daily_proxy_uses_one_engine_for_monthly_and_weekly(monkeypatch) -> None:
    dates = pd.bdate_range("2019-10-01", "2020-03-31")

    def price_levels(name: str, slope: float = 0.0005) -> pd.Series:
        values = 100.0 * np.exp(slope * np.arange(len(dates)))
        return pd.Series(values, index=dates, name=name)

    fred_calls: list[str] = []

    def fake_fred(series_id, **_kwargs):
        fred_calls.append(series_id)
        if series_id == profiled.EURUSD_DAILY_SERIES:
            return price_levels(series_id, slope=0.0001)
        if series_id == profiled.EUR_CASH_DAILY_SERIES:
            return pd.Series(1.0, index=dates, name=series_id)
        pytest.fail(f"unexpected rate/index request: {series_id}")

    def fake_yfinance(definition, **_kwargs):
        result = price_levels(definition.key)
        result.attrs.update(
            {
                "source_identifier": definition.ticker,
                "quote_currency": definition.quote_currency,
                "target_currency": definition.target_currency,
                "currency_operation": definition.currency_operation.value,
                "value_kind": definition.value_kind,
            }
        )
        return result

    def fake_ishares(definition, **_kwargs):
        result = price_levels(definition.key)
        result.attrs.update(
            {
                "source_identifier": definition.source_identifier,
                "quote_currency": definition.quote_currency,
                "target_currency": definition.target_currency,
                "currency_operation": definition.currency_operation.value,
                "value_kind": definition.value_kind,
                "distribution_treatment": definition.distribution_treatment,
            }
        )
        return result

    monkeypatch.setattr(profiled, "_load_fred_daily_observations", fake_fred)
    monkeypatch.setattr(profiled, "_load_yfinance_daily_levels", fake_yfinance)
    monkeypatch.setattr(profiled, "_load_ishares_performance_levels", fake_ishares)

    monthly = legacy_load_data.build_return_panel(
        start="2020-01-31",
        end="2020-03-31",
        frequency="monthly",
        source_profile="daily_proxy_2011",
        strict_no_nan=True,
    )
    weekly = legacy_load_data.build_return_panel(
        start="2020-01-03",
        end="2020-01-31",
        frequency="weekly",
        source_profile="daily_proxy_2011",
        strict_no_nan=True,
    )

    assert set(fred_calls) == {"DEXUSEU", "ECBDFR"}
    assert list(monthly.columns) == get_driver_keys()
    assert monthly.index.equals(pd.date_range("2020-01-31", "2020-03-31", freq="ME"))
    assert monthly.notna().all().all()
    assert monthly.attrs["source_profile"] == "daily_proxy_2011"
    assert monthly.attrs["periods_per_year"] == 12

    assert list(weekly.columns) == get_driver_keys()
    assert weekly.index.equals(pd.date_range("2020-01-03", "2020-01-31", freq="W-FRI"))
    assert weekly.notna().all().all()
    assert weekly.attrs["frequency"] == "weekly_friday"
    assert weekly.attrs["periods_per_year"] == 52
    assert weekly.attrs["hedge_method"] == "native EUR-hedged ETF share classes"
    for driver in get_driver_keys():
        assert driver in weekly.attrs["source_provenance"]
        assert "currency_operation" in weekly.attrs["source_provenance"][driver]

    raw = price_levels("raw")
    raw_monthly, _ = profiled.period_log_returns(
        raw, frequency=PanelFrequency.MONTHLY, end="2020-03-31"
    )
    fx_monthly, _ = profiled.period_log_returns(
        price_levels("fx", slope=0.0001),
        frequency=PanelFrequency.MONTHLY,
        end="2020-03-31",
    )
    assert monthly.loc["2020-02-29", "em_equities"] == pytest.approx(
        raw_monthly.loc["2020-02-29"]
    )
    assert monthly.loc["2020-02-29", "euro_high_yield"] == pytest.approx(
        raw_monthly.loc["2020-02-29"]
    )
    assert monthly.loc["2020-02-29", "gold"] == pytest.approx(
        raw_monthly.loc["2020-02-29"] - fx_monthly.loc["2020-02-29"]
    )
    assert weekly.loc["2020-01-10", "cash"] == pytest.approx(7 * np.log1p(0.01 / 360))


def test_non_anchor_end_is_reported_as_effective_grid_anchor(monkeypatch) -> None:
    dates = pd.bdate_range("2020-09-01", "2020-12-31")

    def levels(name: str) -> pd.Series:
        return pd.Series(
            100.0 + np.arange(len(dates)), index=dates, name=name, dtype=float
        )

    monkeypatch.setattr(
        profiled,
        "_load_fred_daily_observations",
        lambda series_id, **_kwargs: (
            pd.Series(1.1, index=dates, name=series_id)
            if series_id == "DEXUSEU"
            else pd.Series(1.0, index=dates, name=series_id)
        ),
    )
    monkeypatch.setattr(
        profiled,
        "_load_yfinance_daily_levels",
        lambda definition, **_kwargs: levels(definition.key),
    )
    monkeypatch.setattr(
        profiled,
        "_load_ishares_performance_levels",
        lambda definition, **_kwargs: levels(definition.key),
    )

    panel = legacy_load_data.build_return_panel(
        start="2020-10-01",
        end="2020-12-30",
        source_profile="daily_proxy_2011",
    )
    assert panel.index.min() == pd.Timestamp("2020-10-31")
    assert panel.index.max() == pd.Timestamp("2020-11-30")
    assert panel.attrs["effective_completed_end"] == "2020-11-30"


def test_daily_proxy_2011_live_panel_contract() -> None:
    """Opt-in end-to-end network QA for both frequencies."""
    if os.environ.get("BLOCK1_CHECK_DAILY_PROXY_DATA") != "1":
        pytest.skip(
            "Set BLOCK1_CHECK_DAILY_PROXY_DATA=1 to run daily-proxy live-data QA."
        )

    monthly = legacy_load_data.build_return_panel(
        start="2011-01-31",
        end="2025-12-31",
        frequency="monthly",
        source_profile="daily_proxy_2011",
    )
    weekly = legacy_load_data.build_return_panel(
        start="2011-01-07",
        end="2025-12-31",
        frequency="weekly",
        source_profile="daily_proxy_2011",
    )

    assert monthly.shape == (180, 12)
    assert monthly.index[0] == pd.Timestamp("2011-01-31")
    assert monthly.index[-1] == pd.Timestamp("2025-12-31")
    assert monthly.notna().all().all()
    assert weekly.shape == (782, 12)
    assert weekly.index[0] == pd.Timestamp("2011-01-07")
    assert weekly.index[-1] == pd.Timestamp("2025-12-26")
    assert weekly.notna().all().all()
