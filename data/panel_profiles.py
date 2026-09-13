"""Frequency and source-profile contracts for Block-1 return panels.

The published Article-2 panel remains available through ``monthly_legacy``.
New monthly and weekly research panels use ``daily_proxy_2011``.  The latter
uses only sources that can be retrieved without purchasing index files, while
remaining explicit that free access is not the same as an open-data licence.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PanelFrequency(str, Enum):
    """Supported return-panel sampling frequencies."""

    MONTHLY = "monthly"
    WEEKLY = "weekly"


class SourceProfile(str, Enum):
    """Named source contracts with deliberately separate financial semantics."""

    MONTHLY_LEGACY = "monthly_legacy"
    DAILY_PROXY_2011 = "daily_proxy_2011"


class CurrencyOperation(str, Enum):
    """Transformation from the exact source quote currency to EUR returns."""

    IDENTITY = "identity"
    USD_TO_EUR_SPOT = "usd_to_eur_spot"


@dataclass(frozen=True)
class FrequencyDefinition:
    """Pandas resampling and reporting metadata for one panel frequency."""

    frequency: PanelFrequency
    resample_rule: str
    metadata_label: str
    periods_per_year: int


@dataclass(frozen=True)
class RemoteLevelDefinition:
    """Contract for a daily adjusted-close series loaded through yfinance."""

    key: str
    ticker: str
    quote_currency: str
    target_currency: str
    currency_operation: CurrencyOperation
    description: str
    first_level_date: str
    income_treatment: str
    value_kind: str = "adjusted_close_total_return_proxy"
    provider: str = "Yahoo Finance via yfinance"
    license_class: str = "free_access_not_open_data"
    known_limitation: str | None = None


@dataclass(frozen=True)
class ISharesPerformanceDefinition:
    """Contract for a public iShares NAV-performance chart series."""

    key: str
    portfolio_id: int
    source_identifier: str
    quote_currency: str
    target_currency: str
    currency_operation: CurrencyOperation
    description: str
    first_level_date: str
    value_kind: str = "nav_total_return_growth"
    provider: str = "BlackRock/iShares public product page"
    license_class: str = "free_access_not_open_data"
    distribution_treatment: str = "gross_income_reinvested"


FREQUENCY_DEFINITIONS: dict[PanelFrequency, FrequencyDefinition] = {
    PanelFrequency.MONTHLY: FrequencyDefinition(
        frequency=PanelFrequency.MONTHLY,
        resample_rule="ME",
        metadata_label="monthly_month_end",
        periods_per_year=12,
    ),
    PanelFrequency.WEEKLY: FrequencyDefinition(
        frequency=PanelFrequency.WEEKLY,
        resample_rule="W-FRI",
        metadata_label="weekly_friday",
        periods_per_year=52,
    ),
}


# Accumulating share classes are preferred because their level history does not
# depend on a separate dividend feed.  ``EUNW.DE`` is deliberately absent: its
# early Yahoo corporate-action history is incomplete, so Euro HY uses the
# official iShares NAV-performance growth series below.
DAILY_PROXY_2011_YAHOO_LEVEL_SERIES: dict[str, RemoteLevelDefinition] = {
    "euro_equities": RemoteLevelDefinition(
        key="euro_equities",
        ticker="SXR7.DE",
        quote_currency="EUR",
        target_currency="EUR",
        currency_operation=CurrencyOperation.IDENTITY,
        description="iShares Core MSCI EMU UCITS ETF EUR accumulating proxy.",
        first_level_date="2010-03-11",
        income_treatment="accumulating",
    ),
    "global_dm_ex_emu": RemoteLevelDefinition(
        key="global_dm_ex_emu",
        ticker="CM9.PA",
        quote_currency="EUR",
        target_currency="EUR",
        currency_operation=CurrencyOperation.IDENTITY,
        description="Amundi MSCI World ex EMU UCITS ETF EUR accumulating proxy.",
        first_level_date="2009-06-16",
        income_treatment="accumulating",
    ),
    "em_equities": RemoteLevelDefinition(
        key="em_equities",
        ticker="EUNM.DE",
        quote_currency="EUR",
        target_currency="EUR",
        currency_operation=CurrencyOperation.IDENTITY,
        description="iShares MSCI Emerging Markets UCITS ETF accumulating proxy.",
        first_level_date="2009-10-20",
        income_treatment="accumulating",
    ),
    "euro_govt_bond_7_10": RemoteLevelDefinition(
        key="euro_govt_bond_7_10",
        ticker="SXRQ.DE",
        quote_currency="EUR",
        target_currency="EUR",
        currency_operation=CurrencyOperation.IDENTITY,
        description="iShares Euro Government Bond 7-10yr UCITS ETF accumulating proxy.",
        first_level_date="2009-11-25",
        income_treatment="accumulating",
    ),
    "euro_ig_credit": RemoteLevelDefinition(
        key="euro_ig_credit",
        ticker="D5BG.DE",
        quote_currency="EUR",
        target_currency="EUR",
        currency_operation=CurrencyOperation.IDENTITY,
        description="Xtrackers II EUR Corporate Bond UCITS ETF 1C accumulating proxy.",
        first_level_date="2010-02-23",
        income_treatment="accumulating",
        known_limitation=(
            "Yahoo retains the long history under D5BG.DE although the current "
            "Xetra trading code is XBLC; the benchmark changed in 2017."
        ),
    ),
    "global_govt_bond_eur_hedged": RemoteLevelDefinition(
        key="global_govt_bond_eur_hedged",
        ticker="DBZB.DE",
        quote_currency="EUR",
        target_currency="EUR",
        currency_operation=CurrencyOperation.IDENTITY,
        description="Xtrackers global government bond accumulating EUR-hedged proxy.",
        first_level_date="2008-10-20",
        income_treatment="accumulating",
        known_limitation=(
            "Native EUR-hedged developed-government proxy; benchmark family "
            "changed during 2017-2018 and is broader than the legacy G7 proxy."
        ),
    ),
    "em_hc_bond_eur_hedged": RemoteLevelDefinition(
        key="em_hc_bond_eur_hedged",
        ticker="XEMB.DE",
        quote_currency="EUR",
        target_currency="EUR",
        currency_operation=CurrencyOperation.IDENTITY,
        description="Xtrackers USD emerging-markets bond accumulating EUR-hedged proxy.",
        first_level_date="2008-05-06",
        income_treatment="accumulating",
        known_limitation="Native EUR-hedged proxy with documented index changes.",
    ),
    "gold": RemoteLevelDefinition(
        key="gold",
        ticker="GLD",
        quote_currency="USD",
        target_currency="EUR",
        currency_operation=CurrencyOperation.USD_TO_EUR_SPOT,
        description="SPDR Gold Shares adjusted-close proxy.",
        first_level_date="2004-11-18",
        income_treatment="non_distributing",
    ),
    "commodities": RemoteLevelDefinition(
        key="commodities",
        ticker="EXXY.DE",
        quote_currency="EUR",
        target_currency="EUR",
        currency_operation=CurrencyOperation.IDENTITY,
        description="iShares diversified commodity swap EUR proxy.",
        first_level_date="2008-01-02",
        income_treatment="non_distributing",
    ),
}


DAILY_PROXY_2011_ISHARES_PERFORMANCE_SERIES: dict[str, ISharesPerformanceDefinition] = {
    "euro_high_yield": ISharesPerformanceDefinition(
        key="euro_high_yield",
        portfolio_id=251843,
        source_identifier="IE00B66F4759 NAV performance",
        quote_currency="EUR",
        target_currency="EUR",
        currency_operation=CurrencyOperation.IDENTITY,
        description=(
            "iShares EUR High Yield Corporate Bond UCITS ETF NAV performance "
            "with gross income reinvested."
        ),
        first_level_date="2010-09-03",
    )
}


PROFILE_DEFAULT_STARTS: dict[tuple[SourceProfile, PanelFrequency], str] = {
    (SourceProfile.MONTHLY_LEGACY, PanelFrequency.MONTHLY): "2010-09-01",
    (SourceProfile.DAILY_PROXY_2011, PanelFrequency.MONTHLY): "2011-01-31",
    (SourceProfile.DAILY_PROXY_2011, PanelFrequency.WEEKLY): "2011-01-07",
}


def coerce_panel_frequency(value: str | PanelFrequency) -> PanelFrequency:
    """Return a validated frequency enum with an informative error."""
    if isinstance(value, PanelFrequency):
        return value
    try:
        return PanelFrequency(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in PanelFrequency)
        raise ValueError(f"frequency must be one of: {choices}") from exc


def coerce_source_profile(value: str | SourceProfile) -> SourceProfile:
    """Return a validated source-profile enum with an informative error."""
    if isinstance(value, SourceProfile):
        return value
    try:
        return SourceProfile(value)
    except ValueError as exc:
        choices = ", ".join(item.value for item in SourceProfile)
        raise ValueError(f"source_profile must be one of: {choices}") from exc


def get_frequency_definition(
    frequency: str | PanelFrequency,
) -> FrequencyDefinition:
    """Return resampling metadata for ``frequency``."""
    return FREQUENCY_DEFINITIONS[coerce_panel_frequency(frequency)]


def get_profile_default_start(
    source_profile: str | SourceProfile,
    frequency: str | PanelFrequency,
) -> str:
    """Return the documented complete-panel start for a profile/frequency pair."""
    profile = coerce_source_profile(source_profile)
    freq = coerce_panel_frequency(frequency)
    try:
        return PROFILE_DEFAULT_STARTS[(profile, freq)]
    except KeyError as exc:
        raise ValueError(
            f"source_profile={profile.value!r} does not support frequency={freq.value!r}"
        ) from exc


def get_required_local_files(
    source_profile: str | SourceProfile,
) -> tuple[str, ...]:
    """List local files required by a source profile."""
    profile = coerce_source_profile(source_profile)
    if profile is SourceProfile.MONTHLY_LEGACY:
        return (
            "msci_emu_ntr_usd.csv",
            "msci_world_ex_emu_ntr_usd.csv",
        )
    return ()
