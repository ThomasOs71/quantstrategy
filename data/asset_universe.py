"""Asset universe specification for Block-1 data ingestion.

Static registry of all modeled variables used by the scenario pipeline:
- 11 investable assets
- 1 FX driver (EUR/USD), not used directly for optimization
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


N_DRIVERS = 12  # 11 assets + 1 FX driver
N_ASSETS = 11   # only investable assets
COMMON_START = "2010-09-01"  # binding constraint: Euro HY ETF inception


class CurrencyTreatment(Enum):
    """How an asset price series is converted into EUR return convention."""

    EUR_NATIVE = "eur_native"  # priced in EUR
    EUR_HEDGED_FORMULA = "eur_hedged_formula"  # USD ETF + EUR-USD carry carry
    USD_VIA_EURUSD = "usd_via_eurusd"  # USD priced asset, convert via EUR/USD log-return
    EUR_USD_RATE = "eur_usd_rate"  # FX driver itself


class DataSource(Enum):
    """Source-type marker for each registry entry."""

    FRED = "fred"  # FRED API via fredapi
    YFINANCE = "yfinance"  # ETF close data via yfinance
    MSCI_CSV = "msci_csv"  # manual MSCI CSV export in data/raw/msci/
    YFINANCE_HEDGE = "yfinance_hedge"  # USD ETF that is hedged with formula


@dataclass(frozen=True)
class AssetDefinition:
    """Metadata record for one modeled variable."""

    key: str
    display_name: str
    index_proxy: str
    source: DataSource
    source_id: str
    source_id_status: str  # "verified" | "tbd"
    history_start: str
    currency_treatment: CurrencyTreatment
    is_portfolio_asset: bool
    notes: str = ""


# USD-priced assets that need EUR/USD log-return conversion
USD_EXPOSED_KEYS = (
    "global_dm_ex_emu",
    "em_equities",
    "gold",
    "commodities",
)


ASSET_UNIVERSE: dict[str, AssetDefinition] = {
    # Group A: Equities (EUR and USD)
    "euro_equities": AssetDefinition(
        key="euro_equities",
        display_name="Euro Equities",
        index_proxy="MSCI EMU Net Total Return (USD)",
        source=DataSource.MSCI_CSV,
        source_id="msci_emu_ntr_usd.csv",
        source_id_status="verified",
        history_start="1986-12",
        currency_treatment=CurrencyTreatment.USD_VIA_EURUSD,
        is_portfolio_asset=True,
        notes=(
            "Manual download from app2.msci.com: Developed Markets -> EMU -> "
            "Net Total Return -> USD. Save under data/raw/msci/ and do not commit "
            "raw data. Convert with EUR/USD log-additive in load_data.py."
        ),
    ),
    "global_dm_ex_emu": AssetDefinition(
        key="global_dm_ex_emu",
        display_name="Global DM ex-EMU",
        index_proxy="MSCI World ex EMU Net Total Return (USD)",
        source=DataSource.MSCI_CSV,
        source_id="msci_world_ex_emu_ntr_usd.csv",
        source_id_status="verified",
        history_start="1986-12",
        currency_treatment=CurrencyTreatment.USD_VIA_EURUSD,
        is_portfolio_asset=True,
        notes=(
            "Manual download from app2.msci.com: Developed Markets -> World ex EMU -> "
            "Net Total Return -> USD. Save under data/raw/msci/ and do not commit "
            "raw data. Convert with EUR/USD log-additive in load_data.py."
        ),
    ),
    "em_equities": AssetDefinition(
        key="em_equities",
        display_name="EM Equities",
        index_proxy="MSCI Emerging Markets (via ETF proxy, USD)",
        source=DataSource.YFINANCE,
        source_id="IQQE.DE",
        source_id_status="verified",
        history_start="2005-11",
        currency_treatment=CurrencyTreatment.USD_VIA_EURUSD,
        is_portfolio_asset=True,
        notes=(
            "iShares MSCI EM UCITS ETF (Dist), ISIN IE00B0M63177, Xetra. "
            "USD-denominated ETF; convert to EUR via EUR/USD."
        ),
    ),

    # Group B: EUR-native fixed income
    "euro_govt_bond_7_10": AssetDefinition(
        key="euro_govt_bond_7_10",
        display_name="Euro Govt Bond 7-10y",
        index_proxy="Bloomberg Euro Govt Bond 10 (via ETF proxy)",
        source=DataSource.YFINANCE,
        source_id="IBCM.AS",
        source_id_status="verified",
        history_start="2006-12",
        currency_treatment=CurrencyTreatment.EUR_NATIVE,
        is_portfolio_asset=True,
        notes=(
            "iShares Euro Government Bond 7-10yr UCITS ETF EUR (Dist), "
            "ISIN IE00B1FZS806, Euronext Amsterdam. Proxy deviates from ICE BofA "
            "index methodology."
        ),
    ),
    "euro_ig_credit": AssetDefinition(
        key="euro_ig_credit",
        display_name="Euro IG Credit",
        index_proxy="iBoxx EUR Liquid Corporates Large Cap (via ETF proxy)",
        source=DataSource.YFINANCE,
        source_id="IBCS.AS",
        source_id_status="verified",
        history_start="2003-03",
        currency_treatment=CurrencyTreatment.EUR_NATIVE,
        is_portfolio_asset=True,
        notes=(
            "iShares Euro Corporate Bond Large Cap UCITS ETF, "
            "ISIN IE0032523478, Euronext Amsterdam."
        ),
    ),
    "euro_high_yield": AssetDefinition(
        key="euro_high_yield",
        display_name="Euro High Yield",
        index_proxy="iBoxx EUR Liquid High Yield (via ETF proxy)",
        source=DataSource.YFINANCE,
        source_id="EUNW.AS",
        source_id_status="verified",
        history_start="2010-09",  # binding constraint for COMMON_START
        currency_treatment=CurrencyTreatment.EUR_NATIVE,
        is_portfolio_asset=True,
        notes=(
            "iShares EUR High Yield Corporate Bond UCITS ETF EUR (Dist), "
            "ISIN IE00B66F4759. BINDING CONSTRAINT: common start Sep 2010."
        ),
    ),

    # Group C: FX-hedged fixed-income proxies
    "global_govt_bond_eur_hedged": AssetDefinition(
        key="global_govt_bond_eur_hedged",
        display_name="Global Govt Bond (EUR-hedged)",
        index_proxy=(
            "FTSE G7 Government Bond, EUR-hedged via USD ETF + carry formula"
        ),
        source=DataSource.YFINANCE_HEDGE,
        source_id="IGLO.L",
        source_id_status="verified",
        history_start="2009-03",
        currency_treatment=CurrencyTreatment.EUR_HEDGED_FORMULA,
        is_portfolio_asset=True,
        notes=(
            "iShares Global Government Bond UCITS ETF USD (Dist), ISIN IE00B3F81K65. "
            "Unhedged USD ETF then adjusted with "
            "r_hedged = r_usd + (EURIBOR_3M - USD_3M)/12."
        ),
    ),
    "em_hc_bond_eur_hedged": AssetDefinition(
        key="em_hc_bond_eur_hedged",
        display_name="EM HC Bond (EUR-hedged)",
        index_proxy=(
            "JP Morgan EMBI Global Core, EUR-hedged via USD ETF + carry formula"
        ),
        source=DataSource.YFINANCE_HEDGE,
        source_id="IEMB.L",
        source_id_status="verified",
        history_start="2008-02",
        currency_treatment=CurrencyTreatment.EUR_HEDGED_FORMULA,
        is_portfolio_asset=True,
        notes=(
            "iShares J.P. Morgan USD Emerging Markets Bond UCITS ETF (Dist), "
            "ISIN IE00B2NPKV68. Unhedged USD ETF then adjusted with "
            "r_hedged = r_usd + (EURIBOR_3M - USD_3M)/12."
        ),
    ),

    # Group D: Real assets
    "gold": AssetDefinition(
        key="gold",
        display_name="Gold",
        index_proxy="LBMA Gold Price PM Fix (USD)",
        source=DataSource.FRED,
        source_id="GOLDPMGBD228NLBM",
        source_id_status="verified",
        history_start="1968-04",
        currency_treatment=CurrencyTreatment.USD_VIA_EURUSD,
        is_portfolio_asset=True,
        notes=(
            "LBMA PM Fix in USD/troy oz. FRED series GOLDPMGBD228NLBM. "
            "Resample to month end and compute log returns, then convert via EUR/USD."
        ),
    ),
    "commodities": AssetDefinition(
        key="commodities",
        display_name="Commodities",
        index_proxy="Bloomberg Commodity Index (via ETF proxy, USD)",
        source=DataSource.YFINANCE,
        source_id="EXXY.DE",
        source_id_status="verified",
        history_start="2007-08",
        currency_treatment=CurrencyTreatment.USD_VIA_EURUSD,
        is_portfolio_asset=True,
        notes=(
            "iShares Diversified Commodity Swap UCITS ETF (DE), ISIN DE000A0H0728. "
            "USD-denominated proxy converted via EUR/USD."
        ),
    ),

    # Group E: Cash
    "cash": AssetDefinition(
        key="cash",
        display_name="Cash (EURIBOR 3M)",
        index_proxy="EURIBOR 3M",
        source=DataSource.FRED,
        source_id="IR3TIB01EZM156N",
        source_id_status="verified",
        history_start="1994-01",
        currency_treatment=CurrencyTreatment.EUR_NATIVE,
        is_portfolio_asset=True,
        notes=(
            "OECD/EURIBOR series via FRED, monthly level in percent. "
            "Return conversion: r_t = log(1 + rate_t / 100 / 12)."
        ),
    ),

    # FX driver (not an asset in optimization)
    "fx_eurusd": AssetDefinition(
        key="fx_eurusd",
        display_name="EUR/USD (FX Risk Driver)",
        index_proxy="ECB reference rate EUR/USD",
        source=DataSource.FRED,
        source_id="DEXUSEU",
        source_id_status="verified",
        history_start="1999-01",
        currency_treatment=CurrencyTreatment.EUR_USD_RATE,
        is_portfolio_asset=False,
        notes=(
            "FRED DEXUSEU: USD per 1 EUR. Used as risk driver and for USD-to-EUR "
            "conversion: r_eur = r_usd - r_eurusd."
        ),
    ),
}


def get_investable_assets() -> dict[str, AssetDefinition]:
    """Return all 11 investable assets (excluding FX driver)."""
    return {k: v for k, v in ASSET_UNIVERSE.items() if v.is_portfolio_asset}


def get_asset_keys() -> list[str]:
    """Ordered list of investable asset keys."""
    return [k for k, v in ASSET_UNIVERSE.items() if v.is_portfolio_asset]


def get_driver_keys() -> list[str]:
    """Ordered list of all driver keys (including fx_eurusd)."""
    return list(ASSET_UNIVERSE.keys())


def get_usd_exposed_keys() -> tuple[str, ...]:
    """Keys that are converted from USD via EUR/USD log-return conversion."""
    return tuple(k for k in USD_EXPOSED_KEYS if k in ASSET_UNIVERSE)


def get_tbd_sources() -> list[str]:
    """Return keys with non-verified source ids."""
    return [k for k, v in ASSET_UNIVERSE.items() if v.source_id_status == "tbd"]


def n_investable() -> int:
    """Return 11."""
    return len(get_investable_assets())


def n_drivers() -> int:
    """Return 12."""
    return len(ASSET_UNIVERSE)
