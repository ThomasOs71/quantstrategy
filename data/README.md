# Data Layout

This directory contains data code and documentation. Downloaded raw data files
must stay outside version control.

## Expected local data structure

- `data/raw/msci/`
  - Manual monthly MSCI exports used only by `monthly_legacy`
  - See `data/raw/msci/README.md`
- `data/raw/fred_cache/`
  - Optional local FRED caches (ignored by git)
- `data/downloads/daily_proxy_2011/monthly/`
  - Locally persisted monthly return panel and manifest (ignored by git)
- `data/downloads/daily_proxy_2011/weekly/`
  - Locally persisted weekly return panel and manifest (ignored by git)

`daily_proxy_2011` requires no ordered, paid, or manually placed market-data
files. Nine market-proxy levels are retrieved from Yahoo Finance, Euro HY comes
from the public iShares NAV-performance chart, and rate and FX inputs come from
FRED.

## Why no raw data in git

`monthly_legacy` needs two local MSCI monthly index files. They remain excluded
because of licensing and redistribution constraints. `daily_proxy_2011` does
not use those files and does not require the previously considered daily MSCI,
iBoxx, FTSE/LSEG, BlackRock-performance, or Euribor exports.

Yahoo/yfinance and the public iShares performance endpoint can be accessed
without ordering a source export, but their observations are not Open Data.
Raw observations and local caches must not be committed or redistributed. This
repository documents the source and calculation contract; it does not grant or
guarantee publication rights.

## How to run locally

1. Put the FRED key in `FRED_API_KEY`, an ignored supported `FredAPI.txt`, or
   pass it to `build_return_panel`.
2. Supply the monthly MSCI files only when using `monthly_legacy`.
3. Run the common loader for either supported profile.

```python
from data.load_data import build_return_panel

monthly_legacy = build_return_panel()

monthly_research = build_return_panel(
    source_profile="daily_proxy_2011",
    frequency="monthly",
    start="2011-01-31",
    end="2025-12-31",
)

weekly_research = build_return_panel(
    source_profile="daily_proxy_2011",
    frequency="weekly",
    start="2011-01-07",
    end="2025-12-31",
)
```

## Persist local monthly or weekly snapshots

`download_return_panel` accepts exactly the same `monthly` and `weekly`
frequency options as the research builder. It fixes the source profile to
`daily_proxy_2011`, requires complete coverage, and writes an ISO-date CSV plus
a JSON manifest containing provenance and a SHA-256 checksum.

```python
from data.load_data import download_return_panel

monthly_files = download_return_panel(
    frequency="monthly",
    start="2011-01-31",
    end="2025-12-31",
)
weekly_files = download_return_panel(
    frequency="weekly",
    start="2011-01-07",
    end="2025-12-31",
)
```

The returned dictionaries contain `data` and `manifest` paths. Deterministic
filenames include the effective panel boundaries. Existing files are not
replaced unless `overwrite=True` is explicit. Monthly and weekly use separate
default folders below `data/downloads/daily_proxy_2011/`; all contents below
`data/downloads/` are ignored by Git. A custom destination outside the
repository is allowed, while an in-repository destination outside the ignored
tree is rejected.

## Loader contract

Source choice and frequency are independent. `monthly_legacy` is intentionally
the no-argument default and supports no weekly mode. It preserves historical
calculations, including known source-currency mismatches, so Article-2 artifacts
remain reproducible.

`daily_proxy_2011` uses one daily-source engine for monthly and weekly outputs.
Nine market proxies are Yahoo/yfinance adjusted levels, using accumulating
share classes where available. Euro HY is different: it uses the official
iShares performance-chart series for portfolio `251843` / ISIN `IE00B66F4759`,
a EUR-denominated NAV-based growth series with gross income reinvested from
2010-09-03. This avoids relying on the incomplete 2011-2012 Yahoo dividend
history for the distributing `EUNW.DE` listing. The two foreign-bond drivers
use native EUR-hedged accumulating ETFs (`DBZB.DE` and `XEMB.DE`), so this
profile does not construct a synthetic hedge from interest-rate differentials.
`GLD` is the only USD-quoted asset and is converted once through daily
`DEXUSEU` levels.

Cash is built as one daily-compounded ACT/360 level from FRED `ECBDFR`, using
only the last rate known at the start of each calendar day. Monthly and weekly
returns are sampled from that same path. This is an overnight ECB deposit-
facility policy-rate proxy, not 3-month Euribor and not a directly investable
retail cash product.

The complete source, currency, proxy, and access catalog is in
[`SOURCES.md`](SOURCES.md).

For both research frequencies:

- inputs are nine daily Yahoo adjusted market levels, one daily iShares
  NAV-performance level, daily FX levels, or daily rate levels;
- monthly selects the last observation per `ME` bucket;
- weekly selects the last observation per `W-FRI` bucket;
- log differences are calculated only after level aggregation;
- Friday holidays use the latest fresh observation on or before Friday;
- requested coverage is strict by default;
- source IDs, currencies, transformations, retrieval timestamps, and package
  versions are stored in `panel.attrs` provenance.

The iShares Euro-HY series is issuer-reported NAV performance with gross income
reinvested, not the Yahoo adjusted market price for `EUNW.DE`. Its public
product-data endpoint is undocumented, carries no availability SLA, and is not
guaranteed to remain stable or to confer publication rights. These access and
provenance limits remain explicit even though the missing early Yahoo dividend
history is no longer part of the panel.

## Target sample expectations

The published legacy target remains 12 drivers with complete monthly coverage
from **2011-01-31 through 2025-12-31**. Article 2 continues to validate exactly
180 rows x 12 columns.

```powershell
python -c "from data.load_data import build_return_panel; p=build_return_panel(start='2010-01-01', end='2025-12-31'); complete=(~p.isna().any(axis=1)); c=p.index[complete]; print(f'complete_rows={len(c)}', f'first={c.min().date()}', f'last={c.max().date()}')"
```

The `daily_proxy_2011` contracts are:

| Frequency | Complete window | Rows | Drivers |
|---|---:|---:|---:|
| Monthly | 2011-01-31 to 2025-12-31 | 180 | 12 |
| Weekly (`W-FRI`) | 2011-01-07 to 2025-12-26 | 782 | 12 |

Run the end-to-end network contract with:

```powershell
$env:BLOCK1_CHECK_DAILY_PROXY_DATA = "1"
python -m pytest -q tests/test_profiled_load_data.py -k daily_proxy_2011_live_panel_contract
```

Ordinary `pytest` stays offline; live checks are opt-in.
