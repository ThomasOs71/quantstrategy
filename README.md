# QuantStrategy

QuantStrategy is the code companion repository for the Substack flagship series *From Assumptions to Portfolios*, published at [quantstrategy.substack.com](https://quantstrategy.substack.com). The Substack also includes the no-code companion series *The Allocator's Toolkit*; because those practitioner articles are intentionally no-code, this repository does not maintain a separate Toolkit folder.

## Quick Start (data + tests)

- The local files under `data/raw/msci/` are required only by the frozen
  `monthly_legacy` profile. See
  [`data/raw/msci/README.md`](data/raw/msci/README.md) for its download details.
- `daily_proxy_2011` requires no ordered, paid, or manually placed market-data
  files. It retrieves nine daily market proxies from Yahoo Finance, the Euro-HY
  NAV-performance series from the public iShares product endpoint, and its rate
  and FX series from FRED.
- Keep all downloaded data and caches out of git.
- Make sure your FRED API key is available:
  - `API_Keys/FredAPI.txt` (preferred for local setup), or
  - `FRED_API_KEY` environment variable.

The no-argument loader remains the published monthly implementation:

```python
from data.load_data import build_return_panel

monthly_legacy = build_return_panel()
```

It is named `monthly_legacy` because the locally present MSCI exports and two
historical Yahoo listing-currency assumptions do not match the old USD
conversion logic. The numerical behavior is frozen for Article-2
reproducibility and emits a warning. Use `daily_proxy_2011` for the
frequency-neutral research panel with corrected listing-currency semantics.

### Verify local raw MSCI files

This check applies only to `monthly_legacy`:

```powershell
$env:BLOCK1_CHECK_RAW_DATA = "1"
python -m pytest -q tests/test_load_data.py -k msci_download_present_and_parseable
```

Linux/macOS:

```bash
export BLOCK1_CHECK_RAW_DATA=1
python -m pytest -q tests/test_load_data.py -k msci_download_present_and_parseable
```

### Optional: live legacy-panel QA test

```powershell
$env:BLOCK1_CHECK_LIVE_DATA = "1"
python -m pytest -q tests/test_load_data.py -k return_panel_quality_smoke
```

Ordinary `pytest` runs stay offline. Live FRED and Yahoo checks run only when
their explicit environment variables are set.

## Source profiles and frequencies

| Source profile | Monthly | Weekly | Purpose |
|---|---:|---:|---|
| `monthly_legacy` | yes | no | Reproduce the published monthly implementation |
| `daily_proxy_2011` | yes | yes (`W-FRI`) | Research panel built from one no-cost daily proxy set |

Both `daily_proxy_2011` frequencies use the same daily observations and
transformation engine. Period-end levels are selected first and log returns are
calculated afterwards. Weekly Friday holidays use the last observation on or
before Friday; a fully missing week remains missing.

The nine Yahoo/yfinance inputs are `SXR7.DE`, `CM9.PA`, `EUNM.DE`, `SXRQ.DE`,
`D5BG.DE`, `DBZB.DE`, `XEMB.DE`, `GLD`, and `EXXY.DE`. Accumulating share
classes are used where available. Euro HY instead uses the official iShares
performance-chart series for portfolio `251843` / ISIN `IE00B66F4759`: a
EUR-denominated, NAV-based growth series with gross income reinvested, available
from 2010-09-03. It therefore does not depend on the incomplete 2011-2012 Yahoo
dividend history for `EUNW.DE`. `GLD` is converted from USD to EUR with
`DEXUSEU`; cash uses `ECBDFR`, an overnight ECB policy-rate proxy rather than
3-month Euribor.

Yahoo/yfinance observations and the iShares performance-chart endpoint are
accessible without ordering a data license, but they are not Open Data and
their availability does not grant or guarantee redistribution or publication
rights. The iShares endpoint is undocumented and has no availability SLA. Raw
observations must not be committed. See
[`data/SOURCES.md`](data/SOURCES.md) for identifiers, quote currencies,
transformations, proxy limitations, and access terms.

```python
monthly = build_return_panel(
    source_profile="daily_proxy_2011",
    frequency="monthly",
    start="2011-01-31",
    end="2025-12-31",
)
weekly = build_return_panel(
    source_profile="daily_proxy_2011",
    frequency="weekly",
    start="2011-01-07",
    end="2025-12-31",
)
```

To persist either panel locally, use the same frequency names with
`download_return_panel`:

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

Each call writes a CSV plus a provenance/checksum manifest into its own default
folder: `data/downloads/daily_proxy_2011/monthly/` or
`data/downloads/daily_proxy_2011/weekly/`. The entire `data/downloads/` tree is
gitignored because the provider-derived observations are for local research use
and must not be committed. Existing snapshots are protected unless
`overwrite=True` is passed.

### Expected legacy target sample window

After all required legacy sources are present, the intended complete Block-1
driver panel is:

- all 12 driver series, including `fx_eurusd`;
- no missing values for the monthly window **2011-01-31 to 2025-12-31**.

Use this check to confirm:

```powershell
python -c "from data.load_data import build_return_panel; p=build_return_panel(start='2010-01-01', end='2025-12-31'); complete=(~p.isna().any(axis=1)); c=p.index[complete]; print(f'complete_rows={len(c)}', f'first={c.min().date()}', f'last={c.max().date()}')"
```

Linux/macOS:

```bash
python -c "from data.load_data import build_return_panel; p=build_return_panel(start='2010-01-01', end='2025-12-31'); complete=(~p.isna().any(axis=1)); c=p.index[complete]; print(f'complete_rows={len(c)}', f'first={c.min().date()}', f'last={c.max().date()}')"
```

### Expected daily-proxy target windows

With the remote sources available through December 2025, the strict contracts
are:

- monthly: **2011-01-31 to 2025-12-31**, 180 rows x 12 drivers;
- weekly: **2011-01-07 to 2025-12-26**, 782 rows x 12 drivers.

Run the opt-in end-to-end network check with:

```powershell
$env:BLOCK1_CHECK_DAILY_PROXY_DATA = "1"
python -m pytest -q tests/test_profiled_load_data.py -k daily_proxy_2011_live_panel_contract
```
