# MSCI index data - manual download required for `monthly_legacy`

Do not place raw MSCI files into git. Keep them in this folder only for local
execution of the frozen `monthly_legacy` profile.

These monthly files cannot produce weekly returns. `daily_proxy_2011` does not
use local MSCI files and does not require daily MSCI downloads. It builds both
monthly and weekly research panels from the same nine Yahoo market proxies, one
public iShares NAV-performance series, and FRED inputs instead.

## Legacy download instructions

1. Open <https://app2.msci.com/products/index-data-search/>.
2. Export each required index:
   - MSCI EMU
   - MSCI World ex EMU
3. Select:
   - Currency: USD
   - Level: Net Total Return
   - Frequency: Monthly
4. Save each file here with one of the accepted names.

Expected base names:

- `msci_emu_ntr_usd`
- `msci_world_ex_emu_ntr_usd`

Accepted extensions are `.csv`, `.xls`, and `.xlsx`.

Metadata rows are supported as long as each observation ultimately contains a
date and an index level. Typical dates include `Dec-86`, `2023-12-31`, and
`31/12/1986`.

## Currency warning

The filename suffix does not prove the exported currency. Verify that the file
header itself says `Currency: USD`. The files present during the August 2026
audit were named `*_usd.xls` but declared `Currency: EUR`; applying the legacy
USD conversion to those files double-converts EUR/USD.

`monthly_legacy` retains that historical calculation for published-result
reproducibility and marks the mismatch in `DataFrame.attrs`.
`daily_proxy_2011` has a separate remote-source contract and validates the
returned listing currencies; it never reads or silently substitutes these MSCI
files. Its iShares Euro-HY input is the official performance-chart series for
portfolio `251843` / ISIN `IE00B66F4759`, on a NAV basis with gross income
reinvested. The endpoint is free-access but undocumented, has no availability
SLA, and is not guaranteed Open Data or publication-cleared.

MSCI's free web interface does not make its index observations Open Data. MSCI
terms can restrict reproduction, redistribution, and derived use. Keep the raw
exports local and review the applicable terms before publishing results based on
`monthly_legacy`.

## Quick verification

```powershell
$env:BLOCK1_CHECK_RAW_DATA = "1"
python -m pytest -q tests/test_load_data.py -k msci_download_present_and_parseable
```

The published 12-driver monthly panel is expected to be complete from
**2011-01-31 through 2025-12-31** when all legacy sources are available.
