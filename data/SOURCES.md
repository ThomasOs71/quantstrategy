# Block-1 source catalog

This catalog separates economic asset identity from the exact returned series,
quote currency, conversion, proxy limitations, and data-access contract. The
executable definitions for non-legacy panels are in `data/panel_profiles.py`.

## Profiles

### `monthly_legacy`

Compatibility-only monthly implementation. It preserves the original FRED
monthly aggregation, Yahoo `1mo` downloads, `/12` cash/carry formulas, and
historical FX transformations. `em_equities` and `commodities` have permanent
legacy listing-currency mismatches. `euro_equities` and `global_dm_ex_emu` are
additional mismatches when the audited local MSCI headers declare EUR, as the
current local files do; genuine USD exports make those two legacy conversions
consistent. Do not use this profile as the corrected research baseline.

### `daily_proxy_2011`

One daily-source engine feeds either monthly (`ME`) or weekly (`W-FRI`)
returns. It needs no ordered or manually placed market-data files. Nine Yahoo
listing histories are used as market-level proxies, and Euro HY uses the
official iShares NAV-performance chart. Neither is presented as an official
benchmark-index history.

| Driver | Exact source ID | Input | Quote currency | EUR operation | Earliest observed source level | Access |
|---|---|---|---:|---|---:|---|
| `euro_equities` | `SXR7.DE` | iShares Core MSCI EMU UCITS ETF EUR (Acc), daily adjusted level | EUR | Identity | 2010-03-11 | Yahoo listing data |
| `global_dm_ex_emu` | `CM9.PA` | Amundi MSCI World ex EMU UCITS ETF Acc, daily adjusted level | EUR | Identity | 2009-06-16 | Yahoo listing data |
| `em_equities` | `EUNM.DE` | iShares MSCI EM UCITS ETF USD (Acc), EUR listing, daily adjusted level | EUR | Identity | 2009-10-20 | Yahoo listing data |
| `euro_govt_bond_7_10` | `SXRQ.DE` | iShares EUR Govt Bond 7-10yr UCITS ETF EUR Acc, daily adjusted level | EUR | Identity | 2009-11-25 | Yahoo listing data |
| `euro_ig_credit` | `D5BG.DE` | Xtrackers II EUR Corporate Bond UCITS ETF 1C, daily adjusted level | EUR | Identity | 2010-02-23 | Yahoo listing data |
| `euro_high_yield` | iShares portfolio `251843` / `IE00B66F4759 NAV performance` | iShares EUR High Yield Corporate Bond UCITS ETF, daily NAV-based performance growth with gross income reinvested | EUR | Identity | 2010-09-03 | Public iShares product-page performance chart |
| `global_govt_bond_eur_hedged` | `DBZB.DE` | Xtrackers II Global Government Bond UCITS ETF 1C EUR Hedged, daily adjusted level | EUR | Identity; hedge is native to the share class | 2008-10-20 | Yahoo listing data |
| `em_hc_bond_eur_hedged` | `XEMB.DE` | Xtrackers II USD Emerging Markets Bond UCITS ETF 1C EUR Hedged, daily adjusted level | EUR | Identity; hedge is native to the share class | 2008-05-06 | Yahoo listing data |
| `gold` | `GLD` | SPDR Gold Shares, daily adjusted level | USD | Divide daily USD level by `DEXUSEU` before sampling | 2004-11-18 | Yahoo listing data |
| `commodities` | `EXXY.DE` | iShares Diversified Commodity Swap UCITS ETF (DE), daily adjusted level | EUR | Identity | 2008-01-02 | Yahoo listing data |
| `cash` | FRED `ECBDFR` | ECB deposit facility rate, daily annual rate in percent | EUR | Daily ACT/360 compounding, then period-end sampling | 1999-01-01 | FRED; underlying ECB statistics |
| `fx_eurusd` | FRED `DEXUSEU` | Daily USD per EUR level | USD per EUR | Log difference; retained as a risk driver | 1999-01-04 | FRED |

The observed source floors describe the remote histories returned during the
August 2026 source audit; they are not promises by the providers. The iShares
Euro-HY series is the latest-starting input in the shared contract. Its history
from 2010-09-03 supplies the prior level needed for the first January 2011
return.

## Return and currency semantics

- ETF and FX inputs are finite positive levels; asset and FX outputs are log
  returns.
- Listing currency, not the fund name or base currency, determines whether a
  spot conversion is applied.
- `GLD` is converted at the daily level with EUR/USD quoted as USD per EUR:
  `level_EUR = level_USD / EURUSD`. This is equivalent to
  `r_EUR = r_USD - r_EURUSD` in log returns.
- `DBZB.DE` and `XEMB.DE` are native EUR-hedged share classes. No Euribor,
  U.S.-rate input, or modeled carry hedge is added to their returns.
- Cash builds one daily ACT/360 total-return level. Each calendar day uses only
  the last `ECBDFR` fixing known at the start of that day; monthly and weekly
  returns are then sampled from the same level path. No future fixing or
  backfill is used.
- `ECBDFR` is the overnight ECB deposit-facility policy rate. It is not
  3-month Euribor, does not preserve the legacy cash series, and is not a
  directly investable retail cash return.
- Friday holidays use the last available fresh observation on or before Friday.
- Monthly and weekly aggregation selects levels first and differences second.

Both profiles return asset drivers in EUR. If their simulated paths are passed
through `data.fx_mapping.apply_fx_mapping`, set
`input_currency_state="eur_converted"`; otherwise FX would be applied twice.

## Adjusted-price, NAV, and income boundary

Accumulating Yahoo share classes minimize dependence on separate cash-
distribution reconstruction. The Euro-HY driver is not built from the
distributing `EUNW.DE` Yahoo history: it uses iShares portfolio `251843` / ISIN
`IE00B66F4759`, an issuer-reported NAV-performance growth series with gross
income reinvested. This removes the known missing-dividend problem in the early
2011-2012 Yahoo history.

`auto_adjust=True` makes the nine Yahoo inputs adjusted-price proxies. It does
not turn them into official benchmark total-return series and does not eliminate
tracking difference, fees, withholding-tax effects, market-price/NAV
differences, or provider correction risk. The iShares series is NAV-based rather
than a traded market-price history, so it has a different valuation basis from
the Yahoo inputs; that difference is part of the profile contract.

## Access, publication, and repository boundary

`daily_proxy_2011` means that no paid data order is required for the documented
research workflow. It does **not** mean that every observation is open-licensed.
Yahoo/yfinance data and the public iShares product-data endpoint are free-access
research sources, but neither provides an Open-Data or guaranteed publication-
rights contract. Yahoo's terms restrict redistribution and yfinance itself
provides no data license. The iShares endpoint is undocumented, has no
availability SLA, and may change without notice. Raw observations and caches
must not be committed or redistributed.

Locally persisted monthly and weekly panel snapshots are provider-derived
outputs rather than redistributable source data. `download_return_panel` writes
them, together with their provenance manifests, into separate `monthly/` and
`weekly/` folders below the gitignored `data/downloads/daily_proxy_2011/` tree.
Neither snapshots nor manifests belong in version control.

ECB statistics are reusable subject to the ECB's attribution and modification
conditions. FRED series can also carry source-specific terms, so provenance and
source citations must remain with publication artifacts. Provider access today
does not guarantee future continuity, and the repository does not grant or
imply rights to publish raw observations or derived exhibits.

The free MSCI chart portal is deliberately not a source for
`daily_proxy_2011`: MSCI index terms restrict reuse and redistribution. The
Bundesbank Euribor endpoint is also excluded because EMMI licensing can apply
to historical, commercial, and derived use even though the endpoint is
technically accessible without placing an order.
