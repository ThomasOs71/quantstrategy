# From Assumptions to Portfolios

*From Assumptions to Portfolios* is the flagship QuantStrategy series for building portfolios from explicit assumptions: Scenario Atlas -> Views (EP) -> Portfolio Construction -> Decision Layer. The series is empirical-first: concepts are introduced through data, diagnostics, and reproducible implementation patterns before being generalized.

## Block Overview

| Block | Focus | Status |
|---|---|---|
| Block 1 | Scenario Atlas and terminal return foundations | Active from July |
| Block 2 | Views, Entropy Pooling, and practitioner belief integration | Follows after Block 1 |
| Block 3 | Portfolio construction and optimization | Follows after Block 2 |
| Block 4 | Post-processing, robustness checks, and decision preparation | Follows after Block 3 |
| Block 5 | End-to-end workflow and publication-ready implementation | Follows after Block 4 |

## Toolkit Cross-Reference Convention

Flaggschiff-Artikel verlinken so auf Toolkit:

*"For a no-code practitioner version of this idea, see [Toolkit-Artikel]."*

Toolkit-Artikel verlinken so auf Flaggschiff:

*"The full technical version of this idea is developed in [Flaggschiff-Artikel/Block]."*

Use this wording consistently starting with the first article.

## Reproducibility Note

Blocks 2.4 and 2.5 demonstrate the concept, but deliberately do not publish the complete proprietary technology described in the Block 2 plan, section 3. The series still aims to make the research workflow reproducible where possible, but this boundary should be explicit so readers do not mistake the conceptual implementation for a full disclosure of proprietary production technology.

The original Article-2 results remain frozen as the `monthly_legacy` baseline.
Frequency-aware research reruns are stored separately for the
`daily_proxy_2011` monthly and weekly panels; they do not overwrite or silently
reinterpret the published legacy artifacts.

`daily_proxy_2011` uses the same loader API and one daily-source engine for both
monthly and W-FRI research inputs. Its strict historical contracts through
December 2025 are:

- monthly: 2011-01-31 through 2025-12-31, 180 rows x 12 drivers;
- weekly: 2011-01-07 through 2025-12-26, 782 rows x 12 drivers.

The research profile requires no paid data order or manually placed licensed
files. Nine market inputs are no-cost Yahoo/yfinance proxies rather than
official index histories or Open Data. Euro HY instead uses the official
iShares performance-chart series for portfolio `251843` / ISIN `IE00B66F4759`,
on a EUR NAV basis with gross income reinvested from 2010-09-03. This avoids the
incomplete early Yahoo dividend history for `EUNW.DE`. The public iShares
endpoint is nevertheless undocumented, has no availability SLA, and carries no
Open-Data or guaranteed publication-rights contract. Cash uses the overnight
ECB deposit-facility rate (`ECBDFR`), not 3-month Euribor, while the foreign-
bond proxies (`DBZB.DE` and `XEMB.DE`) are native EUR-hedged share classes and
require no modeled carry input.

The Article-2 diagnostic runner now has an explicit frequency contract. Monthly
uses 12 periods/year, 12-/24-month rolling windows and 6/12-month Ljung-Box
lags. Weekly uses 52 periods/year, 52-/104-week rolling windows and 26/52-week
Ljung-Box lags. Weekly ACF tables and figures show every lag from 1 through 52.
Schweizer-Wolff retains twelve predeclared weekly lags that approximate months 1
through 12, keeping its multiple-test family comparable with Monthly.
Full-sample tail thresholds and state terciles remain
retrospective descriptive diagnostics, not live signals.

Local ignored outputs use this layout:

```text
articles/outputs/article2/
├── monthly_legacy/monthly/
└── daily_proxy_2011/
    ├── monthly/
    └── weekly/
```

Comparisons must separate two effects: legacy monthly versus proxy monthly is
primarily a source/proxy change; proxy monthly versus proxy weekly is the
frequency change. Legacy monthly versus proxy weekly is not a pure frequency
comparison. Weekly returns must still not enter scenario code whose horizon
axis is defined in months. Any article using `daily_proxy_2011` must state the
proxy, frequency, adjusted-price, and publication-rights limitations explicitly.
