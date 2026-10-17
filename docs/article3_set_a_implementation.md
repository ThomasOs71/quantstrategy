# Article 3 — Scenario Set A implementation

This implementation follows `ARTICLE3_CODEX_HANDOVER_WEEKLY_V2.md` specification version 2.0. It is a synchronized, non-circular moving-block bootstrap (MBB) over the verified weekly `daily_proxy_2011` return-panel snapshot.

## Contract

- Frequency: weekly Friday observations (`W-FRI`), decimal EUR log returns.
- Verified panel: 782 observations from 2011-01-07 through 2025-12-26; 12 drivers comprising 11 investable assets and `fx_eurusd`.
- Production reference: 10,000 scenarios, 260 weekly steps, overlapping MBB with replacement and block length `L=4`.
- Sensitivity grid: `L in {1, 2, 4, 8, 13, 26, 52}`, 3,000 scenarios and five independent replicates per length.
- Control: independent-IID driver sampling, 3,000 scenarios and five replicates.
- Root seed: `20260913`, using NumPy `PCG64` substreams.

`L=4` is a provisional reference, not an empirically selected optimum. The implementation reports the full grid, replicate variability, and an independent-IID synchronization control rather than presenting a single block length as superior.

## Run and outputs

Run from the repository root:

```powershell
python -m series.from_assumptions_to_portfolios.block1_scenarios.run_article3_set_a --config configs/article3_set_a_weekly.yaml --run-mode full
```

The runner rejects an unverified snapshot and never falls back to synthetic data. It writes a new, timestamped, gitignored directory below `artifacts/article3_set_a/`; existing runs are never overwritten. Each full run contains the reference `L=4` scenario archive (`.npz`), compressed lineage table, all diagnostics and plot data as CSV, six PNG/SVG figures, an English result memo, claims table, reconciliation note, file manifest, and author bundle ZIP.

The scenario archive preserves simulated driver paths and sampled source indices. The lineage table records every selected source segment, its destination interval, realized source occupancy, and boundary/join diagnostics. These records allow independent reconstruction of every reference path.

## Interpretation boundary

Outputs are author-review artifacts, not an automatic publication release. The verified provider snapshot and any downstream materials derived from it remain local and must be reviewed for licensing and disclosure before external distribution.

## Additive block-length research and corrected Figure 5

The [block-length research extension](article3_block_length_research.md) adds
target-specific selector diagnostics, source uncertainty, rolling/terminal
correlation sensitivity and controlled time-grid comparisons without replacing
this reference run. Its new author bundle includes a corrected Figure 5 with
both historical and MBB reference quantiles. Frozen older bundles must not be
used for Figure 5: their exact display-label filter omitted the MBB series.
