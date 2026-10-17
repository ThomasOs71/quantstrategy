# Article 3 — additive block-length research and Figure-5 correction

This extends the existing weekly Set-A run; it does not alter its L=4 reference,
ScenarioSet, source panel, seed stream, or legacy monthly outputs. The user's
two publication priorities are the short/long-block trade-off and a defensible,
target-specific way to choose a good length. No universal optimum is asserted.

## Offline execution

Use Python 3.11 and the existing core dependencies. The corrected plug-in
selector has a separate research-only dependency, `arch==7.2.0`:

```powershell
python -m pip install -r requirements_article3_research.txt
python -m series.from_assumptions_to_portfolios.block1_scenarios.run_article3_block_length --mode smoke
python -m series.from_assumptions_to_portfolios.block1_scenarios.run_article3_block_length --mode full
```

The default is full. The runner verifies the existing W-FRI snapshot and its
manifest, finds a matching full Set-A baseline, checks the entire resolved
Set-A config and relevant baseline artifact hashes, then regenerates the
original seven-length sensitivity grid using the unchanged experiment-101
PCG64 stream. Full-run pooled correlations must match all 35 parent grid
matrices to absolute tolerance 2e-12. A missing or incompatible baseline,
unverified input, or missing optional selector dependency is an explicit error;
there is no download, replacement-source, or synthetic-data fallback.

To select a particular baseline, pass `--parent-run <directory>`. Each run
writes a new unique directory under ignored `artifacts/article3_set_a/`.
Smoke uses 64 paths, one replicate and 19 outer resamples; its claims and
author-review status are explicitly marked unsuitable for article findings.

## Additional numerical contracts

- Input: unchanged decimal EUR log returns, W-FRI, 782 rows x 12 drivers.
  `fx_eurusd` remains a non-investable driver; cash is separate for statistical
  comparisons. No new annualization convention is introduced.
- Same-week pooled correlations, 104-week within-path rolling correlations,
  and across-path terminal correlations are different estimands. Terminal
  correlations cover 52/156/260 weeks in both cumulative-log and compounded-
  simple return representations. Historical rolling-window references enter
  once per unique relevant window; their overlap is disclosed.
- Resolution effects compare the SAME paths at every step, four-week
  endpoints, and last-Friday-per-month endpoints on a declared synthetic
  W-FRI calendar. The latter is not a daily calendar-month-end input. Log
  aggregation preserves terminal wealth, and coarsening cannot increase
  maximum drawdown when initial wealth is included. Paired gaps are reported.
- Corrected Politis–White/Patton lengths are produced by the installed `arch`
  implementation for raw, squared-demeaned and correlation-influence series.
  They target scalar long-run variance of the respective mean/influence.
  Stationary/CBB estimates are not labeled fixed-MBB or drawdown optima.
  Recommendations are exported as continuous values, not silently rounded or
  adopted as a production parameter. The cube-root rule is a separate heuristic.
- Ljung–Box uses only the historical sample, fixed lags 4/13/52, no fitted-model
  degree-of-freedom adjustment, and Holm correction across the 33 non-cash tests
  separately per transform. Cash has its own family. Conventional chi-square
  assumptions, heteroskedasticity and heavy-tail caveats are explicit.
- Source uncertainty: 999 synchronized outer stationary-bootstrap resamples
  of 782 weeks for each mean block 13/26/52. This outer sampler wraps circularly
  and has geometric lengths, but NEVER replaces Set A's non-circular fixed MBB.
  Seed stream: `[20260913,301,outer_mean_block_length,resample_id]`.
  Paired preservation-error contrasts are L4-L1, L4-L8, L4-L13. Percentile
  intervals are exploratory, not simultaneous or out-of-sample validation.
- Exact expected-MBB ACFs include source-edge occupancy, block-offset means,
  independent joins, lag pair counts and final-block truncation. Exhaustive
  small synthetic start libraries validate the estimator. Analytic inner
  scores avoid conflating outer historical uncertainty with simulation noise.

## Figure 5 defect and fix

The previous plot selected exact `reference=reference_L04`; generated reference
summary labels are `reference_L04_L04_r0`. This dropped the MBB observations
while retaining a legend entry. It now selects canonical
`experiment=reference_L04`, requires both experiments for all four assets and
q05/median/q95, rejects duplicates/nonfinite or unordered quantiles, and
exports the full eight-row plot data with source labels and path counts.

The additive author bundle stages the six original figures but replaces
Figure 5's PNG, SVG and CSV with the corrected rendering of the FROZEN parent
summary table. The old run stays unchanged. `figure5_correction.json` records
old and corrected hashes; `baseline/original_artifact_manifest.json` preserves
original ScenarioSet and complete-lineage hashes without copying path arrays.

## Handoff artifacts

- `article3_results.md`: consolidated empirical results and interpretation limits.
- `article_claims.csv`: baseline plus new claim register with evidence paths.
- `block_length_selection.md`: two publication messages and selection procedure.
- `CHATGPT_HANDOVER.md` and `START_PROMPT.txt`: writing-chat handoff.
- Six supplemental PNG/SVG figures and their exact plot data, plus six original
  figures with corrected Figure 5, selected tables, runtime/config/code hashes.
- Raw outer-resample draws are retained locally as CSV.GZ; interval summaries
  are included in the compact writing ZIP.
- `article3_author_bundle.zip`: self-contained derived evidence, excluding raw
  return panels, individual scenario cubes and complete lineage rows. Reference
  arrays and complete lineage remain in the original local research run.

Review underlying source/publication rights before uploading externally. No
automatic publication approval or future-distribution validation is implied.

## Verification

```powershell
python -m pytest tests/test_article3_set_a.py tests/test_article3_block_length.py -q
python -m pytest -q
git diff --check
```

Tests use labeled synthetic fixtures only. If Windows sandbox rename rights in
the default temporary folder prevent snapshot tests, rerun with a verified
nonexistent `--basetemp` directory below ignored `data/downloads/` (the loader
intentionally rejects other in-repository snapshot destinations); do not
modify the data loader to mask an environment-specific permission failure.

## Implemented files and verified local handoff (2026-10-13)

The added implementation is separated into numerical diagnostics
(`article3_block_length_diagnostics.py`), orchestration
(`run_article3_block_length.py`), plotting
(`article3_block_length_plots.py`) and handoff/reporting
(`article3_block_length_reporting.py`), under `block1_scenarios/`.
`article3_plots.py` contains the corrected Figure-5 selector and completeness
guard. Configuration, optional dependencies and regression tests are in
`configs/article3_block_length_weekly.yaml`,
`requirements_article3_research.txt` and `tests/test_article3_block_length.py`.

The final local run is
`artifacts/article3_set_a/article3_block_length_v1_full_20261013T150735Z_d95e8a6c/`.
The original Set-A run remains unchanged. Validation performed:

- Focused Article-3 suites: 61 passed.
- Complete offline suite with a fresh, previously nonexistent `--basetemp`
  under ignored `data/downloads/`: 136 passed, 3 skipped. Skips are the opt-in
  raw MSCI check, legacy live-data QA, and daily-proxy live-data QA. No live
  download path was exercised. Initial default-temp runs hit Windows rename
  restrictions; choosing the data-governance-compatible test directory resolved
  the environment issue without modifying the loader.
- Verified-input smoke and full research execution completed. All 35 grid
  matrices match the frozen parent; max absolute discrepancy <1e-16.
- Reference NPZ and full-lineage hashes match the original manifest; the first
  3,000 L=4 source-index paths are exactly the reference archive prefix.
- All 129 intervals independently recomputed from 128,871 raw score/contrast
  records (2,997 outer source resamples); all interval groups have 999 finite
  values. These are exploratory intervals, not a coverage-calibration claim.
- All 84 listed artifact hashes verified, source-code hashes match current
  producers, ZIP CRC and external SHA-256 verified. Both Figure-5 series and
  the exact frozen-reference quantiles are present in the eight-row CSV.
- Changed figures visually inspected, imports/compile smoke and tracked plus
  new-file whitespace checks passed. Only Git's normal LF/CRLF notices occurred.

The writing ZIP is about 4.3 MB and contains aggregate evidence only; raw
source panels, individual scenario paths and full lineage rows are excluded.
Underlying licensing/publication rights, stationarity assumptions and lack of
out-of-sample validation remain explicit limitations. No commit or push was
performed by this research extension.
