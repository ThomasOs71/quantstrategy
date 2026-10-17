"""Evidence-led author handoff, including a corrected original Figure 5."""

from __future__ import annotations

import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from series.from_assumptions_to_portfolios.block1_scenarios.article3_reporting import sha256_file, write_json


def _value(frame: pd.DataFrame, column: str = "value", **filters) -> float:
    subset = frame
    for key, value in filters.items():
        subset = subset.loc[subset[key] == value]
    values = subset[column].dropna().to_numpy(dtype=float)
    if not len(values):
        raise ValueError(f"report evidence is missing: {filters}, column={column}")
    return float(values.mean())


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(value).replace("|", "/") for value in row) + " |")
    return "\n".join(lines)


def _stage_baseline(run_dir: Path, parent: Path) -> None:
    """Copy only compact, derived baseline evidence; never source panels/paths."""
    baseline = run_dir / "baseline"
    baseline.mkdir()
    names = ["article3_results.md", "article_claims.csv", "figure_captions.md", "resolved_config.yaml",
             "run_metadata.json", "run_status.json", "input_contract.json", "source_reconciliation.md"]
    names += [f"tables/{name}.csv" for name in (
        "correlation_errors", "correlation_matrices", "tail_cooccurrence", "acf_errors", "acf_curves",
        "block_length_tradeoff", "edge_weighting_summary", "path_diversity", "terminal_return_summary",
        "path_risk_summary", "monte_carlo_stability", "reference_prefix_convergence",
    )]
    figures = sorted((parent / "figures").glob("fig*.png")) + sorted((parent / "figures").glob("fig*.svg"))
    figures += sorted((parent / "figures" / "plot_data").glob("fig*.csv"))
    for path in figures:
        if path.stem != "fig05_terminal_outcomes":
            names.append(path.relative_to(parent).as_posix())
    for name in names:
        source = parent / name
        if not source.is_file():
            raise FileNotFoundError(f"required baseline handoff artifact missing: {name}")
        target = baseline / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    # Retain the original reference ScenarioSet/lineage hashes without copying
    # path arrays. This is the parent's manifest, not the corrected bundle's.
    shutil.copyfile(parent / "artifact_manifest.json", baseline / "original_artifact_manifest.json")
    corrections = []
    for suffix, subdir in ((".png", ""), (".svg", ""), (".csv", "plot_data/")):
        name = f"figures/{subdir}fig05_terminal_outcomes{suffix}"
        source, target = run_dir / name, baseline / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        corrections.append({"relative_path": f"baseline/{name}", "original_parent_sha256": sha256_file(parent / name),
                            "corrected_sha256": sha256_file(target),
                            "reason": "canonical_experiment_filter_restores_missing_MBB_reference_series"})
    write_json(run_dir / "figure5_correction.json", {"original_parent_unchanged": True,
        "corrected_plot_rows": 8, "series": ["historical_rolling", "reference_L04"], "files": corrections})


def write_research_handoff(
    run_dir: Path, *, tables: dict[str, pd.DataFrame], parent: Path,
    metadata: dict[str, Any], figure_index: dict[str, str],
) -> None:
    """Write all conclusions from actual numeric evidence; smoke is explicit."""
    _stage_baseline(run_dir, parent)
    mode = metadata["run_mode"]
    warning = "SMOKE ONLY — smaller simulation ensembles and 19 outer resamples; do not use these supplemental numbers as article findings.\n\n" if mode == "smoke" else ""
    analytic = tables["analytic_preservation"]
    squared_metric = "squared_demeaned_acf_rmse_non_cash11_lags1_13"
    acf = {length: _value(analytic, metric=squared_metric, L=length) for length in (1, 4, 8, 13, 26, 52)}
    corr = {length: _value(analytic, metric="weekly_correlation_mae_all12", L=length) for length in (1, 4, 8, 13, 52)}
    selector = tables["automatic_block_lengths"]
    raw = selector.loc[(selector["transform"] == "raw") & (selector.family == "non_cash")]
    squared = selector.loc[(selector["transform"] == "squared_demeaned") & (selector.family == "non_cash")]
    raw_range = f"{raw.circular_length.min():.2f}–{raw.circular_length.max():.2f}"
    squared_range = f"{squared.circular_length.min():.2f}–{squared.circular_length.max():.2f}"
    ljung = tables["ljung_box"]
    counts = []
    for transform in ("raw", "squared_demeaned"):
        for lag in sorted(ljung.lag.unique()):
            subset = ljung.loc[(ljung["transform"] == transform) & (ljung.lag == lag) & (ljung.family == "non_cash11")]
            counts.append({"transform": transform, "lag": int(lag), "rejecting drivers after Holm": int(subset.reject_holm.sum()),
                           "tested drivers": len(subset)})
    count_table = pd.DataFrame.from_records(counts)
    resolution = tables["resolution_summary"]
    fine = _value(resolution, L=4, asset="global_dm_ex_emu", grid="four_week_endpoints", metric="weekly_mdd_q95")
    coarse = _value(resolution, L=4, asset="global_dm_ex_emu", grid="four_week_endpoints", metric="coarse_mdd_q95")
    calendar = _value(resolution, L=4, asset="global_dm_ex_emu", grid="synthetic_last_friday_per_month", metric="coarse_mdd_q95")
    paired_gap = _value(resolution, L=4, asset="global_dm_ex_emu", grid="four_week_endpoints", metric="paired_mdd_gap_mean")
    understated = _value(resolution, L=4, asset="global_dm_ex_emu", grid="four_week_endpoints", metric="fraction_mdd_understated_gt_1pp")
    terminal_difference = resolution.loc[resolution.metric == "max_abs_terminal_log_difference", "value"].max()
    intervals = tables["source_uncertainty_intervals"]
    contrast_intervals = intervals.loc[(intervals.metric == squared_metric) & (intervals.comparator_L > 0)].copy()
    contrast_display = contrast_intervals[["outer_mean_block_length", "comparator_L", "point_estimate", "ci_lower", "ci_upper"]].copy()
    for column in ("point_estimate", "ci_lower", "ci_upper"):
        contrast_display[column] = contrast_display[column].map(lambda value: f"{value:.6f}")
    corrected = pd.read_csv(run_dir / "figures" / "plot_data" / "fig05_terminal_outcomes.csv")
    global_term = corrected.loc[corrected.asset == "global_dm_ex_emu"].copy()
    terminal_display = global_term[["reference", "n_paths", "q05", "median", "q95"]].copy()
    for column in ("q05", "median", "q95"):
        terminal_display[column] = terminal_display[column].map(lambda value: f"{100 * value:.2f}%")

    claims = []
    def claim(identity, text, numbers, path, row_keys, boundary, figure="", status="computed_result"):
        claims.append({"claim_id": identity, "claim_text_en": text, "status": status,
                       "numbers_and_units": numbers, "table_path": path, "row_keys_and_columns": row_keys,
                       "figure_path": figure, "interpretation_boundary": boundary,
                       "run_mode": mode, "input_file_sha256": metadata["input_file_sha256"]})
    claim("S01", "Longer blocks reduce squared-return temporal-preservation error on this frozen panel.",
          f"Exact expected non-cash squared-ACF RMSE lags1..13: {acf}", "tables/analytic_preservation.csv", squared_metric,
          "This metric improves beyond L=4; it does not select an overall scenario optimum.", "figures/supp01_block_length_tradeoff.png")
    claim("S02", "Longer non-circular blocks change source occupancy and the implied one-week correlations.",
          f"Exact expected all12 correlation MAE: {corr}", "tables/analytic_preservation.csv", "metric=weekly_correlation_mae_all12",
          "Synchronization preserves observed vectors, not exact uniform sampling weights for L>1.", "figures/supp01_block_length_tradeoff.png")
    claim("S03", "The corrected automatic selector suggests different lengths for raw and squared-return mean targets.",
          f"Non-cash CBB raw range={raw_range} weeks; squared range={squared_range}; n=782; cube-root floor=9.",
          "tables/automatic_block_lengths.csv", "family=non_cash; transform=raw/squared_demeaned",
          "Continuous stationary/CBB estimates, not exact MBB, quantile, or drawdown optimal lengths.", "figures/supp02_target_specific_lengths.png")
    claim("S04", "Four-week aggregation retains terminal wealth but can understate within-path drawdown.",
          f"Global DM L4 mean replicate MDD q95: weekly={fine:.8f}, 4w={coarse:.8f}; mean paired gap={paired_gap:.8f}; P(gap>0.01)={understated:.8f}; max terminal log error={terminal_difference:.3g}.",
          "tables/resolution_summary.csv", "L=4; asset=global_dm_ex_emu; grid=four_week_endpoints",
          "This is controlled coarsening of the same paths, not a comparison of separately fitted weekly/monthly models.", "figures/supp05_time_resolution.png")
    claim("S05", "Ljung-Box diagnostics distinguish raw-return and squared-return serial dependence.",
          count_table.to_json(orient="records"), "tables/ljung_box.csv", "family=non_cash11; p_holm; reject_holm",
          "Conventional chi-square tests; stationarity/regularity and heavy-tail caveats apply. Rejection is not forecastability.")
    claim("S06", "Source-sample uncertainty is assessed separately from simulation Monte-Carlo variation.",
          f"{metadata['research_config']['outer_resamples']} synchronized outer resamples for each mean block in [13,26,52]; paired L4 contrasts.",
          "tables/source_uncertainty_intervals.csv", f"metric={squared_metric}; comparator_L>0",
          "Exploratory percentile intervals under suitable stationarity; not simultaneous intervals or out-of-sample validation.", "figures/supp06_source_uncertainty.png")
    claim("S07", "Rolling and terminal correlations can depend on block length even when same-week vector synchronization is unchanged.",
          "104-week rolling windows and terminal horizons52/156/260; all grid replicates and both terminal return types exported.",
          "tables/rolling_diagnostics.csv; tables/terminal_correlations.csv", "rolling_correlation; cumulative_log/simple_compounded",
          "Different correlations are different estimands, not evidence that one model is true.", "figures/supp03_rolling_correlations.png")
    claim("S08", "L=4 is an explicit short-block reference, not an empirically established universal optimum.",
          "H260,L4: 65 independently drawn blocks,195/259 within-block adjacent pairs,64 joins; L52:5 drawn blocks and731 overlapping candidate starts.",
          "baseline/tables/block_length_tradeoff.csv", "used_blocks_per_path; candidate_block_starts; within_fraction",
          "The data favor longer lengths for squared-ACF preservation; do not force a narrative that all targets select four weeks.",
          "figures/supp01_block_length_tradeoff.png", status="construction_identity_and_limitation")
    term_numbers = "; ".join(f"{row.reference}: q05={row.q05:.8f}, median={row.median:.8f}, q95={row.q95:.8f}, n={row.n_paths}" for row in global_term.itertuples())
    claim("F05", "The corrected Figure 5 contains historical windows AND the MBB-L=4 reference distribution.",
          term_numbers, "baseline/figures/plot_data/fig05_terminal_outcomes.csv", "asset=global_dm_ex_emu; q05/median/q95; decimal_per_year",
          "New end values assembled from old weekly vectors; a wider distribution is not validated future uncertainty.", "baseline/figures/fig05_terminal_outcomes.png")
    supplementary_claims = pd.DataFrame.from_records(claims)
    supplementary_claims.to_csv(run_dir / "tables" / "supplementary_claims.csv", index=False)
    previous = pd.read_csv(parent / "article_claims.csv")
    for column in ("table_path", "figure_path"):
        previous[column] = previous[column].fillna("").map(lambda path: f"baseline/{path}" if path else "")
    previous["evidence_origin"] = "frozen_Set_A_baseline"
    supplementary_claims["evidence_origin"] = "additive_research"
    pd.concat([previous, supplementary_claims], ignore_index=True).to_csv(run_dir / "article_claims.csv", index=False)

    memo = f"""# Article 3: block length, correlation and time resolution

{warning}## Bottom line for the article

There is no target-free optimal block length. Set A uses synchronized, fixed,
overlapping, **non-circular Moving Block Bootstrap**, with replacement. Its L=4
reference remains unchanged. It provides local continuity and new multiweek
combinations, but does not preserve all volatility memory or validate future
probabilities. This supplement adds evidence for (1) the short/long trade-off
and (2) a defensible procedure for choosing an appropriate length.

Run `{metadata['run_id']}`; parent `{metadata['parent_run_id']}`. Verified input:
782 W-FRI observations, 2011-01-07 to 2025-12-26, 12 synchronized drivers,
already transformed EUR decimal log returns. Source CSV SHA-256:
`{metadata['input_file_sha256']}`. No new downloads, fabricated values,
monthly-legacy inputs, or toy empirical data were used. The calendar for the coarsening comparison is
explicitly synthetic; the RETURN VECTORS are from the verified real panel.

## 1. The short/long trade-off — what the data actually show

For squared demeaned returns, exact expected panel ACF RMSE at lags 1–13,
excluding cash, is {acf[1]:.6f} at L=1, {acf[4]:.6f} at L=4,
{acf[8]:.6f} at L=8, {acf[13]:.6f} at L=13, {acf[26]:.6f} at L=26,
and {acf[52]:.6f} at L=52. L=4 improves on L=1, but longer lengths improve
this preservation metric further. No aggregate quality score or empirical
U-shaped total-MSE curve is estimated here. See `tables/analytic_preservation.csv` and supplement Figure 1.

Long blocks have costs: fewer independently drawn chunks per simulated path
(260 at L=1, 65 at L=4, 5 at L=52), more retention of existing sequences, and
greater non-circular edge underweighting. Exact expected one-week correlation
MAE across all drivers is {corr[4]:.6f} at L=4 versus {corr[52]:.6f} at L=52.
However, L=52 still has **731 overlapping candidate starts**, not just fifteen
or five available historical blocks. The number of draws per path is not an
estimate of the effective independent sample size of the source history.

The textbook bias–variance trade-off concerns the MSE of a specified bootstrap
estimator (for example its variance estimate): short blocks miss relevant
dependence; very long blocks can increase estimator uncertainty. It is not
equivalent to “longer blocks always have more dispersed scenario returns.”
Our recombination/occupancy diagnostics illustrate practical scenario costs;
they do not constitute an estimated universal bias-plus-variance curve.

## 2. How to choose an appropriate block length

Define the target BEFORE choosing the length. For mean-return uncertainty,
examine raw-return dependence; for volatility persistence, examine squared
demeaned returns; for correlations, use correlation influence functions and
rolling/terminal correlation checks; for path risk, assess drawdown stability.

The corrected Politis–White plug-in diagnostic suggests non-cash circular
lengths of {raw_range} weeks for raw means, and {squared_range} for squared
means. These are different statistical targets, and stationary/CBB estimates
are not a certified fixed-MBB or drawdown optimum. The implemented selector
is documented in [arch](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.optimal_block_length.html).
The cube-root heuristic gives floor(782^(1/3))=9; the unknown process- and
target-dependent constant is NOT thereby one. Full estimates, including
stationary lengths, cash separately, and three correlation-influence targets,
are in `tables/automatic_block_lengths.csv`.

Use those estimates to define a search region, then inspect the existing
L={{1,2,4,8,13,26,52}} grid jointly. Report which objectives improve and which
costs grow. Check stability across seeds and outer source resamples. If the
priority is volatility-cluster preservation, this evidence argues for testing
longer blocks rather than declaring L=4 sufficient. If L=4 is retained for
local continuity and flexible recombination, call it an explicit modeling
compromise and show its residual temporal-preservation error. There is no
predeclared tolerance or utility function here that calibrates a single winner.

HHJ/NPPI would be useful after defining a scalar target and MSE/coverage
criterion, with additional pilot/subsample choices; neither is implemented
or claimed to have selected L=4. See the [blocklength implementation documentation](https://alec-stashevsky.github.io/blocklength/).
No measured seasonal period is asserted, and no generic “multiple of the
season plus one” rule is applied. The four-week length is not chosen because
calendar months exist.

## 3. Correlation is not a single object

`weekly_correlation_matrices.csv` contains the pooled one-week Pearson
matrices. Full-run regeneration is reconciled against the parent grid to
absolute tolerance 2e-12. `rolling_diagnostics.csv` adds 104-week correlations
within each path, with q05/q50/q95 for the three predeclared asset pairs, for
every L and replicate (supplement Figure 3). Historical 104-week windows
each enter once; overlapping windows are not independent observations.

`terminal_correlations.csv` adds across-path correlations at 52, 156 and
260 weeks, BOTH cumulative log returns and compounded simple returns.
The two representations need not have identical Pearson correlations.
Historical overlapping horizon windows are descriptive, not independent
replications or future ground truth (supplement Figure 4).

## 4. Weekly blocks are not monthly observations

An L=4 MBB block retains all four synchronized weekly return vectors and
their order. A four-week total retains their log sum but cannot reconstruct
the internal path. Controlled coarsening of the SAME L=4 simulated paths
gives mean replicate-specific global-DM MDD q95 of {fine:.2%} weekly,
{coarse:.2%} at four-week endpoints, and {calendar:.2%} at the last Friday
per synthetic calendar month. The mean paired weekly-minus-4w drawdown gap
is {100 * paired_gap:.2f} percentage points; {understated:.2%} of paths
lose more than one percentage point of visible drawdown.

Across all grid checks, maximum absolute difference in terminal log wealth
is {terminal_difference:.3g} (floating-point round-off). Thus terminal wealth
is preserved, while intraperiod extremes may be hidden. Four weeks are NOT
one calendar month: there are 65 four-week groups in a 260-week path. The
other grid groups the existing weekly steps by a declared artificial W-FRI
calendar and is not a daily-source calendar-month-end dataset. This comparison
does not say monthly data are useless or lose “all information,” and does
not compare two independently fitted bootstrap models.

## 5. Statistical diagnostics and uncertainty

Ljung–Box raw and squared-demeaned checks use only the 782 historical rows,
not millions of simulated observations. The Holm family is 11 non-cash
drivers x 3 fixed lags, separately for each transform; cash is separate.
These are conventional chi-square diagnostics with stationarity/regularity
assumptions, and heavy tails/heteroskedasticity can affect calibration. The
[statsmodels method documentation](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.acorr_ljungbox.html)
describes this calculation. Counts below do not establish forecastability.

{_markdown_table(count_table, list(count_table.columns))}

Outer uncertainty uses {metadata['research_config']['outer_resamples']} stationary-bootstrap source
resamples of length 782 for each mean block length 13,26,52, with synchronized
columns, geometric block lengths and circular wrapping. **This outer sampler
does not replace Set A's non-circular fixed-block scenario sampler.** In each
outer draw, expected-MBB preservation scores are computed analytically; there
is no inner Monte-Carlo noise. Contrasts are paired in the same outer draw.

95% exploratory percentile intervals for squared-ACF RMSE differences
(L=4 minus comparator; negative favors L=4 on this metric):

{_markdown_table(contrast_display, list(contrast_display.columns))}

The intervals describe historical-sample uncertainty under appropriate
stationarity assumptions, are not simultaneous, and do not prove a future
distribution or an optimal scenario length. Five original grid repetitions
measure a different quantity: Monte-Carlo variation conditional on one
fixed input. Raw outer draws and all correlation/raw-ACF intervals are
exported, including findings that do not favor four weeks.

## 6. Corrected Figure 5 — replace the old files

The old exact filter on the display `reference` label omitted the MBB rows:
the stored label is `reference_L04_L04_r0`, not `reference_L04`. The corrected
plot selects the canonical `experiment=reference_L04`, validates every
asset/series/quantile, and fails on missing or duplicate evidence.

The supplied eight-row plot file now contains both series for all four assets.
Global DM ex-EMU, 260 weeks, annualized geometric returns (52 steps/year):

{_markdown_table(terminal_display, list(terminal_display.columns))}

Use `baseline/figures/fig05_terminal_outcomes.png` and its corresponding CSV
in this NEW bundle. The frozen parent's old files remain unchanged for
auditability. This is an example of new end values from old weekly building
blocks, not evidence that wider dispersion is correct future uncertainty.

## Audit boundary and handoff

Read `CHATGPT_HANDOVER.md` and `block_length_selection.md` before drafting.
`article_claims.csv` combines baseline and supplementary claims with evidence
paths and interpretation limits. Six original exhibits (Figure 5 corrected),
six supplemental exhibits, plot data, selected baseline tables, input hashes,
code hashes, runtime and seed settings are included in `article3_author_bundle.zip`.
The reference ScenarioSet and complete lineage remain in the original local
run; they are not copied to the compact writing bundle. No source-return
panel or individual scenario path is in the ZIP. Source availability does
not grant redistribution/publication rights; review those rights before
uploading the derived bundle to an external service or publishing figures.
"""
    (run_dir / "article3_results.md").write_text(memo, encoding="utf-8")

    selection = f"""# Choosing a good block length — proposed Article-3 framing

{warning}## Two distinct questions

1. What dependence must survive? Too-short blocks truncate temporal dependence;
   same-week cross-asset synchronization alone does not retain volatility memory.
2. What does long retention cost? Fewer independent draws per path, less temporal
   recombination, non-circular edge distortion, and target-specific estimator
   variance. Scenario dispersion is not itself estimator variance.

## A reproducible procedure

For a common long-run-variance estimation problem, a leading asymptotic MSE
form is C_bias/L^2 + C_variance*L/n. Minimizing this approximation balances
decreasing squared bias and increasing estimator variance, giving a length
proportional to n^(1/3). Its constants depend on the process, target and
bootstrap variant; this is not a universal law for drawdowns or quantiles,
nor a measured MSE curve for our simulated scenarios.

1. Fix the horizon, scalar statistical target or portfolio/path-risk objective,
   return representation, frequency, and acceptable preservation/stability error.
   Do this before inspecting which length appears favorable.
2. Diagnose historical raw, squared-return and relevant cross-asset dependence.
   Keep cash separate and do not interpret white raw-return ACF as white variance.
3. Use a corrected plug-in selector to suggest a search region for the stated
   scalar target. Its stationary/CBB estimate is not silently an MBB length.
4. Evaluate the grid with the SAME verified input and scenario counts. Inspect
   ACF error, source occupancy, local/terminal correlations, drawdown and effective
   recombination. Do not choose solely the widest return distribution.
5. Separate source-sample uncertainty from simulation-seed variability. Consider
   regime/stationarity sensitivity; a block selector cannot repair nonstationarity.
6. Choose and document a compromise against declared criteria. Use formal HHJ/
   NPPI MSE/coverage calibration only if the estimand and inference problem match.
   An out-of-sample predictive claim would require a separate evaluation design.

## What this evidence permits now

L=4 is a useful SHORT-BLOCK reference with 75.29% constructive adjacent pairs
retained inside blocks and 65 drawn chunks per 260-week path. It improves temporal
preservation over synchronized L=1, but longer blocks improve the squared-ACF
metric. Raw-mean CBB suggestions: {raw_range} weeks; squared-mean suggestions:
{squared_range}. The cube-root heuristic is nine weeks for n=782, not proof of
four or nine. These diagnostics do not produce a single calibrated winner.

Recommended article sentence: “We use four-week blocks as an explicit local-
continuity/recombination reference, then test longer and shorter alternatives.
The suitable length depends on the structure and target we want to retain;
four weeks is not a universal optimum, particularly for volatility persistence.”

Sources: [corrected selector](https://bashtage.github.io/arch/bootstrap/generated/arch.bootstrap.optimal_block_length.html);
[HHJ/PWSD/NPPI](https://alec-stashevsky.github.io/blocklength/).
"""
    (run_dir / "block_length_selection.md").write_text(selection, encoding="utf-8")
    captions = "# Figure inventory\n\nThe six baseline figures are in `baseline/figures/`; Figure 5 is corrected there.\n\n"
    captions += "\n\n".join(f"## {name}\n\n{caption}\n\nPNG/SVG and `figures/plot_data/{name}.csv` are included." for name, caption in figure_index.items())
    (run_dir / "figure_captions.md").write_text(captions + "\n", encoding="utf-8")
    handoff = f"""# ChatGPT writing handoff — Article 3

{warning}## Read in this order

1. This file and `article3_results.md`: consolidated current evidence and limits.
2. `block_length_selection.md`: the two requested article messages and selection procedure.
3. `article_claims.csv`: check every quantitative sentence against its evidence.
4. `figure_captions.md` and figures/plot data as needed.
5. `baseline/article3_results.md`: original Set-A context, not a superseding claim of optimality.

Current supplementary run: `{metadata['run_id']}`. Frozen parent:
`{metadata['parent_run_id']}`. All real return data are the verified 782-row EUR
W-FRI log-return panel. No raw input or individual paths are included here.

## Editorial priorities supplied by the author

- Demonstrate the short/long-block trade-off accurately.
- Explain how to choose a good length for a declared target, not a generic optimum.
- Explain that L=4 retains weekly rows, unlike aggregating to one monthly return.
- Use corrected Figure 5 as “new end values from old building blocks,” not as
  proof that the simulated probability distribution is true future uncertainty.

## Essential corrections and prohibited overclaims

- Replace the OLD Figure 5 and its four-row CSV. The canonical files HERE at
  `baseline/figures/fig05_terminal_outcomes.*` contain both announced series.
- L=4 was specified as a provisional reference, NOT statistically optimized.
- Longer blocks preserve squared-return ACF better in this grid. Report this
  honestly rather than presenting L=4 as the winner on every metric.
- Plug-in estimates differ by target and by stationary versus circular bootstrap.
  They do not automatically calibrate this non-circular fixed MBB.
- L=52 has 731 OVERLAPPING candidate starts; “only fifteen blocks are available”
  would be wrong. Five block draws per path is a different number.
- One-week, 104-week rolling-window, and 52/156/260-week terminal correlations are
  different quantities. Log-sum and simple-compounded Pearson correlations differ.
- Grid-seed repetitions show Monte-Carlo variation. Outer source bootstrap
  intervals show a separate, conditional historical-sample uncertainty diagnostic.
- Ljung–Box is a conventional serial-dependence diagnostic, not a forecastability
  test, stationarity certificate, or model validation.
- Four-week groups are not calendar months; last-Friday-month groups use a declared
  synthetic calendar on the unchanged weekly paths, not daily month-end data.
- Coarsening retains exact terminal wealth but can hide intraperiod drawdowns;
  monthly observations are not thereby useless or stripped of all information.
- No out-of-sample superiority, validated future probabilities, or publication
  data rights are established by this research.

The original six figures, corrected Figure 5, six supplementary figures, plot
data and selected research tables are inside this self-contained writing ZIP.
Use the consolidated root claims; `baseline/article_claims.csv` preserves the
older local-context claim register. Source/ScenarioSet/lineage hashes and seed
contracts are documented in metadata; scenario arrays stay in the frozen local
research run. Check source rights before external upload or publication.
`baseline/original_artifact_manifest.json` preserves the parent's source-array
and lineage hashes, including the OLD Figure 5 hashes for audit only;
`figure5_correction.json` identifies the corrected figure hashes.
"""
    (run_dir / "CHATGPT_HANDOVER.md").write_text(handoff, encoding="utf-8")
    prompt = """Use the attached Article-3 handoff as evidence for drafting the final article.
Read CHATGPT_HANDOVER.md, article3_results.md, block_length_selection.md and
article_claims.csv first. Focus on (1) the short/long-block trade-off and
(2) a target-specific procedure for choosing a good length. Keep L=4 an explicit
reference/compromise, not an empirically certified optimum. Use the corrected
Figure 5 with both historical and MBB series. Distinguish weekly synchronization,
temporal dependence, local and terminal correlations, simulation noise, and
source uncertainty. Preserve null/contrary findings, the exact sampling method,
units and horizons. Do not claim forecasting superiority or validated future
probabilities. Every number must be linked to a supplied table/row; if evidence
is absent, say so rather than inventing it. Start with an evidence-checked outline
and proposed claim-to-figure mapping before drafting polished prose.
"""
    (run_dir / "START_PROMPT.txt").write_text(prompt, encoding="utf-8")
    dictionary = {
        "returns": "EUR decimal log returns; simple terminal returns=expm1(sum(log returns)); 52 weekly steps/year",
        "analytic_preservation": "Exact infinite-ensemble expected Set-A ACF/correlation preservation conditional on source panel; cash excluded from panel ACF",
        "rolling_diagnostics": "104-step correlation and 52-step volatility window quantiles; no cross-path windows; unique historical windows",
        "terminal_correlations": "Across scenarios or overlapping historical horizon windows; both log sums and simple compounded returns; not weekly rho",
        "resolution_summary": "Same paths, retained endpoints, initial wealth included; paired drawdown gap is weekly minus coarse; decimal loss height",
        "automatic_block_lengths": "Continuous corrected PW stationary/CBB lengths for scalar mean/second-moment/correlation-influence targets, not MBB optimality",
        "ljung_box": "Historical 782 rows; Holm across11 noncash drivers x3 lags separately per transform; cash separate",
        "source_uncertainty": "Exploratory95% outer stationary-bootstrap percentile intervals, paired contrasts; not simultaneous; no inner simulation",
        "grid_reconciliation": "Full run regenerated with original101 seed stream; pooled matrices match frozen baseline to2e-12",
        "Figure5": "8 rows=4 assets x2 experiments, q05/median/q95 annualized geometric return; exact canonical experiment selection",
        "limits": "No calibrated unique optimum, stationary-regime certification, out-of-sample prediction validation or data redistribution permission",
    }
    write_json(run_dir / "data_dictionary.json", dictionary)


def build_research_bundle(run_dir: Path, *, parent: Path) -> Path:
    """Self-contained derived-evidence ZIP; no raw financial data or path cubes."""
    destination = run_dir / "article3_author_bundle.zip"
    selected = [path for path in run_dir.rglob("*") if path.is_file()
                and path.suffix in {".md", ".txt", ".json", ".csv", ".yaml", ".png", ".svg"}
                and path.name != "bundle_integrity.json"]
    if len(list((run_dir / "baseline" / "figures").glob("fig*.png"))) != 6:
        raise ValueError("author bundle must contain all six original figures, including corrected Figure 5")
    if len(list((run_dir / "figures").glob("supp*.png"))) != 6:
        raise ValueError("author bundle must contain six supplementary figures")
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(selected):
            archive.write(path, arcname=path.relative_to(run_dir).as_posix())
    return destination
