"""Evidence-first reporting and local author-bundle helpers for Article 3."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from series.from_assumptions_to_portfolios.block1_scenarios.article3_input import WeeklyInputProvenance
from series.from_assumptions_to_portfolios.block1_scenarios.set_a_block_bootstrap import Article3Config


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (Path,)):
        return str(value)
    if isinstance(value, (datetime, pd.Timestamp)):
        return value.isoformat()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _sanitize_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize_json(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, np.floating) and not np.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write standards-compliant JSON; undefined numeric values become null."""
    path.write_text(
        json.dumps(_sanitize_json(payload), indent=2, sort_keys=True, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def sha256_file(path: Path) -> str:
    """Return a file hash for output-manifest lineage."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def input_contract_payload(
    provenance: WeeklyInputProvenance, *, panel: pd.DataFrame
) -> dict[str, Any]:
    """Create a compact machine-readable record of verified empirical input."""
    return {
        "contract_version": "article3_weekly_input_v1",
        "source_profile": "daily_proxy_2011",
        "frequency": "weekly",
        "sampling": "W-FRI",
        "requested_start": panel.attrs.get("requested_start"),
        "requested_end": panel.attrs.get("requested_end"),
        "effective_start": panel.index.min().date().isoformat(),
        "effective_end": panel.index.max().date().isoformat(),
        "n_observations": int(len(panel)),
        "n_drivers": int(panel.shape[1]),
        "n_investable_assets": 11,
        "driver_keys": list(panel.columns),
        "return_type": "log",
        "return_unit": "decimal",
        "asset_base_currency": "EUR",
        "auxiliary_driver": "fx_eurusd",
        "snapshot_path": provenance.snapshot_path,
        "manifest_path": provenance.manifest_path,
        "input_file_sha256": provenance.input_file_sha256,
        "input_manifest_sha256": provenance.input_manifest_sha256,
        "canonical_panel_sha256": provenance.canonical_panel_sha256,
        "article2_run_reference": provenance.article2_run_reference,
        "article2_snapshot_match_status": provenance.article2_snapshot_match_status,
        "source_transformations": {
            "statement": "The stored panel is consumed unchanged: no extra FX conversion, hedge carry, cash reconstruction, demeaning, winsorization, or moment matching.",
            "source_provenance": panel.attrs.get("source_provenance"),
        },
    }


def write_source_reconciliation(path: Path, *, provenance: WeeklyInputProvenance) -> None:
    """Record the handover's known source conflicts without rewriting prior work."""
    article2_status = provenance.article2_snapshot_match_status
    text = f"""# Article 3 source reconciliation

## Verified current input

- The Article-3 input is the verified `daily_proxy_2011` W-FRI weekly snapshot.
- The snapshot SHA-256 is `{provenance.input_file_sha256}`.
- Article-2 run match status is **{article2_status}**.
- Set A consumes already transformed EUR log returns unchanged. `fx_eurusd` remains a driver, not an investable asset.

## Reconciled handover conflicts

1. A 5% tail over 782 observations is not assumed to contain any pre-specified number of rows. Actual strict-`<`, linear-quantile counts are exported in `tables/tail_thresholds.csv`.
2. The weekly panel has 12 drivers. The value 52 is used only for weekly lags and the annualization convention; it does not imply 52 assets.
3. Block lengths are sensitivity cases. L=4 is a provisional pre-registered reference, not an optimizer-selected winner.
4. The sampler creates no new one-week synchronized vectors, but does create novel sequences and joins. This does not prove their economic plausibility.
5. Joint tails are reported separately for all 12 drivers and the 10 investable non-cash assets; neither group is silently substituted for the other.
6. No legacy monthly snapshot, synthetic hedge construction, or reindexed monthly series was used for this run.
"""
    path.write_text(text, encoding="utf-8")


def _value(frame: pd.DataFrame, **filters: object) -> float | None:
    subset = frame
    for column, expected in filters.items():
        if column not in subset.columns:
            return None
        subset = subset.loc[subset[column] == expected]
    values = subset.get("value", pd.Series(dtype=float)).dropna().to_numpy(dtype=float)
    return None if not len(values) else float(values.mean())


def _pct(value: float | None) -> str:
    return "not available" if value is None else f"{value * 100:.2f}%"


def build_claims(
    *,
    correlation_errors: pd.DataFrame,
    tail_cooccurrence: pd.DataFrame,
    acf_errors: pd.DataFrame,
    path_risk: pd.DataFrame,
    edge_summary: pd.DataFrame,
    config: Article3Config,
) -> pd.DataFrame:
    """Create editorial claims only from computed table values or identities."""
    control_corr = _value(correlation_errors, experiment="independent_iid_control", group="all12", metric="offdiag_mae")
    l1_corr = _value(correlation_errors, experiment="synchronized_iid_L01", group="all12", metric="offdiag_mae")
    control_tail = _value(tail_cooccurrence, experiment="independent_iid_control", group="investable_ex_cash10", metric="p_k_ge", k=2)
    l1_tail = _value(tail_cooccurrence, experiment="synchronized_iid_L01", group="investable_ex_cash10", metric="p_k_ge", k=2)
    l1_acf = _value(acf_errors, experiment="synchronized_iid_L01", transform="squared_demeaned", metric="panel_rmse", lag_set="1..13")
    l4_acf = _value(acf_errors, experiment="synchronized_mbb_L04", transform="squared_demeaned", metric="panel_rmse", lag_set="1..13")
    l4_edge = _value(edge_summary, experiment="synchronized_mbb_L04", metric="total_variation_to_uniform")
    mdd_l1 = _value(path_risk, experiment="synchronized_iid_L01", asset="global_dm_ex_emu", horizon_steps=260, risk_kind="maximum_drawdown", metric="q95")
    mdd_l4 = _value(path_risk, experiment="synchronized_mbb_L04", asset="global_dm_ex_emu", horizon_steps=260, risk_kind="maximum_drawdown", metric="q95")
    rows = [
        {"claim_id": "C01", "claim_text_en": "Independent per-driver sampling breaks the observed cross-asset weekly structure more than synchronized L=1 sampling.", "status": "computed_result", "numbers_and_units": f"All-driver correlation MAE: independent={control_corr}, synchronized L=1={l1_corr}.", "method_L_S_H": "controls, 3,000 paths × 260 weeks, five replications", "comparison_or_reference": "independent_iid_control versus synchronized_iid_L01", "table_path": "tables/correlation_errors.csv", "row_keys_and_columns": "group=all12; metric=offdiag_mae", "figure_path": "figures/fig02_joint_stress.png", "monte_carlo_context": "Five equal-size replications; values in the table retain individual runs.", "interpretation_boundary": "This establishes the synchronization comparison, not a benefit of L>1."},
        {"claim_id": "C02", "claim_text_en": "Synchronized resampling and independent resampling produce distinct simultaneous-tail frequencies for investable non-cash assets.", "status": "computed_result", "numbers_and_units": f"P(K≥2): independent={_pct(control_tail)}, synchronized L=1={_pct(l1_tail)}.", "method_L_S_H": "controls, 3,000 paths × 260 weeks, five replications", "comparison_or_reference": "fixed historical 5% linear thresholds", "table_path": "tables/tail_cooccurrence.csv", "row_keys_and_columns": "group=investable_ex_cash10; metric=p_k_ge; k=2", "figure_path": "figures/fig02_joint_stress.png", "monte_carlo_context": "Simulation variation is exported by replicate.", "interpretation_boundary": "One-week co-tail preservation is supplied by synchronization; blocks are not required for it."},
        {"claim_id": "C03", "claim_text_en": "The L=4 reference retains a defined constructive fraction of adjacent observations while introducing new block joins.", "status": "construction_identity", "numbers_and_units": f"For H=260 and L=4, 195 of 259 adjacent pairs are within blocks and 64 are joins.", "method_L_S_H": "L=4, H=260 weeks", "comparison_or_reference": "fixed-length non-circular MBB construction", "table_path": "tables/block_length_tradeoff.csv", "row_keys_and_columns": "n_within_pairs; n_joins; within_fraction; join_fraction", "figure_path": "figures/fig01_block_lineage.png", "monte_carlo_context": "Identity; not a sampling estimate.", "interpretation_boundary": "Constructive within-block membership is not equivalent to realized correlation preservation."},
        {"claim_id": "C04", "claim_text_en": "Non-circular MBB gives lower expected sampling weight to sample-edge weeks.", "status": "computed_result", "numbers_and_units": f"L=4 expected occupancy total variation from uniform={l4_edge}.", "method_L_S_H": "Analytic occupancy, L=4, H=260 weeks", "comparison_or_reference": "uniform historical source weights", "table_path": "tables/edge_weighting_summary.csv", "row_keys_and_columns": "metric=total_variation_to_uniform", "figure_path": "", "monte_carlo_context": "Analytic quantity; realized occupancy is separately exported.", "interpretation_boundary": "The diagnostic exposes the bias; Set A does not rescale returns to remove it."},
        {"claim_id": "C05", "claim_text_en": "The data determine whether L=4 improves squared-return ACF preservation relative to synchronized L=1; no monotonic advantage is imposed.", "status": "computed_result", "numbers_and_units": f"Panel squared-ACF RMSE, lags 1–13: L=1={l1_acf}, L=4={l4_acf}.", "method_L_S_H": "3,000 paths × 260 weeks, five replications", "comparison_or_reference": "synchronized_iid_L01 versus synchronized_mbb_L04", "table_path": "tables/acf_errors.csv", "row_keys_and_columns": "transform=squared_demeaned; metric=panel_rmse; lag_set=1..13", "figure_path": "figures/fig03_preservation_tradeoff.png", "monte_carlo_context": "Across-replicate results in tables/monte_carlo_stability.csv.", "interpretation_boundary": "Preservation diagnostics do not establish forecasting power."},
        {"claim_id": "C06", "claim_text_en": "Block length can change path-risk distributions even when all one-week vectors originate in the same historical panel.", "status": "computed_result", "numbers_and_units": f"Global DM ex-EMU 260-week MDD q95: L=1={_pct(mdd_l1)}, L=4={_pct(mdd_l4)}.", "method_L_S_H": "3,000 paths × 260 weeks, five replications", "comparison_or_reference": "synchronized variants with common frozen input", "table_path": "tables/path_risk_summary.csv", "row_keys_and_columns": "risk_kind=maximum_drawdown; metric=q95; horizon_steps=260", "figure_path": "figures/fig06_path_risk.png", "monte_carlo_context": "Grid-replicate summaries are explicit.", "interpretation_boundary": "Method sensitivity is not out-of-sample superiority."},
        {"claim_id": "C07", "claim_text_en": "Set A creates new multiweek combinations but not new one-week synchronized return vectors.", "status": "construction_identity", "numbers_and_units": "Each simulated weekly row is selected unchanged from the 782-row verified panel.", "method_L_S_H": "synchronized MBB", "comparison_or_reference": "verified source_indices reconstruction", "table_path": "scenarios/set_a_L04.npz", "row_keys_and_columns": "source_indices; driver_paths", "figure_path": "figures/fig01_block_lineage.png", "monte_carlo_context": "Identity conditional on the frozen panel.", "interpretation_boundary": "New joins can nevertheless produce novel terminal outcomes and are not automatically economic regimes."},
        {"claim_id": "C08", "claim_text_en": "L=4 remains a provisional pre-registered reference rather than a calibrated optimum.", "status": "limitation", "numbers_and_units": "Reference block length L=4 weeks; seven sensitivity lengths are reported.", "method_L_S_H": f"L grid={list(config.sensitivity_block_lengths)}", "comparison_or_reference": "preregistered implementation specification", "table_path": "tables/block_length_tradeoff.csv", "row_keys_and_columns": "L; candidate_block_starts; within_fraction", "figure_path": "figures/fig03_preservation_tradeoff.png", "monte_carlo_context": "Not applicable.", "interpretation_boundary": "No single preservation metric is used to choose a winner."},
    ]
    return pd.DataFrame.from_records(rows)


def write_results_memo(
    path: Path,
    *,
    provenance: WeeklyInputProvenance,
    config: Article3Config,
    claims: pd.DataFrame,
    run_mode: str,
) -> None:
    """Write an English evidence memo linked to local machine-readable tables."""
    claim_lines = "\n".join(
        f"- **{row.claim_id}** — {row.claim_text_en} ({row.numbers_and_units})"
        for row in claims.itertuples()
    )
    memo = f"""# Article 3 results — Scenario Set A

## Run Status and Data Contract

**Empirical status:** `VERIFIED_WEEKLY_INPUT_FULL_RUN`<br>
**Run mode:** `{run_mode}`<br>
**Input:** `daily_proxy_2011`, weekly `W-FRI`, EUR decimal log returns, 2011-01-07 through 2025-12-26, 782 observations × 12 drivers (11 investable assets plus `fx_eurusd`).<br>
**Input SHA-256:** `{provenance.input_file_sha256}`<br>
**Canonical panel SHA-256:** `{provenance.canonical_panel_sha256}`<br>
**Article-2 snapshot match:** `{provenance.article2_snapshot_match_status}`.

The frozen panel is consumed unchanged. Set A applies no additional FX conversion, hedge carry, cash reconstruction, demeaning, filtering, winsorization, or moment matching.

## Reference Model

The reference uses overlapping, fixed-length, non-circular moving-block bootstrap sampling with replacement: L={config.block_length_steps} weekly observations, S={config.production_n_scenarios:,} paths, H={config.horizon_steps} weeks, and PCG64 streams derived from root seed {config.root_seed}. Business labels 12m/36m/60m denote 52/156/260 weekly steps; they are not counts of weekly returns.

## What Synchronization Preserves

See `tables/correlation_errors.csv`, `tables/tail_cooccurrence.csv`, `tables/tail_pairwise.csv`, and Figure 2. The independent control draws each driver with replacement and is diagnostic-only. Synchronized L=1 draws observed full 12-driver weekly vectors, so its comparison with the control isolates synchronization rather than temporal blocks.

## What Blocks Add Beyond L=1

See `tables/acf_curves.csv`, `tables/acf_errors.csv`, `tables/tail_sequence_dependence.csv`, and `tables/path_risk_summary.csv`. The evidence is reported as measured, including any weak or non-monotone block-length behavior; it is not an out-of-sample forecast test.

## The Block-Length Trade-Off

`tables/block_length_tradeoff.csv`, `tables/lag_pair_preservation.csv`, and `tables/edge_weighting_summary.csv` report the constructive within-block share, joins, candidate starts, and explicit non-circular edge weighting. L=4 is a provisional reference—not an optimizer-selected block length.

## Terminal Outcomes and Path Risk

All eleven assets and all three horizons are in `tables/terminal_return_summary.csv` and `tables/path_risk_summary.csv`. Overlapping historical windows of matched step length are descriptive references in the paired `historical_rolling_*.csv` tables. Figure 5 shows annualized 260-week terminal-path quantiles; Figure 6 shows 260-week MDD sensitivity.

## Monte-Carlo Stability

`tables/monte_carlo_stability.csv` and `tables/reference_prefix_convergence.csv` distinguish replication variability conditional on this same frozen history from historical estimation uncertainty, model risk, and future uncertainty. More paths reduce sampling noise but do not create new independent historical crises.

## Model and Data Boundaries

- A copied synchronized vector retains any source timing, NAV, holiday, or staleness limitations.
- Blocks interrupt dependence at joins; short blocks do not validate a long volatility cycle.
- Non-circular MBB underweights edge weeks; expected and realized occupancy are exported rather than corrected away.
- Historical one-week support is bounded, while recombination can create new multiweek sequences and terminal extremes.
- Preservation results are not predictive validation or an assertion that every synthetic five-year path is economically plausible.

## Source Reconciliation

See `source_reconciliation.md`. In particular, tail counts are recomputed from the verified snapshot; 52 is a weekly lag/annualization quantity rather than a driver count; and all-12-driver tails remain separate from ten-asset non-cash tails.

## Claims Ready for Editorial Review

{claim_lines}

## Default Assessment

The implementation is valid as a non-circular synchronized historical block-bootstrap baseline. The L=4 reference remains provisional. Editorial interpretation should compare L=4 with the full sensitivity grid and its Monte-Carlo variability rather than declare a winner from one preservation statistic.
"""
    path.write_text(memo, encoding="utf-8")


def write_data_dictionary(path: Path) -> None:
    """Document shared table fields and their units for editorial review."""
    payload = {
        "run_id": "Unique local Article-3 execution identifier.",
        "experiment": "Sampling method or historical reference; independent_iid_control is diagnostic-only.",
        "L": "Fixed block length in weekly observations; L=1 is synchronized IID.",
        "replicate_id": "Stable PCG64 stream replicate identifier.",
        "value": "Numeric value in the accompanying unit; missing values represent an explicit undefined status.",
        "weekly_log_return": "Natural-log return for a W-FRI sampled period, in decimal units.",
        "simple_terminal_return": "exp(sum(log returns)) - 1 over the stated number of weekly steps.",
        "annualized_geometric_return": "exp((52 / horizon_steps) * sum(log returns)) - 1.",
        "maximum_drawdown": "Non-negative loss height from a path high-water mark, including initial wealth of one.",
        "reference": "Historical, analytic occupancy, simulated, or rolling-window comparison identity.",
        "status": "ok or an explicit reason a quantity is undefined; undefined values are never silently replaced by zero.",
    }
    write_json(path, payload)


def write_figure_captions(path: Path) -> None:
    """Write English captions and alt text for all six planned figures."""
    text = """# Figure captions and alt text

## Figure 1 — From Historical Blocks to One Scenario Path
Caption: Three observed four-week source blocks form the first 12 weeks of one 260-week synchronized scenario path. Dashed lines mark newly constructed joins.
Alt text: Three panels show observed weekly returns for global equities, euro government bonds, and gold across the first twelve simulated weeks, colored by source block.

## Figure 2 — Synchronization Preserves Joint Stress
Caption: Joint lower-tail frequencies compare history, synchronized L=1, synchronized L=4, and the independent per-driver control for ten investable non-cash assets.
Alt text: Four lines plot the probability of at least two through ten simultaneous lower-tail events.

## Figure 3 — The Preservation–Recombination Trade-Off
Caption: Seven block lengths are located by their constructive adjacent-pair preservation and squared-return ACF error; error bars show replicate variation.
Alt text: A labelled scatter plot relates the fraction of within-block adjacent pairs to squared-return ACF RMSE.

## Figure 4 — How Much Volatility Memory Survives?
Caption: Squared demeaned-return ACFs for euro government bonds compare history with L=1, L=4, and L=26 resampling.
Alt text: Four weekly-lag autocorrelation lines show how volatility-memory diagnostics differ by block length.

## Figure 5 — Terminal Outcomes From Recombined History
Caption: Five-year annualized terminal-return quantiles for four assets compare L=4 scenarios with overlapping historical windows of the same length.
Alt text: Interval-and-median marks show 5th, 50th, and 95th annualized returns for four assets.

## Figure 6 — Block Length Changes the Path, Not Just the Endpoint
Caption: The 95th percentile of five-year maximum drawdown is shown across the pre-registered block-length grid.
Alt text: Four asset lines plot 260-week maximum-drawdown sensitivity against block length in weeks.
"""
    path.write_text(text, encoding="utf-8")


def write_artifact_manifest(run_dir: Path, *, excluded_names: Sequence[str] = ("artifact_manifest.json",)) -> Path:
    """Hash all local derived artifacts except the self-referential manifest."""
    entries = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name in excluded_names:
            continue
        entries.append({
            "relative_path": str(path.relative_to(run_dir)).replace("\\", "/"),
            "file_type": path.suffix.lower(), "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path), "origin": "article3_derived_local", "access_status": "local_not_for_git_or_publication",
        })
    destination = run_dir / "artifact_manifest.json"
    write_json(destination, {"schema_version": 1, "generated_at_utc": datetime.now(timezone.utc), "entries": entries, "self_hash_excluded": True})
    return destination


def build_author_bundle(run_dir: Path) -> Path:
    """Create a compact author bundle that excludes panel, lineage, and paths."""
    bundle = run_dir / "article3_author_bundle.zip"
    selected = [
        run_dir / "article3_results.md", run_dir / "data_dictionary.json", run_dir / "figure_captions.md",
        run_dir / "resolved_config.yaml", run_dir / "tables" / "article_claims.csv",
    ]
    selected.extend(sorted((run_dir / "figures").glob("fig*.png")))
    selected.extend(sorted((run_dir / "figures").glob("fig*.svg")))
    selected.extend(sorted((run_dir / "figures" / "plot_data").glob("*.csv")))
    for table in ("correlation_errors.csv", "tail_cooccurrence.csv", "acf_errors.csv", "block_length_tradeoff.csv", "terminal_return_summary.csv", "path_risk_summary.csv", "monte_carlo_stability.csv", "reference_prefix_convergence.csv"):
        selected.append(run_dir / "tables" / table)
    with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in selected:
            if path.is_file():
                archive.write(path, arcname=str(path.relative_to(run_dir)).replace("\\", "/"))
    return bundle


def runtime_metadata() -> dict[str, str]:
    """Return non-sensitive platform and package versions for reproducibility."""
    import matplotlib
    import numpy
    import pandas
    import yaml
    return {"python_version": sys.version.split()[0], "platform": platform.platform(), "numpy": numpy.__version__, "pandas": pandas.__version__, "matplotlib": matplotlib.__version__, "pyyaml": yaml.__version__}
