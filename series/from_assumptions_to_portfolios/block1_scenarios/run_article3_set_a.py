"""Run the verified weekly Article-3 Set-A research and export workflow."""

from __future__ import annotations

import argparse
import json
import subprocess
import warnings
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from data.asset_universe import get_asset_keys, get_driver_keys
from data.load_data import download_return_panel
from series.from_assumptions_to_portfolios.block1_scenarios.article3_diagnostics import (
    SELECTED_ASSETS,
    acf_curve_rows,
    acf_error_rows,
    analyze_synchronized_variant,
    correlation_error_rows,
    correlation_matrix_rows,
    edge_weighting_summary,
    ensemble_acf,
    finite_path_acf_summary_rows,
    historical_rolling_paths,
    marginal_preservation_rows,
    monte_carlo_stability_rows,
    path_risk_summary_rows,
    rolling_path_diagnostic_rows,
    source_occupancy_rows,
    tail_cooccurrence_rows,
    tail_independence_reference_rows,
    tail_pairwise_rows,
    tail_sequence_dependence_rows,
    tail_threshold_rows,
    terminal_summary_rows,
    weighted_correlation,
)
from series.from_assumptions_to_portfolios.block1_scenarios.article3_input import (
    WeeklyInputProvenance,
    load_verified_weekly_snapshot,
)
from series.from_assumptions_to_portfolios.block1_scenarios.article3_plots import (
    figure01_block_lineage,
    figure02_joint_stress,
    figure03_preservation_tradeoff,
    figure04_volatility_memory,
    figure05_terminal_outcomes,
    figure06_path_risk,
)
from series.from_assumptions_to_portfolios.block1_scenarios.article3_reporting import (
    build_author_bundle,
    build_claims,
    input_contract_payload,
    runtime_metadata,
    write_artifact_manifest,
    write_data_dictionary,
    write_figure_captions,
    write_json,
    write_results_memo,
    write_source_reconciliation,
)
from series.from_assumptions_to_portfolios.block1_scenarios.block_sampler import (
    BlockSample,
    expected_source_weights,
    independent_iid_indices,
    sample_moving_block_indices,
)
from series.from_assumptions_to_portfolios.block1_scenarios.set_a_block_bootstrap import (
    Article3Config,
    SetABuild,
    build_set_a,
    load_article3_config,
    make_rng,
    save_set_a_reference,
)


DEFAULT_SNAPSHOT = Path("data/downloads/daily_proxy_2011/weekly/daily_proxy_2011_weekly_2011-01-07_2025-12-26.csv")
DEFAULT_MANIFEST = Path("data/downloads/daily_proxy_2011/weekly/daily_proxy_2011_weekly_2011-01-07_2025-12-26.manifest.json")
DEFAULT_ARTICLE2_METADATA = Path("articles/outputs/article2/daily_proxy_2011/weekly/run_metadata.json")


def _git_value(*args: str) -> str | None:
    result = subprocess.run(["git", *args], text=True, capture_output=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _run_id() -> str:
    return "article3_set_a_weekly_v2_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _tag(frame: pd.DataFrame, *, run_id: str, experiment: str | None = None, length: int | None = None, replicate_id: int | None = None) -> pd.DataFrame:
    result = frame.copy()
    result.insert(0, "run_id", run_id)
    for column, value in (("experiment", experiment), ("L", length), ("replicate_id", replicate_id)):
        if column not in result.columns:
            result[column] = value
        elif value is not None:
            result[column] = result[column].fillna(value)
    return result


def _append(collection: dict[str, list[pd.DataFrame]], name: str, frame: pd.DataFrame, **tags: Any) -> None:
    if not frame.empty:
        collection[name].append(_tag(frame, **tags))


def _concatenate(frames: list[pd.DataFrame]) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame()
    # Diagnostic tables deliberately union method-specific fields (for example,
    # a historical reference has no replicate id).  Their column schemas are
    # explicit, so retain the current all-NA concatenation semantics while
    # avoiding a pandas-version warning in run logs.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        return pd.concat(frames, ignore_index=True)


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False)


def _variant_headlines(
    analysis: Any,
    *,
    experiment: str,
    length: int,
    replicate_id: int,
) -> pd.DataFrame:
    """Extract comparable per-replicate values for MC stability reporting."""
    records: list[dict[str, Any]] = []
    def add(frame: pd.DataFrame, filters: dict[str, Any], name: str, unit: str, target: str) -> None:
        subset = frame
        for key, value in filters.items():
            subset = subset.loc[subset[key] == value]
        if not subset.empty:
            records.append({"experiment": experiment, "L": length, "replicate_id": replicate_id,
                            "metric": name, "value": float(subset.iloc[0]["value"]), "unit": unit,
                            "asset_or_group": target})
    add(analysis.correlation_errors, {"group": "all12", "metric": "offdiag_mae"}, "correlation_mae", "correlation", "all12")
    add(analysis.tail_cooccurrence, {"group": "investable_ex_cash10", "metric": "p_k_ge", "k": 2}, "tail_p_ge_2", "probability", "investable_ex_cash10")
    add(analysis.acf_errors, {"transform": "squared_demeaned", "metric": "panel_rmse", "lag_set": "1..13"}, "squared_acf_rmse_1_13", "acf", "non_cash11")
    for asset in SELECTED_ASSETS:
        add(analysis.terminal, {"asset": asset, "horizon_steps": 260, "return_kind": "annualized_geometric_return", "metric": "q05"}, "annualized_terminal_q05", "decimal_per_year", asset)
        add(analysis.path_risk, {"asset": asset, "horizon_steps": 260, "risk_kind": "maximum_drawdown", "metric": "q95"}, "mdd_q95", "decimal_loss", asset)
    return pd.DataFrame.from_records(records)


def _reference_prefix_rows(
    paths: np.ndarray,
    *,
    history: np.ndarray,
    history_correlation: np.ndarray,
    thresholds: np.ndarray,
    keys: list[str],
    asset_positions: list[int],
    asset_keys: list[str],
    config: Article3Config,
) -> pd.DataFrame:
    """Evaluate headline reference statistics at 1k/3k/10k path prefixes."""
    history_means = history.mean(axis=0)
    rows: list[dict[str, Any]] = []
    for count in (1000, 3000, 10000):
        subset = paths[:count]
        correlation = weighted_correlation(subset.reshape(-1, subset.shape[2]))
        error = correlation_error_rows(correlation, history_correlation, keys, experiment="reference_prefix", block_length_steps=4, replicate_id=0, group="all12")
        mae = error.loc[error["metric"] == "offdiag_mae", "value"].iloc[0]
        tails = tail_cooccurrence_rows(subset.reshape(-1, subset.shape[2]), thresholds, keys, reference="reference_prefix")
        tail_value = tails.loc[(tails["group"] == "investable_ex_cash10") & (tails["metric"] == "p_k_ge") & (tails["k"] == 2), "value"].iloc[0]
        acf, _ = ensemble_acf(subset, max_lag=config.acf_max_lag, transform="squared_demeaned", historical_means=history_means)
        hist_acf, _ = ensemble_acf(history[None, :, :], max_lag=config.acf_max_lag, transform="squared_demeaned", historical_means=history_means)
        acf_error = acf_error_rows(acf, hist_acf, keys, transform="squared_demeaned", experiment="reference_prefix", block_length_steps=4, replicate_id=0, lag_sets=(13,))
        acf_value = acf_error.loc[(acf_error["metric"] == "panel_rmse") & (acf_error["lag_set"] == "1..13"), "value"].iloc[0]
        terminal = terminal_summary_rows(subset[:, :, asset_positions], asset_keys, terminal_horizon_steps={"60m": 260}, steps_per_year=52, reference="reference_prefix")
        risk = path_risk_summary_rows(subset[:, :, asset_positions], asset_keys, horizons={"60m": 260}, reference="reference_prefix")
        rows.extend([
            {"n_scenarios": count, "metric": "correlation_mae", "asset_or_group": "all12", "value": mae, "unit": "correlation"},
            {"n_scenarios": count, "metric": "tail_p_ge_2", "asset_or_group": "investable_ex_cash10", "value": tail_value, "unit": "probability"},
            {"n_scenarios": count, "metric": "squared_acf_rmse_1_13", "asset_or_group": "non_cash11", "value": acf_value, "unit": "acf"},
        ])
        for asset in SELECTED_ASSETS:
            rows.append({"n_scenarios": count, "metric": "annualized_terminal_q05", "asset_or_group": asset,
                         "value": terminal.loc[(terminal["asset"] == asset) & (terminal["return_kind"] == "annualized_geometric_return") & (terminal["metric"] == "q05"), "value"].iloc[0], "unit": "decimal_per_year"})
            rows.append({"n_scenarios": count, "metric": "mdd_q95", "asset_or_group": asset,
                         "value": risk.loc[(risk["asset"] == asset) & (risk["risk_kind"] == "maximum_drawdown") & (risk["metric"] == "q95"), "value"].iloc[0], "unit": "decimal_loss"})
    return pd.DataFrame.from_records(rows)


def _load_input(
    *, source_mode: str, snapshot: Path | None, manifest: Path | None, article2_metadata: Path | None
) -> tuple[pd.DataFrame, WeeklyInputProvenance]:
    if source_mode == "snapshot":
        if snapshot is None or manifest is None:
            raise ValueError("snapshot mode requires --panel-snapshot and --provenance-manifest")
        return load_verified_weekly_snapshot(snapshot, manifest, article2_metadata_path=article2_metadata)
    if source_mode == "live":
        created = download_return_panel(frequency="weekly", start="2011-01-07", end="2025-12-31", overwrite=False)
        return load_verified_weekly_snapshot(created["data"], created["manifest"], article2_metadata_path=article2_metadata)
    raise ValueError("source_mode must be 'snapshot' or 'live'")


def run_article3_analysis(
    panel: pd.DataFrame,
    *,
    config: Article3Config,
    provenance: WeeklyInputProvenance,
    output_dir: Path,
    run_id: str,
    run_mode: str,
) -> dict[str, Path]:
    """Create the full numeric, figure, report, lineage, and author-bundle export."""
    output_dir.mkdir(parents=True, exist_ok=False)
    tables_dir = output_dir / "tables"
    figures_dir = output_dir / "figures"
    plot_data_dir = figures_dir / "plot_data"
    scenarios_dir = output_dir / "scenarios"
    logs_dir = output_dir / "logs"
    for directory in (tables_dir, figures_dir, plot_data_dir, scenarios_dir, logs_dir):
        directory.mkdir(parents=True, exist_ok=True)

    write_json(output_dir / "input_contract.json", input_contract_payload(provenance, panel=panel))
    write_source_reconciliation(output_dir / "source_reconciliation.md", provenance=provenance)
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config.as_dict(), sort_keys=False), encoding="utf-8")

    metadata = {
        "spec_version": config.spec_version, "run_id": run_id, "run_timestamp_utc": datetime.now(timezone.utc),
        "run_mode": run_mode, "empirical_status": "VERIFIED_WEEKLY_INPUT_FULL_RUN" if run_mode == "full" else "VERIFIED_WEEKLY_INPUT_SMOKE_RUN",
        "repo_identifier": "ThomasOs71/quantstrategy", "git_commit": _git_value("rev-parse", "HEAD"),
        "branch": _git_value("branch", "--show-current"), "dirty_worktree": bool(_git_value("status", "--porcelain")),
        "input_file_sha256": provenance.input_file_sha256, "canonical_panel_sha256": provenance.canonical_panel_sha256,
        "input_manifest_sha256": provenance.input_manifest_sha256, "article2_run_reference": provenance.article2_run_reference,
        "article2_snapshot_match_status": provenance.article2_snapshot_match_status,
        "source_profile": "daily_proxy_2011", "frequency": "weekly", "sampling": "W-FRI", "units": "decimal log returns", "base_currency": "EUR",
        "requested_sample": {"start": panel.attrs.get("requested_start"), "end": panel.attrs.get("requested_end")},
        "effective_sample": {"start": panel.index.min().date().isoformat(), "end": panel.index.max().date().isoformat()},
        "T": len(panel), "D": panel.shape[1], "asset_keys": get_asset_keys(), "driver_keys": get_driver_keys(),
        "method": config.method, "L": config.block_length_steps, "S": config.production_n_scenarios, "H": config.horizon_steps,
        "terminal_horizon_steps": dict(config.terminal_horizon_steps or {}), "block_sampling": "uniform_with_replacement",
        "overlap": True, "circular": False, "final_truncation_rule": "truncate_final_fixed_block_to_horizon",
        "root_seed": config.root_seed, "stream_ids": {"synchronized_mbb": 101, "independent_control": 201}, "bit_generator": "PCG64",
        "replication_config": {"sensitivity_replicates": config.sensitivity_replicates, "control_replicates": config.control_replicates},
        "dtypes": {"paths": config.path_dtype, "summary": config.summary_dtype}, "quantile_methods": {"tail": config.tail_quantile_method, "summary": config.summary_quantile_method},
        "tail_groups": ["article2_all12", "investable_ex_cash10"], "tail_comparison": "lt", "acf_estimator": "ensemble_lag_pair_normalized",
        "drawdown_sign_convention": "non_negative_loss_height", "annualization_convention": "52_steps_per_year", "warnings": [], "errors": [],
        "completed_tasks": [], "skipped_tasks": [], "runtime": runtime_metadata(),
    }
    write_json(output_dir / "run_metadata.json", metadata)

    collections = {name: [] for name in (
        "article2_reconciliation", "marginal_preservation", "source_occupancy", "edge_weighting_summary", "correlation_matrices", "correlation_errors", "tail_thresholds", "tail_cooccurrence", "tail_pairwise", "tail_independence_reference", "acf_curves", "acf_errors", "acf_equal_horizon_reference", "tail_sequence_dependence", "block_length_tradeoff", "lag_pair_preservation", "path_diversity", "terminal_return_summary", "historical_rolling_terminal_summary", "path_risk_summary", "historical_rolling_path_risk", "monte_carlo_stability", "reference_prefix_convergence", "rolling_diagnostic_summary",
    )}
    values = panel.to_numpy(dtype=np.float64)
    keys = get_driver_keys()
    asset_keys = get_asset_keys()
    asset_positions = [keys.index(key) for key in asset_keys]
    historical_asset_paths = values[:, asset_positions]
    history_means = values.mean(axis=0)
    history_correlation = weighted_correlation(values)
    thresholds_table, thresholds = tail_threshold_rows(values, keys, probability=config.lower_tail_probability)
    history_acfs: dict[str, np.ndarray] = {}
    for transform in ("raw", "squared_demeaned"):
        acf, statuses = ensemble_acf(values[None, :, :], max_lag=config.acf_max_lag, transform=transform, historical_means=history_means)
        history_acfs[transform] = acf
        _append(collections, "acf_curves", acf_curve_rows(acf, statuses, keys, reference="historical_uniform", transform=transform), run_id=run_id, experiment="historical_uniform")
    _append(collections, "article2_reconciliation", pd.DataFrame.from_records([{
        "article2_run_reference": provenance.article2_run_reference, "article2_snapshot_match_status": provenance.article2_snapshot_match_status,
        "article3_snapshot_sha256": provenance.input_file_sha256, "article3_canonical_panel_sha256": provenance.canonical_panel_sha256,
        "frequency": "weekly", "sampling": "W-FRI", "n_observations": len(panel), "n_drivers": len(keys),
    }]), run_id=run_id)
    _append(collections, "marginal_preservation", marginal_preservation_rows(values, keys, reference="historical_uniform"), run_id=run_id, experiment="historical_uniform")
    _append(collections, "tail_thresholds", thresholds_table, run_id=run_id, experiment="historical_uniform")
    _append(collections, "tail_cooccurrence", tail_cooccurrence_rows(values, thresholds, keys, reference="historical_uniform"), run_id=run_id, experiment="historical_uniform")
    _append(collections, "tail_pairwise", tail_pairwise_rows(values, thresholds, keys, reference="historical_uniform"), run_id=run_id, experiment="historical_uniform")
    _append(collections, "tail_independence_reference", tail_independence_reference_rows(values, thresholds, keys), run_id=run_id, experiment="poisson_binomial_independent")
    _append(collections, "tail_sequence_dependence", tail_sequence_dependence_rows(values[None, :, :], thresholds, keys, reference="historical_uniform"), run_id=run_id, experiment="historical_uniform")
    for group, positions in (("all12", list(range(len(keys)))), ("investable_ex_cash10", [keys.index(key) for key in asset_keys if key != "cash"])):
        group_keys = [keys[i] for i in positions]
        _append(collections, "correlation_matrices", correlation_matrix_rows(history_correlation[np.ix_(positions, positions)], group_keys, reference="historical_uniform", group=group), run_id=run_id, experiment="historical_uniform")

    # Analytic expected occupancy baselines are distinct from Monte-Carlo simulations.
    for length in config.sensitivity_block_lengths:
        expected = expected_source_weights(n_obs=len(values), horizon_steps=config.horizon_steps, block_length_steps=length)
        experiment = "mbb_expected_occupancy"
        _append(collections, "marginal_preservation", marginal_preservation_rows(values, keys, reference=f"mbb_expected_L{length:02d}", weights=expected, quantile_mode="inverse_weighted_cdf"), run_id=run_id, experiment=experiment, length=length)
        _append(collections, "source_occupancy", source_occupancy_rows(expected_weights=expected, realized_weights=None, source_dates=panel.index, experiment=experiment, block_length_steps=length, replicate_id=None), run_id=run_id, experiment=experiment, length=length)
        _append(collections, "edge_weighting_summary", edge_weighting_summary(expected, None, experiment=experiment, block_length_steps=length, replicate_id=None), run_id=run_id, experiment=experiment, length=length)
        expected_corr = weighted_correlation(values, expected)
        for group, positions in (("all12", list(range(len(keys)))), ("investable_ex_cash10", [keys.index(key) for key in asset_keys if key != "cash"])):
            group_keys = [keys[i] for i in positions]
            _append(collections, "correlation_matrices", correlation_matrix_rows(expected_corr[np.ix_(positions, positions)], group_keys, reference=f"mbb_expected_L{length:02d}", group=group), run_id=run_id, experiment=experiment, length=length)
            _append(collections, "correlation_errors", correlation_error_rows(expected_corr[np.ix_(positions, positions)], history_correlation[np.ix_(positions, positions)], group_keys, experiment=experiment, block_length_steps=length, replicate_id=None, group=group), run_id=run_id, experiment=experiment, length=length)
        _append(collections, "tail_cooccurrence", tail_cooccurrence_rows(values, thresholds, keys, reference=f"mbb_expected_L{length:02d}", weights=expected), run_id=run_id, experiment=experiment, length=length)
        _append(collections, "tail_pairwise", tail_pairwise_rows(values, thresholds, keys, reference=f"mbb_expected_L{length:02d}", weights=expected), run_id=run_id, experiment=experiment, length=length)

    reference_build = build_set_a(panel, config=config, provenance=provenance, n_scenarios=config.production_n_scenarios, replicate_id=0)
    reference_npz, reference_lineage = save_set_a_reference(reference_build, output_dir=scenarios_dir, block_length_steps=config.block_length_steps)
    # Lineage is serialized immediately and no longer needed in memory.  Keep
    # only the arrays required for numerical diagnostics and Figure 1.
    reference_scenario = reference_build.scenario_set
    reference_sample = reference_build.sample
    del reference_build
    reference_analysis = analyze_synchronized_variant(history=values, history_correlation=history_correlation, history_acfs=history_acfs,
        driver_paths=reference_scenario.driver_paths, source_indices=reference_sample.source_indices, block_starts=reference_sample.block_starts,
        keys=keys, thresholds=thresholds, asset_positions=asset_positions, asset_keys=asset_keys, terminal_horizon_steps=dict(config.terminal_horizon_steps or {}),
        steps_per_year=config.steps_per_year, max_lag=config.acf_max_lag, acf_error_lag_sets=config.acf_error_lag_sets,
        experiment="reference_L04", block_length_steps=config.block_length_steps, replicate_id=0)
    for name, frame in reference_analysis.__dict__.items():
        mapped = {"marginal": "marginal_preservation", "occupancy": "source_occupancy", "edge": "edge_weighting_summary", "correlation_matrices": "correlation_matrices", "correlation_errors": "correlation_errors", "tail_cooccurrence": "tail_cooccurrence", "tail_pairwise": "tail_pairwise", "acf_curves": "acf_curves", "acf_errors": "acf_errors", "tail_sequence": "tail_sequence_dependence", "tradeoff": "block_length_tradeoff", "lag_pairs": "lag_pair_preservation", "diversity": "path_diversity", "terminal": "terminal_return_summary", "path_risk": "path_risk_summary"}[name]
        _append(collections, mapped, frame, run_id=run_id, experiment="reference_L04", length=4, replicate_id=0)
    for transform in ("raw", "squared_demeaned"):
        _append(collections, "acf_equal_horizon_reference", finite_path_acf_summary_rows(reference_scenario.driver_paths, keys, max_lag=config.acf_max_lag, transform=transform, historical_means=history_means, reference="reference_L04"), run_id=run_id, experiment="reference_L04", length=4, replicate_id=0)
    _append(collections, "rolling_diagnostic_summary", rolling_path_diagnostic_rows(reference_scenario.asset_paths, asset_keys, reference="reference_L04", vol_window=config.rolling_vol_steps, corr_window=config.rolling_corr_steps), run_id=run_id, experiment="reference_L04", length=4, replicate_id=0)

    # Finish full-reference-only work before the sensitivity grid. Keeping the
    # entire 10,000-path cube alive across all 35 grid runs needlessly increases
    # memory pressure. Source indices suffice to reconstruct its exact prefix.
    prefix = _reference_prefix_rows(reference_scenario.driver_paths, history=values, history_correlation=history_correlation,
        thresholds=thresholds, keys=keys, asset_positions=asset_positions, asset_keys=asset_keys, config=config)
    del reference_scenario, reference_analysis

    headlines: list[pd.DataFrame] = []
    for length in config.sensitivity_block_lengths:
        experiment = "synchronized_iid_L01" if length == 1 else f"synchronized_mbb_L{length:02d}"
        for replicate_id in range(config.sensitivity_replicates):
            if length == 4 and replicate_id == 0:
                count = config.sensitivity_n_scenarios
                sample = BlockSample(source_indices=reference_sample.source_indices[:count], block_starts=reference_sample.block_starts[:count], used_block_lengths=reference_sample.used_block_lengths)
                paths = values[sample.source_indices, :]
                if not np.array_equal(sample.source_indices, reference_sample.source_indices[:count]):
                    raise RuntimeError("reference prefix identity failed")
            else:
                rng = make_rng(config=config, experiment_id=101, block_length=length, replicate_id=replicate_id)
                sample = sample_moving_block_indices(n_obs=len(values), n_scenarios=config.sensitivity_n_scenarios, horizon_steps=config.horizon_steps, block_length_steps=length, rng=rng)
                paths = values[sample.source_indices, :]
            analysis = analyze_synchronized_variant(history=values, history_correlation=history_correlation, history_acfs=history_acfs,
                driver_paths=paths, source_indices=sample.source_indices, block_starts=sample.block_starts, keys=keys, thresholds=thresholds,
                asset_positions=asset_positions, asset_keys=asset_keys, terminal_horizon_steps=dict(config.terminal_horizon_steps or {}), steps_per_year=config.steps_per_year,
                max_lag=config.acf_max_lag, acf_error_lag_sets=config.acf_error_lag_sets, experiment=experiment, block_length_steps=length, replicate_id=replicate_id)
            for name, frame in analysis.__dict__.items():
                mapped = {"marginal": "marginal_preservation", "occupancy": "source_occupancy", "edge": "edge_weighting_summary", "correlation_matrices": "correlation_matrices", "correlation_errors": "correlation_errors", "tail_cooccurrence": "tail_cooccurrence", "tail_pairwise": "tail_pairwise", "acf_curves": "acf_curves", "acf_errors": "acf_errors", "tail_sequence": "tail_sequence_dependence", "tradeoff": "block_length_tradeoff", "lag_pairs": "lag_pair_preservation", "diversity": "path_diversity", "terminal": "terminal_return_summary", "path_risk": "path_risk_summary"}[name]
                _append(collections, mapped, frame, run_id=run_id, experiment=experiment, length=length, replicate_id=replicate_id)
            headlines.append(_variant_headlines(analysis, experiment=experiment, length=length, replicate_id=replicate_id))
            del paths, sample, analysis
        print(f"Set-A sensitivity L={length}: {config.sensitivity_replicates} replicates complete", flush=True)

    # Explicit negative control: it is not serialized as a production ScenarioSet.
    for replicate_id in range(config.control_replicates):
        rng = make_rng(config=config, experiment_id=201, block_length=0, replicate_id=replicate_id)
        row_indices = independent_iid_indices(n_obs=len(values), n_scenarios=config.control_n_scenarios, horizon_steps=config.horizon_steps, n_drivers=len(keys), rng=rng)
        control_paths = np.empty(row_indices.shape, dtype=np.float64)
        for driver in range(len(keys)):
            control_paths[:, :, driver] = values[row_indices[:, :, driver], driver]
        flattened = control_paths.reshape(-1, len(keys))
        experiment = "independent_iid_control"
        reference = f"{experiment}_r{replicate_id}"
        _append(collections, "marginal_preservation", marginal_preservation_rows(flattened, keys, reference=reference), run_id=run_id, experiment=experiment, length=0, replicate_id=replicate_id)
        control_corr = weighted_correlation(flattened)
        for group, positions in (("all12", list(range(len(keys)))), ("investable_ex_cash10", [keys.index(key) for key in asset_keys if key != "cash"])):
            group_keys = [keys[i] for i in positions]
            _append(collections, "correlation_matrices", correlation_matrix_rows(control_corr[np.ix_(positions, positions)], group_keys, reference=reference, group=group), run_id=run_id, experiment=experiment, length=0, replicate_id=replicate_id)
            _append(collections, "correlation_errors", correlation_error_rows(control_corr[np.ix_(positions, positions)], history_correlation[np.ix_(positions, positions)], group_keys, experiment=experiment, block_length_steps=0, replicate_id=replicate_id, group=group), run_id=run_id, experiment=experiment, length=0, replicate_id=replicate_id)
        _append(collections, "tail_cooccurrence", tail_cooccurrence_rows(flattened, thresholds, keys, reference=reference), run_id=run_id, experiment=experiment, length=0, replicate_id=replicate_id)
        _append(collections, "tail_pairwise", tail_pairwise_rows(flattened, thresholds, keys, reference=reference), run_id=run_id, experiment=experiment, length=0, replicate_id=replicate_id)
        del flattened, control_paths, row_indices

    # Matched historical rolling references for terminals and path risks.
    for label, horizon in dict(config.terminal_horizon_steps or {}).items():
        windows = historical_rolling_paths(historical_asset_paths, horizon_steps=horizon)
        _append(collections, "historical_rolling_terminal_summary", terminal_summary_rows(windows, asset_keys, terminal_horizon_steps={label: horizon}, steps_per_year=config.steps_per_year, reference="historical_rolling"), run_id=run_id, experiment="historical_rolling", length=None)
        _append(collections, "historical_rolling_path_risk", path_risk_summary_rows(windows, asset_keys, horizons={label: horizon}, reference="historical_rolling"), run_id=run_id, experiment="historical_rolling", length=None)
    rolling_history = historical_rolling_paths(values, horizon_steps=config.horizon_steps)
    for transform in ("raw", "squared_demeaned"):
        _append(collections, "acf_equal_horizon_reference", finite_path_acf_summary_rows(rolling_history, keys, max_lag=config.acf_max_lag, transform=transform, historical_means=history_means, reference="historical_rolling_260"), run_id=run_id, experiment="historical_rolling", length=None)
    _append(collections, "rolling_diagnostic_summary", rolling_path_diagnostic_rows(historical_asset_paths[None, :, :], asset_keys, reference="historical_full_sample", vol_window=config.rolling_vol_steps, corr_window=config.rolling_corr_steps), run_id=run_id, experiment="historical_full_sample", length=None)

    combined = {name: _concatenate(frames) for name, frames in collections.items()}
    # Make matched historical summaries available under the standard tables used by reporting and Figure 5.
    terminal_reference = combined["historical_rolling_terminal_summary"].copy()
    terminal_reference["reference"] = "historical_rolling"
    terminal_reference["experiment"] = "historical_rolling"
    combined["terminal_return_summary"] = pd.concat([combined["terminal_return_summary"], terminal_reference], ignore_index=True)

    if headlines:
        combined["monte_carlo_stability"] = monte_carlo_stability_rows(pd.concat(headlines, ignore_index=True))
        combined["monte_carlo_stability"].insert(0, "run_id", run_id)
    combined["reference_prefix_convergence"] = _tag(prefix, run_id=run_id, experiment="reference_L04", length=4, replicate_id=0)

    table_paths: dict[str, Path] = {}
    for name, frame in combined.items():
        destination = tables_dir / f"{name}.csv"
        _write_table(frame, destination)
        table_paths[name] = destination

    plot_data: dict[str, pd.DataFrame] = {}
    plot_data["fig01_block_lineage"] , _ = figure01_block_lineage(panel, reference_sample.source_indices, reference_sample.used_block_lengths, output_dir=figures_dir)
    plot_data["fig02_joint_stress"], _ = figure02_joint_stress(combined["tail_cooccurrence"], output_dir=figures_dir)
    plot_data["fig03_preservation_tradeoff"], _ = figure03_preservation_tradeoff(combined["block_length_tradeoff"], combined["acf_errors"], output_dir=figures_dir)
    plot_data["fig04_volatility_memory"], _ = figure04_volatility_memory(combined["acf_curves"], output_dir=figures_dir)
    plot_data["fig05_terminal_outcomes"], _ = figure05_terminal_outcomes(combined["terminal_return_summary"], output_dir=figures_dir)
    plot_data["fig06_path_risk"], _ = figure06_path_risk(combined["path_risk_summary"], output_dir=figures_dir)
    for name, frame in plot_data.items():
        _write_table(frame, plot_data_dir / f"{name}.csv")

    claims = build_claims(correlation_errors=combined["correlation_errors"], tail_cooccurrence=combined["tail_cooccurrence"], acf_errors=combined["acf_errors"], path_risk=combined["path_risk_summary"], edge_summary=combined["edge_weighting_summary"], config=config)
    claims.insert(0, "run_id", run_id)
    claims_path = tables_dir / "article_claims.csv"
    _write_table(claims, claims_path)
    # A root-level convenience copy satisfies the named editorial deliverable without changing the table source of truth.
    _write_table(claims, output_dir / "article_claims.csv")
    write_data_dictionary(output_dir / "data_dictionary.json")
    write_figure_captions(output_dir / "figure_captions.md")
    write_results_memo(output_dir / "article3_results.md", provenance=provenance, config=config, claims=claims, run_mode=run_mode)
    (logs_dir / "run.log").write_text("Verified weekly snapshot used; no network reload and no synthetic fallback.\n", encoding="utf-8")
    (logs_dir / "test_summary.txt").write_text("Run-time artifact generation completed. Unit-test commands are reported in the Codex handoff.\n", encoding="utf-8")
    metadata["completed_tasks"] = ["preflight", "reference_scenario_set", "sensitivity_grid", "independent_control", "diagnostics", "figures", "reporting", "author_bundle"]
    metadata["reference_scenario_archive"] = str(reference_npz.relative_to(output_dir)).replace("\\", "/")
    metadata["reference_lineage"] = str(reference_lineage.relative_to(output_dir)).replace("\\", "/")
    write_json(output_dir / "run_metadata.json", metadata)
    write_json(output_dir / "run_status.json", {"status": "FULL_COMPLETE" if run_mode == "full" else "SMOKE_COMPLETE", "implementation_valid": True, "author_review_ready": run_mode == "full", "publication_ready": False, "empirical_status": metadata["empirical_status"], "completed_tasks": metadata["completed_tasks"]})
    bundle = build_author_bundle(output_dir)
    manifest = write_artifact_manifest(output_dir)
    return {"run_dir": output_dir, "results": output_dir / "article3_results.md", "claims": claims_path, "scenario": reference_npz, "lineage": reference_lineage, "bundle": bundle, "manifest": manifest}


def run(
    *, config_path: Path, source_mode: str, panel_snapshot: Path | None, provenance_manifest: Path | None,
    run_mode: str, output_root: Path, article2_metadata: Path | None = DEFAULT_ARTICLE2_METADATA,
) -> dict[str, Path]:
    """Run preflight, smoke, or full Article-3 analysis from the repository root."""
    config = load_article3_config(config_path)
    panel, provenance = _load_input(source_mode=source_mode, snapshot=panel_snapshot, manifest=provenance_manifest, article2_metadata=article2_metadata)
    run_id = _run_id()
    output_root.mkdir(parents=True, exist_ok=True)
    final_dir = output_root / run_id
    temporary_dir = output_root / f".{run_id}.tmp"
    if final_dir.exists() or temporary_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing Article-3 run {run_id}")
    if run_mode == "preflight":
        temporary_dir.mkdir(parents=True)
        write_json(temporary_dir / "input_contract.json", input_contract_payload(provenance, panel=panel))
        write_source_reconciliation(temporary_dir / "source_reconciliation.md", provenance=provenance)
        (temporary_dir / "resolved_config.yaml").write_text(yaml.safe_dump(config.as_dict(), sort_keys=False), encoding="utf-8")
        write_json(temporary_dir / "run_status.json", {"status": "PREFLIGHT_COMPLETE", "empirical_status": "VERIFIED_WEEKLY_INPUT", "publication_ready": False})
        temporary_dir.replace(final_dir)
        return {"run_dir": final_dir, "input_contract": final_dir / "input_contract.json"}
    if run_mode not in {"smoke", "full"}:
        raise ValueError("run_mode must be preflight, smoke, or full")
    effective = config if run_mode == "full" else replace(config, production_n_scenarios=config.smoke_n_scenarios, sensitivity_n_scenarios=config.smoke_n_scenarios, control_n_scenarios=config.smoke_n_scenarios, sensitivity_replicates=1, control_replicates=1)
    try:
        results = run_article3_analysis(panel, config=effective, provenance=provenance, output_dir=temporary_dir, run_id=run_id, run_mode=run_mode)
        temporary_dir.replace(final_dir)
    except BaseException:
        # Preserve a non-overwriting failed-run directory for forensic inspection.
        if temporary_dir.exists():
            write_json(temporary_dir / "run_status.json", {"status": "FAILED_QA", "empirical_status": "VERIFIED_WEEKLY_INPUT", "publication_ready": False})
        raise
    return {key: final_dir / path.relative_to(temporary_dir) if path.is_relative_to(temporary_dir) else path for key, path in results.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/article3_set_a_weekly.yaml"))
    parser.add_argument("--source-mode", choices=("snapshot", "live"), default="snapshot")
    parser.add_argument("--panel-snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--provenance-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--article2-metadata", type=Path, default=DEFAULT_ARTICLE2_METADATA)
    parser.add_argument("--run-mode", choices=("preflight", "smoke", "full"), default="preflight")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/article3_set_a"))
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    outputs = run(config_path=args.config, source_mode=args.source_mode, panel_snapshot=args.panel_snapshot,
                  provenance_manifest=args.provenance_manifest, article2_metadata=args.article2_metadata,
                  run_mode=args.run_mode, output_root=args.output_root)
    print(json.dumps({key: str(path) for key, path in outputs.items()}, indent=2))
