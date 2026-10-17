"""Additive, offline Article-3 research; the frozen Set-A run is read-only."""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import yaml

from data.asset_universe import get_asset_keys, get_driver_keys
from series.from_assumptions_to_portfolios.block1_scenarios.article3_block_length_diagnostics import (
    analytic_preservation_scores,
    automatic_block_length_diagnostics,
    four_week_endpoints,
    last_friday_month_endpoints,
    ljung_box_diagnostics,
    resolution_diagnostics,
    source_uncertainty_diagnostics,
    terminal_correlation_rows,
)
from series.from_assumptions_to_portfolios.block1_scenarios.article3_diagnostics import (
    correlation_matrix_rows,
    historical_rolling_paths,
    rolling_path_diagnostic_rows,
    weighted_correlation,
)
from series.from_assumptions_to_portfolios.block1_scenarios.article3_input import load_verified_weekly_snapshot
from series.from_assumptions_to_portfolios.block1_scenarios.article3_reporting import (
    input_contract_payload,
    runtime_metadata,
    sha256_file,
    write_artifact_manifest,
    write_json,
)
from series.from_assumptions_to_portfolios.block1_scenarios.block_sampler import sample_moving_block_indices
from series.from_assumptions_to_portfolios.block1_scenarios.set_a_block_bootstrap import load_article3_config, make_rng


DEFAULT_SNAPSHOT = Path("data/downloads/daily_proxy_2011/weekly/daily_proxy_2011_weekly_2011-01-07_2025-12-26.csv")
DEFAULT_MANIFEST = DEFAULT_SNAPSHOT.with_suffix(".manifest.json")
DEFAULT_ARTICLE2 = Path("articles/outputs/article2/daily_proxy_2011/weekly/run_metadata.json")
DEFAULT_OUTPUT = Path("artifacts/article3_set_a")


@dataclass(frozen=True)
class BlockLengthResearchConfig:
    """Explicit diagnostic settings, without changing Set-A defaults."""

    spec_version: str = "1.0"
    root_seed: int = 20260913
    outer_mean_block_lengths: tuple[int, ...] = (13, 26, 52)
    outer_resamples: int = 999
    outer_confidence: float = .95
    uncertainty_acf_max_lag: int = 13
    ljung_box_lags: tuple[int, ...] = (4, 13, 52)
    ljung_box_alpha: float = .05
    contrasts: tuple[tuple[int, int], ...] = ((4, 1), (4, 8), (4, 13))
    synthetic_calendar_start: str = "2026-01-02"
    smoke_outer_resamples: int = 19

    def __post_init__(self):
        if self.spec_version != "1.0":
            raise ValueError("unsupported block-length research spec_version")
        for name in ("root_seed", "outer_resamples", "uncertainty_acf_max_lag", "smoke_outer_resamples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("outer_mean_block_lengths", "ljung_box_lags"):
            values = getattr(self, name)
            if not values or len(values) != len(set(values)) or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values):
                raise ValueError(f"{name} must contain distinct positive integers")
        for name in ("outer_confidence", "ljung_box_alpha"):
            if not 0 < getattr(self, name) < 1:
                raise ValueError(f"{name} must lie in (0,1)")
        if not self.contrasts or any(len(pair) != 2 or any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in pair) for pair in self.contrasts):
            raise ValueError("contrasts must contain positive integer pairs")
        start = pd.Timestamp(self.synthetic_calendar_start)
        if start.dayofweek != 4 or start.tz is not None:
            raise ValueError("synthetic_calendar_start must be a timezone-naive Friday")

    @classmethod
    def load(cls, path: Path):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or set(payload) - set(cls.__dataclass_fields__):
            raise ValueError("research configuration must be a mapping of known keys")
        for key in ("outer_mean_block_lengths", "ljung_box_lags"):
            if key in payload:
                payload[key] = tuple(payload[key])
        if "contrasts" in payload:
            payload["contrasts"] = tuple(tuple(pair) for pair in payload["contrasts"])
        return cls(**payload)


def verify_parent_run(parent: Path, *, input_hash: str, canonical_hash: str, config) -> dict[str, Any]:
    """Reject incompatible or tampered baselines rather than merging evidence."""
    metadata = json.loads((parent / "run_metadata.json").read_text(encoding="utf-8"))
    status = json.loads((parent / "run_status.json").read_text(encoding="utf-8"))
    if metadata.get("run_mode") != "full" or status.get("status") != "FULL_COMPLETE":
        raise ValueError("parent must be an existing FULL_COMPLETE empirical run")
    expected = {"input_file_sha256": input_hash, "canonical_panel_sha256": canonical_hash,
                "H": config.horizon_steps, "root_seed": config.root_seed,
                "L": config.block_length_steps, "driver_keys": get_driver_keys()}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"parent run does not match current verified input/config: {key}")
    parent_config = load_article3_config(parent / "resolved_config.yaml")
    if parent_config.as_dict() != config.as_dict():
        raise ValueError("parent resolved Set-A config differs; do not silently combine runs")
    manifest = json.loads((parent / "artifact_manifest.json").read_text(encoding="utf-8"))
    entries = {row["relative_path"]: row for row in manifest["entries"]}
    # These are all baseline files copied to the compact bundle or consumed
    # numerically. Hashes protect against mixing regenerated plots and tables.
    relevant = [path for path in parent.rglob("*") if path.is_file()
                and path.suffix in {".csv", ".png", ".svg", ".md", ".yaml", ".json"}
                and "scenarios" not in path.relative_to(parent).parts
                and path.name != "artifact_manifest.json"]
    for path in relevant:
        key = path.relative_to(parent).as_posix()
        if key not in entries or sha256_file(path) != entries[key]["sha256"]:
            raise ValueError(f"parent artifact hash mismatch or missing lineage: {key}")
    return metadata


def _select_parent(root: Path, input_hash: str) -> Path:
    candidates = []
    for path in root.glob("article3_set_a_weekly_v2_*/run_metadata.json"):
        meta = json.loads(path.read_text(encoding="utf-8"))
        if meta.get("run_mode") == "full" and meta.get("input_file_sha256") == input_hash:
            candidates.append(path.parent)
    if not candidates:
        raise FileNotFoundError("No frozen full Set-A run matches this snapshot; run Set A first or specify --parent-run")
    return sorted(candidates)[-1]


def _tag(frame: pd.DataFrame, *, length: int, replicate: int, reference: str) -> pd.DataFrame:
    return frame.assign(L=length, replicate_id=replicate, reference=reference)


def _check_regeneration(frame: pd.DataFrame, parent: pd.DataFrame, keys: list[str], length: int, replicate: int) -> float:
    experiment = "synchronized_iid_L01" if length == 1 else f"synchronized_mbb_L{length:02d}"
    old = parent.loc[(parent.experiment == experiment) & (parent.L == length)
                     & (parent.replicate_id == replicate) & (parent.group == "all12")]
    if len(old) != len(keys) ** 2:
        raise ValueError(f"parent lacks exactly one full correlation matrix for L={length}, replicate={replicate}")
    ordered = ["driver_i", "driver_j"]
    a = frame.sort_values(ordered).correlation.to_numpy()
    b = old.sort_values(ordered).correlation.to_numpy()
    if not np.allclose(a, b, rtol=0, atol=2e-12, equal_nan=True):
        raise ValueError(f"regenerated grid differs from frozen run for L={length}, replicate={replicate}")
    return float(np.nanmax(np.abs(a - b)))


def run_research(
    *, config_path: Path, research_config_path: Path, snapshot: Path,
    manifest: Path, article2_metadata: Path | None, parent_run: Path | None,
    output_root: Path, mode: str = "full",
) -> Path:
    """Run verified-source diagnostics and write an additive, self-contained handoff."""
    from series.from_assumptions_to_portfolios.block1_scenarios.article3_block_length_plots import write_research_figures
    from series.from_assumptions_to_portfolios.block1_scenarios.article3_block_length_reporting import write_research_handoff, build_research_bundle

    if mode not in {"full", "smoke"}:
        raise ValueError("mode must be full or smoke")
    config = load_article3_config(config_path)
    research = BlockLengthResearchConfig.load(research_config_path)
    if research.root_seed != config.root_seed:
        raise ValueError("research and Set-A root seeds must match")
    if any(value not in config.sensitivity_block_lengths for pair in research.contrasts for value in pair):
        raise ValueError("contrast lengths must belong to the frozen sensitivity grid")
    panel, provenance = load_verified_weekly_snapshot(snapshot, manifest, article2_metadata_path=article2_metadata)
    parent = (parent_run or _select_parent(DEFAULT_OUTPUT, provenance.input_file_sha256)).resolve()
    parent_metadata = verify_parent_run(parent, input_hash=provenance.input_file_sha256,
                                        canonical_hash=provenance.canonical_panel_sha256, config=config)
    # Check the optional research dependency before generating any artifacts.
    selectors = automatic_block_length_diagnostics(panel.to_numpy(), list(panel.columns))
    run_id = f"article3_block_length_v1_{mode}_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid4().hex[:8]}"
    run_dir = output_root.resolve() / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "tables").mkdir()
    if mode == "smoke":
        research = replace(research, outer_resamples=research.smoke_outer_resamples)
    n_paths = config.sensitivity_n_scenarios if mode == "full" else config.smoke_n_scenarios
    replicates = config.sensitivity_replicates if mode == "full" else 1
    keys, asset_keys = list(panel.columns), get_asset_keys()
    values = panel.to_numpy(dtype=np.float64)
    source_dir = Path(__file__).resolve().parent
    source_paths = [source_dir / name for name in (
        "article3_block_length_diagnostics.py", "article3_block_length_plots.py",
        "article3_block_length_reporting.py", "run_article3_block_length.py",
        "article3_diagnostics.py", "article3_input.py", "article3_plots.py", "block_sampler.py", "set_a_block_bootstrap.py",
    )] + [config_path.resolve(), research_config_path.resolve()]
    import arch
    import scipy
    import statsmodels
    git_head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    metadata = {"run_id": run_id, "run_mode": mode, "timestamp_utc": datetime.now(timezone.utc),
                "parent_run_id": parent_metadata["run_id"], "parent_run_path": str(parent),
                "parent_metadata_sha256": sha256_file(parent / "run_metadata.json"),
                "parent_manifest_sha256": sha256_file(parent / "artifact_manifest.json"),
                "input_file_sha256": provenance.input_file_sha256,
                "canonical_panel_sha256": provenance.canonical_panel_sha256,
                "git_commit": git_head, "source_code_sha256": {str(p): sha256_file(p) for p in source_paths},
                "runtime": runtime_metadata() | {"arch": arch.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
                "root_seed": config.root_seed, "grid_seed_stream": [config.root_seed, 101, "L", "replicate_id"],
                "outer_seed_stream": [research.root_seed, 301, "outer_mean_block_length", "resample_id"],
                "research_config": asdict(research), "set_a_config": config.as_dict(),
                "n_paths_per_grid_replicate": n_paths, "grid_replicates": replicates,
                "outer_sampler": "stationary_bootstrap_geometric_lengths_circular; uncertainty_only",
                "set_a_sampler": "fixed_overlapping_non_circular_synchronized_MBB; unchanged",
                "monthly_comparison": "same_paths_coarsened; four_week_and_synthetic_last_Friday_of_month; not true daily_EOM",
                "reference_calendar_is_synthetic": True, "smoke_results_not_for_article_claims": mode == "smoke"}
    write_json(run_dir / "run_metadata.json", metadata)
    write_json(run_dir / "input_contract.json", input_contract_payload(provenance, panel=panel))
    write_json(run_dir / "run_status.json", {"status": "RUNNING", "author_review_ready": False, "publication_ready": False})

    def progress(message: str):
        print(message, flush=True)

    try:
        tables = {
            "automatic_block_lengths": selectors,
            "ljung_box": ljung_box_diagnostics(values, keys, lags=research.ljung_box_lags, alpha=research.ljung_box_alpha),
            "analytic_preservation": analytic_preservation_scores(values, keys, lengths=config.sensitivity_block_lengths,
                                                                   horizon_steps=config.horizon_steps, max_lag=research.uncertainty_acf_max_lag),
        }
        collection: dict[str, list[pd.DataFrame]] = {name: [] for name in (
            "weekly_correlation_matrices", "rolling_diagnostics", "terminal_correlations",
            "resolution_summary", "resolution_correlations", "grid_reconciliation",
        )}
        baseline_corr = pd.read_csv(parent / "tables" / "correlation_matrices.csv")
        calendar_dates = pd.date_range(research.synthetic_calendar_start, periods=config.horizon_steps, freq="W-FRI")
        resolutions = {"four_week_endpoints": four_week_endpoints(config.horizon_steps),
                       "synthetic_last_friday_per_month": last_friday_month_endpoints(calendar_dates)}
        tables["resolution_grid"] = pd.DataFrame.from_records([
            {"grid": name, "retained_step": int(step), "synthetic_date": calendar_dates[step - 1].date().isoformat(),
             "calendar_is_synthetic": True, "daily_EOM_claim": False}
            for name, ends in resolutions.items() for step in ends
        ])
        collection["rolling_diagnostics"].append(_tag(
            rolling_path_diagnostic_rows(values[None, :, :len(asset_keys)], asset_keys,
                                         reference="historical_unique_full_sample_windows"),
            length=0, replicate=-1, reference="historical_unique_full_sample_windows"))
        for horizon in config.terminal_horizon_steps.values():
            historical = historical_rolling_paths(values, horizon_steps=horizon)
            collection["terminal_correlations"].append(_tag(
                terminal_correlation_rows(historical, keys, horizons=[horizon]),
                length=0, replicate=-1, reference="historical_overlapping_horizon_windows"))
            del historical
        progress(f"Verified {len(panel)} weeks and frozen parent {parent.name}; starting grid")
        for length in config.sensitivity_block_lengths:
            for replicate in range(replicates):
                sample = sample_moving_block_indices(n_obs=len(values), n_scenarios=n_paths,
                    horizon_steps=config.horizon_steps, block_length_steps=length,
                    rng=make_rng(config=config, experiment_id=101, block_length=length, replicate_id=replicate))
                paths = values[sample.source_indices]
                reference = f"synchronized_mbb_L{length:02d}"
                corr_rows = correlation_matrix_rows(weighted_correlation(paths.reshape(-1, len(keys))), keys,
                                                    reference=reference, group="all12")
                discrepancy = _check_regeneration(corr_rows, baseline_corr, keys, length, replicate) if mode == "full" else np.nan
                collection["grid_reconciliation"].append(pd.DataFrame([{"L": length, "replicate_id": replicate,
                    "max_abs_correlation_difference_to_parent": discrepancy,
                    "status": "matched_within_2e-12" if mode == "full" else "smoke_smaller_ensemble_not_compared"}]))
                collection["weekly_correlation_matrices"].append(_tag(corr_rows, length=length, replicate=replicate, reference=reference))
                collection["rolling_diagnostics"].append(_tag(
                    rolling_path_diagnostic_rows(paths[:, :, :len(asset_keys)], asset_keys, reference=reference),
                    length=length, replicate=replicate, reference=reference))
                collection["terminal_correlations"].append(_tag(
                    terminal_correlation_rows(paths, keys, horizons=list(config.terminal_horizon_steps.values())),
                    length=length, replicate=replicate, reference=reference))
                for grid, ends in resolutions.items():
                    summary, correlations = resolution_diagnostics(paths[:, :, :len(asset_keys)], asset_keys, endpoints=ends, grid_name=grid)
                    collection["resolution_summary"].append(_tag(summary, length=length, replicate=replicate, reference=reference))
                    collection["resolution_correlations"].append(_tag(correlations, length=length, replicate=replicate, reference=reference))
                del paths, sample
            progress(f"Grid L={length}: {replicates} x {n_paths} paths complete")
        tables.update({name: pd.concat(frames, ignore_index=True) for name, frames in collection.items()})
        draws, intervals = source_uncertainty_diagnostics(values, keys, lengths=config.sensitivity_block_lengths,
            horizon_steps=config.horizon_steps, mean_block_lengths=research.outer_mean_block_lengths,
            n_resamples=research.outer_resamples, root_seed=research.root_seed, max_lag=research.uncertainty_acf_max_lag,
            contrasts=research.contrasts, confidence=research.outer_confidence, progress=progress)
        tables["source_uncertainty_intervals"] = intervals
        draws.to_csv(run_dir / "tables" / "source_uncertainty_draws.csv.gz", index=False, compression="gzip", float_format="%.17g")
        for name, table in tables.items():
            table.assign(run_id=run_id).to_csv(run_dir / "tables" / f"{name}.csv", index=False, float_format="%.17g")
        figure_index = write_research_figures(tables, parent=parent, output_dir=run_dir / "figures")
        write_research_handoff(run_dir, tables=tables, parent=parent, metadata=metadata, figure_index=figure_index)
        write_json(run_dir / "run_status.json", {"status": "FULL_COMPLETE" if mode == "full" else "SMOKE_COMPLETE",
            "author_review_ready": mode == "full", "publication_ready": False, "baseline_unchanged": True,
            "calibrated_optimal_block_length": False, "external_data_downloads": 0,
            "licensing_review_required_before_external_redistribution": True})
        write_artifact_manifest(run_dir, excluded_names=("artifact_manifest.json", "article3_author_bundle.zip"))
        bundle = build_research_bundle(run_dir, parent=parent)
        write_json(run_dir / "bundle_integrity.json", {"bundle": bundle.name, "sha256": sha256_file(bundle), "size_bytes": bundle.stat().st_size})
        progress(f"Completed {mode} research: {run_dir}")
        return run_dir
    except Exception as exc:
        write_json(run_dir / "run_status.json", {"status": "FAILED", "author_review_ready": False,
            "publication_ready": False, "error_type": type(exc).__name__, "error": str(exc), "baseline_unchanged": True})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/article3_set_a_weekly.yaml"))
    parser.add_argument("--research-config", type=Path, default=Path("configs/article3_block_length_weekly.yaml"))
    parser.add_argument("--snapshot", type=Path, default=DEFAULT_SNAPSHOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--article2-metadata", type=Path, default=DEFAULT_ARTICLE2)
    parser.add_argument("--parent-run", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--mode", choices=("full", "smoke"), default="full")
    args = parser.parse_args()
    run_research(config_path=args.config, research_config_path=args.research_config, snapshot=args.snapshot,
        manifest=args.manifest, article2_metadata=args.article2_metadata, parent_run=args.parent_run,
        output_root=args.output_root, mode=args.mode)


if __name__ == "__main__":
    main()
