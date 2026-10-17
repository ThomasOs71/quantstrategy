"""Frequency-aware synchronized historical MBB factory for Article-3 Set A."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml

from data.asset_universe import get_asset_keys, get_driver_keys
from series.from_assumptions_to_portfolios.block1_scenarios.article3_input import (
    WeeklyInputProvenance,
    validate_article3_panel,
)
from series.from_assumptions_to_portfolios.block1_scenarios.block_sampler import (
    BlockSample,
    build_lineage_table,
    sample_moving_block_indices,
)
from series.from_assumptions_to_portfolios.block1_scenarios.scenario_schema import ScenarioSet
from series.from_assumptions_to_portfolios.block1_scenarios.terminal_returns import (
    build_terminal_returns_steps,
)


_CONFIG_KEYS = {
    "spec_version", "root_seed", "frequency", "sampling", "steps_per_year",
    "horizon_steps", "terminal_horizon_steps", "method", "block_length_steps",
    "block_length_status", "with_replacement", "overlapping", "circular",
    "production_n_scenarios", "sensitivity_block_lengths", "sensitivity_n_scenarios",
    "sensitivity_replicates", "control_n_scenarios", "control_replicates",
    "smoke_n_scenarios", "path_dtype", "summary_dtype", "acf_max_lag",
    "acf_error_lag_sets", "rolling_vol_steps", "rolling_corr_steps",
    "lower_tail_probability", "tail_quantile_method", "tail_comparison",
    "summary_quantile_method",
}


@dataclass(frozen=True)
class Article3Config:
    """Validated preregistered Set-A configuration."""

    spec_version: str = "2.0"
    root_seed: int = 20260913
    frequency: str = "weekly"
    sampling: str = "W-FRI"
    steps_per_year: int = 52
    horizon_steps: int = 260
    terminal_horizon_steps: Mapping[str, int] | None = None
    method: str = "moving_block_bootstrap"
    block_length_steps: int = 4
    block_length_status: str = "provisional_reference_not_optimal"
    with_replacement: bool = True
    overlapping: bool = True
    circular: bool = False
    production_n_scenarios: int = 10000
    sensitivity_block_lengths: tuple[int, ...] = (1, 2, 4, 8, 13, 26, 52)
    sensitivity_n_scenarios: int = 3000
    sensitivity_replicates: int = 5
    control_n_scenarios: int = 3000
    control_replicates: int = 5
    smoke_n_scenarios: int = 64
    path_dtype: str = "float64"
    summary_dtype: str = "float64"
    acf_max_lag: int = 52
    acf_error_lag_sets: tuple[int, ...] = (4, 13, 52)
    rolling_vol_steps: int = 52
    rolling_corr_steps: int = 104
    lower_tail_probability: float = 0.05
    tail_quantile_method: str = "linear"
    tail_comparison: str = "lt"
    summary_quantile_method: str = "linear"

    def __post_init__(self) -> None:
        if self.terminal_horizon_steps is None:
            object.__setattr__(self, "terminal_horizon_steps", {"12m": 52, "36m": 156, "60m": 260})
        if self.frequency != "weekly" or self.sampling != "W-FRI":
            raise ValueError("Article-3 Set A supports the weekly W-FRI contract only")
        if self.steps_per_year != 52 or self.horizon_steps != 260:
            raise ValueError("Article-3 defaults require 52 steps/year and a 260-step horizon")
        if dict(self.terminal_horizon_steps) != {"12m": 52, "36m": 156, "60m": 260}:
            raise ValueError("terminal_horizon_steps must be {'12m': 52, '36m': 156, '60m': 260}")
        if self.method != "moving_block_bootstrap" or not self.with_replacement or not self.overlapping or self.circular:
            raise ValueError("Set A requires overlapping non-circular MBB with replacement")
        if self.path_dtype != "float64" or self.summary_dtype != "float64":
            raise ValueError("Article-3 paths and summaries must use float64")
        if self.tail_comparison != "lt" or self.tail_quantile_method != "linear":
            raise ValueError("Article-3 tail contract requires linear quantiles and strict <")
        if self.block_length_steps not in self.sensitivity_block_lengths:
            raise ValueError("reference block_length_steps must appear in sensitivity_block_lengths")
        for name in ("root_seed", "block_length_steps", "production_n_scenarios", "sensitivity_n_scenarios", "sensitivity_replicates", "control_n_scenarios", "control_replicates", "smoke_n_scenarios"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any]) -> "Article3Config":
        unknown = set(mapping) - _CONFIG_KEYS
        if unknown:
            raise ValueError(f"unknown Article-3 config keys: {sorted(unknown)}")
        values = dict(mapping)
        for key in ("sensitivity_block_lengths", "acf_error_lag_sets"):
            if key in values:
                values[key] = tuple(values[key])
        return cls(**values)

    def as_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["terminal_horizon_steps"] = dict(self.terminal_horizon_steps or {})
        return result


def load_article3_config(path: str | Path) -> Article3Config:
    """Load and strictly validate the repository YAML configuration."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Article-3 config must be a YAML mapping")
    return Article3Config.from_mapping(payload)


def make_rng(*, config: Article3Config, experiment_id: int, block_length: int, replicate_id: int) -> np.random.Generator:
    """Create an independent documented PCG64 stream from stable integer IDs."""
    sequence = np.random.SeedSequence([config.root_seed, experiment_id, block_length, replicate_id])
    return np.random.Generator(np.random.PCG64(sequence))


@dataclass(frozen=True)
class SetABuild:
    """Production ScenarioSet together with non-serialized construction lineage."""

    scenario_set: ScenarioSet
    sample: BlockSample
    lineage: pd.DataFrame


def build_set_a(
    panel: pd.DataFrame,
    *,
    config: Article3Config,
    provenance: WeeklyInputProvenance,
    n_scenarios: int | None = None,
    replicate_id: int = 0,
) -> SetABuild:
    """Build one synchronized production Set-A MBB ScenarioSet from verified data."""
    validate_article3_panel(panel, provenance=provenance.manifest)
    scenarios = n_scenarios or config.production_n_scenarios
    rng = make_rng(
        config=config,
        experiment_id=101,
        block_length=config.block_length_steps,
        replicate_id=replicate_id,
    )
    sample = sample_moving_block_indices(
        n_obs=len(panel), n_scenarios=scenarios, horizon_steps=config.horizon_steps,
        block_length_steps=config.block_length_steps, rng=rng,
    )
    values = panel.to_numpy(dtype=np.float64)
    driver_paths = values[sample.source_indices, :]
    asset_positions = [get_driver_keys().index(key) for key in get_asset_keys()]
    # The canonical registry places the 11 assets before fx_eurusd.  Retaining
    # this contiguous view avoids a second 229 MB reference-path allocation.
    if asset_positions == list(range(len(asset_positions))):
        asset_paths = driver_paths[:, :, : len(asset_positions)]
    else:  # Defensive fallback should a future registry reorder keys.
        asset_paths = driver_paths[:, :, asset_positions]
    asset_terminal = build_terminal_returns_steps(
        asset_paths, horizon_steps_by_label=dict(config.terminal_horizon_steps or {})
    )
    metadata = {
        "schema_version": config.spec_version,
        "frequency": "weekly",
        "sampling": "W-FRI",
        "steps_per_year": config.steps_per_year,
        "horizon_steps": config.horizon_steps,
        "terminal_horizon_steps": dict(config.terminal_horizon_steps or {}),
        "return_type": "log",
        "terminal_return_type": "simple",
        "asset_base_currency": "EUR",
        "driver_keys": get_driver_keys(),
        "asset_keys": get_asset_keys(),
        "is_production_scenario_set": True,
        "method": config.method,
        "block_length_steps": config.block_length_steps,
        "block_sampling": "uniform_with_replacement",
        "overlapping": True,
        "circular": False,
        "final_truncation_rule": "truncate_final_fixed_block_to_horizon",
        "bit_generator": "PCG64",
        "root_seed": config.root_seed,
        "replicate_id": replicate_id,
        "input_canonical_panel_sha256": provenance.canonical_panel_sha256,
    }
    scenario_set = ScenarioSet(
        label="A", type=f"synchronized_mbb_L{config.block_length_steps:02d}",
        driver_paths=driver_paths, asset_paths=asset_paths, asset_terminal=asset_terminal,
        p=np.full(scenarios, 1.0 / scenarios, dtype=np.float64), metadata=metadata,
    )
    lineage = build_lineage_table(
        block_starts=sample.block_starts, source_dates=panel.index,
        used_block_lengths=sample.used_block_lengths,
    )
    return SetABuild(scenario_set=scenario_set, sample=sample, lineage=lineage)


def save_set_a_reference(build: SetABuild, *, output_dir: Path, block_length_steps: int) -> tuple[Path, Path]:
    """Persist the mandatory reference arrays and full lineage without pickle."""
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"set_a_L{block_length_steps:02d}"
    npz_path = output_dir / f"{stem}.npz"
    scenario_set = build.scenario_set
    arrays: dict[str, np.ndarray] = {
        "driver_paths": scenario_set.driver_paths,
        "asset_paths": scenario_set.asset_paths,
        "p": scenario_set.p,
        "source_indices": build.sample.source_indices,
        "block_starts": build.sample.block_starts,
        "used_block_lengths": np.asarray(build.sample.used_block_lengths, dtype=np.int64),
        "driver_keys": np.asarray(scenario_set.metadata["driver_keys"], dtype="U64"),
        "asset_keys": np.asarray(scenario_set.metadata["asset_keys"], dtype="U64"),
    }
    for label, terminal in scenario_set.asset_terminal.items():
        arrays[f"asset_terminal_{label}"] = terminal
    np.savez_compressed(npz_path, **arrays)
    lineage_path = output_dir / f"{stem}_lineage.csv.gz"
    build.lineage.to_csv(lineage_path, index=False, compression="gzip")
    return npz_path, lineage_path


def load_set_a_reference(path: str | Path) -> dict[str, np.ndarray]:
    """Read a reference archive with NumPy pickle loading explicitly disabled."""
    with np.load(Path(path), allow_pickle=False) as archive:
        return {key: archive[key].copy() for key in archive.files}
