"""Deterministic index sampling and lineage helpers for Article-3 Set A."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Sequence

import numpy as np
import pandas as pd


def _positive_int(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be an integer, not bool")
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{name} must be positive")
    return parsed


@dataclass(frozen=True)
class BlockSample:
    """Source indices and start metadata for one fixed-length MBB draw."""

    source_indices: np.ndarray
    block_starts: np.ndarray
    used_block_lengths: tuple[int, ...]


def used_block_lengths(*, horizon_steps: int, block_length_steps: int) -> tuple[int, ...]:
    """Return the actually used lengths, including a truncated final block."""
    horizon = _positive_int("horizon_steps", horizon_steps)
    length = _positive_int("block_length_steps", block_length_steps)
    return tuple(
        min(length, horizon - block_id * length)
        for block_id in range((horizon + length - 1) // length)
    )


def sample_moving_block_indices(
    *,
    n_obs: int,
    n_scenarios: int,
    horizon_steps: int,
    block_length_steps: int,
    rng: np.random.Generator,
) -> BlockSample:
    """Sample overlapping, fixed-length, non-circular MBB source indices."""
    observations = _positive_int("n_obs", n_obs)
    scenarios = _positive_int("n_scenarios", n_scenarios)
    horizon = _positive_int("horizon_steps", horizon_steps)
    length = _positive_int("block_length_steps", block_length_steps)
    if length > observations:
        raise ValueError("block_length_steps must not exceed n_obs")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    lengths = used_block_lengths(
        horizon_steps=horizon, block_length_steps=length
    )
    n_blocks = len(lengths)
    library_size = observations - length + 1
    starts = rng.integers(
        low=0,
        high=library_size,
        size=(scenarios, n_blocks),
        dtype=np.int64,
    )
    steps = np.arange(horizon, dtype=np.int64)
    indices = starts[:, steps // length] + steps[None, :] % length
    return BlockSample(
        source_indices=indices,
        block_starts=starts,
        used_block_lengths=lengths,
    )


def expected_source_weights(
    *, n_obs: int, horizon_steps: int, block_length_steps: int
) -> np.ndarray:
    """Return exact pooled source-row probabilities for the non-circular MBB."""
    observations = _positive_int("n_obs", n_obs)
    horizon = _positive_int("horizon_steps", horizon_steps)
    length = _positive_int("block_length_steps", block_length_steps)
    if length > observations:
        raise ValueError("block_length_steps must not exceed n_obs")
    library_size = observations - length + 1
    weights = np.zeros(observations, dtype=np.float64)
    for used_length in used_block_lengths(
        horizon_steps=horizon, block_length_steps=length
    ):
        rows = np.arange(observations, dtype=np.int64)
        lower = np.maximum(0, rows - used_length + 1)
        upper = np.minimum(library_size - 1, rows)
        weights += np.maximum(0, upper - lower + 1)
    weights /= float(horizon * library_size)
    if not np.isclose(weights.sum(), 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("expected MBB source weights do not sum to one")
    return weights


def realized_source_weights(source_indices: np.ndarray, *, n_obs: int) -> np.ndarray:
    """Return realized pooled row frequencies from sampled source indices."""
    observations = _positive_int("n_obs", n_obs)
    indices = np.asarray(source_indices)
    if indices.ndim != 2 or indices.size == 0:
        raise ValueError("source_indices must be a non-empty two-dimensional array")
    if not np.issubdtype(indices.dtype, np.integer):
        raise TypeError("source_indices must be integers")
    if indices.min() < 0 or indices.max() >= observations:
        raise ValueError("source_indices are outside the historical sample")
    return np.bincount(indices.ravel(), minlength=observations).astype(np.float64) / indices.size


def block_join_metrics(*, horizon_steps: int, block_length_steps: int) -> dict[str, float | int | None]:
    """Return constructive within-block and join counts for one path."""
    horizon = _positive_int("horizon_steps", horizon_steps)
    lengths = used_block_lengths(
        horizon_steps=horizon, block_length_steps=block_length_steps
    )
    n_blocks = len(lengths)
    joins = n_blocks - 1
    within = horizon - n_blocks
    pairs = horizon - 1
    return {
        "n_blocks": n_blocks,
        "n_joins": joins,
        "n_within_pairs": within,
        "within_fraction": None if pairs == 0 else within / pairs,
        "join_fraction": None if pairs == 0 else joins / pairs,
    }


def lag_within_block_fraction(
    *, horizon_steps: int, block_length_steps: int, lag: int
) -> float | None:
    """Share of lag pairs that are constructively within a sampled block."""
    horizon = _positive_int("horizon_steps", horizon_steps)
    parsed_lag = _positive_int("lag", lag)
    if parsed_lag >= horizon:
        return None
    lengths = used_block_lengths(
        horizon_steps=horizon, block_length_steps=block_length_steps
    )
    return sum(max(length - parsed_lag, 0) for length in lengths) / (horizon - parsed_lag)


def build_lineage_table(
    *,
    block_starts: np.ndarray,
    source_dates: Sequence[pd.Timestamp] | pd.DatetimeIndex,
    used_block_lengths: Sequence[int],
) -> pd.DataFrame:
    """Create full block lineage with actual final-block endpoints."""
    starts = np.asarray(block_starts)
    if starts.ndim != 2 or not np.issubdtype(starts.dtype, np.integer):
        raise ValueError("block_starts must be a two-dimensional integer array")
    dates = pd.DatetimeIndex(source_dates)
    lengths = tuple(int(length) for length in used_block_lengths)
    if starts.shape[1] != len(lengths):
        raise ValueError("block_starts columns must match used_block_lengths")
    # Build column arrays rather than a Python list of dictionaries.  The full
    # 10,000 × 65 reference lineage has 650,000 rows; this avoids a transient
    # object-graph multiple of the final table's memory footprint.
    scenarios, n_blocks = starts.shape
    block_ids = np.repeat(np.arange(n_blocks, dtype=np.int64), scenarios)
    scenario_ids = np.tile(np.arange(scenarios, dtype=np.int64), n_blocks)
    source_start = starts.T.reshape(-1)
    length_array = np.repeat(np.asarray(lengths, dtype=np.int64), scenarios)
    source_end = source_start + length_array - 1
    if source_start.min() < 0 or source_end.max() >= len(dates):
        raise ValueError("lineage block is outside source_dates")
    sim_starts = np.cumsum(np.r_[0, np.asarray(lengths[:-1], dtype=np.int64)])
    sim_start_array = np.repeat(sim_starts, scenarios)
    return pd.DataFrame({
        "scenario_id": scenario_ids,
        "block_id": block_ids,
        "source_start_idx": source_start,
        "source_end_idx_inclusive": source_end,
        "source_start_date": dates[source_start].strftime("%Y-%m-%d"),
        "source_end_date": dates[source_end].strftime("%Y-%m-%d"),
        "steps_used": length_array,
        "sim_start_step_inclusive": sim_start_array,
        "sim_end_step_exclusive": sim_start_array + length_array,
    })


def independent_iid_indices(
    *, n_obs: int, n_scenarios: int, horizon_steps: int, n_drivers: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw the explicit diagnostic-only independent-per-driver control."""
    return rng.integers(
        0,
        _positive_int("n_obs", n_obs),
        size=(
            _positive_int("n_scenarios", n_scenarios),
            _positive_int("horizon_steps", horizon_steps),
            _positive_int("n_drivers", n_drivers),
        ),
        dtype=np.int64,
    )
