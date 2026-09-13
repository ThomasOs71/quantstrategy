"""Canonical historical-return input for Block 1, Article 2.

Article 2 diagnoses the historical building blocks behind later scenario sets.
It does not resample returns or construct scenario paths.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from data.asset_universe import get_driver_keys
from data.load_data import build_return_panel
from data.panel_profiles import PanelFrequency, get_frequency_definition
from series.from_assumptions_to_portfolios.block1_scenarios.article2_frequency import (
    ARTICLE2_END,
    get_article2_frequency_spec,
)


ARTICLE2_START = "2011-01-31"
ARTICLE2_WEEKLY_START = "2011-01-07"


def prepare_article2_panel(
    panel: pd.DataFrame,
    start: str = ARTICLE2_START,
    end: str = ARTICLE2_END,
    driver_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Select and validate the complete EUR monthly log-return panel for Article 2."""
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")

    prepared = panel.copy()
    prepared.index = pd.to_datetime(prepared.index)
    if prepared.index.has_duplicates:
        raise ValueError("panel index must not contain duplicate dates")

    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if start_ts > end_ts:
        raise ValueError("start must not be after end")

    prepared = prepared.loc[(prepared.index >= start_ts) & (prepared.index <= end_ts)]
    validate_article2_panel(
        prepared,
        start=start_ts,
        end=end_ts,
        driver_keys=driver_keys,
    )
    return prepared


def validate_article2_panel(
    panel: pd.DataFrame,
    start: str | pd.Timestamp = ARTICLE2_START,
    end: str | pd.Timestamp = ARTICLE2_END,
    driver_keys: Sequence[str] | None = None,
) -> None:
    """Validate the Article-2 panel contract without changing its contents."""
    validate_article2_research_panel(
        panel,
        frequency=PanelFrequency.MONTHLY,
        start=start,
        end=end,
        driver_keys=driver_keys,
    )


def prepare_article2_research_panel(
    panel: pd.DataFrame,
    *,
    frequency: str | PanelFrequency,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp = ARTICLE2_END,
    driver_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Select and validate a frequency-aware Article-2 research panel."""
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    spec = get_article2_frequency_spec(frequency)
    prepared = panel.copy()
    prepared.index = pd.to_datetime(prepared.index)
    if prepared.index.has_duplicates:
        raise ValueError("panel index must not contain duplicate dates")
    start_ts = pd.Timestamp(start or spec.start)
    end_ts = pd.Timestamp(end)
    if start_ts > end_ts:
        raise ValueError("start must not be after end")
    prepared = prepared.loc[(prepared.index >= start_ts) & (prepared.index <= end_ts)]
    validate_article2_research_panel(
        prepared,
        frequency=spec.frequency,
        start=start_ts,
        end=end_ts,
        driver_keys=driver_keys,
    )
    return prepared


def validate_article2_research_panel(
    panel: pd.DataFrame,
    *,
    frequency: str | PanelFrequency,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp = ARTICLE2_END,
    driver_keys: Sequence[str] | None = None,
) -> None:
    """Validate complete ME or W-FRI Article-2 research coverage."""
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    if panel.empty:
        raise ValueError("Article-2 panel must not be empty")
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise TypeError("panel index must be a DatetimeIndex")
    if not panel.index.is_monotonic_increasing:
        raise ValueError("panel index must be sorted in ascending order")
    if panel.index.has_duplicates:
        raise ValueError("panel index must not contain duplicate dates")

    spec = get_article2_frequency_spec(frequency)
    expected_columns = list(driver_keys or get_driver_keys())
    actual_columns = list(panel.columns)
    if actual_columns != expected_columns:
        raise ValueError(
            "panel columns must exactly match the expected driver order: "
            f"expected {expected_columns}, got {actual_columns}"
        )

    frequency_definition = get_frequency_definition(spec.frequency)
    expected_index = pd.date_range(
        pd.Timestamp(start or spec.start),
        pd.Timestamp(end),
        freq=frequency_definition.resample_rule,
    )
    if not panel.index.equals(expected_index):
        raise ValueError(
            f"panel index must be a complete {spec.period_adjective} "
            f"{frequency_definition.resample_rule} range from "
            f"{expected_index[0].date()} to {expected_index[-1].date()}"
        )

    numeric = panel.to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("panel must contain only finite return values")


def article2_panel_metadata(panel: pd.DataFrame) -> dict[str, Any]:
    """Return JSON-serializable metadata for a validated Article-2 panel."""
    validate_article2_panel(panel)
    metadata: dict[str, Any] = {
        "start": panel.index.min().date().isoformat(),
        "end": panel.index.max().date().isoformat(),
        "n_observations": int(len(panel)),
        "n_drivers": int(panel.shape[1]),
        "driver_keys": list(panel.columns),
        "frequency": "monthly_month_end",
        "return_representation": "EUR monthly log returns",
    }
    for key in (
        "source_profile",
        "known_currency_mismatch",
        "known_currency_mismatch_keys",
        "legacy_source_audit",
    ):
        if key in panel.attrs:
            metadata[key] = panel.attrs[key]
    return metadata


def article2_research_panel_metadata(
    panel: pd.DataFrame,
    *,
    frequency: str | PanelFrequency,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp = ARTICLE2_END,
) -> dict[str, Any]:
    """Return JSON metadata for a validated monthly or weekly research panel."""
    spec = get_article2_frequency_spec(frequency)
    validate_article2_research_panel(
        panel,
        frequency=spec.frequency,
        start=start or spec.start,
        end=end,
    )
    frequency_definition = get_frequency_definition(spec.frequency)
    metadata: dict[str, Any] = {
        "start": panel.index.min().date().isoformat(),
        "end": panel.index.max().date().isoformat(),
        "requested_end": pd.Timestamp(end).date().isoformat(),
        "n_observations": int(len(panel)),
        "n_drivers": int(panel.shape[1]),
        "driver_keys": list(panel.columns),
        "frequency": frequency_definition.metadata_label,
        "periods_per_year": spec.periods_per_year,
        "return_representation": f"EUR {spec.period_adjective} log returns",
    }
    for key in (
        "source_profile",
        "resample_rule",
        "cash_accrual",
        "hedge_method",
        "source_provenance",
        "data_access_note",
        "snapshot_csv_filename",
        "snapshot_manifest_filename",
        "snapshot_csv_sha256",
    ):
        if key in panel.attrs:
            metadata[key] = panel.attrs[key]
    return metadata


def load_article2_panel(
    fred_api_key: str | None = None,
    start: str = ARTICLE2_START,
    end: str = ARTICLE2_END,
) -> pd.DataFrame:
    """Load the existing return panel and select the Article-2 analysis window."""
    source_start = (pd.Timestamp(start) - pd.DateOffset(months=2)).strftime("%Y-%m-%d")
    panel = build_return_panel(
        start=source_start,
        end=end,
        fred_api_key=fred_api_key,
        strict_no_nan=False,
        frequency="monthly",
        source_profile="monthly_legacy",
    )
    return prepare_article2_panel(panel, start=start, end=end)


def load_article2_research_panel(
    *,
    frequency: str | PanelFrequency,
    fred_api_key: str | None = None,
    start: str | None = None,
    end: str = ARTICLE2_END,
) -> pd.DataFrame:
    """Load and validate the no-purchase Article-2 research panel."""
    spec = get_article2_frequency_spec(frequency)
    resolved_start = start or spec.start
    panel = build_return_panel(
        start=resolved_start,
        end=end,
        fred_api_key=fred_api_key,
        strict_no_nan=True,
        frequency=spec.frequency,
        source_profile="daily_proxy_2011",
    )
    return prepare_article2_research_panel(
        panel,
        frequency=spec.frequency,
        start=resolved_start,
        end=end,
    )
