"""Strict verified-weekly input adapter for Article-3 Scenario Set A."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from data.asset_universe import get_asset_keys, get_driver_keys
from data.load_data import load_return_panel_download


WEEKLY_START = pd.Timestamp("2011-01-07")
WEEKLY_END = pd.Timestamp("2025-12-26")
WEEKLY_OBSERVATIONS = 782


@dataclass(frozen=True)
class WeeklyPanelContract:
    """The immutable Article-3 empirical input contract."""

    source_profile: str = "daily_proxy_2011"
    frequency: str = "weekly"
    sampling: str = "W-FRI"
    return_type: str = "log"
    return_unit: str = "decimal"
    base_currency: str = "EUR"
    start: pd.Timestamp = WEEKLY_START
    end: pd.Timestamp = WEEKLY_END


@dataclass(frozen=True)
class WeeklyInputProvenance:
    """File identity and run-match evidence for the frozen weekly panel."""

    snapshot_path: Path
    manifest_path: Path
    input_file_sha256: str
    input_manifest_sha256: str
    canonical_panel_sha256: str
    article2_run_reference: str | None
    article2_snapshot_match_status: str
    manifest: Mapping[str, Any]


def sha256_file(path: Path) -> str:
    """Calculate a streaming SHA-256 without exposing file contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_panel_sha256(panel: pd.DataFrame) -> str:
    """Hash ordered keys, ISO dates, and little-endian Float64 observations."""
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    index = pd.DatetimeIndex(panel.index)
    values = np.asarray(panel.to_numpy(dtype=np.float64), dtype="<f8", order="C")
    digest = hashlib.sha256()
    digest.update(b"article3_canonical_panel_v1\n")
    digest.update("\x1f".join(map(str, panel.columns)).encode("utf-8"))
    digest.update(b"\n")
    digest.update("\n".join(index.strftime("%Y-%m-%d")).encode("utf-8"))
    digest.update(b"\n")
    digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def validate_article3_panel(
    panel: pd.DataFrame,
    *,
    contract: WeeklyPanelContract = WeeklyPanelContract(),
    provenance: Mapping[str, Any] | None = None,
) -> None:
    """Validate the Article-3 weekly panel without altering returns."""
    if not isinstance(panel, pd.DataFrame):
        raise TypeError("panel must be a pandas DataFrame")
    expected_index = pd.date_range(contract.start, contract.end, freq=contract.sampling)
    if not isinstance(panel.index, pd.DatetimeIndex):
        raise TypeError("panel index must be a DatetimeIndex")
    if not panel.index.equals(expected_index):
        raise ValueError("panel index must equal the complete 2011-01-07 to 2025-12-26 W-FRI grid")
    if len(panel) != WEEKLY_OBSERVATIONS:
        raise ValueError("weekly panel must contain exactly 782 observations")
    if list(panel.columns) != get_driver_keys():
        raise ValueError("panel columns must exactly match the canonical driver order")
    if list(panel.columns[:-1]) != get_asset_keys() or panel.columns[-1] != "fx_eurusd":
        raise ValueError("panel must contain eleven investable assets followed by fx_eurusd")
    values = panel.to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("panel must contain only finite decimal log returns")
    attrs = panel.attrs
    if attrs.get("source_profile") != contract.source_profile:
        raise ValueError("panel source_profile is not daily_proxy_2011")
    if attrs.get("frequency") != "weekly_friday" or attrs.get("resample_rule") != contract.sampling:
        raise ValueError("panel metadata does not prove W-FRI sampling")
    if attrs.get("return_representation") != "EUR log returns":
        raise ValueError("panel metadata does not prove EUR log-return representation")
    if attrs.get("periods_per_year") != 52:
        raise ValueError("panel metadata does not prove the 52-step annualization convention")
    if provenance is not None:
        if provenance.get("source_profile") != contract.source_profile:
            raise ValueError("manifest source_profile is not daily_proxy_2011")
        if provenance.get("frequency_option") != contract.frequency:
            raise ValueError("manifest frequency_option is not weekly")
        if int(provenance.get("rows", -1)) != WEEKLY_OBSERVATIONS or int(
            provenance.get("columns", -1)
        ) != len(get_driver_keys()):
            raise ValueError("manifest dimensions do not match the Article-3 contract")
        if list(provenance.get("driver_keys", [])) != get_driver_keys():
            raise ValueError("manifest driver order does not match the canonical registry")


def load_verified_weekly_snapshot(
    snapshot_path: str | Path,
    provenance_manifest: str | Path,
    *,
    article2_metadata_path: str | Path | None = None,
) -> tuple[pd.DataFrame, WeeklyInputProvenance]:
    """Load checksum-verified weekly data and prove the Article-3 contract."""
    csv_path = Path(snapshot_path).resolve()
    manifest_path = Path(provenance_manifest).resolve()
    panel = load_return_panel_download(csv_path, manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_article3_panel(panel, provenance=manifest)
    input_hash = sha256_file(csv_path)
    if manifest.get("csv_sha256") != input_hash:
        raise ValueError("snapshot CSV checksum does not match its provenance manifest")

    run_reference: str | None = None
    match_status = "not_checked"
    if article2_metadata_path is not None:
        metadata_path = Path(article2_metadata_path).resolve()
        run_reference = str(metadata_path)
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        match_status = (
            "matched"
            if metadata.get("snapshot_csv_sha256") == input_hash
            else "different_snapshot"
        )
    provenance = WeeklyInputProvenance(
        snapshot_path=csv_path,
        manifest_path=manifest_path,
        input_file_sha256=input_hash,
        input_manifest_sha256=sha256_file(manifest_path),
        canonical_panel_sha256=canonical_panel_sha256(panel),
        article2_run_reference=run_reference,
        article2_snapshot_match_status=match_status,
        manifest=manifest,
    )
    return panel, provenance
