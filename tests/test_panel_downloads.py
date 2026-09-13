"""Tests for local monthly/weekly panel snapshot downloads."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data.load_data as load_data
from data.asset_universe import get_driver_keys
from data.panel_profiles import PanelFrequency, SourceProfile


def _fake_panel(frequency: str) -> pd.DataFrame:
    if frequency == "monthly":
        index = pd.DatetimeIndex(["2020-01-31", "2020-02-29"])
        frequency_label = "monthly_month_end"
        periods_per_year = 12
    else:
        index = pd.DatetimeIndex(["2020-01-03", "2020-01-10"])
        frequency_label = "weekly_friday"
        periods_per_year = 52
    values = (
        np.arange(len(index) * len(get_driver_keys()), dtype=float).reshape(
            len(index), -1
        )
        / 10_000
    )
    panel = pd.DataFrame(values, index=index, columns=get_driver_keys())
    panel.attrs.update(
        {
            "source_profile": "daily_proxy_2011",
            "frequency": frequency_label,
            "periods_per_year": periods_per_year,
            "source_provenance": {
                key: {"source_identifier": f"synthetic-{key}"}
                for key in get_driver_keys()
            },
        }
    )
    return panel


@pytest.mark.parametrize("frequency", ["monthly", "weekly"])
def test_download_return_panel_writes_csv_and_manifest(
    tmp_path, monkeypatch, frequency
) -> None:
    expected_panel = _fake_panel(frequency)
    captured: dict[str, object] = {}

    def fake_builder(**kwargs):
        captured.update(kwargs)
        return expected_panel.copy()

    monkeypatch.setattr(load_data, "build_return_panel", fake_builder)
    result = load_data.download_return_panel(
        frequency=frequency,
        output_dir=tmp_path,
        start=expected_panel.index.min().date().isoformat(),
        end=expected_panel.index.max().date().isoformat(),
        fred_api_key="TOP_SECRET",
    )

    start = expected_panel.index.min().date().isoformat()
    end = expected_panel.index.max().date().isoformat()
    stem = f"daily_proxy_2011_{frequency}_{start}_{end}"
    assert result["data"] == tmp_path / f"{stem}.csv"
    assert result["manifest"] == tmp_path / f"{stem}.manifest.json"
    assert captured["frequency"] is PanelFrequency(frequency)
    assert captured["source_profile"] is SourceProfile.DAILY_PROXY_2011
    assert captured["strict_no_nan"] is True

    restored = pd.read_csv(result["data"], parse_dates=["date"]).set_index("date")
    pd.testing.assert_frame_equal(
        restored, expected_panel.rename_axis("date"), check_freq=False
    )

    manifest_text = result["manifest"].read_text(encoding="utf-8")
    manifest = json.loads(manifest_text)
    assert "TOP_SECRET" not in manifest_text
    assert manifest["source_profile"] == "daily_proxy_2011"
    assert manifest["frequency_option"] == frequency
    assert manifest["rows"] == 2
    assert manifest["columns"] == 12
    assert manifest["driver_keys"] == get_driver_keys()
    assert manifest["effective_start"] == start
    assert manifest["effective_end"] == end
    assert (
        manifest["csv_sha256"]
        == hashlib.sha256(result["data"].read_bytes()).hexdigest()
    )
    restored_with_manifest = load_data.load_return_panel_download(result["data"])
    restored_values = restored_with_manifest.copy()
    restored_values.attrs = {}
    expected_values = expected_panel.rename_axis("date").copy()
    expected_values.attrs = {}
    pd.testing.assert_frame_equal(
        restored_values,
        expected_values,
        check_freq=False,
    )
    assert restored_with_manifest.attrs["snapshot_csv_sha256"] == manifest["csv_sha256"]


def test_download_return_panel_uses_frequency_specific_default_folder(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(load_data, "DEFAULT_PANEL_DOWNLOAD_DIR", tmp_path)
    monkeypatch.setattr(
        load_data,
        "build_return_panel",
        lambda **_kwargs: _fake_panel("weekly"),
    )

    result = load_data.download_return_panel(frequency="weekly")

    assert result["data"].parent == tmp_path / "weekly"
    assert result["manifest"].parent == tmp_path / "weekly"


def test_load_return_panel_download_rejects_tampered_csv(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        load_data,
        "build_return_panel",
        lambda **_kwargs: _fake_panel("monthly"),
    )
    result = load_data.download_return_panel(frequency="monthly", output_dir=tmp_path)
    result["data"].write_text("tampered\n", encoding="utf-8")

    with pytest.raises(ValueError, match="checksum"):
        load_data.load_return_panel_download(result["data"])


def test_download_return_panel_preserves_existing_snapshot_unless_overwritten(
    tmp_path, monkeypatch
) -> None:
    panel = _fake_panel("monthly")
    monkeypatch.setattr(load_data, "build_return_panel", lambda **_kwargs: panel.copy())

    first = load_data.download_return_panel(frequency="monthly", output_dir=tmp_path)
    original_csv = first["data"].read_bytes()
    original_manifest = first["manifest"].read_bytes()

    with pytest.raises(FileExistsError, match="overwrite=True"):
        load_data.download_return_panel(frequency="monthly", output_dir=tmp_path)
    assert first["data"].read_bytes() == original_csv
    assert first["manifest"].read_bytes() == original_manifest

    replaced = load_data.download_return_panel(
        frequency="monthly", output_dir=tmp_path, overwrite=True
    )
    assert replaced == first
    assert replaced["data"].exists()
    assert replaced["manifest"].exists()


@pytest.mark.parametrize("failure_type", [OSError, KeyboardInterrupt])
def test_download_return_panel_rolls_back_pair_if_manifest_commit_fails(
    tmp_path, monkeypatch, failure_type
) -> None:
    panel = _fake_panel("monthly")
    monkeypatch.setattr(load_data, "build_return_panel", lambda **_kwargs: panel.copy())
    original = load_data.download_return_panel(frequency="monthly", output_dir=tmp_path)
    original_csv = original["data"].read_bytes()
    original_manifest = original["manifest"].read_bytes()

    real_replace = Path.replace
    failure_injected = False

    def fail_manifest_commit_once(source, target):
        nonlocal failure_injected
        target = Path(target)
        if (
            not failure_injected
            and source.name.endswith(".json.tmp")
            and target == original["manifest"]
        ):
            failure_injected = True
            raise failure_type("synthetic manifest commit failure")
        return real_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_manifest_commit_once)
    with pytest.raises(failure_type, match="synthetic manifest commit failure"):
        load_data.download_return_panel(
            frequency="monthly", output_dir=tmp_path, overwrite=True
        )

    assert failure_injected is True
    assert original["data"].read_bytes() == original_csv
    assert original["manifest"].read_bytes() == original_manifest
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob("*.bak"))
    assert not list(tmp_path.glob("*.lock"))


def test_download_return_panel_respects_existing_writer_lock(
    tmp_path, monkeypatch
) -> None:
    panel = _fake_panel("monthly")
    monkeypatch.setattr(load_data, "build_return_panel", lambda **_kwargs: panel.copy())
    original = load_data.download_return_panel(frequency="monthly", output_dir=tmp_path)
    original_csv = original["data"].read_bytes()
    original_manifest = original["manifest"].read_bytes()
    lock_path = original["data"].with_name(f".{original['data'].stem}.lock")
    lock_path.write_text("synthetic active writer\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Another download is committing"):
        load_data.download_return_panel(
            frequency="monthly", output_dir=tmp_path, overwrite=True
        )

    assert lock_path.exists()
    assert original["data"].read_bytes() == original_csv
    assert original["manifest"].read_bytes() == original_manifest
    assert not list(tmp_path.glob("*.tmp"))
    lock_path.unlink()


def test_download_return_panel_rejects_unignored_in_repo_destination() -> None:
    repository_root = Path(load_data.__file__).resolve().parent.parent
    with pytest.raises(ValueError, match="under data/downloads"):
        load_data.download_return_panel(
            frequency="monthly",
            output_dir=repository_root / "data" / "accidental-downloads",
        )


def test_download_return_panel_rejects_invalid_frequency(tmp_path) -> None:
    with pytest.raises(ValueError, match="frequency must be one of"):
        load_data.download_return_panel(frequency="daily", output_dir=tmp_path)
