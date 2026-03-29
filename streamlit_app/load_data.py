from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "articles" / "Optimization_Lexicographic_2" / "streamlit_data"


class StreamlitDataError(RuntimeError):
    """Raised when the precomputed dataset files are unavailable or unreadable."""


def resolve_data_dir(data_dir: str | Path | None = None) -> Path:
    if data_dir is None:
        return DEFAULT_DATA_DIR
    resolved = Path(data_dir)
    if not resolved.is_absolute():
        resolved = (REPO_ROOT / resolved).resolve()
    return resolved


def _read_table(data_dir: Path, base_name: str) -> pd.DataFrame:
    parquet_path = data_dir / f"{base_name}.parquet"
    csv_path = data_dir / f"{base_name}.csv"

    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise StreamlitDataError(
        f"Missing precomputed dataset '{base_name}'. Expected '{parquet_path.name}' or '{csv_path.name}' in {data_dir}."
    )


@st.cache_data(show_spinner=False)
def load_precomputed_tables(data_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    resolved_dir = resolve_data_dir(data_dir)
    if not resolved_dir.exists():
        raise StreamlitDataError(
            f"Precomputed dataset directory not found: {resolved_dir}. Run the dataset builder first."
        )

    return {
        "sweep_results": _read_table(resolved_dir, "sweep_results"),
        "portfolio_snapshots": _read_table(resolved_dir, "portfolio_snapshots"),
        "summary_metrics": _read_table(resolved_dir, "summary_metrics"),
    }


@st.cache_data(show_spinner=False)
def load_data_manifest(data_dir: str | Path | None = None) -> dict[str, Any]:
    resolved_dir = resolve_data_dir(data_dir)
    manifest_path = resolved_dir / "data_manifest.json"
    if not manifest_path.exists():
        return {}
    with manifest_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
