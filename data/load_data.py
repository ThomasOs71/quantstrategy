"""Data loaders and return-panel assembly for Block 1.

The original monthly implementation is preserved as ``monthly_legacy``.  The
public :func:`build_return_panel` facade also dispatches to the no-purchase
``daily_proxy_2011`` engine for monthly or W-FRI research panels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import re
import tempfile
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from data.asset_universe import (
    ASSET_UNIVERSE,
    COMMON_START,
    DataSource,
    KNOWN_LEGACY_CURRENCY_MISMATCH_KEYS,
    get_driver_keys,
    get_tbd_sources,
)
from data.panel_profiles import (
    PanelFrequency,
    SourceProfile,
    coerce_panel_frequency,
    coerce_source_profile,
    get_frequency_definition,
    get_profile_default_start,
)

logger = logging.getLogger(__name__)

MSCI_RAW_DIR = Path(__file__).parent / "raw" / "msci"
DEFAULT_PANEL_DOWNLOAD_DIR = Path(__file__).parent / "downloads" / "daily_proxy_2011"
EURIBOR_3M_SERIES = "IR3TIB01EZM156N"
USD_3M_SERIES = "TB3MS"


def _read_optional_fred_key() -> str | None:
    """Read a local fallback FRED API key from supported ignored helper paths."""
    search_paths = [
        Path(__file__).parent / "FredAPI.txt",
        Path(__file__).resolve().parent.parent / "API_Keys" / "FredAPI.txt",
    ]
    for path in search_paths:
        if path.exists():
            key = path.read_text(encoding="utf-8").strip()
            if key:
                return key
    return None


def _pad_start_window(start: str | None) -> str:
    """Return a slightly earlier start date to allow first lagged return."""
    if not start:
        return COMMON_START
    start_ts = pd.Timestamp(start)
    return (start_ts - pd.DateOffset(months=2)).strftime("%Y-%m-%d")


def _coerce_series_to_datetime_index(series: pd.Series) -> pd.Series:
    values = pd.Series(pd.to_numeric(series, errors="coerce"), index=series.index)
    values.index = pd.to_datetime(values.index)
    values = values.replace([np.inf, -np.inf], np.nan).dropna()
    return values.sort_index()


def _to_month_end(series: pd.Series) -> pd.Series:
    monthly = _coerce_series_to_datetime_index(series).resample("ME").last().dropna()
    return monthly


def load_fred_series(
    series_id: str,
    start: str = COMMON_START,
    end: str | None = None,
    fred_api_key: str | None = None,
) -> pd.Series:
    """Load a FRED series and aggregate to month end."""
    try:
        from fredapi import Fred
    except ImportError as exc:
        raise ImportError(
            "fredapi is required for FRED loading: pip install fredapi"
        ) from exc

    api_key = fred_api_key or os.environ.get("FRED_API_KEY")
    if not api_key:
        api_key = _read_optional_fred_key()
    if not api_key:
        raise ValueError(
            "Missing FRED API key. Pass fred_api_key, set FRED_API_KEY env var, "
            "or place one of the following files (gitignored local helper files): "
            "data/FredAPI.txt or API_Keys/FredAPI.txt."
        )

    fred = Fred(api_key=api_key)
    try:
        raw = fred.get_series(series_id, observation_start=start, observation_end=end)
    except Exception:
        # fredapi embeds its request URL (including the API key) in some network
        # error tracebacks. Keep the public error useful without leaking secrets.
        raise RuntimeError(f"FRED request failed for {series_id}.") from None
    raw.name = series_id
    return _to_month_end(pd.Series(raw))


def _extract_close_series(raw: pd.DataFrame | pd.Series) -> pd.Series:
    """Normalize yfinance return shape to a 1-D price series."""
    if isinstance(raw, pd.Series):
        return raw
    if not isinstance(raw, pd.DataFrame):
        raise TypeError("yfinance output expected as Series or DataFrame")
    if raw.empty:
        raise ValueError("No yfinance rows returned.")

    close_candidates = ["Close", "Adj Close", "Adj_Close", "close", "adj close"]
    for column in close_candidates:
        if column in raw.columns:
            series = raw[column]
            if isinstance(series, pd.DataFrame):
                if series.shape[1] > 1:
                    raise ValueError(
                        f"Expected a single close column for {column}, got {series.shape[1]}."
                    )
                return series.squeeze()
            return series

    if isinstance(raw.columns, pd.MultiIndex):
        for col in raw.columns:
            if col[1] in close_candidates or col[1].lower() in close_candidates:
                return raw[col].squeeze()

    # Fall back to first numeric column.
    return raw.select_dtypes("number").iloc[:, 0].squeeze()


def load_etf_returns(
    ticker: str,
    start: str = COMMON_START,
    end: str | None = None,
) -> pd.Series:
    """Load adjusted monthly log returns for a yfinance ticker."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError(
            "yfinance is required for ETF loading: pip install yfinance"
        ) from exc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = yf.download(
            ticker,
            start=start,
            end=end,
            interval="1mo",
            auto_adjust=True,
            progress=False,
        )

    prices = _extract_close_series(raw)
    prices = _coerce_series_to_datetime_index(prices)
    prices.index = prices.index.to_period("M").to_timestamp("M")
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


def _parse_date_token(value: object) -> pd.Timestamp:
    if value is None:
        return pd.NaT
    token = str(value).strip().strip('"').strip("'")
    if not token:
        return pd.NaT

    # Common MSCI date formats used in exports.
    formats = (
        "%b-%y",
        "%b-%Y",
        "%Y-%m",
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d.%m.%Y",
    )
    for fmt in formats:
        parsed = pd.to_datetime(token, format=fmt, errors="coerce")
        if not pd.isna(parsed):
            return parsed
    parsed = pd.to_datetime(token, dayfirst=False, errors="coerce")
    if pd.isna(parsed):
        parsed = pd.to_datetime(token, dayfirst=True, errors="coerce")
    return parsed


def _parse_float_token(value: object) -> float:
    if value is None:
        return np.nan
    token = str(value).strip()
    if not token:
        return np.nan
    token = token.replace("\u00a0", "").replace(" ", "")
    token = token.replace("N/A", "").replace("NA", "")
    # Keep sign, digits and decimal separators.
    # Convert european thousands separators while preserving decimals:
    if "," in token and "." in token:
        token = token.replace(",", "")
    elif "," in token and "." not in token:
        token = token.replace(",", ".")
    token = re.sub(r"[^0-9.+-eE]", "", token)
    try:
        return float(token)
    except ValueError:
        return np.nan


def _parse_msci_rows(rows: Iterable[list[str]]) -> pd.DataFrame:
    parsed: list[tuple[pd.Timestamp, float]] = []
    for row in rows:
        if len(row) < 2:
            continue

        date = _parse_date_token(row[0])
        if pd.isna(date):
            continue
        value = _parse_float_token(row[1])
        if pd.isna(value):
            continue
        parsed.append((date, value))

    if not parsed:
        raise ValueError("MSCI CSV contains no valid date/value rows.")

    df = pd.DataFrame(parsed, columns=["date", "index_level"])
    return df.sort_values("date").drop_duplicates("date")


def _resolve_msci_file(filename: str) -> Path:
    """Resolve a requested MSCI file, accepting common alternate extensions."""
    requested = MSCI_RAW_DIR / filename
    if requested.exists():
        return requested

    stem = requested.stem
    for extension in (".csv", ".xls", ".xlsx"):
        candidate = MSCI_RAW_DIR / f"{stem}{extension}"
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Missing MSCI file: {requested}\n"
        "Place one of these files under data/raw/msci/:\n"
        f"  - {stem}.csv\n  - {stem}.xls\n  - {stem}.xlsx\n"
    )


def _read_msci_rows_from_file(path: Path) -> list[list[str]]:
    """Read raw date/value rows from MSCI CSV/XLS/XLSX file."""
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", errors="ignore") as handle:
            reader = csv.reader(handle)
            return [list(row) for row in reader]

    try:
        frame = pd.read_excel(path, header=None)
    except Exception as exc:
        raise ValueError(
            f"Could not parse MSCI file '{path}'. Export as plain CSV if parsing fails."
        ) from exc
    return frame.where(pd.notna(frame), None).values.tolist()


def inspect_msci_source_file(filename: str) -> dict[str, str | None]:
    """Return audit-only MSCI file metadata without changing legacy values."""
    path = _resolve_msci_file(filename)
    rows = _read_msci_rows_from_file(path)
    currencies: set[str] = set()
    for row in rows:
        cells = [cell for cell in row if cell is not None and str(cell).strip()]
        for position, cell in enumerate(cells):
            token = str(cell).strip()
            if ":" in token:
                key, value = token.split(":", 1)
            else:
                key, value = token, ""
            normalized_key = re.sub(r"[^a-z]", "", key.lower())
            if normalized_key != "currency":
                continue
            if not value.strip() and position + 1 < len(cells):
                value = str(cells[position + 1])
            currency = value.strip().upper()
            if currency:
                currencies.add(currency)
    if len(currencies) > 1:
        raise ValueError(
            f"MSCI file {path.name!r} declares conflicting currencies: "
            f"{', '.join(sorted(currencies))}"
        )
    currency = next(iter(currencies), None)
    return {
        "filename": path.name,
        "declared_currency": currency,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def load_msci_csv(
    filename: str,
    start: str = COMMON_START,
) -> pd.Series:
    """Load monthly MSCI index CSVs exported from app2.msci.com."""
    filepath = _resolve_msci_file(filename)

    rows = _read_msci_rows_from_file(filepath)

    frame = _parse_msci_rows(rows)
    frame["date"] = pd.to_datetime(frame["date"])
    frame.index = frame["date"].dt.to_period("M").dt.to_timestamp("M")

    prices = pd.Series(frame["index_level"].to_numpy(dtype=float), index=frame.index)
    log_returns = np.log(prices / prices.shift(1)).dropna()
    log_returns = log_returns[log_returns.index >= pd.Timestamp(start)]
    log_returns.name = filename
    return log_returns


def apply_eurusd_hedge_formula(
    usd_log_returns: pd.Series,
    euribor_3m_annual: pd.Series,
    usd_3m_annual: pd.Series,
) -> pd.Series:
    """Apply approximate FX-hedged return formula in log-return space.

    r_hedged_t = r_usd_t + (euribor_3m_t - usd_3m_t) / 12 / 100
    """
    common_idx = usd_log_returns.index.intersection(
        euribor_3m_annual.index
    ).intersection(usd_3m_annual.index)
    carry = (
        (euribor_3m_annual.loc[common_idx] - usd_3m_annual.loc[common_idx]) / 12 / 100
    )
    hedged = usd_log_returns.loc[common_idx] + carry
    base_name = usd_log_returns.name or "usd_asset"
    hedged.name = base_name + "_eur_hedged"
    return hedged


def convert_usd_to_eur(
    usd_log_returns: pd.Series,
    eurusd_log_returns: pd.Series,
) -> pd.Series:
    """Convert USD log returns to EUR with the DEXUSEU convention.

    DEXUSEU is USD per 1 EUR, so the EUR-return is:
        r_eur_t = r_usd_t - r_eurusd_t
    """
    common_idx = usd_log_returns.index.intersection(eurusd_log_returns.index)
    r_eur = usd_log_returns.loc[common_idx] - eurusd_log_returns.loc[common_idx]
    base_name = usd_log_returns.name or "usd_asset"
    r_eur.name = base_name + "_eur"
    return r_eur


def _build_monthly_legacy_return_panel(
    start: str = COMMON_START,
    end: str | None = None,
    fred_api_key: str | None = None,
    warn_tbd: bool = True,
    strict_no_nan: bool = False,
) -> pd.DataFrame:
    """Build the frozen pre-profile monthly panel with all 12 drivers."""
    if warn_tbd:
        tbd = get_tbd_sources()
        if tbd:
            logger.warning(
                "Source IDs still marked tbd: %s. Verify before production.",
                ", ".join(tbd),
            )

    panel_start = pd.Timestamp(start)
    series_start = _pad_start_window(start)
    panel_end = pd.Timestamp(end) if end else None
    series: dict[str, pd.Series] = {}

    logger.info("Loading FRED helper series...")
    euribor_3m = load_fred_series(
        EURIBOR_3M_SERIES,
        start=series_start,
        end=end,
        fred_api_key=fred_api_key,
    )
    usd_3m = load_fred_series(
        USD_3M_SERIES,
        start=series_start,
        end=end,
        fred_api_key=fred_api_key,
    )
    eurusd_raw = load_fred_series(
        "DEXUSEU",
        start=series_start,
        end=end,
        fred_api_key=fred_api_key,
    )
    eurusd_lr = np.log(eurusd_raw / eurusd_raw.shift(1)).dropna()
    series["fx_eurusd"] = eurusd_lr

    logger.info("Loading MSCI index returns from CSV...")
    euro_equities_lr = load_msci_csv("msci_emu_ntr_usd.csv", start=series_start)
    global_dm_ex_emu_lr = load_msci_csv(
        "msci_world_ex_emu_ntr_usd.csv", start=series_start
    )
    series["euro_equities"] = convert_usd_to_eur(euro_equities_lr, eurusd_lr)
    series["global_dm_ex_emu"] = convert_usd_to_eur(global_dm_ex_emu_lr, eurusd_lr)

    logger.info("Loading USD-exposed assets...")
    for key in ("em_equities", "gold", "commodities"):
        asset = ASSET_UNIVERSE[key]
        if asset.source == DataSource.FRED:
            raw_prices = load_fred_series(
                asset.source_id,
                start=series_start,
                end=end,
                fred_api_key=fred_api_key,
            )
            usd_lr = np.log(raw_prices / raw_prices.shift(1)).dropna()
        else:
            if asset.source != DataSource.YFINANCE:
                raise ValueError(
                    f"Unsupported source for USD-exposed asset {key}: {asset.source}"
                )
            usd_lr = load_etf_returns(asset.source_id, start=series_start, end=end)
        series[key] = convert_usd_to_eur(usd_lr, eurusd_lr)

    logger.info("Loading EUR-native ETF assets...")
    for key in ("euro_govt_bond_7_10", "euro_ig_credit", "euro_high_yield"):
        asset = ASSET_UNIVERSE[key]
        if asset.source != DataSource.YFINANCE:
            raise ValueError(
                f"Unsupported source for EUR-native asset {key}: {asset.source}"
            )
        series[key] = load_etf_returns(asset.source_id, start=series_start, end=end)

    logger.info("Applying EUR carry approximation for hedged fixed-income proxies...")
    for key in ("global_govt_bond_eur_hedged", "em_hc_bond_eur_hedged"):
        asset = ASSET_UNIVERSE[key]
        if asset.source != DataSource.YFINANCE_HEDGE:
            raise ValueError(
                f"Unsupported source for hedged asset {key}: {asset.source}"
            )
        usd_lr = load_etf_returns(asset.source_id, start=series_start, end=end)
        series[key] = apply_eurusd_hedge_formula(usd_lr, euribor_3m, usd_3m)

    logger.info("Converting and adding cash return series...")
    cash_rate = euribor_3m.copy()
    series["cash"] = np.log(1 + cash_rate / 100 / 12)

    ordered_keys = get_driver_keys()
    missing = [key for key in ordered_keys if key not in series]
    if missing:
        raise ValueError(f"Missing series for keys: {missing}")

    panel = pd.DataFrame({key: series[key] for key in ordered_keys})
    panel = panel[panel.index >= panel_start]
    if panel_end is not None:
        panel = panel[panel.index <= panel_end]

    if panel.empty:
        raise ValueError("Built panel is empty. Check sources and date range.")

    nan_counts = panel.isna().sum()
    if nan_counts.any():
        nan_summary = nan_counts[nan_counts > 0]
        logger.warning("Panel contains NaN values after merge:\n%s", nan_summary)
        if strict_no_nan:
            raise ValueError(
                "Panel contains NaN values. Set strict_no_nan=False to allow warnings."
            )

    logger.info(
        "Build complete: shape=%s, from=%s to=%s",
        panel.shape,
        panel.index[0].date(),
        panel.index[-1].date(),
    )
    return panel


def build_return_panel(
    start: str | None = None,
    end: str | None = None,
    fred_api_key: str | None = None,
    warn_tbd: bool = True,
    strict_no_nan: bool | None = None,
    *,
    frequency: str | PanelFrequency = PanelFrequency.MONTHLY,
    source_profile: str | SourceProfile = SourceProfile.MONTHLY_LEGACY,
    warn_legacy_currency: bool = True,
) -> pd.DataFrame:
    """Build a Block-1 EUR log-return panel under an explicit source contract.

    With no arguments this preserves the historical monthly implementation.
    ``monthly_legacy`` intentionally reproduces its old transformations and is
    restricted to monthly data.  ``daily_proxy_2011`` uses daily free-access
    research proxies, corrected quote-currency treatments, and the same engine
    for monthly and W-FRI returns. It does not require purchased index files.

    ``strict_no_nan=None`` preserves the permissive legacy default while making
    requested ``daily_proxy_2011`` windows strict by default.
    """
    parsed_frequency = coerce_panel_frequency(frequency)
    parsed_profile = coerce_source_profile(source_profile)
    if (
        parsed_profile is SourceProfile.MONTHLY_LEGACY
        and parsed_frequency is not PanelFrequency.MONTHLY
    ):
        raise ValueError(
            "source_profile='monthly_legacy' supports only frequency='monthly'. "
            "Use source_profile='daily_proxy_2011' for weekly data."
        )
    resolved_start = start or get_profile_default_start(
        parsed_profile, parsed_frequency
    )

    if parsed_profile is SourceProfile.MONTHLY_LEGACY:
        panel = _build_monthly_legacy_return_panel(
            start=resolved_start,
            end=end,
            fred_api_key=fred_api_key,
            warn_tbd=warn_tbd,
            strict_no_nan=bool(strict_no_nan),
        )
        mismatch_keys = ["em_equities", "commodities"]
        source_audit: dict[str, dict[str, str | None]] = {}
        for key, filename in (
            ("euro_equities", "msci_emu_ntr_usd.csv"),
            ("global_dm_ex_emu", "msci_world_ex_emu_ntr_usd.csv"),
        ):
            try:
                audit = inspect_msci_source_file(filename)
            except (FileNotFoundError, ValueError):
                audit = {
                    "filename": filename,
                    "declared_currency": None,
                    "sha256": None,
                }
            source_audit[key] = audit
            if audit["declared_currency"] != "USD":
                mismatch_keys.append(key)
        mismatch_keys = [
            key for key in KNOWN_LEGACY_CURRENCY_MISMATCH_KEYS if key in mismatch_keys
        ]
        if warn_legacy_currency:
            logger.warning(
                "monthly_legacy reproduces known source-currency mismatches for: %s. "
                "Use daily_proxy_2011 for corrected currency semantics.",
                ", ".join(mismatch_keys),
            )
        frequency_definition = get_frequency_definition(parsed_frequency)
        panel.attrs.update(
            {
                "source_profile": parsed_profile.value,
                "frequency": frequency_definition.metadata_label,
                "periods_per_year": frequency_definition.periods_per_year,
                "return_representation": "EUR monthly log returns",
                "known_currency_mismatch": bool(mismatch_keys),
                "known_currency_mismatch_keys": mismatch_keys,
                "legacy_source_audit": source_audit,
            }
        )
        return panel

    from data.profiled_load_data import build_daily_proxy_2011_return_panel

    resolved_strict = True if strict_no_nan is None else strict_no_nan
    return build_daily_proxy_2011_return_panel(
        start=resolved_start,
        end=end,
        frequency=parsed_frequency,
        fred_api_key=fred_api_key,
        strict_no_nan=resolved_strict,
    )


def _commit_panel_download_pair(
    *,
    csv_temp: Path,
    manifest_temp: Path,
    csv_path: Path,
    manifest_path: Path,
    overwrite: bool,
) -> None:
    """Commit a CSV/manifest pair with writer exclusion and rollback."""
    lock_path = csv_path.with_name(f".{csv_path.stem}.lock")
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise RuntimeError(
            f"Another download is committing {csv_path.name}; retry after it finishes."
        ) from None
    try:
        os.close(lock_fd)
        backups: dict[Path, Path] = {}
        installed: list[Path] = []
        try:
            existing = [path for path in (csv_path, manifest_path) if path.exists()]
            if existing and not overwrite:
                joined = ", ".join(str(path) for path in existing)
                raise FileExistsError(
                    f"Panel download already exists: {joined}. "
                    "Pass overwrite=True to replace it."
                )

            for target in existing:
                with tempfile.NamedTemporaryFile(
                    dir=target.parent,
                    prefix=f".{target.name}.",
                    suffix=".bak",
                    delete=False,
                ) as handle:
                    backup = Path(handle.name)
                backup.unlink()
                target.replace(backup)
                backups[target] = backup

            csv_temp.replace(csv_path)
            installed.append(csv_path)
            manifest_temp.replace(manifest_path)
            installed.append(manifest_path)
        except BaseException:
            # Roll back even on KeyboardInterrupt/SystemExit so the CSV and its
            # checksum manifest can never be left as a half-committed pair.
            for installed_path in reversed(installed):
                installed_path.unlink(missing_ok=True)
            for target, backup in backups.items():
                if backup.exists():
                    backup.replace(target)
            raise
        else:
            for backup in backups.values():
                backup.unlink(missing_ok=True)
    finally:
        lock_path.unlink(missing_ok=True)


def download_return_panel(
    *,
    frequency: str | PanelFrequency,
    output_dir: str | Path | None = None,
    start: str | None = None,
    end: str | None = None,
    fred_api_key: str | None = None,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Download and persist one ``daily_proxy_2011`` return panel.

    ``frequency`` must be ``"monthly"`` or ``"weekly"``. The function writes
    an ISO-date CSV and a JSON provenance manifest. By default both files live
    in the frequency-specific ``data/downloads/daily_proxy_2011/monthly/`` or
    ``weekly/`` folder, which is ignored by Git.

    Custom destinations outside the repository are allowed. Within this
    repository, downloads are restricted to ``data/downloads/`` so a caller
    cannot accidentally place provider-derived observations in a tracked path.
    Existing snapshots are preserved unless ``overwrite=True``.
    """
    parsed_frequency = coerce_panel_frequency(frequency)
    destination = (
        Path(output_dir)
        if output_dir is not None
        else DEFAULT_PANEL_DOWNLOAD_DIR / parsed_frequency.value
    )
    destination = destination.expanduser().resolve()
    repository_root = Path(__file__).resolve().parent.parent
    ignored_download_root = (repository_root / "data" / "downloads").resolve()
    if destination.is_relative_to(repository_root) and not destination.is_relative_to(
        ignored_download_root
    ):
        raise ValueError(
            "In-repository panel downloads must be written under data/downloads/, "
            "which is protected by .gitignore."
        )

    panel = build_return_panel(
        start=start,
        end=end,
        fred_api_key=fred_api_key,
        strict_no_nan=True,
        frequency=parsed_frequency,
        source_profile=SourceProfile.DAILY_PROXY_2011,
    )
    effective_start = panel.index.min().date().isoformat()
    effective_end = panel.index.max().date().isoformat()
    stem = (
        f"daily_proxy_2011_{parsed_frequency.value}_"
        f"{effective_start}_{effective_end}"
    )
    csv_path = destination / f"{stem}.csv"
    manifest_path = destination / f"{stem}.manifest.json"
    existing = [path for path in (csv_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(
            f"Panel download already exists: {joined}. Pass overwrite=True to replace it."
        )

    destination.mkdir(parents=True, exist_ok=True)
    csv_temp: Path | None = None
    manifest_temp: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination, prefix=f".{stem}.", suffix=".csv.tmp", delete=False
        ) as handle:
            csv_temp = Path(handle.name)
        panel.to_csv(
            csv_temp,
            index=True,
            index_label="date",
            date_format="%Y-%m-%d",
            float_format="%.17g",
        )
        csv_sha256 = hashlib.sha256(csv_temp.read_bytes()).hexdigest()
        manifest = {
            "schema_version": 1,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_profile": SourceProfile.DAILY_PROXY_2011.value,
            "frequency_option": parsed_frequency.value,
            "rows": int(panel.shape[0]),
            "columns": int(panel.shape[1]),
            "driver_keys": list(panel.columns),
            "effective_start": effective_start,
            "effective_end": effective_end,
            "csv_filename": csv_path.name,
            "csv_sha256": csv_sha256,
            "panel_attrs": dict(panel.attrs),
        }
        manifest_text = (
            json.dumps(
                manifest,
                indent=2,
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            )
            + "\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination,
            prefix=f".{stem}.",
            suffix=".json.tmp",
            delete=False,
        ) as handle:
            handle.write(manifest_text)
            manifest_temp = Path(handle.name)

        _commit_panel_download_pair(
            csv_temp=csv_temp,
            manifest_temp=manifest_temp,
            csv_path=csv_path,
            manifest_path=manifest_path,
            overwrite=overwrite,
        )
        csv_temp = None
        manifest_temp = None
    finally:
        for temporary_path in (csv_temp, manifest_temp):
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    return {"data": csv_path, "manifest": manifest_path}


def load_return_panel_download(
    data_path: str | Path,
    manifest_path: str | Path | None = None,
) -> pd.DataFrame:
    """Load and verify a locally persisted return-panel snapshot."""
    csv_path = Path(data_path).expanduser().resolve()
    resolved_manifest = (
        Path(manifest_path).expanduser().resolve()
        if manifest_path is not None
        else csv_path.with_suffix(".manifest.json")
    )
    if not csv_path.is_file():
        raise FileNotFoundError(f"Panel CSV does not exist: {csv_path}")
    if not resolved_manifest.is_file():
        raise FileNotFoundError(f"Panel manifest does not exist: {resolved_manifest}")

    try:
        manifest = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError(
            f"Panel manifest is not valid JSON: {resolved_manifest}"
        ) from exc
    if not isinstance(manifest, dict):
        raise ValueError("Panel manifest must contain a JSON object.")
    if manifest.get("source_profile") != SourceProfile.DAILY_PROXY_2011.value:
        raise ValueError("Panel manifest has an unsupported source_profile.")
    if manifest.get("csv_filename") != csv_path.name:
        raise ValueError("Panel manifest csv_filename does not match the CSV path.")
    observed_sha256 = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if manifest.get("csv_sha256") != observed_sha256:
        raise ValueError("Panel CSV checksum does not match its manifest.")

    try:
        panel = pd.read_csv(csv_path, parse_dates=["date"]).set_index("date")
    except (ValueError, KeyError) as exc:
        raise ValueError("Panel CSV must contain a parseable date column.") from exc
    panel.index = pd.DatetimeIndex(panel.index)
    if not panel.index.is_monotonic_increasing or panel.index.has_duplicates:
        raise ValueError("Panel CSV dates must be sorted and unique.")
    if list(panel.columns) != get_driver_keys():
        raise ValueError("Panel CSV columns do not match the canonical driver order.")
    if not np.isfinite(panel.to_numpy(dtype=float)).all():
        raise ValueError("Panel CSV must contain only finite returns.")
    if panel.shape != (manifest.get("rows"), manifest.get("columns")):
        raise ValueError("Panel CSV dimensions do not match its manifest.")
    if panel.index.min().date().isoformat() != manifest.get("effective_start"):
        raise ValueError("Panel CSV start date does not match its manifest.")
    if panel.index.max().date().isoformat() != manifest.get("effective_end"):
        raise ValueError("Panel CSV end date does not match its manifest.")
    panel_attrs = manifest.get("panel_attrs")
    if not isinstance(panel_attrs, dict):
        raise ValueError("Panel manifest panel_attrs must be a JSON object.")
    panel.attrs.update(panel_attrs)
    panel.attrs.update(
        {
            "snapshot_csv_filename": csv_path.name,
            "snapshot_manifest_filename": resolved_manifest.name,
            "snapshot_csv_sha256": observed_sha256,
        }
    )
    return panel
