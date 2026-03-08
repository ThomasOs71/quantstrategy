from __future__ import annotations

import json
import math
import re
import urllib.request
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


SHILLER_LANDING_URL = "https://shillerdata.com/"
LEGACY_DATA_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
PAPER_END_QUARTER = pd.Period("2025Q2", freq="Q")
TRAINING_OBSERVATIONS = 20
FORWARD_HORIZON_QUARTERS = 40
FORWARD_HORIZON_YEARS = 10.0
PAPER_REPORT_DIVISOR = 4.0


@dataclass(frozen=True)
class BucketSplit:
    labels: pd.Series
    high_cutoff: float
    low_cutoff: float


def discover_data_url() -> str:
    try:
        html = urllib.request.urlopen(SHILLER_LANDING_URL, timeout=20).read().decode("utf-8", "ignore")
    except Exception:
        return LEGACY_DATA_URL

    match = re.search(r"""href=["'](?P<href>(?:https?:)?//[^"']+/ie_data\.xls[^"']*)["']""", html, re.IGNORECASE)
    if not match:
        return LEGACY_DATA_URL

    href = match.group("href")
    if href.startswith("//"):
        href = "https:" + href
    return href


def load_monthly_data(data_url: str) -> pd.DataFrame:
    raw = pd.read_excel(data_url, sheet_name="Data", header=7)
    renamed = raw.rename(
        columns={
            "Date": "date_code",
            "P": "price",
            "D": "dividend",
            "E": "earnings",
            "CPI": "cpi",
            "Price.1": "real_total_return_index",
            "CAPE": "cape",
        }
    )

    keep = ["date_code", "price", "dividend", "earnings", "cpi", "real_total_return_index", "cape"]
    monthly = renamed.loc[:, keep].copy()

    for column in keep:
        monthly[column] = pd.to_numeric(monthly[column], errors="coerce")

    monthly = monthly[monthly["date_code"].notna()].copy()
    monthly["year"] = np.floor(monthly["date_code"]).astype(int)
    monthly["month"] = np.rint((monthly["date_code"] - monthly["year"]) * 100).astype(int)
    monthly = monthly[monthly["month"].between(1, 12)].copy()
    monthly["period"] = pd.PeriodIndex.from_fields(year=monthly["year"], month=monthly["month"], freq="M")
    monthly = monthly.sort_values("period").reset_index(drop=True)

    return monthly


def build_quarterly_frame(monthly: pd.DataFrame) -> pd.DataFrame:
    quarterly = monthly[monthly["month"].isin([3, 6, 9, 12])].copy()
    quarterly["quarter"] = quarterly["period"].dt.asfreq("Q")
    quarterly["real_tr_index"] = quarterly["real_total_return_index"]
    quarterly["nominal_tr_index"] = quarterly["real_tr_index"] * quarterly["cpi"]
    quarterly["dy"] = quarterly["dividend"].shift(1) / quarterly["price"]
    quarterly["ey"] = quarterly["earnings"].shift(1) / quarterly["price"]
    quarterly["cy"] = 1.0 / quarterly["cape"]

    quarterly["fwd_nominal"] = (
        quarterly["nominal_tr_index"].shift(-FORWARD_HORIZON_QUARTERS) / quarterly["nominal_tr_index"]
    ) ** (1.0 / FORWARD_HORIZON_YEARS) - 1.0
    quarterly["fwd_real"] = (
        quarterly["real_tr_index"].shift(-FORWARD_HORIZON_QUARTERS) / quarterly["real_tr_index"]
    ) ** (1.0 / FORWARD_HORIZON_YEARS) - 1.0

    return quarterly.reset_index(drop=True)


def safe_corr(left: Iterable[float], right: Iterable[float]) -> float:
    x = np.asarray(list(left), dtype=float)
    y = np.asarray(list(right), dtype=float)
    if len(x) < 2:
        return math.nan
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return math.nan
    return float(np.corrcoef(x, y)[0, 1])


def fit_simple_ols(x: pd.Series, y: pd.Series) -> tuple[float, float]:
    x_values = x.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)

    x_mean = float(x_values.mean())
    y_mean = float(y_values.mean())
    sxx = float(np.square(x_values - x_mean).sum())

    if math.isclose(sxx, 0.0):
        return y_mean, 0.0

    sxy = float(((x_values - x_mean) * (y_values - y_mean)).sum())
    beta = sxy / sxx
    alpha = y_mean - beta * x_mean
    return alpha, beta


def assign_quartile_buckets(signal: pd.Series) -> BucketSplit:
    valid = signal.dropna()
    ordered_index = valid.sort_values(kind="mergesort").index.to_numpy()
    buckets = np.array_split(ordered_index, 4)

    labels = pd.Series(index=signal.index, dtype="object")
    labels.loc[buckets[0]] = "low"
    labels.loc[np.concatenate([buckets[1], buckets[2]])] = "middle"
    labels.loc[buckets[3]] = "high"

    low_cutoff = float(valid.loc[buckets[0]].max())
    high_cutoff = float(valid.loc[buckets[3]].min())

    return BucketSplit(labels=labels, high_cutoff=high_cutoff, low_cutoff=low_cutoff)


def summarize_correlations(frame: pd.DataFrame) -> dict[str, float]:
    return {
        "nominal": safe_corr(frame["signal"], frame["fwd_nominal"]),
        "real": safe_corr(frame["signal"], frame["fwd_real"]),
    }


def summarize_forecasts(frame: pd.DataFrame) -> dict[str, float]:
    ordered = frame.sort_values("quarter").reset_index(drop=True)
    if len(ordered) < TRAINING_OBSERVATIONS:
        return {
            "rho": math.nan,
            "md_pct_annualized": math.nan,
            "mad_pct_annualized": math.nan,
            "md_pct_paper": math.nan,
            "mad_pct_paper": math.nan,
            "alpha_raw": math.nan,
            "beta_raw": math.nan,
            "alpha_paper": math.nan,
            "beta_paper": math.nan,
            "forecast_count": 0,
        }

    observed: list[float] = []
    predicted: list[float] = []
    last_alpha = math.nan
    last_beta = math.nan

    for stop in range(TRAINING_OBSERVATIONS - 1, len(ordered)):
        estimation = ordered.iloc[: stop + 1]
        last_alpha, last_beta = fit_simple_ols(estimation["signal"], estimation["fwd_real"])
        observed.append(float(ordered.at[stop, "fwd_real"]))
        predicted.append(last_alpha + last_beta * float(ordered.at[stop, "signal"]))

    observed_array = np.asarray(observed, dtype=float)
    predicted_array = np.asarray(predicted, dtype=float)
    md_pct = float((predicted_array - observed_array).mean() * 100.0)
    mad_pct = float(np.abs(predicted_array - observed_array).mean() * 100.0)
    alpha = float(last_alpha)
    beta = float(last_beta)

    return {
        "rho": safe_corr(predicted_array, observed_array),
        "md_pct_annualized": md_pct,
        "mad_pct_annualized": mad_pct,
        "md_pct_paper": md_pct / PAPER_REPORT_DIVISOR,
        "mad_pct_paper": mad_pct / PAPER_REPORT_DIVISOR,
        "alpha_raw": alpha,
        "beta_raw": beta,
        "alpha_paper": alpha / PAPER_REPORT_DIVISOR,
        "beta_paper": beta / PAPER_REPORT_DIVISOR,
        "forecast_count": int(len(predicted_array)),
    }


def build_signal_frame(quarterly: pd.DataFrame, signal_column: str) -> pd.DataFrame:
    frame = quarterly.loc[:, ["quarter", signal_column, "fwd_nominal", "fwd_real"]].copy()
    frame = frame.rename(columns={signal_column: "signal"})
    frame = frame.dropna(subset=["signal", "fwd_nominal", "fwd_real"]).reset_index(drop=True)
    return frame


def analyze_signal(quarterly: pd.DataFrame, signal_column: str) -> dict[str, object]:
    frame = build_signal_frame(quarterly, signal_column)
    buckets = assign_quartile_buckets(frame["signal"])

    all_frame = frame
    hl_frame = frame[buckets.labels.isin(["high", "low"])].copy()
    middle_frame = frame[buckets.labels.eq("middle")].copy()

    return {
        "cutoffs": {
            "high_ge": buckets.high_cutoff,
            "low_le": buckets.low_cutoff,
        },
        "sample_sizes": {
            "all": int(len(all_frame)),
            "high_low": int(len(hl_frame)),
            "middle": int(len(middle_frame)),
        },
        "correlations": {
            "all": summarize_correlations(all_frame),
            "high_low": summarize_correlations(hl_frame),
            "middle": summarize_correlations(middle_frame),
        },
        "forecasts": {
            "all": summarize_forecasts(all_frame),
            "high_low": summarize_forecasts(hl_frame),
            "middle": summarize_forecasts(middle_frame),
        },
    }


def analyze_window(quarterly: pd.DataFrame, end_quarter: pd.Period) -> dict[str, object]:
    window = quarterly[quarterly["quarter"] <= end_quarter].copy().reset_index(drop=True)
    if window.empty:
        raise ValueError(f"No quarterly observations are available up to {end_quarter}.")

    window["fwd_nominal"] = (
        window["nominal_tr_index"].shift(-FORWARD_HORIZON_QUARTERS) / window["nominal_tr_index"]
    ) ** (1.0 / FORWARD_HORIZON_YEARS) - 1.0
    window["fwd_real"] = (
        window["real_tr_index"].shift(-FORWARD_HORIZON_QUARTERS) / window["real_tr_index"]
    ) ** (1.0 / FORWARD_HORIZON_YEARS) - 1.0

    latest_month = window["period"].iloc[-1]

    return {
        "end_quarter": str(end_quarter),
        "latest_month_in_window": str(latest_month),
        "signals": {
            "dy": analyze_signal(window, "dy"),
            "ey": analyze_signal(window, "ey"),
            "cy": analyze_signal(window, "cy"),
        },
    }


def fmt(value: float, multiplier: float = 1.0, digits: int = 2) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value * multiplier:.{digits}f}"


def render_text_report(results: dict[str, object]) -> str:
    lines: list[str] = []
    lines.append(f"Data source: {results['data_url']}")
    lines.append(f"Latest month online: {results['latest_month_online']}")
    lines.append(f"Latest complete quarter: {results['latest_complete_quarter']}")
    lines.append("Paper-style MD/MAD/beta/alpha = raw result divided by 4.")

    for window_name, window in results["windows"].items():
        lines.append("")
        lines.append(f"[{window_name}] sample end {window['end_quarter']}")
        for signal_name, signal in window["signals"].items():
            lines.append(
                f"{signal_name.upper()} cutoffs high>={fmt(signal['cutoffs']['high_ge'], 100.0)}% "
                f"low<={fmt(signal['cutoffs']['low_le'], 100.0)}%"
            )

            all_corr = signal["correlations"]["all"]
            hl_corr = signal["correlations"]["high_low"]
            mid_corr = signal["correlations"]["middle"]
            lines.append(
                "  Corr(real): "
                f"all {fmt(all_corr['real'])}, "
                f"H+L {fmt(hl_corr['real'])}, "
                f"middle {fmt(mid_corr['real'])}"
            )
            lines.append(
                "  Corr(nom): "
                f"all {fmt(all_corr['nominal'])}, "
                f"H+L {fmt(hl_corr['nominal'])}, "
                f"middle {fmt(mid_corr['nominal'])}"
            )

            all_fc = signal["forecasts"]["all"]
            hl_fc = signal["forecasts"]["high_low"]
            mid_fc = signal["forecasts"]["middle"]
            lines.append(
                "  Forecast rho(real): "
                f"all {fmt(all_fc['rho'])}, "
                f"H+L {fmt(hl_fc['rho'])}, "
                f"middle {fmt(mid_fc['rho'])}"
            )
            lines.append(
                "  Forecast MAD (paper scale, pct pts): "
                f"all {fmt(all_fc['mad_pct_paper'])}, "
                f"H+L {fmt(hl_fc['mad_pct_paper'])}, "
                f"middle {fmt(mid_fc['mad_pct_paper'])}"
            )
            lines.append(
                "  Final beta (paper scale): "
                f"all {fmt(all_fc['beta_paper'])}, "
                f"H+L {fmt(hl_fc['beta_paper'])}, "
                f"middle {fmt(mid_fc['beta_paper'])}"
            )

    return "\n".join(lines)


def main() -> None:
    data_url = discover_data_url()
    monthly = load_monthly_data(data_url)
    quarterly = build_quarterly_frame(monthly)

    latest_complete_quarter = quarterly["quarter"].max()
    windows: dict[str, dict[str, object]] = {}

    if latest_complete_quarter >= PAPER_END_QUARTER:
        windows["paper_aligned"] = analyze_window(quarterly, PAPER_END_QUARTER)
    windows["latest_online"] = analyze_window(quarterly, latest_complete_quarter)

    results = {
        "data_url": data_url,
        "latest_month_online": str(monthly["period"].iloc[-1]),
        "latest_complete_quarter": str(latest_complete_quarter),
        "windows": windows,
    }

    print(render_text_report(results))
    print("")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
