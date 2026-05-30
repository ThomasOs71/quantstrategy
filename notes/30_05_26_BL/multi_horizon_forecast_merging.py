from __future__ import annotations

import argparse
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "articles" / "cov_part2" / "data_raw"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs"
DEFAULT_TICKERS = ("SPY", "EFA", "EEM", "AGG", "GLD")
DEFAULT_HORIZONS = (4, 12)


@dataclass(frozen=True)
class View:
    name: str
    horizon_weeks: int
    exposures: dict[str, float]
    forecast_return: float
    forecast_vol: float


@dataclass(frozen=True)
class SegmentModel:
    tickers: list[str]
    horizons: np.ndarray
    deltas: np.ndarray
    mu: np.ndarray
    cov: np.ndarray
    cumulative_matrix: np.ndarray
    segment_prior_mean: np.ndarray
    segment_prior_cov: np.ndarray


@dataclass(frozen=True)
class MergeResult:
    scenario: str
    views: list[View]
    view_matrix: np.ndarray
    view_values: np.ndarray
    view_noise: np.ndarray
    gain: np.ndarray
    innovation: np.ndarray
    segment_posterior_mean: np.ndarray
    segment_posterior_cov: np.ndarray


def parse_tickers(raw: str) -> list[str]:
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def stooq_file_name(ticker: str, suffix: str = ".us") -> str:
    return f"{ticker.lower()}{suffix}.csv"


def load_local_stooq_prices(tickers: list[str], data_dir: Path, suffix: str = ".us") -> pd.DataFrame:
    series: list[pd.Series] = []
    missing: list[str] = []

    for ticker in tickers:
        path = data_dir / stooq_file_name(ticker, suffix=suffix)
        if not path.exists():
            missing.append(str(path))
            continue

        frame = pd.read_csv(path)
        if "Date" not in frame.columns or "Close" not in frame.columns:
            raise ValueError(f"{path} must contain Date and Close columns.")
        frame["Date"] = pd.to_datetime(frame["Date"])
        close = frame.sort_values("Date").set_index("Date")["Close"].astype(float).rename(ticker)
        series.append(close)

    if missing:
        raise FileNotFoundError("Missing local Stooq files:\n" + "\n".join(missing))

    prices = pd.concat(series, axis=1).sort_index()
    return prices.dropna(how="any")


def weekly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    weekly_prices = prices.resample("W-FRI").last().dropna(how="any")
    returns = weekly_prices.pct_change().dropna(how="any")
    returns.index.name = "date"
    return returns


def estimate_prior(returns: pd.DataFrame, start: str, end: str) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    sample = returns.loc[pd.Timestamp(start) : pd.Timestamp(end)].copy()
    if sample.empty:
        raise ValueError("The requested prior estimation window is empty.")
    mu = sample.mean().to_numpy(dtype=float)
    cov = sample.cov().to_numpy(dtype=float)
    cov = 0.5 * (cov + cov.T)
    return sample, mu, cov


def block_diag(blocks: list[np.ndarray]) -> np.ndarray:
    rows = sum(block.shape[0] for block in blocks)
    cols = sum(block.shape[1] for block in blocks)
    out = np.zeros((rows, cols), dtype=float)
    row = 0
    col = 0
    for block in blocks:
        r, c = block.shape
        out[row : row + r, col : col + c] = block
        row += r
        col += c
    return out


def build_cumulative_matrix(n_assets: int, n_horizons: int) -> np.ndarray:
    out = np.zeros((n_assets * n_horizons, n_assets * n_horizons), dtype=float)
    identity = np.eye(n_assets)
    for horizon_idx in range(n_horizons):
        row = slice(horizon_idx * n_assets, (horizon_idx + 1) * n_assets)
        for segment_idx in range(horizon_idx + 1):
            col = slice(segment_idx * n_assets, (segment_idx + 1) * n_assets)
            out[row, col] = identity
    return out


def build_segment_model(
    tickers: list[str],
    mu: np.ndarray,
    cov: np.ndarray,
    horizons: tuple[int, ...] | list[int] = DEFAULT_HORIZONS,
) -> SegmentModel:
    unique_horizons = np.array(sorted(set(int(h) for h in horizons)), dtype=int)
    if unique_horizons.size < 2:
        raise ValueError("Use at least two horizons for the multi-horizon example.")
    if np.any(unique_horizons <= 0):
        raise ValueError("Horizons must be positive.")

    n_assets = len(tickers)
    if mu.shape != (n_assets,):
        raise ValueError("mu shape does not match tickers.")
    if cov.shape != (n_assets, n_assets):
        raise ValueError("covariance shape does not match tickers.")

    deltas = np.diff(np.concatenate([[0], unique_horizons])).astype(float)
    cumulative_matrix = build_cumulative_matrix(n_assets, len(unique_horizons))
    segment_prior_mean = np.concatenate([delta * mu for delta in deltas])
    segment_prior_cov = block_diag([delta * cov for delta in deltas])
    segment_prior_cov = 0.5 * (segment_prior_cov + segment_prior_cov.T)

    return SegmentModel(
        tickers=tickers,
        horizons=unique_horizons,
        deltas=deltas,
        mu=mu,
        cov=cov,
        cumulative_matrix=cumulative_matrix,
        segment_prior_mean=segment_prior_mean,
        segment_prior_cov=segment_prior_cov,
    )


def exposure_vector(view: View, tickers: list[str]) -> np.ndarray:
    unknown = sorted(set(view.exposures) - set(tickers))
    if unknown:
        raise ValueError(f"View {view.name} references unknown tickers: {unknown}")
    return np.array([float(view.exposures.get(ticker, 0.0)) for ticker in tickers], dtype=float)


def build_view_system(model: SegmentModel, views: list[View]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_assets = len(model.tickers)
    if not views:
        return (
            np.zeros((0, model.segment_prior_mean.size), dtype=float),
            np.zeros(0, dtype=float),
            np.zeros((0, 0), dtype=float),
        )

    z_rows: list[np.ndarray] = []
    values: list[float] = []
    noise_vars: list[float] = []

    for view in views:
        if view.horizon_weeks not in set(model.horizons.tolist()):
            raise ValueError(
                f"View {view.name} uses horizon {view.horizon_weeks}, "
                f"but available horizons are {model.horizons.tolist()}."
            )
        if view.forecast_vol <= 0.0:
            raise ValueError(f"View {view.name} must use a positive forecast_vol.")

        horizon_idx = int(np.where(model.horizons == view.horizon_weeks)[0][0])
        row_block = model.cumulative_matrix[
            horizon_idx * n_assets : (horizon_idx + 1) * n_assets,
            :,
        ]
        z_rows.append(exposure_vector(view, model.tickers) @ row_block)
        values.append(float(view.forecast_return))
        noise_vars.append(float(view.forecast_vol) ** 2)

    return np.vstack(z_rows), np.array(values, dtype=float), np.diag(noise_vars)


def merge_forecasts(model: SegmentModel, scenario: str, views: list[View]) -> MergeResult:
    z, y, h = build_view_system(model, views)
    nu = model.segment_prior_mean
    omega = model.segment_prior_cov

    if not views:
        return MergeResult(
            scenario=scenario,
            views=views,
            view_matrix=z,
            view_values=y,
            view_noise=h,
            gain=np.zeros((nu.size, 0), dtype=float),
            innovation=np.zeros(0, dtype=float),
            segment_posterior_mean=nu.copy(),
            segment_posterior_cov=omega.copy(),
        )

    forecast_cov = z @ omega @ z.T + h
    forecast_cov = 0.5 * (forecast_cov + forecast_cov.T)
    gain = np.linalg.solve(forecast_cov, z @ omega).T
    innovation = y - z @ nu
    posterior_mean = nu + gain @ innovation
    posterior_cov = omega - gain @ z @ omega
    posterior_cov = 0.5 * (posterior_cov + posterior_cov.T)

    return MergeResult(
        scenario=scenario,
        views=views,
        view_matrix=z,
        view_values=y,
        view_noise=h,
        gain=gain,
        innovation=innovation,
        segment_posterior_mean=posterior_mean,
        segment_posterior_cov=posterior_cov,
    )


def default_scenarios() -> dict[str, list[View]]:
    return {
        "single_short_view": [
            View(
                name="SPY_minus_AGG_4w_risk_on",
                horizon_weeks=4,
                exposures={"SPY": 1.0, "AGG": -1.0},
                forecast_return=0.015,
                forecast_vol=0.035,
            )
        ],
        "single_long_view": [
            View(
                name="GLD_12w_positive",
                horizon_weeks=12,
                exposures={"GLD": 1.0},
                forecast_return=0.040,
                forecast_vol=0.060,
            )
        ],
        "mixed_horizon_views": [
            View(
                name="SPY_minus_AGG_4w_risk_on",
                horizon_weeks=4,
                exposures={"SPY": 1.0, "AGG": -1.0},
                forecast_return=0.015,
                forecast_vol=0.035,
            ),
            View(
                name="GLD_12w_positive",
                horizon_weeks=12,
                exposures={"GLD": 1.0},
                forecast_return=0.040,
                forecast_vol=0.060,
            ),
        ],
        "conflicting_horizon_views": [
            View(
                name="SPY_minus_AGG_4w_positive",
                horizon_weeks=4,
                exposures={"SPY": 1.0, "AGG": -1.0},
                forecast_return=0.015,
                forecast_vol=0.025,
            ),
            View(
                name="SPY_minus_AGG_12w_negative",
                horizon_weeks=12,
                exposures={"SPY": 1.0, "AGG": -1.0},
                forecast_return=-0.010,
                forecast_vol=0.040,
            ),
        ],
    }


def format_exposures(exposures: dict[str, float]) -> str:
    return "; ".join(f"{ticker}:{weight:+.2f}" for ticker, weight in sorted(exposures.items()))


def horizon_frame(
    model: SegmentModel,
    result: MergeResult,
    matrix_kind: str,
) -> pd.DataFrame:
    n_assets = len(model.tickers)
    prior_mean = model.cumulative_matrix @ model.segment_prior_mean
    posterior_mean = model.cumulative_matrix @ result.segment_posterior_mean
    prior_cov = model.cumulative_matrix @ model.segment_prior_cov @ model.cumulative_matrix.T
    posterior_cov = model.cumulative_matrix @ result.segment_posterior_cov @ model.cumulative_matrix.T

    records: list[dict[str, float | int | str]] = []
    for horizon_idx, horizon in enumerate(model.horizons):
        for asset_idx, ticker in enumerate(model.tickers):
            row = horizon_idx * n_assets + asset_idx
            if matrix_kind == "means":
                prior_value = float(prior_mean[row])
                posterior_value = float(posterior_mean[row])
                key = "return"
            elif matrix_kind == "vols":
                prior_value = math.sqrt(max(float(prior_cov[row, row]), 0.0))
                posterior_value = math.sqrt(max(float(posterior_cov[row, row]), 0.0))
                key = "vol"
            else:
                raise ValueError(f"Unknown matrix_kind: {matrix_kind}")

            records.append(
                {
                    "scenario": result.scenario,
                    "horizon_weeks": int(horizon),
                    "asset": ticker,
                    f"prior_{key}": prior_value,
                    f"posterior_{key}": posterior_value,
                    "change": posterior_value - prior_value,
                }
            )
    return pd.DataFrame.from_records(records)


def view_fit_frame(model: SegmentModel, result: MergeResult) -> pd.DataFrame:
    if not result.views:
        return pd.DataFrame()

    prior_implied = result.view_matrix @ model.segment_prior_mean
    posterior_implied = result.view_matrix @ result.segment_posterior_mean

    records = []
    for idx, view in enumerate(result.views):
        records.append(
            {
                "scenario": result.scenario,
                "view_name": view.name,
                "horizon_weeks": view.horizon_weeks,
                "exposures": format_exposures(view.exposures),
                "forecast_return": float(result.view_values[idx]),
                "forecast_vol": math.sqrt(float(result.view_noise[idx, idx])),
                "prior_implied_return": float(prior_implied[idx]),
                "posterior_implied_return": float(posterior_implied[idx]),
                "posterior_minus_forecast": float(posterior_implied[idx] - result.view_values[idx]),
                "forecast_minus_prior": float(result.view_values[idx] - prior_implied[idx]),
            }
        )
    return pd.DataFrame.from_records(records)


def source_contributions_frame(model: SegmentModel, result: MergeResult) -> pd.DataFrame:
    n_assets = len(model.tickers)
    horizon_prior = model.cumulative_matrix @ model.segment_prior_mean
    posterior = model.cumulative_matrix @ result.segment_posterior_mean
    view_contributions = [
        model.cumulative_matrix @ result.gain[:, idx] * result.innovation[idx]
        for idx in range(len(result.views))
    ]

    records: list[dict[str, float | int | str]] = []
    for horizon_idx, horizon in enumerate(model.horizons):
        for asset_idx, ticker in enumerate(model.tickers):
            row = horizon_idx * n_assets + asset_idx
            records.append(
                {
                    "scenario": result.scenario,
                    "horizon_weeks": int(horizon),
                    "asset": ticker,
                    "source": "prior",
                    "contribution": float(horizon_prior[row]),
                }
            )
            for view, contribution in zip(result.views, view_contributions):
                records.append(
                    {
                        "scenario": result.scenario,
                        "horizon_weeks": int(horizon),
                        "asset": ticker,
                        "source": view.name,
                        "contribution": float(contribution[row]),
                    }
                )
            records.append(
                {
                    "scenario": result.scenario,
                    "horizon_weeks": int(horizon),
                    "asset": ticker,
                    "source": "posterior_total",
                    "contribution": float(posterior[row]),
                }
            )
    return pd.DataFrame.from_records(records)


def segment_frame(model: SegmentModel, result: MergeResult) -> pd.DataFrame:
    n_assets = len(model.tickers)
    records: list[dict[str, float | int | str]] = []

    for segment_idx, delta in enumerate(model.deltas):
        start = 0 if segment_idx == 0 else int(model.horizons[segment_idx - 1])
        end = int(model.horizons[segment_idx])
        label = f"{start}-{end}W"
        for asset_idx, ticker in enumerate(model.tickers):
            row = segment_idx * n_assets + asset_idx
            prior_vol = math.sqrt(max(float(model.segment_prior_cov[row, row]), 0.0))
            posterior_vol = math.sqrt(max(float(result.segment_posterior_cov[row, row]), 0.0))
            records.append(
                {
                    "scenario": result.scenario,
                    "segment": label,
                    "segment_start_week": start,
                    "segment_end_week": end,
                    "segment_length_weeks": int(delta),
                    "asset": ticker,
                    "prior_return": float(model.segment_prior_mean[row]),
                    "posterior_return": float(result.segment_posterior_mean[row]),
                    "return_change": float(result.segment_posterior_mean[row] - model.segment_prior_mean[row]),
                    "prior_vol": prior_vol,
                    "posterior_vol": posterior_vol,
                    "vol_change": posterior_vol - prior_vol,
                }
            )
    return pd.DataFrame.from_records(records)


def prior_stats_frame(sample: pd.DataFrame) -> pd.DataFrame:
    records = []
    for ticker in sample.columns:
        returns = sample[ticker]
        records.append(
            {
                "asset": ticker,
                "sample_start": sample.index.min().date().isoformat(),
                "sample_end": sample.index.max().date().isoformat(),
                "n_weekly_obs": int(returns.shape[0]),
                "weekly_mean": float(returns.mean()),
                "weekly_vol": float(returns.std(ddof=1)),
                "annualized_mean_approx": float(returns.mean() * 52.0),
                "annualized_vol": float(returns.std(ddof=1) * math.sqrt(52.0)),
            }
        )
    return pd.DataFrame.from_records(records)


def view_definitions_frame(scenarios: dict[str, list[View]]) -> pd.DataFrame:
    records = []
    for scenario, views in scenarios.items():
        for view in views:
            records.append(
                {
                    "scenario": scenario,
                    "view_name": view.name,
                    "horizon_weeks": view.horizon_weeks,
                    "exposures": format_exposures(view.exposures),
                    "forecast_return": view.forecast_return,
                    "forecast_vol": view.forecast_vol,
                    "forecast_variance": view.forecast_vol**2,
                }
            )
    return pd.DataFrame.from_records(records)


def percent_axis(ax: plt.Axes) -> None:
    ax.yaxis.set_major_formatter(lambda value, _: f"{100.0 * value:.1f}%")


def add_bottom_explanation(fig: plt.Figure, text: str, wrap: int = 135) -> None:
    fig.text(
        0.5,
        0.025,
        textwrap.fill(text, width=wrap),
        ha="center",
        va="bottom",
        fontsize=9,
        color="#374151",
    )


def plot_prior_vs_posterior_means(means: pd.DataFrame, out_dir: Path) -> None:
    scenario = "mixed_horizon_views" if "mixed_horizon_views" in set(means["scenario"]) else means["scenario"].iloc[0]
    data = means.loc[means["scenario"] == scenario].copy()
    horizons = sorted(data["horizon_weeks"].unique())
    asset_order = data["asset"].drop_duplicates().tolist()

    fig, axes = plt.subplots(1, len(horizons), figsize=(12, 5), sharey=True)
    if len(horizons) == 1:
        axes = [axes]

    for ax, horizon in zip(axes, horizons):
        sub = data.loc[data["horizon_weeks"] == horizon].set_index("asset").loc[asset_order]
        x = np.arange(sub.shape[0])
        width = 0.36
        ax.bar(x - width / 2, sub["prior_return"], width=width, label="Prior", color="#6b7280")
        ax.bar(x + width / 2, sub["posterior_return"], width=width, label="Posterior", color="#2563eb")
        ax.axhline(0.0, color="#111827", linewidth=0.8)
        ax.set_title(f"{scenario}: {horizon}W")
        ax.set_xticks(x)
        ax.set_xticklabels(sub.index.tolist())
        percent_axis(ax)
        ax.grid(axis="y", alpha=0.25)

    axes[0].set_ylabel("Total return over horizon")
    axes[-1].legend(loc="best")
    fig.suptitle("Prior vs posterior asset-horizon means")
    add_bottom_explanation(
        fig,
        "What to read: each pair of bars compares the historical Gaussian prior with the forecast-updated posterior. "
        "Assets can move even without a direct view because the update uses the prior covariance across markets.",
    )
    fig.tight_layout(rect=(0.0, 0.16, 1.0, 0.94))
    fig.savefig(out_dir / "fig_1_prior_vs_posterior_means.png", dpi=180)
    plt.close(fig)


def plot_view_fit(view_fit: pd.DataFrame, out_dir: Path) -> None:
    if view_fit.empty:
        return

    data = view_fit.copy()
    data["label"] = data["scenario"] + "\n" + data["view_name"]
    x = np.arange(data.shape[0])
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 6.8))
    ax.bar(x - width, data["prior_implied_return"], width=width, label="Prior implied", color="#9ca3af")
    ax.bar(x, data["forecast_return"], width=width, label="Forecast / view", color="#f59e0b")
    ax.bar(x + width, data["posterior_implied_return"], width=width, label="Posterior implied", color="#2563eb")
    ax.axhline(0.0, color="#111827", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(data["label"], rotation=35, ha="right")
    ax.set_ylabel("View portfolio total return")
    ax.set_title("View fit: prior vs forecast vs posterior")
    percent_axis(ax)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best")
    add_bottom_explanation(
        fig,
        "What to read: the posterior-implied view should move from the prior-implied value toward the forecast target. "
        "It does not have to equal the forecast because forecast uncertainty and conflicting information are explicitly modeled.",
    )
    fig.tight_layout(rect=(0.0, 0.26, 1.0, 0.96))
    fig.savefig(out_dir / "fig_2_view_fit.png", dpi=180)
    plt.close(fig)


def plot_segment_adjustments(segments: pd.DataFrame, out_dir: Path) -> None:
    scenarios = [item for item in ["mixed_horizon_views", "conflicting_horizon_views"] if item in set(segments["scenario"])]
    if not scenarios:
        scenarios = [segments["scenario"].iloc[0]]
    segment_labels = sorted(segments["segment"].unique(), key=lambda item: int(item.split("-")[0]))
    asset_order = segments["asset"].drop_duplicates().tolist()

    fig, axes = plt.subplots(
        len(scenarios),
        len(segment_labels),
        figsize=(13, 4 * len(scenarios) + 1),
        sharey=True,
    )
    axes = np.asarray(axes).reshape(len(scenarios), len(segment_labels))

    for row_idx, scenario in enumerate(scenarios):
        for col_idx, segment in enumerate(segment_labels):
            ax = axes[row_idx, col_idx]
            sub = (
                segments.loc[(segments["scenario"] == scenario) & (segments["segment"] == segment)]
                .set_index("asset")
                .loc[asset_order]
            )
            colors = np.where(sub["return_change"] >= 0.0, "#059669", "#dc2626")
            ax.bar(sub.index, sub["return_change"], color=colors)
            ax.axhline(0.0, color="#111827", linewidth=0.8)
            ax.set_title(f"{scenario}: {segment}")
            percent_axis(ax)
            ax.grid(axis="y", alpha=0.25)
            if col_idx == 0:
                ax.set_ylabel("Posterior - prior segment return")

    fig.suptitle("Segment adjustments expose the multi-horizon mechanism")
    add_bottom_explanation(
        fig,
        "What to read: the posterior state is split into non-overlapping segments. A 12W forecast is therefore reconciled "
        "through the 0-4W and 4-12W pieces, which is what a one-period Black-Litterman update cannot show.",
    )
    fig.tight_layout(rect=(0.0, 0.09, 1.0, 0.96))
    fig.savefig(out_dir / "fig_3_segment_adjustments.png", dpi=180)
    plt.close(fig)


def plot_source_contributions(contributions: pd.DataFrame, out_dir: Path) -> None:
    scenario = "mixed_horizon_views" if "mixed_horizon_views" in set(contributions["scenario"]) else contributions["scenario"].iloc[0]
    horizon = int(contributions["horizon_weeks"].max())
    data = contributions.loc[
        (contributions["scenario"] == scenario)
        & (contributions["horizon_weeks"] == horizon)
        & (contributions["source"] != "posterior_total")
    ].copy()
    if data.empty:
        return

    asset_order = data["asset"].drop_duplicates().tolist()
    pivot = data.pivot_table(index="asset", columns="source", values="contribution", aggfunc="sum").loc[asset_order]
    sources = pivot.columns.tolist()
    colors = ["#6b7280", "#2563eb", "#f59e0b", "#059669", "#dc2626", "#7c3aed"]
    x = np.arange(pivot.shape[0])
    pos_base = np.zeros(pivot.shape[0])
    neg_base = np.zeros(pivot.shape[0])

    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, source in enumerate(sources):
        values = pivot[source].to_numpy(dtype=float)
        bottom = np.where(values >= 0.0, pos_base, neg_base)
        ax.bar(x, values, bottom=bottom, label=source, color=colors[idx % len(colors)])
        pos_base = np.where(values >= 0.0, pos_base + values, pos_base)
        neg_base = np.where(values < 0.0, neg_base + values, neg_base)

    ax.axhline(0.0, color="#111827", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist())
    ax.set_ylabel(f"{horizon}W total return contribution")
    ax.set_title(f"Source contributions for {scenario}, {horizon}W horizon")
    percent_axis(ax)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=8)
    add_bottom_explanation(
        fig,
        "What to read: each stacked bar decomposes the posterior mean into the prior plus view-level increments. "
        "This makes the contribution of each forecast visible, including spillovers to correlated assets.",
    )
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 0.96))
    fig.savefig(out_dir / "fig_4_source_contributions.png", dpi=180)
    plt.close(fig)


def write_summary(
    out_dir: Path,
    sample: pd.DataFrame,
    model: SegmentModel,
    view_fit: pd.DataFrame,
    means: pd.DataFrame,
    segments: pd.DataFrame,
) -> None:
    lines = [
        "# Multi-Horizon Forecast Merging Summary",
        "",
        "This run implements the segment-based forecast merging setup from Shah (2019).",
        "It stops at prior/posterior forecasts and does not run portfolio optimization.",
        "",
        "## Data",
        "",
        f"- Assets: {', '.join(model.tickers)}",
        f"- Weekly observations: {len(sample)}",
        f"- Sample: {sample.index.min().date().isoformat()} to {sample.index.max().date().isoformat()}",
        f"- Horizons: {', '.join(str(int(h)) + 'W' for h in model.horizons)}",
        f"- Segments: {', '.join(str(int(d)) + 'W' for d in model.deltas)}",
        "",
        "## Main effects",
        "",
    ]

    for scenario in sorted(means["scenario"].unique()):
        scenario_means = means.loc[means["scenario"] == scenario]
        idx = scenario_means["change"].abs().idxmax()
        strongest = scenario_means.loc[idx]
        fit = view_fit.loc[view_fit["scenario"] == scenario]
        max_residual = fit["posterior_minus_forecast"].abs().max() if not fit.empty else 0.0
        lines.append(
            "- "
            f"`{scenario}`: largest asset-horizon mean shift is "
            f"{10000.0 * strongest['change']:.1f} bp for "
            f"{strongest['asset']} at {int(strongest['horizon_weeks'])}W; "
            f"max posterior view residual is {10000.0 * max_residual:.1f} bp."
        )

    conflict = segments.loc[segments["scenario"] == "conflicting_horizon_views"]
    if not conflict.empty:
        spy_agg = conflict.loc[conflict["asset"].isin(["SPY", "AGG"])]
        lines.extend(
            [
                "",
                "## Multi-horizon reading",
                "",
                "In the conflicting-horizon scenario, the 4W and 12W views are reconciled "
                "through different adjustments to the 0-4W and 4-12W state segments. "
                "This is the key difference from a one-period Black-Litterman update.",
                "",
                f"- Max SPY/AGG segment shift: {10000.0 * spy_agg['return_change'].abs().max():.1f} bp.",
            ]
        )

    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `posterior_asset_horizon_means.csv`: prior vs posterior asset-horizon means.",
            "- `view_fit.csv`: prior-implied, target and posterior-implied view returns.",
            "- `source_contributions.csv`: prior and individual view contributions.",
            "- `segment_posterior.csv`: posterior state by non-overlapping horizon segment.",
        ]
    )

    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_self_test() -> None:
    tickers = ["AAA", "BBB"]
    mu = np.array([0.004, 0.001], dtype=float)
    cov = np.array([[0.0025, 0.0004], [0.0004, 0.0012]], dtype=float)
    model = build_segment_model(tickers=tickers, mu=mu, cov=cov, horizons=(4, 12))

    assert model.cumulative_matrix.shape == (4, 4)
    assert model.segment_prior_mean.shape == (4,)
    assert model.segment_prior_cov.shape == (4, 4)

    no_view = merge_forecasts(model, "no_view", [])
    np.testing.assert_allclose(no_view.segment_posterior_mean, model.segment_prior_mean)
    np.testing.assert_allclose(no_view.segment_posterior_cov, model.segment_prior_cov)

    high_noise = merge_forecasts(
        model,
        "high_noise",
        [View("weak", 4, {"AAA": 1.0, "BBB": -1.0}, 0.20, 1_000_000.0)],
    )
    np.testing.assert_allclose(high_noise.segment_posterior_mean, model.segment_prior_mean, atol=1e-10)

    tight_view = View("tight", 4, {"AAA": 1.0, "BBB": -1.0}, 0.030, 1e-8)
    tight = merge_forecasts(model, "tight", [tight_view])
    implied = float((tight.view_matrix @ tight.segment_posterior_mean).item())
    assert abs(implied - tight_view.forecast_return) < 1e-8

    eigvals = np.linalg.eigvalsh(tight.segment_posterior_cov)
    assert np.min(eigvals) > -1e-10
    np.testing.assert_allclose(tight.segment_posterior_cov, tight.segment_posterior_cov.T, atol=1e-12)

    segments = tight.segment_posterior_mean.reshape(len(model.horizons), len(model.tickers))
    horizons = (model.cumulative_matrix @ tight.segment_posterior_mean).reshape(
        len(model.horizons), len(model.tickers)
    )
    np.testing.assert_allclose(horizons[1], segments[0] + segments[1])
    print("Self-test passed.")


def selected_scenarios(raw: str) -> dict[str, list[View]]:
    scenarios = default_scenarios()
    if raw == "all":
        return scenarios
    if raw not in scenarios:
        raise ValueError(f"Unknown scenario {raw!r}. Choose one of: all, {', '.join(scenarios)}")
    return {raw: scenarios[raw]}


def run(args: argparse.Namespace) -> None:
    tickers = parse_tickers(args.tickers)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prices = load_local_stooq_prices(tickers, Path(args.data_dir))
    returns = weekly_returns(prices)
    sample, mu, cov = estimate_prior(returns, args.start, args.end)
    model = build_segment_model(tickers=tickers, mu=mu, cov=cov, horizons=tuple(args.horizons))
    scenarios = selected_scenarios(args.scenario)
    results = [merge_forecasts(model, scenario, views) for scenario, views in scenarios.items()]

    prior_stats = prior_stats_frame(sample)
    view_definitions = view_definitions_frame(scenarios)
    means = pd.concat([horizon_frame(model, result, "means") for result in results], ignore_index=True)
    vols = pd.concat([horizon_frame(model, result, "vols") for result in results], ignore_index=True)
    view_fit = pd.concat([view_fit_frame(model, result) for result in results], ignore_index=True)
    contributions = pd.concat(
        [source_contributions_frame(model, result) for result in results],
        ignore_index=True,
    )
    segments = pd.concat([segment_frame(model, result) for result in results], ignore_index=True)

    prior_stats.to_csv(out_dir / "prior_stats.csv", index=False)
    view_definitions.to_csv(out_dir / "view_definitions.csv", index=False)
    means.to_csv(out_dir / "posterior_asset_horizon_means.csv", index=False)
    vols.to_csv(out_dir / "posterior_asset_horizon_vols.csv", index=False)
    view_fit.to_csv(out_dir / "view_fit.csv", index=False)
    contributions.to_csv(out_dir / "source_contributions.csv", index=False)
    segments.to_csv(out_dir / "segment_posterior.csv", index=False)

    plot_prior_vs_posterior_means(means, out_dir)
    plot_view_fit(view_fit, out_dir)
    plot_segment_adjustments(segments, out_dir)
    plot_source_contributions(contributions, out_dir)
    write_summary(out_dir, sample, model, view_fit, means, segments)

    print(f"Wrote outputs to {out_dir.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Merge multi-horizon return forecasts with a Gaussian prior using Shah's segment setup.",
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory with local Stooq CSV files.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Directory for CSV/PNG outputs.")
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS), help="Comma-separated tickers.")
    parser.add_argument("--start", default="2015-01-01", help="Prior estimation start date.")
    parser.add_argument("--end", default="2025-12-31", help="Prior estimation end date.")
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=list(DEFAULT_HORIZONS),
        help="Forecast horizons in weeks.",
    )
    parser.add_argument(
        "--scenario",
        default="all",
        help="Scenario to run: all, single_short_view, single_long_view, mixed_horizon_views, conflicting_horizon_views.",
    )
    parser.add_argument("--self-test", action="store_true", help="Run internal consistency checks and exit.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.self_test:
        run_self_test()
        return
    run(args)


if __name__ == "__main__":
    main()
