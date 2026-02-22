from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def make_all_figures(bt: pd.DataFrame, avgw: pd.DataFrame, l1_ab: pd.Series, l1_ac: pd.Series, l1_bc: pd.Series, out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True, parents=True)

    plt.figure(figsize=(10,5))
    plt.plot(bt.index, bt["RV_over_PV_A"], label="A: monthly→monthly")
    plt.plot(bt.index, bt["RV_over_PV_B"], label="B: daily×21→monthly")
    plt.plot(bt.index, bt["RV_over_PV_C"], label="C: mixed-scale")
    plt.yscale("log")
    plt.title("Realized / Predicted Monthly Variance (log scale)")
    plt.xlabel("Rebalance date")
    plt.ylabel("RV / PV")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig1_rv_over_pv.png", dpi=160)
    plt.close()

    tickers = list(avgw.index)
    x = np.arange(len(tickers))
    width = 0.27
    
    plt.figure(figsize=(10,5))
    plt.bar(x - width, avgw["A_monthly"].values, width, label="A: monthly")
    plt.bar(x,         avgw["B_dailyx21"].values, width, label="B: daily×21")
    plt.bar(x + width, avgw["C_mixed"].values, width, label="C: mixed")
    plt.xticks(x, tickers)
    plt.title("Average Portfolio Weights (long-only min-variance)")
    plt.xlabel("Asset")
    plt.ylabel("Weight")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig2_avg_weights.png", dpi=160)
    plt.close()
    
    plt.figure(figsize=(10,5))
    plt.plot(l1_ab.index, l1_ab.values, label="L1(A,B)")
    plt.plot(l1_ac.index, l1_ac.values, label="L1(A,C)")
    plt.plot(l1_bc.index, l1_bc.values, label="L1(B,C)")
    plt.title("Portfolio Disagreement (L1 distance)")
    plt.xlabel("Rebalance date")
    plt.ylabel("L1 distance (0 to 2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig3_l1_distances.png", dpi=160)
    plt.close()


def make_economic_figures(bt: pd.DataFrame, perf: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(exist_ok=True, parents=True)

    # cumulative net wealth for strategies + baselines
    series = {
        "A net": bt["ret_month_A_net"].dropna(),
        "B net": bt["ret_month_B_net"].dropna(),
        "C net": bt["ret_month_C_net"].dropna(),
    }

    plt.figure(figsize=(10, 5))
    for label, r in series.items():
        if len(r) == 0:
            continue
        wealth = (1.0 + r).cumprod()
        plt.plot(wealth.index, wealth.values, label=label)
    plt.title("Cumulative Net Wealth (Monthly Rebalanced)")
    plt.xlabel("Date")
    plt.ylabel("Growth of $1")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "fig4_cumulative_net_wealth.png", dpi=160)
    plt.close()

    # separate forecast quality from economic value for A/B/C
    keys = ["A", "B", "C"]
    male = [bt[f"abs_log_err_{k}"].dropna().mean() for k in keys]
    sharpe = [perf.loc[f"{k}_net", "Sharpe_0rf"] for k in keys]

    x = np.arange(len(keys))
    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    bars = ax1.bar(x - 0.18, male, width=0.36, label="Forecast MALE")
    ax1.set_ylabel("MALE (lower is better)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(["A monthly", "B dailyx21", "C mixed"])
    ax1.set_xlabel("Setup")

    ax2 = ax1.twinx()
    ax2.plot(x + 0.18, sharpe, marker="o", linewidth=2, label="Net Sharpe")
    ax2.set_ylabel("Net Sharpe (higher is better)")

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="best")
    plt.title("Forecast Quality vs Economic Value")
    plt.tight_layout()
    plt.savefig(out_dir / "fig5_forecast_vs_economic.png", dpi=160)
    plt.close()


def plot_fixed_pf_rv_over_pv(eval_df: pd.DataFrame, out_dir: Path | str = "figures") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    plt.figure(figsize=(10, 5))
    plt.plot(eval_df.index, eval_df["RV_over_PV_A"], label="A: monthly->monthly")
    plt.plot(eval_df.index, eval_df["RV_over_PV_B"], label="B: dailyx21->monthly")
    plt.plot(eval_df.index, eval_df["RV_over_PV_C"], label="C: mixed-scale")
    plt.yscale("log")
    plt.title("Realized / Predicted Variance on Fixed 60/40 Portfolio (log scale)")
    plt.xlabel("Rebalance date")
    plt.ylabel("RV / PV")
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / "fig6_rv_over_pv_fixed_6040.png"
    plt.savefig(out_path, dpi=160)
    plt.close()
    return out_path


def _month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    mes = idx.to_series().groupby([idx.year, idx.month]).max().values
    return pd.DatetimeIndex(pd.to_datetime(mes)).sort_values()


def _quarter_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    qes = idx.to_series().groupby([idx.year, idx.quarter]).max().values
    return pd.DatetimeIndex(pd.to_datetime(qes)).sort_values()


def _weekly_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    # last available trading day per ISO week
    wes = idx.to_series().groupby([idx.isocalendar().year, idx.isocalendar().week]).max().values
    return pd.DatetimeIndex(pd.to_datetime(wes)).sort_values()


def _returns_at_ends(px: pd.DataFrame, ends: pd.DatetimeIndex) -> pd.DataFrame:
    p = px.loc[ends].dropna()
    return p.pct_change().dropna()


def plot_equity_bond_correlation(
    px: pd.DataFrame,
    equity: str = "SPY",
    bond: str = "IEF",
    out_dir: Path | str = "figures",
    window_years_daily_weekly: int = 5,
    window_years_quarterly: int = 10,
    *,
    drop_first_years: int = 0,
) -> Path:
    """
    Creates one chart:
      rolling corr(SPY,IEF) computed on non-overlapping daily / weekly / quarterly returns.

    Rolling window is calendar-based (days), so series are comparable.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if equity not in px.columns or bond not in px.columns:
        raise ValueError(f"Missing columns. Need {equity} and {bond} in px.columns.")

    pair_px = px[[equity, bond]].dropna()

    # Build non-overlapping return series at different sampling frequencies
    daily_ends = pair_px.index  # daily
    weekly_ends = _weekly_ends(pair_px.index)
    quarterly_ends = _quarter_ends(pair_px.index)

    r_d = _returns_at_ends(pair_px, daily_ends)
    r_w = _returns_at_ends(pair_px, weekly_ends)
    r_q = _returns_at_ends(pair_px, quarterly_ends)

    # Rolling correlations (calendar windows)
    wd = f"{window_years_daily_weekly * 365}D"
    wq = f"{window_years_quarterly * 365}D"

    corr_d = r_d[equity].rolling(wd).corr(r_d[bond])
    corr_w = r_w[equity].rolling(wd).corr(r_w[bond])
    corr_q = r_q[equity].rolling(wq).corr(r_q[bond])

    # optionally drop the first few years of correlation data
    if drop_first_years > 0:
        cutoff = corr_d.index.min() + pd.DateOffset(years=drop_first_years)
        corr_d = corr_d.loc[cutoff:]
        corr_w = corr_w.loc[cutoff:]
        corr_q = corr_q.loc[cutoff:]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(corr_d.index, corr_d.values, label=f"Daily ({window_years_daily_weekly}y rolling)")
    plt.plot(corr_w.index, corr_w.values, label=f"Weekly ({window_years_daily_weekly}y rolling)")
    plt.plot(corr_q.index, corr_q.values, label=f"Quarterly ({window_years_quarterly}y rolling)")
    plt.axhline(0.0, linewidth=1)
    plt.title(f"Equity–Gov Bond Correlation depends on sampling frequency ({equity} vs {bond})")
    plt.xlabel("Date")
    plt.ylabel("Correlation")
    plt.legend()
    plt.tight_layout()

    out_path = out_dir / f"fig_corr_frequency_{equity.lower()}_{bond.lower()}.png"
    plt.savefig(out_path, dpi=160)
    plt.close()

    return out_path


def plot_volatility_by_frequency(
    px: pd.DataFrame,
    tickers: list[str],
    out_dir: Path | str = "figures",
    *,
    annualize: bool = True,
) -> Path:
    """Create a bar chart comparing volatilities of *tickers* using
    daily, weekly and monthly returns.

    The bars are grouped by frequency with one bar per ticker.  The
    resulting figure is saved under ``out_dir`` and the path returned.
    """

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = px[tickers].dropna()
    if data.empty:
        raise ValueError("No data available for the requested tickers")

    # construct returns at each sampling frequency
    r_d = data.pct_change().dropna()

    weekly_ends = _weekly_ends(data.index)
    r_w = _returns_at_ends(data, weekly_ends)

    monthly_ends = _month_ends(data.index)
    r_m = _returns_at_ends(data, monthly_ends)

    # compute volatilities
    vols = {}
    vols["Daily"] = r_d.std() * (
        np.sqrt(252) if annualize else 1.0
    )
    vols["Weekly"] = r_w.std() * (
        np.sqrt(52) if annualize else 1.0
    )
    vols["Monthly"] = r_m.std() * (
        np.sqrt(12) if annualize else 1.0
    )

    freqs = list(vols.keys())
    n_freq = len(freqs)
    n_tickers = len(tickers)

    # cluster bars by ticker, with a sub-bar per frequency
    x = np.arange(n_tickers)
    width = 0.8 / n_freq

    # friendly labels for the x axis
    ticker_labels = {"SPY": "S&P", "EFA": "Europe+", "EEM": "Emerging Markets"}
    xtick_names = [ticker_labels.get(t, t) for t in tickers]

    plt.figure(figsize=(10, 5))
    # choose distinct colors for each frequency
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    freq_colors = {freqs[i]: colors[i % len(colors)] for i in range(n_freq)}

    for j, f in enumerate(freqs):
        c = freq_colors[f]
        for i, t in enumerate(tickers):
            val = vols[f][t]
            pos = x[i] + (j - (n_freq - 1) / 2) * width
            plt.bar(pos, val, width, color=c)

    plt.xticks(x, xtick_names)
    plt.ylabel("Volatility{}".format(" (annualized)" if annualize else ""))
    plt.title("Return volatility by sampling frequency")

    # "frequency" legend: each entry explains that all three bars of a given
    # colour correspond to that sampling frequency. we no longer use a legend
    # for ticker names (they're on the x‑axis).
    from matplotlib.patches import Patch
    legend_handles = []
    for f in freqs:
        lbl = f + " data (three bars)"
        legend_handles.append(Patch(facecolor=freq_colors[f], label=lbl))
    plt.legend(handles=legend_handles, title="Sampling frequency")

    plt.tight_layout()

    out_path = out_dir / f"fig_volatility_freq_{'_'.join([t.lower() for t in tickers])}.png"
    plt.savefig(out_path, dpi=160)
    plt.close()

    return out_path
