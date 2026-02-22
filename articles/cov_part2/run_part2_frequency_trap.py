import argparse
from pathlib import Path
import yaml
import sys
import numpy as np

# In .py-Dateien existiert __file__, im Notebook nicht
try:
    ROOT = Path(__file__).resolve().parent
except NameError:
    ROOT = Path.cwd()  # Notebook-Working-Directory

# Optional: falls du nicht im Repo-Root bist, laufe nach oben bis "src" gefunden wird
for p in [ROOT, *ROOT.parents]:
    if (p / "src").exists():
        ROOT = p
        break

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

b = {2:2,3:{3:3}}

from src.data import download_stooq_csv_if_missing, load_close_panel
from src.backtest import run_backtest, pf_volatility_evaluation_by_estimator
from src.plots import (
    make_all_figures,
    make_economic_figures,
    plot_fixed_pf_rv_over_pv,
    plot_equity_bond_correlation,
    plot_volatility_by_frequency,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yaml")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    cfg = yaml.safe_load((root / args.config).read_text())

    tickers = cfg["universe"]["tickers"]
    suffix = cfg["universe"].get("stooq_suffix", ".us")

    paths = cfg["paths"]
    data_dir = root / paths["data_raw"]
    fig_dir = root / paths["figures"]
    res_dir = root / paths["results"]

    data_dir.mkdir(exist_ok=True, parents=True)
    fig_dir.mkdir(exist_ok=True, parents=True)
    res_dir.mkdir(exist_ok=True, parents=True)
    # Download Data from Stooq
    for t in tickers:
        download_stooq_csv_if_missing(t, suffix=suffix, out_dir=data_dir)
    
    px = load_close_panel(tickers, data_dir=data_dir, suffix=suffix)

    eval_fixed_6040 = pf_volatility_evaluation_by_estimator(
        px=px,
        equity="SPY",
        bond="AGG",
        w_equity=0.60,
        w_bond=0.40,
        win_months=int(cfg["backtest"]["win_months"]),
        win_days=int(cfg["backtest"]["win_days"]),
        scale_days_to_month=int(cfg["backtest"]["scale_days_to_month"]),
        ewma_lambda=float(cfg["backtest"]["ewma_lambda"]),
    )

    bt, summary_json, avgw, l1_ab, l1_ac, l1_bc, perf = run_backtest(
        px=px,
        cap=float(cfg["backtest"]["cap"]),
        win_months=int(cfg["backtest"]["win_months"]),
        win_days=int(cfg["backtest"]["win_days"]),
        scale_days_to_month=int(cfg["backtest"]["scale_days_to_month"]),
        ewma_lambda=float(cfg["backtest"]["ewma_lambda"]),
        transaction_cost_bps=float(cfg["backtest"].get("transaction_cost_bps", 0.0)),
    )

    bt.to_csv(res_dir / "backtest.csv")
    (res_dir / "summary.json").write_text(summary_json, encoding="utf-8")
    avgw.to_csv(res_dir / "avg_weights.csv")
    l1_ab.to_csv(res_dir / "l1_A_vs_B.csv")
    perf.to_csv(res_dir / "performance_metrics.csv")
    eval_fixed_6040.to_csv(res_dir / "pf_vol_eval_by_estimator_6040.csv")

    make_all_figures(bt=bt, avgw=avgw, l1_ab=l1_ab, l1_ac = l1_ac, l1_bc = l1_bc, out_dir=fig_dir)
    make_economic_figures(bt=bt, perf=perf, out_dir=fig_dir)
    plot_fixed_pf_rv_over_pv(eval_df=eval_fixed_6040, out_dir=fig_dir)
    plot_equity_bond_correlation(
        px,
        equity = "SPY",
        bond = "AGG",
        out_dir = fig_dir,
        window_years_daily_weekly= 10,
        window_years_quarterly = 10,
        drop_first_years = 5,
    )

    # compare volatilities for a few equities at different sampling frequencies
    plot_volatility_by_frequency(
        px,
        tickers=["SPY", "EFA", "EEM"],
        out_dir=fig_dir,
        annualize=True,
    )

    print("Done. See results/ and figures/.")

if __name__ == "__main__":
    main()
