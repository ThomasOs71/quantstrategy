# Covariance Series — Part 2: The Frequency Trap

This repo reproduces the mini case study for **Part 2** of the Covariance Series:
**"The Frequency Trap: Why Daily Covariances Break Monthly Portfolios"**.

## Quickstart

```bash
pip install -r requirements.txt
python run_part2_frequency_trap.py --config config.yaml
```

Outputs:
- `results/backtest.csv`
- `results/summary.json`
- `figures/fig1_rv_over_pv.png`
- `figures/fig2_avg_weights.png`
- `figures/fig3_l1_AB.png`

## Notes
- Data source: Stooq daily CSV close prices (downloaded automatically).
- Portfolio: long-only minimum-variance with per-asset cap (default 60%).
