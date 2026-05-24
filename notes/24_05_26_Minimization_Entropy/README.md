# Return-Entropy Portfolio Optimization

Code companion for a short Substack Note on Mercurio, Wu, and Xie (2020),
"An Entropy-Based Approach to Portfolio Optimization".

This is intentionally a critical implementation, not a pure paper replication.
The study asks whether entropy of the realized portfolio return distribution can
produce meaningfully different multi-asset allocations than mean-variance
optimization.

## Local data universe

The default run uses the CSV files already committed under
`articles/cov_part2/data_raw`.

| Ticker | Role |
| --- | --- |
| SPY | US equity / S&P 500 |
| EFA | Developed ex-US equity |
| EEM | Emerging markets equity |
| AGG | US aggregate bonds |
| LQD | US investment grade credit |
| HYG | US high yield credit |
| GLD | Gold |
| VNQ | US REITs |
| DBC | Broad commodities |

These are local `Close` price series, so the default is reproducible for
subscribers. The script also supports `--data-source yfinance` for a live
Adjusted-Close refresh, but that makes the results depend on network access and
future vendor data revisions.

## Run

From the repository root:

```powershell
python notes\24_05_26_Minimization_Entropy\repo_entropy_study.py
```

Faster smoke run:

```powershell
python notes\24_05_26_Minimization_Entropy\repo_entropy_study.py --n-random 1000 --frontier-points 11
```

Outputs are written to:

```text
notes/24_05_26_Minimization_Entropy/outputs/
```

The default run also performs a monthly walk-forward test:

- At each OOS month, estimate MVPO and REPO on the trailing 156 weekly returns.
- Hold the resulting portfolio for the next month.
- Compare rolling MVPO/REPO against the static in-sample minimum-risk MVPO/REPO
  portfolios held buy-and-hold from 2022 to 2024.
- Export rolling returns, rolling weights, stability metrics, and wealth plots.

## Design choices

- In-sample window: 2015-01-01 to 2021-12-31.
- Out-of-sample window: 2022-01-01 to 2024-12-31.
- Weekly returns are computed from Friday weekly closes.
- The default run uses a 60% per-asset cap to keep the experiment in practical
  multi-asset/SAA territory. Use `--max-weight 1.0` for the paper-like
  unconstrained long-only case.
- REPO entropy is computed on the weighted portfolio return, `R @ w`, not on
  joint entropy across assets.
- Histogram bin edges are fixed before optimization from the in-sample support.
  This is important: changing the bins for every candidate portfolio would make
  the risk measure less comparable.
- Entropy optimization is non-smooth because histogram counts change in jumps.
  The script therefore uses a deterministic random candidate search for the
  entropy frontiers and exact SLSQP polishing only for MVPO.
- Downside REPO is implemented as an experimental extension for a follow-up
  Note. It is not presented as part of the original paper.

## Main files

- `repo_entropy_study.py`: full implementation.
- `outputs/summary.md`: short run summary after execution.
- `outputs/summary_portfolios.csv`: selected min-risk, target-return, and
  high-return portfolios.
- `outputs/bin_sensitivity.csv`: bin-count sensitivity.
- `outputs/bin_weight_distances.csv`: pairwise allocation drift across bin
  choices.
- `outputs/rolling_summary.csv`: rolling monthly rebalance performance and
  turnover/stability diagnostics.
- `outputs/rolling_weights.csv`: rebalance-by-rebalance weights and
  diversification metrics.

This is research code for education and discussion. It is not investment advice.
