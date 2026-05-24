# REPO Entropy Study - Run Summary

- Data source: `local`
- Universe: SPY, EFA, EEM, AGG, LQD, HYG, GLD, VNQ, DBC
- In-sample: 2015-01-01 to 2021-12-31
- Out-of-sample: 2022-01-01 to 2024-12-31

## Minimum-risk portfolios

| Method | OOS ann. return | OOS ann. vol | Sharpe rf=3% | Max drawdown | H(weights) | Eff. N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MVPO | -2.64% | 6.22% | -0.91 | -13.49% | 1.070 | 2.92 |
| REPO | -3.26% | 7.61% | -0.82 | -17.75% | 1.205 | 3.34 |
| Downside REPO | -3.26% | 7.61% | -0.82 | -17.75% | 1.205 | 3.34 |

## Bin governance check

- Rank flips from m=10 to m=30: 3 of 35 comparable asset pairs.
- L1 distance m=10 to m=30: 0.464
- L2 distance m=10 to m=30: 0.230

## Rolling monthly rebalance

| Strategy | Total return | Ann. return | Ann. vol | Sharpe rf=3% | Max DD | Avg turnover | Avg eff. N |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Rolling MVPO | -5.39% | -1.83% | 6.30% | -0.77 | -13.56% | 1.34% | 2.88 |
| Rolling REPO | -9.30% | -3.20% | 7.30% | -0.85 | -18.29% | 11.81% | 4.00 |
| Static MVPO | -7.71% | -2.64% | 6.22% | -0.91 | -13.49% | - | - |
| Static REPO | -9.46% | -3.26% | 7.61% | -0.82 | -17.75% | - | - |

Interpretation guardrail: REPO is non-parametric, but the histogram bins are a modeling choice. Treat bin sensitivity as a governance result, not a nuisance detail.
