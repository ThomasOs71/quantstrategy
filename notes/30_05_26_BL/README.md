# Multi-Horizon Forecast Merging

Code companion for Anish R. Shah, "Easy way to merge return forecasts across
securities and horizons" (`ssrn-3459184.pdf`).

The goal is not portfolio optimization. The script shows how a Gaussian prior
for one-period returns can be combined with noisy views or forecasts that refer
to different securities, portfolios, and horizons.

## Local data

The default run uses local Stooq CSV files already committed in:

```text
articles/cov_part2/data_raw
```

Default universe:

| Ticker | Role |
| --- | --- |
| SPY | US equity |
| EFA | Developed ex-US equity |
| EEM | Emerging markets equity |
| AGG | US aggregate bonds |
| GLD | Gold |

Prices are resampled to Friday weekly closes. The default prior estimation
window is `2015-01-01` to `2025-12-31`.

## Method

The implementation follows the paper's segment representation. With horizons
`4W` and `12W`, the state is not a single 12-week return. It is split into
non-overlapping return segments:

```text
x = [r_0_4, r_4_12]
```

The cumulative horizon returns are obtained through a lower block-triangular
matrix `A`:

```text
[r_0_4, r_0_12] = A x
```

The prior assumes iid weekly returns:

```text
nu    = [4 * mu, 8 * mu]
Omega = diag(4 * C, 8 * C)
```

Views are noisy linear observations:

```text
y = Z x + epsilon
epsilon ~ N(0, H)
```

The posterior is:

```text
a     = nu + Omega Z.T (Z Omega Z.T + H)^-1 (y - Z nu)
Sigma = Omega - Omega Z.T (Z Omega Z.T + H)^-1 Z Omega
```

The script exports both segment posteriors and cumulative asset-horizon
posteriors.

## Examples

The built-in scenarios are intentionally simple and didactic:

| Scenario | Views |
| --- | --- |
| `single_short_view` | `SPY - AGG = +1.5%` over 4 weeks |
| `single_long_view` | `GLD = +4.0%` over 12 weeks |
| `mixed_horizon_views` | 4-week `SPY - AGG` plus 12-week `GLD` |
| `conflicting_horizon_views` | positive 4-week but negative 12-week `SPY - AGG` |

Forecast values are examples, not calibrated signals from the paper.

## Run

From the repository root:

```powershell
python notes\30_05_26_BL\multi_horizon_forecast_merging.py
```

Run internal consistency checks:

```powershell
python notes\30_05_26_BL\multi_horizon_forecast_merging.py --self-test
```

Run one scenario only:

```powershell
python notes\30_05_26_BL\multi_horizon_forecast_merging.py --scenario conflicting_horizon_views
```

Optional arguments:

```powershell
python notes\30_05_26_BL\multi_horizon_forecast_merging.py `
  --tickers SPY,EFA,EEM,AGG,GLD `
  --start 2015-01-01 `
  --end 2025-12-31 `
  --horizons 4 12
```

## Outputs

Outputs are written to:

```text
notes/30_05_26_BL/outputs
```

Main tables:

| File | Content |
| --- | --- |
| `prior_stats.csv` | Weekly prior mean/vol by asset |
| `view_definitions.csv` | Scenario views, horizons, exposures, forecast noise |
| `posterior_asset_horizon_means.csv` | Prior vs posterior expected total returns |
| `posterior_asset_horizon_vols.csv` | Prior vs posterior uncertainty |
| `view_fit.csv` | Prior-implied, target, and posterior-implied view returns |
| `source_contributions.csv` | Prior and individual view contributions |
| `segment_posterior.csv` | Prior vs posterior for `0-4W` and `4-12W` segments |
| `summary.md` | Short interpretation of the run |

Figures:

| File | Content |
| --- | --- |
| `fig_1_prior_vs_posterior_means.png` | Prior vs posterior asset-horizon means |
| `fig_2_view_fit.png` | Forecast target vs prior/posterior implied view |
| `fig_3_segment_adjustments.png` | Posterior shifts by non-overlapping segment |
| `fig_4_source_contributions.png` | Prior and view contributions to posterior means |

## Notes

- Returns are treated additively over horizons, matching the paper's
  simplification.
- `H` is diagonal in this example, so view errors are assumed independent.
- No network access is required.
- This is research code for education and discussion, not investment advice.
