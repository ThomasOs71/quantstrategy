# Multi-Horizon Forecast Merging Summary

This run implements the segment-based forecast merging setup from Shah (2019).
It stops at prior/posterior forecasts and does not run portfolio optimization.

## Data

- Assets: SPY, EFA, EEM, AGG, GLD
- Weekly observations: 574
- Sample: 2015-01-02 to 2025-12-26
- Horizons: 4W, 12W
- Segments: 4W, 8W

## Main effects

- `conflicting_horizon_views`: largest asset-horizon mean shift is -285.3 bp for SPY at 12W; max posterior view residual is 113.7 bp.
- `mixed_horizon_views`: largest asset-horizon mean shift is 60.9 bp for GLD at 12W; max posterior view residual is 45.3 bp.
- `single_long_view`: largest asset-horizon mean shift is 60.8 bp for GLD at 12W; max posterior view residual is 45.5 bp.
- `single_short_view`: largest asset-horizon mean shift is 29.4 bp for SPY at 12W; max posterior view residual is 17.8 bp.

## Multi-horizon reading

In the conflicting-horizon scenario, the 4W and 12W views are reconciled through different adjustments to the 0-4W and 4-12W state segments. This is the key difference from a one-period Black-Litterman update.

- Max SPY/AGG segment shift: 287.9 bp.

## Files

- `posterior_asset_horizon_means.csv`: prior vs posterior asset-horizon means.
- `view_fit.csv`: prior-implied, target and posterior-implied view returns.
- `source_contributions.csv`: prior and individual view contributions.
- `segment_posterior.csv`: posterior state by non-overlapping horizon segment.
