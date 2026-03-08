# Enhanced Allocation Extension

## Broad Market

- USA: underweight (Forward PE 19.2x, spread 0.75%, ETF regime low)
- Europe: underweight (Forward PE 14.9x, spread 1.90%, ETF regime low)

## Preferred Sector Tilts

Top USA ideas:
- Financials: relative_overweight via XLF (direct)
- Consumer Staples: relative_overweight via XLP (direct)
- Materials: relative_overweight via XLB (direct)

Top Europe ideas:
- Financials: relative_overweight via EUFN (direct)
- Utilities: relative_overweight via VGK (broad_region_proxy)
- Consumer Staples: relative_overweight via VGK (broad_region_proxy)

## Proxy Backtest

- Raw signal annual return: 8.03%
- Enhanced signal annual return: 9.05%
- Improvement vs raw: 1.02%

Caveat: the backtest is a US sector ETF proxy. Damodaran does not publish a clean public archive of point-in-time industry snapshots suitable for a true historical replay, so the historical test uses live market proxies (dividend yield, dividend growth, momentum, volatility) rather than archived fundamental snapshots.
