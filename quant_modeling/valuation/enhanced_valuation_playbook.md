# Enhanced Valuation Playbook

Use this as the practical overlay for raw valuation multiples.

## Core Rule

Raw multiples become materially better when you control for:

- growth
- profitability
- payout or cash conversion
- risk
- reinvestment intensity

This reduces classic value traps where a low multiple is justified by weak fundamentals.

## Variable Table

| Multiple | Cheap If | Add These Variables | Main Companion Variable | Practical Screen |
| --- | --- | --- | --- | --- |
| `Forward PE` | Lower | `Expected growth`, `Dividend Payout`, `Beta` | Growth | `-z(Forward PE) + 0.8*z(Expected growth) + 0.4*z(Dividend Payout) - 0.6*z(Beta)` |
| `PBV` | Lower | `ROE`, `ROIC`, `Cost of Equity` | `ROE` | `-z(PBV) + 1.0*z(ROE) + 0.4*z(ROIC) - 0.4*z(Cost of Equity)` |
| `Price/Sales` | Lower | `Net Margin`, `Fundamental Growth`, `Beta` | Margin | `-z(Price/Sales) + 1.0*z(Net Margin) + 0.7*z(Fundamental Growth) - 0.5*z(Beta)` |
| `EV/EBITDA` | Lower | `ROIC`, `Pre-tax Operating Margin`, `Net Cap Ex/Sales`, `Cost of Capital` | Capital efficiency | `-z(EV/EBITDA) + 0.8*z(ROIC) + 0.5*z(Pre-tax Operating Margin) - 0.6*z(Net Cap Ex/Sales) - 0.4*z(Cost of Capital)` |

## Why This Usually Improves Results

- `Forward PE` alone confuses weak growth with cheapness.
- `PBV` alone misses whether book value can compound through `ROE`.
- `Price/Sales` alone ignores whether revenue turns into profits.
- `EV/EBITDA` alone misses capex-heavy businesses and weak returns on capital.

In practice, the extension should improve cross-sectional ranking quality because it removes many false positives. It is most useful for stock, sector, and industry selection. It helps less for broad market timing.

## How To Use It

1. Compute the raw multiple z-score with the correct sign (`lower = cheaper`).
2. Add the companion variables as quality/risk overlays.
3. Rank on the adjusted score.
4. Confirm with a regression residual model:
   `actual multiple` versus `fair multiple implied by fundamentals`.
5. Prefer names or industries that are cheap on both the heuristic score and the regression residual.

## Online Data Used In The Workspace

The live model in [damodaran_enhanced_relative_model.py](/c:/Users/thoma/OneDrive/Dokumente/quantstrategy/quant_modeling/valuation/damodaran_enhanced_relative_model.py) pulls directly from Damodaran's public dataset files for US and Europe.
