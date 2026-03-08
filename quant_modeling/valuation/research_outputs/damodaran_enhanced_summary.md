# Enhanced Relative Valuation Summary

Data pulled online on 2026-02-28 from Damodaran's public datasets.

## Variable Overlay Rules

| Multiple | Add Variables | Primary Driver | Screen Formula |
| --- | --- | --- | --- |
| Forward PE | Expected growth, Dividend Payout, Beta | Growth | `-z(Forward PE) + 0.8*z(Expected growth) + 0.4*z(Dividend Payout) - 0.6*z(Beta)` |
| PBV | ROE, ROIC, Cost of Equity | ROE | `-z(PBV) + 1.0*z(ROE) + 0.4*z(ROIC) - 0.4*z(Cost of Equity)` |
| Price/Sales | Net Margin, Fundamental Growth, Beta | Margin | `-z(Price/Sales) + 1.0*z(Net Margin) + 0.7*z(Fundamental Growth) - 0.5*z(Beta)` |
| EV/EBITDA | ROIC, Pre-tax Operating Margin, Net Cap Ex/Sales, Cost of Capital | Capital efficiency | `-z(EV/EBITDA) + 0.8*z(ROIC) + 0.5*z(Pre-tax Operating Margin) - 0.6*z(Net Cap Ex/Sales) - 0.4*z(Cost of Capital)` |

## Region Snapshots

### US

- Coverage: 89 industries with at least 5 firms.
- Mean regression R^2 across the four models: 0.29
- Average quality overlay delta: 0.05
- Actions: {"market_weight": 43, "overweight": 23, "underweight": 23}

Top cheap industries:
- Tobacco: score=1.87, regime=cheap, action=overweight
- Precious Metals: score=1.82, regime=cheap, action=overweight
- Banks (Regional): score=1.74, regime=cheap, action=overweight
- Bank (Money Center): score=1.50, regime=cheap, action=overweight
- Telecom. Services: score=1.38, regime=cheap, action=overweight

Top expensive industries:
- Software (Internet): score=-3.20, regime=expensive, action=underweight
- Electronics (Consumer & Office): score=-2.97, regime=expensive, action=underweight
- Auto & Truck: score=-2.52, regime=expensive, action=underweight
- Drugs (Biotechnology): score=-2.05, regime=expensive, action=underweight
- Restaurant/Dining: score=-1.93, regime=expensive, action=underweight

### Europe

- Coverage: 93 industries with at least 5 firms.
- Mean regression R^2 across the four models: 0.36
- Average quality overlay delta: 0.02
- Actions: {"market_weight": 45, "overweight": 24, "underweight": 24}

Top cheap industries:
- Information Services: score=1.43, regime=cheap, action=overweight
- Banks (Regional): score=1.38, regime=cheap, action=overweight
- Precious Metals: score=1.28, regime=cheap, action=overweight
- Insurance (General): score=1.20, regime=cheap, action=overweight
- Oil/Gas (Integrated): score=1.18, regime=cheap, action=overweight

Top expensive industries:
- Drugs (Biotechnology): score=-3.06, regime=expensive, action=underweight
- Semiconductor Equip: score=-2.50, regime=expensive, action=underweight
- Entertainment: score=-2.14, regime=expensive, action=underweight
- Electrical Equipment: score=-2.00, regime=expensive, action=underweight
- Semiconductor: score=-1.62, regime=expensive, action=underweight

## Online Sources

- https://pages.stern.nyu.edu/~adamodar/pc/datasets/pedata.xls
- https://pages.stern.nyu.edu/~adamodar/pc/datasets/pbvdata.xls
- https://pages.stern.nyu.edu/~adamodar/pc/datasets/psdata.xls
- https://pages.stern.nyu.edu/~adamodar/pc/datasets/vebitda.xls
- https://pages.stern.nyu.edu/~adamodar/pc/datasets/fundgr.xls
- https://pages.stern.nyu.edu/~adamodar/pc/datasets/betas.xls
- https://pages.stern.nyu.edu/~adamodar/pc/datasets/capex.xls
- https://pages.stern.nyu.edu/~adamodar/pc/datasets/divfund.xls
- https://pages.stern.nyu.edu/~adamodar/pc/datasets/wacc.xls
- https://pages.stern.nyu.edu/~adamodar/pc/datasets/peEurope.xls
- https://pages.stern.nyu.edu/~adamodar/pc/datasets/pbvEurope.xls
- https://pages.stern.nyu.edu/~adamodar/pc/datasets/psEurope.xls
- https://pages.stern.nyu.edu/~adamodar/pc/datasets/vebitdaEurope.xls
- https://pages.stern.nyu.edu/~adamodar/pc/datasets/fundgrEurope.xls
- https://pages.stern.nyu.edu/~adamodar/pc/datasets/betaEurope.xls
- https://pages.stern.nyu.edu/~adamodar/pc/datasets/capexEurope.xls
- https://pages.stern.nyu.edu/~adamodar/pc/datasets/divfundEurope.xls
- https://pages.stern.nyu.edu/~adamodar/pc/datasets/waccEurope.xls

These overlays should usually improve relative valuation screens by reducing value traps, but they remain cross-sectional tools rather than market-timing signals.
