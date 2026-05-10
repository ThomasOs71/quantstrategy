# QuantStrategy

> Companion code for [**QuantStrategy**](https://quantstrategy.substack.com) — a publication on robust portfolio construction, optimization, and risk modeling.

This repository contains the Python code, notebooks, and case studies that accompany each article on the QuantStrategy Substack. Every method is implemented in a way you can adapt to your own data and portfolios — not just toy examples.

📌 **New here?** Start with **[Robust Portfolio Construction Architecture →](https://quantstrategy.substack.com)** — the framework that everything else in this repo builds on.

---

## What you'll find here

Code organized by article. Each folder contains the notebook, supporting modules, and a brief README pointing back to the original write-up.

| Topic | Article |
|---|---|
| Covariance estimation (Ledoit-Wolf, Graphical Lasso, spectral denoising) | [Why Most Covariance Estimations Fail](https://quantstrategy.substack.com/p/why-most-covariance-estimations-fail) |
| 13 practical rules for portfolio optimization | [Wie man Portfolio-Optimierung richtig anwendet](https://quantstrategy.substack.com/p/wie-man-portfolio-optimierung-richtig) |
| _More coming — one article per month_ | |

The full archive lives on [the Substack](https://quantstrategy.substack.com).

---

## Methods covered across articles

- **Robust optimization** — CVaR, LPM, Distributionally Robust Optimization (Wasserstein)
- **Covariance estimation** — Shrinkage, spectral denoising, Graphical Lasso
- **View integration** — Entropy Pooling, Bayesian updating
- **Risk decomposition** — Risk Parity, Minimum Torsion Bets
- **Post-processing** — Resampling, stress testing, lexicographic optimization

---

## Setup

```bash
git clone https://github.com/ThomasOs71/quantstrategy.git
cd quantstrategy
pip install -r requirements.txt
```

Python 3.10+ recommended. Each notebook can be run independently.

---

## Stay in the loop

If you find this code useful, the best way to support the work is to [subscribe to the Substack](https://quantstrategy.substack.com) — new articles drop monthly, each with code added here.

---

## Authors

- **Dr. Thomas Osowski** — Head of Asset Allocation, quantitative portfolio construction
- **Dr. Felix Haase** — Co-author

---

## Disclaimer

All code is for **educational purposes only**. Nothing in this repository constitutes investment advice or a recommendation to buy or sell any financial instrument. The views expressed are our own.

---

## License

MIT — see [LICENSE](LICENSE) for details.
