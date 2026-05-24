# REPO Implementation Brief
## Return-Entropy Portfolio Optimization: Empirical Study & Extensions

---

## Link:
https://www.mdpi.com/1099-4300/22/3/332?type=check_update&version=1


## 1. Objective

This brief specifies the full implementation of a comparative portfolio optimization study. The study covers three components:

1. **REPO vs. MVPO** — Empirical comparison on real asset data
2. **Bin Sensitivity Analysis** — Quantifying the impact of bin choice on REPO
3. **Downside Entropy Extension** — A novel modification addressing the symmetry problem of standard entropy

The output should be a self-contained Python implementation with visualizations and a summary of results.

---

## 2. Background

### 2.1 Mean-Variance Portfolio Optimization (MVPO)

Markowitz (1952) defines the portfolio optimization problem as:

```
minimize  Var(Rₚ) = wᵀ Σ w
maximize  E(Rₚ)  = wᵀ μ
subject to: Σwᵢ = 1, wᵢ ≥ 0
```

Where:
- `w` = vector of portfolio weights (n × 1)
- `Σ` = covariance matrix of returns (n × n)
- `μ` = vector of expected returns (n × 1)

The efficient frontier is traced by solving this for varying risk tolerance parameter `λ`:

```
minimize  Var(Rₚ) - λ · E(Rₚ)
```

### 2.2 Return-Entropy Portfolio Optimization (REPO)

REPO replaces variance with Shannon entropy as the risk measure (Mercurio et al., 2020):

```
minimize  H(Rₚ) = -Σₖ p̂ₖ · log(p̂ₖ)
maximize  E(Rₚ) = wᵀ μ
subject to: Σwᵢ = 1, wᵢ ≥ 0
```

Where `p̂ₖ` is the empirical probability of portfolio return falling in bin `k`.

The combined scalar objective with risk parameter `α`:

```
minimize  H(Rₚ) - α · E(Rₚ)
```

**Key insight:** Entropy is computed directly on the weighted portfolio return `Rₚ = Σ wᵢRᵢ`, not on the joint distribution of individual assets. This reduces the complexity from O(mⁿ) to O(m), regardless of the number of assets `n`.

**Computing portfolio entropy:**

Given portfolio weights `w` and historical return matrix `R` (T × n):
1. Compute portfolio returns: `rₚ = R @ w` (T × 1 vector)
2. Discretize `rₚ` into `m` bins
3. Estimate empirical probabilities: `p̂ₖ = count(rₚ ∈ Aₖ) / T`
4. Compute Shannon entropy: `H = -Σₖ p̂ₖ · log(p̂ₖ)` (use natural log)

### 2.3 Downside Entropy Extension (Novel)

Standard entropy is symmetric — it penalizes upside and downside volatility equally. The proposed extension addresses this:

**Construction:**
1. Define bins as before, but separate negative and positive bins
2. Keep all negative bins individually: `A₁ = (-∞, b₁]`, `A₂ = (b₁, b₂]`, ..., `Aₖ = (bₖ₋₁, 0]`
3. Collapse all positive returns into one single bin: `A₊ = (0, +∞)`
4. Compute entropy over this modified bin structure **without renormalization**

This preserves the absolute loss mass in the risk measure while eliminating upside entropy. In the combined objective, `E(Rₚ)` captures the upside return information.

```
minimize  H_down(Rₚ) - α · E(Rₚ)
```

---

## 3. Data

### 3.1 Asset Universe

Use the following 8 ETFs representing a simplified Strategic Asset Allocation (SAA) universe. Download via `yfinance`.

| Asset Class | Ticker | Description |
|-------------|--------|-------------|
| Global Equities | VT | Vanguard Total World Stock ETF |
| US Equities | SPY | S&P 500 ETF |
| EM Equities | EEM | iShares MSCI Emerging Markets ETF |
| Global Bonds | AGG | iShares Core U.S. Aggregate Bond ETF |
| EM Bonds | EMB | iShares J.P. Morgan USD EM Bond ETF |
| Real Estate | VNQ | Vanguard Real Estate ETF |
| Commodities | GSG | iShares S&P GSCI Commodity ETF |
| Gold | GLD | SPDR Gold Shares |

### 3.2 Time Period

- **In-sample period:** 2015-01-01 to 2021-12-31 (model estimation)
- **Out-of-sample period:** 2022-01-01 to 2024-12-31 (performance evaluation)
- **Frequency:** Weekly returns (adjusted closing prices)
- **Return calculation:** `rᵢₜ = Pᵢₜ / Pᵢ,ₜ₋₁ - 1`

### 3.3 Data Handling

- Drop any weeks with missing data (use `dropna()`)
- Store in-sample and out-of-sample returns as separate DataFrames
- Print summary statistics (mean, std, skewness, kurtosis) per asset

---

## 4. Implementation Specification

### 4.1 Required Libraries

```python
import numpy as np
import pandas as pd
import yfinance as yf
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')
```

### 4.2 Core Functions

#### `compute_portfolio_entropy(weights, returns, m, method='standard')`

Computes portfolio entropy for a given weight vector.

**Parameters:**
- `weights`: np.array (n,) — portfolio weights
- `returns`: pd.DataFrame (T × n) — historical returns
- `m`: int — number of bins
- `method`: str — `'standard'` or `'downside'`

**Logic for `method='standard'`:**
1. Compute `rp = returns.values @ weights`
2. Use `np.histogram(rp, bins=m)` to get counts
3. Convert counts to probabilities: `p = counts / counts.sum()`
4. Remove zero probabilities before entropy calculation
5. Return `H = -np.sum(p * np.log(p))`

**Logic for `method='downside'`:**
1. Compute `rp = returns.values @ weights`
2. Separate negative returns: `rp_neg = rp[rp <= 0]`
3. Compute proportion in positive bin: `p_positive = np.sum(rp > 0) / len(rp)`
4. Discretize negative returns into `m-1` bins using `np.histogram(rp_neg, bins=m-1)`
5. Convert negative bin counts to proportions of **total** returns (not just negative): `p_neg = counts / len(rp)`
6. Combine: `p_all = np.append(p_neg, p_positive)`
7. Remove zero probabilities
8. Return `H_down = -np.sum(p_all * np.log(p_all))`

#### `optimize_portfolio(returns, method, alpha, m=20)`

Solves the portfolio optimization for a given method and risk parameter `alpha`.

**Parameters:**
- `returns`: pd.DataFrame (T × n)
- `method`: str — `'mvpo'`, `'repo'`, or `'downside'`
- `alpha`: float — risk tolerance parameter
- `m`: int — number of bins (only relevant for entropy methods)

**Logic:**
- For `'mvpo'`: minimize `w @ cov @ w - alpha * mu @ w`
- For `'repo'`: minimize `H(Rₚ) - alpha * mu @ w` using `compute_portfolio_entropy(..., method='standard')`
- For `'downside'`: minimize `H_down(Rₚ) - alpha * mu @ w` using `compute_portfolio_entropy(..., method='downside')`

**Constraints and bounds:**
```python
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
bounds = [(0, 1)] * n_assets
w0 = np.ones(n_assets) / n_assets  # equal weight initialization
```

Use `scipy.optimize.minimize` with method `'SLSQP'`.

#### `compute_efficient_frontier(returns, method, n_points=50, m=20)`

Traces the efficient frontier by varying `alpha` from 0 to a reasonable upper bound.

**Logic:**
1. Define `alpha_range = np.linspace(0, 5, n_points)` for entropy methods; `np.linspace(0, 2, n_points)` for MVPO
2. For each alpha: call `optimize_portfolio(...)`, store weights, compute `E(Rₚ)` and risk metric
3. For MVPO: risk metric = `np.sqrt(w @ cov @ w) * np.sqrt(52)` (annualized volatility)
4. For REPO/Downside: risk metric = entropy value
5. Return DataFrame with columns: `['alpha', 'return', 'risk', 'weights']`

#### `backtest_portfolio(weights, returns_oos)`

Computes out-of-sample performance metrics.

**Returns dict with:**
- `total_return`: cumulative return over full OOS period
- `annualized_return`: geometric annualized return
- `annualized_vol`: annualized standard deviation
- `sharpe_ratio`: annualized Sharpe ratio (assume risk-free rate = 0.03)
- `max_drawdown`: maximum drawdown
- `calmar_ratio`: annualized return / abs(max_drawdown)

---

## 5. Study Design

### 5.1 Component 1: REPO vs. MVPO Comparison

**Step 1 — Efficient Frontiers:**
- Compute efficient frontier for both MVPO and REPO (in-sample)
- Plot side-by-side: Risk (x-axis) vs. Expected Return (y-axis)
- Note: x-axis scales differ (volatility vs. entropy) — use two subplots or dual axis

**Step 2 — Minimum Risk Portfolios:**
- Extract the minimum-risk portfolio for each method (alpha = 0)
- Print weights side-by-side as a bar chart
- Compute H(w) (entropy of weight vector) as diversification measure for each

**Step 3 — Fixed Return Comparison:**
- Select a target return equal to the median return on the MVPO frontier
- Find the portfolio on each frontier closest to this target return
- Compare weights, risk metrics, and diversification (H(w))

**Step 4 — Out-of-Sample Backtest:**
- Take 5 representative portfolios from each frontier (evenly spaced by alpha)
- Apply weights to OOS returns
- Compare average performance metrics across all 5 portfolio pairs
- Plot cumulative return paths for min-risk and max-return portfolios

### 5.2 Component 2: Bin Sensitivity Analysis

**Step 1 — Single Portfolio Sensitivity:**
- Fix weights at equal weight (1/n each asset)
- Compute REPO entropy for `m ∈ [5, 10, 15, 20, 30, 50, 100]`
- Plot entropy vs. m to visualize sensitivity
- Also plot normalized entropy `H_norm = H / log(m)` on same chart

**Step 2 — Optimal Portfolio Sensitivity:**
- For each `m` in the same range, solve REPO optimization at alpha = 2.0
- Store the resulting weight vector
- Compute pairwise weight vector distance: `||w(m₁) - w(m₂)||₂`
- Plot heatmap of pairwise distances to show how much the optimal portfolio changes with bin choice

**Step 3 — Governance Implication:**
- Compute: for how many of the 28 asset pairs (n=8) does the rank order of weights change when m goes from 10 to 30?
- Print this as a concrete governance risk statistic

### 5.3 Component 3: Downside Entropy Extension

**Step 1 — Efficient Frontier Comparison:**
- Compute efficient frontiers for all three methods: MVPO, REPO, Downside REPO
- Plot all three frontiers in a single chart (Return vs. Risk)
- Use different colors and line styles

**Step 2 — Risk Profile Analysis:**
- For each method, extract the minimum-risk portfolio
- Compute and compare:
  - Full distribution of portfolio returns (histogram)
  - Standard entropy H(Rₚ)
  - Downside entropy H_down(Rₚ)
  - VaR (5%) and CVaR (5%)
  - Skewness and kurtosis of portfolio returns

**Step 3 — Downside Focus Validation:**
- For all portfolios on the Downside REPO frontier: compute the ratio of negative return probability to total entropy
- Compare this ratio to standard REPO → shows whether Downside REPO actually concentrates risk measurement on the loss side

---

## 6. Visualization Requirements

### Figure 1: Efficient Frontiers
- 2 subplots: MVPO (left), REPO (right)
- Mark minimum-risk portfolio with a red dot
- Mark maximum-return portfolio with a green dot
- Title, axis labels, legend

### Figure 2: Portfolio Weight Comparison
- Grouped bar chart: MVPO vs. REPO weights for min-risk and target-return portfolios
- Include H(w) diversification score in chart title

### Figure 3: Out-of-Sample Cumulative Returns
- Line chart: cumulative return over OOS period for min-risk portfolios (MVPO vs. REPO)
- Include final return and Sharpe ratio in legend

### Figure 4: Bin Sensitivity
- 2-panel figure:
  - Left: H vs. m for equal-weight portfolio (raw + normalized)
  - Right: Heatmap of pairwise optimal weight distances across bin sizes

### Figure 5: All Three Frontiers
- Single chart with three efficient frontiers (MVPO, REPO, Downside REPO)
- Different color per method, same x-axis (return), normalized y-axis for comparability

### Figure 6: Return Distribution Comparison
- 3 histograms side-by-side for min-risk portfolios of each method
- Overlay VaR(5%) and CVaR(5%) as vertical lines
- Include skewness and kurtosis in subtitle

---

## 7. Output Summary Table

At the end of the script, print a formatted summary table:

```
╔══════════════════════════════════════════════════════════════════════╗
║              OUT-OF-SAMPLE PERFORMANCE SUMMARY                      ║
╠══════════════════╦══════════════╦══════════════╦════════════════════╣
║ Metric           ║ MVPO (min)   ║ REPO (min)   ║ Downside (min)     ║
╠══════════════════╬══════════════╬══════════════╬════════════════════╣
║ Total Return     ║              ║              ║                    ║
║ Ann. Return      ║              ║              ║                    ║
║ Ann. Volatility  ║              ║              ║                    ║
║ Sharpe Ratio     ║              ║              ║                    ║
║ Max Drawdown     ║              ║              ║                    ║
║ Calmar Ratio     ║              ║              ║                    ║
║ VaR (5%)         ║              ║              ║                    ║
║ CVaR (5%)        ║              ║              ║                    ║
║ Skewness         ║              ║              ║                    ║
║ H(w) Divers.     ║              ║              ║                    ║
╚══════════════════╩══════════════╩══════════════╩════════════════════╝
```

---

## 8. Implementation Notes

### Numerical Stability
- Always remove zero-probability bins before computing log: `p = p[p > 0]`
- For SLSQP: use multiple random starting points if optimization fails to converge
- Add small regularization to covariance matrix for MVPO: `Σ_reg = Σ + 1e-6 * I`

### Performance
- Pre-compute the covariance matrix and mean returns once before the frontier loop
- The entropy computation involves a histogram per optimizer call — this is the bottleneck
- For bin sensitivity analysis, cache results in a dictionary

### Code Structure
```
repo_study.py
├── 1. Data Download & Preprocessing
├── 2. Core Functions
│   ├── compute_portfolio_entropy()
│   ├── optimize_portfolio()
│   ├── compute_efficient_frontier()
│   └── backtest_portfolio()
├── 3. Component 1: REPO vs. MVPO
├── 4. Component 2: Bin Sensitivity
├── 5. Component 3: Downside Entropy
├── 6. Summary Table
└── 7. Save All Figures
```

### Default Parameters
- Number of bins for main analysis: `m = 20`
- Number of frontier points: `n_points = 50`
- Risk-free rate: `rf = 0.03 / 52` (weekly)
- Random seed: `np.random.seed(42)`

---

## 9. Research Questions to Answer

The implementation should allow answering the following questions:

1. Does REPO produce more diversified portfolios than MVPO at equivalent return levels?
2. How much does the optimal REPO portfolio change when bin size varies from m=10 to m=30?
3. Does Downside REPO produce portfolios with lower CVaR than standard REPO at equivalent expected returns?
4. Which method achieves the best out-of-sample Sharpe ratio over the 2022–2024 period?
5. Is the governance risk of bin sensitivity (measured by weight vector distance) practically significant?

---

*Implementation Brief prepared for Claude Code — based on Mercurio et al. (2020) "An Entropy-Based Approach to Portfolio Optimization", Entropy 22(3), 332.*
