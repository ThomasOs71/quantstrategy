# -*- coding: utf-8 -*-
"""
Created on Sat May 17 14:17:10 2025

@author: Thomas Osowski
"""

# %% Import Packages
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.covariance import LedoitWolf, GraphicalLassoCV, GraphicalLasso, EllipticEnvelope
from statsmodels.multivariate.factor import Factor
from matplotlib.patches import Ellipse

# %% Functoins
def denoise_corr_mp(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Spectral denoising of a correlation matrix via Marchenko-Pastur filtering.

    Parameters:
    - returns: DataFrame of shape (T, N), T observations of N variables.

    Returns:
    - corr_denoised: DataFrame of the cleaned correlation matrix (N x N).
    """
    # Number of observations T and variables N
    T, N = returns.shape
    
    # 1. Compute sample correlation matrix
    corr = returns.corr().values
    
    # 2. Compute Marchenko-Pastur upper bound for correlation matrix (sigma^2 = 1)
    q = N / T
    lambda_plus = (1 + np.sqrt(1/q))**2
    
    # 3. Eigen-decomposition
    eigvals, eigvecs = np.linalg.eigh(corr)
    
    # 4. Identify noise eigenvalues (<= lambda_plus)
    noise_mask = eigvals <= lambda_plus
    
    # 5. Replace noise eigenvalues by their average
    avg_noise = np.mean(eigvals[noise_mask])
    eigvals_denoised = np.where(noise_mask, avg_noise, eigvals)
    
    # 6. Reconstruct denoised correlation matrix
    corr_denoised = eigvecs @ np.diag(eigvals_denoised) @ eigvecs.T
    
    # 7. Rescale to ensure diagonal = 1
    D_inv = np.diag(1 / np.sqrt(np.diag(corr_denoised)))
    corr_denoised = D_inv @ corr_denoised @ D_inv
    
    # Convert back to DataFrame
    return pd.DataFrame(corr_denoised, index=returns.columns, columns=returns.columns)


def cov2corr(cov:np.ndarray) -> np.ndarray:
    vola_inv = np.diag(1/np.sqrt(np.diag(cov)))
    corr = vola_inv@cov@vola_inv
    return corr


def exp_time_filter(nobs:int, alpha:int) -> np.ndarray:
    exp_time_filter = np.exp((np.log(2) / alpha) * np.arange(0,nobs))
    exp_time_filter /= np.sum(exp_time_filter)
    return exp_time_filter


def plot_covariance_ellipse(mean:np.ndarray, 
                            cov:np.ndarray, 
                            n_std:float=1.0, 
                            ax=None, 
                            **kwargs):
    """
    Plots an ellipse representing the covariance of a 2D distribution.

    Parameters:
    - mean: array-like, shape (2,), the center of the ellipse.
    - cov: 2x2 array, covariance matrix.
    - n_std: float, the radius of the ellipse in standard deviations.
    - ax: matplotlib Axes object (optional).
    - kwargs: additional keyword arguments passed to Ellipse patch.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Eigen-decomposition to get ellipse parameters
    eigenvals, eigenvecs = np.linalg.eigh(cov)
    order = eigenvals.argsort()[::-1]
    eigenvals, eigenvecs = eigenvals[order], eigenvecs[:, order]
    
    # Width and height of ellipse = 2 * n_std * sqrt(eigenvals)
    width, height = 2 * n_std * np.sqrt(eigenvals)
    
    # Angle of rotation in degrees
    angle = np.degrees(np.arctan2(eigenvecs[1, 0], eigenvecs[0, 0]))
    
    # Create and add ellipse patch
    ellipse = Ellipse(xy=mean, width=width, height=height, angle=angle, **kwargs)
    ax.add_patch(ellipse)
    
    # Plot settings
    ax.set_aspect('equal')
    ax.set_xlim(mean[0] - width*1.1, mean[0] + width*1.1)
    ax.set_ylim(mean[1] - height*1.1, mean[1] + height*1.1)
    ax.set_xlabel('Variable 1')
    ax.set_ylabel('Variable 2')
    ax.grid(True)
    return ax


# %% Download Data

### Equity Data
tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "META", "NFLX", "ORCL", "IBM", "INTC", "CSCO", "ADBE", "AMD", "PYPL", "CRM", "QCOM", "TXN", "MU", "NOW",
    "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "PNC", "TFC", "USB", "CB", "MMC", "MET", "AIG", "PRU", "ALL", "AFL", "CME",
    "HD", "LOW", "TGT", "WMT", "COST", "MCD", "SBUX", "NKE", "LULU", "TJX", "ULTA", "YUM", "DG", "ROST", "MAR", "HLT", "CCL", "RCL", "UBER", "LYFT",
    "JNJ", "PFE", "MRK", "BMY", "LLY", "ABBV", "GILD", "BIIB", "REGN", "AMGN", "ZTS", "MRNA", "CVS", "UNH", "HUM", "ANTM", "CI", "HCA", "DGX", "LH",
    "XOM", "CVX", "COP", "EOG", "SLB", "MPC", "VLO", "PSX", "KMI", "WMB", "PXD", "OXY", "DUK", "SO", "NEE", "AEP", "D", "EXC", "PEG", "XEL",
    "GE", "HON", "RTX", "BA", "CAT", "DE", "UPS", "FDX", "UNP", "CSX", "NSC", "EMR", "MMM", "LMT", "GD", "NOC", "ITW", "DOV", "ROK", "JCI",
    "DIS", "T", "VZ", "TMUS", "CMCSA", "CHTR", "FOXA", "DISH", "PARA", "WBD", "SPOT", "SIRI", "ROKU", "ATVI", "TTWO", "EA", "MTCH", "ZG", "ANGI", "SNAP",
    "KO", "PEP", "PG", "CL", "KMB", "MDLZ", "KHC", "HSY", "GIS", "SYY", "WBA", "TAP", "EL", "CPB", "CAG", "STZ", "CHD", "KR", "TSN", "MKC",
    "AMT", "PLD", "CCI", "EQIX", "PSA", "SPG", "WELL", "VTR", "DLR", "O", "AVB", "EQR", "ESS", "EXR", "PEAK", "BXP", "FRT", "HST", "MAC", "REG",
    "DUK", "SO", "NEE", "AEP", "D", "EXC", "PEG", "XEL", "WEC", "ED", "EIX", "PPL", "AEE", "CNP", "CMS", "NI", "ES", "LNT", "DTE", "SRE"
    ]

# Download Equity Data - Monthly Frequency
equity_data = yf.download(tickers, 
                          start='2010-01-01', 
                          end='2024-01-01', 
                          interval = "1mo")

closing_prices = equity_data['Close'].ffill()  # Use Simple Forward Fill for Simplicity
closing_prices = closing_prices.dropna(axis = 1).iloc[:,:50]  # Drop Equities with short data sample

# Check for Missing Values
if closing_prices.isna().sum().sum() == 0:
    print("Data Sample is complete without NaN")

# Transform to Log Return Data
equity_logret = (np.log(closing_prices) - np.log(closing_prices.shift(1))).iloc[1:,:]
equity_logret_ = equity_logret.values


### Asset Class Index Data
tickers = {
    'S&P 500':     '^GSPC',   # already in USD
    'MSCI World':  'URTH',    # ETF priced in USD
    'STOXX 600':   '^STOXX',  # index priced in EUR 
    'US Agg (AGG)':    'AGG',  # ETF in USD
    'US 7–10Y (IEF)':  'IEF',  # ETF in USD
}

# Download index prices
start_date = "2010-01-01"
raw = yf.download(list(tickers.values()),
                  start=start_date,
                  auto_adjust=True,
                  progress=False,
                  interval = "1mo")['Close']\
        .rename(columns={v:k for k,v in tickers.items()})

# Download EUR→USD spot rate
# Yahoo ticker for EURUSD is 'EURUSD=X'
fx = yf.download('EURUSD=X',
                 start=start_date,
                 auto_adjust=False,  # spot rate, no adj. close
                 progress=False,
                 interval = "1mo")['Close']

# Align dates
index_data = raw.join(fx, how='inner')  # keeps only dates present in both

# Convert EUR-priced series into USD
euro_cols = ['STOXX 600']
for col in euro_cols:
    index_data[col] = index_data[col] * index_data['EURUSD=X']

# Drop the FX series if you like
index_data = index_data.drop(columns='EURUSD=X')

# Transform to Log Return Data
closing_prices_index = index_data.ffill()  # Use Simple Forward Fill for Simplicity
index_logret = (np.log(closing_prices_index) - np.log(closing_prices_index.shift(1))).dropna()
index_logret_ = index_logret.values


# %% Ledoit - Wolf Example




# %% Graph: Eigenvalues
## Eigenvalues of Covariance and Inverted Covariance
number_of_asset_included = 8
cov_selected = np.cov(equity_logret_[:,:number_of_asset_included].T)

values, vectors = np.linalg.eig(cov_selected)
values_inv, vector_inv = np.linalg.eigh(np.linalg.pinv(cov_selected))

# Test, wheather EV of inverted covariance are identical to reciproical eigenvalues of covariance
if all(abs(1/values - values_inv) < 0.01 ):
    print("Eigenvalues of Inverted Covariance are identical to reciprocial eigenvalues of Covariance")

# Parameter of Graph
labels = ["EV " + str(i) for i in range(len(values))]
x = np.arange(len(values))  # Label positions
width = 0.4
fig, ax1 = plt.subplots()

# Plot the original values on the left y-axis
bar1 = ax1.bar(x - width/2, values, width, label='Eigenvalues of Covariance', color='tab:blue')
ax1.set_ylabel('Eigenvalues of Covariance', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis sharing the same x-axis
ax2 = ax1.twinx()
bar2 = ax2.bar(x + width/2, values_inv, width, label='Eigenvalues of Inverted Covariance', color='tab:red')
ax2.set_ylabel('Eigenvalues of Inverted Covariance', color='tab:red')
ax2.tick_params(axis='y', labelcolor='tab:red')

# X-axis settings
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=45)

# Title and layout
plt.title('Eigenvalues of Original Covariance vs Eigenvalues of Inverted Covariance')
plt.tight_layout()
plt.show()

# %% Graph: Influence of Outliers
# Define mean and covariance
mean = np.array([0, 0])
cov = np.array([[3, 1],
                [1, 2]])

corr = cov2corr(cov)

# Simulate sample data
np.random.seed(0)
data = np.random.multivariate_normal(mean, cov, size=100)
mean_est = np.mean(data,axis=0)
cov_est = np.cov(data.T)

# Outlier-infected data
outlier = np.array([[4,-4],[20,-5],[-5,5],[-6,5]])
data_dirty = np.r_[data,outlier]
mean_est_outlier = np.mean(data_dirty, axis=0)
cov_est_outlier = np.cov(data_dirty.T)

corr_est_outlier = cov2corr(cov_est_outlier)

# Plot data and covariance ellipses
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(data[:, 0], data[:, 1], s=10, alpha=0.6, label='Data')
ax.scatter(outlier[:, 0], outlier[:, 1], s=20, alpha=0.6, color = "red",
            label='Data')

plot_covariance_ellipse(mean, cov, n_std=1.0, ax=ax,
                    edgecolor='red', lw=2, fill=False, label='True Covariance')
plot_covariance_ellipse(mean_est_outlier, cov_est_outlier, n_std=1.0, ax=ax,
                    edgecolor='blue', lw=2, fill=False, label='Estimated Covariance')
ax.annotate(text = f"True Correlation: {np.round(corr[0,1],2)}",xy = (2,-2))
ax.annotate(text = f"Estimated Correlation: {np.round(corr_est_outlier[0,1],2)}",xy = (2,-2.5))

ax.set_title("Impact of Few Outlier on the Covariance Matrix")
ax.legend()
plt.show()

# %% Graph: Weight Comparition of Exponentional and Sample Covariance
time_filter_high_decay = exp_time_filter(equity_logret.shape[0], 50)
time_filter_low_decay = exp_time_filter(equity_logret.shape[0], 100)
time_filter_sample_cov = exp_time_filter(equity_logret.shape[0], 1000000)  # Sample Covariance is equivalent to setting alpha to infinity

plt.plot(np.c_[time_filter_high_decay,
               time_filter_low_decay,
               time_filter_sample_cov])

plt.title("Comparision of Data Point Weightening for each Observation")

plt.legend(["Exp. Time Filter with Fast Decay (low alpha)",
            "Exp. Time Filter with Slow Decay (High alpha)",
            "Time Filter' of Covariance Matrix"])


# %% Graph: Estimation impact of ROBUST Covariance Estimation
ee = EllipticEnvelope(contamination=0.05, support_fraction=None, random_state=0)
ee.fit(data_dirty)
mean_robust = ee.location_
cov_robust = ee.covariance_

fig, ax = plt.subplots(figsize=(8, 6))

plot_covariance_ellipse(mean, cov, n_std=1.0, ax=ax,
                    edgecolor='red', lw=2, fill=False, label='True Sample')

plot_covariance_ellipse(mean_est_outlier, cov_est_outlier, n_std=1.0, ax=ax,
                    edgecolor='blue', lw=2, fill=False, label='Sample Covariance - Outlier - Infected')

plot_covariance_ellipse(mean_robust, cov_robust, n_std=1.0, ax=ax,
                    edgecolor='green', lw=2, fill=False, label='Sample Covariance - Roubust Estimation')

ax.set_title("Robust Estimation of Covariance deals with Outliers")
ax.legend()
plt.show()


# %% Graph: Spectral Denoising
# Spectral
spectral_corr = denoise_corr_mp(equity_logret)
value_sc, vector_sc = np.linalg.eigh(spectral_corr)

# Sample
sample_corr = cov2corr(np.cov(equity_logret_.T))
value_sample, vector_sample = np.linalg.eigh(sample_corr)

fig, ax1 = plt.subplots()
ax1.plot(np.log(value_sample[::-1]),
         label = "Eienvalues from Sample Covariance")

ax1.plot(np.log(value_sc[::-1]),
         label = "Eienvalues from Spectral Denoising")
ax1.set_xlabel("Ranking of Sorted Eigenvalues")
ax1.set_ylabel("Log-Scale of Eigenvalues")
ax1.set_title("Sample Covariance vs. Spectral Denoising: Comparision of Eigenvalues")
ax1.annotate(xy = (10,2.5), text=f"Minimum Eigenvalue in Sample Correlation: {np.round(np.min(value_sample),2)}",)
ax1.annotate(xy = (10,2), text=f"Cond. Number in Sample Correlation: {np.round(np.linalg.cond(sample_corr),2)}",)
ax1.annotate(xy = (10,1.5), text=f"Minimum Eigenvalue in Spectral Denoising: {np.round(np.min(value_sc),2)}",)
ax1.annotate(xy = (10,1), text=f"Cond. Number after Spectral Denoising: {np.round(np.linalg.cond(spectral_corr),2)}",)
ax1.legend()
plt.show()

# %% EQUITY - Estimation of Covariances 
### (Fixed) Sample Covariance
equity_sample_cov = np.cov(equity_logret_.T)
equity_sample_corr = cov2corr(equity_sample_cov)

### Exponentionall-Weighted Sample Covariance
time_filter_cov = exp_time_filter(equity_logret.shape[0], 100)
equity_exp_cov = np.cov(equity_logret.T,aweights = time_filter_cov)
equity_exp_corr = cov2corr(equity_exp_cov)

### EllipticEnvelope (Robust Estimation)
# Direct Estimation of Covariance
ee = EllipticEnvelope(contamination=0.05, support_fraction=None, random_state=0)
ee.fit(equity_logret_)
equity_robust_cov = ee.covariance_
labels = ee.predict(equity_logret_)      # 1 = Inlier, -1 = Outlier
dist2_ee = ee.mahalanobis(equity_logret_)  # quadr. Distanzen
offset = ee.offset_                       # Schwellenwert

print("MinCovDet: robust covariance matrix\n", pd.DataFrame(equity_robust_cov))
print("\nEllipticEnvelope offset (squared MD):", offset)
print("Anzahl Outlier (EE):", np.sum(labels == -1))

'''
Keep in Mind:
Exp Value of Sqrt(Maha) ~ np.sqrt(n) (Approximation)
'''

# Estimation of Correlation - subsequent combination with Volatilites
ee = EllipticEnvelope(contamination=0.05, support_fraction=None, random_state=0)
ee.fit((equity_logret - equity_logret.mean()) / equity_logret.std())
equity_robust_corr = ee.covariance_
equity_robust_corr = cov2corr(equity_robust_corr)
equity_robust_cov_indirect = np.diag(np.sqrt(np.diag(equity_sample_cov))) @ equity_robust_corr @ np.diag(np.sqrt(np.diag(equity_sample_cov)))

# Condition Numbers remain high: Better Combination with Additional Shrinkage (here: Glasso)
'''
Example of "Combination" of Approaches
'''
glasso_cov_estimator = GraphicalLasso(alpha=0.02,
                                      covariance = "precomputed").fit(equity_robust_corr)
equity_robust_glasso_corr = glasso_cov_estimator.covariance_
equity_robust_glasso_cov = np.diag(np.sqrt(np.diag(equity_sample_cov))) @ equity_robust_glasso_corr @ np.diag(np.sqrt(np.diag(equity_sample_cov)))


### Ledoit-Wolf
## Shrinkage of whole Covariance
equity_lw_cov_direct = LedoitWolf().fit(equity_logret).covariance_

## Shrinkage of Correlation 
equity_return_standardized = (equity_logret - equity_logret.mean()) / equity_logret.std()
# Combine Shrinked Correlation with Volatilies
equity_lw_corr = LedoitWolf().fit(equity_return_standardized).covariance_ 
equity_lw_corr = np.diag(1/np.sqrt(np.diag(equity_lw_corr)))@equity_lw_corr@np.diag(1/np.sqrt(np.diag(equity_lw_corr)))
equity_lw_cov_indirect = np.diag(np.sqrt(np.diag(equity_sample_cov))) @ equity_lw_corr @ np.diag(np.sqrt(np.diag(equity_sample_cov)))


### GraphicalLasso
## Estimation of Covariance
glasso_cov_estimator = GraphicalLasso(alpha=0.0002,
                            covariance = "precomputed").fit(equity_sample_cov)
equity_glasso_cov = glasso_cov_estimator.covariance_
'''
Glasso directly on the covariance can make issues...better use on correlation...
'''

## Estimation of Correlation
alphas = np.logspace(-1.5, 1, num=10)
edge_model = GraphicalLassoCV(alphas=alphas)
X = equity_logret.copy()
X /= X.std(axis=0)
edge_model.fit(X)
equity_glasso_corr = edge_model.covariance_
equity_glasso_corr = np.diag(1/np.sqrt(np.diag(edge_model.covariance_)))@equity_glasso_corr@np.diag(1/np.sqrt(np.diag(edge_model.covariance_)))
# Combine with Volatility Estimate
equity_glasso_cov_indirect = np.diag(np.sqrt(np.diag(equity_sample_cov))) @ equity_glasso_corr @ np.diag(np.sqrt(np.diag(equity_sample_cov)))


### Spectral Denosing
equity_spectral_corr = denoise_corr_mp(equity_logret).values
equity_spectral_cov = np.diag(np.sqrt(np.diag(equity_sample_cov))) @ equity_spectral_corr @ np.diag(np.sqrt(np.diag(equity_sample_cov)))

### Factor Model
'''
Estimates a Hidden Factor Model on the Correlation.
Afterwards, the volatilities are added to obtain the covariance

'''
# Factor Model for Correlation Estimatio
vola = np.sqrt(np.diag(equity_sample_cov))
factor_model = Factor(n_factor=3, 
                      corr = equity_sample_corr, 
                      method='pa', 
                      smc=True).fit()  # PAF fitted model

beta = factor_model.loadings  # loadings
delta = factor_model.uniqueness  # variances
equity_factor_cov = beta@beta.T + np.diagflat(delta)

# Adjust to obtain Covariance
beta = np.diag(vola)@beta  # re-scaled loadings (Correlation -> Covariance)
delta = vola**2*delta  # re-scaled variances -> (Correlation -> Covariance)
equity_factor_cov = beta@beta.T + np.diagflat(delta)  # PAF factor analysis covariance matrix


# %% ASSET CLASS INDEX - Estimation of Covariances 
### (Fixed) Sample Covariance
index_sample_cov = np.cov(index_logret_.T)
index_sample_corr = cov2corr(index_sample_cov)

### Exponentionall-Weighted Sample Covariance
time_filter_cov = exp_time_filter(index_logret_.shape[0], 100)
index_exp_cov = np.cov(index_logret.T,aweights = time_filter_cov)
index_exp_corr = cov2corr(equity_exp_cov)

### EllipticEnvelope (Robust Estimation)
# Direct Estimation of Covariance
ee = EllipticEnvelope(contamination=0.05, support_fraction=None, random_state=0)
ee.fit(index_logret_)
index_robust_cov = ee.covariance_
labels = ee.predict(index_logret_)      # 1 = Inlier, -1 = Outlier
dist2_ee = ee.mahalanobis(index_logret_)  # quadr. Distanzen
offset = ee.offset_                       # Schwellenwert

# Estimation of Correlation - subsequent combination with Volatilites
ee = EllipticEnvelope(contamination=0.05, support_fraction=None, random_state=0)
ee.fit((index_logret - index_logret.mean()) / index_logret.std())
index_robust_corr = ee.covariance_
index_robust_corr = cov2corr(index_robust_corr)
index_robust_cov_indirect = np.diag(np.sqrt(np.diag(index_sample_cov))) @ index_robust_corr @ np.diag(np.sqrt(np.diag(index_sample_cov)))

# Condition Numbers remain high: Better Combination with Additional Shrinkage (here: Glasso)
'''
Example of "Combination" of Approaches
'''
glasso_cov_estimator = GraphicalLasso(alpha=0.02,
                                      covariance = "precomputed").fit(index_robust_corr)
index_robust_glasso_corr = glasso_cov_estimator.covariance_
index_robust_glasso_cov = np.diag(np.sqrt(np.diag(index_sample_cov))) @ index_robust_glasso_corr @ np.diag(np.sqrt(np.diag(index_sample_cov)))

### Ledoit-Wolf
## Shrinkage of whole Covariance
index_lw_cov_direct = LedoitWolf().fit(index_logret).covariance_

## Shrinkage of Correlation 
index_return_standardized = (index_logret - index_logret.mean()) / index_logret.std()
# Combine Shrinked Correlation with Volatilies
index_lw_corr = LedoitWolf().fit(index_return_standardized).covariance_ 
index_lw_corr = np.diag(1/np.sqrt(np.diag(index_lw_corr)))@index_lw_corr@np.diag(1/np.sqrt(np.diag(index_lw_corr)))
index_lw_cov_indirect = np.diag(np.sqrt(np.diag(index_sample_cov))) @ index_lw_corr @ np.diag(np.sqrt(np.diag(index_sample_cov)))

### GraphicalLasso
## Estimation of Covariance
glasso_cov_estimator = GraphicalLasso(alpha=0.0002,
                            covariance = "precomputed").fit(index_sample_cov)
index_glasso_cov = glasso_cov_estimator.covariance_
'''
Glasso directly on the covariance can make issues...better use on correlation...
'''

## Estimation of Correlation
alphas = np.logspace(-1.5, 1, num=10)
edge_model = GraphicalLassoCV(alphas=alphas)
X = index_logret.copy()
X /= X.std(axis=0)
edge_model.fit(X)
index_glasso_corr = edge_model.covariance_
index_glasso_corr = np.diag(1/np.sqrt(np.diag(edge_model.covariance_)))@index_glasso_corr@np.diag(1/np.sqrt(np.diag(edge_model.covariance_)))
# Combine with Volatility Estimate
index_glasso_cov_indirect = np.diag(np.sqrt(np.diag(index_sample_cov))) @ index_glasso_corr @ np.diag(np.sqrt(np.diag(index_sample_cov)))

### Spectral Denosing
index_spectral_corr = denoise_corr_mp(index_logret).values
index_spectral_cov = np.diag(np.sqrt(np.diag(index_sample_cov))) @ index_spectral_corr @ np.diag(np.sqrt(np.diag(index_sample_cov)))

### Factor Model
'''
Estimates a Hidden Factor Model on the Correlation.
Afterwards, the volatilities are added to obtain the covariance

'''
# Factor Model for Correlation Estimatio
vola = np.sqrt(np.diag(index_sample_cov))
factor_model = Factor(n_factor=3, 
                      corr = index_sample_corr, 
                      method='pa', 
                      smc=True).fit()  # PAF fitted model

beta = factor_model.loadings  # loadings
delta = factor_model.uniqueness  # variances
index_factor_corr = beta@beta.T + np.diagflat(delta)

# Adjust to obtain Covariance
beta = np.diag(vola)@beta  # re-scaled loadings (Correlation -> Covariance)
delta = vola**2*delta  # re-scaled variances -> (Correlation -> Covariance)
index_factor_cov = beta@beta.T + np.diagflat(delta)  # PAF factor analysis covariance matrix

index_sample_corr




# %% EQUITY - Comparision of Results
### Condition Numbers
equity_cov_estimators = [equity_sample_cov,              #  
                            equity_exp_cov,                 # 
                            equity_robust_cov,              #
                            equity_robust_cov_indirect,     #
                            equity_robust_glasso_cov,       # Robust Estimation + Glasse
                            equity_lw_cov_direct,           # Ledoit-Wolf
                            equity_lw_cov_indirect,         # Ledoit-Wolf - Via Correlation
                            equity_glasso_cov,              # Glasso Covariance Direct
                            equity_glasso_cov_indirect,     #
                            equity_spectral_cov,            #
                            equity_factor_cov]              #                            
                            
equity_condition_number = [np.linalg.cond(i) for i in equity_cov_estimators]

equity_condition_numbers_df = pd.Series(equity_condition_number,
                                        index = ["Sample Covariance",
                                                 "Flexible Probability Appproach",
                                                 "Robust Estimation (direct Estimation)",
                                                 "Robust Estimation (via Correation)",
                                                 "Robust Estimation + Glasso",
                                                 "Ledoit-Wolf (direct Estimation)",
                                                 "Ledoit-Wolf (via Correlation)",
                                                 "Glasso Estimation (direct Estimation)",
                                                 "Glasso Estimation (via Correlation)",
                                                 "Spectral Denoising",
                                                 "Hidden Factor Approach"],
                                        name = "Condition Numbers of various Estimators")


fig, ax = plt.subplots()
# Create horizontal bar chart
ax.barh(
    ["       " + str(i[0]) + " " + str(i[1]) for i in enumerate(equity_condition_numbers_df.index)][::-1],
    equity_condition_numbers_df[::-1],
    color = "lightgreen")

# Set title
ax.set_title("Equity: Covariance Estimator and Number of Conditions", ha='center')

# Left-align y-tick labels
ax.tick_params(axis='y', pad=10)  # Add padding so labels don't overlap bars
for label in ax.get_yticklabels():
    label.set_horizontalalignment('left')
    label.set_color('black')  # 🔴 Set text color to red
plt.show()




### Equity - Frobenius Norms
equity_rel_frob = np.zeros([11,11])

for i in enumerate(equity_condition_numbers_df.index):
    for j in enumerate(equity_condition_numbers_df.index):
        equity_rel_frob[i[0],j[0]] = np.linalg.norm(equity_cov_estimators[i[0]] - equity_cov_estimators[j[0]],ord="fro")


# Plot
fig, ax = plt.subplots(figsize=(8, 8))  # Adjust size as needed
cax = ax.imshow(equity_rel_frob, cmap='coolwarm')

# Annotate cells with values
for i in range(equity_rel_frob.shape[0]):
    for j in range(equity_rel_frob.shape[1]):
        ax.text(j, i, f'{equity_rel_frob[i, j]:.2f}',
                ha='center', va='center', color='black')

# Optional: Add colorbar and remove ticks
fig.colorbar(cax, ax=ax, label='Frobenius Norm')
ax.set_xticks([i for i in range(0,11)])
ax.set_yticks([i for i in range(0,11)])
ax.set_title('Equity - Frobenius Norms Heatmap',size = 20)
plt.legend(list(equity_condition_numbers_df.index))
plt.tight_layout()
plt.show()



# %% ASSET CLASS INDEX - Comparision of Results 
### Condition Numbers
index_cov_estimators = [index_sample_cov,              #  
                            index_exp_cov,                 # 
                            index_robust_cov,              #
                            index_robust_cov_indirect,     #
                            index_robust_glasso_cov,       # Robust Estimation + Glasse
                            index_lw_cov_direct,           # Ledoit-Wolf
                            index_lw_cov_indirect,         # Ledoit-Wolf - Via Correlation
                            index_glasso_cov,              # Glasso Covariance Direct
                            index_glasso_cov_indirect,     #
                            index_spectral_cov,            #
                            index_factor_cov]              #                            
                            
index_condition_number = [np.linalg.cond(i) for i in index_cov_estimators]

index_condition_numbers_df = pd.Series(index_condition_number,
                                        index = ["Sample Covariance",
                                                 "Flexible Probability Appproach",
                                                 "Robust Estimation (direct Estimation)",
                                                 "Robust Estimation (via Correation)",
                                                 "Robust Estimation + Glasso",
                                                 "Ledoit-Wolf (direct Estimation)",
                                                 "Ledoit-Wolf (via Correlation)",
                                                 "Glasso Estimation (direct Estimation)",
                                                 "Glasso Estimation (via Correlation)",
                                                 "Spectral Denoising",
                                                 "Hidden Factor Approach"],
                                        name = "Condition Numbers of various Estimators")


fig, ax = plt.subplots()
# Create horizontal bar chart
ax.barh(
    ["       " + str(i[0]) + " " + str(i[1]) for i in enumerate(index_condition_numbers_df.index)][::-1],
    index_condition_numbers_df[::-1],
    color = "lightgreen")

# Set title
ax.set_title("Asset Class Index: Covariance Estimator and Number of Conditions", ha='center')

# Left-align y-tick labels
ax.tick_params(axis='y', pad=10)  # Add padding so labels don't overlap bars
for label in ax.get_yticklabels():
    label.set_horizontalalignment('left')
    label.set_color('black')  # 🔴 Set text color to red
plt.show()




### Equity - Frobenius Norms
index_rel_frob = np.zeros([11,11])

for i in enumerate(index_condition_numbers_df.index):
    for j in enumerate(index_condition_numbers_df.index):
        index_rel_frob[i[0],j[0]] = np.linalg.norm(index_cov_estimators[i[0]] - index_cov_estimators[j[0]],ord="fro")


# Plot
fig, ax = plt.subplots(figsize=(8, 8))  # Adjust size as needed
cax = ax.imshow(100 * index_rel_frob, cmap='coolwarm')

# Annotate cells with values
for i in range(index_rel_frob.shape[0]):
    for j in range(index_rel_frob.shape[1]):
        ax.text(j, i, f'{100*index_rel_frob[i, j]:.2f}',
                ha='center', va='center', color='black')

# Optional: Add colorbar and remove ticks
fig.colorbar(cax, ax=ax, label='Frobenius Norm')
ax.set_xticks([i for i in range(0,11)])
ax.set_yticks([i for i in range(0,11)])
ax.set_title('Asset Class Index - Frobenius Norms Heatmap',size = 20)
plt.legend(list(index_condition_numbers_df.index))
plt.tight_layout()
plt.show()






