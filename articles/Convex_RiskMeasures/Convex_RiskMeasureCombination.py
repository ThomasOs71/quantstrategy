# A*- coding: utf-8 -*-
"""
Created on Sun Jul 27 13:40:34 2025

@author: thoma
"""

# %% Packages
import numpy as np
from scipy.stats import  norm, multivariate_normal, t, kurtosis
import matplotlib.pyplot as plt
import pandas as pd
import cvxpy as cp 
from scipy.special import gamma
from matplotlib.gridspec import GridSpec

# %% Functions
# Helper: skewed-t inverse CDF (ppf) using transformation method
def skew_t_ppf(u, df, loc, scale, skew):
    # Convert uniform sample to standard t
    z = t.ppf(u, df)
    # Apply skew transformation
    delta = skew / np.sqrt(1 + skew**2)
    return loc + scale * (delta * np.abs(z) + (1 - delta) * z)

# Correct skew-t parameters to preserve mean and variance
def corrected_skewt_params(mu, var_target, df, skew):
    delta = skew / np.sqrt(1 + skew**2)
    mean_bias = delta * np.sqrt(df / np.pi) * gamma((df - 1) / 2) / gamma(df / 2)
    var_factor = (df / (df - 2)) - (delta ** 2) * (df / np.pi) * (gamma((df - 1) / 2) / gamma(df / 2)) ** 2
    sigma2 = var_target / var_factor
    sigma = np.sqrt(sigma2)
    loc = mu - sigma * mean_bias
    return loc, sigma


def pf_characteristics(weights,
                       scenarios_assets_,
                       bm_scenarios_,
                       alpha,
                       tau,
                       T):
    
    ## Portfolio Characteristics
    # Portfolio Weights
    pf_weights = weights
    pf_weights_df = pd.DataFrame(np.round(pf_weights * 100,2), 
                                   index = asset_index)
    
    # Expected Return
    expret = np.mean(scenarios_assets_ @ pf_weights)
    
    ## Calculate Statistics
    pf_scenarios = scenarios_assets_ @ pf_weights
    
    # Std
    std = np.std(pf_scenarios)
    
    # MAD
    mad = np.mean(np.abs(pf_scenarios - np.mean(pf_scenarios)))
    
    # VaR
    var = -np.percentile(pf_scenarios,
                        alpha*100)
        
    # Calculate CVaR
    cvar = -np.mean(pf_scenarios[pf_scenarios<-var])

    # LPM
    lpm = np.mean(np.maximum(tau - pf_scenarios, 0))
    
    # Downside Deviation
    semivariance = 1/(T-1) * np.sum(np.minimum(0,pf_scenarios - expret)**2)
    semideviation = np.sqrt(semivariance)
    
    # Tracking Error
    te = np.sqrt(np.sum((bm_scenarios_ - pf_scenarios)**2)) / np.sqrt(T)

    return pf_weights_df, {"exp_ret": expret,
                           "std": std,
                           "mad": mad,
                           "var": var,
                           "cvar": cvar,
                           "lpm": lpm,
                           "semivariance": semivariance,
                           "semideviation": semideviation,
                           "te": te
                           }

# %% Load Data

# Set Path on your Drive
path_ = r"C:\Users\thoma\OneDrive\Dokumente\quantstrategy\articles\Convex_RiskMeasures" 

### Load Data: Exp_Ret and Covariance
# Expeted Returns
expret = pd.read_excel(path_+ "\\" + r"expected_returns.xlsx",
                       index_col = 0)
# Covariance
covariance = pd.read_excel(path_+ "\\" + r"covariance.xlsx",
                           index_col = 0)

# Covariance
df = pd.read_excel(path_+ "\\" + r"df.xlsx",
                           index_col = 0)

asset_index = expret.index

# Transform to Values (np)
expret = expret.values
covariance = covariance.values
df = df.values.squeeze()

### Extract Correlation and Volatilites
num_assets = np.shape(covariance)[0]
variance = np.diag(covariance)
vola = np.sqrt(variance)

corr = np.diag((1 / vola)) @ covariance @ np.diag((1 / vola))

# %% Marginal - Compula: Marginal: Student t + Gaussian Copula [n Element Test Case]
### Generate Gaussian Copula:
means = np.zeros(num_assets)
 
## Simulation of Normal Distribution Data Points (= Sample)
simulated_normal_data = multivariate_normal.rvs(mean = means, cov = corr, size = 100000)

# Generation of Grades: Transform the simulated data to Uniform(0, 1) scale
uniform_data = norm.cdf(simulated_normal_data)

#######
SET_UNIVARIATE_DISTRIBUTION = "Skew-t"                      # "Student-t"
# SET_UNIVARIATE_DISTRIBUTION = "Student-t"
####

### Include Marginal Information via Univariate Distributions
## STUDENT T DISTRIBUTION
if SET_UNIVARIATE_DISTRIBUTION == "Student-t":
    
    distribution_parameters = {}
    for i, name in enumerate(asset_index):
        param_array= np.empty([3])
        param_array[0] = expret[i].squeeze() # Exp Return = Loc Parameter auf Position 0
        param_array[1] = variance[i] * (df[i] - 2) / df[i] # Scale Parameter auf Position 1
        param_array[2] = df[i] # DoF Parameter auf Position 2
        distribution_parameters[name] = param_array

    scenarios = np.zeros_like(uniform_data)
    for i, name in enumerate(asset_index):
        loc_ = distribution_parameters[name][0]
        scale_ = distribution_parameters[name][1]
        df_ = distribution_parameters[name][2]
        scenarios[:, i] = t.ppf(uniform_data[:, i], df_, loc_, np.sqrt(scale_))

## SKEW - T Distribution
elif SET_UNIVARIATE_DISTRIBUTION == "Skew-t":
    # Define skewness parameters for each asset
    skewness_parameters = {
        "Government Bonds USA": 0.05,
        "Government Bonds EMU": 0.05,
        "Corporate Bonds EMU": -0.08,
        "Corporate Bonds USA": -0.06,
        "Global High Yield Bonds": 0.2,
        "Equity USA": 0.2,
        "Equity Europe": 0.15,
        "Equity Emerging Markets": 0.25,
        "Gold": 0.0
    }

    # Generate skewed scenarios
    scenarios = np.zeros_like(uniform_data)
    for i, name in enumerate(asset_index):
        mu = expret[i]
        var = variance[i]
        df_ = df[i]
        skew = skewness_parameters[name]
        loc_corr, scale_corr = corrected_skewt_params(mu, var, df_, skew)
        # scenarios[:, i] = skew_t_ppf(uniform_data[:, i], df_, loc_corr, scale_corr, skew)
        # Generate skewed data
        x_skewed = skew_t_ppf(uniform_data[:, i], df_, loc_corr, scale_corr, skew)
        
        # Moment-corrected: match exact target mean and std
        emp_mean = np.mean(x_skewed)
        emp_std = np.std(x_skewed)
        target_mean = mu
        target_std = np.sqrt(var)
        
        x_final = target_mean + (x_skewed - emp_mean) * (target_std / emp_std)
            
        scenarios[:, i] = x_final


# pd.DataFrame(scenarios).to_excel(path_ + "\\" + r"market_scenarios.xlsx")


# %% Marginal - Joint Plot

def pairgrid_left_top_scatter_no_diag_no_labels(scenarios, names=None, bins=40, figsize=10, alpha=0.3, s=3):
    """
    Szenario-Plot:
    - Oben: univariate Verteilungen (vertikale Histogramme)
    - Links: univariate Verteilungen (horizontale Histogramme)
    - Mitte: Scatterplots der Paare
    - Keine Achsenbeschriftungen oder Ticks
    """
    X = np.asarray(scenarios)
    T, n = X.shape
    if names is None:
        names = [f"X{i+1}" for i in range(n)]

    fig = plt.figure(figsize=(figsize + 0.5*n, figsize + 0.5*n))
    gs = GridSpec(n+1, n+1, figure=fig, wspace=0.05, hspace=0.05)

    # Gemeinsame Achsenlimits für jede Variable
    xlims = []
    for j in range(n):
        xj = X[:, j]
        lo, hi = np.percentile(xj, [0.2, 99.8])
        pad = 0.05 * (hi - lo + 1e-12)
        xlims.append((lo - pad, hi + pad))

    # Obere Reihe: vertikale Histogramme
    for j in range(n):
        ax = fig.add_subplot(gs[0, j+1])
        ax.hist(X[:, j], bins=bins)
        ax.set_xlim(xlims[j])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(True)

    # Linke Spalte: horizontale Histogramme
    for i in range(n):
        ax = fig.add_subplot(gs[i+1, 0])
        ax.hist(X[:, i], bins=bins, orientation='horizontal')
        ax.set_ylim(xlims[i])
        ax.invert_xaxis()
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(True)
        

    # Innerer Bereich: Scatterplots (ohne Diagonale)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ax = fig.add_subplot(gs[i+1, j+1])
            ax.scatter(X[:, j], X[:, i], alpha=alpha, s=s)
            ax.set_xlim(xlims[j])
            ax.set_ylim(xlims[i])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_frame_on(True)

    # Leere Ecke oben links
    ax00 = fig.add_subplot(gs[0, 0])
    ax00.axis('off')

    plt.show()

# Generate Graph
pairgrid_left_top_scatter_no_diag_no_labels(scenarios, 
                                            names=list(asset_index), 
                                            bins=100, alpha=0.2, s=2)


# %% Graphing: Copula-Marginals
## Graph - Example
# Equity: USA vs Europe
fig,ax = plt.subplots(3,2,figsize=(10,12))
ax = ax.flatten()
ax[0].set_title("Gaussian Grades: \n US Equity vs. Europe Equity")

ax[0].scatter(uniform_data[:,'Equity USA' == asset_index], uniform_data[:,'Equity Europe' == asset_index],
            s=1)

# Equity: USA vs Gold
ax[2].scatter(uniform_data[:,'Equity USA' == asset_index], uniform_data[:,'Gold' == asset_index],
            s=1)
ax[2].set_title("US Equity vs. Gold")
# Governments Bonds: USA vs Europe
ax[4].scatter(uniform_data[:,'Government Bonds USA' == asset_index], uniform_data[:,'Corporate Bonds USA' == asset_index],
            s=1)
ax[4].set_title("US Treasury Bonds vs. US Corporates Bonds")


# Equity: USA vs Europe
ax[1].set_title("Joint Distribution: \n US Equity vs. Europe Equity")

ax[1].scatter(scenarios[:,'Equity USA' == asset_index], scenarios[:,'Equity Europe' == asset_index],
            s=1)

# Equity: USA vs Gold
ax[3].scatter(scenarios[:,'Equity USA' == asset_index], scenarios[:,'Gold' == asset_index],
            s=1)
ax[3].set_title("US Equity vs. Gold")
# Governments Bonds: USA vs Europe
ax[5].scatter(scenarios[:,'Government Bonds USA' == asset_index], scenarios[:,'Corporate Bonds USA' == asset_index],
            s=1)
ax[5].set_title("US Treasury Bonds vs. US Corporates Bonds")



# %% Portfolio Optimization
### Define Benchmark 
bm = pd.Series([0] * len(list(asset_index)), 
                  index = asset_index, 
                  name ="bm_weights")

bm["Government Bonds USA"] = 0.05
bm["Government Bonds EMU"] = 0.15
bm["Corporate Bonds EMU"] = 0.15
bm["Corporate Bonds USA"] = 0.05
bm["Global High Yield Bonds"] = 0.05
bm["Equity USA"] = 0.20
bm["Equity Europe"] = 0.15
bm["Equity Emerging Markets"] = 0.05
bm["Gold"] = 0.15


# Pie Plot of BM
plt.figure(figsize=(8, 8),dpi = 600)
plt.pie(bm, labels=bm.index, autopct='%1.1f%%', startangle=90)
plt.title("Benchmark Composition")

# BM Scenarios
bm_scenarios = scenarios @ bm.values
bm_scenarios = np.atleast_2d(bm_scenarios).T
bm_expret = (expret.reshape(1,-1)@bm)[0]

### Define General Constraints on the Asset Classes
'''
constraints_general = [
    cp.SOC(psi * T**0.5, bm_scenarios - scenarios @ x),
    cp.SOC(1/NEA**0.5, x),   # NEA
    cp.sum(x) == 1,          #
    x >= 0,
    x <= 0.25]
'''

psi = 0.02      # Tracking Erorr Tolerance
alpha = 0.05    # CVaR Level
NEA = 6         # Minimum number of effective assets
tau = 0.010     # Minimum Accepted Return 

### CVAR 
# Define initial parameters
T, n = scenarios.shape

### 1a) CVAR (diffrent Alphas)

# Define Variables
x = cp.Variable((n,1))
t = cp.Variable((1, 1))
u = cp.Variable((T, 1))
# Define risk measure

risk = t + 1/(alpha * T) * cp.sum(u)
# Define risk constraints
constraints = [u >= -scenarios @ x - t,
               u >= 0,
               cp.SOC(psi * T**0.5, bm_scenarios - scenarios @ x),
               cp.SOC(1/NEA**0.5, x),   # NEA
               cp.sum(x) == 1,          #
               x >= 0,
               x <= 0.25,
               bm_expret <= expret.reshape(1,-1) @ x] 

# Solve the problem
objective = cp.Minimize(risk)
prob = cp.Problem(objective, constraints)
prob.solve()

risk.value                  # CVaR (Loss Perspective)
t.value                     # VaR Threshold
np.round(x.value,2)         # portfolio weights

opti_cvar_only = pf_characteristics(x.value,
                                    scenarios,
                                    bm_scenarios,
                                    alpha,
                                    tau,
                                    T)

## Summarize Results
pf_cvar_ = pd.DataFrame(opti_cvar_only[0])
pf_cvar__char = pd.DataFrame([opti_cvar_only[1]])

### 1b)

# Define Variables
x = cp.Variable((n,1))
d = cp.Variable((T,1))

# Define risk measure
risk = cp.sum(d)/(T)
# Define risk constraints
constraints = [d >= (tau - scenarios) @ x,
               d >= 0,
               cp.SOC(psi * T**0.5, bm_scenarios - scenarios @ x),
               cp.SOC(1/NEA**0.5, x),   # NEA
               cp.sum(x) == 1,          #
               x >= 0,
               x <= 0.25,
               bm_expret <= expret.reshape(1,-1) @ x] 

# Solve the problem
objective = cp.Minimize(risk)
prob = cp.Problem(objective, constraints)
prob.solve()

risk.value                  # LPM
np.round(x.value,2)         # portfolio weights

opti_lpm_only = pf_characteristics(x.value,
                                    scenarios,
                                    bm_scenarios,
                                    alpha,
                                    tau,
                                    T)

## Summarize Results
pf_lpm_ = pd.DataFrame(opti_lpm_only[0])
pf_lpm__char = pd.DataFrame([opti_lpm_only[1]])





### 2) CVaR + Lower Partial Moments
'''
Loop over some Weightening Combinations
'''
# Dictionary for Results
cvar_lpm_results = {}
cvar_weights = np.linspace(0,1,9)

## Normalizer - Based on Benchmark
# CVAR_Normalizer
bm_cvar = -np.mean(bm_scenarios[bm_scenarios<np.percentile(scenarios @ bm.values,alpha*100)])

# LPM_Normalizer 
bm_lpm = np.mean(np.maximum(tau - bm_scenarios, 0))

for num,i in enumerate(cvar_weights):
    # Weightening of Risk Measures
    lam_cvar = i
    lam_lpm = (1-i)

    
    # Define Variables
    x = cp.Variable((n,1))
    d = cp.Variable((T,1))
    t = cp.Variable((1, 1))     # CVaR -> VaR Threshhold
    u = cp.Variable((T, 1))     # Counter of Paths Above t
    
    # Define risk measure
    risk = lam_cvar * (t+1/(alpha * T) * cp.sum(u))/bm_cvar + lam_lpm * (cp.sum(d)/(T))/bm_lpm
    
    # Define constraints
    constraints = [u >= -scenarios @ x - t,
                   u >= 0,
                   d >= tau - scenarios @ x,
                   d >= 0,
                   cp.SOC(psi * T**0.5, bm_scenarios - scenarios @ x),
                   cp.SOC(1/NEA**0.5, x),   # NEA
                   cp.sum(x) == 1,          #
                   x >= 0,
                   x <= 0.25,
                   bm_expret <= expret.reshape(1,-1) @ x]
    
    # Solve the problem
    objective = cp.Minimize(risk)
    prob = cp.Problem(objective, constraints)
    prob.solve()

    risk.value                  # Value of Risk Function
    t.value                     # VaR Threshold

    # Generate PF Characteristics
    opti_cvar_lpm = pf_characteristics(x.value,
                                       scenarios,
                                       bm_scenarios,
                                       alpha,
                                       tau,
                                       T)

    # Save PF Characteristics
    cvar_lpm_results[num] = opti_cvar_lpm

## Summarize Results
pf_cvar_lpm = pd.DataFrame()
pf_cvar_lpm_char = pd.DataFrame()
for i in cvar_lpm_results.keys():
    # Weights
    pf_cvar_lpm[i] = cvar_lpm_results[i][0]
    # Char
    pf_cvar_lpm_char[i] = cvar_lpm_results[i][1]
    
    

### 4) CVaR + MAD
'''
Loop over Weightening Combinations
'''
# Dictionary for Results
cvar_mad_results = {}
cvar_weights = np.linspace(0,1,9)

## Normalizer - Based on Benchmark
# MAD_Normalizer 
bm_mad = np.mean(np.abs(bm_scenarios - np.mean(bm_scenarios)))

# Centralizing Matrix
scenario_centered = scenarios - np.mean(scenarios, 
                                        axis=0)

for num,i in enumerate(cvar_weights):
    # Weightening of Risk Measures
    lam_cvar = i
    lam_mad = (1-i)
    # Define Variables
    x = cp.Variable((n,1))
    d = cp.Variable((T,1))
    t = cp.Variable((1, 1))     # CVaR -> VaR Threshhold
    u = cp.Variable((T, 1))     # Counter of Paths Above t
    
    # Define risk measure
    risk = lam_cvar * (t+1/(alpha * T) * cp.sum(u))/bm_cvar + lam_mad * (cp.sum(d)/(T))/bm_mad
    
    # Define constraints
    constraints = [u >= -scenarios @ x - t,
                   u >= 0,
                   # d >= C_T @ scenarios @ x,
                   d >= scenario_centered @ x,
                   d >= -scenario_centered @ x,
                   # d >= -C_T @ scenarios @ x,
                   d >= 0,
                   cp.SOC(psi * T**0.5, bm_scenarios - scenarios @ x),
                   cp.SOC(1/NEA**0.5, x),   # NEA
                   cp.sum(x) == 1,          #
                   x >= 0,
                   x <= 0.25,
                   bm_expret <= expret.reshape(1,-1) @ x] 
    
    # Solve the problem
    objective = cp.Minimize(risk)
    prob = cp.Problem(objective, constraints)
    prob.solve()

    risk.value                  # Value of Risk Function
    t.value                     # VaR Threshold

    # Generate PF Characteristics
    opti_cvar_mad = pf_characteristics(x.value,
                                       scenarios,
                                       bm_scenarios,
                                       alpha,
                                       tau,
                                       T)

    # Save PF Characteristics
    cvar_mad_results[i] = opti_cvar_mad


## Summarize Results
pf_cvar_mad = pd.DataFrame()
pf_cvar_mad_char = pd.DataFrame()
for i in cvar_mad_results.keys():
    # Weights
    pf_cvar_mad[i] = cvar_mad_results[i][0]
    # Char
    pf_cvar_mad_char[i] = cvar_mad_results[i][1]


'''
# Save LPM
with pd.ExcelWriter('cvar_lpm_results.xlsx', engine='openpyxl') as writer:
    pf_cvar_lpm.to_excel(writer, sheet_name='PF', index=True)
    pf_cvar_lpm_char.to_excel(writer, sheet_name='Metrics', index=True)

# Save to one Excel file with two sheets
with pd.ExcelWriter('cvar_mad_results.xlsx', engine='openpyxl') as writer:
    pf_cvar_mad.to_excel(writer, sheet_name='PF', index=True)
    pf_cvar_mad_char.to_excel(writer, sheet_name='Metrics', index=True)
    
'''

