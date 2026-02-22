from __future__ import annotations
import numpy as np

try:
    from sklearn.covariance import LedoitWolf
    _SKLEARN_OK = True
except Exception:
    LedoitWolf = None
    _SKLEARN_OK = False

def ledoit_wolf_cov(X: np.ndarray) -> np.ndarray:
    if _SKLEARN_OK:
        return LedoitWolf().fit(X).covariance_
    return np.cov(X, rowvar=False)

def cov_to_corr(cov: np.ndarray) -> np.ndarray:
    cov = 0.5 * (cov + cov.T)
    std = np.sqrt(np.diag(cov))
    std = np.where(std <= 0, 1.0, std)
    corr = cov / np.outer(std, std)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr
