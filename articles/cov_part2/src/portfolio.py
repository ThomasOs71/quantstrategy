from __future__ import annotations
import numpy as np
import scipy.optimize as opt

def min_var_weights(cov: np.ndarray, cap: float = 0.60) -> np.ndarray:
    n = cov.shape[0]
    cov = 0.5*(cov + cov.T) + np.eye(n)*1e-10

    # Minimum Number of Effective Assets to ensure Diversification
    min_eff = n/3

    def obj(w): 
        return float(w @ cov @ w)

    bounds = [(0.0, cap) for _ in range(n)]
    cons = [{"type":"eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "ineq", "fun": lambda w, K=min_eff: (1.0 / K) - np.dot(w,w)}]
    w0 = np.full(n, 1.0/n)

    res = opt.minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                       options={"maxiter": 200, "ftol": 1e-12})
    if (not res.success) or np.any(np.isnan(res.x)):
        return w0

    w = np.clip(res.x, 0.0, cap)
    w = w / w.sum()
    return w
