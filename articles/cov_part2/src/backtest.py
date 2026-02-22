from __future__ import annotations

import json
from math import sqrt
import numpy as np
import pandas as pd

from src.estimators import ledoit_wolf_cov, cov_to_corr
from src.portfolio import min_var_weights

def _month_ends(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    mes = idx.to_series().groupby([idx.year, idx.month]).max().values
    return pd.DatetimeIndex(pd.to_datetime(mes)).sort_values()

def _ewma_vol(daily_window: pd.DataFrame, lam: float = 0.94) -> np.ndarray:
    alpha = 1.0 - lam
    var = daily_window.ewm(alpha=alpha, adjust=False).var().iloc[-1].values
    var = np.maximum(var, 1e-12)
    return np.sqrt(var)

def run_backtest(
    px: pd.DataFrame,
    cap: float = 0.60,
    win_months: int = 36,
    win_days: int = 756,
    scale_days_to_month: int = 21,
    ewma_lambda: float = 0.94,
    transaction_cost_bps: float = 0.0,
):
    tickers = list(px.columns)
    daily_ret = px.pct_change().dropna(how="any")
    month_ends = _month_ends(daily_ret.index)

    px_me = px.loc[month_ends]
    monthly_ret = px_me.pct_change().dropna(how="any")

    # Eligible rebalance dates
    rebals = []
    for d in month_ends:
        if d not in daily_ret.index or d not in px_me.index:
            continue
        idx_me = px_me.index.get_loc(d)
        idx_dr = daily_ret.index.get_loc(d)
        if idx_me < win_months or idx_dr < win_days:
            continue
        midx = month_ends.get_indexer([d])[0]
        if midx < 0 or midx >= len(month_ends) - 1:
            continue
        rebals.append(d)
    rebals = pd.DatetimeIndex(rebals)

    def slice_next_month(d):
        midx = month_ends.get_indexer([d])[0]
        d_next = month_ends[midx + 1]
        r_next = daily_ret.loc[(daily_ret.index > d) & (daily_ret.index <= d_next)]
        return d_next, r_next

    rows = []
    w_prev = {"A": None, "B": None, "C": None}
    W_A, W_B, W_C = {}, {}, {}
    tc = float(transaction_cost_bps) / 10000.0

    for d in rebals:
        idx_me = px_me.index.get_loc(d)
        idx_dr = daily_ret.index.get_loc(d)

        X_m = monthly_ret.iloc[idx_me - win_months: idx_me].values
        covA = ledoit_wolf_cov(X_m)
        wA = min_var_weights(covA, cap=cap)
        PV_A = float(wA @ covA @ wA)

        X_d = daily_ret.iloc[idx_dr - win_days: idx_dr].values
        covD = ledoit_wolf_cov(X_d)
        covB = covD * scale_days_to_month
        wB = min_var_weights(covB, cap=cap)
        PV_B = float(wB @ covB @ wB)

        vols_d = _ewma_vol(daily_ret.iloc[idx_dr - win_days: idx_dr], lam=ewma_lambda)
        vols_m = vols_d * sqrt(scale_days_to_month)
        corr_m = cov_to_corr(covA)
        covC = np.diag(vols_m) @ corr_m @ np.diag(vols_m)
        wC = min_var_weights(covC, cap=cap)
        PV_C = float(wC @ covC @ wC)

        d_next, r_next = slice_next_month(d)
        rpA = r_next.values @ wA
        rpB = r_next.values @ wB
        rpC = r_next.values @ wC
        RV_A, RV_B, RV_C = float(np.sum(rpA**2)), float(np.sum(rpB**2)), float(np.sum(rpC**2))
        Rm_A = float(np.prod(1.0 + rpA) - 1.0)
        Rm_B = float(np.prod(1.0 + rpB) - 1.0)
        Rm_C = float(np.prod(1.0 + rpC) - 1.0)

        def ratio(rv, pv):
            return max(rv, 1e-12) / max(pv, 1e-12)

        R_A, R_B, R_C = ratio(RV_A, PV_A), ratio(RV_B, PV_B), ratio(RV_C, PV_C)

        def ale(r): return float(abs(np.log(r)))
        aleA, aleB, aleC = ale(R_A), ale(R_B), ale(R_C)

        def turnover(w, wprev):
            if wprev is None:
                return np.nan
            return float(0.5*np.sum(np.abs(w - wprev)))

        toA, toB, toC = turnover(wA, w_prev["A"]), turnover(wB, w_prev["B"]), turnover(wC, w_prev["C"])
        w_prev["A"], w_prev["B"], w_prev["C"] = wA, wB, wC
        toA_f = 0.0 if np.isnan(toA) else toA
        toB_f = 0.0 if np.isnan(toB) else toB
        toC_f = 0.0 if np.isnan(toC) else toC

        Rm_A_net = Rm_A - tc * toA_f
        Rm_B_net = Rm_B - tc * toB_f
        Rm_C_net = Rm_C - tc * toC_f

        w_ew = np.full(len(tickers), 1.0 / len(tickers))
        rp_ew = r_next.values @ w_ew
        Rm_EW = float(np.prod(1.0 + rp_ew) - 1.0)

        if "SPY" in tickers and "AGG" in tickers:
            w_6040 = np.zeros(len(tickers))
            w_6040[tickers.index("SPY")] = 0.60
            w_6040[tickers.index("AGG")] = 0.40
            rp_6040 = r_next.values @ w_6040
            Rm_6040 = float(np.prod(1.0 + rp_6040) - 1.0)
        else:
            Rm_6040 = float("nan")

        W_A[d] = wA
        W_B[d] = wB
        W_C[d] = wC

        row = {
            "rebalance_date": d,
            "eval_month_end": d_next,
            "PV_A": PV_A, "RV_A": RV_A, "RV_over_PV_A": R_A, "abs_log_err_A": aleA, "turnover_A": toA,
            "PV_B": PV_B, "RV_B": RV_B, "RV_over_PV_B": R_B, "abs_log_err_B": aleB, "turnover_B": toB,
            "PV_C": PV_C, "RV_C": RV_C, "RV_over_PV_C": R_C, "abs_log_err_C": aleC, "turnover_C": toC,
            "ret_month_A_gross": Rm_A,
            "ret_month_B_gross": Rm_B,
            "ret_month_C_gross": Rm_C,
            "ret_month_A_net": Rm_A_net,
            "ret_month_B_net": Rm_B_net,
            "ret_month_C_net": Rm_C_net,
            "ret_month_EW": Rm_EW,
            "ret_month_60_40": Rm_6040,
        }
        for i,t in enumerate(tickers):
            row[f"wA_{t}"] = wA[i]
            row[f"wB_{t}"] = wB[i]
            row[f"wC_{t}"] = wC[i]
        rows.append(row)

    bt = pd.DataFrame(rows).set_index("rebalance_date")

    avgA = np.mean(np.vstack([W_A[d] for d in bt.index]), axis=0)
    avgB = np.mean(np.vstack([W_B[d] for d in bt.index]), axis=0)
    avgC = np.mean(np.vstack([W_C[d] for d in bt.index]), axis=0)
    avgw = pd.DataFrame({"A_monthly": avgA, "B_dailyx21": avgB, "C_mixed": avgC}, index=tickers)

   
    # --- Pairwise L1 disagreement ---
    l1_ab = pd.Series([float(np.sum(np.abs(W_A[d] - W_B[d]))) for d in bt.index],
                      index=bt.index, name="L1_A_vs_B")
    
    l1_ac = pd.Series([float(np.sum(np.abs(W_A[d] - W_C[d]))) for d in bt.index],
                      index=bt.index, name="L1_A_vs_C")
    
    l1_bc = pd.Series([float(np.sum(np.abs(W_B[d] - W_C[d]))) for d in bt.index],
                      index=bt.index, name="L1_B_vs_C")


    def summarize(prefix):
        ale = bt[f"abs_log_err_{prefix}"].dropna()
        rr = bt[f"RV_over_PV_{prefix}"].dropna()
        to = bt[f"turnover_{prefix}"].dropna()
        return {
            "MALE": float(ale.mean()),
            "Median_RV_over_PV": float(rr.median()),
            "Mean_RV_over_PV": float(rr.mean()),
            "Mean_Turnover": float(to.mean()) if len(to) else float("nan"),
        }

    summary = {
        "data": {
            "first_date": str(px.index.min().date()),
            "last_date": str(px.index.max().date()),
            "n_daily_obs": int(len(daily_ret)),
            "n_rebalances": int(len(bt)),
            "tickers": tickers,
        },
        "setups": {
            "A_monthly": summarize("A"),
            "B_daily_scaled": summarize("B"),
            "C_mixed": summarize("C"),
        },
        "assumptions": {
            "transaction_cost_bps_per_unit_turnover": float(transaction_cost_bps),
        },
    }

    def _perf_stats(r: pd.Series) -> dict:
        r = r.dropna()
        if len(r) == 0:
            return {
                "Annual_Return": float("nan"),
                "Annual_Volatility": float("nan"),
                "Sharpe_0rf": float("nan"),
                "Downside_Deviation_Ann": float("nan"),
                "Max_Drawdown": float("nan"),
                "Calmar": float("nan"),
            }
        ann_ret = float((1.0 + r).prod() ** (12.0 / len(r)) - 1.0)
        ann_vol = float(r.std(ddof=1) * sqrt(12.0))
        sharpe = ann_ret / ann_vol if ann_vol > 0 else float("nan")
        down = r[r < 0]
        downside = float(down.std(ddof=1) * sqrt(12.0)) if len(down) > 1 else 0.0
        wealth = (1.0 + r).cumprod()
        peak = wealth.cummax()
        dd = wealth / peak - 1.0
        max_dd = float(dd.min()) if len(dd) else float("nan")
        calmar = ann_ret / abs(max_dd) if (not np.isnan(max_dd) and max_dd < 0) else float("nan")
        return {
            "Annual_Return": ann_ret,
            "Annual_Volatility": ann_vol,
            "Sharpe_0rf": sharpe,
            "Downside_Deviation_Ann": downside,
            "Max_Drawdown": max_dd,
            "Calmar": calmar,
        }

    perf_cols = {
        "A_gross": "ret_month_A_gross",
        "B_gross": "ret_month_B_gross",
        "C_gross": "ret_month_C_gross",
        "A_net": "ret_month_A_net",
        "B_net": "ret_month_B_net",
        "C_net": "ret_month_C_net",
        "EqualWeight": "ret_month_EW",
        "60_40_SPY_AGG": "ret_month_60_40",
    }
    perf = pd.DataFrame({k: _perf_stats(bt[v]) for k, v in perf_cols.items()}).T

    summary["economic"] = {
        k: {m: float(perf.loc[k, m]) for m in perf.columns}
        for k in perf.index
    }
    summary_json = json.dumps(summary, indent=2)
    return bt, summary_json, avgw, l1_ab, l1_ac, l1_bc, perf


def pf_volatility_evaluation_by_estimator(
    px: pd.DataFrame,
    equity: str = "SPY",
    bond: str = "AGG",
    w_equity: float = 0.60,
    w_bond: float = 0.40,
    win_months: int = 36,
    win_days: int = 756,
    scale_days_to_month: int = 21,
    ewma_lambda: float = 0.94,
) -> pd.DataFrame:
    """
    Evaluate the three covariance approaches (A/B/C) on a fixed portfolio.

    The fixed portfolio defaults to 60/40 equity-bond (SPY/AGG), so only
    covariance estimation differs across A/B/C.
    """
    tickers = list(px.columns)
    if equity not in tickers or bond not in tickers:
        raise ValueError(f"Need both {equity} and {bond} in px columns.")
    if w_equity < 0 or w_bond < 0:
        raise ValueError("Weights must be non-negative.")
    if abs((w_equity + w_bond) - 1.0) > 1e-12:
        raise ValueError("w_equity + w_bond must sum to 1.")

    w_fix = np.zeros(len(tickers))
    w_fix[tickers.index(equity)] = float(w_equity)
    w_fix[tickers.index(bond)] = float(w_bond)

    daily_ret = px.pct_change().dropna(how="any")
    month_ends = _month_ends(daily_ret.index)
    px_me = px.loc[month_ends]
    monthly_ret = px_me.pct_change().dropna(how="any")

    rebals = []
    for d in month_ends:
        if d not in daily_ret.index or d not in px_me.index:
            continue
        idx_me = px_me.index.get_loc(d)
        idx_dr = daily_ret.index.get_loc(d)
        if idx_me < win_months or idx_dr < win_days:
            continue
        midx = month_ends.get_indexer([d])[0]
        if midx < 0 or midx >= len(month_ends) - 1:
            continue
        rebals.append(d)
    rebals = pd.DatetimeIndex(rebals)

    def slice_next_month(d):
        midx = month_ends.get_indexer([d])[0]
        d_next = month_ends[midx + 1]
        r_next = daily_ret.loc[(daily_ret.index > d) & (daily_ret.index <= d_next)]
        return d_next, r_next

    rows = []
    for d in rebals:
        idx_me = px_me.index.get_loc(d)
        idx_dr = daily_ret.index.get_loc(d)

        X_m = monthly_ret.iloc[idx_me - win_months: idx_me].values
        covA = ledoit_wolf_cov(X_m)

        X_d = daily_ret.iloc[idx_dr - win_days: idx_dr].values
        covD = ledoit_wolf_cov(X_d)
        covB = covD * scale_days_to_month

        vols_d = _ewma_vol(daily_ret.iloc[idx_dr - win_days: idx_dr], lam=ewma_lambda)
        vols_m = vols_d * sqrt(scale_days_to_month)
        corr_m = cov_to_corr(covA)
        covC = np.diag(vols_m) @ corr_m @ np.diag(vols_m)

        PV_A = float(w_fix @ covA @ w_fix)
        PV_B = float(w_fix @ covB @ w_fix)
        PV_C = float(w_fix @ covC @ w_fix)

        d_next, r_next = slice_next_month(d)
        rp = r_next.values @ w_fix
        RV = float(np.sum(rp ** 2))

        def ratio(rv, pv):
            return max(rv, 1e-12) / max(pv, 1e-12)

        R_A = ratio(RV, PV_A)
        R_B = ratio(RV, PV_B)
        R_C = ratio(RV, PV_C)

        rows.append(
            {
                "rebalance_date": d,
                "eval_month_end": d_next,
                "RV_fixed_6040": RV,
                "RV_vol_fixed_6040": float(np.sqrt(max(RV, 1e-12))),
                "PV_A": PV_A,
                "PV_B": PV_B,
                "PV_C": PV_C,
                "PV_vol_A": float(np.sqrt(max(PV_A, 1e-12))),
                "PV_vol_B": float(np.sqrt(max(PV_B, 1e-12))),
                "PV_vol_C": float(np.sqrt(max(PV_C, 1e-12))),
                "RV_over_PV_A": R_A,
                "RV_over_PV_B": R_B,
                "RV_over_PV_C": R_C,
                "abs_log_err_A": float(abs(np.log(R_A))),
                "abs_log_err_B": float(abs(np.log(R_B))),
                "abs_log_err_C": float(abs(np.log(R_C))),
            }
        )

    if len(rows) == 0:
        return pd.DataFrame(
            columns=[
                "eval_month_end",
                "RV_fixed_6040",
                "RV_vol_fixed_6040",
                "PV_A",
                "PV_B",
                "PV_C",
                "PV_vol_A",
                "PV_vol_B",
                "PV_vol_C",
                "RV_over_PV_A",
                "RV_over_PV_B",
                "RV_over_PV_C",
                "abs_log_err_A",
                "abs_log_err_B",
                "abs_log_err_C",
            ]
        )

    return pd.DataFrame(rows).set_index("rebalance_date")
