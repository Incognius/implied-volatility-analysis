import sys, os, numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scipy.optimize import brentq
from pricing_models.bsm import bsm_price

def bsm_bounds_call_put(S, K, r, q, T, option):
    disc_S = S * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)
    if option == "call":
        lower = max(S - K, 0.0) if T <= 1e-8 else max(disc_S - disc_K, 0.0)
        upper = disc_S
    else:
        lower = max(K - S, 0.0) if T <= 1e-8 else max(disc_K - disc_S, 0.0)
        upper = disc_K
    return lower, upper

def implied_vol_brent(
    market_price: float,
    S: float, K: float,
    r: float, q: float,
    T: float,
    option: str = "call",
    lo: float = 1e-8, hi: float = 5.0,
    tol: float = 1e-8, maxiter: int = 200,
    strict_bounds: bool = False
) -> float:
    option = option.lower()
    if option not in {"call", "put"}:
        return np.nan
    lb, ub = bsm_bounds_call_put(S, K, r, q, T, option)
    if strict_bounds:
        if not (lb <= market_price <= ub):
            return np.nan
    else:
        market_price = float(np.clip(market_price, lb + 1e-12, ub - 1e-12))
    def f(sig):
        sig = max(sig, 1e-12)
        return bsm_price(S, K, r, q, sig, T, option) - market_price
    f_lo, f_hi = f(lo), f(hi)
    tries, hi_cap = 0, 50.0
    while f_lo * f_hi > 0 and hi < hi_cap and tries < 8:
        hi *= 2.0
        f_hi = f(hi)
        tries += 1
    if f_lo * f_hi > 0:
        return np.nan
    return brentq(f, lo, hi, xtol=tol, maxiter=maxiter)
