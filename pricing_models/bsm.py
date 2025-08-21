from __future__ import annotations
import numpy as np
from scipy.stats import norm
from numpy.typing import ArrayLike

def bsm_price(
    S: ArrayLike,
    K: ArrayLike,
    r: float,
    q: float,
    sigma: ArrayLike,
    T: ArrayLike,
    option: str = "call",
):
    option = option.lower()
    if option not in {"call", "put"}:
        raise ValueError("Option type must be 'call' or 'put'.")
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    S, K, sigma, T = np.broadcast_arrays(S, K, sigma, T)
    intrinsic_now = np.where(option == "call", np.maximum(S - K, 0), np.maximum(K - S, 0))
    if np.any(T <= 0):
        mask_T0 = (T <= 0)
        out = np.empty_like(S, dtype=float)
        out[mask_T0] = intrinsic_now[mask_T0]
    else:
        out = None
    disc_S = S * np.exp(-q * T)
    disc_K = K * np.exp(-r * T)
    intrinsic_forward = np.where(option == "call", np.maximum(disc_S - disc_K, 0), np.maximum(disc_K - disc_S, 0))
    eps = 1e-10
    with np.errstate(divide="ignore", invalid="ignore"):
        d1 = (np.log((S + eps) / (K + eps)) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T + eps))
        d2 = d1 - sigma * np.sqrt(T + eps)
    if option == "call":
        price = disc_S * norm.cdf(d1) - disc_K * norm.cdf(d2)
    else:
        price = disc_K * norm.cdf(-d2) - disc_S * norm.cdf(-d1)
    if out is None:
        out = price
    else:
        out[~(T <= 0)] = price
    zero_sigma_mask = (sigma <= 0) & (T > 0)
    if np.any(zero_sigma_mask):
        out[zero_sigma_mask] = intrinsic_forward[zero_sigma_mask]
    return out.item() if out.shape == () else out
