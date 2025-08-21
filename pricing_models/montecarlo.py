from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike
from typing import Optional, Tuple, Union

def mc_price(
    S: ArrayLike,
    K: ArrayLike,
    r: float,
    q: float,
    sigma: ArrayLike,
    T: ArrayLike,
    option: str = "call",
    *,
    num_paths: int = 200_000,
    antithetic: bool = True,
    seed: Optional[int] = None,
    return_std: bool = False,
) -> Union[float, np.ndarray, Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]]:
    option = option.lower()
    if option not in {"call", "put"}:
        raise ValueError("Option type must be 'call' or 'put'.")
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    S, K, sigma, T = np.broadcast_arrays(S, K, sigma, T)
    out_shape = S.shape
    scalar_output = out_shape == ()
    S_flat = S.ravel()[np.newaxis, :]
    K_flat = K.ravel()[np.newaxis, :]
    sigma_flat = sigma.ravel()[np.newaxis, :]
    T_flat = T.ravel()[np.newaxis, :]
    rng = np.random.default_rng(seed)
    if antithetic:
        half = (num_paths + 1) // 2
        Z_half = rng.standard_normal(size=(half, 1))
        Z_full = np.vstack((Z_half, -Z_half))
        Z = Z_full[:num_paths, :]
    else:
        Z = rng.standard_normal(size=(num_paths, 1))
    sqrtT = np.sqrt(np.maximum(T_flat, 0.0))
    drift = (r - q - 0.5 * sigma_flat**2) * T_flat
    diffusion = sigma_flat * sqrtT
    exponent = drift + diffusion * Z
    S_T = S_flat * np.exp(exponent)
    if option == "call":
        payoff = np.maximum(S_T - K_flat, 0.0)
    else:
        payoff = np.maximum(K_flat - S_T, 0.0)
    discounted = np.exp(-r * T_flat) * payoff
    mc_mean = discounted.mean(axis=0)
    mc_std_err = discounted.std(axis=0, ddof=0) / np.sqrt(discounted.shape[0])
    price = mc_mean.reshape(out_shape)
    stderr = mc_std_err.reshape(out_shape)
    intrinsic_now = np.where(option == "call", np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    if np.any(T <= 0):
        mask_T0 = (T <= 0)
        price[mask_T0] = intrinsic_now[mask_T0]
        stderr[mask_T0] = 0.0
    zero_sigma_mask = (sigma <= 0) & (T > 0)
    if np.any(zero_sigma_mask):
        disc_S = S * np.exp(-q * T)
        disc_K = K * np.exp(-r * T)
        intrinsic_forward = np.where(option == "call", np.maximum(disc_S - disc_K, 0.0), np.maximum(disc_K - disc_S, 0.0))
        price[zero_sigma_mask] = intrinsic_forward[zero_sigma_mask]
        stderr[zero_sigma_mask] = 0.0
    if scalar_output:
        price = float(price)
        stderr = float(stderr)
    return (price, stderr) if return_std else price
