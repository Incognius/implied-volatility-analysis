from __future__ import annotations
import numpy as np
from numpy.typing import ArrayLike

def binomial_price(
    S: ArrayLike,
    K: ArrayLike,
    r: float,
    q: float,
    sigma: ArrayLike,
    T: ArrayLike,
    N: int = 200,
    option: str = "call",
    style: str = "european",
    u: float | None = None,
    d: float | None = None,
    p: float | None = None,
):
    option = option.lower()
    if option not in {"call", "put"}:
        raise ValueError("option must be 'call' or 'put'.")
    style = style.lower()
    if style not in {"european", "american"}:
        raise ValueError("style must be 'european' or 'american'.")
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    S, K, sigma, T = np.broadcast_arrays(S, K, sigma, T)
    intrinsic_now = np.where(option == "call", np.maximum(S - K, 0.0), np.maximum(K - S, 0.0))
    out = np.empty_like(S, dtype=np.float64)
    mask_live = T > 0
    out[~mask_live] = intrinsic_now[~mask_live]
    if not np.any(mask_live):
        return out.item() if out.shape == () else out
    idxs = np.argwhere(mask_live)
    idxs = [tuple(i) for i in idxs]
    for ix in idxs:
        Si, Ki, sigi, Ti = float(S[ix]), float(K[ix]), float(sigma[ix]), float(T[ix])
        if N <= 0 or Ti <= 0:
            out[ix] = np.maximum(Si - Ki, 0.0) if option == "call" else np.maximum(Ki - Si, 0.0)
            continue
        dt = Ti / N
        disc = np.exp(-r * dt)
        if u is None or d is None or p is None:
            if sigi <= 0:
                fwd_ST = Si * np.exp((r - q) * Ti)
                if option == "call":
                    out[ix] = disc**N * max(fwd_ST - Ki, 0.0)
                else:
                    out[ix] = disc**N * max(Ki - fwd_ST, 0.0)
                continue
            ui = np.exp(sigi * np.sqrt(dt))
            di = 1.0 / ui
            pu = (np.exp((r - q) * dt) - di) / (ui - di)
        else:
            ui, di, pu = float(u), float(d), float(p)
        pu = min(max(pu, 0.0), 1.0)
        j = np.arange(N + 1, dtype=np.float64)
        S_term = Si * (ui**j) * (di ** (N - j))
        if option == "call":
            values = np.maximum(S_term - Ki, 0.0)
        else:
            values = np.maximum(Ki - S_term, 0.0)
        for k in range(N, 0, -1):
            values = disc * (pu * values[1:] + (1.0 - pu) * values[:-1])
            if style == "american":
                j_k = np.arange(k, dtype=np.float64)
                S_k = Si * (ui**j_k) * (di ** (k - 1 - j_k))
                if option == "call":
                    exercise = np.maximum(S_k - Ki, 0.0)
                else:
                    exercise = np.maximum(Ki - S_k, 0.0)
                values = np.maximum(values, exercise)
        out[ix] = values[0]
    return out.item() if out.shape == () else out
