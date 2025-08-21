from pathlib import Path
import numpy as np
from utils.helpers import load_option_chain_long
from pricing_models.bsm import bsm_price
from pricing_models.binomial import binomial_price
from pricing_models.montecarlo import mc_price
import pandas as pd
from iv_calculation.implied_vol import implied_vol_brent

project_root = Path(__file__).resolve().parents[1]
csv_path = project_root / "Phase0" / "data" / "NIFTY_2025-08-21_option_chain_2025-08-15-05-58-13.csv"
df_long = load_option_chain_long(str(csv_path), spot=24631)

r = 0.067
q = 1.2
sigma = 0.2
T = 6/365

def price_row_bsm(row):
    return bsm_price(S=row["spot"], K=row["strike"], r=r, q=q, sigma=sigma, T=T, option=row["type"])

df_long["bsm_theo"] = df_long.apply(price_row_bsm, axis=1)
df_long["bsm_edge"] = df_long["ltp"] - df_long["bsm_theo"]

def price_row_binom(row):
    return binomial_price(S=row["spot"], K=row["strike"], r=r, q=q, sigma=sigma, T=T, N=500, option=row["type"], style="european")

df_long["binom_theo"] = df_long.apply(price_row_binom, axis=1)
df_long["binom_edge"] = df_long["ltp"] - df_long["binom_theo"]

def price_row_mc(row):
    return mc_price(S=row["spot"], K=row["strike"], r=r, q=q, sigma=sigma, T=T, option=row["type"], num_paths=100_000, antithetic=True, seed=123)

df_long["mc_theo"] = df_long.apply(price_row_mc, axis=1)
df_long["mc_edge"] = df_long["ltp"] - df_long["mc_theo"]

r_default = 0.067
q_default = 0.012
T_default = 6/365

def make_row_iv_bsm(r_fallback: float, q_fallback: float, T_fallback: float):
    def _row_iv(row):
        r = float(row.get("r", r_fallback))
        q = float(row.get("q", q_fallback))
        T = float(row.get("T", T_fallback))
        return implied_vol_brent(market_price=row["ltp"], S=row["spot"], K=row["strike"], r=r, q=q, T=T, option=row["type"], strict_bounds=False)
    return _row_iv

df_long["iv_bsm"] = df_long.apply(make_row_iv_bsm(r_default, q_default, T_default), axis=1)
df_long["bsm_theo_from_iv"] = df_long.apply(lambda row: bsm_price(S=row["spot"], K=row["strike"], r=r_default, q=q_default, sigma=row["iv_bsm"], T=T_default, option=row["type"]), axis=1)
df_long["iv_fit_err"] = df_long["bsm_theo_from_iv"] - df_long["ltp"]

df_long.to_csv(project_root / "Phase0" / "data" / "Priced_Options_IV.csv", index=False)

print(df_long[76:90])