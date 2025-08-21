import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
csv_path = project_root / "Phase0" / "data" / "Priced_Options_IV.csv"
df_long = pd.read_csv(csv_path)

# Separate calls and puts
calls = df_long[df_long["type"] == "call"].copy()
puts = df_long[df_long["type"] == "put"].copy()

fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)

# --- Plot 1: Market vs BSM (from IV)
axes[0].scatter(df_long["strike"], df_long["ltp"], label="Market LTP")
axes[0].scatter(df_long["strike"], df_long["bsm_theo_from_iv"], label="BSM Theo (from IV)")
axes[0].axvline(df_long["spot"].iloc[0], linestyle="--", color="black", label="ATM")
axes[0].set_ylabel("Option Price")
axes[0].set_title("Market vs BSM (from IV)")
axes[0].legend()
axes[0].grid(alpha=0.3)

# --- Plot 2: Distortion Across Strikes
axes[1].plot(df_long["strike"], df_long["iv_fit_err"], marker="o", label="Fit Error")
axes[1].axhline(0, linestyle="--", color="black")
axes[1].axvline(df_long["spot"].iloc[0], linestyle="--", color="black", label="ATM")
axes[1].set_ylabel("Fit Error")
axes[1].set_title("Distortion Across Strikes")
axes[1].legend()
axes[1].grid(alpha=0.3)

# --- Plot 3: Implied Volatility Smile (Calls vs Puts)
axes[2].plot(calls["strike"], calls["iv_bsm"], marker="o", label="Call IV", color="blue")
axes[2].plot(puts["strike"], puts["iv_bsm"], marker="o", label="Put IV", color="red")
axes[2].axvline(df_long["spot"].iloc[0], linestyle="--", color="black", label="ATM")
axes[2].set_xlabel("Strike Price")
axes[2].set_ylabel("Implied Volatility")
axes[2].set_title("Implied Volatility Smile - Calls vs Puts")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()
