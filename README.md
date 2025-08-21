# Option Pricing & Implied Volatility Analysis

This project implements and compares multiple option pricing models, and then uses **Brent’s method** to reverse-engineer implied volatility (IV) from market prices.  
The focus is on both **computational methods** (vectorization, variance reduction) and **practical insights** (why theoretical vs market prices differ, how IV surfaces behave).

---

## Features

- **Pricing Models**
  - **Black–Scholes–Merton (BSM)**  
    - Closed-form analytical model under risk-neutral assumptions.  
    - Implemented with vectorization for efficient batch pricing.  

  - **Binomial Tree (CRR)**  
    - Cox-Ross-Rubinstein (CRR) tree formulation.  
    - Handles American-style extensions (though here used for European).  
    - Prices converge toward BSM as steps increase.

  - **Monte Carlo Simulation**  
    - Risk-neutral path simulation for European calls/puts.  
    - Supports **antithetic variates** for variance reduction.  
    - Fully vectorized to handle thousands of contracts simultaneously.

- **Implied Volatility (IV)**
  - **Brent’s Method Root-Finding**  
    - Robust numerical solver to back out the volatility that matches a given market option price under BSM.  
    - Handles edge cases with no-arbitrage bounds and expands search region if necessary.  

- **Helpers**
  - Option chain CSV parsing into a **long-format DataFrame** (strike, type, LTP, expiry, spot).  
  - Row-wise application of pricing models and IV estimation.  
  - Automated error metrics: theoretical vs observed LTP, IV misfit.

- **Plotting & Analysis**
  - Compare **market LTP vs model predictions**.  
  - Visualize **implied volatility surface distortions** across strikes.  
  - Highlight why ATM options tend to align closely with theory, while ITM/OTM deviate.

---

##  Key Insights:

1. **Model Agreement**  
   - BSM, Binomial (CRR), and Monte Carlo give nearly identical results for European options when parameters are consistent.  
   - Minor differences arise from discretization (Binomial) or finite sampling (MC).

2. **Variance Reduction (Antithetic Variates)**  
   - Using paired random draws (`Z` and `-Z`) in Monte Carlo dramatically stabilizes estimates.  
   - Intuitively, this balances “lucky” and “unlucky” paths, reducing noise without more simulations.

3. **Implied Volatility Extraction**  
   - Brent’s method reliably recovers IV from market prices in the ATM region.  
   - For ITM/OTM strikes, recovered IVs flatten unnaturally because:  
     - Market frictions (bid–ask spread, low liquidity).  
     - Option prices near intrinsic value leave little sensitivity to volatility.  
     - Deep ITM/OTM options carry less information about volatility.

4. **Why Prices Differ from Market LTP**  
   - **Theta decay** (time value erosion).  
   - **Market factors** (supply/demand, hedging pressure, margin rules).  
   - **Simplifying assumptions** in BSM (constant volatility, lognormal returns, no transaction costs).

---

## Tech Notes

- Fully vectorized `numpy` implementation.  
- Modular design: each pricing model isolated in `pricing_models/`.  
- IV solver in `iv_calculation/`.  
- Data prep in `helpers.py`.  
- Analysis/visualization in `plotting.py`.

---

## Analysis  

- **At-the-Money (ATM) Options**  
  - The Black–Scholes model prices are most accurate for ATM options because the **vega** (sensitivity of option price to volatility) is highest at the money.  
  - ATM prices are highly responsive to volatility, so the implied volatility (IV) backed out from market prices is **stable and well-defined**.  
  - Market participants also trade ATM contracts most actively for hedging, which tightens bid–ask spreads and aligns observed prices with theoretical values.  

- **In-the-Money (ITM) and Out-of-the-Money (OTM) Options**  
  - Away from ATM, **vega decays quickly**:  
    - ITM options behave more like stock (delta → 1), so their extrinsic value is small and IV estimates become unstable.  
    - OTM options approach zero value, so small price ticks cause huge swings in implied volatility.  
  - This makes IV surfaces look **flat or noisy in the wings**, even when market dynamics imply otherwise.  

- **Asymmetry Between Calls and Puts**  
  - **OTM puts** distort more because they are widely used for crash-hedging, creating **skew** and demand-driven distortions. They also suffer from wider spreads and lower liquidity.  
  - **ITM calls** distort because their price is mostly intrinsic (stock-like), leaving little extrinsic value to back out a meaningful IV.  
  - This explains why in your plots, puts wiggle at OTM strikes while calls wiggle at ITM strikes.  

- **Why Distortions Appear**  
  - The BSM model assumes lognormal returns and constant volatility, which fails to capture the **smile/skew** observed in real markets.  
  - For ATM, this simplification is “good enough.”  
  - For ITM/OTM, structural skew and low vega make IVs appear **flattened or distorted**, even though market pricing reflects risk premia and demand imbalances.  

- **Takeaway**  
  - **ATM IVs are reliable** because they are volatility-sensitive and liquid.  
  - **OTM puts distort** due to skew, crash-hedging demand, and low liquidity.  
  - **ITM calls distort** because they behave stock-like, with little extrinsic value.  


This project demonstrates not only the **theory of option pricing**, but also the **practical challenges** when mapping theory onto real market data.
