# Structural Default Prediction Engine (Merton Model)

## 📌 Project Overview
This project implements a **Structural Credit Risk Model** based on the Merton (1974) framework. Unlike traditional accounting-based models (Altman Z-Score), this model treats a company's Equity as a **Call Option on its Assets** (strike price = debt liabilities).

By reverse-engineering the Black-Scholes-Merton formula, this engine solves for unobservable **Asset Value** and **Asset Volatility** to calculate a market-implied **Distance to Default (DD)** and **Probability of Default (PD)**.

## 🚀 Key Features
- **Reverse Engineering:** Solves the non-linear system of equations linking Equity Volatility to Asset Volatility.
- **Dynamic Debt Barriers:** Automatically parses Balance Sheets to construct the KMV-style default barrier ($Short Term Debt + 0.5 \times Long Term Debt$).
- **Automated Data Pipeline:** Fetches real-time equity data and risk-free rates (Treasury Yields) via `yfinance`.
- **Production-Grade Code:** Includes robust error handling, logging, and modular architecture.

## 🛠️ Tech Stack
- **Language:** Python 3.x
- **Libraries:** `scipy.optimize` (fsolve), `numpy` (Numerical integration), `pandas`, `yfinance`
- **Math:** Stochastic Calculus (Ito's Lemma), Black-Scholes Pricing Model

## 🧮 The Mathematics
The model solves the following system simultaneously:

$$E = V_A N(d_1) - D e^{-rT} N(d_2)$$
$$\sigma_E = \frac{V_A}{E} N(d_1) \sigma_A$$

Where:
- $E$: Market Value of Equity (Observable)
- $\sigma_E$: Equity Volatility (Observable)
- $D$: Debt Face Value (Balance Sheet)
- $V_A$: Asset Value (Solved for)
- $\sigma_A$: Asset Volatility (Solved for)

## 📜 Disclaimer
This tool is for educational and research purposes only. It does not constitute financial advice.