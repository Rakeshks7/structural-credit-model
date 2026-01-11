import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from scipy.optimize import fsolve
from dataclasses import dataclass
import logging
from typing import Tuple, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class CreditRiskOutput:
    ticker: str
    equity_value: float
    equity_vol: float
    debt_barrier: float
    asset_value: float
    asset_vol: float
    distance_to_default: float
    prob_default: float

class MarketDataProvider:
    def __init__(self, ticker: str):
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)

    def get_risk_free_rate(self) -> float:
        try:

            treasury = yf.Ticker("^IRX")
            hist = treasury.history(period="1d")
            if not hist.empty:
                rate = hist['Close'].iloc[-1] / 100
                return rate
            return 0.04
        except Exception as e:
            logger.warning(f"Could not fetch Risk Free Rate, defaulting to 4%: {e}")
            return 0.04

    def get_equity_data(self) -> Tuple[float, float]:
        try:
            fast_info = self.stock.fast_info
            market_cap = fast_info['market_cap']

            hist = self.stock.history(period="1y")
            if hist.empty:
                raise ValueError("No price history found")

            hist['pct_change'] = hist['Close'].pct_change()
            hist['log_ret'] = np.log(hist['Close'] / hist['Close'].shift(1))

            daily_vol = hist['log_ret'].std()
            annualized_vol = daily_vol * np.sqrt(252)

            return market_cap, annualized_vol

        except Exception as e:
            logger.error(f"Error fetching equity data for {self.ticker}: {e}")
            raise

    def get_debt_structure(self) -> float:
        try:
            bs = self.stock.balance_sheet
            q_bs = self.stock.quarterly_balance_sheet

            data = q_bs if not q_bs.empty else bs

            if data.empty:
                raise ValueError("Balance Sheet data unavailable via API.")

            latest = data.iloc[:, 0]

            st_debt = latest.get('Current Debt And Capital Lease Obligation', 0)
            if st_debt == 0:
                 st_debt = latest.get('Current Debt', 0)

            lt_debt = latest.get('Long Term Debt And Capital Lease Obligation', 0)
            if lt_debt == 0:
                lt_debt = latest.get('Long Term Debt', 0)

            if st_debt == 0 and lt_debt == 0:
                total_debt = latest.get('Total Debt', 0)
                st_debt = total_debt * 0.2
                lt_debt = total_debt * 0.8
                logger.warning(f"Detailed debt breakdown missing for {self.ticker}. Using Total Debt heuristic.")

            default_point = st_debt + (0.5 * lt_debt)

            if default_point == 0:
                raise ValueError("Calculated Debt is 0. Model invalid for debt-free firms.")

            return default_point

        except Exception as e:
            logger.error(f"Error fetching debt data for {self.ticker}: {e}")
            raise

class MertonSolver:
    def __init__(self, E: float, sigma_E: float, D: float, r: float, T: float = 1.0):
        self.E = E
        self.sigma_E = sigma_E
        self.D = D
        self.r = r
        self.T = T

    def _d1_d2(self, A, sigma_A):
        d1 = (np.log(A / self.D) + (self.r + 0.5 * sigma_A**2) * self.T) / (sigma_A * np.sqrt(self.T))
        d2 = d1 - sigma_A * np.sqrt(self.T)
        return d1, d2

    def _equations(self, vars):
        A, sigma_A = vars

        if A <= 0 or sigma_A <= 1e-4:
            return [1e10, 1e10]

        d1, d2 = self._d1_d2(A, sigma_A)

        eq1 = A * norm.cdf(d1) - self.D * np.exp(-self.r * self.T) * norm.cdf(d2) - self.E

        eq2 = (A / self.E) * norm.cdf(d1) * sigma_A - self.sigma_E

        return [eq1, eq2]

    def run(self) -> dict:
        x0 = [self.E + self.D, self.sigma_E * (self.E / (self.E + self.D))]

        try:
            A_solved, sigma_A_solved = fsolve(self._equations, x0, xtol=1e-6)
        except RuntimeWarning:
            logger.error("Optimization failed to converge.")
            return None


        numerator = np.log(A_solved / self.D) + (self.r - 0.5 * sigma_A_solved**2) * self.T
        denominator = sigma_A_solved * np.sqrt(self.T)

        distance_to_default = numerator / denominator
        prob_default = norm.cdf(-distance_to_default)

        return {
            "A": A_solved,
            "sigma_A": sigma_A_solved,
            "DD": distance_to_default,
            "PD": prob_default
        }


def analyze_credit_risk(ticker: str) -> Optional[CreditRiskOutput]:
    logger.info(f"Starting Merton Analysis for {ticker}...")

    provider = MarketDataProvider(ticker)

    try:
        E, sigma_E = provider.get_equity_data()
        D = provider.get_debt_structure()
        r = provider.get_risk_free_rate()

        logger.info(f"Data Loaded: Mkt Cap={E/1e9:.2f}B, Vol={sigma_E:.2%}, Barrier={D/1e9:.2f}B, r={r:.2%}")

    except Exception as e:
        logger.error(f"Data ingestion failed: {e}")
        return None

    solver = MertonSolver(E, sigma_E, D, r)
    res = solver.run()

    if not res:
        return None

    return CreditRiskOutput(
        ticker=ticker,
        equity_value=E,
        equity_vol=sigma_E,
        debt_barrier=D,
        asset_value=res['A'],
        asset_vol=res['sigma_A'],
        distance_to_default=res['DD'],
        prob_default=res['PD']
    )


if __name__ == "__main__":
    target_company = "BA"

    result = analyze_credit_risk(target_company)

    if result:
        print("\n" + "="*50)
        print(f"STRUCTURAL CREDIT RISK REPORT: {result.ticker}")
        print("="*50)
        print(f"{'Metric':<25} | {'Value':>15}")
        print("-" * 43)
        print(f"{'Market Cap (Equity)':<25} | ${result.equity_value/1e9:,.2f} B")
        print(f"{'Default Barrier (Debt)':<25} | ${result.debt_barrier/1e9:,.2f} B")
        print(f"{'Risk-Free Rate':<25} | {MarketDataProvider(target_company).get_risk_free_rate():.2%}")
        print("-" * 43)
        print(f"{'Implied Asset Value':<25} | ${result.asset_value/1e9:,.2f} B")
        print(f"{'Implied Asset Vol':<25} | {result.asset_vol:.2%}")
        print(f"{'Equity Volatility':<25} | {result.equity_vol:.2%}")
        print("-" * 43)
        print(f"{'Distance to Default (DD)':<25} | {result.distance_to_default:.4f} σ")
        print(f"{'Prob. of Default (1Y)':<25} | {result.prob_default:.6%}")
        print("="*50)

        if result.prob_default > 0.05:
            print(">>> ALERT: HIGH DEFAULT PROBABILITY DETECTED")
        elif result.distance_to_default < 2.0:
            print(">>> WARNING: DISTANCE TO DEFAULT IS LOW")
        else:
            print(">>> SIGNAL: CREDIT PROFILE APPEARS STABLE")