"""
Portfolio Simulator - Refactored UI Entry Point

Author: Mohit Saharan
Email: mohit@msaharan.com
"""

from src.portfolio_simulator.ui import PortfolioDashboard

# Configuration constants from original file
DEFAULT_TICKERS = ['IWDA.AS', 'QDV5.DE', 'PPFB.DE', 'XEON.DE']

ISIN_TO_TICKER = {
    'IE00B4L5Y983': 'IWDA.AS',
    'IE00BHZRQZ17': 'FLXI.DE',
    'IE00B4ND3602': 'PPFB.DE',
    'IE00BZCQB185': 'QDV5.DE',
    'IE00B5BMR087': 'SXR8.DE',
    'IE00B3XXRP09': 'VUSA.AS',
    'US67066G1040': 'NVD.DE',
    'IE00BFY0GT14': 'SPPW.DE',
    'NL0010273215': 'ASML.AS',
    'IE000RHYOR04': 'ERNX.DE',
    'IE00B3FH7618': 'IEGE.MI',
    'LU0290358497': 'XEON.DE'
}

# Explanations for tooltips
EXPLANATIONS = {
    'Historical Annual Return': "The average annual return based on historical data.",
    'Historical Annual Volatility': "The standard deviation of annual returns, measuring risk.",
    'Historical Sharpe Ratio': "Risk-adjusted return: (return - risk-free rate) / volatility.",
    'Historical Sortino Ratio': "Similar to Sharpe but only considers downside volatility.",
    'Historical Max Drawdown': "The largest peak-to-trough decline in portfolio value.",
    'Mean Final Value (Inflation-Adjusted, DCA)': "Average ending value after simulations, adjusted for inflation, using Dollar-Cost Averaging.",
    'Median Final Value (Inflation-Adjusted, DCA)': "Median ending value after simulations, adjusted for inflation, using DCA.",
    'Mean Final Value (Lump-Sum Comparison)': "Average ending value if invested all at once, for comparison.",
    'Std Dev of Final Values (DCA)': "Variability in the simulated ending values.",
    '95% VaR (Absolute Loss, DCA)': "There is a 5% chance of losing more than this amount over the horizon.",
    '95% CVaR (Absolute Loss, DCA)': "The average loss in the worst 5% of scenarios.",
    'Effective Cost Drag (%)': "The percentage reduction in returns due to fees and taxes.",
    'Total Historical Return (DCA)': "Total return from backtest using Dollar-Cost Averaging.",
    'Total Historical Return (Lump-Sum)': "Total return from backtest if invested all at once.",
    'Annualized Return': "Average annual return from historical backtest.",
    'Annualized Volatility': "Annualized standard deviation of returns from backtest.",
    'Sharpe Ratio': "Risk-adjusted return from backtest.",
    'Sortino Ratio': "Downside risk-adjusted return from backtest.",
    'Max Drawdown': "Largest decline in backtest."
}


def main():
    """Main entry point for the portfolio simulator dashboard."""
    # Initialize and render dashboard
    dashboard = PortfolioDashboard(
        default_tickers=DEFAULT_TICKERS,
        isin_to_ticker=ISIN_TO_TICKER,
        explanations=EXPLANATIONS
    )
    
    dashboard.render()


if __name__ == "__main__":
    main() 