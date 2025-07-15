"""
Portfolio Simulator Application

This is a modular Python application for simulating portfolio performance using bootstrap Monte Carlo methods.
It fetches historical data for UCITS-compliant ETFs (MSCI World, MSCI India, Gold, Cash) and simulates future outcomes
based on user-defined weights and parameters.

Key Features:
- Fetches data via yfinance.
- Calculates historical stats (return, vol, Sharpe).
- Runs bootstrap simulations for risk metrics (VaR, CVaR).
- Plots distribution of outcomes.
- Command-line interface for easy customization.

Usage:
python portfolio_simulator.py --weights 0.4 0.3 0.2 0.1 --horizon 5 --simulations 5000 --initial 100000

For continued development:
- Add GUI (e.g., via Tkinter or Streamlit).
- Implement weight optimization (e.g., using scipy.optimize).
- Add more assets or custom data sources.
- Incorporate live data updates or API integrations.
- Add saving results to CSV or database.

Dependencies: yfinance, numpy, pandas, matplotlib, argparse
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Default tickers for the assets (UCITS-compliant, EUR-traded)
DEFAULT_TICKERS = ['IWDA.AS', 'QDV5.DE', 'PPFB.DE', 'XEON.DE']  # MSCI World, MSCI India, Gold, Cash

# Default start date for historical data
DEFAULT_START_DATE = '2015-01-01'

def fetch_data(tickers, start_date):
    """
    Fetch historical adjusted close prices for given tickers.
    
    Args:
        tickers (list): List of ticker symbols.
        start_date (str): Start date for data download (YYYY-MM-DD).
    
    Returns:
        pd.DataFrame: DataFrame with closing prices.
    """
    data = yf.download(tickers, start=start_date)['Close'].dropna()
    if len(data) < 252:
        print("Warning: Limited historical data available. Simulations may not be reliable.")
    return data

def calculate_returns(data):
    """
    Calculate daily returns from price data.
    
    Args:
        data (pd.DataFrame): Price data.
    
    Returns:
        pd.DataFrame: Daily returns.
    """
    return data.pct_change().dropna()

def portfolio_stats(weights, returns, cash_ticker=DEFAULT_TICKERS[3]):
    """
    Calculate historical portfolio statistics.
    
    Args:
        weights (np.array): Portfolio weights.
        returns (pd.DataFrame): Daily returns.
        cash_ticker (str): Ticker for cash proxy.
    
    Returns:
        tuple: (annual_return, annual_vol, sharpe)
    """
    port_returns = np.dot(returns, weights)
    annual_return = np.mean(port_returns) * 252
    annual_vol = np.std(port_returns) * np.sqrt(252)
    rf_rate = returns[cash_ticker].mean() * 252
    sharpe = (annual_return - rf_rate) / annual_vol if annual_vol != 0 else 0
    return annual_return, annual_vol, sharpe

def bootstrap_simulation(returns, weights, num_simulations, time_horizon_years, initial_investment):
    """
    Perform bootstrap Monte Carlo simulation.
    
    Args:
        returns (pd.DataFrame): Daily returns.
        weights (np.array): Portfolio weights.
        num_simulations (int): Number of simulations.
        time_horizon_years (int): Simulation horizon in years.
        initial_investment (float): Initial portfolio value.
    
    Returns:
        dict: Results dictionary.
        np.array: Simulated final values.
    """
    days = int(252 * time_horizon_years)
    sim_final_values = []
    sim_port_returns = []
    
    if len(returns) == 0:
        raise ValueError("No historical returns data available. Check tickers and data download.")
    
    for _ in range(num_simulations):
        boot_sample = returns.sample(days, replace=True)
        boot_port_returns = np.dot(boot_sample, weights)
        compounded_return = np.prod(1 + boot_port_returns) - 1
        final_value = initial_investment * (1 + compounded_return)
        sim_final_values.append(final_value)
        sim_port_returns.append(compounded_return)
    
    sim_final_values = np.array(sim_final_values)
    sim_port_returns = np.array(sim_port_returns)
    
    mean_final = np.mean(sim_final_values)
    median_final = np.median(sim_final_values)
    std_final = np.std(sim_final_values)
    var_95 = np.percentile(sim_port_returns, 5) * initial_investment
    cvar_95 = np.mean(sim_port_returns[sim_port_returns <= np.percentile(sim_port_returns, 5)]) * initial_investment
    
    hist_return, hist_vol, hist_sharpe = portfolio_stats(weights, returns)
    
    results = {
        'Mean Final Value': mean_final,
        'Median Final Value': median_final,
        'Std Dev of Final Values': std_final,
        '95% VaR (Absolute Loss)': var_95,
        '95% CVaR (Absolute Loss)': cvar_95,
        'Historical Annual Return': hist_return,
        'Historical Annual Volatility': hist_vol,
        'Historical Sharpe Ratio': hist_sharpe
    }
    
    return results, sim_final_values

def plot_results(sim_final_values, time_horizon_years, results):
    """
    Plot histogram of simulated final portfolio values.
    
    Args:
        sim_final_values (np.array): Simulated values.
        time_horizon_years (int): Horizon.
        results (dict): Simulation results.
    """
    plt.figure(figsize=(10, 6))
    plt.hist(sim_final_values, bins=50, alpha=0.75, color='blue')
    plt.title(f'Distribution of Simulated Portfolio Values after {time_horizon_years} Years')
    plt.xlabel('Final Portfolio Value')
    plt.ylabel('Frequency')
    plt.axvline(results['Mean Final Value'], color='red', linestyle='--', label='Mean')
    plt.axvline(results['Median Final Value'], color='green', linestyle='--', label='Median')
    plt.legend()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Portfolio Performance Simulator")
    parser.add_argument('--weights', type=float, nargs='+', default=[0.40, 0.30, 0.20, 0.10],
                        help='Portfolio weights (must sum to 1, e.g., 0.4 0.3 0.2 0.1)')
    parser.add_argument('--horizon', type=int, default=5, help='Simulation horizon in years')
    parser.add_argument('--simulations', type=int, default=5000, help='Number of simulations')
    parser.add_argument('--initial', type=float, default=100000, help='Initial investment amount')
    parser.add_argument('--tickers', type=str, nargs='+', default=DEFAULT_TICKERS,
                        help='List of tickers (default: IWDA.AS QDV5.DE PPFB.DE XEON.DE)')
    parser.add_argument('--start_date', type=str, default=DEFAULT_START_DATE,
                        help='Start date for historical data (YYYY-MM-DD)')
    
    args = parser.parse_args()
    
    weights = np.array(args.weights)
    if np.sum(weights) != 1.0:
        weights = weights / np.sum(weights)
        print("Warning: Weights normalized to sum to 1.")
    
    data = fetch_data(args.tickers, args.start_date)
    returns = calculate_returns(data)
    
    results, sim_final_values = bootstrap_simulation(
        returns, weights, args.simulations, args.horizon, args.initial
    )
    
    print("Portfolio Simulation Results:")
    for key, value in results.items():
        print(f"{key}: {value:.2f}")
    
    plot_results(sim_final_values, args.horizon, results)

if __name__ == "__main__":
    main()