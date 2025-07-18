"""
Financial Calculations Module

Core financial mathematics for portfolio analysis including statistics and optimization.

Author: Mohit Saharan
Email: mohit@msaharan.com
"""

import numpy as np
from scipy.optimize import minimize
from .data_operations import DEFAULT_TICKERS


def portfolio_stats(weights, returns, cash_ticker=DEFAULT_TICKERS[3]):
    """
    Calculate historical portfolio statistics, including Sortino and max drawdown.
    """
    port_returns = np.dot(returns, weights)
    mean_return = np.mean(port_returns)
    annual_return = mean_return * 252
    annual_vol = np.std(port_returns) * np.sqrt(252)
    rf_rate = returns[cash_ticker].mean() * 252 if cash_ticker in returns.columns else 0
    sharpe = (annual_return - rf_rate) / annual_vol if annual_vol != 0 else 0
    
    # Sortino Ratio
    downside_returns = port_returns.copy()
    downside_returns[port_returns > 0] = 0
    downside_std = np.std(downside_returns) * np.sqrt(252)
    sortino = (annual_return - rf_rate) / downside_std if downside_std > 0 else 0
    
    # Max Drawdown
    cum_returns = np.cumprod(1 + port_returns)
    peaks = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns / peaks) - 1
    max_dd = drawdowns.min()
    
    return annual_return, annual_vol, sharpe, sortino, max_dd


def optimize_weights(returns, cash_ticker=DEFAULT_TICKERS[3]):
    """
    Optimize portfolio weights to maximize Sharpe ratio.
    """
    def objective(weights):
        ann_ret, ann_vol, sharpe, _, _ = portfolio_stats(weights, returns, cash_ticker)
        return -sharpe  # Minimize negative Sharpe
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(len(returns.columns))]
    initial_guess = np.array([1/len(returns.columns)] * len(returns.columns))
    result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
    return result.x if result.success else None