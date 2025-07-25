"""
Visualization Module

Contains all visualization functions for the portfolio simulator.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from .data_operations import calculate_returns


def plot_results(sim_final_values, time_horizon_years, results):
    """
    Generate histogram plot of simulated final portfolio values.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sim_final_values, bins=50, alpha=0.75, color='blue')
    ax.set_title(f'Distribution of Simulated Portfolio Values after {time_horizon_years} Years (Inflation-Adjusted, DCA)')
    ax.set_xlabel('Final Portfolio Value')
    ax.set_ylabel('Frequency')
    ax.axvline(results['Mean Final Value (Inflation-Adjusted, DCA)'], color='red', linestyle='--', label='Mean (DCA)')
    ax.axvline(results['Median Final Value (Inflation-Adjusted, DCA)'], color='green', linestyle='--', label='Median (DCA)')
    ax.legend()
    return fig


def plot_historical_performance(data, weights, tickers):
    """
    Plot cumulative historical returns for portfolio and individual assets.
    """
    returns = calculate_returns(data)
    cum_returns = (1 + returns).cumprod()
    port_cum_returns = (1 + np.dot(returns, weights)).cumprod()
    
    fig = go.Figure()
    for i, ticker in enumerate(tickers):
        fig.add_trace(go.Scatter(x=cum_returns.index, y=cum_returns[ticker], mode='lines', name=ticker))
    fig.add_trace(go.Scatter(x=cum_returns.index, y=port_cum_returns, mode='lines', name='Portfolio', line=dict(width=3, dash='dash')))
    fig.update_layout(title='Historical Cumulative Returns', xaxis_title='Date', yaxis_title='Cumulative Return')
    return fig


def plot_weight_drift(returns, target_weights, rebalance=False, rebalance_frequency='annual', rebalance_threshold=0.05):
    """
    Plot weight drift over time, with optional rebalancing.
    """
    num_assets = len(target_weights)
    values = target_weights  # Start with weights (total 1)
    drift_data = pd.DataFrame(index=returns.index, columns=['Drift'] + [f'Weight {i}' for i in range(num_assets)])
    rebalance_days = 252 if rebalance_frequency == 'annual' else 63
    day_count = 0
    
    for date, daily_returns in returns.iterrows():
        values *= (1 + daily_returns)
        total_value = np.sum(values)
        current_weights = values / total_value if total_value > 0 else target_weights
        max_drift = np.max(np.abs(current_weights - target_weights))
        drift_data.at[date, 'Drift'] = max_drift
        for i in range(num_assets):
            drift_data.at[date, f'Weight {i}'] = current_weights[i]
        
        if rebalance and (day_count % rebalance_days == 0):
            if max_drift > rebalance_threshold:
                values = total_value * target_weights  # Rebalance
        day_count += 1
    
    fig = go.Figure()
    for col in drift_data.columns[1:]:
        fig.add_trace(go.Scatter(x=drift_data.index, y=drift_data[col], mode='lines', name=col))
    fig.add_trace(go.Scatter(x=drift_data.index, y=drift_data['Drift'], mode='lines', name='Max Drift', line=dict(dash='dot')))
    fig.update_layout(title='Portfolio Weight Drift Over Time', xaxis_title='Date', yaxis_title='Weight')
    return fig


def plot_drawdowns(returns, weights):
    """
    Plot historical drawdown curve for the portfolio.
    """
    port_returns = np.dot(returns, weights)
    cum_returns = np.cumprod(1 + port_returns)
    peaks = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns / peaks) - 1
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=returns.index, y=drawdowns, mode='lines', name='Drawdown', fill='tozeroy'))
    fig.update_layout(title='Historical Portfolio Drawdown', xaxis_title='Date', yaxis_title='Drawdown')
    return fig