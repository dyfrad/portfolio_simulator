"""
Portfolio Simulator

Author: Mohit Saharan
Email: mohit@msaharan.com
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go

# Default tickers for the assets (UCITS-compliant, EUR-traded)
DEFAULT_TICKERS = ['IWDA.AS', 'QDV5.DE', 'PPFB.DE', 'XEON.DE']  # MSCI World, MSCI India, Gold, Cash

# ISIN to Ticker mapping
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

# Default start date for historical data
DEFAULT_START_DATE = '2015-01-01'



def fetch_data(tickers, start_date, end_date=None):
    """
    Fetch historical adjusted close prices for given tickers.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Close'].dropna()
    if len(data) < 252:
        print("Warning: Limited historical data available. Simulations may not be reliable.")
    return data

def calculate_returns(data, ter=0.0):
    """
    Calculate daily returns from price data, adjusted for TER.
    """
    returns = data.pct_change().dropna()
    daily_ter = (ter / 100) / 252  # Annual TER to daily deduction
    returns -= daily_ter
    return returns

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

def bootstrap_simulation(returns, weights, num_simulations, time_horizon_years, initial_investment, inflation_rate=0.0, periodic_contrib=0.0, contrib_frequency='monthly', transaction_fee=0.0, tax_rate=0.0, rebalance=False, rebalance_frequency='annual', rebalance_threshold=0.05, shock_factors=None, base_invested=None, progress_callback=None):
    """
    Perform bootstrap Monte Carlo simulation with optional inflation adjustment, DCA, fees, taxes, rebalancing, and stress shocks.
    """
    if base_invested is None:
        base_invested = initial_investment
    days = int(252 * time_horizon_years)
    contrib_days = 21 if contrib_frequency == 'monthly' else 63  # Approx trading days per month/quarter
    rebalance_days = 252 if rebalance_frequency == 'annual' else 63  # Annual or quarterly
    sim_final_values = []
    sim_port_returns = []
    sim_final_values_lump = []  # For lump-sum comparison
    
    if len(returns) == 0:
        raise ValueError("No historical returns data available.")
    
    for i in range(num_simulations):
        # Update progress if callback provided
        if progress_callback:
            progress_callback((i + 1) / num_simulations)
            
        boot_sample = returns.sample(days, replace=True).reset_index(drop=True)
        
        # Apply stress shock if specified
        if shock_factors is not None:
            shock_start = np.random.randint(0, days - 251)
            daily_shock = shock_factors / 252
            for dd in range(252):
                boot_sample.iloc[shock_start + dd] = daily_shock
        
        # Lump-sum
        boot_port_returns = np.dot(boot_sample, weights)
        compounded_return = np.prod(1 + boot_port_returns) - 1
        adjusted_return_lump = (1 + compounded_return) / (1 + inflation_rate)**time_horizon_years - 1
        final_value_lump = initial_investment * (1 + adjusted_return_lump)
        gains_lump = final_value_lump - base_invested
        net_final_lump = base_invested + gains_lump * (1 - tax_rate) if gains_lump > 0 else final_value_lump
        sim_final_values_lump.append(net_final_lump)
        
        # DCA with optional rebalancing
        values = initial_investment * weights
        total_value = initial_investment
        total_invested = base_invested
        for d in range(0, days, contrib_days):
            period_returns = boot_sample.iloc[d:d+contrib_days]
            values *= np.prod(1 + period_returns, axis=0)
            total_value = np.sum(values)
            if d + contrib_days < days:
                effective_contrib = periodic_contrib - transaction_fee
                contrib_per_asset = effective_contrib * weights
                values += contrib_per_asset
                total_invested += periodic_contrib
            # Rebalance if enabled and at interval
            if rebalance and (d % rebalance_days == 0 or d + contrib_days >= days):
                current_weights = values / total_value if total_value > 0 else weights
                if np.any(np.abs(current_weights - weights) > rebalance_threshold):
                    values = total_value * weights  # Reset to target weights (virtual sell/buy)
        final_value_dca = np.sum(values)
        gains_dca = final_value_dca - total_invested
        net_final_dca = total_invested + gains_dca * (1 - tax_rate) if gains_dca > 0 else final_value_dca
        adjusted_net_final_dca = net_final_dca / (1 + inflation_rate)**time_horizon_years
        sim_final_values.append(adjusted_net_final_dca)
        sim_port_returns.append((adjusted_net_final_dca / total_invested) - 1 if total_invested > 0 else 0)
    
    sim_final_values = np.array(sim_final_values)
    sim_port_returns = np.array(sim_port_returns)
    sim_final_values_lump = np.array(sim_final_values_lump)
    
    mean_final = np.mean(sim_final_values)
    median_final = np.median(sim_final_values)
    std_final = np.std(sim_final_values)
    var_95 = np.percentile(sim_port_returns, 5) * (initial_investment + periodic_contrib * (days // contrib_days))
    cvar_95 = np.mean(sim_port_returns[sim_port_returns <= np.percentile(sim_port_returns, 5)]) * (initial_investment + periodic_contrib * (days // contrib_days))
    
    mean_final_lump = np.mean(sim_final_values_lump)
    
    hist_return, hist_vol, hist_sharpe, hist_sortino, hist_max_dd = portfolio_stats(weights, returns)
    
    # Calculate effective cost drag (using mean final DCA gross vs net)
    gross_mean_final = total_invested * (1 + np.mean([np.prod(1 + np.dot(returns.sample(days, replace=True), weights)) - 1 for _ in range(10)]))  # Approximate gross
    cost_drag = ((gross_mean_final - mean_final) / gross_mean_final * 100) if gross_mean_final > 0 else 0
    
    results = {
        'Mean Final Value (Inflation-Adjusted, DCA)': mean_final,
        'Median Final Value (Inflation-Adjusted, DCA)': median_final,
        'Mean Final Value (Lump-Sum Comparison)': mean_final_lump,
        'Std Dev of Final Values (DCA)': std_final,
        '95% VaR (Absolute Loss, DCA)': var_95,
        '95% CVaR (Absolute Loss, DCA)': cvar_95,
        'Historical Annual Return': hist_return,
        'Historical Annual Volatility': hist_vol,
        'Historical Sharpe Ratio': hist_sharpe,
        'Historical Sortino Ratio': hist_sortino,
        'Historical Max Drawdown': hist_max_dd,
        'Effective Cost Drag (%)': cost_drag
    }
    
    return results, sim_final_values

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

def backtest_portfolio(data, weights, initial_investment, periodic_contrib=0.0, contrib_frequency='monthly', transaction_fee=0.0, tax_rate=0.0, rebalance=False, rebalance_frequency='annual', rebalance_threshold=0.05):
    """
    Backtest portfolio over historical data with optional DCA, fees, taxes, and rebalancing.
    """
    returns = calculate_returns(data)
    num_assets = len(weights)
    values = initial_investment * weights
    total_value = initial_investment
    total_invested = initial_investment
    rebalance_days = 252 if rebalance_frequency == 'annual' else 63
    day_count = 0
    
    for _, daily_returns in returns.iterrows():
        values *= (1 + daily_returns)
        total_value = np.sum(values)
        current_weights = values / total_value if total_value > 0 else weights
        if rebalance and (day_count % rebalance_days == 0):
            max_drift = np.max(np.abs(current_weights - weights))
            if max_drift > rebalance_threshold:
                values = total_value * weights  # Rebalance
        day_count += 1
    
    cum_port_returns = (total_value / initial_investment) - 1 if initial_investment > 0 else 0
    ann_return, ann_vol, sharpe, sortino, max_dd = portfolio_stats(weights, returns)
    
    # Lump-sum with tax
    gains_lump = initial_investment * cum_port_returns
    net_cum_port_returns = cum_port_returns - (gains_lump / initial_investment) * tax_rate if gains_lump > 0 else cum_port_returns
    
    # DCA backtest
    cum_port_returns_dca = net_cum_port_returns  # Default to lump if no DCA
    if periodic_contrib > 0:
        freq = 'M' if contrib_frequency == 'monthly' else 'Q'
        monthly_data = data.resample(freq).last()
        monthly_returns = monthly_data.pct_change().dropna()
        port_monthly_returns = np.dot(monthly_returns, weights)
        value = 0.0
        total_invested = 0.0
        for ret in port_monthly_returns:
            effective_contrib = periodic_contrib - transaction_fee
            value = (value + effective_contrib) * (1 + ret)
            total_invested += periodic_contrib
        cum_port_returns_dca = (value / total_invested) - 1 if total_invested > 0 else 0
        # Apply tax on gains for DCA
        gains_dca = value - total_invested
        net_value_dca = total_invested + gains_dca * (1 - tax_rate) if gains_dca > 0 else value
        cum_port_returns_dca = (net_value_dca / total_invested) - 1 if total_invested > 0 else 0
    
    return {
        'Total Return (DCA)': cum_port_returns_dca,
        'Total Return (Lump-Sum)': net_cum_port_returns,
        'Annualized Return': ann_return,
        'Annualized Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_dd
    }

def optimize_weights(returns, cash_ticker=DEFAULT_TICKERS[3]):
    def objective(weights):
        ann_ret, ann_vol, sharpe, _, _ = portfolio_stats(weights, returns, cash_ticker)
        return -sharpe  # Minimize negative Sharpe
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(len(returns.columns))]
    initial_guess = np.array([1/len(returns.columns)] * len(returns.columns))
    result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
    return result.x if result.success else None

