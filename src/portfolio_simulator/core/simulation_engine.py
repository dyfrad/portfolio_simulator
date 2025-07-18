"""
Simulation Engine Module

Monte Carlo simulation engine for portfolio analysis with DCA, rebalancing, and stress testing.

Author: Mohit Saharan
Email: mohit@msaharan.com
"""

import numpy as np
from .financial_calculations import portfolio_stats


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