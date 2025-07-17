import numpy as np
import pandas as pd
import plotly.graph_objects as go

DEFAULT_TICKERS = ['IWDA.AS', 'QDV5.DE', 'PPFB.DE', 'XEON.DE']

# Copied relevant functions from the app for testing
def calculate_returns(data, ter=0.0):
    returns = data.pct_change().dropna()
    daily_ter = (ter / 100) / 252
    returns -= daily_ter
    return returns

def portfolio_stats(weights, returns, cash_ticker=DEFAULT_TICKERS[3]):
    port_returns = np.dot(returns, weights)
    annual_return = np.mean(port_returns) * 252
    annual_vol = np.std(port_returns) * np.sqrt(252)
    rf_rate = returns[cash_ticker].mean() * 252 if cash_ticker in returns.columns else 0
    sharpe = (annual_return - rf_rate) / annual_vol if annual_vol != 0 else 0
    return annual_return, annual_vol, sharpe

def bootstrap_simulation(returns, weights, num_simulations, time_horizon_years, initial_investment, inflation_rate=0.0, monthly_contrib=0.0, contrib_frequency='monthly', transaction_fee=0.0, tax_rate=0.0, rebalance=False, rebalance_frequency='annual', rebalance_threshold=0.05):
    days = int(252 * time_horizon_years)
    contrib_days = 21 if contrib_frequency == 'monthly' else 63
    rebalance_days = 252 if rebalance_frequency == 'annual' else 63
    sim_final_values = []
    sim_port_returns = []
    sim_final_values_lump = []
    
    if len(returns) == 0:
        raise ValueError("No historical returns data available.")
    
    for i in range(num_simulations):
        boot_sample = returns.sample(days, replace=True)
        
        # Lump-sum
        boot_port_returns = np.dot(boot_sample, weights)
        compounded_return = np.prod(1 + boot_port_returns) - 1
        adjusted_return_lump = (1 + compounded_return) / (1 + inflation_rate)**time_horizon_years - 1
        final_value_lump = initial_investment * (1 + adjusted_return_lump)
        gains_lump = final_value_lump - initial_investment
        net_final_lump = initial_investment + gains_lump * (1 - tax_rate) if gains_lump > 0 else final_value_lump
        sim_final_values_lump.append(net_final_lump)
        
        # DCA with optional rebalancing
        values = np.full(len(weights), initial_investment / len(weights)) if initial_investment > 0 else np.zeros(len(weights))
        total_value = initial_investment
        total_invested = initial_investment
        for d in range(0, days, contrib_days):
            period_returns = boot_sample.iloc[d:d+contrib_days]
            values *= np.prod(1 + period_returns, axis=0)
            total_value = np.sum(values)
            if d + contrib_days < days:
                effective_contrib = monthly_contrib - transaction_fee
                contrib_per_asset = effective_contrib * weights
                values += contrib_per_asset
                total_invested += monthly_contrib
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
    var_95 = np.percentile(sim_port_returns, 5) * (initial_investment + monthly_contrib * (days // contrib_days))
    cvar_95 = np.mean(sim_port_returns[sim_port_returns <= np.percentile(sim_port_returns, 5)]) * (initial_investment + monthly_contrib * (days // contrib_days))
    
    mean_final_lump = np.mean(sim_final_values_lump)
    
    hist_return, hist_vol, hist_sharpe = portfolio_stats(weights, returns)
    
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
        'Effective Cost Drag (%)': cost_drag
    }
    
    return results, sim_final_values

def plot_weight_drift(returns, target_weights, rebalance=False, rebalance_frequency='annual', rebalance_threshold=0.05):
    num_assets = len(target_weights)
    values = np.full(num_assets, 1.0 / num_assets)  # Start with equal value per asset (total 1)
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

def backtest_portfolio(data, weights, monthly_contrib=0.0, contrib_frequency='monthly', transaction_fee=0.0, tax_rate=0.0, rebalance=False, rebalance_frequency='annual', rebalance_threshold=0.05, initial_investment=10000):
    returns = calculate_returns(data)
    num_assets = len(weights)
    values = np.full(num_assets, initial_investment / num_assets) if initial_investment > 0 else np.zeros(num_assets)
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
    ann_return, ann_vol, sharpe = portfolio_stats(weights, returns)
    
    # Lump-sum with tax
    gains_lump = initial_investment * cum_port_returns
    net_cum_port_returns = cum_port_returns - (gains_lump / initial_investment) * tax_rate if gains_lump > 0 else cum_port_returns
    
    # DCA backtest
    cum_port_returns_dca = net_cum_port_returns  # Default to lump if no DCA
    if monthly_contrib > 0:
        freq = 'M' if contrib_frequency == 'monthly' else 'Q'
        monthly_data = data.resample(freq).last()
        monthly_returns = monthly_data.pct_change().dropna()
        port_monthly_returns = np.dot(monthly_returns, weights)
        value = 0.0
        total_invested = 0.0
        for ret in port_monthly_returns:
            effective_contrib = monthly_contrib - transaction_fee
            value = (value + effective_contrib) * (1 + ret)
            total_invested += monthly_contrib
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
        'Sharpe Ratio': sharpe
    }

# Mock data for tests: Varying returns to induce drift (Asset 0 grows faster)
np.random.seed(42)  # For reproducibility
dates = pd.date_range(start='2024-01-01', periods=252, freq='B')
varying_returns = pd.DataFrame(np.random.normal(0.0005, 0.02, (252, 4)), index=dates, columns=['IWDA.AS', 'QDV5.DE', 'PPFB.DE', 'XEON.DE'])
varying_returns.iloc[:, 0] += 0.002  # Even stronger bias for asset 0 to ensure drift
mock_data = (1 + varying_returns).cumprod() * 100
mock_returns = varying_returns

# Weights for tests
target_weights = np.array([0.25, 0.25, 0.25, 0.25])

# Test Suite for Rebalancing
def run_tests():
    passed = 0
    total = 5

    # Test 1: No rebalancing in simulation (final value without reset)
    try:
        results_no_rebal, _ = bootstrap_simulation(mock_returns, target_weights, num_simulations=5, time_horizon_years=1, initial_investment=10000, rebalance=False)
        print("Test 1 (No rebalance sim runs): PASSED")
        passed += 1
    except Exception as e:
        print(f"Test 1: FAILED - {e}")

    # Test 2: Rebalancing in simulation (should run, final value close but different due to resets)
    try:
        results_rebal, _ = bootstrap_simulation(mock_returns, target_weights, num_simulations=5, time_horizon_years=1, initial_investment=10000, rebalance=True, rebalance_threshold=0.01)
        assert abs(results_rebal['Mean Final Value (Inflation-Adjusted, DCA)'] - results_no_rebal['Mean Final Value (Inflation-Adjusted, DCA)']) < 1000, "Failed: Rebal value too different"
        print("Test 2 (Rebalance sim runs and value reasonable): PASSED")
        passed += 1
    except AssertionError as e:
        print(f"Test 2: FAILED - {e}")

    # Test 3: Weight drift plot without rebalance (drift > 0)
    try:
        fig_no_rebal = plot_weight_drift(mock_returns, target_weights, rebalance=False)
        drift_values = [fig_no_rebal.data[-1].y[-1]]  # Last max drift
        assert drift_values[0] > 0, "Failed: No drift in biased returns"
        print("Test 3 (Drift plot shows positive drift without rebal): PASSED")
        passed += 1
    except AssertionError as e:
        print(f"Test 3: FAILED - {e}")

    # Test 4: Weight drift plot with rebalance (lower final drift)
    try:
        fig_rebal = plot_weight_drift(mock_returns, target_weights, rebalance=True, rebalance_threshold=0.01)
        drift_no_rebal = fig_no_rebal.data[-1].y[-1]
        drift_rebal = fig_rebal.data[-1].y[-1]
        assert drift_rebal < drift_no_rebal, f"Failed: Rebal drift {drift_rebal} not < no rebal {drift_no_rebal}"
        print("Test 4 (Drift plot shows reduced drift with rebal): PASSED")
        passed += 1
    except AssertionError as e:
        print(f"Test 4: FAILED - {e}")

    # Test 5: Backtest with rebalance (runs, return slightly different)
    try:
        back_no_rebal = backtest_portfolio(mock_data, target_weights, rebalance=False)
        back_rebal = backtest_portfolio(mock_data, target_weights, rebalance=True, rebalance_threshold=0.01)
        assert abs(back_rebal['Total Return (Lump-Sum)'] - back_no_rebal['Total Return (Lump-Sum)']) < 0.05, "Failed: Rebal return too different"
        print("Test 5 (Backtest with rebal runs and return reasonable): PASSED")
        passed += 1
    except AssertionError as e:
        print(f"Test 5: FAILED - {e}")

    print(f"\n{passed}/{total} tests passed.")

run_tests()