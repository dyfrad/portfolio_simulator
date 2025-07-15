import numpy as np
import pandas as pd

# Copied relevant functions from the app for testing
def calculate_returns(data, ter=0.0):
    returns = data.pct_change().dropna()
    daily_ter = (ter / 100) / 252
    returns -= daily_ter
    return returns

def portfolio_stats(weights, returns, cash_ticker='XEON.DE'):
    port_returns = np.dot(returns, weights)
    annual_return = np.mean(port_returns) * 252
    annual_vol = np.std(port_returns) * np.sqrt(252)
    rf_rate = returns[cash_ticker].mean() * 252 if cash_ticker in returns.columns else 0
    sharpe = (annual_return - rf_rate) / annual_vol if annual_vol != 0 else 0
    return annual_return, annual_vol, sharpe

def bootstrap_simulation(returns, weights, num_simulations, time_horizon_years, initial_investment, inflation_rate=0.0, monthly_contrib=0.0, contrib_frequency='monthly', transaction_fee=0.0, tax_rate=0.0):
    days = int(252 * time_horizon_years)
    contrib_days = 21 if contrib_frequency == 'monthly' else 63
    sim_final_values = []
    sim_port_returns = []
    sim_final_values_lump = []
    
    if len(returns) == 0:
        raise ValueError("No historical returns data available.")
    
    for _ in range(num_simulations):
        boot_sample = returns.sample(days, replace=True)
        boot_port_returns = np.dot(boot_sample, weights)
        
        # Lump-sum
        compounded_return = np.prod(1 + boot_port_returns) - 1
        adjusted_return_lump = (1 + compounded_return) / (1 + inflation_rate)**time_horizon_years - 1
        final_value_lump = initial_investment * (1 + adjusted_return_lump)
        gains_lump = final_value_lump - initial_investment
        net_final_lump = initial_investment + gains_lump * (1 - tax_rate) if gains_lump > 0 else final_value_lump
        sim_final_values_lump.append(net_final_lump)
        
        # DCA
        value = initial_investment
        total_invested = initial_investment
        num_contribs = 0
        for d in range(0, days, contrib_days):
            period_returns = boot_port_returns[d:d+contrib_days]
            value *= np.prod(1 + period_returns)
            if d + contrib_days < days:
                effective_contrib = monthly_contrib - transaction_fee
                value += effective_contrib
                total_invested += monthly_contrib
                num_contribs += 1
        compounded_return_dca = (value / total_invested) - 1 if total_invested > 0 else 0
        adjusted_return_dca = (1 + compounded_return_dca) / (1 + inflation_rate)**time_horizon_years - 1
        final_value_dca = total_invested * (1 + adjusted_return_dca)
        gains_dca = final_value_dca - total_invested
        net_final_dca = total_invested + gains_dca * (1 - tax_rate) if gains_dca > 0 else final_value_dca
        sim_final_values.append(net_final_dca)
        sim_port_returns.append(adjusted_return_dca)
    
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

def backtest_portfolio(data, weights, monthly_contrib=0.0, contrib_frequency='monthly', transaction_fee=0.0, tax_rate=0.0, initial_investment=10000):
    returns = data.pct_change().dropna()
    port_returns = np.dot(returns, weights)
    cum_port_returns = (1 + port_returns).cumprod()[-1] - 1
    ann_return, ann_vol, sharpe = portfolio_stats(weights, returns)
    
    gains_lump = initial_investment * cum_port_returns
    net_cum_port_returns = cum_port_returns - (gains_lump / initial_investment) * tax_rate if gains_lump > 0 else cum_port_returns
    
    cum_port_returns_dca = net_cum_port_returns
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

# Mock data with fixed positive returns for predictability (1 year, ~12% ann return)
dates = pd.date_range(start='2024-01-01', periods=252, freq='B')  # 252 business days
fixed_returns = pd.DataFrame(0.0005 * np.ones((252, 4)), index=dates, columns=['IWDA.AS', 'QDV5.DE', 'PPFB.DE', 'XEON.DE'])  # Fixed 0.05% daily
mock_data = (1 + fixed_returns).cumprod() * 100  # Prices starting at 100
mock_returns = fixed_returns

# Weights for tests (equal allocation)
weights = np.array([0.25, 0.25, 0.25, 0.25])

# Test Suite for Fees and Taxes
def run_tests():
    passed = 0
    total = 6

    # Test 1: TER reduces returns in calculate_returns
    try:
        returns_no_ter = calculate_returns(mock_data, ter=0.0)
        returns_with_ter = calculate_returns(mock_data, ter=0.5)  # 0.5% TER
        assert np.all(returns_with_ter < returns_no_ter), "Failed: TER did not reduce returns"
        print("Test 1 (TER reduces daily returns): PASSED")
        passed += 1
    except AssertionError as e:
        print(f"Test 1: FAILED - {e}")

    # Test 2: Simulation with no fees/tax (matches base final value)
    try:
        results_no_fee, _ = bootstrap_simulation(mock_returns, weights, num_simulations=10, time_horizon_years=1, initial_investment=10000, inflation_rate=0.0, monthly_contrib=0, transaction_fee=0, tax_rate=0)
        expected_final = 10000 * np.prod(1 + 0.0005) ** 252  # Approx with fixed returns
        assert abs(results_no_fee['Mean Final Value (Inflation-Adjusted, DCA)'] - expected_final) < 100, "Failed: No fee/tax doesn't match expected"
        print("Test 2 (No fees/tax, sim matches base): PASSED")
        passed += 1
    except AssertionError as e:
        print(f"Test 2: FAILED - {e}")

    # Test 3: Simulation with tax reduces gains
    try:
        results_tax, _ = bootstrap_simulation(mock_returns, weights, num_simulations=10, time_horizon_years=1, initial_investment=10000, inflation_rate=0.0, monthly_contrib=0, transaction_fee=0, tax_rate=0.25)
        results_no_tax, _ = bootstrap_simulation(mock_returns, weights, num_simulations=10, time_horizon_years=1, initial_investment=10000, inflation_rate=0.0, monthly_contrib=0, transaction_fee=0, tax_rate=0)
        assert results_tax['Mean Final Value (Inflation-Adjusted, DCA)'] < results_no_tax['Mean Final Value (Inflation-Adjusted, DCA)'], "Failed: Tax did not reduce final value"
        print("Test 3 (Tax reduces sim gains): PASSED")
        passed += 1
    except AssertionError as e:
        print(f"Test 3: FAILED - {e}")

    # Test 4: Simulation with transaction fees in DCA
    try:
        results_fee, _ = bootstrap_simulation(mock_returns, weights, num_simulations=10, time_horizon_years=1, initial_investment=0, inflation_rate=0.0, monthly_contrib=1000, transaction_fee=5, tax_rate=0)
        results_no_fee, _ = bootstrap_simulation(mock_returns, weights, num_simulations=10, time_horizon_years=1, initial_investment=0, inflation_rate=0.0, monthly_contrib=1000, transaction_fee=0, tax_rate=0)
        assert results_fee['Mean Final Value (Inflation-Adjusted, DCA)'] < results_no_fee['Mean Final Value (Inflation-Adjusted, DCA)'], "Failed: Fees did not reduce DCA value"
        print("Test 4 (Transaction fees reduce DCA sim): PASSED")
        passed += 1
    except AssertionError as e:
        print(f"Test 4: FAILED - {e}")

    # Test 5: Cost drag positive with fees/tax
    try:
        results_cost, _ = bootstrap_simulation(mock_returns, weights, num_simulations=10, time_horizon_years=1, initial_investment=10000, inflation_rate=0.0, monthly_contrib=0, transaction_fee=0, tax_rate=0.25)
        assert results_cost['Effective Cost Drag (%)'] > 0, f"Failed: Cost drag {results_cost['Effective Cost Drag (%)']} <= 0"
        print("Test 5 (Cost drag positive): PASSED")
        passed += 1
    except AssertionError as e:
        print(f"Test 5: FAILED - {e}")

    # Test 6: Backtest with tax and fees reduces returns
    try:
        back_results_fee_tax = backtest_portfolio(mock_data, weights, monthly_contrib=1000, transaction_fee=5, tax_rate=0.25, initial_investment=10000)
        back_results_no_fee_tax = backtest_portfolio(mock_data, weights, monthly_contrib=1000, transaction_fee=0, tax_rate=0, initial_investment=10000)
        assert back_results_fee_tax['Total Return (DCA)'] < back_results_no_fee_tax['Total Return (DCA)'], "Failed: Fees/tax did not reduce backtest DCA return"
        assert back_results_fee_tax['Total Return (Lump-Sum)'] < back_results_no_fee_tax['Total Return (Lump-Sum)'], "Failed: Tax did not reduce backtest lump-sum return"
        print("Test 6 (Fees/tax reduce backtest returns): PASSED")
        passed += 1
    except AssertionError as e:
        print(f"Test 6: FAILED - {e}")

    print(f"\n{passed}/{total} tests passed.")

run_tests()