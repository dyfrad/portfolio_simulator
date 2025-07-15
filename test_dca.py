import numpy as np
import pandas as pd

# Copied relevant functions from the app for testing
def portfolio_stats(weights, returns, cash_ticker='XEON.DE'):
    port_returns = np.dot(returns, weights)
    annual_return = np.mean(port_returns) * 252
    annual_vol = np.std(port_returns) * np.sqrt(252)
    rf_rate = returns[cash_ticker].mean() * 252 if cash_ticker in returns.columns else 0
    sharpe = (annual_return - rf_rate) / annual_vol if annual_vol != 0 else 0
    return annual_return, annual_vol, sharpe

def bootstrap_simulation(returns, weights, num_simulations, time_horizon_years, initial_investment, inflation_rate=0.0, monthly_contrib=0.0, contrib_frequency='monthly'):
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
        sim_final_values_lump.append(final_value_lump)
        
        # DCA
        value = initial_investment
        total_invested = initial_investment
        for d in range(0, days, contrib_days):
            period_returns = boot_port_returns[d:d+contrib_days]
            value *= np.prod(1 + period_returns)
            if d + contrib_days < days:
                value += monthly_contrib
                total_invested += monthly_contrib
        compounded_return_dca = (value / total_invested) - 1 if total_invested > 0 else 0
        adjusted_return_dca = (1 + compounded_return_dca) / (1 + inflation_rate)**time_horizon_years - 1
        final_value_dca = total_invested * (1 + adjusted_return_dca)
        sim_final_values.append(final_value_dca)
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
    
    results = {
        'Mean Final Value (Inflation-Adjusted, DCA)': mean_final,
        'Median Final Value (Inflation-Adjusted, DCA)': median_final,
        'Mean Final Value (Lump-Sum Comparison)': mean_final_lump,
        'Std Dev of Final Values (DCA)': std_final,
        '95% VaR (Absolute Loss, DCA)': var_95,
        '95% CVaR (Absolute Loss, DCA)': cvar_95,
        'Historical Annual Return': hist_return,
        'Historical Annual Volatility': hist_vol,
        'Historical Sharpe Ratio': hist_sharpe
    }
    
    return results, sim_final_values

def backtest_portfolio(data, weights, monthly_contrib=0.0, contrib_frequency='monthly'):
    returns = data.pct_change().dropna()
    port_returns = np.dot(returns, weights)
    cum_port_returns = (1 + port_returns).cumprod()[-1] - 1
    ann_return, ann_vol, sharpe = portfolio_stats(weights, returns)
    
    # DCA backtest
    if monthly_contrib > 0:
        freq = 'M' if contrib_frequency == 'monthly' else 'Q'
        monthly_data = data.resample(freq).last()
        monthly_returns = monthly_data.pct_change().dropna()
        port_monthly_returns = np.dot(monthly_returns, weights)
        value = 0.0
        total_invested = 0.0
        for ret in port_monthly_returns:
            value = (value + monthly_contrib) * (1 + ret)
            total_invested += monthly_contrib
        cum_port_returns_dca = (value / total_invested) - 1 if total_invested > 0 else 0
    else:
        cum_port_returns_dca = cum_port_returns
    
    return {
        'Total Return (DCA)': cum_port_returns_dca,
        'Total Return (Lump-Sum)': cum_port_returns,
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

# Test Suite
def run_tests():
    passed = 0
    total = 5

    # Test 1: Simulation with no DCA (should match lump-sum)
    try:
        results_no_dca, _ = bootstrap_simulation(mock_returns, weights, num_simulations=10, time_horizon_years=1, initial_investment=10000, inflation_rate=0.0, monthly_contrib=0.0)
        diff = abs(results_no_dca['Mean Final Value (Inflation-Adjusted, DCA)'] - results_no_dca['Mean Final Value (Lump-Sum Comparison)'])
        assert diff < 1e-3, f"Failed: Diff {diff} > tolerance"
        print("Test 1 (No DCA, sim values match): PASSED")
        passed += 1
    except AssertionError as e:
        print(f"Test 1: FAILED - {e}")

    # Test 2: Simulation with DCA (final > total invested, due to positive returns)
    try:
        results_dca, _ = bootstrap_simulation(mock_returns, weights, num_simulations=10, time_horizon_years=1, initial_investment=0, inflation_rate=0.0, monthly_contrib=1000, contrib_frequency='monthly')
        num_periods = 252 // 21  # ~12
        total_invested_approx = 1000 * (num_periods - 1)  # 11 adds for monthly
        assert results_dca['Mean Final Value (Inflation-Adjusted, DCA)'] > total_invested_approx, f"Failed: {results_dca['Mean Final Value (Inflation-Adjusted, DCA)']} <= {total_invested_approx}"
        assert results_dca['Mean Final Value (Inflation-Adjusted, DCA)'] != results_dca['Mean Final Value (Lump-Sum Comparison)'], "Failed: DCA matches lump-sum"
        print("Test 2 (DCA sim > total contrib and differs from lump-sum): PASSED")
        passed += 1
    except AssertionError as e:
        print(f"Test 2: FAILED - {e}")

    # Test 3: Backtest with no DCA (should match lump-sum)
    try:
        back_results_no_dca = backtest_portfolio(mock_data, weights, monthly_contrib=0.0)
        diff = abs(back_results_no_dca['Total Return (DCA)'] - back_results_no_dca['Total Return (Lump-Sum)'])
        assert diff < 1e-5, f"Failed: Diff {diff} > tolerance"
        print("Test 3 (No DCA, backtest returns match): PASSED")
        passed += 1
    except AssertionError as e:
        print(f"Test 3: FAILED - {e}")

    # Test 4: Backtest with DCA (positive return, differs from lump-sum)
    try:
        back_results_dca = backtest_portfolio(mock_data, weights, monthly_contrib=1000, contrib_frequency='monthly')
        num_months = len(mock_data.resample('M').last()) - 1  # ~11 returns for 12 months
        total_invested_approx = 1000 * num_months  # Adds per return loop
        assert back_results_dca['Total Return (DCA)'] > 0, f"Failed: Return {back_results_dca['Total Return (DCA)']} <= 0"
        assert back_results_dca['Total Return (DCA)'] != back_results_dca['Total Return (Lump-Sum)'], "Failed: DCA matches lump-sum"
        print("Test 4 (DCA backtest positive and different): PASSED")
        passed += 1
    except AssertionError as e:
        print(f"Test 4: FAILED - {e}")

    # Test 5: Inflation adjustment reduces value in simulation
    try:
        results_inf, _ = bootstrap_simulation(mock_returns, weights, num_simulations=10, time_horizon_years=1, initial_investment=10000, inflation_rate=0.05, monthly_contrib=0)
        results_no_inf, _ = bootstrap_simulation(mock_returns, weights, num_simulations=10, time_horizon_years=1, initial_investment=10000, inflation_rate=0.0, monthly_contrib=0)
        assert results_inf['Mean Final Value (Inflation-Adjusted, DCA)'] < results_no_inf['Mean Final Value (Inflation-Adjusted, DCA)'], "Failed: Inflation did not reduce value"
        print("Test 5 (Inflation reduces sim value): PASSED")
        passed += 1
    except AssertionError as e:
        print(f"Test 5: FAILED - {e}")

    print(f"\n{passed}/{total} tests passed.")

run_tests()