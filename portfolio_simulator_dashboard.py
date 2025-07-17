"""
Portfolio Simulator Dashboard

This is a Streamlit-based web dashboard for the Portfolio Simulator application.
It allows users to interactively adjust portfolio weights, simulation parameters,
and view results and visualizations locally.

New Features Added:
- Inflation Adjustment: Input for expected annual inflation rate; adjusts final values.
- Custom Assets: Allow adding custom tickers (UCITS ETFs) via text input.
- More Visuals: Line chart for historical cumulative returns of the portfolio and assets.
- Backtesting: Simulate historical portfolio performance over a selected period.
- Deployment Instructions: See below for deploying to Streamlit Community Cloud.
- DCA Simulation and Backtesting: Support for monthly/quarterly contributions in simulations and backtests.
- Fees and Taxes: Incorporate TER, transaction fees, and capital gains tax for more realistic projections.
- Automatic Rebalancing: Option to simulate rebalancing at specified frequency and threshold, with drift visualization.

To run locally: `streamlit run this_file.py`

For Deployment to Streamlit Community Cloud (as of July 2025):
1. Ensure your GitHub repo is public (or private with access granted).
2. Go to https://share.streamlit.io/, sign in with GitHub.
3. Click 'Deploy an app' > Connect to GitHub > Select your repo and branch.
4. Specify the file path (e.g., portfolio_simulator_dashboard.py).
5. Add a requirements.txt file in your repo with dependencies:
   streamlit
   yfinance
   numpy
   pandas
   matplotlib
   scipy
   plotly
6. Deploy—it's free for public apps. For secrets (if any), use the app's settings.

Dependencies: yfinance, numpy, pandas, matplotlib, streamlit, scipy, plotly
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import io
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go

# Default tickers for the assets (UCITS-compliant, EUR-traded)
DEFAULT_TICKERS = ['IWDA.AS', 'QDV5.DE', 'PPFB.DE', 'XEON.DE']  # MSCI World, MSCI India, Gold, Cash

# Default start date for historical data
DEFAULT_START_DATE = '2015-01-01'

@st.cache_data
def fetch_data(tickers, start_date, end_date=None):
    """
    Fetch historical adjusted close prices for given tickers.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Close'].dropna()
    if len(data) < 252:
        st.warning("Limited historical data available. Simulations may not be reliable.")
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
    Calculate historical portfolio statistics.
    """
    port_returns = np.dot(returns, weights)
    annual_return = np.mean(port_returns) * 252
    annual_vol = np.std(port_returns) * np.sqrt(252)
    rf_rate = returns[cash_ticker].mean() * 252 if cash_ticker in returns.columns else 0
    sharpe = (annual_return - rf_rate) / annual_vol if annual_vol != 0 else 0
    return annual_return, annual_vol, sharpe

def bootstrap_simulation(returns, weights, num_simulations, time_horizon_years, initial_investment, inflation_rate=0.0, monthly_contrib=0.0, contrib_frequency='monthly', transaction_fee=0.0, tax_rate=0.0, rebalance=False, rebalance_frequency='annual', rebalance_threshold=0.05):
    """
    Perform bootstrap Monte Carlo simulation with optional inflation adjustment, DCA, fees, taxes, and rebalancing.
    """
    days = int(252 * time_horizon_years)
    contrib_days = 21 if contrib_frequency == 'monthly' else 63  # Approx trading days per month/quarter
    rebalance_days = 252 if rebalance_frequency == 'annual' else 63  # Annual or quarterly
    sim_final_values = []
    sim_port_returns = []
    sim_final_values_lump = []  # For lump-sum comparison
    
    if len(returns) == 0:
        raise ValueError("No historical returns data available.")
    
    progress_bar = st.progress(0)
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
        
        # Update progress
        progress_bar.progress((i + 1) / num_simulations)
    
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

def backtest_portfolio(data, weights, monthly_contrib=0.0, contrib_frequency='monthly', transaction_fee=0.0, tax_rate=0.0, rebalance=False, rebalance_frequency='annual', rebalance_threshold=0.05):
    """
    Backtest portfolio over historical data with optional DCA, fees, taxes, and rebalancing.
    """
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

def optimize_weights(returns, cash_ticker=DEFAULT_TICKERS[3]):
    def objective(weights):
        ann_ret, ann_vol, sharpe = portfolio_stats(weights, returns, cash_ticker)
        return -sharpe  # Minimize negative Sharpe
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(len(returns.columns))]
    initial_guess = np.array([1/len(returns.columns)] * len(returns.columns))
    result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
    return result.x if result.success else None

# Streamlit Dashboard
st.title('Portfolio Simulator Dashboard')

st.markdown("""
This dashboard simulates the performance of a portfolio consisting of default assets:
- MSCI World (IWDA.AS)
- MSCI India (QDV5.DE)
- iShares Gold (PPFB.DE)
- Cash (XEON.DE)
Add custom tickers below for more flexibility.
""")

# Sidebar for inputs
st.sidebar.header('Simulation Parameters')

optimize = st.sidebar.checkbox('Optimize Weights for Max Sharpe')

custom_tickers = st.sidebar.text_input('Custom Tickers (comma-separated, e.g., VUSA.AS)', '')
all_tickers = DEFAULT_TICKERS.copy()
if custom_tickers:
    all_tickers.extend([t.strip() for t in custom_tickers.split(',')])

# Weights input (dynamic based on tickers)
st.sidebar.subheader('Portfolio Weights')
weights = []
cols = st.sidebar.columns(2)
for i, ticker in enumerate(all_tickers):
    col = cols[i % 2]
    weight = col.number_input(ticker, min_value=0.0, max_value=1.0, value=1.0/len(all_tickers), step=0.05)
    weights.append(weight)

weights = np.array(weights)
total_weight = np.sum(weights)
if total_weight != 1.0:
    weights = weights / total_weight
    st.sidebar.warning(f'Weights normalized to sum to 1: {weights.round(2)}')

horizon = st.sidebar.slider('Time Horizon (Years)', min_value=1, max_value=20, value=5)
simulations = st.sidebar.slider('Number of Simulations', min_value=100, max_value=10000, value=1000, step=100)  # Reduced default for speed
initial_investment = st.sidebar.number_input('Initial Investment (€)', min_value=0.0, value=100000.0, step=1000.0)  # Allow 0 for pure DCA
monthly_contrib = st.sidebar.number_input('Monthly Contribution (€)', min_value=0.0, value=0.0, step=100.0)
contrib_frequency = st.sidebar.selectbox('Contribution Frequency', ['monthly', 'quarterly'])
inflation_rate = st.sidebar.slider('Expected Annual Inflation Rate (%)', min_value=0.0, max_value=10.0, value=2.0, step=0.1) / 100

# New inputs for fees and taxes
ter = st.sidebar.slider('Annual TER (%)', min_value=0.0, max_value=1.0, value=0.2, step=0.05)
transaction_fee = st.sidebar.number_input('Transaction Fee per Buy (€)', min_value=0.0, value=5.0, step=1.0)
tax_rate = st.sidebar.slider('Capital Gains Tax Rate (%)', min_value=0.0, max_value=50.0, value=25.0, step=1.0) / 100

# New inputs for rebalancing
rebalance = st.sidebar.checkbox('Enable Rebalancing')
rebalance_frequency = st.sidebar.selectbox('Rebalancing Frequency', ['annual', 'quarterly'])
rebalance_threshold = st.sidebar.slider('Rebalancing Threshold (%)', min_value=0.0, max_value=20.0, value=5.0, step=0.5) / 100

start_date = st.sidebar.text_input('Start Date (YYYY-MM-DD)', DEFAULT_START_DATE)
backtest_end_date = st.sidebar.text_input('Backtest End Date (YYYY-MM-DD, optional)', '')

# Run button
if st.sidebar.button('Run Simulation'):
    try:
        # Fetch data
        data = fetch_data(all_tickers, start_date)
        returns = calculate_returns(data, ter)  # Adjust for TER

        # Apply optimization if checked
        if optimize:
            optimal_weights = optimize_weights(returns)
            if optimal_weights is not None:
                weights = optimal_weights
                st.sidebar.info(f'Optimized Weights: {dict(zip(all_tickers, weights.round(4)))}')
            else:
                raise ValueError("Weight optimization failed. Using default equal weights.")

        # Display allocation pie chart
        fig_pie = px.pie(values=weights, names=all_tickers, title='Portfolio Allocation')
        st.plotly_chart(fig_pie)

        # Run simulation with inflation, DCA, fees, tax, rebalancing
        results, sim_final_values = bootstrap_simulation(
            returns, weights, simulations, horizon, initial_investment, inflation_rate, monthly_contrib, contrib_frequency, transaction_fee, tax_rate, rebalance, rebalance_frequency, rebalance_threshold
        )
        
        # Display simulation results
        st.header('Simulation Results')
        col1, col2, col3 = st.columns(3)
        col1.metric('Historical Annual Return', f"{results['Historical Annual Return']:.2%}")
        col1.metric('Historical Annual Volatility', f"{results['Historical Annual Volatility']:.2%}")
        col1.metric('Historical Sharpe Ratio', f"{results['Historical Sharpe Ratio']:.2f}")
        
        col2.metric('Mean Final Value (Inflation-Adjusted, DCA)', f"€{results['Mean Final Value (Inflation-Adjusted, DCA)']:.2f}")
        col2.metric('Median Final Value (Inflation-Adjusted, DCA)', f"€{results['Median Final Value (Inflation-Adjusted, DCA)']:.2f}")
        col2.metric('Mean Final Value (Lump-Sum)', f"€{results['Mean Final Value (Lump-Sum Comparison)']:.2f}")
        
        col3.metric('Std Dev of Final Values (DCA)', f"€{results['Std Dev of Final Values (DCA)']:.2f}")
        col3.metric('95% VaR (Absolute Loss, DCA)', f"€{results['95% VaR (Absolute Loss, DCA)']:.2f}")
        col3.metric('95% CVaR (Absolute Loss, DCA)', f"€{results['95% CVaR (Absolute Loss, DCA)']:.2f}")
        col3.metric('Effective Cost Drag (%)', f"{results['Effective Cost Drag (%)']:.2f}%")
        
        # Plot simulation distribution
        st.header('Distribution of Outcomes')
        fig = plot_results(sim_final_values, horizon, results)
        st.pyplot(fig)

        # Historical performance plot
        st.header('Historical Performance')
        fig_hist = plot_historical_performance(data, weights, all_tickers)
        st.plotly_chart(fig_hist)

        # Weight drift plot
        st.header('Weight Drift Analysis')
        fig_drift = plot_weight_drift(returns, weights, rebalance, rebalance_frequency, rebalance_threshold)
        st.plotly_chart(fig_drift)

        # Backtesting with DCA, fees, tax, rebalancing
        st.header('Backtesting Results')
        backtest_end = backtest_end_date if backtest_end_date else None
        backtest_data = fetch_data(all_tickers, start_date, backtest_end)
        backtest_results = backtest_portfolio(backtest_data, weights, monthly_contrib, contrib_frequency, transaction_fee, tax_rate, rebalance, rebalance_frequency, rebalance_threshold)
        col1, col2 = st.columns(2)
        col1.metric('Total Historical Return (DCA)', f"{backtest_results['Total Return (DCA)']:.2%}")
        col1.metric('Total Historical Return (Lump-Sum)', f"{backtest_results['Total Return (Lump-Sum)']:.2%}")
        col1.metric('Annualized Return', f"{backtest_results['Annualized Return']:.2%}")
        col2.metric('Annualized Volatility', f"{backtest_results['Annualized Volatility']:.2%}")
        col2.metric('Sharpe Ratio', f"{backtest_results['Sharpe Ratio']:.2f}")

    except ValueError as e:
        st.error(str(e))

# CSV Download (shows after successful run)
if 'results' in locals():
    csv_buffer = io.StringIO()
    pd.DataFrame([results]).to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Simulation Results as CSV",
        data=csv_buffer.getvalue(),
        file_name="portfolio_simulation_results.csv",
        mime="text/csv"
    )