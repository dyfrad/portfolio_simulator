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

def calculate_returns(data):
    """
    Calculate daily returns from price data.
    """
    return data.pct_change().dropna()

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

def bootstrap_simulation(returns, weights, num_simulations, time_horizon_years, initial_investment, inflation_rate=0.0):
    """
    Perform bootstrap Monte Carlo simulation with optional inflation adjustment.
    """
    days = int(252 * time_horizon_years)
    sim_final_values = []
    sim_port_returns = []
    
    if len(returns) == 0:
        raise ValueError("No historical returns data available.")
    
    with st.spinner('Running simulations...'):
        for _ in range(num_simulations):
            boot_sample = returns.sample(days, replace=True)
            boot_port_returns = np.dot(boot_sample, weights)
            compounded_return = np.prod(1 + boot_port_returns) - 1
            # Adjust for inflation
            adjusted_return = (1 + compounded_return) / (1 + inflation_rate)**time_horizon_years - 1
            final_value = initial_investment * (1 + adjusted_return)
            sim_final_values.append(final_value)
            sim_port_returns.append(adjusted_return)
    
    sim_final_values = np.array(sim_final_values)
    sim_port_returns = np.array(sim_port_returns)
    
    mean_final = np.mean(sim_final_values)
    median_final = np.median(sim_final_values)
    std_final = np.std(sim_final_values)
    var_95 = np.percentile(sim_port_returns, 5) * initial_investment
    cvar_95 = np.mean(sim_port_returns[sim_port_returns <= np.percentile(sim_port_returns, 5)]) * initial_investment
    
    hist_return, hist_vol, hist_sharpe = portfolio_stats(weights, returns)
    
    results = {
        'Mean Final Value (Inflation-Adjusted)': mean_final,
        'Median Final Value (Inflation-Adjusted)': median_final,
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
    Generate histogram plot of simulated final portfolio values.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sim_final_values, bins=50, alpha=0.75, color='blue')
    ax.set_title(f'Distribution of Simulated Portfolio Values after {time_horizon_years} Years (Inflation-Adjusted)')
    ax.set_xlabel('Final Portfolio Value')
    ax.set_ylabel('Frequency')
    ax.axvline(results['Mean Final Value (Inflation-Adjusted)'], color='red', linestyle='--', label='Mean')
    ax.axvline(results['Median Final Value (Inflation-Adjusted)'], color='green', linestyle='--', label='Median')
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

def backtest_portfolio(data, weights):
    """
    Backtest portfolio over historical data.
    """
    returns = calculate_returns(data)
    port_returns = np.dot(returns, weights)
    cum_port_returns = (1 + port_returns).cumprod()[-1] - 1
    ann_return, ann_vol, sharpe = portfolio_stats(weights, returns)
    return {
        'Total Return': cum_port_returns,
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
simulations = st.sidebar.slider('Number of Simulations', min_value=100, max_value=10000, value=5000, step=100)
initial_investment = st.sidebar.number_input('Initial Investment (€)', min_value=1000.0, value=100000.0, step=1000.0)
inflation_rate = st.sidebar.slider('Expected Annual Inflation Rate (%)', min_value=0.0, max_value=10.0, value=2.0, step=0.1) / 100

start_date = st.sidebar.text_input('Start Date (YYYY-MM-DD)', DEFAULT_START_DATE)
backtest_end_date = st.sidebar.text_input('Backtest End Date (YYYY-MM-DD, optional)', '')

# Run button
if st.sidebar.button('Run Simulation'):
    try:
        # Fetch data
        data = fetch_data(all_tickers, start_date)
        returns = calculate_returns(data)

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

        # Run simulation with inflation
        results, sim_final_values = bootstrap_simulation(
            returns, weights, simulations, horizon, initial_investment, inflation_rate
        )
        
        # Display simulation results
        st.header('Simulation Results')
        col1, col2, col3 = st.columns(3)
        col1.metric('Historical Annual Return', f"{results['Historical Annual Return']:.2%}")
        col1.metric('Historical Annual Volatility', f"{results['Historical Annual Volatility']:.2%}")
        col1.metric('Historical Sharpe Ratio', f"{results['Historical Sharpe Ratio']:.2f}")
        
        col2.metric('Mean Final Value (Inflation-Adjusted)', f"€{results['Mean Final Value (Inflation-Adjusted)']:.2f}")
        col2.metric('Median Final Value (Inflation-Adjusted)', f"€{results['Median Final Value (Inflation-Adjusted)']:.2f}")
        col2.metric('Std Dev of Final Values', f"€{results['Std Dev of Final Values']:.2f}")
        
        col3.metric('95% VaR (Absolute Loss)', f"€{results['95% VaR (Absolute Loss)']:.2f}")
        col3.metric('95% CVaR (Absolute Loss)', f"€{results['95% CVaR (Absolute Loss)']:.2f}")
        
        # Plot simulation distribution
        st.header('Distribution of Outcomes')
        fig = plot_results(sim_final_values, horizon, results)
        st.pyplot(fig)

        # Historical performance plot
        st.header('Historical Performance')
        fig_hist = plot_historical_performance(data, weights, all_tickers)
        st.plotly_chart(fig_hist)

        # Backtesting
        st.header('Backtesting Results')
        backtest_end = backtest_end_date if backtest_end_date else None
        backtest_data = fetch_data(all_tickers, start_date, backtest_end)
        backtest_results = backtest_portfolio(backtest_data, weights)
        col1, col2 = st.columns(2)
        col1.metric('Total Historical Return', f"{backtest_results['Total Return']:.2%}")
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