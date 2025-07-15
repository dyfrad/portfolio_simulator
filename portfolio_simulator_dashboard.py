"""
Portfolio Simulator Dashboard

This is a Streamlit-based web dashboard for the Portfolio Simulator application.
It allows users to interactively adjust portfolio weights, simulation parameters,
and view results and visualizations locally.

To run: Install Streamlit if needed (`pip install streamlit`), then `streamlit run this_file.py`

Features:
- Interactive inputs for weights (auto-normalized), horizon, simulations, initial investment.
- Fetches historical data on demand.
- Displays simulation results and histogram plot.
- Modern, responsive UI with Streamlit.

For continued development:
- Add caching for data fetching (st.cache_data).
- Implement advanced features like weight optimization.
- Export results to CSV.
- Deploy to cloud (e.g., Streamlit Sharing) when ready.

Dependencies: yfinance, numpy, pandas, matplotlib, streamlit
"""

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import io
from scipy.optimize import minimize
import plotly.express as px




# Default tickers for the assets (UCITS-compliant, EUR-traded)
DEFAULT_TICKERS = ['IWDA.AS', 'QDV5.DE', 'PPFB.DE', 'XEON.DE']  # MSCI World, MSCI India, Gold, Cash

# Default start date for historical data
DEFAULT_START_DATE = '2015-01-01'

@st.cache_data
def fetch_data(tickers, start_date):
    """
    Fetch historical adjusted close prices for given tickers.
    """
    data = yf.download(tickers, start=start_date)['Close'].dropna()
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
    rf_rate = returns[cash_ticker].mean() * 252
    sharpe = (annual_return - rf_rate) / annual_vol if annual_vol != 0 else 0
    return annual_return, annual_vol, sharpe

def bootstrap_simulation(returns, weights, num_simulations, time_horizon_years, initial_investment):
    """
    Perform bootstrap Monte Carlo simulation.
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
    Generate histogram plot of simulated final portfolio values.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(sim_final_values, bins=50, alpha=0.75, color='blue')
    ax.set_title(f'Distribution of Simulated Portfolio Values after {time_horizon_years} Years')
    ax.set_xlabel('Final Portfolio Value')
    ax.set_ylabel('Frequency')
    ax.axvline(results['Mean Final Value'], color='red', linestyle='--', label='Mean')
    ax.axvline(results['Median Final Value'], color='green', linestyle='--', label='Median')
    ax.legend()
    return fig

def optimize_weights(returns, cash_ticker=DEFAULT_TICKERS[3]):
    def objective(weights):
        ann_ret, ann_vol, sharpe = portfolio_stats(weights, returns, cash_ticker)
        return -sharpe  # Minimize negative Sharpe
    
    constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
    bounds = [(0, 1) for _ in range(len(DEFAULT_TICKERS))]
    initial_guess = np.array([1/len(DEFAULT_TICKERS)] * len(DEFAULT_TICKERS))
    result = minimize(objective, initial_guess, bounds=bounds, constraints=constraints)
    return result.x if result.success else None

# Streamlit Dashboard
st.title('Portfolio Simulator Dashboard')

st.markdown("""
This dashboard simulates the performance of a portfolio consisting of:
- MSCI World (IWDA.AS)
- MSCI India (QDV5.DE)
- iShares Gold (PPFB.DE)
- Cash (XEON.DE)
""")

# Sidebar for inputs
st.sidebar.header('Simulation Parameters')

optimize = st.sidebar.checkbox('Optimize Weights for Max Sharpe')

# Weights input
st.sidebar.subheader('Portfolio Weights')
col1, col2 = st.sidebar.columns(2)
weight_world = col1.number_input('MSCI World', min_value=0.0, max_value=1.0, value=0.40, step=0.05)
weight_india = col1.number_input('MSCI India', min_value=0.0, max_value=1.0, value=0.30, step=0.05)
weight_gold = col2.number_input('Gold', min_value=0.0, max_value=1.0, value=0.20, step=0.05)
weight_cash = col2.number_input('Cash', min_value=0.0, max_value=1.0, value=0.10, step=0.05)

weights = np.array([weight_world, weight_india, weight_gold, weight_cash])
total_weight = np.sum(weights)
if total_weight != 1.0:
    weights = weights / total_weight
    st.sidebar.warning(f'Weights normalized to sum to 1: {weights.round(2)}')

fig_pie = px.pie(values=weights, names=DEFAULT_TICKERS, title='Portfolio Allocation')
st.plotly_chart(fig_pie)    

horizon = st.sidebar.slider('Time Horizon (Years)', min_value=1, max_value=20, value=5)
simulations = st.sidebar.slider('Number of Simulations', min_value=100, max_value=10000, value=5000, step=100)
initial_investment = st.sidebar.number_input('Initial Investment (€)', min_value=1000.0, value=100000.0, step=1000.0)

start_date = st.sidebar.text_input('Start Date (YYYY-MM-DD)', DEFAULT_START_DATE)


# Fetch data (could cache this)
data = fetch_data(DEFAULT_TICKERS, DEFAULT_START_DATE)
returns = calculate_returns(data)

# Run button
if st.sidebar.button('Run Simulation'):
    try:
        results, sim_final_values = bootstrap_simulation(
            returns, weights, simulations, horizon, initial_investment
        )
        
        # Display results
        st.header('Simulation Results')
        col1, col2, col3 = st.columns(3)
        col1.metric('Historical Annual Return', f"{results['Historical Annual Return']:.2%}")
        col1.metric('Historical Annual Volatility', f"{results['Historical Annual Volatility']:.2%}")
        col1.metric('Historical Sharpe Ratio', f"{results['Historical Sharpe Ratio']:.2f}")
        
        col2.metric('Mean Final Value', f"€{results['Mean Final Value']:.2f}")
        col2.metric('Median Final Value', f"€{results['Median Final Value']:.2f}")
        col2.metric('Std Dev of Final Values', f"€{results['Std Dev of Final Values']:.2f}")
        
        col3.metric('95% VaR (Absolute Loss)', f"€{results['95% VaR (Absolute Loss)']:.2f}")
        col3.metric('95% CVaR (Absolute Loss)', f"€{results['95% CVaR (Absolute Loss)']:.2f}")
        
        # Plot
        st.header('Distribution of Outcomes')
        fig = plot_results(sim_final_values, horizon, results)
        st.pyplot(fig)
    except ValueError as e:
        st.error(str(e))

if 'results' in locals():
    csv_buffer = io.StringIO()
    pd.DataFrame([results]).to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Results as CSV",
        data=csv_buffer.getvalue(),
        file_name="portfolio_simulation_results.csv",
        mime="text/csv"
    )