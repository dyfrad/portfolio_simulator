"""
Portfolio Simulator

Author: Mohit Saharan
Email: mohit@msaharan.com
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
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

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

# Initialize session state
if 'ran_simulation' not in st.session_state:
    st.session_state.ran_simulation = False

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

def bootstrap_simulation(returns, weights, num_simulations, time_horizon_years, initial_investment, inflation_rate=0.0, periodic_contrib=0.0, contrib_frequency='monthly', transaction_fee=0.0, tax_rate=0.0, rebalance=False, rebalance_frequency='annual', rebalance_threshold=0.05, shock_factors=None, base_invested=None):
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
    
    progress_bar = st.progress(0)
    for i in range(num_simulations):
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
        
        # Update progress
        progress_bar.progress((i + 1) / num_simulations)
    
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

def generate_pdf_report(all_tickers, weights, results, backtest_results, fig_pie, fig_hist, fig_dd, fig_drift, fig_dist, horizon):
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    y = height - 50

    c.drawString(100, y, "Portfolio Simulator Report")
    y -= 30

    c.drawString(100, y, "Portfolio Allocation:")
    y -= 20
    pie_img = io.BytesIO()
    fig_pie.write_image(pie_img, format='png')
    pie_img.seek(0)
    c.drawImage(ImageReader(pie_img), 100, y - 200, width=400, height=200)
    y -= 220

    c.drawString(100, y, "Simulation Results:")
    y -= 20
    for key, value in results.items():
        c.drawString(120, y, f"{key}: {value:.2f}" if isinstance(value, float) else f"{key}: {value}")
        y -= 15
    y -= 20

    c.drawString(100, y, "Backtesting Results:")
    y -= 20
    for key, value in backtest_results.items():
        c.drawString(120, y, f"{key}: {value:.2%}" if 'Return' in key or 'Volatility' in key or 'Drawdown' in key else f"{key}: {value:.2f}")
        y -= 15

    c.showPage()
    y = height - 50

    c.drawString(100, y, "Charts:")
    y -= 20

    # Historical Performance
    hist_img = io.BytesIO()
    fig_hist.write_image(hist_img, format='png')
    hist_img.seek(0)
    c.drawImage(ImageReader(hist_img), 100, y - 200, width=400, height=200)
    y -= 220

    # Drawdown
    dd_img = io.BytesIO()
    fig_dd.write_image(dd_img, format='png')
    dd_img.seek(0)
    c.drawImage(ImageReader(dd_img), 100, y - 200, width=400, height=200)
    y -= 220

    c.showPage()
    y = height - 50

    # Weight Drift
    drift_img = io.BytesIO()
    fig_drift.write_image(drift_img, format='png')
    drift_img.seek(0)
    c.drawImage(ImageReader(drift_img), 100, y - 200, width=400, height=200)
    y -= 220

    # Distribution
    dist_img = io.BytesIO()
    fig_dist.savefig(dist_img, format='png')
    dist_img.seek(0)
    c.drawImage(ImageReader(dist_img), 100, y - 200, width=400, height=200)

    c.save()
    buffer.seek(0)
    return buffer

# Explanations for tooltips
explanations = {
    'Historical Annual Return': "The average annual return based on historical data.",
    'Historical Annual Volatility': "The standard deviation of annual returns, measuring risk.",
    'Historical Sharpe Ratio': "Risk-adjusted return: (return - risk-free rate) / volatility.",
    'Historical Sortino Ratio': "Similar to Sharpe but only considers downside volatility.",
    'Historical Max Drawdown': "The largest peak-to-trough decline in portfolio value.",
    'Mean Final Value (Inflation-Adjusted, DCA)': "Average ending value after simulations, adjusted for inflation, using Dollar-Cost Averaging.",
    'Median Final Value (Inflation-Adjusted, DCA)': "Median ending value after simulations, adjusted for inflation, using DCA.",
    'Mean Final Value (Lump-Sum)': "Average ending value if invested all at once, for comparison.",
    'Std Dev of Final Values (DCA)': "Variability in the simulated ending values.",
    '95% VaR (Absolute Loss, DCA)': "There is a 5% chance of losing more than this amount over the horizon.",
    '95% CVaR (Absolute Loss, DCA)': "The average loss in the worst 5% of scenarios.",
    'Effective Cost Drag (%)': "The percentage reduction in returns due to fees and taxes.",
    'Total Historical Return (DCA)': "Total return from backtest using Dollar-Cost Averaging.",
    'Total Historical Return (Lump-Sum)': "Total return from backtest if invested all at once.",
    'Annualized Return': "Average annual return from historical backtest.",
    'Annualized Volatility': "Annualized standard deviation of returns from backtest.",
    'Sharpe Ratio': "Risk-adjusted return from backtest.",
    'Sortino Ratio': "Downside risk-adjusted return from backtest.",
    'Max Drawdown': "Largest decline in backtest."
}

# Streamlit Dashboard
st.title('Portfolio Simulator')

st.markdown("""
This dashboard simulates the performance of a portfolio consisting of default assets:
- MSCI World (IWDA.AS)
- MSCI India (QDV5.DE)
- iShares Gold (PPFB.DE)
- Cash (XEON.DE)
Add custom tickers below for more flexibility.
""")

# Sidebar for inputs
# Author information
st.sidebar.markdown("**Author Information**")
st.sidebar.markdown("**Mohit Saharan**")
st.sidebar.markdown("mohit@msaharan.com")
st.sidebar.markdown("---")

st.sidebar.header('Simulation Parameters')

optimize = st.sidebar.checkbox('Optimize Weights for Max Sharpe')

custom_tickers = st.sidebar.text_input('Custom Tickers (comma-separated, e.g., VUSA.AS)', '')
all_tickers = DEFAULT_TICKERS.copy()
if custom_tickers:
    all_tickers.extend([t.strip() for t in custom_tickers.split(',')])

uploaded_file = st.sidebar.file_uploader("Upload Portfolio CSV (Ticker, Shares, Cost Basis) or Transactions CSV", type="csv")

initial_investment = 100000.0
base_invested = None
weights = None

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if set(['Ticker', 'Shares', 'Cost Basis']).issubset(df.columns):
        # Holdings CSV
        required_cols = ['Ticker', 'Shares', 'Cost Basis']
        df_group = df.groupby('Ticker').agg({'Shares': 'sum', 'Cost Basis': 'sum'}).reset_index()
        try:
            current_prices = yf.download(list(df_group['Ticker']), period='1d')['Close'].iloc[-1]
            df_group['Current Price'] = df_group['Ticker'].map(current_prices)
            df_group['Current Value'] = df_group['Shares'] * df_group['Current Price']
            total_value = df_group['Current Value'].sum()
            df_group['Weight'] = df_group['Current Value'] / total_value
            df_group['Unrealized Gain'] = df_group['Current Value'] - df_group['Cost Basis']
            st.header('Current Portfolio')
            st.dataframe(df_group)
            all_tickers = list(df_group['Ticker'])
            weights = df_group['Weight'].values
            initial_investment = total_value
            base_invested = df_group['Cost Basis'].sum()
        except Exception as e:
            st.error(f"Error fetching current prices: {e}")
    elif set(['Date', 'Product', 'ISIN', 'Quantity', 'Price', 'Local value', 'Transaction and/or third']).issubset(df.columns):
        # Transactions CSV
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
        df = df.dropna(subset=['Date'])
        df['Local value'] = pd.to_numeric(df['Local value'], errors='coerce')
        df['Transaction and/or third'] = pd.to_numeric(df['Transaction and/or third'], errors='coerce').fillna(0)
        df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
        df = df.sort_values(['Date', 'Time'])
        
        holdings = {}
        for _, row in df.iterrows():
            if pd.isna(row['ISIN']): continue
            isin = row['ISIN']
            if isin not in holdings:
                holdings[isin] = {'shares': 0.0, 'total_cost': 0.0, 'product': row['Product']}
            quantity = row['Quantity']
            local_value = row['Local value']
            fees = row['Transaction and/or third']
            if quantity > 0:  # buy
                cost = -local_value - fees
                holdings[isin]['total_cost'] += cost
                holdings[isin]['shares'] += quantity
            elif quantity < 0:  # sell
                if holdings[isin]['shares'] > 0:
                    avg_cost = holdings[isin]['total_cost'] / holdings[isin]['shares']
                    cost_reduction = avg_cost * (-quantity)
                    holdings[isin]['total_cost'] -= cost_reduction
                holdings[isin]['shares'] += quantity
            if abs(holdings[isin]['shares']) < 1e-6:
                holdings[isin]['shares'] = 0.0
                holdings[isin]['total_cost'] = 0.0
        
        df_holdings = pd.DataFrame(
            [{'ISIN': k, 'Ticker': ISIN_TO_TICKER.get(k), 'Shares': v['shares'], 'Cost Basis': v['total_cost'], 'Product': v['product']}
             for k, v in holdings.items() if v['shares'] > 0]
        )
        st.header('Computed Portfolio Holdings from Transactions')
        st.dataframe(df_holdings)
        
        valid_holdings = df_holdings[df_holdings['Ticker'].notna()]
        all_tickers = valid_holdings['Ticker'].tolist()
        if all_tickers:
            try:
                current_prices = yf.download(all_tickers, period='1d')['Close'].iloc[-1]
                df_holdings['Current Price'] = df_holdings['Ticker'].map(current_prices)
                df_holdings['Current Value'] = df_holdings['Shares'] * df_holdings['Current Price']
                total_value = df_holdings['Current Value'].sum()
                df_holdings['Weight'] = df_holdings['Current Value'] / total_value
                df_holdings['Unrealized Gain'] = df_holdings['Current Value'] - df_holdings['Cost Basis']
                st.dataframe(df_holdings)
                weights = df_holdings['Weight'].values
                initial_investment = total_value
                base_invested = df_holdings['Cost Basis'].sum()
            except Exception as e:
                st.error(f"Error fetching current prices: {e}")
        else:
            st.error("No valid tickers found for holdings.")
    else:
        st.error("CSV must contain either columns: Ticker, Shares, Cost Basis or transaction columns: Date, Product, ISIN, Quantity, Price, Local value, Transaction and/or third")

# Weights input (dynamic based on tickers) - only if not uploaded
if uploaded_file is None:
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

horizon = st.sidebar.number_input('Time Horizon (1 - 10 Years, in steps of 0.25 year):', min_value=1.0, max_value=10.0, value=5.0, step=0.25)
simulations = st.sidebar.number_input('Number of Simulations (100 - 10000, in steps of 100)', min_value=100, max_value=10000, value=1000, step=100)
initial_investment = st.sidebar.number_input('Initial Investment (€)', min_value=0.0, value=100000.0 if uploaded_file is None else initial_investment, step=1000.0, disabled=uploaded_file is not None)  # Allow 0 for pure DCA
periodic_contrib = st.sidebar.number_input('Periodic Contribution (€)', min_value=0.0, value=0.0, step=100.0)
contrib_frequency = st.sidebar.selectbox('Contribution Frequency', ['monthly', 'quarterly'])
inflation_rate = st.sidebar.number_input('Expected Annual Inflation Rate (0 - 20 %, in steps of 0.1%)', min_value=0.0, max_value=20.0, value=2.0, step=0.1) / 100

# New inputs for fees and taxes
ter = st.sidebar.number_input('Annual TER (0 - 2 %, in steps of 0.01%)', min_value=0.0, max_value=2.0, value=0.2, step=0.01)
transaction_fee = st.sidebar.number_input('Transaction Fee per Buy (€)', min_value=0.0, value=5.0, step=1.0)
tax_rate = st.sidebar.number_input('Capital Gains Tax Rate (0 - 50 %, in steps of 0.1%)', min_value=0.0, max_value=50.0, value=25.0, step=0.1) / 100

# New inputs for rebalancing
rebalance = st.sidebar.checkbox('Enable Rebalancing')
rebalance_frequency = st.sidebar.selectbox('Rebalancing Frequency', ['quarterly', 'annual'])
rebalance_threshold = st.sidebar.number_input('Rebalancing Threshold (0 - 20 %, in steps of 0.5%)', min_value=0.0, max_value=20.0, value=5.0, step=0.5) / 100

# New input for stress scenarios
stress_scenario = st.sidebar.selectbox('Stress Scenario', ['None', '2008 Recession', 'COVID Crash', '2022 Bear Market', 'Inflation Spike'])

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

        # Handle stress scenario
        shock_factors = None
        if stress_scenario != 'None':
            scenarios = {
                '2008 Recession': [-0.40, -0.55, 0.05, 0.02],
                'COVID Crash': [-0.34, -0.35, 0.15, 0.00],
                '2022 Bear Market': [-0.18, -0.08, 0.00, 0.00],
                'Inflation Spike': [0.05, 0.05, 0.30, 0.05],
            }
            shock_factors = scenarios[stress_scenario]
            if len(all_tickers) > len(shock_factors):
                avg_stock_shock = np.mean(shock_factors[:2])
                shock_factors += [avg_stock_shock] * (len(all_tickers) - len(shock_factors))
            if len(shock_factors) > len(all_tickers):
                shock_factors = shock_factors[:len(all_tickers)]
            shock_factors = np.array(shock_factors)
            if stress_scenario == 'Inflation Spike':
                inflation_rate = 0.08
                st.info('Inflation rate overridden to 8% for Inflation Spike scenario.')
            st.info(f'Simulating under {stress_scenario} stress conditions.')

        # Display allocation pie chart
        fig_pie = px.pie(values=weights, names=all_tickers, title='Portfolio Allocation')

        # Run simulation with inflation, DCA, fees, tax, rebalancing, stress
        results, sim_final_values = bootstrap_simulation(
            returns, weights, simulations, horizon, initial_investment, inflation_rate, periodic_contrib, contrib_frequency, transaction_fee, tax_rate, rebalance, rebalance_frequency, rebalance_threshold, shock_factors, base_invested
        )
        
        fig_dist = plot_results(sim_final_values, horizon, results)

        fig_hist = plot_historical_performance(data, weights, all_tickers)

        fig_dd = plot_drawdowns(returns, weights)

        fig_drift = plot_weight_drift(returns, weights, rebalance, rebalance_frequency, rebalance_threshold)

        # Backtesting with DCA, fees, tax, rebalancing
        backtest_end = backtest_end_date if backtest_end_date else None
        backtest_data = fetch_data(all_tickers, start_date, backtest_end)
        backtest_results = backtest_portfolio(backtest_data, weights, initial_investment, periodic_contrib, contrib_frequency, transaction_fee, tax_rate, rebalance, rebalance_frequency, rebalance_threshold)

        # Store in session state
        st.session_state.results = results
        st.session_state.sim_final_values = sim_final_values
        st.session_state.fig_pie = fig_pie
        st.session_state.fig_dist = fig_dist
        st.session_state.fig_hist = fig_hist
        st.session_state.fig_dd = fig_dd
        st.session_state.fig_drift = fig_drift
        st.session_state.backtest_results = backtest_results
        st.session_state.all_tickers = all_tickers
        st.session_state.weights = weights
        st.session_state.horizon = horizon
        st.session_state.ran_simulation = True

    except ValueError as e:
        st.error(str(e))

if st.session_state.ran_simulation:
    # Display allocation pie chart
    st.plotly_chart(st.session_state.fig_pie)

    # Display simulation results
    st.header('Simulation Results')
    col1, col2, col3 = st.columns(3)
    col1.metric('Historical Annual Return', f"{st.session_state.results['Historical Annual Return']:.2%}", help=explanations['Historical Annual Return'])
    col1.metric('Historical Annual Volatility', f"{st.session_state.results['Historical Annual Volatility']:.2%}", help=explanations['Historical Annual Volatility'])
    col1.metric('Historical Sharpe Ratio', f"{st.session_state.results['Historical Sharpe Ratio']:.2f}", help=explanations['Historical Sharpe Ratio'])
    col1.metric('Historical Sortino Ratio', f"{st.session_state.results['Historical Sortino Ratio']:.2f}", help=explanations['Historical Sortino Ratio'])
    col1.metric('Historical Max Drawdown', f"{st.session_state.results['Historical Max Drawdown']:.2%}", help=explanations['Historical Max Drawdown'])
    
    col2.metric('Mean Final Value (Inflation-Adjusted, DCA)', f"€{st.session_state.results['Mean Final Value (Inflation-Adjusted, DCA)']:.2f}", help=explanations['Mean Final Value (Inflation-Adjusted, DCA)'])
    col2.metric('Median Final Value (Inflation-Adjusted, DCA)', f"€{st.session_state.results['Median Final Value (Inflation-Adjusted, DCA)']:.2f}", help=explanations['Median Final Value (Inflation-Adjusted, DCA)'])
    col2.metric('Mean Final Value (Lump-Sum)', f"€{st.session_state.results['Mean Final Value (Lump-Sum Comparison)']:.2f}", help=explanations['Mean Final Value (Lump-Sum)'])
    
    col3.metric('Std Dev of Final Values (DCA)', f"€{st.session_state.results['Std Dev of Final Values (DCA)']:.2f}", help=explanations['Std Dev of Final Values (DCA)'])
    col3.metric('95% VaR (Absolute Loss, DCA)', f"€{st.session_state.results['95% VaR (Absolute Loss, DCA)']:.2f}", help=explanations['95% VaR (Absolute Loss, DCA)'])
    col3.metric('95% CVaR (Absolute Loss, DCA)', f"€{st.session_state.results['95% CVaR (Absolute Loss, DCA)']:.2f}", help=explanations['95% CVaR (Absolute Loss, DCA)'])
    col3.metric('Effective Cost Drag (%)', f"{st.session_state.results['Effective Cost Drag (%)']:.2f}%", help=explanations['Effective Cost Drag (%)'])
    
    # Plot simulation distribution
    st.header('Distribution of Outcomes')
    st.pyplot(st.session_state.fig_dist)

    # Historical performance plot
    st.header('Historical Performance')
    st.plotly_chart(st.session_state.fig_hist)

    # Historical drawdown plot
    st.header('Historical Drawdown')
    st.plotly_chart(st.session_state.fig_dd)

    # Weight drift plot
    st.header('Weight Drift Analysis')
    st.plotly_chart(st.session_state.fig_drift)

    # Backtesting results
    st.header('Backtesting Results')
    col1, col2 = st.columns(2)
    col1.metric('Total Historical Return (DCA)', f"{st.session_state.backtest_results['Total Return (DCA)']:.2%}", help=explanations['Total Historical Return (DCA)'])
    col1.metric('Total Historical Return (Lump-Sum)', f"{st.session_state.backtest_results['Total Return (Lump-Sum)']:.2%}", help=explanations['Total Historical Return (Lump-Sum)'])
    col1.metric('Annualized Return', f"{st.session_state.backtest_results['Annualized Return']:.2%}", help=explanations['Annualized Return'])
    col2.metric('Annualized Volatility', f"{st.session_state.backtest_results['Annualized Volatility']:.2%}", help=explanations['Annualized Volatility'])
    col2.metric('Sharpe Ratio', f"{st.session_state.backtest_results['Sharpe Ratio']:.2f}", help=explanations['Sharpe Ratio'])
    col2.metric('Sortino Ratio', f"{st.session_state.backtest_results['Sortino Ratio']:.2f}", help=explanations['Sortino Ratio'])
    col2.metric('Max Drawdown', f"{st.session_state.backtest_results['Max Drawdown']:.2%}", help=explanations['Max Drawdown'])

    # Generate PDF Report Button
    if st.button('Generate PDF Report'):
        pdf_buffer = generate_pdf_report(
            st.session_state.all_tickers, 
            st.session_state.weights, 
            st.session_state.results, 
            st.session_state.backtest_results, 
            st.session_state.fig_pie, 
            st.session_state.fig_hist, 
            st.session_state.fig_dd, 
            st.session_state.fig_drift, 
            st.session_state.fig_dist, 
            st.session_state.horizon
        )
        st.download_button(
            label="Download PDF Report",
            data=pdf_buffer,
            file_name="portfolio_report.pdf",
            mime="application/pdf"
        )

# CSV Download
if st.session_state.ran_simulation:
    csv_buffer = io.StringIO()
    pd.DataFrame([st.session_state.results]).to_csv(csv_buffer, index=False)
    st.download_button(
        label="Download Simulation Results as CSV",
        data=csv_buffer.getvalue(),
        file_name="portfolio_simulation_results.csv",
        mime="text/csv"
    )