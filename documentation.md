# Portfolio Simulator Dashboard Documentation

## Introduction

The Portfolio Simulator Dashboard is a web-based application built using Streamlit, designed to simulate and analyze the performance of investment portfolios. It focuses on UCITS-compliant ETFs traded in EUR, allowing users to model portfolios with assets such as global equities (e.g., MSCI World via IWDA.AS), emerging markets (e.g., MSCI India via QDV5.DE), commodities (e.g., gold via PPFB.DE), and cash equivalents (e.g., XEON.DE). Users can customize portfolios by adding tickers, uploading holdings or transaction CSVs, and adjusting parameters like weights, time horizons, contributions, inflation, fees, taxes, rebalancing, and stress scenarios.

The application performs bootstrap Monte Carlo simulations for forward-looking projections, historical backtesting, portfolio optimization, and generates visualizations and reports. It incorporates realistic elements such as Dollar-Cost Averaging (DCA), Total Expense Ratio (TER), transaction fees, capital gains taxes, automatic rebalancing, and predefined stress tests.

This documentation provides a detailed explanation of the financial concepts, mathematical formulations, and programming implementation. Code snippets are included for key functions, with explanations of logic and potential edge cases.

## Dependencies and Setup

### Required Libraries
The application relies on the following Python libraries:
- `yfinance`: For fetching historical financial data from Yahoo Finance.
- `numpy`: For numerical computations, including array operations and statistics.
- `pandas`: For data manipulation, time series handling, and DataFrame operations.
- `matplotlib`: For static plotting (e.g., histograms).
- `streamlit`: For building the interactive web dashboard.
- `scipy`: For optimization (e.g., minimizing negative Sharpe ratio).
- `plotly`: For interactive visualizations (e.g., line charts, pie charts).
- `reportlab`: For generating PDF reports.
- `kaleido`: For exporting Plotly figures to static images in PDFs.

These are listed in a `requirements.txt` file for deployment. No additional installations are needed beyond these, as the code interpreter environment includes them.

### Running the Application
To run locally:
```
streamlit run portfolio_simulator_dashboard.py
```
This launches a web server at `http://localhost:8501`.

For deployment to Streamlit Community Cloud (as of July 2025):
1. Host the code in a GitHub repository (public or private with access granted).
2. Navigate to https://share.streamlit.io/, sign in with GitHub.
3. Deploy the app by selecting the repository, branch, and file path.
4. Ensure `requirements.txt` is in the repository root.
5. The service is free for public apps; manage secrets via app settings if needed.

Potential issues: Yahoo Finance API rate limits may cause delays for large datasets. Handle by caching data with `@st.cache_data`.

## Data Fetching and Preprocessing

### Financial Concept: Historical Price Data
Historical adjusted close prices are used to compute returns, reflecting dividends and splits. This assumes efficient markets where past performance informs future volatility but not expected returns (per the Efficient Market Hypothesis, Fama, 1970). Data is fetched from Yahoo Finance, which provides reliable, free access to ETF data.

Mathematical Detail: Adjusted close \( P_{t, adj} \) accounts for corporate actions:
\[
P_{t, adj} = P_t \times \prod_{k=1}^{m} (1 + d_k + s_k)
\]
where \( d_k \) are dividends and \( s_k \) are split factors over \( m \) events.

Reference: Yahoo Finance documentation; "A Random Walk Down Wall Street" by Burton Malkiel for market efficiency.

### Code Implementation
The `fetch_data` function downloads data:
```python
@st.cache_data
def fetch_data(tickers, start_date, end_date=None):
    data = yf.download(tickers, start=start_date, end=end_date)['Close'].dropna()
    if len(data) < 252:
        st.warning("Limited historical data available. Simulations may not be reliable.")
    return data
```
- **Logic**: Uses `yfinance.download` for multi-ticker data. Drops NaNs to ensure complete datasets. Caches for performance. Warns if less than one year of data (252 trading days), as short histories reduce statistical reliability.
- **Edge Cases**: Invalid tickers raise exceptions (handled upstream). No data before ETF inception leads to empty DataFrames, triggering errors in simulations.

Returns are calculated with TER adjustment:
```python
def calculate_returns(data, ter=0.0):
    returns = data.pct_change().dropna()
    daily_ter = (ter / 100) / 252
    returns -= daily_ter
    return returns
```
- **Logic**: Computes daily percentage changes. Subtracts daily TER (annual TER prorated over 252 trading days) to model expense drag.
- **Financial Concept**: TER represents ongoing fund costs (management fees, etc.). Subtracting it assumes constant drag, simplifying reality where TER varies.
- **Math**: Daily return \( r_d = \frac{P_{t} - P_{t-1}}{P_{t-1}} - \frac{TER}{252} \).

## Portfolio Statistics

### Financial Concepts
- **Annual Return**: Geometric mean of returns, annualized.
- **Volatility**: Standard deviation of returns, measuring dispersion (risk).
- **Sharpe Ratio**: Excess return per unit of volatility, using cash as risk-free rate.
- **Sortino Ratio**: Similar to Sharpe but penalizes only downside volatility.
- **Max Drawdown**: Largest peak-to-trough loss, indicating worst-case resilience.

Mathematical Formulations:
- Annual Return: \( R_a = \bar{r_d} \times 252 \), where \( \bar{r_d} \) is mean daily return.
- Volatility: \( \sigma_a = \sigma_d \sqrt{252} \), assuming i.i.d. returns (not always true due to autocorrelation).
- Sharpe: \( S = \frac{R_a - R_f}{\sigma_a} \), \( R_f \) from cash returns.
- Sortino: \( So = \frac{R_a - R_f}{\sigma_{down}} \), where \( \sigma_{down} = \sqrt{\frac{\sum (r_d < 0) r_d^2}{n}} \sqrt{252} \).
- Max Drawdown: \( MDD = \min \left( \frac{C_t}{P_t} - 1 \right) \), \( C_t \) cumulative return, \( P_t \) peak.

References: Sharpe (1966) for Sharpe ratio; Sortino & van der Meer (1991) for Sortino; "Fooled by Randomness" by Nassim Taleb for drawdown risks.

### Code Implementation
```python
def portfolio_stats(weights, returns, cash_ticker=DEFAULT_TICKERS[3]):
    port_returns = np.dot(returns, weights)
    mean_return = np.mean(port_returns)
    annual_return = mean_return * 252
    annual_vol = np.std(port_returns) * np.sqrt(252)
    rf_rate = returns[cash_ticker].mean() * 252 if cash_ticker in returns.columns else 0
    sharpe = (annual_return - rf_rate) / annual_vol if annual_vol != 0 else 0
    
    downside_returns = port_returns.copy()
    downside_returns[port_returns > 0] = 0
    downside_std = np.std(downside_returns) * np.sqrt(252)
    sortino = (annual_return - rf_rate) / downside_std if downside_std > 0 else 0
    
    cum_returns = np.cumprod(1 + port_returns)
    peaks = np.maximum.accumulate(cum_returns)
    drawdowns = (cum_returns / peaks) - 1
    max_dd = drawdowns.min()
    
    return annual_return, annual_vol, sharpe, sortino, max_dd
```
- **Logic**: Computes weighted portfolio returns via dot product. Annualizes assuming 252 days. Handles zero volatility/downside with conditionals. Uses cumulative products for drawdowns.
- **Edge Cases**: Zero weights or no data yield zero stats. Negative rf_rate possible in low-interest environments.

## Bootstrap Monte Carlo Simulation

### Financial Concept: Bootstrap Simulation
Bootstrap resampling draws random samples with replacement from historical returns to simulate future paths, preserving empirical distributions without assuming normality (Efron, 1979). It's non-parametric, suitable for fat-tailed financial returns. Incorporates DCA (periodic investments to mitigate timing risk), inflation adjustment (real returns), fees/taxes (drag on growth), rebalancing (maintain allocation), and stress shocks (exogenous events).

Mathematical Detail:
- Simulated path: Sample \( r_{1:d} \) from historical \( R \), compute \( V_T = V_0 \prod (1 + r_t) + \sum C_k \prod (1 + r_{k:d}) \), adjusted for inflation \( V_T / (1 + i)^T \).
- VaR/CVaR: \( VaR_{95} = P_5(\tilde{r}) \times Inv \), \( CVaR_{95} = \bar{\tilde{r} | \tilde{r} \leq P_5} \times Inv \), where \( \tilde{r} \) are simulated total returns.
- Rebalancing: If drift \( \max |w_c - w_t| > \theta \), reset \( v = V \cdot w_t \).

References: "Bootstrap Methods and Their Application" by Davison & Hinkley; Investopedia on DCA and VaR.

### Code Implementation
The core function is `bootstrap_simulation` (abridged for brevity):
```python
def bootstrap_simulation(returns, weights, num_simulations, time_horizon_years, initial_investment, inflation_rate=0.0, monthly_contrib=0.0, contrib_frequency='monthly', transaction_fee=0.0, tax_rate=0.0, rebalance=False, rebalance_frequency='annual', rebalance_threshold=0.05, shock_factors=None, base_invested=None):
    # ... (initialization)
    for i in range(num_simulations):
        boot_sample = returns.sample(days, replace=True).reset_index(drop=True)
        if shock_factors is not None:
            # Apply shocks
        # Lump-sum calculation
        # DCA loop with contributions, rebalancing
    # Compute stats: mean, median, VaR, CVaR, etc.
    return results, sim_final_values
```
- **Logic**: Resamples daily returns. Applies shocks as additive factors. Simulates lump-sum and DCA separately. Rebalances by resetting values to target weights if threshold exceeded at frequency. Adjusts for inflation post-compounding. Taxes applied on gains only. Progress bar for UX.
- **Edge Cases**: Zero contributions revert to lump-sum. Empty returns raise ValueError. Rebalancing ignores if total_value=0.

## Visualizations

### Concepts
- Histogram: Distribution of simulated outcomes, highlighting mean/median.
- Cumulative Returns: \( CR_t = \prod (1 + r_{1:t}) \), for trend analysis.
- Drawdowns: As above, filled area for visualization.
- Weight Drift: Tracks \( w_t = v_t / V_t \), max deviation.

Uses Plotly for interactivity, Matplotlib for static.

### Code
Functions like `plot_results`, `plot_historical_performance`, `plot_drawdowns`, `plot_weight_drift` return figures. Logic: Compute cumulatives via `cumprod`, traces added to Plotly figures.

## Backtesting

### Concept
Backtesting applies strategy to historical data for out-of-sample validation, avoiding lookahead bias. Includes DCA, rebalancing, fees/taxes.

Math: Similar to simulation but deterministic: \( V_T = V_0 (1 + R_{hist}) + \sum C_k (1 + R_{k:T}) \), net of costs.

Reference: "Evaluating Trading Strategies" by Campbell R. Harvey.

### Code
`backtest_portfolio` iterates over historical returns, applying contributions/rebalancing/taxes.

## Portfolio Optimization

### Concept
Maximizes Sharpe via constrained optimization (Markowitz, 1952). Objective: Minimize -Sharpe, subject to \( \sum w = 1 \), \( 0 \leq w \leq 1 \).

Reference: "Portfolio Selection" by Harry Markowitz.

### Code
Uses `scipy.optimize.minimize` with SLSQP solver implicitly.

## PDF Report Generation

Uses Reportlab to canvas text and images from figures. Logic: Draw metrics, embed PNGs from Plotly/Matplotlib.

## User Interface and Session State

Streamlit sidebar for inputs, columns for metrics with tooltips (explanations dict). Session state persists results post-simulation. Uploads handle CSVs for real portfolios, mapping ISIN to tickers.

## Limitations and Scrutiny Points
- Assumptions: Stationary returns (questionable in regime shifts); no slippage in rebalancing; simplified tax (flat on gains at end).
- Financial Correctness: Bootstrap may underestimate tail risks; stress scenarios are stylized.
- Programming: Potential overflows in large simulations; reliance on Yahoo data accuracy.
- Extensions: Incorporate correlations explicitly or machine learning for return predictions.

## References
- Fama, E. F. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. Journal of Finance.
- Sharpe, W. F. (1966). Mutual Fund Performance. Journal of Business.
- Sortino, F. A., & van der Meer, R. (1991). Downside Risk. Journal of Portfolio Management.
- Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife. Annals of Statistics.
- Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
- Websites: Investopedia (for metrics), Yahoo Finance API docs.
- Books: "Quantitative Investment Analysis" by DeFusco et al. (CFA Institute).