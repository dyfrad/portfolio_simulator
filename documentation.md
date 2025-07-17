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

### Financial Concept: Backtesting
Backtesting evaluates an investment strategy by applying it to historical data, simulating how it would have performed in the past. This helps validate strategy viability, identify weaknesses, and estimate risk-adjusted returns without lookahead bias. In portfolio contexts, it incorporates asset allocation, rebalancing, contributions (DCA), fees, taxes, and comparisons to benchmarks. Advanced techniques include sensitivity analysis (varying parameters), walk-forward optimization (in-sample training, out-of-sample testing), and incorporation of transaction costs for realism.

Key limitations: Overfitting (curve-fitting to history), survivorship bias (excluding delisted assets), and regime shifts (changing market conditions). To mitigate, use out-of-sample data and robust statistics.

Mathematical Detail:
- Portfolio Value Evolution (Lump-Sum): \( V_t = V_{t-1} (1 + \sum w_i r_{i,t}) \), net of fees/taxes at end.
- DCA: Add contributions periodically, e.g., monthly: \( V_t = (V_{t-1} + C - f) (1 + \sum w_i r_{i,t}) \), where \( C \) is contribution, \( f \) transaction fee.
- Rebalancing: At intervals, if \( \max |w_{current} - w_{target}| > \theta \), reset weights, simulating buys/sells (ignoring slippage for simplicity).
- Taxes: Applied on realized gains at end: \( G = V_T - Inv \), net \( V_T^{net} = Inv + G (1 - \tau) \) if \( G > 0 \).
- Advanced Metrics:
  - Total Return: \( TR = \frac{V_T - Inv}{Inv} \).
  - Annualized Return: As above.
  - Calmar Ratio: \( Calmar = \frac{R_a}{|MDD|} \), measures return per unit of drawdown risk.
  - Beta (\( \beta \)): Systematic risk vs benchmark: \( \beta = \frac{Cov(r_p, r_b)}{Var(r_b)} \), via linear regression.
  - Alpha (\( \alpha \)): Excess return: From CAPM, \( r_p - r_f = \alpha + \beta (r_b - r_f) + \epsilon \); solve for \( \alpha \).
  - Information Ratio: \( IR = \frac{R_a - R_{b,a}}{TE} \), where \( TE = \sigma(r_p - r_b) \) (tracking error).
  - Exposure: \( Exp = \frac{\sum t_{invested}}{T} \times 100\% \), % time with positions (typically 100% for buy-and-hold portfolios).
  - Standard Deviation of Losses: \( \sigma_{loss} = \sqrt{\frac{\sum (r_t < 0) (r_t - \bar{r}_{loss})^2}{n_{loss}}} \sqrt{252} \).

References: "Evaluating Trading Strategies" by Campbell R. Harvey; "Portfolio Selection" by Harry Markowitz; TrendSpider Learning Center for advanced metrics like Beta, Exposure, and Calmar; Portfolio Visualizer documentation for practical backtesting tools.

### Code Implementation
The `backtest_portfolio` function has been enhanced to include advanced metrics (Calmar, Beta, Alpha, Information Ratio, Exposure) and returns time series for portfolio values (equity curves) for both lump-sum and DCA. A benchmark ticker (e.g., 'IWDA.AS' for MSCI World) is now optional input for relative metrics. Exposure is computed as the percentage of days with positive portfolio value (assuming always invested unless zero).

Updated code (abridged):
```python
def backtest_portfolio(data, weights, initial_investment, monthly_contrib=0.0, contrib_frequency='monthly', transaction_fee=0.0, tax_rate=0.0, rebalance=False, rebalance_frequency='annual', rebalance_threshold=0.05, benchmark_ticker=None):
    returns = calculate_returns(data)
    num_assets = len(weights)
    values = initial_investment * weights
    total_value = initial_investment
    total_invested = initial_investment
    rebalance_days = 252 if rebalance_frequency == 'annual' else 63
    day_count = 0
    port_value_series = pd.Series(index=returns.index, dtype=float)  # For lump-sum equity curve
    dca_value_series = pd.Series(index=returns.index, dtype=float) if monthly_contrib > 0 else None
    
    for idx, (_, daily_returns) in enumerate(returns.iterrows()):
        values *= (1 + daily_returns)
        total_value = np.sum(values)
        port_value_series.iloc[idx] = total_value
        current_weights = values / total_value if total_value > 0 else weights
        if rebalance and (day_count % rebalance_days == 0):
            max_drift = np.max(np.abs(current_weights - weights))
            if max_drift > rebalance_threshold:
                values = total_value * weights  # Rebalance
        day_count += 1
    
    # ... (original total return calculations for lump-sum and DCA)
    
    # Advanced metrics
    ann_return, ann_vol, sharpe, sortino, max_dd = portfolio_stats(weights, returns)
    calmar = ann_return / -max_dd if max_dd < 0 else 0
    
    # Benchmark metrics if provided
    beta, alpha, info_ratio = None, None, None
    exposure = (port_value_series > 0).mean() * 100  # % days invested
    if benchmark_ticker:
        bench_data = fetch_data(benchmark_ticker, data.index[0], data.index[-1])
        bench_returns = calculate_returns(bench_data)
        if benchmark_ticker in bench_returns.columns:
            bench_ret = bench_returns[benchmark_ticker]
            port_ret = np.dot(returns, weights)
            cov = np.cov(port_ret, bench_ret)[0,1]
            var_b = np.var(bench_ret)
            beta = cov / var_b if var_b != 0 else 0
            ann_bench = bench_ret.mean() * 252
            alpha = ann_return - (rf_rate + beta * (ann_bench - rf_rate))
            tracking_error = np.std(port_ret - bench_ret) * np.sqrt(252)
            info_ratio = (ann_return - ann_bench) / tracking_error if tracking_error != 0 else 0
    
    # For DCA equity curve (resampled for efficiency)
    if monthly_contrib > 0:
        freq = 'M' if contrib_frequency == 'monthly' else 'Q'
        monthly_data = data.resample(freq).last()
        monthly_returns = monthly_data.pct_change().dropna()
        port_monthly_returns = np.dot(monthly_returns, weights)
        value = 0.0
        total_invested = 0.0
        dca_idx = 0
        for ret in port_monthly_returns:
            effective_contrib = monthly_contrib - transaction_fee
            value = (value + effective_contrib) * (1 + ret)
            total_invested += monthly_contrib
            # Map back to daily index (approximate)
            dca_value_series.iloc[dca_idx:dca_idx + 21] = value  # Fill forward approx
            dca_idx += 21  # Avg days per month
        # ... (tax adjustment)
    
    return {
        'Total Return (DCA)': cum_port_returns_dca,
        'Total Return (Lump-Sum)': net_cum_port_returns,
        'Annualized Return': ann_return,
        'Annualized Volatility': ann_vol,
        'Sharpe Ratio': sharpe,
        'Sortino Ratio': sortino,
        'Max Drawdown': max_dd,
        'Calmar Ratio': calmar,
        'Beta': beta,
        'Alpha': alpha,
        'Information Ratio': info_ratio,
        'Exposure (%)': exposure,
    }, port_value_series, dca_value_series
```
- **Logic**: Iterates daily for lump-sum, tracking value series. For DCA, resamples to contribution frequency and approximates daily series by filling forward. Computes advanced metrics using covariance for beta, CAPM for alpha, tracking error for IR. Exposure assumes always invested if value >0.
- **Edge Cases**: No benchmark skips relative metrics. Zero max_dd avoids division by zero. Short data may inflate annualized stats; warn if <252 days.
- **Enhancements for Details**: Returns equity curves for plotting (e.g., via Plotly line chart). In dashboard, add benchmark input, display new metrics with tooltips, and plot equity curves (lump-sum vs DCA vs benchmark).

To visualize backtest:
```python
def plot_backtest_performance(port_value_series, dca_value_series, bench_data=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=port_value_series.index, y=port_value_series, mode='lines', name='Lump-Sum Portfolio'))
    if dca_value_series is not None:
        fig.add_trace(go.Scatter(x=dca_value_series.index, y=dca_value_series, mode='lines', name='DCA Portfolio'))
    if bench_data is not None:
        bench_cum = (1 + calculate_returns(bench_data)).cumprod() * initial_investment
        fig.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum.iloc[:,0], mode='lines', name='Benchmark'))
    fig.update_layout(title='Backtest Equity Curves', xaxis_title='Date', yaxis_title='Portfolio Value')
    return fig
```
- **Integration**: In dashboard, after running, fetch benchmark if selected, compute, plot, and update report.

## Portfolio Optimization

### Concept
Maximizes Sharpe via constrained optimization (Markowitz, 1952). Objective: Minimize -Sharpe, subject to \( \sum w = 1 \), \( 0 \leq w \leq 1 \).

Reference: "Portfolio Selection" by Harry Markowitz.

### Code
Uses `scipy.optimize.minimize` with SLSQP solver implicitly.

## PDF Report Generation

Uses Reportlab to canvas text and images from figures. Logic: Draw metrics, embed PNGs from Plotly/Matplotlib. Enhanced to include backtest plot and new metrics.

## User Interface and Session State

Streamlit sidebar for inputs, columns for metrics with tooltips (explanations dict). Session state persists results post-simulation. Uploads handle CSVs for real portfolios, mapping ISIN to tickers. Add sidebar input for benchmark_ticker (default 'IWDA.AS').

## Limitations and Scrutiny Points
- Assumptions: Stationary returns (questionable in regime shifts); no slippage in rebalancing; simplified tax (flat on gains at end); always 100% exposure unless zero value.
- Financial Correctness: Backtesting may overfit; bootstrap underestimates tails; add out-of-sample for robustness.
- Programming: Potential overflows in large datasets; reliance on Yahoo data accuracy; approximate DCA series filling.
- Extensions: Walk-forward backtesting, sensitivity to parameters, machine learning for predictions.

## References
- Fama, E. F. (1970). Efficient Capital Markets: A Review of Theory and Empirical Work. Journal of Finance.
- Sharpe, W. F. (1966). Mutual Fund Performance. Journal of Business.
- Sortino, F. A., & van der Meer, R. (1991). Downside Risk. Journal of Portfolio Management.
- Efron, B. (1979). Bootstrap Methods: Another Look at the Jackknife. Annals of Statistics.
- Markowitz, H. (1952). Portfolio Selection. Journal of Finance.
- Harvey, C. R. (2014). Evaluating Trading Strategies. Journal of Portfolio Management.
- Websites: Investopedia (metrics), Yahoo Finance API docs, TrendSpider Learning Center (advanced metrics), Portfolio Visualizer (backtesting tools).
- Books: "Quantitative Investment Analysis" by DeFusco et al. (CFA Institute).