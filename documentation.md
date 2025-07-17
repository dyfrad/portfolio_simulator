# Portfolio Simulator Dashboard Documentation

## Overview

This documentation provides a comprehensive guide to the Portfolio Simulator Dashboard, a Streamlit-based web application for simulating and analyzing investment portfolios. The dashboard supports interactive adjustments to portfolio weights, simulation parameters, and visualizations. It incorporates advanced features like inflation adjustment, Dollar-Cost Averaging (DCA), fees, taxes, rebalancing, stress scenarios, and backtesting.

The application is designed for users interested in financial planning, particularly with UCITS-compliant ETFs traded in EUR. It uses historical data from Yahoo Finance via `yfinance` to perform bootstrap Monte Carlo simulations and historical backtests.

Key aspects covered:
- **Code-wise**: Breakdown of the Python code structure, key functions, and logic.
- **Mathematics-wise**: Equations for risk metrics, returns, and simulations.
- **Finance-wise**: Explanations of concepts like TER, DCA, rebalancing, stress testing, and tax implications.
- **Technicalities**: Dependencies, setup, data handling, and limitations.

This documentation is provided in Markdown format. Save this content as `portfolio_simulator_docs.md` on your laptop and open it with a Markdown viewer (e.g., VS Code with Markdown preview, or convert to HTML/PDF using Pandoc). For equations, it uses LaTeX syntax (renderable in tools like Obsidian or Jupyter). Code snippets are highlighted for readability. Figures are described with placeholders; you can generate them by running the code and screenshotting the outputs.

## Installation and Setup

### Dependencies
The application requires the following Python libraries (as listed in the code):
- `yfinance`: For fetching historical financial data.
- `numpy`: For numerical computations.
- `pandas`: For data manipulation.
- `matplotlib`: For basic plotting.
- `streamlit`: For the web dashboard.
- `scipy`: For optimization (e.g., portfolio weight optimization).
- `plotly`: For interactive charts.
- `reportlab`: For PDF report generation.
- `kaleido`: For exporting Plotly figures to images.

Install them via pip (create a `requirements.txt` file with the list above and run `pip install -r requirements.txt`).

### How to Run Locally
1. Save the provided code as `portfolio_simulator_dashboard.py`.
2. Open a terminal in the file's directory.
3. Run: `streamlit run portfolio_simulator_dashboard.py`.
4. The dashboard will open in your browser (e.g., http://localhost:8501).

For deployment to Streamlit Community Cloud, follow the instructions in the code's docstring (requires a GitHub repo).

### Data Sources
- Historical prices: Fetched from Yahoo Finance using `yfinance.download()`.
- Default tickers: IWDA.AS (MSCI World), QDV5.DE (MSCI India), PPFB.DE (Gold), XEON.DE (Cash).
- ISIN to Ticker mapping: Hardcoded dictionary for transaction CSV support.
- Limitations: No internet access in code execution tool (per guidelines), but `yfinance` requires internet when running locally.

## Code Structure

The code is structured as follows:
- **Imports**: Libraries for data, math, plotting, and UI.
- **Constants**: Default tickers, ISIN mappings, start date.
- **Session State Initialization**: For caching simulation runs.
- **Functions**: Data fetching, calculations, simulations, plotting, backtesting, optimization, PDF generation.
- **Explanations Dictionary**: Tooltips for metrics.
- **Streamlit UI**: Sidebar inputs, main display, buttons.
- **Logic Flow**: Fetch data → Calculate returns → Run simulation/backtest → Display results/plots → Generate report.

Key global variables:
- `DEFAULT_TICKERS`: List of default assets.
- `ISIN_TO_TICKER`: Mapping for transaction processing.

## Key Functions and Explanations

### 1. Data Fetching: `fetch_data(tickers, start_date, end_date=None)`
   - **Code Snippet**:
     ```python
     @st.cache_data
     def fetch_data(tickers, start_date, end_date=None):
         data = yf.download(tickers, start=start_date, end=end_date)['Close'].dropna()
         if len(data) < 252:
             st.warning("Limited historical data available. Simulations may not be reliable.")
         return data
     ```
   - **Code Explanation**: Uses Streamlit's caching decorator for efficiency. Downloads adjusted close prices, drops NaNs, and warns if less than 1 year of data (252 trading days).
   - **Finance Explanation**: Focuses on 'Close' prices for total returns (including dividends via adjusted close). Assumes daily data for accuracy in simulations.
   - **Technicalities**: Handles multiple tickers; end_date optional for full history.

### 2. Returns Calculation: `calculate_returns(data, ter=0.0)`
   - **Code Snippet**:
     ```python
     def calculate_returns(data, ter=0.0):
         returns = data.pct_change().dropna()
         daily_ter = (ter / 100) / 252  # Annual TER to daily deduction
         returns -= daily_ter
         return returns
     ```
   - **Code Explanation**: Computes percentage changes, subtracts daily TER (Total Expense Ratio) uniformly across assets.
   - **Mathematics**:
     Daily return: \( r_t = \frac{P_t - P_{t-1}}{P_{t-1}} \)
     Adjusted: \( r_t' = r_t - \frac{\text{TER}}{252 \times 100} \)
   - **Finance Explanation**: TER represents fund management fees (e.g., 0.2% annual). Subtracting it simulates net returns. Assumes constant TER; in reality, it varies by asset.

### 3. Portfolio Statistics: `portfolio_stats(weights, returns, cash_ticker=...)`
   - **Code Snippet** (truncated):
     ```python
     def portfolio_stats(weights, returns, cash_ticker=DEFAULT_TICKERS[3]):
         port_returns = np.dot(weights)
         mean_return = np.mean(port_returns))
         annual_return = mean_return * 252
         annual_vol = np.std(port_returns) * np.sqrt(252)
         rf_rate = returns[cash_ticker].mean() * 252 if ... else 0
         sharpe = (annual_return - rf_rate) / annual_vol if annual_vol != 0 else 0
         # Sortino: downside std
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
     ```
   - **Code Explanation**: Computes weighted returns, annualizes metrics, uses cash as risk-free rate. Handles edge cases like zero vol.
   - **Mathematics**:
     - Annual Return: \( \bar{r} \times 252 \)
     - Volatility: \( \sigma = \sqrt{\(var(r_p) \times 252} \)
     - Sharpe Ratio: \( S = \frac{\bar{r} - r_f}{\sigma} \)
     - \( r_p = \) portfolio return
     - \( r_f = \) risk-free rate
     - \( \sigma = \) volatility
     - Sortino Ratio: Similar to Sharpe but \( \sigma_down = \sqrt{\var(r_p[r_p < 0])} \times 252 \)
     - Max Drawdown: \( \left( \frac{P_t}{P_\peak} - 1 \right)_{\min} \)
   - **Finance Explanation**: Sharpe measures reward per unit risk; Sortino focuses on bad risk. Max DD shows worst loss, important for investor tolerance. Risk-free rate from cash proxy.

### 4. Bootstrap Simulation: `bootstrap_simulation(...)`
   - core function, long and complex.
   - **Code Explanation**: Performs N bootstrap resamples of historical returns. Supports DCA (adds contributions periodically), inflation adjustment, fees (subtract per contrib), taxes (on gains), rebalancing (reset weights if drift > threshold), stress shocks (apply return shocks). Computes lump-sum for comparison, VaR/CVaR.
   - **Mathematics**: Bootstrap: Sample with replacement from empirical distribution. Final value: \( V = V_0 \prod (1 + r) / (1 + i)^y + \sum contribs \), adjusted.
     - VaR 95%: 5th percentile of losses.
     - CVaR 95%: Mean of losses below VaR.
   - **Finance Explanation**: Bootstrap assumes future like past (stationary). DCA reduces timing risk. Rebalancing maintains allocation but incurs (virtual) costs. Stress: Applies shocks e.g., -40% in equities for 2008 scenario. Taxes: Applied to realized gains at end (simplified, assumes no withdrawals).

### 5. Plotting Functions
- e.g., `plot_results`: Histogram of final values with mean/median (Matplotlib).
- `plot_historical_performance`: Cumulative returns line chart (Plotly).
- `plot_weight_drift`: Tracks asset weights over time, shows max drift (Plotly).
- `plot_drawdowns`: Filled area for drawdowns (Plotly).
- **Figure Example**: For historical performance, the plot shows lines for each asset and dashed for portfolio. (Run code to view; save as PNG for local reference.)

### 6. Backtesting: `backtest_portfolio(...)`
   - Simulates actual historical path with DCA, rebalancing, fees/taxes.
   - **Finance Explanation**: Unlike simulation (random paths), backtest uses real sequence for "what if" analysis.

### 7. Optimization: `optimize_weights(returns)`
   - Uses `scipy.minimize` to maximize Sharpe (negative objective).
   - Constraints: Weights sum to 1, bounds [0,1].
   - **Mathematics**: Quadratic optimization: argmax \( S(w) \) s.t. \( \sum w = 1 \).

### 8. PDF Report: `generate_pdf_report(...)`
   - Uses ReportLab to create PDF with metrics and embedded charts (via Kaleido for image export).

### 9. UI Logic
   - Sidebar: Inputs for params, file upload (holdings/transactions CSV).
   - Handles uploads: Computes current value/weights from prices/cost basis.
   - Button triggers simulation, stores in session state for persistence.

## Mathematical Foundations

- **Returns and Compounding**: Cumulative return: \( \prod (1 + r_t) - 1 \).
- **Inflation Adjustment**: Real value: \( V / (1 + i)^y \), where i = inflation rate, y = years.
- **VaR/CVaR**: From simulated returns distribution.
- **Rebalancing**: If |current_weight - target| > threshold, reset to target (assumes costless; real-world has transaction costs).

## Finance Concepts

- **DCA**: Invest fixed amount periodically to average costs.
- **TER/Fees/Taxes**: Reduce net returns; e.g., TER deducted daily, transaction fee per buy, capital gains tax on profits.
- **Rebalancing**: Prevents drift from target allocation, e.g., quarterly if drift >5%.
- **Stress Scenarios**: Predefined shocks based on historical events (e.g., 2008: -40% equities).
- **Backtesting vs Simulation**: Backtest = historical path; Simulation = random resamples for forward-looking.

## Limitations and Technicalities

- Assumptions: Stationary returns, no dividends beyond adjusted close, simplified taxes (end-only, no losses carryover).
- Performance: High simulations (e.g., 10k) may be slow; progress bar included.
- Errors: Handles empty data, optimization failure.
- Security: Local run only; no API keys needed.

## References

- Sharpe Ratio: Sharpe, W. F. (1966). "Mutual Fund Performance". Journal of Business.
- Sortino Ratio: Sortino, F. A., & Price, L. N. (1994). "Performance Measurement in a Downside Risk Framework". Journal of Investing.
- Bootstrap Simulation: Efron, B. (1979). "Bootstrap Methods: Another Look at the Jackknife". Annals of Statistics.
- Yahoo Finance API: https://pypi.org/project/yfinance/
- Streamlit Docs: https://docs.streamlit.io/
- Financial Concepts: Investopedia (e.g., https://www.investopedia.com/terms/d/dollarcostaveraging.asp for DCA).

For figures, run the dashboard and export plots (e.g., Plotly has download button). If you need updates or clarifications, rerun with modifications.