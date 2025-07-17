# Portfolio Simulator Dashboard

## Overview

This is a personal, locally-run Streamlit-based web dashboard for simulating and analyzing investment portfolios. It allows interactive adjustments to portfolio weights, simulation parameters, and visualizations of results. The application supports features like inflation adjustment, Dollar-Cost Averaging (DCA), custom assets, backtesting, fees/taxes, rebalancing, stress scenarios, and report generation.

Originally cloned from the private repository at `github.com/dyfrad/portfolio_simulator` for personal use. This project is not intended for open-source distribution or public sharing. All modifications and usage are for private financial planning purposes only.

## Features

- **Portfolio Allocation**: Adjust weights for default UCITS-compliant ETFs (e.g., IWDA.AS for MSCI World, QDV5.DE for MSCI India, PPFB.DE for Gold, XEON.DE for Cash) or add custom tickers.
- **Inflation Adjustment**: Input expected annual inflation rate to adjust final values.
- **Custom Assets**: Add custom tickers via text input.
- **Visualizations**: Line charts for historical cumulative returns, drawdown curves, weight drift, and simulation outcome distributions.
- **Backtesting**: Simulate historical performance over a selected period, including DCA.
- **DCA Support**: Monthly/quarterly contributions in simulations and backtests.
- **Fees and Taxes**: Incorporate Total Expense Ratio (TER), transaction fees, and capital gains tax.
- **Rebalancing**: Automatic rebalancing at specified frequency and threshold, with drift visualization.
- **Advanced Metrics**: Sharpe ratio, Sortino ratio, max drawdown, VaR, CVaR.
- **Stress Scenarios**: Predefined tests like 2008 Recession or COVID Crash.
- **Portfolio Upload**: Upload CSV for holdings (Ticker, Shares, Cost Basis) or transaction history to compute current value and weights.
- **Educational Tooltips**: Help explanations for metrics.
- **Report Generation**: Download PDF reports with metrics and charts.

## Installation

Since this is a cloned repository for personal use:

1. Ensure you have Python 3.8+ installed.
2. Clone the repository (already done):
   ```
   git clone git@github.com:dyfrad/portfolio_simulator.git
   cd portfolio_simulator
   ```
3. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies from `requirements.txt` (create one if not present):
   ```
   pip install -r requirements.txt
   ```

### requirements.txt Content
```
streamlit
yfinance
numpy
pandas
matplotlib
scipy
plotly
reportlab
kaleido
```

## Running the Application

Run the dashboard locally:
```
streamlit run portfolio_simulator_dashboard.py
```

This will launch the app in your default web browser (typically at `http://localhost:8501`).

- For development or debugging, you can edit the code in `portfolio_simulator_dashboard.py`.
- Data is fetched from Yahoo Finance, so an internet connection is required for simulations.

## Usage

1. **Sidebar Inputs**:
   - Adjust weights, time horizon, simulations, initial investment, contributions, inflation, fees/taxes, rebalancing, and stress scenarios.
   - Upload a CSV for portfolio holdings or transactions to auto-populate weights and values.
   - Optional: Optimize weights for maximum Sharpe ratio.

2. **Run Simulation**:
   - Click "Run Simulation" to generate results, metrics, and charts.
   - View historical performance, drawdowns, weight drift, and outcome distributions.

3. **Backtesting**:
   - Automatically runs with simulation; shows historical returns, volatility, etc.

4. **Reports**:
   - Generate and download a PDF report with key metrics and visualizations.
   - Download simulation results as CSV.

5. **Customization**:
   - Add custom tickers in the sidebar (comma-separated).
   - Modify default tickers or ISIN mappings in the code if needed.
   - Extend features by editing functions (e.g., add new stress scenarios).

## Configuration

- **Default Start Date**: '2015-01-01' (editable in sidebar).
- **ISIN to Ticker Mapping**: Hardcoded in the code; update the `ISIN_TO_TICKER` dictionary for additional assets.
- **Cache**: Uses Streamlit's `@st.cache_data` for data fetching to improve performance.
- **Session State**: Tracks simulation runs to avoid re-computation.

## Troubleshooting

- **Data Fetch Errors**: Ensure tickers are valid and Yahoo Finance is accessible. If data is limited (<252 days), a warning appears.
- **Optimization Failure**: Falls back to equal weights if Sharpe optimization fails.
- **PDF Generation**: Requires images to be exportable; ensure Kaleido is installed correctly.
- **Performance**: High simulation counts (e.g., 10,000) may take time; reduce for quicker runs.

## Notes for Personal Use

- This tool uses historical data for simulations and assumes no guarantees on future performance. It's for educational and planning purposes only—not financial advice.
- Backup your local repository regularly.
- If modifying the code, test changes locally before relying on outputs.
- No external APIs or keys are required, but respect Yahoo Finance's terms of service for data usage.

## Deployment (Optional, for Personal Cloud Use)

If you want to deploy to Streamlit Community Cloud for easier access (e.g., on mobile):

1. Make your GitHub repo private (or use a personal access token).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Deploy the app, specifying the file path (e.g., `portfolio_simulator_dashboard.py`).
4. Add `requirements.txt` to the repo.
5. Note: Free for personal apps, but keep it private.

For questions or issues, refer to the code comments or Streamlit documentation.