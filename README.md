# Portfolio Simulator Dashboard

## Overview

This is a Streamlit-based web application for simulating and analyzing portfolio performance. It supports Monte Carlo simulations using bootstrap resampling to account for uncertainties and risks in asset returns. The dashboard focuses on UCITS-compliant ETFs traded in EUR, with default assets including MSCI World, MSCI India, Gold, and Cash. Users can customize portfolios, optimize weights, adjust for inflation, backtest historical performance, and visualize results.

The application is designed to run locally on your computer but can be deployed to the cloud for sharing.

## Features

- **Interactive Portfolio Configuration**: Adjust weights for default or custom assets, with automatic normalization.
- **Monte Carlo Simulations**: Bootstrap-based simulations for future performance, including risk metrics like VaR, CVaR, expected returns, and volatility.
- **Inflation Adjustment**: Input an expected annual inflation rate to compute real (inflation-adjusted) final values.
- **Weight Optimization**: Maximize Sharpe ratio using historical data via SciPy.
- **Custom Assets**: Add your own ETF tickers (e.g., other UCITS funds) for flexible portfolios.
- **Historical Visualizations**: Line charts showing cumulative returns for the portfolio and individual assets.
- **Backtesting**: Evaluate historical performance over a user-defined period, with metrics like total return, annualized return/volatility, and Sharpe.
- **Results Export**: Download simulation results as CSV.
- **Caching**: Efficient data fetching with Streamlit caching for faster reruns.

## Installation

1. **Clone the Repository**:
   ```
   git clone https://github.com/dyfrad/portfolio_simulator.git
   cd portfolio_simulator
   ```

2. **Set Up a Virtual Environment** (Recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   Create a `requirements.txt` file with the following content:
   ```
   streamlit
   yfinance
   numpy
   pandas
   matplotlib
   scipy
   plotly
   ```
   Then run:
   ```
   pip install -r requirements.txt
   ```

## Usage

1. **Run Locally**:
   ```
   streamlit run portfolio_simulator_dashboard.py
   ```
   This will launch the dashboard in your browser (usually at http://localhost:8501).

2. **Interact with the Dashboard**:
   - Use the sidebar to configure parameters: weights, time horizon, simulations, initial investment, inflation rate, start/end dates, and custom tickers.
   - Check "Optimize Weights for Max Sharpe" for automatic allocation.
   - Click "Run Simulation" to generate results, visualizations, and backtest metrics.
   - Download results via the CSV button after a successful run.

3. **Example Scenario**:
   - Set weights: 40% MSCI World, 30% MSCI India, 20% Gold, 10% Cash.
   - Horizon: 5 years, Simulations: 5000, Inflation: 2%.
   - Start Date: 2015-01-01, Backtest End Date: 2024-12-31.
   - Run to see simulated outcomes, historical charts, and backtest performance.

## Backtesting Guide

- **Purpose**: Test how the portfolio would have performed historically.
- **Inputs**: Use "Start Date" and "Backtest End Date" in the sidebar.
- **Output**: Metrics like Total Return, Annualized Return, Volatility, and Sharpe Ratio appear in the "Backtesting Results" section after running.
- **Tip**: Leave End Date blank for data up to the current date. Ensure the period has enough data for accurate results.

## Deployment

To deploy to Streamlit Community Cloud (free for public apps):
1. Make your GitHub repo public.
2. Visit [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Deploy a new app: Select your repo, branch, and file (e.g., `portfolio_simulator_dashboard.py`).
4. Ensure `requirements.txt` is in the repo root.
5. Deploy—the app will be live at a URL like `https://your-app-name.streamlit.app`.

For other platforms (e.g., Heroku, Render), follow their Python/Streamlit guides.

## Dependencies

- Python 3.8+
- See `requirements.txt` for libraries.

## Limitations

- Relies on yfinance for data—may have rate limits or occasional downtime.
- Simulations are historical-based; not financial advice—use at your own risk.
- Custom tickers must be valid yfinance symbols with overlapping data.

