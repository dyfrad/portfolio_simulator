# Portfolio Simulator

## Overview

This is a personal, locally-run Streamlit-based web dashboard for simulating and analyzing investment portfolios. It allows interactive adjustments to portfolio weights, simulation parameters, and visualizations of results. The application supports features like inflation adjustment, Dollar-Cost Averaging (DCA), custom assets, backtesting, fees/taxes, rebalancing, stress scenarios, and report generation. Key use cases include setting custom tickers for flexibility and uploading CSV files tuned for Degiro transaction history to automatically compute current holdings, cost basis, and portfolio weights.

Originally cloned from the private repository at `github.com/dyfrad/portfolio_simulator` for personal use. This project is not intended for open-source distribution or public sharing. All modifications and usage are for private financial planning purposes only.

## Project Structure

The codebase has a modular structure:

```
portfolio_simulator/
├── portfolio_simulator.py          # Core simulation logic and financial calculations
├── portfolio_simulator_ui.py       # Streamlit UI and main entry point
├── ui/                             # UI components
│   ├── __init__.py
│   ├── dashboard.py                # Main dashboard component
│   └── components/                 # Reusable UI components
│       ├── __init__.py
│       ├── results_display.py      # Results visualization
│       ├── sidebar_inputs.py       # Input controls
│       └── state_manager.py        # State management
├── reports/                        # Report generation
│   ├── __init__.py
│   ├── data_models.py              # Data structures for reports
│   ├── factory.py                  # Report factory pattern
│   └── pdf_generator.py            # PDF generation logic
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Container configuration
└── README.md                       # This file
```

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
- **Portfolio Upload**: Upload CSV for holdings (Ticker, Shares, Cost Basis) or transaction history (tuned for Degiro transactions) to compute current value and weights.
- **Educational Tooltips**: Help explanations for metrics.
- **Report Generation**: Download PDF reports with metrics and charts.

## Installation

Since this is a cloned repository for personal use:

1. Ensure you have Python 3.8+ installed (recommended: 3.12 for better performance and security).
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
streamlit run portfolio_simulator_ui.py
```

This will launch the app in your default web browser (typically at `http://localhost:8501`).

- For development or debugging, you can edit the code in the modular structure:
  - Main UI logic: `portfolio_simulator_ui.py`
  - Core simulation: `portfolio_simulator.py`
  - UI components: `ui/dashboard.py` and `ui/components/`
  - Report generation: `reports/`
- Data is fetched from Yahoo Finance, so an internet connection is required for simulations.

## Running with Docker

For a containerized environment (useful for isolation or deployment):

1. Ensure Docker is installed on your system.
2. Build the Docker image:
   ```
   docker build -t portfolio-simulator .
   ```
3. Run the container:
   ```
   docker run -p 8501:8501 portfolio-simulator
   ```
4. Access the app in your browser at `http://localhost:8501`.

This setup ensures all dependencies are contained and the app runs consistently across environments.

## Usage

1. **Sidebar Inputs**:
   - Adjust weights, time horizon, simulations, initial investment, contributions, inflation, fees/taxes, rebalancing, and stress scenarios.
   - Upload a CSV for portfolio holdings or transactions (tuned for Degiro transactions) to auto-populate weights and values.
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

## Development

The codebase follows a modular architecture:

- **Core Logic**: `portfolio_simulator.py` contains the main simulation engine and financial calculations
- **UI Layer**: `portfolio_simulator_ui.py` serves as the entry point, importing from the `ui` module
- **Components**: Reusable UI components are organized in `ui/components/`
- **Reports**: PDF report generation logic is in the `reports/` module

To extend functionality:
- Add new UI components in `ui/components/`
- Modify simulation logic in `portfolio_simulator.py`
- Update report templates in `reports/`
- Configure defaults in `portfolio_simulator_ui.py`

## Troubleshooting

- **Data Fetch Errors**: Ensure tickers are valid and Yahoo Finance is accessible. If data is limited (<252 days), a warning appears.
- **yfinance Issues**: As an unofficial wrapper for Yahoo Finance API, it may break if Yahoo changes their endpoints. Update to the latest version with `pip install yfinance --upgrade` or visit the GitHub repo for issues: https://github.com/ranaroussi/yfinance.
- **Optimization Failure**: Falls back to equal weights if Sharpe optimization fails.
- **PDF Generation**: Requires images to be exportable; ensure Kaleido is installed correctly.
- **Performance**: High simulation counts (e.g., 10,000) may take time; reduce for quicker runs.
- **Docker Issues**: Ensure Docker is running and ports are free. If building fails, check for missing files or network issues during pip install.
- **Import Errors**: If you encounter module import issues, ensure all `__init__.py` files are present in the `ui/` and `reports/` directories.

## Notes for Personal Use

- This tool uses historical data for simulations and assumes no guarantees on future performance. It's for educational and planning purposes only—not financial advice.
- Backup your local repository regularly.
- If modifying the code, test changes locally before relying on outputs.
- No external APIs or keys are required, but respect Yahoo Finance's terms of service for data usage.
- For privacy, avoid uploading sensitive financial data to public deployments; keep everything local or in private repos.

## Deployment (Optional, for Personal Cloud Use)

If you want to deploy to Streamlit Community Cloud for easier access (e.g., on mobile):

1. Make your GitHub repo private (or use a personal access token).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Deploy the app, specifying the file path (e.g., `portfolio_simulator_ui.py`).
4. Add `requirements.txt` to the repo.
5. Note: Free for personal apps, but keep it private.

For questions or issues, refer to the code comments or Streamlit documentation.