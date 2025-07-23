# Portfolio Simulator

## Overview

This is a personal, locally-run Streamlit-based web dashboard for simulating and analyzing investment portfolios. It allows interactive adjustments to portfolio weights, simulation parameters, and visualizations of results. The application supports features like inflation adjustment, Dollar-Cost Averaging (DCA), custom assets, backtesting, fees/taxes, rebalancing, and stress scenarios. Key use cases include setting custom tickers for flexibility and uploading CSV files tuned for Degiro transaction history to automatically compute current holdings, cost basis, and portfolio weights.

Originally cloned from the private repository at `github.com/dyfrad/portfolio_simulator` for personal use. This project is not intended for open-source distribution or public sharing. All modifications and usage are for private financial planning purposes only.

## Project Structure

The codebase follows modern Python packaging standards with a well-organized modular structure:

```
portfolio_simulator/
├── src/                            # Source code package
│   └── portfolio_simulator/        # Main package
│       ├── __init__.py
│       ├── core/                   # Business logic modules
│       │   ├── __init__.py
│       │   ├── data_operations.py  # Data fetching and processing
│       │   ├── financial_calculations.py # Portfolio statistics and optimization
│       │   ├── simulation_engine.py # Monte Carlo simulation engine
│       │   ├── backtesting.py      # Historical backtesting
│       │   └── visualization.py    # Chart generation
│       ├── config/                 # Configuration management
│       │   ├── __init__.py
│       │   ├── constants.py        # Default tickers and constants
│       │   ├── settings.py         # Application settings
│       │   └── environments/       # Environment-specific configs
│       │       ├── __init__.py
│       │       ├── development.py
│       │       ├── production.py
│       │       └── testing.py
│       └── ui/                     # UI components
│           ├── __init__.py
│           ├── dashboard.py        # Main dashboard component
│           └── components/         # Reusable UI components
│               ├── __init__.py
│               ├── results_display.py # Results visualization
│               ├── sidebar_inputs.py  # Input controls
│               └── state_manager.py   # State management
├── scripts/                        # Entry point scripts
│   ├── run_simulator.py
│   └── run_ui.py
├── tests/                          # Comprehensive test suite
│   ├── __init__.py
│   ├── conftest.py                 # Pytest configuration
│   ├── fixtures/                   # Test data and fixtures
│   │   ├── __init__.py
│   │   └── sample_data.py
│   ├── unit/                       # Unit tests
│   │   ├── __init__.py
│   │   ├── test_data_operations.py
│   │   ├── test_financial_calculations.py
│   │   ├── test_simulation_engine.py
│   │   ├── test_backtesting.py
│   │   ├── test_visualization.py
│   │   └── test_config.py
│   ├── integration/                # Integration tests
│   │   ├── __init__.py
│   │   └── test_portfolio_simulator.py
│   └── utils.py                    # Test utilities and helpers
├── pyproject.toml                  # Modern Python packaging configuration
├── requirements.txt                # Legacy dependencies (maintained for compatibility)
├── Dockerfile                      # Container configuration
├── portfolio_simulator_ui.py       # Legacy UI entry point (maintained for compatibility)
└── README.md                       # This file
```

## Features and Usage

- **Portfolio Allocation**: Adjust weights for default UCITS-compliant ETFs or add custom tickers.
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

**How to Use:**
- Adjust sidebar inputs (weights, horizon, contributions, etc.)
- Upload CSV for portfolio holdings or transactions (Degiro format supported)
- Click "Run Simulation" to generate results, metrics, and charts
- View historical performance, drawdowns, weight drift, and outcome distributions
- Add custom tickers or optimize weights for maximum Sharpe ratio

## Installation

1. Ensure you have Python 3.8+ installed (recommended: 3.12 for better performance and security).
2. Clone the repository (already done):
   ```bash
   git clone git@github.com:dyfrad/portfolio_simulator.git
   cd portfolio_simulator
   ```
3. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install the package in development mode (recommended):
   ```bash
   pip install -e .
   ```
   Or, for legacy compatibility:
   ```bash
   pip install -r requirements.txt
   ```
5. For development and testing dependencies:
   ```bash
   pip install -e ".[dev,test]"
   ```

## Running the Application

- **Entry Point Scripts (Recommended):**
  ```bash
  portfolio-ui
  portfolio-simulator
  ```
- **Legacy Entry Point:**
  ```bash
  streamlit run portfolio_simulator_ui.py
  ```
- **Script Files:**
  ```bash
  python scripts/run_ui.py
  ```

The application will launch in your default web browser (typically at `http://localhost:8501`).

**Note:** Data is fetched from Yahoo Finance, so an internet connection is required for simulations.

## Docker

1. Ensure Docker is installed.
2. Build the Docker image:
   ```bash
   docker build -t portfolio-simulator .
   ```
3. Run the container:
   ```bash
   docker run -p 8501:8501 portfolio-simulator
   ```
4. Access the app at `http://localhost:8501`.

## Configuration

- **Constants:** Default tickers and mappings in `src/portfolio_simulator/config/constants.py`
- **Settings:** Application settings in `src/portfolio_simulator/config/settings.py`
- **Environments:** Environment-specific configurations in `src/portfolio_simulator/config/environments/`
- **Key Options:** Default start date, ISIN to ticker mapping, environment settings, cache, session state
- **Environment Variables:**
  ```bash
  export ENVIRONMENT=development  # or production, testing
  export DEBUG=true              # Enable debug mode
  ```

## Development

- **Architecture:** Modular, separation of concerns, testable code, environment-specific config
- **Where to Add Code:**
  - Core logic: `src/portfolio_simulator/core/`
  - UI: `src/portfolio_simulator/ui/components/`
  - Config: `src/portfolio_simulator/config/`
  - Tests: `tests/unit/`, `tests/integration/`
- **Code Quality Tools:** black, flake8, mypy, pre-commit (install with dev dependencies)

## Testing

Run the comprehensive test suite from the project root:
```bash
pytest
```
- For coverage:
  ```bash
  pytest --cov=src/portfolio_simulator
  ```
- For specific test types:
  ```bash
  pytest tests/unit/          # Unit tests only
  pytest tests/integration/   # Integration tests only
  ```

## Troubleshooting

### Common Issues

- **Data Fetch Errors**: Ensure tickers are valid and Yahoo Finance is accessible. If data is limited (<252 days), a warning appears.
- **yfinance Issues**: As an unofficial wrapper for Yahoo Finance API, it may break if Yahoo changes their endpoints. Update to the latest version with `pip install yfinance --upgrade` or visit the GitHub repo for issues: https://github.com/ranaroussi/yfinance.
- **Optimization Failure**: Falls back to equal weights if Sharpe optimization fails.
- **Performance**: High simulation counts (e.g., 10,000) may take time; reduce for quicker runs.
- **Docker Issues**: Ensure Docker is running and ports are free. If building fails, check for missing files or network issues during pip install.

### Installation Issues

- **Import Errors**: If you encounter module import issues after restructuring:
  1. Reinstall the package: `pip install -e .`
  2. Ensure you're using the new import paths: `from portfolio_simulator.core import ...`
  3. Check that all `__init__.py` files are present

- **Entry Point Issues**: If `portfolio-ui` or `portfolio-simulator` commands don't work:
  1. Reinstall with entry points: `pip install -e .`
  2. Check your PATH includes the Python scripts directory
  3. Use the legacy method: `streamlit run portfolio_simulator_ui.py`

### Testing Issues

- **Test Failures**: If tests fail after installation:
  1. Install test dependencies: `pip install -e ".[test]"`
  2. Run tests from the project root: `pytest`
  3. Check that the package is properly installed: `pip show portfolio-simulator`

### Performance Optimization

- **Slow Simulations**: 
  - Reduce number of simulations in development environment
  - Use the testing environment config for faster iterations
  - Consider caching data when running multiple simulations

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

## License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0). See the [LICENSE](LICENSE) file for details.