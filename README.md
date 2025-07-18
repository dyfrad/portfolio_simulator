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

## Installation

Since this is a cloned repository for personal use:

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
   
   **Alternative**: Install from requirements.txt for legacy compatibility:
   ```bash
   pip install -r requirements.txt
   ```

### Modern Package Installation

The project now uses modern Python packaging with `pyproject.toml`. This provides:
- **Proper dependency management** with optional dependencies for development and testing
- **Entry point scripts** accessible from anywhere after installation
- **Editable installs** for development with `pip install -e .`

### Development Dependencies

For development and testing, install with development dependencies:
```bash
pip install -e ".[dev,test]"
```

This includes tools for:
- **Testing**: pytest, pytest-cov, pytest-mock
- **Code Quality**: black, flake8, mypy, pre-commit
- **Documentation**: sphinx, sphinx-rtd-theme

## Running the Application

### Method 1: Using Entry Point Scripts (Recommended)
After installing the package, you can run the application from anywhere:
```bash
# Run the Streamlit UI
portfolio-ui

# Or run the simulation engine directly
portfolio-simulator
```

### Method 2: Using Legacy Entry Point
For backward compatibility, you can still use the legacy entry point:
```bash
streamlit run portfolio_simulator_ui.py
```

### Method 3: Using Script Files
Navigate to the scripts directory and run:
```bash
python scripts/run_ui.py
```

### Application Access
The application will launch in your default web browser (typically at `http://localhost:8501`).

### Development Guidelines
- **Core Logic**: Located in `src/portfolio_simulator/core/` modules
- **UI Components**: Organized in `src/portfolio_simulator/ui/` 
- **Configuration**: Managed through `src/portfolio_simulator/config/`
- **Testing**: Comprehensive test suite in `tests/` directory

**Note**: Data is fetched from Yahoo Finance, so an internet connection is required for simulations.

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

4. **Customization**:
   - Add custom tickers in the sidebar (comma-separated).
   - Modify default tickers or ISIN mappings in the code if needed.
   - Extend features by editing functions (e.g., add new stress scenarios).

## Configuration

The application uses a sophisticated configuration management system:

### Configuration Structure
- **Constants**: Default tickers and mappings in `src/portfolio_simulator/config/constants.py`
- **Settings**: Application settings in `src/portfolio_simulator/config/settings.py`
- **Environments**: Environment-specific configurations in `src/portfolio_simulator/config/environments/`

### Key Configuration Options
- **Default Start Date**: '2015-01-01' (configurable)
- **ISIN to Ticker Mapping**: Extensible mapping in `constants.py`
- **Environment Settings**: Separate configs for development, production, and testing
- **Cache**: Uses Streamlit's `@st.cache_data` for performance optimization
- **Session State**: Intelligent state management to avoid re-computation

### Environment Variables
Set environment-specific behavior using:
```bash
export ENVIRONMENT=development  # or production, testing
export DEBUG=true              # Enable debug mode
```

## Development

The codebase follows modern Python packaging standards and clean architecture principles:

### Architecture Overview
- **Separation of Concerns**: Business logic separated from UI and configuration
- **Modular Design**: Each module has a single responsibility
- **Testable Code**: Comprehensive test coverage with unit and integration tests
- **Configuration Management**: Environment-specific settings and constants

### Development Workflow
1. **Core Financial Logic**: Implement in `src/portfolio_simulator/core/`
2. **UI Components**: Create reusable components in `src/portfolio_simulator/ui/components/`
3. **Configuration**: Add settings to appropriate config modules
4. **Testing**: Write tests in `tests/unit/` and `tests/integration/`

### Testing
Run the comprehensive test suite:
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/portfolio_simulator

# Run specific test types
pytest tests/unit/          # Unit tests only
pytest tests/integration/   # Integration tests only
```

### Code Quality
The project includes tools for maintaining code quality:
```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

## Testing Infrastructure

The project includes a comprehensive testing framework with 135+ tests achieving 44% code coverage:

### Test Structure
```
tests/
├── conftest.py                 # Pytest configuration and shared fixtures
├── fixtures/                   # Test data and mock objects
│   └── sample_data.py         # Sample portfolio and market data
├── unit/                      # Unit tests (104 tests)
│   ├── test_data_operations.py       # Data fetching and processing tests
│   ├── test_financial_calculations.py # Portfolio statistics and optimization tests  
│   ├── test_simulation_engine.py     # Monte Carlo simulation tests
│   ├── test_backtesting.py          # Historical backtesting tests
│   ├── test_visualization.py        # Chart generation tests
│   └── test_config.py              # Configuration management tests
├── integration/               # Integration tests (10 tests)
│   └── test_portfolio_simulator.py  # End-to-end workflow tests
└── utils.py                   # Test utilities and assertion helpers
```

### Test Categories

- **Unit Tests**: Test individual functions and methods in isolation
- **Integration Tests**: Test complete workflows from data fetch to visualization
- **Mock Objects**: Comprehensive mocking of external dependencies (Yahoo Finance, etc.)
- **Parametrized Tests**: Multiple test scenarios with different inputs
- **Financial Assertions**: Specialized assertions for validating financial calculations

### Test Execution

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test categories
pytest tests/unit/                    # Unit tests only
pytest tests/integration/             # Integration tests only
pytest -m "not slow"                  # Skip slow tests

# Run with coverage reporting
pytest --cov=src/portfolio_simulator --cov-report=html
```

### Test Coverage
Current test coverage focuses on core financial functionality:
- **Core Modules**: 98-100% coverage for financial calculations, simulation, and backtesting
- **Configuration**: 100% coverage for settings and environment management  
- **Overall**: 44% total coverage with comprehensive testing of critical financial logic

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