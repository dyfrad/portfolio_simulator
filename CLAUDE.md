# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Style
- Act as a senior developer with expertise in the codebase
- Do the job when asked without trying to teach or explain unless specifically requested
- Answer with low verbosity - be direct and concise
- Focus on delivering results rather than educational content

## Commands

### Streamlit Application
```bash
# Activate conda environment
conda activate pyfin

# Run the Streamlit application
streamlit run portfolio_simulator_ui.py

# Install dependencies
pip install -r requirements.txt
```

## Architecture

### Repository Structure
This is a portfolio simulator application built with Python and Streamlit:

- **Root Directory**: Contains the complete Streamlit portfolio simulator application
- **Core Module**: `portfolio_simulator.py` - Main simulation engine and financial calculations
- **UI Module**: `portfolio_simulator_ui.py` - Streamlit user interface

### Core Components

#### Main Application (`portfolio_simulator.py`)
- Monte Carlo simulation engine
- Financial calculations (Sharpe ratio, VaR, CVaR, etc.)
- Portfolio optimization using scipy
- Yahoo Finance data fetching
- Report generation (PDF)

#### User Interface (`portfolio_simulator_ui.py`)
- Streamlit-based web interface
- Interactive portfolio builder
- Real-time simulation results
- Chart visualizations
- CSV import functionality

### Key Features
- **Monte Carlo Simulations**: Up to 50,000 iterations for portfolio analysis
- **Portfolio Optimization**: Sharpe ratio maximization using modern portfolio theory
- **Historical Backtesting**: Performance analysis with Dollar-Cost Averaging
- **Risk Metrics**: VaR, CVaR, Sortino ratio, maximum drawdown
- **Interactive Charts**: Plotly visualizations for distributions and historical data
- **CSV Import**: Support for Degiro transaction history parsing
- **PDF Reports**: Automated report generation with charts and metrics

### Financial Calculations
The core financial logic includes:
- Historical data fetching from Yahoo Finance
- Daily returns calculation with TER (Total Expense Ratio) adjustment
- Portfolio weight optimization using scipy.optimize
- Monte Carlo simulation with bootstrap sampling
- Risk metrics calculations (VaR, CVaR, Sharpe ratio, etc.)

### Configuration
- **Python Environment**: Uses conda environment named `pyfin` for Python dependencies
- **External APIs**: Yahoo Finance for market data (yfinance library)
- **Data Storage**: Session state management for user portfolios and results

### Common Development Tasks
When working on this codebase:
1. **Adding new financial metrics**: Implement in `portfolio_simulator.py` and update UI in `portfolio_simulator_ui.py`
2. **UI improvements**: Modify Streamlit components in `portfolio_simulator_ui.py`
3. **New chart types**: Add Plotly visualizations to both core and UI modules
4. **Performance optimization**: Focus on Monte Carlo simulation efficiency
5. **Feature development**: Ensure all changes maintain the application's financial accuracy

### Technology Stack
- **Backend**: Python, pandas, numpy, scipy (for optimization)
- **Frontend**: Streamlit for web interface
- **Financial**: yfinance for market data
- **Visualization**: Plotly for interactive charts
- **Reports**: matplotlib, reportlab for PDF generation

This codebase represents a sophisticated financial analysis tool with complex Monte Carlo simulations, portfolio optimization, and comprehensive risk analysis capabilities.