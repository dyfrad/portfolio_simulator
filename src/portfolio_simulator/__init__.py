"""
Portfolio Simulator Package

A comprehensive Monte Carlo portfolio simulation tool with modern portfolio theory,
risk analysis, and interactive visualizations.
"""

__version__ = "0.1.0"
__author__ = "Mohit Saharan"
__email__ = "mohit@msaharan.com"

# Import core functions to make them available at package level
from .core.data_operations import fetch_data, calculate_returns
from .core.simulation_engine import bootstrap_simulation
from .core.backtesting import backtest_portfolio
from .core.financial_calculations import optimize_weights