"""
Data Operations Module

Handles data fetching, processing, and constants for the portfolio simulator.

Author: Mohit Saharan
Email: mohit@msaharan.com
"""

import yfinance as yf
import pandas as pd

# Import constants from config
from ..config.constants import DEFAULT_TICKERS, ISIN_TO_TICKER, DEFAULT_START_DATE


def fetch_data(tickers, start_date, end_date=None):
    """
    Fetch historical adjusted close prices for given tickers.
    """
    data = yf.download(tickers, start=start_date, end=end_date)['Close'].dropna()
    if len(data) < 252:
        print("Warning: Limited historical data available. Simulations may not be reliable.")
    return data


def calculate_returns(data, ter=0.0):
    """
    Calculate daily returns from price data, adjusted for TER.
    """
    returns = data.pct_change().dropna()
    daily_ter = (ter / 100) / 252  # Annual TER to daily deduction
    returns -= daily_ter
    return returns