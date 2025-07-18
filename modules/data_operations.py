"""
Data Operations Module

Handles data fetching, processing, and constants for the portfolio simulator.

Author: Mohit Saharan
Email: mohit@msaharan.com
"""

import yfinance as yf
import pandas as pd

# Default tickers for the assets (UCITS-compliant, EUR-traded)
DEFAULT_TICKERS = ['IWDA.AS', 'QDV5.DE', 'PPFB.DE', 'XEON.DE']  # MSCI World, MSCI India, Gold, Cash

# ISIN to Ticker mapping
ISIN_TO_TICKER = {
    'IE00B4L5Y983': 'IWDA.AS',
    'IE00BHZRQZ17': 'FLXI.DE',
    'IE00B4ND3602': 'PPFB.DE',
    'IE00BZCQB185': 'QDV5.DE',
    'IE00B5BMR087': 'SXR8.DE',
    'IE00B3XXRP09': 'VUSA.AS',
    'US67066G1040': 'NVD.DE',
    'IE00BFY0GT14': 'SPPW.DE',
    'NL0010273215': 'ASML.AS',
    'IE000RHYOR04': 'ERNX.DE',
    'IE00B3FH7618': 'IEGE.MI',
    'LU0290358497': 'XEON.DE'
}

# Default start date for historical data
DEFAULT_START_DATE = '2015-01-01'


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