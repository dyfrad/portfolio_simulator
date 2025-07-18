"""Sample data for portfolio simulator tests."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_sample_stock_data(ticker: str, days: int = 252) -> pd.DataFrame:
    """Create sample stock price data for testing."""
    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate sample price data with some volatility
    np.random.seed(42)  # For reproducible tests
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    return pd.DataFrame({
        'Date': dates,
        'Open': prices,
        'High': [p * 1.02 for p in prices],
        'Low': [p * 0.98 for p in prices],
        'Close': prices,
        'Volume': np.random.randint(1000000, 10000000, len(dates))
    })

def create_sample_portfolio_data():
    """Create sample portfolio data for testing."""
    return {
        'tickers': ['IWDA.AS', 'QDV5.DE', 'PPFB.DE', 'XEON.DE'],
        'weights': [0.3, 0.3, 0.2, 0.2],
        'ter_rates': [0.001, 0.001, 0.001, 0.001],
        'initial_investment': 10000
    }

def create_sample_market_data():
    """Create sample market data for multiple tickers."""
    tickers = ['IWDA.AS', 'QDV5.DE', 'PPFB.DE', 'XEON.DE']
    market_data = {}
    
    for ticker in tickers:
        market_data[ticker] = create_sample_stock_data(ticker)
    
    return market_data

def create_sample_degiro_csv():
    """Create sample Degiro CSV data for testing."""
    return """Date,Time,Product,ISIN,Description,FX,Change,Balance,Order Id
01-01-2023,09:00,IWDA.AS,IE00B4L5Y983,ISHARES CORE MSCI WORLD UCITS ETF,EUR,500.00,500.00,
01-02-2023,09:00,QDV5.DE,IE00BZCQB185,AMUNDI MSCI INDIA UCITS ETF,EUR,300.00,800.00,
01-03-2023,09:00,PPFB.DE,IE00B4ND3602,LYXOR GOLD BULLION SECURITIES,EUR,200.00,1000.00,
01-04-2023,09:00,XEON.DE,LU0290358497,XTRACKERS EURO OVERNIGHT RATE,EUR,200.00,1200.00,"""

def create_sample_simulation_results():
    """Create sample simulation results for testing."""
    np.random.seed(42)
    return {
        'final_values': np.random.normal(12000, 2000, 1000),
        'returns': np.random.normal(0.2, 0.15, 1000),
        'sharpe_ratio': 1.2,
        'var_95': -0.15,
        'cvar_95': -0.22,
        'max_drawdown': -0.18
    }