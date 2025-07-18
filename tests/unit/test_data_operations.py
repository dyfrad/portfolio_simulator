"""Unit tests for data_operations module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock
from datetime import datetime, timedelta

from portfolio_simulator.core.data_operations import fetch_data, calculate_returns
from tests.fixtures.sample_data import create_sample_stock_data, create_sample_market_data
from tests.utils import MockYahooFinance, assert_returns_reasonable


class TestFetchData:
    """Test cases for fetch_data function."""
    
    def test_fetch_data_single_ticker(self, mock_yfinance_data):
        """Test fetching data for a single ticker."""
        with patch('yfinance.download') as mock_download:
            # Setup mock
            ticker = 'IWDA.AS'
            mock_series = mock_yfinance_data[ticker]['Close']
            mock_series.index = mock_yfinance_data[ticker]['Date']
            mock_download.return_value = {'Close': mock_series}
            
            # Test
            result = fetch_data([ticker], '2023-01-01', '2023-12-31')
            
            # Assertions
            assert isinstance(result, pd.Series)
            assert len(result) > 0
            mock_download.assert_called_once_with([ticker], start='2023-01-01', end='2023-12-31')
    
    def test_fetch_data_multiple_tickers(self, mock_yfinance_data):
        """Test fetching data for multiple tickers."""
        with patch('yfinance.download') as mock_download:
            # Setup mock
            tickers = ['IWDA.AS', 'QDV5.DE']
            mock_data = pd.DataFrame({
                ticker: mock_yfinance_data[ticker]['Close']
                for ticker in tickers
            })
            mock_data.index = mock_yfinance_data[tickers[0]]['Date']
            mock_download.return_value = {'Close': mock_data}
            
            # Test
            result = fetch_data(tickers, '2023-01-01', '2023-12-31')
            
            # Assertions
            assert isinstance(result, pd.DataFrame)
            assert list(result.columns) == tickers
            assert len(result) > 0
            mock_download.assert_called_once_with(tickers, start='2023-01-01', end='2023-12-31')
    
    def test_fetch_data_no_end_date(self, mock_yfinance_data):
        """Test fetching data with no end date specified."""
        with patch('yfinance.download') as mock_download:
            ticker = 'IWDA.AS'
            mock_series = mock_yfinance_data[ticker]['Close']
            mock_series.index = mock_yfinance_data[ticker]['Date']
            mock_download.return_value = {'Close': mock_series}
            
            result = fetch_data([ticker], '2023-01-01')
            
            assert isinstance(result, pd.Series)
            mock_download.assert_called_once_with([ticker], start='2023-01-01', end=None)
    
    def test_fetch_data_limited_data_warning(self, mock_yfinance_data, capsys):
        """Test warning when limited historical data is available."""
        with patch('yfinance.download') as mock_download:
            # Create limited data (less than 252 days)
            limited_data = mock_yfinance_data['IWDA.AS'].head(100)
            mock_series = limited_data['Close']
            mock_series.index = limited_data['Date']
            mock_download.return_value = {'Close': mock_series}
            
            result = fetch_data(['IWDA.AS'], '2023-01-01')
            
            # Check warning was printed
            captured = capsys.readouterr()
            assert "Warning: Limited historical data available" in captured.out
            assert len(result) < 252
    
    def test_fetch_data_empty_result(self):
        """Test handling of empty data result."""
        with patch('yfinance.download') as mock_download:
            mock_download.return_value = {'Close': pd.Series(dtype=float)}
            
            result = fetch_data(['INVALID_TICKER'], '2023-01-01')
            
            assert len(result) == 0


class TestCalculateReturns:
    """Test cases for calculate_returns function."""
    
    def test_calculate_returns_single_series(self):
        """Test calculating returns for a single price series."""
        # Create sample price data
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        prices = pd.Series([100, 101, 102, 101, 103, 102, 104, 105, 104, 106], index=dates)
        
        # Test
        returns = calculate_returns(prices)
        
        # Assertions
        assert isinstance(returns, pd.Series)
        assert len(returns) == len(prices) - 1  # One less due to pct_change
        assert not returns.isna().any()
        assert_returns_reasonable(returns.values)
    
    def test_calculate_returns_with_ter(self):
        """Test calculating returns with TER adjustment."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        prices = pd.Series([100, 101, 102, 101, 103, 102, 104, 105, 104, 106], index=dates)
        ter = 0.1  # 0.1% annual TER
        
        # Test
        returns_no_ter = calculate_returns(prices, ter=0.0)
        returns_with_ter = calculate_returns(prices, ter=ter)
        
        # Assertions
        assert isinstance(returns_with_ter, pd.Series)
        assert len(returns_with_ter) == len(returns_no_ter)
        
        # Returns with TER should be slightly lower
        daily_ter = (ter / 100) / 252
        expected_adjustment = returns_no_ter - daily_ter
        pd.testing.assert_series_equal(returns_with_ter, expected_adjustment)
    
    def test_calculate_returns_dataframe(self):
        """Test calculating returns for multiple series (DataFrame)."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        prices = pd.DataFrame({
            'IWDA.AS': [100, 101, 102, 101, 103, 102, 104, 105, 104, 106],
            'QDV5.DE': [50, 51, 50, 52, 51, 53, 52, 54, 53, 55]
        }, index=dates)
        
        # Test
        returns = calculate_returns(prices)
        
        # Assertions
        assert isinstance(returns, pd.DataFrame)
        assert list(returns.columns) == ['IWDA.AS', 'QDV5.DE']
        assert len(returns) == len(prices) - 1
        assert not returns.isna().any().any()
        
        # Check each column
        for col in returns.columns:
            assert_returns_reasonable(returns[col].values)
    
    def test_calculate_returns_with_nan_values(self):
        """Test handling of NaN values in price data."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        prices = pd.Series([100, 101, np.nan, 101, 103, 102, 104, 105, 104, 106], index=dates)
        
        # Test
        returns = calculate_returns(prices)
        
        # Assertions
        assert isinstance(returns, pd.Series)
        assert not returns.isna().any()  # NaN values should be dropped
        assert len(returns) == len(prices) - 1  # One less due to pct_change, NaN handling doesn't change length
    
    def test_calculate_returns_empty_data(self):
        """Test handling of empty price data."""
        empty_prices = pd.Series(dtype=float)
        
        # Test
        returns = calculate_returns(empty_prices)
        
        # Assertions
        assert isinstance(returns, pd.Series)
        assert len(returns) == 0
    
    def test_calculate_returns_single_price_point(self):
        """Test handling of single price point (should return empty)."""
        single_price = pd.Series([100])
        
        # Test
        returns = calculate_returns(single_price)
        
        # Assertions
        assert isinstance(returns, pd.Series)
        assert len(returns) == 0
    
    @pytest.mark.parametrize("ter", [0.0, 0.1, 0.5, 1.0, 2.0])
    def test_calculate_returns_various_ter_values(self, ter):
        """Test calculating returns with various TER values."""
        dates = pd.date_range('2023-01-01', periods=10, freq='D')
        prices = pd.Series([100, 101, 102, 101, 103, 102, 104, 105, 104, 106], index=dates)
        
        # Test
        returns = calculate_returns(prices, ter=ter)
        
        # Assertions
        assert isinstance(returns, pd.Series)
        assert len(returns) == len(prices) - 1
        assert not returns.isna().any()
        
        # Check TER adjustment
        raw_returns = prices.pct_change().dropna()
        daily_ter = (ter / 100) / 252
        expected_returns = raw_returns - daily_ter
        pd.testing.assert_series_equal(returns, expected_returns)
    
    def test_calculate_returns_negative_ter(self):
        """Test that negative TER values work (though unrealistic)."""
        dates = pd.date_range('2023-01-01', periods=5, freq='D')
        prices = pd.Series([100, 101, 102, 101, 103], index=dates)
        
        # Test with negative TER (unrealistic but should work)
        returns = calculate_returns(prices, ter=-0.1)
        
        # Assertions
        assert isinstance(returns, pd.Series)
        assert len(returns) == 4
        assert not returns.isna().any()