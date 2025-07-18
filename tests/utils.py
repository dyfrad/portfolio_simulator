"""Test utilities and helpers for portfolio simulator tests."""

import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
from typing import Dict, List, Any
import pytest


class MockYahooFinance:
    """Mock Yahoo Finance data provider for testing."""
    
    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data
    
    def download(self, tickers: List[str], start: str = None, end: str = None, **kwargs):
        """Mock yfinance download function."""
        if isinstance(tickers, str):
            tickers = [tickers]
        
        result = {}
        for ticker in tickers:
            if ticker in self.data:
                df = self.data[ticker].copy()
                if start:
                    df = df[df['Date'] >= start]
                if end:
                    df = df[df['Date'] <= end]
                result[ticker] = df
        
        return result


def assert_portfolio_weights_valid(weights: List[float], tolerance: float = 1e-6):
    """Assert that portfolio weights are valid (sum to 1, all positive)."""
    assert abs(sum(weights) - 1.0) < tolerance, f"Weights sum to {sum(weights)}, not 1.0"
    assert all(w >= 0 for w in weights), f"Negative weights found: {weights}"


def assert_returns_reasonable(returns: np.ndarray, min_return: float = -0.9, max_return: float = 5.0):
    """Assert that returns are within reasonable bounds."""
    assert np.all(returns >= min_return), f"Returns below {min_return}: {returns[returns < min_return]}"
    assert np.all(returns <= max_return), f"Returns above {max_return}: {returns[returns > max_return]}"


def assert_financial_metric_valid(metric: float, metric_name: str, min_val: float = None, max_val: float = None):
    """Assert that a financial metric is valid."""
    assert not np.isnan(metric), f"{metric_name} is NaN"
    assert not np.isinf(metric), f"{metric_name} is infinite"
    
    if min_val is not None:
        assert metric >= min_val, f"{metric_name} {metric} below minimum {min_val}"
    
    if max_val is not None:
        assert metric <= max_val, f"{metric_name} {metric} above maximum {max_val}"


def create_mock_streamlit_session():
    """Create a mock Streamlit session state for testing."""
    session_state = Mock()
    session_state.portfolio_data = None
    session_state.simulation_results = None
    session_state.current_portfolio = []
    return session_state


@pytest.fixture
def mock_yfinance_data():
    """Fixture providing mock Yahoo Finance data."""
    from tests.fixtures.sample_data import create_sample_market_data
    return create_sample_market_data()


@pytest.fixture
def mock_yahoo_finance(mock_yfinance_data):
    """Fixture providing mocked Yahoo Finance downloader."""
    return MockYahooFinance(mock_yfinance_data)


def parametrize_tickers():
    """Decorator for parametrizing tests with different ticker combinations."""
    return pytest.mark.parametrize("tickers", [
        ['IWDA.AS'],
        ['IWDA.AS', 'QDV5.DE'],
        ['IWDA.AS', 'QDV5.DE', 'PPFB.DE'],
        ['IWDA.AS', 'QDV5.DE', 'PPFB.DE', 'XEON.DE'],
    ])


def parametrize_simulation_params():
    """Decorator for parametrizing tests with different simulation parameters."""
    return pytest.mark.parametrize("num_simulations,simulation_period", [
        (100, 30),    # Small test
        (500, 60),    # Medium test
        (1000, 252),  # Full year
    ])


class FinancialAssertions:
    """Helper class for financial calculation assertions."""
    
    @staticmethod
    def assert_sharpe_ratio_valid(sharpe_ratio: float):
        """Assert Sharpe ratio is within reasonable bounds."""
        assert_financial_metric_valid(sharpe_ratio, "Sharpe ratio", min_val=-5.0, max_val=5.0)
    
    @staticmethod
    def assert_var_valid(var: float):
        """Assert Value at Risk is valid (negative value)."""
        assert_financial_metric_valid(var, "VaR", max_val=0.0)
    
    @staticmethod
    def assert_cvar_valid(cvar: float, var: float):
        """Assert Conditional VaR is valid (more negative than VaR)."""
        assert_financial_metric_valid(cvar, "CVaR", max_val=0.0)
        assert cvar <= var, f"CVaR {cvar} should be <= VaR {var}"
    
    @staticmethod
    def assert_correlation_matrix_valid(corr_matrix: np.ndarray):
        """Assert correlation matrix is valid."""
        assert corr_matrix.shape[0] == corr_matrix.shape[1], "Correlation matrix must be square"
        assert np.allclose(np.diag(corr_matrix), 1.0), "Diagonal elements must be 1.0"
        assert np.allclose(corr_matrix, corr_matrix.T), "Correlation matrix must be symmetric"
        assert np.all(corr_matrix >= -1.0) and np.all(corr_matrix <= 1.0), "Correlations must be [-1, 1]"


# Export commonly used assertions for convenience
assert_sharpe_ratio_valid = FinancialAssertions.assert_sharpe_ratio_valid
assert_var_valid = FinancialAssertions.assert_var_valid
assert_cvar_valid = FinancialAssertions.assert_cvar_valid
assert_correlation_matrix_valid = FinancialAssertions.assert_correlation_matrix_valid