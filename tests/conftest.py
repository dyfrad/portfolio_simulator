"""Pytest configuration and fixtures for portfolio simulator tests."""

import pytest
import sys
from pathlib import Path

# Add src directory to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

@pytest.fixture
def sample_tickers():
    """Sample ticker list for testing."""
    return ["AAPL", "GOOGL", "MSFT", "TSLA"]

@pytest.fixture
def sample_weights():
    """Sample portfolio weights for testing."""
    return [0.3, 0.3, 0.2, 0.2]

@pytest.fixture
def sample_simulation_params():
    """Sample simulation parameters for testing."""
    return {
        "num_simulations": 1000,
        "simulation_period": 252,  # 1 year
        "initial_investment": 10000
    }

@pytest.fixture
def mock_yfinance_data():
    """Fixture providing mock Yahoo Finance data."""
    from tests.fixtures.sample_data import create_sample_market_data
    return create_sample_market_data()

@pytest.fixture
def mock_yahoo_finance(mock_yfinance_data):
    """Fixture providing mocked Yahoo Finance downloader."""
    from tests.utils import MockYahooFinance
    return MockYahooFinance(mock_yfinance_data)