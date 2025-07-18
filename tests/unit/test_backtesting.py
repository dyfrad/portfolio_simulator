"""Unit tests for backtesting module."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from portfolio_simulator.core.backtesting import backtest_portfolio
from tests.fixtures.sample_data import create_sample_stock_data
from tests.utils import (
    assert_portfolio_weights_valid, 
    assert_sharpe_ratio_valid,
    assert_financial_metric_valid
)


class TestBacktestPortfolio:
    """Test cases for backtest_portfolio function."""
    
    @pytest.fixture
    def sample_price_data(self):
        """Sample price data for backtesting."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # Create realistic price data with some trend
        initial_prices = [100, 150, 80, 50]
        price_data = {}
        
        for i, ticker in enumerate(['IWDA.AS', 'QDV5.DE', 'PPFB.DE', 'XEON.DE']):
            prices = [initial_prices[i]]
            for _ in range(251):
                # Add some trend and volatility
                daily_return = np.random.normal(0.0005, 0.015)
                new_price = prices[-1] * (1 + daily_return)
                prices.append(new_price)
            price_data[ticker] = prices
        
        return pd.DataFrame(price_data, index=dates)
    
    @pytest.fixture
    def sample_weights(self):
        """Sample portfolio weights."""
        return np.array([0.4, 0.3, 0.2, 0.1])
    
    def test_backtest_portfolio_basic(self, sample_price_data, sample_weights):
        """Test basic portfolio backtesting."""
        results = backtest_portfolio(
            data=sample_price_data,
            weights=sample_weights,
            initial_investment=10000
        )
        
        # Check results structure
        assert isinstance(results, dict)
        
        # Check required keys
        required_keys = [
            'Total Return (DCA)',
            'Total Return (Lump-Sum)',
            'Annualized Return',
            'Annualized Volatility',
            'Sharpe Ratio',
            'Sortino Ratio',
            'Max Drawdown'
        ]
        
        for key in required_keys:
            assert key in results
            assert isinstance(results[key], (int, float))
            assert not np.isnan(results[key])
    
    def test_backtest_portfolio_with_dca(self, sample_price_data, sample_weights):
        """Test backtesting with Dollar-Cost Averaging."""
        results = backtest_portfolio(
            data=sample_price_data,
            weights=sample_weights,
            initial_investment=10000,
            periodic_contrib=500,
            contrib_frequency='monthly'
        )
        
        # DCA and lump sum results should be different
        dca_return = results['Total Return (DCA)']
        lump_sum_return = results['Total Return (Lump-Sum)']
        
        assert isinstance(dca_return, (int, float))
        assert isinstance(lump_sum_return, (int, float))
        assert not np.isnan(dca_return)
        assert not np.isnan(lump_sum_return)
    
    def test_backtest_portfolio_with_fees(self, sample_price_data, sample_weights):
        """Test backtesting with transaction fees."""
        results_no_fees = backtest_portfolio(
            data=sample_price_data,
            weights=sample_weights,
            initial_investment=10000,
            periodic_contrib=500,
            transaction_fee=0.0
        )
        
        results_with_fees = backtest_portfolio(
            data=sample_price_data,
            weights=sample_weights,
            initial_investment=10000,
            periodic_contrib=500,
            transaction_fee=5.0  # €5 per transaction
        )
        
        # Fees should reduce returns
        assert results_with_fees['Total Return (DCA)'] <= results_no_fees['Total Return (DCA)']
    
    def test_backtest_portfolio_with_taxes(self, sample_price_data, sample_weights):
        """Test backtesting with tax on gains."""
        results_no_tax = backtest_portfolio(
            data=sample_price_data,
            weights=sample_weights,
            initial_investment=10000,
            tax_rate=0.0
        )
        
        results_with_tax = backtest_portfolio(
            data=sample_price_data,
            weights=sample_weights,
            initial_investment=10000,
            tax_rate=0.15  # 15% tax
        )
        
        # Tax should reduce returns (assuming positive gains)
        if results_no_tax['Total Return (Lump-Sum)'] > 0:
            assert results_with_tax['Total Return (Lump-Sum)'] <= results_no_tax['Total Return (Lump-Sum)']
    
    def test_backtest_portfolio_with_rebalancing(self, sample_price_data, sample_weights):
        """Test backtesting with rebalancing."""
        results_no_rebalance = backtest_portfolio(
            data=sample_price_data,
            weights=sample_weights,
            initial_investment=10000,
            rebalance=False
        )
        
        results_with_rebalance = backtest_portfolio(
            data=sample_price_data,
            weights=sample_weights,
            initial_investment=10000,
            rebalance=True,
            rebalance_frequency='annual',
            rebalance_threshold=0.05
        )
        
        # Both should produce valid results
        assert not np.isnan(results_no_rebalance['Total Return (Lump-Sum)'])
        assert not np.isnan(results_with_rebalance['Total Return (Lump-Sum)'])
    
    def test_backtest_portfolio_quarterly_contributions(self, sample_price_data, sample_weights):
        """Test backtesting with quarterly contributions."""
        results = backtest_portfolio(
            data=sample_price_data,
            weights=sample_weights,
            initial_investment=10000,
            periodic_contrib=1500,  # Quarterly contribution
            contrib_frequency='quarterly'
        )
        
        assert isinstance(results['Total Return (DCA)'], (int, float))
        assert not np.isnan(results['Total Return (DCA)'])
    
    def test_backtest_portfolio_quarterly_rebalancing(self, sample_price_data, sample_weights):
        """Test backtesting with quarterly rebalancing."""
        results = backtest_portfolio(
            data=sample_price_data,
            weights=sample_weights,
            initial_investment=10000,
            rebalance=True,
            rebalance_frequency='quarterly',
            rebalance_threshold=0.1
        )
        
        assert isinstance(results['Total Return (Lump-Sum)'], (int, float))
        assert not np.isnan(results['Total Return (Lump-Sum)'])
    
    def test_backtest_portfolio_single_asset(self, sample_price_data):
        """Test backtesting with single asset portfolio."""
        single_asset_data = sample_price_data[['IWDA.AS']]
        single_asset_weights = np.array([1.0])
        
        results = backtest_portfolio(
            data=single_asset_data,
            weights=single_asset_weights,
            initial_investment=10000
        )
        
        # Should produce valid results for single asset
        assert isinstance(results['Total Return (Lump-Sum)'], (int, float))
        assert not np.isnan(results['Total Return (Lump-Sum)'])
        assert_sharpe_ratio_valid(results['Sharpe Ratio'])
    
    def test_backtest_portfolio_equal_weights(self, sample_price_data):
        """Test backtesting with equal weights."""
        n_assets = len(sample_price_data.columns)
        equal_weights = np.array([1/n_assets] * n_assets)
        
        results = backtest_portfolio(
            data=sample_price_data,
            weights=equal_weights,
            initial_investment=10000
        )
        
        # Should produce valid results
        assert not np.isnan(results['Total Return (Lump-Sum)'])
        assert results['Annualized Volatility'] > 0
    
    def test_backtest_portfolio_zero_initial_investment(self, sample_price_data, sample_weights):
        """Test backtesting with zero initial investment."""
        results = backtest_portfolio(
            data=sample_price_data,
            weights=sample_weights,
            initial_investment=0,
            periodic_contrib=1000
        )
        
        # Should handle zero initial investment gracefully
        # DCA should still work with contributions
        assert isinstance(results['Total Return (DCA)'], (int, float))
        # Lump sum return should be 0 or handle division by zero
        assert not np.isnan(results['Total Return (Lump-Sum)'])
    
    def test_backtest_portfolio_short_period(self, sample_weights):
        """Test backtesting with short time period."""
        # Create short period data (1 month)
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        np.random.seed(42)
        
        short_price_data = pd.DataFrame({
            'IWDA.AS': 100 * np.cumprod(1 + np.random.normal(0.001, 0.01, 30)),
            'QDV5.DE': 150 * np.cumprod(1 + np.random.normal(0.001, 0.015, 30)),
            'PPFB.DE': 80 * np.cumprod(1 + np.random.normal(0.0005, 0.008, 30)),
            'XEON.DE': 50 * np.cumprod(1 + np.random.normal(0.0001, 0.003, 30))
        }, index=dates)
        
        results = backtest_portfolio(
            data=short_price_data,
            weights=sample_weights,
            initial_investment=10000
        )
        
        # Should handle short periods
        assert isinstance(results['Total Return (Lump-Sum)'], (int, float))
        assert not np.isnan(results['Total Return (Lump-Sum)'])
    
    def test_backtest_portfolio_volatile_market(self, sample_weights):
        """Test backtesting in highly volatile market conditions."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        # Create highly volatile price data
        volatile_price_data = pd.DataFrame({
            'IWDA.AS': 100 * np.cumprod(1 + np.random.normal(0.001, 0.05, 252)),  # 5% daily vol
            'QDV5.DE': 150 * np.cumprod(1 + np.random.normal(0.001, 0.06, 252)),
            'PPFB.DE': 80 * np.cumprod(1 + np.random.normal(0.0005, 0.04, 252)),
            'XEON.DE': 50 * np.cumprod(1 + np.random.normal(0.0001, 0.01, 252))
        }, index=dates)
        
        results = backtest_portfolio(
            data=volatile_price_data,
            weights=sample_weights,
            initial_investment=10000
        )
        
        # Should handle high volatility
        assert isinstance(results['Annualized Volatility'], (int, float))
        assert results['Annualized Volatility'] > 0.1  # Should reflect high volatility
        assert results['Max Drawdown'] <= 0  # Drawdown should be negative
    
    def test_backtest_portfolio_bear_market(self, sample_weights):
        """Test backtesting in bear market conditions."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        # Create declining price data (bear market)
        bear_price_data = pd.DataFrame({
            'IWDA.AS': 100 * np.cumprod(1 + np.random.normal(-0.002, 0.02, 252)),  # Negative trend
            'QDV5.DE': 150 * np.cumprod(1 + np.random.normal(-0.003, 0.025, 252)),
            'PPFB.DE': 80 * np.cumprod(1 + np.random.normal(-0.001, 0.015, 252)),
            'XEON.DE': 50 * np.cumprod(1 + np.random.normal(0.0001, 0.005, 252))  # Cash-like
        }, index=dates)
        
        results = backtest_portfolio(
            data=bear_price_data,
            weights=sample_weights,
            initial_investment=10000
        )
        
        # Should handle bear market
        assert isinstance(results['Total Return (Lump-Sum)'], (int, float))
        assert results['Max Drawdown'] < 0  # Should have significant drawdown
        assert results['Annualized Return'] <= 0  # Likely negative return
    
    def test_backtest_portfolio_missing_data(self, sample_weights):
        """Test backtesting with missing data points."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        # Create data with some missing values
        price_data_with_nan = pd.DataFrame({
            'IWDA.AS': 100 * np.cumprod(1 + np.random.normal(0.001, 0.015, 252)),
            'QDV5.DE': 150 * np.cumprod(1 + np.random.normal(0.001, 0.02, 252)),
            'PPFB.DE': 80 * np.cumprod(1 + np.random.normal(0.0005, 0.012, 252)),
            'XEON.DE': 50 * np.cumprod(1 + np.random.normal(0.0001, 0.005, 252))
        }, index=dates)
        
        # Introduce some NaN values
        price_data_with_nan.iloc[10:15, 0] = np.nan
        price_data_with_nan.iloc[50:55, 1] = np.nan
        
        results = backtest_portfolio(
            data=price_data_with_nan,
            weights=sample_weights,
            initial_investment=10000
        )
        
        # Should handle missing data gracefully
        assert isinstance(results['Total Return (Lump-Sum)'], (int, float))
        assert not np.isnan(results['Total Return (Lump-Sum)'])
    
    @pytest.mark.parametrize("rebalance_threshold", [0.01, 0.05, 0.1, 0.2])
    def test_backtest_portfolio_various_rebalance_thresholds(self, sample_price_data, sample_weights, rebalance_threshold):
        """Test backtesting with various rebalancing thresholds."""
        results = backtest_portfolio(
            data=sample_price_data,
            weights=sample_weights,
            initial_investment=10000,
            rebalance=True,
            rebalance_frequency='annual',
            rebalance_threshold=rebalance_threshold
        )
        
        assert isinstance(results['Total Return (Lump-Sum)'], (int, float))
        assert not np.isnan(results['Total Return (Lump-Sum)'])
    
    @pytest.mark.parametrize("tax_rate", [0.0, 0.1, 0.15, 0.2, 0.3])
    def test_backtest_portfolio_various_tax_rates(self, sample_price_data, sample_weights, tax_rate):
        """Test backtesting with various tax rates."""
        results = backtest_portfolio(
            data=sample_price_data,
            weights=sample_weights,
            initial_investment=10000,
            tax_rate=tax_rate
        )
        
        assert isinstance(results['Total Return (Lump-Sum)'], (int, float))
        assert not np.isnan(results['Total Return (Lump-Sum)'])
    
    def test_backtest_portfolio_risk_metrics_validity(self, sample_price_data, sample_weights):
        """Test that all risk metrics are within valid ranges."""
        results = backtest_portfolio(
            data=sample_price_data,
            weights=sample_weights,
            initial_investment=10000
        )
        
        # Test individual metrics
        assert_sharpe_ratio_valid(results['Sharpe Ratio'])
        assert_financial_metric_valid(results['Sortino Ratio'], "Sortino Ratio", min_val=-10.0, max_val=10.0)
        assert_financial_metric_valid(results['Annualized Volatility'], "Annualized Volatility", min_val=0.0)
        assert_financial_metric_valid(results['Max Drawdown'], "Max Drawdown", max_val=0.0)
    
    def test_backtest_portfolio_large_contributions(self, sample_price_data, sample_weights):
        """Test backtesting with large periodic contributions."""
        results = backtest_portfolio(
            data=sample_price_data,
            weights=sample_weights,
            initial_investment=10000,
            periodic_contrib=5000,  # Large monthly contribution
            contrib_frequency='monthly'
        )
        
        # Should handle large contributions
        assert isinstance(results['Total Return (DCA)'], (int, float))
        assert not np.isnan(results['Total Return (DCA)'])
    
    def test_backtest_portfolio_high_transaction_fees(self, sample_price_data, sample_weights):
        """Test backtesting with high transaction fees."""
        results = backtest_portfolio(
            data=sample_price_data,
            weights=sample_weights,
            initial_investment=10000,
            periodic_contrib=500,
            transaction_fee=50.0  # High fee
        )
        
        # Should handle high fees (might result in negative returns)
        assert isinstance(results['Total Return (DCA)'], (int, float))
        assert not np.isnan(results['Total Return (DCA)'])
        # High fees should significantly impact returns
        assert results['Total Return (DCA)'] < 0.5  # Should be much lower due to fees