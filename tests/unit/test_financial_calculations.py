"""Unit tests for financial_calculations module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch

from portfolio_simulator.core.financial_calculations import portfolio_stats, optimize_weights
from tests.fixtures.sample_data import create_sample_market_data
from tests.utils import (
    assert_portfolio_weights_valid, 
    assert_returns_reasonable, 
    assert_sharpe_ratio_valid,
    assert_financial_metric_valid
)


class TestPortfolioStats:
    """Test cases for portfolio_stats function."""
    
    @pytest.fixture
    def sample_returns(self):
        """Sample returns data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        return pd.DataFrame({
            'IWDA.AS': np.random.normal(0.0004, 0.015, 252),
            'QDV5.DE': np.random.normal(0.0005, 0.020, 252),
            'PPFB.DE': np.random.normal(0.0002, 0.012, 252),
            'XEON.DE': np.random.normal(0.0001, 0.005, 252)
        }, index=dates)
    
    def test_portfolio_stats_basic(self, sample_returns):
        """Test basic portfolio statistics calculation."""
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        
        annual_ret, annual_vol, sharpe, sortino, max_dd = portfolio_stats(weights, sample_returns)
        
        # Basic assertions
        assert isinstance(annual_ret, float)
        assert isinstance(annual_vol, float)
        assert isinstance(sharpe, float)
        assert isinstance(sortino, float)
        assert isinstance(max_dd, float)
        
        # Value range assertions
        assert annual_vol > 0, "Annual volatility should be positive"
        assert max_dd <= 0, "Maximum drawdown should be negative or zero"
        assert_sharpe_ratio_valid(sharpe)
        assert_financial_metric_valid(sortino, "Sortino ratio", min_val=-10.0, max_val=10.0)
    
    def test_portfolio_stats_equal_weights(self, sample_returns):
        """Test portfolio statistics with equal weights."""
        n_assets = len(sample_returns.columns)
        weights = np.array([1/n_assets] * n_assets)
        
        annual_ret, annual_vol, sharpe, sortino, max_dd = portfolio_stats(weights, sample_returns)
        
        # Should produce valid statistics
        assert not np.isnan(annual_ret)
        assert not np.isnan(annual_vol)
        assert not np.isnan(sharpe)
        assert not np.isnan(sortino)
        assert not np.isnan(max_dd)
    
    def test_portfolio_stats_single_asset(self, sample_returns):
        """Test portfolio statistics with single asset (100% weight)."""
        weights = np.array([1.0, 0.0, 0.0, 0.0])
        
        annual_ret, annual_vol, sharpe, sortino, max_dd = portfolio_stats(weights, sample_returns)
        
        # Should match single asset statistics
        single_asset_returns = sample_returns.iloc[:, 0]
        expected_annual_ret = single_asset_returns.mean() * 252
        expected_annual_vol = single_asset_returns.std() * np.sqrt(252)
        
        assert abs(annual_ret - expected_annual_ret) < 1e-10
        assert abs(annual_vol - expected_annual_vol) < 1e-10
    
    def test_portfolio_stats_with_cash_ticker(self, sample_returns):
        """Test portfolio statistics with cash ticker for risk-free rate."""
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        cash_ticker = 'XEON.DE'
        
        annual_ret, annual_vol, sharpe, sortino, max_dd = portfolio_stats(
            weights, sample_returns, cash_ticker=cash_ticker
        )
        
        # Should use cash ticker as risk-free rate
        expected_rf_rate = sample_returns[cash_ticker].mean() * 252
        expected_sharpe = (annual_ret - expected_rf_rate) / annual_vol if annual_vol != 0 else 0
        
        assert abs(sharpe - expected_sharpe) < 1e-10
    
    def test_portfolio_stats_no_cash_ticker(self, sample_returns):
        """Test portfolio statistics without cash ticker."""
        weights = np.array([0.4, 0.3, 0.2, 0.1])
        
        annual_ret, annual_vol, sharpe, sortino, max_dd = portfolio_stats(
            weights, sample_returns, cash_ticker='NONEXISTENT'
        )
        
        # Should use 0 as risk-free rate
        expected_sharpe = annual_ret / annual_vol if annual_vol != 0 else 0
        assert abs(sharpe - expected_sharpe) < 1e-10
    
    def test_portfolio_stats_zero_volatility(self):
        """Test portfolio statistics with zero volatility (constant returns)."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        constant_returns = pd.DataFrame({
            'ASSET1': np.zeros(252),
            'ASSET2': np.zeros(252)
        }, index=dates)
        
        weights = np.array([0.5, 0.5])
        
        annual_ret, annual_vol, sharpe, sortino, max_dd = portfolio_stats(weights, constant_returns)
        
        assert annual_vol == 0
        assert sharpe == 0  # Should handle division by zero
        assert sortino == 0  # Should handle division by zero
        assert max_dd == 0  # No drawdown with constant returns
    
    def test_portfolio_stats_negative_returns(self):
        """Test portfolio statistics with consistently negative returns."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        negative_returns = pd.DataFrame({
            'ASSET1': np.full(252, -0.001),
            'ASSET2': np.full(252, -0.002)
        }, index=dates)
        
        weights = np.array([0.5, 0.5])
        
        annual_ret, annual_vol, sharpe, sortino, max_dd = portfolio_stats(weights, negative_returns)
        
        assert annual_ret < 0
        assert max_dd < 0
        assert sharpe < 0
    
    @pytest.mark.parametrize("weights", [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.25, 0.25, 0.25, 0.25],
        [0.5, 0.3, 0.2, 0.0],
    ])
    def test_portfolio_stats_various_weights(self, sample_returns, weights):
        """Test portfolio statistics with various weight combinations."""
        weights = np.array(weights)
        
        annual_ret, annual_vol, sharpe, sortino, max_dd = portfolio_stats(weights, sample_returns)
        
        # All statistics should be finite
        assert np.isfinite(annual_ret)
        assert np.isfinite(annual_vol)
        assert np.isfinite(sharpe)
        assert np.isfinite(sortino)
        assert np.isfinite(max_dd)
        
        # Volatility should be non-negative
        assert annual_vol >= 0
        assert max_dd <= 0


class TestOptimizeWeights:
    """Test cases for optimize_weights function."""
    
    @pytest.fixture
    def sample_returns(self):
        """Sample returns data for testing."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        return pd.DataFrame({
            'IWDA.AS': np.random.normal(0.0004, 0.015, 252),
            'QDV5.DE': np.random.normal(0.0005, 0.020, 252),
            'PPFB.DE': np.random.normal(0.0002, 0.012, 252),
            'XEON.DE': np.random.normal(0.0001, 0.005, 252)
        }, index=dates)
    
    def test_optimize_weights_basic(self, sample_returns):
        """Test basic weight optimization."""
        optimal_weights = optimize_weights(sample_returns)
        
        assert optimal_weights is not None
        assert len(optimal_weights) == len(sample_returns.columns)
        assert_portfolio_weights_valid(optimal_weights)
    
    def test_optimize_weights_with_cash_ticker(self, sample_returns):
        """Test weight optimization with cash ticker."""
        cash_ticker = 'XEON.DE'
        optimal_weights = optimize_weights(sample_returns, cash_ticker=cash_ticker)
        
        assert optimal_weights is not None
        assert_portfolio_weights_valid(optimal_weights)
    
    def test_optimize_weights_improves_sharpe(self, sample_returns):
        """Test that optimization improves Sharpe ratio."""
        # Equal weights as baseline
        equal_weights = np.array([0.25, 0.25, 0.25, 0.25])
        _, _, equal_sharpe, _, _ = portfolio_stats(equal_weights, sample_returns)
        
        # Optimized weights
        optimal_weights = optimize_weights(sample_returns)
        _, _, optimal_sharpe, _, _ = portfolio_stats(optimal_weights, sample_returns)
        
        # Optimal should be better than or equal to equal weights
        assert optimal_sharpe >= equal_sharpe - 1e-6  # Allow for numerical precision
    
    def test_optimize_weights_two_assets(self, sample_returns):
        """Test optimization with only two assets."""
        two_asset_returns = sample_returns[['IWDA.AS', 'QDV5.DE']]
        
        optimal_weights = optimize_weights(two_asset_returns)
        
        assert optimal_weights is not None
        assert len(optimal_weights) == 2
        assert_portfolio_weights_valid(optimal_weights)
    
    def test_optimize_weights_single_asset(self, sample_returns):
        """Test optimization with single asset."""
        single_asset_returns = sample_returns[['IWDA.AS']]
        
        optimal_weights = optimize_weights(single_asset_returns)
        
        assert optimal_weights is not None
        assert len(optimal_weights) == 1
        assert abs(optimal_weights[0] - 1.0) < 1e-10
    
    def test_optimize_weights_identical_assets(self):
        """Test optimization with identical assets."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        returns = np.random.normal(0.0004, 0.015, 252)
        
        identical_returns = pd.DataFrame({
            'ASSET1': returns,
            'ASSET2': returns,
            'ASSET3': returns
        }, index=dates)
        
        optimal_weights = optimize_weights(identical_returns)
        
        assert optimal_weights is not None
        assert_portfolio_weights_valid(optimal_weights)
        # With identical assets, weights should be approximately equal
        assert all(abs(w - 1/3) < 0.1 for w in optimal_weights)
    
    def test_optimize_weights_negative_correlation(self):
        """Test optimization with negatively correlated assets."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        base_returns = np.random.normal(0.0004, 0.015, 252)
        
        corr_returns = pd.DataFrame({
            'ASSET1': base_returns,
            'ASSET2': -base_returns * 0.8 + np.random.normal(0, 0.005, 252)
        }, index=dates)
        
        optimal_weights = optimize_weights(corr_returns)
        
        assert optimal_weights is not None
        assert_portfolio_weights_valid(optimal_weights)
        # With negative correlation, both assets should have significant weights
        assert all(w > 0.1 for w in optimal_weights)
    
    def test_optimize_weights_high_volatility_asset(self):
        """Test optimization with one very high volatility asset."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        mixed_vol_returns = pd.DataFrame({
            'LOW_VOL': np.random.normal(0.0003, 0.008, 252),
            'HIGH_VOL': np.random.normal(0.0005, 0.050, 252)
        }, index=dates)
        
        optimal_weights = optimize_weights(mixed_vol_returns)
        
        assert optimal_weights is not None
        assert_portfolio_weights_valid(optimal_weights)
        # Low volatility asset should typically get higher weight
        assert optimal_weights[0] > optimal_weights[1]
    
    def test_optimize_weights_convergence_failure(self):
        """Test handling of optimization convergence failure."""
        # Create problematic data that might cause convergence issues
        dates = pd.date_range('2023-01-01', periods=10, freq='D')  # Very limited data
        problematic_returns = pd.DataFrame({
            'ASSET1': np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1, 0.1, -0.1]),
            'ASSET2': np.array([np.nan, np.nan, np.nan, np.nan, np.nan, 0.01, 0.01, 0.01, 0.01, 0.01])
        }, index=dates).dropna()
        
        # Even with problematic data, should handle gracefully
        optimal_weights = optimize_weights(problematic_returns)
        
        if optimal_weights is not None:
            assert_portfolio_weights_valid(optimal_weights)
        # If optimization fails, should return None (handled by the function)
    
    @pytest.mark.parametrize("n_assets", [2, 3, 4, 5])
    def test_optimize_weights_various_asset_counts(self, n_assets):
        """Test optimization with various numbers of assets."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        returns_dict = {}
        for i in range(n_assets):
            returns_dict[f'ASSET{i+1}'] = np.random.normal(0.0004, 0.015, 252)
        
        returns = pd.DataFrame(returns_dict, index=dates)
        optimal_weights = optimize_weights(returns)
        
        assert optimal_weights is not None
        assert len(optimal_weights) == n_assets
        assert_portfolio_weights_valid(optimal_weights)