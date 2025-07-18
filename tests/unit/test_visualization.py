"""Unit tests for visualization module."""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from unittest.mock import patch, Mock

from portfolio_simulator.core.visualization import (
    plot_results, 
    plot_historical_performance, 
    plot_weight_drift, 
    plot_drawdowns
)
from tests.fixtures.sample_data import create_sample_simulation_results


class TestPlotResults:
    """Test cases for plot_results function."""
    
    @pytest.fixture
    def sample_simulation_data(self):
        """Sample simulation data for testing."""
        return create_sample_simulation_results()
    
    def test_plot_results_basic(self, sample_simulation_data):
        """Test basic plot results functionality."""
        sim_final_values = sample_simulation_data['final_values']
        time_horizon_years = 5
        results = {
            'Mean Final Value (Inflation-Adjusted, DCA)': np.mean(sim_final_values),
            'Median Final Value (Inflation-Adjusted, DCA)': np.median(sim_final_values)
        }
        
        fig = plot_results(sim_final_values, time_horizon_years, results)
        
        # Check that a matplotlib figure is returned
        assert isinstance(fig, plt.Figure)
        
        # Check that the figure has the expected elements
        ax = fig.get_axes()[0]
        assert ax.get_title() == 'Distribution of Simulated Portfolio Values after 5 Years (Inflation-Adjusted, DCA)'
        assert ax.get_xlabel() == 'Final Portfolio Value'
        assert ax.get_ylabel() == 'Frequency'
        
        # Check that vertical lines are added for mean and median
        lines = ax.get_lines()
        assert len(lines) >= 2  # At least mean and median lines
        
        # Clean up
        plt.close(fig)
    
    def test_plot_results_with_different_time_horizons(self, sample_simulation_data):
        """Test plot results with different time horizons."""
        sim_final_values = sample_simulation_data['final_values']
        results = {
            'Mean Final Value (Inflation-Adjusted, DCA)': np.mean(sim_final_values),
            'Median Final Value (Inflation-Adjusted, DCA)': np.median(sim_final_values)
        }
        
        for time_horizon in [1, 3, 5, 10]:
            fig = plot_results(sim_final_values, time_horizon, results)
            
            ax = fig.get_axes()[0]
            expected_title = f'Distribution of Simulated Portfolio Values after {time_horizon} Years (Inflation-Adjusted, DCA)'
            assert ax.get_title() == expected_title
            
            plt.close(fig)
    
    def test_plot_results_with_extreme_values(self):
        """Test plot results with extreme values."""
        # Create data with extreme values
        sim_final_values = np.array([1000, 5000, 10000, 15000, 1000000])  # One extreme outlier
        results = {
            'Mean Final Value (Inflation-Adjusted, DCA)': np.mean(sim_final_values),
            'Median Final Value (Inflation-Adjusted, DCA)': np.median(sim_final_values)
        }
        
        fig = plot_results(sim_final_values, 1, results)
        
        # Should handle extreme values gracefully
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)
    
    def test_plot_results_with_small_dataset(self):
        """Test plot results with small dataset."""
        # Very small dataset
        sim_final_values = np.array([10000, 10500, 11000])
        results = {
            'Mean Final Value (Inflation-Adjusted, DCA)': np.mean(sim_final_values),
            'Median Final Value (Inflation-Adjusted, DCA)': np.median(sim_final_values)
        }
        
        fig = plot_results(sim_final_values, 1, results)
        
        # Should handle small datasets
        assert isinstance(fig, plt.Figure)
        
        plt.close(fig)


class TestPlotHistoricalPerformance:
    """Test cases for plot_historical_performance function."""
    
    @pytest.fixture
    def sample_price_data(self):
        """Sample price data for testing."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        return pd.DataFrame({
            'IWDA.AS': 100 * np.cumprod(1 + np.random.normal(0.0004, 0.015, 252)),
            'QDV5.DE': 150 * np.cumprod(1 + np.random.normal(0.0005, 0.020, 252)),
            'PPFB.DE': 80 * np.cumprod(1 + np.random.normal(0.0002, 0.012, 252)),
            'XEON.DE': 50 * np.cumprod(1 + np.random.normal(0.0001, 0.005, 252))
        }, index=dates)
    
    @pytest.fixture
    def sample_weights(self):
        """Sample portfolio weights."""
        return np.array([0.4, 0.3, 0.2, 0.1])
    
    @pytest.fixture
    def sample_tickers(self):
        """Sample ticker list."""
        return ['IWDA.AS', 'QDV5.DE', 'PPFB.DE', 'XEON.DE']
    
    def test_plot_historical_performance_basic(self, sample_price_data, sample_weights, sample_tickers):
        """Test basic historical performance plotting."""
        fig = plot_historical_performance(sample_price_data, sample_weights, sample_tickers)
        
        # Check that a plotly figure is returned
        assert isinstance(fig, go.Figure)
        
        # Check that we have the expected number of traces (4 assets + 1 portfolio)
        assert len(fig.data) == 5
        
        # Check that the portfolio trace is included
        trace_names = [trace.name for trace in fig.data]
        assert 'Portfolio' in trace_names
        
        # Check that all tickers are included
        for ticker in sample_tickers:
            assert ticker in trace_names
    
    def test_plot_historical_performance_single_asset(self, sample_price_data, sample_tickers):
        """Test historical performance with single asset."""
        single_asset_data = sample_price_data[['IWDA.AS']]
        single_asset_weights = np.array([1.0])
        single_ticker = ['IWDA.AS']
        
        fig = plot_historical_performance(single_asset_data, single_asset_weights, single_ticker)
        
        # Should have 2 traces (1 asset + 1 portfolio)
        assert len(fig.data) == 2
        
        trace_names = [trace.name for trace in fig.data]
        assert 'IWDA.AS' in trace_names
        assert 'Portfolio' in trace_names
    
    def test_plot_historical_performance_equal_weights(self, sample_price_data, sample_tickers):
        """Test historical performance with equal weights."""
        n_assets = len(sample_tickers)
        equal_weights = np.array([1/n_assets] * n_assets)
        
        fig = plot_historical_performance(sample_price_data, equal_weights, sample_tickers)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == len(sample_tickers) + 1  # Assets + Portfolio
    
    def test_plot_historical_performance_short_period(self, sample_weights, sample_tickers):
        """Test historical performance with short time period."""
        # Create short period data
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        np.random.seed(42)
        short_data = pd.DataFrame({
            'IWDA.AS': 100 * np.cumprod(1 + np.random.normal(0.001, 0.01, 30)),
            'QDV5.DE': 150 * np.cumprod(1 + np.random.normal(0.001, 0.015, 30)),
            'PPFB.DE': 80 * np.cumprod(1 + np.random.normal(0.0005, 0.008, 30)),
            'XEON.DE': 50 * np.cumprod(1 + np.random.normal(0.0001, 0.003, 30))
        }, index=dates)
        
        fig = plot_historical_performance(short_data, sample_weights, sample_tickers)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 5


class TestPlotWeightDrift:
    """Test cases for plot_weight_drift function."""
    
    @pytest.fixture
    def sample_returns(self):
        """Sample returns data for testing."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        return pd.DataFrame({
            'IWDA.AS': np.random.normal(0.0004, 0.015, 252),
            'QDV5.DE': np.random.normal(0.0005, 0.020, 252),
            'PPFB.DE': np.random.normal(0.0002, 0.012, 252),
            'XEON.DE': np.random.normal(0.0001, 0.005, 252)
        }, index=dates)
    
    @pytest.fixture
    def sample_target_weights(self):
        """Sample target weights."""
        return np.array([0.4, 0.3, 0.2, 0.1])
    
    def test_plot_weight_drift_basic(self, sample_returns, sample_target_weights):
        """Test basic weight drift plotting."""
        fig = plot_weight_drift(sample_returns, sample_target_weights)
        
        # Check that a plotly figure is returned
        assert isinstance(fig, go.Figure)
        
        # Should have traces for each weight + max drift
        # 4 weights + 1 max drift = 5 traces
        assert len(fig.data) == 5
        
        # Check trace names
        trace_names = [trace.name for trace in fig.data]
        assert 'Max Drift' in trace_names
        for i in range(4):
            assert f'Weight {i}' in trace_names
    
    def test_plot_weight_drift_with_rebalancing(self, sample_returns, sample_target_weights):
        """Test weight drift with rebalancing."""
        fig = plot_weight_drift(
            sample_returns, 
            sample_target_weights, 
            rebalance=True, 
            rebalance_frequency='annual', 
            rebalance_threshold=0.05
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 5
    
    def test_plot_weight_drift_quarterly_rebalancing(self, sample_returns, sample_target_weights):
        """Test weight drift with quarterly rebalancing."""
        fig = plot_weight_drift(
            sample_returns, 
            sample_target_weights, 
            rebalance=True, 
            rebalance_frequency='quarterly', 
            rebalance_threshold=0.1
        )
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 5
    
    def test_plot_weight_drift_single_asset(self, sample_returns):
        """Test weight drift with single asset."""
        single_asset_returns = sample_returns[['IWDA.AS']]
        single_target_weights = np.array([1.0])
        
        fig = plot_weight_drift(single_asset_returns, single_target_weights)
        
        # Should have 2 traces (1 weight + 1 max drift)
        assert len(fig.data) == 2
    
    def test_plot_weight_drift_equal_weights(self, sample_returns):
        """Test weight drift with equal weights."""
        n_assets = len(sample_returns.columns)
        equal_weights = np.array([1/n_assets] * n_assets)
        
        fig = plot_weight_drift(sample_returns, equal_weights)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == n_assets + 1  # Weights + Max Drift
    
    def test_plot_weight_drift_short_period(self, sample_target_weights):
        """Test weight drift with short time period."""
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        np.random.seed(42)
        short_returns = pd.DataFrame({
            'IWDA.AS': np.random.normal(0.001, 0.01, 30),
            'QDV5.DE': np.random.normal(0.001, 0.015, 30),
            'PPFB.DE': np.random.normal(0.0005, 0.008, 30),
            'XEON.DE': np.random.normal(0.0001, 0.003, 30)
        }, index=dates)
        
        fig = plot_weight_drift(short_returns, sample_target_weights)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 5


class TestPlotDrawdowns:
    """Test cases for plot_drawdowns function."""
    
    @pytest.fixture
    def sample_returns(self):
        """Sample returns data for testing."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        return pd.DataFrame({
            'IWDA.AS': np.random.normal(0.0004, 0.015, 252),
            'QDV5.DE': np.random.normal(0.0005, 0.020, 252),
            'PPFB.DE': np.random.normal(0.0002, 0.012, 252),
            'XEON.DE': np.random.normal(0.0001, 0.005, 252)
        }, index=dates)
    
    @pytest.fixture
    def sample_weights(self):
        """Sample portfolio weights."""
        return np.array([0.4, 0.3, 0.2, 0.1])
    
    def test_plot_drawdowns_basic(self, sample_returns, sample_weights):
        """Test basic drawdown plotting."""
        fig = plot_drawdowns(sample_returns, sample_weights)
        
        # Check that a plotly figure is returned
        assert isinstance(fig, go.Figure)
        
        # Should have 1 trace (drawdown)
        assert len(fig.data) == 1
        
        # Check trace properties
        trace = fig.data[0]
        assert trace.name == 'Drawdown'
        assert trace.fill == 'tozeroy'  # Should be filled to zero
        
        # Check that drawdown values are <= 0
        drawdown_values = trace.y
        assert all(val <= 0 for val in drawdown_values)
    
    def test_plot_drawdowns_single_asset(self, sample_returns):
        """Test drawdown plotting with single asset."""
        single_asset_returns = sample_returns[['IWDA.AS']]
        single_asset_weights = np.array([1.0])
        
        fig = plot_drawdowns(single_asset_returns, single_asset_weights)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        
        # Drawdown values should be <= 0
        drawdown_values = fig.data[0].y
        assert all(val <= 0 for val in drawdown_values)
    
    def test_plot_drawdowns_equal_weights(self, sample_returns):
        """Test drawdown plotting with equal weights."""
        n_assets = len(sample_returns.columns)
        equal_weights = np.array([1/n_assets] * n_assets)
        
        fig = plot_drawdowns(sample_returns, equal_weights)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
    
    def test_plot_drawdowns_volatile_returns(self, sample_weights):
        """Test drawdown plotting with volatile returns."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        # Create volatile returns with some large negative days
        volatile_returns = pd.DataFrame({
            'IWDA.AS': np.random.normal(0.001, 0.05, 252),  # High volatility
            'QDV5.DE': np.random.normal(0.001, 0.06, 252),
            'PPFB.DE': np.random.normal(0.0005, 0.04, 252),
            'XEON.DE': np.random.normal(0.0001, 0.01, 252)
        }, index=dates)
        
        # Add some extreme negative days
        volatile_returns.iloc[50:55] = -0.1  # Market crash period
        
        fig = plot_drawdowns(volatile_returns, sample_weights)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        
        # Should show significant drawdowns
        drawdown_values = fig.data[0].y
        assert min(drawdown_values) < -0.1  # Should have significant drawdown
    
    def test_plot_drawdowns_bear_market(self, sample_weights):
        """Test drawdown plotting in bear market conditions."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        # Create declining market (bear market)
        bear_returns = pd.DataFrame({
            'IWDA.AS': np.random.normal(-0.002, 0.02, 252),  # Negative trend
            'QDV5.DE': np.random.normal(-0.003, 0.025, 252),
            'PPFB.DE': np.random.normal(-0.001, 0.015, 252),
            'XEON.DE': np.random.normal(0.0001, 0.005, 252)
        }, index=dates)
        
        fig = plot_drawdowns(bear_returns, sample_weights)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        
        # Should show continuous drawdown in bear market
        drawdown_values = fig.data[0].y
        assert min(drawdown_values) < -0.2  # Should have large drawdown
    
    def test_plot_drawdowns_short_period(self, sample_weights):
        """Test drawdown plotting with short time period."""
        dates = pd.date_range('2023-01-01', periods=30, freq='D')
        np.random.seed(42)
        short_returns = pd.DataFrame({
            'IWDA.AS': np.random.normal(0.001, 0.01, 30),
            'QDV5.DE': np.random.normal(0.001, 0.015, 30),
            'PPFB.DE': np.random.normal(0.0005, 0.008, 30),
            'XEON.DE': np.random.normal(0.0001, 0.003, 30)
        }, index=dates)
        
        fig = plot_drawdowns(short_returns, sample_weights)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
    
    def test_plot_drawdowns_positive_only_returns(self, sample_weights):
        """Test drawdown plotting with only positive returns."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # Create only positive returns (bull market)
        positive_returns = pd.DataFrame({
            'IWDA.AS': np.full(252, 0.001),  # Constant positive returns
            'QDV5.DE': np.full(252, 0.0012),
            'PPFB.DE': np.full(252, 0.0008),
            'XEON.DE': np.full(252, 0.0002)
        }, index=dates)
        
        fig = plot_drawdowns(positive_returns, sample_weights)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        
        # With only positive returns, drawdown should be minimal (close to 0)
        drawdown_values = fig.data[0].y
        assert all(val >= -0.01 for val in drawdown_values)  # Very small drawdowns only
    
    def test_plot_drawdowns_zero_returns(self, sample_weights):
        """Test drawdown plotting with zero returns."""
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        
        # Create zero returns
        zero_returns = pd.DataFrame({
            'IWDA.AS': np.zeros(252),
            'QDV5.DE': np.zeros(252),
            'PPFB.DE': np.zeros(252),
            'XEON.DE': np.zeros(252)
        }, index=dates)
        
        fig = plot_drawdowns(zero_returns, sample_weights)
        
        assert isinstance(fig, go.Figure)
        assert len(fig.data) == 1
        
        # With zero returns, drawdown should be exactly 0
        drawdown_values = fig.data[0].y
        assert all(abs(val) < 1e-10 for val in drawdown_values)  # Essentially zero