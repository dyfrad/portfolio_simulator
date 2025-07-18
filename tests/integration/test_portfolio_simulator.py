"""Integration tests for portfolio simulator."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, Mock
import tempfile
import os

from portfolio_simulator.core.data_operations import fetch_data, calculate_returns
from portfolio_simulator.core.financial_calculations import portfolio_stats, optimize_weights
from portfolio_simulator.core.simulation_engine import bootstrap_simulation
from portfolio_simulator.core.backtesting import backtest_portfolio
from portfolio_simulator.core.visualization import plot_results, plot_historical_performance
from portfolio_simulator.config.constants import DEFAULT_TICKERS, ISIN_TO_TICKER
from portfolio_simulator.config.settings import get_settings

from tests.fixtures.sample_data import create_sample_market_data, create_sample_portfolio_data
from tests.utils import (
    assert_portfolio_weights_valid, 
    assert_sharpe_ratio_valid,
    MockYahooFinance
)


class TestEndToEndPortfolioSimulation:
    """Integration tests for complete portfolio simulation workflow."""
    
    @pytest.fixture
    def sample_market_data(self):
        """Sample market data for integration testing."""
        return create_sample_market_data()
    
    @pytest.fixture
    def sample_portfolio(self):
        """Sample portfolio configuration."""
        return create_sample_portfolio_data()
    
    def test_complete_simulation_workflow(self, sample_market_data, sample_portfolio):
        """Test complete simulation workflow from data to results."""
        # Mock yfinance data
        with patch('yfinance.download') as mock_download:
            # Setup mock data
            mock_data = {}
            for ticker in sample_portfolio['tickers']:
                mock_data[ticker] = sample_market_data[ticker].set_index('Date')['Close']
            mock_download.return_value = {'Close': pd.DataFrame(mock_data)}
            
            # Step 1: Fetch data
            data = fetch_data(
                sample_portfolio['tickers'], 
                '2023-01-01', 
                '2023-12-31'
            )
            
            assert isinstance(data, pd.DataFrame)
            assert len(data) > 0
            
            # Step 2: Calculate returns
            returns = calculate_returns(data)
            
            assert isinstance(returns, pd.DataFrame)
            assert len(returns) == len(data) - 1
            
            # Step 3: Optimize weights
            optimal_weights = optimize_weights(returns)
            
            assert optimal_weights is not None
            assert_portfolio_weights_valid(optimal_weights)
            
            # Step 4: Calculate portfolio statistics
            ann_ret, ann_vol, sharpe, sortino, max_dd = portfolio_stats(optimal_weights, returns)
            
            assert isinstance(ann_ret, float)
            assert isinstance(ann_vol, float)
            assert_sharpe_ratio_valid(sharpe)
            assert max_dd <= 0
            
            # Step 5: Run simulation
            sim_results, sim_values = bootstrap_simulation(
                returns=returns,
                weights=optimal_weights,
                num_simulations=100,
                time_horizon_years=1,
                initial_investment=10000
            )
            
            assert isinstance(sim_results, dict)
            assert isinstance(sim_values, np.ndarray)
            assert len(sim_values) == 100
            
            # Step 6: Run backtest
            backtest_results = backtest_portfolio(
                data=data,
                weights=optimal_weights,
                initial_investment=10000
            )
            
            assert isinstance(backtest_results, dict)
            assert 'Total Return (DCA)' in backtest_results
            assert 'Total Return (Lump-Sum)' in backtest_results
    
    def test_portfolio_optimization_and_simulation_integration(self, sample_market_data):
        """Test integration between portfolio optimization and simulation."""
        # Create returns data
        returns_data = {}
        for ticker, data in sample_market_data.items():
            returns_data[ticker] = data.set_index('Date')['Close'].pct_change().dropna()
        
        returns = pd.DataFrame(returns_data)
        
        # Optimize portfolio
        optimal_weights = optimize_weights(returns)
        assert optimal_weights is not None
        
        # Run simulation with optimized weights
        sim_results, sim_values = bootstrap_simulation(
            returns=returns,
            weights=optimal_weights,
            num_simulations=50,
            time_horizon_years=1,
            initial_investment=10000
        )
        
        # Optimized portfolio should have reasonable performance
        mean_return = sim_results['Historical Annual Return']
        sharpe_ratio = sim_results['Historical Sharpe Ratio']
        
        assert isinstance(mean_return, float)
        assert_sharpe_ratio_valid(sharpe_ratio)
        
        # Simulation results should be consistent with historical stats
        assert sim_results['Mean Final Value (Inflation-Adjusted, DCA)'] > 0
        assert len(sim_values) == 50
    
    def test_backtesting_and_simulation_consistency(self, sample_market_data):
        """Test consistency between backtesting and simulation results."""
        # Use sample data
        price_data = pd.DataFrame({
            ticker: data.set_index('Date')['Close'] 
            for ticker, data in sample_market_data.items()
        })
        
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        
        # Run backtest
        backtest_results = backtest_portfolio(
            data=price_data,
            weights=weights,
            initial_investment=10000
        )
        
        # Run simulation
        returns = calculate_returns(price_data)
        sim_results, _ = bootstrap_simulation(
            returns=returns,
            weights=weights,
            num_simulations=100,
            time_horizon_years=1,
            initial_investment=10000
        )
        
        # Historical stats should be consistent
        backtest_sharpe = backtest_results['Sharpe Ratio']
        sim_sharpe = sim_results['Historical Sharpe Ratio']
        
        assert abs(backtest_sharpe - sim_sharpe) < 0.1  # Should be very close
        
        # Volatility should be consistent
        backtest_vol = backtest_results['Annualized Volatility']
        sim_vol = sim_results['Historical Annual Volatility']
        
        assert abs(backtest_vol - sim_vol) < 0.01  # Should be very close
    
    def test_visualization_integration(self, sample_market_data):
        """Test integration with visualization components."""
        # Create sample data
        price_data = pd.DataFrame({
            ticker: data.set_index('Date')['Close'] 
            for ticker, data in sample_market_data.items()
        })
        
        weights = np.array([0.25, 0.25, 0.25, 0.25])
        tickers = list(sample_market_data.keys())
        
        # Run simulation
        returns = calculate_returns(price_data)
        sim_results, sim_values = bootstrap_simulation(
            returns=returns,
            weights=weights,
            num_simulations=100,
            time_horizon_years=1,
            initial_investment=10000
        )
        
        # Test visualization functions
        # Plot results
        fig1 = plot_results(sim_values, 1, sim_results)
        assert fig1 is not None
        
        # Plot historical performance
        fig2 = plot_historical_performance(price_data, weights, tickers)
        assert fig2 is not None
        
        # Clean up matplotlib figures
        import matplotlib.pyplot as plt
        plt.close(fig1)
    
    def test_configuration_integration(self):
        """Test integration with configuration system."""
        # Test settings loading
        settings = get_settings()
        
        assert isinstance(settings, dict)
        assert 'default_tickers' in settings
        assert 'isin_to_ticker' in settings
        
        # Test that default tickers are usable
        default_tickers = settings['default_tickers']
        assert isinstance(default_tickers, list)
        assert len(default_tickers) > 0
        
        # Test ISIN mapping
        isin_mapping = settings['isin_to_ticker']
        assert isinstance(isin_mapping, dict)
        
        # Test that all default tickers have ISIN mappings
        mapped_tickers = set(isin_mapping.values())
        for ticker in default_tickers:
            assert ticker in mapped_tickers
    
    def test_error_handling_integration(self):
        """Test error handling across integrated components."""
        # Test with empty data
        empty_data = pd.DataFrame()
        
        # Should handle empty data gracefully
        returns = calculate_returns(empty_data)
        assert len(returns) == 0
        
        # Test with invalid weights
        valid_returns = pd.DataFrame({
            'A': [0.01, 0.02, -0.01],
            'B': [0.02, -0.01, 0.01]
        })
        
        invalid_weights = np.array([0.7, 0.7])  # Sums to > 1
        
        # Portfolio stats should handle invalid weights
        try:
            portfolio_stats(invalid_weights, valid_returns)
            # Should not raise an exception, just produce results
        except Exception as e:
            pytest.fail(f"Should handle invalid weights gracefully: {e}")
    
    def test_data_pipeline_integration(self):
        """Test complete data pipeline from raw data to analysis."""
        # Create sample raw price data
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        raw_data = pd.DataFrame({
            'IWDA.AS': 100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, 100)),
            'QDV5.DE': 150 * np.cumprod(1 + np.random.normal(0.0007, 0.020, 100))
        }, index=dates)
        
        # Pipeline: Raw data -> Returns -> Stats -> Optimization -> Simulation
        
        # Step 1: Calculate returns
        returns = calculate_returns(raw_data)
        assert len(returns) == len(raw_data) - 1
        
        # Step 2: Calculate basic stats
        equal_weights = np.array([0.5, 0.5])
        ann_ret, ann_vol, sharpe, sortino, max_dd = portfolio_stats(equal_weights, returns)
        
        assert isinstance(ann_ret, float)
        assert isinstance(ann_vol, float)
        
        # Step 3: Optimize
        optimal_weights = optimize_weights(returns)
        assert optimal_weights is not None
        
        # Step 4: Simulate
        sim_results, sim_values = bootstrap_simulation(
            returns=returns,
            weights=optimal_weights,
            num_simulations=50,
            time_horizon_years=1,
            initial_investment=10000
        )
        
        assert len(sim_values) == 50
        assert sim_results['Mean Final Value (Inflation-Adjusted, DCA)'] > 0
    
    def test_multi_asset_integration(self):
        """Test integration with multiple assets."""
        # Create data for multiple assets
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        # Create correlated returns
        base_returns = np.random.normal(0.0005, 0.015, 252)
        
        multi_asset_data = pd.DataFrame({
            'ASSET1': 100 * np.cumprod(1 + base_returns + np.random.normal(0, 0.005, 252)),
            'ASSET2': 150 * np.cumprod(1 + base_returns * 0.8 + np.random.normal(0, 0.008, 252)),
            'ASSET3': 80 * np.cumprod(1 + base_returns * 0.6 + np.random.normal(0, 0.012, 252)),
            'ASSET4': 50 * np.cumprod(1 + np.random.normal(0.0002, 0.003, 252)),  # Low correlation
            'ASSET5': 200 * np.cumprod(1 + base_returns * 1.2 + np.random.normal(0, 0.020, 252))  # High correlation
        }, index=dates)
        
        # Test full pipeline with multiple assets
        returns = calculate_returns(multi_asset_data)
        
        # Optimize portfolio
        optimal_weights = optimize_weights(returns)
        assert len(optimal_weights) == 5
        assert_portfolio_weights_valid(optimal_weights)
        
        # Run simulation
        sim_results, sim_values = bootstrap_simulation(
            returns=returns,
            weights=optimal_weights,
            num_simulations=100,
            time_horizon_years=2,
            initial_investment=10000
        )
        
        # Should handle multi-asset portfolio
        assert len(sim_values) == 100
        assert sim_results['Historical Sharpe Ratio'] is not None
        
        # Run backtest
        backtest_results = backtest_portfolio(
            data=multi_asset_data,
            weights=optimal_weights,
            initial_investment=10000
        )
        
        assert isinstance(backtest_results['Total Return (Lump-Sum)'], float)
    
    def test_stress_scenario_integration(self):
        """Test integration under stress scenarios."""
        # Create data with stress scenarios
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        np.random.seed(42)
        
        # Normal market data
        normal_returns = np.random.normal(0.0005, 0.015, 200)
        
        # Add market crash (stress scenario)
        crash_returns = np.random.normal(-0.05, 0.03, 52)  # 52 days of crash
        
        all_returns = np.concatenate([normal_returns, crash_returns])
        
        stress_data = pd.DataFrame({
            'ASSET1': 100 * np.cumprod(1 + all_returns + np.random.normal(0, 0.005, 252)),
            'ASSET2': 150 * np.cumprod(1 + all_returns * 0.8 + np.random.normal(0, 0.008, 252))
        }, index=dates)
        
        # Test pipeline under stress
        returns = calculate_returns(stress_data)
        
        # Optimize (should handle stress periods)
        optimal_weights = optimize_weights(returns)
        assert optimal_weights is not None
        
        # Simulate (should handle stress scenarios)
        sim_results, sim_values = bootstrap_simulation(
            returns=returns,
            weights=optimal_weights,
            num_simulations=100,
            time_horizon_years=1,
            initial_investment=10000
        )
        
        # Should complete simulation despite stress
        assert len(sim_values) == 100
        
        # VaR should reflect stress scenarios
        var_95 = sim_results['95% VaR (Absolute Loss, DCA)']
        assert var_95 < 0  # Should show potential losses
        
        # Max drawdown should be significant
        max_dd = sim_results['Historical Max Drawdown']
        assert max_dd < -0.1  # Should show significant drawdown
    
    def test_performance_integration(self):
        """Test performance characteristics of integrated system."""
        # Create larger dataset to test performance
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')  # ~3 years
        np.random.seed(42)
        
        large_dataset = pd.DataFrame({
            f'ASSET{i}': 100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, 1000))
            for i in range(10)  # 10 assets
        }, index=dates)
        
        # Test that system handles larger datasets efficiently
        import time
        
        start_time = time.time()
        
        # Full pipeline
        returns = calculate_returns(large_dataset)
        optimal_weights = optimize_weights(returns)
        
        # Smaller simulation for performance test
        sim_results, sim_values = bootstrap_simulation(
            returns=returns,
            weights=optimal_weights,
            num_simulations=100,  # Reasonable number for performance test
            time_horizon_years=1,
            initial_investment=10000
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # Should complete within reasonable time (less than 30 seconds)
        assert elapsed_time < 30, f"Integration test took too long: {elapsed_time:.2f} seconds"
        
        # Should produce valid results
        assert len(sim_values) == 100
        assert sim_results['Mean Final Value (Inflation-Adjusted, DCA)'] > 0