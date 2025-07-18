"""Unit tests for simulation_engine module."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch

from portfolio_simulator.core.simulation_engine import bootstrap_simulation
from tests.fixtures.sample_data import create_sample_market_data
from tests.utils import assert_returns_reasonable, assert_financial_metric_valid


class TestBootstrapSimulation:
    """Test cases for bootstrap_simulation function."""
    
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
    
    @pytest.fixture
    def sample_weights(self):
        """Sample portfolio weights."""
        return np.array([0.4, 0.3, 0.2, 0.1])
    
    def test_bootstrap_simulation_basic(self, sample_returns, sample_weights):
        """Test basic bootstrap simulation."""
        results, final_values = bootstrap_simulation(
            returns=sample_returns,
            weights=sample_weights,
            num_simulations=100,
            time_horizon_years=1,
            initial_investment=10000
        )
        
        # Check results structure
        assert isinstance(results, dict)
        assert isinstance(final_values, np.ndarray)
        assert len(final_values) == 100
        
        # Check required keys
        required_keys = [
            'Mean Final Value (Inflation-Adjusted, DCA)',
            'Median Final Value (Inflation-Adjusted, DCA)',
            'Mean Final Value (Lump-Sum Comparison)',
            'Std Dev of Final Values (DCA)',
            '95% VaR (Absolute Loss, DCA)',
            '95% CVaR (Absolute Loss, DCA)',
            'Historical Annual Return',
            'Historical Annual Volatility',
            'Historical Sharpe Ratio',
            'Historical Sortino Ratio',
            'Historical Max Drawdown',
            'Effective Cost Drag (%)'
        ]
        
        for key in required_keys:
            assert key in results
            assert isinstance(results[key], (int, float))
            assert not np.isnan(results[key])
    
    def test_bootstrap_simulation_with_periodic_contributions(self, sample_returns, sample_weights):
        """Test simulation with periodic contributions."""
        results, final_values = bootstrap_simulation(
            returns=sample_returns,
            weights=sample_weights,
            num_simulations=50,
            time_horizon_years=1,
            initial_investment=10000,
            periodic_contrib=500,
            contrib_frequency='monthly'
        )
        
        # DCA should generally result in higher final values due to contributions
        mean_final = results['Mean Final Value (Inflation-Adjusted, DCA)']
        assert mean_final > 10000  # Should be higher than initial investment
        
        # All final values should be positive
        assert np.all(final_values > 0)
    
    def test_bootstrap_simulation_with_inflation(self, sample_returns, sample_weights):
        """Test simulation with inflation adjustment."""
        results_no_inflation, _ = bootstrap_simulation(
            returns=sample_returns,
            weights=sample_weights,
            num_simulations=50,
            time_horizon_years=1,
            initial_investment=10000,
            inflation_rate=0.0
        )
        
        results_with_inflation, _ = bootstrap_simulation(
            returns=sample_returns,
            weights=sample_weights,
            num_simulations=50,
            time_horizon_years=1,
            initial_investment=10000,
            inflation_rate=0.03  # 3% inflation
        )
        
        # Inflation-adjusted results should be lower
        assert results_with_inflation['Mean Final Value (Inflation-Adjusted, DCA)'] < \
               results_no_inflation['Mean Final Value (Inflation-Adjusted, DCA)']
    
    def test_bootstrap_simulation_with_fees_and_taxes(self, sample_returns, sample_weights):
        """Test simulation with transaction fees and taxes."""
        results_no_costs, _ = bootstrap_simulation(
            returns=sample_returns,
            weights=sample_weights,
            num_simulations=50,
            time_horizon_years=1,
            initial_investment=10000,
            periodic_contrib=500,
            transaction_fee=0.0,
            tax_rate=0.0
        )
        
        results_with_costs, _ = bootstrap_simulation(
            returns=sample_returns,
            weights=sample_weights,
            num_simulations=50,
            time_horizon_years=1,
            initial_investment=10000,
            periodic_contrib=500,
            transaction_fee=5.0,  # €5 per transaction
            tax_rate=0.15  # 15% tax on gains
        )
        
        # Results with costs should be lower
        assert results_with_costs['Mean Final Value (Inflation-Adjusted, DCA)'] < \
               results_no_costs['Mean Final Value (Inflation-Adjusted, DCA)']
    
    def test_bootstrap_simulation_with_rebalancing(self, sample_returns, sample_weights):
        """Test simulation with rebalancing."""
        results_no_rebalance, _ = bootstrap_simulation(
            returns=sample_returns,
            weights=sample_weights,
            num_simulations=50,
            time_horizon_years=1,
            initial_investment=10000,
            rebalance=False
        )
        
        results_with_rebalance, _ = bootstrap_simulation(
            returns=sample_returns,
            weights=sample_weights,
            num_simulations=50,
            time_horizon_years=1,
            initial_investment=10000,
            rebalance=True,
            rebalance_frequency='annual',
            rebalance_threshold=0.05
        )
        
        # Both should produce valid results
        assert results_no_rebalance['Mean Final Value (Inflation-Adjusted, DCA)'] > 0
        assert results_with_rebalance['Mean Final Value (Inflation-Adjusted, DCA)'] > 0
    
    def test_bootstrap_simulation_with_shock_factors(self, sample_returns, sample_weights):
        """Test simulation with market shock factors."""
        results_no_shock, _ = bootstrap_simulation(
            returns=sample_returns,
            weights=sample_weights,
            num_simulations=50,
            time_horizon_years=1,
            initial_investment=10000,
            shock_factors=None
        )
        
        # Apply negative shock (market crash)
        shock_factors = np.array([-0.3, -0.35, -0.2, -0.05])  # Different shocks per asset
        results_with_shock, _ = bootstrap_simulation(
            returns=sample_returns,
            weights=sample_weights,
            num_simulations=50,
            time_horizon_years=1,
            initial_investment=10000,
            shock_factors=shock_factors
        )
        
        # Shock should generally reduce final values
        assert results_with_shock['Mean Final Value (Inflation-Adjusted, DCA)'] < \
               results_no_shock['Mean Final Value (Inflation-Adjusted, DCA)']
    
    def test_bootstrap_simulation_progress_callback(self, sample_returns, sample_weights):
        """Test simulation with progress callback."""
        progress_values = []
        
        def progress_callback(progress):
            progress_values.append(progress)
        
        results, final_values = bootstrap_simulation(
            returns=sample_returns,
            weights=sample_weights,
            num_simulations=10,
            time_horizon_years=1,
            initial_investment=10000,
            progress_callback=progress_callback
        )
        
        # Progress callback should have been called
        assert len(progress_values) == 10
        assert progress_values[0] == 0.1
        assert progress_values[-1] == 1.0
        assert all(0 <= p <= 1 for p in progress_values)
    
    def test_bootstrap_simulation_empty_returns(self, sample_weights):
        """Test simulation with empty returns data."""
        empty_returns = pd.DataFrame()
        
        with pytest.raises(ValueError, match="No historical returns data available"):
            bootstrap_simulation(
                returns=empty_returns,
                weights=sample_weights,
                num_simulations=10,
                time_horizon_years=1,
                initial_investment=10000
            )
    
    def test_bootstrap_simulation_short_time_horizon(self, sample_returns, sample_weights):
        """Test simulation with very short time horizon."""
        results, final_values = bootstrap_simulation(
            returns=sample_returns,
            weights=sample_weights,
            num_simulations=50,
            time_horizon_years=0.1,  # ~1 month
            initial_investment=10000
        )
        
        # Should still produce valid results
        assert len(final_values) == 50
        assert all(v > 0 for v in final_values)
        assert results['Mean Final Value (Inflation-Adjusted, DCA)'] > 0
    
    def test_bootstrap_simulation_long_time_horizon(self, sample_returns, sample_weights):
        """Test simulation with long time horizon."""
        results, final_values = bootstrap_simulation(
            returns=sample_returns,
            weights=sample_weights,
            num_simulations=20,  # Fewer simulations for speed
            time_horizon_years=10,
            initial_investment=10000,
            periodic_contrib=500
        )
        
        # Should produce valid results
        assert len(final_values) == 20
        assert results['Mean Final Value (Inflation-Adjusted, DCA)'] > 10000
    
    def test_bootstrap_simulation_risk_metrics(self, sample_returns, sample_weights):
        """Test that risk metrics are calculated correctly."""
        results, final_values = bootstrap_simulation(
            returns=sample_returns,
            weights=sample_weights,
            num_simulations=1000,
            time_horizon_years=1,
            initial_investment=10000
        )
        
        # VaR should be negative (represents losses)
        var_95 = results['95% VaR (Absolute Loss, DCA)']
        cvar_95 = results['95% CVaR (Absolute Loss, DCA)']
        
        assert var_95 <= 0
        assert cvar_95 <= 0
        assert cvar_95 <= var_95  # CVaR should be more negative than VaR
    
    def test_bootstrap_simulation_different_contrib_frequencies(self, sample_returns, sample_weights):
        """Test simulation with different contribution frequencies."""
        for freq in ['monthly', 'quarterly']:
            results, final_values = bootstrap_simulation(
                returns=sample_returns,
                weights=sample_weights,
                num_simulations=50,
                time_horizon_years=1,
                initial_investment=10000,
                periodic_contrib=500,
                contrib_frequency=freq
            )
            
            assert len(final_values) == 50
            assert results['Mean Final Value (Inflation-Adjusted, DCA)'] > 10000
    
    def test_bootstrap_simulation_different_rebalance_frequencies(self, sample_returns, sample_weights):
        """Test simulation with different rebalancing frequencies."""
        for freq in ['annual', 'quarterly']:
            results, final_values = bootstrap_simulation(
                returns=sample_returns,
                weights=sample_weights,
                num_simulations=50,
                time_horizon_years=1,
                initial_investment=10000,
                rebalance=True,
                rebalance_frequency=freq,
                rebalance_threshold=0.05
            )
            
            assert len(final_values) == 50
            assert results['Mean Final Value (Inflation-Adjusted, DCA)'] > 0
    
    def test_bootstrap_simulation_lump_sum_comparison(self, sample_returns, sample_weights):
        """Test that lump sum comparison is included in results."""
        results, final_values = bootstrap_simulation(
            returns=sample_returns,
            weights=sample_weights,
            num_simulations=100,
            time_horizon_years=1,
            initial_investment=10000
        )
        
        dca_mean = results['Mean Final Value (Inflation-Adjusted, DCA)']
        lump_sum_mean = results['Mean Final Value (Lump-Sum Comparison)']
        
        # Both should be positive
        assert dca_mean > 0
        assert lump_sum_mean > 0
        
        # They should be reasonably close (within an order of magnitude)
        assert 0.1 < dca_mean / lump_sum_mean < 10
    
    @pytest.mark.parametrize("num_simulations", [10, 50, 100])
    def test_bootstrap_simulation_various_simulation_counts(self, sample_returns, sample_weights, num_simulations):
        """Test simulation with various numbers of simulations."""
        results, final_values = bootstrap_simulation(
            returns=sample_returns,
            weights=sample_weights,
            num_simulations=num_simulations,
            time_horizon_years=1,
            initial_investment=10000
        )
        
        assert len(final_values) == num_simulations
        assert results['Mean Final Value (Inflation-Adjusted, DCA)'] > 0
    
    def test_bootstrap_simulation_zero_initial_investment(self, sample_returns, sample_weights):
        """Test simulation with zero initial investment but contributions."""
        results, final_values = bootstrap_simulation(
            returns=sample_returns,
            weights=sample_weights,
            num_simulations=50,
            time_horizon_years=1,
            initial_investment=0,
            periodic_contrib=1000,
            base_invested=0
        )
        
        # Should still produce valid results from contributions
        assert len(final_values) == 50
        assert results['Mean Final Value (Inflation-Adjusted, DCA)'] > 0
    
    def test_bootstrap_simulation_high_volatility_returns(self, sample_weights):
        """Test simulation with high volatility returns."""
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', periods=252, freq='D')
        high_vol_returns = pd.DataFrame({
            'IWDA.AS': np.random.normal(0.001, 0.05, 252),  # High volatility
            'QDV5.DE': np.random.normal(0.001, 0.06, 252),
            'PPFB.DE': np.random.normal(0.0005, 0.04, 252),
            'XEON.DE': np.random.normal(0.0001, 0.01, 252)
        }, index=dates)
        
        results, final_values = bootstrap_simulation(
            returns=high_vol_returns,
            weights=sample_weights,
            num_simulations=100,
            time_horizon_years=1,
            initial_investment=10000
        )
        
        # Should handle high volatility gracefully
        assert len(final_values) == 100
        assert results['Std Dev of Final Values (DCA)'] > 0
        assert results['Historical Annual Volatility'] > 0.1  # High volatility should be reflected