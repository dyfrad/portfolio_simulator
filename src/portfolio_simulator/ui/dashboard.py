"""
Main dashboard controller for portfolio simulator.
"""

import streamlit as st
import numpy as np
from typing import Dict, Any, List

from .components import SidebarInputs, ResultsDisplay, StateManager
from .components.sidebar_inputs import SimulationConfig


class PortfolioDashboard:
    """Main dashboard controller for portfolio simulation UI."""
    
    def __init__(
        self, 
        default_tickers: List[str],
        isin_to_ticker: Dict[str, str],
        explanations: Dict[str, str]
    ):
        """Initialize dashboard with configuration."""
        self.default_tickers = default_tickers
        self.isin_to_ticker = isin_to_ticker
        self.explanations = explanations
        
        # Initialize UI components
        self.sidebar_inputs = SidebarInputs(default_tickers, isin_to_ticker)
        self.results_display = ResultsDisplay(explanations)
        self.state_manager = StateManager()
        
        # Initialize session state
        self.state_manager.initialize_state()
    
    def render(self):
        """Render the complete dashboard."""
        self._render_header()
        
        # Render sidebar inputs
        config = self.sidebar_inputs.render()
        
        # Render run simulation button
        if st.sidebar.button('Run Simulation'):
            self._execute_simulation(config)
        
        # Render results if available
        if self.state_manager.has_simulation_results():
            session_data = self.state_manager.get_simulation_results()
            self.results_display.render_all_results(session_data)
    
    def _render_header(self):
        """Render dashboard header and description."""
        st.title('Portfolio Simulator')
        
        # Author information in main section
        st.markdown("**Author: Mohit Saharan (mohit@msaharan.com)**")
        st.markdown("---")
        
        st.markdown("""
        This dashboard simulates the performance of a portfolio consisting of default assets:
        - MSCI World (IWDA.AS)
        - MSCI India (QDV5.DE)
        - iShares Gold (PPFB.DE)
        - Cash (XEON.DE)
        Add custom tickers below for more flexibility.
        """)
    
    def _execute_simulation(self, config: SimulationConfig):
        """Execute portfolio simulation with given configuration."""
        try:
            # Import business logic functions
            from portfolio_simulator import (
                fetch_data, calculate_returns, bootstrap_simulation, 
                backtest_portfolio, optimize_weights
            )
            from ..core.visualization import (
                plot_results, plot_historical_performance,
                plot_drawdowns, plot_weight_drift
            )
            import plotly.express as px
            
            # Fetch data
            data = fetch_data(config.all_tickers, config.start_date)
            returns = calculate_returns(data, config.ter)
            
            # Apply optimization if requested
            weights = config.weights
            if config.optimize_weights:
                optimal_weights = optimize_weights(returns)
                if optimal_weights is not None:
                    weights = optimal_weights
                    st.sidebar.info(f'Optimized Weights: {dict(zip(config.all_tickers, weights.round(4)))}')
                else:
                    raise ValueError("Weight optimization failed. Using default equal weights.")
            
            # Handle stress scenario
            shock_factors = self._get_stress_factors(config.stress_scenario, config.all_tickers)
            inflation_rate = config.inflation_rate
            
            if config.stress_scenario == 'Inflation Spike':
                inflation_rate = 0.08
                st.info('Inflation rate overridden to 8% for Inflation Spike scenario.')
            
            if config.stress_scenario != 'None':
                st.info(f'Simulating under {config.stress_scenario} stress conditions.')
            
            # Create original pie chart and display it immediately
            fig_pie_original = px.pie(values=config.weights, names=config.all_tickers, title='Original Portfolio Allocation')
            st.plotly_chart(fig_pie_original, key="pie_chart_original")
            st.session_state.fig_pie_original = fig_pie_original
            
            # Store optimization info for later display
            if config.optimize_weights:
                fig_pie_optimized = px.pie(values=weights, names=config.all_tickers, title='Optimized Portfolio Allocation')
                st.session_state.fig_pie_optimized = fig_pie_optimized
                st.session_state.optimized_weights = weights
                st.session_state.all_tickers = config.all_tickers
            
            # Show progress indicator
            progress_bar = st.progress(0)
            st.text("Running Monte Carlo simulation...")
            
            # Progress callback function
            def update_progress(progress):
                progress_bar.progress(progress)
                if progress >= 1.0:
                    st.text("Simulation complete! Generating charts...")
            
            # Run simulation
            results, sim_final_values = bootstrap_simulation(
                returns, weights, config.simulations, config.horizon, 
                config.initial_investment, inflation_rate, config.periodic_contrib, 
                config.contrib_frequency, config.transaction_fee, config.tax_rate, 
                config.rebalance, config.rebalance_frequency, config.rebalance_threshold, 
                shock_factors, config.base_invested, progress_callback=update_progress
            )
            
            # Create visualization charts
            fig_dist = plot_results(sim_final_values, config.horizon, results)
            fig_hist = plot_historical_performance(data, weights, config.all_tickers)
            fig_dd = plot_drawdowns(returns, weights)
            fig_drift = plot_weight_drift(
                returns, weights, config.rebalance, 
                config.rebalance_frequency, config.rebalance_threshold
            )
            
            # Backtesting
            backtest_end = config.backtest_end_date if config.backtest_end_date else None
            backtest_data = fetch_data(config.all_tickers, config.start_date, backtest_end)
            backtest_results = backtest_portfolio(
                backtest_data, weights, config.initial_investment, config.periodic_contrib,
                config.contrib_frequency, config.transaction_fee, config.tax_rate,
                config.rebalance, config.rebalance_frequency, config.rebalance_threshold
            )
            
            # Store results in session state
            self.state_manager.store_simulation_results(
                results=results,
                sim_final_values=sim_final_values,
                all_tickers=config.all_tickers,
                weights=weights,
                horizon=config.horizon,
                backtest_results=backtest_results,
                fig_pie_original=st.session_state.fig_pie_original,
                fig_pie_optimized=st.session_state.get('fig_pie_optimized'),
                fig_dist=fig_dist,
                fig_hist=fig_hist,
                fig_dd=fig_dd,
                fig_drift=fig_drift
            )
            
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"An error occurred during simulation: {str(e)}")
    
    def _get_stress_factors(self, stress_scenario: str, all_tickers: List[str]) -> np.ndarray:
        """Get stress factors for given scenario."""
        if stress_scenario == 'None':
            return None
            
        scenarios = {
            '2008 Recession': [-0.40, -0.55, 0.05, 0.02],
            'COVID Crash': [-0.34, -0.35, 0.15, 0.00],
            '2022 Bear Market': [-0.18, -0.08, 0.00, 0.00],
            'Inflation Spike': [0.05, 0.05, 0.30, 0.05],
        }
        
        shock_factors = scenarios[stress_scenario]
        
        # Adjust for number of tickers
        if len(all_tickers) > len(shock_factors):
            avg_stock_shock = np.mean(shock_factors[:2])
            shock_factors += [avg_stock_shock] * (len(all_tickers) - len(shock_factors))
        if len(shock_factors) > len(all_tickers):
            shock_factors = shock_factors[:len(all_tickers)]
            
        return np.array(shock_factors) 