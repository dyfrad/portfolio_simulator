"""
Session state management for portfolio simulator.
"""

import streamlit as st
from typing import Dict, Any, Optional
import numpy as np


class StateManager:
    """Manages Streamlit session state for portfolio simulation."""
    
    @staticmethod
    def initialize_state():
        """Initialize session state variables."""
        if 'ran_simulation' not in st.session_state:
            st.session_state.ran_simulation = False
    
    @staticmethod
    def store_simulation_results(
        results: Dict[str, float],
        sim_final_values: np.ndarray,
        all_tickers: list,
        weights: np.ndarray,
        horizon: float,
        backtest_results: Dict[str, float],
        fig_pie: Any,
        fig_dist: Any,
        fig_hist: Any,
        fig_dd: Any,
        fig_drift: Any
    ):
        """Store all simulation results in session state."""
        st.session_state.results = results
        st.session_state.sim_final_values = sim_final_values
        st.session_state.all_tickers = all_tickers
        st.session_state.weights = weights
        st.session_state.horizon = horizon
        st.session_state.backtest_results = backtest_results
        st.session_state.fig_pie = fig_pie
        st.session_state.fig_dist = fig_dist
        st.session_state.fig_hist = fig_hist
        st.session_state.fig_dd = fig_dd
        st.session_state.fig_drift = fig_drift
        st.session_state.ran_simulation = True
    
    @staticmethod
    def has_simulation_results() -> bool:
        """Check if simulation results are available."""
        return st.session_state.get('ran_simulation', False)
    
    @staticmethod
    def get_simulation_results() -> Optional[Dict[str, Any]]:
        """Get all simulation results from session state."""
        if not StateManager.has_simulation_results():
            return None
            
        return {
            'results': st.session_state.results,
            'sim_final_values': st.session_state.sim_final_values,
            'all_tickers': st.session_state.all_tickers,
            'weights': st.session_state.weights,
            'horizon': st.session_state.horizon,
            'backtest_results': st.session_state.backtest_results,
            'fig_pie': st.session_state.fig_pie,
            'fig_dist': st.session_state.fig_dist,
            'fig_hist': st.session_state.fig_hist,
            'fig_dd': st.session_state.fig_dd,
            'fig_drift': st.session_state.fig_drift
        }
    
    @staticmethod
    def clear_results():
        """Clear simulation results from session state."""
        st.session_state.ran_simulation = False
        for key in ['results', 'sim_final_values', 'all_tickers', 'weights', 
                   'horizon', 'backtest_results', 'fig_pie', 'fig_dist', 
                   'fig_hist', 'fig_dd', 'fig_drift']:
            if key in st.session_state:
                del st.session_state[key] 