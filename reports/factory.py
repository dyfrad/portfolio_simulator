"""
Factory for creating report data objects.
"""

from typing import List, Dict, Any
import numpy as np

from .data_models import ReportData


class ReportFactory:
    """Factory for creating structured report data from simulation components."""
    
    @staticmethod
    def create_simulation_report(
        tickers: List[str], 
        weights: np.ndarray,
        simulation_results: Dict[str, float],
        backtest_results: Dict[str, float],
        charts: Dict[str, Any],
        horizon_years: float
    ) -> ReportData:
        """Create a simulation report data object from individual components."""
        chart_mapping = {
            'pie': charts.get('fig_pie'),
            'hist': charts.get('fig_hist'), 
            'dd': charts.get('fig_dd'),
            'drift': charts.get('fig_drift'),
            'dist': charts.get('fig_dist')
        }
        
        # Filter out None values
        chart_mapping = {k: v for k, v in chart_mapping.items() if v is not None}
        
        return ReportData(
            tickers=tickers,
            weights=weights,
            simulation_results=simulation_results,
            backtest_results=backtest_results,
            charts=chart_mapping,
            horizon_years=horizon_years
        )
    
    @staticmethod
    def create_charts_dict(
        fig_pie=None, 
        fig_hist=None, 
        fig_dd=None, 
        fig_drift=None, 
        fig_dist=None
    ) -> Dict[str, Any]:
        """Helper method to create charts dictionary from individual chart objects."""
        return {
            'fig_pie': fig_pie,
            'fig_hist': fig_hist,
            'fig_dd': fig_dd,
            'fig_drift': fig_drift,
            'fig_dist': fig_dist
        } 