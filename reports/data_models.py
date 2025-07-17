"""
Data models for report generation.
"""

from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np


@dataclass
class ReportData:
    """Structured data container for portfolio simulation reports."""
    tickers: List[str]
    weights: np.ndarray
    simulation_results: Dict[str, float]
    backtest_results: Dict[str, float]
    charts: Dict[str, Any]
    horizon_years: float
    
    def format_simulation_result(self, key: str) -> str:
        """Format simulation result value for display."""
        value = self.simulation_results.get(key, 0)
        if isinstance(value, float):
            if 'Return' in key or 'Volatility' in key or 'Drawdown' in key:
                return f"{value:.2%}"
            elif 'Value' in key or 'VaR' in key or 'CVaR' in key:
                return f"€{value:,.2f}"
            elif 'Ratio' in key:
                return f"{value:.2f}"
            elif 'Drag' in key:
                return f"{value:.2f}%"
        return str(value)
    
    def format_backtest_result(self, key: str) -> str:
        """Format backtest result value for display."""
        value = self.backtest_results.get(key, 0)
        if isinstance(value, float):
            if 'Return' in key or 'Volatility' in key or 'Drawdown' in key:
                return f"{value:.2%}"
            elif 'Ratio' in key:
                return f"{value:.2f}"
        return str(value) 