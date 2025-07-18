"""
Portfolio Simulator

Author: Mohit Saharan
Email: mohit@msaharan.com
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from scipy.optimize import minimize
import plotly.express as px
import plotly.graph_objects as go

from src.portfolio_simulator.core.data_operations import (
    fetch_data,
    calculate_returns,
    DEFAULT_TICKERS,
    ISIN_TO_TICKER,
    DEFAULT_START_DATE
)
from src.portfolio_simulator.core.financial_calculations import (
    portfolio_stats,
    optimize_weights
)
from src.portfolio_simulator.core.simulation_engine import (
    bootstrap_simulation
)
from src.portfolio_simulator.core.backtesting import (
    backtest_portfolio
)
from src.portfolio_simulator.core.visualization import (
    plot_results,
    plot_historical_performance,
    plot_weight_drift,
    plot_drawdowns
)









