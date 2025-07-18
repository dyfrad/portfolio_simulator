"""
Financial calculation utilities migrated from original portfolio_simulator.py.
"""

import numpy as np
import pandas as pd
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from scipy.optimize import minimize
from .data_fetcher import DataFetcher


class FinancialCalculator:
    """Financial calculation utilities for portfolio analysis."""
    
    def __init__(self):
        self.data_fetcher = DataFetcher()
    
    async def fetch_data(
        self, 
        tickers: List[str], 
        start_date: str, 
        end_date: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical adjusted close prices for given tickers.
        Migrated from original fetch_data function.
        """
        try:
            result = await self.data_fetcher.fetch_yahoo_finance_data(tickers, start_date, end_date)
            return result
        finally:
            # Ensure session is closed to prevent warnings
            await self.data_fetcher.close()
    
    def calculate_returns(self, data: Dict[str, pd.DataFrame], ter: float = 0.0) -> Dict[str, List[float]]:
        """
        Calculate daily returns from price data.
        Migrated from original calculate_returns function.
        """
        returns = {}
        daily_ter = ter / 252  # Convert annual TER to daily
        
        for ticker, prices in data.items():
            if len(prices) < 2:
                continue
                
            # Calculate percentage returns
            price_series = prices['close'] if isinstance(prices, pd.DataFrame) else prices
            daily_returns = price_series.pct_change().dropna()
            
            # Subtract daily TER if specified
            if ter > 0:
                daily_returns = daily_returns - daily_ter
            
            returns[ticker] = daily_returns.tolist()
        
        return returns
    
    def optimize_weights(self, returns: Dict[str, List[float]]) -> Optional[np.ndarray]:
        """
        Optimize portfolio weights for maximum Sharpe ratio.
        Migrated from original optimize_weights function.
        """
        if not returns or len(returns) < 2:
            return None
        
        # Convert to DataFrame for easier handling
        returns_df = pd.DataFrame(returns)
        
        # Check for sufficient data
        if len(returns_df) < 252:  # Less than 1 year of data
            return None
        
        # Calculate annualized metrics
        mean_returns = returns_df.mean() * 252
        cov_matrix = returns_df.cov() * 252
        
        num_assets = len(returns_df.columns)
        
        def negative_sharpe(weights: np.ndarray) -> float:
            """Objective function: negative Sharpe ratio."""
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            
            if portfolio_volatility == 0:
                return -999
            
            # Using risk-free rate of 0 for simplicity
            sharpe_ratio = portfolio_return / portfolio_volatility
            return -sharpe_ratio
        
        # Constraints: weights sum to 1
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        
        # Bounds: weights between 0 and 1 (no short selling)
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess: equal weights
        initial_guess = np.array([1.0 / num_assets] * num_assets)
        
        try:
            # Optimize
            result = minimize(
                negative_sharpe,
                initial_guess,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000}
            )
            
            if result.success and np.allclose(np.sum(result.x), 1.0, atol=1e-3):
                return result.x
            else:
                return None
                
        except Exception as e:
            print(f"Optimization failed: {str(e)}")
            return None
    
    def calculate_portfolio_metrics(
        self, 
        returns: Dict[str, List[float]], 
        weights: List[float]
    ) -> Dict[str, float]:
        """Calculate portfolio performance metrics."""
        returns_df = pd.DataFrame(returns)
        weights_array = np.array(weights)
        
        # Portfolio returns
        portfolio_returns = (returns_df * weights_array).sum(axis=1)
        
        # Annualized metrics
        mean_return = portfolio_returns.mean() * 252
        volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Sharpe ratio (assuming risk-free rate = 0)
        sharpe_ratio = mean_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = mean_return / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cumulative_returns = (1 + portfolio_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'expected_return': mean_return,
            'expected_volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': abs(max_drawdown)
        }
    
    def calculate_var_cvar(self, returns: List[float], confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate Value at Risk and Conditional Value at Risk.
        """
        returns_array = np.array(returns)
        
        # Sort returns in ascending order
        sorted_returns = np.sort(returns_array)
        
        # Calculate VaR
        var_index = int((1 - confidence_level) * len(sorted_returns))
        var = sorted_returns[var_index] if var_index < len(sorted_returns) else sorted_returns[-1]
        
        # Calculate CVaR (average of returns worse than VaR)
        cvar_returns = sorted_returns[:var_index+1]
        cvar = np.mean(cvar_returns) if len(cvar_returns) > 0 else var
        
        return var, cvar
    
    def backtest_portfolio(
        self,
        data: Dict[str, pd.DataFrame],
        weights: List[float],
        initial_investment: float,
        periodic_contrib: float = 0,
        contrib_frequency: str = "monthly",
        transaction_fee: float = 0,
        tax_rate: float = 0,
        rebalance: bool = False,
        rebalance_frequency: str = "quarterly",
        rebalance_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Backtest portfolio performance using historical data.
        Migrated from original backtest_portfolio function.
        """
        # Convert data to aligned DataFrame
        price_data = {}
        for ticker, df in data.items():
            if isinstance(df, pd.DataFrame):
                price_data[ticker] = df['close']
            else:
                price_data[ticker] = df
        
        prices_df = pd.DataFrame(price_data).dropna()
        
        if len(prices_df) < 2:
            raise ValueError("Insufficient price data for backtesting")
        
        # Initialize portfolio
        weights_array = np.array(weights)
        num_shares = (initial_investment * weights_array) / prices_df.iloc[0].values
        cash = 0
        total_invested = initial_investment
        
        # Tracking variables
        portfolio_values = []
        dates = []
        
        # Contribution frequency mapping
        contrib_days = {
            'monthly': 21,
            'quarterly': 63,
            'annually': 252
        }.get(contrib_frequency, 21)
        
        # Rebalancing frequency mapping
        rebalance_days = {
            'monthly': 21,
            'quarterly': 63,
            'annually': 252
        }.get(rebalance_frequency, 63)
        
        for i, (date, prices) in enumerate(prices_df.iterrows()):
            # Calculate current portfolio value
            current_value = np.sum(num_shares * prices.values) + cash
            portfolio_values.append(current_value)
            dates.append(date)
            
            # Add periodic contributions
            if i > 0 and i % contrib_days == 0 and periodic_contrib > 0:
                effective_contrib = periodic_contrib - transaction_fee
                contribution_per_asset = effective_contrib * weights_array
                shares_to_buy = contribution_per_asset / prices.values
                num_shares += shares_to_buy
                total_invested += periodic_contrib
            
            # Rebalancing
            if rebalance and i > 0 and i % rebalance_days == 0:
                current_portfolio_value = np.sum(num_shares * prices.values)
                current_weights = (num_shares * prices.values) / current_portfolio_value
                
                # Check if rebalancing is needed
                weight_drift = np.abs(current_weights - weights_array)
                if np.any(weight_drift > rebalance_threshold):
                    # Rebalance to target weights
                    target_values = current_portfolio_value * weights_array
                    num_shares = target_values / prices.values
        
        # Calculate final metrics
        final_value = portfolio_values[-1]
        total_return = (final_value / total_invested) - 1 if total_invested > 0 else 0
        
        # Calculate annualized return
        years = len(portfolio_values) / 252
        annualized_return = ((final_value / total_invested) ** (1 / years) - 1) if years > 0 and total_invested > 0 else 0
        
        # Calculate volatility
        portfolio_returns = pd.Series(portfolio_values).pct_change().dropna()
        annualized_volatility = portfolio_returns.std() * np.sqrt(252)
        
        # Calculate Sharpe ratio
        sharpe_ratio = annualized_return / annualized_volatility if annualized_volatility > 0 else 0
        
        # Calculate max drawdown
        portfolio_series = pd.Series(portfolio_values)
        running_max = portfolio_series.expanding().max()
        drawdown = (portfolio_series - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'total_return_dca': total_return,
            'total_return_lump_sum': (portfolio_values[-1] / initial_investment) - 1,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': abs(max_drawdown),
            'final_value': final_value,
            'total_invested': total_invested,
            'portfolio_values': portfolio_values,
            'dates': dates
        }
    
    def generate_stress_scenarios(self, tickers: List[str]) -> Dict[str, List[float]]:
        """Generate stress test scenarios for portfolio simulation."""
        scenarios = {
            '2008 Recession': {
                'description': 'Global financial crisis with severe market decline',
                'equity_shock': -0.40,
                'emerging_shock': -0.55,
                'gold_shock': 0.05,
                'cash_shock': 0.02
            },
            'COVID Crash': {
                'description': '2020 pandemic-induced market crash',
                'equity_shock': -0.34,
                'emerging_shock': -0.35,
                'gold_shock': 0.15,
                'cash_shock': 0.00
            },
            '2022 Bear Market': {
                'description': 'Inflation and rate hike induced bear market',
                'equity_shock': -0.18,
                'emerging_shock': -0.08,
                'gold_shock': 0.00,
                'cash_shock': 0.00
            },
            'Inflation Spike': {
                'description': 'High inflation environment',
                'equity_shock': 0.05,
                'emerging_shock': 0.05,
                'gold_shock': 0.30,
                'cash_shock': 0.05
            }
        }
        
        # Map default shocks based on asset type assumptions
        stress_factors = {}
        
        for scenario_name, scenario in scenarios.items():
            factors = []
            for ticker in tickers:
                ticker_upper = ticker.upper()
                
                # Classify asset type based on ticker patterns
                if any(pattern in ticker_upper for pattern in ['IWDA', 'MSCI', 'WORLD', 'VUSA', 'SPY']):
                    # Global/US equity
                    factors.append(scenario['equity_shock'])
                elif any(pattern in ticker_upper for pattern in ['QDV5', 'EMERGING', 'EM', 'INDIA']):
                    # Emerging markets
                    factors.append(scenario['emerging_shock'])
                elif any(pattern in ticker_upper for pattern in ['PPFB', 'GOLD', 'GLD']):
                    # Gold/commodities
                    factors.append(scenario['gold_shock'])
                elif any(pattern in ticker_upper for pattern in ['XEON', 'CASH', 'MONEY']):
                    # Cash/money market
                    factors.append(scenario['cash_shock'])
                else:
                    # Default to equity shock for unknown assets
                    factors.append(scenario['equity_shock'])
            
            stress_factors[scenario_name] = factors
        
        return stress_factors 