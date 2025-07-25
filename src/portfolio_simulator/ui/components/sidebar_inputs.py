"""
Sidebar input components for portfolio simulator.
"""

import streamlit as st
import pandas as pd
import yfinance as yf
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from ...config.settings import settings


@dataclass
class SimulationConfig:
    """Configuration parameters for portfolio simulation."""
    # Portfolio configuration
    all_tickers: List[str]
    weights: np.ndarray
    optimize_weights: bool
    initial_investment: float
    base_invested: Optional[float]
    has_uploaded_file: bool
    
    # Simulation parameters
    horizon: float
    simulations: int
    periodic_contrib: float
    contrib_frequency: str
    inflation_rate: float
    
    # Costs and taxes
    ter: float
    transaction_fee: float
    tax_rate: float
    
    # Rebalancing
    rebalance: bool
    rebalance_frequency: str
    rebalance_threshold: float
    
    # Stress testing
    stress_scenario: str
    
    # Data parameters
    start_date: str
    backtest_end_date: str


class SidebarInputs:
    """Handles all sidebar input widgets and portfolio configuration."""
    
    def __init__(self, default_tickers: List[str], isin_to_ticker: dict):
        self.default_tickers = default_tickers
        self.isin_to_ticker = isin_to_ticker
        
    def render(self) -> SimulationConfig:
        """Render all sidebar inputs and return configuration."""
        # Simulation Parameters header
        st.sidebar.header('Simulation Parameters')
        st.sidebar.markdown("---")
        
        # Portfolio configuration
        portfolio_config = self._render_portfolio_config()
        
        # Simulation parameters (pass initial_investment and file upload status from portfolio config)
        sim_params = self._render_simulation_parameters(
            portfolio_config.get('initial_investment', 100000.0),
            portfolio_config.get('has_uploaded_file', False)
        )
        
        # Costs and fees
        cost_params = self._render_cost_parameters()
        
        # Rebalancing
        rebalance_params = self._render_rebalancing_parameters()
        
        # Stress testing
        stress_params = self._render_stress_parameters()
        
        # Data parameters
        data_params = self._render_data_parameters()
        
        # Combine all parameters
        return SimulationConfig(
            **portfolio_config,
            **sim_params,
            **cost_params,
            **rebalance_params,
            **stress_params,
            **data_params
        )
    
    def _render_portfolio_config(self) -> dict:
        """Render portfolio configuration inputs."""
        optimize = st.sidebar.checkbox('Optimize Weights for Max Sharpe')
        
        custom_tickers = st.sidebar.text_input(
            'Custom Tickers (comma-separated, e.g., VUSA.AS)', ''
        )
        
        all_tickers = self.default_tickers.copy()
        if custom_tickers:
            all_tickers.extend([t.strip() for t in custom_tickers.split(',')])
        
        uploaded_file = st.sidebar.file_uploader(
            "Upload Portfolio CSV (Ticker, Shares, Cost Basis) or Degiro Transactions CSV", 
            type="csv"
        )
        
        initial_investment = 100000.0
        base_invested = None
        weights = None
        
        if uploaded_file is not None:
            portfolio_data = self._process_uploaded_file(uploaded_file, all_tickers)
            if portfolio_data:
                all_tickers, weights, initial_investment, base_invested = portfolio_data
        
        # Weights input (dynamic based on tickers) - only if not uploaded
        if uploaded_file is None:
            weights = self._render_weight_inputs(all_tickers)
        
        return {
            'all_tickers': all_tickers,
            'weights': weights,
            'optimize_weights': optimize,
            'initial_investment': initial_investment,
            'base_invested': base_invested,
            'has_uploaded_file': uploaded_file is not None
        }
    
    def _render_weight_inputs(self, all_tickers: List[str]) -> np.ndarray:
        """Render weight input controls for tickers."""
        st.sidebar.subheader('Portfolio Weights')
        
        weights = []
        cols = st.sidebar.columns(2)
        
        for i, ticker in enumerate(all_tickers):
            col = cols[i % 2]
            weight = col.number_input(
                ticker, 
                min_value=settings.portfolio_weight_min, 
                max_value=settings.portfolio_weight_max, 
                value=1.0/len(all_tickers), 
                step=settings.portfolio_weight_step
            )
            weights.append(weight)
        
        weights = np.array(weights)
        total_weight = np.sum(weights)
        if total_weight != 1.0:
            weights = weights / total_weight
            st.sidebar.warning(f'Weights normalized to sum to 1: {weights.round(2)}')
        
        return weights
    
    def _render_simulation_parameters(self, calculated_initial_investment: float = 100000.0, has_uploaded_file: bool = False) -> dict:
        """Render simulation parameter inputs."""
        
        horizon = st.sidebar.number_input(
            f'Time Horizon ({settings.time_horizon_min} - {settings.time_horizon_max} Years, in steps of {settings.time_horizon_step} year):', 
            min_value=settings.time_horizon_min, max_value=settings.time_horizon_max, value=settings.default_time_horizon_years, step=settings.time_horizon_step
        )
        
        simulations = st.sidebar.number_input(
            f'Number of Simulations ({settings.num_simulations_min} - {settings.num_simulations_max}, in steps of {settings.num_simulations_step})', 
            min_value=settings.num_simulations_min, max_value=settings.num_simulations_max, value=settings.default_num_simulations, step=settings.num_simulations_step
        )
        
        initial_investment = st.sidebar.number_input(
            'Initial Investment (€)', 
            min_value=settings.initial_investment_min, 
            value=calculated_initial_investment, 
            step=settings.initial_investment_step,
            disabled=has_uploaded_file
        )
        
        periodic_contrib = st.sidebar.number_input(
            'Periodic Contribution (€)', 
            min_value=settings.periodic_contrib_min, value=settings.default_periodic_contrib, step=settings.periodic_contrib_step
        )
        
        contrib_frequency = st.sidebar.selectbox(
            'Contribution Frequency', 
            ['monthly', 'quarterly'],
            index=0 if settings.default_contrib_frequency == 'monthly' else 1
        )
        
        inflation_rate = st.sidebar.number_input(
            f'Expected Annual Inflation Rate ({settings.inflation_rate_min} - {settings.inflation_rate_max} %, in steps of {settings.inflation_rate_step}%)', 
            min_value=settings.inflation_rate_min, max_value=settings.inflation_rate_max, value=settings.default_inflation_rate, step=settings.inflation_rate_step
        ) / 100
        
        return {
            'horizon': horizon,
            'simulations': simulations,
            'periodic_contrib': periodic_contrib,
            'contrib_frequency': contrib_frequency,
            'inflation_rate': inflation_rate
        }
    
    def _render_cost_parameters(self) -> dict:
        """Render cost and fee parameter inputs."""
        
        ter = st.sidebar.number_input(
            f'Annual TER ({settings.ter_min} - {settings.ter_max} %, in steps of {settings.ter_step}%)', 
            min_value=settings.ter_min, max_value=settings.ter_max, value=settings.default_transaction_fee, step=settings.ter_step
        )
        
        transaction_fee = st.sidebar.number_input(
            'Transaction Fee per Buy (€)', 
            min_value=settings.transaction_fee_min, value=settings.default_transaction_fee_fixed, step=settings.transaction_fee_step
        )
        
        tax_rate = st.sidebar.number_input(
            f'Capital Gains Tax Rate ({settings.tax_rate_min} - {settings.tax_rate_max} %, in steps of {settings.tax_rate_step}%)', 
            min_value=settings.tax_rate_min, max_value=settings.tax_rate_max, value=settings.default_tax_rate, step=settings.tax_rate_step
        ) / 100
        
        return {
            'ter': ter,
            'transaction_fee': transaction_fee,
            'tax_rate': tax_rate
        }
    
    def _render_rebalancing_parameters(self) -> dict:
        """Render rebalancing parameter inputs."""
        
        rebalance = st.sidebar.checkbox('Enable Rebalancing', value=settings.default_rebalance)
        
        rebalance_frequency = st.sidebar.selectbox(
            'Rebalancing Frequency', 
            ['quarterly', 'annual'],
            index=0 if settings.default_rebalance_frequency == 'quarterly' else 1
        )
        
        rebalance_threshold = st.sidebar.number_input(
            f'Rebalancing Threshold ({settings.rebalance_threshold_min} - {settings.rebalance_threshold_max} %, in steps of {settings.rebalance_threshold_step}%)', 
            min_value=settings.rebalance_threshold_min, max_value=settings.rebalance_threshold_max, value=settings.default_rebalance_threshold * 100, step=settings.rebalance_threshold_step
        ) / 100
        
        return {
            'rebalance': rebalance,
            'rebalance_frequency': rebalance_frequency,
            'rebalance_threshold': rebalance_threshold
        }
    
    def _render_stress_parameters(self) -> dict:
        """Render stress testing parameter inputs."""
        stress_scenario = st.sidebar.selectbox(
            'Stress Scenario', 
            ['None', '2008 Recession', 'COVID Crash', '2022 Bear Market', 'Inflation Spike']
        )
        
        return {'stress_scenario': stress_scenario}
    
    def _render_data_parameters(self) -> dict:
        """Render data parameter inputs."""
        start_date = st.sidebar.text_input('Start Date (YYYY-MM-DD)', '2015-01-01')
        backtest_end_date = st.sidebar.text_input('Backtest End Date (YYYY-MM-DD, optional)', '')
        
        return {
            'start_date': start_date,
            'backtest_end_date': backtest_end_date
        }
    
    def _process_uploaded_file(self, uploaded_file, all_tickers: List[str]) -> Optional[Tuple]:
        """Process uploaded portfolio or transaction CSV file."""
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check if it's a holdings CSV
            if set(['Ticker', 'Shares', 'Cost Basis']).issubset(df.columns):
                return self._process_holdings_csv(df)
            
            # Check if it's a transactions CSV
            elif set(['Date', 'Product', 'ISIN', 'Quantity', 'Price', 'Local value', 'Transaction and/or third']).issubset(df.columns):
                return self._process_transactions_csv(df)
            
            else:
                st.error("CSV must contain either columns: Ticker, Shares, Cost Basis or transaction columns")
                return None
                
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
            return None
    
    def _process_holdings_csv(self, df: pd.DataFrame) -> Optional[Tuple]:
        """Process holdings CSV and return portfolio data."""
        try:
            df_group = df.groupby('Ticker').agg({'Shares': 'sum', 'Cost Basis': 'sum'}).reset_index()
            current_prices = yf.download(list(df_group['Ticker']), period='1d')['Close'].iloc[-1]
            df_group['Current Price'] = df_group['Ticker'].map(current_prices)
            df_group['Current Value'] = df_group['Shares'] * df_group['Current Price']
            total_value = df_group['Current Value'].sum()
            df_group['Weight'] = df_group['Current Value'] / total_value
            df_group['Unrealized Gain'] = df_group['Current Value'] - df_group['Cost Basis']
            
            st.header('Current Portfolio')
            st.dataframe(df_group)
            
            all_tickers = list(df_group['Ticker'])
            weights = df_group['Weight'].values
            initial_investment = total_value
            base_invested = df_group['Cost Basis'].sum()
            
            return all_tickers, weights, initial_investment, base_invested
            
        except Exception as e:
            st.error(f"Error processing holdings CSV: {e}")
            return None
    
    def _process_transactions_csv(self, df: pd.DataFrame) -> Optional[Tuple]:
        """Process transactions CSV and return portfolio data."""
        try:
            # Process transactions logic (simplified for space)
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
            df = df.dropna(subset=['Date'])
            df['Local value'] = pd.to_numeric(df['Local value'], errors='coerce')
            df['Transaction and/or third'] = pd.to_numeric(df['Transaction and/or third'], errors='coerce').fillna(0)
            df['Quantity'] = pd.to_numeric(df['Quantity'], errors='coerce')
            df = df.sort_values(['Date', 'Time'])
            
            # Calculate holdings from transactions
            holdings = {}
            for _, row in df.iterrows():
                if pd.isna(row['ISIN']): 
                    continue
                isin = row['ISIN']
                if isin not in holdings:
                    holdings[isin] = {'shares': 0.0, 'total_cost': 0.0, 'product': row['Product']}
                
                quantity = row['Quantity']
                local_value = row['Local value']
                fees = row['Transaction and/or third']
                
                if quantity > 0:  # buy
                    cost = -local_value - fees
                    holdings[isin]['total_cost'] += cost
                    holdings[isin]['shares'] += quantity
                elif quantity < 0:  # sell
                    if holdings[isin]['shares'] > 0:
                        avg_cost = holdings[isin]['total_cost'] / holdings[isin]['shares']
                        cost_reduction = avg_cost * (-quantity)
                        holdings[isin]['total_cost'] -= cost_reduction
                    holdings[isin]['shares'] += quantity
                
                if abs(holdings[isin]['shares']) < 1e-6:
                    holdings[isin]['shares'] = 0.0
                    holdings[isin]['total_cost'] = 0.0
            
            df_holdings = pd.DataFrame([
                {'ISIN': k, 'Ticker': self.isin_to_ticker.get(k), 'Shares': v['shares'], 
                 'Cost Basis': v['total_cost'], 'Product': v['product']}
                for k, v in holdings.items() if v['shares'] > 0
            ])
            
            valid_holdings = df_holdings[df_holdings['Ticker'].notna()]
            all_tickers = valid_holdings['Ticker'].tolist()
            
            if all_tickers:
                # Fetch current prices and calculate live portfolio data
                current_prices = yf.download(all_tickers, period='1d')['Close'].iloc[-1]
                df_holdings['Current Price'] = df_holdings['Ticker'].map(current_prices)
                df_holdings['Current Value'] = df_holdings['Shares'] * df_holdings['Current Price']
                total_value = df_holdings['Current Value'].sum()
                df_holdings['Weight'] = df_holdings['Current Value'] / total_value
                df_holdings['Unrealized Gain'] = df_holdings['Current Value'] - df_holdings['Cost Basis']
                df_holdings['Unrealized Gain %'] = (df_holdings['Unrealized Gain'] / df_holdings['Cost Basis'] * 100).round(2)
                
                # Display single comprehensive table
                st.header('Portfolio Holdings from Transactions')
                st.dataframe(df_holdings, use_container_width=True)
                
                weights = df_holdings['Weight'].values
                initial_investment = total_value
                base_invested = df_holdings['Cost Basis'].sum()
                
                return all_tickers, weights, initial_investment, base_invested
            else:
                st.error("No valid tickers found for holdings.")
                return None
                
        except Exception as e:
            st.error(f"Error processing transactions CSV: {e}")
            return None 