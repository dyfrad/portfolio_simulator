"""
Portfolio simulation service for Monte Carlo analysis.
"""

import uuid
import time
import asyncio
from typing import List, Optional, Dict, Any, Callable
from sqlalchemy.orm import Session
from app.models.portfolio import Portfolio
from app.models.simulation import SimulationResult
from app.schemas.simulation import SimulationRequest, SimulationResponse, SimulationResults
from app.utils.financial_calcs import FinancialCalculator
from app.core.config import settings


class SimulationService:
    """Service for portfolio simulation operations."""
    
    def __init__(self, db: Session):
        self.db = db
        self.calculator = FinancialCalculator()
    
    async def run_monte_carlo_simulation(
        self,
        portfolio: Portfolio,
        config: SimulationRequest,
        user_id: int,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> SimulationResponse:
        """Run Monte Carlo simulation for a portfolio."""
        simulation_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            # Fetch historical data
            from datetime import datetime, timedelta
            
            if config.start_date:
                start_date = config.start_date
            else:
                # Default to 10 years of historical data
                start_date = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')
            
            data = await self.calculator.fetch_data(portfolio.tickers, start_date)
            returns = self.calculator.calculate_returns(data, config.ter)
            
            # Apply stress scenario if specified
            stress_factors = self._get_stress_factors(config.stress_scenario, portfolio.tickers)
            
            # Run Monte Carlo simulation
            simulation_results = await self._run_simulation_engine(
                returns=returns,
                weights=portfolio.weights,
                config=config,
                stress_factors=stress_factors,
                progress_callback=progress_callback
            )
            
            execution_time = time.time() - start_time
            
            # Create response
            results = SimulationResults(
                mean_final_value=simulation_results['mean_final_value'],
                median_final_value=simulation_results['median_final_value'],
                std_final_value=simulation_results['std_final_value'],
                min_final_value=simulation_results['min_final_value'],
                max_final_value=simulation_results['max_final_value'],
                var_95=simulation_results['var_95'],
                cvar_95=simulation_results['cvar_95'],
                mean_total_return=simulation_results['mean_total_return'],
                median_total_return=simulation_results['median_total_return'],
                probability_of_loss=simulation_results['probability_of_loss'],
                mean_final_value_lump_sum=simulation_results.get('mean_final_value_lump_sum'),
                dca_vs_lump_sum_difference=simulation_results.get('dca_vs_lump_sum_difference'),
                final_values=simulation_results['final_values'],
                simulations_completed=simulation_results['simulations_completed'],
                execution_time_seconds=execution_time
            )
            
            response = SimulationResponse(
                simulation_id=simulation_id,
                portfolio_id=portfolio.id,
                config=config,
                results=results,
                created_at=datetime.now()
            )
            
            return response
            
        except Exception as e:
            raise Exception(f"Simulation failed: {str(e)}")
    
    async def _run_simulation_engine(
        self,
        returns: Dict[str, List[float]],
        weights: List[float],
        config: SimulationRequest,
        stress_factors: Optional[List[float]] = None,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """Core simulation engine."""
        import numpy as np
        
        tickers = list(returns.keys())
        num_simulations = config.num_simulations
        horizon_days = int(config.horizon_years * 252)  # Trading days
        
        # Convert returns to numpy arrays
        returns_matrix = np.array([returns[ticker] for ticker in tickers]).T
        weights_array = np.array(weights)
        
        # Simulation containers
        final_values_dca = []
        final_values_lump_sum = []
        
        # DCA parameters
        contrib_frequency_days = {
            'monthly': 21,
            'quarterly': 63,
            'annually': 252
        }.get(config.contribution_frequency, 21)
        
        # Rebalancing parameters
        rebalance_frequency_days = {
            'monthly': 21,
            'quarterly': 63,
            'annually': 252
        }.get(config.rebalance_frequency, 63)
        
        for sim in range(num_simulations):
            # Update progress
            if progress_callback and sim % 100 == 0:
                await asyncio.sleep(0)  # Allow other tasks to run
                progress_callback(sim / num_simulations)
            
            # Bootstrap sample
            sample_indices = np.random.choice(len(returns_matrix), horizon_days, replace=True)
            sampled_returns = returns_matrix[sample_indices]
            
            # Apply stress factors if specified
            if stress_factors is not None:
                stress_start = np.random.randint(0, max(1, horizon_days - 252))
                stress_end = min(stress_start + 252, horizon_days)
                daily_stress = np.array(stress_factors) / 252
                sampled_returns[stress_start:stress_end] += daily_stress
            
            # Lump-sum simulation
            portfolio_value_lump = portfolio.initial_investment
            for day_return in sampled_returns:
                portfolio_return = np.dot(day_return, weights_array)
                portfolio_value_lump *= (1 + portfolio_return)
            
            # Adjust for inflation
            inflation_adjusted_lump = portfolio_value_lump / ((1 + config.inflation_rate) ** config.horizon_years)
            final_values_lump_sum.append(inflation_adjusted_lump)
            
            # DCA simulation
            portfolio_values = np.array(weights_array) * portfolio.initial_investment
            total_invested = portfolio.initial_investment
            
            for day in range(horizon_days):
                # Apply daily returns
                day_return = sampled_returns[day]
                portfolio_values *= (1 + day_return)
                
                # Add periodic contribution
                if day % contrib_frequency_days == 0 and day > 0:
                    contribution = config.periodic_contribution - config.transaction_fee
                    portfolio_values += np.array(weights_array) * contribution
                    total_invested += config.periodic_contribution
                
                # Rebalancing
                if config.rebalance and day % rebalance_frequency_days == 0:
                    total_value = np.sum(portfolio_values)
                    current_weights = portfolio_values / total_value if total_value > 0 else weights_array
                    
                    # Check if rebalancing is needed
                    weight_drift = np.abs(current_weights - weights_array)
                    if np.any(weight_drift > config.rebalance_threshold):
                        portfolio_values = total_value * weights_array
            
            # Calculate final DCA value
            final_value_dca = np.sum(portfolio_values)
            
            # Apply taxes on gains
            gains = final_value_dca - total_invested
            if gains > 0:
                final_value_dca = total_invested + gains * (1 - config.tax_rate)
            
            # Adjust for inflation
            inflation_adjusted_dca = final_value_dca / ((1 + config.inflation_rate) ** config.horizon_years)
            final_values_dca.append(inflation_adjusted_dca)
        
        # Calculate statistics
        final_values_dca = np.array(final_values_dca)
        final_values_lump_sum = np.array(final_values_lump_sum)
        
        # Calculate returns
        total_returns_dca = (final_values_dca / total_invested) - 1 if total_invested > 0 else 0
        total_returns_lump = (final_values_lump_sum / portfolio.initial_investment) - 1
        
        # Risk metrics
        var_95 = np.percentile(final_values_dca, 5)
        cvar_95 = np.mean(final_values_dca[final_values_dca <= var_95])
        
        return {
            'mean_final_value': float(np.mean(final_values_dca)),
            'median_final_value': float(np.median(final_values_dca)),
            'std_final_value': float(np.std(final_values_dca)),
            'min_final_value': float(np.min(final_values_dca)),
            'max_final_value': float(np.max(final_values_dca)),
            'var_95': float(var_95),
            'cvar_95': float(cvar_95),
            'mean_total_return': float(np.mean(total_returns_dca)),
            'median_total_return': float(np.median(total_returns_dca)),
            'probability_of_loss': float(np.mean(final_values_dca < total_invested)),
            'mean_final_value_lump_sum': float(np.mean(final_values_lump_sum)),
            'dca_vs_lump_sum_difference': float(np.mean(final_values_dca) - np.mean(final_values_lump_sum)),
            'final_values': final_values_dca.tolist(),
            'simulations_completed': num_simulations
        }
    
    def _get_stress_factors(self, stress_scenario: str, tickers: List[str]) -> Optional[List[float]]:
        """Get stress factors for given scenario."""
        if stress_scenario == 'None':
            return None
        
        scenarios = {
            '2008 Recession': [-0.40, -0.55, 0.05, 0.02],
            'COVID Crash': [-0.34, -0.35, 0.15, 0.00],
            '2022 Bear Market': [-0.18, -0.08, 0.00, 0.00],
            'Inflation Spike': [0.05, 0.05, 0.30, 0.05],
        }
        
        shock_factors = scenarios.get(stress_scenario, [0.0] * len(tickers))
        
        # Adjust for number of tickers
        if len(tickers) > len(shock_factors):
            avg_stock_shock = np.mean(shock_factors[:2]) if len(shock_factors) >= 2 else 0.0
            shock_factors.extend([avg_stock_shock] * (len(tickers) - len(shock_factors)))
        elif len(shock_factors) > len(tickers):
            shock_factors = shock_factors[:len(tickers)]
        
        return shock_factors
    
    async def save_simulation_result(
        self,
        result: SimulationResponse,
        user_id: int,
        portfolio_id: int
    ) -> None:
        """Save simulation result to database."""
        simulation_result = SimulationResult(
            simulation_id=result.simulation_id,
            user_id=user_id,
            portfolio_id=portfolio_id,
            config=result.config.dict(),
            results=result.results.dict(),
            mean_final_value=result.results.mean_final_value,
            median_final_value=result.results.median_final_value,
            std_final_value=result.results.std_final_value,
            var_95=result.results.var_95,
            cvar_95=result.results.cvar_95,
            execution_time_seconds=result.results.execution_time_seconds
        )
        
        self.db.add(simulation_result)
        self.db.commit()
    
    def get_user_simulation_history(self, user_id: int) -> List[Dict[str, Any]]:
        """Get simulation history for a user."""
        results = self.db.query(SimulationResult).filter(
            SimulationResult.user_id == user_id
        ).order_by(SimulationResult.created_at.desc()).limit(50).all()
        
        return [
            {
                'simulation_id': result.simulation_id,
                'portfolio_id': result.portfolio_id,
                'mean_final_value': result.mean_final_value,
                'created_at': result.created_at,
                'config': result.config
            }
            for result in results
        ]
    
    def get_simulation_result(self, simulation_id: str, user_id: int) -> Optional[Dict[str, Any]]:
        """Get specific simulation result."""
        result = self.db.query(SimulationResult).filter(
            SimulationResult.simulation_id == simulation_id,
            SimulationResult.user_id == user_id
        ).first()
        
        if not result:
            return None
        
        return {
            'simulation_id': result.simulation_id,
            'portfolio_id': result.portfolio_id,
            'config': result.config,
            'results': result.results,
            'created_at': result.created_at
        } 