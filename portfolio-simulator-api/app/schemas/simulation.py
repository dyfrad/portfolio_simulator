"""
Simulation schemas for request/response validation.
"""

from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class StressScenario(str, Enum):
    """Available stress test scenarios."""
    NONE = "None"
    RECESSION_2008 = "2008 Recession"
    COVID_CRASH = "COVID Crash"
    BEAR_MARKET_2022 = "2022 Bear Market"
    INFLATION_SPIKE = "Inflation Spike"


class SimulationRequest(BaseModel):
    """Schema for simulation request."""
    portfolio_id: int
    
    # Time parameters
    horizon_years: float
    num_simulations: int
    
    # Economic parameters
    inflation_rate: float = 0.025
    
    # DCA parameters
    periodic_contribution: float = 0.0
    contribution_frequency: str = "monthly"  # monthly, quarterly, annually
    
    # Cost parameters
    ter: float = 0.001  # Total Expense Ratio
    transaction_fee: float = 0.0
    tax_rate: float = 0.0
    
    # Rebalancing parameters
    rebalance: bool = False
    rebalance_frequency: str = "quarterly"  # monthly, quarterly, annually
    rebalance_threshold: float = 0.05
    
    # Stress testing
    stress_scenario: StressScenario = StressScenario.NONE
    
    # Data parameters
    start_date: Optional[str] = None  # YYYY-MM-DD, defaults to 10 years ago
    
    @validator('horizon_years')
    def validate_horizon(cls, v):
        if v <= 0:
            raise ValueError('Horizon must be positive')
        if v > 50:
            raise ValueError('Maximum horizon is 50 years')
        return v
    
    @validator('num_simulations')
    def validate_num_simulations(cls, v):
        if v < 100:
            raise ValueError('Minimum 100 simulations required')
        if v > 50000:
            raise ValueError('Maximum 50,000 simulations allowed')
        return v
    
    @validator('inflation_rate')
    def validate_inflation_rate(cls, v):
        if v < -0.1 or v > 0.2:
            raise ValueError('Inflation rate must be between -10% and 20%')
        return v
    
    @validator('periodic_contribution')
    def validate_periodic_contribution(cls, v):
        if v < 0:
            raise ValueError('Periodic contribution cannot be negative')
        if v > 100000:
            raise ValueError('Maximum periodic contribution is €100,000')
        return v
    
    @validator('contribution_frequency')
    def validate_contribution_frequency(cls, v):
        allowed = ["monthly", "quarterly", "annually"]
        if v not in allowed:
            raise ValueError(f'Contribution frequency must be one of: {allowed}')
        return v
    
    @validator('ter', 'tax_rate')
    def validate_rates(cls, v):
        if v < 0 or v > 1:
            raise ValueError('Rate must be between 0 and 1')
        return v
    
    @validator('rebalance_threshold')
    def validate_rebalance_threshold(cls, v):
        if v < 0.01 or v > 0.5:
            raise ValueError('Rebalance threshold must be between 1% and 50%')
        return v


class SimulationResults(BaseModel):
    """Schema for simulation results."""
    # Final value statistics
    mean_final_value: float
    median_final_value: float
    std_final_value: float
    min_final_value: float
    max_final_value: float
    
    # Risk metrics
    var_95: float  # Value at Risk (95% confidence)
    cvar_95: float  # Conditional Value at Risk (95% confidence)
    
    # Return statistics
    mean_total_return: float
    median_total_return: float
    probability_of_loss: float
    
    # Comparison metrics
    mean_final_value_lump_sum: Optional[float] = None
    dca_vs_lump_sum_difference: Optional[float] = None
    
    # Raw data (for charts)
    final_values: List[float]
    
    # Metadata
    simulations_completed: int
    execution_time_seconds: float


class SimulationResponse(BaseModel):
    """Schema for simulation response."""
    simulation_id: str
    portfolio_id: int
    config: SimulationRequest
    results: SimulationResults
    created_at: datetime
    
    class Config:
        from_attributes = True


class SimulationStatus(str, Enum):
    """Simulation execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SimulationStatusResponse(BaseModel):
    """Schema for simulation status response."""
    simulation_id: str
    status: SimulationStatus
    progress_percentage: Optional[float] = None
    message: Optional[str] = None
    estimated_completion_time: Optional[datetime] = None
    
    
class SimulationHistoryResponse(BaseModel):
    """Schema for simulation history response."""
    simulation_id: str
    portfolio_name: str
    horizon_years: float
    num_simulations: int
    mean_final_value: float
    created_at: datetime
    
    class Config:
        from_attributes = True 