"""
Simulation result database model.
"""

from sqlalchemy import Column, Integer, String, JSON, ForeignKey, DateTime, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class SimulationResult(Base):
    """Model for storing Monte Carlo simulation results."""
    
    __tablename__ = "simulation_results"
    
    id = Column(Integer, primary_key=True, index=True)
    simulation_id = Column(String, unique=True, index=True, nullable=False)  # UUID
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    portfolio_id = Column(Integer, ForeignKey("portfolios.id"), nullable=False)
    
    # Simulation configuration
    config = Column(JSON, nullable=False)  # Simulation parameters
    
    # Simulation results
    results = Column(JSON, nullable=False)  # Complete simulation results
    
    # Quick access metrics (for faster queries)
    mean_final_value = Column(Float)
    median_final_value = Column(Float)
    std_final_value = Column(Float)
    var_95 = Column(Float)  # Value at Risk (95%)
    cvar_95 = Column(Float)  # Conditional Value at Risk (95%)
    
    # Execution metadata
    execution_time_seconds = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="simulation_results")
    portfolio = relationship("Portfolio", back_populates="simulation_results")
    
    def __repr__(self):
        return f"<SimulationResult(id={self.simulation_id}, user_id={self.user_id}, portfolio_id={self.portfolio_id})>" 