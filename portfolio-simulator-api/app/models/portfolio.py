"""
Portfolio database model.
"""

from sqlalchemy import Column, Integer, String, Float, JSON, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base


class Portfolio(Base):
    """Portfolio model for storing user portfolio configurations."""
    
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    name = Column(String, nullable=False)
    description = Column(String)
    
    # Portfolio composition
    tickers = Column(JSON, nullable=False)  # ["IWDA.AS", "QDV5.DE", ...]
    weights = Column(JSON, nullable=False)  # [0.4, 0.3, 0.2, 0.1]
    initial_investment = Column(Float, nullable=False)
    
    # Metadata
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    owner = relationship("User", back_populates="portfolios")
    simulation_results = relationship("SimulationResult", back_populates="portfolio", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Portfolio(id={self.id}, name='{self.name}', user_id={self.user_id})>"
    
    @property
    def ticker_count(self) -> int:
        """Get number of assets in portfolio."""
        return len(self.tickers) if self.tickers else 0
    
    @property
    def is_weights_valid(self) -> bool:
        """Check if weights sum to approximately 1."""
        if not self.weights:
            return False
        return abs(sum(self.weights) - 1.0) < 0.01 