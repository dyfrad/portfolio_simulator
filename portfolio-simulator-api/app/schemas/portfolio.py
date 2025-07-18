"""
Portfolio schemas for request/response validation.
"""

from pydantic import BaseModel, validator
from typing import List, Optional
from datetime import datetime


class PortfolioBase(BaseModel):
    """Base portfolio schema."""
    name: str
    description: Optional[str] = None
    tickers: List[str]
    weights: List[float]
    initial_investment: float
    
    @validator('name')
    def validate_name(cls, v):
        if len(v.strip()) == 0:
            raise ValueError('Portfolio name cannot be empty')
        if len(v) > 100:
            raise ValueError('Portfolio name cannot exceed 100 characters')
        return v.strip()
    
    @validator('tickers')
    def validate_tickers(cls, v):
        if len(v) == 0:
            raise ValueError('At least one ticker is required')
        if len(v) > 20:
            raise ValueError('Maximum 20 tickers allowed')
        # Remove duplicates while preserving order
        seen = set()
        unique_tickers = []
        for ticker in v:
            if ticker.upper() not in seen:
                seen.add(ticker.upper())
                unique_tickers.append(ticker.upper())
        return unique_tickers
    
    @validator('weights')
    def validate_weights(cls, v, values):
        if 'tickers' in values:
            if len(v) != len(values['tickers']):
                raise ValueError('Number of weights must match number of tickers')
        
        if any(w < 0 for w in v):
            raise ValueError('All weights must be non-negative')
        
        if any(w > 1 for w in v):
            raise ValueError('All weights must be <= 1')
        
        weight_sum = sum(v)
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(f'Weights must sum to 1.0 (current sum: {weight_sum:.3f})')
        
        return v
    
    @validator('initial_investment')
    def validate_initial_investment(cls, v):
        if v <= 0:
            raise ValueError('Initial investment must be positive')
        if v < 100:
            raise ValueError('Minimum initial investment is €100')
        if v > 10_000_000:
            raise ValueError('Maximum initial investment is €10,000,000')
        return v


class PortfolioCreate(PortfolioBase):
    """Schema for portfolio creation."""
    pass


class PortfolioUpdate(BaseModel):
    """Schema for portfolio updates."""
    name: Optional[str] = None
    description: Optional[str] = None
    tickers: Optional[List[str]] = None
    weights: Optional[List[float]] = None
    initial_investment: Optional[float] = None
    
    @validator('name')
    def validate_name(cls, v):
        if v is not None:
            if len(v.strip()) == 0:
                raise ValueError('Portfolio name cannot be empty')
            if len(v) > 100:
                raise ValueError('Portfolio name cannot exceed 100 characters')
            return v.strip()
        return v
    
    @validator('weights')
    def validate_weights(cls, v, values):
        if v is not None:
            if 'tickers' in values and values['tickers'] is not None:
                if len(v) != len(values['tickers']):
                    raise ValueError('Number of weights must match number of tickers')
            
            if any(w < 0 for w in v):
                raise ValueError('All weights must be non-negative')
            
            if any(w > 1 for w in v):
                raise ValueError('All weights must be <= 1')
            
            weight_sum = sum(v)
            if abs(weight_sum - 1.0) > 0.01:
                raise ValueError(f'Weights must sum to 1.0 (current sum: {weight_sum:.3f})')
        
        return v


class PortfolioListResponse(BaseModel):
    """Schema for portfolio list response."""
    id: int
    name: str
    description: Optional[str]
    ticker_count: int
    initial_investment: float
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class PortfolioResponse(PortfolioBase):
    """Schema for detailed portfolio response."""
    id: int
    user_id: int
    created_at: datetime
    updated_at: Optional[datetime]
    
    class Config:
        from_attributes = True


class PortfolioOptimizationResponse(BaseModel):
    """Schema for portfolio optimization response."""
    portfolio_id: int
    original_weights: List[float]
    optimized_weights: List[float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    improvement_percentage: float 