"""
Configuration settings for the Portfolio Simulator API.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import List, Optional
import os

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = "sqlite:///./portfolio.db"
    
    # Security
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # CORS
    ALLOWED_HOSTS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://yourportfolio.vercel.app",
        "https://yourportfolio.com"
    ]
    
    # External APIs
    YAHOO_FINANCE_TIMEOUT: int = 30
    YAHOO_FINANCE_BASE_URL: str = "https://query1.finance.yahoo.com"
    
    # Simulation defaults
    MAX_SIMULATIONS: int = 50000
    DEFAULT_SIMULATIONS: int = 10000
    MAX_SIMULATION_TIME_SECONDS: int = 300  # 5 minutes
    
    # Redis for caching
    REDIS_URL: str = "redis://localhost:6379"
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    
    # Celery (background tasks)
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    
    # File storage
    UPLOAD_MAX_SIZE: int = 10 * 1024 * 1024  # 10MB
    REPORTS_STORAGE_PATH: str = "/tmp/reports"
    
    # Email (for notifications)
    SMTP_HOST: Optional[str] = None
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    
    # Stripe (payments)
    STRIPE_PUBLISHABLE_KEY: Optional[str] = None
    STRIPE_SECRET_KEY: Optional[str] = None
    STRIPE_WEBHOOK_SECRET: Optional[str] = None
    
    # Rate limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    PREMIUM_RATE_LIMIT_PER_MINUTE: int = 300
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Global settings instance
settings = Settings() 