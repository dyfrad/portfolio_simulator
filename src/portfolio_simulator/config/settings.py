"""
Settings management for portfolio simulator.
Just organizes existing configuration values, no new features.
"""

import os
from .constants import DEFAULT_TICKERS, DEFAULT_START_DATE


def load_environment_config(environment):
    """Load environment-specific configuration."""
    config = {}
    try:
        if environment == "development":
            from .environments.development import DEBUG, NUM_SIMULATIONS, CACHE_ENABLED
            config = {"debug": DEBUG, "num_simulations": NUM_SIMULATIONS, "cache_enabled": CACHE_ENABLED}
        elif environment == "production":
            from .environments.production import DEBUG, NUM_SIMULATIONS, CACHE_ENABLED
            config = {"debug": DEBUG, "num_simulations": NUM_SIMULATIONS, "cache_enabled": CACHE_ENABLED}
        elif environment == "testing":
            from .environments.testing import DEBUG, NUM_SIMULATIONS, CACHE_ENABLED
            config = {"debug": DEBUG, "num_simulations": NUM_SIMULATIONS, "cache_enabled": CACHE_ENABLED}
    except ImportError:
        # Fall back to defaults if environment config not found
        pass
    return config


class Settings:
    """Application settings - organizes existing configuration."""
    
    def __init__(self):
        # Environment
        self.environment = os.getenv("PORTFOLIO_ENV", "development")
        
        # Load environment-specific config
        env_config = load_environment_config(self.environment)
        self.debug = env_config.get("debug", os.getenv("DEBUG", "False").lower() == "true")
        self.cache_enabled = env_config.get("cache_enabled", True)
        
        # Data settings (existing values)
        self.default_tickers = DEFAULT_TICKERS
        self.default_start_date = DEFAULT_START_DATE
        
        # Simulation defaults (matching actual UI defaults)
        self.default_num_simulations = env_config.get("num_simulations", 1000)
        self.default_time_horizon_years = 5.0  # matches UI default
        self.default_initial_investment = 100000.0  # matches UI calculated value
        self.default_periodic_contrib = 0.0  # matches UI default
        self.default_contrib_frequency = 'monthly'  # existing: monthly, quarterly
        self.default_inflation_rate = 2.0  # matches UI default
        self.default_transaction_fee = 0.2  # matches UI default
        self.default_tax_rate = 25.0  # matches UI default
        
        # Rebalancing defaults (existing values)
        self.default_rebalance = True
        self.default_rebalance_frequency = 'quarterly'  # existing: quarterly, annual
        self.default_rebalance_threshold = 0.05
        
        # Risk calculation defaults (existing values)
        self.default_confidence_level = 0.95
        self.default_risk_free_rate = 0.02
        self.trading_days_per_year = 252
        
        # UI Configuration Parameters
        # Time Horizon settings
        self.time_horizon_min = 1.0
        self.time_horizon_max = 10.0
        self.time_horizon_step = 0.25
        
        # Simulation settings
        self.num_simulations_min = 100
        self.num_simulations_max = 10000
        self.num_simulations_step = 100
        
        # Initial Investment settings
        self.initial_investment_min = 0.0
        self.initial_investment_step = 1000.0
        
        # Periodic Contribution settings
        self.periodic_contrib_min = 0.0
        self.periodic_contrib_step = 100.0
        
        # Inflation Rate settings
        self.inflation_rate_min = 0.0
        self.inflation_rate_max = 20.0
        self.inflation_rate_step = 0.1
        
        # TER settings
        self.ter_min = 0.0
        self.ter_max = 2.0
        self.ter_step = 0.01
        
        # Transaction Fee settings
        self.transaction_fee_min = 0.0
        self.transaction_fee_step = 1.0
        self.default_transaction_fee_fixed = 5.0  # separate from TER
        
        # Tax Rate settings
        self.tax_rate_min = 0.0
        self.tax_rate_max = 50.0
        self.tax_rate_step = 0.1
        
        # Rebalancing Threshold settings
        self.rebalance_threshold_min = 0.0
        self.rebalance_threshold_max = 20.0
        self.rebalance_threshold_step = 0.5
        
        # Portfolio Weight settings
        self.portfolio_weight_min = 0.0
        self.portfolio_weight_max = 1.0
        self.portfolio_weight_step = 0.01


# Global settings instance
settings = Settings()


def get_settings():
    """Get the current application settings."""
    return settings