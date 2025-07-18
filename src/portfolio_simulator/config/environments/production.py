"""
Production environment configuration.
"""

# Production-specific settings
DEBUG = False
NUM_SIMULATIONS = 10000
CACHE_ENABLED = True
LOGGING_LEVEL = 'INFO'
PERFORMANCE_MONITORING = True


def get_config():
    """Get production configuration."""
    return {
        'environment': 'production',
        'debug': DEBUG,
        'logging_level': LOGGING_LEVEL,
        'cache_enabled': CACHE_ENABLED,
        'performance_monitoring': PERFORMANCE_MONITORING,
        'num_simulations': NUM_SIMULATIONS,
    }