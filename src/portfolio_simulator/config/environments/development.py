"""
Development environment configuration.
"""

# Development-specific settings
DEBUG = True
NUM_SIMULATIONS = 1000
CACHE_ENABLED = True
LOGGING_LEVEL = 'DEBUG'
PERFORMANCE_MONITORING = True


def get_config():
    """Get development configuration."""
    return {
        'environment': 'development',
        'debug': DEBUG,
        'logging_level': LOGGING_LEVEL,
        'cache_enabled': CACHE_ENABLED,
        'performance_monitoring': PERFORMANCE_MONITORING,
        'num_simulations': NUM_SIMULATIONS,
    }