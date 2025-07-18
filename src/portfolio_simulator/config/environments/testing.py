"""
Testing environment configuration.
"""

# Testing-specific settings
DEBUG = True
NUM_SIMULATIONS = 100
CACHE_ENABLED = False
LOGGING_LEVEL = 'DEBUG'
PERFORMANCE_MONITORING = False


def get_config():
    """Get testing configuration."""
    return {
        'environment': 'testing',
        'debug': DEBUG,
        'logging_level': LOGGING_LEVEL,
        'cache_enabled': CACHE_ENABLED,
        'performance_monitoring': PERFORMANCE_MONITORING,
        'num_simulations': NUM_SIMULATIONS,
    }