"""Unit tests for config modules."""

import pytest
import os
from unittest.mock import patch, Mock

from portfolio_simulator.config.constants import (
    DEFAULT_TICKERS, 
    ISIN_TO_TICKER, 
    DEFAULT_START_DATE
)
from portfolio_simulator.config.settings import get_settings
from portfolio_simulator.config.environments import development, production, testing


class TestConstants:
    """Test cases for constants module."""
    
    def test_default_tickers_structure(self):
        """Test that default tickers are properly structured."""
        assert isinstance(DEFAULT_TICKERS, list)
        assert len(DEFAULT_TICKERS) > 0
        
        # Check that all tickers are strings
        for ticker in DEFAULT_TICKERS:
            assert isinstance(ticker, str)
            assert len(ticker) > 0
    
    def test_default_tickers_content(self):
        """Test specific content of default tickers."""
        # These are the expected default tickers based on the constants file
        expected_tickers = ['IWDA.AS', 'QDV5.DE', 'PPFB.DE', 'XEON.DE']
        assert DEFAULT_TICKERS == expected_tickers
    
    def test_isin_to_ticker_mapping(self):
        """Test ISIN to ticker mapping structure."""
        assert isinstance(ISIN_TO_TICKER, dict)
        assert len(ISIN_TO_TICKER) > 0
        
        # Check that all keys are ISINs (12-character strings)
        for isin in ISIN_TO_TICKER.keys():
            assert isinstance(isin, str)
            assert len(isin) == 12, f"ISIN {isin} should be 12 characters"
        
        # Check that all values are ticker strings
        for ticker in ISIN_TO_TICKER.values():
            assert isinstance(ticker, str)
            assert len(ticker) > 0
    
    def test_isin_to_ticker_specific_mappings(self):
        """Test specific ISIN to ticker mappings."""
        # Test some known mappings
        expected_mappings = {
            'IE00B4L5Y983': 'IWDA.AS',
            'IE00BZCQB185': 'QDV5.DE',
            'IE00B4ND3602': 'PPFB.DE',
            'LU0290358497': 'XEON.DE'
        }
        
        for isin, expected_ticker in expected_mappings.items():
            assert isin in ISIN_TO_TICKER
            assert ISIN_TO_TICKER[isin] == expected_ticker
    
    def test_default_start_date(self):
        """Test default start date format."""
        assert isinstance(DEFAULT_START_DATE, str)
        assert DEFAULT_START_DATE == '2015-01-01'
        
        # Test that it's a valid date format
        from datetime import datetime
        try:
            datetime.strptime(DEFAULT_START_DATE, '%Y-%m-%d')
        except ValueError:
            pytest.fail(f"DEFAULT_START_DATE {DEFAULT_START_DATE} is not a valid date format")
    
    def test_all_default_tickers_in_isin_mapping(self):
        """Test that all default tickers have corresponding ISIN mappings."""
        ticker_values = set(ISIN_TO_TICKER.values())
        
        for ticker in DEFAULT_TICKERS:
            assert ticker in ticker_values, f"Default ticker {ticker} not found in ISIN mapping"


class TestSettings:
    """Test cases for settings module."""
    
    def test_get_settings_default(self):
        """Test getting default settings."""
        settings = get_settings()
        
        assert isinstance(settings, dict)
        assert 'environment' in settings
        assert 'debug' in settings
        assert 'default_tickers' in settings
        assert 'isin_to_ticker' in settings
        assert 'default_start_date' in settings
    
    def test_get_settings_development(self):
        """Test getting development settings."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'development'}):
            settings = get_settings()
            
            assert settings['environment'] == 'development'
            assert settings['debug'] is True
    
    def test_get_settings_production(self):
        """Test getting production settings."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'production'}):
            settings = get_settings()
            
            assert settings['environment'] == 'production'
            assert settings['debug'] is False
    
    def test_get_settings_testing(self):
        """Test getting testing settings."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'testing'}):
            settings = get_settings()
            
            assert settings['environment'] == 'testing'
            assert settings['debug'] is True
    
    def test_get_settings_includes_constants(self):
        """Test that settings include all necessary constants."""
        settings = get_settings()
        
        # Should include all constants
        assert settings['default_tickers'] == DEFAULT_TICKERS
        assert settings['isin_to_ticker'] == ISIN_TO_TICKER
        assert settings['default_start_date'] == DEFAULT_START_DATE
    
    def test_get_settings_with_custom_environment(self):
        """Test getting settings with custom environment variable."""
        with patch.dict(os.environ, {'ENVIRONMENT': 'custom'}):
            settings = get_settings()
            
            # Should fall back to development settings
            assert settings['environment'] == 'development'
            assert settings['debug'] is True


class TestEnvironmentConfigs:
    """Test cases for environment configuration modules."""
    
    def test_development_config(self):
        """Test development environment configuration."""
        config = development.get_config()
        
        assert isinstance(config, dict)
        assert config['environment'] == 'development'
        assert config['debug'] is True
        assert 'logging_level' in config
        assert 'cache_enabled' in config
        assert 'performance_monitoring' in config
    
    def test_production_config(self):
        """Test production environment configuration."""
        config = production.get_config()
        
        assert isinstance(config, dict)
        assert config['environment'] == 'production'
        assert config['debug'] is False
        assert 'logging_level' in config
        assert 'cache_enabled' in config
        assert 'performance_monitoring' in config
    
    def test_testing_config(self):
        """Test testing environment configuration."""
        config = testing.get_config()
        
        assert isinstance(config, dict)
        assert config['environment'] == 'testing'
        assert config['debug'] is True
        assert 'logging_level' in config
        assert 'cache_enabled' in config
        assert 'performance_monitoring' in config
    
    def test_environment_specific_differences(self):
        """Test that different environments have appropriate differences."""
        dev_config = development.get_config()
        prod_config = production.get_config()
        test_config = testing.get_config()
        
        # Debug should be different
        assert dev_config['debug'] is True
        assert prod_config['debug'] is False
        assert test_config['debug'] is True
        
        # Logging levels should be appropriate
        assert dev_config['logging_level'] == 'DEBUG'
        assert prod_config['logging_level'] == 'INFO'
        assert test_config['logging_level'] == 'DEBUG'
        
        # Cache settings should be appropriate
        assert dev_config['cache_enabled'] is True
        assert prod_config['cache_enabled'] is True
        assert test_config['cache_enabled'] is False  # Disable cache in tests
    
    def test_all_configs_have_required_keys(self):
        """Test that all environment configs have required keys."""
        configs = [
            development.get_config(),
            production.get_config(),
            testing.get_config()
        ]
        
        required_keys = [
            'environment',
            'debug',
            'logging_level',
            'cache_enabled',
            'performance_monitoring'
        ]
        
        for config in configs:
            for key in required_keys:
                assert key in config, f"Config missing required key: {key}"
    
    def test_config_values_are_appropriate_types(self):
        """Test that config values are appropriate types."""
        configs = [
            development.get_config(),
            production.get_config(),
            testing.get_config()
        ]
        
        for config in configs:
            assert isinstance(config['environment'], str)
            assert isinstance(config['debug'], bool)
            assert isinstance(config['logging_level'], str)
            assert isinstance(config['cache_enabled'], bool)
            assert isinstance(config['performance_monitoring'], bool)
    
    def test_performance_monitoring_settings(self):
        """Test performance monitoring settings across environments."""
        dev_config = development.get_config()
        prod_config = production.get_config()
        test_config = testing.get_config()
        
        # Development should have performance monitoring enabled
        assert dev_config['performance_monitoring'] is True
        
        # Production should have performance monitoring enabled
        assert prod_config['performance_monitoring'] is True
        
        # Testing should have performance monitoring disabled for speed
        assert test_config['performance_monitoring'] is False
    
    def test_config_immutability(self):
        """Test that config objects are not accidentally modified."""
        config1 = development.get_config()
        config2 = development.get_config()
        
        # Should be independent copies
        config1['test_key'] = 'test_value'
        assert 'test_key' not in config2
    
    def test_environment_config_validation(self):
        """Test that environment configs are valid."""
        configs = [
            ('development', development.get_config()),
            ('production', production.get_config()),
            ('testing', testing.get_config())
        ]
        
        for env_name, config in configs:
            # Environment name should match
            assert config['environment'] == env_name
            
            # Logging level should be valid
            valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
            assert config['logging_level'] in valid_levels
            
            # Boolean values should be actual booleans
            assert isinstance(config['debug'], bool)
            assert isinstance(config['cache_enabled'], bool)
            assert isinstance(config['performance_monitoring'], bool)