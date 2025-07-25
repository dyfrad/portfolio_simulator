# Portfolio Simulator Restructuring Plan

## Overview
This document outlines the plan to restructure the portfolio simulator project to comply with industry standards for Python projects.

## Current Issues

### 1. Package Structure
- Root-level Python files (`portfolio_simulator.py`, `portfolio_simulator_ui.py`)
- Mixed concerns (business logic and UI entry points at root)
- Non-standard package organization

### 2. Packaging & Dependencies
- Using outdated `requirements.txt` instead of modern `pyproject.toml`
- Missing project metadata and build configuration
- No development dependencies specification

### 3. Testing Infrastructure
- No test suite exists
- No testing configuration
- Missing test fixtures and sample data

### 4. Configuration Management
- Constants scattered across files (`DEFAULT_TICKERS`, `ISIN_TO_TICKER`)
- No environment-specific configuration
- Hardcoded values throughout codebase

### 5. Code Quality
- Missing type hints throughout codebase
- No linting or formatting configuration
- No pre-commit hooks for code quality

### 6. Documentation Structure
- Documentation files scattered at root level
- No API documentation
- Missing developer guide

## Recommended Target Structure

```
portfolio_simulator/
├── src/
│   └── portfolio_simulator/
│       ├── __init__.py
│       ├── core/                    # Business logic (renamed from modules/)
│       │   ├── __init__.py
│       │   ├── data_operations.py
│       │   ├── financial_calculations.py
│       │   ├── simulation_engine.py
│       │   ├── backtesting.py
│       │   └── visualization.py
│       ├── config/                  # Configuration management
│       │   ├── __init__.py
│       │   ├── settings.py
│       │   ├── constants.py
│       │   └── environments/
│       │       ├── __init__.py
│       │       ├── development.py
│       │       ├── production.py
│       │       └── testing.py
│       └── ui/                      # UI components (existing)
│           ├── __init__.py
│           ├── dashboard.py
│           └── components/
│               ├── __init__.py
│               ├── results_display.py
│               ├── sidebar_inputs.py
│               └── state_manager.py
├── tests/                           # Test suite
│   ├── __init__.py
│   ├── conftest.py                  # Pytest configuration
│   ├── unit/
│   │   ├── __init__.py
│   │   ├── test_data_operations.py
│   │   ├── test_financial_calculations.py
│   │   ├── test_simulation_engine.py
│   │   ├── test_backtesting.py
│   │   └── test_visualization.py
│   ├── integration/
│   │   ├── __init__.py
│   │   └── test_portfolio_simulator.py
│   └── fixtures/
│       ├── __init__.py
│       └── sample_data.py
├── docs/                            # Documentation
│   ├── api/                         # API documentation
│   ├── user_guide/                  # User documentation
│   ├── developer_guide/             # Development documentation
│   └── examples/                    # Usage examples
├── scripts/                         # Entry point scripts
│   ├── run_simulator.py
│   ├── run_ui.py
│   └── generate_report.py
├── .github/                         # CI/CD pipeline
│   └── workflows/
│       ├── ci.yml                   # Testing, linting
│       ├── cd.yml                   # Deployment
│       └── release.yml              # Release automation
├── pyproject.toml                   # Modern Python packaging
├── setup.py                         # Fallback packaging
├── .pre-commit-config.yaml          # Pre-commit hooks
├── .flake8                          # Linting configuration
├── .gitignore                       # Git ignore rules
├── .env.example                     # Environment variables template
├── README.md                        # Project overview
└── CHANGELOG.md                     # Version history
```

## Implementation Plan

### Phase 1: Core Restructuring (High Priority)
1. **Create src/portfolio_simulator package structure**
   - Move `modules/` to `src/portfolio_simulator/core/`
   - Move `ui/` to `src/portfolio_simulator/ui/`
   - Create proper `__init__.py` files

   **Incremental Implementation Plan:**
   - Phase 1a: Create src structure alongside existing
     1. Create src/portfolio_simulator/ structure
     2. Copy modules to new location
     3. Keep existing modules/ intact
     4. Result: App still runs from current entry points
   
   - Phase 1b: Gradual import migration
     1. Update one import at a time
     2. Test after each change
     3. Result: App runs throughout transition
   
   - Phase 1c: Configuration extraction
     1. Create config system
     2. Gradually move constants
     3. Result: App runs with new config system
   
   - Phase 1d: Entry point updates
     1. Create new entry points
     2. Keep old ones working
     3. Result: Both old and new entry points work

2. **Create configuration system**
   - Extract constants to `src/portfolio_simulator/config/constants.py`
   - Create settings management in `src/portfolio_simulator/config/settings.py`
   - Add environment-specific configurations

   **Incremental Implementation Plan:**
   - Phase 2a: Complete configuration extraction
     1. Create `src/portfolio_simulator/config/settings.py` for dynamic settings
     2. Create environment-specific config files
     3. Update modules to use centralized configuration
     4. Result: All configuration is centralized and manageable
   
   - Phase 2b: Add environment support
     1. Create `src/portfolio_simulator/config/environments/` directory
     2. Add development.py, production.py, testing.py configs
     3. Add environment detection and loading logic
     4. Result: App supports multiple environments seamlessly

3. **Create modern packaging**
   - Replace `requirements.txt` with `pyproject.toml`
   - Add project metadata and build configuration
   - Define development dependencies

   **Incremental Implementation Plan:**
   - Phase 3a: Create pyproject.toml alongside requirements.txt
     1. Create `pyproject.toml` with existing dependencies from `requirements.txt`
     2. Add project metadata (name, version, description, etc.)
     3. Define build system configuration
     4. Result: Modern packaging config exists alongside old one
   
   - Phase 3b: Add development and optional dependencies
     1. Add development dependencies section (testing, linting, etc.)
     2. Add optional dependencies for different features
     3. Define console scripts for entry points
     4. Result: Complete packaging configuration ready for use
   
   - Phase 3c: Test and validate packaging
     1. Test package installation in clean environment
     2. Verify all dependencies are correctly specified
     3. Test console scripts work properly
     4. Result: Packaging system is production-ready

### Phase 2: Testing Infrastructure (High Priority)
1. **Set up testing framework**
   - Create `tests/` directory structure
   - Add `conftest.py` with pytest configuration
   - Create test fixtures and sample data

   **Incremental Implementation Plan:**
   - Phase 2a: Create basic test structure
     1. Create `tests/` directory with proper subdirectories (unit/, integration/, fixtures/)
     2. Add `tests/__init__.py` and `tests/conftest.py` with basic pytest configuration
     3. Create `tests/fixtures/__init__.py` and `tests/fixtures/sample_data.py` for test data
     4. Result: Basic test framework structure is in place
   
   - Phase 2b: Set up pytest configuration and dependencies
     1. Add pytest and testing dependencies to `pyproject.toml` dev dependencies
     2. Configure pytest settings in `pyproject.toml` or `pytest.ini`
     3. Add test markers for different test types (unit, integration, slow)
     4. Result: Pytest is properly configured and ready for use
   
   - Phase 2c: Create test fixtures and sample data
     1. Create sample portfolio data in `tests/fixtures/sample_data.py`
     2. Create mock market data for testing financial calculations
     3. Add fixtures for different simulation scenarios
     4. Create helper functions for test data generation
     5. Result: Comprehensive test data available for all test scenarios
   
   - Phase 2d: Set up test utilities and helpers
     1. Create `tests/utils.py` with common test utilities
     2. Add assertion helpers for financial calculations
     3. Create mock objects for external dependencies (Yahoo Finance)
     4. Add test decorators for parametrized tests
     5. Result: Test utilities support efficient test development

2. **Write unit tests**
   - Test each module in `core/`
   - Test configuration management
   - Test UI components

3. **Write integration tests**
   - End-to-end simulation tests
   - UI workflow tests

### Phase 3: Code Quality (Medium Priority)
1. **Add type hints**
   - Annotate all function parameters and return types
   - Use `typing` module for complex types
   - Add mypy configuration

2. **Set up code quality tools**
   - Configure Black for formatting
   - Configure flake8 for linting
   - Add pre-commit hooks
   - Configure mypy for type checking

### Phase 4: Documentation (Medium Priority)
1. **Reorganize documentation**
   - Move existing docs to `docs/` directory
   - Create API documentation
   - Write developer guide
   - Add usage examples

2. **Create entry points**
   - Move root-level Python files to `scripts/`
   - Create proper CLI entry points
   - Add setup for console scripts

### Phase 5: CI/CD (Low Priority)
1. **GitHub Actions**
   - Automated testing pipeline
   - Code quality checks
   - Automated releases

2. **Environment management**
   - Docker configuration updates
   - Environment variable management

## Benefits of Restructuring

1. **Industry Standard Compliance**: Follows Python packaging best practices
2. **Better Organization**: Clear separation of concerns
3. **Improved Testing**: Comprehensive test suite for reliability
4. **Code Quality**: Automated linting and formatting
5. **Better Documentation**: Clear API and usage documentation
6. **Easier Maintenance**: Modular structure for easier updates
7. **Professional Appearance**: Attracts contributors and users
8. **CI/CD Ready**: Automated testing and deployment
9. **Type Safety**: Better IDE support and error catching
10. **Configuration Management**: Environment-specific settings

## Migration Strategy

1. **Incremental Migration**: Implement changes in phases to avoid breaking existing functionality
2. **Backward Compatibility**: Maintain existing entry points during transition
3. **Testing**: Ensure all functionality works after each phase
4. **Documentation**: Update documentation as structure changes

## Status
- [ ] Phase 1: Core Restructuring
- [ ] Phase 2: Testing Infrastructure  
- [ ] Phase 3: Code Quality
- [ ] Phase 4: Documentation
- [ ] Phase 5: CI/CD

## Notes
- All existing functionality must be preserved during restructuring
- Entry points should maintain backward compatibility
- Consider creating migration scripts for smooth transition
- Update CLAUDE.md with new structure and commands