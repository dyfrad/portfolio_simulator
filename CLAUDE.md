# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Working Style
- Act as a senior developer with expertise in the codebase
- Do the job when asked without trying to teach or explain unless specifically requested
- Answer with low verbosity - be direct and concise
- Focus on delivering results rather than educational content

## Commands

### Legacy Streamlit Application
```bash
# Activate conda environment
conda activate pyfin

# Run the legacy Streamlit application
streamlit run portfolio_simulator_ui.py

# Install dependencies for legacy app
pip install -r requirements.txt
```

### FastAPI Backend (portfolio-simulator-api/)
```bash
# Use direct python path for conda environment
/Users/msaharan/products/anaconda3/envs/pyfin/bin/python

# Development server
/Users/msaharan/products/anaconda3/envs/pyfin/bin/python -m uvicorn app.main:app --reload

# Run tests
/Users/msaharan/products/anaconda3/envs/pyfin/bin/python -m pytest
/Users/msaharan/products/anaconda3/envs/pyfin/bin/python -m pytest --cov=app  # with coverage
/Users/msaharan/products/anaconda3/envs/pyfin/bin/python -m pytest app/tests/test_auth.py -v  # specific test file

# Database migrations
/Users/msaharan/products/anaconda3/envs/pyfin/bin/python -m alembic upgrade head          # apply migrations
/Users/msaharan/products/anaconda3/envs/pyfin/bin/python -m alembic revision --autogenerate -m "description"  # create new migration
/Users/msaharan/products/anaconda3/envs/pyfin/bin/python -m alembic downgrade -1          # rollback one migration

# Docker setup
docker-compose up -d          # start all services
docker-compose exec api alembic upgrade head  # run migrations in container

# Install dependencies
/Users/msaharan/products/anaconda3/envs/pyfin/bin/pip install -r requirements.txt
```

### Next.js Frontend (portfolio-simulator-web/)
```bash
# Development server
npm run dev

# Build for production
npm run build

# Production server
npm start

# Linting
npm run lint

# Install dependencies
npm install
```

## Architecture

### Repository Structure
This is a portfolio simulator application undergoing migration from Streamlit to a modern FastAPI + React stack:

1. **Legacy Streamlit App**: Root directory contains the original Python + Streamlit portfolio simulator that should be kept functional during migration
2. **FastAPI Backend**: `portfolio-simulator-api/` - New backend with authentication, database, and API endpoints
3. **React Frontend**: `portfolio-simulator-web/` - New Next.js frontend with modern UI

**Important**: The root directory Streamlit application should remain fully functional throughout the migration process. It serves as the reference implementation and fallback while the new API/frontend architecture is being developed.

### Core Components

#### Legacy Application (`portfolio_simulator.py`)
- Monte Carlo simulation engine
- Financial calculations (Sharpe ratio, VaR, CVaR, etc.)
- Portfolio optimization using scipy
- Yahoo Finance data fetching
- Report generation (PDF)

#### FastAPI Backend Architecture
```
app/
├── main.py              # FastAPI application entry point
├── core/                # Core configuration and utilities
│   ├── config.py        # Environment settings
│   ├── security.py      # JWT authentication
│   └── database.py      # Database connection
├── api/v1/              # API endpoints
│   ├── auth.py          # Authentication routes
│   ├── portfolios.py    # Portfolio CRUD operations
│   ├── simulations.py   # Monte Carlo simulation endpoints
│   └── reports.py       # PDF report generation
├── models/              # SQLAlchemy database models
├── schemas/             # Pydantic request/response schemas
├── services/            # Business logic layer
└── utils/               # Utility functions (financial calculations)
```

#### Frontend Architecture
```
components/
├── ui/                  # Reusable UI components
├── layout/              # Layout components
├── portfolio/           # Portfolio management components
├── simulation/          # Simulation runner and results
└── charts/              # Interactive charts (Plotly.js)
```

### Key Features
- **Monte Carlo Simulations**: Up to 50,000 iterations for portfolio analysis
- **Portfolio Optimization**: Sharpe ratio maximization using modern portfolio theory
- **Historical Backtesting**: Performance analysis with Dollar-Cost Averaging
- **Risk Metrics**: VaR, CVaR, Sortino ratio, maximum drawdown
- **Interactive Charts**: Plotly.js visualizations for distributions and historical data
- **CSV Import**: Support for Degiro transaction history parsing
- **PDF Reports**: Automated report generation with charts and metrics
- **Authentication**: JWT-based user management with premium tiers

### Database Models
- **User**: Authentication, premium status, created portfolios
- **Portfolio**: Tickers, weights, initial investment, optimization settings
- **SimulationResult**: Cached simulation results and historical data

### Financial Calculations
The `utils/financial_calcs.py` module contains migrated logic from the original Streamlit app:
- Historical data fetching from Yahoo Finance
- Daily returns calculation with TER (Total Expense Ratio) adjustment
- Portfolio weight optimization using scipy.optimize
- Monte Carlo simulation with bootstrap sampling
- Risk metrics calculations (VaR, CVaR, Sharpe ratio, etc.)

### Testing Strategy
- **Backend**: pytest with async support, database fixtures
- **Frontend**: Jest and React Testing Library
- **Integration**: TestClient for API endpoint testing
- **Load Testing**: Locust for performance validation

### Migration Status
This repository is in active migration from Streamlit to FastAPI + React. **The legacy Streamlit app in the root directory must remain fully functional and should continue to be used during the migration process.** Key migration priorities:
1. Preserve all financial calculation logic from the root directory application
2. Maintain data accuracy and simulation reliability
3. Add user authentication and premium features
4. Improve performance and scalability
5. Enable mobile-responsive design

**Migration Strategy**: The root directory contains the production-ready Streamlit application that should be kept operational while building the new API/frontend architecture. This ensures continuity of service and serves as the reference implementation for feature parity validation.

### Development Notes
- The legacy app uses session state management - this is being replaced with proper database persistence
- All financial calculations are being migrated to maintain backward compatibility
- The new architecture supports background job processing for large simulations
- Redis caching is implemented for performance optimization
- Premium features include larger simulation counts and advanced reporting

### Configuration
- **Environment Variables**: Stored in `.env` files for each service
- **Database**: PostgreSQL for production, SQLite for development
- **Caching**: Redis for simulation results and session data
- **External APIs**: Yahoo Finance for market data (yfinance library)
- **Python Environment**: Uses conda environment named `pyfin` for Python dependencies

### Common Development Tasks
When working on this codebase:
1. **Adding new financial metrics**: First implement in the root directory Streamlit app, then migrate to `portfolio-simulator-api/utils/financial_calcs.py` and corresponding tests
2. **New API endpoints**: Create in `api/v1/` with proper authentication and validation, ensuring parity with Streamlit functionality
3. **Database changes**: Use Alembic migrations, test with both SQLite and PostgreSQL
4. **Frontend components**: Follow existing patterns in `components/` directory, replicating Streamlit UI behavior
5. **Chart integration**: Use Plotly.js for consistency with legacy visualizations in the root directory app
6. **Feature development**: Always maintain the root directory Streamlit app as the primary reference and keep it functional

### Technology Stack
- **Backend**: FastAPI, SQLAlchemy, Alembic, PostgreSQL, Redis, Celery
- **Frontend**: Next.js, React, TypeScript, Tailwind CSS, Plotly.js
- **Financial**: yfinance, pandas, numpy, scipy (for optimization)
- **Testing**: pytest, Jest, React Testing Library
- **Deployment**: Docker, Docker Compose, intended for DigitalOcean

This codebase represents a sophisticated financial analysis tool with complex Monte Carlo simulations, portfolio optimization, and comprehensive risk analysis capabilities.