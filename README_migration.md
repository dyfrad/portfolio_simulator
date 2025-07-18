# Portfolio Simulator Migration Guide

This document provides comprehensive instructions for using the migrated Portfolio Simulator application, which has been successfully migrated from a Streamlit monolith to a modern FastAPI + Next.js architecture.

## 🏗️ Architecture Overview

The application now consists of three main components:

1. **Legacy Streamlit App** (Root directory) - Original application, kept functional during migration
2. **FastAPI Backend** (`portfolio-simulator-api/`) - Modern REST API with authentication
3. **Next.js Frontend** (`portfolio-simulator-web/`) - Modern React-based web interface

## 🚀 Quick Start

### Prerequisites

- **Python**: Conda environment named `pyfin` with required packages
- **Node.js**: Version 18+ for the frontend
- **Database**: PostgreSQL (production) or SQLite (development)

### 1. Environment Setup

```bash
# Activate the conda environment
conda activate pyfin

# Verify Python path
which python  # Should show: /Users/msaharan/products/anaconda3/envs/pyfin/bin/python
```

### 2. Backend Setup (FastAPI)

```bash
# Navigate to backend directory
cd portfolio-simulator-api/

# Install dependencies (if needed) - Using UV (recommended for speed)
uv pip install -r requirements.txt --python /Users/msaharan/products/anaconda3/envs/pyfin/bin/python

# Alternative: Using pip
# /Users/msaharan/products/anaconda3/envs/pyfin/bin/pip install -r requirements.txt

# Run database migrations
/Users/msaharan/products/anaconda3/envs/pyfin/bin/python -m alembic upgrade head

# Start the backend server
/Users/msaharan/products/anaconda3/envs/pyfin/bin/python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at:
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 3. Frontend Setup (Next.js)

```bash
# Navigate to frontend directory
cd portfolio-simulator-web/

# Install dependencies (if needed)
npm install

# Start the development server
npm run dev
```

The frontend will be available at:
- **Application**: http://localhost:3000
- **Network Access**: http://172.20.10.2:3000

### 4. Legacy Streamlit App (Backup)

```bash
# From root directory
conda activate pyfin
streamlit run portfolio_simulator_ui.py
```

## 📋 Usage Guide

### User Registration & Authentication

1. **Visit the Homepage**: http://localhost:3000
2. **Create Account**: Click "Get Started" or navigate to `/auth/register`
3. **Login**: Use your credentials at `/auth/login`
4. **Dashboard**: Access your portfolio management at `/dashboard`

### Portfolio Management

1. **Create Portfolio**:
   - Click "Create New" in the dashboard
   - Enter portfolio name and initial investment
   - Add asset tickers (e.g., `IWDA.AS`, `QDV5.DE`)
   - Set allocation weights (must sum to 100%)
   - Optionally enable weight optimization

2. **Manage Portfolios**:
   - View all portfolios in the left panel
   - Click to select and view details
   - Edit or delete existing portfolios

### Monte Carlo Simulations

1. **Select Portfolio**: Choose from your created portfolios
2. **Configure Simulation**:
   - Set investment horizon (years)
   - Choose number of simulations (up to 50,000)
   - Configure periodic contributions
   - Set inflation rate and expense ratios
   - Select stress test scenarios

3. **Run Simulation**: Click "Run Simulation" and monitor progress
4. **View Results**: Interactive charts showing:
   - Return distribution histograms
   - Percentile analysis
   - Risk metrics (VaR, CVaR, Sharpe ratio)
   - Historical performance

### Report Generation

1. **Generate PDF Report**: Click "Generate Report" after simulation
2. **Download**: Access comprehensive PDF with charts and analysis
3. **Share**: Reports include all key metrics and visualizations

## 🔧 Development Commands

### Backend Commands

```bash
# Run tests
/Users/msaharan/products/anaconda3/envs/pyfin/bin/python -m pytest

# Run tests with coverage
/Users/msaharan/products/anaconda3/envs/pyfin/bin/python -m pytest --cov=app

# Run specific test file
/Users/msaharan/products/anaconda3/envs/pyfin/bin/python -m pytest app/tests/test_auth.py -v

# Create new migration
/Users/msaharan/products/anaconda3/envs/pyfin/bin/python -m alembic revision --autogenerate -m "description"

# Rollback migration
/Users/msaharan/products/anaconda3/envs/pyfin/bin/python -m alembic downgrade -1
```

### Frontend Commands

```bash
# Development server
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Run linting
npm run lint

# Type checking
npm run type-check  # (if configured)
```

### Docker Setup (Optional)

```bash
# Start all services
docker-compose up -d

# Run migrations in container
docker-compose exec api alembic upgrade head

# View logs
docker-compose logs -f api
docker-compose logs -f web
```

## 📊 Key Features

### Financial Calculations
- **Monte Carlo Simulations**: Up to 50,000 iterations
- **Portfolio Optimization**: Sharpe ratio maximization
- **Risk Metrics**: VaR, CVaR, Sortino ratio, maximum drawdown
- **Historical Backtesting**: Dollar-cost averaging analysis

### User Experience
- **Authentication**: JWT-based with refresh tokens
- **Premium Features**: Enhanced simulation limits
- **Interactive Charts**: Plotly.js visualizations
- **CSV Import**: Degiro transaction history support
- **Mobile Responsive**: Optimized for all devices

### Technical Features
- **Real-time Updates**: Live simulation progress
- **Caching**: Redis for performance optimization
- **Background Jobs**: Celery for large simulations
- **API Documentation**: Auto-generated OpenAPI specs

## 🔍 API Endpoints

### Authentication
- `POST /api/v1/auth/register` - User registration
- `POST /api/v1/auth/login` - User login
- `POST /api/v1/auth/refresh` - Token refresh
- `GET /api/v1/auth/me` - Current user profile

### Portfolios
- `GET /api/v1/portfolios/` - List user portfolios
- `POST /api/v1/portfolios/` - Create new portfolio
- `GET /api/v1/portfolios/{id}` - Get portfolio details
- `PUT /api/v1/portfolios/{id}` - Update portfolio
- `DELETE /api/v1/portfolios/{id}` - Delete portfolio

### Simulations
- `POST /api/v1/simulations/run` - Run Monte Carlo simulation
- `GET /api/v1/simulations/progress/{id}` - Get simulation progress
- `GET /api/v1/simulations/{id}` - Get simulation results

### Reports
- `POST /api/v1/reports/generate/{simulation_id}` - Generate PDF report
- `GET /api/v1/reports/download/{report_id}` - Download report

## 🐛 Troubleshooting

### Common Issues

1. **Backend not starting**:
   - Verify conda environment is activated
   - Check Python path: `/Users/msaharan/products/anaconda3/envs/pyfin/bin/python`
   - Run database migrations: `alembic upgrade head`

2. **Frontend not connecting to backend**:
   - Ensure backend is running on port 8000
   - Check `NEXT_PUBLIC_API_URL` environment variable
   - Verify no CORS issues in browser console

3. **Database issues**:
   - Check database connection in `app/core/database.py`
   - Verify migrations are up to date
   - Check SQLite file permissions (development)

4. **Authentication problems**:
   - Clear browser localStorage
   - Check token expiration in dev tools
   - Verify JWT secret configuration

### Performance Optimization

1. **Large Simulations**:
   - Use background jobs for >10,000 simulations
   - Monitor memory usage during calculations
   - Consider Redis caching for repeated calculations

2. **Database Performance**:
   - Index frequently queried columns
   - Use connection pooling for production
   - Consider read replicas for analytics

## 📈 Migration Status

### ✅ Completed Features
- User authentication and authorization
- Portfolio CRUD operations
- Monte Carlo simulation engine
- Interactive charting and visualization
- PDF report generation
- CSV import functionality
- Mobile-responsive design

### 🔄 In Progress
- Premium subscription features
- Advanced optimization algorithms
- Real-time market data integration
- Enhanced analytics dashboard

### 📋 Future Enhancements
- Social features (portfolio sharing)
- Advanced backtesting strategies
- Machine learning-based predictions
- Multi-currency support
- API rate limiting and monitoring

## 📞 Support

For issues or questions regarding the migrated application:

1. **Check API Documentation**: http://localhost:8000/docs
2. **Review Logs**: Check both backend and frontend console logs
3. **Database Issues**: Verify migrations and connection settings
4. **Performance**: Monitor resource usage during simulations

## 🎯 Next Steps

1. **Test Core Functionality**: Register, create portfolios, run simulations
2. **Validate Data Accuracy**: Compare results with legacy Streamlit app
3. **Performance Testing**: Test with large portfolios and simulations
4. **Security Review**: Audit authentication and authorization
5. **Production Deployment**: Configure for production environment

The migrated application maintains full feature parity with the original Streamlit app while providing a modern, scalable architecture for future enhancements.