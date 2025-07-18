# Portfolio Simulator API

FastAPI backend for the Portfolio Simulator application - a sophisticated financial analysis tool for portfolio Monte Carlo simulations, optimization, and backtesting.

## Features

- **Monte Carlo Simulations**: Run up to 50,000 simulations for portfolio analysis
- **Portfolio Optimization**: Sharpe ratio optimization using modern portfolio theory
- **Backtesting**: Historical performance analysis with Dollar-Cost Averaging
- **Risk Analysis**: VaR, CVaR, Sortino ratio, and drawdown analysis
- **Stress Testing**: Predefined scenarios (2008 recession, COVID crash, etc.)
- **PDF Reports**: Automated report generation with charts and metrics
- **CSV Import**: Support for Degiro transaction history and generic portfolio files
- **User Authentication**: JWT-based auth with premium tier support
- **Real-time Progress**: WebSocket updates for long-running simulations

## Technology Stack

- **Framework**: FastAPI 0.104.1 with async/await support
- **Database**: PostgreSQL with SQLAlchemy ORM
- **Caching**: Redis for performance optimization
- **Background Tasks**: Celery for report generation
- **Data Sources**: Yahoo Finance API integration
- **Authentication**: JWT tokens with bcrypt password hashing
- **Testing**: Pytest with async support
- **Documentation**: Auto-generated OpenAPI/Swagger docs

## Quick Start

### Prerequisites

- Python 3.11+
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose (optional)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd portfolio-simulator-api
   ```

2. **Set up virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Set up database**
   ```bash
   # Create database
   createdb portfolio_db
   
   # Run migrations
   alembic upgrade head
   ```

6. **Start the application**
   ```bash
   uvicorn app.main:app --reload
   ```

### Docker Setup (Recommended)

1. **Start all services**
   ```bash
   docker-compose up -d
   ```

2. **Run database migrations**
   ```bash
   docker-compose exec api alembic upgrade head
   ```

3. **Access the API**
   - API: http://localhost:8000
   - Docs: http://localhost:8000/docs
   - Flower (Celery monitoring): http://localhost:5555

## API Documentation

### Authentication

```bash
# Register user
curl -X POST "http://localhost:8000/api/v1/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "securepassword123",
    "full_name": "John Doe"
  }'

# Login
curl -X POST "http://localhost:8000/api/v1/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=securepassword123"
```

### Portfolio Management

```bash
# Create portfolio
curl -X POST "http://localhost:8000/api/v1/portfolios" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "My Portfolio",
    "tickers": ["IWDA.AS", "QDV5.DE", "PPFB.DE"],
    "weights": [0.6, 0.3, 0.1],
    "initial_investment": 100000
  }'

# Get portfolios
curl -H "Authorization: Bearer YOUR_TOKEN" \
  "http://localhost:8000/api/v1/portfolios"
```

### Run Simulation

```bash
curl -X POST "http://localhost:8000/api/v1/simulations/run" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "portfolio_id": 1,
    "horizon_years": 20,
    "num_simulations": 10000,
    "inflation_rate": 0.025,
    "periodic_contribution": 1000,
    "stress_scenario": "None"
  }'
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DATABASE_URL` | PostgreSQL connection string | Required |
| `REDIS_URL` | Redis connection string | Required |
| `SECRET_KEY` | JWT signing key | Required |
| `MAX_SIMULATIONS` | Maximum simulations per request | 50000 |
| `YAHOO_FINANCE_TIMEOUT` | API timeout in seconds | 30 |

### Premium Features

Premium users get access to:
- Up to 50,000 simulations (vs 10,000 for free users)
- Advanced stress testing scenarios
- Detailed PDF reports with charts
- Priority API access

## Development

### Project Structure

```
app/
├── main.py                 # FastAPI application entry
├── core/                   # Core configuration
│   ├── config.py          # Settings management
│   ├── security.py        # Authentication utilities
│   └── database.py        # Database connection
├── api/v1/                # API endpoints
│   ├── auth.py            # Authentication routes
│   ├── portfolios.py      # Portfolio management
│   ├── simulations.py     # Simulation endpoints
│   └── reports.py         # Report generation
├── models/                # Database models
├── schemas/               # Pydantic request/response models
├── services/              # Business logic
└── utils/                 # Utility functions
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest app/tests/test_auth.py -v
```

### Database Migrations

```bash
# Create new migration
alembic revision --autogenerate -m "Add new table"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

### Adding New Features

1. Add database models in `app/models/`
2. Create Pydantic schemas in `app/schemas/`
3. Implement business logic in `app/services/`
4. Add API endpoints in `app/api/v1/`
5. Write tests in `app/tests/`

## Performance Optimization

### Caching Strategy

- **Market Data**: 1-hour cache for historical prices
- **Simulation Results**: 30-minute cache for identical parameters
- **User Sessions**: Redis-based session storage

### Background Processing

Long-running operations are processed asynchronously:
- Monte Carlo simulations > 10,000 iterations
- PDF report generation
- Large CSV file processing

### Rate Limiting

- Free users: 60 requests/minute
- Premium users: 300 requests/minute
- Simulation endpoints: Additional throttling based on complexity

## Deployment

### Production Checklist

- [ ] Set strong `SECRET_KEY`
- [ ] Configure production database
- [ ] Set up Redis cluster
- [ ] Enable HTTPS
- [ ] Configure monitoring (logs, metrics)
- [ ] Set up backup strategy
- [ ] Configure email notifications
- [ ] Set up Stripe webhooks (if using payments)

### Environment-Specific Settings

```bash
# Production
export ENVIRONMENT=production
export DEBUG=false
export DATABASE_URL=postgresql://user:pass@prod-db:5432/portfolio_db

# Staging
export ENVIRONMENT=staging
export DEBUG=true
export DATABASE_URL=postgresql://user:pass@staging-db:5432/portfolio_db
```

## Monitoring

### Health Checks

- `/health` - Basic health check
- `/` - API status and version

### Logging

Structured JSON logging with:
- Request/response logging
- Error tracking with stack traces
- Performance metrics
- Security events

### Metrics

Key metrics to monitor:
- Request latency (p95, p99)
- Simulation execution time
- Database connection pool usage
- Redis cache hit rate
- Background task queue length

## Security

### Authentication

- JWT tokens with configurable expiration
- Refresh token rotation
- Password hashing with bcrypt
- Rate limiting per user

### Data Protection

- Input validation with Pydantic
- SQL injection prevention with SQLAlchemy
- CORS configuration
- Request size limits
- API versioning

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is private and proprietary. All rights reserved.

## Support

For technical support or questions:
- Email: mohit@msaharan.com
- Documentation: http://localhost:8000/docs
- Health Status: http://localhost:8000/health 