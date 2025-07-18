# Portfolio Simulator Migration Roadmap
## From Streamlit to FastAPI + React/Next.js

### Overview
This roadmap details the migration of your sophisticated portfolio simulator from Streamlit to a production-ready FastAPI backend with React/Next.js frontend. The migration preserves all existing functionality while adding enterprise-grade features for commercial deployment.

### Current Application Analysis
**Existing Features to Preserve:**
- Monte Carlo simulations with 10,000+ iterations
- Portfolio optimization using Sharpe ratio maximization
- Historical backtesting with DCA support
- Interactive visualizations (Plotly charts)
- PDF report generation
- CSV file upload and parsing
- Stress testing scenarios
- Rebalancing strategies
- Complex financial calculations (VaR, CVaR, Sortino ratio)

**Technical Debt to Address:**
- Single-threaded Streamlit bottlenecks
- No user authentication/authorization
- Limited API capabilities
- Session state management limitations
- Deployment scalability constraints

---

## Phase 1: Backend Foundation (Weeks 1-4)

### Week 1: Project Setup and Architecture

#### 1.1 Initialize FastAPI Project Structure
```bash
portfolio-simulator-api/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py           # Environment configuration
│   │   ├── security.py         # JWT auth, password hashing
│   │   └── database.py         # Database connection
│   ├── api/
│   │   ├── __init__.py
│   │   ├── deps.py             # Dependencies (auth, db)
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── auth.py         # Authentication endpoints
│   │       ├── portfolios.py   # Portfolio CRUD
│   │       ├── simulations.py  # Simulation endpoints
│   │       └── reports.py      # Report generation
│   ├── models/
│   │   ├── __init__.py
│   │   ├── user.py             # User database models
│   │   ├── portfolio.py        # Portfolio models
│   │   └── simulation.py       # Simulation result models
│   ├── schemas/
│   │   ├── __init__.py
│   │   ├── auth.py             # Auth request/response schemas
│   │   ├── portfolio.py        # Portfolio schemas
│   │   └── simulation.py       # Simulation schemas
│   ├── services/
│   │   ├── __init__.py
│   │   ├── auth_service.py     # Authentication logic
│   │   ├── portfolio_service.py # Portfolio management
│   │   ├── simulation_service.py # Core simulation logic
│   │   └── report_service.py   # Report generation
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── financial_calcs.py  # Migrated calculation functions
│   │   └── data_fetcher.py     # Yahoo Finance integration
│   └── tests/
│       ├── __init__.py
│       ├── test_auth.py
│       ├── test_portfolios.py
│       └── test_simulations.py
├── alembic/                    # Database migrations
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

#### 1.2 Core Configuration Setup
Create `app/core/config.py`:
```python
from pydantic import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql://user:password@localhost/portfolio_db"
    
    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # External APIs
    YAHOO_FINANCE_TIMEOUT: int = 30
    
    # Simulation defaults
    MAX_SIMULATIONS: int = 50000
    DEFAULT_SIMULATIONS: int = 10000
    
    # Redis for caching
    REDIS_URL: str = "redis://localhost:6379"
    
    class Config:
        env_file = ".env"

settings = Settings()
```

### Week 2: Database and Authentication

#### 2.1 Database Models
Create `app/models/user.py`:
```python
from sqlalchemy import Column, Integer, String, DateTime, Boolean, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    is_premium = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_login = Column(DateTime)
    
    # Relationships
    portfolios = relationship("Portfolio", back_populates="owner")
    simulation_results = relationship("SimulationResult", back_populates="user")
```

Create `app/models/portfolio.py`:
```python
from sqlalchemy import Column, Integer, String, Float, JSON, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from .user import Base
import datetime

class Portfolio(Base):
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String, nullable=False)
    tickers = Column(JSON)  # ["IWDA.AS", "QDV5.DE", ...]
    weights = Column(JSON)  # [0.4, 0.3, 0.2, 0.1]
    initial_investment = Column(Float)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow)
    
    # Relationships
    owner = relationship("User", back_populates="portfolios")
    simulation_results = relationship("SimulationResult", back_populates="portfolio")
```

#### 2.2 Authentication Service
Create `app/services/auth_service.py`:
```python
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from app.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class AuthService:
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        return pwd_context.hash(password)
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: timedelta = None):
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=15)
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt
```

### Week 3: Core Business Logic Migration

#### 3.1 Migrate Financial Calculations
Create `app/utils/financial_calcs.py` - Migrate your existing functions:
```python
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from scipy.optimize import minimize

class FinancialCalculator:
    """Migrated financial calculation logic from original portfolio_simulator.py"""
    
    @staticmethod
    def fetch_data(tickers: List[str], start_date: str, end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch historical data - migrated from original fetch_data function"""
        data = yf.download(tickers, start=start_date, end=end_date)['Close'].dropna()
        if len(data) < 252:
            raise ValueError("Insufficient historical data for reliable simulation")
        return data
    
    @staticmethod
    def calculate_returns(data: pd.DataFrame, ter: float = 0.0) -> pd.DataFrame:
        """Calculate daily returns - migrated from original calculate_returns function"""
        returns = data.pct_change().dropna()
        if ter > 0:
            daily_ter = ter / 252
            returns = returns - daily_ter
        return returns
    
    @staticmethod
    def optimize_weights(returns: pd.DataFrame) -> Optional[np.ndarray]:
        """Portfolio optimization - migrated from original optimize_weights function"""
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252
        num_assets = len(returns.columns)
        
        def negative_sharpe(weights):
            portfolio_return = np.sum(mean_returns * weights)
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
            return -portfolio_return / portfolio_volatility if portfolio_volatility > 0 else -999
        
        constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
        bounds = tuple((0, 1) for _ in range(num_assets))
        initial_guess = np.array([1.0 / num_assets] * num_assets)
        
        try:
            result = minimize(negative_sharpe, initial_guess, method='SLSQP',
                            bounds=bounds, constraints=constraints)
            return result.x if result.success else None
        except Exception:
            return None
```

#### 3.2 Simulation Service
Create `app/services/simulation_service.py`:
```python
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Callable
from app.utils.financial_calcs import FinancialCalculator
from app.schemas.simulation import SimulationConfig, SimulationResults

class SimulationService:
    """Handles Monte Carlo simulations and backtesting"""
    
    def __init__(self):
        self.calculator = FinancialCalculator()
    
    async def run_monte_carlo_simulation(
        self, 
        config: SimulationConfig,
        progress_callback: Optional[Callable] = None
    ) -> SimulationResults:
        """
        Run Monte Carlo simulation - migrated from bootstrap_simulation function
        """
        # Fetch data
        data = self.calculator.fetch_data(config.tickers, config.start_date)
        returns = self.calculator.calculate_returns(data, config.ter)
        
        # Optimize weights if requested
        weights = np.array(config.weights)
        if config.optimize_weights:
            optimal_weights = self.calculator.optimize_weights(returns)
            if optimal_weights is not None:
                weights = optimal_weights
        
        # Run simulation logic (migrate your existing bootstrap_simulation logic)
        sim_final_values = []
        num_simulations = config.num_simulations
        
        for i in range(num_simulations):
            if progress_callback:
                await progress_callback(i / num_simulations)
            
            # Migrate your existing simulation logic here
            # ... (bootstrap sampling, DCA calculations, etc.)
            
        return SimulationResults(
            mean_final_value=np.mean(sim_final_values),
            median_final_value=np.median(sim_final_values),
            std_final_value=np.std(sim_final_values),
            # ... other results
        )
```

### Week 4: API Endpoints

#### 4.1 Portfolio Endpoints
Create `app/api/v1/portfolios.py`:
```python
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from sqlalchemy.orm import Session
from typing import List
from app.api.deps import get_current_user, get_db
from app.schemas.portfolio import PortfolioCreate, PortfolioResponse
from app.services.portfolio_service import PortfolioService

router = APIRouter()

@router.post("/", response_model=PortfolioResponse)
async def create_portfolio(
    portfolio: PortfolioCreate,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new portfolio"""
    service = PortfolioService(db)
    return await service.create_portfolio(portfolio, current_user.id)

@router.get("/", response_model=List[PortfolioResponse])
async def get_user_portfolios(
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all portfolios for current user"""
    service = PortfolioService(db)
    return await service.get_user_portfolios(current_user.id)

@router.post("/upload-csv")
async def upload_portfolio_csv(
    file: UploadFile = File(...),
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Upload and parse portfolio CSV file (Degiro format)"""
    service = PortfolioService(db)
    return await service.process_csv_upload(file, current_user.id)
```

#### 4.2 Simulation Endpoints
Create `app/api/v1/simulations.py`:
```python
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.api.deps import get_current_user, get_db
from app.schemas.simulation import SimulationRequest, SimulationResponse
from app.services.simulation_service import SimulationService
import asyncio

router = APIRouter()

@router.post("/run", response_model=SimulationResponse)
async def run_simulation(
    simulation_request: SimulationRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Run Monte Carlo portfolio simulation"""
    service = SimulationService()
    
    # For premium users, allow larger simulations
    max_sims = 50000 if current_user.is_premium else 10000
    if simulation_request.num_simulations > max_sims:
        raise HTTPException(400, f"Maximum {max_sims} simulations allowed")
    
    result = await service.run_monte_carlo_simulation(simulation_request.config)
    
    # Store result in database
    background_tasks.add_task(service.save_simulation_result, result, current_user.id)
    
    return result

@router.get("/progress/{simulation_id}")
async def get_simulation_progress(simulation_id: str):
    """WebSocket endpoint for real-time simulation progress"""
    # Implementation for real-time progress updates
    pass
```

---

## Phase 2: Frontend Development (Weeks 5-10)

### Week 5-6: Next.js Project Setup

#### 5.1 Initialize Frontend Project
```bash
npx create-next-app@latest portfolio-simulator-web --typescript --tailwind --eslint
cd portfolio-simulator-web
npm install @tanstack/react-query axios @headlessui/react @heroicons/react
npm install react-hook-form @hookform/resolvers zod
npm install recharts plotly.js react-plotly.js  # For charts
npm install @stripe/stripe-js @stripe/react-stripe-js  # For payments
```

#### 5.2 Project Structure
```
portfolio-simulator-web/
├── components/
│   ├── ui/                     # Reusable UI components
│   │   ├── Button.tsx
│   │   ├── Input.tsx
│   │   ├── Modal.tsx
│   │   └── LoadingSpinner.tsx
│   ├── layout/
│   │   ├── Header.tsx
│   │   ├── Sidebar.tsx
│   │   └── Layout.tsx
│   ├── portfolio/
│   │   ├── PortfolioBuilder.tsx    # Replaces sidebar_inputs.py
│   │   ├── WeightOptimizer.tsx
│   │   └── PortfolioList.tsx
│   ├── simulation/
│   │   ├── SimulationRunner.tsx
│   │   ├── ProgressTracker.tsx
│   │   └── ResultsDisplay.tsx      # Replaces results_display.py
│   └── charts/
│       ├── DistributionChart.tsx
│       ├── HistoricalChart.tsx
│       ├── DrawdownChart.tsx
│       └── WeightDriftChart.tsx
├── pages/
│   ├── api/                    # Next.js API routes (proxy to FastAPI)
│   ├── auth/
│   │   ├── login.tsx
│   │   └── register.tsx
│   ├── dashboard.tsx
│   ├── portfolios.tsx
│   └── simulation.tsx
├── lib/
│   ├── api.ts                  # API client
│   ├── auth.ts                 # Authentication utilities
│   └── types.ts                # TypeScript types
├── hooks/
│   ├── useAuth.ts
│   ├── usePortfolio.ts
│   └── useSimulation.ts
└── styles/
    └── globals.css
```

### Week 7-8: Core Components Migration

#### 7.1 Portfolio Builder Component
Create `components/portfolio/PortfolioBuilder.tsx`:
```typescript
import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

const portfolioSchema = z.object({
  name: z.string().min(1, 'Portfolio name required'),
  tickers: z.array(z.string()).min(1, 'At least one ticker required'),
  weights: z.array(z.number()).min(1),
  initialInvestment: z.number().min(100, 'Minimum €100 investment'),
  optimizeWeights: z.boolean().default(false),
});

type PortfolioFormData = z.infer<typeof portfolioSchema>;

export const PortfolioBuilder: React.FC = () => {
  const [isOptimizing, setIsOptimizing] = useState(false);
  
  const { register, handleSubmit, watch, setValue, formState: { errors } } = useForm<PortfolioFormData>({
    resolver: zodResolver(portfolioSchema),
    defaultValues: {
      tickers: ['IWDA.AS', 'QDV5.DE', 'PPFB.DE', 'XEON.DE'],
      weights: [0.25, 0.25, 0.25, 0.25],
      initialInvestment: 100000,
      optimizeWeights: false,
    }
  });

  const onSubmit = async (data: PortfolioFormData) => {
    if (data.optimizeWeights) {
      setIsOptimizing(true);
      // Call optimization API
      try {
        const optimizedWeights = await optimizePortfolioWeights(data.tickers);
        setValue('weights', optimizedWeights);
      } finally {
        setIsOptimizing(false);
      }
    }
    // Save portfolio
  };

  return (
    <div className="max-w-2xl mx-auto p-6">
      <h2 className="text-2xl font-bold mb-6">Portfolio Configuration</h2>
      
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        {/* Portfolio Name */}
        <div>
          <label className="block text-sm font-medium mb-2">Portfolio Name</label>
          <input
            {...register('name')}
            className="w-full p-3 border border-gray-300 rounded-lg"
            placeholder="My Portfolio"
          />
          {errors.name && <p className="text-red-500 text-sm mt-1">{errors.name.message}</p>}
        </div>

        {/* Tickers Configuration */}
        <TickerSelector 
          register={register} 
          watch={watch} 
          setValue={setValue} 
          errors={errors}
        />

        {/* Weight Optimization */}
        <div className="flex items-center space-x-3">
          <input
            {...register('optimizeWeights')}
            type="checkbox"
            className="h-4 w-4 text-blue-600"
          />
          <label className="text-sm font-medium">Optimize for Maximum Sharpe Ratio</label>
        </div>

        <button
          type="submit"
          disabled={isOptimizing}
          className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 disabled:opacity-50"
        >
          {isOptimizing ? 'Optimizing...' : 'Create Portfolio'}
        </button>
      </form>
    </div>
  );
};
```

#### 7.2 Simulation Runner Component
Create `components/simulation/SimulationRunner.tsx`:
```typescript
import React, { useState } from 'react';
import { useSimulation } from '@/hooks/useSimulation';
import { ProgressTracker } from './ProgressTracker';
import { ResultsDisplay } from './ResultsDisplay';

interface SimulationConfig {
  portfolioId: string;
  horizon: number;
  numSimulations: number;
  periodicContrib: number;
  inflationRate: number;
  stressScenario: string;
  // ... other config options
}

export const SimulationRunner: React.FC = () => {
  const [config, setConfig] = useState<SimulationConfig>({
    horizon: 20,
    numSimulations: 10000,
    periodicContrib: 1000,
    inflationRate: 0.025,
    stressScenario: 'None',
  });
  
  const { 
    runSimulation, 
    isRunning, 
    progress, 
    results, 
    error 
  } = useSimulation();

  const handleRunSimulation = async () => {
    await runSimulation(config);
  };

  return (
    <div className="space-y-6">
      {/* Simulation Configuration Panel */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Simulation Parameters</h3>
        
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium mb-2">Time Horizon (years)</label>
            <input
              type="number"
              value={config.horizon}
              onChange={(e) => setConfig({...config, horizon: Number(e.target.value)})}
              className="w-full p-2 border border-gray-300 rounded"
              min="1"
              max="50"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium mb-2">Number of Simulations</label>
            <select
              value={config.numSimulations}
              onChange={(e) => setConfig({...config, numSimulations: Number(e.target.value)})}
              className="w-full p-2 border border-gray-300 rounded"
            >
              <option value={1000}>1,000 (Fast)</option>
              <option value={10000}>10,000 (Standard)</option>
              <option value={50000}>50,000 (Premium)</option>
            </select>
          </div>
        </div>

        <button
          onClick={handleRunSimulation}
          disabled={isRunning}
          className="mt-4 bg-green-600 text-white px-6 py-2 rounded hover:bg-green-700 disabled:opacity-50"
        >
          {isRunning ? 'Running Simulation...' : 'Run Monte Carlo Simulation'}
        </button>
      </div>

      {/* Progress Tracker */}
      {isRunning && <ProgressTracker progress={progress} />}

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
          {error}
        </div>
      )}

      {/* Results Display */}
      {results && <ResultsDisplay results={results} />}
    </div>
  );
};
```

### Week 9-10: Charts and Advanced Features

#### 9.1 Interactive Charts Migration
Create `components/charts/DistributionChart.tsx`:
```typescript
import React from 'react';
import Plot from 'react-plotly.js';
import { SimulationResults } from '@/lib/types';

interface Props {
  results: SimulationResults;
  horizon: number;
}

export const DistributionChart: React.FC<Props> = ({ results, horizon }) => {
  const data = [{
    x: results.finalValues,
    type: 'histogram' as const,
    nbinsx: 50,
    marker: {
      color: 'rgba(59, 130, 246, 0.7)',
      line: {
        color: 'rgba(59, 130, 246, 1)',
        width: 1
      }
    },
    name: 'Final Portfolio Values'
  }];

  const layout = {
    title: `Portfolio Value Distribution (${horizon} years)`,
    xaxis: {
      title: 'Final Portfolio Value (€)',
      tickformat: ',.0f'
    },
    yaxis: {
      title: 'Frequency'
    },
    showlegend: false,
    margin: { t: 50, r: 50, b: 50, l: 50 }
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow">
      <Plot
        data={data}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '400px' }}
      />
    </div>
  );
};
```

---

## Phase 3: Advanced Features & Production (Weeks 11-16)

### Week 11-12: Authentication & User Management

#### 11.1 Authentication Integration
Create `lib/auth.ts`:
```typescript
import axios from 'axios';
import { jwtDecode } from 'jwt-decode';

interface User {
  id: string;
  email: string;
  fullName: string;
  isPremium: boolean;
}

interface AuthTokens {
  accessToken: string;
  refreshToken: string;
}

class AuthService {
  private static instance: AuthService;
  private tokens: AuthTokens | null = null;

  static getInstance(): AuthService {
    if (!AuthService.instance) {
      AuthService.instance = new AuthService();
    }
    return AuthService.instance;
  }

  async login(email: string, password: string): Promise<User> {
    const response = await axios.post('/api/auth/login', {
      email,
      password
    });

    this.tokens = response.data.tokens;
    localStorage.setItem('auth_tokens', JSON.stringify(this.tokens));
    
    return response.data.user;
  }

  async register(email: string, password: string, fullName: string): Promise<User> {
    const response = await axios.post('/api/auth/register', {
      email,
      password,
      full_name: fullName
    });

    this.tokens = response.data.tokens;
    localStorage.setItem('auth_tokens', JSON.stringify(this.tokens));
    
    return response.data.user;
  }

  getAccessToken(): string | null {
    if (!this.tokens) {
      const stored = localStorage.getItem('auth_tokens');
      if (stored) {
        this.tokens = JSON.parse(stored);
      }
    }
    return this.tokens?.accessToken || null;
  }

  async refreshToken(): Promise<string | null> {
    if (!this.tokens?.refreshToken) return null;

    try {
      const response = await axios.post('/api/auth/refresh', {
        refresh_token: this.tokens.refreshToken
      });

      this.tokens.accessToken = response.data.access_token;
      localStorage.setItem('auth_tokens', JSON.stringify(this.tokens));
      
      return this.tokens.accessToken;
    } catch (error) {
      this.logout();
      return null;
    }
  }

  logout(): void {
    this.tokens = null;
    localStorage.removeItem('auth_tokens');
    window.location.href = '/auth/login';
  }

  getCurrentUser(): User | null {
    const token = this.getAccessToken();
    if (!token) return null;

    try {
      const decoded = jwtDecode<any>(token);
      return {
        id: decoded.sub,
        email: decoded.email,
        fullName: decoded.full_name,
        isPremium: decoded.is_premium
      };
    } catch {
      return null;
    }
  }
}

export default AuthService.getInstance();
```

#### 11.2 Premium Features & Billing
Create `components/billing/SubscriptionManager.tsx`:
```typescript
import React, { useState } from 'react';
import { loadStripe } from '@stripe/stripe-js';
import { Elements, CardElement, useStripe, useElements } from '@stripe/react-stripe-js';

const stripePromise = loadStripe(process.env.NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY!);

export const SubscriptionManager: React.FC = () => {
  return (
    <Elements stripe={stripePromise}>
      <SubscriptionForm />
    </Elements>
  );
};

const SubscriptionForm: React.FC = () => {
  const stripe = useStripe();
  const elements = useElements();
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    setIsLoading(true);

    if (!stripe || !elements) return;

    const cardElement = elements.getElement(CardElement);
    if (!cardElement) return;

    try {
      // Create payment method
      const { error, paymentMethod } = await stripe.createPaymentMethod({
        type: 'card',
        card: cardElement,
      });

      if (error) {
        console.error('Payment method error:', error);
        return;
      }

      // Create subscription
      const response = await fetch('/api/billing/create-subscription', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${authService.getAccessToken()}`
        },
        body: JSON.stringify({
          payment_method_id: paymentMethod.id,
          plan: 'premium_monthly'
        })
      });

      const subscription = await response.json();

      if (subscription.status === 'active') {
        // Update user premium status
        window.location.reload();
      }
    } catch (error) {
      console.error('Subscription error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto bg-white p-6 rounded-lg shadow">
      <h3 className="text-xl font-semibold mb-4">Upgrade to Premium</h3>
      
      <div className="mb-6">
        <h4 className="font-medium mb-2">Premium Features:</h4>
        <ul className="text-sm text-gray-600 space-y-1">
          <li>• Up to 50,000 Monte Carlo simulations</li>
          <li>• Advanced portfolio optimization</li>
          <li>• Detailed PDF reports</li>
          <li>• Priority support</li>
          <li>• Custom stress testing scenarios</li>
        </ul>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="p-3 border border-gray-300 rounded">
          <CardElement />
        </div>
        
        <button
          type="submit"
          disabled={!stripe || isLoading}
          className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {isLoading ? 'Processing...' : 'Subscribe - €29.99/month'}
        </button>
      </form>
    </div>
  );
};
```

### Week 13-14: Performance & Caching

#### 13.1 Redis Caching for Simulations
Update `app/services/simulation_service.py`:
```python
import redis
import json
import hashlib
from typing import Optional

class SimulationService:
    def __init__(self):
        self.redis_client = redis.Redis.from_url(settings.REDIS_URL)
        self.calculator = FinancialCalculator()
    
    def _generate_cache_key(self, config: SimulationConfig) -> str:
        """Generate cache key from simulation configuration"""
        config_str = json.dumps({
            'tickers': sorted(config.tickers),
            'weights': config.weights,
            'horizon': config.horizon,
            'num_simulations': config.num_simulations,
            'inflation_rate': config.inflation_rate,
            'stress_scenario': config.stress_scenario,
            # Add other relevant config
        }, sort_keys=True)
        return f"simulation:{hashlib.md5(config_str.encode()).hexdigest()}"
    
    async def run_monte_carlo_simulation(
        self, 
        config: SimulationConfig,
        progress_callback: Optional[Callable] = None
    ) -> SimulationResults:
        """Run simulation with caching"""
        cache_key = self._generate_cache_key(config)
        
        # Check cache first
        cached_result = self.redis_client.get(cache_key)
        if cached_result:
            return SimulationResults.parse_raw(cached_result)
        
        # Run simulation
        result = await self._execute_simulation(config, progress_callback)
        
        # Cache result for 1 hour
        self.redis_client.setex(
            cache_key, 
            3600, 
            result.json()
        )
        
        return result
```

#### 13.2 Background Job Processing
Create `app/services/background_jobs.py`:
```python
from celery import Celery
from app.core.config import settings
from app.services.simulation_service import SimulationService

celery_app = Celery(
    "portfolio_simulator",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL
)

@celery_app.task
def run_simulation_background(simulation_id: str, config_dict: dict):
    """Run simulation in background"""
    config = SimulationConfig(**config_dict)
    service = SimulationService()
    
    # Run simulation
    result = service.run_monte_carlo_simulation(config)
    
    # Store result in database
    # Update simulation status
    
    return result.dict()

@celery_app.task
def generate_pdf_report(user_id: int, simulation_id: str):
    """Generate PDF report in background"""
    from app.services.report_service import ReportService
    
    service = ReportService()
    pdf_buffer = service.generate_simulation_report(simulation_id)
    
    # Upload to cloud storage (S3/DO Spaces)
    # Send email notification
    
    return {"status": "completed", "download_url": "..."}
```

### Week 15-16: Deployment & Production

#### 15.1 Docker Configuration
Create `docker-compose.production.yml`:
```yaml
version: '3.8'

services:
  api:
    build:
      context: ./portfolio-simulator-api
      dockerfile: Dockerfile
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/portfolio_db
      - REDIS_URL=redis://redis:6379
      - SECRET_KEY=${SECRET_KEY}
    depends_on:
      - db
      - redis
    ports:
      - "8000:8000"

  web:
    build:
      context: ./portfolio-simulator-web
      dockerfile: Dockerfile
    environment:
      - NEXT_PUBLIC_API_URL=https://api.yourportfolio.com
      - NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=${STRIPE_PUBLISHABLE_KEY}
    ports:
      - "3000:3000"

  db:
    image: postgres:14
    environment:
      - POSTGRES_DB=portfolio_db
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

  worker:
    build:
      context: ./portfolio-simulator-api
      dockerfile: Dockerfile
    command: celery -A app.services.background_jobs worker --loglevel=info
    environment:
      - DATABASE_URL=postgresql://user:password@db:5432/portfolio_db
      - REDIS_URL=redis://redis:6379
    depends_on:
      - db
      - redis

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/nginx/ssl
    depends_on:
      - api
      - web

volumes:
  postgres_data:
```

#### 15.2 DigitalOcean Deployment Script
Create `deploy.sh`:
```bash
#!/bin/bash

# DigitalOcean Deployment Script
echo "Deploying Portfolio Simulator to DigitalOcean..."

# Create droplet if not exists
doctl compute droplet create portfolio-simulator \
    --region nyc1 \
    --size s-2vcpu-4gb \
    --image ubuntu-20-04-x64 \
    --ssh-keys YOUR_SSH_KEY_ID

# Get droplet IP
DROPLET_IP=$(doctl compute droplet list --format Name,PublicIPv4 --no-header | grep portfolio-simulator | awk '{print $2}')

echo "Droplet IP: $DROPLET_IP"

# Setup server
ssh root@$DROPLET_IP << 'EOF'
    # Install Docker
    apt update
    apt install -y docker.io docker-compose
    systemctl start docker
    systemctl enable docker

    # Install Node.js
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash -
    apt install -y nodejs

    # Clone repository
    git clone https://github.com/yourusername/portfolio-simulator.git
    cd portfolio-simulator

    # Setup environment
    cp .env.example .env
    # Edit .env with production values

    # Build and start services
    docker-compose -f docker-compose.production.yml up -d

    # Setup SSL with Let's Encrypt
    apt install -y certbot python3-certbot-nginx
    certbot --nginx -d yourportfolio.com -d api.yourportfolio.com
EOF

echo "Deployment complete! Visit https://yourportfolio.com"
```

---

## Phase 4: Testing & Launch (Weeks 17-20)

### Week 17-18: Comprehensive Testing

#### 17.1 Backend Testing
Create comprehensive test suite in `app/tests/`:
```python
# test_simulations.py
import pytest
import numpy as np
from app.services.simulation_service import SimulationService
from app.schemas.simulation import SimulationConfig

@pytest.mark.asyncio
async def test_monte_carlo_simulation():
    """Test Monte Carlo simulation accuracy"""
    config = SimulationConfig(
        tickers=['IWDA.AS', 'QDV5.DE'],
        weights=[0.7, 0.3],
        horizon=10,
        num_simulations=1000,
        inflation_rate=0.025
    )
    
    service = SimulationService()
    results = await service.run_monte_carlo_simulation(config)
    
    assert results.mean_final_value > 0
    assert results.std_final_value > 0
    assert len(results.final_values) == 1000

@pytest.mark.asyncio 
async def test_portfolio_optimization():
    """Test portfolio optimization logic"""
    from app.utils.financial_calcs import FinancialCalculator
    
    calc = FinancialCalculator()
    data = calc.fetch_data(['IWDA.AS', 'QDV5.DE'], '2020-01-01')
    returns = calc.calculate_returns(data)
    
    optimal_weights = calc.optimize_weights(returns)
    
    assert optimal_weights is not None
    assert abs(sum(optimal_weights) - 1.0) < 0.01  # Weights sum to 1
    assert all(w >= 0 for w in optimal_weights)     # No short selling
```

#### 17.2 Frontend Testing
Create `__tests__/components/PortfolioBuilder.test.tsx`:
```typescript
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { PortfolioBuilder } from '@/components/portfolio/PortfolioBuilder';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

const createWrapper = () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });
  
  return ({ children }: { children: React.ReactNode }) => (
    <QueryClientProvider client={queryClient}>
      {children}
    </QueryClientProvider>
  );
};

describe('PortfolioBuilder', () => {
  it('renders with default tickers', () => {
    render(<PortfolioBuilder />, { wrapper: createWrapper() });
    
    expect(screen.getByDisplayValue('IWDA.AS')).toBeInTheDocument();
    expect(screen.getByDisplayValue('QDV5.DE')).toBeInTheDocument();
  });

  it('handles weight optimization', async () => {
    render(<PortfolioBuilder />, { wrapper: createWrapper() });
    
    const optimizeCheckbox = screen.getByLabelText(/optimize for maximum sharpe/i);
    fireEvent.click(optimizeCheckbox);
    
    const submitButton = screen.getByRole('button', { name: /create portfolio/i });
    fireEvent.click(submitButton);
    
    await waitFor(() => {
      expect(screen.getByText(/optimizing/i)).toBeInTheDocument();
    });
  });
});
```

### Week 19: Performance Testing & Optimization

#### 19.1 Load Testing
Create `load_test.py` using Locust:
```python
from locust import HttpUser, task, between
import json

class PortfolioSimulatorUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Login user"""
        response = self.client.post("/api/auth/login", json={
            "email": "test@example.com",
            "password": "testpass123"
        })
        self.token = response.json()["access_token"]
        self.headers = {"Authorization": f"Bearer {self.token}"}
    
    @task(3)
    def get_portfolios(self):
        """Get user portfolios"""
        self.client.get("/api/portfolios", headers=self.headers)
    
    @task(1)
    def run_simulation(self):
        """Run small simulation"""
        self.client.post("/api/simulations/run", 
            headers=self.headers,
            json={
                "config": {
                    "tickers": ["IWDA.AS", "QDV5.DE"],
                    "weights": [0.7, 0.3],
                    "horizon": 10,
                    "num_simulations": 100  # Small for load testing
                }
            }
        )
```

### Week 20: Launch Preparation

#### 20.1 Production Checklist
```markdown
## Pre-Launch Checklist

### Security
- [ ] HTTPS enabled with valid SSL certificates
- [ ] API rate limiting configured
- [ ] Input validation on all endpoints
- [ ] SQL injection protection verified
- [ ] XSS protection enabled
- [ ] CORS properly configured
- [ ] Environment variables secured

### Performance
- [ ] Database indexes optimized
- [ ] Redis caching implemented
- [ ] CDN configured for static assets
- [ ] Image optimization enabled
- [ ] Gzip compression enabled
- [ ] Background job processing working

### Monitoring
- [ ] Application logging configured
- [ ] Error tracking (Sentry) setup
- [ ] Performance monitoring enabled
- [ ] Database monitoring active
- [ ] Uptime monitoring configured

### Business
- [ ] Stripe payment processing tested
- [ ] Email notifications working
- [ ] Terms of service and privacy policy
- [ ] GDPR compliance measures
- [ ] Customer support system ready

### Testing
- [ ] End-to-end tests passing
- [ ] Load testing completed
- [ ] Security testing performed
- [ ] Browser compatibility verified
- [ ] Mobile responsiveness tested
```

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **Phase 1: Backend** | 4 weeks | FastAPI application with core simulation logic |
| **Phase 2: Frontend** | 6 weeks | React/Next.js application with full UI |
| **Phase 3: Advanced** | 6 weeks | Authentication, payments, performance optimization |
| **Phase 4: Launch** | 4 weeks | Testing, deployment, production readiness |

**Total Duration: 20 weeks (5 months)**

## Expected Benefits Post-Migration

1. **Scalability**: Handle 100+ concurrent users vs 5-10 with Streamlit
2. **Performance**: 10x faster simulation execution with background processing
3. **User Management**: Full authentication, subscription management
4. **Mobile Support**: Responsive design works on all devices
5. **API-First**: Easy to add mobile apps, integrations
6. **Commercial Ready**: Payment processing, usage analytics, reporting
7. **Maintainability**: Modern architecture, comprehensive testing
8. **Future-Proof**: Technology stack with 5+ year viability

## Risk Mitigation

1. **Technical Risks**: Maintain parallel Streamlit deployment during migration
2. **Timeline Risks**: Focus on MVP first, add premium features later
3. **User Adoption**: Provide migration assistance and training
4. **Performance Risks**: Implement caching and optimization from day 1
5. **Security Risks**: Security audit before production launch

This migration roadmap preserves all your sophisticated financial modeling capabilities while building a commercial-grade SaaS platform suitable for paid users on DigitalOcean infrastructure. 