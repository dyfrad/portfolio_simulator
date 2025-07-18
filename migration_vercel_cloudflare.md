# Portfolio Simulator Migration: Vercel & Cloudflare Approaches
## Leveraging Platform-as-a-Service for Simplified Deployment

### Overview
Instead of building and managing your own infrastructure, this guide explores using Vercel or Cloudflare's platform services to deploy your portfolio simulator. These approaches can significantly reduce development time, infrastructure complexity, and operational overhead while providing enterprise-grade performance and scalability.

---

## Approach Comparison

| Feature | Custom FastAPI + React | Vercel Full-Stack | Cloudflare Full-Stack | Hybrid (Platform Frontend + Custom Backend) |
|---------|------------------------|-------------------|----------------------|-------------------------------------------|
| **Development Time** | 20 weeks | 8-12 weeks | 10-14 weeks | 6-10 weeks |
| **Infrastructure Management** | Full control | Minimal | Minimal | Moderate |
| **Scalability** | Manual scaling | Auto-scaling | Auto-scaling | Mixed |
| **Cost (MVP)** | $50-200/month | $20-100/month | $5-50/month | $30-150/month |
| **Performance** | Optimizable | Very Good | Excellent (Edge) | Good |
| **Vendor Lock-in** | None | Medium | Medium | Low |
| **Learning Curve** | High | Medium | Medium-High | Medium |

---

## Option 1: Vercel Full-Stack Approach

### Architecture Overview
```
User → Vercel Edge Network → Next.js App → Vercel Serverless Functions → Vercel Postgres
                                      ↓
                                Vercel Blob Storage (PDFs)
                                      ↓
                                External APIs (Yahoo Finance)
```

### Week 1-2: Vercel Setup & Project Structure

#### 1.1 Initialize Vercel Project
```bash
npx create-next-app@latest portfolio-simulator --typescript --tailwind --app
cd portfolio-simulator

# Install Vercel CLI
npm i -g vercel

# Install additional dependencies
npm install @vercel/postgres @vercel/blob
npm install @auth/nextjs-auth0 # or @clerk/nextjs for auth
npm install stripe recharts plotly.js react-plotly.js
npm install @tanstack/react-query zod react-hook-form
```

#### 1.2 Project Structure
```
portfolio-simulator/
├── app/                          # Next.js 14 App Router
│   ├── (auth)/
│   │   ├── login/
│   │   └── register/
│   ├── dashboard/
│   │   ├── page.tsx
│   │   └── portfolio/
│   │       └── [id]/page.tsx
│   ├── simulation/
│   │   └── page.tsx
│   ├── api/                      # Serverless API routes
│   │   ├── auth/
│   │   │   └── route.ts
│   │   ├── portfolios/
│   │   │   ├── route.ts
│   │   │   └── [id]/route.ts
│   │   ├── simulations/
│   │   │   ├── route.ts
│   │   │   └── run/route.ts
│   │   ├── reports/
│   │   │   └── route.ts
│   │   └── webhooks/
│   │       └── stripe/route.ts
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx
├── components/
│   ├── ui/                       # Reusable components
│   ├── portfolio/
│   │   ├── PortfolioBuilder.tsx
│   │   └── PortfolioList.tsx
│   ├── simulation/
│   │   ├── SimulationRunner.tsx
│   │   └── ResultsDisplay.tsx
│   └── charts/
│       ├── DistributionChart.tsx
│       └── HistoricalChart.tsx
├── lib/
│   ├── db.ts                     # Vercel Postgres client
│   ├── auth.ts                   # Authentication
│   ├── financial-calcs.ts        # Migrated calculation logic
│   └── types.ts
├── hooks/
├── utils/
└── vercel.json                   # Vercel configuration
```

### Week 3-4: Database & Authentication

#### 1.3 Database Setup with Vercel Postgres
Create `lib/db.ts`:
```typescript
import { sql } from '@vercel/postgres';

export interface User {
  id: string;
  email: string;
  name: string;
  created_at: Date;
  is_premium: boolean;
}

export interface Portfolio {
  id: string;
  user_id: string;
  name: string;
  tickers: string[];
  weights: number[];
  initial_investment: number;
  created_at: Date;
}

export class DatabaseService {
  static async createUser(email: string, name: string): Promise<User> {
    const result = await sql`
      INSERT INTO users (email, name, is_premium)
      VALUES (${email}, ${name}, false)
      RETURNING *
    `;
    return result.rows[0] as User;
  }

  static async createPortfolio(userId: string, portfolio: Omit<Portfolio, 'id' | 'user_id' | 'created_at'>): Promise<Portfolio> {
    const result = await sql`
      INSERT INTO portfolios (user_id, name, tickers, weights, initial_investment)
      VALUES (${userId}, ${portfolio.name}, ${JSON.stringify(portfolio.tickers)}, ${JSON.stringify(portfolio.weights)}, ${portfolio.initial_investment})
      RETURNING *
    `;
    return {
      ...result.rows[0],
      tickers: JSON.parse(result.rows[0].tickers),
      weights: JSON.parse(result.rows[0].weights),
    } as Portfolio;
  }

  static async getUserPortfolios(userId: string): Promise<Portfolio[]> {
    const result = await sql`
      SELECT * FROM portfolios WHERE user_id = ${userId}
      ORDER BY created_at DESC
    `;
    return result.rows.map(row => ({
      ...row,
      tickers: JSON.parse(row.tickers),
      weights: JSON.parse(row.weights),
    })) as Portfolio[];
  }

  static async getPortfolio(id: string, userId: string): Promise<Portfolio | null> {
    const result = await sql`
      SELECT * FROM portfolios WHERE id = ${id} AND user_id = ${userId}
    `;
    if (result.rows.length === 0) return null;
    
    const row = result.rows[0];
    return {
      ...row,
      tickers: JSON.parse(row.tickers),
      weights: JSON.parse(row.weights),
    } as Portfolio;
  }
}
```

#### 1.4 Authentication with Auth0 or Clerk
Create `lib/auth.ts` (using Clerk):
```typescript
import { auth } from '@clerk/nextjs';
import { redirect } from 'next/navigation';

export async function requireAuth() {
  const { userId } = auth();
  if (!userId) {
    redirect('/login');
  }
  return userId;
}

export async function getUser() {
  const { userId } = auth();
  return userId;
}
```

### Week 5-6: Core Financial Logic Migration

#### 1.5 Financial Calculations (Serverless-Optimized)
Create `lib/financial-calcs.ts`:
```typescript
// Migrate your core calculation logic with optimizations for serverless
export class FinancialCalculator {
  private static cache = new Map<string, any>();
  private static CACHE_TTL = 5 * 60 * 1000; // 5 minutes

  static async fetchData(tickers: string[], startDate: string): Promise<any> {
    const cacheKey = `data-${tickers.join(',')}-${startDate}`;
    const cached = this.cache.get(cacheKey);
    
    if (cached && Date.now() - cached.timestamp < this.CACHE_TTL) {
      return cached.data;
    }

    // Fetch from Yahoo Finance (with timeout for serverless)
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 25000); // 25s timeout

    try {
      const promises = tickers.map(async ticker => {
        const response = await fetch(
          `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?range=max&interval=1d`,
          { signal: controller.signal }
        );
        const data = await response.json();
        return { ticker, data: data.chart.result[0] };
      });

      const results = await Promise.all(promises);
      clearTimeout(timeoutId);

      const processedData = this.processYahooFinanceData(results, startDate);
      
      // Cache result
      this.cache.set(cacheKey, {
        data: processedData,
        timestamp: Date.now()
      });

      return processedData;
    } catch (error) {
      clearTimeout(timeoutId);
      throw new Error(`Failed to fetch data: ${error.message}`);
    }
  }

  static processYahooFinanceData(results: any[], startDate: string): any {
    // Process raw Yahoo Finance data into your required format
    // This replaces your yfinance.download() functionality
    const data = {};
    
    results.forEach(({ ticker, data: tickerData }) => {
      const timestamps = tickerData.timestamp;
      const closes = tickerData.indicators.quote[0].close;
      
      const startTimestamp = Math.floor(new Date(startDate).getTime() / 1000);
      const filteredData = timestamps
        .map((ts: number, idx: number) => ({ timestamp: ts, close: closes[idx] }))
        .filter((item: any) => item.timestamp >= startTimestamp && item.close !== null);

      data[ticker] = filteredData;
    });

    return data;
  }

  static calculateReturns(data: any, ter: number = 0): any {
    // Migrate your calculate_returns function
    const returns = {};
    const dailyTer = ter / 252;

    Object.keys(data).forEach(ticker => {
      const prices = data[ticker];
      const tickerReturns = [];

      for (let i = 1; i < prices.length; i++) {
        const prevPrice = prices[i - 1].close;
        const currPrice = prices[i].close;
        const return_ = (currPrice - prevPrice) / prevPrice - dailyTer;
        tickerReturns.push(return_);
      }

      returns[ticker] = tickerReturns;
    });

    return returns;
  }

  static async runMonteCarloSimulation(
    returns: any,
    weights: number[],
    config: {
      numSimulations: number;
      horizon: number;
      initialInvestment: number;
      inflationRate: number;
      periodicContrib: number;
      // ... other config
    }
  ): Promise<any> {
    // Migrate your bootstrap_simulation function
    // Optimized for serverless execution time limits
    
    const tickers = Object.keys(returns);
    const simResults = [];
    const maxExecutionTime = 25000; // 25 seconds for serverless
    const startTime = Date.now();

    for (let sim = 0; sim < config.numSimulations; sim++) {
      // Check execution time limit
      if (Date.now() - startTime > maxExecutionTime) {
        console.log(`Simulation stopped early due to time limit. Completed ${sim} of ${config.numSimulations} simulations.`);
        break;
      }

      // Monte Carlo simulation logic
      const simReturns = [];
      const totalDays = config.horizon * 252;

      // Bootstrap sample
      for (let day = 0; day < totalDays; day++) {
        const randomDay = Math.floor(Math.random() * returns[tickers[0]].length);
        const dayReturn = tickers.reduce((acc, ticker, idx) => {
          return acc + returns[ticker][randomDay] * weights[idx];
        }, 0);
        simReturns.push(dayReturn);
      }

      // Calculate final value with DCA
      let portfolioValue = config.initialInvestment;
      let totalInvested = config.initialInvestment;

      for (let day = 0; day < totalDays; day++) {
        portfolioValue *= (1 + simReturns[day]);
        
        // Add monthly contribution (simplified)
        if (day % 21 === 0 && day > 0) { // ~monthly
          portfolioValue += config.periodicContrib;
          totalInvested += config.periodicContrib;
        }
      }

      // Adjust for inflation
      const inflationAdjustedValue = portfolioValue / Math.pow(1 + config.inflationRate, config.horizon);
      simResults.push(inflationAdjustedValue);
    }

    return {
      finalValues: simResults,
      meanFinalValue: simResults.reduce((a, b) => a + b, 0) / simResults.length,
      medianFinalValue: simResults.sort((a, b) => a - b)[Math.floor(simResults.length / 2)],
      stdFinalValue: this.calculateStandardDeviation(simResults),
      // ... other metrics
    };
  }

  private static calculateStandardDeviation(values: number[]): number {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map(value => Math.pow(value - mean, 2));
    const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
    return Math.sqrt(avgSquaredDiff);
  }
}
```

### Week 7-8: API Routes & Simulation Logic

#### 1.6 Portfolio API Routes
Create `app/api/portfolios/route.ts`:
```typescript
import { NextRequest, NextResponse } from 'next/server';
import { requireAuth } from '@/lib/auth';
import { DatabaseService } from '@/lib/db';
import { z } from 'zod';

const createPortfolioSchema = z.object({
  name: z.string().min(1).max(100),
  tickers: z.array(z.string()).min(1).max(10),
  weights: z.array(z.number().min(0).max(1)),
  initialInvestment: z.number().min(100),
});

export async function GET() {
  try {
    const userId = await requireAuth();
    const portfolios = await DatabaseService.getUserPortfolios(userId);
    return NextResponse.json(portfolios);
  } catch (error) {
    return NextResponse.json({ error: 'Unauthorized' }, { status: 401 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const userId = await requireAuth();
    const body = await request.json();
    
    const validatedData = createPortfolioSchema.parse(body);
    
    // Validate weights sum to 1
    const weightSum = validatedData.weights.reduce((sum, weight) => sum + weight, 0);
    if (Math.abs(weightSum - 1) > 0.01) {
      return NextResponse.json({ error: 'Weights must sum to 1' }, { status: 400 });
    }

    const portfolio = await DatabaseService.createPortfolio(userId, validatedData);
    return NextResponse.json(portfolio, { status: 201 });
  } catch (error) {
    if (error instanceof z.ZodError) {
      return NextResponse.json({ error: 'Invalid data', details: error.errors }, { status: 400 });
    }
    return NextResponse.json({ error: 'Failed to create portfolio' }, { status: 500 });
  }
}
```

#### 1.7 Simulation API Route
Create `app/api/simulations/run/route.ts`:
```typescript
import { NextRequest, NextResponse } from 'next/server';
import { requireAuth } from '@/lib/auth';
import { DatabaseService } from '@/lib/db';
import { FinancialCalculator } from '@/lib/financial-calcs';
import { z } from 'zod';

const simulationSchema = z.object({
  portfolioId: z.string(),
  horizon: z.number().min(1).max(50),
  numSimulations: z.number().min(100).max(10000),
  inflationRate: z.number().min(-0.1).max(0.2),
  periodicContrib: z.number().min(0),
  // ... other parameters
});

export async function POST(request: NextRequest) {
  try {
    const userId = await requireAuth();
    const body = await request.json();
    
    const config = simulationSchema.parse(body);
    
    // Get portfolio
    const portfolio = await DatabaseService.getPortfolio(config.portfolioId, userId);
    if (!portfolio) {
      return NextResponse.json({ error: 'Portfolio not found' }, { status: 404 });
    }

    // Fetch market data
    const startDate = new Date();
    startDate.setFullYear(startDate.getFullYear() - 10); // 10 years of data
    
    const data = await FinancialCalculator.fetchData(
      portfolio.tickers,
      startDate.toISOString().split('T')[0]
    );

    const returns = FinancialCalculator.calculateReturns(data, 0.001); // 0.1% TER

    // Run simulation
    const results = await FinancialCalculator.runMonteCarloSimulation(
      returns,
      portfolio.weights,
      {
        numSimulations: config.numSimulations,
        horizon: config.horizon,
        initialInvestment: portfolio.initial_investment,
        inflationRate: config.inflationRate,
        periodicContrib: config.periodicContrib,
      }
    );

    // Store results (optional)
    // await DatabaseService.storeSimulationResult(userId, config.portfolioId, results);

    return NextResponse.json({
      portfolioId: config.portfolioId,
      config,
      results,
      generatedAt: new Date().toISOString(),
    });

  } catch (error) {
    console.error('Simulation error:', error);
    if (error instanceof z.ZodError) {
      return NextResponse.json({ error: 'Invalid data', details: error.errors }, { status: 400 });
    }
    return NextResponse.json({ error: 'Simulation failed' }, { status: 500 });
  }
}

// Add response timeout handling
export const maxDuration = 30; // Vercel Pro plan allows up to 30s
```

### Week 9-10: Frontend Components & Deployment

#### 1.8 Main Dashboard Component
Create `app/dashboard/page.tsx`:
```typescript
'use client';

import { useState, useEffect } from 'react';
import { useUser } from '@clerk/nextjs';
import { PortfolioList } from '@/components/portfolio/PortfolioList';
import { SimulationRunner } from '@/components/simulation/SimulationRunner';

export default function Dashboard() {
  const { user, isLoaded } = useUser();
  const [selectedPortfolio, setSelectedPortfolio] = useState<string | null>(null);

  if (!isLoaded) {
    return <div className="flex justify-center items-center h-screen">Loading...</div>;
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900">Portfolio Simulator</h1>
        <p className="text-gray-600">Welcome back, {user?.firstName || 'User'}</p>
      </div>

      <div className="grid lg:grid-cols-3 gap-8">
        {/* Portfolio Selection */}
        <div className="lg:col-span-1">
          <PortfolioList
            selectedPortfolio={selectedPortfolio}
            onSelectPortfolio={setSelectedPortfolio}
          />
        </div>

        {/* Simulation & Results */}
        <div className="lg:col-span-2">
          {selectedPortfolio ? (
            <SimulationRunner portfolioId={selectedPortfolio} />
          ) : (
            <div className="bg-gray-50 rounded-lg p-8 text-center">
              <p className="text-gray-500">Select a portfolio to run simulations</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
```

#### 1.9 Vercel Deployment Configuration
Create `vercel.json`:
```json
{
  "functions": {
    "app/api/simulations/run/route.ts": {
      "maxDuration": 30
    }
  },
  "env": {
    "POSTGRES_URL": "@postgres_url",
    "CLERK_SECRET_KEY": "@clerk_secret_key",
    "STRIPE_SECRET_KEY": "@stripe_secret_key"
  },
  "headers": [
    {
      "source": "/api/(.*)",
      "headers": [
        {
          "key": "Access-Control-Allow-Origin",
          "value": "*"
        }
      ]
    }
  ]
}
```

#### 1.10 Database Schema (SQL)
Create `schema.sql`:
```sql
-- Run this in Vercel Postgres dashboard
CREATE TABLE users (
  id SERIAL PRIMARY KEY,
  email VARCHAR(255) UNIQUE NOT NULL,
  name VARCHAR(255),
  is_premium BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE portfolios (
  id SERIAL PRIMARY KEY,
  user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
  name VARCHAR(100) NOT NULL,
  tickers JSONB NOT NULL,
  weights JSONB NOT NULL,
  initial_investment DECIMAL(15,2) NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE simulation_results (
  id SERIAL PRIMARY KEY,
  user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
  portfolio_id INTEGER REFERENCES portfolios(id) ON DELETE CASCADE,
  config JSONB NOT NULL,
  results JSONB NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_portfolios_user_id ON portfolios(user_id);
CREATE INDEX idx_simulation_results_user_id ON simulation_results(user_id);
CREATE INDEX idx_simulation_results_portfolio_id ON simulation_results(portfolio_id);
```

---

## Option 2: Cloudflare Full-Stack Approach

### Architecture Overview
```
User → Cloudflare Edge → Cloudflare Pages → Cloudflare Workers → Cloudflare D1 Database
                                        ↓
                                Cloudflare R2 Storage (PDFs)
                                        ↓
                                External APIs (Yahoo Finance)
```

### Week 1-2: Cloudflare Setup

#### 2.1 Initialize Cloudflare Project
```bash
npm create cloudflare@latest portfolio-simulator -- --framework=next

cd portfolio-simulator

# Install Cloudflare-specific packages
npm install @cloudflare/next-on-pages
npm install wrangler --save-dev

# Install app dependencies
npm install @auth/core @auth/d1-adapter
npm install recharts plotly.js react-plotly.js
npm install @tanstack/react-query zod react-hook-form
```

#### 2.2 Wrangler Configuration
Create `wrangler.toml`:
```toml
name = "portfolio-simulator"
compatibility_date = "2024-01-15"

[env.production]
name = "portfolio-simulator"
routes = [
  { pattern = "yourportfolio.com/*", zone_name = "yourportfolio.com" }
]

[[env.production.d1_databases]]
binding = "DB"
database_name = "portfolio-simulator-db"
database_id = "your-database-id"

[[env.production.r2_buckets]]
binding = "STORAGE"
bucket_name = "portfolio-reports"

[env.production.vars]
NEXTAUTH_URL = "https://yourportfolio.com"
NEXTAUTH_SECRET = "your-secret"
STRIPE_PUBLISHABLE_KEY = "pk_live_..."

[env.development]
name = "portfolio-simulator-dev"

[[env.development.d1_databases]]
binding = "DB"
database_name = "portfolio-simulator-dev"
database_id = "your-dev-database-id"
```

#### 2.3 Database Setup with D1
Create `schema.sql`:
```sql
-- Cloudflare D1 Database Schema
CREATE TABLE users (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email TEXT UNIQUE NOT NULL,
  name TEXT,
  password_hash TEXT,
  is_premium INTEGER DEFAULT 0,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE portfolios (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL,
  name TEXT NOT NULL,
  tickers TEXT NOT NULL, -- JSON string
  weights TEXT NOT NULL, -- JSON string
  initial_investment REAL NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (user_id) REFERENCES users (id)
);

CREATE TABLE simulation_cache (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  cache_key TEXT UNIQUE NOT NULL,
  results TEXT NOT NULL, -- JSON string
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  expires_at DATETIME NOT NULL
);
```

Deploy database:
```bash
# Create database
wrangler d1 create portfolio-simulator-db

# Apply schema
wrangler d1 execute portfolio-simulator-db --file=schema.sql
```

### Week 3-4: Cloudflare Workers & API

#### 2.4 Financial Calculations Worker
Create `workers/financial-calculator.ts`:
```typescript
export interface Env {
  DB: D1Database;
  STORAGE: R2Bucket;
}

export interface SimulationConfig {
  tickers: string[];
  weights: number[];
  horizon: number;
  numSimulations: number;
  initialInvestment: number;
  inflationRate: number;
}

export class CloudflareFinancialCalculator {
  private env: Env;

  constructor(env: Env) {
    this.env = env;
  }

  async fetchMarketData(tickers: string[], startDate: string): Promise<any> {
    const cacheKey = `market-data-${tickers.join(',')}-${startDate}`;
    
    // Check D1 cache
    const cached = await this.env.DB.prepare(
      'SELECT results FROM simulation_cache WHERE cache_key = ? AND expires_at > datetime("now")'
    ).bind(cacheKey).first();

    if (cached) {
      return JSON.parse(cached.results);
    }

    // Fetch from Yahoo Finance with Cloudflare's global network
    const promises = tickers.map(async ticker => {
      const response = await fetch(
        `https://query1.finance.yahoo.com/v8/finance/chart/${ticker}?range=5y&interval=1d`,
        {
          cf: {
            cacheEverything: true,
            cacheTtl: 3600, // 1 hour cache
          },
        }
      );
      const data = await response.json();
      return { ticker, data: data.chart.result[0] };
    });

    const results = await Promise.all(promises);
    const processedData = this.processMarketData(results, startDate);

    // Cache for 1 hour
    await this.env.DB.prepare(
      'INSERT OR REPLACE INTO simulation_cache (cache_key, results, expires_at) VALUES (?, ?, datetime("now", "+1 hour"))'
    ).bind(cacheKey, JSON.stringify(processedData)).run();

    return processedData;
  }

  processMarketData(results: any[], startDate: string): any {
    const data: any = {};
    const startTimestamp = Math.floor(new Date(startDate).getTime() / 1000);

    results.forEach(({ ticker, data: tickerData }) => {
      if (!tickerData || !tickerData.timestamp) return;

      const timestamps = tickerData.timestamp;
      const closes = tickerData.indicators.quote[0].close;

      const filteredData = timestamps
        .map((ts: number, idx: number) => ({ timestamp: ts, close: closes[idx] }))
        .filter((item: any) => item.timestamp >= startTimestamp && item.close !== null);

      data[ticker] = filteredData;
    });

    return data;
  }

  calculateReturns(data: any, ter: number = 0): any {
    const returns: any = {};
    const dailyTer = ter / 252;

    Object.keys(data).forEach(ticker => {
      const prices = data[ticker];
      if (!prices || prices.length < 2) return;

      const tickerReturns = [];
      for (let i = 1; i < prices.length; i++) {
        const prevPrice = prices[i - 1].close;
        const currPrice = prices[i].close;
        const return_ = (currPrice - prevPrice) / prevPrice - dailyTer;
        tickerReturns.push(return_);
      }
      returns[ticker] = tickerReturns;
    });

    return returns;
  }

  async runMonteCarloSimulation(config: SimulationConfig): Promise<any> {
    // Check cache first
    const cacheKey = `simulation-${JSON.stringify(config)}`;
    const cached = await this.env.DB.prepare(
      'SELECT results FROM simulation_cache WHERE cache_key = ? AND expires_at > datetime("now")'
    ).bind(cacheKey).first();

    if (cached) {
      return JSON.parse(cached.results);
    }

    // Fetch market data
    const startDate = new Date();
    startDate.setFullYear(startDate.getFullYear() - 5);
    
    const data = await this.fetchMarketData(config.tickers, startDate.toISOString().split('T')[0]);
    const returns = this.calculateReturns(data, 0.001);

    // Run simulation (optimized for Workers CPU limits)
    const results = this.runSimulation(returns, config);

    // Cache results for 30 minutes
    await this.env.DB.prepare(
      'INSERT OR REPLACE INTO simulation_cache (cache_key, results, expires_at) VALUES (?, ?, datetime("now", "+30 minutes"))'
    ).bind(cacheKey, JSON.stringify(results)).run();

    return results;
  }

  private runSimulation(returns: any, config: SimulationConfig): any {
    const tickers = Object.keys(returns);
    const simResults = [];
    const maxSimulations = Math.min(config.numSimulations, 5000); // Limit for Workers

    for (let sim = 0; sim < maxSimulations; sim++) {
      const totalDays = config.horizon * 252;
      let portfolioValue = config.initialInvestment;

      // Simple Monte Carlo simulation
      for (let day = 0; day < totalDays; day++) {
        const randomDay = Math.floor(Math.random() * returns[tickers[0]].length);
        const dayReturn = tickers.reduce((acc, ticker, idx) => {
          return acc + (returns[ticker][randomDay] || 0) * config.weights[idx];
        }, 0);
        
        portfolioValue *= (1 + dayReturn);
      }

      // Adjust for inflation
      const inflationAdjustedValue = portfolioValue / Math.pow(1 + config.inflationRate, config.horizon);
      simResults.push(inflationAdjustedValue);
    }

    return {
      finalValues: simResults,
      meanFinalValue: simResults.reduce((a, b) => a + b, 0) / simResults.length,
      medianFinalValue: simResults.sort((a, b) => a - b)[Math.floor(simResults.length / 2)],
      stdFinalValue: this.calculateStandardDeviation(simResults),
      simulationsRun: maxSimulations,
    };
  }

  private calculateStandardDeviation(values: number[]): number {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const squaredDiffs = values.map(value => Math.pow(value - mean, 2));
    const avgSquaredDiff = squaredDiffs.reduce((a, b) => a + b, 0) / values.length;
    return Math.sqrt(avgSquaredDiff);
  }
}

// Worker event handler
export default {
  async fetch(request: Request, env: Env): Promise<Response> {
    const url = new URL(request.url);
    
    if (url.pathname === '/api/simulation' && request.method === 'POST') {
      try {
        const config: SimulationConfig = await request.json();
        const calculator = new CloudflareFinancialCalculator(env);
        const results = await calculator.runMonteCarloSimulation(config);
        
        return new Response(JSON.stringify(results), {
          headers: {
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
          },
        });
      } catch (error) {
        return new Response(JSON.stringify({ error: 'Simulation failed' }), {
          status: 500,
          headers: { 'Content-Type': 'application/json' },
        });
      }
    }

    return new Response('Not found', { status: 404 });
  },
};
```

### Week 5-6: Next.js Frontend with Cloudflare Pages

#### 2.5 API Integration
Create `lib/api.ts`:
```typescript
const API_BASE = process.env.NODE_ENV === 'production' 
  ? 'https://yourportfolio.com'
  : 'http://localhost:3000';

export class CloudflareAPIClient {
  static async runSimulation(config: any) {
    const response = await fetch(`${API_BASE}/api/simulation`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      throw new Error('Simulation failed');
    }

    return response.json();
  }

  static async savePortfolio(portfolio: any) {
    const response = await fetch(`${API_BASE}/api/portfolios`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(portfolio),
    });

    return response.json();
  }
}
```

---

## Option 3: Hybrid Approach (Platform Frontend + Custom Backend)

### Architecture Overview
```
User → Vercel/Cloudflare Frontend → DigitalOcean FastAPI Backend → PostgreSQL
                                              ↓
                                        Redis Cache
                                              ↓
                                        Background Workers
```

### Advantages of Hybrid Approach
1. **Best of both worlds**: Platform-managed frontend with full control over backend
2. **Reduced vendor lock-in**: Can migrate frontend or backend independently
3. **Optimal performance**: Complex calculations on dedicated servers
4. **Cost-effective**: Pay for compute only when needed

### Week 1-3: Vercel Frontend Setup

#### 3.1 Frontend-Only Next.js Application
```typescript
// lib/api.ts - API client pointing to your FastAPI backend
const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'https://api.yourportfolio.com';

export class APIClient {
  private static getAuthHeaders() {
    const token = localStorage.getItem('auth_token');
    return token ? { Authorization: `Bearer ${token}` } : {};
  }

  static async runSimulation(config: any) {
    const response = await fetch(`${API_BASE}/api/v1/simulations/run`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...this.getAuthHeaders(),
      },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Simulation failed');
    }

    return response.json();
  }

  static async getPortfolios() {
    const response = await fetch(`${API_BASE}/api/v1/portfolios`, {
      headers: this.getAuthHeaders(),
    });

    return response.json();
  }

  static async createPortfolio(portfolio: any) {
    const response = await fetch(`${API_BASE}/api/v1/portfolios`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        ...this.getAuthHeaders(),
      },
      body: JSON.stringify(portfolio),
    });

    return response.json();
  }
}
```

### Week 4-6: FastAPI Backend on DigitalOcean

Deploy your FastAPI backend following the original migration guide but optimized for Vercel frontend:

#### 3.2 CORS Configuration for Vercel
```python
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Portfolio Simulator API")

# Configure CORS for Vercel deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourportfolio.vercel.app",
        "https://yourportfolio.com",
        "http://localhost:3000"  # Development
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Cost Comparison

### Vercel Full-Stack (Monthly Costs)
- **Hobby Plan**: $0 (limited to 100GB bandwidth)
- **Pro Plan**: $20 (includes Vercel Postgres, 1TB bandwidth)
- **Enterprise**: $40+ (dedicated support, advanced features)

**Additional costs:**
- Vercel Postgres: Included in Pro plan
- Vercel Blob Storage: $0.15/GB
- Function executions: Included in plan limits

**Total estimated cost for MVP**: $20-50/month

### Cloudflare Full-Stack (Monthly Costs)
- **Workers**: $5/month (10M requests)
- **Pages**: Free (unlimited static sites)
- **D1 Database**: Free tier (25GB database)
- **R2 Storage**: $0.015/GB stored

**Additional costs:**
- Workers KV: $0.50/million operations
- Email routing: Free
- Analytics: Free

**Total estimated cost for MVP**: $5-20/month

### Hybrid Approach (Monthly Costs)
- **Frontend (Vercel)**: $0-20
- **Backend (DigitalOcean)**: $24-50 (2-4GB droplet)
- **Database**: $15 (managed PostgreSQL)
- **Redis**: $15 (managed Redis)

**Total estimated cost for MVP**: $54-100/month

---

## Decision Matrix

| Factor | Vercel Full-Stack | Cloudflare Full-Stack | Hybrid | Custom (Original) |
|--------|-------------------|----------------------|---------|-------------------|
| **Time to Market** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Scalability** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Cost (MVP)** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Performance** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ |
| **Flexibility** | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Vendor Lock-in** | ⭐⭐ | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Learning Curve** | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

## Recommendation

**For immediate MVP launch**: **Vercel Full-Stack**
- Fastest time to market (8-12 weeks)
- Excellent developer experience
- Built-in authentication options
- Automatic scaling and optimization

**For maximum performance at scale**: **Cloudflare Full-Stack**
- Best global performance (edge computing)
- Lowest costs
- Excellent for international users

**For flexibility and control**: **Hybrid Approach**
- Balance between ease and control
- Can optimize each component independently
- Easier to migrate or scale individual parts

Start with Vercel full-stack for MVP, then consider migrating to hybrid or Cloudflare as you scale and need more control. 