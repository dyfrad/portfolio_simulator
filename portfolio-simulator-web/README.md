# Portfolio Simulator Frontend

A modern React/Next.js frontend for the Portfolio Simulator application, providing advanced Monte Carlo simulation capabilities for investment portfolio analysis.

## Features

- **Portfolio Management**: Create, edit, and manage multiple investment portfolios
- **Monte Carlo Simulations**: Run sophisticated simulations with up to 50,000 iterations
- **Portfolio Optimization**: Automatic asset allocation optimization using Sharpe ratio maximization
- **Interactive Charts**: Visualize simulation results with Plotly.js charts
- **User Authentication**: Secure login and registration system
- **Responsive Design**: Modern UI that works on desktop and mobile devices
- **Real-time Progress**: Live progress tracking for long-running simulations

## Tech Stack

- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **State Management**: React Query (TanStack Query)
- **Forms**: React Hook Form with Zod validation
- **Charts**: Plotly.js with react-plotly.js
- **HTTP Client**: Axios
- **Authentication**: JWT with automatic token refresh

## Getting Started

### Prerequisites

- Node.js 18+ 
- npm or yarn
- Backend API running (see portfolio-simulator-api)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd portfolio-simulator-web
```

2. Install dependencies:
```bash
npm install
```

3. Set up environment variables:
```bash
cp .env.example .env.local
```

Edit `.env.local` and add your backend API URL:
```
NEXT_PUBLIC_API_URL=http://localhost:8000
```

4. Run the development server:
```bash
npm run dev
```

5. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
portfolio-simulator-web/
├── app/                    # Next.js App Router pages
│   ├── auth/              # Authentication pages
│   │   ├── login/         # Login page
│   │   └── register/      # Registration page
│   ├── dashboard/         # Main dashboard page
│   ├── globals.css        # Global styles
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Landing page
├── components/            # React components
│   ├── ui/               # Reusable UI components
│   │   ├── Button.tsx
│   │   ├── Input.tsx
│   │   └── LoadingSpinner.tsx
│   ├── portfolio/        # Portfolio-related components
│   │   └── PortfolioBuilder.tsx
│   ├── simulation/       # Simulation components
│   │   └── SimulationRunner.tsx
│   └── charts/           # Chart components
│       └── DistributionChart.tsx
├── hooks/                # Custom React hooks
│   ├── useAuth.ts        # Authentication hook
│   ├── usePortfolio.ts   # Portfolio management hook
│   └── useSimulation.ts  # Simulation hook
├── lib/                  # Utility libraries
│   ├── api.ts            # API client
│   ├── auth.ts           # Authentication service
│   ├── types.ts          # TypeScript type definitions
│   └── utils.ts          # Utility functions
└── public/               # Static assets
```

## Key Components

### PortfolioBuilder
Allows users to create and configure investment portfolios with:
- Multiple asset tickers
- Weight allocation with automatic normalization
- Initial investment amount
- Optional weight optimization

### SimulationRunner
Runs Monte Carlo simulations with configurable parameters:
- Time horizon (1-50 years)
- Number of simulations (1,000-50,000)
- Monthly contributions
- Inflation rate
- Stress scenarios
- Total expense ratio (TER)

### DistributionChart
Displays simulation results using Plotly.js:
- Histogram of final portfolio values
- Summary statistics (mean, median, percentiles)
- Interactive zoom and pan capabilities

## API Integration

The frontend communicates with the FastAPI backend through the `ApiService` class in `lib/api.ts`. Key endpoints include:

- **Authentication**: `/api/v1/auth/*`
- **Portfolios**: `/api/v1/portfolios/*`
- **Simulations**: `/api/v1/simulations/*`
- **Reports**: `/api/v1/reports/*`

## Authentication Flow

1. User registers/logs in through auth pages
2. JWT tokens are stored in localStorage
3. API requests automatically include Authorization header
4. Token refresh happens automatically on 401 responses
5. User is redirected to login on authentication failure

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint

### Code Style

- TypeScript strict mode enabled
- ESLint with Next.js configuration
- Prettier for code formatting
- Component props typed with interfaces
- Custom hooks for state management

### Testing

To add tests:
1. Create test files in `__tests__/` directory
2. Use Jest and React Testing Library
3. Run tests with `npm test`

## Deployment

### Vercel (Recommended)

1. Connect your GitHub repository to Vercel
2. Set environment variables in Vercel dashboard
3. Deploy automatically on push to main branch

### Other Platforms

The app can be deployed to any platform that supports Next.js:
- Netlify
- Railway
- DigitalOcean App Platform
- AWS Amplify

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | `http://localhost:8000` |

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support and questions:
- Create an issue in the GitHub repository
- Check the documentation in the backend repository
- Review the migration guide for implementation details
