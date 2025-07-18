'use client';

import React, { useState } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { PortfolioBuilder } from '@/components/portfolio/PortfolioBuilder';
import { SimulationRunner } from '@/components/simulation/SimulationRunner';
import { DistributionChart } from '@/components/charts/DistributionChart';
import { Button } from '@/components/ui/Button';
import { useAuth } from '@/hooks/useAuth';
import { usePortfolio } from '@/hooks/usePortfolio';
import { Portfolio, SimulationResults } from '@/lib/types';
import { formatCurrency } from '@/lib/utils';

// Create a client
const queryClient = new QueryClient();

function DashboardContent() {
  const { user, isAuthenticated, isLoading } = useAuth();
  const { portfolios, isLoadingPortfolios } = usePortfolio();
  const [selectedPortfolio, setSelectedPortfolio] = useState<Portfolio | null>(null);
  const [simulationResults, setSimulationResults] = useState<SimulationResults | null>(null);
  const [showPortfolioBuilder, setShowPortfolioBuilder] = useState(false);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  if (!isAuthenticated) {
    return (
      <div className="flex items-center justify-center min-h-screen">
        <div className="text-center">
          <h1 className="text-2xl font-bold mb-4">Welcome to Portfolio Simulator</h1>
          <p className="text-gray-600 mb-6">Please log in to access your portfolios and simulations.</p>
          <Button onClick={() => window.location.href = '/auth/login'}>
            Go to Login
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div>
              <h1 className="text-3xl font-bold text-gray-900">Portfolio Simulator</h1>
              <p className="text-gray-600">Welcome back, {user?.fullName}</p>
            </div>
            <div className="flex items-center space-x-4">
              <span className="text-sm text-gray-600">
                {user?.isPremium ? 'Premium User' : 'Free User'}
              </span>
              <Button variant="ghost" onClick={() => window.location.href = '/auth/logout'}>
                Logout
              </Button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          {/* Left Column - Portfolio Management */}
          <div className="lg:col-span-1 space-y-6">
            {/* Portfolio List */}
            <div className="bg-white rounded-lg shadow p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold">Your Portfolios</h2>
                <Button
                  size="sm"
                  onClick={() => setShowPortfolioBuilder(true)}
                >
                  Create New
                </Button>
              </div>

              {isLoadingPortfolios ? (
                <div className="text-center py-4">
                  <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-blue-600 mx-auto"></div>
                </div>
              ) : portfolios.length === 0 ? (
                <div className="text-center py-8">
                  <p className="text-gray-500 mb-4">No portfolios yet</p>
                  <Button
                    variant="secondary"
                    onClick={() => setShowPortfolioBuilder(true)}
                  >
                    Create Your First Portfolio
                  </Button>
                </div>
              ) : (
                <div className="space-y-3">
                  {portfolios.map((portfolio) => (
                    <div
                      key={portfolio.id}
                      className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                        selectedPortfolio?.id === portfolio.id
                          ? 'border-blue-500 bg-blue-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                      onClick={() => setSelectedPortfolio(portfolio)}
                    >
                      <h3 className="font-medium">{portfolio.name}</h3>
                      <p className="text-sm text-gray-600">
                        {portfolio.tickers.join(', ')}
                      </p>
                      <p className="text-sm text-gray-600">
                        Initial: {formatCurrency(portfolio.initialInvestment)}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Portfolio Builder Modal */}
            {showPortfolioBuilder && (
              <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
                <div className="bg-white rounded-lg max-w-2xl w-full mx-4 max-h-[90vh] overflow-y-auto">
                  <PortfolioBuilder
                    onSuccess={() => {
                      setShowPortfolioBuilder(false);
                      // Refresh portfolio list - the newly created portfolio will appear
                      window.location.reload();
                    }}
                    onCancel={() => setShowPortfolioBuilder(false)}
                  />
                </div>
              </div>
            )}
          </div>

          {/* Right Column - Simulation */}
          <div className="lg:col-span-2 space-y-6">
            {selectedPortfolio ? (
              <>
                {/* Selected Portfolio Info */}
                <div className="bg-white rounded-lg shadow p-6">
                  <h2 className="text-xl font-semibold mb-4">
                    {selectedPortfolio.name}
                  </h2>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div>
                      <div className="text-sm text-gray-600">Assets</div>
                      <div className="font-medium">{selectedPortfolio.tickers.length}</div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">Initial Investment</div>
                      <div className="font-medium">
                        {formatCurrency(selectedPortfolio.initialInvestment)}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">Created</div>
                      <div className="font-medium">
                        {new Date(selectedPortfolio.createdAt).toLocaleDateString()}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-gray-600">Last Updated</div>
                      <div className="font-medium">
                        {new Date(selectedPortfolio.updatedAt).toLocaleDateString()}
                      </div>
                    </div>
                  </div>
                </div>

                {/* Simulation Runner */}
                <SimulationRunner
                  portfolioId={selectedPortfolio.id}
                  tickers={selectedPortfolio.tickers}
                  weights={selectedPortfolio.weights}
                  initialInvestment={selectedPortfolio.initialInvestment}
                  onComplete={setSimulationResults}
                />

                {/* Simulation Results */}
                {simulationResults && (
                  <DistributionChart
                    results={simulationResults}
                    horizon={20} // This should come from the simulation config
                  />
                )}
              </>
            ) : (
              <div className="bg-white rounded-lg shadow p-12 text-center">
                <h2 className="text-xl font-semibold mb-4">Select a Portfolio</h2>
                <p className="text-gray-600 mb-6">
                  Choose a portfolio from the left panel to run simulations and view results.
                </p>
                {portfolios.length === 0 && (
                  <Button onClick={() => setShowPortfolioBuilder(true)}>
                    Create Your First Portfolio
                  </Button>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default function DashboardPage() {
  return (
    <QueryClientProvider client={queryClient}>
      <DashboardContent />
    </QueryClientProvider>
  );
} 