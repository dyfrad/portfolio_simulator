import React from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { useSimulation } from '@/hooks/useSimulation';
import { SimulationConfig, SimulationResults } from '@/lib/types';
import { formatNumber } from '@/lib/utils';

const simulationSchema = z.object({
  horizon: z.number().min(1, 'Minimum 1 year').max(10, 'Maximum 10 years'),
  numSimulations: z.number().min(100, 'Minimum 100 simulations').max(10000, 'Maximum 10,000 simulations'),
  periodicContrib: z.number().min(0, 'Must be non-negative'),
  inflationRate: z.number().min(0, 'Must be non-negative').max(0.5, 'Maximum 50%'),
  stressScenario: z.string(),
  ter: z.number().min(0, 'Must be non-negative').max(0.1, 'Maximum 10%'),
  contributionFrequency: z.string(),
});

type SimulationFormData = z.infer<typeof simulationSchema>;

interface SimulationRunnerProps {
  portfolioId?: string;
  tickers?: string[];
  weights?: number[];
  initialInvestment?: number;
  onComplete?: (results: SimulationResults) => void;
}

export const SimulationRunner: React.FC<SimulationRunnerProps> = ({
  portfolioId,
  tickers = ['IWDA.AS', 'QDV5.DE', 'PPFB.DE', 'XEON.DE'],
  weights = [0.25, 0.25, 0.25, 0.25],
  initialInvestment = 100000,
  onComplete,
}) => {
  const { runSimulation, isRunning, progress, results, simulationError } = useSimulation();

  const {
    register,
    handleSubmit,
    watch,
    formState: { errors },
  } = useForm<SimulationFormData>({
    resolver: zodResolver(simulationSchema),
    defaultValues: {
      horizon: 5,
      numSimulations: 1000,
      periodicContrib: 0,
      inflationRate: 0.025,
      stressScenario: 'None',
      ter: 0.002,
      contributionFrequency: 'monthly',
    },
  });

  const onSubmit = async (data: SimulationFormData) => {
    try {
      const config: SimulationConfig = {
        portfolioId,
        tickers,
        weights,
        initialInvestment,
        horizon: data.horizon,
        numSimulations: data.numSimulations,
        periodicContrib: data.periodicContrib,
        inflationRate: data.inflationRate,
        stressScenario: data.stressScenario,
        ter: data.ter,
        optimizeWeights: false,
        startDate: '2015-01-01', // Default start date
        contributionFrequency: data.contributionFrequency,
      };

      const simulationResults = await runSimulation(config);
      onComplete?.(simulationResults);
    } catch (error) {
      console.error('Error running simulation:', error);
    }
  };

  return (
    <div className="space-y-6">
      {/* Simulation Configuration Panel */}
      <div className="bg-white p-6 rounded-lg shadow">
        <h3 className="text-lg font-semibold mb-4">Simulation Parameters</h3>
        
        <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Input
              label="Time Horizon (years)"
              type="number"
              min="1"
              max="50"
              {...register('horizon', { valueAsNumber: true })}
              error={errors.horizon?.message}
            />
            
            <Input
              label="Number of Simulations"
              type="number"
              min="1000"
              max="50000"
              step="1000"
              {...register('numSimulations', { valueAsNumber: true })}
              error={errors.numSimulations?.message}
              helperText="Higher number = more accurate results"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <Input
              label="Periodic Contribution (€)"
              type="number"
              min="0"
              step="100"
              {...register('periodicContrib', { valueAsNumber: true })}
              error={errors.periodicContrib?.message}
            />
            
            <Input
              label="Inflation Rate"
              type="number"
              min="0"
              max="0.5"
              step="0.001"
              {...register('inflationRate', { valueAsNumber: true })}
              error={errors.inflationRate?.message}
              helperText="Annual inflation rate (e.g., 0.025 = 2.5%)"
            />
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Contribution Frequency
              </label>
              <select
                {...register('contributionFrequency')}
                className="block w-full rounded-lg border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
              >
                <option value="monthly">Monthly</option>
                <option value="quarterly">Quarterly</option>
              </select>
            </div>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Stress Scenario
              </label>
              <select
                {...register('stressScenario')}
                className="block w-full rounded-lg border-gray-300 shadow-sm focus:border-blue-500 focus:ring-blue-500 sm:text-sm"
              >
                <option value="None">None</option>
                <option value="2008_Crisis">2008 Financial Crisis</option>
                <option value="2020_Covid">2020 COVID-19 Crash</option>
                <option value="High_Volatility">High Volatility Period</option>
                <option value="Low_Returns">Low Returns Period</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            
            <Input
              label="Total Expense Ratio (TER)"
              type="number"
              min="0"
              max="0.1"
              step="0.001"
              {...register('ter', { valueAsNumber: true })}
              error={errors.ter?.message}
              helperText="Annual fund expenses (e.g., 0.002 = 0.2%)"
            />
          </div>

          <Button
            type="submit"
            loading={isRunning}
            disabled={isRunning}
            className="w-full"
          >
            {isRunning ? 'Running Simulation...' : 'Run Monte Carlo Simulation'}
          </Button>
        </form>
      </div>

      {/* Progress Tracker */}
      {isRunning && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Simulation Progress</h3>
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span>Progress</span>
              <span>{Math.round(progress)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div
                className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                style={{ width: `${progress}%` }}
              />
            </div>
            <p className="text-sm text-gray-600">
              Running {formatNumber(watch('numSimulations') || 1000)} Monte Carlo simulations...
            </p>
          </div>
        </div>
      )}

      {/* Error Display */}
      {simulationError && (
        <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded">
          <strong>Error:</strong> {simulationError.message}
        </div>
      )}

      {/* Quick Results Preview */}
      {results && (
        <div className="bg-white p-6 rounded-lg shadow">
          <h3 className="text-lg font-semibold mb-4">Simulation Results</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                €{formatNumber(results.meanFinalValue)}
              </div>
              <div className="text-sm text-gray-600">Mean Value</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                €{formatNumber(results.medianFinalValue)}
              </div>
              <div className="text-sm text-gray-600">Median Value</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {formatNumber(results.sharpeRatio, 2)}
              </div>
              <div className="text-sm text-gray-600">Sharpe Ratio</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {formatNumber(results.maxDrawdown * 100, 1)}%
              </div>
              <div className="text-sm text-gray-600">Max Drawdown</div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}; 