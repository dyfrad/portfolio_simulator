import React, { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Button } from '@/components/ui/Button';
import { Input } from '@/components/ui/Input';
import { usePortfolio } from '@/hooks/usePortfolio';
import { PortfolioCreate } from '@/lib/types';
import { formatPercentage, normalizeWeights } from '@/lib/utils';

const portfolioSchema = z.object({
  name: z.string().min(1, 'Portfolio name is required'),
  initialInvestment: z.number().min(100, 'Minimum investment is €100'),
  optimizeWeights: z.boolean(),
});

type PortfolioFormData = z.infer<typeof portfolioSchema>;

interface PortfolioBuilderProps {
  onSuccess?: (portfolio: PortfolioCreate) => void;
  onCancel?: () => void;
}

export const PortfolioBuilder: React.FC<PortfolioBuilderProps> = ({
  onSuccess,
  onCancel,
}) => {
  const [isOptimizing, setIsOptimizing] = useState(false);
  const [tickers, setTickers] = useState(['IWDA.AS', 'QDV5.DE', 'PPFB.DE', 'XEON.DE']);
  const [weights, setWeights] = useState([0.25, 0.25, 0.25, 0.25]);
  const { createPortfolio, isCreating } = usePortfolio();

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
  } = useForm<PortfolioFormData>({
    resolver: zodResolver(portfolioSchema),
    defaultValues: {
      name: '',
      initialInvestment: 100000,
      optimizeWeights: false,
    },
  });

  const onSubmit = async (data: PortfolioFormData) => {
    try {
      if (data.optimizeWeights) {
        setIsOptimizing(true);
        // TODO: Call optimization API
        // const optimizedWeights = await optimizePortfolioWeights(tickers);
        // setWeights(optimizedWeights);
        setIsOptimizing(false);
      }

      const portfolio: PortfolioCreate = {
        name: data.name,
        tickers: tickers.filter(t => t.trim() !== ''),
        weights: weights,
        initialInvestment: data.initialInvestment,
        optimizeWeights: data.optimizeWeights,
      };

      const result = await createPortfolio(portfolio);
      reset();
      onSuccess?.(result);
    } catch (error) {
      console.error('Error creating portfolio:', error);
    }
  };

  const addTicker = () => {
    setTickers([...tickers, '']);
    setWeights([...weights, 0]);
  };

  const removeTicker = (index: number) => {
    const newTickers = tickers.filter((_, i) => i !== index);
    const newWeights = weights.filter((_, i) => i !== index);
    setTickers(newTickers);
    setWeights(normalizeWeights(newWeights));
  };

  const updateTicker = (index: number, value: string) => {
    const newTickers = [...tickers];
    newTickers[index] = value;
    setTickers(newTickers);
  };

  const updateWeight = (index: number, value: number) => {
    const newWeights = [...weights];
    newWeights[index] = value / 100; // Convert percentage to decimal
    setWeights(normalizeWeights(newWeights));
  };

  const totalWeight = weights.reduce((sum, weight) => sum + weight, 0);

  return (
    <div className="max-w-2xl mx-auto p-6 bg-white rounded-lg shadow">
      <h2 className="text-2xl font-bold mb-6">Create New Portfolio</h2>
      
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
        {/* Portfolio Name */}
        <Input
          label="Portfolio Name"
          {...register('name')}
          placeholder="My Portfolio"
          error={errors.name?.message}
        />

        {/* Tickers and Weights */}
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-lg font-semibold">Assets</h3>
            <Button
              type="button"
              variant="secondary"
              size="sm"
              onClick={addTicker}
            >
              Add Asset
            </Button>
          </div>

          {tickers.map((ticker, index) => (
            <div key={index} className="flex items-center space-x-4">
              <div className="flex-1">
                <Input
                  label={`Asset ${index + 1}`}
                  value={ticker}
                  onChange={(e) => updateTicker(index, e.target.value)}
                  placeholder="e.g., IWDA.AS"
                />
              </div>
              
              <div className="w-32">
                <Input
                  label="Weight (%)"
                  type="number"
                  step="0.01"
                  min="0"
                  max="100"
                  value={Math.round(weights[index] * 100)}
                  onChange={(e) => updateWeight(index, parseFloat(e.target.value) || 0)}
                />
              </div>
              
              {tickers.length > 1 && (
                <Button
                  type="button"
                  variant="danger"
                  size="sm"
                  onClick={() => removeTicker(index)}
                  className="mt-6"
                >
                  Remove
                </Button>
              )}
            </div>
          ))}

          {/* Weight Summary */}
          <div className="bg-gray-50 p-4 rounded-lg">
            <div className="flex justify-between items-center">
              <span className="font-medium">Total Weight:</span>
              <span className={totalWeight === 1 ? 'text-green-600' : 'text-red-600'}>
                {formatPercentage(totalWeight)}
              </span>
            </div>
            {totalWeight !== 1 && (
              <p className="text-sm text-red-600 mt-1">
                Weights must sum to 100%
              </p>
            )}
          </div>
        </div>

        {/* Initial Investment */}
        <Input
          label="Initial Investment (€)"
          type="number"
          step="100"
          min="100"
          {...register('initialInvestment', { valueAsNumber: true })}
          error={errors.initialInvestment?.message}
        />

        {/* Weight Optimization */}
        <div className="flex items-center space-x-3">
          <input
            {...register('optimizeWeights')}
            type="checkbox"
            className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
          />
          <label className="text-sm font-medium">
            Optimize for Maximum Sharpe Ratio
          </label>
        </div>

        {/* Action Buttons */}
        <div className="flex space-x-4">
          <Button
            type="submit"
            loading={isCreating || isOptimizing}
            disabled={totalWeight !== 1 || tickers.some(t => t.trim() === '')}
            className="flex-1"
          >
            {isOptimizing ? 'Optimizing...' : 'Create Portfolio'}
          </Button>
          
          {onCancel && (
            <Button
              type="button"
              variant="secondary"
              onClick={onCancel}
            >
              Cancel
            </Button>
          )}
        </div>
      </form>
    </div>
  );
}; 