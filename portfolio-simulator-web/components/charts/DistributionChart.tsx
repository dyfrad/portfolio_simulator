'use client';

import React, { useState, useEffect } from 'react';
import dynamic from 'next/dynamic';
import { SimulationResults } from '@/lib/types';
import { formatCurrency } from '@/lib/utils';

const Plot = dynamic(() => import('react-plotly.js'), { 
  ssr: false,
  loading: () => <div className="h-96 bg-gray-100 animate-pulse rounded-lg flex items-center justify-center">
    <div className="text-gray-500">Loading chart...</div>
  </div>
});

interface DistributionChartProps {
  results: SimulationResults;
  horizon: number;
  className?: string;
}

export const DistributionChart: React.FC<DistributionChartProps> = ({
  results,
  horizon,
  className,
}) => {
  const [isClient, setIsClient] = useState(false);

  useEffect(() => {
    setIsClient(true);
  }, []);

  if (!isClient) {
    return (
      <div className={`bg-white p-6 rounded-lg shadow ${className || ''}`}>
        <div className="h-96 bg-gray-100 animate-pulse rounded-lg flex items-center justify-center">
          <div className="text-gray-500">Loading chart...</div>
        </div>
      </div>
    );
  }
  const data = [
    {
      x: results.finalValues,
      type: 'histogram' as const,
      nbinsx: 50,
      marker: {
        color: 'rgba(59, 130, 246, 0.7)',
        line: {
          color: 'rgba(59, 130, 246, 1)',
          width: 1,
        },
      },
      name: 'Final Portfolio Values',
    },
  ];

  const layout = {
    title: {
      text: `Portfolio Value Distribution (${horizon} years)`,
      font: { size: 18 },
    },
    xaxis: {
      title: 'Final Portfolio Value (€)',
      tickformat: ',.0f',
      tickprefix: '€',
    },
    yaxis: {
      title: 'Frequency',
    },
    showlegend: false,
    margin: { t: 60, r: 50, b: 60, l: 60 },
    plot_bgcolor: 'rgba(0,0,0,0)',
    paper_bgcolor: 'rgba(0,0,0,0)',
    font: {
      family: 'Inter, system-ui, sans-serif',
    },
  };

  const config = {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
    displaylogo: false,
  };

  return (
    <div className={`bg-white p-6 rounded-lg shadow ${className || ''}`}>
      <Plot
        data={data}
        layout={layout}
        config={config}
        style={{ width: '100%', height: '400px' }}
        useResizeHandler={true}
      />
      
      {/* Summary Statistics */}
      <div className="mt-4 grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
        <div>
          <div className="text-sm text-gray-600">Mean</div>
          <div className="text-lg font-semibold text-green-600">
            {formatCurrency(results.meanFinalValue)}
          </div>
        </div>
        <div>
          <div className="text-sm text-gray-600">Median</div>
          <div className="text-lg font-semibold text-blue-600">
            {formatCurrency(results.medianFinalValue)}
          </div>
        </div>
        <div>
          <div className="text-sm text-gray-600">5th Percentile</div>
          <div className="text-lg font-semibold text-orange-600">
            {formatCurrency(results.percentile5)}
          </div>
        </div>
        <div>
          <div className="text-sm text-gray-600">95th Percentile</div>
          <div className="text-lg font-semibold text-purple-600">
            {formatCurrency(results.percentile95)}
          </div>
        </div>
      </div>
    </div>
  );
}; 