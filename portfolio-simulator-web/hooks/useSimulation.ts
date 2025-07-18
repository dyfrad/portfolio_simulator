import { useState, useCallback, useEffect } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { SimulationConfig, SimulationResults } from '@/lib/types';
import { ApiService, handleApiError } from '@/lib/api';

export const useSimulation = () => {
  const queryClient = useQueryClient();
  const [progress, setProgress] = useState(0);
  const [isRunning, setIsRunning] = useState(false);

  // Run simulation mutation
  const runSimulationMutation = useMutation({
    mutationFn: ApiService.runSimulation,
    onSuccess: (results) => {
      setIsRunning(false);
      setProgress(100);
      // Invalidate simulations list
      queryClient.invalidateQueries({ queryKey: ['simulations'] });
      return results;
    },
    onError: (error) => {
      setIsRunning(false);
      setProgress(0);
      console.error('Error running simulation:', handleApiError(error));
    },
  });

  // Get user simulations
  const {
    data: simulations = [],
    isLoading: isLoadingSimulations,
    error: simulationsError,
    refetch: refetchSimulations,
  } = useQuery({
    queryKey: ['simulations'],
    queryFn: ApiService.getUserSimulations,
    enabled: true, // Will be controlled by auth state
  });

  // Run simulation with progress tracking
  const runSimulation = useCallback(async (config: SimulationConfig) => {
    setIsRunning(true);
    setProgress(0);
    
    try {
      const results = await runSimulationMutation.mutateAsync(config);
      
      // Start progress polling
      const pollProgress = async () => {
        if (results.simulationId) {
          try {
            const progressData = await ApiService.getSimulationProgress(results.simulationId);
            setProgress(progressData.progress * 100);
            
            if (progressData.status === 'completed') {
              setProgress(100);
              setIsRunning(false);
              return;
            }
            
            // Continue polling
            setTimeout(pollProgress, 1000);
          } catch (error) {
            console.error('Error polling progress:', error);
          }
        }
      };
      
      pollProgress();
      
      return results;
    } catch (error) {
      setIsRunning(false);
      setProgress(0);
      throw error;
    }
  }, [runSimulationMutation]);

  // Generate report mutation
  const generateReportMutation = useMutation({
    mutationFn: ApiService.generateReport,
    onError: (error) => {
      console.error('Error generating report:', handleApiError(error));
    },
  });

  // Download report mutation
  const downloadReportMutation = useMutation({
    mutationFn: ApiService.downloadReport,
    onError: (error) => {
      console.error('Error downloading report:', handleApiError(error));
    },
  });

  // Generate report
  const generateReport = useCallback(async (simulationId: string) => {
    return generateReportMutation.mutateAsync(simulationId);
  }, [generateReportMutation]);

  // Download report
  const downloadReport = useCallback(async (reportId: string) => {
    const blob = await downloadReportMutation.mutateAsync(reportId);
    
    // Create download link
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `portfolio-simulation-report-${reportId}.pdf`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    window.URL.revokeObjectURL(url);
  }, [downloadReportMutation]);

  return {
    // Simulation state
    isRunning,
    progress,
    results: runSimulationMutation.data,
    
    // Data
    simulations,
    isLoadingSimulations,
    simulationsError,
    
    // Actions
    runSimulation,
    generateReport,
    downloadReport,
    refetchSimulations,
    
    // Mutation states
    isGeneratingReport: generateReportMutation.isPending,
    isDownloadingReport: downloadReportMutation.isPending,
    
    // Errors
    simulationError: runSimulationMutation.error,
    generateReportError: generateReportMutation.error,
    downloadReportError: downloadReportMutation.error,
  };
};

// Hook for a single simulation
export const useSimulationById = (id: string | null) => {
  const {
    data: simulation,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['simulation', id],
    queryFn: () => ApiService.getSimulation(id!),
    enabled: !!id,
  });

  return {
    simulation,
    isLoading,
    error,
    refetch,
  };
}; 