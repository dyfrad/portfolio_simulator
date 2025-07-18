import { useState, useCallback } from 'react';
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { Portfolio, PortfolioCreate } from '@/lib/types';
import { ApiService, handleApiError } from '@/lib/api';

export const usePortfolio = () => {
  const queryClient = useQueryClient();

  // Get user portfolios
  const {
    data: portfolios = [],
    isLoading: isLoadingPortfolios,
    error: portfoliosError,
    refetch: refetchPortfolios,
  } = useQuery({
    queryKey: ['portfolios'],
    queryFn: ApiService.getUserPortfolios,
    enabled: true, // Will be controlled by auth state
  });

  // Create portfolio mutation
  const createPortfolioMutation = useMutation({
    mutationFn: ApiService.createPortfolio,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['portfolios'] });
    },
    onError: (error) => {
      console.error('Error creating portfolio:', handleApiError(error));
    },
  });

  // Update portfolio mutation
  const updatePortfolioMutation = useMutation({
    mutationFn: ({ id, portfolio }: { id: string; portfolio: Partial<PortfolioCreate> }) =>
      ApiService.updatePortfolio(id, portfolio),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['portfolios'] });
    },
    onError: (error) => {
      console.error('Error updating portfolio:', handleApiError(error));
    },
  });

  // Delete portfolio mutation
  const deletePortfolioMutation = useMutation({
    mutationFn: ApiService.deletePortfolio,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['portfolios'] });
    },
    onError: (error) => {
      console.error('Error deleting portfolio:', handleApiError(error));
    },
  });

  // Upload CSV mutation
  const uploadCsvMutation = useMutation({
    mutationFn: ApiService.uploadPortfolioCsv,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['portfolios'] });
    },
    onError: (error) => {
      console.error('Error uploading CSV:', handleApiError(error));
    },
  });

  // Create portfolio
  const createPortfolio = useCallback(async (portfolio: PortfolioCreate) => {
    return createPortfolioMutation.mutateAsync(portfolio);
  }, [createPortfolioMutation]);

  // Update portfolio
  const updatePortfolio = useCallback(async (id: string, portfolio: Partial<PortfolioCreate>) => {
    return updatePortfolioMutation.mutateAsync({ id, portfolio });
  }, [updatePortfolioMutation]);

  // Delete portfolio
  const deletePortfolio = useCallback(async (id: string) => {
    return deletePortfolioMutation.mutateAsync(id);
  }, [deletePortfolioMutation]);

  // Upload CSV
  const uploadCsv = useCallback(async (file: File) => {
    return uploadCsvMutation.mutateAsync(file);
  }, [uploadCsvMutation]);

  return {
    // Data
    portfolios,
    isLoadingPortfolios,
    portfoliosError,
    
    // Mutations
    createPortfolio,
    updatePortfolio,
    deletePortfolio,
    uploadCsv,
    
    // Mutation states
    isCreating: createPortfolioMutation.isPending,
    isUpdating: updatePortfolioMutation.isPending,
    isDeleting: deletePortfolioMutation.isPending,
    isUploading: uploadCsvMutation.isPending,
    
    // Errors
    createError: createPortfolioMutation.error,
    updateError: updatePortfolioMutation.error,
    deleteError: deletePortfolioMutation.error,
    uploadError: uploadCsvMutation.error,
    
    // Actions
    refetchPortfolios,
  };
};

// Hook for a single portfolio
export const usePortfolioById = (id: string | null) => {
  const {
    data: portfolio,
    isLoading,
    error,
    refetch,
  } = useQuery({
    queryKey: ['portfolio', id],
    queryFn: () => ApiService.getPortfolio(id!),
    enabled: !!id,
  });

  return {
    portfolio,
    isLoading,
    error,
    refetch,
  };
}; 