'use client';

import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { deleteAlgorithm, getAlgorithms, reviewAlgorithm, submitAlgorithms } from '@/lib/api/edgequake';
import type { Algorithm } from '@/types/algorithms';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { CheckCircle, Loader2, Send } from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { toast } from 'sonner';
import { AlgorithmCard } from './algorithm-card';

type StatusFilter = 'all' | 'pending' | 'approved' | 'rejected';

interface AlgorithmListProps {
  documentId: string;
}

const STATUS_FILTERS: { label: string; value: StatusFilter }[] = [
  { label: 'All', value: 'all' },
  { label: 'Pending', value: 'pending' },
  { label: 'Approved', value: 'approved' },
  { label: 'Rejected', value: 'rejected' },
];

export function AlgorithmList({ documentId }: AlgorithmListProps) {
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');
  const [submitted, setSubmitted] = useState(() => {
    if (typeof window === 'undefined') return false;
    return localStorage.getItem(`algo-submitted-${documentId}`) === 'true';
  });
  const queryClient = useQueryClient();

  const { data, isLoading, isError, error } = useQuery({
    queryKey: ['algorithms', documentId],
    queryFn: () => getAlgorithms(documentId),
    enabled: !!documentId,
    staleTime: 30 * 1000,
  });

  const reviewMutation = useMutation({
    mutationFn: ({ algorithmId, status }: { algorithmId: string; status: 'approved' | 'rejected' }) =>
      reviewAlgorithm(algorithmId, status),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['algorithms', documentId] });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (algorithmId: string) => deleteAlgorithm(algorithmId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['algorithms', documentId] });
      toast.success('Algorithm deleted');
    },
    onError: (error) => {
      toast.error('Failed to delete algorithm', {
        description: error instanceof Error ? error.message : 'Unknown error',
      });
    },
  });

  const submitMutation = useMutation({
    mutationFn: () => submitAlgorithms(documentId),
    onSuccess: (result) => {
      toast.success(
        `Submitted: ${result.approved_count} approved, ${result.rejected_count} rejected`,
      );
      setSubmitted(true);
      localStorage.setItem(`algo-submitted-${documentId}`, 'true');
      queryClient.invalidateQueries({ queryKey: ['algorithms', documentId] });
      queryClient.invalidateQueries({ queryKey: ['documents'] });
    },
    onError: (error) => {
      toast.error('Failed to submit algorithms', {
        description: error instanceof Error ? error.message : 'Unknown error',
      });
    },
  });

  const handleApprove = useCallback(
    (id: string) => reviewMutation.mutate({ algorithmId: id, status: 'approved' }),
    [reviewMutation],
  );

  const handleReject = useCallback(
    (id: string) => reviewMutation.mutate({ algorithmId: id, status: 'rejected' }),
    [reviewMutation],
  );

  const handleDelete = useCallback(
    (id: string) => {
      if (window.confirm('Delete this algorithm? This cannot be undone.')) {
        deleteMutation.mutate(id);
      }
    },
    [deleteMutation],
  );

  const handleApproveAll = useCallback(() => {
    const pending = data?.algorithms.filter((a) => a.status === 'pending') ?? [];
    if (pending.length === 0) return;
    pending.forEach((a) => reviewMutation.mutate({ algorithmId: a.id, status: 'approved' }));
  }, [data, reviewMutation]);

  const handleRejectAll = useCallback(() => {
    const pending = data?.algorithms.filter((a) => a.status === 'pending') ?? [];
    if (pending.length === 0) return;
    pending.forEach((a) => reviewMutation.mutate({ algorithmId: a.id, status: 'rejected' }));
  }, [data, reviewMutation]);

  const filtered = useMemo<Algorithm[]>(() => {
    if (!data?.algorithms) return [];
    if (statusFilter === 'all') return data.algorithms;
    return data.algorithms.filter((a) => a.status === statusFilter);
  }, [data, statusFilter]);

  if (isLoading) {
    return (
      <div className="space-y-3">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-40 w-full" />
        <Skeleton className="h-40 w-full" />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-4 text-sm text-destructive">
        Failed to load algorithms: {(error as Error)?.message || 'Unknown error'}
      </div>
    );
  }

  if (!data?.algorithms || data.algorithms.length === 0) {
    return (
      <div className="text-center py-8 text-sm text-muted-foreground">
        No algorithms have been extracted from this document yet.
      </div>
    );
  }

  const pendingCount = data.algorithms.filter((a) => a.status === 'pending').length;
  const approvedCount = data.algorithms.filter((a) => a.status === 'approved').length;
  const allReviewed = pendingCount === 0;

  // Reset submitted flag if new pending algorithms appear (re-extraction)
  if (pendingCount > 0 && submitted) {
    setSubmitted(false);
    localStorage.removeItem(`algo-submitted-${documentId}`);
  }

  return (
    <div className="space-y-4">
      {/* Status filter bar + bulk actions */}
      <div className="flex items-center justify-between gap-2 flex-wrap">
        <div className="flex items-center gap-1">
          {STATUS_FILTERS.map((filter) => {
            const count =
              filter.value === 'all'
                ? data.algorithms.length
                : data.algorithms.filter((a) => a.status === filter.value).length;
            return (
              <button
                key={filter.value}
                onClick={() => setStatusFilter(filter.value)}
                className={`rounded-md px-3 py-1.5 text-xs font-medium transition-colors ${
                  statusFilter === filter.value
                    ? 'bg-primary text-primary-foreground'
                    : 'bg-muted text-muted-foreground hover:bg-muted/80'
                }`}
              >
                {filter.label} ({count})
              </button>
            );
          })}
        </div>
        <div className="flex items-center gap-1.5">
          {pendingCount > 0 && (
            <>
              <button
                onClick={handleApproveAll}
                disabled={reviewMutation.isPending}
                className="rounded-md px-3 py-1.5 text-xs font-medium bg-green-500/15 text-green-700 dark:text-green-400 hover:bg-green-500/25 transition-colors disabled:opacity-50"
              >
                Approve All ({pendingCount})
              </button>
              <button
                onClick={handleRejectAll}
                disabled={reviewMutation.isPending}
                className="rounded-md px-3 py-1.5 text-xs font-medium bg-red-500/15 text-red-700 dark:text-red-400 hover:bg-red-500/25 transition-colors disabled:opacity-50"
              >
                Reject All ({pendingCount})
              </button>
            </>
          )}
        </div>
      </div>

      {/* Algorithm cards */}
      <div className="space-y-3">
        {filtered.map((algorithm) => (
          <AlgorithmCard
            key={algorithm.id}
            algorithm={algorithm}
            onApprove={handleApprove}
            onReject={handleReject}
            onDelete={handleDelete}
          />
        ))}
      </div>

      {filtered.length === 0 && (
        <div className="text-center py-6 text-sm text-muted-foreground">
          No algorithms match the selected filter.
        </div>
      )}

      {/* Submit panel — hidden after successful submission */}
      {!submitted && (
        <div className="flex items-center justify-between pt-3 border-t">
          <div className="text-sm text-muted-foreground">
            {submitMutation.isPending ? (
              <span className="flex items-center gap-1.5">
                <Loader2 className="h-4 w-4 animate-spin" />
                Embedding algorithms...
              </span>
            ) : allReviewed ? (
              <span className="flex items-center gap-1.5">
                <CheckCircle className="h-4 w-4 text-green-600" />
                All algorithms reviewed — {approvedCount} approved, {data.algorithms.length - approvedCount} rejected
              </span>
            ) : (
              <span>{pendingCount} algorithm(s) still pending review</span>
            )}
          </div>
          <Button
            onClick={() => submitMutation.mutate()}
            disabled={!allReviewed || approvedCount === 0 || submitMutation.isPending}
            size="sm"
          >
            {submitMutation.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
            {submitMutation.isPending ? 'Processing...' : `Submit (${approvedCount})`}
          </Button>
        </div>
      )}
    </div>
  );
}
