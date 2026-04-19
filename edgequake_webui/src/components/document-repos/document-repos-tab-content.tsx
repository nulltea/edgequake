'use client';

import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import { detectRepos, getDocumentRepos, reviewRepo } from '@/lib/api/edgequake';
import type {
  DetectionRun,
  RepoCandidate,
} from '@/types/document-repos';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { GitBranch, Loader2, RefreshCw } from 'lucide-react';
import { useCallback } from 'react';
import { toast } from 'sonner';
import { RepoCard } from './repo-card';

interface DocumentReposTabContentProps {
  documentId: string;
}

function detectionRunSummary(run: DetectionRun | null | undefined): string {
  if (!run) {
    return 'Detection has not been run yet for this document.';
  }
  if (run.status === 'running') {
    return 'Detection is running — refresh in a moment.';
  }
  if (run.status === 'failed') {
    return (
      run.error_message ??
      'Detection failed. Try running it again — SearXNG or Crawl4AI may have been unavailable.'
    );
  }
  const a = run.layer_a_candidates;
  const b = run.layer_b_candidates;
  if (a === 0 && b === 0) {
    return 'Detection ran and found no reference repositories.';
  }
  const parts: string[] = [];
  if (a > 0) parts.push(`${a} from PDF links`);
  if (b > 0) parts.push(`${b} from web search`);
  return `Detection complete — ${parts.join(' + ')}.`;
}

export function DocumentReposTabContent({
  documentId,
}: DocumentReposTabContentProps) {
  const queryClient = useQueryClient();

  const { data, isLoading, isError, error, refetch, isFetching } = useQuery({
    queryKey: ['document-repos', documentId],
    queryFn: () => getDocumentRepos(documentId),
    enabled: !!documentId,
    staleTime: 30 * 1000,
    // Auto-refresh while a run is in progress so the UI catches completion
    // without the user having to hit Refresh manually.
    refetchInterval: (query) => {
      const d = query.state.data;
      return d?.detection_run?.status === 'running' ? 3_000 : false;
    },
  });

  const reviewMutation = useMutation({
    mutationFn: ({
      repoId,
      status,
    }: {
      repoId: string;
      status: 'approved' | 'rejected';
    }) => reviewRepo(repoId, status),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['document-repos', documentId] });
    },
    onError: (err) => {
      toast.error('Failed to update status', {
        description: err instanceof Error ? err.message : 'Unknown error',
      });
    },
  });

  const detectMutation = useMutation({
    mutationFn: () => detectRepos(documentId),
    onSuccess: (r) => {
      toast.success('Detection queued', {
        description: `Track ID: ${r.track_id}`,
      });
      queryClient.invalidateQueries({ queryKey: ['document-repos', documentId] });
    },
    onError: (err) => {
      toast.error('Failed to queue detection', {
        description: err instanceof Error ? err.message : 'Unknown error',
      });
    },
  });

  const handleApprove = useCallback(
    (id: string) => reviewMutation.mutate({ repoId: id, status: 'approved' }),
    [reviewMutation],
  );
  const handleReject = useCallback(
    (id: string) => reviewMutation.mutate({ repoId: id, status: 'rejected' }),
    [reviewMutation],
  );

  if (isLoading) {
    return (
      <div className="p-4 space-y-3">
        <Skeleton className="h-6 w-48" />
        <Skeleton className="h-24 w-full" />
        <Skeleton className="h-24 w-full" />
      </div>
    );
  }

  if (isError) {
    return (
      <div className="p-4">
        <div className="rounded-lg border border-destructive/30 bg-destructive/5 p-4 text-sm">
          Failed to load reference repositories:{' '}
          {error instanceof Error ? error.message : 'Unknown error'}
        </div>
      </div>
    );
  }

  const candidates: RepoCandidate[] = data?.candidates ?? [];
  const run = data?.detection_run ?? null;
  const summary = detectionRunSummary(run);
  const empty = candidates.length === 0;

  return (
    <div className="p-4 space-y-4 overflow-auto h-full">
      {/* Header: summary + actions */}
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-muted-foreground flex items-center gap-1.5">
            <GitBranch className="h-4 w-4" />
            {candidates.length > 0
              ? `${candidates.length} reference repository candidate${candidates.length === 1 ? '' : 's'}`
              : 'Reference repositories'}
          </h2>
          <p className="text-xs text-muted-foreground mt-1">{summary}</p>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Button
            variant="outline"
            size="sm"
            onClick={() => refetch()}
            disabled={isFetching}
          >
            <RefreshCw
              className={`h-4 w-4 ${isFetching ? 'animate-spin' : ''}`}
            />
            Refresh
          </Button>
          <Button
            size="sm"
            onClick={() => detectMutation.mutate()}
            disabled={detectMutation.isPending || run?.status === 'running'}
          >
            {detectMutation.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <RefreshCw className="h-4 w-4" />
            )}
            {run ? 'Re-detect' : 'Detect'}
          </Button>
        </div>
      </div>

      {/* Empty state */}
      {empty && (
        <div className="flex flex-col items-center justify-center py-12 px-4 rounded-lg border border-dashed">
          <div className="rounded-full bg-muted p-3 mb-3">
            <GitBranch className="h-6 w-6 text-muted-foreground" />
          </div>
          <p className="text-sm text-muted-foreground text-center max-w-sm">
            No reference repositories detected for this document.
            {!run &&
              ' Detection runs automatically after PDF ingestion — or click "Detect" to run it now.'}
          </p>
        </div>
      )}

      {/* Candidates */}
      {!empty && (
        <div className="space-y-3">
          {candidates.map((c) => (
            <RepoCard
              key={c.id}
              repo={c}
              onApprove={handleApprove}
              onReject={handleReject}
            />
          ))}
        </div>
      )}
    </div>
  );
}
