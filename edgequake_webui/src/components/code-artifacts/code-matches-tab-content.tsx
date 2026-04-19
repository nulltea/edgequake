'use client';

import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import {
  analyzeCodeReference,
  getAlgorithms,
  getDocumentRepos,
  getCodeReferences,
  reviewCodeArtifact,
} from '@/lib/api/edgequake';
import type {
  CodeArtifact,
  CodeReferenceRun,
} from '@/types/code-artifacts';
import type { RepoCandidate } from '@/types/document-repos';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { FileCode2, Loader2, RefreshCw } from 'lucide-react';
import { useCallback, useMemo } from 'react';
import { toast } from 'sonner';
import { CodeArtifactCard } from './code-artifact-card';

interface CodeMatchesTabContentProps {
  documentId: string;
}

function runSummary(runs: CodeReferenceRun[]): string {
  if (runs.length === 0) {
    return 'Approve a reference repository in the References tab to run code analysis.';
  }
  if (runs.some((r) => ['queued', 'cloning', 'analyzing', 'embedding'].includes(r.status))) {
    return 'Analyzing — refresh in a moment.';
  }
  const failed = runs.filter((r) => r.status === 'failed');
  if (failed.length > 0) {
    return failed[0]!.error_message ?? 'Last analysis failed.';
  }
  const totalFindings = runs.reduce((n, r) => n + r.finding_count, 0);
  if (totalFindings === 0) {
    return 'Analysis complete — no code matches found for the approved algorithms.';
  }
  return `Analysis complete — ${totalFindings} candidate match${totalFindings === 1 ? '' : 'es'} across ${runs.length} repo${runs.length === 1 ? '' : 's'}.`;
}

export function CodeMatchesTabContent({
  documentId,
}: CodeMatchesTabContentProps) {
  const queryClient = useQueryClient();

  const { data, isLoading, isError, error, refetch, isFetching } = useQuery({
    queryKey: ['code-references', documentId],
    queryFn: () => getCodeReferences(documentId),
    enabled: !!documentId,
    staleTime: 30 * 1000,
    refetchInterval: (query) => {
      const d = query.state.data;
      if (!d) return false;
      return d.runs.some((r) =>
        ['queued', 'cloning', 'analyzing', 'embedding'].includes(r.status),
      )
        ? 3_000
        : false;
    },
  });

  // Algorithm names for group headers — mirror the list the document detail page fetches.
  const { data: algorithms } = useQuery({
    queryKey: ['algorithms', documentId],
    queryFn: () => getAlgorithms(documentId),
    enabled: !!documentId,
    staleTime: 60 * 1000,
  });

  // Repo URLs keyed by document_repo_id — for the "view on GitHub" link-out.
  const { data: repos } = useQuery({
    queryKey: ['document-repos', documentId],
    queryFn: () => getDocumentRepos(documentId),
    enabled: !!documentId,
    staleTime: 60 * 1000,
  });

  const repoUrlById = useMemo(() => {
    const map = new Map<string, string>();
    for (const r of (repos?.candidates ?? []) as RepoCandidate[]) {
      map.set(r.id, r.url);
    }
    return map;
  }, [repos]);

  const algorithmNameById = useMemo(() => {
    const map = new Map<string, string>();
    for (const a of algorithms?.algorithms ?? []) {
      map.set(a.id, a.name);
    }
    return map;
  }, [algorithms]);

  // Group candidates by algorithm_id.
  const grouped = useMemo(() => {
    const by: Map<string, CodeArtifact[]> = new Map();
    for (const c of data?.candidates ?? []) {
      const arr = by.get(c.algorithm_id) ?? [];
      arr.push(c);
      by.set(c.algorithm_id, arr);
    }
    return by;
  }, [data]);

  const reviewMutation = useMutation({
    mutationFn: ({
      id,
      status,
    }: {
      id: string;
      status: 'approved' | 'rejected';
    }) => reviewCodeArtifact(id, status),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: ['code-references', documentId],
      });
    },
    onError: (err) => {
      toast.error('Failed to update status', {
        description: err instanceof Error ? err.message : 'Unknown error',
      });
    },
  });

  const analyzeMutation = useMutation({
    mutationFn: (documentRepoId: string) =>
      analyzeCodeReference(documentRepoId),
    onSuccess: (r) => {
      toast.success('Analysis queued', {
        description: `Track ID: ${r.track_id}`,
      });
      queryClient.invalidateQueries({
        queryKey: ['code-references', documentId],
      });
    },
    onError: (err) => {
      toast.error('Failed to queue analysis', {
        description: err instanceof Error ? err.message : 'Unknown error',
      });
    },
  });

  const handleApprove = useCallback(
    (id: string) => reviewMutation.mutate({ id, status: 'approved' }),
    [reviewMutation],
  );
  const handleReject = useCallback(
    (id: string) => reviewMutation.mutate({ id, status: 'rejected' }),
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
          Failed to load code matches:{' '}
          {error instanceof Error ? error.message : 'Unknown error'}
        </div>
      </div>
    );
  }

  const candidates = data?.candidates ?? [];
  const runs = data?.runs ?? [];
  const summary = runSummary(runs);
  const empty = candidates.length === 0;

  const approvedRepos = (repos?.candidates ?? []).filter(
    (r) => r.status === 'approved',
  );

  return (
    <div className="p-4 space-y-4 overflow-auto h-full">
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h2 className="text-sm font-semibold text-muted-foreground flex items-center gap-1.5">
            <FileCode2 className="h-4 w-4" />
            {candidates.length > 0
              ? `${candidates.length} code-match candidate${candidates.length === 1 ? '' : 's'}`
              : 'Code matches'}
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
          {approvedRepos.map((r) => (
            <Button
              key={r.id}
              size="sm"
              variant="outline"
              onClick={() => analyzeMutation.mutate(r.id)}
              disabled={analyzeMutation.isPending}
              title={`Re-analyze ${r.owner}/${r.repo}`}
            >
              {analyzeMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <RefreshCw className="h-4 w-4" />
              )}
              {r.owner}/{r.repo}
            </Button>
          ))}
        </div>
      </div>

      {/* Empty state */}
      {empty && (
        <div className="flex flex-col items-center justify-center py-12 px-4 rounded-lg border border-dashed">
          <div className="rounded-full bg-muted p-3 mb-3">
            <FileCode2 className="h-6 w-6 text-muted-foreground" />
          </div>
          <p className="text-sm text-muted-foreground text-center max-w-sm">
            {approvedRepos.length === 0
              ? 'Approve a reference repository in the References tab first, then analysis runs automatically.'
              : 'Analysis runs automatically after repo approval. Check back in a moment, or click the re-analyze button above.'}
          </p>
        </div>
      )}

      {/* Groups */}
      {!empty && (
        <div className="space-y-6">
          {Array.from(grouped.entries()).map(([algoId, items]) => (
            <div key={algoId}>
              <h3 className="text-sm font-semibold mb-2">
                {algorithmNameById.get(algoId) ?? 'Unknown algorithm'}{' '}
                <span className="text-muted-foreground font-normal">
                  ({items.length} match{items.length === 1 ? '' : 'es'})
                </span>
              </h3>
              <div className="space-y-3">
                {items.map((c) => (
                  <CodeArtifactCard
                    key={c.id}
                    artifact={c}
                    repoUrl={repoUrlById.get(c.document_repo_id)}
                    onApprove={handleApprove}
                    onReject={handleReject}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
