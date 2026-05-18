'use client';

import { Button } from '@/components/ui/button';
import { Skeleton } from '@/components/ui/skeleton';
import {
  analyzeCodeReference,
  approveAllCodeReferences,
  deleteCodeArtifact,
  getAlgorithms,
  getDocumentRepos,
  getCodeReferences,
  reviewCodeArtifact,
  submitCodeReferences,
} from '@/lib/api/edgequake';
import type {
  CodeArtifact,
  CodeReferenceListResponse,
  CodeReferenceRun,
} from '@/types/code-artifacts';
import type { RepoCandidate } from '@/types/document-repos';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import {
  CheckCheck,
  CheckCircle,
  FileCode2,
  Loader2,
  RefreshCw,
  Send,
} from 'lucide-react';
import { useCallback, useMemo, useState } from 'react';
import { toast } from 'sonner';
import { InlineMath } from '@/components/algorithms/inline-math';
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

  // Submission lock: once the user has submitted, hide the submit panel
  // until a re-analysis repopulates pending rows (handled below).
  const [submitted, setSubmitted] = useState(() => {
    if (typeof window === 'undefined') return false;
    return (
      localStorage.getItem(`code-refs-submitted-${documentId}`) === 'true'
    );
  });

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

  // Group candidates by algorithm_id. Orphan candidates (algorithm_id === null,
  // e.g. after document reprocessing re-created algorithms under new UUIDs)
  // are collected under the sentinel key "__orphan__" so the UI can render
  // them in a dedicated "Orphan matches" section.
  const ORPHAN_KEY = '__orphan__';
  const grouped = useMemo(() => {
    const by: Map<string, CodeArtifact[]> = new Map();
    for (const c of data?.candidates ?? []) {
      const key = c.algorithm_id ?? ORPHAN_KEY;
      const arr = by.get(key) ?? [];
      arr.push(c);
      by.set(key, arr);
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
    // Optimistic update: flip the row's status in the cache before the
    // request returns. Without this, the badge + action buttons don't
    // update until the next refetch — which earlier could take ~30s
    // because of the query's staleTime, so users had to hit the browser
    // refresh to see their click take effect.
    onMutate: async ({ id, status }) => {
      await queryClient.cancelQueries({
        queryKey: ['code-references', documentId],
      });
      const previous = queryClient.getQueryData<CodeReferenceListResponse>([
        'code-references',
        documentId,
      ]);
      if (previous) {
        queryClient.setQueryData<CodeReferenceListResponse>(
          ['code-references', documentId],
          {
            ...previous,
            candidates: previous.candidates.map((c) =>
              c.id === id ? { ...c, status } : c,
            ),
          },
        );
      }
      return { previous };
    },
    onError: (err, _vars, ctx) => {
      if (ctx?.previous) {
        queryClient.setQueryData(
          ['code-references', documentId],
          ctx.previous,
        );
      }
      toast.error('Failed to update status', {
        description: err instanceof Error ? err.message : 'Unknown error',
      });
    },
    onSettled: () => {
      // Always reconcile with the server — covers the case where the
      // backend's graph/embedding sync surfaces information we want
      // (e.g. updated counts in `runs`).
      queryClient.invalidateQueries({
        queryKey: ['code-references', documentId],
      });
    },
  });

  const deleteMutation = useMutation({
    mutationFn: (id: string) => deleteCodeArtifact(id),
    onMutate: async (id: string) => {
      // Optimistically remove the card from the list so the UI feels
      // instant. If the request fails, we restore from `previous`.
      await queryClient.cancelQueries({
        queryKey: ['code-references', documentId],
      });
      const previous = queryClient.getQueryData<CodeReferenceListResponse>([
        'code-references',
        documentId,
      ]);
      if (previous) {
        queryClient.setQueryData<CodeReferenceListResponse>(
          ['code-references', documentId],
          {
            ...previous,
            candidates: previous.candidates.filter((c) => c.id !== id),
          },
        );
      }
      return { previous };
    },
    onError: (err, _id, ctx) => {
      if (ctx?.previous) {
        queryClient.setQueryData(
          ['code-references', documentId],
          ctx.previous,
        );
      }
      toast.error('Failed to delete match', {
        description: err instanceof Error ? err.message : 'Unknown error',
      });
    },
    onSettled: () => {
      queryClient.invalidateQueries({
        queryKey: ['code-references', documentId],
      });
    },
  });

  const approveAllMutation = useMutation({
    mutationFn: () => approveAllCodeReferences(documentId),
    onMutate: async () => {
      await queryClient.cancelQueries({
        queryKey: ['code-references', documentId],
      });
      const previous = queryClient.getQueryData<CodeReferenceListResponse>([
        'code-references',
        documentId,
      ]);
      if (previous) {
        queryClient.setQueryData<CodeReferenceListResponse>(
          ['code-references', documentId],
          {
            ...previous,
            candidates: previous.candidates.map((c) =>
              c.status === 'pending' ? { ...c, status: 'approved' as const } : c,
            ),
          },
        );
      }
      return { previous };
    },
    onSuccess: (result) => {
      if (result.sync_failures > 0) {
        toast.warning(
          `Approved ${result.approved_count}, ${result.sync_failures} sync warning${result.sync_failures === 1 ? '' : 's'}`,
          {
            description:
              'Graph/embedding sync had partial failures — check server logs.',
          },
        );
      } else if (result.approved_count > 0) {
        toast.success(
          `Approved ${result.approved_count} match${result.approved_count === 1 ? '' : 'es'}`,
        );
      } else {
        toast.info('Nothing to approve — no pending matches.');
      }
    },
    onError: (err, _vars, ctx) => {
      if (ctx?.previous) {
        queryClient.setQueryData(
          ['code-references', documentId],
          ctx.previous,
        );
      }
      toast.error('Failed to approve all', {
        description: err instanceof Error ? err.message : 'Unknown error',
      });
    },
    onSettled: () => {
      queryClient.invalidateQueries({
        queryKey: ['code-references', documentId],
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

  const submitMutation = useMutation({
    mutationFn: () => submitCodeReferences(documentId),
    onSuccess: (result) => {
      if (result.status === 'no_approved_matches') {
        toast.success('Submitted — no approved matches to index', {
          description: `${result.rejected_count} rejected.`,
        });
      } else if (result.status === 'auto_index_disabled') {
        toast.success(
          `Submitted: ${result.approved_count} approved, ${result.rejected_count} rejected`,
          {
            description:
              'Auto-index is off — run POST /reference-codebase/indexes or the Re-index button per repo to build the graph.',
          },
        );
      } else {
        toast.success(
          `Submitted: ${result.approved_count} approved, ${result.rejected_count} rejected`,
          {
            description: `${result.indexes_queued} code-graph index${result.indexes_queued === 1 ? '' : 'es'} queued.`,
          },
        );
      }
      setSubmitted(true);
      localStorage.setItem(`code-refs-submitted-${documentId}`, 'true');
      queryClient.invalidateQueries({
        queryKey: ['code-references', documentId],
      });
      queryClient.invalidateQueries({
        queryKey: ['reference-codebase-indexes', documentId],
      });
    },
    onError: (err) => {
      toast.error('Failed to submit code matches', {
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
  const handleDelete = useCallback(
    (id: string) => deleteMutation.mutate(id),
    [deleteMutation],
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

  // Pick the active run's phase, if any, so the loading card can show a
  // slightly more specific status ("cloning" / "analyzing" / "embedding").
  const ACTIVE_STATUSES = ['queued', 'cloning', 'analyzing', 'embedding'] as const;
  const activeRun = runs.find((r) =>
    (ACTIVE_STATUSES as readonly string[]).includes(r.status),
  );
  const analyzing = !!activeRun || analyzeMutation.isPending;

  const approvedRepos = (repos?.candidates ?? []).filter(
    (r) => r.status === 'approved',
  );

  // Submit-panel bookkeeping. Same semantics as algorithm-list.tsx:
  // enabled only when every candidate is approved or rejected AND at
  // least one is approved. If new pending rows appear after a prior
  // submission (e.g. the user hit "Re-analyze"), reset the locked-out
  // state so the panel becomes actionable again.
  const pendingCount = candidates.filter((c) => c.status === 'pending').length;
  const approvedCount = candidates.filter(
    (c) => c.status === 'approved',
  ).length;
  const allReviewed = candidates.length > 0 && pendingCount === 0;
  if (pendingCount > 0 && submitted) {
    setSubmitted(false);
    localStorage.removeItem(`code-refs-submitted-${documentId}`);
  }

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
          {pendingCount > 0 && (
            <Button
              size="sm"
              variant="outline"
              className="text-green-700 dark:text-green-400 hover:bg-green-500/10"
              onClick={() => approveAllMutation.mutate()}
              disabled={approveAllMutation.isPending}
              title={`Approve all ${pendingCount} pending match${pendingCount === 1 ? '' : 'es'}`}
            >
              {approveAllMutation.isPending ? (
                <Loader2 className="h-4 w-4 animate-spin" />
              ) : (
                <CheckCheck className="h-4 w-4" />
              )}
              Approve all ({pendingCount})
            </Button>
          )}
          {approvedRepos.map((r) => (
            <Button
              key={r.id}
              size="sm"
              variant="outline"
              onClick={() => analyzeMutation.mutate(r.id)}
              disabled={analyzeMutation.isPending}
              title={`Re-analyze ${r.owner}/${r.repo} — clears existing matches first`}
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

      {/* Loading state — a CodeReferenceAnalysis task is in flight */}
      {empty && analyzing && (
        <div className="flex flex-col items-center justify-center py-12 px-4 rounded-lg border border-dashed">
          <Loader2 className="h-8 w-8 animate-spin text-primary mb-3" />
          <p className="text-sm font-medium">Analyzing repository code</p>
          <p className="text-xs text-muted-foreground text-center max-w-sm mt-1">
            {activeRun
              ? `Status: ${activeRun.status} — matches will appear here as they are found.`
              : 'Queueing analysis — matches will appear here as they are found.'}
          </p>
        </div>
      )}

      {/* Empty state — no active analysis */}
      {empty && !analyzing && (
        <div className="flex flex-col items-center justify-center py-12 px-4 rounded-lg border border-dashed">
          <div className="rounded-full bg-muted p-3 mb-3">
            <FileCode2 className="h-6 w-6 text-muted-foreground" />
          </div>
          <p className="text-sm text-muted-foreground text-center max-w-sm">
            {approvedRepos.length === 0
              ? 'Approve a reference repository in the References tab first, then analysis runs automatically.'
              : 'No code matches yet. Click the re-analyze button above to run again.'}
          </p>
        </div>
      )}

      {/* Groups */}
      {!empty && (
        <div className="space-y-6">
          {Array.from(grouped.entries()).map(([algoId, items]) => {
            const isOrphan = algoId === ORPHAN_KEY;
            const heading = isOrphan
              ? 'Orphan matches'
              : (algorithmNameById.get(algoId) ?? 'Unknown algorithm');
            const subtitle = isOrphan
              ? 'Algorithm was re-extracted; re-link or reject below'
              : null;
            return (
              <div key={algoId}>
                <h3 className="text-sm font-semibold mb-2 flex items-baseline gap-1.5 flex-wrap">
                  {isOrphan ? (
                    heading
                  ) : (
                    <InlineMath text={heading} />
                  )}
                  <span className="text-muted-foreground font-normal">
                    ({items.length} match{items.length === 1 ? '' : 'es'})
                  </span>
                </h3>
                {subtitle && (
                  <p className="text-xs text-muted-foreground mb-2">{subtitle}</p>
                )}
                <div className="space-y-3">
                  {items.map((c) => (
                    <CodeArtifactCard
                      key={c.id}
                      artifact={c}
                      repoUrl={repoUrlById.get(c.document_repo_id)}
                      onApprove={handleApprove}
                      onReject={handleReject}
                      onDelete={handleDelete}
                      busy={
                        (reviewMutation.isPending &&
                          reviewMutation.variables?.id === c.id) ||
                        (deleteMutation.isPending &&
                          deleteMutation.variables === c.id)
                      }
                    />
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Submit panel — mirrors algorithms-list. Hidden after a
          successful submission until new pending rows appear. */}
      {!empty && !submitted && (
        <div className="flex items-center justify-between pt-3 border-t">
          <div className="text-sm text-muted-foreground">
            {submitMutation.isPending ? (
              <span className="flex items-center gap-1.5">
                <Loader2 className="h-4 w-4 animate-spin" />
                Queueing code-graph indexing...
              </span>
            ) : allReviewed ? (
              <span className="flex items-center gap-1.5">
                <CheckCircle className="h-4 w-4 text-green-600" />
                All matches reviewed — {approvedCount} approved,{' '}
                {candidates.length - approvedCount} rejected
              </span>
            ) : (
              <span>{pendingCount} match(es) still pending review</span>
            )}
          </div>
          <Button
            onClick={() => submitMutation.mutate()}
            disabled={
              !allReviewed || approvedCount === 0 || submitMutation.isPending
            }
            size="sm"
          >
            {submitMutation.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
            {submitMutation.isPending
              ? 'Processing...'
              : `Submit (${approvedCount})`}
          </Button>
        </div>
      )}
    </div>
  );
}
