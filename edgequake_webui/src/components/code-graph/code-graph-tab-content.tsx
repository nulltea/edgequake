/**
 * Code Graph tab — Phase 2 of the Reference Code GraphRAG extension.
 *
 * For each approved `code_artifact` in the document, this tab fetches the
 * anchor-centric N-hop subgraph from the indexed reference repo and
 * renders it. Clicking a node opens the chunk content in a side drawer.
 *
 * Hidden until at least one reference-codebase index exists for the
 * document's approved repo — until indexing completes, the tab is a
 * non-actionable placeholder.
 */
"use client";

import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import {
  createReferenceCodebaseIndex,
  getAlgorithms,
  getCodeReferences,
  getDocumentRepos,
  getReferenceCodebaseGraph,
  indexRepo,
  listReferenceCodebaseIndexesForRepo,
} from "@/lib/api/edgequake";
import type { CodeArtifact } from "@/types/code-artifacts";
import type {
  ReferenceCodebaseGraphNode,
  ReferenceCodebaseIndex,
} from "@/types/reference-codebase";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlertTriangle, FileCode2, GitBranch, Loader2, RefreshCw } from "lucide-react";
import { useMemo, useState } from "react";
import { toast } from "sonner";
import { InlineMath } from "@/components/algorithms/inline-math";
import { CodeGraphRenderer } from "./code-graph-renderer";

interface CodeGraphTabContentProps {
  documentId: string;
}

export function CodeGraphTabContent({ documentId }: CodeGraphTabContentProps) {
  // Approved code_artifacts — these are the anchor list on the left.
  const { data: codeRefs, isLoading: loadingRefs } = useQuery({
    queryKey: ["code-references", documentId],
    queryFn: () => getCodeReferences(documentId),
    enabled: !!documentId,
    staleTime: 30 * 1000,
  });

  const approvedArtifacts: CodeArtifact[] = useMemo(
    () => (codeRefs?.candidates ?? []).filter((c) => c.status === "approved"),
    [codeRefs],
  );

  // Algorithm names for card headers.
  const { data: algorithms } = useQuery({
    queryKey: ["algorithms", documentId],
    queryFn: () => getAlgorithms(documentId),
    enabled: !!documentId,
    staleTime: 60 * 1000,
  });
  const algorithmNameById = useMemo(() => {
    const m = new Map<string, string>();
    for (const a of algorithms?.algorithms ?? []) m.set(a.id, a.name);
    return m;
  }, [algorithms]);

  // Detected repos — we need the document_repo_id for each approved
  // artifact to resolve to an index.
  const { data: repos } = useQuery({
    queryKey: ["document-repos", documentId],
    queryFn: () => getDocumentRepos(documentId),
    enabled: !!documentId,
    staleTime: 60 * 1000,
  });

  // Approved repos — used both when artifacts exist (look up url) and when
  // they don't (offer a no-anchors "Index this repo" path).
  const approvedRepos = useMemo(
    () => (repos?.candidates ?? []).filter((r) => r.status === "approved"),
    [repos],
  );

  // Repo ids we need indexes for: union of (repos artifacts point at) and
  // (approved repos overall). The second set is what makes the no-anchors
  // flow possible — without it we couldn't show index status or wire an
  // Index button for docs that have a repo but no algorithms.
  const repoIdsToFetch = useMemo(() => {
    const ids = new Set<string>();
    for (const a of approvedArtifacts) ids.add(a.document_repo_id);
    for (const r of approvedRepos) ids.add(r.id);
    return Array.from(ids);
  }, [approvedArtifacts, approvedRepos]);

  // Per-repo indexes. Poll while any index is in-flight.
  const { data: indexesByRepo, isLoading: loadingIndexes } = useQuery({
    queryKey: ["reference-codebase-indexes", documentId, repoIdsToFetch],
    queryFn: async () => {
      const entries = await Promise.all(
        repoIdsToFetch.map(async (rid) => {
          try {
            const { indexes } = await listReferenceCodebaseIndexesForRepo(rid);
            return [rid, indexes] as const;
          } catch {
            return [rid, [] as ReferenceCodebaseIndex[]] as const;
          }
        }),
      );
      return new Map(entries);
    },
    enabled: repoIdsToFetch.length > 0,
    staleTime: 60 * 1000,
    refetchInterval: (query) => {
      const m = query.state.data;
      if (!m) return false;
      for (const list of m.values()) {
        if (
          list.some((i) =>
            ["queued", "scanning", "parsing", "chunking", "embedding"].includes(
              i.status,
            ),
          )
        ) {
          return 5_000;
        }
      }
      return false;
    },
  });

  const [selectedArtifactId, setSelectedArtifactId] = useState<string | null>(
    null,
  );
  const [selectedNode, setSelectedNode] =
    useState<ReferenceCodebaseGraphNode | null>(null);
  // Default to 3 hops — 1 surfaces only direct neighbours which is
  // almost always too thin for a porting context; 3 typically shows
  // the callers-of-callers that matter.
  const [hops, setHops] = useState(3);
  // Whole-repo toggle: skip BFS, show top-N by degree + every edge
  // between them. Useful for "give me the lay of the land" before
  // drilling into an anchor.
  const [whole, setWhole] = useState(false);

  const selectedArtifact = useMemo(
    () => approvedArtifacts.find((a) => a.id === selectedArtifactId) ?? null,
    [approvedArtifacts, selectedArtifactId],
  );

  // Pick the first completed index for the artifact's repo. Failed /
  // in-flight rows fall through and show a banner.
  const selectedIndex: ReferenceCodebaseIndex | null = useMemo(() => {
    if (!selectedArtifact) return null;
    const list = indexesByRepo?.get(selectedArtifact.document_repo_id);
    if (!list) return null;
    return list.find((i) => i.status === "complete") ?? list[0] ?? null;
  }, [selectedArtifact, indexesByRepo]);

  const {
    data: graph,
    isLoading: loadingGraph,
    isError: graphError,
    error: graphErrorObj,
  } = useQuery({
    queryKey: [
      "reference-codebase-graph",
      selectedIndex?.id,
      selectedArtifact?.id,
      hops,
      whole,
    ],
    queryFn: () =>
      getReferenceCodebaseGraph(selectedIndex!.id, {
        // Whole mode: no anchor, top-N by degree with a much higher
        // node cap so the full repo fits. Anchor mode: 200 is plenty
        // for a N-hop subgraph around one function.
        ...(whole
          ? { whole: true, max_nodes: 2000 }
          : {
              anchor_artifact_id: selectedArtifact?.id,
              hops,
              max_nodes: 500,
            }),
      }),
    enabled:
      !!selectedIndex &&
      selectedIndex.status === "complete" &&
      (whole || !!selectedArtifact),
    staleTime: 60 * 1000,
  });

  const repoUrlById = useMemo(() => {
    const map = new Map<string, string>();
    for (const r of repos?.candidates ?? []) map.set(r.id, r.url);
    return map;
  }, [repos]);

  const queryClient = useQueryClient();
  const reindexMutation = useMutation({
    mutationFn: (documentRepoId: string) =>
      createReferenceCodebaseIndex({
        document_repo_id: documentRepoId,
        mode: "algorithm_focused",
        force_reindex: true,
      }),
    onSuccess: (r) => {
      toast.success("Re-index queued", { description: `Track ID: ${r.track_id}` });
      queryClient.invalidateQueries({
        queryKey: ["reference-codebase-indexes", documentId],
      });
    },
    onError: (err) => {
      toast.error("Failed to queue re-index", {
        description: err instanceof Error ? err.message : "Unknown error",
      });
    },
  });

  // Doc-centric "Index this repo" trigger. Used in the no-anchors empty
  // state (and could be used elsewhere later). Posts to /repos/{id}/index;
  // the server auto-picks `full` mode when no approved code_artifacts
  // exist, sidestepping the `algorithm_focused requires artifacts`
  // rejection that blocks the auto-trigger path.
  const indexRepoMutation = useMutation({
    mutationFn: (repoId: string) => indexRepo(repoId, {}),
    onSuccess: (r) => {
      toast.success(`Indexing queued (${r.mode})`, {
        description: `Track ID: ${r.track_id}`,
      });
      queryClient.invalidateQueries({
        queryKey: ["reference-codebase-indexes", documentId],
      });
    },
    onError: (err) => {
      toast.error("Failed to queue indexing", {
        description: err instanceof Error ? err.message : "Unknown error",
      });
    },
  });

  // Auto-select the first anchor once data lands.
  if (!selectedArtifactId && approvedArtifacts.length > 0) {
    setSelectedArtifactId(approvedArtifacts[0]!.id);
  }

  if (loadingRefs || loadingIndexes) {
    return (
      <div className="p-4 space-y-3">
        <Skeleton className="h-6 w-40" />
        <Skeleton className="h-[400px] w-full" />
      </div>
    );
  }

  if (approvedArtifacts.length === 0) {
    // No anchors — but there may still be approved reference repos. In that
    // case the user can index a repo without algorithms by hitting the
    // doc-centric endpoint, which the server auto-picks `full` mode for.
    // Closes the "doc has repo but no algorithms" gap where this tab used
    // to be a dead end.
    if (approvedRepos.length > 0) {
      return (
        <NoAnchorsButReposState
          repos={approvedRepos}
          indexesByRepo={indexesByRepo}
          onIndex={(repoId) => indexRepoMutation.mutate(repoId)}
          pending={indexRepoMutation.isPending}
        />
      );
    }
    return (
      <EmptyState
        title="No approved code anchors"
        body="Approve a code match in the Code Matches tab to seed the reference-codebase graph, or approve a reference repo to enable whole-repo indexing."
      />
    );
  }

  const inFlight =
    selectedIndex &&
    ["queued", "scanning", "parsing", "chunking", "embedding"].includes(
      selectedIndex.status,
    );

  return (
    <div className="flex h-full overflow-hidden">
      {/* Left pane: anchor list */}
      <aside className="w-72 shrink-0 border-r overflow-y-auto p-3 space-y-2">
        <h3 className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-2">
          Approved anchors
        </h3>
        {approvedArtifacts.map((a) => (
          <button
            type="button"
            key={a.id}
            onClick={() => {
              setSelectedArtifactId(a.id);
              setSelectedNode(null);
            }}
            className={`w-full text-left rounded-md border p-2 text-xs transition hover:bg-muted ${
              a.id === selectedArtifactId
                ? "bg-muted border-primary"
                : "bg-background"
            }`}
          >
            <div className="font-medium truncate">
              {a.algorithm_id ? (
                <InlineMath
                  text={
                    algorithmNameById.get(a.algorithm_id) ?? "Unknown algorithm"
                  }
                />
              ) : (
                "Orphan match"
              )}
            </div>
            <div className="text-muted-foreground mt-1 truncate">
              <FileCode2 className="inline h-3 w-3 mr-1" />
              {a.file_path}:{a.start_line}-{a.end_line}
            </div>
          </button>
        ))}
      </aside>

      {/* Right pane: graph + drawer */}
      <section className="flex-1 flex flex-col overflow-hidden">
        <div className="flex items-center gap-3 p-3 border-b text-xs">
          <label className="flex items-center gap-1.5 cursor-pointer select-none">
            <input
              type="checkbox"
              checked={whole}
              onChange={(e) => setWhole(e.target.checked)}
              className="h-3.5 w-3.5"
            />
            <span>Whole repo</span>
          </label>
          <div className={`flex items-center gap-2 ${whole ? "opacity-40" : ""}`}>
            <GitBranch className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">Hops</span>
            <input
              type="range"
              min={1}
              // Slider upper bound = the indexed codebase's approximate
              // graph diameter, returned by the backend. Falls back to
              // 10 while loading or for indexes too small to compute a
              // diameter (e.g. a single-file repo with no edges).
              max={Math.max(1, selectedIndex?.max_depth ?? 10)}
              value={Math.min(hops, selectedIndex?.max_depth ?? 10)}
              onChange={(e) => setHops(parseInt(e.target.value, 10))}
              disabled={whole}
              className="w-24"
            />
            <span className="tabular-nums">
              {hops}
              {selectedIndex?.max_depth
                ? ` / ${selectedIndex.max_depth}`
                : ""}
            </span>
          </div>
          {selectedArtifact && (
            <div className="ml-auto flex items-center gap-2 text-muted-foreground">
              {selectedIndex ? (
                <span>
                  Index: <span className="font-mono">{selectedIndex.id.slice(0, 8)}</span>{" "}
                  · {selectedIndex.status}
                  {selectedIndex.symbol_count > 0 && (
                    <>
                      {" · "}
                      {selectedIndex.symbol_count} symbols ·{" "}
                      {selectedIndex.edge_count} edges
                    </>
                  )}
                </span>
              ) : (
                <span>No index yet for this repo</span>
              )}
              <Button
                variant="outline"
                size="sm"
                onClick={() =>
                  reindexMutation.mutate(selectedArtifact.document_repo_id)
                }
                disabled={reindexMutation.isPending}
                title={
                  selectedIndex
                    ? "Re-run indexing with force_reindex=true"
                    : "Build the reference-codebase index for this repo"
                }
              >
                {reindexMutation.isPending ? (
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                ) : (
                  <RefreshCw className="h-3.5 w-3.5" />
                )}
                {selectedIndex ? "Re-index" : "Index"}
              </Button>
            </div>
          )}
        </div>

        {/* Index-in-flight banner is replaced by the in-graph loading card
            below (see Building code graph block) — that one is more visible
            while the index runs and keeps the control bar uncluttered. */}
        {selectedIndex?.status === "failed" && (
          <InlineBanner
            tone="error"
            text={`Indexing failed: ${selectedIndex.error_message ?? "unknown error"}`}
          />
        )}
        {graph?.truncated && (
          <InlineBanner
            tone="warn"
            text={`Truncated at ${graph.nodes.length} nodes — lower hops or pick a smaller anchor.`}
          />
        )}
        {/* TODO(Chunk 6 follow-up): parse-error banner — needs a
            total_parse_errors aggregate on GET /indexes/{id} so we can
            surface "tree-sitter failed on N% of files". The per-file
            `reference_codebase_files.parse_errors` column is already
            populated; a server-side SUM(parse_errors) join into the
            index response DTO is the only missing piece. */}

        <div className="flex-1 overflow-hidden p-3">
          {inFlight ? (
            <div className="h-full flex flex-col items-center justify-center rounded-lg border border-dashed">
              <Loader2 className="h-8 w-8 animate-spin text-primary mb-3" />
              <p className="text-sm font-medium">Building code graph</p>
              <p className="text-xs text-muted-foreground text-center max-w-sm mt-1">
                Status: {selectedIndex!.status} — the graph will render as
                soon as indexing completes.
              </p>
            </div>
          ) : loadingGraph ? (
            <Skeleton className="h-full w-full" />
          ) : graphError ? (
            <div className="rounded-md border border-destructive/30 bg-destructive/5 p-4 text-sm">
              Failed to load graph:{" "}
              {graphErrorObj instanceof Error
                ? graphErrorObj.message
                : "unknown error"}
            </div>
          ) : graph && graph.nodes.length > 0 ? (
            <CodeGraphRenderer
              nodes={graph.nodes}
              edges={graph.edges}
              onNodeClick={setSelectedNode}
            />
          ) : !selectedIndex && selectedArtifact ? (
            <div className="h-full flex flex-col items-center justify-center rounded-lg border border-dashed text-center px-6">
              <FileCode2 className="h-8 w-8 text-muted-foreground mb-3" />
              <p className="text-sm font-medium">No code graph yet</p>
              <p className="text-xs text-muted-foreground max-w-sm mt-1">
                Click <span className="font-medium">Index</span> above to build
                the reference-codebase index for this repo. This clones the
                repo, parses the code, and embeds chunks — usually a few
                minutes for a typical repository.
              </p>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-sm text-muted-foreground">
              {whole
                ? "Empty index — try toggling whole-repo off and selecting an anchor."
                : "Select an anchor to load its subgraph, or toggle \"Whole repo\"."}
            </div>
          )}
        </div>

        {selectedNode && selectedIndex && (
          <NodeDetailsDrawer
            node={selectedNode}
            repoUrl={repoUrlById.get(selectedArtifact?.document_repo_id ?? "")}
            repoCommit={selectedIndex.repo_commit}
            onClose={() => setSelectedNode(null)}
          />
        )}
      </section>
    </div>
  );
}

function EmptyState({ title, body }: { title: string; body: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 px-4 rounded-lg border border-dashed m-4">
      <div className="rounded-full bg-muted p-3 mb-3">
        <FileCode2 className="h-6 w-6 text-muted-foreground" />
      </div>
      <p className="text-sm font-medium">{title}</p>
      <p className="text-xs text-muted-foreground text-center max-w-sm mt-1">
        {body}
      </p>
    </div>
  );
}

/**
 * Shown when a doc has approved reference repos but no approved code
 * artifacts. Surfaces one row per repo with an "Index" button (auto-picks
 * `full` mode on the server) plus the current status of any existing
 * indexes for that repo. Polling for in-flight status is handled by the
 * parent's `indexesByRepo` query.
 */
function NoAnchorsButReposState({
  repos,
  indexesByRepo,
  onIndex,
  pending,
}: {
  repos: Array<{ id: string; url: string; owner: string; repo: string }>;
  indexesByRepo: Map<string, ReferenceCodebaseIndex[]> | undefined;
  onIndex: (repoId: string) => void;
  pending: boolean;
}) {
  return (
    <div className="m-4 space-y-4">
      <div className="rounded-lg border border-dashed p-6 text-center">
        <div className="inline-flex rounded-full bg-muted p-3 mb-3">
          <FileCode2 className="h-6 w-6 text-muted-foreground" />
        </div>
        <p className="text-sm font-medium">No approved code anchors</p>
        <p className="text-xs text-muted-foreground max-w-md mx-auto mt-1">
          This document has approved reference repos but no approved code
          matches yet. Index a repo directly to build a whole-repo code graph,
          or approve code matches in the Code Matches tab for anchor-focused
          indexing.
        </p>
      </div>

      <div className="rounded-lg border">
        <div className="border-b px-4 py-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Approved repos
        </div>
        <ul className="divide-y">
          {repos.map((r) => {
            const indexes = indexesByRepo?.get(r.id) ?? [];
            const latest =
              indexes.find((i) => i.status === "complete") ?? indexes[0];
            const inFlight =
              latest &&
              ["queued", "scanning", "parsing", "chunking", "embedding"].includes(
                latest.status,
              );
            return (
              <li
                key={r.id}
                className="flex items-center gap-3 px-4 py-3 text-sm"
              >
                <GitBranch className="h-4 w-4 text-muted-foreground shrink-0" />
                <div className="min-w-0 flex-1">
                  <div className="font-medium truncate">
                    {r.owner}/{r.repo}
                  </div>
                  <div className="text-xs text-muted-foreground truncate">
                    {latest ? (
                      <>
                        Index{" "}
                        <span className="font-mono">
                          {latest.id.slice(0, 8)}
                        </span>{" "}
                        · {latest.status}
                        {latest.symbol_count > 0 && (
                          <>
                            {" · "}
                            {latest.symbol_count} symbols ·{" "}
                            {latest.edge_count} edges
                          </>
                        )}
                      </>
                    ) : (
                      "No index yet"
                    )}
                  </div>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => onIndex(r.id)}
                  disabled={pending || inFlight}
                  title={
                    inFlight
                      ? "Indexing already in progress for this repo"
                      : latest
                        ? "Build a new index (server auto-picks 'full' when no anchors exist)"
                        : "Index this repo — server auto-picks 'full' mode when no anchors exist"
                  }
                >
                  {pending || inFlight ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <RefreshCw className="h-3.5 w-3.5" />
                  )}
                  {latest ? "Re-index" : "Index"}
                </Button>
              </li>
            );
          })}
        </ul>
      </div>
    </div>
  );
}

function InlineBanner({
  tone,
  text,
}: {
  tone: "info" | "warn" | "error";
  text: string;
}) {
  const style =
    tone === "error"
      ? "border-destructive/30 bg-destructive/5 text-destructive"
      : tone === "warn"
        ? "border-amber-500/30 bg-amber-50 text-amber-900 dark:bg-amber-900/20 dark:text-amber-200"
        : "border-muted bg-muted/50";
  return (
    <div
      className={`mx-3 mt-3 rounded-md border px-3 py-2 text-xs flex items-start gap-2 ${style}`}
    >
      {tone !== "info" && <AlertTriangle className="h-4 w-4 shrink-0 mt-0.5" />}
      <span>{text}</span>
    </div>
  );
}

function NodeDetailsDrawer({
  node,
  repoUrl,
  repoCommit,
  onClose,
}: {
  node: ReferenceCodebaseGraphNode;
  repoUrl: string | undefined;
  repoCommit: string;
  onClose: () => void;
}) {
  // The graph endpoint gives us chunk_id; the chunk content would
  // require a follow-up fetch. Phase 2 v1 skips fetching the body and
  // just shows metadata + a GitHub link so the user can jump out.
  const githubUrl =
    repoUrl && repoCommit
      ? `${repoUrl}/blob/${repoCommit}/${node.file_path}#L${node.start_line}-L${node.end_line}`
      : null;

  return (
    <div className="border-t p-4 bg-muted/30">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="font-mono text-sm font-semibold">
            {node.name}
            {node.is_anchor && (
              <span className="ml-2 rounded bg-primary/15 text-primary px-1.5 py-0.5 text-xs">
                anchor
              </span>
            )}
          </div>
          <div className="text-xs text-muted-foreground mt-1">
            {node.kind} · {node.language} · depth {node.depth} ·{" "}
            <span className="font-mono">
              {node.file_path}:{node.start_line}-{node.end_line}
            </span>
          </div>
        </div>
        <Button variant="ghost" size="sm" onClick={onClose}>
          Close
        </Button>
      </div>
      {githubUrl && (
        <a
          href={githubUrl}
          target="_blank"
          rel="noreferrer"
          className="inline-flex items-center gap-1 mt-3 text-xs text-primary hover:underline"
        >
          <RefreshCw className="h-3 w-3" />
          Open in GitHub (at pinned commit)
        </a>
      )}
    </div>
  );
}
