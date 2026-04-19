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

  // Distinct repo ids the approved artifacts point at.
  const distinctRepoIds = useMemo(
    () => Array.from(new Set(approvedArtifacts.map((a) => a.document_repo_id))),
    [approvedArtifacts],
  );

  // Per-repo indexes. Poll while any index is in-flight.
  const { data: indexesByRepo, isLoading: loadingIndexes } = useQuery({
    queryKey: ["reference-codebase-indexes", documentId, distinctRepoIds],
    queryFn: async () => {
      const entries = await Promise.all(
        distinctRepoIds.map(async (rid) => {
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
    enabled: distinctRepoIds.length > 0,
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
  const [hops, setHops] = useState(1);

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
    ],
    queryFn: () =>
      getReferenceCodebaseGraph(selectedIndex!.id, {
        anchor_artifact_id: selectedArtifact!.id,
        hops,
        max_nodes: 200,
      }),
    enabled: !!selectedIndex && !!selectedArtifact && selectedIndex.status === "complete",
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
    return (
      <EmptyState
        title="No approved code anchors"
        body="Approve a code match in the Code Matches tab to seed the reference-codebase graph."
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
              {algorithmNameById.get(a.algorithm_id) ?? "Unknown algorithm"}
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
          <div className="flex items-center gap-2">
            <GitBranch className="h-4 w-4 text-muted-foreground" />
            <span className="text-muted-foreground">Hops</span>
            <input
              type="range"
              min={1}
              max={3}
              value={hops}
              onChange={(e) => setHops(parseInt(e.target.value, 10))}
              className="w-24"
            />
            <span className="tabular-nums">{hops}</span>
          </div>
          {selectedIndex && (
            <div className="ml-auto flex items-center gap-2 text-muted-foreground">
              <span>
                Index: <span className="font-mono">{selectedIndex.id.slice(0, 8)}</span>{" "}
                · {selectedIndex.status}
                {selectedIndex.symbol_count > 0 && (
                  <>
                    {" · "}
                    {selectedIndex.symbol_count} symbols · {selectedIndex.edge_count}{" "}
                    edges
                  </>
                )}
              </span>
              {selectedArtifact && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() =>
                    reindexMutation.mutate(selectedArtifact.document_repo_id)
                  }
                  disabled={reindexMutation.isPending}
                  title="Re-run indexing with force_reindex=true"
                >
                  {reindexMutation.isPending ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  ) : (
                    <RefreshCw className="h-3.5 w-3.5" />
                  )}
                  Re-index
                </Button>
              )}
            </div>
          )}
        </div>

        {inFlight && (
          <InlineBanner
            tone="info"
            text={`Index ${selectedIndex!.status} — graph will render once it completes.`}
          />
        )}
        {selectedIndex?.status === "failed" && (
          <InlineBanner
            tone="error"
            text={`Indexing failed: ${selectedIndex.error_message ?? "unknown error"}`}
          />
        )}
        {!selectedIndex && selectedArtifact && (
          <InlineBanner
            tone="info"
            text="No reference-codebase index for this repo yet. Approving the code match with auto-index enabled kicks one off; otherwise POST /api/v1/reference-codebase/indexes to build one."
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
          {loadingGraph ? (
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
          ) : (
            <div className="h-full flex items-center justify-center text-sm text-muted-foreground">
              Select an anchor to load its subgraph.
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
