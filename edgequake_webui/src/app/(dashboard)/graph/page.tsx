/**
 * @module GraphPage
 * @description Knowledge graph visualization page route.
 *
 * @implements FEAT0601 - Interactive graph visualization
 * @see GraphViewer component for full implementation
 */
'use client';

import { GraphLoadingOverlay } from '@/components/graph/graph-loading-overlay';
import { getSubgraph } from '@/lib/api/edgequake';
import { useGraphStore } from '@/stores/use-graph-store';
import { useTenantStore } from '@/stores/use-tenant-store';
import dynamic from 'next/dynamic';
import { useSearchParams } from 'next/navigation';
import { useEffect } from 'react';

/**
 * WHY: Both dynamic imports use GraphLoadingOverlay as their loading fallback.
 * Without this, the main content area is completely EMPTY during JS bundle loading
 * (GraphTourWrapper had no fallback → rendered null → blank screen for seconds).
 * The overlay provides immediate visual feedback from the moment the user navigates
 * to /graph, through bundle loading, through data fetching, to final graph render.
 */
const GraphLoadingFallback = () => (
  <div className="relative h-full w-full">
    <GraphLoadingOverlay visible={true} phase="Loading graph viewer..." />
  </div>
);

// Dynamic import for GraphViewer since it uses browser APIs (Sigma.js)
const GraphViewer = dynamic(
  () => import('@/components/graph/graph-viewer'),
  {
    ssr: false,
    loading: GraphLoadingFallback,
  }
);

// Dynamic import for tour wrapper (client-only)
// WHY: Previously had NO loading fallback → rendered null → empty main area
const GraphTourWrapper = dynamic(
  () => import('@/components/graph/graph-tour-wrapper'),
  {
    ssr: false,
    loading: GraphLoadingFallback,
  }
);

// Matches a canonical UUID (8-4-4-4-12 hex). We use this to recognise
// when the legacy `?entity=<uuid>` URL was actually meant as a document
// filter — entity *names* are never UUID-shaped, so the heuristic is
// safe.
const UUID_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

export default function GraphPage() {
  const searchParams = useSearchParams();
  const {
    setSearchQuery,
    setStartNode,
    setDocumentId,
    nodes,
    setGraph,
    setStartingSet,
    setQueryText,
    setLoading,
    setError,
  } = useGraphStore();

  // SPEC-006 P3 — handle `?start_nodes=A,B,C&depth=1&q=…` deep-links
  // from chat surfaces (OpenWebUI tool, future Claude integrations).
  // When present, this short-circuits the normal "load the global
  // graph" path: we POST `/graph/subgraph` with the parsed starting
  // set and inject the response directly. The starting-set IDs are
  // recorded in the store so the renderer can style them distinctly.
  useEffect(() => {
    const startNodesParam = searchParams.get('start_nodes');
    if (!startNodesParam) return;

    const ids = startNodesParam
      .split(',')
      .map((s) => {
        try {
          return decodeURIComponent(s);
        } catch {
          return s; // best-effort fallback if the segment was already decoded
        }
      })
      .map((s) => s.trim())
      .filter(Boolean);
    if (ids.length === 0) return;

    const depthRaw = searchParams.get('depth');
    // SPEC-006 P3 — default to depth=0 (seeds-only) so the user lands
    // on a clean, focused canvas showing just what they searched for.
    // Inter-seed edges still render (the backend's
    // get_edges_for_node_set fills them in). Further exploration is
    // explicit via right-click → "Expand Neighborhood" on a node.
    const depth = depthRaw ? Math.max(0, Math.min(2, parseInt(depthRaw, 10))) : 0;
    const q = searchParams.get('q');

    // Workspace propagation. Chat-surface deep-links carry the
    // tenant/workspace the entities came from so the canvas resolves
    // to the SAME workspace — otherwise the user's currently-selected
    // workspace wins, every seed fails the tenant filter, and the
    // canvas renders empty. We switch the tenant store synchronously
    // before the fetch so the API client picks up the override on the
    // very next call.
    const urlWorkspaceId = searchParams.get('workspace_id');
    const urlTenantId = searchParams.get('tenant_id');
    if (urlWorkspaceId) {
      const tenantStore = useTenantStore.getState();
      if (urlTenantId && tenantStore.selectedTenantId !== urlTenantId) {
        tenantStore.selectTenant(urlTenantId);
      }
      tenantStore.selectWorkspace(urlWorkspaceId);
    }

    setQueryText(q);
    setStartingSet(ids);
    setLoading(true);
    setError(null);

    let cancelled = false;
    getSubgraph({ start_nodes: ids, depth, max_nodes: 60 })
      .then((resp) => {
        if (cancelled) return;
        // Derive entity_types / relationship_types from the returned
        // nodes/edges so legend + filter pills work without an extra
        // round-trip. `/graph/subgraph` doesn't emit metadata directly
        // — the response is intentionally lean.
        const entityTypes = Array.from(
          new Set(resp.nodes.map((n) => n.node_type).filter(Boolean)),
        );
        const relationshipTypes = Array.from(
          new Set(resp.edges.map((e) => e.relationship_type).filter(Boolean)),
        );
        setGraph({
          nodes: resp.nodes,
          edges: resp.edges,
          metadata: {
            node_count: resp.stats.total_nodes,
            edge_count: resp.stats.total_edges,
            entity_types: entityTypes,
            relationship_types: relationshipTypes,
          },
          is_truncated: resp.stats.truncated,
          total_nodes: resp.stats.total_nodes,
          total_edges: resp.stats.total_edges,
        });
      })
      .catch((err) => {
        if (cancelled) return;
        setError(
          err instanceof Error
            ? err.message
            : 'Failed to load subgraph for deep-link',
        );
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });

    return () => {
      cancelled = true;
    };
    // We only want to react to URL changes — `setGraph` et al. are
    // store actions and stable across renders, so listing them would
    // just add noise.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams]);

  // Handle URL parameters for deep linking from query results
  useEffect(() => {
    // SPEC-006: skip the legacy URL-param flow when we are entering
    // via the new `start_nodes` deep-link — the dedicated effect above
    // already populated the canvas, and the legacy handler would
    // clobber it by setting a single startNode / search query.
    if (searchParams.get('start_nodes')) return;

    const entities = searchParams.get('entities');
    const focus = searchParams.get('focus');
    const entity = searchParams.get('entity');
    const documentIdParam = searchParams.get('document_id');

    // Document scoping. Canonical param is `?document_id=<uuid>`. We
    // also accept the legacy `?entity=<uuid>` shape — users had been
    // typing document UUIDs into `?entity=` hoping for a filter, and
    // the viewer silently ignored them. Treating a UUID-shaped `entity`
    // as a document_id matches that intuition without breaking the
    // entity-focus use case (entity names are not UUID-shaped).
    const resolvedDocumentId =
      documentIdParam ??
      (entity && UUID_RE.test(entity) ? entity : null);
    setDocumentId(resolvedDocumentId);

    // If entities filter is provided, set as search query
    if (entities) {
      // Use the first entity as a search filter
      const entityList = entities.split(',');
      if (entityList.length > 0) {
        setSearchQuery(entityList[0]);
      }
    }

    // If focus or entity is specified (and not a UUID swallowed by the
    // document filter above), try to set it as the start node.
    const targetEntity = focus || (entity && !UUID_RE.test(entity) ? entity : null);
    if (targetEntity && nodes.length > 0) {
      // Find matching node
      const matchingNode = nodes.find(
        n => n.label?.toLowerCase() === targetEntity.toLowerCase() ||
             n.id?.toLowerCase() === targetEntity.toLowerCase()
      );
      if (matchingNode) {
        setStartNode(matchingNode.id);
      }
    }
  }, [searchParams, setSearchQuery, setStartNode, setDocumentId, nodes]);
  
  return (
    <div className="h-full overflow-hidden">
      <GraphTourWrapper>
        <GraphViewer />
      </GraphTourWrapper>
    </div>
  );
}
