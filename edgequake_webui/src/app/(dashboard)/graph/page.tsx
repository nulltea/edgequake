/**
 * @module GraphPage
 * @description Knowledge graph visualization page route.
 *
 * @implements FEAT0601 - Interactive graph visualization
 * @see GraphViewer component for full implementation
 */
'use client';

import { GraphLoadingOverlay } from '@/components/graph/graph-loading-overlay';
import { useGraphStore } from '@/stores/use-graph-store';
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
  const { setSearchQuery, setStartNode, setDocumentId, nodes } = useGraphStore();

  // Handle URL parameters for deep linking from query results
  useEffect(() => {
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
