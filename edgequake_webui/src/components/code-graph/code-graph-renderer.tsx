/**
 * CodeGraphRenderer — sigma.js wrapper for the Phase 2 reference-codebase
 * subgraph endpoint.
 *
 * Deliberately separate from `components/graph/graph-renderer.tsx`
 * (which is tuned for paper knowledge-graph entity types + community
 * clustering) — code graphs colour by language and highlight anchor
 * symbols; reusing the entity renderer would mean bolting on kind-aware
 * overrides for every signal it already consumes.
 *
 * Layout: circular seed placement + ForceAtlas2 settle so anchors stay
 * central. Kept small — the whole point of the anchor-centric graph is
 * that N is ~10-100 nodes, not 5000.
 */
"use client";

import type {
  ReferenceCodebaseGraphEdge,
  ReferenceCodebaseGraphNode,
} from "@/types/reference-codebase";
import Graph from "graphology";
import circular from "graphology-layout/circular";
import forceAtlas2 from "graphology-layout-forceatlas2";
import { useTheme } from "next-themes";
import { useEffect, useRef } from "react";
import Sigma from "sigma";

interface CodeGraphRendererProps {
  nodes: ReferenceCodebaseGraphNode[];
  edges: ReferenceCodebaseGraphEdge[];
  /** Fires with the clicked node's payload; drives the chunk drawer. */
  onNodeClick?: (node: ReferenceCodebaseGraphNode) => void;
}

// One hue per language; fall back to slate for unknowns. Anchor symbols
// get a brighter variant and a border to stand out from neighbours.
const LANGUAGE_COLOR: Record<string, string> = {
  rust: "#dea584",
  python: "#3572a5",
  typescript: "#2b7489",
  javascript: "#f1e05a",
  go: "#00add8",
  c: "#555555",
  cpp: "#f34b7d",
  text: "#64748b",
};

function colorForNode(node: ReferenceCodebaseGraphNode): string {
  const base =
    LANGUAGE_COLOR[node.language.toLowerCase()] ?? LANGUAGE_COLOR.text!;
  // Anchor → keep base, bright. Non-anchor → desaturate by mixing with a
  // neutral. sigma doesn't ship a mixer; a simple alpha trick works.
  if (node.is_anchor) return base;
  return base + "aa"; // 67% alpha
}

function edgeStyle(kind: string): { color: string; type: string } {
  // kind → visual style. Dashed for cross-file / imports, solid for
  // intra-module edges. Distinct hues keep them readable in mixed graphs.
  switch (kind) {
    case "imports":
      return { color: "#94a3b8", type: "dashed" };
    case "calls":
      return { color: "#2563eb", type: "arrow" };
    case "defines":
      return { color: "#10b981", type: "arrow" };
    case "references":
      return { color: "#a855f7", type: "arrow" };
    case "implements":
      return { color: "#f97316", type: "arrow" };
    case "inherits":
      return { color: "#ef4444", type: "arrow" };
    default:
      return { color: "#64748b", type: "arrow" };
  }
}

export function CodeGraphRenderer({
  nodes,
  edges,
  onNodeClick,
}: CodeGraphRendererProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const sigmaRef = useRef<Sigma | null>(null);
  const { resolvedTheme } = useTheme();

  useEffect(() => {
    if (!containerRef.current || nodes.length === 0) return;

    const graph = new Graph();
    for (const n of nodes) {
      graph.addNode(n.symbol_id, {
        label: n.name,
        size: n.is_anchor ? 14 : 8 + Math.min(n.algorithm_focus * 6, 6),
        color: colorForNode(n),
        borderColor: n.is_anchor ? "#111827" : undefined,
        // Stash the full payload so node-click handlers have context.
        payload: n,
      });
    }
    for (const e of edges) {
      if (!graph.hasNode(e.source_symbol_id) || !graph.hasNode(e.target_symbol_id)) {
        continue;
      }
      const { color, type } = edgeStyle(e.kind);
      const key = `${e.source_symbol_id}→${e.target_symbol_id}:${e.kind}`;
      if (graph.hasEdge(key)) continue;
      graph.addEdgeWithKey(key, e.source_symbol_id, e.target_symbol_id, {
        label: e.kind,
        size: 1.5,
        color,
        type: type === "dashed" ? "line" : "arrow",
      });
    }

    circular.assign(graph);
    forceAtlas2.assign(graph, {
      iterations: 150,
      settings: {
        gravity: 1.2,
        scalingRatio: 6,
        slowDown: 5,
        barnesHutOptimize: graph.order > 50,
      },
    });

    const sigma = new Sigma(graph, containerRef.current, {
      labelSize: 11,
      labelWeight: "500",
      labelColor: {
        color: resolvedTheme === "dark" ? "#e2e8f0" : "#374151",
      },
      renderLabels: true,
      defaultEdgeType: "arrow",
    });
    sigmaRef.current = sigma;

    sigma.on("clickNode", ({ node }) => {
      const payload = graph.getNodeAttribute(node, "payload") as
        | ReferenceCodebaseGraphNode
        | undefined;
      if (payload && onNodeClick) onNodeClick(payload);
    });

    return () => {
      sigma.kill();
      sigmaRef.current = null;
    };
  }, [nodes, edges, onNodeClick, resolvedTheme]);

  return (
    <div
      ref={containerRef}
      className="h-full w-full rounded-md border bg-background"
      style={{ minHeight: 400 }}
    />
  );
}
