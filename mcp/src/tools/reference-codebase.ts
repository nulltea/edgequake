/**
 * Reference-codebase tools — Phase 2 of the Reference Code GraphRAG
 * extension. Surfaces the separate code RAG for agents that are porting
 * or integrating a paper's reference implementation.
 *
 * Two tools in one file because they share DTOs, client stub, and
 * env-bound tenant/workspace scope — splitting them would be a two-file
 * diff where one does. Tenant/workspace come from the MCP server's env
 * singleton; do not expose them as tool arguments (multi-tenant leak).
 */

import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { getClient } from "../client.js";
import { formatError } from "../errors.js";

export function registerReferenceCodebaseTools(server: McpServer): void {
  // ──────────────── query_code ────────────────
  //
  // Natural-language search over approved code chunks from a paper's
  // reference implementation repo. Not for finding papers — for
  // pulling up the actual function bodies when the user is porting,
  // integrating, or validating an implementation.
  server.tool(
    "query_code",
    [
      "Semantic search over reference-implementation code snippets that were approved during paper review.",
      "",
      "Use this when the user is porting, integrating, or validating a reference implementation of an algorithm described in a paper — e.g. 'how does the capacity-constrained cluster rebalancing work in this repo?' or 'find the attention kernel'.",
      "",
      "Returns code chunks (file_path, line range, symbol name, language, content) ranked by embedding similarity. Each chunk is anchored to an approved review decision, so results are human-validated implementations, not arbitrary repo text.",
      "",
      "Do NOT use this for paper-prose questions — use `query` for that.",
    ].join("\n"),
    {
      query: z
        .string()
        .describe("Natural-language description of what to find in code"),
      document_repo_id: z
        .string()
        .uuid()
        .optional()
        .describe("Restrict to a specific paper's repo (UUID)"),
      index_id: z
        .string()
        .uuid()
        .optional()
        .describe(
          "Restrict to a specific index row (tighter than document_repo_id)",
        ),
      algorithm_ids: z
        .array(z.string().uuid())
        .optional()
        .describe(
          "Restrict to chunks anchored to these approved algorithm ids",
        ),
      limit: z
        .number()
        .int()
        .positive()
        .max(50)
        .optional()
        .describe("Max hits (default 12)"),
      max_distance: z
        .number()
        .positive()
        .optional()
        .describe("Cosine-distance ceiling. Lower = stricter. Default 0.65"),
    },
    async (params) => {
      try {
        const client = await getClient();
        const result = await client.referenceCodebase.query({
          query: params.query,
          document_repo_id: params.document_repo_id,
          index_id: params.index_id,
          algorithm_ids: params.algorithm_ids,
          limit: params.limit,
          max_distance: params.max_distance,
        });

        if (result.coding_context.length === 0) {
          return {
            content: [
              {
                type: "text" as const,
                text: "No approved code snippets matched. Try a more specific query, broaden `max_distance`, or confirm the paper's repo has been indexed.",
              },
            ],
          };
        }

        // Render as Markdown with one fenced block per hit. The
        // `{repo_url}@{commit}` citation + `L{start}-L{end}` line range
        // lets the agent quote-cite exactly the code it saw.
        const parts: string[] = [
          `**${result.coding_context.length} reference-code hit(s):**`,
        ];
        for (const hit of result.coding_context) {
          const header = [
            `**${hit.symbol_name ?? hit.file_path}**`,
            `\`${hit.file_path}:${hit.start_line}-${hit.end_line}\``,
            `(${hit.language}, ${hit.chunk_kind}, distance ${hit.cosine_distance.toFixed(3)})`,
          ].join(" — ");
          const citation =
            hit.repo_url && hit.repo_commit
              ? `\n[source](${hit.repo_url}/blob/${hit.repo_commit}/${hit.file_path}#L${hit.start_line}-L${hit.end_line})`
              : "";
          parts.push(
            `${header}${citation}\n\n\`\`\`${hit.language}\n${hit.content}\n\`\`\``,
          );
        }

        return {
          content: [{ type: "text" as const, text: parts.join("\n\n") }],
        };
      } catch (error) {
        return formatError(error);
      }
    },
  );

  // ──────────────── get_symbol_neighborhood ────────────────
  //
  // Walk the symbol graph out from a known anchor. Useful when the
  // agent has a symbol name (from `query_code` results or the user
  // mentioning a function) and wants callers/callees/imports for
  // porting context.
  server.tool(
    "get_symbol_neighborhood",
    [
      "Walk the symbol graph of an indexed reference codebase outward from a seed symbol.",
      "",
      "Returns the N-hop subgraph: nodes (functions/classes/structs with file+line info and chunk ids), edges (defines/calls/imports/references/implements/inherits). Anchor nodes that overlap an approved paper algorithm are flagged with `is_anchor=true` — use those as natural entry points.",
      "",
      "Use this AFTER `query_code` when the user asks 'what calls this' or 'what does this depend on' — it's strictly for navigating the code graph, not searching prose.",
      "",
      "Seed resolution: pass `anchor_artifact_id` (an approved code_artifact UUID) or `anchor_symbol` (a symbol name, exact or case-insensitive). Omit both to fall back to top-degree symbols — only useful on `mode='full'` indexes when you have no better entry point.",
    ].join("\n"),
    {
      index_id: z
        .string()
        .uuid()
        .describe("Reference-codebase index UUID (from `query_code` results)"),
      anchor_symbol: z
        .string()
        .optional()
        .describe("Symbol name to seed from (e.g. 'rebalance_clusters')"),
      anchor_artifact_id: z
        .string()
        .uuid()
        .optional()
        .describe("Approved code_artifact UUID to seed from"),
      hops: z
        .number()
        .int()
        .min(1)
        .max(3)
        .optional()
        .describe("BFS depth. Default 1, max 3."),
      max_nodes: z
        .number()
        .int()
        .positive()
        .max(1000)
        .optional()
        .describe("Hard cap on returned node count. Default 200."),
    },
    async (params) => {
      try {
        const client = await getClient();
        const graph = await client.referenceCodebase.graph(params.index_id, {
          anchor_artifact_id: params.anchor_artifact_id,
          anchor_symbol: params.anchor_symbol,
          hops: params.hops,
          max_nodes: params.max_nodes,
        });

        if (graph.nodes.length === 0) {
          return {
            content: [
              {
                type: "text" as const,
                text: "No symbols found in neighborhood.",
              },
            ],
          };
        }

        // Group edges by source for a readable summary: for each node,
        // list its outgoing edges. This maps to the "what does this
        // call?" and "what does this import?" questions the agent is
        // actually asking.
        const outgoing = new Map<string, typeof graph.edges>();
        for (const e of graph.edges) {
          const list = outgoing.get(e.source_symbol_id) ?? [];
          list.push(e);
          outgoing.set(e.source_symbol_id, list);
        }

        const lines: string[] = [
          `Seed: ${graph.seed_symbol_ids.length} symbol(s), walked ${graph.hops} hop(s)${graph.truncated ? ` — truncated at ${graph.nodes.length} nodes` : ""}.`,
          `Repo: ${graph.repo_url}@${graph.repo_commit}`,
          "",
        ];
        for (const n of graph.nodes) {
          const anchor = n.is_anchor ? " ⭐" : "";
          lines.push(
            `**${n.name}** [${n.kind}]${anchor} — \`${n.file_path}:${n.start_line}-${n.end_line}\` (depth ${n.depth})`,
          );
          const outs = outgoing.get(n.symbol_id) ?? [];
          for (const e of outs) {
            const tgt = graph.nodes.find((m) => m.symbol_id === e.target_symbol_id);
            const label = tgt ? tgt.name : (e.target_name ?? "<external>");
            lines.push(`  ↳ ${e.kind}: ${label}`);
          }
        }

        return {
          content: [{ type: "text" as const, text: lines.join("\n") }],
        };
      } catch (error) {
        return formatError(error);
      }
    },
  );
}
