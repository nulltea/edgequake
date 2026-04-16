/**
 * Query tool — the primary tool for agents to retrieve knowledge.
 *
 * Uses context_only mode: retrieves relevant chunks, entities, and
 * relationships from the knowledge graph WITHOUT generating an answer.
 * The calling model synthesizes the answer from the returned context.
 * This saves one LLM call per query.
 */
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { getClient } from "../client.js";
import { formatError } from "../errors.js";

export function registerQueryTools(server: McpServer): void {
  server.tool(
    "query",
    "Search the EdgeQuake knowledge graph. Returns retrieved text chunks, entities, and relationships. Use the returned context to answer the user's question. Use 'hybrid' mode (default) for best results.",
    {
      query: z.string().describe("Natural language question"),
      mode: z
        .enum(["naive", "local", "global", "hybrid", "mix"])
        .optional()
        .describe(
          "Query mode: naive (vector-only), local (entity graph), global (community search), hybrid (local+global, default), mix (weighted blend)",
        ),
    },
    async (params) => {
      try {
        const client = await getClient();
        const result = await client.query.execute({
          query: params.query,
          mode: params.mode,
          context_only: true,
        });

        const chunks: string[] = [];
        const entities: string[] = [];
        const relationships: string[] = [];
        const sourceDocs = new Set<string>();

        for (const s of result.sources) {
          if (s.source_type === "chunk" && s.snippet) {
            const doc = s.file_path || s.document_id || "";
            chunks.push(`[${doc}]: ${s.snippet}`);
            if (doc) sourceDocs.add(doc);
          } else if (s.source_type === "entity" && s.snippet) {
            entities.push(`- ${s.id}: ${s.snippet}`);
          } else if (s.source_type === "relationship" && s.snippet) {
            relationships.push(`- ${s.snippet}`);
          }
        }

        const parts: string[] = [];
        if (chunks.length > 0) {
          parts.push("**Text chunks:**\n" + chunks.slice(0, 10).join("\n\n"));
        }
        if (entities.length > 0) {
          parts.push("**Entities:**\n" + entities.slice(0, 15).join("\n"));
        }
        if (relationships.length > 0) {
          parts.push(
            "**Relationships:**\n" + relationships.slice(0, 10).join("\n"),
          );
        }
        if (sourceDocs.size > 0) {
          parts.push("**Source documents:** " + [...sourceDocs].join(", "));
        }

        const text =
          parts.length > 0
            ? parts.join("\n\n")
            : "No relevant context found in the knowledge base.";

        return {
          content: [{ type: "text" as const, text }],
        };
      } catch (error) {
        return formatError(error);
      }
    },
  );
}
