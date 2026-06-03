/**
 * Query tool — the primary tool for agents to retrieve knowledge.
 *
 * Uses context_only mode: retrieves relevant chunks, entities, and
 * relationships from the knowledge graph WITHOUT generating an answer.
 * The calling model synthesizes the answer from the returned context.
 * This saves one LLM call per query.
 *
 * In addition to chunks/entities/relationships, the response may include
 * two curated enrichment blocks:
 *   - `approved_algorithms` — structured algorithm definitions (steps +
 *     pseudocode) extracted from the paper.
 *   - `reference_code` — code implementations of paper algorithms,
 *     discovered in linked reference repositories.
 * Both are rendered inline so the chat model can cite them directly.
 * (The JSON field name `approved_algorithms` is the backend contract —
 * the consumer-facing section header avoids the workflow term.)
 */
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import type { ApprovedAlgorithm, ReferenceCodeSnippet } from "edgequake-sdk";
import { EdgeQuake } from "edgequake-sdk";
import { z } from "zod";
import { getClient, getConfig } from "../client.js";
import { formatError } from "../errors.js";

function renderAlgorithms(algorithms: ApprovedAlgorithm[]): string {
  const lines: string[] = ["**Algorithms:**"];
  algorithms.forEach((a, i) => {
    const idx = i + 1;
    const header = `[A${idx}] **${a.name}** (${a.algorithm_type}, confidence: ${a.confidence})`;
    lines.push(header);
    if (a.description) lines.push(`_description:_ ${a.description}`);
    if (a.complexity) lines.push(`_complexity:_ ${a.complexity}`);
    if (a.steps && a.steps.length > 0) {
      lines.push("_steps:_");
      for (const step of a.steps) {
        const action = (step.action ?? "").trim();
        const details = (step.details ?? "").trim();
        lines.push(`  ${step.number}. ${action} ${details}`.trimEnd());
      }
    }
    if (a.pseudocode) {
      lines.push("_pseudocode:_");
      lines.push("```\n" + a.pseudocode.replace(/\s+$/, "") + "\n```");
    }
  });
  return lines.join("\n");
}

function renderReferenceCode(snippets: ReferenceCodeSnippet[]): string {
  const lines: string[] = ["**Reference code implementations:**"];
  snippets.forEach((s, i) => {
    const idx = i + 1;
    const name = s.algorithm_name || s.algorithm_id;
    let header = `[C${idx}] **${name}** — \`${s.file_path}:${s.start_line}-${s.end_line}\``;
    if (s.repo_url && s.repo_commit) {
      header += ` ([source](${s.repo_url}/blob/${s.repo_commit}/${s.file_path}#L${s.start_line}-L${s.end_line}))`;
    }
    lines.push(header);
    if (s.match_rationale) lines.push(`_rationale:_ ${s.match_rationale}`);
    const lang = s.language ?? "";
    lines.push("```" + lang + "\n" + s.snippet + "\n```");
  });
  return lines.join("\n");
}

export function registerQueryTools(server: McpServer): void {
  server.tool(
    "query",
    "Search the EdgeQuake knowledge graph. Returns retrieved text chunks, entities, and relationships, plus any curated algorithm definitions and reference code implementations matched semantically to the question. Use 'hybrid' mode (default) for best results. Retrieval is strongest for single-aspect queries — split a multi-part question into one query per aspect and merge the results; a single query bundling unrelated aspects retrieves only the dominant one and misses the rest.",
    {
      query: z
        .string()
        .describe(
          "A focused question about ONE topic/aspect. Retrieval is embedding-based: a query that bundles several unrelated sub-questions is averaged into a single vector that matches only the dominant aspect, so chunks for the other aspects are silently dropped (you'll get false 'not reported'). Ask one aspect per call and run multiple queries when you need several, combining the results yourself. Good: 'What hardware/GPUs did ObfuscaTune use in experiments?' Bad: 'ObfuscaTune threat model and trusted hardware and TEE-vs-GPU split and overhead and security basis?'",
        ),
      mode: z
        .enum(["naive", "local", "global", "hybrid", "mix"])
        .optional()
        .describe(
          "Query mode: naive (vector-only), local (entity graph), global (community search), hybrid (local+global, default), mix (weighted blend)",
        ),
      workspace: z
        .string()
        .optional()
        .describe(
          "Target workspace slug. If omitted, uses the default configured workspace.",
        ),
    },
    async (params) => {
      try {
        const client = await getClient();

        let queryClient = client;
        if (params.workspace) {
          const config = getConfig();
          if (!config.defaultTenant) {
            throw new Error(
              "Cannot resolve workspace slug: no tenant configured",
            );
          }
          const workspace = await client.tenants.getWorkspaceBySlug(
            config.defaultTenant,
            params.workspace,
          );
          queryClient = new EdgeQuake({
            baseUrl: config.baseUrl,
            apiKey: config.apiKey,
            tenantId: config.defaultTenant,
            workspaceId: workspace.id,
            timeout: 60_000,
          });
        }

        const result = await queryClient.query.execute({
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

        const approvedAlgorithms = result.approved_algorithms ?? [];
        const referenceCode = result.reference_code ?? [];

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
        if (approvedAlgorithms.length > 0) {
          parts.push(renderAlgorithms(approvedAlgorithms));
        }
        if (referenceCode.length > 0) {
          parts.push(renderReferenceCode(referenceCode));
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
