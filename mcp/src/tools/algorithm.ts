/**
 * Algorithm extraction tools.
 */
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { getClient } from "../client.js";
import { formatError } from "../errors.js";

export function registerAlgorithmTools(server: McpServer): void {
  // algorithm_list
  server.tool(
    "algorithm_list",
    "List algorithms extracted from a specific document, with optional status filter.",
    {
      document_id: z.string().describe("Document UUID"),
      status: z
        .enum(["pending", "approved", "rejected"])
        .optional()
        .describe("Filter by review status"),
    },
    async (params) => {
      try {
        const client = await getClient();
        const result = await client.algorithms.listByDocument(
          params.document_id,
          params.status,
        );

        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify(
                {
                  algorithms: result.algorithms.map((a) => ({
                    id: a.id,
                    name: a.name,
                    description: a.description,
                    status: a.status,
                    confidence: a.confidence,
                    tags: a.tags,
                    steps_count: a.steps.length,
                    inputs_count: a.inputs.length,
                    outputs_count: a.outputs.length,
                  })),
                  total: result.total,
                },
                null,
                2,
              ),
            },
          ],
        };
      } catch (error) {
        return formatError(error);
      }
    },
  );

  // algorithm_search
  server.tool(
    "algorithm_search",
    "Search approved algorithms across the workspace by semantic, lexical, and graph-entity signals.",
    {
      query: z.string().optional().describe("Search query string"),
      document_id: z.string().optional().describe("Filter by document UUID"),
      limit: z.number().optional().describe("Max results to return (default: 20)"),
      offset: z.number().optional().describe("Offset for pagination (default: 0)"),
    },
    async (params) => {
      try {
        const client = await getClient();
        const result = await client.algorithms.search({
          query: params.query,
          document_id: params.document_id,
          limit: params.limit,
          offset: params.offset,
        });

        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify(
                {
                  algorithms: result.algorithms.map((a) => ({
                    id: a.id,
                    name: a.name,
                    document_id: a.document_id,
                    description: a.description,
                    status: a.status,
                    confidence: a.confidence,
                    tags: a.tags,
                  })),
                  total: result.total,
                  limit: result.limit,
                  offset: result.offset,
                },
                null,
                2,
              ),
            },
          ],
        };
      } catch (error) {
        return formatError(error);
      }
    },
  );

}
