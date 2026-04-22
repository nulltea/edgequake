/**
 * Workspace management tools.
 */
import type { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { z } from "zod";
import { getClient, getConfig } from "../client.js";
import { formatError } from "../errors.js";

export function registerWorkspaceTools(server: McpServer): void {
  // workspace_list
  server.tool(
    "workspace_list",
    "List all workspaces in the current tenant",
    {},
    async () => {
      try {
        const client = await getClient();
        const config = getConfig();
        const tenantId = config.defaultTenant;

        if (!tenantId) {
          // Try listing tenants to find one
          const tenants = await client.tenants.list();
          if (tenants.length === 0) {
            return {
              content: [
                {
                  type: "text" as const,
                  text: "No tenants found. Create a tenant first.",
                },
              ],
              isError: true,
            };
          }
          const workspaces = await client.tenants.listWorkspaces(tenants[0].id);
          return {
            content: [
              {
                type: "text" as const,
                text: JSON.stringify(
                  workspaces.map((w) => ({
                    id: w.id,
                    name: w.name,
                    slug: w.slug,
                    description: w.description,
                    llm_provider: w.llm_provider,
                    llm_model: w.llm_model,
                  })),
                  null,
                  2,
                ),
              },
            ],
          };
        }

        const workspaces = await client.tenants.listWorkspaces(tenantId);
        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify(
                workspaces.map((w) => ({
                  id: w.id,
                  name: w.name,
                  slug: w.slug,
                  description: w.description,
                  llm_provider: w.llm_provider,
                  llm_model: w.llm_model,
                })),
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

  // workspace_get
  server.tool(
    "workspace_get",
    "Get workspace details including document and entity counts",
    {
      workspace_id: z.string().describe("Workspace UUID"),
    },
    async (params) => {
      try {
        const client = await getClient();
        const detail = await client.workspaces.get(params.workspace_id);
        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify(detail, null, 2),
            },
          ],
        };
      } catch (error) {
        return formatError(error);
      }
    },
  );

  // workspace_stats
  server.tool(
    "workspace_stats",
    "Get statistics for a workspace (document, entity, relationship, chunk counts)",
    {
      workspace_id: z.string().describe("Workspace UUID"),
    },
    async (params) => {
      try {
        const client = await getClient();
        const stats = await client.workspaces.stats(params.workspace_id);
        return {
          content: [
            {
              type: "text" as const,
              text: JSON.stringify(stats, null, 2),
            },
          ],
        };
      } catch (error) {
        return formatError(error);
      }
    },
  );
}
