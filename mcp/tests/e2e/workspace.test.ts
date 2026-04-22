/**
 * E2E: Workspace tools test.
 */
import type { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { callTool, createTestClient, isServerRunning } from "./helpers.js";

describe("workspace tools (e2e)", () => {
  let client: Client;
  let cleanup: () => Promise<void>;
  let serverUp: boolean;

  beforeAll(async () => {
    serverUp = await isServerRunning();
    if (!serverUp) return;
    const ctx = await createTestClient();
    client = ctx.client;
    cleanup = ctx.cleanup;
  });

  afterAll(async () => {
    if (cleanup) await cleanup();
  });

  it("should list workspaces", async () => {
    if (!serverUp) {
      console.log("SKIP: EdgeQuake server not running");
      return;
    }
    const result = await callTool(client, "workspace_list");
    expect(Array.isArray(result)).toBe(true);
  });

  it("should get details and stats for an existing workspace", async () => {
    if (!serverUp) {
      console.log("SKIP: EdgeQuake server not running");
      return;
    }

    const workspaces = (await callTool(client, "workspace_list")) as Array<{
      id: string;
    }>;
    if (workspaces.length === 0) {
      console.log("SKIP: no workspaces available");
      return;
    }
    const workspaceId = workspaces[0].id;

    // Get
    const detail = (await callTool(client, "workspace_get", {
      workspace_id: workspaceId,
    })) as Record<string, unknown>;
    expect(detail.id).toBe(workspaceId);

    // Stats
    const stats = (await callTool(client, "workspace_stats", {
      workspace_id: workspaceId,
    })) as Record<string, unknown>;
    expect(stats).toHaveProperty("document_count");
    expect(stats).toHaveProperty("entity_count");
  });
});
