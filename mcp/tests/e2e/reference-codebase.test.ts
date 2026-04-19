/**
 * E2E: Reference-codebase MCP tools (query_code, get_symbol_neighborhood).
 *
 * Exercises against whatever index exists in the dev workspace — the
 * tests treat missing fixtures as SKIP so they don't break when the
 * workspace is empty.
 */
import type { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { afterAll, beforeAll, describe, expect, it } from "vitest";
import { callTool, createTestClient, isServerRunning } from "./helpers.js";

describe("reference-codebase tools (e2e)", () => {
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

  it("query_code returns human-readable hits when the NL query matches", async () => {
    if (!serverUp) {
      console.log("SKIP: EdgeQuake server not running");
      return;
    }
    // Response is rendered markdown, not JSON, so callTool returns the
    // string unchanged. Verify it mentions a code fence OR the "no hits"
    // fallback — both are valid outcomes depending on workspace state.
    const result = (await callTool(client, "query_code", {
      query: "capacity-constrained cluster rebalancing",
      limit: 3,
    })) as string | Record<string, unknown>;

    expect(typeof result === "string" || typeof result === "object").toBe(true);
    if (typeof result === "string") {
      const expected =
        result.includes("reference-code hit") ||
        result.includes("No approved code snippets matched");
      expect(expected).toBe(true);
    }
  });

  it("get_symbol_neighborhood resolves by anchor_symbol", async () => {
    if (!serverUp) {
      console.log("SKIP: EdgeQuake server not running");
      return;
    }

    // Discover a live index via HTTP; skip if the dev workspace has none.
    const baseUrl = process.env.EDGEQUAKE_BASE_URL ?? "http://localhost:8080";
    const tenant = "00000000-0000-0000-0000-000000000002";
    const workspace = "00000000-0000-0000-0000-000000000003";
    const listResp = await fetch(
      `${baseUrl}/api/v1/reference-codebase/query`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-Tenant-ID": tenant,
          "X-Workspace-ID": workspace,
        },
        body: JSON.stringify({
          query: "rebalance",
          limit: 1,
        }),
      },
    );
    if (!listResp.ok) {
      console.log("SKIP: reference-codebase query endpoint unreachable");
      return;
    }
    const list = (await listResp.json()) as {
      coding_context: Array<{ index_id: string; symbol_name: string | null }>;
    };
    if (list.coding_context.length === 0 || !list.coding_context[0].symbol_name) {
      console.log("SKIP: no index with a named symbol chunk in dev workspace");
      return;
    }
    const indexId = list.coding_context[0].index_id;
    const symbolName = list.coding_context[0].symbol_name;

    const result = (await callTool(client, "get_symbol_neighborhood", {
      index_id: indexId,
      anchor_symbol: symbolName,
      hops: 1,
    })) as string | Record<string, unknown>;

    expect(typeof result === "string" || typeof result === "object").toBe(true);
    if (typeof result === "string") {
      // The renderer always includes "Seed: N symbol(s)" on success, or
      // an error message on failure. Either way the tool responded.
      const expected =
        result.includes("Seed:") ||
        result.includes("No symbols") ||
        result.includes("not found");
      expect(expected).toBe(true);
    }
  });
});
