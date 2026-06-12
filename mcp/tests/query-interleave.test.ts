/**
 * Regression test for the `query` tool's figure interleaving (issue 2).
 *
 * Figures must be hydrated as base64 image blocks AT their retrieved chunk
 * position (right after the caption), de-duplicated by (document_id, figure_id)
 * — not appended at the array tail where the harness's output cap truncates
 * them first.
 */
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";

const execute = vi.fn();
const getFigureMedia = vi.fn();

vi.mock("../src/client.js", () => ({
  getClient: async () => ({
    query: { execute },
    documents: { getFigureMedia },
  }),
  getConfig: () => ({ baseUrl: "http://x", apiKey: undefined }),
}));

const { createServer } = await import("../src/server.js");

interface ContentItem {
  type: string;
  text?: string;
  data?: string;
  mimeType?: string;
}
interface ToolResult {
  isError?: boolean;
  content: ContentItem[];
}

function fakeBlob(bytes: number[], type = "image/webp"): Blob {
  return {
    type,
    arrayBuffer: async () => new Uint8Array(bytes).buffer,
  } as unknown as Blob;
}

describe("query figure interleave", () => {
  let client: Client;
  let cleanup: () => Promise<void>;

  beforeAll(async () => {
    const server = createServer();
    client = new Client({ name: "test-client", version: "0.1.0" });
    const [c, s] = InMemoryTransport.createLinkedPair();
    await Promise.all([server.connect(s), client.connect(c)]);
    cleanup = async () => {
      await client.close();
      await server.close();
    };
  });

  afterAll(async () => {
    if (cleanup) await cleanup();
  });

  it("swaps backend figure markdown for a base64 block at position, deduped", async () => {
    // The backend injects `![caption](/api/v1/documents/{doc}/figures/{id})`
    // into chunk snippets (figure-chunk snippets + matched prose divs). The MCP
    // tool turns the FIRST occurrence of each figure_id into a base64 block and
    // leaves later occurrences as caption text.
    execute.mockResolvedValueOnce({
      sources: [
        { source_type: "chunk", snippet: "Intro prose.", file_path: "paper.pdf", document_id: "doc1" },
        {
          source_type: "chunk",
          kind: "figure",
          figure_id: "fig_1_0",
          document_id: "doc1",
          file_path: "paper.pdf",
          snippet: "![Figure 1: Overview](/api/v1/documents/doc1/figures/fig_1_0)",
        },
        { source_type: "chunk", snippet: "Middle prose.", file_path: "paper.pdf", document_id: "doc1" },
        // duplicate reference to the same figure → must NOT yield a 2nd image
        {
          source_type: "chunk",
          snippet: "As in ![Figure 1: Overview](/api/v1/documents/doc1/figures/fig_1_0) above.",
          file_path: "paper.pdf",
          document_id: "doc1",
        },
        { source_type: "entity", id: "E1", snippet: "entity one" },
      ],
    });
    getFigureMedia.mockResolvedValue(fakeBlob([1, 2, 3]));

    const result = (await client.callTool({
      name: "query",
      arguments: { query: "overview figure" },
    })) as ToolResult;

    expect(result.isError).toBeFalsy();
    const c = result.content;

    // Exactly one image block (duplicate figure_id deduped to first occurrence).
    const images = c.filter((x) => x.type === "image");
    expect(images.length).toBe(1);
    expect(images[0].mimeType).toBe("image/webp");
    expect(images[0].data).toBe(Buffer.from([1, 2, 3]).toString("base64"));
    // getFigureMedia fetched once (deduped before fetching).
    expect(getFigureMedia).toHaveBeenCalledTimes(1);
    expect(getFigureMedia).toHaveBeenCalledWith("doc1", "fig_1_0");

    // The image sits immediately AFTER a text block carrying its caption…
    const imgIdx = c.findIndex((x) => x.type === "image");
    expect(imgIdx).toBeGreaterThan(0);
    expect(c[imgIdx - 1].type).toBe("text");
    expect(c[imgIdx - 1].text).toContain("Figure 1: Overview");

    // …no raw relative URL leaks into any text block…
    expect(c.every((x) => !(x.text ?? "").includes("/api/v1/documents/"))).toBe(true);

    // …and it's NOT at the tail — trailing text (entities) comes after it.
    expect(imgIdx).toBeLessThan(c.length - 1);
    const entIdx = c.findIndex((x) => x.type === "text" && (x.text ?? "").includes("entity one"));
    expect(entIdx).toBeGreaterThan(imgIdx);
  });
});
