/**
 * Regression tests for the `document_get_md` tool.
 *
 * Bug: a doc whose PDF backing had `markdown_content` missing on the wire
 * (the SDK's `PdfContentResponse` previously declared a non-existent
 * `markdown: string` field) crashed the handler with
 * `Cannot read properties of undefined (reading 'trim')`.
 */
import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { InMemoryTransport } from "@modelcontextprotocol/sdk/inMemory.js";
import { afterAll, beforeAll, describe, expect, it, vi } from "vitest";

const getDoc = vi.fn();
const getPdfContent = vi.fn();

vi.mock("../src/client.js", () => ({
  getClient: async () => ({
    documents: {
      get: getDoc,
      pdf: { getContent: getPdfContent },
    },
  }),
  getConfig: () => ({ baseUrl: "http://x", apiKey: undefined }),
}));

const { createServer } = await import("../src/server.js");

interface ToolResult {
  isError?: boolean;
  content: Array<{ type: string; text: string }>;
}

describe("document_get_md", () => {
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

  it("returns PDF markdown_content when present", async () => {
    getDoc.mockResolvedValueOnce({
      id: "doc-1",
      pdf_id: "pdf-1",
      status: "completed",
      content: null,
    });
    getPdfContent.mockResolvedValueOnce({
      pdf_id: "pdf-1",
      filename: "paper.pdf",
      file_size_bytes: 1024,
      content_type: "application/pdf",
      markdown_content: "# Paper title\n\nBody.",
      is_processed: true,
    });

    const result = (await client.callTool({
      name: "document_get_md",
      arguments: { document_id: "doc-1" },
    })) as ToolResult;

    expect(result.isError).toBeFalsy();
    expect(result.content[0].text).toContain("# Paper title");
  });

  it("does not crash when markdown_content is missing (the original bug)", async () => {
    getDoc.mockResolvedValueOnce({
      id: "doc-2",
      pdf_id: "pdf-2",
      status: "completed",
      content: null,
    });
    getPdfContent.mockResolvedValueOnce({
      pdf_id: "pdf-2",
      filename: "paper.pdf",
      file_size_bytes: 1024,
      content_type: "application/pdf",
      is_processed: false,
    });

    const result = (await client.callTool({
      name: "document_get_md",
      arguments: { document_id: "doc-2" },
    })) as ToolResult;

    expect(result.isError).toBe(true);
    const body = result.content[0].text;
    expect(body).not.toContain("trim");
    expect(body).toContain("Document markdown not available");
  });

  it("falls back to doc.content when there is no PDF backing", async () => {
    getDoc.mockResolvedValueOnce({
      id: "doc-3",
      pdf_id: null,
      status: "completed",
      content: "Plain markdown body.",
    });

    const result = (await client.callTool({
      name: "document_get_md",
      arguments: { document_id: "doc-3" },
    })) as ToolResult;

    expect(result.isError).toBeFalsy();
    expect(result.content[0].text).toBe("Plain markdown body.");
  });
});
