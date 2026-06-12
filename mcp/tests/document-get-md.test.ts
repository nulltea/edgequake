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
const getFigureMedia = vi.fn();

vi.mock("../src/client.js", () => ({
  getClient: async () => ({
    documents: {
      get: getDoc,
      pdf: { getContent: getPdfContent },
      getFigureMedia,
    },
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

  it("rewrites figure sentinels to fetchable media URLs (no base64 inlining)", async () => {
    getDoc.mockResolvedValueOnce({
      id: "doc-fig",
      pdf_id: "pdf-fig",
      status: "completed",
      content: null,
    });
    getPdfContent.mockResolvedValueOnce({
      pdf_id: "pdf-fig",
      filename: "paper.pdf",
      file_size_bytes: 2048,
      content_type: "application/pdf",
      markdown_content:
        "Intro.\n\n![fig_1_0](edgequake-figure)\n\nMiddle.\n\n![fig_2_0](edgequake-figure)\n\nEnd.",
      is_processed: true,
    });

    const result = (await client.callTool({
      name: "document_get_md",
      arguments: { document_id: "doc-fig" },
    })) as ToolResult;

    expect(result.isError).toBeFalsy();
    // Single text block — no base64 image blocks, no figure fetches.
    expect(result.content).toHaveLength(1);
    expect(result.content[0].type).toBe("text");
    expect(result.content.some((c) => c.type === "image")).toBe(false);
    expect(getFigureMedia).not.toHaveBeenCalled();

    const text = result.content[0].text ?? "";
    // Opaque sentinel replaced by a real markdown image link to the media URL.
    expect(text).not.toContain("edgequake-figure");
    expect(text).toContain(
      "![fig_1_0](http://x/api/v1/documents/doc-fig/figures/fig_1_0)",
    );
    expect(text).toContain(
      "![fig_2_0](http://x/api/v1/documents/doc-fig/figures/fig_2_0)",
    );
    // Surrounding prose preserved.
    expect(text).toContain("Intro.");
    expect(text).toContain("Middle.");
    expect(text).toContain("End.");
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
