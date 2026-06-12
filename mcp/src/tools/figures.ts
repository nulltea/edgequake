/**
 * Figure inlining helpers for the MCP read paths.
 *
 * Extracted figures live as raw image bytes keyed by `(documentId, figureId)`
 * and are served by `GET /api/v1/documents/{id}/figures/{figureId}`. In the
 * extracted markdown each figure is a sentinel `![<id>](edgequake-figure)`;
 * retrieved chunks carry `kind === "figure"` plus `figure_id`. Neither form
 * carries pixels, so on their own the model never sees the figure.
 *
 * These helpers fetch the bytes (as WEBP — ~70% smaller than the stored PNG
 * for chart/line-art figures, which keeps the base64 payload manageable) and
 * turn them into MCP `image` content blocks the model can actually view.
 */
import type { EdgeQuake } from "edgequake-sdk";

/** An MCP image content block. */
export interface ImageBlock {
  type: "image";
  data: string;
  mimeType: string;
}

/**
 * Cap on how many figures a single tool response will inline as base64 image
 * blocks (the `query` path). A retrieval can surface many figure hits; inlining
 * all of them would bloat the response. Callers should note when this truncates.
 */
export const MAX_INLINE_FIGURES = 20;

/** Sentinel placeholder the VLM-OCR pipeline writes for each captured figure. */
const FIGURE_SENTINEL = /!\[([A-Za-z0-9_]+)\]\(edgequake-figure\)/g;

/**
 * Rewrite every `![<id>](edgequake-figure)` sentinel in `markdown` to a real,
 * fetchable media URL: `![<id>](<baseUrl>/api/v1/documents/<documentId>/figures/<id>)`.
 *
 * This is the `document_get_md` strategy: keeping figures as markdown image
 * links (rather than inlining base64 blocks) leaves the response valid markdown,
 * avoids the per-figure base64 payload, and sidesteps the response token cap
 * that would otherwise truncate a large document's trailing images.
 */
export function rewriteFigureSentinelsToUrls(
  markdown: string,
  documentId: string,
  baseUrl: string,
): string {
  const base = baseUrl.replace(/\/+$/, "");
  return markdown.replace(
    FIGURE_SENTINEL,
    (_match, id: string) =>
      `![${id}](${base}/api/v1/documents/${documentId}/figures/${id})`,
  );
}

/**
 * Fetch a single figure as a WEBP image block. Best-effort: returns `null` on
 * any failure (404, transcode hiccup, empty body, network) so an absent image
 * never fails the whole document/query read.
 */
export async function fetchFigureBlock(
  client: EdgeQuake,
  documentId: string,
  figureId: string,
): Promise<ImageBlock | null> {
  try {
    const blob = await client.documents.getFigureMedia(documentId, figureId);
    const buf = Buffer.from(await blob.arrayBuffer());
    if (buf.length === 0) return null;
    return {
      type: "image" as const,
      data: buf.toString("base64"),
      // The endpoint serves WEBP; fall back to the blob's reported type on the
      // rare path where a legacy figure failed to transcode and was streamed
      // in its stored format.
      mimeType: blob.type || "image/webp",
    };
  } catch {
    return null;
  }
}

/**
 * Fetch the given figures as WEBP image blocks (used by the `query` path,
 * where figures come from retrieval hits rather than markdown positions).
 *
 * Returns the blocks plus how many ids were dropped because of the
 * {@link MAX_INLINE_FIGURES} cap, so callers can disclose the truncation.
 */
export async function fetchFigureBlocks(
  client: EdgeQuake,
  documentId: string,
  figureIds: string[],
): Promise<{ blocks: ImageBlock[]; truncated: number }> {
  const truncated = Math.max(0, figureIds.length - MAX_INLINE_FIGURES);
  const wanted = figureIds.slice(0, MAX_INLINE_FIGURES);
  const blocks = await Promise.all(
    wanted.map((id) => fetchFigureBlock(client, documentId, id)),
  );
  return { blocks: blocks.filter((b): b is ImageBlock => b !== null), truncated };
}
