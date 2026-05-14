/**
 * Resolve VLM-OCR figure placeholders in markdown.
 *
 * The Rust figure extractor in `edgequake-pdf/src/backend/figure_extract.rs`
 * replaces each `<div><img src="imgs/img_in_..."/></div>` block in the
 * vendor-produced markdown with a sentinel placeholder of the form
 *
 *   ![<figure_id>](edgequake-figure)
 *
 * The figure bytes are stored in Postgres (`chunks.media_bytes`) and served by
 * `GET /api/v1/documents/{documentId}/figures/{figureId}`. This helper rewrites
 * the sentinel placeholders to that URL so the renderer's `<img>` tag actually
 * resolves to the cropped figure.
 *
 * Why client-side rewriting:
 *   - keeps the markdown blob in storage portable (no URLs baked in),
 *   - lets the chunker keep pattern-matching the stable sentinel,
 *   - lets different deployments hit different `SERVER_BASE_URL`s without
 *     migrating the markdown.
 */

import { getFigureMediaUrl } from "@/lib/api/edgequake";

/**
 * Matches `![<id>](edgequake-figure)` where `<id>` is the figure id the
 * extractor put in the alt text (e.g. `fig_3_5`). The id pattern is the
 * extractor's `fig_{page}_{order_index}`, matched conservatively as
 * `[A-Za-z0-9_]+` so an id-format tweak doesn't silently stop matching.
 */
const SENTINEL_RE = /!\[([A-Za-z0-9_]+)\]\(edgequake-figure\)/g;

/**
 * Replace every `![<id>](edgequake-figure)` with `![<id>](<media-fetch URL>)`.
 *
 * No-op when `documentId` is undefined/empty — keeps the sentinel intact so
 * the broken-image fallback in the renderer surfaces the placeholder rather
 * than silently swallowing it.
 */
export function resolveFigureSentinels(
  markdown: string,
  documentId: string | undefined,
): string {
  if (!markdown || !documentId) return markdown;
  return markdown.replace(SENTINEL_RE, (_match, figureId: string) => {
    const url = getFigureMediaUrl(documentId, figureId);
    return `![${figureId}](${url})`;
  });
}
