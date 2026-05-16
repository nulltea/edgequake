/**
 * Resolve VLM-OCR table placeholders in markdown.
 *
 * The Rust table extractor in `edgequake-pdf/src/backend/table_extract.rs`
 * replaces each `<div><table border="1">…</table></div>` block in the
 * vendor-produced markdown with a sentinel placeholder of the form
 *
 *   ![<table_id>](edgequake-table)
 *
 * The HTML and parsed rows are stored in Postgres (`chunks.table_html`,
 * `chunks.table_rows`) and served by
 * `GET /api/v1/documents/{documentId}/tables/{tableId}`.
 *
 * Unlike figures (which resolve to an `<img>` URL), tables need to inline
 * their HTML directly into the rendered markdown so the table actually
 * shows up styled. So this helper takes a pre-fetched `tables` list
 * (caption-only is fine — for inline render we use the document's
 * `getDocumentTable` payload, but the list-row form works too as long as it
 * carries `html`) and string-replaces each sentinel with a
 * `<div class="edgequake-table-inline">…</div>` containing the HTML.
 *
 * Why client-side: same reasoning as the figure resolver — keeps the
 * stored markdown portable and lets the chunker keep matching a stable
 * sentinel string.
 */

/**
 * Matches `![<id>](edgequake-table)` where `<id>` is the table id the
 * extractor put in the alt text (e.g. `tbl_3_5`). Conservative
 * `[A-Za-z0-9_]+` so an id-format tweak doesn't silently stop matching.
 */
const SENTINEL_RE = /!\[([A-Za-z0-9_]+)\]\(edgequake-table\)/g;

export interface InlineTablePayload {
  /** Stable extractor id (`tbl_{page}_{order_index}`). */
  table_id: string;
  /** Rendered table HTML — `<table…>…</table>`. */
  html: string;
}

/**
 * Replace every `![<id>](edgequake-table)` with the inlined HTML from
 * `tables`, keyed by `table_id`. Sentinels with no matching entry are left
 * intact (downstream the markdown renderer will show the placeholder so
 * the gap is visible during reprocess windows).
 */
export function resolveTableSentinels(
  markdown: string,
  tables: InlineTablePayload[],
): string {
  if (!markdown || tables.length === 0) return markdown;
  const byId = new Map<string, string>();
  for (const t of tables) {
    if (t.html) byId.set(t.table_id, t.html);
  }
  if (byId.size === 0) return markdown;
  return markdown.replace(SENTINEL_RE, (match, tableId: string) => {
    const html = byId.get(tableId);
    if (!html) return match;
    // Wrap so we can target the inlined table without touching tables that
    // might appear in the raw markdown for other reasons.
    return `<div class="edgequake-table-inline" data-table-id="${tableId}">${html}</div>`;
  });
}
