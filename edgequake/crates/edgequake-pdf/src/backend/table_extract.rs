//! Table extraction for VLM-OCR.
//!
//! Companion to `figure_extract`. `StructureResult::to_markdown()` renders
//! detected `LayoutElementType::Table` regions as a centered HTML block:
//!
//! ```html
//! <div style="text-align: center;"><table border="1">…</table></div>
//! ```
//!
//! The HTML inside that div is the VLM's table recognition output
//! (`RecognitionTask::Table`, "Parse the table in this image into HTML.").
//! For the gallery / classification feature we want:
//!
//!   1. The HTML preserved verbatim for inline display and as the input to
//!      the classifier.
//!   2. The HTML algorithmically parsed into headers + rows so structured
//!      consumers (e.g. comparison views) don't need to re-parse it.
//!   3. The nearby caption (TableTitle / FigureTableChartTitle / FigureTitle
//!      below or above the table) stitched in so the gallery card has a
//!      label.
//!   4. A stable `![tbl_{page}_{order_index}](edgequake-table)` placeholder
//!      inserted in place of the table div so the chunker can pair markdown
//!      sites with the structured payload, and so frontend rendering can
//!      swap the placeholder back for the rendered HTML on read.

use std::sync::{Arc, LazyLock, Mutex};

use oar_ocr_core::domain::structure::{LayoutElement, LayoutElementType};
use regex::Regex;
use tracing::warn;

use super::ExtractedTable;

/// Matches a `<div style="text-align: center;"><table border="1">…</table></div>`
/// block, the exact shape `StructureResult::to_markdown()` emits for Table
/// elements (see oar-ocr-core domain/structure.rs ~line 586). `(?s)` enables
/// DOTALL so the inner table HTML (which contains newlines after
/// `clean_ocr_text`) is consumed by `.*?`. Non-greedy so two tables on the
/// same page get matched independently.
static TABLE_DIV_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(
        r#"(?s)<div style="text-align: center;"><table border="1">.*?</table></div>"#,
    )
    .unwrap()
});

/// `<tr>...</tr>` cell-row, DOTALL for multi-line cells.
static TR_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?si)<tr[^>]*>(.*?)</tr>"#).unwrap());
/// `<th>...</th>` header cell.
static TH_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?si)<th[^>]*>(.*?)</th>"#).unwrap());
/// `<td>...</td>` body cell.
static TD_RE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#"(?si)<td[^>]*>(.*?)</td>"#).unwrap());
/// Strips any HTML tag — used to clean cell contents to plain text.
static TAG_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r#"<[^>]+>"#).unwrap());

/// Walk `elements`, find Table regions, pair each with its matching
/// `<div…><table…>` block in `markdown`, parse the HTML into headers + rows,
/// stitch in the nearest caption, push an `ExtractedTable` into `sink`, and
/// replace the matched block with `![tbl_{page}_{i}](edgequake-table)`.
///
/// `page_num` is 1-based; `i` is the layout element's index within the page.
/// Tables on a page get patched in document order (sequence of regex matches
/// vs sequence of Table-class elements in `elements`).
pub fn extract_and_patch(
    elements: &[LayoutElement],
    page_num: u32,
    markdown: &str,
    sink: &Arc<Mutex<Vec<ExtractedTable>>>,
) -> String {
    // 1) Collect Table-class elements with their reading-order index. The
    //    order_index field on ExtractedTable mirrors how figure_extract uses
    //    the layout-element vector index so each id (`tbl_{page}_{i}`) is
    //    stable across reprocess as long as the layout detector is.
    let table_elements: Vec<(usize, &LayoutElement)> = elements
        .iter()
        .enumerate()
        .filter(|(_, e)| e.element_type == LayoutElementType::Table)
        .collect();

    if table_elements.is_empty() {
        return markdown.to_string();
    }

    // 2) Pre-collect caption-class elements for the caption stitcher.
    //    LayoutElementType::is_caption() returns true for FigureTitle /
    //    ChartTitle / TableTitle / FigureTableChartTitle, which is what we
    //    want — tables can be captioned by any of those depending on the
    //    paper style.
    let captions: Vec<&LayoutElement> =
        elements.iter().filter(|e| e.element_type.is_caption()).collect();

    // 3) Find each `<div…><table…>` block in the markdown in source order
    //    and pair it with the next Table-class element. If the counts
    //    diverge (e.g. one table emitted as `[Table]` because its
    //    html_structure was missing, or vice versa), we pair what we can
    //    and warn — the remaining tables stay inline as raw HTML/`[Table]`
    //    placeholders, which is the existing pre-extraction behaviour.
    let match_ranges: Vec<(usize, usize, String)> = TABLE_DIV_RE
        .find_iter(markdown)
        .map(|m| (m.start(), m.end(), m.as_str().to_string()))
        .collect();

    if match_ranges.len() != table_elements.len() {
        warn!(
            page = page_num,
            md_table_blocks = match_ranges.len(),
            layout_table_elements = table_elements.len(),
            "table_extract: table-block count diverges from layout-element count; pairing what we can"
        );
    }

    let pair_count = match_ranges.len().min(table_elements.len());
    if pair_count == 0 {
        return markdown.to_string();
    }

    // 4) Build the patched markdown by splicing each matched div with a
    //    placeholder. Iterate by byte range to avoid second-pass regex
    //    rematching (the inner HTML is unpredictable so a needle-replace
    //    pattern would have to re-escape every metachar).
    let mut patched = String::with_capacity(markdown.len());
    let mut cursor = 0usize;
    let mut new_tables: Vec<ExtractedTable> = Vec::with_capacity(pair_count);

    for (k, (m_start, m_end, m_text)) in
        match_ranges.iter().take(pair_count).enumerate()
    {
        let (elem_idx, elem) = table_elements[k];
        let table_id = format!("tbl_{page_num}_{elem_idx}");
        let placeholder = format!("![{table_id}](edgequake-table)");

        patched.push_str(&markdown[cursor..*m_start]);
        patched.push_str(&placeholder);
        cursor = *m_end;

        // Extract just the inner `<table…>…</table>` (drop the wrapping
        // `<div>`). Frontend rendering re-wraps it with its own styling.
        let html = inner_table_html(m_text).to_string();
        let (headers, rows) = parse_table_html(&html);
        let caption = find_caption_nearby(elem, &captions);

        new_tables.push(ExtractedTable {
            id: table_id,
            html,
            headers,
            rows,
            caption,
            page: page_num,
            order_index: elem_idx as u32,
        });
    }
    patched.push_str(&markdown[cursor..]);

    if !new_tables.is_empty() {
        if let Ok(mut guard) = sink.lock() {
            guard.extend(new_tables);
        } else {
            warn!("table_extract: sink mutex poisoned; tables lost for this page");
        }
    }

    patched
}

/// Return the `<table…>…</table>` substring of a matched div block. Falls
/// back to the full block if the trim fails (defensive — the regex
/// guarantees the shape).
fn inner_table_html(div_block: &str) -> &str {
    let start = div_block.find("<table").unwrap_or(0);
    let end_marker = "</table>";
    let end = div_block.rfind(end_marker).map(|i| i + end_marker.len());
    match end {
        Some(e) if e > start => &div_block[start..e],
        _ => div_block,
    }
}

/// Parse a `<table>…</table>` HTML fragment into headers + rows.
///
/// Heuristic:
///   - If the first `<tr>` contains `<th>` cells, treat them as headers and
///     the remaining `<tr>`s as body rows. Body rows are parsed as `<td>` (or
///     `<th>` if the row happens to be all-header — some VLM outputs put
///     `<th>` on every row, in which case the row goes into `rows` as-is).
///   - Otherwise treat all `<tr>`s as body rows and leave headers empty.
///
/// Cell content is stripped of HTML tags and collapsed whitespace. Returns
/// `(Vec::new(), Vec::new())` if the input has no `<tr>` at all.
fn parse_table_html(table_html: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let trs: Vec<&str> = TR_RE
        .captures_iter(table_html)
        .filter_map(|c| c.get(1).map(|m| m.as_str()))
        .collect();
    if trs.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let parse_row = |row_html: &str| -> Vec<String> {
        let ths: Vec<String> = TH_RE
            .captures_iter(row_html)
            .filter_map(|c| c.get(1).map(|m| clean_cell(m.as_str())))
            .collect();
        if !ths.is_empty() {
            return ths;
        }
        TD_RE
            .captures_iter(row_html)
            .filter_map(|c| c.get(1).map(|m| clean_cell(m.as_str())))
            .collect()
    };

    let first_has_th = TH_RE.is_match(trs[0]);
    if first_has_th {
        let headers = parse_row(trs[0]);
        let rows: Vec<Vec<String>> = trs.iter().skip(1).map(|r| parse_row(r)).collect();
        (headers, rows)
    } else {
        let rows: Vec<Vec<String>> = trs.iter().map(|r| parse_row(r)).collect();
        (Vec::new(), rows)
    }
}

/// Strip tags, collapse whitespace, trim. The VLM occasionally emits
/// `<br/>` inside cells — we replace those with a space rather than dropping
/// them so multi-line cells stay distinguishable in the parsed text.
fn clean_cell(s: &str) -> String {
    let with_breaks = s.replace("<br>", " ").replace("<br/>", " ").replace("<br />", " ");
    let no_tags = TAG_RE.replace_all(&with_breaks, "");
    let mut out = String::with_capacity(no_tags.len());
    let mut prev_ws = false;
    for c in no_tags.chars() {
        if c.is_whitespace() {
            if !prev_ws && !out.is_empty() {
                out.push(' ');
            }
            prev_ws = true;
        } else {
            out.push(c);
            prev_ws = false;
        }
    }
    out.trim().to_string()
}

/// Find the nearest caption element above or below `table` within the same
/// column. Tables in IEEE/ACM-style papers usually have the caption *above*
/// the table; arxiv/Springer often have it *below*. We check both with the
/// same gap and centroid tolerance as figure caption stitching.
fn find_caption_nearby(
    table: &LayoutElement,
    captions: &[&LayoutElement],
) -> String {
    const MAX_V_GAP: f32 = 100.0;
    const MAX_H_CENTER_DIFF: f32 = 80.0;

    let t_top = table.bbox.y_min();
    let t_bottom = table.bbox.y_max();
    let t_cx = (table.bbox.x_min() + table.bbox.x_max()) * 0.5;

    let mut best: Option<(f32, &LayoutElement)> = None;
    for cap in captions {
        let cap_top = cap.bbox.y_min();
        let cap_bottom = cap.bbox.y_max();
        // Gap above (caption sits above table) or below.
        let gap_above = t_top - cap_bottom;
        let gap_below = cap_top - t_bottom;
        let gap = gap_above
            .max(0.0)
            .min(if gap_below >= 0.0 { gap_below } else { f32::INFINITY });
        let in_range = (0.0..=MAX_V_GAP).contains(&gap_above)
            || (0.0..=MAX_V_GAP).contains(&gap_below);
        if !in_range {
            continue;
        }
        let cap_cx = (cap.bbox.x_min() + cap.bbox.x_max()) * 0.5;
        if (cap_cx - t_cx).abs() > MAX_H_CENTER_DIFF {
            continue;
        }
        let effective_gap = if (0.0..=MAX_V_GAP).contains(&gap_above) {
            gap_above
        } else {
            gap_below
        };
        if best.map(|(g, _)| effective_gap < g).unwrap_or(true) {
            best = Some((effective_gap, *cap));
        }
        let _ = gap;
    }
    best.and_then(|(_, c)| c.text.clone())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use oar_ocr_core::processors::BoundingBox;

    fn make_element(
        element_type: LayoutElementType,
        text: Option<&str>,
        x0: f32,
        y0: f32,
        x1: f32,
        y1: f32,
    ) -> LayoutElement {
        let bbox = BoundingBox::from_coords(x0, y0, x1, y1);
        let mut el = LayoutElement::new(bbox, element_type, 1.0);
        el.text = text.map(|t| t.to_string());
        el
    }

    #[test]
    fn no_tables_leaves_markdown_untouched() {
        let sink: Arc<Mutex<Vec<ExtractedTable>>> = Arc::new(Mutex::new(Vec::new()));
        let md = "# Title\n\nSome text.";
        let elements = vec![make_element(
            LayoutElementType::Text,
            Some("Some text."),
            0.0,
            0.0,
            10.0,
            10.0,
        )];
        let out = extract_and_patch(&elements, 1, md, &sink);
        assert_eq!(out, md);
        assert!(sink.lock().unwrap().is_empty());
    }

    #[test]
    fn single_table_with_header_row_parsed_and_patched() {
        let sink: Arc<Mutex<Vec<ExtractedTable>>> = Arc::new(Mutex::new(Vec::new()));
        let table_div = "<div style=\"text-align: center;\"><table border=\"1\">\
            <tr><th>Model</th><th>Acc</th></tr>\
            <tr><td>GPT-4</td><td>0.92</td></tr>\
            <tr><td>Llama</td><td>0.87</td></tr>\
            </table></div>";
        let md = format!("Body text.\n\n{table_div}\n\nMore text.");
        // Table element above the caption (caption below).
        let elements = vec![
            make_element(LayoutElementType::Text, Some("Body text."), 0.0, 0.0, 100.0, 30.0),
            make_element(LayoutElementType::Table, None, 30.0, 60.0, 170.0, 140.0),
            make_element(
                LayoutElementType::TableTitle,
                Some("Table 1: Model accuracy on benchmark X."),
                40.0,
                145.0,
                160.0,
                160.0,
            ),
        ];
        let out = extract_and_patch(&elements, 2, &md, &sink);
        assert!(out.contains("![tbl_2_1](edgequake-table)"), "out:\n{out}");
        assert!(!out.contains("<table border=\"1\">"));

        let tables = sink.lock().unwrap();
        assert_eq!(tables.len(), 1);
        let t = &tables[0];
        assert_eq!(t.id, "tbl_2_1");
        assert_eq!(t.page, 2);
        assert_eq!(t.order_index, 1);
        assert_eq!(t.headers, vec!["Model", "Acc"]);
        assert_eq!(
            t.rows,
            vec![
                vec!["GPT-4".to_string(), "0.92".to_string()],
                vec!["Llama".to_string(), "0.87".to_string()],
            ]
        );
        assert_eq!(t.caption, "Table 1: Model accuracy on benchmark X.");
        assert!(t.html.contains("<table"));
    }

    #[test]
    fn table_without_header_row_has_empty_headers() {
        let sink: Arc<Mutex<Vec<ExtractedTable>>> = Arc::new(Mutex::new(Vec::new()));
        let table_div = "<div style=\"text-align: center;\"><table border=\"1\">\
            <tr><td>a</td><td>b</td></tr>\
            <tr><td>c</td><td>d</td></tr>\
            </table></div>";
        let elements = vec![make_element(
            LayoutElementType::Table,
            None,
            0.0,
            0.0,
            100.0,
            100.0,
        )];
        let out = extract_and_patch(&elements, 1, table_div, &sink);
        assert!(out.contains("![tbl_1_0](edgequake-table)"));
        let tables = sink.lock().unwrap();
        assert_eq!(tables.len(), 1);
        assert!(tables[0].headers.is_empty());
        assert_eq!(tables[0].rows.len(), 2);
    }

    #[test]
    fn caption_above_table_is_stitched() {
        let sink: Arc<Mutex<Vec<ExtractedTable>>> = Arc::new(Mutex::new(Vec::new()));
        let table_div = "<div style=\"text-align: center;\"><table border=\"1\">\
            <tr><td>x</td></tr></table></div>";
        // Caption above (gap=10), table below.
        let elements = vec![
            make_element(
                LayoutElementType::TableTitle,
                Some("Tab. 3: Caption above."),
                40.0,
                40.0,
                160.0,
                50.0,
            ),
            make_element(LayoutElementType::Table, None, 30.0, 60.0, 170.0, 140.0),
        ];
        let _ = extract_and_patch(&elements, 4, table_div, &sink);
        let tables = sink.lock().unwrap();
        assert_eq!(tables[0].caption, "Tab. 3: Caption above.");
    }

    #[test]
    fn two_tables_paired_in_order() {
        let sink: Arc<Mutex<Vec<ExtractedTable>>> = Arc::new(Mutex::new(Vec::new()));
        let div = |body: &str| {
            format!(
                "<div style=\"text-align: center;\"><table border=\"1\">{body}</table></div>"
            )
        };
        let md = format!(
            "intro\n\n{}\n\nmiddle\n\n{}\n\nend",
            div("<tr><td>a</td></tr>"),
            div("<tr><td>b</td></tr>"),
        );
        let elements = vec![
            make_element(LayoutElementType::Table, None, 0.0, 10.0, 100.0, 50.0),
            make_element(LayoutElementType::Text, Some("middle"), 0.0, 60.0, 100.0, 80.0),
            make_element(LayoutElementType::Table, None, 0.0, 90.0, 100.0, 150.0),
        ];
        let out = extract_and_patch(&elements, 7, &md, &sink);
        assert!(out.contains("![tbl_7_0](edgequake-table)"));
        assert!(out.contains("![tbl_7_2](edgequake-table)"));
        let tables = sink.lock().unwrap();
        assert_eq!(tables.len(), 2);
        assert_eq!(tables[0].order_index, 0);
        assert_eq!(tables[1].order_index, 2);
        assert_eq!(tables[0].rows[0], vec!["a".to_string()]);
        assert_eq!(tables[1].rows[0], vec!["b".to_string()]);
    }

    #[test]
    fn cell_cleaner_strips_tags_and_collapses_whitespace() {
        assert_eq!(clean_cell("  foo  bar  "), "foo bar");
        assert_eq!(clean_cell("<b>foo</b> <i>bar</i>"), "foo bar");
        assert_eq!(clean_cell("line1<br/>line2"), "line1 line2");
        assert_eq!(clean_cell("a\n\n\tb"), "a b");
    }
}
