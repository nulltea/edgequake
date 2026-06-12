//! Figure extraction for VLM-OCR.
//!
//! `StructureResult::to_markdown()` emits each `Image` / `Chart` element as
//! one HTML img div:
//!
//! ```html
//! <div style="text-align: center;"><img src="imgs/img_in_image_box_X_Y_W_H.jpg" alt="Image" width="N%" /></div>
//! ```
//!
//! and its caption (a sibling `LayoutElementType::FigureTitle` directly below
//! it) as a separate centered div. The image file referenced by `src=` is a
//! placeholder — it never gets created — so when the markdown is rendered
//! downstream the user sees the `alt="Image"` text leak through plus the
//! caption div as plain text.
//!
//! This module:
//!
//! 1. Walks the layout elements and identifies `Image` / `Chart` / `Seal`
//!    regions.
//! 2. For each, locates the nearest caption-class element directly below it
//!    in the same column (mirrors the spatial heuristic used by
//!    `build_algorithm_blocks`'s `find_caption_below`).
//! 3. Re-crops the figure region from a clone of the page image, WEBP-encodes
//!    it (lossy, q≈80 — paper figures are line-art/charts where WEBP is ~70%
//!    smaller than PNG at visually-lossless quality, which keeps the inlined
//!    base64 payload on agent read paths small), and pushes an
//!    `ExtractedFigure` payload into the caller's sink.
//! 4. Replaces the on-disk img div with `![fig_{page}_{i}](edgequake-figure)`
//!    so the chunker can pair markdown sites with PNG payloads. The caption
//!    div is left intact so the rendered markdown still has human-readable
//!    text next to the placeholder.

use std::sync::{Arc, LazyLock, Mutex};

use image::RgbImage;
use oar_ocr_core::domain::structure::{LayoutElement, LayoutElementType};
use oar_ocr_core::utils::BBoxCrop;
use regex::Regex;
use tracing::warn;

use super::ExtractedFigure;

/// `StructureResult::to_markdown()` builds the img filename from element
/// coordinates rounded to integers; we recompute the exact same string to
/// anchor the search. The whole `<div ...><img src="<file>" ... /></div>`
/// block gets replaced in one pass.
///
/// The regex spans the wrapping div — there isn't always whitespace between
/// the inner tags so `[^>]*` is the right granularity.
static IMG_DIV_TEMPLATE: LazyLock<&str> =
    LazyLock::new(|| r#"<div style="text-align: center;"><img src="{SRC}"[^>]*/></div>"#);

/// Walk `elements`, crop each captioned figure-class region from `page_image`,
/// PNG-encode, push into `sink`, and return the patched markdown.
///
/// `page_num` is 1-based and used as the page prefix in figure ids. `markdown`
/// is the output of `result.to_markdown()` from the vendor parser.
pub fn extract_and_patch(
    page_image: &RgbImage,
    elements: &[LayoutElement],
    page_num: u32,
    markdown: &str,
    sink: &Arc<Mutex<Vec<ExtractedFigure>>>,
) -> String {
    // Pre-collect caption-class elements (FigureTitle / ChartTitle) so we
    // can stitch each figure to its caption with a single scan per figure.
    let captions: Vec<&LayoutElement> = elements
        .iter()
        .filter(|e| e.element_type.is_caption())
        .collect();

    let mut patched = markdown.to_string();
    let mut new_figures: Vec<ExtractedFigure> = Vec::new();

    for (i, element) in elements.iter().enumerate() {
        if !is_figure_type(element.element_type) {
            continue;
        }

        let img_src = expected_img_src(element);
        // Escape the literal src for regex use (only `.` is hazardous in
        // practice; we splice the precomputed src into a fixed template
        // rather than re-escaping every metachar).
        let escaped = regex::escape(&img_src);
        let pattern = IMG_DIV_TEMPLATE.replace("{SRC}", &escaped);
        let re = match Regex::new(&pattern) {
            Ok(re) => re,
            Err(e) => {
                warn!(error = %e, "figure_extract: failed to compile img-div regex");
                continue;
            }
        };

        if !re.is_match(&patched) {
            // Either to_markdown skipped this element (rare for Image/Chart
            // but possible for Seal which uses a different ![Seal]... form)
            // or a later post-processing step rewrote the tag. Don't emit a
            // stranded figure chunk that has no markdown anchor.
            continue;
        }

        let figure_id = format!("fig_{page_num}_{i}");

        // Crop + WEBP-encode FIRST. The placeholder write and the sink push
        // must be atomic — either both happen for this figure or neither.
        // The previous ordering emitted the `![fig_X_Y](edgequake-figure)`
        // sentinel before attempting the crop, so a crop / encode failure
        // would leave a dangling sentinel in the markdown with no matching
        // entry in the sink (and therefore no chunks row, no media-fetch
        // hit, broken link in the rendered doc).
        let crop = match BBoxCrop::crop_bounding_box(page_image, &element.bbox) {
            Ok(c) => c,
            Err(e) => {
                warn!(figure_id = %figure_id, error = %e, "figure_extract: crop failed");
                continue;
            }
        };
        let Some(image_bytes) = encode_webp(&crop) else {
            continue;
        };

        // Pull the caption text from the nearest figure_title below this
        // figure (same column, vertical gap ≤ 100px). Falls back to the
        // figure's own .text if the VLM put a description there; ultimately
        // an empty caption is acceptable — the placeholder still anchors
        // the chunk.
        let caption = find_caption_below(element, &captions)
            .or_else(|| element.text.as_deref().map(|s| s.trim().to_string()))
            .filter(|s| !s.is_empty())
            .unwrap_or_default();

        // Crop+encode succeeded: now commit the markdown rewrite and the
        // sink entry together.
        let placeholder = format!("![{figure_id}](edgequake-figure)");
        patched = re.replace(&patched, placeholder.as_str()).into_owned();

        new_figures.push(ExtractedFigure {
            id: figure_id,
            image_bytes,
            mime: "image/webp".to_string(),
            caption,
            page: page_num,
            order_index: i as u32,
        });
    }

    if !new_figures.is_empty() {
        if let Ok(mut guard) = sink.lock() {
            guard.extend(new_figures);
        } else {
            warn!("figure_extract: sink mutex poisoned; figures lost for this page");
        }
    }

    patched
}

fn is_figure_type(t: LayoutElementType) -> bool {
    matches!(
        t,
        LayoutElementType::Image | LayoutElementType::Chart | LayoutElementType::Seal
    )
}

/// Recompute the exact `imgs/img_in_*_box_X_Y_W_H.jpg` filename that
/// `StructureResult::to_markdown` produces for this element. Mirrors the
/// `format!` in vendor/oar-ocr-core domain/structure.rs ~line 692-704.
fn expected_img_src(element: &LayoutElement) -> String {
    let kind = if element.element_type == LayoutElementType::Chart {
        "chart"
    } else {
        "image"
    };
    format!(
        "imgs/img_in_{}_box_{:.0}_{:.0}_{:.0}_{:.0}.jpg",
        kind,
        element.bbox.x_min(),
        element.bbox.y_min(),
        element.bbox.x_max(),
        element.bbox.y_max()
    )
}

/// Find the nearest caption element directly below `figure` in the same
/// column. Constants mirror the algorithm-extraction stitcher in vlm_ocr.rs.
/// Returns the trimmed caption text if matched.
fn find_caption_below(
    figure: &LayoutElement,
    captions: &[&LayoutElement],
) -> Option<String> {
    const MAX_V_GAP: f32 = 100.0;
    const MAX_H_CENTER_DIFF: f32 = 60.0;

    let fig_bottom = figure.bbox.y_max();
    let fig_cx = (figure.bbox.x_min() + figure.bbox.x_max()) * 0.5;

    let mut best: Option<(f32, &LayoutElement)> = None;
    for cap in captions {
        let cap_top = cap.bbox.y_min();
        let gap = cap_top - fig_bottom;
        if !(0.0..=MAX_V_GAP).contains(&gap) {
            continue;
        }
        let cap_cx = (cap.bbox.x_min() + cap.bbox.x_max()) * 0.5;
        if (cap_cx - fig_cx).abs() > MAX_H_CENTER_DIFF {
            continue;
        }
        if best.map(|(g, _)| gap < g).unwrap_or(true) {
            best = Some((gap, *cap));
        }
    }
    best.and_then(|(_, c)| c.text.clone())
        .map(|s| s.trim().to_string())
}

/// Lossy WEBP quality for figure crops. 80 keeps chart axes / small-font
/// labels legible to a vision model while cutting ~70% off the PNG size.
const WEBP_QUALITY: f32 = 80.0;

/// Lossy-WEBP-encode an RGB crop. Returns `None` (and warns) on encode
/// failure so the caller drops the figure rather than committing a markdown
/// placeholder with no matching sink entry.
fn encode_webp(image: &RgbImage) -> Option<Vec<u8>> {
    // `webp::Encoder::from_rgb` borrows the raw RGB8 buffer directly; the
    // returned `WebPMemory` owns the encoded bytes which we copy into a Vec.
    let encoder = webp::Encoder::from_rgb(image.as_raw(), image.width(), image.height());
    let encoded = encoder.encode(WEBP_QUALITY);
    if encoded.is_empty() {
        warn!("figure_extract: WEBP encode produced empty output");
        return None;
    }
    Some(encoded.to_vec())
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
    fn no_figure_elements_leaves_markdown_untouched() {
        let img = RgbImage::new(64, 64);
        let sink: Arc<Mutex<Vec<ExtractedFigure>>> = Arc::new(Mutex::new(Vec::new()));
        let markdown = "# Title\n\nSome text.";
        let elements = vec![make_element(
            LayoutElementType::Text,
            Some("Some text."),
            0.0,
            0.0,
            10.0,
            10.0,
        )];
        let out = extract_and_patch(&img, &elements, 1, markdown, &sink);
        assert_eq!(out, markdown);
        assert!(sink.lock().unwrap().is_empty());
    }

    #[test]
    fn captioned_image_div_gets_replaced_and_figure_captured() {
        let img = RgbImage::from_pixel(200, 200, image::Rgb([10, 20, 30]));
        let sink: Arc<Mutex<Vec<ExtractedFigure>>> = Arc::new(Mutex::new(Vec::new()));

        // Image at bbox (50, 60, 150, 130). FigureTitle directly below it
        // (gap = 5 px, centers aligned).
        let elements = vec![
            make_element(LayoutElementType::Text, Some("intro paragraph"), 0.0, 0.0, 200.0, 50.0),
            make_element(LayoutElementType::Image, None, 50.0, 60.0, 150.0, 130.0),
            make_element(
                LayoutElementType::FigureTitle,
                Some("Figure 1: Overview of the system."),
                40.0, 135.0, 160.0, 150.0,
            ),
        ];

        // Build the markdown the way to_markdown() would (the relevant
        // fragment, anyway).
        let img_div = r#"<div style="text-align: center;"><img src="imgs/img_in_image_box_50_60_150_130.jpg" alt="Image" width="50%" /></div>"#;
        let cap_div = r#"<div style="text-align: center;">Figure 1: Overview of the system. </div>"#;
        let md = format!("intro paragraph\n\n{img_div}\n\n{cap_div}");

        let out = extract_and_patch(&img, &elements, 3, &md, &sink);

        // Placeholder replaced the image div.
        assert!(
            out.contains("![fig_3_1](edgequake-figure)"),
            "expected placeholder in patched markdown:\n{out}"
        );
        // Image div is gone.
        assert!(!out.contains("imgs/img_in_image_box_50_60_150_130.jpg"));
        // Caption div is preserved.
        assert!(out.contains("Figure 1: Overview of the system."));

        let figs = sink.lock().unwrap();
        assert_eq!(figs.len(), 1);
        assert_eq!(figs[0].id, "fig_3_1");
        assert_eq!(figs[0].caption, "Figure 1: Overview of the system.");
        assert_eq!(figs[0].mime, "image/webp");
        assert!(!figs[0].image_bytes.is_empty());
        // Sanity: the bytes are a real WEBP that decodes back to the crop size.
        let decoded = image::load_from_memory(&figs[0].image_bytes)
            .expect("encoded figure should decode as WEBP");
        assert_eq!((decoded.width(), decoded.height()), (100, 70));
        assert_eq!(figs[0].page, 3);
    }

    #[test]
    fn chart_element_uses_chart_filename() {
        let img = RgbImage::new(200, 200);
        let sink: Arc<Mutex<Vec<ExtractedFigure>>> = Arc::new(Mutex::new(Vec::new()));

        let elements = vec![make_element(
            LayoutElementType::Chart,
            None,
            10.0, 20.0, 110.0, 80.0,
        )];
        let chart_div = r#"<div style="text-align: center;"><img src="imgs/img_in_chart_box_10_20_110_80.jpg" alt="Image" width="50%" /></div>"#;
        let out = extract_and_patch(&img, &elements, 1, chart_div, &sink);
        assert!(out.contains("![fig_1_0](edgequake-figure)"));
        assert_eq!(sink.lock().unwrap().len(), 1);
    }

    #[test]
    fn distant_caption_is_not_stitched() {
        let img = RgbImage::new(400, 400);
        let sink: Arc<Mutex<Vec<ExtractedFigure>>> = Arc::new(Mutex::new(Vec::new()));

        // Caption is >100 px below the figure → should NOT be stitched.
        let elements = vec![
            make_element(LayoutElementType::Image, None, 50.0, 60.0, 150.0, 130.0),
            make_element(
                LayoutElementType::FigureTitle,
                Some("Figure 9: unrelated."),
                40.0, 250.0, 160.0, 265.0,
            ),
        ];
        let img_div = r#"<div style="text-align: center;"><img src="imgs/img_in_image_box_50_60_150_130.jpg" alt="Image" width="50%" /></div>"#;
        let _out = extract_and_patch(&img, &elements, 1, img_div, &sink);
        let figs = sink.lock().unwrap();
        assert_eq!(figs.len(), 1);
        assert_eq!(
            figs[0].caption, "",
            "distant caption shouldn't be stitched"
        );
    }
}
