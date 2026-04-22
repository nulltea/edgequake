//! VLM-OCR backend: PP-DocLayoutV2 layout detection + remote VLM recognition.
//!
//! Uses the [`oar-ocr-core`] layout predictor (PP-DocLayoutV2, 25 classes, 8-dim
//! reading order) for fast ONNX-based layout detection, then sends each cropped
//! region to an external VLM server (llama.cpp, vLLM, Ollama, etc.) via the
//! [`VlmClientBackend`] HTTP client for text/formula/table recognition.
//!
//! Configuration:
//!   - Layout ONNX: `$EDGEQUAKE_OAR_OCR_MODEL_DIR/pp-doclayoutv2.onnx`
//!   - VLM server: `$EDGEQUAKE_LLAMACPP_URL` (default `http://localhost:8081`)
//!   - Model name: `$EDGEQUAKE_LLAMACPP_MODEL` (optional)
//!   - Timeout: `$EDGEQUAKE_LLAMACPP_TIMEOUT` (default 120s)

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use async_trait::async_trait;
use tracing::{info, warn};

use super::vlm_client::{VlmClientBackend, VlmClientConfig};
use super::{PdfConversionConfig, PdfConverter};
use crate::error::PdfConversionError;

// pp-doclayout_plus-l (20-class): outperforms V2/V3 on protocol/algorithm boxes.
// On B2A.pdf V2 missed Protocol 1 entirely; plus-L detects all 5 algorithmic
// blocks (2 protocols + 3 functionality figures) at confidence >= 0.79.
const LAYOUT_MODEL: &str = "pp-doclayout_plus-l.onnx";
const LAYOUT_MODEL_NAME: &str = "pp-doclayout_plus-l";

/// VLM-OCR powered PDF → Markdown converter (PP-DocLayoutV2 + remote VLM).
#[derive(Debug, Default)]
pub struct VlmOcrConverter;

#[async_trait]
impl PdfConverter for VlmOcrConverter {
    async fn convert(
        &self,
        pdf_bytes: &[u8],
        config: &PdfConversionConfig,
    ) -> Result<String, PdfConversionError> {
        let pdf_bytes = pdf_bytes.to_vec();
        let vlm_base_url = config.vlm_base_url.clone();
        let vlm_model = config.vlm_model.clone();
        let concurrency = config
            .vision
            .as_ref()
            .and_then(|v| v.concurrency)
            .unwrap_or_else(|| {
                std::env::var("EDGEQUAKE_PDF_CONCURRENCY")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(4)
            });
        let progress_cb = config
            .vision
            .as_ref()
            .and_then(|v| v.progress_callback.clone());
        let algo_sink = config.algorithm_block_sink.clone();

        tokio::task::spawn_blocking(move || {
            use rayon::prelude::*;

            // 1. Resolve layout model directory.
            let model_dir = model_cache_dir()?;
            let layout_path = model_dir.join(LAYOUT_MODEL);
            if !layout_path.exists() {
                return Err(PdfConversionError::Backend(format!(
                    "VLM-OCR: layout model not found at {}. \
                     Download pp-doclayoutv2.onnx to $EDGEQUAKE_OAR_OCR_MODEL_DIR.",
                    layout_path.display()
                )));
            }

            // 2. Render PDF pages to images.
            let images = render_pdf_to_images(&pdf_bytes)?;
            if images.is_empty() {
                return Err(PdfConversionError::EmptyOutput("VLM-OCR: PDF has no pages"));
            }
            let page_count = images.len();
            info!(pages = page_count, "VLM-OCR: rendered PDF pages");

            if let Some(ref cb) = progress_cb {
                cb.on_conversion_start(page_count);
            }

            // 3. Build layout predictor (ONNX, fast).
            let layout_predictor = oar_ocr_core::predictors::LayoutDetectionPredictor::builder()
                .model_name(LAYOUT_MODEL_NAME)
                .build(&layout_path)
                .map_err(|e| {
                    PdfConversionError::Backend(format!("VLM-OCR: layout predictor failed: {e}"))
                })?;

            // 4. Create VLM client backend.
            let mut vlm_config = VlmClientConfig::from_env();
            if let Some(url) = vlm_base_url {
                vlm_config.base_url = url;
            }
            if let Some(model) = vlm_model {
                vlm_config.model = Some(model);
            }
            info!(
                base_url = %vlm_config.base_url,
                model = ?vlm_config.model,
                concurrency = concurrency,
                "VLM-OCR: connecting to VLM server"
            );
            let backend = VlmClientBackend::new(vlm_config);
            // crop_pad_ratio=0.04 adds ~4% padding around each layout element's
            // bbox before VLM recognition. Without it, plus-L's tight algorithm
            // bboxes can crop just below a protocol box's title line (e.g.
            // "**Protocol 1 Bit2A protocol (Π₁)**" or "**Protocol 2 B2A
            // protocol (Π₂)**"), causing GLM-OCR to return only the math body
            // without the title. Verified with inspect_page5_vlm_output and
            // inspect_page4_functionalities tests on B2A.pdf: pad 0.02 is
            // borderline (Protocol 1 sometimes loses its title); pad 0.04
            // reliably captures both Protocol 1 and Protocol 2 headers.
            let parser_cfg = oar_ocr_vl::doc_parser::DocParserConfig {
                crop_pad_ratio: 0.04,
                ..Default::default()
            };
            let parser = oar_ocr_vl::doc_parser::DocParser::with_config(&backend, parser_cfg);

            // 5. Process pages concurrently with rayon.
            let completed = Arc::new(AtomicUsize::new(0));
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(concurrency)
                .build()
                .map_err(|e| PdfConversionError::Internal(format!("rayon pool: {e}")))?;

            let results: Vec<(usize, Option<String>)> = pool.install(|| {
                images
                    .into_par_iter()
                    .enumerate()
                    .map(|(i, image)| {
                        let page_num = i + 1;

                        if let Some(ref cb) = progress_cb {
                            cb.on_page_start(page_num, page_count);
                        }

                        let md = match parser.parse(&layout_predictor, image) {
                            Ok(result) => {
                                // Capture algorithm blocks if sink is provided.
                                // One AlgorithmBlock per detected layout element (not per page) so
                                // Pass 2 gets one algorithm per chunk and progress count matches
                                // the number of detected algorithms.
                                if let Some(ref sink) = algo_sink {
                                    let algo_blocks_for_page =
                                        build_algorithm_blocks(&result.layout_elements, page_num);
                                    info!(
                                        page = page_num,
                                        algo_count = algo_blocks_for_page.len(),
                                        total_elements = result.layout_elements.len(),
                                        "VLM-OCR: algorithm elements on page"
                                    );
                                    if let Ok(mut blocks) = sink.lock() {
                                        for b in algo_blocks_for_page {
                                            blocks.push(b);
                                        }
                                    }
                                }

                                let md = result.to_markdown();
                                // Apply the same deterministic LaTeX repair we run on
                                // algorithm blocks (\mathbb{S} from sampling-$, unbraced
                                // ^\theta scripts, orphan $ delimiters, unbalanced [[…]],
                                // JSON-escape collisions). Without this pass the full
                                // page markdown stored in pdf_documents.markdown_content
                                // keeps OCR artefacts that the algorithm path avoids.
                                let md = crate::latex_repair::repair_latex(&md);
                                if md.trim().is_empty() {
                                    None
                                } else {
                                    Some(md)
                                }
                            }
                            Err(e) => {
                                warn!(page = page_num, error = %e, "VLM-OCR: page failed");
                                if let Some(ref cb) = progress_cb {
                                    cb.on_page_error(page_num, page_count, e.to_string());
                                }
                                None
                            }
                        };

                        let done = completed.fetch_add(1, Ordering::Relaxed) + 1;
                        // Only report page_complete on success; failures are already
                        // reported via on_page_error above. Reporting both inflates
                        // the completed_pages counter and makes the UI show progress
                        // for pages that actually failed.
                        if let (Some(ref cb), Some(ref s)) = (progress_cb.as_ref(), md.as_ref()) {
                            cb.on_page_complete(page_num, page_count, s.len());
                        }
                        info!(
                            page = page_num,
                            done = done,
                            total = page_count,
                            "VLM-OCR: page completed"
                        );

                        (i, md)
                    })
                    .collect()
            });

            // 6. Assemble markdown in page order.
            let mut sorted = results;
            sorted.sort_by_key(|(i, _)| *i);

            let mut markdown = String::new();
            let mut succeeded = 0usize;
            for (_, md) in sorted {
                if let Some(page_md) = md {
                    if !markdown.is_empty() {
                        markdown.push_str("\n\n");
                    }
                    markdown.push_str(&page_md);
                    succeeded += 1;
                }
            }

            if markdown.trim().is_empty() {
                return Err(PdfConversionError::EmptyOutput("VLM-OCR returned no text"));
            }

            if let Some(ref cb) = progress_cb {
                cb.on_conversion_complete(page_count, succeeded);
            }

            info!(
                markdown_len = markdown.len(),
                pages_succeeded = succeeded,
                pages_total = page_count,
                "VLM-OCR conversion completed"
            );

            Ok(markdown)
        })
        .await
        .map_err(|e| PdfConversionError::Internal(e.to_string()))?
    }

    fn backend_name(&self) -> &'static str {
        "vlmocr"
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Algorithm block detection (used by algorithm extraction pipeline)
// ─────────────────────────────────────────────────────────────────────────────

/// Build algorithm blocks from a page's layout elements.
///
/// For each `Algorithm`-class element, produces one `AlgorithmBlock` whose
/// markdown is the VLM-recognized body prefixed by the nearest `FigureTitle`
/// caption that sits *directly below* the algorithm in the same column (e.g.
/// "Fig. 2: FUNCTIONALITY F_Bit2A").
///
/// Why prepend the caption:
///
/// PP-DocLayout_plus-L classifies the centered figure caption below each
/// Functionality box as a separate `figure_title` element, so the algorithm
/// element's crop contains only the numbered steps ("1) F_X receives...").
/// Without the caption, Pass 2's LLM can't identify the functionality by name
/// and invents generic titles like "Additive Secret Sharing Computation" or
/// "Function F_Bit2A". Verified via the `inspect_page4_all_elements` test on
/// B2A.pdf.
fn build_algorithm_blocks(
    elements: &[oar_ocr_core::domain::structure::LayoutElement],
    page_num: usize,
) -> Vec<AlgorithmBlock> {
    use oar_ocr_core::domain::structure::LayoutElementType;

    let figure_titles: Vec<&oar_ocr_core::domain::structure::LayoutElement> = elements
        .iter()
        .filter(|e| e.element_type == LayoutElementType::FigureTitle)
        .collect();

    let mut blocks = Vec::new();
    for elem in elements
        .iter()
        .filter(|e| e.element_type == LayoutElementType::Algorithm)
    {
        let algo_md = oar_ocr_vl::utils::to_markdown(std::slice::from_ref(elem), &[]);
        if algo_md.trim().is_empty() {
            continue;
        }

        let caption = find_caption_below(elem, &figure_titles);
        let markdown = match caption {
            Some(cap) => format!("## {cap}\n\n{algo_md}"),
            None => algo_md,
        };

        // Repair known OCR-stage LaTeX corruption (GLM-OCR's \mathbb{S} for
        // uniform-sampling `$`, orphan delimiters, unbalanced [[...]], bare
        // super/sub-scripts) before handing the chunk to the LLM. JSON-escape
        // reconstruction also runs, but is a no-op here because the text hasn't
        // been JSON-round-tripped yet.
        let markdown = crate::latex_repair::repair_latex(&markdown);

        blocks.push(AlgorithmBlock {
            page: page_num,
            markdown,
        });
    }
    blocks
}

/// Find the nearest `figure_title` element directly below `algo` in the same
/// column (centroids within 60px horizontally; title top within 100px of
/// algo bottom). Returns the caption text if matched.
fn find_caption_below(
    algo: &oar_ocr_core::domain::structure::LayoutElement,
    figure_titles: &[&oar_ocr_core::domain::structure::LayoutElement],
) -> Option<String> {
    const MAX_V_GAP: f32 = 100.0;
    const MAX_H_CENTER_DIFF: f32 = 60.0;

    let algo_bottom = algo.bbox.y_max();
    let algo_cx = (algo.bbox.x_min() + algo.bbox.x_max()) * 0.5;

    let mut best: Option<(f32, &oar_ocr_core::domain::structure::LayoutElement)> = None;
    for title in figure_titles {
        let title_top = title.bbox.y_min();
        let gap = title_top - algo_bottom;
        if !(0.0..=MAX_V_GAP).contains(&gap) {
            continue;
        }
        let title_cx = (title.bbox.x_min() + title.bbox.x_max()) * 0.5;
        if (title_cx - algo_cx).abs() > MAX_H_CENTER_DIFF {
            continue;
        }
        if best.map(|(g, _)| gap < g).unwrap_or(true) {
            best = Some((gap, *title));
        }
    }

    best.and_then(|(_, t)| t.text.clone())
        .map(|s| s.trim().to_string())
}

/// A detected algorithm block from layout detection + VLM recognition.
#[derive(Debug, Clone)]
pub struct AlgorithmBlock {
    /// 1-based page number.
    pub page: usize,
    /// VLM-recognized markdown content of the algorithm block.
    pub markdown: String,
}

/// Detect algorithm blocks in a PDF using layout detection + VLM recognition.
///
/// 1. Renders PDF pages to images
/// 2. Runs PP-DocLayoutV2 layout detection
/// 3. Filters for `Algorithm` elements
/// 4. Recognizes each algorithm block via VLM (DocParser)
/// 5. Returns clean markdown per algorithm block
pub fn detect_algorithm_blocks(
    pdf_bytes: &[u8],
    vlm_config: VlmClientConfig,
) -> Result<Vec<AlgorithmBlock>, PdfConversionError> {
    // 1. Resolve layout model
    let model_dir = model_cache_dir()?;
    let layout_path = model_dir.join(LAYOUT_MODEL);
    if !layout_path.exists() {
        return Err(PdfConversionError::Backend(format!(
            "Algorithm detection: layout model not found at {}",
            layout_path.display()
        )));
    }

    // 2. Render PDF pages
    let images = render_pdf_to_images(pdf_bytes)?;
    if images.is_empty() {
        return Ok(Vec::new());
    }
    info!(
        pages = images.len(),
        "Algorithm detection: rendered PDF pages"
    );

    // 3. Build layout predictor
    let layout_predictor = oar_ocr_core::predictors::LayoutDetectionPredictor::builder()
        .model_name(LAYOUT_MODEL_NAME)
        .build(&layout_path)
        .map_err(|e| {
            PdfConversionError::Backend(format!(
                "Algorithm detection: layout predictor failed: {e}"
            ))
        })?;

    // 4. Build VLM backend + DocParser
    info!(
        base_url = %vlm_config.base_url,
        model = ?vlm_config.model,
        "Algorithm detection: using VLM server"
    );
    let backend = VlmClientBackend::new(vlm_config);
    // crop_pad_ratio=0.04 — see comment in VlmOcrConverter::convert for why.
    let parser_cfg = oar_ocr_vl::doc_parser::DocParserConfig {
        crop_pad_ratio: 0.04,
        ..Default::default()
    };
    let parser = oar_ocr_vl::doc_parser::DocParser::with_config(&backend, parser_cfg);

    // 5. Process pages concurrently: detect layout → filter algorithms → recognize
    let concurrency = std::env::var("EDGEQUAKE_PDF_CONCURRENCY")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(concurrency)
        .build()
        .map_err(|e| PdfConversionError::Internal(format!("rayon pool: {e}")))?;

    let page_count = images.len();
    let mut blocks: Vec<AlgorithmBlock> = pool.install(|| {
        use rayon::prelude::*;
        images
            .into_par_iter()
            .enumerate()
            .flat_map(|(page_idx, image)| {
                let page_num = page_idx + 1;

                let result = match parser.parse(&layout_predictor, image) {
                    Ok(r) => r,
                    Err(e) => {
                        warn!(page = page_num, error = %e, "Algorithm detection: page failed, skipping");
                        return Vec::new();
                    }
                };

                let page_blocks = build_algorithm_blocks(&result.layout_elements, page_num);
                info!(
                    page = page_num,
                    count = page_blocks.len(),
                    done = page_num,
                    total = page_count,
                    "Algorithm detection: found algorithm blocks"
                );
                page_blocks
            })
            .collect()
    });

    // Sort by page order (rayon may return out of order)
    blocks.sort_by_key(|b| b.page);

    info!(
        total_blocks = blocks.len(),
        "Algorithm detection: completed"
    );
    Ok(blocks)
}

// ─────────────────────────────────────────────────────────────────────────────
// Layout model directory
// ─────────────────────────────────────────────────────────────────────────────

fn model_cache_dir() -> Result<PathBuf, PdfConversionError> {
    if let Ok(dir) = std::env::var("EDGEQUAKE_OAR_OCR_MODEL_DIR") {
        if !dir.is_empty() {
            let p = PathBuf::from(dir);
            std::fs::create_dir_all(&p).map_err(|e| {
                PdfConversionError::Internal(format!("Failed to create model dir: {e}"))
            })?;
            return Ok(p);
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let p = PathBuf::from(home).join(".cache/edgequake/oar-ocr-models");
        std::fs::create_dir_all(&p).map_err(|e| {
            PdfConversionError::Internal(format!("Failed to create model dir: {e}"))
        })?;
        return Ok(p);
    }
    let p = PathBuf::from("/tmp/edgequake-oar-ocr-models");
    std::fs::create_dir_all(&p)
        .map_err(|e| PdfConversionError::Internal(format!("Failed to create model dir: {e}")))?;
    Ok(p)
}

// ─────────────────────────────────────────────────────────────────────────────
// PDF rendering (hayro, pure Rust)
// ─────────────────────────────────────────────────────────────────────────────

fn render_pdf_to_images(pdf_bytes: &[u8]) -> Result<Vec<image::RgbImage>, PdfConversionError> {
    use hayro::hayro_syntax::Pdf;

    let pdf_data: Arc<Vec<u8>> = Arc::new(pdf_bytes.to_vec());
    let pdf = Pdf::new(pdf_data)
        .map_err(|e| PdfConversionError::Backend(format!("hayro parse failed: {e:?}")))?;

    let pages = pdf.pages();
    if pages.is_empty() {
        warn!("VLM-OCR: PDF has zero pages");
        return Ok(Vec::new());
    }

    let mut images = Vec::with_capacity(pages.len());
    for (i, page) in pages.iter().enumerate() {
        match render_single_page(page) {
            Ok(img) => images.push(img),
            Err(e) => {
                warn!(page = i + 1, error = %e, "VLM-OCR: failed to render page, skipping");
            }
        }
    }
    Ok(images)
}

fn render_single_page(page: &hayro::hayro_syntax::page::Page) -> Result<image::RgbImage, String> {
    use hayro::RenderSettings;

    let media_box = page.media_box();
    let width = (media_box.x1 - media_box.x0) as f32;
    let height = (media_box.y1 - media_box.y0) as f32;

    if width <= 0.0 || height <= 0.0 {
        return Err(format!("Invalid page size: {}x{}", width, height));
    }

    let scale = std::env::var("EDGEQUAKE_OAR_OCR_SCALE")
        .ok()
        .and_then(|s| s.parse::<f32>().ok())
        .filter(|s| *s > 0.0 && *s <= 8.0)
        .unwrap_or(3.0f32);

    let settings = RenderSettings {
        x_scale: scale,
        y_scale: scale,
        bg_color: hayro::vello_cpu::color::palette::css::WHITE,
        ..Default::default()
    };

    let interpreter_settings = hayro::hayro_interpret::InterpreterSettings::default();
    let pixmap = hayro::render(page, &interpreter_settings, &settings);

    // Convert RGBA → RGB.
    let rgba = pixmap.data_as_u8_slice();
    let mut rgb = Vec::with_capacity((pixmap.width() as usize) * (pixmap.height() as usize) * 3);
    for chunk in rgba.chunks(4) {
        rgb.push(chunk[0]);
        rgb.push(chunk[1]);
        rgb.push(chunk[2]);
    }

    image::RgbImage::from_raw(u32::from(pixmap.width()), u32::from(pixmap.height()), rgb)
        .ok_or_else(|| "Failed to construct RgbImage from pixmap".to_string())
}

#[cfg(test)]
mod layout_model_comparison {
    use super::*;

    /// Compare layout detection across V2, V3, and plus-L for a given PDF.
    /// Prints per-page label counts so we can see which model detects Protocol
    /// boxes as Algorithm-class elements (vs text/image/etc).
    ///
    /// Run with:
    ///   EDGEQUAKE_OAR_OCR_MODEL_DIR=/home/timo/.cache/edgequake/oar-ocr-models \
    ///   EDGEQUAKE_TEST_PDF=/home/timo/edgequake/B2A.pdf \
    ///   cargo test -p edgequake-pdf --release --lib layout_model_comparison::compare_layout_models -- --ignored --nocapture
    #[test]
    #[ignore]
    fn compare_layout_models() {
        let pdf_path = std::env::var("EDGEQUAKE_TEST_PDF")
            .unwrap_or_else(|_| "/home/timo/edgequake/B2A.pdf".to_string());
        let pdf_bytes = std::fs::read(&pdf_path).expect("read PDF");
        let images = render_pdf_to_images(&pdf_bytes).expect("render pages");
        let model_dir = model_cache_dir().expect("model dir");

        let candidates = [
            ("V2", "pp-doclayoutv2", "pp-doclayoutv2.onnx"),
            ("V3", "pp-doclayoutv3", "pp-doclayoutv3.onnx"),
            ("plus-L", "pp-doclayout_plus-l", "pp-doclayout_plus-l.onnx"),
        ];

        for (label, model_name, file) in candidates {
            let path = model_dir.join(file);
            if !path.exists() {
                eprintln!("[{label}] model not found at {}, skipping", path.display());
                continue;
            }
            let predictor = match oar_ocr_core::predictors::LayoutDetectionPredictor::builder()
                .model_name(model_name)
                .build(&path)
            {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("[{label}] failed to build predictor: {e}");
                    continue;
                }
            };

            println!("\n========== {label} ({model_name}) ==========");
            for (i, img) in images.iter().enumerate() {
                let page = i + 1;
                let result = match predictor.predict(vec![img.clone()]) {
                    Ok(r) => r,
                    Err(e) => {
                        eprintln!("[{label}] page {page} predict failed: {e}");
                        continue;
                    }
                };
                let page_elements = match result.elements.first() {
                    Some(v) => v,
                    None => continue,
                };
                let mut by_type: std::collections::BTreeMap<String, usize> = Default::default();
                let mut algo_details = Vec::new();
                for (idx, elem) in page_elements.iter().enumerate() {
                    *by_type.entry(elem.element_type.clone()).or_insert(0) += 1;
                    // Flag anything that looks algorithm-like or pseudocode-like.
                    let raw = elem.element_type.to_lowercase();
                    if raw.contains("algorithm")
                        || raw.contains("pseudocode")
                        || raw.contains("code")
                    {
                        algo_details.push(format!(
                            "  #{idx} label={} score={:.3} points={:?}",
                            elem.element_type, elem.score, elem.bbox.points,
                        ));
                    }
                }
                println!(
                    "page {page}: total={} | {}",
                    page_elements.len(),
                    by_type
                        .iter()
                        .map(|(k, v)| format!("{k}={v}"))
                        .collect::<Vec<_>>()
                        .join(", ")
                );
                for line in algo_details {
                    println!("{line}");
                }
            }
        }
    }

    /// Verify the figure-caption stitching: after running DocParser, call
    /// build_algorithm_blocks() and print each block so we can confirm the
    /// "Fig. N: FUNCTIONALITY F_X" captions get prepended correctly.
    #[test]
    #[ignore]
    fn verify_caption_stitching() {
        use super::super::vlm_client::{VlmClientBackend, VlmClientConfig};
        use oar_ocr_vl::doc_parser::{DocParser, DocParserConfig};

        let pdf_path = std::env::var("EDGEQUAKE_TEST_PDF")
            .unwrap_or_else(|_| "/home/timo/edgequake/B2A.pdf".to_string());
        let pdf_bytes = std::fs::read(&pdf_path).expect("read PDF");
        let images = render_pdf_to_images(&pdf_bytes).expect("render pages");
        let model_dir = model_cache_dir().expect("model dir");
        let mut vlm_cfg = VlmClientConfig::from_env();
        if let Ok(m) = std::env::var("EDGEQUAKE_TEST_VLM_MODEL") {
            vlm_cfg.model = Some(m);
        }
        let backend = VlmClientBackend::new(vlm_cfg);
        let path = model_dir.join("pp-doclayout_plus-l.onnx");
        let predictor = oar_ocr_core::predictors::LayoutDetectionPredictor::builder()
            .model_name("pp-doclayout_plus-l")
            .build(&path)
            .expect("build predictor");
        let cfg = DocParserConfig {
            crop_pad_ratio: 0.02,
            ..Default::default()
        };
        let parser = DocParser::with_config(&backend, cfg);

        for (page_idx, img) in images.iter().enumerate() {
            let page_num = page_idx + 1;
            let result = parser.parse(&predictor, img.clone()).expect("parse");
            let blocks = super::build_algorithm_blocks(&result.layout_elements, page_num);
            if blocks.is_empty() {
                continue;
            }
            println!("\n===== page {page_num}: {} blocks =====", blocks.len());
            for (i, b) in blocks.iter().enumerate() {
                let preview: String = b.markdown.chars().take(200).collect();
                println!("  block #{i} (len={}): {:?}", b.markdown.len(), preview);
            }
        }
    }

    /// Print every page-4 element with bbox + element type, so we can see
    /// how figure_title captions are positioned relative to Algorithm boxes.
    #[test]
    #[ignore]
    fn inspect_page4_all_elements() {
        use super::super::vlm_client::{VlmClientBackend, VlmClientConfig};
        use oar_ocr_vl::doc_parser::{DocParser, DocParserConfig};

        let pdf_path = std::env::var("EDGEQUAKE_TEST_PDF")
            .unwrap_or_else(|_| "/home/timo/edgequake/B2A.pdf".to_string());
        let pdf_bytes = std::fs::read(&pdf_path).expect("read PDF");
        let images = render_pdf_to_images(&pdf_bytes).expect("render pages");
        let page4 = images.get(3).expect("page 4").clone();
        let model_dir = model_cache_dir().expect("model dir");

        let mut vlm_cfg = VlmClientConfig::from_env();
        if let Ok(m) = std::env::var("EDGEQUAKE_TEST_VLM_MODEL") {
            vlm_cfg.model = Some(m);
        }
        let backend = VlmClientBackend::new(vlm_cfg);

        let path = model_dir.join("pp-doclayout_plus-l.onnx");
        let predictor = oar_ocr_core::predictors::LayoutDetectionPredictor::builder()
            .model_name("pp-doclayout_plus-l")
            .build(&path)
            .expect("build predictor");

        let cfg = DocParserConfig {
            crop_pad_ratio: 0.02,
            ..Default::default()
        };
        let parser = DocParser::with_config(&backend, cfg);
        let result = parser.parse(&predictor, page4.clone()).expect("parse");
        println!("\n========== plus-L page 4, all elements ==========");
        for (idx, elem) in result.layout_elements.iter().enumerate() {
            let text = elem.text.as_deref().unwrap_or("<no-text>");
            let preview: String = text.chars().take(120).collect();
            println!(
                "  #{idx} type={:?} label={:?} bbox=[x:{:.0}-{:.0} y:{:.0}-{:.0}] textlen={} text={:?}",
                elem.element_type,
                elem.label,
                elem.bbox.x_min(),
                elem.bbox.x_max(),
                elem.bbox.y_min(),
                elem.bbox.y_max(),
                text.len(),
                preview
            );
        }
    }

    /// Inspect page 4's algorithm elements from both models at the current
    /// crop_pad_ratio=0.02 used in production, so we can see what GLM-OCR
    /// returns for each Functionality box (F_CR, F_Bit2A, F_B2A) and
    /// Protocol 1.
    #[test]
    #[ignore]
    fn inspect_page4_functionalities() {
        use super::super::vlm_client::{VlmClientBackend, VlmClientConfig};
        use oar_ocr_vl::doc_parser::{DocParser, DocParserConfig};

        let pdf_path = std::env::var("EDGEQUAKE_TEST_PDF")
            .unwrap_or_else(|_| "/home/timo/edgequake/B2A.pdf".to_string());
        let pdf_bytes = std::fs::read(&pdf_path).expect("read PDF");
        let images = render_pdf_to_images(&pdf_bytes).expect("render pages");
        let page4 = images.get(3).expect("page 4").clone();
        let model_dir = model_cache_dir().expect("model dir");

        let mut vlm_cfg = VlmClientConfig::from_env();
        if let Ok(m) = std::env::var("EDGEQUAKE_TEST_VLM_MODEL") {
            vlm_cfg.model = Some(m);
        }
        let backend = VlmClientBackend::new(vlm_cfg);

        let path = model_dir.join("pp-doclayout_plus-l.onnx");
        let predictor = oar_ocr_core::predictors::LayoutDetectionPredictor::builder()
            .model_name("pp-doclayout_plus-l")
            .build(&path)
            .expect("build predictor");

        for pad in [0.02_f32, 0.04, 0.06] {
            let cfg = DocParserConfig {
                crop_pad_ratio: pad,
                ..Default::default()
            };
            let parser = DocParser::with_config(&backend, cfg);
            let result = parser.parse(&predictor, page4.clone()).expect("parse");
            println!("\n========== plus-L page 4, pad={pad} ==========");
            for (idx, elem) in result.layout_elements.iter().enumerate() {
                if elem.element_type
                    != oar_ocr_core::domain::structure::LayoutElementType::Algorithm
                {
                    continue;
                }
                let text = elem.text.as_deref().unwrap_or("<no-text>");
                let preview: String = text.chars().take(320).collect();
                println!(
                    "  #{idx} bbox=[x:{:.0}-{:.0} y:{:.0}-{:.0}] textlen={}",
                    elem.bbox.x_min(),
                    elem.bbox.x_max(),
                    elem.bbox.y_min(),
                    elem.bbox.y_max(),
                    text.len(),
                );
                println!("     text: {preview:?}");
            }
        }
    }

    /// Run DocParser (layout + VLM) against page 5 of B2A.pdf and print the
    /// content of every detected element, so we can see exactly what GLM-OCR
    /// returns for the Protocol 2 algorithm crop under each layout model.
    ///
    /// Run with (after tailscale login):
    ///   OPENAI_COMPATIBLE_BASE_URL=https://ai.tail59ea6b.ts.net/v1 \
    ///   EDGEQUAKE_TEST_VLM_MODEL=GLM-OCR \
    ///   EDGEQUAKE_OAR_OCR_MODEL_DIR=/home/timo/.edgequake/oar-ocr-models \
    ///   EDGEQUAKE_TEST_PDF=/home/timo/edgequake/B2A.pdf \
    ///   cargo test -p edgequake-pdf --release --lib layout_model_comparison::inspect_page5_vlm_output -- --ignored --nocapture
    #[test]
    #[ignore]
    fn inspect_page5_vlm_output() {
        use super::super::vlm_client::{VlmClientBackend, VlmClientConfig};
        use oar_ocr_vl::doc_parser::DocParser;

        let pdf_path = std::env::var("EDGEQUAKE_TEST_PDF")
            .unwrap_or_else(|_| "/home/timo/edgequake/B2A.pdf".to_string());
        let pdf_bytes = std::fs::read(&pdf_path).expect("read PDF");
        let images = render_pdf_to_images(&pdf_bytes).expect("render pages");
        let page5 = images.get(4).expect("page 5").clone();
        let model_dir = model_cache_dir().expect("model dir");

        let mut vlm_cfg = VlmClientConfig::from_env();
        if let Ok(m) = std::env::var("EDGEQUAKE_TEST_VLM_MODEL") {
            vlm_cfg.model = Some(m);
        }
        println!(
            "VLM: base_url={} model={:?}",
            vlm_cfg.base_url, vlm_cfg.model
        );
        let backend = VlmClientBackend::new(vlm_cfg);

        let candidates = [
            ("V2", "pp-doclayoutv2", "pp-doclayoutv2.onnx"),
            ("plus-L", "pp-doclayout_plus-l", "pp-doclayout_plus-l.onnx"),
        ];

        for (label, model_name, file) in candidates {
            let path = model_dir.join(file);
            if !path.exists() {
                eprintln!("[{label}] model not found, skipping");
                continue;
            }
            let predictor = match oar_ocr_core::predictors::LayoutDetectionPredictor::builder()
                .model_name(model_name)
                .build(&path)
            {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("[{label}] build predictor failed: {e}");
                    continue;
                }
            };
            let parser = DocParser::new(&backend);

            println!("\n========== {label} page 5 DocParser output (default pad 0.0) ==========");
            let result = match parser.parse(&predictor, page5.clone()) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[{label}] DocParser.parse failed: {e}");
                    continue;
                }
            };
            for (idx, elem) in result.layout_elements.iter().enumerate() {
                let text = elem.text.as_deref().unwrap_or("<no-text>");
                let preview: String = text.chars().take(500).collect();
                println!(
                    "  #{idx} type={:?} label={:?} bbox=[x:{:.0}-{:.0} y:{:.0}-{:.0}] score={:.3} textlen={}",
                    elem.element_type,
                    elem.label,
                    elem.bbox.x_min(),
                    elem.bbox.x_max(),
                    elem.bbox.y_min(),
                    elem.bbox.y_max(),
                    elem.confidence,
                    text.len()
                );
                if !preview.is_empty() && text != "<no-text>" {
                    println!("     text: {preview:?}");
                }
            }

            println!("\n--- {label} page 5 full markdown ---");
            println!("{}", result.to_markdown());

            // --- Same model with crop_pad_ratio=0.02 (restores missing title) ---
            let padded_cfg = oar_ocr_vl::doc_parser::DocParserConfig {
                crop_pad_ratio: 0.02,
                ..Default::default()
            };
            let padded_parser = DocParser::with_config(&backend, padded_cfg);
            println!("\n========== {label} page 5 DocParser output (pad 0.02) ==========");
            let padded = match padded_parser.parse(&predictor, page5.clone()) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[{label}] padded parse failed: {e}");
                    continue;
                }
            };
            for (idx, elem) in padded.layout_elements.iter().enumerate() {
                if elem.element_type
                    != oar_ocr_core::domain::structure::LayoutElementType::Algorithm
                {
                    continue;
                }
                let text = elem.text.as_deref().unwrap_or("<no-text>");
                let preview: String = text.chars().take(800).collect();
                println!(
                    "  ALGO #{idx} bbox=[x:{:.0}-{:.0} y:{:.0}-{:.0}] textlen={}",
                    elem.bbox.x_min(),
                    elem.bbox.x_max(),
                    elem.bbox.y_min(),
                    elem.bbox.y_max(),
                    text.len()
                );
                println!("     text: {preview:?}");
            }
        }
    }

    /// VLM penalty-sweep harness. Runs the full `VlmOcrConverter` pipeline
    /// (PP-DocLayoutV2 + remote VLM per layout element) on the whole PDF
    /// for each candidate penalty configuration. Each config's full
    /// markdown lands in `/tmp/sweep-<label>-full.md`; a summary table
    /// prints at the end.
    ///
    /// Configs exercised via env vars picked up by `VlmClientConfig::from_env`:
    ///   - `EDGEQUAKE_VLM_FREQUENCY_PENALTY`
    ///   - `EDGEQUAKE_VLM_PRESENCE_PENALTY`
    ///   - `EDGEQUAKE_VLM_REPEAT_PENALTY`
    ///   - `EDGEQUAKE_VLM_STOP_SEQUENCES` (pipe-separated)
    /// Set inside the loop per config; restored after each run.
    ///
    /// Goal: find a config that kills Algorithm 1's `(\underbrace{1, \ldots,
    /// 0, \ldots, ...}` runaway without distorting other algorithms.
    ///
    /// Run with:
    ///   OPENAI_COMPATIBLE_BASE_URL=https://ai.tail59ea6b.ts.net/v1 \
    ///   EDGEQUAKE_TEST_VLM_MODEL=GLM-OCR \
    ///   EDGEQUAKE_OAR_OCR_MODEL_DIR=/home/timo/.edgequake/oar-ocr-models \
    ///   EDGEQUAKE_TEST_PDF=/tmp/euston.pdf \
    ///   cargo test -p edgequake-pdf --release --lib \
    ///     layout_model_comparison::vlm_penalty_sweep -- --ignored --nocapture
    ///
    /// Expect ~5–10 min per config on a 20-page paper (20 pages × ~10
    /// layout elements × ~2–6 s per VLM call, concurrency-limited by
    /// `EDGEQUAKE_PDF_CONCURRENCY`). 4 configs → 20–40 min total.
    #[test]
    #[ignore]
    fn vlm_penalty_sweep() {
        use super::super::PdfConversionConfig;
        use std::sync::Arc;

        let pdf_path =
            std::env::var("EDGEQUAKE_TEST_PDF").unwrap_or_else(|_| "/tmp/euston.pdf".to_string());
        let vlm_model = std::env::var("EDGEQUAKE_TEST_VLM_MODEL").ok();
        let vlm_base_url = std::env::var("OPENAI_COMPATIBLE_BASE_URL")
            .or_else(|_| std::env::var("OPENAI_BASE_URL"))
            .ok();
        let pdf_bytes = std::fs::read(&pdf_path).expect("read PDF");

        eprintln!(
            "[sweep] pdf={pdf_path} bytes={} model={vlm_model:?} base_url={vlm_base_url:?}",
            pdf_bytes.len()
        );

        // Kept small (4 configs) so the whole sweep on a 20-page paper
        // finishes in ~30 min. Each tuple: (label, freq, pres, repeat,
        // stops) — 4 candidates chosen to cover the search space:
        //   - baseline      : reference (no penalties)
        //   - presence-mid  : OpenAI presence_penalty@0.3 (binary, less
        //                     destructive than frequency on legit math
        //                     repetition like `\ldots` enumerations)
        //   - light-repeat  : llama.cpp repeat_penalty@1.03 (native;
        //                     independent knob if GLM-OCR backend
        //                     respects llama.cpp flags over OpenAI ones)
        //   - stop-runaway  : no penalty, just a hard `stop` on the
        //                     exact pathological 3-rep pattern — the
        //                     least-invasive option if the backend
        //                     honours `stop` arrays
        let configs: Vec<(&str, f32, f32, f32, Vec<String>)> = vec![
            ("baseline", 0.0, 0.0, 1.0, vec![]),
            ("presence-mid", 0.0, 0.3, 1.0, vec![]),
            ("light-repeat", 0.0, 0.0, 1.03, vec![]),
            (
                "stop-runaway",
                0.0,
                0.0,
                1.0,
                vec!["0, \\ldots, 0, \\ldots, 0, \\ldots".to_string()],
            ),
        ];

        #[derive(Debug)]
        struct Metric {
            label: &'static str,
            chars: usize,
            ldots_count: usize,
            longest_ldots_run: usize,
            display_math_blocks: usize,
            for_loops: usize,
            has_algo1_heading: bool,
            has_algo2_heading: bool,
            has_algo3_heading: bool,
            elapsed_secs: f32,
        }

        let mut metrics: Vec<Metric> = Vec::new();

        // Shared tokio runtime — reuse across configs.
        let rt = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .expect("build tokio runtime");

        for (label, freq, pres, repeat, stops) in &configs {
            // Per-config env vars — VlmClientConfig::from_env picks them
            // up inside `VlmOcrConverter::convert`. Preserve originals so
            // subsequent configs don't leak state.
            let orig_freq = std::env::var("EDGEQUAKE_VLM_FREQUENCY_PENALTY").ok();
            let orig_pres = std::env::var("EDGEQUAKE_VLM_PRESENCE_PENALTY").ok();
            let orig_rep = std::env::var("EDGEQUAKE_VLM_REPEAT_PENALTY").ok();
            let orig_stop = std::env::var("EDGEQUAKE_VLM_STOP_SEQUENCES").ok();

            // SAFETY: test runs single-threaded by default (Cargo serial
            // test) and the env mutation is scoped to one iteration.
            unsafe {
                std::env::set_var("EDGEQUAKE_VLM_FREQUENCY_PENALTY", freq.to_string());
                std::env::set_var("EDGEQUAKE_VLM_PRESENCE_PENALTY", pres.to_string());
                std::env::set_var("EDGEQUAKE_VLM_REPEAT_PENALTY", repeat.to_string());
                if stops.is_empty() {
                    std::env::remove_var("EDGEQUAKE_VLM_STOP_SEQUENCES");
                } else {
                    std::env::set_var("EDGEQUAKE_VLM_STOP_SEQUENCES", stops.join("|"));
                }
            }

            eprintln!("\n[sweep] ============================================================");
            eprintln!("[sweep] {label}: freq={freq} pres={pres} repeat={repeat} stops={stops:?}");

            let converter = VlmOcrConverter;
            let cfg = PdfConversionConfig {
                vlm_base_url: vlm_base_url.clone(),
                vlm_model: vlm_model.clone(),
                vision: None,
                algorithm_block_sink: None,
                ..Default::default()
            };

            let t0 = std::time::Instant::now();
            let md = match rt.block_on(converter.convert(&pdf_bytes, &cfg)) {
                Ok(m) => m,
                Err(e) => {
                    eprintln!("[sweep] {label}: convert FAILED: {e}");
                    // Restore env before aborting.
                    restore_env("EDGEQUAKE_VLM_FREQUENCY_PENALTY", orig_freq);
                    restore_env("EDGEQUAKE_VLM_PRESENCE_PENALTY", orig_pres);
                    restore_env("EDGEQUAKE_VLM_REPEAT_PENALTY", orig_rep);
                    restore_env("EDGEQUAKE_VLM_STOP_SEQUENCES", orig_stop);
                    continue;
                }
            };
            let elapsed = t0.elapsed();

            // Restore env for next iteration.
            restore_env("EDGEQUAKE_VLM_FREQUENCY_PENALTY", orig_freq);
            restore_env("EDGEQUAKE_VLM_PRESENCE_PENALTY", orig_pres);
            restore_env("EDGEQUAKE_VLM_REPEAT_PENALTY", orig_rep);
            restore_env("EDGEQUAKE_VLM_STOP_SEQUENCES", orig_stop);

            let out_path = format!("/tmp/sweep-{label}-full.md");
            std::fs::write(&out_path, &md).expect("write sweep output");

            let ldots_total = md.matches("\\ldots").count();
            let (longest_run, _) = longest_runaway_run(&md);
            let display_math = md.matches("$$").count();
            let for_loops = md.matches("for ").count() + md.matches("**for**").count();
            let has_algo1 = md.contains("Algorithm 1 Component");
            let has_algo2 = md.contains("Algorithm 2 Homomorphic");
            let has_algo3 = md.contains("Algorithm 3 Homomorphic");

            eprintln!(
                "[sweep] {label}: elapsed={:.0}s chars={} ldots={} longest_run={} $$={} for={} algo1={} algo2={} algo3={} → {}",
                elapsed.as_secs_f32(),
                md.len(),
                ldots_total,
                longest_run,
                display_math,
                for_loops,
                has_algo1,
                has_algo2,
                has_algo3,
                out_path
            );

            metrics.push(Metric {
                label: Box::leak(label.to_string().into_boxed_str()),
                chars: md.len(),
                ldots_count: ldots_total,
                longest_ldots_run: longest_run,
                display_math_blocks: display_math,
                for_loops,
                has_algo1_heading: has_algo1,
                has_algo2_heading: has_algo2,
                has_algo3_heading: has_algo3,
                elapsed_secs: elapsed.as_secs_f32(),
            });
        }

        let _ = Arc::new(rt); // keep rt alive until after the loop

        println!("\n========== VLM penalty sweep summary ==========\n");
        println!(
            "{:<14} | {:>5} | {:>7} | {:>6} | {:>12} | {:>5} | {:>5} | {:>2} {:>2} {:>2}",
            "config", "t(s)", "chars", "\\ldots", "longest_run", "$$blk", "for", "a1", "a2", "a3"
        );
        println!("{:-<90}", "");
        for m in &metrics {
            println!(
                "{:<14} | {:>5.0} | {:>7} | {:>6} | {:>12} | {:>5} | {:>5} | {:>2} {:>2} {:>2}",
                m.label,
                m.elapsed_secs,
                m.chars,
                m.ldots_count,
                m.longest_ldots_run,
                m.display_math_blocks,
                m.for_loops,
                m.has_algo1_heading as u8,
                m.has_algo2_heading as u8,
                m.has_algo3_heading as u8,
            );
        }
        println!(
            "\n(Full per-config markdown: /tmp/sweep-<label>-full.md — diff any two with `diff -u`.)"
        );
    }

    fn restore_env(key: &str, original: Option<String>) {
        // SAFETY: same constraints as the set_var above — test is single-
        // threaded.
        unsafe {
            match original {
                Some(v) => std::env::set_var(key, v),
                None => std::env::remove_var(key),
            }
        }
    }

    /// Return the longest unbroken run of `X, \ldots,` patterns in `s`
    /// (counts reps). Used by the penalty sweep to quantify runaway
    /// severity per config. Threshold-free: baseline hits ≥500 reps,
    /// a healthy config drops to single digits.
    fn longest_runaway_run(s: &str) -> (usize, usize) {
        // Slide through the string looking for `(0|1), \ldots, ` tokens
        // chained back-to-back. `best` = longest run found; `best_start`
        // = byte offset of its first token.
        let needle_0 = "0, \\ldots, ";
        let needle_1 = "1, \\ldots, ";
        let mut best = 0usize;
        let mut best_start = 0usize;
        let mut i = 0;
        while i < s.len() {
            let at = &s[i..];
            if at.starts_with(needle_0) || at.starts_with(needle_1) {
                let start = i;
                let mut count = 0;
                let mut j = i;
                while j < s.len() {
                    let seg = &s[j..];
                    if seg.starts_with(needle_0) {
                        count += 1;
                        j += needle_0.len();
                    } else if seg.starts_with(needle_1) {
                        count += 1;
                        j += needle_1.len();
                    } else {
                        break;
                    }
                }
                if count > best {
                    best = count;
                    best_start = start;
                }
                i = j.max(i + 1);
            } else {
                // Advance by one char (UTF-8 safe).
                i += s[i..].chars().next().map(|c| c.len_utf8()).unwrap_or(1);
            }
        }
        (best, best_start)
    }
}
