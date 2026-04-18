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
            // crop_pad_ratio=0.02 restores ~2% top/bottom padding around each
            // layout element's bbox. Without this, the plus-L layout model's
            // tight algorithm bboxes can crop JUST BELOW the box's title line
            // (e.g. "**Protocol 2** B2A protocol (Π₂)"), causing GLM-OCR to
            // return only the math body without the title. Verified with the
            // inspect_page5_vlm_output test on B2A.pdf: default pad 0.0 gives
            // 471 chars (no title); pad 0.02 gives 1448 chars (full structured
            // box). See commit message for details.
            let parser_cfg = oar_ocr_vl::doc_parser::DocParserConfig {
                crop_pad_ratio: 0.02,
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
                                    use oar_ocr_core::domain::structure::LayoutElementType;
                                    let algo_elements: Vec<_> = result
                                        .layout_elements
                                        .iter()
                                        .filter(|e| e.element_type == LayoutElementType::Algorithm)
                                        .cloned()
                                        .collect();
                                    info!(
                                        page = page_num,
                                        algo_count = algo_elements.len(),
                                        total_elements = result.layout_elements.len(),
                                        "VLM-OCR: algorithm elements on page"
                                    );
                                    if !algo_elements.is_empty() {
                                        if let Ok(mut blocks) = sink.lock() {
                                            for (idx, elem) in algo_elements.iter().enumerate() {
                                                let algo_md = oar_ocr_vl::utils::to_markdown(
                                                    std::slice::from_ref(elem),
                                                    &[],
                                                );
                                                let preview: String = algo_md
                                                    .chars()
                                                    .take(240)
                                                    .collect();
                                                info!(
                                                    page = page_num,
                                                    algo_idx = idx,
                                                    score = elem.confidence,
                                                    md_len = algo_md.len(),
                                                    preview = %preview,
                                                    "VLM-OCR: captured algorithm block"
                                                );
                                                if !algo_md.trim().is_empty() {
                                                    blocks.push(AlgorithmBlock {
                                                        page: page_num,
                                                        markdown: algo_md,
                                                    });
                                                }
                                            }
                                        }
                                    }
                                }

                                let md = result.to_markdown();
                                if md.trim().is_empty() { None } else { Some(md) }
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
                        info!(page = page_num, done = done, total = page_count, "VLM-OCR: page completed");

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
    use oar_ocr_core::domain::structure::LayoutElementType;

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
    info!(pages = images.len(), "Algorithm detection: rendered PDF pages");

    // 3. Build layout predictor
    let layout_predictor = oar_ocr_core::predictors::LayoutDetectionPredictor::builder()
        .model_name(LAYOUT_MODEL_NAME)
        .build(&layout_path)
        .map_err(|e| {
            PdfConversionError::Backend(format!("Algorithm detection: layout predictor failed: {e}"))
        })?;

    // 4. Build VLM backend + DocParser
    info!(
        base_url = %vlm_config.base_url,
        model = ?vlm_config.model,
        "Algorithm detection: using VLM server"
    );
    let backend = VlmClientBackend::new(vlm_config);
    // crop_pad_ratio=0.02 — see comment in VlmOcrConverter::convert for why.
    let parser_cfg = oar_ocr_vl::doc_parser::DocParserConfig {
        crop_pad_ratio: 0.02,
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

                let algo_elements: Vec<_> = result
                    .layout_elements
                    .iter()
                    .filter(|e| e.element_type == LayoutElementType::Algorithm)
                    .cloned()
                    .collect();

                if algo_elements.is_empty() {
                    return Vec::new();
                }

                info!(
                    page = page_num,
                    count = algo_elements.len(),
                    done = page_num,
                    total = page_count,
                    "Algorithm detection: found algorithm blocks"
                );

                // One block per element for granular progress + cleaner LLM chunks.
                let mut page_blocks = Vec::new();
                for elem in &algo_elements {
                    let md = oar_ocr_vl::utils::to_markdown(std::slice::from_ref(elem), &[]);
                    if !md.trim().is_empty() {
                        page_blocks.push(AlgorithmBlock {
                            page: page_num,
                            markdown: md,
                        });
                    }
                }
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
    std::fs::create_dir_all(&p).map_err(|e| {
        PdfConversionError::Internal(format!("Failed to create model dir: {e}"))
    })?;
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

    image::RgbImage::from_raw(
        u32::from(pixmap.width()),
        u32::from(pixmap.height()),
        rgb,
    )
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
                let mut by_type: std::collections::BTreeMap<String, usize> =
                    Default::default();
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
            println!(
                "\n========== {label} page 5 DocParser output (pad 0.02) =========="
            );
            let padded = match padded_parser.parse(&predictor, page5.clone()) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[{label}] padded parse failed: {e}");
                    continue;
                }
            };
            for (idx, elem) in padded.layout_elements.iter().enumerate() {
                if elem.element_type != oar_ocr_core::domain::structure::LayoutElementType::Algorithm {
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
}
