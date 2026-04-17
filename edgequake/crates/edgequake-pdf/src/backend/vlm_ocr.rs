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
use std::sync::Arc;

use async_trait::async_trait;
use tracing::{info, warn};

use super::vlm_client::{VlmClientBackend, VlmClientConfig};
use super::{PdfConversionConfig, PdfConverter};
use crate::error::PdfConversionError;

const LAYOUT_MODEL: &str = "pp-doclayoutv2.onnx";
const LAYOUT_MODEL_NAME: &str = "pp-doclayoutv2";

/// VLM-OCR powered PDF → Markdown converter (PP-DocLayoutV2 + remote VLM).
#[derive(Debug, Default)]
pub struct VlmOcrConverter;

#[async_trait]
impl PdfConverter for VlmOcrConverter {
    async fn convert(
        &self,
        pdf_bytes: &[u8],
        _config: &PdfConversionConfig,
    ) -> Result<String, PdfConversionError> {
        let pdf_bytes = pdf_bytes.to_vec();

        tokio::task::spawn_blocking(move || {
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

            // 3. Build layout predictor (ONNX, fast).
            let layout_predictor = oar_ocr_core::predictors::LayoutDetectionPredictor::builder()
                .model_name(LAYOUT_MODEL_NAME)
                .build(&layout_path)
                .map_err(|e| {
                    PdfConversionError::Backend(format!("VLM-OCR: layout predictor failed: {e}"))
                })?;

            // 4. Create VLM client backend from env vars.
            let vlm_config = VlmClientConfig::from_env();
            info!(
                base_url = %vlm_config.base_url,
                model = ?vlm_config.model,
                "VLM-OCR: connecting to VLM server"
            );
            let backend = VlmClientBackend::new(vlm_config);

            // 5. Create document parser.
            let parser = oar_ocr_vl::doc_parser::DocParser::new(&backend);

            // 6. Process each page.
            let mut markdown = String::new();
            let mut succeeded = 0usize;
            for (i, image) in images.into_iter().enumerate() {
                match parser.parse(&layout_predictor, image) {
                    Ok(result) => {
                        if !markdown.is_empty() {
                            markdown.push_str("\n\n");
                        }
                        markdown.push_str(&result.to_markdown());
                        succeeded += 1;
                    }
                    Err(e) => {
                        warn!(page = i + 1, error = %e, "VLM-OCR: page failed, skipping");
                    }
                }
            }

            if markdown.trim().is_empty() {
                return Err(PdfConversionError::EmptyOutput("VLM-OCR returned no text"));
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
