//! OAR-OCR-VL backend: PP-DocLayoutV2 + UniRec VLM for unified document recognition.
//!
//! Uses the [`oar-ocr-vl`](https://crates.io/crates/oar-ocr-vl) crate with:
//!   - **PP-DocLayoutV2** (25 classes, 8-dim reading order) for layout detection
//!   - **UniRec** (0.1B, ~536 MB) for unified text/formula/table recognition
//!
//! UniRec is a unified Vision-Language model that handles text, formulas, and tables
//! in a single pass — it doesn't need task-specific prompts. This means algorithm
//! boxes get the same quality as formula regions, because UniRec natively understands
//! mixed text+math content.
//!
//! Runs on CPU via Candle (pure Rust). Expect ~5-15 seconds per page depending on
//! content density. Significantly slower than the staged OAR-OCR pipeline but with
//! better LaTeX output for inline math.
//!
//! Model directories:
//!   - Layout ONNX: shared with OAR-OCR (`$EDGEQUAKE_OAR_OCR_MODEL_DIR/pp-doclayoutv2.onnx`)
//!   - UniRec weights: `$EDGEQUAKE_OAR_OCR_VL_MODEL_DIR` (default `$HOME/.cache/edgequake/unirec-0.1b`)
//!     Expected contents: `config.json`, `tokenizer.json`, `model.safetensors`

use std::path::PathBuf;

use async_trait::async_trait;
use tracing::{info, warn};

use super::{PdfConversionConfig, PdfConverter};
use crate::error::PdfConversionError;

/// OAR-OCR-VL powered PDF → Markdown converter (PP-DocLayoutV2 + UniRec).
#[derive(Debug, Default)]
pub struct OarOcrVlConverter;

const LAYOUT_MODEL: &str = "pp-doclayoutv2.onnx";
const LAYOUT_MODEL_NAME: &str = "pp-doclayoutv2";

#[async_trait]
impl PdfConverter for OarOcrVlConverter {
    async fn convert(
        &self,
        pdf_bytes: &[u8],
        _config: &PdfConversionConfig,
    ) -> Result<String, PdfConversionError> {
        let pdf_bytes = pdf_bytes.to_vec();

        tokio::task::spawn_blocking(move || {
            // 1. Resolve model directories.
            let oar_model_dir = super::oar_ocr::model_cache_dir()?;
            let vl_model_dir = vl_model_dir()?;

            let layout_path = oar_model_dir.join(LAYOUT_MODEL);
            if !layout_path.exists() {
                return Err(PdfConversionError::Backend(format!(
                    "OAR-OCR-VL: layout model not found at {}. Run `scripts/download-oar-ocr-models.sh`.",
                    layout_path.display()
                )));
            }

            let unirec_config = vl_model_dir.join("config.json");
            if !unirec_config.exists() {
                return Err(PdfConversionError::Backend(format!(
                    "OAR-OCR-VL: UniRec model not found at {}. Download from HuggingFace (topdu/unirec-0.1b).",
                    vl_model_dir.display()
                )));
            }

            // 2. Render PDF pages to images.
            let images = super::oar_ocr::render_pdf_to_images(&pdf_bytes)?;
            if images.is_empty() {
                return Err(PdfConversionError::EmptyOutput(
                    "OAR-OCR-VL: PDF has no pages",
                ));
            }
            let page_count = images.len();
            info!(pages = page_count, "OAR-OCR-VL: rendered PDF pages");

            // 3. Build layout predictor (ONNX, fast).
            let layout_predictor =
                oar_ocr_core::predictors::LayoutDetectionPredictor::builder()
                    .model_name(LAYOUT_MODEL_NAME)
                    .build(&layout_path)
                    .map_err(|e| {
                        PdfConversionError::Backend(format!(
                            "OAR-OCR-VL: layout predictor failed: {e}"
                        ))
                    })?;

            // 4. Load UniRec VLM (Candle, CPU).
            let device = candle_core::Device::Cpu;
            let unirec = oar_ocr_vl::unirec::UniRec::from_dir(&vl_model_dir, device).map_err(
                |e| PdfConversionError::Backend(format!("OAR-OCR-VL: UniRec load failed: {e}")),
            )?;
            info!("OAR-OCR-VL: UniRec model loaded");

            // 5. Create document parser.
            let parser = oar_ocr_vl::doc_parser::DocParser::new(&unirec);

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
                        warn!(page = i + 1, error = %e, "OAR-OCR-VL: page failed, skipping");
                    }
                }
            }

            if markdown.trim().is_empty() {
                return Err(PdfConversionError::EmptyOutput(
                    "OAR-OCR-VL returned no text",
                ));
            }

            info!(
                markdown_len = markdown.len(),
                pages_succeeded = succeeded,
                pages_total = page_count,
                "OAR-OCR-VL conversion completed"
            );

            Ok(markdown)
        })
        .await
        .map_err(|e| PdfConversionError::Internal(e.to_string()))?
    }

    fn backend_name(&self) -> &'static str {
        "oarocrvl"
    }
}

fn vl_model_dir() -> Result<PathBuf, PdfConversionError> {
    if let Ok(dir) = std::env::var("EDGEQUAKE_OAR_OCR_VL_MODEL_DIR") {
        if !dir.is_empty() {
            let p = PathBuf::from(dir);
            std::fs::create_dir_all(&p).map_err(|e| {
                PdfConversionError::Internal(format!(
                    "Failed to create OAR-OCR-VL model dir: {e}"
                ))
            })?;
            return Ok(p);
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        let p = PathBuf::from(home).join(".cache/edgequake/unirec-0.1b");
        std::fs::create_dir_all(&p).map_err(|e| {
            PdfConversionError::Internal(format!(
                "Failed to create OAR-OCR-VL model dir: {e}"
            ))
        })?;
        return Ok(p);
    }
    let p = PathBuf::from("/tmp/edgequake-unirec-0.1b");
    std::fs::create_dir_all(&p).map_err(|e| {
        PdfConversionError::Internal(format!("Failed to create OAR-OCR-VL model dir: {e}"))
    })?;
    Ok(p)
}
