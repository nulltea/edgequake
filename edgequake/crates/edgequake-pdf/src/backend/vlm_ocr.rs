//! VLM-OCR backend: PP-DocLayoutV2 layout detection + remote VLM recognition.
//!
//! Uses the [`oar-ocr-core`] layout predictor (PP-DocLayoutV2, 25 classes, 8-dim
//! reading order) for fast ONNX-based layout detection, then sends each cropped
//! region to an external VLM server (llama.cpp, vLLM, Ollama, etc.) via the
//! [`VlmClientBackend`] HTTP client for text/formula/table recognition.
//!
//! Configuration:
//!   - Layout ONNX: shared with OAR-OCR (`$EDGEQUAKE_OAR_OCR_MODEL_DIR/pp-doclayoutv2.onnx`)
//!   - VLM server: `$EDGEQUAKE_LLAMACPP_URL` (default `http://localhost:8081`)
//!   - Model name: `$EDGEQUAKE_LLAMACPP_MODEL` (optional)
//!   - Timeout: `$EDGEQUAKE_LLAMACPP_TIMEOUT` (default 120s)

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
            // 1. Resolve layout model directory (shared with OAR-OCR).
            let oar_model_dir = super::oar_ocr::model_cache_dir()?;
            let layout_path = oar_model_dir.join(LAYOUT_MODEL);
            if !layout_path.exists() {
                return Err(PdfConversionError::Backend(format!(
                    "VLM-OCR: layout model not found at {}. Run `scripts/download-oar-ocr-models.sh`.",
                    layout_path.display()
                )));
            }

            // 2. Render PDF pages to images.
            let images = super::oar_ocr::render_pdf_to_images(&pdf_bytes)?;
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
