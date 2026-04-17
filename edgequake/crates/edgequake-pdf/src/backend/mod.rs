mod edgeparse;
mod kreuzberg;
mod oar_ocr;
mod oar_ocr_vl;
mod vision;

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::PdfConversionError;

pub use edgeparse::EdgeParsePdfConverter;
pub use kreuzberg::KreuzbergConverter;
pub use oar_ocr::OarOcrConverter;
pub use oar_ocr_vl::OarOcrVlConverter;
pub use vision::VisionPdfConverter;

/// Runtime-selectable PDF parser backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PdfParserBackend {
    #[default]
    Vision,
    EdgeParse,
    /// Kreuzberg-based backend (PDFium + RT-DETR v2 layout + TATR tables).
    /// Was previously named "PdfExtract".
    Kreuzberg,
    /// OAR-OCR backend (PP-DocLayoutV2 + PP-OCRv5 + PP-FormulaNet + tables).
    OarOcr,
    /// OAR-OCR-VL backend (PP-DocLayoutV2 + UniRec 0.1B VLM).
    /// Slower but better LaTeX output for inline math.
    OarOcrVl,
}

impl PdfParserBackend {
    pub fn from_env_str(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "vision" | "llm" => Some(Self::Vision),
            "edgeparse" | "edge-parse" | "edge_parse" => Some(Self::EdgeParse),
            // Accept both new "kreuzberg" and legacy "pdfextract" names for backward compat.
            "kreuzberg" | "pdfextract" | "pdf-extract" | "pdf_extract" => Some(Self::Kreuzberg),
            "oarocr" | "oar-ocr" | "oar_ocr" => Some(Self::OarOcr),
            "oarocrvl" | "oar-ocr-vl" | "oar_ocr_vl" => Some(Self::OarOcrVl),
            _ => None,
        }
    }

    pub fn from_env() -> Option<Self> {
        std::env::var("EDGEQUAKE_PDF_PARSER_BACKEND")
            .ok()
            .filter(|value| !value.trim().is_empty())
            .and_then(|value| Self::from_env_str(&value))
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Vision => "vision",
            Self::EdgeParse => "edgeparse",
            Self::Kreuzberg => "kreuzberg",
            Self::OarOcr => "oarocr",
            Self::OarOcrVl => "oarocrvl",
        }
    }
}

/// Per-task vision conversion options preserved from the existing processor.
#[derive(Clone, Default)]
pub struct VisionConversionConfig {
    pub model: Option<String>,
    pub concurrency: Option<usize>,
    pub dpi: Option<u32>,
    pub checkpoint_dir: Option<String>,
    pub no_resume: bool,
    pub progress_callback: Option<Arc<dyn edgequake_pdf2md::ConversionProgressCallback>>,
}

impl std::fmt::Debug for VisionConversionConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VisionConversionConfig")
            .field("model", &self.model)
            .field("concurrency", &self.concurrency)
            .field("dpi", &self.dpi)
            .field("checkpoint_dir", &self.checkpoint_dir)
            .field("no_resume", &self.no_resume)
            .field(
                "progress_callback",
                &self.progress_callback.as_ref().map(|_| "<callback>"),
            )
            .finish()
    }
}

/// Configuration shared by PDF conversion backends.
#[derive(Clone, Debug, Default)]
pub struct PdfConversionConfig {
    pub page_count_hint: Option<usize>,
    pub table_method: Option<String>,
    pub filename: Option<String>,
    pub vision: Option<VisionConversionConfig>,
}

#[async_trait]
pub trait PdfConverter: Send + Sync {
    async fn convert(
        &self,
        pdf_bytes: &[u8],
        config: &PdfConversionConfig,
    ) -> Result<String, PdfConversionError>;

    fn backend_name(&self) -> &'static str;
}

pub fn create_pdf_converter(
    backend: PdfParserBackend,
    llm_provider: Option<Arc<dyn edgequake_llm::traits::LLMProvider>>,
) -> Arc<dyn PdfConverter> {
    match backend {
        PdfParserBackend::Vision => Arc::new(VisionPdfConverter::new(llm_provider)),
        PdfParserBackend::EdgeParse => Arc::new(EdgeParsePdfConverter),
        PdfParserBackend::Kreuzberg => Arc::new(KreuzbergConverter),
        PdfParserBackend::OarOcr => Arc::new(OarOcrConverter),
        PdfParserBackend::OarOcrVl => Arc::new(OarOcrVlConverter),
    }
}

#[cfg(test)]
mod tests {
    use super::PdfParserBackend;

    #[test]
    fn backend_env_aliases_roundtrip() {
        assert_eq!(
            PdfParserBackend::from_env_str("vision"),
            Some(PdfParserBackend::Vision)
        );
        assert_eq!(
            PdfParserBackend::from_env_str("edge-parse"),
            Some(PdfParserBackend::EdgeParse)
        );
        assert_eq!(PdfParserBackend::Vision.as_str(), "vision");
        assert_eq!(PdfParserBackend::EdgeParse.as_str(), "edgeparse");
        assert_eq!(PdfParserBackend::Kreuzberg.as_str(), "kreuzberg");
        assert_eq!(PdfParserBackend::OarOcr.as_str(), "oarocr");
        // Legacy "pdfextract" parses as Kreuzberg
        assert_eq!(
            PdfParserBackend::from_env_str("pdfextract"),
            Some(PdfParserBackend::Kreuzberg)
        );
        assert_eq!(
            PdfParserBackend::from_env_str("kreuzberg"),
            Some(PdfParserBackend::Kreuzberg)
        );
        assert_eq!(
            PdfParserBackend::from_env_str("oar-ocr"),
            Some(PdfParserBackend::OarOcr)
        );
    }
}
