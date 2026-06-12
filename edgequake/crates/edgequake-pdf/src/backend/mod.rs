mod edgeparse;
mod figure_extract;
mod table_extract;
mod vision;
pub mod vlm_client;
mod vlm_ocr;

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::PdfConversionError;

pub use edgeparse::EdgeParsePdfConverter;
pub use table_extract::{
    inline_table_placeholders, mark_table_placeholders, render_table_embed_text,
    render_table_markdown, render_table_rerank_text, table_refs_in,
};
pub use vision::VisionPdfConverter;
pub use vlm_ocr::{detect_algorithm_blocks, AlgorithmBlock, VlmOcrConverter};

/// A figure extracted from a PDF during VLM-OCR conversion: encoded image
/// bytes of the cropped layout region plus the VLM-recognized caption and the
/// metadata needed to re-anchor it to the source markdown (page, reading
/// order).
///
/// The pipeline writes a stable `![figure:<id>](caption: …)` placeholder into
/// the markdown for each extracted figure; the chunker pairs that placeholder
/// against this struct to emit a media-bearing chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedFigure {
    /// Stable extractor-side id, `fig_{page}_{order_index}`.
    pub id: String,
    /// Encoded crop of the figure region (lossy WEBP — see `figure_extract`).
    pub image_bytes: Vec<u8>,
    /// MIME of `image_bytes` — currently always `"image/webp"`.
    pub mime: String,
    /// VLM-generated caption / description for the figure.
    pub caption: String,
    /// 1-based page number the figure was cropped from.
    pub page: u32,
    /// Reading-order index within the page (preserves intra-page order).
    pub order_index: u32,
}

/// A table extracted from a PDF during VLM-OCR conversion: VLM-rendered HTML
/// plus the algorithmically-parsed `{ headers, rows }` structure and the
/// caption needed to display the table in the gallery. The HTML lives in
/// `chunks.table_html`; `rows` lives in `chunks.table_rows` as JSONB.
///
/// The pipeline writes a `![tbl_{page}_{order_index}](edgequake-table)`
/// placeholder into the markdown so the chunker can emit a Table-kind chunk
/// at the table's source position. Inline rendering on the frontend swaps the
/// placeholder back for the rendered HTML on read.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedTable {
    /// Stable extractor-side id, `tbl_{page}_{order_index}`.
    pub id: String,
    /// VLM-rendered HTML for the table (e.g. `<table border="1">…</table>`).
    pub html: String,
    /// Algorithmically-parsed header row, if a `<th>` row was detected.
    pub headers: Vec<String>,
    /// Algorithmically-parsed body rows. Each inner Vec is one `<tr>`.
    /// Empty when the parser couldn't make sense of the HTML; the raw HTML
    /// in `html` is still usable for display and classification.
    pub rows: Vec<Vec<String>>,
    /// Caption text (TableTitle / FigureTableChartTitle / FigureTitle below or
    /// above the table). Empty when no nearby caption was detected.
    pub caption: String,
    /// 1-based page number the table was extracted from.
    pub page: u32,
    /// Reading-order index within the page (preserves intra-page order).
    pub order_index: u32,
}

/// Runtime-selectable PDF parser backend.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PdfParserBackend {
    #[default]
    Vision,
    EdgeParse,
    /// VLM-OCR backend (PP-DocLayoutV2 layout + remote VLM recognition).
    /// Uses an external VLM server (llama.cpp, vLLM, Ollama) for GPU-accelerated
    /// text/formula/table recognition.
    VlmOcr,
}

impl PdfParserBackend {
    pub fn from_env_str(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "vision" | "llm" => Some(Self::Vision),
            "edgeparse" | "edge-parse" | "edge_parse" => Some(Self::EdgeParse),
            "vlmocr" | "vlm-ocr" | "vlm_ocr" => Some(Self::VlmOcr),
            // Legacy aliases — map removed backends to closest alternatives.
            "kreuzberg" | "pdfextract" | "pdf-extract" | "pdf_extract" => Some(Self::EdgeParse),
            "oarocr" | "oar-ocr" | "oar_ocr" | "oarocrvl" | "oar-ocr-vl" | "oar_ocr_vl" => {
                Some(Self::VlmOcr)
            }
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
            Self::VlmOcr => "vlmocr",
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
#[derive(Clone, Default)]
pub struct PdfConversionConfig {
    pub page_count_hint: Option<usize>,
    pub table_method: Option<String>,
    pub filename: Option<String>,
    pub vision: Option<VisionConversionConfig>,
    /// VLM-OCR: base URL of the OpenAI-compatible VLM server.
    pub vlm_base_url: Option<String>,
    /// VLM-OCR: model name to send in the API request.
    pub vlm_model: Option<String>,
    /// VLM-OCR: if set, algorithm blocks detected during conversion are collected here.
    /// The processor reads this after conversion to trigger automatic algorithm extraction.
    pub algorithm_block_sink: Option<Arc<std::sync::Mutex<Vec<AlgorithmBlock>>>>,
    /// VLM-OCR: if set, image/chart/seal figures cropped during conversion are
    /// pushed here as PNG-encoded payloads with their VLM caption. Each figure
    /// also appears in the returned markdown as
    /// `![figure:<id>](caption: <caption>)` so the chunker can pair the two.
    /// Setting this opts the converter into figure extraction; leaving it
    /// `None` preserves the legacy text-only behaviour.
    pub figure_sink: Option<Arc<std::sync::Mutex<Vec<ExtractedFigure>>>>,
    /// VLM-OCR: if set, tables detected during layout parsing are recorded
    /// here as VLM-rendered HTML + parsed rows + caption. The returned
    /// markdown gets a `![tbl_{page}_{order_index}](edgequake-table)`
    /// placeholder in the table's source position so the chunker can emit a
    /// Table-kind chunk paired by id. `None` preserves text-only behaviour
    /// (raw HTML stays inline in the markdown).
    pub table_sink: Option<Arc<std::sync::Mutex<Vec<ExtractedTable>>>>,
}

impl std::fmt::Debug for PdfConversionConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PdfConversionConfig")
            .field("page_count_hint", &self.page_count_hint)
            .field("table_method", &self.table_method)
            .field("filename", &self.filename)
            .field("vlm_base_url", &self.vlm_base_url)
            .field("vlm_model", &self.vlm_model)
            .field(
                "algorithm_block_sink",
                &self.algorithm_block_sink.as_ref().map(|_| "<sink>"),
            )
            .field(
                "figure_sink",
                &self.figure_sink.as_ref().map(|_| "<sink>"),
            )
            .field(
                "table_sink",
                &self.table_sink.as_ref().map(|_| "<sink>"),
            )
            .finish()
    }
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
        PdfParserBackend::VlmOcr => Arc::new(VlmOcrConverter),
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
        assert_eq!(PdfParserBackend::VlmOcr.as_str(), "vlmocr");
        // Legacy aliases map to closest backends
        assert_eq!(
            PdfParserBackend::from_env_str("kreuzberg"),
            Some(PdfParserBackend::EdgeParse)
        );
        assert_eq!(
            PdfParserBackend::from_env_str("pdfextract"),
            Some(PdfParserBackend::EdgeParse)
        );
        assert_eq!(
            PdfParserBackend::from_env_str("oarocr"),
            Some(PdfParserBackend::VlmOcr)
        );
        assert_eq!(
            PdfParserBackend::from_env_str("oarocrvl"),
            Some(PdfParserBackend::VlmOcr)
        );
        assert_eq!(
            PdfParserBackend::from_env_str("vlm-ocr"),
            Some(PdfParserBackend::VlmOcr)
        );
    }
}
