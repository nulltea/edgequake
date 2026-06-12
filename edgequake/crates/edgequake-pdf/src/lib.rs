pub mod backend;
pub mod error;
pub mod latex_repair;
pub mod links;
pub mod repos;

pub use backend::{
    create_pdf_converter, detect_algorithm_blocks, inline_table_placeholders,
    mark_table_placeholders, render_table_embed_text, render_table_markdown,
    render_table_rerank_text, table_refs_in, AlgorithmBlock, ExtractedFigure, ExtractedTable,
    PdfConversionConfig, PdfConverter, PdfParserBackend, VisionConversionConfig,
};
pub use error::PdfConversionError;
pub use links::{
    extract_links, strip_references_section, LinkExtraction, PdfLinkAnnotation, PdfLinkError,
    ReferenceBoundary,
};
pub use repos::{detect_repos, DetectedRepo, RepoHost};
