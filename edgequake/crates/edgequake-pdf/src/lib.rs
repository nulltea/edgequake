pub mod backend;
pub mod error;
pub mod latex_repair;
pub mod links;
pub mod repos;

pub use backend::{
    create_pdf_converter, detect_algorithm_blocks, AlgorithmBlock, ExtractedFigure,
    ExtractedTable, PdfConversionConfig, PdfConverter, PdfParserBackend, VisionConversionConfig,
};
pub use error::PdfConversionError;
pub use links::{
    extract_links, strip_references_section, LinkExtraction, PdfLinkAnnotation, PdfLinkError,
    ReferenceBoundary,
};
pub use repos::{detect_repos, DetectedRepo, RepoHost};
