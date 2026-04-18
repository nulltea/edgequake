pub mod backend;
pub mod error;
pub mod links;
pub mod repos;

pub use backend::{
    create_pdf_converter, detect_algorithm_blocks, AlgorithmBlock, PdfConversionConfig,
    PdfConverter, PdfParserBackend, VisionConversionConfig,
};
pub use error::PdfConversionError;
pub use links::{
    extract_links, LinkExtraction, PdfLinkAnnotation, PdfLinkError, ReferenceBoundary,
};
pub use repos::{detect_repos, DetectedRepo, RepoHost};
