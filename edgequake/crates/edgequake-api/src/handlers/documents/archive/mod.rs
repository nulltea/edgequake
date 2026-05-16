//! Document archive handlers.
//!
//! Archive is a soft-delete that preserves the original document row, its PDF,
//! the converted Markdown, extracted algorithms, and `document_repos` entries,
//! while dropping the bulky derived data: chunks, embeddings, KG entity/edge
//! contributions, and `reference_codebase_*` rows.

mod single;

pub use single::*;
