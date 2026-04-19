//! Phase 2 reference-codebase RAG indexing.
//!
//! This module builds an EdgeQuake-native, separate codebase index over the
//! persisted clones owned by the `code-analyzer` sidecar. The implementation
//! follows the shape of Code-RAG/codegraph-rust systems: source files →
//! semantic symbols → structural edges → AST-aware chunks → vector index.

mod indexer;
mod storage;
mod treesitter;
mod types;

pub use indexer::{IndexLimits, ReferenceCodebaseIndexer};
pub use treesitter::{parse_file, PendingEdge, ParseOutput};
pub use storage::{PostgresReferenceCodebaseStorage, ReferenceCodebaseStorage};
pub use types::{
    CodebaseChunk, CodebaseEdge, CodebaseFile, CodebaseIndex, CodebaseIndexMode,
    CodebaseIndexStatus, CodebaseQueryHit, CodebaseSubgraph, CodebaseSymbol, IndexBuildOutput,
    SubgraphEdge, SubgraphNode,
};
