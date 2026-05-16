//! Task status and type enums.
//!
//! Defines the lifecycle states (TaskStatus) and classification (TaskType)
//! for background tasks in the processing pipeline.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Task status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskStatus {
    Pending,
    Processing,
    Indexed,
    Failed,
    Cancelled,
}

impl fmt::Display for TaskStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Processing => write!(f, "processing"),
            Self::Indexed => write!(f, "indexed"),
            Self::Failed => write!(f, "failed"),
            Self::Cancelled => write!(f, "cancelled"),
        }
    }
}

/// Task type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskType {
    Upload,
    Insert,
    Scan,
    Reindex,
    PdfProcessing,
    AlgorithmExtraction,
    AlgorithmEmbedding,
    RepoDetection,
    CodeReferenceAnalysis,
    ReferenceCodebaseIndex,
    /// Classify extracted tables (performance | quality | complexity | other)
    /// after algorithm extraction has populated the document. Document-scoped
    /// and idempotent: runs over `chunks` rows with `kind='table'` and
    /// `table_type IS NULL`.
    TableClassification,
}

impl fmt::Display for TaskType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Upload => write!(f, "upload"),
            Self::Insert => write!(f, "insert"),
            Self::Scan => write!(f, "scan"),
            Self::Reindex => write!(f, "reindex"),
            Self::PdfProcessing => write!(f, "pdf_processing"),
            Self::AlgorithmExtraction => write!(f, "algorithm_extraction"),
            Self::AlgorithmEmbedding => write!(f, "algorithm_embedding"),
            Self::RepoDetection => write!(f, "repo_detection"),
            Self::CodeReferenceAnalysis => write!(f, "code_reference_analysis"),
            Self::ReferenceCodebaseIndex => write!(f, "reference_codebase_index"),
            Self::TableClassification => write!(f, "table_classification"),
        }
    }
}
