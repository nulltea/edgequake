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

impl TaskType {
    /// Whether tasks of this type bypass the per-tenant concurrency limit
    /// (`MAX_TASKS_PER_TENANT`). Bypassed tasks are still bounded by the
    /// worker-pool size, but they don't compete with — or get gated by —
    /// the slot reserved for document-ingest work.
    ///
    /// Why: the document-upload pipeline runs with a low per-tenant cap
    /// (default 1) so it doesn't saturate LLM / embedding backends. The
    /// reference-code workflows (analyzer match-detection and codebase
    /// indexing) call out to a *different* sidecar / embedding model, so
    /// gating them behind the same 1-slot lock just makes them wait
    /// behind PDF ingests for no resource reason.
    pub const fn bypasses_tenant_concurrency_limit(&self) -> bool {
        matches!(
            self,
            Self::CodeReferenceAnalysis | Self::ReferenceCodebaseIndex
        )
    }
}

#[cfg(test)]
mod task_type_tests {
    use super::TaskType;

    #[test]
    fn bypass_flag_only_covers_code_tasks() {
        assert!(TaskType::CodeReferenceAnalysis.bypasses_tenant_concurrency_limit());
        assert!(TaskType::ReferenceCodebaseIndex.bypasses_tenant_concurrency_limit());

        assert!(!TaskType::Upload.bypasses_tenant_concurrency_limit());
        assert!(!TaskType::Insert.bypasses_tenant_concurrency_limit());
        assert!(!TaskType::PdfProcessing.bypasses_tenant_concurrency_limit());
        assert!(!TaskType::AlgorithmExtraction.bypasses_tenant_concurrency_limit());
        assert!(!TaskType::AlgorithmEmbedding.bypasses_tenant_concurrency_limit());
        assert!(!TaskType::RepoDetection.bypasses_tenant_concurrency_limit());
        assert!(!TaskType::TableClassification.bypasses_tenant_concurrency_limit());
    }
}
