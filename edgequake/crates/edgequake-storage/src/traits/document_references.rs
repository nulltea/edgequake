//! Storage for parsed document references (citations).
//!
//! References are deterministic regex-parser output (see
//! `edgequake_pipeline::references`) — there is no review/approval status.
//! Rows are scoped by `(tenant_id, workspace_id, document_id)` and replaced
//! wholesale when a document is reprocessed.

use async_trait::async_trait;
use uuid::Uuid;

use crate::error::Result;

/// One persisted reference row.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DocumentReference {
    pub id: Uuid,
    pub document_id: String,
    /// Reference number from the source list marker (`[n]`, `n.`, `n)`).
    pub reference_number: i32,
    pub raw_text: String,
    pub doi: Option<String>,
    pub url: Option<String>,
}

/// A reference to be inserted (no id/timestamp yet).
#[derive(Debug, Clone)]
pub struct NewDocumentReference {
    pub reference_number: i32,
    pub raw_text: String,
    pub doi: Option<String>,
    pub url: Option<String>,
}

/// CRUD over the `document_references` table.
#[async_trait]
pub trait ReferenceStorage: Send + Sync {
    /// Replace all references for a document with `references` in a single
    /// transaction (delete-then-insert). Used by the inline ingestion step,
    /// which re-parses on every reprocess.
    async fn replace_references(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        document_id: &str,
        references: &[NewDocumentReference],
    ) -> Result<()>;

    /// List all references for a document, ordered by `reference_number`.
    async fn list_references(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        document_id: &str,
    ) -> Result<Vec<DocumentReference>>;

    /// Fetch the references for a document whose `reference_number` is in
    /// `numbers`. Used by retrieval-time chunk enrichment. Empty `numbers`
    /// returns an empty vec without hitting the database.
    async fn references_by_numbers(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        document_id: &str,
        numbers: &[i32],
    ) -> Result<Vec<DocumentReference>>;

    /// Delete all references for a document (e.g. on document delete).
    async fn delete_references(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        document_id: &str,
    ) -> Result<()>;
}
