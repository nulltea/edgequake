//! Request / response DTOs for code-reference endpoints.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use edgequake_agents::code_analysis::{
    ArtifactStatus, CodeArtifact, CodeReferenceRun, MatchConfidence, RunStatus,
};

// ── Requests ────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ReviewCodeArtifactRequest {
    pub status: ArtifactStatus,
}

// ── Responses ───────────────────────────────────────────────────────────────

/// Stable UI-facing shape for a single candidate. Enum-string serialisation
/// keeps the WebUI contract decoupled from the storage enum tags.
#[derive(Debug, Serialize)]
pub struct CodeArtifactResponse {
    pub id: Uuid,
    pub document_id: String,
    /// Null when the referenced algorithm row was deleted (reprocess flow);
    /// the candidate remains as an orphan pending manual re-link or reject.
    pub algorithm_id: Option<Uuid>,
    pub document_repo_id: Uuid,
    pub repo_commit: String,
    pub repo_license: Option<String>,
    pub language: String,
    pub file_path: String,
    pub symbol_name: Option<String>,
    pub start_line: i32,
    pub end_line: i32,
    pub snippet: String,
    pub match_rationale: Option<String>,
    pub match_confidence: &'static str,
    pub status: &'static str,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl From<CodeArtifact> for CodeArtifactResponse {
    fn from(a: CodeArtifact) -> Self {
        Self {
            id: a.id,
            document_id: a.document_id,
            algorithm_id: a.algorithm_id,
            document_repo_id: a.document_repo_id,
            repo_commit: a.repo_commit,
            repo_license: a.repo_license,
            language: a.language,
            file_path: a.file_path,
            symbol_name: a.symbol_name,
            start_line: a.start_line,
            end_line: a.end_line,
            snippet: a.snippet,
            match_rationale: a.match_rationale,
            match_confidence: match a.match_confidence {
                MatchConfidence::High => "high",
                MatchConfidence::Medium => "medium",
                MatchConfidence::Low => "low",
            },
            status: status_str(a.status),
            created_at: a.created_at,
            updated_at: a.updated_at,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct CodeReferenceRunResponse {
    pub document_id: String,
    pub document_repo_id: Uuid,
    pub status: &'static str,
    pub algorithm_count: i32,
    pub finding_count: i32,
    pub cost_usd_equivalent: Option<f32>,
    pub error_message: Option<String>,
    pub attempted_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

impl From<CodeReferenceRun> for CodeReferenceRunResponse {
    fn from(r: CodeReferenceRun) -> Self {
        Self {
            document_id: r.document_id,
            document_repo_id: r.document_repo_id,
            status: run_status_str(r.status),
            algorithm_count: r.algorithm_count,
            finding_count: r.finding_count,
            cost_usd_equivalent: r.cost_usd_equivalent,
            error_message: r.error_message,
            attempted_at: r.attempted_at,
            completed_at: r.completed_at,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct CodeReferenceListResponse {
    pub document_id: String,
    pub candidates: Vec<CodeArtifactResponse>,
    pub runs: Vec<CodeReferenceRunResponse>,
}

#[derive(Debug, Serialize)]
pub struct CodeArtifactCountEntry {
    pub document_id: String,
    pub count: i64,
}

#[derive(Debug, Serialize)]
pub struct CodeArtifactCountsResponse {
    pub counts: Vec<CodeArtifactCountEntry>,
}

#[derive(Debug, Serialize)]
pub struct CodeArtifactReviewResponse {
    pub id: Uuid,
    pub status: &'static str,
}

#[derive(Debug, Serialize)]
pub struct DeleteCodeArtifactResponse {
    pub id: Uuid,
    pub deleted: bool,
}

#[derive(Debug, Serialize)]
pub struct ApproveAllCodeReferencesResponse {
    pub document_id: String,
    /// Number of rows that were pending before this call and are now
    /// approved. Zero is a valid response (everything was already
    /// reviewed) and is not an error.
    pub approved_count: usize,
    /// Per-artifact graph-sync / embedding-sync failures encountered
    /// while flipping. The flip itself still succeeds; this field lets
    /// the UI surface a partial-failure toast instead of silently
    /// dropping the warnings on the server log.
    pub sync_failures: usize,
}

#[derive(Debug, Serialize)]
pub struct AnalyzeCodeReferenceResponse {
    pub document_id: String,
    pub document_repo_id: Uuid,
    pub track_id: String,
    pub status: &'static str,
}

/// Response from `POST /code-reference/by-document/{id}/submit`. Mirrors
/// `algorithms/submit` so the UI can render the same success toast shape:
/// "Submitted: N approved, M rejected — K indexes queued".
#[derive(Debug, Serialize)]
pub struct CodeReferenceSubmitResponse {
    pub document_id: String,
    pub approved_count: usize,
    pub rejected_count: usize,
    /// Number of distinct (document_repo, commit) pairs for which a
    /// reference-codebase index task was enqueued as a result of this
    /// submit. Zero when `auto_index_enabled()` is off, or when every
    /// repo already has a matching index row (idempotent).
    pub indexes_queued: usize,
    pub status: &'static str,
}

// ── Enum → str ──────────────────────────────────────────────────────────────

pub(super) fn status_str(s: ArtifactStatus) -> &'static str {
    match s {
        ArtifactStatus::Pending => "pending",
        ArtifactStatus::Approved => "approved",
        ArtifactStatus::Rejected => "rejected",
    }
}

fn run_status_str(s: RunStatus) -> &'static str {
    match s {
        RunStatus::Queued => "queued",
        RunStatus::Cloning => "cloning",
        RunStatus::Analyzing => "analyzing",
        RunStatus::Embedding => "embedding",
        RunStatus::AwaitingReview => "awaiting_review",
        RunStatus::Complete => "complete",
        RunStatus::Failed => "failed",
    }
}
