//! Request/response DTOs for reference-repository endpoints.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use edgequake_agents::repo_detection::{
    Confidence, DetectionMethod, DetectionRun, DetectionRunStatus, DocumentRepo, RepoHost,
    RepoStatus,
};

// ── Requests ────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ReviewRepoRequest {
    pub status: RepoStatus,
}

#[derive(Debug, Deserialize)]
pub struct DetectReposRequest {
    /// PDF id for Layer A. Optional — if unset, only Layer B runs (against
    /// the document's stored markdown).
    #[serde(default)]
    pub pdf_id: Option<String>,
}

// ── Responses ───────────────────────────────────────────────────────────────

/// Stable, UI-friendly serialisation of a single candidate. We don't serialise
/// [`DocumentRepo`] directly because its enum `serde` tags go lowercase which
/// doesn't match the UI's preferred shape (string values for host/method/etc).
#[derive(Debug, Serialize)]
pub struct RepoCandidateResponse {
    pub id: Uuid,
    pub document_id: String,
    pub host: &'static str,
    pub owner: String,
    pub repo: String,
    pub url: String,
    pub detection_method: &'static str,
    pub pdf_page_index: Option<i32>,
    pub search_rank: Option<i32>,
    pub source_url: Option<String>,
    pub confidence: &'static str,
    pub status: &'static str,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

impl From<DocumentRepo> for RepoCandidateResponse {
    fn from(r: DocumentRepo) -> Self {
        Self {
            id: r.id,
            document_id: r.document_id,
            host: host_str(r.host),
            owner: r.owner,
            repo: r.repo,
            url: r.url,
            detection_method: method_str(r.detection_method),
            pdf_page_index: r.pdf_page_index,
            search_rank: r.search_rank,
            source_url: r.source_url,
            confidence: confidence_str(r.confidence),
            status: status_str(r.status),
            created_at: r.created_at,
            updated_at: r.updated_at,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct DetectionRunResponse {
    pub document_id: String,
    pub status: &'static str,
    pub layer_a_candidates: i32,
    pub layer_b_candidates: i32,
    pub error_message: Option<String>,
    pub attempted_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

impl From<DetectionRun> for DetectionRunResponse {
    fn from(r: DetectionRun) -> Self {
        Self {
            document_id: r.document_id,
            status: run_status_str(r.status),
            layer_a_candidates: r.layer_a_candidates,
            layer_b_candidates: r.layer_b_candidates,
            error_message: r.error_message,
            attempted_at: r.attempted_at,
            completed_at: r.completed_at,
        }
    }
}

#[derive(Debug, Serialize)]
pub struct RepoListResponse {
    pub document_id: String,
    pub candidates: Vec<RepoCandidateResponse>,
    pub detection_run: Option<DetectionRunResponse>,
}

#[derive(Debug, Serialize)]
pub struct RepoReviewResponse {
    pub id: Uuid,
    pub status: &'static str,
}

#[derive(Debug, Serialize)]
pub struct DetectReposResponse {
    pub document_id: String,
    pub track_id: String,
    pub status: &'static str,
}

// ── Enum → &str mapping ─────────────────────────────────────────────────────

fn host_str(h: RepoHost) -> &'static str {
    match h {
        RepoHost::Github => "github",
        RepoHost::Gitlab => "gitlab",
        RepoHost::Bitbucket => "bitbucket",
    }
}

fn method_str(m: DetectionMethod) -> &'static str {
    match m {
        DetectionMethod::PdfLink => "pdf_link",
        DetectionMethod::WebSearch => "web_search",
    }
}

fn confidence_str(c: Confidence) -> &'static str {
    match c {
        Confidence::High => "high",
        Confidence::Medium => "medium",
        Confidence::Low => "low",
    }
}

pub(super) fn status_str(s: RepoStatus) -> &'static str {
    match s {
        RepoStatus::Pending => "pending",
        RepoStatus::Approved => "approved",
        RepoStatus::Rejected => "rejected",
    }
}

fn run_status_str(s: DetectionRunStatus) -> &'static str {
    match s {
        DetectionRunStatus::Running => "running",
        DetectionRunStatus::Complete => "complete",
        DetectionRunStatus::Failed => "failed",
    }
}
