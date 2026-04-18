//! Domain types for reference-repository detection.
//!
//! Mirrors the `document_repos` + `document_repo_detections` tables declared
//! in `migrations/042_add_document_repos.sql`. Lives in the domain crate so
//! the storage trait and HTTP handlers share one definition.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RepoHost {
    Github,
    Gitlab,
    Bitbucket,
}

impl RepoHost {
    pub fn as_str(self) -> &'static str {
        match self {
            RepoHost::Github => "github",
            RepoHost::Gitlab => "gitlab",
            RepoHost::Bitbucket => "bitbucket",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "github" => Some(RepoHost::Github),
            "gitlab" => Some(RepoHost::Gitlab),
            "bitbucket" => Some(RepoHost::Bitbucket),
            _ => None,
        }
    }

    /// Map from the Layer A parser's host enum in `edgequake-pdf`.
    pub fn from_pdf(h: edgequake_pdf::RepoHost) -> Self {
        match h {
            edgequake_pdf::RepoHost::GitHub => RepoHost::Github,
            edgequake_pdf::RepoHost::GitLab => RepoHost::Gitlab,
            edgequake_pdf::RepoHost::Bitbucket => RepoHost::Bitbucket,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectionMethod {
    PdfLink,
    WebSearch,
}

impl DetectionMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            DetectionMethod::PdfLink => "pdf_link",
            DetectionMethod::WebSearch => "web_search",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "pdf_link" => Some(DetectionMethod::PdfLink),
            "web_search" => Some(DetectionMethod::WebSearch),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Confidence {
    High,
    Medium,
    Low,
}

impl Confidence {
    pub fn as_str(self) -> &'static str {
        match self {
            Confidence::High => "high",
            Confidence::Medium => "medium",
            Confidence::Low => "low",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "high" => Some(Confidence::High),
            "medium" => Some(Confidence::Medium),
            "low" => Some(Confidence::Low),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RepoStatus {
    Pending,
    Approved,
    Rejected,
}

impl RepoStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            RepoStatus::Pending => "pending",
            RepoStatus::Approved => "approved",
            RepoStatus::Rejected => "rejected",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "pending" => Some(RepoStatus::Pending),
            "approved" => Some(RepoStatus::Approved),
            "rejected" => Some(RepoStatus::Rejected),
            _ => None,
        }
    }
}

/// A row in `document_repos`. Created by the orchestrator; persisted via
/// [`crate::repo_detection::storage::RepoStorage`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentRepo {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub workspace_id: Uuid,
    pub document_id: String,
    pub host: RepoHost,
    pub owner: String,
    pub repo: String,
    pub url: String,
    pub detection_method: DetectionMethod,
    pub pdf_page_index: Option<i32>,
    pub search_rank: Option<i32>,
    pub source_url: Option<String>,
    pub confidence: Confidence,
    pub status: RepoStatus,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DetectionRunStatus {
    Running,
    Complete,
    Failed,
}

impl DetectionRunStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            DetectionRunStatus::Running => "running",
            DetectionRunStatus::Complete => "complete",
            DetectionRunStatus::Failed => "failed",
        }
    }
}

/// A row in `document_repo_detections`: per-document job-state record.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectionRun {
    pub tenant_id: Uuid,
    pub workspace_id: Uuid,
    pub document_id: String,
    pub status: DetectionRunStatus,
    pub layer_a_candidates: i32,
    pub layer_b_candidates: i32,
    pub error_message: Option<String>,
    pub attempted_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

/// A candidate produced by the orchestrator before persistence — carries the
/// same fields as a [`DocumentRepo`] row minus server-generated ones.
#[derive(Debug, Clone)]
pub struct RepoCandidate {
    pub host: RepoHost,
    pub owner: String,
    pub repo: String,
    pub url: String,
    pub detection_method: DetectionMethod,
    pub pdf_page_index: Option<i32>,
    pub search_rank: Option<i32>,
    pub source_url: Option<String>,
    pub confidence: Confidence,
}
