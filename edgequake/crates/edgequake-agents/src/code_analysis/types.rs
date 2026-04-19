//! Domain types for reference-code analysis (Phase 1).
//!
//! Mirrors the `code_artifacts` + `code_reference_runs` tables in
//! `migrations/043_add_code_artifacts.sql`, plus the DTOs we exchange with
//! the `code-analyzer` sidecar service.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

// ── Review state ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ArtifactStatus {
    Pending,
    Approved,
    Rejected,
}

impl ArtifactStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            ArtifactStatus::Pending => "pending",
            ArtifactStatus::Approved => "approved",
            ArtifactStatus::Rejected => "rejected",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "pending" => Some(Self::Pending),
            "approved" => Some(Self::Approved),
            "rejected" => Some(Self::Rejected),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MatchConfidence {
    High,
    Medium,
    Low,
}

impl MatchConfidence {
    pub fn as_str(self) -> &'static str {
        match self {
            MatchConfidence::High => "high",
            MatchConfidence::Medium => "medium",
            MatchConfidence::Low => "low",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "high" => Some(Self::High),
            "medium" => Some(Self::Medium),
            "low" => Some(Self::Low),
            _ => None,
        }
    }
}

// ── Code artifact row ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeArtifact {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub workspace_id: Uuid,
    pub document_id: String,
    pub algorithm_id: Uuid,
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
    pub match_confidence: MatchConfidence,

    pub status: ArtifactStatus,
    pub embedding_id: Option<Uuid>,

    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Not-yet-persisted candidate produced by the orchestrator.
#[derive(Debug, Clone)]
pub struct CodeArtifactCandidate {
    pub algorithm_id: Uuid,
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
    pub match_confidence: MatchConfidence,
}

// ── Run-state row ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunStatus {
    Queued,
    Cloning,
    Analyzing,
    Embedding,
    AwaitingReview,
    Complete,
    Failed,
}

impl RunStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            RunStatus::Queued => "queued",
            RunStatus::Cloning => "cloning",
            RunStatus::Analyzing => "analyzing",
            RunStatus::Embedding => "embedding",
            RunStatus::AwaitingReview => "awaiting_review",
            RunStatus::Complete => "complete",
            RunStatus::Failed => "failed",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "queued" => Some(Self::Queued),
            "cloning" => Some(Self::Cloning),
            "analyzing" => Some(Self::Analyzing),
            "embedding" => Some(Self::Embedding),
            "awaiting_review" => Some(Self::AwaitingReview),
            "complete" => Some(Self::Complete),
            "failed" => Some(Self::Failed),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeReferenceRun {
    pub tenant_id: Uuid,
    pub workspace_id: Uuid,
    pub document_id: String,
    pub document_repo_id: Uuid,
    pub status: RunStatus,
    pub algorithm_count: i32,
    pub finding_count: i32,
    pub cost_usd_equivalent: Option<f32>,
    pub error_message: Option<String>,
    pub attempted_at: DateTime<Utc>,
    pub completed_at: Option<DateTime<Utc>>,
}

// ── Wire DTOs with the code-analyzer sidecar ───────────────────────────────
//
// The code-analyzer speaks the shapes in `code-analyzer/src/code_analyzer/schema.py`.

#[derive(Debug, Clone, Serialize)]
pub struct AnalyzerAlgorithmInput {
    pub id: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "String::is_empty")]
    pub description: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pseudocode: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub languages_hint: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AnalyzerRequest {
    pub repo_url: String,
    pub repo_commit: String,
    pub algorithms: Vec<AnalyzerAlgorithmInput>,
    pub size_cap_mb: u32,
    pub timeout_s: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnalyzerFinding {
    pub algorithm_id: String,
    pub file: String,
    pub start_line: i32,
    pub end_line: i32,
    pub rationale: String,
    pub confidence: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct AnalyzerResponse {
    pub repo_path: String,
    pub repo_commit: String,
    pub repo_license: Option<String>,
    pub findings: Vec<AnalyzerFinding>,
    #[serde(default)]
    pub usage_cost_usd_equivalent: Option<f32>,
    pub duration_ms: i64,
    #[serde(default)]
    pub num_turns: Option<i32>,
}
