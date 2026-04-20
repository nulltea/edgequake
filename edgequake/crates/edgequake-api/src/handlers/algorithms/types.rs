//! Request/response DTOs for algorithm endpoints.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use edgequake_algorithms::{Algorithm, AlgorithmStatus};

// ── Requests ────────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ExtractAlgorithmsRequest {
    pub document_id: String,
}

#[derive(Debug, Deserialize)]
pub struct ReviewAlgorithmRequest {
    pub status: AlgorithmStatus,
}

#[derive(Debug, Deserialize)]
pub struct ListAlgorithmsParams {
    pub status: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct SearchAlgorithmsParams {
    pub query: Option<String>,
    pub status: Option<String>,
    pub document_id: Option<String>,
    pub limit: Option<i64>,
    pub offset: Option<i64>,
}

// ── Responses ───────────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct ExtractAlgorithmsResponse {
    pub document_id: String,
    pub status: String,
    pub message: String,
}

#[derive(Debug, Serialize)]
pub struct AlgorithmListResponse {
    pub algorithms: Vec<Algorithm>,
    pub total: usize,
}

#[derive(Debug, Serialize)]
pub struct AlgorithmReviewResponse {
    pub id: Uuid,
    pub status: String,
}

#[derive(Debug, Serialize)]
pub struct AlgorithmDeleteResponse {
    pub deleted: u64,
}

#[derive(Debug, Serialize)]
pub struct AlgorithmSubmitResponse {
    pub document_id: String,
    pub approved_count: usize,
    pub rejected_count: usize,
    pub status: String,
}

#[derive(Debug, Serialize)]
pub struct AlgorithmSearchResponse {
    pub algorithms: Vec<Algorithm>,
    pub total: i64,
    pub limit: i64,
    pub offset: i64,
}

#[derive(Debug, Serialize)]
pub struct AlgorithmCountEntry {
    pub document_id: String,
    pub count: i64,
}

#[derive(Debug, Serialize)]
pub struct AlgorithmCountsResponse {
    pub counts: Vec<AlgorithmCountEntry>,
}
