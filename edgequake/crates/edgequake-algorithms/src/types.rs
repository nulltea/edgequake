//! Data types for algorithm extraction.
//!
//! Ported from RAGSearcher's `algorithm_types.rs`, adapted for EdgeQuake's
//! multi-tenant model (tenant_id, workspace_id, document_id).
//!
//! Includes robust deserialization helpers that tolerate LLM output quirks
//! (numbers as strings, objects where strings expected, etc.).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Deserializer, Serialize};
use uuid::Uuid;

// ─────────────────────────────────────────────────────────────────────────────
//                         DESERIALIZATION HELPERS
// ─────────────────────────────────────────────────────────────────────────────
// LLMs occasionally return numbers as strings, objects where strings are expected,
// etc. These helpers tolerate those quirks.

fn string_or_json<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let val = serde_json::Value::deserialize(deserializer)?;
    match val {
        serde_json::Value::String(s) => Ok(s),
        serde_json::Value::Null => Ok(String::new()),
        other => Ok(other.to_string()),
    }
}

fn string_or_json_default<'de, D>(deserializer: D) -> Result<String, D::Error>
where
    D: Deserializer<'de>,
{
    let val = Option::<serde_json::Value>::deserialize(deserializer)?;
    match val {
        None | Some(serde_json::Value::Null) => Ok(String::new()),
        Some(serde_json::Value::String(s)) => Ok(s),
        Some(other) => Ok(other.to_string()),
    }
}

fn opt_string_or_json<'de, D>(deserializer: D) -> Result<Option<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let val = Option::<serde_json::Value>::deserialize(deserializer)?;
    match val {
        None | Some(serde_json::Value::Null) => Ok(None),
        Some(serde_json::Value::String(s)) => Ok(Some(s)),
        Some(other) => Ok(Some(other.to_string())),
    }
}

fn usize_or_string<'de, D>(deserializer: D) -> Result<usize, D::Error>
where
    D: Deserializer<'de>,
{
    let val = serde_json::Value::deserialize(deserializer)?;
    match val {
        serde_json::Value::Number(n) => n
            .as_u64()
            .map(|v| v as usize)
            .ok_or_else(|| serde::de::Error::custom("expected unsigned integer")),
        serde_json::Value::String(s) => s.parse::<usize>().map_err(serde::de::Error::custom),
        _ => Err(serde::de::Error::custom("expected number or string")),
    }
}

fn flexible_string_vec<'de, D>(deserializer: D) -> Result<Vec<String>, D::Error>
where
    D: Deserializer<'de>,
{
    let val = serde_json::Value::deserialize(deserializer)?;
    match val {
        serde_json::Value::Array(arr) => Ok(arr
            .into_iter()
            .map(|v| match v {
                serde_json::Value::String(s) => s,
                other => other.to_string(),
            })
            .collect()),
        serde_json::Value::Null => Ok(Vec::new()),
        _ => Ok(vec![val.to_string()]),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//                         PASS 1: ALGORITHM INVENTORY
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmCandidate {
    #[serde(deserialize_with = "string_or_json")]
    pub id: String,
    #[serde(deserialize_with = "string_or_json")]
    pub name: String,
    #[serde(default, deserialize_with = "string_or_json_default")]
    pub description: String,
    #[serde(default, deserialize_with = "string_or_json_default")]
    pub location: String,
    #[serde(default, rename = "type", deserialize_with = "string_or_json_default")]
    pub algorithm_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmInventory {
    #[serde(deserialize_with = "string_or_json")]
    pub paper_title: String,
    pub algorithms: Vec<AlgorithmCandidate>,
    #[serde(default, deserialize_with = "string_or_json_default")]
    pub paper_type: String,
}

// ─────────────────────────────────────────────────────────────────────────────
//                      PASS 2: ALGORITHM EXTRACTION
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmStep {
    #[serde(default, deserialize_with = "usize_or_string")]
    pub number: usize,
    #[serde(deserialize_with = "string_or_json")]
    pub action: String,
    #[serde(default, deserialize_with = "string_or_json_default")]
    pub details: String,
    #[serde(default, deserialize_with = "opt_string_or_json")]
    pub math: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmIO {
    #[serde(deserialize_with = "string_or_json")]
    pub name: String,
    #[serde(default, rename = "type", deserialize_with = "string_or_json_default")]
    pub io_type: String,
    #[serde(default, deserialize_with = "string_or_json_default")]
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedAlgorithm {
    #[serde(default, deserialize_with = "usize_or_string")]
    pub rank: usize,
    #[serde(deserialize_with = "string_or_json")]
    pub name: String,
    /// One of: "Algorithm", "Protocol", "Functionality", "Theorem",
    /// "Definition", "Lemma", "Scheme". Defaults to "Algorithm" when the
    /// LLM does not provide a value.
    #[serde(default = "default_algorithm_type", rename = "type", alias = "algorithm_type",
            deserialize_with = "string_or_json")]
    pub algorithm_type: String,
    #[serde(default, deserialize_with = "string_or_json_default")]
    pub description: String,
    pub steps: Vec<AlgorithmStep>,
    #[serde(default)]
    pub inputs: Vec<AlgorithmIO>,
    #[serde(default)]
    pub outputs: Vec<AlgorithmIO>,
    #[serde(default, deserialize_with = "flexible_string_vec")]
    pub preconditions: Vec<String>,
    #[serde(default, deserialize_with = "opt_string_or_json")]
    pub complexity: Option<String>,
    #[serde(default, deserialize_with = "opt_string_or_json")]
    pub mathematical_notation: Option<String>,
    #[serde(default, deserialize_with = "opt_string_or_json")]
    pub pseudocode: Option<String>,
    #[serde(default, deserialize_with = "flexible_string_vec")]
    pub tags: Vec<String>,
    #[serde(default = "default_confidence", deserialize_with = "string_or_json")]
    pub confidence: String,
}

fn default_confidence() -> String {
    "medium".to_string()
}

fn default_algorithm_type() -> String {
    "Algorithm".to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmExtractionOutput {
    pub algorithms: Vec<ExtractedAlgorithm>,
}

// ─────────────────────────────────────────────────────────────────────────────
//                        PASS 3: VERIFICATION
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletenessIssue {
    #[serde(default)]
    pub algorithm_rank: usize,
    #[serde(default)]
    pub issue: String,
    #[serde(default)]
    pub severity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmVerificationResult {
    #[serde(default)]
    pub verification_status: String,
    #[serde(default)]
    pub completeness_issues: Vec<CompletenessIssue>,
    #[serde(default)]
    pub overall_quality: String,
    #[serde(default, deserialize_with = "flexible_string_vec")]
    pub improvement_suggestions: Vec<String>,
}

/// Combined result from the 3-pass extraction pipeline.
#[derive(Debug, Clone)]
pub struct AlgorithmExtractionResult {
    pub algorithms: Vec<ExtractedAlgorithm>,
    pub verification: Option<AlgorithmVerificationResult>,
}

// ─────────────────────────────────────────────────────────────────────────────
//                    STORED ALGORITHM (DATABASE ROW)
// ─────────────────────────────────────────────────────────────────────────────

/// Review status for an extracted algorithm.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum AlgorithmStatus {
    Pending,
    Approved,
    Rejected,
}

impl std::fmt::Display for AlgorithmStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pending => write!(f, "pending"),
            Self::Approved => write!(f, "approved"),
            Self::Rejected => write!(f, "rejected"),
        }
    }
}

impl std::str::FromStr for AlgorithmStatus {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "pending" => Ok(Self::Pending),
            "approved" => Ok(Self::Approved),
            "rejected" => Ok(Self::Rejected),
            _ => Err(format!("invalid algorithm status: {s}")),
        }
    }
}

/// A fully extracted algorithm stored in PostgreSQL.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Algorithm {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub workspace_id: Uuid,
    pub document_id: String,
    pub name: String,
    /// Kind of algorithmic construct: "Algorithm", "Protocol", "Functionality",
    /// "Theorem", "Definition", "Lemma", "Scheme", or custom.
    pub algorithm_type: String,
    pub description: Option<String>,
    pub steps: Vec<AlgorithmStep>,
    pub inputs: Vec<AlgorithmIO>,
    pub outputs: Vec<AlgorithmIO>,
    pub preconditions: Vec<String>,
    pub complexity: Option<String>,
    pub mathematical_notation: Option<String>,
    pub pseudocode: Option<String>,
    pub tags: Vec<String>,
    pub confidence: String,
    pub status: AlgorithmStatus,
    pub verification_status: Option<String>,
    pub verification_details: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Task data payload for algorithm extraction (serialized into task.task_data).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlgorithmExtractionTaskData {
    pub document_id: String,
    pub tenant_id: Uuid,
    pub workspace_id: Uuid,
}
