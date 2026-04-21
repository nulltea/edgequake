use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodebaseIndexMode {
    AlgorithmFocused,
    Full,
}

impl CodebaseIndexMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AlgorithmFocused => "algorithm_focused",
            Self::Full => "full",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "algorithm_focused" => Some(Self::AlgorithmFocused),
            "full" => Some(Self::Full),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CodebaseIndexStatus {
    Queued,
    Scanning,
    Parsing,
    Chunking,
    Embedding,
    Complete,
    Failed,
}

impl CodebaseIndexStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Queued => "queued",
            Self::Scanning => "scanning",
            Self::Parsing => "parsing",
            Self::Chunking => "chunking",
            Self::Embedding => "embedding",
            Self::Complete => "complete",
            Self::Failed => "failed",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "queued" => Some(Self::Queued),
            "scanning" => Some(Self::Scanning),
            "parsing" => Some(Self::Parsing),
            "chunking" => Some(Self::Chunking),
            "embedding" => Some(Self::Embedding),
            "complete" => Some(Self::Complete),
            "failed" => Some(Self::Failed),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebaseIndex {
    pub id: Uuid,
    pub tenant_id: Uuid,
    pub workspace_id: Uuid,
    pub document_id: String,
    pub document_repo_id: Uuid,
    pub repo_url: String,
    pub repo_commit: String,
    pub repo_path: String,
    pub repo_license: Option<String>,
    pub mode: CodebaseIndexMode,
    pub status: CodebaseIndexStatus,
    pub language_set: Vec<String>,
    pub file_count: i32,
    pub symbol_count: i32,
    pub chunk_count: i32,
    pub edge_count: i32,
    /// Approximate graph diameter (max BFS depth from the highest-degree
    /// symbol). Computed on-demand in `get_index` so it matches the
    /// current edge set without requiring reindex. `None` on indexes
    /// that aren't `complete` yet.
    pub max_depth: Option<i32>,
    pub error_message: Option<String>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone)]
pub struct CodebaseFile {
    pub id: Uuid,
    pub file_path: String,
    pub language: String,
    pub checksum: String,
    pub line_count: i32,
    pub byte_count: i32,
    pub skipped_reason: Option<String>,
}

#[derive(Debug, Clone)]
pub struct CodebaseSymbol {
    pub id: Uuid,
    pub file_id: Uuid,
    pub symbol_kind: String,
    pub name: String,
    pub qualified_name: String,
    pub parent_symbol_id: Option<Uuid>,
    pub file_path: String,
    pub language: String,
    pub start_line: i32,
    pub end_line: i32,
    pub start_byte: i32,
    pub end_byte: i32,
    /// Per-language rich metadata extracted at tree-sitter time.
    /// Shape: `{"parameters": [...], "return_type": "...", "docstring": "...",
    /// "visibility": "public|private|crate|protected", "is_async": bool,
    /// "is_test": bool}`. Keys present only when the grammar exposes them.
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct CodebaseEdge {
    pub edge_type: String,
    pub source_symbol_id: Option<Uuid>,
    pub target_symbol_id: Option<Uuid>,
    pub source_file_id: Option<Uuid>,
    pub target_file_id: Option<Uuid>,
    pub target_name: Option<String>,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone)]
pub struct CodebaseChunk {
    pub id: Uuid,
    pub file_id: Uuid,
    pub symbol_id: Option<Uuid>,
    pub algorithm_id: Option<Uuid>,
    pub code_artifact_id: Option<Uuid>,
    pub chunk_kind: String,
    pub language: String,
    pub file_path: String,
    pub symbol_name: Option<String>,
    pub start_line: i32,
    pub end_line: i32,
    pub token_estimate: i32,
    pub algorithm_focus: f32,
    pub content: String,
    pub content_hash: String,
}

#[derive(Debug, Clone)]
pub struct IndexBuildOutput {
    pub files: Vec<CodebaseFile>,
    pub symbols: Vec<CodebaseSymbol>,
    pub edges: Vec<CodebaseEdge>,
    pub chunks: Vec<CodebaseChunk>,
    pub language_set: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebaseQueryHit {
    pub chunk_id: Uuid,
    pub index_id: Uuid,
    pub document_id: String,
    pub document_repo_id: Uuid,
    pub repo_url: String,
    pub repo_commit: String,
    pub file_path: String,
    pub language: String,
    pub symbol_name: Option<String>,
    pub start_line: i32,
    pub end_line: i32,
    pub chunk_kind: String,
    pub algorithm_id: Option<Uuid>,
    pub content: String,
    pub cosine_distance: f64,
}

/// One node in a graph subgraph returned by
/// [`ReferenceCodebaseStorage::fetch_subgraph`].
///
/// `chunk_id` is populated when the symbol has a matching chunk in the
/// current index — lets the UI go one hop from graph click to chunk
/// content without a second round-trip.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubgraphNode {
    pub symbol_id: Uuid,
    pub name: String,
    pub qualified_name: String,
    pub kind: String,
    pub language: String,
    pub file_path: String,
    pub start_line: i32,
    pub end_line: i32,
    /// Depth from the BFS seed set. Seeds are 0, their neighbours 1, etc.
    pub depth: i32,
    /// Best-matching chunk id, when one exists. `algorithm_anchor`
    /// chunks win over plain `symbol` chunks.
    pub chunk_id: Option<Uuid>,
    /// True when the symbol overlaps an approved `code_artifact` (i.e.
    /// is one of the user's review anchors).
    pub is_anchor: bool,
    /// `algorithm_focus` copied from the best-matching chunk (0.0–1.0).
    pub algorithm_focus: f32,
    /// Rich symbol metadata extracted at index time: `{parameters, return_type,
    /// docstring, visibility, is_async, is_test}`. Keys present only when
    /// the source grammar exposes them. Empty object `{}` otherwise.
    pub metadata: serde_json::Value,
}

/// One edge in a subgraph. Emitted when both endpoints are symbols inside
/// the returned node set — file-only edges are filtered out because they
/// have no renderable node on the client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SubgraphEdge {
    pub source_symbol_id: Uuid,
    pub target_symbol_id: Uuid,
    pub kind: String,
    pub target_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodebaseSubgraph {
    pub nodes: Vec<SubgraphNode>,
    pub edges: Vec<SubgraphEdge>,
    /// True when the result was clipped by `max_nodes` — UI can surface a
    /// "not all neighbours shown" hint.
    pub truncated: bool,
}
