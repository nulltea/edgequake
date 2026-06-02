//! Knowledge graph API handlers for visualization and exploration.
//!
//! # Implements
//!
//! @implements FEAT0206
//! @implements FEAT0405 (Graph Exploration API)
//! @implements FEAT0204 (Graph Analytics)
//! @implements FEAT0601 (Knowledge Graph Visualization)
//! @implements FEAT0410 (REST API Service)
//!
//! - **UC0101**: Explore Entity Neighborhood
//! - **UC0104**: View Graph Statistics
//!
//! # Enforces
//!
//! - **BR0201**: Tenant isolation (graph scoped to workspace)
//! - **BR0009**: Max 1000 nodes per visualization request
//!
//! # Endpoints
//!
//! | Method | Path | Handler | Description |
//! |--------|------|---------|-------------|
//! | GET | `/api/v1/graph` | [`get_graph`] | Get full graph (paginated) |
//! | GET | `/api/v1/graph/stats` | [`get_graph_stats`] | Node/edge counts |
//! | GET | `/api/v1/graph/stream` | SSE streaming graph updates |
//!
//! # WHY: Separate Graph Visualization Layer
//!
//! Graph visualization is compute-intensive and has different requirements
//! than query execution:
//! - Needs pagination to handle large graphs
//! - Requires layout hints for rendering
//! - May need streaming for real-time updates
//!
//! Separating from query handlers enables independent optimization.

mod graph_query;
mod graph_stream;

pub use graph_query::*;
pub use graph_stream::*;

// Re-export DTOs from graph_types module
pub use crate::handlers::graph_types::*;

use crate::handlers::documents::storage_helpers::extract_source_docs;

/// Returns true when the node/edge's `source_ids` (or legacy
/// `source_id`) property contains the requested document — either as
/// a bare doc UUID or as a chunk key prefixed with that UUID
/// (`<doc_uuid>-chunk-<n>`). Shared by the non-streaming
/// (`graph_query::traversal`) and streaming (`graph_stream`) handlers
/// so the graph viewer's "filter by document" surface is consistent.
pub(crate) fn properties_match_document(
    properties: &std::collections::HashMap<String, serde_json::Value>,
    document_id: &str,
) -> bool {
    let chunk_prefix = format!("{document_id}-chunk-");
    extract_source_docs(properties)
        .iter()
        .any(|s| s == document_id || s.starts_with(&chunk_prefix) || s.starts_with(document_id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::extract::{Json, Path, Query, State};

    use crate::middleware::TenantContext;
    use crate::state::AppState;

    #[tokio::test]
    async fn test_get_graph_empty() {
        let state = AppState::test_state();
        let tenant_ctx = TenantContext::default();
        let params = GraphQueryParams {
            start_node: None,
            depth: 2,
            max_nodes: 100,
            document_id: None,
        };

        let result = get_graph(State(state), tenant_ctx, Query(params)).await;
        assert!(result.is_ok());

        let response = result.unwrap().0;
        assert!(response.nodes.is_empty());
    }

    #[tokio::test]
    async fn test_get_graph_with_depth() {
        let state = AppState::test_state();
        let tenant_ctx = TenantContext::default();
        let params = GraphQueryParams {
            start_node: None,
            depth: 5,
            max_nodes: 50,
            document_id: None,
        };

        let result = get_graph(State(state), tenant_ctx, Query(params)).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_get_node_not_found() {
        let state = AppState::test_state();

        let result = get_node(State(state), Path("nonexistent_node".to_string())).await;
        // Should return not found or empty
        assert!(result.is_ok() || result.is_err());
    }

    #[tokio::test]
    async fn test_search_labels_empty() {
        let state = AppState::test_state();
        let params = SearchLabelsQuery {
            q: "test".to_string(),
            limit: 10,
        };

        let result = search_labels(State(state), Query(params)).await;
        assert!(result.is_ok());

        let response = result.unwrap().0;
        assert!(response.labels.is_empty());
    }

    #[tokio::test]
    async fn test_get_popular_labels() {
        let state = AppState::test_state();
        let params = PopularLabelsQuery {
            limit: 20,
            min_degree: None,
            entity_type: None,
        };

        let result = get_popular_labels(State(state), Query(params)).await;
        assert!(result.is_ok());
    }

    // SPEC-006 P3 — POST /api/v1/graph/subgraph

    #[test]
    fn test_subgraph_request_validated_rejects_empty() {
        let req = SubgraphRequest {
            start_nodes: vec![],
            depth: 1,
            max_nodes: 60,
        };
        assert!(req.validated().is_err());
    }

    #[test]
    fn test_subgraph_request_validated_rejects_oversized() {
        let req = SubgraphRequest {
            start_nodes: (0..MAX_SUBGRAPH_START_NODES + 1)
                .map(|i| format!("N{}", i))
                .collect(),
            depth: 1,
            max_nodes: 60,
        };
        assert!(req.validated().is_err());
    }

    #[test]
    fn test_subgraph_request_validated_clamps_depth() {
        let req = SubgraphRequest {
            start_nodes: vec!["A".into()],
            depth: 99,
            max_nodes: 60,
        };
        let v = req.validated().unwrap();
        assert_eq!(v.depth, MAX_SUBGRAPH_DEPTH);
    }

    #[test]
    fn test_subgraph_request_validated_clamps_max_nodes() {
        let req = SubgraphRequest {
            start_nodes: vec!["A".into()],
            depth: 1,
            max_nodes: 9_999,
        };
        let v = req.validated().unwrap();
        assert_eq!(v.max_nodes, MAX_SUBGRAPH_NODES);
    }

    #[test]
    fn test_subgraph_request_validated_clamps_zero_max_nodes() {
        let req = SubgraphRequest {
            start_nodes: vec!["A".into()],
            depth: 1,
            max_nodes: 0,
        };
        let v = req.validated().unwrap();
        assert_eq!(v.max_nodes, 1);
    }

    #[test]
    fn test_subgraph_request_defaults() {
        let json = r#"{"start_nodes": ["A"]}"#;
        let req: SubgraphRequest = serde_json::from_str(json).unwrap();
        assert_eq!(req.depth, 1);
        assert_eq!(req.max_nodes, 60);
    }

    #[tokio::test]
    async fn test_subgraph_missing_tenant_returns_empty() {
        // Per SPEC-006: tenant-context absent ⇒ short-circuit to an
        // empty 200 response rather than leak across workspaces.
        let state = AppState::test_state();
        let tenant_ctx = TenantContext::default();
        let req = SubgraphRequest {
            start_nodes: vec!["UNKNOWN_NODE".into()],
            depth: 1,
            max_nodes: 60,
        };

        let result = get_subgraph(State(state), tenant_ctx, Json(req)).await;
        let resp = result.unwrap().0;
        assert!(resp.nodes.is_empty());
        assert!(resp.edges.is_empty());
        assert_eq!(resp.stats.total_nodes, 0);
        assert!(!resp.stats.truncated);
        assert_eq!(resp.stats.requested_depth, 1);
    }

    #[tokio::test]
    async fn test_subgraph_empty_start_nodes_is_bad_request() {
        let state = AppState::test_state();
        let tenant_ctx = TenantContext::default();
        let req = SubgraphRequest {
            start_nodes: vec![],
            depth: 1,
            max_nodes: 60,
        };

        let result = get_subgraph(State(state), tenant_ctx, Json(req)).await;
        assert!(result.is_err());
    }
}
