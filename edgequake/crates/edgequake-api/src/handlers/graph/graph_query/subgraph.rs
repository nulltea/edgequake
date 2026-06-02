//! Multi-source subgraph handler (`POST /api/v1/graph/subgraph`).
//!
//! SPEC-006 P3 (search → render): the caller supplies a starting set
//! of entity IDs — typically extracted from `POST /api/v1/query`'s
//! `sources[].source_type == "entity"` — and gets back a merged BFS
//! subgraph for direct rendering on the Sigma.js canvas.
//!
//! # Implements
//!
//! - **UC0101**: Explore Entity Neighborhood (multi-source variant)
//! - **FEAT0601**: Knowledge Graph Visualization
//!
//! # Enforces
//!
//! - **BR0201**: Tenant isolation (filters by tenant + workspace)
//! - **BR0009**: Node-count ceiling via clamped `max_nodes`
//!
//! Distinct from `traversal::get_graph` (`GET /api/v1/graph`) which
//! accepts at most a single `start_node` query parameter. The combined
//! Cypher-equivalent the storage layer offers per starting node
//! (`get_knowledge_graph`) is invoked once per start node here and the
//! results are merged with BFS-truncation at the global `max_nodes`
//! ceiling.

use axum::{extract::State, Json};
use std::collections::{HashMap, HashSet};
use tracing::{debug, warn};

use crate::error::{ApiError, ApiResult};
use crate::handlers::graph_types::*;
use crate::handlers::isolation::properties_match_tenant_context;
use crate::middleware::TenantContext;
use crate::state::AppState;

/// Extract a multi-source subgraph for rendering.
#[utoipa::path(
    post,
    path = "/api/v1/graph/subgraph",
    tag = "Graph",
    request_body = SubgraphRequest,
    responses(
        (status = 200, description = "Subgraph extracted", body = SubgraphResponse),
        (status = 400, description = "Invalid request (empty or oversized start_nodes)"),
        (status = 404, description = "No supplied start_nodes are visible in this workspace")
    )
)]
pub async fn get_subgraph(
    State(state): State<AppState>,
    tenant_ctx: TenantContext,
    Json(req): Json<SubgraphRequest>,
) -> ApiResult<Json<SubgraphResponse>> {
    let req = req.validated().map_err(ApiError::BadRequest)?;
    let requested_depth = req.depth;

    debug!(
        start_count = req.start_nodes.len(),
        depth = req.depth,
        max_nodes = req.max_nodes,
        tenant_id = ?tenant_ctx.tenant_id,
        workspace_id = ?tenant_ctx.workspace_id,
        "subgraph request"
    );

    // SECURITY: same strict tenant-context check as traversal::get_graph.
    // Without a workspace we cannot meaningfully scope a graph response.
    if tenant_ctx.tenant_id.is_none() || tenant_ctx.workspace_id.is_none() {
        warn!(
            tenant_id = ?tenant_ctx.tenant_id,
            workspace_id = ?tenant_ctx.workspace_id,
            "Tenant context missing — returning empty subgraph for security"
        );
        return Ok(Json(SubgraphResponse {
            nodes: vec![],
            edges: vec![],
            stats: SubgraphStats {
                total_nodes: 0,
                total_edges: 0,
                truncated: false,
                requested_depth,
                effective_depth: 0,
            },
        }));
    }

    // BFS from each start node independently, then merge. The storage
    // adapter's `get_knowledge_graph` already enforces a per-call node
    // ceiling; we feed it the remaining budget so the final merged set
    // never exceeds the requested cap.
    let mut merged_nodes: HashMap<String, GraphNodeResponse> = HashMap::new();
    let mut raw_edges: Vec<GraphEdgeResponse> = Vec::new();
    let mut truncated = false;

    for start in &req.start_nodes {
        if merged_nodes.len() >= req.max_nodes {
            truncated = true;
            break;
        }
        let remaining = req.max_nodes - merged_nodes.len();
        let kg = state
            .graph_storage
            .get_knowledge_graph(start, req.depth, remaining)
            .await?;
        if kg.is_truncated {
            truncated = true;
        }

        for n in kg.nodes {
            if !properties_match_tenant_context(&n.properties, &tenant_ctx) {
                continue;
            }
            if merged_nodes.contains_key(&n.id) {
                continue;
            }
            if merged_nodes.len() >= req.max_nodes {
                truncated = true;
                break;
            }
            let node_type = n
                .properties
                .get("entity_type")
                .and_then(|v| v.as_str())
                .unwrap_or("UNKNOWN")
                .to_string();
            let description = n
                .properties
                .get("description")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();
            merged_nodes.insert(
                n.id.clone(),
                GraphNodeResponse {
                    id: n.id.clone(),
                    label: n.id.clone(),
                    node_type,
                    description,
                    degree: 0,
                    properties: serde_json::to_value(&n.properties).unwrap_or_default(),
                },
            );
        }

        for e in kg.edges {
            if !properties_match_tenant_context(&e.properties, &tenant_ctx) {
                continue;
            }
            raw_edges.push(GraphEdgeResponse {
                source: e.source,
                target: e.target,
                relationship_type: e
                    .properties
                    .get("relation_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("RELATED_TO")
                    .to_string(),
                weight: e
                    .properties
                    .get("weight")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0) as f32,
                properties: serde_json::to_value(&e.properties).unwrap_or_default(),
            });
        }
    }

    // Fallback: any seed that didn't surface via BFS gets fetched via
    // single-node lookup. The storage's `get_knowledge_graph` returns
    // empty subgraphs for some otherwise-valid IDs (parameter-sensitive
    // behaviour in the AGE adapter), which used to make P3 404 whenever
    // every supplied ID hit that quirk — even though `get_node` finds
    // them fine. Looking the seeds up directly gives the canvas
    // something to render in that case (just the seeds, no neighbours).
    for start in &req.start_nodes {
        if merged_nodes.contains_key(start) {
            continue;
        }
        if merged_nodes.len() >= req.max_nodes {
            truncated = true;
            break;
        }
        match state.graph_storage.get_node(start).await {
            Ok(Some(n)) => {
                if !properties_match_tenant_context(&n.properties, &tenant_ctx) {
                    continue;
                }
                let node_type = n
                    .properties
                    .get("entity_type")
                    .and_then(|v| v.as_str())
                    .unwrap_or("UNKNOWN")
                    .to_string();
                let description = n
                    .properties
                    .get("description")
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                merged_nodes.insert(
                    n.id.clone(),
                    GraphNodeResponse {
                        id: n.id.clone(),
                        label: n.id.clone(),
                        node_type,
                        description,
                        degree: 0,
                        properties: serde_json::to_value(&n.properties).unwrap_or_default(),
                    },
                );
            }
            Ok(None) => {}
            Err(e) => {
                // A single-node lookup failure shouldn't sink the whole
                // request — log and move on to the next seed.
                warn!(start, error = ?e, "subgraph seed fallback lookup failed");
            }
        }
    }

    // 404 only when no supplied ID is visible at all — distinguishes
    // "your IDs don't exist or aren't yours" from "your IDs exist but
    // their neighbourhoods came back empty" (the latter still returns
    // 200 with the seed nodes rendered).
    let start_set: HashSet<&str> = req.start_nodes.iter().map(|s| s.as_str()).collect();
    let any_seed_kept = merged_nodes.keys().any(|id| start_set.contains(id.as_str()));
    if !any_seed_kept {
        return Err(ApiError::NotFound(format!(
            "None of the {} supplied start_nodes are visible in this workspace",
            req.start_nodes.len()
        )));
    }

    // Always fetch inter-seed edges explicitly. At depth=0 the BFS
    // returns each seed in isolation with no edges, so without this
    // step the canvas renders seeds as a disconnected dust cloud even
    // when the seeds are tightly linked in the graph. At depth>=1 the
    // call is largely redundant (BFS would already include inter-seed
    // edges where the seeds are neighbours), but it's still useful as
    // a backstop for any seed-pair edge that BFS missed due to the
    // node-budget truncation.
    let kept_ids: Vec<String> = merged_nodes.keys().cloned().collect();
    match state
        .graph_storage
        .get_edges_for_node_set(
            &kept_ids,
            tenant_ctx.tenant_id.as_deref(),
            tenant_ctx.workspace_id.as_deref(),
        )
        .await
    {
        Ok(extra) => {
            for e in extra {
                raw_edges.push(GraphEdgeResponse {
                    source: e.source,
                    target: e.target,
                    relationship_type: e
                        .properties
                        .get("relation_type")
                        .and_then(|v| v.as_str())
                        .unwrap_or("RELATED_TO")
                        .to_string(),
                    weight: e
                        .properties
                        .get("weight")
                        .and_then(|v| v.as_f64())
                        .unwrap_or(1.0) as f32,
                    properties: serde_json::to_value(&e.properties).unwrap_or_default(),
                });
            }
        }
        Err(e) => {
            warn!(error = ?e, "inter-seed edge fetch failed; rendering with BFS edges only");
        }
    }

    // Drop edges whose endpoints were truncated away, then dedup edges
    // that appeared in multiple per-source subgraphs / the inter-seed
    // fetch above.
    let kept: HashSet<String> = merged_nodes.keys().cloned().collect();
    let mut seen_edges: HashSet<(String, String, String)> = HashSet::new();
    let edges: Vec<GraphEdgeResponse> = raw_edges
        .into_iter()
        .filter(|e| kept.contains(&e.source) && kept.contains(&e.target))
        .filter(|e| {
            seen_edges.insert((
                e.source.clone(),
                e.target.clone(),
                e.relationship_type.clone(),
            ))
        })
        .collect();

    let nodes: Vec<GraphNodeResponse> = merged_nodes.into_values().collect();
    let total_nodes = nodes.len();
    let total_edges = edges.len();

    Ok(Json(SubgraphResponse {
        nodes,
        edges,
        stats: SubgraphStats {
            total_nodes,
            total_edges,
            truncated,
            requested_depth,
            // v1 does not introspect the actual BFS frontier reached
            // per call — `effective_depth` is the depth the storage
            // adapter was permitted to explore, which equals the
            // requested depth.
            effective_depth: requested_depth,
        },
    }))
}
