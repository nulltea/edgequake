//! Entity CRUD operations for manual knowledge graph management.
//!
//! # Implements
//!
//! - **UC0101**: Explore Entity Neighborhood
//! - **UC0102**: Search Entities by Name
//! - **UC0103**: Delete Entity from Graph
//! - **FEAT0002**: Entity Extraction (view extracted entities)
//! - **FEAT0202**: Graph Traversal
//! - **FEAT0203**: Graph Mutation Operations
//! - **FEAT0401**: REST API Service
//!
//! # Enforces
//!
//! - **BR0008**: Entity names normalized (UPPERCASE with underscores)
//! - **BR0005**: Entity description max 512 tokens
//! - **BR0201**: Tenant isolation
//!
//! # Endpoints
//!
//! | Method | Path | Handler | Description |
//! |--------|------|---------|-------------|
//! | GET | `/api/v1/graph/entities` | [`list_entities`] | List with pagination |
//! | GET | `/api/v1/graph/entities/:id` | [`get_entity`] | Get single entity |
//! | POST | `/api/v1/graph/entities` | [`create_entity`] | Manually create entity |
//! | PUT | `/api/v1/graph/entities/:id` | [`update_entity`] | Update entity |
//! | DELETE | `/api/v1/graph/entities/:id` | [`delete_entity`] | Delete with cascade |
//! | GET | `/api/v1/graph/entities/:id/neighbors` | [`get_entity_neighbors`] | Get connected entities |
//!
//! # WHY: Manual Entity Management
//!
//! While entities are typically extracted automatically from documents, users need
//! manual CRUD operations for:
//! - Correcting extraction errors
//! - Adding domain knowledge not in documents
//! - Merging duplicate entities
//! - Curating the knowledge graph

pub mod embed;
mod entity_body;
mod entity_crud;
mod entity_ops;

pub use entity_body::*;
pub use entity_crud::*;
pub use entity_ops::*;

// Re-export DTOs from entities_types module
pub use crate::handlers::entities_types::*;

use edgequake_storage::GraphNode;

// ============================================================================
// Shared Helper Functions
// ============================================================================

/// Normalize entity name to the same form the pipeline uses.
///
/// Title-cases each word and joins with underscores (e.g.
/// `"John Smith"` → `"John_Smith"`, `"john smith"` → `"John_Smith"`).
/// Case is **not** forced uppercase — the result stays readable in
/// UIs while still merging case variants ("john doe", "JOHN DOE",
/// "John Doe" all map to the same key).
///
/// Delegates to `edgequake_pipeline::prompts::normalize_entity_name` so
/// API writes and pipeline writes produce identical entity IDs.
pub(super) fn normalize_entity_name(name: &str) -> String {
    edgequake_pipeline::prompts::normalize_entity_name(name)
}

/// Convert GraphNode to EntityResponse.
pub(super) fn node_to_entity_response(node: GraphNode, degree: usize) -> EntityResponse {
    let props = &node.properties;

    EntityResponse {
        id: node.id.clone(),
        entity_name: node.id.clone(),
        entity_type: props
            .get("entity_type")
            .and_then(|v| v.as_str())
            .unwrap_or("UNKNOWN")
            .to_string(),
        description: props
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        source_id: props
            .get("source_id")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string(),
        created_at: props
            .get("created_at")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        updated_at: props
            .get("updated_at")
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string(),
        degree,
        metadata: props
            .get("metadata")
            .cloned()
            .unwrap_or(serde_json::json!({})),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_entity_name() {
        // Delegates to pipeline normalizer — case-preserving for acronyms /
        // CamelCase, title-cased otherwise, joined with single space.
        assert_eq!(
            normalize_entity_name("quantum computing"),
            "Quantum Computing"
        );
        // AI has uppercase past position 0 → preserved verbatim.
        assert_eq!(normalize_entity_name("AI"), "AI");
        assert_eq!(
            normalize_entity_name("Machine Learning"),
            "Machine Learning"
        );
    }

    #[test]
    fn test_normalize_entity_name_edge_cases() {
        // Words joined with single space.
        assert_eq!(normalize_entity_name("hello world"), "Hello World");
        // Multiple spaces collapse.
        assert_eq!(normalize_entity_name("hello  world"), "Hello World");
        // Empty string
        assert_eq!(normalize_entity_name(""), "");
        // All-caps → preserved verbatim (acronym treatment).
        assert_eq!(
            normalize_entity_name("ALREADY UPPERCASE"),
            "ALREADY UPPERCASE"
        );
    }

    #[test]
    fn test_normalize_preserves_acronyms_and_camelcase() {
        assert_eq!(normalize_entity_name("FFN"), "FFN");
        assert_eq!(normalize_entity_name("TMFA"), "TMFA");
        assert_eq!(normalize_entity_name("ISA-HiddenState"), "ISA-HiddenState");
        assert_eq!(normalize_entity_name("iPhone"), "iPhone");
        assert_eq!(normalize_entity_name("GraphQL"), "GraphQL");
    }

    #[test]
    fn test_create_entity_request_deserialization() {
        let json = r#"{
            "entity_name": "test entity",
            "entity_type": "CONCEPT",
            "description": "A test entity",
            "source_id": "manual_entry"
        }"#;
        let request: Result<CreateEntityRequest, _> = serde_json::from_str(json);
        assert!(request.is_ok());
        let req = request.unwrap();
        assert_eq!(req.entity_name, "test entity");
        assert_eq!(req.entity_type, "CONCEPT");
    }

    #[test]
    fn test_update_entity_request_partial() {
        // Only description
        let json = r#"{"description": "Updated description"}"#;
        let request: Result<UpdateEntityRequest, _> = serde_json::from_str(json);
        assert!(request.is_ok());
        let req = request.unwrap();
        assert!(req.entity_type.is_none());
        assert_eq!(req.description, Some("Updated description".to_string()));
    }

    #[test]
    fn test_entity_response_serialization() {
        let response = EntityResponse {
            id: "test-id".to_string(),
            entity_name: "TEST_ENTITY".to_string(),
            entity_type: "CONCEPT".to_string(),
            description: "A test".to_string(),
            source_id: "doc-1".to_string(),
            created_at: "2024-01-01T00:00:00Z".to_string(),
            updated_at: "2024-01-01T00:00:00Z".to_string(),
            degree: 5,
            metadata: serde_json::Value::Null,
        };
        let json = serde_json::to_string(&response);
        assert!(json.is_ok());
        assert!(json.unwrap().contains("TEST_ENTITY"));
    }

    #[test]
    fn test_merge_entities_request_deserialization() {
        let json = r#"{
            "source_entity": "ENTITY_A",
            "target_entity": "ENTITY_B"
        }"#;
        let request: Result<MergeEntitiesRequest, _> = serde_json::from_str(json);
        assert!(request.is_ok());
        let req = request.unwrap();
        assert_eq!(req.source_entity, "ENTITY_A");
        assert_eq!(req.target_entity, "ENTITY_B");
    }

    #[test]
    fn test_delete_entity_query_deserialization() {
        let json = r#"{"delete_relationships": true, "confirm": true}"#;
        let query: Result<DeleteEntityQuery, _> = serde_json::from_str(json);
        assert!(query.is_ok());
        let q = query.unwrap();
        assert!(q.delete_relationships);
        assert!(q.confirm);
    }

    #[test]
    fn test_entity_statistics_serialization() {
        let stats = EntityStatistics {
            total_relationships: 100,
            outgoing_count: 50,
            incoming_count: 50,
            document_references: 10,
        };
        let json = serde_json::to_string(&stats);
        assert!(json.is_ok());
    }
}
