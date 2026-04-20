//! Live integration test for `PostgresRepoStorage`.
//!
//! Skipped unless `DATABASE_URL` points at a Postgres that has migration 042
//! applied. No cleanup between test cases — we namespace by a random UUID per
//! tenant/workspace/document so tests don't collide with each other or with
//! the running backend's data.

#![cfg(feature = "postgres")]

use edgequake_agents::repo_detection::{
    storage::{PostgresRepoStorage, RepoStorage},
    types::{Confidence, DetectionMethod, RepoCandidate, RepoHost, RepoStatus},
};
use sqlx::PgPool;
use uuid::Uuid;

async fn setup() -> Option<(PostgresRepoStorage, Uuid, Uuid, String)> {
    let url = std::env::var("DATABASE_URL")
        .ok()
        .filter(|v| !v.is_empty())?;
    let pool = PgPool::connect(&url).await.expect("connect postgres");
    let storage = PostgresRepoStorage::new(pool);
    Some((
        storage,
        Uuid::new_v4(),
        Uuid::new_v4(),
        format!("test-doc-{}", Uuid::new_v4()),
    ))
}

fn sample_candidate(layer: DetectionMethod, rank: i32) -> RepoCandidate {
    RepoCandidate {
        host: RepoHost::Github,
        owner: "foo".into(),
        repo: "bar".into(),
        url: "https://github.com/foo/bar".into(),
        detection_method: layer,
        pdf_page_index: match layer {
            DetectionMethod::PdfLink => Some(2),
            DetectionMethod::WebSearch => None,
        },
        search_rank: match layer {
            DetectionMethod::WebSearch => Some(rank),
            DetectionMethod::PdfLink => None,
        },
        source_url: match layer {
            DetectionMethod::WebSearch => Some("https://github.com/foo/bar".into()),
            DetectionMethod::PdfLink => None,
        },
        confidence: Confidence::High,
        verification: None,
    }
}

#[tokio::test]
async fn upsert_then_list_round_trip() {
    let Some((storage, tenant, workspace, doc)) = setup().await else {
        eprintln!("SKIP: DATABASE_URL not set");
        return;
    };
    let cand = sample_candidate(DetectionMethod::PdfLink, 0);
    storage
        .upsert_candidates(tenant, workspace, &doc, &[cand])
        .await
        .unwrap();

    let rows = storage
        .list_for_document(tenant, workspace, &doc)
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    let row = &rows[0];
    assert_eq!(row.owner, "foo");
    assert_eq!(row.repo, "bar");
    assert!(matches!(row.detection_method, DetectionMethod::PdfLink));
    assert!(matches!(row.confidence, Confidence::High));
    assert!(matches!(row.status, RepoStatus::Pending));
}

#[tokio::test]
async fn upsert_is_idempotent_on_natural_key() {
    let Some((storage, tenant, workspace, doc)) = setup().await else {
        eprintln!("SKIP: DATABASE_URL not set");
        return;
    };
    let cand = sample_candidate(DetectionMethod::PdfLink, 0);
    storage
        .upsert_candidates(tenant, workspace, &doc, std::slice::from_ref(&cand))
        .await
        .unwrap();
    storage
        .upsert_candidates(tenant, workspace, &doc, &[cand])
        .await
        .unwrap();

    let rows = storage
        .list_for_document(tenant, workspace, &doc)
        .await
        .unwrap();
    assert_eq!(rows.len(), 1, "natural-key upsert should not duplicate");
}

#[tokio::test]
async fn update_status_flips_review_state() {
    let Some((storage, tenant, workspace, doc)) = setup().await else {
        eprintln!("SKIP: DATABASE_URL not set");
        return;
    };
    storage
        .upsert_candidates(
            tenant,
            workspace,
            &doc,
            &[sample_candidate(DetectionMethod::WebSearch, 1)],
        )
        .await
        .unwrap();
    let row = storage
        .list_for_document(tenant, workspace, &doc)
        .await
        .unwrap()[0]
        .clone();

    let flipped = storage
        .update_status(row.id, tenant, workspace, RepoStatus::Approved)
        .await
        .unwrap();
    assert!(flipped);

    let rows = storage
        .list_for_document(tenant, workspace, &doc)
        .await
        .unwrap();
    assert!(matches!(rows[0].status, RepoStatus::Approved));

    // Wrong tenant/workspace = no-op, not an error.
    let other = storage
        .update_status(row.id, Uuid::new_v4(), Uuid::new_v4(), RepoStatus::Rejected)
        .await
        .unwrap();
    assert!(!other);
}

#[tokio::test]
async fn detection_run_lifecycle() {
    let Some((storage, tenant, workspace, doc)) = setup().await else {
        eprintln!("SKIP: DATABASE_URL not set");
        return;
    };
    assert!(storage
        .get_detection_run(tenant, workspace, &doc)
        .await
        .unwrap()
        .is_none());

    storage
        .mark_detection_running(tenant, workspace, &doc)
        .await
        .unwrap();
    let r = storage
        .get_detection_run(tenant, workspace, &doc)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(
        r.status,
        edgequake_agents::repo_detection::types::DetectionRunStatus::Running
    ));

    storage
        .mark_detection_complete(tenant, workspace, &doc, 2, 0)
        .await
        .unwrap();
    let r = storage
        .get_detection_run(tenant, workspace, &doc)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(r.layer_a_candidates, 2);
    assert_eq!(r.layer_b_candidates, 0);
    assert!(r.completed_at.is_some());
    assert!(matches!(
        r.status,
        edgequake_agents::repo_detection::types::DetectionRunStatus::Complete
    ));

    // Re-running flips state back to Running with cleared error/completed_at.
    storage
        .mark_detection_running(tenant, workspace, &doc)
        .await
        .unwrap();
    let r = storage
        .get_detection_run(tenant, workspace, &doc)
        .await
        .unwrap()
        .unwrap();
    assert!(r.completed_at.is_none());
    assert!(r.error_message.is_none());

    storage
        .mark_detection_failed(tenant, workspace, &doc, "searxng down")
        .await
        .unwrap();
    let r = storage
        .get_detection_run(tenant, workspace, &doc)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(r.error_message.as_deref(), Some("searxng down"));
    assert!(matches!(
        r.status,
        edgequake_agents::repo_detection::types::DetectionRunStatus::Failed
    ));
}
