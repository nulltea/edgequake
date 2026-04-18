//! Live integration test for `PostgresCodeArtifactStorage`.
//!
//! Skipped unless `DATABASE_URL` points at a Postgres with migration 043
//! applied. Tests namespace by random UUIDs so they don't collide with
//! running backend data.
//!
//! Seeds a row in `algorithms` + `document_repos` (required by FK) before
//! the test runs, and cleans up at the end.

#![cfg(feature = "postgres")]

use edgequake_agents::code_analysis::{
    storage::{CodeArtifactStorage, PostgresCodeArtifactStorage},
    types::{ArtifactStatus, CodeArtifactCandidate, CodeReferenceRun, MatchConfidence, RunStatus},
};
use sqlx::PgPool;
use uuid::Uuid;

struct Fixture {
    storage: PostgresCodeArtifactStorage,
    pool: PgPool,
    tenant: Uuid,
    workspace: Uuid,
    doc: String,
    algorithm_id: Uuid,
    document_repo_id: Uuid,
}

async fn setup() -> Option<Fixture> {
    let url = std::env::var("DATABASE_URL")
        .ok()
        .filter(|v| !v.is_empty())?;
    let pool = PgPool::connect(&url).await.expect("connect postgres");
    let tenant = Uuid::new_v4();
    let workspace = Uuid::new_v4();
    let doc = format!("test-code-{}", Uuid::new_v4());

    // Seed algorithm row (FK target).
    let algorithm_id = Uuid::new_v4();
    sqlx::query(
        r#"INSERT INTO algorithms (id, tenant_id, workspace_id, document_id, name, status, confidence)
           VALUES ($1, $2, $3, $4, $5, 'approved', 'high')"#,
    )
    .bind(algorithm_id)
    .bind(tenant)
    .bind(workspace)
    .bind(&doc)
    .bind("test-algorithm")
    .execute(&pool)
    .await
    .expect("seed algorithm");

    // Seed document_repos row (FK target).
    let document_repo_id = Uuid::new_v4();
    sqlx::query(
        r#"INSERT INTO document_repos
            (id, tenant_id, workspace_id, document_id, host, owner, repo, url, detection_method, confidence, status)
           VALUES ($1, $2, $3, $4, 'github', 'foo', 'bar', 'https://github.com/foo/bar', 'pdf_link', 'high', 'approved')"#,
    )
    .bind(document_repo_id)
    .bind(tenant)
    .bind(workspace)
    .bind(&doc)
    .execute(&pool)
    .await
    .expect("seed document_repo");

    Some(Fixture {
        storage: PostgresCodeArtifactStorage::new(pool.clone()),
        pool,
        tenant,
        workspace,
        doc,
        algorithm_id,
        document_repo_id,
    })
}

async fn teardown(f: &Fixture) {
    // ON DELETE CASCADE on document_repos → code_artifacts + code_reference_runs,
    // and algorithms → code_artifacts, so deleting the two parents clears everything.
    let _ = sqlx::query(r#"DELETE FROM algorithms WHERE id = $1"#)
        .bind(f.algorithm_id)
        .execute(&f.pool)
        .await;
    let _ = sqlx::query(r#"DELETE FROM document_repos WHERE id = $1"#)
        .bind(f.document_repo_id)
        .execute(&f.pool)
        .await;
}

fn candidate(f: &Fixture, start: i32, end: i32) -> CodeArtifactCandidate {
    CodeArtifactCandidate {
        algorithm_id: f.algorithm_id,
        document_repo_id: f.document_repo_id,
        repo_commit: "abc123".into(),
        repo_license: Some("MIT".into()),
        language: "python".into(),
        file_path: "model.py".into(),
        symbol_name: Some("CausalSelfAttention".into()),
        start_line: start,
        end_line: end,
        snippet: format!("def ...\n# lines {start}..{end}\n"),
        match_rationale: Some("because".into()),
        match_confidence: MatchConfidence::High,
    }
}

#[tokio::test]
async fn upsert_then_list() {
    let Some(f) = setup().await else {
        eprintln!("SKIP: DATABASE_URL not set");
        return;
    };
    let rows = f
        .storage
        .upsert_candidates(f.tenant, f.workspace, &f.doc, &[candidate(&f, 29, 76)])
        .await
        .unwrap();
    assert_eq!(rows.len(), 1);
    assert!(matches!(rows[0].status, ArtifactStatus::Pending));

    let listed = f
        .storage
        .list_for_document(f.tenant, f.workspace, &f.doc)
        .await
        .unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0].file_path, "model.py");
    assert_eq!(listed[0].start_line, 29);
    assert_eq!(listed[0].end_line, 76);

    teardown(&f).await;
}

#[tokio::test]
async fn upsert_is_idempotent_on_natural_key() {
    let Some(f) = setup().await else {
        eprintln!("SKIP: DATABASE_URL not set");
        return;
    };
    let c = candidate(&f, 29, 76);
    f.storage
        .upsert_candidates(f.tenant, f.workspace, &f.doc, std::slice::from_ref(&c))
        .await
        .unwrap();
    f.storage
        .upsert_candidates(f.tenant, f.workspace, &f.doc, &[c])
        .await
        .unwrap();
    let listed = f
        .storage
        .list_for_document(f.tenant, f.workspace, &f.doc)
        .await
        .unwrap();
    assert_eq!(listed.len(), 1);
    teardown(&f).await;
}

#[tokio::test]
async fn update_status_survives_reupsert() {
    let Some(f) = setup().await else {
        eprintln!("SKIP: DATABASE_URL not set");
        return;
    };
    let [row]: [_; 1] = f
        .storage
        .upsert_candidates(f.tenant, f.workspace, &f.doc, &[candidate(&f, 29, 76)])
        .await
        .unwrap()
        .try_into()
        .unwrap();
    let flipped = f
        .storage
        .update_status(row.id, f.tenant, f.workspace, ArtifactStatus::Approved)
        .await
        .unwrap();
    assert!(flipped);

    // Re-run the analyzer → upsert with same natural key. Status must stick.
    f.storage
        .upsert_candidates(f.tenant, f.workspace, &f.doc, &[candidate(&f, 29, 76)])
        .await
        .unwrap();
    let listed = f
        .storage
        .list_for_document(f.tenant, f.workspace, &f.doc)
        .await
        .unwrap();
    assert!(matches!(listed[0].status, ArtifactStatus::Approved));
    teardown(&f).await;
}

#[tokio::test]
async fn run_lifecycle() {
    let Some(f) = setup().await else {
        eprintln!("SKIP: DATABASE_URL not set");
        return;
    };
    assert!(f
        .storage
        .get_run(f.tenant, f.workspace, &f.doc, f.document_repo_id)
        .await
        .unwrap()
        .is_none());

    f.storage
        .mark_run_status(
            f.tenant,
            f.workspace,
            &f.doc,
            f.document_repo_id,
            RunStatus::Cloning,
        )
        .await
        .unwrap();
    let r = f
        .storage
        .get_run(f.tenant, f.workspace, &f.doc, f.document_repo_id)
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(r.status, RunStatus::Cloning));

    f.storage
        .mark_run_complete(
            f.tenant,
            f.workspace,
            &f.doc,
            f.document_repo_id,
            3,
            5,
            Some(0.12),
        )
        .await
        .unwrap();
    let r = f
        .storage
        .get_run(f.tenant, f.workspace, &f.doc, f.document_repo_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(r.algorithm_count, 3);
    assert_eq!(r.finding_count, 5);
    assert_eq!(r.cost_usd_equivalent, Some(0.12));
    assert!(matches!(r.status, RunStatus::AwaitingReview));
    assert!(r.completed_at.is_some());

    f.storage
        .mark_run_failed(f.tenant, f.workspace, &f.doc, f.document_repo_id, "boom")
        .await
        .unwrap();
    let r = f
        .storage
        .get_run(f.tenant, f.workspace, &f.doc, f.document_repo_id)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(r.error_message.as_deref(), Some("boom"));
    assert!(matches!(r.status, RunStatus::Failed));

    // Suppress unused CodeReferenceRun import check.
    let _ = std::mem::size_of::<CodeReferenceRun>();

    teardown(&f).await;
}
