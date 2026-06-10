//! Postgres impl of [`ReferenceStorage`] over the `document_references` table.

use async_trait::async_trait;
use sqlx::{PgPool, Row as SqlxRow};
use std::sync::Arc;
use uuid::Uuid;

use crate::error::{Result, StorageError};
use crate::traits::{DocumentReference, NewDocumentReference, ReferenceStorage};

pub struct PgReferenceStorage {
    pool: Arc<PgPool>,
}

impl PgReferenceStorage {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool: Arc::new(pool),
        }
    }

    pub fn from_arc(pool: Arc<PgPool>) -> Self {
        Self { pool }
    }
}

fn row_to_reference(row: &sqlx::postgres::PgRow) -> DocumentReference {
    DocumentReference {
        id: row.get("id"),
        document_id: row.get("document_id"),
        reference_number: row.get("reference_number"),
        raw_text: row.get("raw_text"),
        doi: row.get("doi"),
        url: row.get("url"),
    }
}

#[async_trait]
impl ReferenceStorage for PgReferenceStorage {
    async fn replace_references(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        document_id: &str,
        references: &[NewDocumentReference],
    ) -> Result<()> {
        let mut tx = self
            .pool
            .begin()
            .await
            .map_err(|e| StorageError::Database(e.to_string()))?;

        sqlx::query(
            "DELETE FROM document_references \
             WHERE tenant_id = $1 AND workspace_id = $2 AND document_id = $3",
        )
        .bind(tenant_id)
        .bind(workspace_id)
        .bind(document_id)
        .execute(&mut *tx)
        .await
        .map_err(|e| StorageError::Database(e.to_string()))?;

        for r in references {
            sqlx::query(
                "INSERT INTO document_references \
                 (tenant_id, workspace_id, document_id, reference_number, raw_text, doi, url) \
                 VALUES ($1, $2, $3, $4, $5, $6, $7)",
            )
            .bind(tenant_id)
            .bind(workspace_id)
            .bind(document_id)
            .bind(r.reference_number)
            .bind(&r.raw_text)
            .bind(&r.doi)
            .bind(&r.url)
            .execute(&mut *tx)
            .await
            .map_err(|e| StorageError::Database(e.to_string()))?;
        }

        tx.commit()
            .await
            .map_err(|e| StorageError::Database(e.to_string()))?;
        Ok(())
    }

    async fn list_references(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        document_id: &str,
    ) -> Result<Vec<DocumentReference>> {
        let rows = sqlx::query(
            "SELECT id, document_id, reference_number, raw_text, doi, url \
             FROM document_references \
             WHERE tenant_id = $1 AND workspace_id = $2 AND document_id = $3 \
             ORDER BY reference_number ASC",
        )
        .bind(tenant_id)
        .bind(workspace_id)
        .bind(document_id)
        .fetch_all(&*self.pool)
        .await
        .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(rows.iter().map(row_to_reference).collect())
    }

    async fn references_by_numbers(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        document_id: &str,
        numbers: &[i32],
    ) -> Result<Vec<DocumentReference>> {
        if numbers.is_empty() {
            return Ok(Vec::new());
        }
        let rows = sqlx::query(
            "SELECT id, document_id, reference_number, raw_text, doi, url \
             FROM document_references \
             WHERE tenant_id = $1 AND workspace_id = $2 AND document_id = $3 \
               AND reference_number = ANY($4) \
             ORDER BY reference_number ASC",
        )
        .bind(tenant_id)
        .bind(workspace_id)
        .bind(document_id)
        .bind(numbers)
        .fetch_all(&*self.pool)
        .await
        .map_err(|e| StorageError::Database(e.to_string()))?;

        Ok(rows.iter().map(row_to_reference).collect())
    }

    async fn delete_references(
        &self,
        tenant_id: Uuid,
        workspace_id: Uuid,
        document_id: &str,
    ) -> Result<()> {
        sqlx::query(
            "DELETE FROM document_references \
             WHERE tenant_id = $1 AND workspace_id = $2 AND document_id = $3",
        )
        .bind(tenant_id)
        .bind(workspace_id)
        .bind(document_id)
        .execute(&*self.pool)
        .await
        .map_err(|e| StorageError::Database(e.to_string()))?;
        Ok(())
    }
}
