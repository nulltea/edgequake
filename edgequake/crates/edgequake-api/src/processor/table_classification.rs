//! Table-classification task processor.
//!
//! Reads every `chunks` row for the document where `kind='table'` and
//! `table_type IS NULL`, asks the workspace's configured LLM to bucket each
//! one as `performance` / `quality` / `complexity` / `other`, and stores the
//! label + rationale back on the row. Best-effort per row: one failure
//! doesn't poison the rest of the document.
//!
//! Runs as a follow-up to algorithm extraction in `process_pdf_processing`
//! (not inline with PDF conversion) so the document is fully visible in the
//! UI before classifications start trickling in.

use super::*;
use tokio_util::sync::CancellationToken;

/// Permitted classification labels. Must stay in sync with the prompt below
/// and the gallery filter chips on the frontend.
const TABLE_TYPES: &[&str] = &["performance", "quality", "complexity", "other"];

impl DocumentTaskProcessor {
    /// Entry point wired into `TaskProcessor::process` for
    /// `TaskType::TableClassification`. Wraps `run_table_classification_inline`
    /// with task progress updates.
    #[cfg(feature = "postgres")]
    pub(super) async fn process_table_classification(
        &self,
        task: &mut Task,
        data: edgequake_tasks::TableClassificationData,
        cancel_token: CancellationToken,
    ) -> TaskResult<serde_json::Value> {
        info!(
            document_id = %data.document_id,
            workspace_id = %data.workspace_id,
            "Processing table classification task"
        );
        task.update_progress("table_classification".to_string(), 1, 10);
        let document_uuid = uuid::Uuid::parse_str(&data.document_id)
            .map_err(|e| TaskError::Process(format!("Invalid document_id: {e}")))?;
        let (classified, total) = self
            .run_table_classification_inline(
                &data.workspace_id,
                task.workspace_id,
                document_uuid,
                cancel_token,
            )
            .await
            .map_err(TaskError::Process)?;
        task.update_progress("completed".to_string(), 1, 100);
        Ok(serde_json::json!({
            "classified": classified,
            "total": total,
        }))
    }

    /// Shared helper: classify every unlabeled table row for the document.
    /// Safe to call inline from other processors (e.g. `process_pdf_processing`
    /// kicks this off after algorithm extraction). Returns `(classified,
    /// total_unclassified_at_start)`. Per-row failures are logged and the
    /// loop continues; only failures to talk to Postgres at all abort.
    #[cfg(feature = "postgres")]
    pub(super) async fn run_table_classification_inline(
        &self,
        workspace_id_str: &str,
        workspace_id_uuid: uuid::Uuid,
        document_uuid: uuid::Uuid,
        cancel_token: CancellationToken,
    ) -> Result<(usize, usize), String> {
        let workspace_id_opt = if !workspace_id_str.is_empty() && workspace_id_str != "default" {
            Some(workspace_id_str)
        } else {
            None
        };
        let llm = self
            .resolve_table_classification_llm(workspace_id_opt, workspace_id_uuid)
            .await;

        let database_url =
            std::env::var("DATABASE_URL").map_err(|_| "DATABASE_URL not set".to_string())?;
        let pool = sqlx::PgPool::connect(&database_url)
            .await
            .map_err(|e| format!("Failed to connect to database: {e}"))?;

        let rows = sqlx::query_as::<_, (String, String, Option<String>)>(
            r#"
            SELECT table_id, COALESCE(table_html, ''), content
            FROM chunks
            WHERE document_id = $1
              AND kind = 'table'
              AND table_type IS NULL
            ORDER BY chunk_index
            "#,
        )
        .bind(document_uuid)
        .fetch_all(&pool)
        .await
        .map_err(|e| format!("Failed to query tables: {e}"))?;

        let total = rows.len();
        if total == 0 {
            info!(
                document_id = %document_uuid,
                "Table classification: no unclassified tables for document"
            );
            return Ok((0, 0));
        }

        let mut classified = 0usize;
        for (table_id, html, caption) in rows.into_iter() {
            if cancel_token.is_cancelled() {
                info!(
                    document_id = %document_uuid,
                    classified = classified,
                    "Table classification: cancelled mid-batch"
                );
                break;
            }
            let caption_text = caption.unwrap_or_default();
            let prompt = build_classification_prompt(&caption_text, &html);
            let (label, rationale) = match llm.complete(&prompt).await {
                Ok(resp) => parse_classification_response(&resp.content),
                Err(e) => {
                    warn!(
                        document_id = %document_uuid,
                        table_id = %table_id,
                        error = %e,
                        "Table classification: LLM call failed; leaving row unlabelled"
                    );
                    continue;
                }
            };
            let res = sqlx::query(
                r#"
                UPDATE chunks
                SET table_type = $1,
                    table_classification_rationale = $2
                WHERE document_id = $3 AND table_id = $4 AND kind = 'table'
                "#,
            )
            .bind(&label)
            .bind(&rationale)
            .bind(document_uuid)
            .bind(&table_id)
            .execute(&pool)
            .await;
            match res {
                Ok(_) => {
                    classified += 1;
                }
                Err(e) => warn!(
                    document_id = %document_uuid,
                    table_id = %table_id,
                    error = %e,
                    "Table classification: row update failed; continuing"
                ),
            }
        }
        info!(
            document_id = %document_uuid,
            classified = classified,
            total = total,
            "Table classification: completed"
        );
        Ok((classified, total))
    }

    /// Workspace LLM resolution for table classification. Mirrors
    /// `resolve_workspace_llm` in repo_detection.rs — but standalone here so
    /// we don't expose the private method across modules.
    #[cfg(feature = "postgres")]
    async fn resolve_table_classification_llm(
        &self,
        workspace_id_str: Option<&str>,
        workspace_id_uuid: uuid::Uuid,
    ) -> std::sync::Arc<dyn edgequake_llm::traits::LLMProvider> {
        use crate::safety_limits::create_safe_llm_provider;
        let default = std::sync::Arc::clone(&self.llm_provider);
        if workspace_id_str.is_none() {
            return default;
        }
        let Some(ws_svc) = self.workspace_service.as_ref() else {
            return default;
        };
        match ws_svc.get_workspace(workspace_id_uuid).await {
            Ok(Some(ws)) => match create_safe_llm_provider(&ws.llm_provider, &ws.llm_model) {
                Ok(p) => {
                    info!(
                        workspace_id = %workspace_id_uuid,
                        provider = %ws.llm_provider,
                        model = %ws.llm_model,
                        "table-classification: using workspace LLM"
                    );
                    p
                }
                Err(e) => {
                    warn!(
                        workspace_id = %workspace_id_uuid,
                        error = %e,
                        "table-classification: workspace LLM init failed; falling back to server default"
                    );
                    default
                }
            },
            _ => default,
        }
    }
}

/// Compose the prompt for one table. Kept short and JSON-strict so a small
/// classifier model (e.g. the workspace's default chat model) can answer
/// reliably. We pass both the caption (often "Table 3: Latency on …") and
/// the rendered HTML so the LLM has the column headers and a sample of the
/// data — that combination is what disambiguates "performance" from
/// "quality" cleanly.
fn build_classification_prompt(caption: &str, html: &str) -> String {
    let trimmed_html = if html.len() > 8000 {
        &html[..8000]
    } else {
        html
    };
    let caption_line = if caption.trim().is_empty() {
        "(no caption detected)".to_string()
    } else {
        caption.trim().to_string()
    };
    format!(
        "Classify the following table from an academic / technical paper into exactly \
one of these categories:\n\
- \"performance\": runtime / throughput / latency / memory / cost benchmarks.\n\
- \"quality\":     accuracy / F1 / BLEU / win-rate / human-eval / robustness scores.\n\
- \"complexity\":  big-O / step counts / round complexity / asymptotic comparisons.\n\
- \"other\":       anything else (datasets, hyperparameters, ablation knobs, etc.).\n\
\n\
Reply with ONLY a single-line JSON object of this exact shape:\n\
{{\"table_type\": \"<one of the four labels above>\", \"rationale\": \"<one short sentence>\"}}\n\
\n\
Caption: {caption_line}\n\
HTML: {trimmed_html}\n"
    )
}

/// Parse the LLM's reply. Tolerates the model wrapping the JSON in code
/// fences or trailing prose — extracts the first balanced `{…}` block and
/// validates the `table_type` field against the allowlist. Falls back to
/// `("other", raw_reply)` when nothing parseable comes back so the row still
/// gets a label.
fn parse_classification_response(reply: &str) -> (String, String) {
    let trimmed = reply.trim();
    let json_blob = extract_first_json_object(trimmed).unwrap_or(trimmed);
    let parsed: serde_json::Value = match serde_json::from_str(json_blob) {
        Ok(v) => v,
        Err(_) => {
            return (
                "other".to_string(),
                format!("unparseable LLM reply: {}", short(reply, 240)),
            );
        }
    };
    let raw_type = parsed
        .get("table_type")
        .and_then(|v| v.as_str())
        .unwrap_or("other")
        .trim()
        .to_lowercase();
    let label = if TABLE_TYPES.contains(&raw_type.as_str()) {
        raw_type
    } else {
        "other".to_string()
    };
    let rationale = parsed
        .get("rationale")
        .and_then(|v| v.as_str())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| short(reply, 240));
    (label, rationale)
}

/// Find the first balanced `{…}` substring. Naive: matches braces without
/// respecting strings / escapes. Sufficient for short, well-formed JSON-only
/// replies; the parse failure path catches the rest.
fn extract_first_json_object(s: &str) -> Option<&str> {
    let start = s.find('{')?;
    let mut depth = 0i32;
    for (i, c) in s[start..].char_indices() {
        match c {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[start..start + i + 1]);
                }
            }
            _ => {}
        }
    }
    None
}

fn short(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let cut: String = s.chars().take(max).collect();
        format!("{cut}…")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_strict_json_response() {
        let raw = r#"{"table_type":"performance","rationale":"Reports throughput in tok/s."}"#;
        let (label, why) = parse_classification_response(raw);
        assert_eq!(label, "performance");
        assert!(why.starts_with("Reports throughput"));
    }

    #[test]
    fn parse_code_fenced_response() {
        let raw = "```json\n{\"table_type\":\"quality\",\"rationale\":\"Accuracy across benchmarks.\"}\n```";
        let (label, _) = parse_classification_response(raw);
        assert_eq!(label, "quality");
    }

    #[test]
    fn parse_with_trailing_prose() {
        let raw = "Sure! {\"table_type\":\"complexity\",\"rationale\":\"Round complexity O(log n).\"} Hope that helps.";
        let (label, why) = parse_classification_response(raw);
        assert_eq!(label, "complexity");
        assert!(why.contains("Round complexity"));
    }

    #[test]
    fn parse_unknown_label_falls_back_to_other() {
        let raw = r#"{"table_type":"performance-vs-quality","rationale":"…"}"#;
        let (label, _) = parse_classification_response(raw);
        assert_eq!(label, "other");
    }

    #[test]
    fn parse_unparseable_returns_other_with_raw_in_rationale() {
        let raw = "I think this is a performance table.";
        let (label, why) = parse_classification_response(raw);
        assert_eq!(label, "other");
        assert!(why.contains("unparseable") || why.contains("performance"));
    }
}
