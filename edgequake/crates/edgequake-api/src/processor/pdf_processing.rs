use super::*;
use tokio_util::sync::CancellationToken;

#[cfg(feature = "postgres")]
fn strip_nul_bytes(text: String) -> String {
    if !text.contains('\0') {
        return text;
    }

    let nul_count = text.chars().filter(|&ch| ch == '\0').count();
    let sanitized = text.replace('\0', "");
    warn!(
        nul_count,
        sanitized_len = sanitized.len(),
        "Removed NUL bytes from extracted PDF markdown before persistence"
    );
    sanitized
}

impl DocumentTaskProcessor {
    /// Process PDF processing task (SPEC-007).
    ///
    /// This method handles the complete PDF processing pipeline:
    /// 1. Load PDF from storage using pdf_id
    /// 2. Extract content (text mode only for now, vision TODO)
    /// 3. Convert to markdown
    /// 4. Create document and trigger standard ingestion
    /// 5. Update PDF status with results
    ///
    /// @implements SPEC-007: PDF Upload Support with Vision LLM Integration
    /// @implements FEAT0704: PDF processing worker
    /// @implements UC0704: System processes PDF in background
    /// @enforces BR0704: PDF processed async with retry logic
    #[cfg(feature = "postgres")]
    pub(super) async fn process_pdf_processing(
        &self,
        task: &mut Task,
        data: edgequake_tasks::PdfProcessingData,
        cancel_token: CancellationToken,
    ) -> TaskResult<serde_json::Value> {
        use edgequake_storage::{
            ExtractionMethod, PdfProcessingStatus, UpdatePdfProcessingRequest,
        };

        info!(
            pdf_id = %data.pdf_id,
            workspace_id = %data.workspace_id,
            enable_vision = data.enable_vision,
            "Starting PDF processing task"
        );

        // 1. Get PDF storage
        let pdf_storage = self.pdf_storage.as_ref().ok_or_else(|| {
            edgequake_tasks::TaskError::UnsupportedOperation(
                "PDF storage not available (postgres feature enabled but storage not initialized)"
                    .to_string(),
            )
        })?;

        // 2. Load PDF from storage
        let pdf = pdf_storage.get_pdf(&data.pdf_id).await.map_err(|e| {
            edgequake_tasks::TaskError::Storage(format!(
                "Failed to load PDF {}: {}",
                data.pdf_id, e
            ))
        })?;

        // Handle case where PDF not found
        let pdf = pdf.ok_or_else(|| {
            edgequake_tasks::TaskError::NotFound(format!("PDF not found: {}", data.pdf_id))
        })?;

        info!(
            pdf_id = %data.pdf_id,
            filename = %pdf.filename,
            size = pdf.file_size_bytes,
            pages = ?pdf.page_count,
            "Loaded PDF from storage"
        );

        // 3. Update status to processing
        pdf_storage
            .update_pdf_status(&data.pdf_id, PdfProcessingStatus::Processing)
            .await
            .map_err(|e| edgequake_tasks::TaskError::Storage(e.to_string()))?;

        // == Progress: loading complete, preparing for conversion ==
        task.update_progress("pdf_loading".to_string(), 1, 5);

        // 3.1 Create document metadata early with "converting" stage
        // WHY: Users need to see the document appear in the UI immediately with visual feedback
        // showing that PDF → Markdown conversion is happening.
        // OODA-ITERATION-03: Include track_id for cancel button support
        // WHY: Frontend cancel button requires doc.track_id to call POST /tasks/{track_id}/cancel
        // FIX-REBUILD: When rebuilding/reprocessing, reuse the existing document ID
        // to avoid creating orphaned duplicates. Without this, the old document still
        // references the same pdf_id whose markdown_content gets overwritten, causing
        // it to display wrong/hallucinated content from the new extraction.
        let is_reprocess = data.existing_document_id.is_some();
        let early_doc_id = data
            .existing_document_id
            .clone()
            .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

        // FIX-DUPLICATE-BUG: Persist the generated document ID back into task_data
        // so that worker retries (including restart-driven auto-recovery) reuse the
        // same document ID instead of creating a new UUID on each attempt. Without
        // this, a restart mid-processing leaves the on-disk task with
        // existing_document_id=None; the recovered task generates a fresh UUID and
        // we end up with duplicate documents — one orphan at the old UUID, one live
        // at the new one.
        //
        // Critically, the patched task_data must be persisted to the DB NOW — before
        // any work happens — so that a crash/restart right after this point does not
        // lose the assignment.
        if !is_reprocess {
            if let Ok(mut task_data_map) = serde_json::from_value::<
                serde_json::Map<String, serde_json::Value>,
            >(task.task_data.clone())
            {
                task_data_map.insert(
                    "existing_document_id".to_string(),
                    serde_json::json!(early_doc_id.clone()),
                );
                task.task_data = serde_json::Value::Object(task_data_map);

                if let Some(ref ts) = self.task_storage {
                    if let Err(e) = ts.update_task(task).await {
                        // Non-fatal: we can still process this task, but a restart
                        // before task completion will create a duplicate document.
                        warn!(
                            document_id = %early_doc_id,
                            track_id = %task.track_id,
                            error = %e,
                            "Failed to persist existing_document_id to task_data — restart safety degraded"
                        );
                    }
                }
            }
        }
        let metadata_key = format!("{}-metadata", early_doc_id);
        // OODA-04: Include file_size_bytes and sha256_checksum in early metadata
        // WHY: Enables complete lineage from the moment the document appears in UI.
        // Without these, users see metadata gaps until processing completes.
        let metadata_json = json!({
            "id": early_doc_id,
            "title": pdf.filename.clone(),
            "file_name": pdf.filename.clone(),
            "source_type": "pdf",
            "document_type": "pdf",
            "status": "processing",
            "current_stage": "converting",
            "stage_message": match pdf.page_count {
                Some(n) if n > 0 => format!("Converting PDF to Markdown (0/{} pages)", n),
                _ => "Converting PDF to Markdown (detecting pages...)".to_string(),
            },
            "stage_progress": 0.0,
            "pdf_id": data.pdf_id.to_string(),
            "file_size_bytes": pdf.file_size_bytes,
            "sha256_checksum": pdf.sha256_checksum,
            "page_count": pdf.page_count,
            "tenant_id": data.tenant_id.to_string(),
            "workspace_id": data.workspace_id.to_string(),
            "track_id": task.track_id.clone(),
            "created_at": chrono::Utc::now().to_rfc3339(),
            "updated_at": chrono::Utc::now().to_rfc3339(),
        });

        self.kv_storage
            .upsert(&[(metadata_key.clone(), metadata_json.clone())])
            .await
            .map_err(|e| edgequake_tasks::TaskError::Storage(e.to_string()))?;

        // Create the relational `documents` row UP FRONT so every downstream
        // write that references `documents.id` (chunks.figure_id rows from the
        // figure / table backfills, pdf_documents.document_id link, algorithm
        // rows, repo rows) finds an anchor to FK against. The previous
        // ordering deferred this to step 7 (~`ensure_document_record` at the
        // end of process_pdf_processing) — every chunks INSERT issued by the
        // early-backfill path then failed with FK violation, was logged at
        // warn-level, and swallowed, leaving figures and tables invisible on
        // first-upload PDFs.
        //
        // Idempotent: the late-stage `ensure_document_record` call at the end
        // of this function uses INSERT … ON CONFLICT (id) DO UPDATE, so
        // re-running it with the real content + final status just refreshes
        // the row we created here.
        #[cfg(feature = "postgres")]
        if let Ok(document_uuid) = uuid::Uuid::parse_str(&early_doc_id) {
            if let Err(e) = pdf_storage
                .ensure_document_record(
                    &document_uuid,
                    &data.workspace_id,
                    Some(&data.tenant_id),
                    &pdf.filename,
                    "",
                    "processing",
                )
                .await
            {
                // Treat as fatal: without the documents row in place, every
                // downstream FK-dependent write will silently lose data, so
                // failing here is the loud signal the previous design
                // suppressed.
                let msg = format!("Failed to create early documents row: {e}");
                error!(
                    document_id = %early_doc_id,
                    pdf_id = %data.pdf_id,
                    error = %e,
                    "{msg}"
                );
                return Err(edgequake_tasks::TaskError::Storage(msg));
            }
        }

        // FIX-REBUILD: When reprocessing, clean up old content and chunk KV entries
        // WHY: Old chunks with stale content must be removed before the pipeline
        // creates new ones, otherwise the document ends up with a mix of old and new chunks.
        if is_reprocess {
            info!(
                document_id = %early_doc_id,
                pdf_id = %data.pdf_id,
                "Reprocessing: cleaning up old content, chunks, vectors, and graph data before re-extraction"
            );
            // Remove old content entry
            let content_key = format!("{}-content", early_doc_id);
            let _ = self.kv_storage.delete(&[content_key]).await;

            // Remove old chunk entries
            let all_keys = self.kv_storage.keys().await.unwrap_or_default();
            let chunk_prefix = format!("{}-chunk-", early_doc_id);
            let chunk_keys: Vec<String> = all_keys
                .into_iter()
                .filter(|k| k.starts_with(&chunk_prefix))
                .collect();
            if !chunk_keys.is_empty() {
                info!(
                    document_id = %early_doc_id,
                    chunk_count = chunk_keys.len(),
                    "Removing old chunk entries"
                );
                let _ = self.kv_storage.delete(&chunk_keys).await;
            }

            // Purge vector rows for this document from the workspace
            // vector store. Without this, every reprocess accumulates
            // entity embeddings on top of the previous extraction — the
            // new extraction inserts fresh rows but the old ones stick
            // around, keeping pre-fix noise ("Theorem 1", single-letter
            // entities, PERSON entries) visible in RAG retrieval and the
            // graph view even though the current extraction would no
            // longer emit them.
            //
            // The `document_id` column on the workspace vectors table is
            // indexed, so `DELETE WHERE document_id = $1` is cheap.
            // Best-effort: failures here are logged and ignored so the
            // reprocess pipeline still runs — stale rows are tolerable,
            // a wedged reprocess is not.
            match self
                .resolve_workspace_vector_storage_for_reprocess(&data.workspace_id)
                .await
            {
                Some(ws_storage) => match ws_storage.delete_by_document_id(&early_doc_id).await {
                    Ok(n) => {
                        info!(
                            document_id = %early_doc_id,
                            rows_deleted = n,
                            "Purged workspace vector rows from prior extraction"
                        );
                    }
                    Err(e) => {
                        warn!(
                            document_id = %early_doc_id,
                            error = %e,
                            "Failed to purge workspace vector rows before reprocess (continuing)"
                        );
                    }
                },
                None => {
                    warn!(
                        document_id = %early_doc_id,
                        workspace_id = %data.workspace_id,
                        "Workspace vector storage unavailable; stale entity rows may persist"
                    );
                }
            }

            // Purge prior figure / table rows from the public `chunks` table
            // for this document. Without this, a reprocess that yields a
            // different page count or layout (media at new bboxes →
            // different `chunk_index`) leaves stale rows from the previous
            // run that the media-fetch / gallery endpoints would still
            // happily serve. Best-effort: failures are logged and the
            // reprocess continues.
            #[cfg(feature = "postgres")]
            if let Ok(doc_uuid) = uuid::Uuid::parse_str(&early_doc_id) {
                if let Ok(database_url) = std::env::var("DATABASE_URL") {
                    if let Ok(pool) = sqlx::PgPool::connect(&database_url).await {
                        match sqlx::query(
                            "DELETE FROM chunks WHERE document_id = $1 AND kind IN ('figure', 'table')",
                        )
                            .bind(doc_uuid)
                            .execute(&pool)
                            .await
                        {
                            Ok(r) => {
                                info!(
                                    document_id = %early_doc_id,
                                    rows_deleted = r.rows_affected(),
                                    "Reprocess: purged figure + table rows from chunks table"
                                );
                            }
                            Err(e) => warn!(
                                document_id = %early_doc_id,
                                error = %e,
                                "Reprocess: figure/table row purge failed (continuing)"
                            ),
                        }
                    }
                }
            }

            // Clean the AGE graph — entities/relationships whose only
            // source is this document get deleted outright; edges
            // referencing deleted nodes are dropped. Keeps the graph
            // view in sync with the fresh extraction.
            match crate::handlers::documents::storage_helpers::cleanup_document_graph_data(
                &early_doc_id,
                &self.graph_storage,
                None,
            )
            .await
            {
                Ok(stats) => {
                    info!(
                        document_id = %early_doc_id,
                        entities_removed = stats.entities_removed,
                        entities_updated = stats.entities_updated,
                        relationships_removed = stats.relationships_removed,
                        "Purged graph data from prior extraction"
                    );
                }
                Err(e) => {
                    warn!(
                        document_id = %early_doc_id,
                        error = %e,
                        "Failed to purge graph data before reprocess (continuing)"
                    );
                }
            }
        }

        info!(
            document_id = %early_doc_id,
            pdf_id = %data.pdf_id,
            is_reprocess = is_reprocess,
            "{}document metadata with 'converting' stage",
            if is_reprocess { "Updated existing " } else { "Created early " }
        );

        // OODA-09: Create progress callback for real-time page-by-page feedback
        // WHY: Users need to see extraction progress like "Extracting page 5/10..."
        // OODA-10: Also attach progress broadcaster if available for WebSocket delivery
        // OODA-16: Add filename for progress display
        let mut callback = PipelineProgressCallback::new(
            self.pipeline_state.clone(),
            data.pdf_id.to_string(),
            task.track_id.clone(),
        )
        .with_filename(pdf.filename.clone())
        .with_document_metadata(early_doc_id.clone(), Arc::clone(&self.kv_storage));

        if let Some(ref broadcaster) = self.progress_broadcaster {
            callback = callback.with_broadcaster(broadcaster.clone());
        }
        let progress_callback: Arc<dyn edgequake_pdf2md::ConversionProgressCallback> =
            Arc::new(callback);

        // 4. Extract content (vision or text mode)
        // == Progress: starting conversion (this can take 5-10+ minutes) ==
        task.update_progress("pdf_converting".to_string(), 2, 10);

        // ── CANCELLATION GATE: before vision extraction (most expensive PDF stage) ──
        self.check_cancelled(&cancel_token, "pre-vision-extraction", &early_doc_id)
            .await?;

        let backend = data.pdf_parser_backend;
        let page_count = pdf.page_count.unwrap_or(0) as usize;
        let extraction_method = match backend {
            edgequake_pdf::PdfParserBackend::Vision => ExtractionMethod::Vision,
            edgequake_pdf::PdfParserBackend::EdgeParse => ExtractionMethod::EdgeParse,
            edgequake_pdf::PdfParserBackend::VlmOcr => ExtractionMethod::VlmOcr,
        };

        let default_vision_model = || {
            use crate::handlers::pdf_upload::types::default_vision_model_for_provider;
            data.vision_model
                .clone()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| default_vision_model_for_provider(&data.vision_provider))
        };

        let vision_model = match backend {
            edgequake_pdf::PdfParserBackend::Vision => Some(default_vision_model()),
            edgequake_pdf::PdfParserBackend::EdgeParse
            | edgequake_pdf::PdfParserBackend::VlmOcr => None,
        };

        let converter = match backend {
            edgequake_pdf::PdfParserBackend::Vision => {
                if !data.enable_vision {
                    return Err(edgequake_tasks::TaskError::UnsupportedOperation(
                        "Vision PDF extraction requires enable_vision=true.".to_string(),
                    ));
                }
                #[cfg(feature = "vision")]
                {
                    // WHY: Use create_safe_vision_provider (not create_safe_llm_provider)
                    // so that local providers (Ollama, LM Studio) receive a
                    // per-page timeout derived from EDGEQUAKE_VISION_PAGE_TIMEOUT_SECS
                    // rather than being hard-capped at MAXIMUM_TIMEOUT_SECS (600s).
                    // See ADR-04-001 in mission/04-heavy-pdf.md.
                    use crate::safety_limits::create_safe_vision_provider;

                    let provider = create_safe_vision_provider(
                        &data.vision_provider,
                        vision_model.as_deref().unwrap_or_default(),
                    )
                    .map_err(|e| {
                        edgequake_tasks::TaskError::Processing(format!(
                            "Failed to create vision provider '{}': {e}",
                            data.vision_provider
                        ))
                    })?;
                    edgequake_pdf::create_pdf_converter(backend, Some(provider))
                }
                #[cfg(not(feature = "vision"))]
                {
                    return Err(edgequake_tasks::TaskError::UnsupportedOperation(
                        "Vision extraction requires the 'vision' feature flag".to_string(),
                    ));
                }
            }
            edgequake_pdf::PdfParserBackend::EdgeParse
            | edgequake_pdf::PdfParserBackend::VlmOcr => {
                edgequake_pdf::create_pdf_converter(backend, None)
            }
        };

        // WHY: Local providers (Ollama, LM Studio) run on a single GPU that is
        // memory-bound. High concurrency causes VRAM thrashing and *increases*
        // total conversion time. Cap local concurrency at 2. Cloud providers
        // retain the original scale-with-page-count formula.
        // See ADR-04-003 in mission/04-heavy-pdf.md.
        let concurrency = std::env::var("EDGEQUAKE_PDF_CONCURRENCY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or_else(|| {
                use crate::safety_limits::is_local_provider;
                if is_local_provider(&data.vision_provider) {
                    2 // Local GPU: sequential-ish to avoid VRAM thrashing
                } else {
                    match page_count {
                        0..=49 => 10,
                        50..=199 => 8,
                        200..=499 => 5,
                        _ => 3,
                    }
                }
            });
        let dpi = std::env::var("EDGEQUAKE_PDF_DPI")
            .ok()
            .and_then(|v| v.parse::<u32>().ok())
            .unwrap_or(match page_count {
                0..=499 => 150,
                500..=999 => 120,
                _ => 100,
            });
        let checkpoint_dir = std::env::var("EDGEQUAKE_CHECKPOINT_DIR").unwrap_or_else(|_| {
            let mut dir = std::env::temp_dir();
            dir.push("edgequake-checkpoints");
            dir.to_string_lossy().to_string()
        });
        // VLM-OCR: resolve base URL and model from vision settings.
        // Always use OPENAI_COMPATIBLE_BASE_URL (aperture) — OCR models are served
        // through aperture and routed by model name. The vision_model from workspace
        // settings (e.g. "GLM-OCR") selects the backend.
        let (vlm_base_url, vlm_model) = if backend == edgequake_pdf::PdfParserBackend::VlmOcr {
            let base_url = std::env::var("OPENAI_COMPATIBLE_BASE_URL").ok();
            let model = data.vision_model.clone().filter(|s| !s.is_empty());
            (base_url, model)
        } else {
            (None, None)
        };

        // VisionConversionConfig carries the progress callback + concurrency.
        // Both Vision and VLM-OCR backends read from it.
        let vision_config = edgequake_pdf::VisionConversionConfig {
            model: vision_model.clone(),
            concurrency: Some(concurrency),
            dpi: Some(dpi),
            checkpoint_dir: Some(checkpoint_dir),
            no_resume: is_reprocess,
            progress_callback: Some(progress_callback),
        };

        // VLM-OCR: create a sink to capture algorithm blocks during conversion.
        let algo_block_sink = if backend == edgequake_pdf::PdfParserBackend::VlmOcr {
            Some(Arc::new(std::sync::Mutex::new(Vec::<
                edgequake_pdf::AlgorithmBlock,
            >::new())))
        } else {
            None
        };

        // VLM-OCR: sink to capture extracted figure crops (Image/Chart/Seal
        // regions). When set, the converter rewrites the page markdown so
        // each captured figure becomes a `![<id>](edgequake-figure)` placeholder
        // and pushes the PNG bytes + caption into the sink, ready for the
        // chunker to materialise as a figure chunk.
        let figure_sink = if backend == edgequake_pdf::PdfParserBackend::VlmOcr {
            Some(Arc::new(std::sync::Mutex::new(Vec::<
                edgequake_pdf::ExtractedFigure,
            >::new())))
        } else {
            None
        };

        // VLM-OCR: parallel sink for extracted tables. The converter rewrites
        // each `<div…><table…>` block in the page markdown to a
        // `![tbl_…](edgequake-table)` placeholder so the chunker can emit a
        // Table-kind chunk; the HTML + parsed rows + caption land here.
        let table_sink = if backend == edgequake_pdf::PdfParserBackend::VlmOcr {
            Some(Arc::new(std::sync::Mutex::new(Vec::<
                edgequake_pdf::ExtractedTable,
            >::new())))
        } else {
            None
        };

        let conversion_config = edgequake_pdf::PdfConversionConfig {
            page_count_hint: pdf.page_count.map(|count| count as usize),
            table_method: None,
            filename: Some(pdf.filename.clone()),
            vision: Some(vision_config),
            vlm_base_url,
            vlm_model,
            algorithm_block_sink: algo_block_sink.clone(),
            figure_sink: figure_sink.clone(),
            table_sink: table_sink.clone(),
        };

        let markdown = match backend {
            edgequake_pdf::PdfParserBackend::Vision => {
                // WHY: EDGEQUAKE_VISION_TIMEOUT_SECS is kept for backwards
                // compatibility. When not set, use the provider-aware formula:
                //   120 + (page_count × secs_per_page_for_provider)
                // This gives ~3 720s for 120 pages with Ollama vs the previous
                // 660s, matching the real hardware requirement.
                // See ADR-04-002 in mission/04-heavy-pdf.md.
                let base_timeout_secs: u64 = std::env::var("EDGEQUAKE_VISION_TIMEOUT_SECS")
                    .ok()
                    .and_then(|v| v.parse().ok())
                    .unwrap_or(0);
                let vision_timeout_secs = if base_timeout_secs > 0 {
                    base_timeout_secs
                } else {
                    use crate::safety_limits::vision_outer_timeout_secs;
                    vision_outer_timeout_secs(&data.vision_provider, page_count)
                };
                let vision_timeout = std::time::Duration::from_secs(vision_timeout_secs);

                info!(
                    pdf_id = %data.pdf_id,
                    vision_provider = %data.vision_provider,
                    vision_model = %vision_model.clone().unwrap_or_default(),
                    page_count = page_count,
                    concurrency = concurrency,
                    dpi = dpi,
                    timeout_secs = vision_timeout_secs,
                    "Starting Vision PDF conversion"
                );

                match tokio::time::timeout(
                    vision_timeout,
                    converter.convert(&pdf.pdf_data, &conversion_config),
                )
                .await
                {
                    Ok(result) => result.map_err(|e| {
                        edgequake_tasks::TaskError::Processing(format!(
                            "PDF conversion failed: {e}"
                        ))
                    })?,
                    Err(_elapsed) => {
                        error!(
                            pdf_id = %data.pdf_id,
                            timeout_secs = vision_timeout.as_secs(),
                            "Vision extraction timed out - LLM provider may be unresponsive"
                        );
                        let _ = self
                            .update_document_status(
                                &early_doc_id,
                                "failed",
                                Some(&format!(
                                    "Vision extraction timed out after {}s. Check that the LLM provider ({}) is reachable.",
                                    vision_timeout.as_secs(),
                                    data.vision_provider
                                )),
                            )
                            .await;
                        return Err(edgequake_tasks::TaskError::Timeout(format!(
                            "Vision extraction timed out after {}s for PDF {}. Provider '{}' may be unresponsive.",
                            vision_timeout.as_secs(),
                            data.pdf_id,
                            data.vision_provider
                        )));
                    }
                }
            }
            edgequake_pdf::PdfParserBackend::EdgeParse => {
                info!(
                    pdf_id = %data.pdf_id,
                    page_count = page_count,
                    "Starting EdgeParse PDF conversion"
                );
                converter
                    .convert(&pdf.pdf_data, &conversion_config)
                    .await
                    .map_err(|e| {
                        edgequake_tasks::TaskError::Processing(format!(
                            "PDF conversion failed: {e}"
                        ))
                    })?
            }
            edgequake_pdf::PdfParserBackend::VlmOcr => {
                info!(
                    pdf_id = %data.pdf_id,
                    page_count = page_count,
                    "Starting VLM-OCR PDF conversion (PP-DocLayoutV2 + remote VLM)"
                );
                converter
                    .convert(&pdf.pdf_data, &conversion_config)
                    .await
                    .map_err(|e| {
                        edgequake_tasks::TaskError::Processing(format!(
                            "PDF conversion failed: {e}"
                        ))
                    })?
            }
        };

        let markdown = strip_nul_bytes(markdown);

        // Low-content warning for deterministic (no-LLM) backends that may fail silently
        // on scanned / image-only PDFs.
        let is_deterministic = matches!(
            backend,
            edgequake_pdf::PdfParserBackend::EdgeParse | edgequake_pdf::PdfParserBackend::VlmOcr
        );
        let extraction_errors = if is_deterministic {
            let avg_chars_per_page = markdown.len() / page_count.max(1);
            if avg_chars_per_page < 50 {
                warn!(
                    pdf_id = %data.pdf_id,
                    backend = backend.as_str(),
                    avg_chars_per_page,
                    "Low text content — PDF may be scanned/image-only"
                );
                Some(json!({
                    "low_content_warning": {
                        "avg_chars_per_page": avg_chars_per_page,
                        "message": "Low text content detected. This PDF may be image-only. Consider using Vision extraction."
                    }
                }))
            } else {
                None
            }
        } else {
            None
        };
        let extraction_warning = extraction_errors
            .as_ref()
            .and_then(|value| value.get("low_content_warning"))
            .and_then(|value| value.get("message"))
            .and_then(|value| value.as_str())
            .map(str::to_string);

        info!(
            pdf_id = %data.pdf_id,
            markdown_len = markdown.len(),
            extraction_method = ?extraction_method,
            "Extracted markdown from PDF"
        );

        // == Progress: conversion done, storing markdown ==
        task.update_progress("storing_markdown".to_string(), 3, 45);

        // 5. Store markdown in pdf_documents with extraction method
        let update_req = UpdatePdfProcessingRequest {
            pdf_id: data.pdf_id,
            processing_status: PdfProcessingStatus::Completed,
            markdown_content: Some(markdown.clone()),
            extraction_method: Some(extraction_method),
            extraction_errors: extraction_errors.clone(),
            document_id: None, // Will be set after document creation
            vision_model: vision_model.clone(),
        };

        pdf_storage
            .update_pdf_processing(update_req.clone())
            .await
            .map_err(|e| edgequake_tasks::TaskError::Storage(e.to_string()))?;

        // 5.5. Post-OCR rename: parse front-matter and replace the
        // upload-time filename with a citation-style name ("Author et al.
        // - Year - Title.pdf"). Applies to all PDF uploads (multipart +
        // URL) — only skipped when front-matter extraction fails or the
        // new name matches the old one. The rename also syncs the KV
        // `{doc_id}-metadata` `title`/`file_name` fields so the UI list
        // reflects the new name without another round-trip. Best-effort:
        // any failure keeps the original filename.
        let pdf = self
            .maybe_rename_from_front_matter(
                &pdf,
                &markdown,
                data.source_url.as_deref(),
                &early_doc_id,
                &pdf_storage,
            )
            .await;

        // 6a. Figure extraction (VLM-OCR path only). Drain the sink the
        //     converter populated for each Image/Chart/Seal layout region
        //     it captioned. The page markdown was rewritten in-flight to
        //     replace each `*Figure N: caption*` line with
        //     `![<id>](edgequake-figure)` so the chunker can pair these
        //     placeholders with the PNG payloads. Currently we just log
        //     the totals — the chunker integration that materialises
        //     figure chunks lives behind task #7 (see plan file).
        let extracted_figures: Vec<edgequake_pdf::ExtractedFigure> =
            if let Some(ref sink) = figure_sink {
                let figs: Vec<edgequake_pdf::ExtractedFigure> = sink
                    .lock()
                    .map(|g| g.clone())
                    .unwrap_or_else(|e| e.into_inner().clone());
                if !figs.is_empty() {
                    let total_bytes: usize = figs.iter().map(|f| f.image_bytes.len()).sum();
                    info!(
                        pdf_id = %data.pdf_id,
                        figure_count = figs.len(),
                        total_image_bytes = total_bytes,
                        "VLM-OCR: captured figures (chunker integration pending)"
                    );
                }
                figs
            } else {
                Vec::new()
            };
        // Suppress unused-variable warning until task #7 lands and the chunker
        // consumes this list.
        let _ = &extracted_figures;

        // Drain the table sink — parallel to figures. The HTML + parsed rows
        // are written to the chunks table by `backfill_table_content` below,
        // after `process_text_insert` has created the placeholder Table-kind
        // chunks via the chunker.
        let extracted_tables: Vec<edgequake_pdf::ExtractedTable> =
            if let Some(ref sink) = table_sink {
                let tabs: Vec<edgequake_pdf::ExtractedTable> = sink
                    .lock()
                    .map(|g| g.clone())
                    .unwrap_or_else(|e| e.into_inner().clone());
                if !tabs.is_empty() {
                    info!(
                        pdf_id = %data.pdf_id,
                        table_count = tabs.len(),
                        "VLM-OCR: captured tables"
                    );
                }
                tabs
            } else {
                Vec::new()
            };
        let _ = &extracted_tables;

        // 5b. Early media backfill — write figure PNGs and table HTML/rows
        //     into the chunks table NOW, before algorithm extraction +
        //     entity extraction run. Those steps are slow (multi-minute LLM
        //     calls) and the gallery / inline-render endpoints only need
        //     the chunks rows to serve media. Doing the inserts up front
        //     means users see figures and tables within seconds of
        //     conversion finishing instead of waiting for the whole
        //     pipeline to drain.
        //
        //     The vector-store / multimodal-embedding / entity-link steps
        //     for figures still run later (after process_text_insert)
        //     because they depend on the workspace pipeline being warm.
        //     Same for table classification.
        #[cfg(feature = "postgres")]
        {
            if !extracted_figures.is_empty() {
                if let Err(e) = backfill_figure_media(
                    &early_doc_id,
                    data.tenant_id,
                    data.workspace_id,
                    &extracted_figures,
                )
                .await
                {
                    warn!(
                        pdf_id = %data.pdf_id,
                        error = %e,
                        figure_count = extracted_figures.len(),
                        "Early figure-bytes backfill failed (non-fatal); figures will be retried after text insert"
                    );
                } else {
                    info!(
                        pdf_id = %data.pdf_id,
                        figure_count = extracted_figures.len(),
                        "Early figure-bytes backfill: wrote PNG payloads to chunks table"
                    );
                }
            }
            if !extracted_tables.is_empty() {
                if let Err(e) = backfill_table_content(
                    &early_doc_id,
                    data.tenant_id,
                    data.workspace_id,
                    &extracted_tables,
                )
                .await
                {
                    warn!(
                        pdf_id = %data.pdf_id,
                        error = %e,
                        table_count = extracted_tables.len(),
                        "Early table backfill failed (non-fatal); will be retried after text insert"
                    );
                } else {
                    info!(
                        pdf_id = %data.pdf_id,
                        table_count = extracted_tables.len(),
                        "Early table backfill: wrote HTML + parsed rows to chunks table"
                    );
                }
            }
        }

        // Flag that survives across both enrichment stages (algorithm
        // extraction, repo detection, late media backfill) and trips the
        // final document status from `completed` to `partial_failure` if
        // any of them fail. Declared up here so the pre-text_insert blocks
        // can set it; phase-2 (post-text_insert) backfills set it too.
        let mut enrichment_persistence_failed = false;

        // ── 6. process_text_insert (chunking → embed → persist → extract) ──
        //
        // Runs FIRST among the post-conversion stages so the document
        // reaches a queryable state ASAP. process_text_insert is itself
        // a two-phase pipeline (see Pipeline::chunk_and_embed_chunks +
        // Pipeline::extract_and_embed_graph): phase 1 chunks + embeds +
        // flushes chunks to KV and the workspace vector store; phase 2
        // runs the long LLM entity-extraction step. The moment phase 1
        // completes, hybrid query already returns chunks for this doc —
        // without waiting for entity extraction or any of the algorithm
        // / repo-detection enrichment that follows.
        //
        // Previously this block ran AFTER algorithm extraction (Pass 2+3)
        // and reference-repo detection, which delayed phase 1 by
        // however long those steps took (typically minutes on a paper
        // with multiple algorithm blocks). Reordering costs nothing:
        // algorithm extraction reads the markdown and writes its own
        // `algorithms` table; entity extraction reads the markdown and
        // writes the graph. No cross-dependency required algorithms to
        // land first — the chunker still re-processes the full markdown
        // (including algorithm blocks), and any cross-referencing
        // between algorithms and chunks/entities can happen at query
        // time against the populated tables.
        // == Progress: markdown stored, starting entity extraction ==
        task.update_progress("entity_extraction".to_string(), 5, 60);

        // ── CANCELLATION GATE: before handing off to text_insert pipeline ──
        self.check_cancelled(&cancel_token, "pre-text-insert", &early_doc_id)
            .await?;

        // SPEC-002: Include source_type: "pdf" for unified pipeline tracking
        // OODA-05: Include tenant_id/workspace_id for multi-tenant document visibility
        // OODA-04: Include sha256_checksum for end-to-end lineage traceability
        //
        // References-section strip. Academic bibliographies are dense with
        // author names + journal titles + URLs that all behave as noise in
        // the knowledge graph and pollute retrieval with citation-only
        // matches. We truncate at the first `References` / `Bibliography` /
        // `Works Cited` heading so the chunker, entity extractor, embedder,
        // and retrieval layer never see them. The FULL markdown (with the
        // bibliography) was already persisted to `pdf_documents.markdown_content`
        // above, so the document Content tab still renders the citations
        // for human reading.
        let stripped = edgequake_pdf::strip_references_section(&markdown).to_string();
        if stripped.len() < markdown.len() {
            info!(
                pdf_id = %data.pdf_id,
                full_len = markdown.len(),
                kept_len = stripped.len(),
                "Stripped references section before chunking ({} bytes dropped)",
                markdown.len() - stripped.len()
            );
        }

        // Parse + persist references from the FULL markdown (with the
        // bibliography) BEFORE it is stripped for chunking. process_text_insert
        // only sees the stripped text, so the PDF path must do this itself.
        self.parse_and_store_references(
            &early_doc_id,
            Some(&data.tenant_id.to_string()),
            &data.workspace_id.to_string(),
            &markdown,
        )
        .await;
        // Table POINTER into the chunker input (not inline GFM). Each table is
        // embedded as its own dedicated `kind='table'` vector chunk (see
        // `backfill_table_embeddings`, caption-led embed + full GFM as content),
        // so expanding the full GFM into the neighbouring prose chunk here would
        // (a) bloat that chunk's embedding with symbol-heavy grid noise and
        // (b) double-count the table in the index. Instead we leave a lightweight
        // `[[table:<id>]]` pointer in the prose chunk; at query-assembly the
        // referenced table is hydrated from its dedicated chunk and deduped by id
        // (so a referenced table still reaches context, exactly once). The stored
        // `markdown_content` keeps the original `![tbl_…](edgequake-table)`
        // placeholder for the frontend, so this only affects what the
        // chunker/embedder sees.
        let markdown_for_chunking = if extracted_tables.is_empty() {
            stripped
        } else {
            edgequake_pdf::mark_table_placeholders(&stripped, &extracted_tables)
        };
        let text_data = edgequake_tasks::TextInsertData {
            text: markdown_for_chunking,
            file_source: pdf.filename.clone(),
            workspace_id: data.workspace_id.to_string(),
            metadata: Some(json!({
                "document_id": early_doc_id.clone(),  // Reuse early document ID
                "source": "pdf_upload",
                "source_type": "pdf",
                "document_type": "pdf",
                "pdf_id": data.pdf_id.to_string(),
                "filename": pdf.filename,
                "page_count": pdf.page_count,
                "file_size_bytes": pdf.file_size_bytes,
                "sha256_checksum": pdf.sha256_checksum,
                "tenant_id": data.tenant_id.to_string(),
                "workspace_id": data.workspace_id.to_string(),
                // SPEC-040: Store PDF extraction lineage for document detail view
                "pdf_vision_model": vision_model,
                "pdf_extraction_method": extraction_method.as_str(),
                "pdf_extraction_warning": extraction_warning,
                // Chunks-only mode: process_text_insert stops after phase 1
                // (chunk + embed) and skips entity extraction.
                "skip_extraction": data.skip_extraction,
            })),
        };

        // Clone the cancellation token so the inline table-classification
        // step below (which needs it to bail on user cancel) can share the
        // same signal as the text-insert call that consumes it here.
        let cancel_token_for_classification = cancel_token.clone();
        // finalize_status: false — algorithm extraction, repo detection,
        // and the media-enrichment block all still run after this call;
        // we don't want the document to flip to "completed" while those
        // steps are pending. The caller (us) writes the final status at
        // the end of this function.
        let result = self
            .process_text_insert(task, text_data, cancel_token_for_classification.clone(), false)
            .await?;
        // Pull the final-status hint that process_text_insert returned —
        // either "completed" or "partial_failure" depending on what it
        // saw. We honour it when writing the final status so a partial
        // failure doesn't get paved over with "completed".
        let text_insert_final_status: String = result
            .get("final_status")
            .and_then(|v| v.as_str())
            .unwrap_or("completed")
            .to_string();

        // ── 7. Algorithm extraction (VLM-OCR path only — Pass 2+3) ──
        //
        // Runs AFTER process_text_insert (was: before). chunks + entities
        // are already in storage at this point, so a missing algorithm
        // row degrades the result to `partial_failure` instead of failing
        // the whole doc.
        //
        // Pass 2+3 writes the `algorithms` table directly; when auto-
        // approve is on (workspace setting), it also embeds and upserts
        // the algorithm vectors into the workspace vector store — at
        // which point algorithms are queryable alongside the chunks
        // that landed in phase 1 above.
        // Chunks-only mode: skip algorithm extraction (Pass 2+3) and
        // reference-repo detection — both are heavy LLM stages. They can be
        // triggered later via the extract endpoint.
        if !data.skip_extraction {
        if let Some(ref sink) = algo_block_sink {
            let blocks = sink.lock().unwrap_or_else(|e| e.into_inner()).clone();
            if !blocks.is_empty() {
                info!(
                    pdf_id = %data.pdf_id,
                    block_count = blocks.len(),
                    "Auto algorithm extraction: detected {} blocks during conversion, running Pass 2+3",
                    blocks.len()
                );

                self.update_document_status(&early_doc_id, "algo_extracting", None)
                    .await
                    .ok();
                task.update_progress("algo_extracting".to_string(), 5, 70);

                let workspace_id_str = data.workspace_id.to_string();
                let ws_id = if workspace_id_str != "default" && !workspace_id_str.is_empty() {
                    Some(workspace_id_str.as_str())
                } else {
                    None
                };

                // finalize_status=false: media enrichment + final status
                // are still owned by this caller — let the embedding
                // step finish without stamping `completed`.
                let expected_blocks = blocks.len();
                match self
                    .run_algorithm_pass2_pass3(&early_doc_id, ws_id, &blocks, task, false)
                    .await
                {
                    Ok((count, pass2_failures)) => {
                        info!(
                            pdf_id = %data.pdf_id,
                            algorithm_count = count,
                            expected_blocks,
                            pass2_failures,
                            "Auto algorithm extraction completed"
                        );
                        // Only flip partial_failure when at least one block's
                        // Pass 2 LLM call errored — that's information loss
                        // vs. the layout detector. `count < expected_blocks`
                        // alone is a false positive: it conflates legitimate
                        // dedup (two blocks were the same algorithm) and
                        // legitimate "LLM returned no algorithms for this
                        // block" outcomes with real failures.
                        if pass2_failures > 0 {
                            warn!(
                                pdf_id = %data.pdf_id,
                                algorithm_count = count,
                                expected_blocks,
                                pass2_failures,
                                "Pass 2 LLM failed on {pass2_failures} of {expected_blocks} block(s) — marking partial_failure"
                            );
                            enrichment_persistence_failed = true;
                        }
                    }
                    Err(e) => {
                        warn!(
                            pdf_id = %data.pdf_id,
                            error = %e,
                            expected_blocks,
                            "Auto algorithm extraction failed — marking partial_failure"
                        );
                        enrichment_persistence_failed = true;
                    }
                }
            }
        }

        // ── 8. Reference-repo detection (Phase 0 of Reference Code GraphRAG) ──
        //
        // Runs after algorithm extraction so the `document_repos` table
        // sits next to the algorithm rows on the document detail page.
        // Layer A works on PDF bytes; if nothing, Layer B falls through
        // to SearXNG + Crawl4AI.
        let pdf_data_for_detection = pdf_storage
            .get_pdf(&data.pdf_id)
            .await
            .ok()
            .flatten()
            .map(|p| p.pdf_data)
            .unwrap_or_default();
        if let Err(e) = self
            .run_repo_detection_inline(
                task.tenant_id,
                task.workspace_id,
                &early_doc_id,
                &pdf_data_for_detection,
                &markdown,
            )
            .await
        {
            warn!(
                pdf_id = %data.pdf_id,
                error = %e,
                "Reference-repo detection failed — marking partial_failure"
            );
            enrichment_persistence_failed = true;
        }
        } // end: skip algorithm + repo detection in chunks-only mode

        // (enrichment_persistence_failed is declared up at the algorithm
        // extraction step so the pre-text_insert enrichment can set it.
        // The figure/table backfills below mutate the same flag.)

        // Surface the post-text-insert sub-stage to both the document
        // detail page (`current_stage` field) and the task tracker
        // (`task.update_progress` percentages). Until this fix landed, the
        // document flipped to "completed" inside process_text_insert and
        // then sat there silently while figure/table enrichment ran for
        // another several minutes.
        self.update_document_status(&early_doc_id, "media_enrichment", None)
            .await
            .ok();
        task.update_progress("media_enrichment".to_string(), 6, 75);

        // 6b. Figure backfill. process_text_insert created figure chunk markers
        //     in the workspace vector store (kind=figure, figure_id set) but
        //     left them WITHOUT embeddings — the pipeline's text-embedding
        //     loop skips figure chunks because the placeholder text would yield
        //     a meaningless vector. The PNG bytes have been sitting in
        //     extracted_figures since the converter ran.
        //
        //     This step does two things:
        //       (a) Write the PNG bytes into the `chunks` table (BYTEA columns
        //           from migration 052) so the media-fetch endpoint can serve
        //           them by (document_id, figure_id).
        //       (b) Compute a fused multimodal embedding (caption + PNG bytes)
        //           via MultimodalEmbeddingClient and upsert it to the
        //           workspace vector store under id `{doc_id}-figure-{fid}` so
        //           queries can retrieve figures in the same vector space as
        //           text chunks.
        //
        //     Both are best-effort: failures are logged and swallowed so a
        //     downstream-service hiccup doesn't kill the PDF ingest run.
        #[cfg(feature = "postgres")]
        if !extracted_figures.is_empty() {
            if let Err(e) = backfill_figure_media(
                &early_doc_id,
                data.tenant_id,
                data.workspace_id,
                &extracted_figures,
            )
            .await
            {
                warn!(
                    pdf_id = %data.pdf_id,
                    error = %e,
                    figure_count = extracted_figures.len(),
                    "Figure-bytes backfill failed; figures captured but bytes not persisted — document will be marked partial_failure"
                );
                enrichment_persistence_failed = true;
            } else {
                info!(
                    pdf_id = %data.pdf_id,
                    figure_count = extracted_figures.len(),
                    "Figure-bytes backfill: wrote PNG payloads to chunks table"
                );
            }

            // (b) Multimodal embedding via the workspace's configured
            //     embedding provider — but only if its model card declares
            //     supports_vision = true. Otherwise the workspace is using a
            //     text-only embedder (e.g. embeddinggemma 768 / OpenAI 1536)
            //     and a 1024-dim figure vector would (a) be in a different
            //     dim from the rest of the workspace, breaking unified
            //     retrieval, and (b) fail the vector-table dimension check.
            //     Bytes are still served via the media-fetch endpoint either
            //     way; only searchability needs vision support.
            match self
                .backfill_figure_embeddings(
                    &early_doc_id,
                    &data.tenant_id.to_string(),
                    &data.workspace_id.to_string(),
                    &extracted_figures,
                )
                .await
            {
                Ok(Some(stored)) => {
                    info!(
                        pdf_id = %data.pdf_id,
                        embedded = stored,
                        total = extracted_figures.len(),
                        "Figure-embedding backfill: wrote vectors to workspace vector store"
                    );
                }
                Ok(None) => {
                    info!(
                        pdf_id = %data.pdf_id,
                        figure_count = extracted_figures.len(),
                        "Figure-embedding backfill: workspace's embedding model isn't multimodal (supports_vision=false) — figures retrievable by media only, not via vector search"
                    );
                }
                Err(e) => {
                    warn!(
                        pdf_id = %data.pdf_id,
                        error = %e,
                        figure_count = extracted_figures.len(),
                        "Figure-embedding backfill failed (non-fatal); figures retrievable by media only"
                    );
                }
            }

            // 6c. Vision entity extraction. Workspace-scoped hybrid query
            //     retrieves chunks via entity.source_chunk_ids — figures
            //     are unreachable unless an entity points at them. Run each
            //     figure (caption + PNG) through the same entity-extraction
            //     prompt the text body uses, but against a vision-capable
            //     LLM. Resulting entities get source_chunk_ids containing
            //     the figure's chunk id; when merged into AGE via upsert,
            //     they enrich existing entities (e.g. a "GPT-2" entity
            //     extracted from text now also has the figure chunk-id
            //     in its source_chunk_ids array, so retrieval surfaces the
            //     figure when querying GPT-2).
            //     Best-effort: failures are logged and swallowed.
            //     Skipped in chunks-only mode (heavy vision-LLM extraction).
            if !data.skip_extraction {
            match self
                .backfill_figure_entities(
                    &early_doc_id,
                    &data.tenant_id.to_string(),
                    &data.workspace_id.to_string(),
                    &extracted_figures,
                )
                .await
            {
                Ok(Some((entities, relationships))) => {
                    info!(
                        pdf_id = %data.pdf_id,
                        entities,
                        relationships,
                        figure_count = extracted_figures.len(),
                        "Figure-entity backfill: linked figure chunks to graph entities"
                    );
                }
                Ok(None) => {
                    info!(
                        pdf_id = %data.pdf_id,
                        figure_count = extracted_figures.len(),
                        "Figure-entity backfill: workspace's extraction LLM isn't vision-capable (supports_vision=false) — figures retrievable but not graph-linked"
                    );
                }
                Err(e) => {
                    warn!(
                        pdf_id = %data.pdf_id,
                        error = %e,
                        figure_count = extracted_figures.len(),
                        "Figure-entity backfill failed (non-fatal); figures retrievable but not graph-linked"
                    );
                }
            }
            } // end: skip figure-entity backfill in chunks-only mode
        }

        // 6d. Table backfill + classification (VLM-OCR path only). Writes
        //     the HTML / parsed rows for each extracted table into the
        //     `chunks` table (mirrors the figure flow), then asks the
        //     workspace's LLM to label each as performance / quality /
        //     complexity / other. Runs *after* `process_text_insert` so the
        //     placeholder Table-kind chunks the chunker emitted are already
        //     in place; we just enrich the rows with HTML+rows and then
        //     classify them. All steps are best-effort.
        #[cfg(feature = "postgres")]
        if !extracted_tables.is_empty() {
            if let Err(e) = backfill_table_content(
                &early_doc_id,
                data.tenant_id,
                data.workspace_id,
                &extracted_tables,
            )
            .await
            {
                warn!(
                    pdf_id = %data.pdf_id,
                    error = %e,
                    table_count = extracted_tables.len(),
                    "Table backfill failed; tables captured but HTML/rows not persisted — document will be marked partial_failure"
                );
                enrichment_persistence_failed = true;
            } else {
                info!(
                    pdf_id = %data.pdf_id,
                    table_count = extracted_tables.len(),
                    "Table backfill: wrote HTML + parsed rows to chunks table"
                );
            }

            // (b) Embed each table (caption + cells) as a vector chunk so its
            //     numbers are reachable via dense / naive / hybrid retrieval.
            //     Runs even in chunks-only mode (mirrors figure embedding) —
            //     it's a plain text embedding, no LLM.
            match self
                .backfill_table_embeddings(
                    &early_doc_id,
                    &data.tenant_id.to_string(),
                    &data.workspace_id.to_string(),
                    &extracted_tables,
                )
                .await
            {
                Ok(stored) => info!(
                    pdf_id = %data.pdf_id,
                    embedded = stored,
                    total = extracted_tables.len(),
                    "Table-embedding backfill: wrote vectors to workspace vector store"
                ),
                Err(e) => warn!(
                    pdf_id = %data.pdf_id,
                    error = %e,
                    table_count = extracted_tables.len(),
                    "Table-embedding backfill failed (non-fatal); tables not dense-retrievable"
                ),
            }

            // (c) Graph entities for tables so cells are reachable via
            //     local/global retrieval. Heavy-ish (per-table embed) — gated
            //     out of chunks-only mode like the figure-entity backfill.
            if !data.skip_extraction {
                match self
                    .backfill_table_entities(
                        &early_doc_id,
                        &data.tenant_id.to_string(),
                        &data.workspace_id.to_string(),
                        &extracted_tables,
                    )
                    .await
                {
                    Ok(n) => info!(
                        pdf_id = %data.pdf_id,
                        entities = n,
                        "Table-entity backfill: upserted table nodes + vectors"
                    ),
                    Err(e) => warn!(
                        pdf_id = %data.pdf_id,
                        error = %e,
                        "Table-entity backfill failed (non-fatal); tables not graph-linked"
                    ),
                }
            }

            // Table classification is a heavy per-row LLM call — skipped in
            // chunks-only mode. The HTML/rows backfill above still runs so
            // tables render; classification can be triggered later.
            if !data.skip_extraction {
            if let Ok(doc_uuid) = uuid::Uuid::parse_str(&early_doc_id) {
                // Surface the classifier sub-stage so the documents list
                // and task tracker stop showing a misleading "completed"
                // while the LLM is still labelling tables one by one.
                self.update_document_status(&early_doc_id, "classifying_tables", None)
                    .await
                    .ok();
                task.update_progress("classifying_tables".to_string(), 6, 90);
                match self
                    .run_table_classification_inline(
                        &data.workspace_id.to_string(),
                        data.workspace_id,
                        doc_uuid,
                        cancel_token_for_classification,
                    )
                    .await
                {
                    Ok((classified, total)) => info!(
                        pdf_id = %data.pdf_id,
                        classified,
                        total,
                        "Table classification (inline): completed"
                    ),
                    Err(e) => warn!(
                        pdf_id = %data.pdf_id,
                        error = %e,
                        "Table classification (inline) failed (non-fatal); rows remain unlabelled"
                    ),
                }
            }
            } // end: skip table classification in chunks-only mode
        }

        // == Progress: extraction complete, linking PDF ==
        task.update_progress("linking".to_string(), 5, 95);

        // 7. Link PDF to created document (use early_doc_id)
        if let Ok(document_uuid) = uuid::Uuid::parse_str(&early_doc_id) {
            // FIX-ISSUE-74: Ensure a row in the `documents` relational table exists
            // BEFORE setting pdf_documents.document_id (which has a FK constraint).
            // WHY: Without this, the UPDATE violates the foreign key constraint
            // "pdf_documents_document_id_fkey" because no matching documents(id) row exists.
            let workspace_uuid = data.workspace_id;
            let tenant_uuid = Some(data.tenant_id);
            // WHY: Truncate content to 64KB for the relational record to avoid bloat.
            // Full content lives in KV storage. Use floor_char_boundary to avoid
            // splitting a multi-byte UTF-8 codepoint, which would panic.
            let truncate_at = if markdown.len() > 65_536 {
                // Find the largest char boundary <= 65_536
                markdown
                    .char_indices()
                    .map(|(i, _)| i)
                    .take_while(|&i| i <= 65_536)
                    .last()
                    .unwrap_or(0)
            } else {
                markdown.len()
            };
            if let Err(e) = pdf_storage
                .ensure_document_record(
                    &document_uuid,
                    &workspace_uuid,
                    tenant_uuid.as_ref(),
                    &pdf.filename,
                    &markdown[..truncate_at],
                    // WHY: The relational `documents` table has a CHECK constraint
                    // that only allows 'pending', 'processing', 'indexed', 'failed'.
                    // KV storage uses 'completed' but the relational table uses 'indexed'.
                    "indexed",
                )
                .await
            {
                error!(
                    "Failed to ensure document record: {} - continuing anyway",
                    e
                );
            }

            if let Err(e) = pdf_storage
                .link_pdf_to_document(&data.pdf_id, &document_uuid)
                .await
            {
                error!("Failed to link PDF to document: {} - continuing anyway", e);
                // Non-fatal - PDF still processed successfully
            }
        }

        // NOTE: algorithm + reference-repo extraction now run BEFORE
        // `process_text_insert` (see steps 6+7 above). Keeping this
        // comment as a signpost — older log grep patterns and design
        // docs still reference the previous ordering.

        // Finalize the document status now that every late-stage step has
        // run. We delegated this to `process_text_insert` with
        // `finalize_status: false`, then advanced through the
        // `media_enrichment` and `classifying_tables` substages. Honour the
        // hint it returned so a partial_failure outcome doesn't get paved
        // over with `completed`. If the figure/table backfill reported a
        // non-zero per-row failure (fix B), demote from `completed` to
        // `partial_failure` here — text extraction may have succeeded but
        // some media artefacts didn't reach the chunks table, so the doc
        // shouldn't claim full success.
        let final_status = if enrichment_persistence_failed
            && text_insert_final_status == "completed"
        {
            "partial_failure"
        } else {
            text_insert_final_status.as_str()
        };
        self.update_document_status(&early_doc_id, final_status, None)
            .await
            .ok();
        // Chunks-only mode: mark the document as awaiting extraction so it
        // appears in the "needs extraction" set and the UI shows the trigger.
        if data.skip_extraction {
            self.set_extraction_skipped_flag(&early_doc_id, true)
                .await
                .ok();
        }
        task.update_progress("completed".to_string(), 7, 100);

        info!(
            pdf_id = %data.pdf_id,
            "PDF processing completed successfully"
        );

        // OODA-16: Clean up progress tracking (fire-and-forget)
        // WHY: Free memory for completed uploads. GET endpoint will return 404.
        let state = self.pipeline_state.clone();
        let track_id = task.track_id.clone();
        tokio::spawn(async move {
            state.remove_pdf_progress(&track_id).await;
        });

        Ok(result)
    }

    #[cfg(not(feature = "postgres"))]
    pub(super) async fn process_pdf_processing(
        &self,
        _task: &mut Task,
        data: edgequake_tasks::PdfProcessingData,
        _cancel_token: CancellationToken,
    ) -> TaskResult<serde_json::Value> {
        warn!(
            pdf_id = %data.pdf_id,
            "PDF processing not available (postgres feature disabled)"
        );
        Err(edgequake_tasks::TaskError::UnsupportedOperation(
            "PDF processing requires postgres feature".to_string(),
        ))
    }

    /// Resolve the workspace-specific vector storage for a reprocess
    /// cleanup pass. Lenient: returns `None` when the workspace row is
    /// gone or the registry can't provision storage (a zombie document
    /// must still be reprocessable even if the workspace metadata
    /// degraded). Falls back to the default storage when the registry
    /// exposes one — this matches what
    /// `get_workspace_vector_storage_for_delete` does in the delete
    /// path, so orphan rows go to a consistent place.
    #[cfg(feature = "postgres")]
    async fn resolve_workspace_vector_storage_for_reprocess(
        &self,
        workspace_id: &uuid::Uuid,
    ) -> Option<Arc<dyn edgequake_storage::traits::VectorStorage>> {
        use edgequake_storage::traits::WorkspaceVectorConfig;

        // The registry caches instances per workspace; if this document
        // ever got ingested, the cache hit path gives us the same
        // storage without needing a fresh workspace-service lookup.
        if let Some(cached) = self.vector_registry.get(workspace_id).await {
            return Some(cached);
        }

        // Cache miss — provision via the workspace service to pick up
        // the correct embedding dimension. If the service isn't wired
        // or the workspace row is missing, fall back to the registry's
        // default storage so we at least attempt a purge.
        if let Some(ws_svc) = self.workspace_service.as_ref() {
            if let Ok(Some(ws)) = ws_svc.get_workspace(*workspace_id).await {
                let cfg = WorkspaceVectorConfig::new(*workspace_id, ws.embedding_dimension);
                if let Ok(storage) = self.vector_registry.get_or_create(cfg).await {
                    return Some(storage);
                }
            }
        }

        Some(self.vector_registry.default_storage())
    }

    /// Post-OCR rename helper. Returns an updated `PdfDocument` with
    /// either the freshly-composed citation filename or the original one
    /// untouched. **Never fails** — the caller continues regardless so
    /// a broken rename can't wedge ingestion.
    ///
    /// Behavior is gated by the opt-in flag `rename_after_parse = true`
    /// stamped into the PDF metadata JSON by callers that want automatic
    /// renaming (today: the `/documents/pdf/from-url` endpoint). All
    /// other uploads short-circuit immediately.
    #[cfg(feature = "postgres")]
    async fn maybe_rename_from_front_matter(
        &self,
        pdf: &edgequake_storage::PdfDocument,
        markdown: &str,
        source_url: Option<&str>,
        document_id: &str,
        pdf_storage: &std::sync::Arc<dyn edgequake_storage::PdfDocumentStorage>,
    ) -> edgequake_storage::PdfDocument {
        use edgequake_agents::web_search::extract_front_matter;

        let Some(fm) = extract_front_matter(markdown) else {
            info!(
                pdf_id = %pdf.pdf_id,
                "rename skipped: no front-matter extractable"
            );
            return pdf.clone();
        };

        let Some(author) = fm
            .first_author
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        else {
            info!(
                pdf_id = %pdf.pdf_id,
                "rename skipped: front-matter has no first author"
            );
            return pdf.clone();
        };

        // Year: prefer arxiv-id in the source URL; otherwise leave off.
        // Non-arxiv uploads still get `"Author - Title.pdf"`.
        let year = source_url.and_then(derive_arxiv_year);

        let new_filename = format_citation_filename(author, year.as_deref(), &fm.title);
        if new_filename == pdf.filename {
            return pdf.clone();
        }

        // 1. Rewrite the pdf_documents row so any future consumer
        //    (reprocess, download, delete) sees the new name.
        if let Err(e) = pdf_storage
            .update_pdf_filename(&pdf.pdf_id, &new_filename)
            .await
        {
            warn!(
                pdf_id = %pdf.pdf_id,
                error = %e,
                "rename failed at pdf_documents — keeping original filename"
            );
            return pdf.clone();
        }

        // 2. Sync KV `{doc_id}-metadata` so the documents list + detail
        //    views pick up the new title. The list renderer reads
        //    `title` from this blob; the detail view reads `file_name`.
        //    Failure here is non-fatal — log it and keep the storage-
        //    level rename.
        if let Err(e) = self
            .sync_document_metadata_title(document_id, &new_filename)
            .await
        {
            warn!(
                pdf_id = %pdf.pdf_id,
                document_id,
                error = %e,
                "rename: pdf_documents updated but KV title sync failed"
            );
        }

        info!(
            pdf_id = %pdf.pdf_id,
            old = %pdf.filename,
            new = %new_filename,
            "post-OCR rename applied"
        );

        // Clone-with-override so the caller's downstream metadata blob
        // picks up the new filename.
        let mut updated = pdf.clone();
        updated.filename = new_filename;
        updated
    }

    /// Read the `{doc_id}-metadata` KV blob, overwrite `title` +
    /// `file_name` with the new filename, and write it back. We don't
    /// touch any other fields so concurrent status writes from the
    /// progress tracker aren't clobbered (last-writer-wins on this
    /// single blob is a pre-existing limitation of the KV design; the
    /// progress tracker runs after this rename call so the race is
    /// narrow in practice).
    #[cfg(feature = "postgres")]
    async fn sync_document_metadata_title(
        &self,
        document_id: &str,
        new_filename: &str,
    ) -> Result<(), String> {
        let key = format!("{document_id}-metadata");
        let Some(mut value) = self
            .kv_storage
            .get_by_id(&key)
            .await
            .map_err(|e| format!("kv get: {e}"))?
        else {
            // Metadata blob isn't there yet — probably a race with the
            // early-metadata write. Not fatal for this rename pass;
            // downstream progress writes will overwrite with their own
            // title field sourced from pdf.filename, which we've just
            // shadowed above.
            return Ok(());
        };
        if let Some(obj) = value.as_object_mut() {
            obj.insert(
                "title".to_string(),
                serde_json::Value::String(new_filename.to_string()),
            );
            obj.insert(
                "file_name".to_string(),
                serde_json::Value::String(new_filename.to_string()),
            );
        }
        self.kv_storage
            .upsert(&[(key, value)])
            .await
            .map_err(|e| format!("kv upsert: {e}"))
    }

    /// Compute fused multimodal embeddings (caption + PNG bytes) for each
    /// extracted figure and upsert them to the workspace vector store under
    /// id `{document_id}-figure-{figure_id}` — same id the chunker uses for
    /// its placeholder chunk, so the embedding lands on the same record.
    ///
    /// Driven entirely by **workspace settings**: looks up the workspace's
    /// configured embedding provider + model, checks the model card's
    /// `supports_vision` flag, and (when true) routes figure inputs to that
    /// provider's `/v1/embeddings` endpoint via `MultimodalEmbeddingClient`.
    /// No env-var overrides — `/workspace` settings are the single source of
    /// truth for which model to use.
    ///
    /// Returns:
    /// - `Ok(Some(n))` — embeddings stored for `n` of `figures.len()` figures
    /// - `Ok(None)`    — workspace's embedding model isn't multimodal
    ///                   (`supports_vision = false`); figures stay
    ///                   retrievable via the media-fetch endpoint only
    /// - `Err(_)`      — unexpected failure (workspace not found, provider
    ///                   missing base_url, vector store unreachable)
    ///
    /// Best-effort per figure: a single embed/upsert failure is logged and
    /// the rest of the batch proceeds.
    #[cfg(feature = "postgres")]
    async fn backfill_figure_embeddings(
        &self,
        document_id: &str,
        tenant_id: &str,
        workspace_id: &str,
        figures: &[edgequake_pdf::ExtractedFigure],
    ) -> Result<Option<usize>, String> {
        use edgequake_pipeline::embedding::{
            EmbeddingInput, EmbeddingRole, MultimodalEmbeddingClient,
        };
        use serde_json::json;

        // 1. Resolve workspace → find embedding provider + model.
        let workspace_uuid = uuid::Uuid::parse_str(workspace_id)
            .map_err(|e| format!("invalid workspace_id {workspace_id:?}: {e}"))?;
        let ws = self
            .workspace_service
            .as_ref()
            .ok_or_else(|| "no workspace_service".to_string())?
            .get_workspace(workspace_uuid)
            .await
            .map_err(|e| format!("get_workspace: {e}"))?
            .ok_or_else(|| format!("workspace {workspace_uuid} not found"))?;

        // 2. Look up the provider/model in the loaded ModelsConfig to (a)
        //    check supports_vision and (b) get the base_url so we can talk
        //    to the same /v1/embeddings the trait-based text path uses.
        let cfg = self
            .models_config
            .as_ref()
            .ok_or_else(|| "no models_config".to_string())?;
        let provider_cfg = cfg
            .get_provider(&ws.embedding_provider)
            .ok_or_else(|| format!("provider {} not in models.toml", ws.embedding_provider))?;
        let model_card = cfg
            .get_model(&ws.embedding_provider, &ws.embedding_model)
            .ok_or_else(|| {
                format!(
                    "model {} not in provider {}",
                    ws.embedding_model, ws.embedding_provider
                )
            })?;

        if !model_card.capabilities.supports_vision {
            // Workspace's embedding model is text-only. Skip — bytes remain
            // accessible via the media-fetch endpoint; only vector search
            // over figures is unavailable until a multimodal model is
            // selected in /workspace.
            return Ok(None);
        }

        // Resolve the OpenAI-compat /v1 base URL for this provider. Prefer
        // the explicit TOML `base_url`; fall back to the provider's per-type
        // env-var convention (mirrors what the SDK does internally for the
        // text path so users who never set `base_url` still work). The
        // lmstudio path commonly only has `LMSTUDIO_HOST` set — it doesn't
        // include `/v1` so we append it here.
        let base_url = provider_cfg
            .base_url
            .clone()
            .or_else(|| {
                provider_cfg
                    .base_url_env
                    .as_ref()
                    .and_then(|var| std::env::var(var).ok())
                    .map(|u| ensure_v1_suffix(&u))
            })
            .or_else(|| match ws.embedding_provider.to_ascii_lowercase().as_str() {
                "lmstudio" | "lm-studio" | "lm_studio" => std::env::var("LMSTUDIO_HOST")
                    .ok()
                    .map(|u| ensure_v1_suffix(&u)),
                "ollama" => std::env::var("OLLAMA_HOST")
                    .ok()
                    .map(|u| ensure_v1_suffix(&u)),
                "openai" => Some("https://api.openai.com/v1".to_string()),
                _ => None,
            })
            .ok_or_else(|| {
                format!(
                    "provider {} has no base_url and no env-var fallback resolved; \
                     set `base_url` in models.toml or the provider's *_HOST env var",
                    ws.embedding_provider
                )
            })?;

        let client = MultimodalEmbeddingClient::new(&base_url, &ws.embedding_model);
        let store = self
            .get_workspace_vector_storage_strict(workspace_id)
            .await?;

        let mut stored = 0usize;
        for fig in figures {
            let embedding = match client
                .embed(
                    EmbeddingInput::Figure {
                        caption: &fig.caption,
                        bytes: &fig.image_bytes,
                        mime: &fig.mime,
                    },
                    EmbeddingRole::Document,
                )
                .await
            {
                Ok(v) => v,
                Err(e) => {
                    warn!(
                        figure_id = %fig.id,
                        error = %e,
                        "figure-embedding: multimodal embed call failed"
                    );
                    continue;
                }
            };

            let chunk_id = format!("{document_id}-figure-{}", fig.id);
            // `content` mirrors what text-chunk rows store: it's what the BM25
            // reranker scores and what build_chunk_from_result reads into
            // RetrievedChunk.content. Without it, figure chunks score ~0 in
            // BM25 and get dropped below min_rerank_score. Caption is the only
            // searchable text we have for a figure, so it doubles as content.
            let metadata = json!({
                "type": "chunk",
                "kind": "figure",
                "content": fig.caption,
                "document_id": document_id,
                "figure_id": fig.id,
                "page": fig.page,
                "order_index": fig.order_index,
                "caption": fig.caption,
                "tenant_id": tenant_id,
                "workspace_id": workspace_id,
                "media_mime": fig.mime,
                "embedding_provider": ws.embedding_provider,
                "embedding_model": ws.embedding_model,
            });

            if let Err(e) = store
                .upsert(&[(chunk_id.clone(), embedding, metadata)])
                .await
            {
                warn!(
                    figure_id = %fig.id,
                    chunk_id = %chunk_id,
                    error = %e,
                    "figure-embedding: vector upsert failed"
                );
                continue;
            }
            stored += 1;
        }
        Ok(Some(stored))
    }

    /// Embed each extracted table as a vector-store chunk so its cell values
    /// (communication cost, latency, accuracy) are reachable via dense /
    /// naive / hybrid retrieval. Mirrors [`backfill_figure_embeddings`] but
    /// uses a **text** embedding (tables are text, not images), so there is
    /// no `supports_vision` gate — the workspace's normal embedding model is
    /// used, matching the text-chunk vector dimension.
    ///
    /// The table text embedded is the GFM rendering (caption + cells) from
    /// [`edgequake_pdf::render_table_markdown`], stored under id
    /// `{document_id}-table-{table_id}` with `kind=table`. A table that
    /// renders to nothing is skipped. Best-effort per table.
    #[cfg(feature = "postgres")]
    async fn backfill_table_embeddings(
        &self,
        document_id: &str,
        tenant_id: &str,
        workspace_id: &str,
        tables: &[edgequake_pdf::ExtractedTable],
    ) -> Result<usize, String> {
        use edgequake_pipeline::embedding::{
            EmbeddingInput, EmbeddingRole, MultimodalEmbeddingClient,
        };
        use serde_json::json;

        let workspace_uuid = uuid::Uuid::parse_str(workspace_id)
            .map_err(|e| format!("invalid workspace_id {workspace_id:?}: {e}"))?;
        let ws = self
            .workspace_service
            .as_ref()
            .ok_or_else(|| "no workspace_service".to_string())?
            .get_workspace(workspace_uuid)
            .await
            .map_err(|e| format!("get_workspace: {e}"))?
            .ok_or_else(|| format!("workspace {workspace_uuid} not found"))?;

        let base_url = self.resolve_embedding_base_url(&ws)?;
        let client = MultimodalEmbeddingClient::new(&base_url, &ws.embedding_model);
        let store = self
            .get_workspace_vector_storage_strict(workspace_id)
            .await?;

        let mut stored = 0usize;
        for table in tables {
            // Decouple the three texts so each stage gets the right signal:
            //   embed_text  — caption + column headers (the vector; the raw grid
            //                 dilutes cosine similarity for caption-style queries)
            //   rerank_text — caption + both axes (headers + first-column labels);
            //                 the workspace reranker (qwen3) scores this instead
            //                 of `content`, ranking the table by what it's about
            //                 rather than its dense grid (see reranking.rs)
            //   content     — the full GFM, returned into context + displayed
            let Some(embed_text) = edgequake_pdf::render_table_embed_text(table) else {
                continue;
            };
            let content =
                edgequake_pdf::render_table_markdown(table).unwrap_or_else(|| embed_text.clone());
            let rerank_text = edgequake_pdf::render_table_rerank_text(table);
            let embedding = match client
                .embed(EmbeddingInput::Text(&embed_text), EmbeddingRole::Document)
                .await
            {
                Ok(v) => v,
                Err(e) => {
                    warn!(table_id = %table.id, error = %e, "table-embedding: text embed call failed");
                    continue;
                }
            };

            let chunk_id = format!("{document_id}-table-{}", table.id);
            let metadata = json!({
                "type": "chunk",
                "kind": "table",
                "content": content,
                "rerank_text": rerank_text,
                "document_id": document_id,
                "table_id": table.id,
                "page": table.page,
                "order_index": table.order_index,
                "caption": table.caption,
                "tenant_id": tenant_id,
                "workspace_id": workspace_id,
                "embedding_provider": ws.embedding_provider,
                "embedding_model": ws.embedding_model,
            });

            if let Err(e) = store
                .upsert(&[(chunk_id.clone(), embedding, metadata)])
                .await
            {
                warn!(table_id = %table.id, chunk_id = %chunk_id, error = %e, "table-embedding: vector upsert failed");
                continue;
            }
            stored += 1;
        }
        Ok(stored)
    }

    /// Upsert one graph entity per extracted table so table content is
    /// reachable via local/global (graph) retrieval, not just dense search.
    /// The entity's `description` is the GFM rendering (caption + cells), and
    /// its `source_chunk_ids` points at the `{document_id}-table-{table_id}`
    /// vector chunk (embedded by [`backfill_table_embeddings`]). The entity is
    /// also embedded (`entity:{name}` vector) so local-mode ANN can match it
    /// by its cell content. Best-effort; never fatal.
    #[cfg(feature = "postgres")]
    async fn backfill_table_entities(
        &self,
        document_id: &str,
        tenant_id: &str,
        workspace_id: &str,
        tables: &[edgequake_pdf::ExtractedTable],
    ) -> Result<usize, String> {
        use edgequake_pipeline::embedding::{
            EmbeddingInput, EmbeddingRole, MultimodalEmbeddingClient,
        };
        use serde_json::json;

        let workspace_uuid = uuid::Uuid::parse_str(workspace_id)
            .map_err(|e| format!("invalid workspace_id {workspace_id:?}: {e}"))?;
        let ws = self
            .workspace_service
            .as_ref()
            .ok_or_else(|| "no workspace_service".to_string())?
            .get_workspace(workspace_uuid)
            .await
            .map_err(|e| format!("get_workspace: {e}"))?
            .ok_or_else(|| format!("workspace {workspace_uuid} not found"))?;

        let base_url = self.resolve_embedding_base_url(&ws)?;
        let client = MultimodalEmbeddingClient::new(&base_url, &ws.embedding_model);
        let store = self
            .get_workspace_vector_storage_strict(workspace_id)
            .await?;

        let mut nodes_batch: Vec<(String, std::collections::HashMap<String, serde_json::Value>)> =
            Vec::new();
        // (entity_name, caption-led embed_text, GFM description, table_chunk_id)
        // for the entity-vector upsert below.
        let mut to_embed: Vec<(String, String, String, String)> = Vec::new();

        for table in tables {
            let Some(rendered) = edgequake_pdf::render_table_markdown(table) else {
                continue;
            };
            // Embed the entity vector on the caption-led text (consistent with the
            // dedicated table chunk — the raw grid dilutes the vector); keep the
            // full GFM as the entity `description` returned into context.
            let embed_text =
                edgequake_pdf::render_table_embed_text(table).unwrap_or_else(|| rendered.clone());
            // Name the node by its caption (paper-specific, descriptive); fall
            // back to a page/order id when the table has no caption. Cap the
            // length so the graph node name stays sane.
            let caption = table.caption.trim();
            let name = if caption.is_empty() {
                format!("Table {}.{}", table.page, table.order_index)
            } else {
                caption.chars().take(120).collect::<String>()
            };
            let table_chunk_id = format!("{document_id}-table-{}", table.id);

            let mut props = std::collections::HashMap::new();
            props.insert("entity_type".to_string(), json!("TABLE"));
            props.insert("description".to_string(), json!(rendered));
            props.insert("importance".to_string(), json!(0.5));
            props.insert("source_ids".to_string(), json!(vec![document_id.to_string()]));
            props.insert(
                "source_chunk_ids".to_string(),
                json!(vec![table_chunk_id.clone()]),
            );
            props.insert("tenant_id".to_string(), json!(tenant_id));
            props.insert("workspace_id".to_string(), json!(workspace_id));
            nodes_batch.push((name.clone(), props));
            to_embed.push((name, embed_text, rendered, table_chunk_id));
        }

        if nodes_batch.is_empty() {
            return Ok(0);
        }

        self.graph_storage
            .upsert_nodes_batch(&nodes_batch)
            .await
            .map_err(|e| format!("upsert_nodes_batch (tables): {e}"))?;

        // Embed each table entity so local-mode ANN over entity vectors can
        // match it by cell content. Best-effort per entity.
        let mut embedded = 0usize;
        for (name, embed_text, description, _chunk_id) in &to_embed {
            let embedding = match client
                .embed(EmbeddingInput::Text(embed_text), EmbeddingRole::Document)
                .await
            {
                Ok(v) => v,
                Err(e) => {
                    warn!(entity = %name, error = %e, "table-entity: embed call failed");
                    continue;
                }
            };
            let vec_id = format!("entity:{name}");
            let metadata = json!({
                "type": "entity",
                "name": name,
                "entity_type": "TABLE",
                "description": description,
                "document_id": document_id,
                "tenant_id": tenant_id,
                "workspace_id": workspace_id,
            });
            if let Err(e) = store.upsert(&[(vec_id, embedding, metadata)]).await {
                warn!(entity = %name, error = %e, "table-entity: vector upsert failed");
                continue;
            }
            embedded += 1;
        }
        Ok(embedded)
    }

    /// Resolve the OpenAI-compatible `/v1` base URL for the workspace's
    /// embedding provider — explicit TOML `base_url`, then the provider's
    /// `*_HOST` env var, then well-known defaults. Shared by the table
    /// embedding/entity backfills (the figure path inlines its own copy).
    #[cfg(feature = "postgres")]
    fn resolve_embedding_base_url(
        &self,
        ws: &edgequake_core::types::Workspace,
    ) -> Result<String, String> {
        let cfg = self
            .models_config
            .as_ref()
            .ok_or_else(|| "no models_config".to_string())?;
        let provider_cfg = cfg
            .get_provider(&ws.embedding_provider)
            .ok_or_else(|| format!("provider {} not in models.toml", ws.embedding_provider))?;
        provider_cfg
            .base_url
            .clone()
            .or_else(|| {
                provider_cfg
                    .base_url_env
                    .as_ref()
                    .and_then(|var| std::env::var(var).ok())
                    .map(|u| ensure_v1_suffix(&u))
            })
            .or_else(|| match ws.embedding_provider.to_ascii_lowercase().as_str() {
                "lmstudio" | "lm-studio" | "lm_studio" => {
                    std::env::var("LMSTUDIO_HOST").ok().map(|u| ensure_v1_suffix(&u))
                }
                "ollama" => std::env::var("OLLAMA_HOST").ok().map(|u| ensure_v1_suffix(&u)),
                "openai" => Some("https://api.openai.com/v1".to_string()),
                _ => None,
            })
            .ok_or_else(|| {
                format!(
                    "provider {} has no base_url and no env-var fallback resolved",
                    ws.embedding_provider
                )
            })
    }

    /// Run each figure (caption + PNG bytes) through the workspace's
    /// configured extraction LLM with the same entity-extraction prompt the
    /// text body uses, then merge the resulting entities/relationships into
    /// AGE.
    ///
    /// **Purpose:** the workspace-scoped hybrid retrieval
    /// (`edgequake-query::sota_engine::vector_queries::query_hybrid_with_vector_storage`)
    /// pulls chunks via `entity.source_chunk_ids`. Figures live in pgvector but
    /// no entity references them, so they're unreachable through hybrid even
    /// though they're indexed. This pass writes graph entries whose
    /// `source_chunk_ids` contain `{document_id}-figure-{figure_id}`, plugging
    /// the figures into the graph.
    ///
    /// **LLM resolution:** uses the workspace's `llm_provider` + `llm_model`
    /// (same model the text-extraction path uses, kept in lockstep so chunks
    /// and figures land in the same naming/typing convention). Gated on
    /// `model_card.capabilities.supports_vision` — text-only models can't
    /// see the figure, so the pass is skipped cleanly with `Ok(None)`.
    ///
    /// **Merge behaviour:** `graph_storage.upsert_nodes_batch` keys on entity
    /// name. An entity like "GPT-2" already extracted from text gets its
    /// `source_chunk_ids` enriched with the figure id — Postgres-AGE merges
    /// the array under the hood (mirroring the same OODA-07 source_ids merge
    /// the text path does at `text_insert.rs:756-806`). A brand-new entity
    /// from a figure is added as a fresh node; it won't be ANN-discoverable
    /// (we don't backfill its entity embedding here) but it'll surface via
    /// relationship-graph traversal from connected entities.
    ///
    /// Best-effort per figure: a single extract failure logs+continues.
    #[cfg(feature = "postgres")]
    async fn backfill_figure_entities(
        &self,
        document_id: &str,
        tenant_id: &str,
        workspace_id: &str,
        figures: &[edgequake_pdf::ExtractedFigure],
    ) -> Result<Option<(usize, usize)>, String> {
        use crate::safety_limits::create_safe_llm_provider;
        use edgequake_pipeline::extractor::VisionExtractionClient;

        // 1. Resolve workspace → llm_provider + llm_model. Use the same
        //    factory the text-extraction path uses — see
        //    `workspace_resolver.rs:73` — so URL/api-key/timeout resolution
        //    flows through ProviderFactory + models.toml exactly the same way
        //    for both paths.
        let workspace_uuid = uuid::Uuid::parse_str(workspace_id)
            .map_err(|e| format!("invalid workspace_id {workspace_id:?}: {e}"))?;
        let ws = self
            .workspace_service
            .as_ref()
            .ok_or_else(|| "no workspace_service".to_string())?
            .get_workspace(workspace_uuid)
            .await
            .map_err(|e| format!("get_workspace: {e}"))?
            .ok_or_else(|| format!("workspace {workspace_uuid} not found"))?;

        // Gate on `supports_vision` from models.toml — text-only models
        // can't see the figure, so skip cleanly with `Ok(None)`.
        let cfg = self
            .models_config
            .as_ref()
            .ok_or_else(|| "no models_config".to_string())?;
        let model_card = cfg
            .get_model(&ws.llm_provider, &ws.llm_model)
            .ok_or_else(|| {
                format!(
                    "model {} not in provider {}",
                    ws.llm_model, ws.llm_provider
                )
            })?;
        if !model_card.capabilities.supports_vision {
            return Ok(None);
        }

        let llm_provider = create_safe_llm_provider(&ws.llm_provider, &ws.llm_model)
            .map_err(|e| format!("create_safe_llm_provider: {e}"))?;
        let client = VisionExtractionClient::new(llm_provider);

        let mut nodes_batch: Vec<(String, std::collections::HashMap<String, serde_json::Value>)> =
            Vec::new();
        let mut edges_batch: Vec<(
            String,
            String,
            std::collections::HashMap<String, serde_json::Value>,
        )> = Vec::new();
        // (entity_id, figure_chunk_id) pairs, accurate per-figure (entities
        // extracted from fig_3_0 don't get incorrectly linked to fig_4_0).
        let mut entity_chunk_pairs: Vec<(String, String)> = Vec::new();

        let mut entities_total = 0usize;
        let mut rels_total = 0usize;

        for fig in figures {
            let chunk_id = format!("{document_id}-figure-{}", fig.id);
            let mut result = match client
                .extract(&fig.caption, &fig.image_bytes, &fig.mime, &chunk_id)
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    warn!(
                        figure_id = %fig.id,
                        error = %e,
                        "figure-entity: vision-extract call failed"
                    );
                    continue;
                }
            };

            // Mirror `pipeline::helpers::link_extractions_to_chunks` — that
            // helper is `pub(super)`, so we replicate the two-line linkage
            // here. The parser already populates `result.source_chunk_id`;
            // we propagate it onto each entity's source_chunk_ids array and
            // each relationship's source_chunk_id slot.
            for entity in &mut result.entities {
                entity.add_source_chunk_id(&chunk_id);
            }
            for rel in &mut result.relationships {
                if rel.source_chunk_id.is_none() {
                    rel.source_chunk_id = Some(chunk_id.clone());
                }
            }

            entities_total += result.entities.len();
            rels_total += result.relationships.len();

            for entity in &result.entities {
                // Track this specific entity → figure chunk_id link for the
                // vector-store merge step below. Same entity extracted from
                // different figures yields multiple pairs — the SQL update
                // is idempotent so duplicates are harmless.
                entity_chunk_pairs.push((format!("entity:{}", entity.name), chunk_id.clone()));

                let mut props = std::collections::HashMap::new();
                props.insert(
                    "entity_type".to_string(),
                    serde_json::json!(entity.entity_type),
                );
                props.insert(
                    "description".to_string(),
                    serde_json::json!(entity.description),
                );
                props.insert("importance".to_string(), serde_json::json!(entity.importance));
                // Note: source_ids merge with existing entity is handled by
                // graph_storage.upsert_nodes_batch's MERGE semantics — we
                // contribute the current document_id; the storage layer
                // combines with whatever's already there.
                props.insert(
                    "source_ids".to_string(),
                    serde_json::json!(vec![document_id.to_string()]),
                );
                props.insert(
                    "source_chunk_ids".to_string(),
                    serde_json::json!(&entity.source_chunk_ids),
                );
                props.insert("tenant_id".to_string(), serde_json::json!(tenant_id));
                props.insert("workspace_id".to_string(), serde_json::json!(workspace_id));
                nodes_batch.push((entity.name.clone(), props));
            }

            for rel in &result.relationships {
                let mut props = std::collections::HashMap::new();
                props.insert(
                    "relation_type".to_string(),
                    serde_json::json!(rel.relation_type),
                );
                props.insert("description".to_string(), serde_json::json!(rel.description));
                props.insert("weight".to_string(), serde_json::json!(rel.weight));
                props.insert("keywords".to_string(), serde_json::json!(rel.keywords));
                props.insert(
                    "source_ids".to_string(),
                    serde_json::json!(vec![document_id.to_string()]),
                );
                if let Some(ref c) = rel.source_chunk_id {
                    props.insert("source_chunk_ids".to_string(), serde_json::json!(vec![c]));
                }
                props.insert("tenant_id".to_string(), serde_json::json!(tenant_id));
                props.insert("workspace_id".to_string(), serde_json::json!(workspace_id));
                edges_batch.push((rel.source.clone(), rel.target.clone(), props));
            }
        }

        if !nodes_batch.is_empty() {
            if let Err(e) = self.graph_storage.upsert_nodes_batch(&nodes_batch).await {
                return Err(format!("upsert_nodes_batch: {e}"));
            }
        }
        if !edges_batch.is_empty() {
            if let Err(e) = self.graph_storage.upsert_edges_batch(&edges_batch).await {
                return Err(format!("upsert_edges_batch: {e}"));
            }
        }

        // Merge figure chunk IDs into the workspace vector store's existing
        // entity rows. Hybrid retrieval (vector_queries.rs::query_local_with_
        // vector_storage) reads `metadata.source_chunk_ids` directly off the
        // entity-vector row — graph mutation alone isn't enough. We don't
        // re-embed: the entity's existing vector (from the text-extraction
        // pass) stays as-is; we only append the figure chunk_ids to the
        // metadata JSONB array.
        //
        // Figure-only entities (names that text never extracted) have no
        // row in the vector store, so the UPDATE silently does nothing for
        // them. They remain reachable via relationship-graph traversal from
        // any text-derived entity that's been newly connected to them.
        //
        // Best-effort: a missing DATABASE_URL or table-lookup failure logs
        // and continues; the graph edges are already written.
        if !entity_chunk_pairs.is_empty() {
            if let Err(e) = self
                .merge_figure_chunk_ids_into_entity_vectors(workspace_id, &entity_chunk_pairs)
                .await
            {
                warn!(
                    document_id,
                    error = %e,
                    "figure-entity: workspace-vector source_chunk_ids merge failed (non-fatal); graph still updated"
                );
            }
        }

        Ok(Some((entities_total, rels_total)))
    }

    /// Append figure chunk IDs to `metadata.source_chunk_ids` on existing
    /// entity-vector rows in the workspace vector store.
    ///
    /// Why this exists: `graph_storage.upsert_nodes_batch` updates AGE, but
    /// the workspace-scoped hybrid retrieval path
    /// (`vector_queries.rs::query_local_with_vector_storage`) reads
    /// `metadata.source_chunk_ids` directly off the vector row, not from the
    /// graph node. Without this merge, figures stay invisible to hybrid even
    /// though the graph and figure vector store both know about them.
    ///
    /// **JSONB merge semantics:** uses `jsonb_set` + array concat, gated by
    /// `NOT @>` so re-runs are idempotent (existing entries don't duplicate).
    ///
    /// Best-effort: each (entity_id, figure_chunk_id) pair runs in its own
    /// statement so one bad row doesn't poison the rest.
    #[cfg(feature = "postgres")]
    async fn merge_figure_chunk_ids_into_entity_vectors(
        &self,
        workspace_id: &str,
        pairs: &[(String, String)],
    ) -> Result<(), String> {
        let workspace_uuid = uuid::Uuid::parse_str(workspace_id)
            .map_err(|e| format!("invalid workspace_id {workspace_id:?}: {e}"))?;
        let short_id = &workspace_uuid.to_string()[..8];
        let database_url = std::env::var("DATABASE_URL")
            .map_err(|_| "DATABASE_URL not set for figure-entity merge".to_string())?;
        let pool = sqlx::PgPool::connect(&database_url)
            .await
            .map_err(|e| format!("PgPool::connect: {e}"))?;

        // Resolve the workspace's vector table by pattern. The exact name is
        // `eq_{namespace}_ws_{short_id}_vectors` where namespace is set in
        // PostgresConfig (`with_namespace(...)` in state/postgres.rs). We
        // dynamically discover it instead of duplicating the namespace
        // convention — keeps this code resilient to namespace renames.
        let row: Option<(String,)> = sqlx::query_as(
            r#"
            SELECT tablename FROM pg_tables
            WHERE schemaname = 'public'
              AND tablename LIKE $1
            LIMIT 1
            "#,
        )
        .bind(format!("eq_%_ws_{short_id}_vectors"))
        .fetch_optional(&pool)
        .await
        .map_err(|e| format!("table lookup: {e}"))?;
        let table = row
            .ok_or_else(|| format!("no workspace vector table found for {short_id}"))?
            .0;

        let mut updated = 0usize;
        for (entity_id, chunk_id) in pairs {
            // jsonb_set on source_chunk_ids: append chunk_id if absent. The
            // `NOT (... @> ...)` clause keeps re-runs idempotent and prevents
            // duplicate array entries.
            let sql = format!(
                r#"
                UPDATE public.{table} SET metadata = jsonb_set(
                    metadata,
                    '{{source_chunk_ids}}',
                    COALESCE(metadata->'source_chunk_ids', '[]'::jsonb) || to_jsonb($2::text)
                )
                WHERE id = $1
                  AND metadata->>'type' = 'entity'
                  AND NOT COALESCE(metadata->'source_chunk_ids', '[]'::jsonb) @> to_jsonb($2::text)
                "#
            );
            match sqlx::query(&sql)
                .bind(entity_id)
                .bind(chunk_id)
                .execute(&pool)
                .await
            {
                Ok(r) => updated += r.rows_affected() as usize,
                Err(e) => {
                    warn!(
                        entity_id,
                        chunk_id,
                        error = %e,
                        "figure-entity: source_chunk_ids UPDATE failed (continuing)"
                    );
                }
            }
        }

        info!(
            workspace_id,
            updated_rows = updated,
            pairs_attempted = pairs.len(),
            "Figure-entity backfill: merged figure chunk IDs into entity-vector source_chunk_ids"
        );
        Ok(())
    }
}

/// Ensure a URL ends with `/v1` (the OpenAI-compat path prefix). LMSTUDIO_HOST
/// and OLLAMA_HOST conventions omit the suffix; the multimodal client expects
/// the full `/v1` base because it appends `/embeddings` to it.
fn ensure_v1_suffix(url: &str) -> String {
    let trimmed = url.trim_end_matches('/');
    if trimmed.ends_with("/v1") {
        trimmed.to_string()
    } else {
        format!("{trimmed}/v1")
    }
}

/// Derive a 4-digit year from an arxiv id embedded in `source_url` (e.g.
/// `.../pdf/2506.09452` → `"2025"`). Returns `None` when the URL isn't an
/// arxiv link or the id is malformed.
#[cfg(feature = "postgres")]
fn derive_arxiv_year(source_url: &str) -> Option<String> {
    if source_url.is_empty() {
        return None;
    }
    // Arxiv ids look like `YYMM.NNNNN` where YY is the 2-digit year.
    // Scan the whole URL because the id can live in the path (arxiv.org)
    // or a query fragment (semanticscholar, etc.).
    let mut chars = source_url.chars().peekable();
    while chars.peek().is_some() {
        let mut digits = String::new();
        while let Some(&c) = chars.peek() {
            if c.is_ascii_digit() {
                digits.push(c);
                chars.next();
            } else {
                break;
            }
        }
        if digits.len() == 4 {
            // Expect `.NNNNN` immediately after.
            if chars.peek() == Some(&'.') {
                chars.next();
                let post: String = chars.by_ref().take(5).collect();
                if post.chars().all(|c| c.is_ascii_digit()) {
                    let yy: u32 = digits[..2].parse().ok()?;
                    // arxiv introduced the new-format ids in 2007; a YY
                    // of `07..=99` → 2007..2099, `00..=06` → 2100+ which
                    // would be future — but stick to the simple 20xx
                    // mapping because arxiv won't hit 2100 this decade.
                    return Some(format!("20{yy:02}"));
                }
            }
        }
        // Advance past a non-digit if no match
        if chars.peek().is_some_and(|c| !c.is_ascii_digit()) {
            chars.next();
        }
    }
    None
}

/// Compose `"{author} et al. - {year} - {title}.pdf"`, dropping the
/// `{year} - ` segment when year is absent. Title is truncated to 120
/// chars and path-unsafe characters are replaced with `_` so the result
/// is safe for filesystems and display.
#[cfg(feature = "postgres")]
fn format_citation_filename(author: &str, year: Option<&str>, title: &str) -> String {
    const TITLE_CAP: usize = 120;

    let clean_author = sanitize_filename_segment(author);
    let clean_title = sanitize_filename_segment(title);
    let trimmed_title: String = clean_title.chars().take(TITLE_CAP).collect();

    match year {
        Some(y) if !y.is_empty() => format!("{clean_author} et al. - {y} - {trimmed_title}.pdf"),
        _ => format!("{clean_author} et al. - {trimmed_title}.pdf"),
    }
}

/// Replace filesystem-hostile characters with `_` and collapse runs of
/// whitespace. Preserves unicode (we want author names like `Özkan` to
/// survive) and `:` (academic subtitles like "Euston: Efficient..." —
/// legal on Linux/macOS; Windows users may have to rename on download,
/// which is the better trade-off than mangled titles everywhere else).
/// Also strips leading/trailing whitespace and `.`.
#[cfg(feature = "postgres")]
fn sanitize_filename_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '/' | '\\' | '<' | '>' | '"' | '|' | '?' | '*' => out.push('_'),
            c if c.is_control() => out.push('_'),
            c => out.push(c),
        }
    }
    // Collapse whitespace runs to single spaces.
    let collapsed: String = out.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed.trim_matches('.').to_string()
}

/// Write figure PNG bytes + caption + provenance into the `chunks` table for
/// each captured figure. One row per figure, keyed by `(document_id, figure_id)`.
/// Used by the media-fetch endpoint to stream the raw PNG back to clients.
///
/// Opens its own `PgPool` from `DATABASE_URL` (mirrors the algorithm-extraction
/// path's pattern). Best-effort — caller logs and swallows errors so a backfill
/// failure doesn't kill the PDF ingest run.
///
/// chunk_index is set to `1_000_000 + page * 1_000 + order_index` to stay clear
/// of any future text-chunk rows the same document might pick up — text chunks
/// today live in KV, not this table, but the unique `(document_id, chunk_index)`
/// constraint means we want a stable, collision-free range.
/// Insert / upsert one `chunks` row per extracted table. Mirrors
/// `backfill_figure_media`: `kind='table'`, `table_id` keys the row, the
/// rendered HTML and parsed rows go into the table_html / table_rows JSONB
/// columns. The caption goes into `content` so retrieval has something
/// human-readable to surface and the classifier has its label.
///
/// `table_type` is left NULL — populated later by
/// `run_table_classification_inline`. Chunk-index lives in
/// [2_000_000, 2_999_999] so it never collides with text (≤ ~1M) or figure
/// rows (1_000_000 + …).
#[cfg(feature = "postgres")]
async fn backfill_table_content(
    early_doc_id: &str,
    tenant_id: uuid::Uuid,
    workspace_id: uuid::Uuid,
    tables: &[edgequake_pdf::ExtractedTable],
) -> Result<(), String> {
    let document_id = uuid::Uuid::parse_str(early_doc_id)
        .map_err(|e| format!("invalid early_doc_id {early_doc_id:?}: {e}"))?;

    let database_url = std::env::var("DATABASE_URL")
        .map_err(|_| "DATABASE_URL not set for table backfill".to_string())?;
    let pool = sqlx::PgPool::connect(&database_url)
        .await
        .map_err(|e| format!("connect failed: {e}"))?;

    let total = tables.len();
    let mut failures: usize = 0;
    let mut first_error: Option<String> = None;
    for t in tables {
        // Bump the chunk_index by 500_000 above the figure range to avoid
        // colliding on a page with the same order_index. Figures live in
        // [1_000_000, 1_999_999]; tables live in [2_000_000, 2_999_999].
        let chunk_index: i32 = 2_000_000
            + (t.page as i32).saturating_mul(1_000)
            + t.order_index as i32;
        let rows_json = serde_json::json!({
            "headers": t.headers,
            "rows": t.rows,
        });
        let metadata = serde_json::json!({
            "kind": "table",
            "table_id": t.id,
            "page": t.page,
            "order_index": t.order_index,
        });
        let res = sqlx::query(
            r#"
            INSERT INTO chunks (
                document_id, tenant_id, workspace_id,
                content, chunk_index,
                kind, table_id, table_html, table_rows,
                metadata
            ) VALUES ($1, $2, $3, $4, $5, 'table', $6, $7, $8, $9)
            ON CONFLICT (document_id, chunk_index) DO UPDATE SET
                content    = EXCLUDED.content,
                kind       = EXCLUDED.kind,
                table_id   = EXCLUDED.table_id,
                table_html = EXCLUDED.table_html,
                table_rows = EXCLUDED.table_rows,
                metadata   = EXCLUDED.metadata
            "#,
        )
        .bind(document_id)
        .bind(tenant_id)
        .bind(workspace_id)
        .bind(&t.caption)
        .bind(chunk_index)
        .bind(&t.id)
        .bind(&t.html)
        .bind(&rows_json)
        .bind(&metadata)
        .execute(&pool)
        .await;
        if let Err(e) = res {
            let err_str = e.to_string();
            warn!(
                table_id = %t.id,
                document_id = %document_id,
                error = %err_str,
                "table backfill: row insert failed"
            );
            if first_error.is_none() {
                first_error = Some(err_str);
            }
            failures += 1;
        }
    }
    if failures > 0 {
        return Err(format!(
            "table backfill: {failures}/{total} inserts failed (first error: {})",
            first_error.as_deref().unwrap_or("<unknown>")
        ));
    }
    Ok(())
}

async fn backfill_figure_media(
    early_doc_id: &str,
    tenant_id: uuid::Uuid,
    workspace_id: uuid::Uuid,
    figures: &[edgequake_pdf::ExtractedFigure],
) -> Result<(), String> {
    let document_id = uuid::Uuid::parse_str(early_doc_id)
        .map_err(|e| format!("invalid early_doc_id {early_doc_id:?}: {e}"))?;

    let database_url = std::env::var("DATABASE_URL")
        .map_err(|_| "DATABASE_URL not set for figure-bytes backfill".to_string())?;
    let pool = sqlx::PgPool::connect(&database_url)
        .await
        .map_err(|e| format!("connect failed: {e}"))?;

    // `ON CONFLICT (document_id, chunk_index) DO UPDATE` makes the backfill
    // idempotent across PDF reprocess runs — a re-extracted figure with the
    // same (page, order_index) overwrites the previous row's bytes/caption.
    let total = figures.len();
    let mut failures: usize = 0;
    let mut first_error: Option<String> = None;
    for fig in figures {
        let chunk_index: i32 = 1_000_000
            + (fig.page as i32).saturating_mul(1_000)
            + fig.order_index as i32;
        let metadata = serde_json::json!({
            "kind": "figure",
            "figure_id": fig.id,
            "page": fig.page,
            "order_index": fig.order_index,
        });
        let res = sqlx::query(
            r#"
            INSERT INTO chunks (
                document_id, tenant_id, workspace_id,
                content, chunk_index,
                kind, figure_id, media_bytes, media_mime,
                metadata
            ) VALUES ($1, $2, $3, $4, $5, 'figure', $6, $7, $8, $9)
            ON CONFLICT (document_id, chunk_index) DO UPDATE SET
                content      = EXCLUDED.content,
                kind         = EXCLUDED.kind,
                figure_id    = EXCLUDED.figure_id,
                media_bytes  = EXCLUDED.media_bytes,
                media_mime   = EXCLUDED.media_mime,
                metadata     = EXCLUDED.metadata
            "#,
        )
        .bind(document_id)
        .bind(tenant_id)
        .bind(workspace_id)
        .bind(&fig.caption)
        .bind(chunk_index)
        .bind(&fig.id)
        .bind(&fig.image_bytes)
        .bind(&fig.mime)
        .bind(&metadata)
        .execute(&pool)
        .await;
        if let Err(e) = res {
            let err_str = e.to_string();
            // Single-row failure shouldn't abort the rest of the batch — log
            // and keep going so subsequent figures still have a chance. The
            // outer Err return below ensures the caller still sees the batch
            // as failed (previously this loop returned Ok(()) even when every
            // INSERT errored, which is how this whole bug went unnoticed).
            warn!(
                figure_id = %fig.id,
                document_id = %document_id,
                error = %err_str,
                "figure backfill: row insert failed"
            );
            if first_error.is_none() {
                first_error = Some(err_str);
            }
            failures += 1;
        }
    }
    if failures > 0 {
        return Err(format!(
            "figure backfill: {failures}/{total} inserts failed (first error: {})",
            first_error.as_deref().unwrap_or("<unknown>")
        ));
    }
    Ok(())
}

#[cfg(all(test, feature = "postgres"))]
mod rename_tests {
    use super::*;

    #[test]
    fn year_from_arxiv_pdf_url() {
        assert_eq!(
            derive_arxiv_year("https://arxiv.org/pdf/2506.09452"),
            Some("2025".to_string())
        );
        assert_eq!(
            derive_arxiv_year("https://arxiv.org/abs/2312.01234v2"),
            Some("2023".to_string())
        );
    }

    #[test]
    fn year_none_for_non_arxiv() {
        assert_eq!(derive_arxiv_year("https://example.com/paper.pdf"), None);
        assert_eq!(derive_arxiv_year(""), None);
    }

    #[test]
    fn format_with_year() {
        assert_eq!(
            format_citation_filename("Roberts", Some("2025"), "Stained Glass Transform"),
            "Roberts et al. - 2025 - Stained Glass Transform.pdf"
        );
    }

    #[test]
    fn format_without_year() {
        assert_eq!(
            format_citation_filename("Smith", None, "Paper Title"),
            "Smith et al. - Paper Title.pdf"
        );
    }

    #[test]
    fn sanitize_strips_forbidden_chars_but_keeps_colon() {
        // `:` is preserved (valid on Linux/macOS, common in academic
        // subtitles). Other Windows-reserved + path-separator chars are
        // replaced with `_`.
        assert_eq!(sanitize_filename_segment("A/B\\C:D?E*F"), "A_B_C:D_E_F");
        assert_eq!(sanitize_filename_segment("  spaces   out  "), "spaces out");
        assert_eq!(
            sanitize_filename_segment("Euston: Efficient Inference"),
            "Euston: Efficient Inference"
        );
    }

    #[test]
    fn truncates_long_title() {
        let long = "a".repeat(200);
        let f = format_citation_filename("X", Some("2020"), &long);
        // "{author} et al. - {year} - " is 17 chars ("X et al. - 2020 - ");
        // title is capped at 120; total = "X et al. - 2020 - " + 120 + ".pdf"
        assert!(f.ends_with(".pdf"));
        let title_part = f
            .strip_prefix("X et al. - 2020 - ")
            .and_then(|s| s.strip_suffix(".pdf"))
            .unwrap();
        assert_eq!(title_part.len(), 120);
    }
}
