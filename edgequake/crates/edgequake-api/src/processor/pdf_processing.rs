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

        // FIX-REBUILD: When reprocessing, clean up old content and chunk KV entries
        // WHY: Old chunks with stale content must be removed before the pipeline
        // creates new ones, otherwise the document ends up with a mix of old and new chunks.
        if is_reprocess {
            info!(
                document_id = %early_doc_id,
                pdf_id = %data.pdf_id,
                "Reprocessing: cleaning up old content and chunks before re-extraction"
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

        let conversion_config = edgequake_pdf::PdfConversionConfig {
            page_count_hint: pdf.page_count.map(|count| count as usize),
            table_method: None,
            filename: Some(pdf.filename.clone()),
            vision: Some(vision_config),
            vlm_base_url,
            vlm_model,
            algorithm_block_sink: algo_block_sink.clone(),
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

        // 5.5. Post-OCR rename: if this upload came from the URL path
        // (which stamps `rename_after_parse=true` + `source_url` on the
        // task payload), parse front-matter and replace the opaque
        // upload-time filename with a citation-style name. Best-effort
        // — on any failure we keep the original filename so the row
        // doesn't end up with a confusing half-renamed state.
        let pdf = self
            .maybe_rename_from_front_matter(
                &pdf,
                &markdown,
                data.rename_after_parse,
                data.source_url.as_deref(),
                &pdf_storage,
            )
            .await;

        // 6. Create document via standard pipeline
        // == Progress: markdown stored, starting entity extraction + indexing ==
        task.update_progress("entity_extraction".to_string(), 4, 50);

        // ── CANCELLATION GATE: before handing off to text_insert pipeline ──
        self.check_cancelled(&cancel_token, "pre-text-insert", &early_doc_id)
            .await?;

        // SPEC-002: Include source_type: "pdf" for unified pipeline tracking
        // OODA-05: Include tenant_id/workspace_id for multi-tenant document visibility
        // Pass the early_doc_id so we reuse the same document that's already showing in UI
        // OODA-04: Include sha256_checksum for end-to-end lineage traceability
        // WHY: Downstream ensure_document_source_type needs checksum for integrity verification
        let text_data = edgequake_tasks::TextInsertData {
            text: markdown.clone(),
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
                // WHY: The lineage builder in documents.rs reads from this metadata JSON.
                // vision_model and extraction_method are stored in pdf_documents table but
                // not in the KV document metadata, making them invisible in the lineage view.
                "pdf_vision_model": vision_model,
                "pdf_extraction_method": extraction_method.as_str(),
                "pdf_extraction_warning": extraction_warning,
            })),
        };

        let result = self
            .process_text_insert(task, text_data, cancel_token)
            .await?;

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

        // 8. Automatic algorithm extraction (VLM-OCR only).
        // If algorithm blocks were detected during conversion, run Pass 2+3 inline.
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
                task.update_progress("algo_extracting".to_string(), 6, 80);

                let workspace_id_str = data.workspace_id.to_string();
                let ws_id = if workspace_id_str != "default" && !workspace_id_str.is_empty() {
                    Some(workspace_id_str.as_str())
                } else {
                    None
                };

                match self
                    .run_algorithm_pass2_pass3(&early_doc_id, ws_id, &blocks, task)
                    .await
                {
                    Ok(count) => {
                        info!(
                            pdf_id = %data.pdf_id,
                            algorithm_count = count,
                            "Auto algorithm extraction completed"
                        );
                    }
                    Err(e) => {
                        warn!(
                            pdf_id = %data.pdf_id,
                            error = %e,
                            "Auto algorithm extraction failed (non-fatal)"
                        );
                    }
                }

                // Restore status to completed after algorithm extraction.
                self.update_document_status(&early_doc_id, "completed", None)
                    .await
                    .ok();
            }
        }

        // 9. Reference-repo detection (Phase 0 of the Reference Code GraphRAG
        // extension). Layer A runs against the PDF bytes; if nothing is found,
        // Layer B falls through to SearXNG + Crawl4AI when those env vars are
        // set. Failures are swallowed — detection is opportunistic, not
        // required for a successful ingest.
        //
        // We need the raw PDF bytes again plus the markdown. Re-fetching from
        // storage (rather than holding onto the earlier `pdf.pdf_data` clone)
        // keeps the hot path memory footprint small even when this step is a
        // no-op.
        let pdf_data_for_detection = pdf_storage
            .get_pdf(&data.pdf_id)
            .await
            .ok()
            .flatten()
            .map(|p| p.pdf_data)
            .unwrap_or_default();
        // `markdown` is the freshly extracted content local to this task —
        // identical to what we just persisted into `pdf_documents`. Pass it
        // directly instead of round-tripping through KV (the old
        // `{doc_id}-content` key is no longer written, so that lookup would
        // leave Layer B with an empty string and trigger `NoFrontMatter`).
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
                "Reference-repo detection failed (non-fatal)"
            );
        }

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
        rename_requested: bool,
        source_url: Option<&str>,
        pdf_storage: &std::sync::Arc<dyn edgequake_storage::PdfDocumentStorage>,
    ) -> edgequake_storage::PdfDocument {
        use edgequake_agents::web_search::extract_front_matter;

        if !rename_requested {
            return pdf.clone();
        }

        let Some(fm) = extract_front_matter(markdown) else {
            info!(
                pdf_id = %pdf.pdf_id,
                "rename skipped: no front-matter extractable"
            );
            return pdf.clone();
        };

        let Some(author) = fm.first_author.as_deref().map(str::trim).filter(|s| !s.is_empty())
        else {
            info!(
                pdf_id = %pdf.pdf_id,
                "rename skipped: front-matter has no first author"
            );
            return pdf.clone();
        };

        let year = source_url.and_then(derive_arxiv_year);

        let new_filename = format_citation_filename(author, year.as_deref(), &fm.title);
        if new_filename == pdf.filename {
            return pdf.clone();
        }

        if let Err(e) = pdf_storage
            .update_pdf_filename(&pdf.pdf_id, &new_filename)
            .await
        {
            warn!(
                pdf_id = %pdf.pdf_id,
                error = %e,
                "rename failed at storage — keeping original filename"
            );
            return pdf.clone();
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
/// survive). Also strips leading/trailing whitespace and `.`.
#[cfg(feature = "postgres")]
fn sanitize_filename_segment(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '/' | '\\' | '<' | '>' | ':' | '"' | '|' | '?' | '*' => out.push('_'),
            c if c.is_control() => out.push('_'),
            c => out.push(c),
        }
    }
    // Collapse whitespace runs to single spaces.
    let collapsed: String = out.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed.trim_matches('.').to_string()
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
    fn sanitize_strips_forbidden_chars() {
        assert_eq!(sanitize_filename_segment("A/B\\C:D?E*F"), "A_B_C_D_E_F");
        assert_eq!(sanitize_filename_segment("  spaces   out  "), "spaces out");
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
