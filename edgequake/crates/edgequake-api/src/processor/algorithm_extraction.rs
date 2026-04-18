//! Algorithm extraction task processor.
//!
//! Processes AlgorithmExtraction tasks through the 3-pass LLM pipeline
//! with per-stage status updates visible in the documents UI.
//! Supports workspace-level LLM overrides: separate models for
//! analysis (passes 1,3) and extraction (pass 2).

use super::*;
use futures::stream::{self, StreamExt};
use tokio_util::sync::CancellationToken;

impl DocumentTaskProcessor {
    /// Process an algorithm extraction task through the 3-pass LLM pipeline.
    pub(super) async fn process_algorithm_extraction(
        &self,
        task: &mut Task,
        data: edgequake_tasks::AlgorithmExtractionData,
        cancel_token: CancellationToken,
    ) -> TaskResult<serde_json::Value> {
        let document_id = &data.document_id;

        info!(
            document_id = %document_id,
            workspace_id = %data.workspace_id,
            chunk_count = data.chunks.len(),
            pdf_id = ?data.pdf_id,
            "Processing algorithm extraction task"
        );

        // Resolve per-pass LLM providers from workspace config
        let workspace_id = if !data.workspace_id.is_empty() && data.workspace_id != "default" {
            Some(data.workspace_id.as_str())
        } else {
            None
        };

        let (analysis_provider, extraction_provider) = self
            .resolve_algorithm_llm_providers(workspace_id)
            .await
            .map_err(|e| TaskError::Process(format!("LLM provider error: {e}")))?;

        let analysis_extractor =
            edgequake_algorithms::AlgorithmExtractor::new(analysis_provider);
        let extraction_extractor =
            edgequake_algorithms::AlgorithmExtractor::new(extraction_provider);

        // === Pass 1: Identify algorithm blocks ===
        self.check_cancelled(&cancel_token, "algo_identifying", document_id)
            .await?;
        self.update_document_status(document_id, "algo_identifying", None)
            .await
            .ok();
        task.update_progress("algo_identifying".to_string(), 3, 15);

        // Two paths: layout detection (PDF) or LLM chunk scanning (text).
        // `from_layout_detection` tracks which we took — layout-detected blocks
        // are self-contained per-algorithm chunks, so Pass 2 must NOT feed the
        // LLM overlapping (chunk_a, chunk_b) pairs the way the text path does
        // (that pattern re-extracts the same algorithm on consecutive iterations
        // and produces near-duplicate names).
        let from_layout_detection = data.pdf_id.is_some();
        let (algorithm_chunks, flagged_inventories) = if let Some(ref pdf_id) = data.pdf_id {
            // ── PDF path: layout detection + VLM recognition ──
            // Precise, no hallucination — finds actual algorithm bounding boxes.
            self.update_stage_detail(
                document_id,
                "algo_identifying",
                "Detecting algorithm blocks via layout detection + VLM",
                0.1,
            )
            .await
            .ok();

            let pdf_uuid = uuid::Uuid::parse_str(pdf_id)
                .map_err(|e| TaskError::Process(format!("Invalid pdf_id: {e}")))?;
            let pdf_storage = self
                .pdf_storage
                .as_ref()
                .ok_or_else(|| TaskError::Process("PDF storage not available".to_string()))?;
            let pdf = pdf_storage
                .get_pdf(&pdf_uuid)
                .await
                .map_err(|e| TaskError::Process(format!("Failed to fetch PDF: {e}")))?
                .ok_or_else(|| TaskError::Process("PDF not found".to_string()))?;

            // Resolve VLM config: always use aperture, model from workspace settings.
            let vlm_config = {
                let mut cfg = edgequake_pdf::backend::vlm_client::VlmClientConfig::from_env();
                if let Ok(url) = std::env::var("OPENAI_COMPATIBLE_BASE_URL") {
                    cfg.base_url = url;
                }
                if let Some(ref model) = data.vision_model {
                    if !model.is_empty() {
                        cfg.model = Some(model.clone());
                    }
                }
                cfg
            };

            let blocks = tokio::task::spawn_blocking(move || {
                edgequake_pdf::detect_algorithm_blocks(&pdf.pdf_data, vlm_config)
            })
            .await
            .map_err(|e| TaskError::Process(format!("Layout detection task panicked: {e}")))?
            .map_err(|e| TaskError::Process(format!("Algorithm block detection failed: {e}")))?;

            let block_pages: Vec<_> = blocks.iter().map(|b| b.page.to_string()).collect();
            info!(
                document_id = %document_id,
                block_count = blocks.len(),
                pages = %block_pages.join(","),
                "Pass 1: detected algorithm blocks via layout detection"
            );

            task.update_progress("algo_identifying".to_string(), 3, 40);

            if blocks.is_empty() {
                info!(document_id = %document_id, "No algorithm blocks detected in PDF");
                self.update_document_status(document_id, "completed", None)
                    .await
                    .ok();
                task.update_progress("completed".to_string(), 3, 100);
                return Ok(json!({
                    "document_id": document_id,
                    "algorithm_count": 0,
                    "method": "layout_detection",
                }));
            }

            // Each block becomes a "chunk" for Pass 2. Build a synthetic inventory
            // so the extraction prompt knows what to extract.
            let chunks: Vec<String> = blocks.iter().map(|b| b.markdown.clone()).collect();
            let inventories: Vec<(usize, edgequake_algorithms::AlgorithmInventory)> = blocks
                .iter()
                .enumerate()
                .map(|(i, block)| {
                    let candidate = edgequake_algorithms::AlgorithmCandidate {
                        id: format!("A{}", i + 1),
                        name: format!("Algorithm block (page {})", block.page),
                        description: "Detected by layout analysis".to_string(),
                        location: format!("Page {}", block.page),
                        algorithm_type: "Algorithm".to_string(),
                    };
                    (
                        i,
                        edgequake_algorithms::AlgorithmInventory {
                            paper_title: String::new(),
                            algorithms: vec![candidate],
                            paper_type: String::new(),
                        },
                    )
                })
                .collect();

            (chunks, inventories)
        } else {
            // ── Text path: LLM chunk scanning (original flow) ──

            let chunks = &data.chunks;
            if chunks.is_empty() {
                return Err(TaskError::Process("No chunks provided".to_string()));
            }
            let pair_count = if chunks.len() == 1 { 1 } else { chunks.len() - 1 };

            info!(
                document_id = %document_id,
                chunk_count = chunks.len(),
                pair_count = pair_count,
                "Pass 1: scanning text chunks with LLM"
            );

            let pass1_concurrency = std::env::var("EDGEQUAKE_PDF_CONCURRENCY")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(4)
                .max(1);

            let pass1_futures: Vec<_> = (0..pair_count)
                .map(|pair_idx| {
                    let extractor = analysis_extractor.clone();
                    let chunk_a = chunks[pair_idx].clone();
                    let chunk_b = chunks.get(pair_idx + 1).cloned();
                    async move {
                        let result = extractor
                            .run_inventory_chunk_pair(&chunk_a, chunk_b.as_deref(), pair_idx)
                            .await;
                        (pair_idx, result)
                    }
                })
                .collect();

            let mut pass1_stream =
                stream::iter(pass1_futures).buffer_unordered(pass1_concurrency);

            let mut flagged_pairs: Vec<(usize, edgequake_algorithms::AlgorithmInventory)> =
                Vec::new();
            let mut pass1_completed: usize = 0;

            while let Some((pair_idx, result)) = pass1_stream.next().await {
                self.check_cancelled(&cancel_token, "algo_identifying", document_id)
                    .await?;
                pass1_completed += 1;

                let progress = 10 + (pass1_completed * 30 / pair_count.max(1)) as u8;
                task.update_progress("algo_identifying".to_string(), 3, progress.min(40));

                let stage_message = format!(
                    "Identifying algorithms: chunks {}/{}",
                    pass1_completed, pair_count
                );
                let stage_progress = pass1_completed as f64 / pair_count as f64;
                self.update_stage_detail(
                    document_id,
                    "algo_identifying",
                    &stage_message,
                    stage_progress,
                )
                .await
                .ok();

                match result {
                    Ok(inv) => {
                        if !inv.algorithms.is_empty() {
                            flagged_pairs.push((pair_idx, inv));
                        }
                    }
                    Err(e) => {
                        warn!(
                            document_id = %document_id,
                            pair_index = pair_idx,
                            error = %e,
                            "Pass 1 failed for chunk pair (skipping)"
                        );
                    }
                }
            }

            flagged_pairs.sort_by_key(|(idx, _)| *idx);

            if flagged_pairs.is_empty() {
                info!(document_id = %document_id, "No algorithms found across any chunk pair");
                self.update_document_status(document_id, "completed", None)
                    .await
                    .ok();
                task.update_progress("completed".to_string(), 3, 100);
                return Ok(json!({
                    "document_id": document_id,
                    "algorithm_count": 0,
                    "pairs_scanned": pair_count,
                }));
            }

            (data.chunks.clone(), flagged_pairs)
        };

        let algo_count: usize = flagged_inventories
            .iter()
            .map(|(_, inv)| inv.algorithms.len())
            .sum();

        // === Pass 2: Extraction (uses extraction LLM) ===
        self.check_cancelled(&cancel_token, "algo_extracting", document_id)
            .await?;
        self.update_document_status(document_id, "algo_extracting", None)
            .await
            .ok();
        task.update_progress("algo_extracting".to_string(), 3, 40);

        // Algorithm Pass 2 is heavier per call than entity extraction: each
        // block is a full algorithm definition (32k max_tokens, reasoning on).
        // Heavy vision+extraction models like gemma 4 (26B A4B) saturate at
        // ~2–4 concurrent requests and start timing out past that. Use a
        // dedicated env var; defaults to 2.
        let pass2_concurrency = std::env::var("EDGEQUAKE_ALGO_EXTRACTION_CONCURRENCY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2)
            .max(1);

        let flagged_count = flagged_inventories.len();
        info!(
            document_id = %document_id,
            concurrency = pass2_concurrency,
            flagged_pairs = flagged_count,
            "Pass 2: running extraction concurrently"
        );

        let pass2_futures: Vec<_> = flagged_inventories
            .iter()
            .cloned()
            .map(|(pair_idx, inventory)| {
                let extractor = extraction_extractor.clone();
                let chunk_a = algorithm_chunks[pair_idx].clone();
                // Only use sliding (chunk_a, chunk_b) pairs for the text path,
                // where algorithms may straddle chunk boundaries. Layout-detected
                // blocks are self-contained per-algorithm VLM crops — pairing
                // them causes each algorithm to be re-extracted on two
                // consecutive iterations and produce duplicate names.
                let chunk_b = if from_layout_detection {
                    None
                } else {
                    algorithm_chunks.get(pair_idx + 1).cloned()
                };
                async move {
                    let result = extractor
                        .run_extraction_chunk_pair(
                            &chunk_a,
                            chunk_b.as_deref(),
                            &inventory,
                            pair_idx,
                        )
                        .await;
                    (pair_idx, result)
                }
            })
            .collect();

        let mut pass2_stream =
            stream::iter(pass2_futures).buffer_unordered(pass2_concurrency);

        let mut all_extracted: Vec<edgequake_algorithms::ExtractedAlgorithm> = Vec::new();
        let mut pass2_completed: usize = 0;

        while let Some((pair_idx, result)) = pass2_stream.next().await {
            self.check_cancelled(&cancel_token, "algo_extracting", document_id)
                .await?;
            pass2_completed += 1;

            let progress = 40 + (pass2_completed * 30 / flagged_count.max(1)) as u8;
            task.update_progress("algo_extracting".to_string(), 3, progress.min(70));

            let stage_message = format!(
                "Extracting algorithms: {}/{}",
                pass2_completed, flagged_count
            );
            let stage_progress = pass2_completed as f64 / flagged_count as f64;
            self.update_stage_detail(
                document_id,
                "algo_extracting",
                &stage_message,
                stage_progress,
            )
            .await
            .ok();

            match result {
                Ok(ext) => all_extracted.extend(ext.algorithms),
                Err(e) => {
                    warn!(
                        document_id = %document_id,
                        pair_index = pair_idx,
                        error = %e,
                        "Pass 2 failed for block (skipping)"
                    );
                }
            }
        }

        // Deduplicate by normalized algorithm name
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut deduped: Vec<edgequake_algorithms::ExtractedAlgorithm> = Vec::new();
        for algo in all_extracted {
            let key = algo
                .name
                .trim()
                .to_lowercase()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            if seen.insert(key) {
                deduped.push(algo);
            }
        }

        let extraction = edgequake_algorithms::AlgorithmExtractionOutput {
            algorithms: deduped,
        };

        info!(
            document_id = %document_id,
            flagged_pairs = flagged_count,
            unique_algorithms = extraction.algorithms.len(),
            "Pass 2 complete after deduplication"
        );

        // === Pass 3: Verification (uses analysis LLM) ===
        self.check_cancelled(&cancel_token, "algo_verifying", document_id)
            .await?;
        self.update_document_status(document_id, "algo_verifying", None)
            .await
            .ok();
        task.update_progress("algo_verifying".to_string(), 3, 70);

        let verification = analysis_extractor.run_verification(&extraction).await;

        // === Store results ===
        task.update_progress("storing".to_string(), 3, 90);

        let count = extraction.algorithms.len();

        // Check workspace review mode
        let auto_approve = self
            .resolve_algorithm_review_mode(workspace_id)
            .await
            .unwrap_or(false);

        #[cfg(feature = "postgres")]
        {
            use edgequake_algorithms::{
                Algorithm, AlgorithmStatus, AlgorithmStorage, PostgresAlgorithmStorage,
            };

            let database_url = std::env::var("DATABASE_URL").map_err(|_| {
                TaskError::Process("DATABASE_URL not set for algorithm storage".to_string())
            })?;
            let pool = sqlx::PgPool::connect(&database_url).await.map_err(|e| {
                TaskError::Process(format!("Failed to connect to database: {e}"))
            })?;
            let storage = PostgresAlgorithmStorage::new(std::sync::Arc::new(pool));

            let tenant_id = task.tenant_id;
            let workspace_id_uuid = task.workspace_id;

            // Delete existing algorithms (re-extraction)
            let _ = storage
                .delete_algorithms_by_document(document_id, tenant_id, workspace_id_uuid)
                .await;

            let initial_status = if auto_approve {
                AlgorithmStatus::Approved
            } else {
                AlgorithmStatus::Pending
            };

            let now = chrono::Utc::now();
            let algorithms: Vec<Algorithm> = extraction
                .algorithms
                .into_iter()
                .map(|ea| Algorithm {
                    id: uuid::Uuid::new_v4(),
                    tenant_id,
                    workspace_id: workspace_id_uuid,
                    document_id: document_id.clone(),
                    name: ea.name,
                    algorithm_type: ea.algorithm_type,
                    description: if ea.description.is_empty() {
                        None
                    } else {
                        Some(ea.description)
                    },
                    steps: ea.steps,
                    inputs: ea.inputs,
                    outputs: ea.outputs,
                    preconditions: ea.preconditions,
                    complexity: ea.complexity,
                    mathematical_notation: ea.mathematical_notation,
                    pseudocode: ea.pseudocode,
                    tags: ea.tags,
                    confidence: ea.confidence,
                    status: initial_status,
                    verification_status: verification
                        .as_ref()
                        .map(|v| v.verification_status.clone()),
                    verification_details: verification
                        .as_ref()
                        .and_then(|v| serde_json::to_value(v).ok()),
                    created_at: now,
                    updated_at: now,
                })
                .collect();

            let stored_count = algorithms.len();
            let algorithm_ids: Vec<String> =
                algorithms.iter().map(|a| a.id.to_string()).collect();

            storage
                .create_algorithms(&algorithms)
                .await
                .map_err(|e| TaskError::Process(format!("Failed to store algorithms: {e}")))?;

            info!(
                document_id = %document_id,
                count = stored_count,
                auto_approve = auto_approve,
                "Algorithm extraction complete — stored in database"
            );

            // Auto-approve mode: run embedding inline as a continuation
            if auto_approve && !algorithm_ids.is_empty() {
                info!(document_id = %document_id, count = algorithm_ids.len(), "Auto-approve: embedding algorithms inline");

                self.update_document_status(document_id, "algo_embedding", None)
                    .await
                    .ok();
                task.update_progress("algo_embedding".to_string(), 4, 92);

                let embed_data = edgequake_tasks::AlgorithmEmbeddingData {
                    document_id: document_id.clone(),
                    workspace_id: data.workspace_id.clone(),
                    algorithm_ids,
                };

                // Reuse the embedding processor method via a sub-task
                // We run it inline since the processor is already active
                match self
                    .process_algorithm_embedding(task, embed_data, cancel_token.clone())
                    .await
                {
                    Ok(_) => {
                        info!(document_id = %document_id, "Auto-approve: embedding complete");
                    }
                    Err(e) => {
                        warn!(error = %e, "Auto-approve: embedding failed (non-fatal, algorithms still stored)");
                    }
                }
                // process_algorithm_embedding already sets completed status
                return Ok(json!({
                    "document_id": document_id,
                    "algorithm_count": count,
                    "algorithms_identified": algo_count,
                    "auto_approved": true,
                    "embedded": true,
                    "verification_status": verification.as_ref().map(|v| &v.verification_status),
                }));
            }
        }

        // Restore document to completed status
        self.update_document_status(document_id, "completed", None)
            .await
            .ok();
        task.update_progress("completed".to_string(), 3, 100);

        self.pipeline_state
            .info(format!(
                "Algorithm extraction complete: {} algorithms from document {}",
                count, document_id
            ))
            .await;

        Ok(json!({
            "document_id": document_id,
            "algorithm_count": count,
            "algorithms_identified": algo_count,
            "auto_approved": auto_approve,
            "verification_status": verification.as_ref().map(|v| &v.verification_status),
        }))
    }

    /// Resolve per-pass LLM providers for algorithm extraction.
    ///
    /// Returns (analysis_provider, extraction_provider) where:
    /// - analysis_provider is used for passes 1 (inventory) and 3 (verification)
    /// - extraction_provider is used for pass 2 (detailed extraction)
    ///
    /// Falls back to workspace default LLM, then server default.
    async fn resolve_algorithm_llm_providers(
        &self,
        workspace_id: Option<&str>,
    ) -> Result<(
        std::sync::Arc<dyn edgequake_llm::traits::LLMProvider>,
        std::sync::Arc<dyn edgequake_llm::traits::LLMProvider>,
    ), String> {
        use crate::safety_limits::create_safe_llm_provider;

        let default_provider = std::sync::Arc::clone(&self.llm_provider);

        let workspace_service = match &self.workspace_service {
            Some(ws) => ws,
            _ => return Ok((default_provider.clone(), default_provider)),
        };

        let workspace_id = match workspace_id {
            Some(id) if !id.is_empty() && id != "default" => id,
            _ => return Ok((default_provider.clone(), default_provider)),
        };

        let workspace_uuid = uuid::Uuid::parse_str(workspace_id)
            .map_err(|e| format!("Invalid workspace ID: {e}"))?;

        let ws = workspace_service
            .get_workspace(workspace_uuid)
            .await
            .map_err(|e| format!("Failed to get workspace: {e}"))?
            .ok_or_else(|| format!("Workspace not found: {workspace_id}"))?;

        // Resolve analysis provider: algorithm_analysis_llm > workspace default > server default
        let analysis_provider = if let (Some(ref provider), Some(ref model)) = (
            &ws.algorithm_analysis_llm_provider,
            &ws.algorithm_analysis_llm_model,
        ) {
            create_safe_llm_provider(provider, model)
                .map_err(|e| format!("Failed to create analysis LLM provider: {e}"))?
        } else if self.strict_workspace_mode {
            create_safe_llm_provider(&ws.llm_provider, &ws.llm_model)
                .map_err(|e| format!("Failed to create workspace LLM provider: {e}"))?
        } else {
            default_provider.clone()
        };

        // Resolve extraction provider: algorithm_extraction_llm > workspace default > server default
        let extraction_provider = if let (Some(ref provider), Some(ref model)) = (
            &ws.algorithm_extraction_llm_provider,
            &ws.algorithm_extraction_llm_model,
        ) {
            create_safe_llm_provider(provider, model)
                .map_err(|e| format!("Failed to create extraction LLM provider: {e}"))?
        } else if self.strict_workspace_mode {
            create_safe_llm_provider(&ws.llm_provider, &ws.llm_model)
                .map_err(|e| format!("Failed to create workspace LLM provider: {e}"))?
        } else {
            default_provider.clone()
        };

        Ok((analysis_provider, extraction_provider))
    }

    /// Check if workspace uses auto-approve mode for algorithms.
    async fn resolve_algorithm_review_mode(
        &self,
        workspace_id: Option<&str>,
    ) -> Result<bool, String> {
        let workspace_service = match &self.workspace_service {
            Some(ws) => ws,
            _ => return Ok(false),
        };

        let workspace_id = match workspace_id {
            Some(id) if !id.is_empty() && id != "default" => id,
            _ => return Ok(false),
        };

        let workspace_uuid = uuid::Uuid::parse_str(workspace_id)
            .map_err(|e| format!("Invalid workspace ID: {e}"))?;

        let ws = workspace_service
            .get_workspace(workspace_uuid)
            .await
            .map_err(|e| format!("Failed to get workspace: {e}"))?
            .ok_or_else(|| format!("Workspace not found: {workspace_id}"))?;

        Ok(ws
            .algorithm_review_mode
            .as_deref()
            .map(|m| m == "auto")
            .unwrap_or(false))
    }

    /// Run algorithm extraction Pass 2+3 on pre-detected algorithm blocks.
    ///
    /// Called from PDF processing when VLM-OCR detects algorithm blocks during
    /// conversion. Skips Pass 1 (already done via layout detection).
    /// Returns the number of algorithms stored.
    pub(super) async fn run_algorithm_pass2_pass3(
        &self,
        document_id: &str,
        workspace_id: Option<&str>,
        blocks: &[edgequake_pdf::AlgorithmBlock],
        task: &mut Task,
    ) -> TaskResult<usize> {
        use futures::stream::{self, StreamExt};

        let (analysis_provider, extraction_provider) = self
            .resolve_algorithm_llm_providers(workspace_id)
            .await
            .map_err(|e| TaskError::Process(format!("LLM provider error: {e}")))?;

        let analysis_extractor =
            edgequake_algorithms::AlgorithmExtractor::new(analysis_provider);
        let extraction_extractor =
            edgequake_algorithms::AlgorithmExtractor::new(extraction_provider);

        // Build chunks + synthetic inventories from detected blocks.
        let chunks: Vec<String> = blocks.iter().map(|b| b.markdown.clone()).collect();
        let inventories: Vec<(usize, edgequake_algorithms::AlgorithmInventory)> = blocks
            .iter()
            .enumerate()
            .map(|(i, block)| {
                let candidate = edgequake_algorithms::AlgorithmCandidate {
                    id: format!("A{}", i + 1),
                    name: format!("Algorithm block (page {})", block.page),
                    description: "Detected by layout analysis".to_string(),
                    location: format!("Page {}", block.page),
                    algorithm_type: "Algorithm".to_string(),
                };
                (
                    i,
                    edgequake_algorithms::AlgorithmInventory {
                        paper_title: String::new(),
                        algorithms: vec![candidate],
                        paper_type: String::new(),
                    },
                )
            })
            .collect();

        // === Pass 2: Extraction ===
        // Dedicated env var — see comment in the other pass2_concurrency site.
        let pass2_concurrency = std::env::var("EDGEQUAKE_ALGO_EXTRACTION_CONCURRENCY")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .unwrap_or(2)
            .max(1);

        let block_count = inventories.len();
        info!(
            document_id = %document_id,
            block_count = block_count,
            concurrency = pass2_concurrency,
            "Auto algorithm extraction: running Pass 2"
        );

        // Layout-detected blocks are already per-algorithm and self-contained
        // (each block = one cropped algorithm region's VLM text). The sliding-pair
        // chunk_a+chunk_b pattern used by the text path is WRONG here — it would
        // feed the LLM overlapping pairs of algorithms, causing the same algorithm
        // to be re-extracted on consecutive iterations and producing duplicates.
        // Pass None for chunk_b so each block is processed exactly once.
        let pass2_futures: Vec<_> = inventories
            .iter()
            .cloned()
            .map(|(idx, inventory)| {
                let extractor = extraction_extractor.clone();
                let chunk_a = chunks[idx].clone();
                async move {
                    let result = extractor
                        .run_extraction_chunk_pair(&chunk_a, None, &inventory, idx)
                        .await;
                    (idx, result)
                }
            })
            .collect();

        let mut pass2_stream =
            stream::iter(pass2_futures).buffer_unordered(pass2_concurrency);

        let mut all_extracted: Vec<edgequake_algorithms::ExtractedAlgorithm> = Vec::new();
        let mut completed: usize = 0;
        while let Some((idx, result)) = pass2_stream.next().await {
            completed += 1;
            let stage_progress = completed as f64 / block_count.max(1) as f64;
            self.update_stage_detail(
                document_id,
                "algo_extracting",
                &format!("Extracting algorithms: {}/{}", completed, block_count),
                stage_progress,
            )
            .await
            .ok();

            match result {
                Ok(ext) => all_extracted.extend(ext.algorithms),
                Err(e) => {
                    warn!(
                        document_id = %document_id,
                        block = idx,
                        error = %e,
                        "Auto algo Pass 2 failed for block (skipping)"
                    );
                }
            }
        }

        // Deduplicate
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut deduped: Vec<edgequake_algorithms::ExtractedAlgorithm> = Vec::new();
        for algo in all_extracted {
            let key = algo.name.trim().to_lowercase().split_whitespace().collect::<Vec<_>>().join(" ");
            if seen.insert(key) {
                deduped.push(algo);
            }
        }

        let extraction = edgequake_algorithms::AlgorithmExtractionOutput { algorithms: deduped };

        info!(
            document_id = %document_id,
            unique_algorithms = extraction.algorithms.len(),
            "Auto algorithm extraction: Pass 2 complete"
        );

        if extraction.algorithms.is_empty() {
            return Ok(0);
        }

        // === Pass 3: Verification ===
        self.update_document_status(document_id, "algo_verifying", None)
            .await
            .ok();
        task.update_progress("algo_verifying".to_string(), 6, 85);

        let verification = analysis_extractor.run_verification(&extraction).await;

        // === Store results ===
        let count = extraction.algorithms.len();
        let auto_approve = self
            .resolve_algorithm_review_mode(workspace_id)
            .await
            .unwrap_or(false);

        #[cfg(feature = "postgres")]
        {
            use edgequake_algorithms::{
                Algorithm, AlgorithmStatus, AlgorithmStorage, PostgresAlgorithmStorage,
            };

            let database_url = std::env::var("DATABASE_URL").map_err(|_| {
                TaskError::Process("DATABASE_URL not set for algorithm storage".to_string())
            })?;
            let pool = sqlx::PgPool::connect(&database_url).await.map_err(|e| {
                TaskError::Process(format!("Failed to connect to database: {e}"))
            })?;
            let storage = PostgresAlgorithmStorage::new(std::sync::Arc::new(pool));

            let tenant_id = task.tenant_id;
            let workspace_id_uuid = task.workspace_id;

            // Delete existing algorithms (re-extraction)
            let _ = storage
                .delete_algorithms_by_document(document_id, tenant_id, workspace_id_uuid)
                .await;

            let initial_status = if auto_approve {
                AlgorithmStatus::Approved
            } else {
                AlgorithmStatus::Pending
            };

            let now = chrono::Utc::now();
            let algorithms: Vec<Algorithm> = extraction
                .algorithms
                .into_iter()
                .map(|ea| Algorithm {
                    id: uuid::Uuid::new_v4(),
                    tenant_id,
                    workspace_id: workspace_id_uuid,
                    document_id: document_id.to_string(),
                    name: ea.name,
                    algorithm_type: ea.algorithm_type,
                    description: if ea.description.is_empty() { None } else { Some(ea.description) },
                    steps: ea.steps,
                    inputs: ea.inputs,
                    outputs: ea.outputs,
                    preconditions: ea.preconditions,
                    complexity: ea.complexity,
                    mathematical_notation: ea.mathematical_notation,
                    pseudocode: ea.pseudocode,
                    tags: ea.tags,
                    confidence: ea.confidence,
                    status: initial_status,
                    verification_status: verification.as_ref().map(|v| v.verification_status.clone()),
                    verification_details: verification.as_ref().and_then(|v| serde_json::to_value(v).ok()),
                    created_at: now,
                    updated_at: now,
                })
                .collect();

            storage
                .create_algorithms(&algorithms)
                .await
                .map_err(|e| TaskError::Process(format!("Failed to store algorithms: {e}")))?;

            info!(
                document_id = %document_id,
                count = count,
                auto_approve = auto_approve,
                "Auto algorithm extraction stored in database"
            );
        }

        Ok(count)
    }
}
