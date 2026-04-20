use axum::extract::State;
use axum::Json;
use axum_extra::extract::Multipart;
use serde::Deserialize;
use tracing::{debug, info, warn};

use super::helpers::{
    clear_document_derived_data, create_pdf_processing_task, estimate_processing_time,
    extract_page_count, get_pdf_storage,
};
use super::types::*;
use crate::error::{ApiError, ApiResult};
use crate::middleware::TenantContext;
use crate::state::AppState;
use edgequake_pdf::PdfParserBackend;
use edgequake_storage::{
    calculate_pdf_checksum, validate_pdf_data, CreatePdfRequest, PdfProcessingStatus,
};

/// Upper bound on the bytes we'll pull from a remote URL in the
/// `from-url` ingestion path. Matches the practical cap of the multipart
/// route (100 MB) and guards against adversarial hosts returning huge
/// streams.
const MAX_URL_FETCH_BYTES: u64 = 100 * 1024 * 1024;

/// Magic bytes every PDF must start with. Used after HEAD-check to reject
/// servers that report `application/pdf` but actually ship HTML / garbage.
const PDF_MAGIC: &[u8] = b"%PDF-";

// ============================================================================
// Handlers
// ============================================================================

/// Upload a PDF document.
///
/// @implements SPEC-007: PDF Upload Support
/// @implements UC0701: Upload PDF for processing
/// @implements BR0702: 100MB file size limit
/// @implements BR0703: Deduplication via SHA-256
///
/// # Flow
///
/// 1. Parse multipart form data
/// 2. Validate PDF file (size, format, signature)
/// 3. Calculate SHA-256 checksum
/// 4. Check for duplicates
/// 5. Store raw PDF in database
/// 6. Create background processing task
/// 7. Return response with task ID
///
/// # Arguments
///
/// * `state` - Application state with PDF storage
/// * `context` - Tenant context (workspace, tenant IDs)
/// * `multipart` - Multipart form data with PDF file
///
/// # Returns
///
/// * `Ok(Json(PdfUploadResponse))` - Upload successful
/// * `Err(ApiError)` - Validation or storage failure
///
/// # Errors
///
/// - `ApiError::PayloadTooLarge` - File exceeds 100MB
/// - `ApiError::BadRequest` - Invalid PDF format
/// - `ApiError::Conflict` - Duplicate PDF detected
/// - `ApiError::Internal` - Storage failure
#[utoipa::path(
    post,
    path = "/api/v1/documents/pdf",
    request_body(content_type = "multipart/form-data"),
    responses(
        (status = 200, description = "PDF uploaded successfully", body = PdfUploadResponse),
        (status = 400, description = "Invalid PDF or request"),
        (status = 409, description = "Duplicate PDF"),
        (status = 413, description = "File too large"),
        (status = 500, description = "Internal server error")
    ),
    tag = "Documents"
)]
pub async fn upload_pdf_document(
    State(state): State<AppState>,
    context: TenantContext,
    mut multipart: Multipart,
) -> ApiResult<Json<PdfUploadResponse>> {
    info!(
        "PDF upload request: workspace={:?}, tenant={:?}",
        context.workspace_id, context.tenant_id
    );

    // 1. Parse multipart fields
    let mut file_data: Option<Vec<u8>> = None;
    let mut filename = String::from("document.pdf");
    let mut options = PdfUploadOptions {
        enable_vision: true,
        vision_provider: None, // None = apply workspace config then server default
        vision_model: None,
        title: None,
        metadata: None,
        track_id: None,
        force_reindex: false,
        pdf_parser_backend: None,
    };

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| ApiError::BadRequest(format!("Failed to parse multipart: {}", e)))?
    {
        match field.name() {
            Some("file") => {
                filename = field.file_name().unwrap_or("document.pdf").to_string();
                file_data = Some(
                    field
                        .bytes()
                        .await
                        .map_err(|e| ApiError::BadRequest(format!("Failed to read file: {}", e)))?
                        .to_vec(),
                );
            }
            Some("enable_vision") => {
                if let Ok(text) = field.text().await {
                    options.enable_vision = text.parse().unwrap_or(true);
                }
            }
            Some("vision_provider") => {
                if let Ok(text) = field.text().await {
                    options.vision_provider = Some(text);
                }
            }
            Some("vision_model") => {
                if let Ok(text) = field.text().await {
                    options.vision_model = Some(text);
                }
            }
            Some("title") => {
                if let Ok(text) = field.text().await {
                    options.title = Some(text);
                }
            }
            Some("metadata") => {
                if let Ok(text) = field.text().await {
                    if let Ok(json) = serde_json::from_str(&text) {
                        options.metadata = Some(json);
                    }
                }
            }
            Some("track_id") => {
                if let Ok(text) = field.text().await {
                    options.track_id = Some(text);
                }
            }
            Some("force_reindex") => {
                // OODA-08: Parse force_reindex parameter
                // WHY: Allows re-processing of duplicate documents
                if let Ok(text) = field.text().await {
                    options.force_reindex = text.parse().unwrap_or(false);
                }
            }
            Some("pdf_parser_backend") => {
                if let Ok(text) = field.text().await {
                    options.pdf_parser_backend = PdfParserBackend::from_env_str(&text);
                }
            }
            _ => {}
        }
    }

    // 2. Validate file data
    let file_data = file_data.ok_or_else(|| {
        ApiError::BadRequest("Missing 'file' field in multipart request".to_string())
    })?;

    // Delegate to the shared ingestion flow — everything below used to
    // live inline here and is identical for the `from-url` path, which
    // also arrives as `(bytes, filename, options)`.
    ingest_pdf_bytes(state, context, file_data, filename, options).await
}

/// Shared ingestion path for any already-buffered PDF bytes, regardless
/// of how they were obtained (multipart upload vs. server-side URL
/// fetch). Performs validation → dedup → storage → task creation and
/// returns the standard `PdfUploadResponse`.
async fn ingest_pdf_bytes(
    state: AppState,
    context: TenantContext,
    file_data: Vec<u8>,
    filename: String,
    mut options: PdfUploadOptions,
) -> ApiResult<Json<PdfUploadResponse>> {
    validate_pdf_data(&file_data)
        .map_err(|e| ApiError::BadRequest(format!("Invalid PDF: {}", e)))?;

    // 3. Calculate checksum
    let checksum = calculate_pdf_checksum(&file_data);

    debug!(
        "PDF validation passed: size={}, checksum={}",
        file_data.len(),
        checksum
    );

    // 4. Get PDF storage (platform-specific)
    let pdf_storage = get_pdf_storage(&state)?;

    // 5. Extract workspace_id as UUID
    let workspace_id = context
        .workspace_id_uuid()
        .ok_or_else(|| ApiError::BadRequest("Workspace ID required".to_string()))?;

    // 5b. SPEC-040: Apply workspace-level vision LLM config as defaults.
    // Priority: form explicit > workspace config > server default.
    // WHY: Workspace can pin a specific vision provider/model for all PDF uploads,
    // avoiding the need for callers to pass vision_provider/vision_model every time.
    // 5b. SPEC-040: Apply workspace-level vision LLM config as defaults.
    // Priority: form explicit > workspace vision config > workspace main LLM > server env default.
    // WHY: When vision_llm_provider is not set in the workspace, fall back to the workspace's
    // main llm_provider so that Ollama users don't silently hit the "openai" hard-coded default.
    let workspace = state
        .workspace_service
        .get_workspace(workspace_id)
        .await
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    let resolved_backend = options.resolved_backend(workspace.as_ref());

    if matches!(
        resolved_backend,
        PdfParserBackend::Vision | PdfParserBackend::VlmOcr
    ) && (options.vision_provider.is_none() || options.vision_model.is_none())
    {
        if let Some(ws) = workspace.as_ref() {
            if options.vision_provider.is_none() {
                let effective_vision_provider = ws
                    .vision_llm_provider
                    .as_deref()
                    .filter(|p| !p.is_empty())
                    .unwrap_or(&ws.llm_provider);
                debug!(
                    "SPEC-040: Resolved vision_provider={} (workspace vision={:?}, main={})",
                    effective_vision_provider, ws.vision_llm_provider, ws.llm_provider
                );
                options.vision_provider = Some(effective_vision_provider.to_string());
            }
            if options.vision_model.is_none() {
                if let Some(ref wm) = ws.vision_llm_model {
                    debug!(
                        "SPEC-040: Applying workspace vision_model={} from workspace config",
                        wm
                    );
                    options.vision_model = Some(wm.clone());
                }
                // Note: if ws.vision_llm_model is also None, resolved_vision_provider() +
                // default_vision_model_for_provider() will derive the right default at task creation.
            }
        }
    }

    // 6. Check for duplicates
    if let Some(existing) = pdf_storage
        .find_pdf_by_checksum(&workspace_id, &checksum)
        .await
        .map_err(|e| ApiError::Internal(format!("Failed to check for duplicates: {}", e)))?
    {
        // OODA-08: Handle force_reindex parameter
        // WHY: When user explicitly requests re-indexing, we should:
        //      1. Clear existing graph/vector data for this document
        //      2. Reset PDF processing status
        //      3. Create new processing task
        if options.force_reindex {
            info!(
                "OODA-08: Force re-indexing requested for existing PDF: id={}, document_id={:?}",
                existing.pdf_id, existing.document_id
            );

            // Clear existing document data if document_id exists
            if let Some(document_id) = existing.document_id {
                if let Err(e) = clear_document_derived_data(&state, &document_id.to_string()).await
                {
                    warn!(
                        "Failed to clear document data during re-index: {} (continuing anyway)",
                        e
                    );
                }
            }

            // Reset PDF processing status to pending
            pdf_storage
                .update_pdf_status(&existing.pdf_id, PdfProcessingStatus::Processing)
                .await
                .map_err(|e| ApiError::Internal(format!("Failed to reset PDF status: {}", e)))?;

            // Create new processing task
            let task_id = create_pdf_processing_task(
                &state,
                &context,
                existing.pdf_id,
                &options,
                workspace.as_ref(),
            )
            .await?;

            // Initialize progress tracking
            let effective_track_id = options.track_id.clone().unwrap_or_else(|| task_id.clone());
            info!(
                "OODA-08: Re-indexing PDF progress for track_id={}, pdf_id={}, filename={}",
                effective_track_id, existing.pdf_id, existing.filename
            );
            state
                .pipeline_state
                .start_pdf_progress(
                    &effective_track_id,
                    &existing.pdf_id.to_string(),
                    &existing.filename,
                )
                .await;

            let estimated_time = estimate_processing_time(&[], existing.page_count);

            return Ok(Json(PdfUploadResponse {
                pdf_id: existing.pdf_id.to_string(),
                document_id: None, // Will be set after re-processing
                status: "reindexing".to_string(),
                task_id: task_id.to_string(),
                track_id: options.track_id.clone(),
                message: "Re-indexing document. Previous graph/vector data cleared.".to_string(),
                estimated_time_seconds: estimated_time,
                metadata: PdfMetadata {
                    filename: existing.filename,
                    file_size_bytes: existing.file_size_bytes,
                    page_count: existing.page_count,
                    sha256_checksum: existing.sha256_checksum,
                    vision_enabled: options.enable_vision,
                    vision_model: if options.enable_vision {
                        Some(options.vision_model())
                    } else {
                        None
                    },
                },
                duplicate_of: None, // Re-indexing = already decided to replace
            }));
        }

        // Default: Return duplicate status (no re-indexing)
        warn!(
            "Duplicate PDF upload detected: existing_id={}",
            existing.pdf_id
        );

        // OODA-01 FIX: Initialize progress even for duplicates
        //
        // WHY: Frontend polls /pdf/progress/{track_id} immediately after upload.
        //      Even for duplicates, we need to return a valid progress entry
        //      so the frontend doesn't get a 404 error.
        //
        // The duplicate response tells the frontend it's already processed,
        // but the progress entry needs to exist for the initial poll.
        if let Some(ref track_id) = options.track_id {
            info!(
                "OODA-01: Initializing PDF progress for duplicate, track_id={}, pdf_id={}, filename={}",
                track_id, existing.pdf_id, existing.filename
            );
            state
                .pipeline_state
                .start_pdf_progress(track_id, &existing.pdf_id.to_string(), &existing.filename)
                .await;
        }

        let existing_pdf_id = existing.pdf_id.to_string();
        return Ok(Json(PdfUploadResponse {
            pdf_id: existing_pdf_id.clone(),
            document_id: existing.document_id.map(|id| id.to_string()),
            status: "duplicate".to_string(),
            task_id: "".to_string(),
            track_id: options.track_id.clone(),
            message: format!("PDF already uploaded with ID: {}", existing_pdf_id),
            estimated_time_seconds: 0,
            metadata: PdfMetadata {
                filename: existing.filename,
                file_size_bytes: existing.file_size_bytes,
                page_count: existing.page_count,
                sha256_checksum: existing.sha256_checksum,
                vision_enabled: options.enable_vision,
                vision_model: existing.vision_model,
            },
            // WHY: This field is what the frontend checks to trigger the
            // DuplicateUploadDialog, enabling the user to reprocess or skip.
            duplicate_of: Some(existing_pdf_id),
        }));
    }

    // 6. Extract page count (simple PDF parse)
    let page_count = extract_page_count(&file_data);

    // 7. Store raw PDF
    let vision_model = if matches!(
        resolved_backend,
        PdfParserBackend::Vision | PdfParserBackend::VlmOcr
    ) && (options.enable_vision
        || resolved_backend == PdfParserBackend::VlmOcr)
    {
        Some(options.vision_model())
    } else {
        None
    };

    // Use explicit title as filename if provided (e.g. citation-format name from API clients)
    let filename = options.title.clone().unwrap_or(filename);

    let pdf_id = match pdf_storage
        .create_pdf(CreatePdfRequest {
            workspace_id,
            filename: filename.clone(),
            content_type: "application/pdf".to_string(),
            file_size_bytes: file_data.len() as i64,
            sha256_checksum: checksum.clone(),
            page_count,
            pdf_data: file_data.clone(),
            vision_model: vision_model.clone(),
        })
        .await
    {
        Ok(id) => id,
        Err(e) => {
            // FIX-DUPLICATE-BUG: Handle concurrent upload race condition gracefully.
            // WHY: If the unique constraint fires (two uploads of the same PDF arrived
            // simultaneously), re-fetch the existing PDF and return a duplicate response
            // instead of a 500 error.
            let err_msg = format!("{}", e);
            if err_msg.contains("already exists") || err_msg.contains("concurrent upload") {
                warn!(
                    "Concurrent duplicate PDF detected via DB constraint: checksum={}",
                    checksum
                );
                if let Ok(Some(existing)) = pdf_storage
                    .find_pdf_by_checksum(&workspace_id, &checksum)
                    .await
                {
                    let existing_pdf_id = existing.pdf_id.to_string();
                    return Ok(Json(PdfUploadResponse {
                        pdf_id: existing_pdf_id.clone(),
                        document_id: existing.document_id.map(|id| id.to_string()),
                        status: "duplicate".to_string(),
                        task_id: "".to_string(),
                        track_id: options.track_id.clone(),
                        message: format!(
                            "PDF already uploaded with ID: {} (concurrent upload detected)",
                            existing_pdf_id
                        ),
                        estimated_time_seconds: 0,
                        metadata: PdfMetadata {
                            filename: existing.filename,
                            file_size_bytes: existing.file_size_bytes,
                            page_count: existing.page_count,
                            sha256_checksum: existing.sha256_checksum,
                            vision_enabled: options.enable_vision,
                            vision_model: existing.vision_model,
                        },
                        duplicate_of: Some(existing_pdf_id),
                    }));
                }
            }
            return Err(ApiError::Internal(format!("Failed to store PDF: {}", e)));
        }
    };

    info!(
        "PDF stored: id={}, size={}, pages={:?}",
        pdf_id,
        file_data.len(),
        page_count
    );

    // 8. Create background task
    let task_id =
        create_pdf_processing_task(&state, &context, pdf_id, &options, workspace.as_ref()).await?;

    // 9. OODA-01: Initialize progress tracking immediately
    //
    // WHY: Frontend polls /pdf/progress/{track_id} immediately after upload.
    //      Previously, progress was only initialized when the task callback
    //      fired (on_extraction_start), causing a race condition → 404 errors.
    //
    // FIX: Initialize progress here, before returning. The callback will
    //      update phases as processing proceeds, but the entry now exists.
    let effective_track_id = options.track_id.clone().unwrap_or_else(|| task_id.clone());
    info!(
        "OODA-01: Initializing PDF progress for track_id={}, pdf_id={}, filename={}",
        effective_track_id, pdf_id, filename
    );
    state
        .pipeline_state
        .start_pdf_progress(&effective_track_id, &pdf_id.to_string(), &filename)
        .await;

    // 10. Estimate processing time (rough heuristic)
    let estimated_time = estimate_processing_time(&file_data, page_count);

    Ok(Json(PdfUploadResponse {
        pdf_id: pdf_id.to_string(),
        document_id: None,
        status: "processing".to_string(),
        task_id: task_id.to_string(),
        track_id: options.track_id,
        message: "PDF uploaded successfully. Processing in background.".to_string(),
        estimated_time_seconds: estimated_time,
        metadata: PdfMetadata {
            filename,
            file_size_bytes: file_data.len() as i64,
            page_count,
            sha256_checksum: checksum,
            vision_enabled: options.enable_vision,
            vision_model,
        },
        duplicate_of: None,
    }))
}

// ============================================================================
// URL Upload
// ============================================================================

/// JSON body for `POST /api/v1/documents/pdf/from-url`.
#[derive(Debug, Deserialize, utoipa::ToSchema)]
pub struct UploadPdfFromUrlRequest {
    /// Direct URL to a PDF. Must be `http(s)`. Will be HEAD-checked for
    /// `application/pdf` (or `application/octet-stream` + `.pdf` path)
    /// before downloading.
    pub url: String,
    /// Optional override for the initial filename. If absent, we derive
    /// from the URL's basename (falling back to `Paper-<hash>.pdf`).
    /// The post-OCR rename step will replace this with a citation-style
    /// name once the paper's front-matter is extractable.
    #[serde(default)]
    pub title: Option<String>,
    /// Opt-in force-reindex for the shared dedup/reprocessing path.
    #[serde(default)]
    pub force_reindex: bool,
    /// Propagated to the task so the front-end progress poller can
    /// correlate. Server auto-generates one if absent.
    #[serde(default)]
    pub track_id: Option<String>,
}

/// Upload a PDF identified by URL.
///
/// # Flow
///
/// 1. Validate URL (scheme + SSRF guard)
/// 2. HEAD → verify `Content-Type` is PDF-ish (best-effort; falls through
///    to magic-byte check when HEAD is unsupported)
/// 3. GET → stream into memory with a hard size cap
/// 4. Magic-byte check (`%PDF-`)
/// 5. Delegate to the shared [`ingest_pdf_bytes`] path with
///    `rename_after_parse = true` and `source_url` in metadata so the
///    post-OCR step can rename to a citation-style filename.
#[utoipa::path(
    post,
    path = "/api/v1/documents/pdf/from-url",
    request_body = UploadPdfFromUrlRequest,
    responses(
        (status = 200, description = "PDF fetched and queued", body = PdfUploadResponse),
        (status = 400, description = "Invalid URL, non-PDF content, upstream fetch failure, or payload too large"),
        (status = 500, description = "Internal server error")
    ),
    tag = "Documents"
)]
pub async fn upload_pdf_from_url(
    State(state): State<AppState>,
    context: TenantContext,
    Json(request): Json<UploadPdfFromUrlRequest>,
) -> ApiResult<Json<PdfUploadResponse>> {
    info!(
        url = %request.url,
        workspace = ?context.workspace_id,
        tenant = ?context.tenant_id,
        "PDF upload-from-url request"
    );

    // 1. URL sanity + SSRF guard.
    let url = validate_external_http_url(&request.url)?;

    // HTTP client with tight timeouts — 10s for HEAD, 60s for the body
    // fetch. Buffered into a local `Vec<u8>` but bounded.
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .user_agent("edgequake-pdf-fetcher/1.0")
        .redirect(reqwest::redirect::Policy::limited(5))
        .build()
        .map_err(|e| ApiError::Internal(format!("Failed to build HTTP client: {e}")))?;

    // 2. HEAD check — best-effort. Skipped silently when the server
    //    returns 405 / connection reset / anything else we can't read —
    //    the magic-byte check in step 4 is the hard gate.
    head_check_pdf_content_type(&client, url.as_str()).await?;

    // 3. Download with size cap.
    let file_data = stream_pdf_bytes(&client, url.as_str(), MAX_URL_FETCH_BYTES).await?;

    // 4. Magic-byte gate. `validate_pdf_data` inside `ingest_pdf_bytes`
    //    will also reject non-PDF bytes, but checking here gives a
    //    clearer error message distinguishing "server lied about
    //    content-type" from "PDF parser rejected the bytes".
    if !file_data.starts_with(PDF_MAGIC) {
        return Err(ApiError::BadRequest(format!(
            "URL content is not a PDF (expected magic '%PDF-', got {:?})",
            String::from_utf8_lossy(&file_data[..file_data.len().min(8)])
        )));
    }

    // 5. Derive the initial filename.
    let filename = derive_initial_filename(&request.title, &url, &file_data);

    // 6. Build options. Signal to the post-OCR rename step via metadata
    //    that we'd like a citation-style rename; include the source URL
    //    so the rename can parse arxiv-id → year when applicable.
    let mut metadata = serde_json::Map::new();
    metadata.insert("rename_after_parse".to_string(), serde_json::json!(true));
    metadata.insert("source_url".to_string(), serde_json::json!(url.as_str()));
    let options = PdfUploadOptions {
        enable_vision: true,
        vision_provider: None,
        vision_model: None,
        title: request.title,
        metadata: Some(serde_json::Value::Object(metadata)),
        track_id: request.track_id,
        force_reindex: request.force_reindex,
        pdf_parser_backend: None,
    };

    ingest_pdf_bytes(state, context, file_data, filename, options).await
}

/// Parse and sanity-check a user-supplied URL before we hit the network.
/// Rejects non-`http(s)` schemes and private / loopback / link-local
/// hostnames (best-effort SSRF guard — not a replacement for network-
/// level egress controls, but catches the obvious mistakes).
fn validate_external_http_url(raw: &str) -> ApiResult<reqwest::Url> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(ApiError::BadRequest("URL is empty".to_string()));
    }
    let url = reqwest::Url::parse(trimmed)
        .map_err(|e| ApiError::BadRequest(format!("Invalid URL: {e}")))?;
    match url.scheme() {
        "http" | "https" => {}
        other => {
            return Err(ApiError::BadRequest(format!(
                "URL scheme '{other}' not allowed — only http(s)"
            )));
        }
    }
    let host = url
        .host_str()
        .ok_or_else(|| ApiError::BadRequest("URL has no host".to_string()))?;
    let lower = host.to_ascii_lowercase();
    if lower == "localhost"
        || lower.ends_with(".localhost")
        || lower.starts_with("127.")
        || lower == "0.0.0.0"
        || lower.starts_with("10.")
        || lower.starts_with("192.168.")
        || lower.starts_with("169.254.")
        || lower == "::1"
        || lower.starts_with("[::1")
        || lower.starts_with("fe80")
    {
        return Err(ApiError::BadRequest(format!(
            "URL points to a private or loopback host ('{host}')"
        )));
    }
    // Also reject the 172.16/12 private range (rough — accept when first
    // octet mismatches). Only enforced for plain-dotted IPv4 literals.
    if let Some(rest) = lower.strip_prefix("172.") {
        if let Some((second, _)) = rest.split_once('.') {
            if let Ok(n) = second.parse::<u32>() {
                if (16..=31).contains(&n) {
                    return Err(ApiError::BadRequest(format!(
                        "URL points to a private host ('{host}')"
                    )));
                }
            }
        }
    }
    Ok(url)
}

/// Best-effort HEAD check. Returns `Ok(())` when the response headers
/// look like a PDF OR when HEAD is unsupported; returns an error only
/// when the server explicitly replies with non-PDF content. The magic-
/// byte check downstream is the non-negotiable gate.
async fn head_check_pdf_content_type(
    client: &reqwest::Client,
    url: &str,
) -> ApiResult<()> {
    let resp = match client
        .head(url)
        .timeout(std::time::Duration::from_secs(10))
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            // Network error on HEAD — don't fail the whole request; the
            // subsequent GET will surface the same issue with a clearer
            // error.
            debug!(error = %e, "HEAD request failed; proceeding to GET");
            return Ok(());
        }
    };

    if resp.status() == reqwest::StatusCode::METHOD_NOT_ALLOWED || !resp.status().is_success() {
        debug!(status = %resp.status(), "HEAD inconclusive; proceeding to GET");
        return Ok(());
    }

    let ctype = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_ascii_lowercase();
    let path = reqwest::Url::parse(url).ok().map(|u| u.path().to_string());
    let path_ends_pdf = path
        .as_deref()
        .map(|p| p.to_ascii_lowercase().ends_with(".pdf"))
        .unwrap_or(false);

    if ctype.starts_with("application/pdf") {
        return Ok(());
    }
    if ctype.starts_with("application/octet-stream") && path_ends_pdf {
        return Ok(());
    }
    if ctype.is_empty() {
        // Some CDNs omit Content-Type on HEAD — defer to magic bytes.
        return Ok(());
    }
    Err(ApiError::BadRequest(format!(
        "URL content-type is '{ctype}', not application/pdf"
    )))
}

/// Stream `url` into a `Vec<u8>`, aborting if the accumulated size
/// exceeds `max_bytes`. Uses chunked reads so we don't allocate the full
/// payload when the response is already rejected by HEAD / redirect.
async fn stream_pdf_bytes(
    client: &reqwest::Client,
    url: &str,
    max_bytes: u64,
) -> ApiResult<Vec<u8>> {
    let mut resp = client
        .get(url)
        .send()
        .await
        .map_err(|e| ApiError::BadRequest(format!("Failed to fetch URL: {e}")))?;
    if !resp.status().is_success() {
        return Err(ApiError::BadRequest(format!(
            "Upstream returned {} for {url}",
            resp.status()
        )));
    }

    // Honour Content-Length when provided — lets us reject huge payloads
    // before we read a single byte.
    if let Some(len) = resp.content_length() {
        if len > max_bytes {
            return Err(ApiError::BadRequest(format!(
                "URL content length {len} exceeds {} byte cap",
                max_bytes
            )));
        }
    }

    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = resp
        .chunk()
        .await
        .map_err(|e| ApiError::BadRequest(format!("Stream read failed: {e}")))?
    {
        if (buf.len() as u64).saturating_add(chunk.len() as u64) > max_bytes {
            return Err(ApiError::BadRequest(format!(
                "URL body exceeds {max_bytes} byte cap; aborting download"
            )));
        }
        buf.extend_from_slice(&chunk);
    }
    if buf.is_empty() {
        return Err(ApiError::BadRequest("Upstream returned empty body".to_string()));
    }
    Ok(buf)
}

/// Pick a reasonable initial filename for the stored PDF:
///   1. Caller-supplied `title` wins (user knows best).
///   2. Else the URL's last path segment, if it ends `.pdf`.
///   3. Else `Paper-<checksum[..8]>.pdf` so every row still has a name.
fn derive_initial_filename(
    title_override: &Option<String>,
    url: &reqwest::Url,
    file_data: &[u8],
) -> String {
    if let Some(t) = title_override.as_deref() {
        let trimmed = t.trim();
        if !trimmed.is_empty() {
            return if trimmed.to_ascii_lowercase().ends_with(".pdf") {
                trimmed.to_string()
            } else {
                format!("{trimmed}.pdf")
            };
        }
    }
    if let Some(last_segment) = url.path_segments().and_then(|segs| segs.last()) {
        let decoded = percent_decode(last_segment);
        let lower = decoded.to_ascii_lowercase();
        if lower.ends_with(".pdf") && decoded.len() > 4 {
            return decoded;
        }
    }
    let checksum = calculate_pdf_checksum(file_data);
    format!("Paper-{}.pdf", &checksum[..checksum.len().min(8)])
}

/// Very small percent-decoder for path segments. Good enough for arxiv
/// (`%20` → space) without pulling in a dependency.
fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(h), Some(l)) = (
                (bytes[i + 1] as char).to_digit(16),
                (bytes[i + 2] as char).to_digit(16),
            ) {
                out.push((h * 16 + l) as u8);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reject_non_http_scheme() {
        assert!(validate_external_http_url("file:///etc/passwd").is_err());
        assert!(validate_external_http_url("data:application/pdf,abc").is_err());
        assert!(validate_external_http_url("ftp://host/file.pdf").is_err());
    }

    #[test]
    fn reject_loopback_and_private_hosts() {
        for url in [
            "http://127.0.0.1/x.pdf",
            "https://localhost/x.pdf",
            "http://10.0.0.5/x.pdf",
            "http://192.168.1.2/x.pdf",
            "http://172.16.5.5/x.pdf",
            "http://172.31.1.1/x.pdf",
            "http://169.254.169.254/x.pdf", // AWS metadata
        ] {
            assert!(
                validate_external_http_url(url).is_err(),
                "should reject {url}"
            );
        }
    }

    #[test]
    fn accept_public_host() {
        assert!(validate_external_http_url("https://arxiv.org/pdf/2506.09452").is_ok());
        assert!(validate_external_http_url("https://example.com/paper.pdf").is_ok());
        // 172.15.x is outside the private range
        assert!(validate_external_http_url("http://172.15.1.1/x.pdf").is_ok());
    }

    #[test]
    fn derive_filename_prefers_title() {
        let u = reqwest::Url::parse("https://arxiv.org/pdf/2506.09452").unwrap();
        let f = derive_initial_filename(
            &Some("Smith - 2024 - X.pdf".to_string()),
            &u,
            b"%PDF-1.7 ...",
        );
        assert_eq!(f, "Smith - 2024 - X.pdf");
    }

    #[test]
    fn derive_filename_title_without_extension() {
        let u = reqwest::Url::parse("https://arxiv.org/pdf/2506.09452").unwrap();
        let f =
            derive_initial_filename(&Some("MyPaper".to_string()), &u, b"%PDF-1.7 ...");
        assert_eq!(f, "MyPaper.pdf");
    }

    #[test]
    fn derive_filename_from_url_basename() {
        let u = reqwest::Url::parse("https://example.com/papers/Hello%20World.pdf").unwrap();
        let f = derive_initial_filename(&None, &u, b"%PDF-1.7 ...");
        assert_eq!(f, "Hello World.pdf");
    }

    #[test]
    fn derive_filename_fallback_to_hash() {
        // URL basename doesn't end in .pdf
        let u = reqwest::Url::parse("https://arxiv.org/abs/2506.09452").unwrap();
        let f = derive_initial_filename(&None, &u, b"%PDF-1.7 hello");
        assert!(f.starts_with("Paper-"));
        assert!(f.ends_with(".pdf"));
    }
}
