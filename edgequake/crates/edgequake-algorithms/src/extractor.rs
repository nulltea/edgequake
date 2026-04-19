//! 3-pass algorithm extraction pipeline using EdgeQuake's LLM provider system.
//!
//! Adapted from RAGSearcher's `AlgorithmExtractor` — uses `edgequake_llm::LLMProvider`
//! instead of Claude CLI subprocess calls.

use std::sync::Arc;
use std::time::Instant;

use crate::prompts;
use crate::types::{
    AlgorithmExtractionOutput, AlgorithmExtractionResult, AlgorithmInventory,
    AlgorithmVerificationResult, ExtractedAlgorithm,
};

/// Errors that can occur during algorithm extraction.
#[derive(Debug, thiserror::Error)]
pub enum AlgorithmExtractionError {
    #[error("LLM call failed: {0}")]
    LlmError(String),
    #[error("Failed to parse LLM response as JSON: {0}")]
    ParseError(String),
    #[error("Extraction was cancelled")]
    Cancelled,
}

/// Extracts structured algorithm definitions from document text using a 3-pass LLM pipeline.
#[derive(Clone)]
pub struct AlgorithmExtractor {
    llm_provider: Arc<dyn edgequake_llm::traits::LLMProvider>,
}

impl AlgorithmExtractor {
    pub fn new(llm_provider: Arc<dyn edgequake_llm::traits::LLMProvider>) -> Self {
        Self { llm_provider }
    }

    /// Run all 3 passes as a convenience wrapper.
    pub async fn extract_algorithms(
        &self,
        source_text: &str,
    ) -> Result<AlgorithmExtractionResult, AlgorithmExtractionError> {
        let total_start = Instant::now();
        tracing::info!("Starting 3-pass algorithm extraction");

        let inventory = self.run_inventory(source_text).await?;

        if inventory.algorithms.is_empty() {
            tracing::info!("No algorithms found — returning empty result");
            return Ok(AlgorithmExtractionResult {
                algorithms: Vec::new(),
                verification: None,
            });
        }

        let extraction = self.run_extraction(source_text, &inventory).await?;
        let verification = self.run_verification(&extraction).await;

        tracing::info!(
            "Algorithm extraction complete in {:.1}s total ({} algorithms)",
            total_start.elapsed().as_secs_f64(),
            extraction.algorithms.len(),
        );

        Ok(AlgorithmExtractionResult {
            algorithms: extraction.algorithms,
            verification,
        })
    }

    /// Pass 1: Identify algorithms in a single contiguous text block.
    pub async fn run_inventory(
        &self,
        source_text: &str,
    ) -> Result<AlgorithmInventory, AlgorithmExtractionError> {
        self.run_inventory_on_text(source_text, "Pass 1").await
    }

    /// Pass 1 over a chunk pair. Returns algorithms identified in the concatenated chunk pair.
    pub async fn run_inventory_chunk_pair(
        &self,
        chunk_a: &str,
        chunk_b: Option<&str>,
        pair_index: usize,
    ) -> Result<AlgorithmInventory, AlgorithmExtractionError> {
        let combined = match chunk_b {
            Some(b) => format!("{}\n\n{}", chunk_a, b),
            None => chunk_a.to_string(),
        };
        self.run_inventory_on_text(&combined, &format!("Pass 1 [pair {pair_index}]"))
            .await
    }

    async fn run_inventory_on_text(
        &self,
        source_text: &str,
        label: &str,
    ) -> Result<AlgorithmInventory, AlgorithmExtractionError> {
        tracing::info!("{label}: Identifying algorithms...");
        let start = Instant::now();

        let inventory_prompt = prompts::algorithm_inventory_prompt();
        let full_prompt = format!(
            "## Paper Excerpt\n{}\n\n## Instructions\n{}",
            source_text, inventory_prompt
        );

        let options = edgequake_llm::traits::CompletionOptions {
            max_tokens: Some(4096),
            temperature: Some(0.0),
            reasoning_effort: Some("none".to_string()),
            ..Default::default()
        };

        let response = self
            .llm_provider
            .complete_with_options(&full_prompt, &options)
            .await
            .map_err(|e| AlgorithmExtractionError::LlmError(format!("{label} failed: {e}")))?;

        let inventory: AlgorithmInventory =
            parse_json_response(&response.content).map_err(|e| {
                AlgorithmExtractionError::ParseError(format!("{label} parse error: {e}"))
            })?;

        tracing::info!(
            "{label} complete in {:.1}s: {} algorithms identified",
            start.elapsed().as_secs_f64(),
            inventory.algorithms.len(),
        );

        Ok(inventory)
    }

    /// Pass 2: Extract detailed algorithm definitions from a single text block.
    pub async fn run_extraction(
        &self,
        source_text: &str,
        inventory: &AlgorithmInventory,
    ) -> Result<AlgorithmExtractionOutput, AlgorithmExtractionError> {
        self.run_extraction_on_text(source_text, inventory, "Pass 2")
            .await
    }

    /// Pass 2 over a chunk pair with an inventory scoped to that pair.
    pub async fn run_extraction_chunk_pair(
        &self,
        chunk_a: &str,
        chunk_b: Option<&str>,
        inventory: &AlgorithmInventory,
        pair_index: usize,
    ) -> Result<AlgorithmExtractionOutput, AlgorithmExtractionError> {
        let combined = match chunk_b {
            Some(b) => format!("{}\n\n{}", chunk_a, b),
            None => chunk_a.to_string(),
        };
        self.run_extraction_on_text(&combined, inventory, &format!("Pass 2 [pair {pair_index}]"))
            .await
    }

    async fn run_extraction_on_text(
        &self,
        source_text: &str,
        inventory: &AlgorithmInventory,
        label: &str,
    ) -> Result<AlgorithmExtractionOutput, AlgorithmExtractionError> {
        tracing::info!("{label}: Extracting algorithm definitions...");
        let start = Instant::now();

        let inventory_json = serde_json::to_string_pretty(inventory).map_err(|e| {
            AlgorithmExtractionError::ParseError(format!("Inventory serialization: {e}"))
        })?;
        let extraction_prompt = prompts::algorithm_extraction_prompt(&inventory_json);
        let full_prompt = format!(
            "## Paper Excerpt\n{}\n\n## Instructions\n{}",
            source_text, extraction_prompt
        );

        // reasoning_effort=low. The extraction schema is rigid and the prompt
        // is explicit; we don't need a lot of think-aloud. Leaving it at the
        // provider default lets heavy reasoning models (e.g. gemma 4 26B A4B)
        // eat the entire timeout budget with reasoning tokens and return
        // truncated output — observed 26B producing 2 incomplete algos where
        // gemma 4 (E4B) produced 5 clean ones at the same setting. `"low"`
        // keeps small focused reasoning for light models without letting
        // larger ones over-think.
        let options = edgequake_llm::traits::CompletionOptions {
            max_tokens: Some(32768),
            temperature: Some(0.0),
            reasoning_effort: Some("low".to_string()),
            ..Default::default()
        };

        // Retry transient failures (network errors, timeouts, empty responses).
        // Heavy vision+extraction models like gemma 4 (26B A4B) routinely drop
        // connections or time out under concurrent load; without retries a
        // single burst of failures silently loses most of the algorithm blocks.
        const MAX_ATTEMPTS: usize = 3;
        let mut last_err: Option<AlgorithmExtractionError> = None;
        let mut extraction: Option<AlgorithmExtractionOutput> = None;
        for attempt in 1..=MAX_ATTEMPTS {
            match self
                .llm_provider
                .complete_with_options(&full_prompt, &options)
                .await
            {
                Ok(response) => {
                    match parse_json_response::<AlgorithmExtractionOutput>(&response.content) {
                        Ok(mut parsed) => {
                            // Post-parse LaTeX repair: reconstruct
                            // JSON-escape collisions (tab+ext → \text, BS+eta
                            // → \beta, …), balance [[…]] and $…$, brace
                            // ^\theta, strip \mathbb{S} OCR artefacts.
                            // Applied here, before returning, so every caller
                            // of Pass 2 gets clean data and the storage
                            // layer never sees corrupted LaTeX.
                            repair_extraction(&mut parsed);
                            extraction = Some(parsed);
                            break;
                        }
                        Err(e) => {
                            // Empty or malformed response — retry.
                            let err = AlgorithmExtractionError::ParseError(format!(
                                "{label} parse error (attempt {attempt}/{MAX_ATTEMPTS}): {e}"
                            ));
                            tracing::warn!(
                                attempt = attempt,
                                max = MAX_ATTEMPTS,
                                label = label,
                                error = %err,
                                "Pass 2 parse failure — retrying"
                            );
                            last_err = Some(err);
                        }
                    }
                }
                Err(e) => {
                    let err = AlgorithmExtractionError::LlmError(format!(
                        "{label} failed (attempt {attempt}/{MAX_ATTEMPTS}): {e}"
                    ));
                    tracing::warn!(
                        attempt = attempt,
                        max = MAX_ATTEMPTS,
                        label = label,
                        error = %err,
                        "Pass 2 LLM call failed — retrying"
                    );
                    last_err = Some(err);
                }
            }

            if attempt < MAX_ATTEMPTS {
                // Backoff: 2s, 5s. Keeps retries short enough for an 8-block
                // extraction to still finish within reasonable wall time.
                let delay_secs = if attempt == 1 { 2 } else { 5 };
                tokio::time::sleep(std::time::Duration::from_secs(delay_secs)).await;
            }
        }

        let extraction = match extraction {
            Some(e) => e,
            None => {
                return Err(last_err.unwrap_or_else(|| {
                    AlgorithmExtractionError::LlmError(format!(
                        "{label} failed with no error recorded"
                    ))
                }))
            }
        };

        tracing::info!(
            "{label} complete in {:.1}s: {} algorithms extracted",
            start.elapsed().as_secs_f64(),
            extraction.algorithms.len(),
        );

        Ok(extraction)
    }

    /// Pass 3: Verify algorithm quality (non-fatal).
    pub async fn run_verification(
        &self,
        extraction: &AlgorithmExtractionOutput,
    ) -> Option<AlgorithmVerificationResult> {
        tracing::info!("Pass 3/3: Verifying algorithm definitions...");
        let start = Instant::now();

        let algorithms_json = match serde_json::to_string_pretty(extraction) {
            Ok(j) => j,
            Err(e) => {
                tracing::warn!("Pass 3 skipped — serialization error: {e}");
                return None;
            }
        };
        let verification_prompt = prompts::algorithm_verification_prompt(&algorithms_json);

        let options = edgequake_llm::traits::CompletionOptions {
            max_tokens: Some(4096),
            temperature: Some(0.0),
            reasoning_effort: Some("none".to_string()),
            ..Default::default()
        };

        match self
            .llm_provider
            .complete_with_options(&verification_prompt, &options)
            .await
        {
            Ok(resp) => match parse_json_response::<AlgorithmVerificationResult>(&resp.content) {
                Ok(vr) => {
                    tracing::info!(
                        "Pass 3 complete in {:.1}s: status={}, quality={}, issues={}",
                        start.elapsed().as_secs_f64(),
                        vr.verification_status,
                        vr.overall_quality,
                        vr.completeness_issues.len(),
                    );
                    Some(vr)
                }
                Err(e) => {
                    tracing::warn!(
                        "Pass 3 parse failed in {:.1}s (non-fatal): {e}",
                        start.elapsed().as_secs_f64(),
                    );
                    None
                }
            },
            Err(e) => {
                tracing::warn!(
                    "Pass 3 call failed in {:.1}s (non-fatal): {e}",
                    start.elapsed().as_secs_f64(),
                );
                None
            }
        }
    }
}

/// Sanitize LLM JSON output by escaping unrecognized backslash sequences.
///
/// LLMs frequently emit raw LaTeX like `\mathcal`, `\theta` inside JSON strings.
/// JSON only allows `\n \r \t \\ \" \/ \b \f \uXXXX` — everything else is invalid.
/// This function doubles unrecognized backslashes so serde_json can parse them.
///
/// Retained as a first-line fallback *before* handing off to `llm_json` —
/// `llm_json` handles structural issues (unquoted keys, trailing commas,
/// unclosed brackets) but does nothing about bare `\mathcal` becoming an
/// invalid JSON escape. This pass covers that gap.
fn sanitize_llm_json(input: &str) -> String {
    let mut result = String::with_capacity(input.len() + 64);
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    let mut in_string = false;

    while i < chars.len() {
        let c = chars[i];

        if c == '"' && (i == 0 || chars[i - 1] != '\\') {
            in_string = !in_string;
            result.push(c);
            i += 1;
            continue;
        }

        if in_string && c == '\\' && i + 1 < chars.len() {
            let next = chars[i + 1];
            match next {
                // Valid JSON escapes — pass through as-is
                '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't' => {
                    result.push('\\');
                    result.push(next);
                    i += 2;
                }
                'u' => {
                    // \uXXXX — pass through
                    result.push('\\');
                    result.push('u');
                    i += 2;
                }
                _ => {
                    // Unrecognized escape (e.g. \m from \mathcal) — double the backslash
                    result.push('\\');
                    result.push('\\');
                    result.push(next);
                    i += 2;
                }
            }
            continue;
        }

        result.push(c);
        i += 1;
    }

    result
}

/// Parse a JSON response from the LLM.
///
/// Pipeline:
///   1. Strip markdown code fences.
///   2. Fast path: try `serde_json::from_str` as-is — most clean outputs parse here.
///   3. On failure: apply `sanitize_llm_json` (doubles unrecognised backslashes
///      so LaTeX commands like `\mathcal` survive), then `llm_json::repair_json`
///      (battle-tested structural repair for unquoted keys, trailing commas,
///      unclosed brackets, single quotes, stray prose).
///   4. Retry `serde_json::from_str` on the repaired string.
///
/// Post-parse LaTeX repair (reconstructing `\text` from tab+ext collisions,
/// `\beta` from backspace+eta, balancing `[[...]]`, closing orphan `$`,
/// bracing `^\theta`) is applied by [`repair_extracted`] on the deserialised
/// `ExtractedAlgorithm` — see its caller in `run_extraction_on_text`.
pub fn parse_json_response<T: serde::de::DeserializeOwned>(content: &str) -> Result<T, String> {
    let trimmed = content.trim();

    // Strip markdown code fences if the LLM wrapped the JSON
    let json_str = if trimmed.starts_with("```") {
        let without_start = trimmed
            .strip_prefix("```json")
            .or_else(|| trimmed.strip_prefix("```"))
            .unwrap_or(trimmed);
        without_start
            .strip_suffix("```")
            .unwrap_or(without_start)
            .trim()
    } else {
        trimmed
    };

    // Try parsing as-is first (fast path)
    if let Ok(v) = serde_json::from_str(json_str) {
        return Ok(v);
    }

    // Stage A: double unrecognised backslashes so LaTeX commands parse.
    let sanitized = sanitize_llm_json(json_str);
    if let Ok(v) = serde_json::from_str(&sanitized) {
        return Ok(v);
    }

    // Stage B: structural repair (trailing commas, unquoted keys,
    // unclosed brackets, single quotes, stray prose). llm_json is a
    // Rust port of Python json_repair — handles the bulk of
    // malformed-LLM-JSON cases we don't.
    match llm_json::repair_json(&sanitized, &Default::default()) {
        Ok(repaired) => serde_json::from_str(&repaired).map_err(|e| {
            let preview: String = json_str.chars().take(200).collect();
            format!("{e} (response preview after repair: {preview})")
        }),
        Err(repair_err) => {
            let preview: String = json_str.chars().take(200).collect();
            Err(format!(
                "JSON parse + llm_json repair both failed: {repair_err} (response preview: {preview})"
            ))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Post-deserialisation LaTeX repair
// ─────────────────────────────────────────────────────────────────────────────

/// Walk every math-bearing field of an `AlgorithmExtractionOutput` and apply
/// [`edgequake_pdf::latex_repair::repair_latex`] to reconstruct JSON-escape
/// collisions, balance brackets and `$` delimiters, brace single-token
/// super/sub-scripts, and strip OCR artefacts.
///
/// Pass-2 output contains LaTeX in many places; we repair them all before
/// the data is persisted.
pub fn repair_extraction(extraction: &mut AlgorithmExtractionOutput) {
    for algo in &mut extraction.algorithms {
        repair_extracted(algo);
    }
}

fn repair_extracted(algo: &mut ExtractedAlgorithm) {
    algo.name = edgequake_pdf::latex_repair::repair_latex(&algo.name);
    algo.description = edgequake_pdf::latex_repair::repair_latex(&algo.description);
    if let Some(s) = algo.mathematical_notation.as_mut() {
        *s = edgequake_pdf::latex_repair::repair_latex(s);
    }
    if let Some(s) = algo.pseudocode.as_mut() {
        *s = edgequake_pdf::latex_repair::repair_latex(s);
    }
    if let Some(s) = algo.complexity.as_mut() {
        *s = edgequake_pdf::latex_repair::repair_latex(s);
    }
    for step in &mut algo.steps {
        step.action = edgequake_pdf::latex_repair::repair_latex(&step.action);
        step.details = edgequake_pdf::latex_repair::repair_latex(&step.details);
        if let Some(s) = step.math.as_mut() {
            *s = edgequake_pdf::latex_repair::repair_latex(s);
        }
    }
    for io in algo.inputs.iter_mut().chain(algo.outputs.iter_mut()) {
        io.name = edgequake_pdf::latex_repair::repair_latex(&io.name);
        io.io_type = edgequake_pdf::latex_repair::repair_latex(&io.io_type);
        io.description = edgequake_pdf::latex_repair::repair_latex(&io.description);
    }
    for pre in &mut algo.preconditions {
        *pre = edgequake_pdf::latex_repair::repair_latex(pre);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_json_response_plain() {
        let json = r#"{"paper_title": "Test", "algorithms": [], "paper_type": "empirical"}"#;
        let result: AlgorithmInventory = parse_json_response(json).unwrap();
        assert_eq!(result.paper_title, "Test");
        assert!(result.algorithms.is_empty());
    }

    #[test]
    fn test_parse_json_response_with_fences() {
        let json =
            "```json\n{\"paper_title\": \"Test\", \"algorithms\": [], \"paper_type\": \"empirical\"}\n```";
        let result: AlgorithmInventory = parse_json_response(json).unwrap();
        assert_eq!(result.paper_title, "Test");
    }

    #[test]
    fn test_parse_json_with_latex_escapes() {
        // Simulates LLM output with unescaped LaTeX
        let json =
            r#"{"paper_title": "Test \mathcal{D}", "algorithms": [], "paper_type": "empirical"}"#;
        let result: AlgorithmInventory = parse_json_response(json).unwrap();
        assert!(result.paper_title.contains("\\mathcal"));
    }

    #[test]
    fn test_sanitize_preserves_valid_escapes() {
        let input = r#"{"key": "line1\nline2\ttab"}"#;
        let sanitized = sanitize_llm_json(input);
        assert_eq!(input, sanitized);
    }

    #[test]
    fn test_sanitize_doubles_unrecognised_escapes() {
        // `\m` is not a valid JSON escape → should be doubled so the string
        // parses cleanly. `\t` IS a valid JSON escape (tab), so it is passed
        // through — the collision is then repaired post-parse by
        // `edgequake_pdf::latex_repair::reconstruct_escape_collisions` walking
        // the deserialised fields via `repair_extraction`.
        let input = r#"{"desc": "uses \mathcal{D}"}"#;
        let sanitized = sanitize_llm_json(input);
        assert!(
            sanitized.contains(r"\\mathcal"),
            "expected doubled backslash, got: {sanitized}"
        );
    }
}
