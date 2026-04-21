//! Shared helpers for pipeline processing stages.
//!
//! These functions eliminate duplication across `process`, `process_with_progress`,
//! and `process_with_resilience` by extracting common logic for:
//! - Linking entities/relationships to source chunks
//! - Aggregating extraction statistics
//! - Generating embeddings (chunk, entity, relationship)
//! - Building document lineage

use std::collections::HashSet;
use std::sync::Arc;

use crate::chunker::TextChunk;
use crate::error::Result;
use crate::extractor::ExtractionResult;
use crate::lineage::{DocumentLineage, ExtractionMetadata, LineageBuilder, SourceSpan};

use super::{CostBreakdownStats, Pipeline, ProcessingStats};

// ─────────────────────────────────────────────────────────────────────────────
//                       EXTRACTION POST-PROCESSING
// ─────────────────────────────────────────────────────────────────────────────

/// Link extracted entities and relationships to their source chunks.
///
/// WHY: Without chunk linkage, Local/Global query modes cannot find
/// related chunks during retrieval — entities would be "orphaned" nodes
/// in the knowledge graph with no provenance trail.
pub(super) fn link_extractions_to_chunks(extractions: &mut [ExtractionResult]) {
    for extraction in extractions.iter_mut() {
        let chunk_id = extraction.source_chunk_id.clone();
        tracing::debug!(
            "Linking {} entities and {} relationships to chunk {}",
            extraction.entities.len(),
            extraction.relationships.len(),
            chunk_id
        );
        for entity in &mut extraction.entities {
            entity.add_source_chunk_id(&chunk_id);
        }
        for rel in &mut extraction.relationships {
            if rel.source_chunk_id.is_none() {
                rel.source_chunk_id = Some(chunk_id.clone());
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//                       STATISTICS AGGREGATION
// ─────────────────────────────────────────────────────────────────────────────

/// Aggregate extraction statistics from all successful extractions.
///
/// Populates entity/relationship counts, token usage, unique types/keywords,
/// and extraction cost in the provided `ProcessingStats`.
///
/// WHY UNIFIED: This logic was duplicated verbatim across `process`,
/// `process_with_progress`, and `process_with_resilience`. Extracting it
/// ensures consistent cost calculation and keyword collection.
pub(super) fn aggregate_extraction_stats(
    extractions: &[ExtractionResult],
    extractor: &Arc<dyn crate::extractor::EntityExtractor>,
    stats: &mut ProcessingStats,
) {
    let mut entity_types_set = HashSet::new();
    let mut relationship_types_set = HashSet::new();
    let mut keywords_set = HashSet::new();
    let mut total_input_tokens = 0usize;
    let mut total_output_tokens = 0usize;

    // Capture LLM model and provider names
    // @implements SPEC-032/OODA-226: Provider tracking in ProcessingStats
    stats.llm_model = Some(extractor.model_name().to_string());
    stats.llm_provider = Some(extractor.provider_name().to_string());

    for extraction in extractions {
        stats.entity_count += extraction.entities.len();
        stats.relationship_count += extraction.relationships.len();
        stats.llm_calls += 1;
        total_input_tokens += extraction.input_tokens;
        total_output_tokens += extraction.output_tokens;

        // Surface the structural-noise filter's drop counts so they
        // appear in the final `ProcessingStats`. The counters are
        // written by the tuple parser (see
        // `prompts::parser::tuple_parser::parse`) into the per-chunk
        // extraction metadata. Silently ignore missing / malformed
        // entries — the filter is belt-and-braces, not load-bearing.
        if let Some(n) = extraction
            .metadata
            .get("dropped_structural_entities")
            .and_then(|v| v.as_u64())
        {
            stats.dropped_structural_entities += n as usize;
        }
        if let Some(n) = extraction
            .metadata
            .get("dropped_structural_relationships")
            .and_then(|v| v.as_u64())
        {
            stats.dropped_structural_relationships += n as usize;
        }

        for entity in &extraction.entities {
            entity_types_set.insert(entity.entity_type.clone());
        }
        for rel in &extraction.relationships {
            relationship_types_set.insert(rel.relation_type.clone());
            for keyword in &rel.keywords {
                keywords_set.insert(keyword.clone());
            }
        }
    }

    stats.total_tokens = total_input_tokens + total_output_tokens;
    stats.input_tokens = total_input_tokens;
    stats.output_tokens = total_output_tokens;

    // Store collected types and keywords
    if !entity_types_set.is_empty() {
        stats.entity_types = Some(entity_types_set.into_iter().collect());
    }
    if !relationship_types_set.is_empty() {
        stats.relationship_types = Some(relationship_types_set.into_iter().collect());
    }
    if !keywords_set.is_empty() {
        let mut keywords: Vec<String> = keywords_set.into_iter().collect();
        keywords.sort();
        // Limit to top 50 keywords
        keywords.truncate(50);
        stats.keywords = Some(keywords);
    }

    // Calculate extraction cost using model pricing
    let model_name = extractor.model_name();
    let pricing = crate::progress::default_model_pricing();
    let model_pricing = pricing
        .get(model_name)
        .cloned()
        .unwrap_or_else(|| crate::progress::ModelPricing::new("gpt-4.1-nano", 0.00015, 0.0006));

    let extraction_cost = model_pricing.calculate_cost(total_input_tokens, total_output_tokens);
    stats.cost_usd += extraction_cost;

    let cost_breakdown = CostBreakdownStats {
        extraction_cost_usd: extraction_cost,
        extraction_input_tokens: total_input_tokens,
        extraction_output_tokens: total_output_tokens,
        ..CostBreakdownStats::default()
    };
    stats.cost_breakdown = Some(cost_breakdown);
}

// ─────────────────────────────────────────────────────────────────────────────
//                       EMBEDDING GENERATION HELPERS
// ─────────────────────────────────────────────────────────────────────────────

/// Conservative chars-per-true-token for dense technical content.
///
/// WHY 2.5: The chunker uses 4 chars/token (English prose).
/// Scientific PDFs contain tables with numbers, gene IDs, p-values, and
/// formulas where tokenizers split aggressively — real density can reach
/// 1.5–2.0 chars/token. Using 2.5 provides a safe intermediate buffer.
const EMBED_CHARS_PER_TOKEN: f64 = 2.5;

/// Safety headroom factor applied to the embedding context limit.
///
/// WHY 0.85: Leaves 15% slack for tokenizer variance, whitespace tokens,
/// and any prompt overhead the embedding endpoint may add.
const EMBED_SAFETY_FACTOR: f64 = 0.85;

/// Fallback maximum characters when `provider.max_tokens()` returns 0 (unknown).
///
/// 6 000 chars ≈ 2 400 tokens at 2.5 chars/token, keeping chunks well within
/// the 2 048-token limit of models like embeddinggemma.
const EMBED_FALLBACK_MAX_CHARS: usize = 6_000;

/// Compute the maximum safe character count for a single embedding input.
///
/// When the provider exposes its context limit, we derive the char cap from it.
/// When the limit is unknown (0), we fall back to `EMBED_FALLBACK_MAX_CHARS`.
fn embed_max_chars(max_tokens: usize) -> usize {
    if max_tokens == 0 {
        EMBED_FALLBACK_MAX_CHARS
    } else {
        (max_tokens as f64 * EMBED_CHARS_PER_TOKEN * EMBED_SAFETY_FACTOR) as usize
    }
}

/// Truncate `s` to at most `max_bytes`, preserving UTF-8 character boundaries.
fn truncate_at_char_boundary(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    // Walk back to the nearest valid UTF-8 boundary.
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// Guard a text batch before sending to the embedding provider.
///
/// Truncates any string that exceeds `max_chars` and logs a WARNING so
/// operators know chunks are being trimmed and can tune `chunk_size` or
/// switch to an embedding model with a larger context window.
///
/// WHY: A partial embedding is more useful than a pipeline failure.
/// The 400 "input length exceeds context length" error from Ollama would
/// otherwise abort the entire document ingestion.
fn guard_for_embedding(texts: &[String], max_chars: usize) -> Vec<String> {
    texts
        .iter()
        .enumerate()
        .map(|(i, text)| {
            if text.len() > max_chars {
                tracing::warn!(
                    input_index = i,
                    original_chars = text.len(),
                    cap_chars = max_chars,
                    "Embedding input truncated: text exceeds the safe token limit for the \
                     embedding model. Consider reducing chunk_size in PipelineConfig or \
                     switching to an embedding model with a larger context window."
                );
                truncate_at_char_boundary(text, max_chars).to_string()
            } else {
                text.clone()
            }
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
//                       EMBEDDING GENERATION
// ─────────────────────────────────────────────────────────────────────────────

impl Pipeline {
    /// Generate embeddings for chunks, entities, and relationships.
    ///
    /// WHY UNIFIED: All three processing methods shared identical embedding
    /// logic (~120 lines each). This single implementation handles:
    /// - Chunk embeddings (content → vector)
    /// - Entity embeddings (name: description → vector)
    /// - Relationship embeddings (keywords + source→target + description → vector)
    /// - Embedding cost calculation
    pub(super) async fn generate_all_embeddings(
        &self,
        chunks: &mut [TextChunk],
        extractions: &mut [ExtractionResult],
        stats: &mut ProcessingStats,
    ) -> Result<()> {
        let provider = match &self.embedding_provider {
            Some(p) => p,
            None => return Ok(()),
        };

        // Capture embedding model and provider info
        // @implements SPEC-032/OODA-226: Provider tracking in ProcessingStats
        stats.embedding_model = Some(provider.model().to_string());
        stats.embedding_provider = Some(provider.name().to_string());
        stats.embedding_dimensions = Some(provider.dimension());

        // Pre-compute the safe character limit for this provider once.
        // WHY: Avoids repeated calls to max_tokens() in tight loops and keeps
        // the guard logic in a single reusable helper (DRY).
        let max_chars = embed_max_chars(provider.max_tokens());

        // ── Chunk embeddings ──
        if self.config.enable_chunk_embeddings {
            let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
            if !texts.is_empty() {
                let safe_texts = guard_for_embedding(&texts, max_chars);
                let embeddings = provider
                    .embed(&safe_texts)
                    .await
                    .map_err(|e| crate::error::PipelineError::EmbeddingError(e.to_string()))?;

                for (chunk, embedding) in chunks.iter_mut().zip(embeddings) {
                    chunk.embedding = Some(embedding);
                }
            }
        }

        // ── Entity embeddings (batched) ──
        if self.config.enable_entity_embeddings {
            let mut all_entity_texts: Vec<String> = Vec::new();
            let mut entity_indices: Vec<(usize, usize)> = Vec::new(); // (extraction_idx, entity_idx)

            for (ext_idx, extraction) in extractions.iter().enumerate() {
                for (ent_idx, entity) in extraction.entities.iter().enumerate() {
                    all_entity_texts.push(format!("{}: {}", entity.name, entity.description));
                    entity_indices.push((ext_idx, ent_idx));
                }
            }

            if !all_entity_texts.is_empty() {
                let safe_entity_texts = guard_for_embedding(&all_entity_texts, max_chars);
                let all_embeddings = provider
                    .embed(&safe_entity_texts)
                    .await
                    .map_err(|e| crate::error::PipelineError::EmbeddingError(e.to_string()))?;

                // Validate embedding count matches input count
                // WHY: If provider returns fewer embeddings than inputs, zip() silently drops
                // entities without embeddings, causing graph nodes with missing vectors.
                if all_embeddings.len() != all_entity_texts.len() {
                    tracing::warn!(
                        expected = all_entity_texts.len(),
                        actual = all_embeddings.len(),
                        "Entity embedding count mismatch - some entities may lack embeddings"
                    );
                }

                for (embedding, (ext_idx, ent_idx)) in
                    all_embeddings.into_iter().zip(entity_indices)
                {
                    extractions[ext_idx].entities[ent_idx].embedding = Some(embedding);
                }
            }
        }

        // ── Relationship embeddings (batched) ──
        if self.config.enable_relationship_embeddings {
            let mut all_relationship_texts: Vec<String> = Vec::new();
            let mut relationship_indices: Vec<(usize, usize)> = Vec::new();

            for (ext_idx, extraction) in extractions.iter().enumerate() {
                for (rel_idx, r) in extraction.relationships.iter().enumerate() {
                    // Format: "keywords\tsource->target\ndescription"
                    // Matches LightRAG's relationship embedding format
                    all_relationship_texts.push(format!(
                        "{}\t{}->{}\n{}",
                        r.keywords.join(", "),
                        r.source,
                        r.target,
                        r.description
                    ));
                    relationship_indices.push((ext_idx, rel_idx));
                }
            }

            if !all_relationship_texts.is_empty() {
                let safe_rel_texts = guard_for_embedding(&all_relationship_texts, max_chars);
                let all_embeddings = provider
                    .embed(&safe_rel_texts)
                    .await
                    .map_err(|e| crate::error::PipelineError::EmbeddingError(e.to_string()))?;

                if all_embeddings.len() != all_relationship_texts.len() {
                    tracing::warn!(
                        expected = all_relationship_texts.len(),
                        actual = all_embeddings.len(),
                        "Relationship embedding count mismatch - some relationships may lack embeddings"
                    );
                }

                for (embedding, (ext_idx, rel_idx)) in
                    all_embeddings.into_iter().zip(relationship_indices)
                {
                    extractions[ext_idx].relationships[rel_idx].embedding = Some(embedding);
                }
            }
        }

        // ── Embedding cost calculation ──
        let mut total_embed_tokens = 0usize;

        if self.config.enable_chunk_embeddings {
            let chunk_text_len: usize = chunks.iter().map(|c| c.content.len()).sum();
            // Estimate token count (approx 4 chars per token)
            total_embed_tokens += chunk_text_len / 4;
        }
        if self.config.enable_entity_embeddings {
            for extraction in extractions.iter() {
                for entity in &extraction.entities {
                    total_embed_tokens += (entity.name.len() + entity.description.len()) / 4;
                }
            }
        }
        if self.config.enable_relationship_embeddings {
            for extraction in extractions.iter() {
                for rel in &extraction.relationships {
                    total_embed_tokens +=
                        (rel.source.len() + rel.target.len() + rel.description.len()) / 4;
                }
            }
        }

        let embed_model_name = provider.model();
        let pricing = crate::progress::default_model_pricing();
        let embed_pricing = pricing.get(embed_model_name).cloned().unwrap_or_else(|| {
            crate::progress::ModelPricing::new("text-embedding-3-small", 0.00002, 0.0)
        });

        let embedding_cost = embed_pricing.calculate_cost(total_embed_tokens, 0);
        stats.cost_usd += embedding_cost;

        if let Some(ref mut breakdown) = stats.cost_breakdown {
            breakdown.embedding_cost_usd = embedding_cost;
            breakdown.embedding_tokens = total_embed_tokens;
        } else {
            let breakdown = CostBreakdownStats {
                embedding_cost_usd: embedding_cost,
                embedding_tokens: total_embed_tokens,
                ..CostBreakdownStats::default()
            };
            stats.cost_breakdown = Some(breakdown);
        }

        Ok(())
    }

    // ─────────────────────────────────────────────────────────────────────────
    //                       LINEAGE BUILDING
    // ─────────────────────────────────────────────────────────────────────────

    /// Build document lineage from chunks and extractions.
    ///
    /// Returns `None` if lineage tracking is disabled in config.
    ///
    /// WHY UNIFIED: All three processing methods had identical lineage
    /// building code (~40 lines each). This single implementation ensures
    /// consistent entity/relationship ID generation and span recording.
    pub(super) fn build_lineage(
        &self,
        document_id: &str,
        chunks: &[TextChunk],
        extractions: &[ExtractionResult],
        stats: &ProcessingStats,
    ) -> Option<DocumentLineage> {
        if !self.config.enable_lineage_tracking {
            return None;
        }

        let job_id = uuid::Uuid::new_v4().to_string();
        let mut builder = LineageBuilder::new(document_id, document_id, &job_id);

        // Record chunks with their line numbers
        for chunk in chunks {
            let metadata = ExtractionMetadata::new(stats.llm_model.as_deref().unwrap_or("unknown"));
            builder.record_chunk(
                &chunk.id,
                chunk.index,
                chunk.start_line,
                chunk.end_line,
                chunk.start_offset,
                chunk.end_offset,
                metadata,
            );
        }

        // Record entities and relationships from extractions
        for extraction in extractions {
            for entity in &extraction.entities {
                let entity_id = format!("{}_{}", extraction.source_chunk_id, entity.name);
                let span = SourceSpan::new(0, 0, 0, 0);
                builder.record_entity(
                    &entity_id,
                    &entity.name,
                    &extraction.source_chunk_id,
                    span,
                    &entity.description,
                );
            }

            for rel in &extraction.relationships {
                let rel_id = format!(
                    "{}_{}_{}",
                    extraction.source_chunk_id, rel.source, rel.target
                );
                let span = SourceSpan::new(0, 0, 0, 0);
                builder.record_relationship(
                    &rel_id,
                    &rel.source,
                    &rel.target,
                    &rel.relation_type,
                    &extraction.source_chunk_id,
                    span,
                    &rel.description,
                );
            }
        }

        Some(builder.build())
    }

    /// Initialize processing stats from chunked document.
    ///
    /// Sets chunk_count, chunking_strategy, and avg_chunk_size.
    pub(super) fn init_chunk_stats(&self, chunks: &[TextChunk]) -> ProcessingStats {
        let avg_chunk_size = if chunks.is_empty() {
            None
        } else {
            let total_chars: usize = chunks.iter().map(|c| c.content.len()).sum();
            Some(total_chars / chunks.len())
        };

        ProcessingStats {
            chunk_count: chunks.len(),
            chunking_strategy: Some(format!("sliding_window_{}", self.config.chunker.chunk_size)),
            avg_chunk_size,
            ..ProcessingStats::default()
        }
    }
}
