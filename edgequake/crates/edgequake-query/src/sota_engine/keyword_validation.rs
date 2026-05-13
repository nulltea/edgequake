use crate::keywords::ExtractedKeywords;

use super::SOTAQueryEngine;

impl SOTAQueryEngine {
    /// Sort entities by degree (descending) for importance-based ranking.
    ///
    /// High-degree entities are more connected in the knowledge graph
    /// and typically represent more important/central concepts.
    pub(super) fn sort_entities_by_degree(&self, entities: &mut [crate::context::RetrievedEntity]) {
        entities.sort_by(|a, b| {
            // Sort by degree descending (higher degree = more important)
            b.degree.cmp(&a.degree)
        });
        tracing::debug!(
            entity_count = entities.len(),
            top_degree = entities.first().map(|e| e.degree).unwrap_or(0),
            "Sorted entities by degree"
        );
    }

    /// Validate keywords against the knowledge graph.
    ///
    /// WHY: When a query contains terms that don't exist in the knowledge base
    /// (e.g., "STLA Medium"), including them in the embedding computation dilutes
    /// the semantic search and reduces retrieval quality for terms that DO exist.
    ///
    /// This method checks each low-level keyword against the graph and drops
    /// those with zero entity matches, preventing embedding dilution.
    pub(super) async fn validate_keywords(
        &self,
        keywords: &ExtractedKeywords,
    ) -> ExtractedKeywords {
        if keywords.low_level.is_empty() {
            return keywords.clone();
        }

        let mut validated_low_level = Vec::new();
        let mut dropped_keywords = Vec::new();

        for keyword in &keywords.low_level {
            // Check cache first
            let cache_key = keyword.to_lowercase();
            let cached_result = {
                let cache = self.keyword_validation_cache.read().await;
                cache.get(&cache_key).copied()
            };

            let exists = if let Some(exists) = cached_result {
                // Cache hit
                exists
            } else {
                // Cache miss - check graph
                let matches = self.graph_storage.search_labels(keyword, 1).await;
                let exists = matches.map(|labels| !labels.is_empty()).unwrap_or(false);

                // Update cache
                {
                    let mut cache = self.keyword_validation_cache.write().await;
                    // Limit cache size to prevent unbounded growth
                    if cache.len() < 10000 {
                        cache.insert(cache_key, exists);
                    }
                }
                exists
            };

            if exists {
                validated_low_level.push(keyword.clone());
            } else {
                dropped_keywords.push(keyword.clone());
            }
        }

        if !dropped_keywords.is_empty() {
            tracing::info!(
                dropped = ?dropped_keywords,
                kept = ?validated_low_level,
                "Dropped keywords with no graph matches"
            );
        }

        // If ALL keywords were dropped, fall back to original to avoid empty search
        if validated_low_level.is_empty() {
            tracing::warn!(
                original = ?keywords.low_level,
                "All keywords dropped - falling back to original keywords"
            );
            return keywords.clone();
        }

        ExtractedKeywords::new(
            keywords.high_level.clone(),
            validated_low_level,
            keywords.query_intent,
        )
    }
}
